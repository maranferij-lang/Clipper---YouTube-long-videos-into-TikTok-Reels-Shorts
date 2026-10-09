"""
render_opus_tc_v4.py  (19.09.2026)  9:16 render "like Opus Clip" for Clipper.
Version history: _tools/CHANGELOG.md. This is v3.5.11 without dead code, with profiles and no change to the frame logic.

HOW IT WORKS (short, so you do not have to read the code)
 1. Source shots: a cut = a peak in the frame difference; also a boundary where there are no faces for longer than 1.2 s (slide, graphic).
    The camera stays still inside a shot. The speaker switches only at a shot boundary or in a pause (Light-ASD via --asd).
 2. Mode per shot:  slide without faces -> WIDE (the whole 16:9 frame as a band on a blurred background) for its full length;
    short wide shot (1.2-4.5 s, face < --band-wide) -> WIDE, but not right next to a splice or the clip edge;
    long wide shot -> SPEAKER (crop on the speaker); close-up -> SPEAKER full-height crop.
 3. One crop scale per source shot: 10th percentile of the face height over the shot, zoomed in so the face is
    >= 17 % of the crop, floor 0.62 of the frame height (upscale <= 1.4x), ceiling: the largest face of the shot <= 50 % of the crop.
 4. Own cuts: a segment longer than --hold-max is cut at a pause/punctuation; if there are none and the segment is > 1.5*hold-max,
    between words. The final phrase (2-5 s) is a single frame. Segments < --min-seg are merged within a shot; a stub < 1 s
    at the clip edge or near a splice is merged even across a source cut. One "breath" band (--band 1) on the quietest
    wide segment, not near a splice.
 5. Captions: UPPERCASE, Montserrat ExtraBold, up to 3 words per line, up to 2 lines, a pill on the active word,
    block at --sub-y; in a pause > 1 s the caption disappears after --cap-hold. A title plate at the top only with --title (per clip).
 6. Audio is left untouched: highpass/loudnorm is done by the pipeline in the final stage.

PROFILES (--profile; an explicitly given parameter takes priority over the profile)
  podcast (default): multicamera, three people in frame. punch 1.0 (no punch-in), hold-max 9, band 1, short-shot 1.5,
                    min-crop 1.0 (full-height crop, upscale only in a wide shot per item 3), hl yellow.
  solo:             one person, the camera does not cut. band 0, band-wide 0, switch-hold 60, min-dwell 60, face-frac 0.22,
                    punch 0.76, cuts only at sentence boundaries with a step >= --punch-gap (9 s), hl #C82828.

RUN (usually via the pipeline):
  python render_opus_tc_v4.py --raw raw.mp4 --words words.json --out clip.mp4 --font Mv.ttf --yunet yunet.onnx \
      --asd asd/ --splices 12.4,28.0 --path-json path.json [--profile podcast] [--frames A:B] [--title "..."]
"""
import sys, os, json, subprocess, argparse, math, collections
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument('--raw', required=True); ap.add_argument('--words', required=True)
ap.add_argument('--out', required=True); ap.add_argument('--font', required=True)
ap.add_argument('--yunet', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yunet.onnx'))
ap.add_argument('--sub-y', type=int, default=1480); ap.add_argument('--title', default='')
ap.add_argument('--det-every', type=int, default=3)
ap.add_argument('--face-frac', type=float, default=0.29, help='target face height / crop height (zoom on wide shots). v3.1: 0.29, in Opus the face is 0.25-0.30 of the frame height (was 0.32)')
ap.add_argument('--min-crop', type=float, default=-1, help='minimum crop height as a fraction of the frame; -1 = auto (4K: down to 1080 px, 1080p: 1.0 without extra zoom); 0.6 = zoom to face-frac')
ap.add_argument('--sface', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face_recognition_sface_2021dec.onnx'), help='v1.4: SFace model for person identity across shots')
ap.add_argument('--sface-thr', type=float, default=0.45, help='v1.4: cosine-similarity threshold for "same person"')
ap.add_argument('--path-json', default='', help='camera trajectory cache: if the file exists, detection is skipped')
ap.add_argument('--frames', default='', help='A:B - render only frames [A,B) (for chunked rendering)')
ap.add_argument('--path-only', action='store_true', help='only compute the trajectory and exit')
ap.add_argument('--switch-hold', type=float, default=2.6, help='v3.0: the camera switches to another person only if they lead for at least this many seconds. Shorter interjections are ignored, as Opus does (the visible person is the speaker only 62 %% of the time)')
ap.add_argument('--band', type=int, default=2, help='v3.0: how many times per clip to go to a band frame with blurred top/bottom (Opus reference: 1-3 times, 10 %% of the time)')
ap.add_argument('--band-len', type=float, default=3.0, help='v3.0: target duration of such a frame (reference 1.5-4.5 s)')
ap.add_argument('--band-end', type=int, default=1, help='v3.0: one of the band shots is at the end of the clip')
ap.add_argument('--min-dwell', type=float, default=1.6, help='v1.3: a new speaker must lead for at least this many seconds, otherwise the camera does not switch (exclamations like "of course", micro-sounds)')
ap.add_argument('--min-hold', type=float, default=2.5, help='v1.3: after a switch, hold the frame for at least this many seconds')
ap.add_argument('--snap', type=float, default=0.6, help='v1.3: snap the switch moment to the nearest speech pause (+-s, 0 = off)')
ap.add_argument('--sharpen', type=float, default=-1, help='v1.3: unsharp after scaling (0..1); -1 = auto (0.35 when upscaling, 0 when downscaling from 4K)')
ap.add_argument('--title-sec', type=float, default=0, help='v1.3: how many seconds to hold the hook plate (0 = the whole clip, as in ep2)')
ap.add_argument('--caps', type=int, default=1, help='v2.0: subtitles in CAPITAL letters (as Opus)')
ap.add_argument('--hl', default='yellow', help='v2.0: highlight colour of the active word: yellow | green')
ap.add_argument('--weight', default='ExtraBold', help='v3.1: named instance of the variable font for captions and the plate (ExtraBold | Bold | Black)')
ap.add_argument('--band-wide', type=float, default=0.22, help='v3.1: a shot counts as wide (a band-frame candidate) if the speaker face is smaller than this fraction of the source height')
ap.add_argument('--cap-hold', type=float, default=0.6, help='v3.1: how many seconds a caption stays after the end of the phrase if the next phrase is not close (Opus: captions do not hang in pauses)')
ap.add_argument('--font-max', type=int, default=72, help='v2.0: maximum subtitle font size (Opus ~64-72 px)')
ap.add_argument('--face-y', type=float, default=0.19, help='v2.4: shift the crop down so the face sits higher in the frame; 0.19 puts the face center at 0.31 of the height (as in Opus), 0.10 gave 0.40')
ap.add_argument('--min-face', type=float, default=0.11, help='v2.0: speaker face smaller than this fraction of the height -> WIDE mode')
ap.add_argument('--split', action='store_true', help='v2.0: allow SPLIT mode (two people one above the other)')
ap.add_argument('--hold-max', type=float, default=5.5, help='v2.2: if one angle is held longer, insert an own cut at a speech pause (Opus reference: 90 %% of shots <= 6.4 s)')
ap.add_argument('--hold-min', type=float, default=2.0, help='v2.2: we do not hold an angle shorter than this (reference: 19 %% of shots < 1.5 s are source cuts, our own cuts are never that short)')
ap.add_argument('--punch', type=float, default=0.84, help='v2.2: how much tighter the punch-in is on an own cut (0.84 = 16 %% closer)')
ap.add_argument('--cut-peak', type=float, default=1.8, help='v2.1: how many times a difference must exceed its neighbours to count as a cut (a peak, not motion)')
ap.add_argument('--hold-tail', type=int, default=1, help='v2.3: do not cut during the final phrase (the punchline is held as one frame)')
ap.add_argument('--min-seg', type=float, default=2.0, help='v2.1: shorter frame segments are merged into the previous one (to avoid flicker)')
ap.add_argument('--min-shot', type=float, default=0.3, help='v2.0: shots shorter than this are merged into the previous one (v3.3: 0.3, even a short source shot gets its own frame, not a crop of another one)')
ap.add_argument('--short-shot', type=float, default=2.0, help='v3.3: a source shot shorter than this and not a close-up (face < band-wide) is shown as a band (full frame, blurred top/bottom) instead of a crop: a crop of a wide shot for 1-2 s reads as a jump')
ap.add_argument('--cut-thr', type=float, default=0, help='v2.0: threshold for shot-change detection (0 = auto)')
ap.add_argument('--wide-auto', type=int, default=1, help='v2.0: with a source >= 1440p, WIDE mode turns itself off (a full-height crop is sharp enough anyway)')
ap.add_argument('--no-wide', action='store_true', help='v2.0: turn off WIDE mode (always crop)')
ap.add_argument('--asd', default='', help='v1.5: folder with the output of run_asd_cpu_patched.py (Light-ASD) for the same clip, speaker by real ASD')
ap.add_argument('--splices', default='', help='v1.5: moments of internal splices of the clip (s, comma-separated): a speaker switch within 2 s after a splice is pulled to it')
ap.add_argument('--debug', default='', help='A:B - print AV correlation/motion per person for seconds [A,B]')
ap.add_argument('--det-width', type=int, default=960, help='frame width for face detection (same for 4K and 1080p)')
ap.add_argument('--profile', default='podcast', choices=['podcast', 'solo'], help='parameter set for the source format (see the docstring)')
ap.add_argument('--punch-gap', type=float, default=9.0, help='solo: minimum step between own cuts (s); cuts only at sentence boundaries')
ap.add_argument('--words-per-line', type=int, default=3, help='words per caption line (selection-standard: 2-3)')
ap.add_argument('--crf', type=int, default=16, help='x264 crf of the final encode (15 = maximum, 17 = like ep3 v3.2)')
_argv = sys.argv[1:]
a = ap.parse_args()
# ---- profiles: profile values apply only to parameters NOT given explicitly on the command line ----
PROFILES = {
    'podcast': dict(punch=1.0, hold_max=9.0, band=1, short_shot=1.5, min_crop=1.0, hl='yellow'),
    'solo':    dict(band=0, band_wide=0.0, switch_hold=60.0, min_dwell=60.0, face_frac=0.22, punch=0.76, hl='#C82828', hold_max=9.0),
}
_given = {x.lstrip('-').split('=')[0].replace('-', '_') for x in _argv if x.startswith('--')}
for _k, _v in PROFILES[a.profile].items():
    if _k not in _given: setattr(a, _k, _v)
PUNCH_SENTENCE = (a.profile == 'solo')
OW, OH = 1080, 1920

# ---------------- pass 1: faces, identity, speaker, camera path ----------------
cap = cv2.VideoCapture(a.raw); fps = cap.get(5) or 30; n = int(cap.get(7)); W = int(cap.get(3)); H = int(cap.get(4))
CW = int(round(H * 9 / 16)); CW = min(CW, W)            # crop width at full height
MIN_CROP = a.min_crop if a.min_crop > 0 else (0.62 if H >= 1440 else 1.0)   # auto: from 4K zoom down to 0.62 of the frame (1.4x upscale, a frame like Opus); from 1080p, no extra zoom
AUTO_NOWIDE = (not a.no_wide) and (a.wide_auto and H >= 1440)   # 4K/1440p: a wide shot is cropped too (sharpness is enough)
CACHED = None
if a.path_json and os.path.exists(a.path_json):
    CACHED = json.load(open(a.path_json)); n = CACHED['n']; path = CACHED['path']; cap.release()
if CACHED is None:
  # three scales: mid (<= 1920 wide) - mouth and embeddings; small (det-width) - detection
  MS = min(1.0, 1920.0 / W); MW, MH = int(round(W * MS)), int(round(H * MS)); MINV = 1.0 / MS
  DS = min(1.0, a.det_width / MW); DW, DH = int(round(MW * DS)), int(round(MH * DS)); DINV = 1.0 / DS
  det = cv2.FaceDetectorYN.create(a.yunet, '', (DW, DH), 0.6, 0.3, 200)
  rec = None
  if a.sface and os.path.exists(a.sface):
      rec = cv2.FaceRecognizerSF.create(a.sface, '')
  def iou(a_, b_):
      ax0, ay0, aw, ah = a_; bx0, by0, bw, bh = b_
      ix = max(0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0)); iy = max(0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
      inter = ix * iy; return inter / (aw * ah + bw * bh - inter + 1e-6)
  tracks = {}; ARCH = {}   # id -> dict(box(mid), patch, score, last, embs, last_emb, frames); ARCH - finished tracks
  CUTS = []; _pg = None; DIFFS = []   # v2.0: source shot boundaries (adaptive frame-difference threshold)
  next_id = 0; samples = []   # (frame_idx, {tid: (score, box_mid)})
  fi = 0
  while True:
      ok = cap.grab()
      if not ok: break
      if fi % a.det_every == 0:
          ok, fr = cap.retrieve()
          if not ok: break
          mid = cv2.resize(fr, (MW, MH), interpolation=cv2.INTER_AREA) if MS < 1.0 else fr
          small = cv2.resize(mid, (DW, DH), interpolation=cv2.INTER_AREA) if DS < 1.0 else mid
          _tiny = cv2.cvtColor(cv2.resize(small, (135, 240), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
          DIFFS.append((fi, float(np.mean(np.abs(_tiny - _pg))) if _pg is not None else 0.0))
          _pg = _tiny
          gray = cv2.cvtColor(mid, cv2.COLOR_BGR2GRAY)
          det.setInputSize((DW, DH)); _, faces = det.detect(small)
          faces = [] if faces is None else [f for f in faces if f[2] * f[3] > 0.0012 * DW * DH]
          seen = set()
          for f in faces:
              fm = f.copy(); fm[:14] *= DINV            # coordinates in mid
              box = tuple(float(v) for v in fm[:4]); lm = fm[4:14]
              rmx, rmy, lmx, lmy = lm[6], lm[7], lm[8], lm[9]     # mouth corners
              mcx, mcy = (rmx + lmx) / 2, (rmy + lmy) / 2; mw = max(10, abs(lmx - rmx) * 1.6); mh = max(8, mw * 0.55)
              x0, y0 = int(max(0, mcx - mw / 2)), int(max(0, mcy - mh / 2)); x1, y1 = int(min(MW, mcx + mw / 2)), int(min(MH, mcy + mh / 2))
              patch = cv2.resize(gray[y0:y1, x0:x1], (32, 16)).astype(np.float32) if x1 > x0 + 3 and y1 > y0 + 3 else None
              best, bid = 0.0, None
              for tid, t in tracks.items():
                  v = iou(box, t['box'])
                  if v > best: best, bid = v, tid
              if bid is None or best < 0.25:
                  bid = next_id; next_id += 1
                  tracks[bid] = {'box': box, 'patch': patch, 'score': 0.0, 'last': fi, 'embs': [], 'last_emb': -10**9, 'frames': 0, 'row': fm}
                  motion = 0.0
              else:
                  t = tracks[bid]
                  motion = float(np.mean(np.abs(patch - t['patch']))) if (patch is not None and t['patch'] is not None) else 0.0
                  t['box'] = box; t['patch'] = patch; t['last'] = fi; t['row'] = fm
              t = tracks[bid]; t['score'] = 0.6 * t['score'] + 0.4 * motion; t['motion'] = motion; t['frames'] += 1; seen.add(bid)
              # face embedding: when a track appears and then every ~2 s (up to 6 per track)
              if rec is not None and (fi - t['last_emb']) >= 2 * fps and len(t['embs']) < 6 and box[2] * box[3] > 900:
                  try:
                      al = rec.alignCrop(mid, fm); e = rec.feature(al).flatten(); e = e / (np.linalg.norm(e) + 1e-9)
                      t['embs'].append(e); t['last_emb'] = fi
                  except Exception: pass
          for tid in [k for k, t in tracks.items() if fi - t['last'] > fps * 1.5]: ARCH[tid] = tracks.pop(tid)
          live = {tid: (tracks[tid]['score'], tracks[tid]['box'], tracks[tid].get('motion', 0.0)) for tid in tracks if tid in seen}
          samples.append((fi, live))
      fi += 1
  cap.release(); n = fi
  ARCH.update(tracks)
  # ---- v1.4: identity via SFace: track -> person (greedy clustering, cos >= sface-thr) ----
  person_of = {}; persons = []   # persons[k] = centroid
  for tid in sorted(ARCH, key=lambda k: -ARCH[k]['frames']):
      E = ARCH[tid]['embs']
      if not E: person_of[tid] = None; continue
      c = np.mean(E, axis=0); c = c / (np.linalg.norm(c) + 1e-9)
      best, bk = -1, None
      for k, pc in enumerate(persons):
          v = float(np.dot(c, pc))
          if v > best: best, bk = v, k
      if bk is not None and best >= a.sface_thr:
          person_of[tid] = bk; persons[bk] = persons[bk] * 0.7 + c * 0.3; persons[bk] /= (np.linalg.norm(persons[bk]) + 1e-9)
      else:
          person_of[tid] = len(persons); persons.append(c)
  # tracks without an embedding (small/profile faces) become their own person, to avoid mixing them up
  npers = len(persons)
  for tid in ARCH:
      if person_of.get(tid) is None: person_of[tid] = npers; npers += 1
  # ---- speaker offline at the person level ----
  words_all = json.load(open(a.words, encoding='utf-8')); WORDS_RAW = words_all
  gaps = []
  for i in range(1, len(words_all)):
      pe = words_all[i - 1].get('e', words_all[i - 1]['t'] + 0.2)
      if words_all[i]['t'] - pe >= 0.3: gaps.append((pe + words_all[i]['t']) / 2)
  plive_all = []   # per sample: {pid: (score, box, tid, motion)}
  for (f, live) in samples:
      pl = {}
      for tid, (sc, box, mo) in live.items():
          pid = person_of[tid]
          if pid not in pl or sc > pl[pid][0]: pl[pid] = (sc, box, tid, mo)
      plive_all.append((f, pl))
  # ---- v1.4: audio-visual correlation (light ASD): the speaker mouth motion correlates with speech energy ----
  try:
      raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', a.raw, '-ac', '1', '-ar', '8000', '-f', 's16le', '-'], capture_output=True).stdout
      au = np.abs(np.frombuffer(raw, np.int16).astype(np.float32))
      env = np.array([au[int(f / fps * 8000):int((f + a.det_every) / fps * 8000)].mean() if int((f + a.det_every) / fps * 8000) <= len(au) else 0.0 for (f, _) in plive_all])
  except Exception:
      env = np.zeros(len(plive_all))
  env = (env - env.mean()) / (env.std() + 1e-6)
  pids_all = sorted({pid for _, pl in plive_all for pid in pl})
  M = {pid: np.array([pl[pid][3] if pid in pl else np.nan for _, pl in plive_all]) for pid in pids_all}
  HW = max(3, int(round(0.75 * fps / a.det_every)))   # +-0.75 s
  def avcorr(pid, i):
      m = M[pid][max(0, i - HW):i + HW + 1]; e = env[max(0, i - HW):i + HW + 1]
      ok = ~np.isnan(m)
      if ok.sum() < 5: return 0.0
      m = m[ok]; e = e[ok]
      if m.std() < 1e-6 or e.std() < 1e-6: return 0.0
      return float(np.clip(np.corrcoef(m, e)[0, 1], 0, 1))
  AV = [{pid: avcorr(pid, i) for pid in pl} for i, (_, pl) in enumerate(plive_all)]
  # combined per-sample score: correlation (main) + normalized motion (support)
  mmax = max([v for pid in M for v in M[pid][~np.isnan(M[pid])]] + [1e-6])
  for i, (_, pl) in enumerate(plive_all):
      for pid in list(pl):
          sc, box, tid, mo = pl[pid]
          pl[pid] = (AV[i][pid] + 0.3 * mo / mmax, box, tid, mo)
  MARGIN = 0.15
  # ---- v1.5: real ASD (Light-ASD, run_asd_cpu_patched.py) instead of the proxy, if available ----
  USE_ASD = False
  if a.asd and os.path.exists(os.path.join(a.asd, 'pywork', 'tracks.pckl')) and os.path.exists(os.path.join(a.asd, 'pywork', 'scores.pckl')):
      import pickle
      atr = pickle.load(open(os.path.join(a.asd, 'pywork', 'tracks.pckl'), 'rb'))
      asc = pickle.load(open(os.path.join(a.asd, 'pywork', 'scores.pckl'), 'rb'))
      AW, AFPS = 0, 0
      if os.path.exists(os.path.join(a.asd, 'pyavi', 'video.avi')):
          capA = cv2.VideoCapture(os.path.join(a.asd, 'pyavi', 'video.avi')); AW = capA.get(3) or 0; AFPS = capA.get(5) or 0; capA.release()
      if not AW and os.path.exists(os.path.join(a.asd, 'dims.json')):      # v3.1: pyavi is removed after render, so take the dimensions from dims.json
          _dm = json.load(open(os.path.join(a.asd, 'dims.json'))); AW = float(_dm.get('w', 0)); AFPS = float(_dm.get('fps', 25))
      if not AW:
          print('!!! WARNING: ASD without pyavi/video.avi and without dims.json, using width 1920/25 fps, ASD scores may not match the faces', flush=True); AW = 1920
      AFPS = AFPS or 25
      ASC = MW / AW
      # per ASD-frame: list of (bbox_mid, score)
      per_k = {}
      for tr, sc in zip(atr, asc):
          fr = tr['track']['frame']; bb = tr['track']['bbox']
          for j, k in enumerate(fr):
              if j < len(sc):
                  b = bb[j] * ASC; per_k.setdefault(int(k), []).append(((b[0], b[1], b[2] - b[0], b[3] - b[1]), float(sc[j])))
      def asd_score(box, k):
          best, bs = 0.0, None
          for kk in (k, k - 1, k + 1):
              for bb, sc in per_k.get(kk, []):
                  v = iou(box, bb)
                  if v > best: best, bs = v, sc
          return bs if best >= 0.3 else None
      hit = 0
      for i, (f, pl) in enumerate(plive_all):
          k = int(round(f / fps * AFPS))
          for pid in list(pl):
              sc0, box, tid, mo = pl[pid]
              sc = asd_score(box, k)
              if sc is not None: hit += 1
              pl[pid] = ((sc if sc is not None else -1.5), box, tid, mo)
      USE_ASD = True; MARGIN = 0.4
      print(f'ASD: {len(atr)} tracks, coverage {hit}/{sum(len(pl) for _, pl in plive_all)} faces', flush=True)
  if a.debug:
      d0, d1 = [float(x) for x in a.debug.split(':')]
      for i, (f, pl) in enumerate(plive_all):
          if d0 <= f / fps <= d1:
              print(f'{f/fps:5.2f}s env={env[i]:+.2f} ' + ' '.join(f'P{pid}:sc={pl[pid][0]:.2f},mo={pl[pid][3]:.2f}' for pid in sorted(pl)))
  # leader by sample: in the first 1.5 s of a new shot (all tracks are new) look ahead - total motion over 1.5 s
  lead = []; cur = None; prev_ids = set()
  win = int(round(1.5 * fps / a.det_every))
  for i, (f, pl) in enumerate(plive_all):
      if not pl: lead.append((f, None)); continue
      ids = set(pl)
      new_shot = not (ids & prev_ids); prev_ids = ids
      if new_shot and len(pl) > 1:
          acc = {pid: 0.0 for pid in pl}
          for j in range(i, min(len(plive_all), i + win)):
              for pid, v in plive_all[j][1].items():
                  if pid in acc: acc[pid] += v[0]
          top = max(acc, key=acc.get)
          # if the current speaker is visible and clearly not losing, keep them (continuity across a shot change)
          if cur in acc and acc[top] <= acc[cur] + MARGIN * win: top = cur
      else:
          top = max(pl, key=lambda k: pl[k][0])
          if cur in pl and pl[top][0] <= pl[cur][0] + MARGIN: top = cur   # ASD: compare relatively, even if both are < 0 (small faces in a wide shot)
          elif cur is not None and cur not in pl and len(pl) == 1: top = next(iter(pl))
          elif cur is not None and cur not in pl: top = top
      cur = top; lead.append((f, top))
  for i in range(len(lead)):
      if lead[i][1] is None and i > 0: lead[i] = (lead[i][0], lead[i - 1][1])
  runs = []
  for f, pid in lead:
      if runs and runs[-1][0] == pid: runs[-1][2] = f
      else: runs.append([pid, f, f])
  def merge_short(runs, min_len):
      changed = True
      while changed and len(runs) > 1:
          changed = False
          k = min(range(len(runs)), key=lambda i: runs[i][2] - runs[i][1])
          if runs[k][2] - runs[k][1] < min_len:
              if 0 < k < len(runs) - 1 and runs[k - 1][0] == runs[k + 1][0]:
                  runs[k - 1][2] = runs[k + 1][2]; del runs[k:k + 2]
              elif k == 0: runs[1][1] = runs[0][1]; del runs[0]
              elif k == len(runs) - 1: runs[k - 1][2] = runs[k][2]; del runs[k]
              else:
                  if runs[k - 1][2] - runs[k - 1][1] >= runs[k + 1][2] - runs[k + 1][1]: runs[k - 1][2] = runs[k][2]
                  else: runs[k + 1][1] = runs[k][1]
                  del runs[k]
              changed = True
      return runs
  runs = merge_short(runs, max(a.min_dwell, a.switch_hold) * fps)   # v3.0: interjections shorter than switch-hold do not move the camera
  runs = merge_short(runs, a.min_hold * fps)
  SPL = [float(x) for x in a.splices.split(',') if x.strip()] if a.splices else []
  for i in range(1, len(runs)):
      t = runs[i][1] / fps
      cand = [x for x in SPL if -0.3 <= t - x <= 2.0]      # a splice slightly before the switch -> cut at the splice
      if cand:
          nf = int(round(max(cand) * fps)); runs[i - 1][2] = nf - 1; runs[i][1] = nf; continue
      if a.snap > 0 and gaps:
          g = min(gaps, key=lambda x: abs(x - t))
          if abs(g - t) <= a.snap:
              nf = int(round(g * fps)); runs[i - 1][2] = nf - 1; runs[i][1] = nf
  spk = [None] * n
  for pid, f0, f1 in runs:
      for f in range(max(0, f0), min(n, f1 + 1)): spk[f] = pid
  # camera target: the speaker box in the current shot; if the speaker is not in the shot (a reaction), the largest face
  samples2 = []
  for (f, pl) in plive_all:
      pid = spk[f]
      reaction = False
      if pid in pl: sc, box, tid, _mo = pl[pid]
      elif pl: sc, box, tid, _mo = max(pl.values(), key=lambda v: v[1][2] * v[1][3]); reaction = True   # the speaker is not in the shot: this is a reaction
      else: samples2.append((f, None, None, 0, None, None)); continue
      cx = (box[0] + box[2] / 2) * MINV; cy = (box[1] + box[3] / 2) * MINV; fh = box[3] * MINV   # full-res
      if reaction: fh = H * a.face_frac * 2   # reaction shot: no extra zoom, widest crop (reads as a cutaway, not as a camera mistake)
      samples2.append((f, cx, tid, len(pl), cy, fh))
  samples = samples2
  print(f'persons: {len(persons)} (+{npers - len(persons)} without embedding) | speaker runs: {len(runs)} → ' + ', '.join(f'P{pid}@{f0/fps:.1f}-{f1/fps:.1f}' for pid, f0, f1 in runs), flush=True)
  # ---- v2.1: shot cut = a PEAK in the frame difference (a moving person gives a plateau, a cut is a single spike) ----
  fr_idx = [f for f, _ in DIFFS]; dvals = [d for _, d in DIFFS]
  dv = np.array(dvals[1:]) if len(dvals) > 2 else np.array([0.0])
  med = float(np.median(dv))
  base = a.cut_thr if a.cut_thr > 0 else max(3.0, med * 2.5)
  CUTS = []
  for i in range(1, len(dvals)):
      d = dvals[i]
      nb = max(dvals[i - 1], dvals[i + 1] if i + 1 < len(dvals) else 0.0)
      if d > base and d > a.cut_peak * max(nb, 0.5):
          CUTS.append(fr_idx[i])
  CUTS = [c for i, c in enumerate(CUTS) if i == 0 or c - CUTS[i - 1] > 0.35 * fps]
  print(f'source cuts: {len(CUTS)} (threshold {base:.1f}, peak x{a.cut_peak}, median difference {med:.1f})', flush=True)
  # ---- v2.0: source shots + frame mode for each shot ----
  SM = {f: (cx, tid, cy, fh, nf) for (f, cx, tid, nf, cy, fh) in samples}
  face_by_frame = {f: pl for (f, pl) in plive_all}
  # v3.5.9 (16.09): this podcast has slide/graphics inserts. The frame difference does not always catch them,
  # and then the slide ends up inside a shot with the speaker, and the 9:16 crop cuts the graphic in half (seen on e4c03
  # 42-54 s). So we add a shot boundary where faces disappear (or stay negligible) for longer than 1.2 s.
  NOFACE = []
  _sampled = sorted(face_by_frame)                      # faces are counted only on sampled frames
  _step = max(1, int(np.median(np.diff(_sampled)))) if len(_sampled) > 2 else 1
  # strict: no face of ANY size. A wide studio shot (faces small, but present) -
  # is not a graphic, it goes the usual way (band/crop by the v3.5 rules).
  _noface_f = [f for f in _sampled if not face_by_frame[f]]
  if _noface_f:
      _run0 = _prev = _noface_f[0]
      for _f in _noface_f[1:] + [None]:
          if _f is not None and _f - _prev <= 2 * _step: _prev = _f; continue
          if (_prev - _run0) / fps >= 1.2: NOFACE.append((_run0, _prev + 1))
          if _f is None: break
          _run0 = _prev = _f
  if NOFACE: print(f'inserts without faces: {len(NOFACE)} (' + ', '.join(f'{x/fps:.1f}-{y/fps:.1f}s' for x, y in NOFACE) + ')', flush=True)
  shots = []
  bounds = sorted({0, n} | {c for c in CUTS if 0 < c < n}
                  | {x for pair in NOFACE for x in pair if 0 < x < n})
  for i in range(len(bounds) - 1):
      f0, f1 = bounds[i], bounds[i + 1]
      if shots and (f1 - f0) < a.min_shot * fps: shots[-1][1] = f1; continue   # too short a shot: merge into the previous one
      shots.append([f0, f1])
  if not shots: shots = [[0, n]]
  def shot_of(f):
      for k, (f0, f1) in enumerate(shots):
          if f0 <= f < f1: return k
      return len(shots) - 1
  # for each shot: mode + geometry
  MODES = []
  for (f0, f1) in shots:
      fr_list = [f for f in SM if f0 <= f < f1]
      pid_here = collections.Counter(spk[f] for f in range(f0, min(n, f1)) if spk[f] is not None).most_common(1)
      pid_here = pid_here[0][0] if pid_here else None
      boxes = []          # speaker faces in this shot (mid coordinates)
      allf = []
      for f in fr_list:
          pl = face_by_frame.get(f, {})
          for pid, v in pl.items(): allf.append((pid, v[1]))
          if pid_here in pl: boxes.append(pl[pid_here][1])
      if boxes:
          bx = np.median([b[0] + b[2] / 2 for b in boxes]) * MINV
          by = np.median([b[1] + b[3] / 2 for b in boxes]) * MINV
          bh = np.median([b[3] for b in boxes]) * MINV
      else:
          bx = by = bh = None
      # can everyone fit into 9:16?
      xs = [(b[0] + b[2] / 2) * MINV for _, b in allf]
      spread = (max(xs) - min(xs)) if len(xs) > 1 else 0
      npers_here = len({pid for pid, _ in allf})
      # v3.4 (rule from 06.09, hard): NEVER CROP A WIDE SHOT, regardless of duration.
      # A wide shot is shown whole: a 16:9 band at full width with blurred top/bottom. A crop of a wide shot
      # (speaker tiny, the table and background in frame) followed by a jump to a close-up is the worst artifact.
      # v3.5 (correction from 07.09): a band only for a SHORT wide shot (a quick cutaway, a crop would read
      # as a jump). If the wide shot lasts longer and the speaker is visible, we crop it instead of showing a band.
      sdur = (f1 - f0) / float(fps)
      # v3.5.1 (07.09): a band only if the shot is SHORT, or there are no faces in frame at all.
      # A long wide shot where the speaker was not recognized (bh is None) but faces exist: we crop the largest face,
      # rather than show a band: otherwise 4 s of a far shot and then a jump to a close-up.
      # v3.5.2 (07.09): a far shot is a band (as in Opus), but NOT when it runs into a splice:
      # the scene changes there anyway, and a band next to it reads as a jump, so we crop in that case.
      _t0, _t1 = f0 / float(fps), f1 / float(fps)
      near_splice = any(abs(_t1 - sp) < 0.6 or abs(_t0 - sp) < 0.6 for sp in SPL)
      tiny = (bh is None) or (bh < a.band_wide * H)
      # v3.5.6 (16.09): a band shorter than 1.2 s, or right at the start/end of the clip, is a flash,
      # it reads as a glitch, not as a cutaway. Such a piece is cropped like a long wide shot.
      near_edge = (_t0 < 0.6) or (_t1 > (n / float(fps)) - 0.6)
      if fr_list and not allf and any(x <= f0 and f1 <= y + 1 for x, y in NOFACE):
          # v3.5.11 (16.09): the shot lies entirely in a no-face zone: it is a slide/graphic. Show it as a whole
          # 16:9 frame (a band) regardless of duration: a 9:16 crop cuts a graphic in half.
          # The check is on NOFACE specifically, not simply on an empty allf: a short shot may have
          # no samples at all, and then an empty allf would mean "unknown", not "graphic".
          MODES.append(('WIDE', None, None, None)); continue
      if tiny and (near_splice or near_edge or sdur > 4.5 or sdur < 1.2) and allf:
          _big = max(allf, key=lambda t: t[1][3])[1]
          bx = (_big[0] + _big[2] / 2) * MINV; by = (_big[1] + _big[3] / 2) * MINV; bh = _big[3] * MINV
          short_wide = False
      else:
          short_wide = tiny and 1.2 <= sdur <= 4.5 and not near_splice and not near_edge
      if short_wide:
          MODES.append(('WIDE', None, None, None))
      elif bh is not None and bh >= a.min_face * H and not (npers_here >= 2 and spread > 0.62 * W and bh < 0.16 * H):
          MODES.append(('SPEAKER', bx, by, bh))
      elif a.split and npers_here == 2 and spread > 0.45 * W:
          per = {}
          for pid, b in allf: per.setdefault(pid, []).append(b)
          cs = sorted(((np.median([x[0] + x[2] / 2 for x in v]) * MINV, np.median([x[1] + x[3] / 2 for x in v]) * MINV,
                        np.median([x[3] for x in v]) * MINV) for v in per.values()), key=lambda t: t[0])
          MODES.append(('SPLIT', cs, None, None))
      elif (a.no_wide or AUTO_NOWIDE) and bh is not None:
          MODES.append(('SPEAKER', bx, by, bh))
      elif a.no_wide or AUTO_NOWIDE:
          xs2 = xs if xs else [W / 2]
          MODES.append(('SPEAKER', float(np.median(xs2)), H / 2, H * a.face_frac))
      else:
          MODES.append(('WIDE', None, None, None))
  nsp = sum(1 for m in MODES if m[0] == 'SPEAKER'); nw = sum(1 for m in MODES if m[0] == 'WIDE'); nsl = sum(1 for m in MODES if m[0] == 'SPLIT')
  tw = sum((shots[k][1] - shots[k][0]) for k, m in enumerate(MODES) if m[0] == 'WIDE') / max(1, n)
  print(f'source shots: {len(shots)} ({len(shots)/(n/fps)*60:.1f}/min) | SPEAKER {nsp}, WIDE {nw} ({tw:.0%} of the time), SPLIT {nsl}', flush=True)
  # ---- v2.1: FRAME SEGMENTS = union of source shot boundaries, speaker boundaries and splices ----
  # One static frame per segment. Inside a segment the camera does not move or reframe:
  # a person may lean or gesture, the frame stays. The frame changes only when the source cuts
  # or when another person starts speaking (review 05.09: "there is a wide shot, and you are immediately on the speaker").
  target = [None] * n
  last = None
  for (f, cx, tid, nf, cy, fh) in samples: target[f] = (cx, tid, cy, fh)
  for i in range(n):
      if target[i] is None or target[i][0] is None: target[i] = last if last is not None else (W / 2, None, H / 2, H * a.face_frac)
      else: last = target[i]
  box_by_frame = {}
  for (f, pl) in plive_all: box_by_frame[f] = pl
  SPL_F = sorted({int(round(x * fps)) for x in SPL if 0 < x * fps < n})
  bset = {0, n}
  bset |= {f0 for f0, f1 in shots}
  prev_pid = None
  for f in range(n):
      if spk[f] != prev_pid and spk[f] is not None: bset.add(f); prev_pid = spk[f]
  cut_set = sorted({f0 for f0, f1 in shots})
  SPL_SNAP = []
  for f in SPL_F:
      near = [c for c in cut_set if abs(c - f) <= 0.8 * fps]
      SPL_SNAP.append(min(near, key=lambda c: abs(c - f)) if near else f)   # a splice next to a source cut = one cut
  SPL_F = sorted(set(SPL_SNAP))
  bset |= set(SPL_F)
  # v2.3: the punchline is held as one frame. In a fast exchange (three people finishing a phrase in turns)
  # the camera cannot keep up with the speaker and chops up the finale, so there are no cuts in the last sentence.
  TAIL_CUT = n
  if PUNCH_SENTENCE: a.hold_max = max(a.hold_max, a.punch_gap)
  if a.hold_tail and WORDS_RAW:
      te = max(w.get('e', w['t'] + 0.2) for w in WORDS_RAW)
      tail = te
      for i in range(len(WORDS_RAW) - 1, 0, -1):
          w = WORDS_RAW[i - 1]; nx = WORDS_RAW[i]
          gap = nx['t'] - w.get('e', w['t'] + 0.2)
          if w['w'].rstrip()[-1:] in '.!?…' or gap >= 0.35:
              tail = nx['t']
              if te - tail >= 2.5: break
      tail = min(max(tail, te - 5.0), te - 2.0)   # punchline: the last 2-5 s
      cutoff = int(round(tail * fps)); TAIL_CUT = cutoff
      drop = {x for x in bset if x > cutoff}
      if drop or True:
          bset -= drop
          bset.add(cutoff)          # the start of the final phrase itself is a cut: the punchline gets its own frame
          bset.add(n)               # v3.2: the end of the clip also fell into drop, so the final phrase lived in the frame of the previous segment (shots of 8-12 s)
          print(f'final phrase from {tail:.1f} s: held as one frame ({len(drop) - 1} cuts removed)', flush=True)
  # v2.2: own cuts in long static shots, otherwise the picture sits for 20+ s and the viewer leaves.
  # Reference (6 Opus clips, 74 shots): median 3.4 s, p75 5.0, p90 6.4, almost nothing > 8 s.
  # points where we can cut: a speech pause OR a phrase boundary by punctuation.
  # (YouTube auto-subtitles have almost no pauses, words run together, so punctuation is the main signal here)
  gaps_f = []
  for i in range(1, len(WORDS_RAW)):
      pw = WORDS_RAW[i - 1]; pe = pw.get('e', pw['t'] + 0.2)
      gap = WORDS_RAW[i]['t'] - pe
      if gap >= 0.15 or str(pw.get('w', '')).rstrip()[-1:] in '.,!?…:;':
          gaps_f.append(int(round(((pe + WORDS_RAW[i]['t']) / 2) * fps)))
  def snap_to_pause(f, win=0.7):
      cand = [g for g in gaps_f if abs(g - f) <= win * fps]
      return min(cand, key=lambda g: abs(g - f)) if cand else None
  # v3.2: boundaries between words (even with no pause) as a fallback when there is neither a pause nor punctuation nearby:
  # otherwise the shot sat for 9-12 s (Opus: almost nothing longer than 8 s; its source cuts also fall mid-sentence)
  wb_f = sorted({int(round(((WORDS_RAW[i - 1].get('e', WORDS_RAW[i - 1]['t'] + 0.2) + WORDS_RAW[i]['t']) / 2) * fps)) for i in range(1, len(WORDS_RAW))})
  # solo: cut only at a sentence boundary (punctuation or a pause >= 0.35 s), within a 2.6 s radius; otherwise no cut
  sent_f = []
  for i in range(1, len(WORDS_RAW)):
      pw = WORDS_RAW[i - 1]; pe = pw.get('e', pw['t'] + 0.2)
      if str(pw.get('w', '')).rstrip()[-1:] in '.!?…' or WORDS_RAW[i]['t'] - pe >= 0.35:
          sent_f.append(int(round(((pe + WORDS_RAW[i]['t']) / 2) * fps)))
  def snap_any(f):
      if PUNCH_SENTENCE:
          cand = [g for g in sent_f if abs(g - f) <= 2.6 * fps]
          return min(cand, key=lambda g: abs(g - f)) if cand else None
      g = snap_to_pause(f, 0.7) or snap_to_pause(f, 1.6) or snap_to_pause(f, 2.6)
      if g: return g
      cand = [w for w in wb_f if abs(w - f) <= 1.2 * fps]
      return min(cand, key=lambda w: abs(w - f)) if cand else None
  base_b = sorted(x for x in bset if 0 <= x <= n)
  extra = []
  for i in range(len(base_b) - 1):
      f0, f1 = base_b[i], base_b[i + 1]
      L = (f1 - f0) / fps
      if L <= a.hold_max: continue
      k = int(L // a.hold_max) + 1           # split into equal parts of <= hold_max
      for j in range(1, k):
          f = f0 + int(round((f1 - f0) * j / k))
          g = snap_any(f)
          if g and f0 + a.hold_min * fps < g < f1 - a.hold_min * fps and g <= TAIL_CUT: extra.append(g)
  bset |= set(extra)
  if extra: print(f'own cuts in long shots: {len(extra)} (no angle is held longer than {a.hold_max:.0f} s)', flush=True)
  bnds = sorted(x for x in bset if 0 <= x <= n)
  segs = []
  for i in range(len(bnds) - 1):
      f0, f1 = bnds[i], bnds[i + 1]
      if f1 - f0 < 2: continue
      segs.append([f0, f1])
  def split_long(segs):
      out = []
      for f0, f1 in segs:
          L = (f1 - f0) / fps
          if L <= a.hold_max: out.append([f0, f1]); continue
          k = int(L // a.hold_max) + 1
          pts = [f0]
          for j in range(1, k):
              f = f0 + int(round((f1 - f0) * j / k))
              g = snap_any(f)
              ok_ = lambda x: x and x > pts[-1] + a.min_seg * fps and x < f1 - a.min_seg * fps and x <= TAIL_CUT
              if ok_(g): pts.append(g)
              # v3.5.8 (16.09): a frame cut is not an audio cut. If there is no word boundary nearby and the shot is three times
              # longer than hold-max, we cut anyway: the viewer notices 15 s on one frame, a cut they do not.
              elif (not PUNCH_SENTENCE) and L > 1.5 * a.hold_max and ok_(f): pts.append(f)
          pts.append(f1)
          for i2 in range(len(pts) - 1): out.append([pts[i2], pts[i2 + 1]])
      return out
  # no segment shorter than min_seg, but merge ONLY within one source shot (v3.3):
  # a crop that crosses a source cut shows another camera in another framing ("a wide shot, but cropped").
  # A short shot without a neighbour in the same shot stays a separate segment with its own frame.
  locked = set()
  while len(segs) > 1:
      cands = [i for i in range(len(segs)) if (segs[i][1] - segs[i][0]) < a.min_seg * fps and tuple(segs[i]) not in locked]
      if not cands: break
      k = min(cands, key=lambda i: segs[i][1] - segs[i][0])
      sh_k = shot_of(segs[k][0])
      left = k - 1 if k > 0 and shot_of(segs[k - 1][0]) == sh_k else None
      right = k + 1 if k + 1 < len(segs) and shot_of(segs[k + 1][0]) == sh_k else None
      if left is None and right is None:
          # v3.5.5 (16.09): a stub < 1 s at the clip edge or right next to a splice is flicker.
          # The v3.3 rule (do not merge across a source cut) loses here: the viewer will not notice 0.7 s of another framing,
          # but will definitely notice a 0.7 s flash.
          Lk = (segs[k][1] - segs[k][0]) / fps
          at_edge = (k == 0) or (k == len(segs) - 1) or (segs[k][0] in SPL_F) or (segs[k][1] in SPL_F)
          if Lk < 1.0 and at_edge and len(segs) > 1:
              if k == 0: segs[1][0] = segs[0][0]; del segs[0]
              else: segs[k - 1][1] = segs[k][1]; del segs[k]
              continue
          locked.add(tuple(segs[k])); continue
      if right is None or (left is not None and (segs[left][1] - segs[left][0]) <= (segs[right][1] - segs[right][0])):
          segs[left][1] = segs[k][1]; del segs[k]
      else:
          segs[right][0] = segs[k][0]; del segs[k]
  segs = split_long(segs)                               # after merging, segments that are too long may have appeared
  if not segs: segs = [[0, n]]
  extra = set(extra) | {f0 for f0, f1 in segs}
  # frame of each segment: the median SPEAKER box within the segment; if the speaker is not in frame, keep
  # the previous frame (same shot) or take the largest face
  # v3.5.6 (16.09): collecting speaker boxes per segment is factored out so the scale is computed TWICE:
  # first over all segments (to set one zoom per source shot), then the frames.
  def seg_boxes(f0, f1):
      per = {}
      for f in range(f0, min(n, f1)):
          pl = box_by_frame.get(f)
          if not pl: continue
          for pid2, v in pl.items(): per.setdefault(pid2, []).append((v[0], v[1]))
      pid = None
      if per:
          cnt = collections.Counter(spk[f] for f in range(f0, min(n, f1)) if spk[f] is not None)
          if cnt and cnt.most_common(1)[0][0] in per: pid = cnt.most_common(1)[0][0]
          else:
              scored = {k2: float(np.median([sc for sc, _ in v])) for k2, v in per.items()}
              pid = max(scored, key=scored.get)
      bx = []; by = []; bh = []
      for sc, b in per.get(pid, []):
          bx.append((b[0] + b[2] / 2) * MINV); by.append((b[1] + b[3] / 2) * MINV); bh.append(b[3] * MINV)
      if not bx:
          for f in range(f0, min(n, f1)):
              pl = box_by_frame.get(f)
              if not pl: continue
              b = max(pl.values(), key=lambda v: v[1][2] * v[1][3])[1]
              bx.append((b[0] + b[2] / 2) * MINV); by.append((b[1] + b[3] / 2) * MINV); bh.append(b[3] * MINV)
      return bx, by, bh
  ZOOM_FLOOR = 0.62      # like the auto MIN_CROP for 4K in OPUS_FRAMEWORK: 1.4x upscale, a frame like Opus
  def tch_for(fh_):
      t = min(H, max(MIN_CROP * H, fh_ / a.face_frac))
      if fh_ > 0 and fh_ / t < 0.17: t = min(H, max(ZOOM_FLOOR * H, fh_ / 0.17))
      return t
  # ONE scale per source shot: the median face height over the whole shot (weighted by segment length).
  # The zoom no longer "flips" between neighbouring segments inside one shot: that is what read as
  # "the camera moving closer/away" (correction from 06.09), and at the same time in a long wide
  # shot the speaker does not stay tiny.
  _shot_fh = {}
  for (f0, f1) in segs:
      _b = seg_boxes(f0, f1)[2]
      if not _b: continue
      _shot_fh.setdefault(shot_of(f0), []).extend([float(np.median(_b))] * max(1, f1 - f0))
  SHOT_TCH = {}
  for _k, _v in _shot_fh.items():
      if not _v: continue
      _t = tch_for(float(np.percentile(_v, 10)))     # reference point: the widest moment of the shot, not the average
      _t = max(_t, float(np.max(_v)) / 0.50)         # the largest face of the same shot must fit in the frame
      SHOT_TCH[_k] = float(min(H, _t))
  if _shot_fh:
      print('shot scale: ' + ' | '.join(
          f"{shots[k][0]/fps:.1f}-{shots[k][1]/fps:.1f}s p10={np.percentile(v,10)/H:.0%} max={np.max(v)/H:.0%} crop={SHOT_TCH[k]:.0f}"
          for k, v in sorted(_shot_fh.items()) if k in SHOT_TCH), flush=True)
  CROPS = []; prev_crop = None; prev_punch = False; SEG_FH = []
  for (f0, f1) in segs:
      # v2.2: pick the face by the ASD score inside the segment: more reliable than "the speaker person"
      # (in a wide shot the face is small, the embedding can give another person, and the frame went to the wrong one)
      bx, by, bh = seg_boxes(f0, f1)
      if not bx:
          CROPS.append(prev_crop if prev_crop else (W / 2, H / 2, H, 1.0)); SEG_FH.append(SEG_FH[-1] if SEG_FH else 1.0); continue
      fh_ = float(np.median(bh)); SEG_FH.append(fh_ / H)
      tch = SHOT_TCH.get(shot_of(f0)) or tch_for(fh_)   # v3.5.6: one scale per source shot
      cx_ = float(np.median(bx)); cy_ = float(np.median(by)) + a.face_y * tch
      # v2.2: if the frame barely changed compared to the previous segment (a splice or an own cut in the same
      # shot), alternate "normal <-> punch-in" so that the cut reads as editing, not as a glitch/twitch
      same_as_prev = prev_crop and abs(cx_ - prev_crop[0]) < 0.06 * W and abs(tch - prev_crop[2]) < 0.08 * H
      # v3.5.7 (16.09): on the calm profile (--punch 1.0) the alternation is fully off.
      # Before, the punch-in branch still raised the crop to MIN_CROP*H*0.80 = 1728 px and
      # overrode the zoom on a wide shot, so the face stayed small.
      if a.punch != 1.0 and same_as_prev and (f0 in SPL_F or f0 in extra):
          if prev_punch: tch = min(H, tch / a.punch)
          else: tch = max(MIN_CROP * H * 0.80, tch * a.punch)
          prev_punch = not prev_punch
          cy_ = float(np.median(by)) + a.face_y * tch
      elif not same_as_prev:
          prev_punch = False
      CROPS.append((cx_, cy_, tch, 1.0)); prev_crop = CROPS[-1]
  # v3.0: 1-3 trips to a band frame (blurred top/bottom): that is what Opus does (10 % of the time, 1.5-4.5 s each).
  # v3.1: in Opus a band is a WIDE source shot (faces small, a crop makes no sense), so the candidates are
  # first of all wide shots (speaker face < band-wide of the height), among them those where the visible person is silent (listening);
  # at the end of the clip only if the final shot is wide (a close-up in a band does not read as a "cut to the wide").
  BAND_SEGS = set()
  if a.band > 0 and len(segs) > 2:
      cand = []
      for k2, (f0, f1) in enumerate(segs):
          L = (f1 - f0) / fps
          if L < 1.4: continue
          sc_ = []
          for f in range(f0, min(n, f1), max(1, a.det_every)):
              pl = box_by_frame.get(f)
              if pl: sc_.append(max(v[0] for v in pl.values()))
          quiet = float(np.median(sc_)) if sc_ else 0.0
          wide = (SEG_FH[k2] if k2 < len(SEG_FH) else 1.0) < a.band_wide
          cand.append((k2, L, quiet, wide))
      chosen = []
      last_wide = (SEG_FH[-1] if SEG_FH else 1.0) < a.band_wide
      if a.band_end and len(segs) >= 2 and last_wide: chosen.append(len(segs) - 1)
      # v3.5.10 (16.09): a segment that runs into a splice is not a candidate: a band there
      # gives a double join (frame change + scene change) and reads as a jump.
      def _at_splice(k3):
          t0_, t1_ = segs[k3][0] / fps, segs[k3][1] / fps
          return any(abs(t1_ - sp) < 0.8 or abs(t0_ - sp) < 0.8 for sp in SPL)
      rest = [c for c in cand if c[0] not in chosen and 0 < c[0] < len(segs) - 1 and not _at_splice(c[0])]
      rest.sort(key=lambda c: (not c[3], c[2]))              # wide shots first, among them the quietest
      wide_rest = [c for c in rest if c[3]]
      pool = wide_rest if wide_rest else rest                # no wide shot at all: take the quietest
      for c in pool:                                         # v3.2: neighbouring segments are not taken, otherwise two bands merge into one of 7-9 s
          if len(chosen) >= a.band: break
          if any(abs(c[0] - k3) <= 1 for k3 in chosen): continue
          chosen.append(c[0])
      if not last_wide and a.band_end: print('final shot is a close-up: no band at the end', flush=True)
      # a band lives exactly band-len seconds inside a segment (Opus reference: 1.5-4.5 s), not the whole segment
      L2 = int(round(a.band_len * fps))
      for k2 in chosen:
          f0, f1 = segs[k2]
          if (f1 - f0) - L2 < a.min_seg * fps:
              BAND_SEGS.add((f0, f1))                                  # the segment is short anyway: all of it goes to the band
          else:
              BAND_SEGS.add((f1 - L2, f1))                             # the band runs into the next cut, no stubs left
      if BAND_SEGS: print('band shots (blurred top/bottom): ' + ', '.join(f'{x/fps:.1f}-{y/fps:.1f} s' for x, y in sorted(BAND_SEGS)), flush=True)
  path = []
  si = 0
  for i in range(n):
      while si + 1 < len(segs) and i >= segs[si][1]: si += 1
      k = shot_of(i); mode = MODES[k]
      if any(x <= i < y for x, y in BAND_SEGS) or mode[0] == 'WIDE':
          path.append(['WIDE', 0, 0, 0]); continue
      if mode[0] == 'SPLIT':
          path.append(['SPLIT'] + [list(c) for c in mode[1]]); continue
      cx_, cy_, ch_, pl_ = CROPS[si]; cw_ = ch_ * 9 / 16
      path.append(['SPEAKER', min(max(cx_, cw_ / 2), W - cw_ / 2), min(max(cy_, ch_ / 2), H - ch_ / 2), ch_, pl_])
  nsw = sum(1 for i in range(1, len(segs)))
  print(f'frame segments: {len(segs)} ({len(segs)/(n/fps)*60:.1f}/min), boundaries: source cuts + speaker change + splices', flush=True)
  print(f'faces: tracks={next_id} frames={n} crop={CW}x{H} min-crop={MIN_CROP:.2f}', flush=True)
  if a.path_json: json.dump({'n': n, 'path': path, 'fps': fps, 'W': W, 'H': H, 'CW': CW, 'src_cuts': [round(c / fps, 2) for c in CUTS]}, open(a.path_json, 'w'))
  if a.path_only: sys.exit(0)

# ---------------- captions (same style as render_blurfit_tc.py) ----------------
WORDS = json.load(open(a.words, encoding='utf-8'))
# v1.5: subtitle cleanup: speaker markers >>, tags like [laughter]/[sniffing], pure fillers (e, ee, m), auto-subtitle artifacts ("до$" = dollars)
import re as _re
_FILL = {'е', 'ее', 'еее', 'е-е', 'м', 'мм', 'ем', 'а-а', 'ээ', 'э'}
def _clean(t):
    t = t.replace('&gt;', '>').replace('>>', '').strip()
    t = _re.sub(r'\[[^\]]*\]', '', t).strip()
    t = t.replace('до$', 'доларів').replace('$', ' $')
    return t.strip()
_out = []
for w in WORDS:
    t = _clean(str(w.get('w', '')))
    if not t or t.strip('.,!?…:;«»"()—–-').lower() in _FILL: continue
    w = dict(w); w['w'] = t; _out.append(w)
WORDS = [w for w in _out if w['w'].strip()]
if WORDS and WORDS[-1].get('e', 0) > n / fps + 0.5:
    print(f"!!! WARNING: subtitles run to {WORDS[-1]['e']:.1f} s but the video is only {n/fps:.1f} s: the clip is cut short, the end of the thought is lost", flush=True)
for i in range(len(WORDS)):
    if WORDS[i]['t'] < 0: WORDS[i]['t'] = 0.0
    if i > 0 and WORDS[i]['t'] < WORDS[i - 1]['t']: WORDS[i]['t'] = WORDS[i - 1]['t']
    if 'e' not in WORDS[i] or WORDS[i]['e'] <= WORDS[i]['t']: WORDS[i]['e'] = WORDS[i]['t'] + 0.2
def group(words):
    lines = []; cur = []
    for i, w in enumerate(words):
        cur.append(w); punct = w['w'].rstrip()[-1:] in '.!?...'; gap = (words[i + 1]['t'] - w['e']) if i + 1 < len(words) else 9
        if len(cur) >= a.words_per_line or punct or gap > 0.5 or i == len(words) - 1: lines.append(cur); cur = []
    return lines
LINES = group(WORDS)
LEND = []
for i in range(len(LINES)):
    e_ = LINES[i][-1]['e']
    if i + 1 < len(LINES):
        nt = LINES[i + 1][0]['t']
        LEND.append(nt if nt - e_ < 1.0 else e_ + a.cap_hold)   # v3.1: in a long pause the caption disappears instead of hanging until the next phrase
    else: LEND.append(e_ + 0.35)
MAXW = 940; ROW = int(a.font_max * 1.20); FCACHE = {}
def gf(sz):
    if sz not in FCACHE:
        f = ImageFont.truetype(a.font, sz)
        for _v in (a.weight, 'Bold'):                      # v3.1: ExtraBold as in Opus; if the font has no such instance, Bold
            try: f.set_variation_by_name(_v); break
            except Exception: continue
        FCACHE[sz] = f
    return FCACHE[sz]
def split_rows(words, widths, spf):
    tot = sum(widths) + spf * (len(words) - 1)
    if tot <= MAXW or len(words) < 2: return [list(range(len(words)))]
    best = None
    for k in range(1, len(words)):
        w1 = sum(widths[:k]) + spf * (k - 1); w2 = sum(widths[k:]) + spf * (len(words) - k - 1)
        if best is None or abs(w1 - w2) < best[0]: best = (abs(w1 - w2), k)
    k = best[1]; return [list(range(k)), list(range(k, len(words)))]
def fit(words):
    for sz in range(a.font_max, a.font_max - 26, -2):
        f = gf(sz); spf = f.getlength(' '); widths = [f.getlength(w) for w in words]
        rows = split_rows(words, widths, spf); maxrow = max(sum(widths[i] for i in r) + spf * (len(r) - 1) for r in rows)
        if maxrow <= MAXW: return sz, rows, widths, spf
    sz0 = a.font_max - 26; f = gf(sz0); spf = f.getlength(' '); widths = [f.getlength(w) for w in words]
    return sz0, split_rows(words, widths, spf), widths, spf
if a.caps:
    for ln in LINES:
        for x in ln: x['w'] = x['w'].upper()
LAYOUT = [fit([x['w'] for x in ln]) for ln in LINES]
def _hex(c):
    c = c.lstrip('#'); return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))
HLC = {'yellow': (236, 187, 28), 'green': (60, 200, 70)}.get(a.hl) or (_hex(a.hl) if a.hl.startswith('#') else (236, 187, 28))
YELLOW = HLC; WHITE = (255, 255, 255); BLACK = (0, 0, 0); SUB_Y = a.sub_y
BAND_Y0 = SUB_Y - int(1.9 * ROW); BAND_H = int(4.0 * ROW); OVC = {}
def active_idx(L, t):
    for i, w in enumerate(L):
        is_last = (i == len(L) - 1)
        if (w['t'] <= t <= w['e']) or (is_last and t > w['e']) or (i > 0 and L[i - 1]['e'] < t < w['t']): return i
    return -1
def overlay_for(ci, aidx):
    key = (ci, aidx)
    if key in OVC: return OVC[key]
    L = LINES[ci]; sz, rows, widths, spf = LAYOUT[ci]; f = gf(sz); words = [x['w'] for x in L]
    ys = [SUB_Y] if len(rows) == 1 else [SUB_Y - ROW // 2, SUB_Y + ROW // 2]
    band = Image.new('RGBA', (OW, BAND_H), (0, 0, 0, 0)); dr = ImageDraw.Draw(band); dy = -BAND_Y0
    for ri, r in enumerate(rows):
        tot = sum(widths[i] for i in r) + spf * (len(r) - 1); x = (OW - tot) / 2; y = ys[ri] + dy; xx = x
        for i in r:
            if i == aidx: dr.rounded_rectangle([xx - 15, y - sz * 0.74, xx + widths[i] + 15, y + sz * 0.60], radius=16, fill=YELLOW)
            xx += widths[i] + spf
    for ri, r in enumerate(rows):
        tot = sum(widths[i] for i in r) + spf * (len(r) - 1); x = (OW - tot) / 2; y = ys[ri] + dy; xx = x
        for i in r:
            dr.text((xx, y - sz * 0.60), words[i], font=f, fill=WHITE, stroke_width=max(5, int(sz * 0.09)), stroke_fill=BLACK); xx += widths[i] + spf
    OVC[key] = (band.convert('RGB'), band.split()[3]); return OVC[key]
def draw(img, t):
    ci = None
    for k, ln in enumerate(LINES):
        if ln[0]['t'] - 0.05 <= t < LEND[k]: ci = k; break
    if ci is None: return img
    rgb, mask = overlay_for(ci, active_idx(LINES[ci], t)); img.paste(rgb, (0, BAND_Y0), mask); return img
TITLE_BAND = None
if a.title:
    ttl = a.title.upper()
    def tfont(sz):
        f = ImageFont.truetype(a.font, sz)
        for _v in (a.weight, 'Bold'):
            try: f.set_variation_by_name(_v); break
            except Exception: continue
        return f
    TMAXW = 900; sz = 52
    while sz >= 36:
        f = tfont(sz); words = ttl.split(); lines = []; cur = ''
        for w in words:
            t = (cur + ' ' + w).strip()
            if f.getlength(t) <= TMAXW: cur = t
            else:
                if cur: lines.append(cur)
                cur = w
        if cur: lines.append(cur)
        if len(lines) <= 2 and max(f.getlength(l) for l in lines) <= TMAXW: break
        sz -= 3
    lh = int(sz * 1.28); padx = 30; pady = 22
    boxw = int(max(f.getlength(l) for l in lines)) + 2 * padx; boxh = len(lines) * lh + 2 * pady
    TY = 170; bx0 = (OW - boxw) // 2
    band = Image.new('RGBA', (OW, TY + boxh + 30), (0, 0, 0, 0)); d = ImageDraw.Draw(band)
    d.rounded_rectangle([bx0, TY, bx0 + boxw, TY + boxh], radius=26, fill=(0, 0, 0, 232))
    for i, l in enumerate(lines):
        lw = f.getlength(l); d.text(((OW - lw) / 2, TY + pady + i * lh), l, font=f, fill=(255, 255, 255))
    TITLE_BAND = (band.convert('RGB'), band.split()[3])

# ---------------- pass 2: render ----------------
cap = cv2.VideoCapture(a.raw)
FA, FB = 0, n
if a.frames:
    FA, FB = [int(x) for x in a.frames.split(':')]; FB = min(FB, n)
au = a.out + '.wav'
subprocess.run(['ffmpeg', '-y', '-i', a.raw, '-ss', f'{FA / fps:.4f}', '-t', f'{(FB - FA) / fps:.4f}', '-vn', '-ac', '2', '-ar', '48000', au, '-loglevel', 'panic'], check=True)
vout = subprocess.Popen(['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{OW}x{OH}', '-r', str(fps),
                         '-i', '-', '-i', au, '-c:v', 'libx264', '-preset', 'medium', '-crf', str(a.crf), '-pix_fmt', 'yuv420p', '-c:a', 'alac', '-shortest', a.out, '-loglevel', 'error'], stdin=subprocess.PIPE)
SHARP = a.sharpen if a.sharpen >= 0 else (0.35 if OH > H else 0.0)
WIDE_Y = 0.42     # center of the wide-shot band (Opus puts it slightly above center, captions below it)
ot = FA / fps; fi = 0
while fi < FA:
    if not cap.grab(): break
    fi += 1
while fi < FB:
    ret, frame = cap.read()
    if not ret: break
    P = path[fi]; mode = P[0] if isinstance(P[0], str) else 'SPEAKER'
    def _sharp(im):
        return cv2.addWeighted(im, 1 + SHARP, cv2.GaussianBlur(im, (0, 0), 1.2), -SHARP, 0) if SHARP > 0 else im
    def _crop(cx, cy, ch):
        ch = int(round(ch)); cw = int(round(ch * 9 / 16))
        x0 = max(0, min(W - cw, int(round(cx - cw / 2)))); y0 = max(0, min(H - ch, int(round(cy - ch / 2))))
        c = frame[y0:y0 + ch, x0:x0 + cw]
        return cv2.resize(c, (OW, OH), interpolation=(cv2.INTER_LANCZOS4 if OH > ch else cv2.INTER_AREA))
    if mode == 'WIDE':
        bw = int(round(OH * W / H)); bg = cv2.resize(frame, (bw, OH), interpolation=cv2.INTER_AREA)
        x0 = max(0, (bw - OW) // 2); bg = bg[:, x0:x0 + OW]
        bg = cv2.GaussianBlur(bg, (0, 0), 26); bg = (bg * 0.55).astype(np.uint8)
        bh = int(round(OW * H / W)); band = cv2.resize(frame, (OW, bh), interpolation=cv2.INTER_AREA)
        y0 = int(round(OH * WIDE_Y - bh / 2)); y0 = max(0, min(OH - bh, y0))
        fg = bg.copy(); fg[y0:y0 + bh] = _sharp(band)
    elif mode == 'SPLIT':
        cs = P[1:]; half = OH // 2; fg = np.zeros((OH, OW, 3), np.uint8)
        for j, (bx, by, bh_) in enumerate(cs[:2]):
            ch = min(H, max(0.5 * H, bh_ / (a.face_frac * 1.35))); cw = ch * 9 / 32
            x0 = max(0, min(W - int(cw), int(round(bx - cw / 2)))); y0 = max(0, min(H - int(ch), int(round(by + 0.10 * ch - ch / 2))))
            c = frame[y0:y0 + int(ch), x0:x0 + int(cw)]
            fg[j * half:(j + 1) * half] = _sharp(cv2.resize(c, (OW, half), interpolation=cv2.INTER_LANCZOS4))
    else:
        fg = _sharp(_crop(P[1], P[2], P[3]))
    img = draw(Image.fromarray(cv2.cvtColor(fg, cv2.COLOR_BGR2RGB)), ot)
    if TITLE_BAND is not None and (a.title_sec <= 0 or ot < a.title_sec): img.paste(TITLE_BAND[0], (0, 0), TITLE_BAND[1])
    vout.stdin.write(img.tobytes()); ot += 1.0 / fps; fi += 1
cap.release(); vout.stdin.close(); vout.wait(); os.remove(au); print('OK', a.out, 'frames %d-%d' % (FA, FB))
