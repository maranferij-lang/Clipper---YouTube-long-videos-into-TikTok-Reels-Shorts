#!/usr/bin/env python3
"""
pipeline_v3.py (v3.1, 19.09.2026)  pipeline "4K ranges -> raw -> Light-ASD -> render_opus_tc_v4 -> loudnorm -> QC" for a batch of clips.
Resumable: state in <work>/state.json, stages hq -> cut -> asd -> path -> render -> final -> (qc) -> done | qc_fail.
Call it repeatedly until "ALL DONE" (in a sandbox with a per-call limit <= 180 s use --budget 165; natively on a Mac use --budget 100000).

  python3 pipeline_v3.py jobs.json --work ~/ep5/w --budget 165 [--profile podcast|solo] [--no-asd] [--keep-raw] [--no-qc]

jobs.json: {"video":"<youtube id>","words":"<abs path yt_words.json>","out":"<preview dir>",
            "jobs":[{"name":..,"ranges":[[a,b],..],"title":"..", "fix":{"recieve":"receive"}, "plate":false}]}
  ranges: absolute seconds, exactly on words (start = t of the first word, end = e of the last); several ranges =
          an internal splice (forward only). The +-0.5 s margin for moving boundaries into silence is handled by the pipeline (MARGIN),
          do not build it into the selection: otherwise someone else's word ends up in the range.
  fix:    spelling corrections of the auto-subtitles for this clip (word -> word). Does not change the meaning.
  plate:  title plate at the top for this clip only (off by default, as in Opus).

What each stage does:
  hq     each range is downloaded separately with a +-0.5 s margin (--force-keyframes-at-cuts) in chunks of <= 28 s and joined
         without re-encoding (long ranges would otherwise not fit into one call).
  cut    stubs of another shot at the edges are trimmed to the source cut (only if no word starts there); start and
         end are pulled into silence by RMS (YouTube timecodes lag by ~0.08 s); parts with ALAC (lossless) ->
         raw.mp4; words.json in the clip timeline with the fix corrections applied; meta.json (splices, boundaries).
  asd    Light-ASD on a 1440p copy (run_asd_cpu_q4_v1.2.py, resumable).
  path   camera trajectory (render --path-only) -> path.json + path_info.json.
  render in chunks of RENDER_CH frames (260) into r_*.mp4 with ALAC audio.
  final  join, highpass 85 + loudnorm I=-15 TP=-1.5 LRA=11 in two passes, a single AAC encode; qc/<name>.json and a contact sheet;
         then qccheck_v2: clean -> done (raw and the 4K windows are removed), otherwise qc_fail (everything is kept for a re-render).
After ALL DONE it writes <out>/TITLES_for_descriptions.md (the clip titles table).
"""
import os, sys, json, time, subprocess, math, glob, shutil, argparse

ap = argparse.ArgumentParser()
ap.add_argument('jobs'); ap.add_argument('--budget', type=float, default=150)
ap.add_argument('--work', default=os.path.expanduser('~/ep3/pipe'))
ap.add_argument('--only', default='', help='only these clips (comma-separated)')
ap.add_argument('--asd-res', type=int, default=1440, help='height of the copy used for ASD (2160 = from the 4K frames, 1440 is faster)')
ap.add_argument('--no-asd', action='store_true', help='do not run Light-ASD (solo channel: one person in frame)')
ap.add_argument('--plate', action='store_true', help='title plate at the top for ALL clips (by default only where jobs.json has "plate": true)')
ap.add_argument('--renderer', default='render_opus_tc_v4.py')
ap.add_argument('--asd-script', default='run_asd_cpu_q4_v1.2.py')
ap.add_argument('--qc-script', default='qccheck_v2.py')
ap.add_argument('--profile', default='podcast', choices=['podcast', 'solo'])
ap.add_argument('--keep-raw', action='store_true', help='do not remove raw.mp4 and the 4K windows after done')
ap.add_argument('--no-qc', action='store_true', help='do not run qccheck in final')
ap.add_argument('--chunk', type=int, default=int(os.environ.get('RENDER_CH', 260)), help='frames per render chunk')
a = ap.parse_args()
T0 = time.time()
HOME = os.path.expanduser('~'); TOOLS = os.environ.get('CLIPPER_TOOLS') or os.path.dirname(os.path.abspath(__file__))
ENV = dict(os.environ, PATH=os.path.join(HOME, '.local', 'bin') + ':' + os.environ.get('PATH', ''), PYTHONPATH=os.path.join(HOME, 'Light-ASD'))
J = json.load(open(a.jobs, encoding='utf-8')); VID = J['video']
os.makedirs(a.work, exist_ok=True); os.makedirs(os.path.join(a.work, 'hq'), exist_ok=True); os.makedirs(J['out'], exist_ok=True)
SP = os.path.join(a.work, 'state.json')
S = json.load(open(SP)) if os.path.exists(SP) else {}
def save(): json.dump(S, open(SP, 'w'), indent=1)
def left(): return a.budget - (time.time() - T0)
def sh(cmd, timeout=None, **kw):
    return subprocess.run(cmd, env=ENV, timeout=timeout, **kw)
def log(*x): print(f'[{time.time()-T0:5.0f}s]', *x, flush=True)

# ---------- 4K windows (shared between clips) ----------
MARGIN = 0.5   # v3.1 (19.09): the 4K window is downloaded with a +-0.5 s margin so a boundary can be moved into silence in both directions;
               # the ranges in jobs.json stay exact (on words), the margin is not built into the selection
def windows_for(ranges):
    return [(round(max(0.0, s - MARGIN), 2), round(e + MARGIN, 2)) for s, e in ranges]
def hq_path(s, e): return os.path.join(a.work, 'hq', f'{VID}_{s:.2f}_{e:.2f}.mp4')
def vdur(p):
    try: return float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', p], capture_output=True, text=True).stdout.strip())
    except Exception: return 0.0
HQ_CHUNK = 28.0   # s; a 4K range longer than this does not finish downloading within one call
def _grab(s, e, out, tmo):
    tmp = out[:-4] + '.dl.mp4'
    for f in glob.glob(tmp[:-4] + '*'): os.remove(f)
    for fmt in ('313+140-1', '401+140-1', '313+bestaudio', '401+bestaudio'):
        try:
            sh(['yt-dlp', '-q', '--no-warnings', '-f', fmt, '--download-sections', f'*{s:.2f}-{e:.2f}', '--force-keyframes-at-cuts',
                '--downloader-args', 'ffmpeg_o:-preset ultrafast -crf 14 -threads 2', '--merge-output-format', 'mp4', '-o', tmp,
                f'https://youtu.be/{VID}'], timeout=tmo, capture_output=True, text=True)
        except subprocess.TimeoutExpired: pass
        d = vdur(tmp) if os.path.exists(tmp) else 0.0
        if d and abs(d - (e - s)) < 0.6: os.replace(tmp, out); return True      # whole file of the required length
        for f in glob.glob(tmp[:-4] + '*'): os.remove(f)
    return False
def ensure_hq(s, e):
    p = hq_path(s, e)
    if os.path.exists(p) and os.path.getsize(p) > 1e5: return True
    n = max(1, math.ceil((e - s) / HQ_CHUNK)); step = (e - s) / n
    bits = [(round(s + i * step, 2), round(s + (i + 1) * step, 2) if i < n - 1 else e) for i in range(n)]
    pdir = os.path.join(a.work, 'hq', 'parts'); os.makedirs(pdir, exist_ok=True); paths = []
    for bs, be in bits:
        bp = os.path.join(pdir, f'{VID}_{bs:.2f}_{be:.2f}.mp4'); paths.append(bp)
        if os.path.exists(bp) and os.path.getsize(bp) > 1e5: continue
        if left() < 12 + 1.3 * (be - bs): return False
        log(f'hq {bs:.2f}-{be:.2f} ({be-bs:.0f} c)')
        if not _grab(bs, be, bp, max(25, left() - 6)): log('hq FAIL', bs, be); return False
    if n == 1: os.replace(paths[0], p); return True
    lst = os.path.join(pdir, 'l.txt'); open(lst, 'w').write(''.join(f"file '{x}'\n" for x in paths))
    sh(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy', p], capture_output=True)
    d = vdur(p)
    if abs(d - (e - s)) > 0.8:
        log(f'!!! hq concat {d:.2f} != {e-s:.2f}');
        if os.path.exists(p): os.remove(p)
        return False
    for x in paths: os.remove(x)
    return True

EDGE = 1.6      # v2.1: within how many seconds of the range edge we look for a source cut
def edge_cuts(src, tail=False):
    """source cuts in the first (or last, tail=True) EDGE seconds of the file: list of times from the start of the window"""
    import numpy as np
    W_, H_ = 96, 54
    span = vdur(src); t0 = (span - MARGIN - EDGE) if tail else MARGIN
    cmd = ['ffmpeg', '-v', 'error', '-ss', f'{max(0.0, t0):.3f}', '-t', f'{EDGE}', '-i', src, '-vf', f'scale={W_}:{H_}', '-pix_fmt', 'gray', '-f', 'rawvideo', '-']
    raw = subprocess.run(cmd, capture_output=True).stdout
    n = len(raw) // (W_ * H_)
    if n < 4: return [], n
    fr = np.frombuffer(raw[:n * W_ * H_], np.uint8).reshape(n, H_, W_).astype(np.float32)
    dd = [float(np.mean(np.abs(fr[i] - fr[i - 1]))) for i in range(1, n)]
    med = float(np.median(dd)); base = max(8.0, med * 2.5); cuts = []
    for i in range(len(dd)):
        nb = max(dd[i - 1] if i > 0 else 0.0, dd[i + 1] if i + 1 < len(dd) else 0.0)
        if dd[i] > base and dd[i] > 1.8 * max(nb, 0.5): cuts.append((i + 1) / 30.0)
    return cuts, n

def trim_stubs(job, W):
    """v2.1: if a range boundary is 0.06-1.6 s from a source cut and that stub contains no words,
    move the boundary to the cut (the frame will not show 0.3-1.5 s of another shot, which reads as a jump). Returns the new ranges."""
    out = []
    for i, (s, e) in enumerate(job['ranges']):
        src = hq_path(*windows_for([[s, e]])[0]); s2, e2 = s, e
        starts = [w['t'] for w in W if s <= w['t'] < e]
        cs, n1 = edge_cuts(src, tail=False)
        cs = [c for c in cs if 0.06 <= c <= EDGE]
        if cs:
            c = max(cs)                                     # last cut within the first 1.6 s
            # no words on the stub: shift; if a word "starts" on it but the stub is <= 0.4 s, shift too
            # (I put the range boundary at the word start per the auto-subtitles +-0.2 s, while the editor cuts at the real voice onset)
            if not any(s <= t < s + c + 0.05 for t in starts): s2 = round(s + c, 2)   # v2.2 (07.09): removed "or c <= 0.4":
            # it moved the start even when a word began on the stub, and the word vanished from the audio (while staying in the subtitles)
        ce, n2 = edge_cuts(src, tail=True)
        ce = [EDGE - c for c in ce if 0.06 <= EDGE - c <= EDGE]    # distance from the end of the range (analysis window [e-EDGE, e])
        if ce:
            c = max(ce)                                     # first cut within the last 1.6 s
            cabs = e - c
            before = [t for t in starts if t < cabs]
            if not any(cabs - 0.05 <= t < e for t in starts) and (not before or before[-1] <= cabs - 0.4): e2 = round(cabs - 0.03, 2)
        if (s2, e2) != (s, e): log(f"  stub: r{i} {s:.2f}-{e:.2f} → {s2:.2f}-{e2:.2f} (start +{s2-s:.2f}, end -{e-e2:.2f})")
        out.append([s2, e2])
    return out

def snap_silence(src, ranges0, ranges, W):
    """v3.1: start and end into silence in both directions, within the MARGIN allowance.
    Start: if there is already voice at the boundary, check whether there is silence between the boundary and the first word: yes -> this is the tail of another phrase, start
    forward to the last silence before the word; no -> the word began before the boundary (YouTube timecodes lag), start
    back to the last silence before the boundary (up to 0.35 s). End is the mirror: voice at the boundary and silence after the last word ->
    back to the first silence after the word; voice continuous from the word to the boundary -> forward to the first silence (up to 0.4 s).
    Silence = RMS of a 20 ms window < 12 % of the median speech RMS. Words are not touched: the start never goes past the first word."""
    import numpy as np
    out = []
    for i, ((s0, e0), (s, e)) in enumerate(zip(ranges0, ranges)):
        try:
            w0 = windows_for([[s0, e0]])[0][0]                       # the file starts at w0 = s0 - MARGIN
            raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', src[i], '-ac', '1', '-ar', '16000', '-f', 's16le', '-'], capture_output=True).stdout
            au = np.abs(np.frombuffer(raw, np.int16).astype(np.float32)); win = 320   # 20 ms
            nfr = len(au) // win
            if nfr < 30: out.append([s, e]); continue
            rms = np.sqrt((au[:nfr * win].reshape(nfr, win) ** 2).mean(axis=1))
            speech = np.median(rms[rms > np.percentile(rms, 40)]) if nfr else 0
            thr = 0.12 * max(speech, 1.0)
            K = lambda t: int(round((t - w0) / 0.02))               # episode time -> window index
            wt = [w['t'] for w in W if s0 <= w['t'] < e0]; we = [w.get('e', w['t'] + 0.3) for w in W if s0 <= w['t'] < e0]
            first_w = min(wt) if wt else s0; last_e = max(we) if we else e0
            s2, e2 = s, e
            def last_run(q):     # last continuous quiet run in a list of indices
                r = [q[-1]]
                for k in reversed(q[:-1]):
                    if k == r[0] - 1: r.insert(0, k)
                    else: break
                return r
            def first_run(q):
                r = [q[0]]
                for k in q[1:]:
                    if k == r[-1] + 1: r.append(k)
                    else: break
                return r
            PAD = 4              # how many quiet windows (20 ms each) to leave before / after a word: 80 ms
            # start
            if rms[K(s):K(s) + 3].mean() >= thr:
                q_fwd = [k for k in range(K(s), min(K(first_w - 0.03), nfr)) if rms[k] < thr]
                if q_fwd:                                                            # tail of another phrase -> forward, into the silence before the word
                    r = last_run(q_fwd); s2 = round(w0 + max(r[0], r[-1] - PAD + 1) * 0.02, 2)
                else:
                    q_back = [k for k in range(max(0, K(s - 0.35)), K(s)) if rms[k] < thr]
                    if q_back:                                                       # word began earlier -> back, keeping the silence
                        r = last_run(q_back); s2 = round(w0 + max(r[0], r[-1] - PAD + 1) * 0.02, 2)
            # end
            if rms[max(0, K(e) - 3):K(e)].mean() >= thr:
                q_back2 = [k for k in range(max(0, K(last_e + 0.03)), K(e)) if rms[k] < thr]
                if q_back2:                                                          # another speaker's lead-in -> back, into the silence after the word
                    r = first_run(q_back2); e2 = round(w0 + (min(r[-1], r[0] + PAD - 1) + 1) * 0.02, 2)
                else:
                    q_fwd2 = [k for k in range(K(e), min(K(e + 0.40), nfr)) if rms[k] < thr]
                    if q_fwd2:                                                       # word is still sounding -> forward
                        r = first_run(q_fwd2); e2 = round(w0 + (min(r[-1], r[0] + PAD - 1) + 1) * 0.02, 2)
            if (s2, e2) != (s, e): log(f'  silence: r{i} {s:.2f}-{e:.2f} -> {s2:.2f}-{e2:.2f} (start {s2-s:+.2f}, end {e2-e:+.2f})')
            out.append([s2, e2])
        except Exception as ex:
            log('snap_silence:', ex); out.append([s, e])
    return out

def cut(job):
    name = job['name']; d = os.path.join(a.work, name); os.makedirs(d, exist_ok=True)
    raw = os.path.join(d, 'raw.mp4')
    if os.path.exists(raw): return True
    if left() < 15 + 0.6 * sum(e - s for s, e in job['ranges']): return False
    log('cut', name)
    W = json.load(open(J['words'], encoding='utf-8'))
    ranges0 = [list(r) for r in job['ranges']]
    ranges = trim_stubs(job, W) if not job.get('no_trim') else ranges0
    srcs = [hq_path(*windows_for([[s0, e0]])[0]) for s0, e0 in ranges0]
    ranges = snap_silence(srcs, ranges0, ranges, W) if not job.get('no_trim') else ranges
    parts = []
    for i, ((s0, e0), (s, e)) in enumerate(zip(ranges0, ranges)):
        w0 = windows_for([[s0, e0]])[0][0]; src = hq_path(*windows_for([[s0, e0]])[0]); p = os.path.join(d, f'p{i}.mp4')
        AF = f'aresample=async=1,afade=in:d=0.015,afade=out:st={max(0.0, (e - s) - 0.015):.3f}:d=0.015'   # fade on both edges against clicks
        # the window is wider than the range by MARGIN, so the part is cut exactly at the boundaries with a single re-encode (ultrafast crf 14);
        # audio is ALAC lossless, AAC happens once in final
        sh(['ffmpeg', '-y', '-v', 'error', '-ss', f'{s - w0:.3f}', '-t', f'{e - s:.3f}', '-i', src, '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '14', '-threads', '2',
            '-pix_fmt', 'yuv420p', '-c:a', 'alac', '-ar', '48000', '-af', AF, p], check=True)
        parts.append(p)
    job['ranges'] = ranges
    open(os.path.join(d, 'list.txt'), 'w').write(''.join(f"file '{p}'\n" for p in parts))
    sh(['ffmpeg', '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', os.path.join(d, 'list.txt'), '-c', 'copy', raw], check=True)
    for p in parts: os.remove(p)
    rd = vdur(raw); want = sum(e - s for s, e in job['ranges'])
    if abs(rd - want) > 0.5: log(f'!!! cut: raw {rd:.2f} c, expected {want:.2f} c')
    # words in the clip timeline + splices
    W = json.load(open(J['words'], encoding='utf-8')); out = []; off = 0.0; spl = []; lost = []
    for i, (s, e) in enumerate(job['ranges']):
        s0, e0 = ranges0[i]
        for k, w in enumerate(W):
            if min(s, s0) <= w['t'] < max(e, e0):
                if w['t'] >= e + 0.05 or w['t'] < s - 0.12: lost.append(w['w']); continue   # word outside the new boundaries: not sent to the subtitles
                we = w.get('e', min(W[k + 1]['t'], w['t'] + 1.0) if k + 1 < len(W) else w['t'] + 0.5)
                ww = w['w']
                for bad, good in job.get('fix', {}).items(): ww = ww.replace(bad, good)
                t0 = max(w['t'], s)
                out.append({'t': round(t0 - s + off, 2), 'e': round(max(min(we, e), t0 + 0.15) - s + off, 2), 'w': ww})
        off += e - s
        if i < len(job['ranges']) - 1: spl.append(round(off, 2))
    json.dump(out, open(os.path.join(d, 'words.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    multi = any(('&gt;&gt;' in w['w']) or ('>>' in w['w']) for w in out)
    if lost: log('   words outside the new boundaries (not in subtitles):', ' '.join(lost)[:120])
    json.dump({'splices': spl, 'dur': round(off, 2), 'multi_speaker': multi, 'ranges_used': job['ranges'], 'ranges_orig': ranges0, 'lost_words': lost}, open(os.path.join(d, 'meta.json'), 'w'), ensure_ascii=False)
    log('cut ok', name, f'{off:.1f} s, splices {spl}')
    return True

def asd(job):
    name = job['name']; d = os.path.join(a.work, name); ad = os.path.join(d, 'asd')
    def dims(src_):
        try:
            wh = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=p=0', src_], capture_output=True, text=True).stdout.strip().split(',')
            json.dump({'w': int(wh[0]), 'h': int(wh[1]), 'fps': 25}, open(os.path.join(ad, 'dims.json'), 'w'))
        except Exception as ex: log('dims.json:', ex)
    meta = json.load(open(os.path.join(d, 'meta.json')))
    rj = os.path.join(ad, 'ranges.json'); cur = meta.get('ranges_used', None)
    if os.path.exists(os.path.join(ad, 'pywork', 'scores.pckl')):
        old = json.load(open(rj)) if os.path.exists(rj) else meta.get('ranges_orig')   # without ranges.json the old ASD was computed on the original boundaries
        if cur is not None and old != cur:      # v2.1: ASD was computed on a different timeline (before the stubs were trimmed), recompute
            log('asd: timeline changed (stubs trimmed), recomputing', name); shutil.rmtree(ad, ignore_errors=True)
        else:
            s2 = os.path.join(d, f'raw_{a.asd_res}.mp4') if a.asd_res < 2160 else os.path.join(d, 'raw.mp4')
            if not os.path.exists(os.path.join(ad, 'dims.json')) and os.path.exists(s2): dims(s2)
            return True
    if a.no_asd:
        log('asd skipped (--no-asd)', name); return True
    if left() < 45: return False
    src = os.path.join(d, 'raw.mp4')
    if a.asd_res < 2160:
        src2 = os.path.join(d, f'raw_{a.asd_res}.mp4')
        if not os.path.exists(src2):
            sh(['ffmpeg', '-y', '-v', 'error', '-i', src, '-vf', f'scale=-2:{a.asd_res}:flags=area', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '16', '-c:a', 'copy', src2], check=True)
        src = src2
    dur = json.load(open(os.path.join(d, 'meta.json')))['dur']
    log('asd', name, f'(budget {left():.0f} c)')
    try:
        sh([sys.executable, os.path.join(TOOLS, a.asd_script), '--src', src, '--start', '0', '--dur', str(math.ceil(dur) + 1), '--save', ad,
            '--weight', os.path.join(TOOLS, 'weight', 'pretrain_AVA_CVPR.model'), '--yunet', os.path.join(TOOLS, 'yunet.onnx')],
           timeout=max(20, left() - 8), cwd=os.path.join(HOME, 'Light-ASD'), capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        log('asd: budget exhausted, will continue on the next call'); return False
    ok = os.path.exists(os.path.join(ad, 'pywork', 'scores.pckl'))
    if ok:
        dims(src)   # dimensions of the copy ASD was computed on: pyavi/video.avi is removed after render, and render needs the width
        json.dump(cur, open(rj, 'w'))
    return ok

def render_cmd(job, extra):
    d = os.path.join(a.work, job['name'])
    return [sys.executable, os.environ.get('RENDERER_PATH') or os.path.join(TOOLS, a.renderer), '--raw', os.path.join(d, 'raw.mp4'), '--words', os.path.join(d, 'words.json'),
            '--font', os.path.join(TOOLS, 'Mv.ttf'), '--yunet', os.path.join(TOOLS, 'yunet.onnx'), '--path-json', os.path.join(d, 'path.json'),
            '--profile', a.profile] + __import__('shlex').split(os.environ.get('RENDER_EXTRA', '')) \
           + (['--asd', os.path.join(d, 'asd')] if os.path.exists(os.path.join(d, 'asd', 'pywork', 'scores.pckl')) else []) + extra

def path(job):
    d = os.path.join(a.work, job['name']); pj = os.path.join(d, 'path.json')
    if os.path.exists(pj): return True
    if left() < 45: return False
    spl = json.load(open(os.path.join(d, 'meta.json')))['splices']
    log('path', job['name'])
    r = sh(render_cmd(job, ['--out', os.path.join(d, 'x.mp4'), '--path-only', '--splices', ','.join(str(x) for x in spl)]), capture_output=True, text=True, timeout=max(20, left() - 5))
    info = [l for l in r.stdout.splitlines() if l.startswith(('persons', 'ASD', 'speaker', 'source cuts', 'source shots', 'frame segments', 'band shots', 'inserts', 'final', 'own cuts', 'shot scale', '!!!'))]
    if r.returncode != 0: log('path FAIL', r.stderr[-400:])
    json.dump(info, open(os.path.join(d, 'path_info.json'), 'w'), ensure_ascii=False)
    for l in info: log('  ', l[:160])
    return os.path.exists(pj)

def render(job):
    d = os.path.join(a.work, job['name']); pj = os.path.join(d, 'path.json')
    n = json.load(open(pj))['n']; CH = a.chunk
    chunks = [(k, min(k + CH, n)) for k in range(0, n, CH)]
    for A, B in chunks:
        p = os.path.join(d, f'r_{A}.mp4')
        if os.path.exists(p): continue
        if left() < 12 + 0.105 * (B - A): return False
        log('render', job['name'], f'{A}:{B}')
        tmp = p + '.tmp.mp4'
        try:
            sh(render_cmd(job, ['--out', tmp, '--frames', f'{A}:{B}'] + (['--title', job['title']] if (a.plate or job.get('plate')) and job.get('title') else [])), check=True, timeout=max(20, left() - 3), capture_output=True)
        except subprocess.TimeoutExpired:
            if os.path.exists(tmp): os.remove(tmp)
            log('render: budget exhausted on chunk', f'{A}:{B}'); return False
        os.replace(tmp, p)
    return True

def final(job):
    d = os.path.join(a.work, job['name']); outp = os.path.join(J['out'], job['name'] + '.mp4')
    if left() < 30: return False
    n = json.load(open(os.path.join(d, 'path.json')))['n']; CH = a.chunk
    parts = [os.path.join(d, f'r_{k}.mp4') for k in range(0, n, CH)]
    open(os.path.join(d, 'cat.txt'), 'w').write(''.join(f"file '{p}'\n" for p in parts))
    cat = os.path.join(d, 'cat.mp4')
    sh(['ffmpeg', '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', os.path.join(d, 'cat.txt'), '-c', 'copy', cat], check=True)
    # highpass 85 Hz + loudnorm I=-15 TP=-1.5 LRA=11 (selection-standard), two passes (measure -> linear normalization without "pumping")
    AF = 'highpass=f=85,loudnorm=I=-15:TP=-1.5:LRA=11'
    m = sh(['ffmpeg', '-v', 'info', '-i', cat, '-af', AF + ':print_format=json', '-f', 'null', '-'], capture_output=True, text=True).stderr
    try:
        js = json.loads(m[m.rindex('{'):m.rindex('}') + 1])
        AF2 = AF + f":measured_I={js['input_i']}:measured_TP={js['input_tp']}:measured_LRA={js['input_lra']}:measured_thresh={js['input_thresh']}:offset={js['target_offset']}:linear=true"
    except Exception as ex:
        log('loudnorm: measurement failed, single pass', ex); AF2 = AF
    sh(['ffmpeg', '-y', '-v', 'error', '-i', cat, '-c:v', 'copy', '-af', AF2, '-ar', '48000', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', outp], check=True)
    os.remove(cat)
    # QC: video duration vs subtitles, layout
    dur = json.load(open(os.path.join(d, 'meta.json')))['dur']; ts = [round(dur * k / 11 + 0.3, 2) for k in range(11)]
    try:
        Wd = json.load(open(os.path.join(d, 'words.json'), encoding='utf-8')); wend = max(w.get('e', w['t']) for w in Wd)
        P = json.load(open(os.path.join(d, 'path.json'))); pth = P['path']; fps = P.get('fps', 30)
        nseg = sum(1 for i in range(1, len(pth)) if pth[i][:4] != pth[i - 1][:4]) + 1
        wide = sum(1 for x in pth if x[0] == 'WIDE') / max(1, len(pth))
        vd = vdur(outp)
        qc = {'name': job['name'], 'video_s': round(vd, 2), 'subs_end_s': round(wend, 2), 'ok_len': (vd + 0.3 >= wend), 'segments': nseg,
              'seg_per_min': round(nseg / (vd / 60), 1) if vd else None, 'wide_frac': round(wide, 3), 'info': json.load(open(os.path.join(d, 'path_info.json'))) if os.path.exists(os.path.join(d, 'path_info.json')) else []}
        os.makedirs(os.path.join(J['out'], 'qc'), exist_ok=True)
        json.dump(qc, open(os.path.join(J['out'], 'qc', job['name'] + '.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        if not qc['ok_len']: log(f"!!! {job['name']}: video {vd:.1f} c is shorter than subtitles {wend:.1f} c")
    except Exception as ex: log('qc json:', ex)
    fl = []
    for i, t in enumerate(ts):
        f = os.path.join(d, f'g_{i:02d}.png'); fl.append(f)
        sh(['ffmpeg', '-y', '-v', 'error', '-ss', str(t), '-i', outp, '-frames:v', '1', '-vf', f"scale=216:384,drawtext=text='{t}':x=5:y=5:fontsize=22:fontcolor=yellow:box=1:boxcolor=black@0.6", f])
    os.makedirs(os.path.join(J['out'], 'qc'), exist_ok=True)
    cmd = ['ffmpeg', '-y', '-v', 'error']
    for f in fl: cmd += ['-i', f]
    cmd += ['-filter_complex', ''.join(f'[{i}]' for i in range(len(fl))) + f'hstack={len(fl)}', os.path.join(J['out'], 'qc', job['name'] + '.jpg')]
    sh(cmd)
    # clean up the heavy files
    for f in glob.glob(os.path.join(d, 'r_*.mp4')) + glob.glob(os.path.join(d, 'g_*.png')) + glob.glob(os.path.join(d, 'raw_*.mp4')): os.remove(f)
    shutil.rmtree(os.path.join(d, 'asd', 'pyframes'), ignore_errors=True); shutil.rmtree(os.path.join(d, 'asd', 'pycrop'), ignore_errors=True)
    shutil.rmtree(os.path.join(d, 'asd', 'pyavi'), ignore_errors=True)
    log('FINAL', outp, f'{os.path.getsize(outp)/1e6:.0f} MB')
    return True

def qc(job):
    """qccheck_v2 on a finished clip: clean -> True (done), problems -> state qc_fail, report in qc/<name>.qc.txt"""
    if a.no_qc: return True
    if left() < 60: return False
    d = os.path.join(a.work, job['name'])
    r = sh([sys.executable, os.path.join(TOOLS, a.qc_script), '--work', a.work, '--out', J['out'], '--name', job['name'],
            '--yunet', os.path.join(TOOLS, 'yunet.onnx')], capture_output=True, text=True, timeout=max(30, left() - 5))
    txt = '\n'.join(l for l in (r.stdout + r.stderr).splitlines() if 'WARN:0@' not in l and l.strip())
    os.makedirs(os.path.join(J['out'], 'qc'), exist_ok=True)
    open(os.path.join(J['out'], 'qc', job['name'] + '.qc.txt'), 'w', encoding='utf-8').write(txt)
    bad = ('❌' in txt) or (r.returncode != 0)
    for l in txt.splitlines():
        if l.strip(): log('  ', l[:160])
    if bad:
        S[job['name']] = 'qc_fail'; save(); log('QC FAIL', job['name'], '- raw and 4K are kept for a re-render'); return 'fail'
    return True

STAGES = [('hq', None), ('cut', cut), ('asd', asd), ('path', path), ('render', render), ('final', final), ('qc', qc)]
only = set(a.only.split(',')) if a.only else None
jobs = [j for j in J['jobs'] if not only or j['name'] in only]
for job in jobs:
    st = S.get(job['name'], 'hq')
    if st in ('done', 'qc_fail'): continue
    ok = True
    while ok and st != 'done':
        if st == 'hq':
            ok = all(ensure_hq(s, e) for s, e in windows_for(job['ranges']))
        else:
            ok = dict(STAGES)[st](job)
        if ok == 'fail': break            # qc_fail: state already saved, move on to the next clip
        if ok:
            st = STAGES[[s for s, _ in STAGES].index(st) + 1][0] if st != 'qc' else 'done'
            S[job['name']] = st; save()
    if ok == 'fail': continue
    if not ok:
        log('STOP at', job['name'], st, f'(left {left():.0f} c)'); break
    # done with clean QC: remove the heavy files (4K windows only if no other clip still needs them)
    if not a.keep_raw:
        used = {hq_path(*w) for j2 in jobs if S.get(j2['name'], 'hq') not in ('done',) and j2 is not job for w in windows_for(j2['ranges'])}
        for w in windows_for(job['ranges']):
            p = hq_path(*w)
            if p not in used and os.path.exists(p): os.remove(p)
        for f in glob.glob(os.path.join(a.work, job['name'], 'raw*.mp4')) + glob.glob(os.path.join(a.work, job['name'], 'x.mp4')): os.remove(f)
if all(S.get(j['name']) in ('done', 'qc_fail') for j in jobs):
    names = [j['name'] for j in jobs if S.get(j['name']) == 'done']
    if names:
        open(os.path.join(J['out'], 'TITLES_for_descriptions.md'), 'w', encoding='utf-8').write(
            '| File | Shorts description title | Length, s |\n|---|---|---|\n' +
            '\n'.join(f"| {j['name']}.mp4 | {j.get('title', '')} | {sum(e - s for s, e in j['ranges']):.0f} |" for j in jobs if j['name'] in names) + '\n')
    fails = [j['name'] for j in jobs if S.get(j['name']) == 'qc_fail']
    print('ALL DONE' + (f'  (qc_fail: {", ".join(fails)})' if fails else ''))
print('state:', {j['name']: S.get(j['name'], 'hq') for j in jobs})
