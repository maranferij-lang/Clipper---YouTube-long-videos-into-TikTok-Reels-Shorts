"""
render_opus_tc_v4.py  (19.09.2026)  рендер 9:16 «як Opus Clip» для Clipper.
Історія версій: _tools/CHANGELOG.md. Це v3.5.11 без мертвого коду, з профілями і без змін у логіці кадру.

ЯК ПРАЦЮЄ (коротко, щоб не читати код)
 1. Плани джерела: різ = пік різниці кадрів; додатково межа там, де облич немає довше 1.2 с (слайд, графіка).
    Усередині плану камера стоїть. Перемикання спікера лише на межі плану або в паузі (Light-ASD через --asd).
 2. Режим на план:  слайд без облич -> WIDE (весь кадр 16:9 смугою на розмитому фоні) на всю довжину;
    короткий загальний (1.2-4.5 с, обличчя < --band-wide) -> WIDE, але не впритул до склейки/краю кліпу;
    довгий загальний -> SPEAKER (кроп на спікера); крупний -> SPEAKER кроп повної висоти.
 3. Масштаб кропу один на план джерела: 10-й перцентиль висоти обличчя по плану, дозум так, щоб обличчя було
    >= 17 % кропу, підлога 0.62 висоти кадру (апскейл <= 1.4x), стеля: найбільше обличчя плану <= 50 % кропу.
 4. Власні різи: сегмент довший за --hold-max ріжеться в паузу/пунктуацію; якщо їх немає і сегмент > 1.5*hold-max,
    між словами. Фінальна фраза (2-5 с) одним кадром. Сегменти < --min-seg зливаються в межах плану; огризок < 1 с
    на краю кліпу чи біля склейки зливається навіть через різ джерела. Одна смуга-«видих» (--band 1) на найтихішому
    широкому сегменті, не біля склейки.
 5. Підписи: ВЕЛИКИМИ, Montserrat ExtraBold, до 3 слів у рядку, до 2 рядків, пігулка на активному слові,
    блок на --sub-y; у паузі > 1 с підпис зникає через --cap-hold. Плашка вгорі лише з --title (покліпово).
 6. Звук не чіпаємо: highpass/loudnorm робить pipeline у стадії final.

ПРОФІЛІ (--profile; параметр, заданий явно, має пріоритет над профілем)
  podcast (дефолт): мультикамера, троє в кадрі. punch 1.0 (без підрізу), hold-max 9, band 1, short-shot 1.5,
                    min-crop 1.0 (кроп повної висоти, апскейл лише у широкому плані за п. 3), hl yellow.
  solo:             одна людина, камера не ріже. band 0, band-wide 0, switch-hold 60, min-dwell 60, face-frac 0.22,
                    punch 0.76, різи лише на межах речень з кроком >= --punch-gap (9 с), hl #C82828.

ЗАПУСК (зазвичай через pipeline):
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
ap.add_argument('--face-frac', type=float, default=0.29, help='цільова висота обличчя / висота кропу (зум у широких планах). v3.1: 0.29 — в Opus обличчя 0.25–0.30 висоти кадру (було 0.32)')
ap.add_argument('--min-crop', type=float, default=-1, help='мінімальна висота кропу як частка кадру; -1 = авто (4K → до 1080 px, 1080p → 1.0 без дозуму); 0.6 = дозум за face-frac')
ap.add_argument('--sface', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face_recognition_sface_2021dec.onnx'), help='v1.4: модель SFace для ідентичності людей між планами')
ap.add_argument('--sface-thr', type=float, default=0.45, help='v1.4: поріг cos-схожості «та сама людина»')
ap.add_argument('--path-json', default='', help='кеш траєкторії камери: якщо файл є — пропускаємо детекцію')
ap.add_argument('--frames', default='', help='A:B — рендерити лише кадри [A,B) (для рендеру частинами)')
ap.add_argument('--path-only', action='store_true', help='лише порахувати траєкторію і вийти')
ap.add_argument('--switch-hold', type=float, default=2.6, help='v3.0: камера переходить на іншу людину, лише якщо вона веде ≥ стільки секунд. Коротші вкидання ігноруємо — Opus так робить (видимий = мовець лише 62 %% часу)')
ap.add_argument('--band', type=int, default=2, help='v3.0: скільки разів за кліп виходити на кадр-смугу з розмитим верхом/низом (референс Opus: 1–3 рази, 10 %% часу)')
ap.add_argument('--band-len', type=float, default=3.0, help='v3.0: цільова тривалість такого кадру (референс 1.5–4.5 с)')
ap.add_argument('--band-end', type=int, default=1, help='v3.0: один із виходів на смугу — у фіналі кліпу')
ap.add_argument('--min-dwell', type=float, default=1.6, help='v1.3: новий спікер має вести ≥ стільки секунд, інакше камера не перемикається (вигуки «звичайно», мікрозвуки)')
ap.add_argument('--min-hold', type=float, default=2.5, help='v1.3: після перемикання тримаємо кадр ≥ стільки секунд')
ap.add_argument('--snap', type=float, default=0.6, help='v1.3: підтягувати момент перемикання до найближчої паузи в мовленні (±с, 0 = вимкнути)')
ap.add_argument('--sharpen', type=float, default=-1, help='v1.3: unsharp після масштабування (0..1); -1 = авто (0.35 при апскейлі, 0 при даунскейлі з 4K)')
ap.add_argument('--title-sec', type=float, default=0, help='v1.3: скільки секунд тримати плашку-хук (0 = увесь кліп, як у ep2)')
ap.add_argument('--caps', type=int, default=1, help='v2.0: субтитри ВЕЛИКИМИ літерами (як Opus)')
ap.add_argument('--hl', default='yellow', help='v2.0: колір підсвітки активного слова: yellow | green')
ap.add_argument('--weight', default='ExtraBold', help='v3.1: named instance змінного шрифту для підписів і плашки (ExtraBold | Bold | Black)')
ap.add_argument('--band-wide', type=float, default=0.22, help='v3.1: план вважається широким (кандидат на кадр-смугу), якщо обличчя мовця менше за цю частку висоти джерела')
ap.add_argument('--cap-hold', type=float, default=0.6, help='v3.1: скільки секунд підпис лишається після кінця фрази, якщо наступна фраза не поруч (Opus: підпис не висить у паузах)')
ap.add_argument('--font-max', type=int, default=72, help='v2.0: максимальний кегль субтитрів (Opus ~64-72 px)')
ap.add_argument('--face-y', type=float, default=0.19, help='v2.4: зсув кропу вниз → обличчя вище в кадрі; 0.19 дає центр обличчя на 0.31 висоти (як в Opus), 0.10 було на 0.40')
ap.add_argument('--min-face', type=float, default=0.11, help='v2.0: обличчя спікера менше за цю частку висоти → режим WIDE')
ap.add_argument('--split', action='store_true', help='v2.0: дозволити режим SPLIT (двоє один над одним)')
ap.add_argument('--hold-max', type=float, default=5.5, help='v2.2: якщо один ракурс тримається довше — вставляємо власний різ на паузі в мовленні (референс Opus: 90 %% витримок ≤ 6.4 с)')
ap.add_argument('--hold-min', type=float, default=2.0, help='v2.2: коротше за це ракурс не тримаємо (референс: 19 %% витримок < 1.5 с — це різи самого джерела, свої такими не робимо)')
ap.add_argument('--punch', type=float, default=0.84, help='v2.2: наскільки тісніший «підріз» при власному різі (0.84 = на 16 %% крупніше)')
ap.add_argument('--cut-peak', type=float, default=1.8, help='v2.1: у скільки разів різниця має перевищувати сусідні, щоб вважатись різом (пік, а не рух)')
ap.add_argument('--hold-tail', type=int, default=1, help='v2.3: не різати під час фінальної фрази (панчлайн тримаємо одним кадром)')
ap.add_argument('--min-seg', type=float, default=2.0, help='v2.1: коротші сегменти кадру приліплюються до попереднього (щоб не блимало)')
ap.add_argument('--min-shot', type=float, default=0.3, help='v2.0: плани коротші за це приліплюються до попереднього (v3.3: 0.3 — навіть короткий план джерела має власний кадр, а не чужий кроп)')
ap.add_argument('--short-shot', type=float, default=2.0, help='v3.3: план джерела коротший за це і не крупний (обличчя < band-wide) → показуємо смугою (весь кадр, розмитий верх/низ), а не кропом: кроп широкого плану на 1–2 с читається як стрибок')
ap.add_argument('--cut-thr', type=float, default=0, help='v2.0: поріг детекції зміни плану (0 = авто)')
ap.add_argument('--wide-auto', type=int, default=1, help='v2.0: при джерелі ≥1440p режим WIDE вимикається сам (кроп повної висоти й так різкий)')
ap.add_argument('--no-wide', action='store_true', help='v2.0: вимкнути режим WIDE (завжди кроп)')
ap.add_argument('--asd', default='', help='v1.5: тека з результатом run_asd_cpu_patched.py (Light-ASD) для цього ж кліпу — спікер за справжнім ASD')
ap.add_argument('--splices', default='', help='v1.5: моменти внутрішніх склейок кліпу (с, через кому): перемикання спікера в межах 2 с після склейки притягується до неї')
ap.add_argument('--debug', default='', help='A:B — вивести AV-кореляцію/рух по персонах у секундах [A,B]')
ap.add_argument('--det-width', type=int, default=960, help='ширина кадру для детекції облич (4K і 1080p однаково)')
ap.add_argument('--profile', default='podcast', choices=['podcast', 'solo'], help='набір параметрів під формат джерела (див. докстрінг)')
ap.add_argument('--punch-gap', type=float, default=9.0, help='solo: мінімальний крок між власними різами (с); різи лише на межах речень')
ap.add_argument('--words-per-line', type=int, default=3, help='скільки слів у рядку підпису (selection-standard: 2-3)')
ap.add_argument('--crf', type=int, default=16, help='x264 crf фінального кодування (15 = максимум, 17 = як ep3 v3.2)')
_argv = sys.argv[1:]
a = ap.parse_args()
# ---- профілі: значення профілю застосовуються лише до параметрів, яких НЕ задано явно в командному рядку ----
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
CW = int(round(H * 9 / 16)); CW = min(CW, W)            # ширина кропу при повній висоті
MIN_CROP = a.min_crop if a.min_crop > 0 else (0.62 if H >= 1440 else 1.0)   # auto: з 4K дозум до 0.62 кадру (апскейл 1.4×, кадр як в Opus); з 1080p — без дозуму
AUTO_NOWIDE = (not a.no_wide) and (a.wide_auto and H >= 1440)   # 4K/1440p: широкий план теж кропимо (різкості вистачає)
CACHED = None
if a.path_json and os.path.exists(a.path_json):
    CACHED = json.load(open(a.path_json)); n = CACHED['n']; path = CACHED['path']; cap.release()
if CACHED is None:
  # три масштаби: mid (≤1920 завширшки) — рот і ембеддинги; small (det-width) — детекція
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
  tracks = {}; ARCH = {}   # id -> dict(box(mid), patch, score, last, embs, last_emb, frames); ARCH — завершені треки
  CUTS = []; _pg = None; DIFFS = []   # v2.0: межі планів джерела (адаптивний поріг кадрової різниці)
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
              fm = f.copy(); fm[:14] *= DINV            # координати в mid
              box = tuple(float(v) for v in fm[:4]); lm = fm[4:14]
              rmx, rmy, lmx, lmy = lm[6], lm[7], lm[8], lm[9]     # кути рота
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
              # ембеддинг особи: на появі треку і далі кожні ~2 с (до 6 на трек)
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
  # ---- v1.4: ідентичність через SFace: трек → персона (жадібна кластеризація, cos ≥ sface-thr) ----
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
  # треки без ембеддингу (малі/бічні обличчя) — власна персона, щоб не плутати
  npers = len(persons)
  for tid in ARCH:
      if person_of.get(tid) is None: person_of[tid] = npers; npers += 1
  # ---- спікер офлайн на рівні персон ----
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
  # ---- v1.4: аудіо-візуальна кореляція (легкий ASD): рух рота спікера корелює з енергією мови ----
  try:
      raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', a.raw, '-ac', '1', '-ar', '8000', '-f', 's16le', '-'], capture_output=True).stdout
      au = np.abs(np.frombuffer(raw, np.int16).astype(np.float32))
      env = np.array([au[int(f / fps * 8000):int((f + a.det_every) / fps * 8000)].mean() if int((f + a.det_every) / fps * 8000) <= len(au) else 0.0 for (f, _) in plive_all])
  except Exception:
      env = np.zeros(len(plive_all))
  env = (env - env.mean()) / (env.std() + 1e-6)
  pids_all = sorted({pid for _, pl in plive_all for pid in pl})
  M = {pid: np.array([pl[pid][3] if pid in pl else np.nan for _, pl in plive_all]) for pid in pids_all}
  HW = max(3, int(round(0.75 * fps / a.det_every)))   # ±0.75 с
  def avcorr(pid, i):
      m = M[pid][max(0, i - HW):i + HW + 1]; e = env[max(0, i - HW):i + HW + 1]
      ok = ~np.isnan(m)
      if ok.sum() < 5: return 0.0
      m = m[ok]; e = e[ok]
      if m.std() < 1e-6 or e.std() < 1e-6: return 0.0
      return float(np.clip(np.corrcoef(m, e)[0, 1], 0, 1))
  AV = [{pid: avcorr(pid, i) for pid in pl} for i, (_, pl) in enumerate(plive_all)]
  # combined per-sample score: кореляція (головне) + нормований рух (підтримка)
  mmax = max([v for pid in M for v in M[pid][~np.isnan(M[pid])]] + [1e-6])
  for i, (_, pl) in enumerate(plive_all):
      for pid in list(pl):
          sc, box, tid, mo = pl[pid]
          pl[pid] = (AV[i][pid] + 0.3 * mo / mmax, box, tid, mo)
  MARGIN = 0.15
  # ---- v1.5: справжній ASD (Light-ASD, run_asd_cpu_patched.py) замість проксі, якщо є ----
  USE_ASD = False
  if a.asd and os.path.exists(os.path.join(a.asd, 'pywork', 'tracks.pckl')) and os.path.exists(os.path.join(a.asd, 'pywork', 'scores.pckl')):
      import pickle
      atr = pickle.load(open(os.path.join(a.asd, 'pywork', 'tracks.pckl'), 'rb'))
      asc = pickle.load(open(os.path.join(a.asd, 'pywork', 'scores.pckl'), 'rb'))
      AW, AFPS = 0, 0
      if os.path.exists(os.path.join(a.asd, 'pyavi', 'video.avi')):
          capA = cv2.VideoCapture(os.path.join(a.asd, 'pyavi', 'video.avi')); AW = capA.get(3) or 0; AFPS = capA.get(5) or 0; capA.release()
      if not AW and os.path.exists(os.path.join(a.asd, 'dims.json')):      # v3.1: pyavi прибрано після рендеру — розміри з dims.json
          _dm = json.load(open(os.path.join(a.asd, 'dims.json'))); AW = float(_dm.get('w', 0)); AFPS = float(_dm.get('fps', 25))
      if not AW:
          print('!!! УВАГА: ASD без pyavi/video.avi і без dims.json — беру ширину 1920/25 fps, оцінки ASD можуть не збігтися з обличчями', flush=True); AW = 1920
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
      print(f'ASD: {len(atr)} треків, покриття {hit}/{sum(len(pl) for _, pl in plive_all)} облич', flush=True)
  if a.debug:
      d0, d1 = [float(x) for x in a.debug.split(':')]
      for i, (f, pl) in enumerate(plive_all):
          if d0 <= f / fps <= d1:
              print(f'{f/fps:5.2f}s env={env[i]:+.2f} ' + ' '.join(f'P{pid}:sc={pl[pid][0]:.2f},mo={pl[pid][3]:.2f}' for pid in sorted(pl)))
  # лідер за вибіркою: у перші 1.5 с нового плану (усі треки нові) дивимося вперед — сумарний рух за 1.5 с
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
          # якщо поточний спікер видимий і не програє явно — лишаємо його (тяглість через зміну плану)
          if cur in acc and acc[top] <= acc[cur] + MARGIN * win: top = cur
      else:
          top = max(pl, key=lambda k: pl[k][0])
          if cur in pl and pl[top][0] <= pl[cur][0] + MARGIN: top = cur   # ASD: порівнюємо відносно, навіть якщо обидва < 0 (малі обличчя в широкому плані)
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
  runs = merge_short(runs, max(a.min_dwell, a.switch_hold) * fps)   # v3.0: вкидання коротші за switch-hold камеру не рухають
  runs = merge_short(runs, a.min_hold * fps)
  SPL = [float(x) for x in a.splices.split(',') if x.strip()] if a.splices else []
  for i in range(1, len(runs)):
      t = runs[i][1] / fps
      cand = [x for x in SPL if -0.3 <= t - x <= 2.0]      # склейка трохи раніше за перемикання → ріжемо на склейці
      if cand:
          nf = int(round(max(cand) * fps)); runs[i - 1][2] = nf - 1; runs[i][1] = nf; continue
      if a.snap > 0 and gaps:
          g = min(gaps, key=lambda x: abs(x - t))
          if abs(g - t) <= a.snap:
              nf = int(round(g * fps)); runs[i - 1][2] = nf - 1; runs[i][1] = nf
  spk = [None] * n
  for pid, f0, f1 in runs:
      for f in range(max(0, f0), min(n, f1 + 1)): spk[f] = pid
  # ціль камери: бокс спікера в поточному плані; якщо спікера в плані нема (реакція) — найбільше обличчя
  samples2 = []
  for (f, pl) in plive_all:
      pid = spk[f]
      reaction = False
      if pid in pl: sc, box, tid, _mo = pl[pid]
      elif pl: sc, box, tid, _mo = max(pl.values(), key=lambda v: v[1][2] * v[1][3]); reaction = True   # спікера в плані нема — це реакція
      else: samples2.append((f, None, None, 0, None, None)); continue
      cx = (box[0] + box[2] / 2) * MINV; cy = (box[1] + box[3] / 2) * MINV; fh = box[3] * MINV   # full-res
      if reaction: fh = H * a.face_frac * 2   # реакційний план — без дозуму, найширший кроп (читається як cutaway, а не як помилка камери)
      samples2.append((f, cx, tid, len(pl), cy, fh))
  samples = samples2
  print(f'persons: {len(persons)} (+{npers - len(persons)} без ембеддингу) | speaker runs: {len(runs)} → ' + ', '.join(f'P{pid}@{f0/fps:.1f}-{f1/fps:.1f}' for pid, f0, f1 in runs), flush=True)
  # ---- v2.1: різ плану = ПІК у різниці кадрів (рух людини дає плато, різ — одиничний сплеск) ----
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
  print(f'різи джерела: {len(CUTS)} (поріг {base:.1f}, пік ×{a.cut_peak}, медіанна різниця {med:.1f})', flush=True)
  # ---- v2.0: плани джерела + режим кадру на кожен план ----
  SM = {f: (cx, tid, cy, fh, nf) for (f, cx, tid, nf, cy, fh) in samples}
  face_by_frame = {f: pl for (f, pl) in plive_all}
  # v3.5.9 (16.09): у цьому подкасті є вставки слайдів/графіки. Різниця кадрів їх не завжди ловить,
  # і тоді слайд опиняється всередині плану зі спікером — кроп 9:16 ріже графік навпіл (видно на e4c03
  # 42-54 с). Тому додаємо межу плану там, де обличчя зникають (або лишаються мізерні) довше за 1.2 с.
  NOFACE = []
  _sampled = sorted(face_by_frame)                      # обличчя рахуються лише на семплованих кадрах
  _step = max(1, int(np.median(np.diff(_sampled)))) if len(_sampled) > 2 else 1
  # строго: жодного обличчя БУДЬ-ЯКОГО розміру. Загальний план студії (обличчя дрібні, але є) —
  # це не графіка, він іде звичайним шляхом (смуга/кроп за правилами v3.5).
  _noface_f = [f for f in _sampled if not face_by_frame[f]]
  if _noface_f:
      _run0 = _prev = _noface_f[0]
      for _f in _noface_f[1:] + [None]:
          if _f is not None and _f - _prev <= 2 * _step: _prev = _f; continue
          if (_prev - _run0) / fps >= 1.2: NOFACE.append((_run0, _prev + 1))
          if _f is None: break
          _run0 = _prev = _f
  if NOFACE: print(f'вставки без облич: {len(NOFACE)} (' + ', '.join(f'{x/fps:.1f}-{y/fps:.1f}с' for x, y in NOFACE) + ')', flush=True)
  shots = []
  bounds = sorted({0, n} | {c for c in CUTS if 0 < c < n}
                  | {x for pair in NOFACE for x in pair if 0 < x < n})
  for i in range(len(bounds) - 1):
      f0, f1 = bounds[i], bounds[i + 1]
      if shots and (f1 - f0) < a.min_shot * fps: shots[-1][1] = f1; continue   # надто короткий план — приліплюємо до попереднього
      shots.append([f0, f1])
  if not shots: shots = [[0, n]]
  def shot_of(f):
      for k, (f0, f1) in enumerate(shots):
          if f0 <= f < f1: return k
      return len(shots) - 1
  # для кожного плану: режим + геометрія
  MODES = []
  for (f0, f1) in shots:
      fr_list = [f for f in SM if f0 <= f < f1]
      pid_here = collections.Counter(spk[f] for f in range(f0, min(n, f1)) if spk[f] is not None).most_common(1)
      pid_here = pid_here[0][0] if pid_here else None
      boxes = []          # обличчя спікера в цьому плані (mid-координати)
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
      # чи можна вмістити всіх у 9:16?
      xs = [(b[0] + b[2] / 2) * MINV for _, b in allf]
      spread = (max(xs) - min(xs)) if len(xs) > 1 else 0
      npers_here = len({pid for pid, _ in allf})
      # v3.4 (правило Маріка 06.09, тверде): у ЗАГАЛЬНИЙ ПЛАН НІКОЛИ НЕ КРОПИМО — незалежно від тривалості.
      # Загальний показуємо цілим: смуга 16:9 на всю ширину з розмитим верхом/низом. Кроп на загальному
      # (спікер дрібний, у кадрі стіл і фон) із наступним стрибком на крупний — найгірший артефакт для нього.
      # v3.5 (правка Маріка 07.09): смуга — лише для КОРОТКОГО загального плану (швидка перебивка, кроп читався б
      # як стрибок). Якщо загальний тримається довше і спікера видно — кропимо його, а не показуємо смугу.
      sdur = (f1 - f0) / float(fps)
      # v3.5.1 (07.09): смуга лише якщо план КОРОТКИЙ, або в кадрі взагалі немає облич.
      # Довгий загальний, де мовця не впізнали (bh is None), але обличчя є — кропимо найбільше обличчя,
      # а не показуємо смугу: інакше 4 с далекого плану, а потім стрибок на крупний.
      # v3.5.2 (07.09): далекий план — смугою (як в Opus), але НЕ коли він впирається у склейку:
      # там і так зміна сцени, і смуга поруч читається як стрибок — у цьому разі кропимо.
      _t0, _t1 = f0 / float(fps), f1 / float(fps)
      near_splice = any(abs(_t1 - sp) < 0.6 or abs(_t0 - sp) < 0.6 for sp in SPL)
      tiny = (bh is None) or (bh < a.band_wide * H)
      # v3.5.6 (16.09): смуга коротша за 1.2 с або впритул до початку/кінця кліпу — це спалах,
      # читається як глюк, а не як перебивка. Такий шматок кропимо, як довгий загальний план.
      near_edge = (_t0 < 0.6) or (_t1 > (n / float(fps)) - 0.6)
      if fr_list and not allf and any(x <= f0 and f1 <= y + 1 for x, y in NOFACE):
          # v3.5.11 (16.09): план цілком лежить у зоні без облич — це слайд/графіка. Показуємо цілим
          # кадром 16:9 (смуга), незалежно від тривалості: кроп 9:16 ріже графік навпіл.
          # Перевірка саме по NOFACE, а не просто по порожньому allf: короткий план може не мати
          # жодного семпла, і тоді порожній allf означав би «не знаю», а не «графіка».
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
  print(f'планів джерела: {len(shots)} ({len(shots)/(n/fps)*60:.1f}/хв) | SPEAKER {nsp}, WIDE {nw} ({tw:.0%} часу), SPLIT {nsl}', flush=True)
  # ---- v2.1: СЕГМЕНТИ КАДРУ = межі планів джерела ∪ межі мовця ∪ склейки ----
  # Один нерухомий кадр на сегмент. Усередині сегмента камера не рухається й не перекадровується:
  # людина може нахилятися, жестикулювати — кадр стоїть. Кадр міняється лише коли ріже джерело
  # або коли починає говорити інша людина (рев'ю Маріка 05.09: «є загальний план — і ти зразу ж на спікері»).
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
      SPL_SNAP.append(min(near, key=lambda c: abs(c - f)) if near else f)   # склейка поруч із різом джерела = один різ
  SPL_F = sorted(set(SPL_SNAP))
  bset |= set(SPL_F)
  # v2.3: панчлайн тримаємо одним кадром. У швидкому обміні (троє домовляють фразу по черзі) камера
  # не встигає за спікером і кришить фінал — тому в останньому реченні різів немає.
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
      tail = min(max(tail, te - 5.0), te - 2.0)   # панчлайн — останні 2–5 с
      cutoff = int(round(tail * fps)); TAIL_CUT = cutoff
      drop = {x for x in bset if x > cutoff}
      if drop or True:
          bset -= drop
          bset.add(cutoff)          # сам початок фінальної фрази — це різ: панчлайн отримує власний кадр
          bset.add(n)               # v3.2: кінець кліпу теж вилітав у drop → фінальна фраза жила в кадрі попереднього сегмента (витримки 8–12 с)
          print(f'фінальна фраза з {tail:.1f} с — тримаємо одним кадром (прибрано {len(drop) - 1} різів)', flush=True)
  # v2.2: власні різи в довгих статичних планах — інакше картинка стоїть 20+ с і глядач іде.
  # Референс (6 кліпів Opus, 74 витримки): медіана 3.4 с, p75 5.0, p90 6.4, майже нічого > 8 с.
  # точки, де можна різати: пауза в мовленні АБО межа фрази за пунктуацією.
  # (в авто-субтитрах YouTube пауз майже немає — слова стикуються, тож пунктуація тут головний сигнал)
  gaps_f = []
  for i in range(1, len(WORDS_RAW)):
      pw = WORDS_RAW[i - 1]; pe = pw.get('e', pw['t'] + 0.2)
      gap = WORDS_RAW[i]['t'] - pe
      if gap >= 0.15 or str(pw.get('w', '')).rstrip()[-1:] in '.,!?…:;':
          gaps_f.append(int(round(((pe + WORDS_RAW[i]['t']) / 2) * fps)))
  def snap_to_pause(f, win=0.7):
      cand = [g for g in gaps_f if abs(g - f) <= win * fps]
      return min(cand, key=lambda g: abs(g - f)) if cand else None
  # v3.2: межі між словами (навіть без паузи) — запасний варіант, коли поруч немає ні паузи, ні пунктуації:
  # інакше план стояв 9–12 с (Opus: майже нічого довше 8 с; його різи джерела теж падають посеред речень)
  wb_f = sorted({int(round(((WORDS_RAW[i - 1].get('e', WORDS_RAW[i - 1]['t'] + 0.2) + WORDS_RAW[i]['t']) / 2) * fps)) for i in range(1, len(WORDS_RAW))})
  # solo: різ лише на межі речення (пунктуація або пауза >= 0.35 с), у радіусі 2.6 с; інакше різу немає
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
      k = int(L // a.hold_max) + 1           # ділимо на рівні частини по ≤ hold_max
      for j in range(1, k):
          f = f0 + int(round((f1 - f0) * j / k))
          g = snap_any(f)
          if g and f0 + a.hold_min * fps < g < f1 - a.hold_min * fps and g <= TAIL_CUT: extra.append(g)
  bset |= set(extra)
  if extra: print(f'власних різів у довгих планах: {len(extra)} (ракурс не тримаємо довше {a.hold_max:.0f} с)', flush=True)
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
              # v3.5.8 (16.09): різ кадру — не різ звуку. Якщо межі слова поруч немає, а план утричі
              # довший за hold-max, ріжемо все одно: 15 с на одному кадрі глядач помічає, різ — ні.
              elif (not PUNCH_SENTENCE) and L > 1.5 * a.hold_max and ok_(f): pts.append(f)
          pts.append(f1)
          for i2 in range(len(pts) - 1): out.append([pts[i2], pts[i2 + 1]])
      return out
  # жодного сегмента коротшого за min_seg — але зливаємо ТІЛЬКИ в межах одного плану джерела (v3.3):
  # кроп, що перетинає різ джерела, показує чужу камеру чужим кадруванням («загальний план, але обрізаний»).
  # Короткий план без сусіда в тому ж плані лишається окремим сегментом зі своїм кадром.
  locked = set()
  while len(segs) > 1:
      cands = [i for i in range(len(segs)) if (segs[i][1] - segs[i][0]) < a.min_seg * fps and tuple(segs[i]) not in locked]
      if not cands: break
      k = min(cands, key=lambda i: segs[i][1] - segs[i][0])
      sh_k = shot_of(segs[k][0])
      left = k - 1 if k > 0 and shot_of(segs[k - 1][0]) == sh_k else None
      right = k + 1 if k + 1 < len(segs) and shot_of(segs[k + 1][0]) == sh_k else None
      if left is None and right is None:
          # v3.5.5 (16.09): огризок < 1 с на краю кліпу або впритул до склейки — це блимання.
          # Правило v3.3 (не зливати через різ джерела) тут програє: 0.7 с чужого кадрування
          # глядач не помітить, а блиск на 0.7 с помітить обов'язково.
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
  segs = split_long(segs)                               # після злиття могли утворитись задовгі сегменти
  if not segs: segs = [[0, n]]
  extra = set(extra) | {f0 for f0, f1 in segs}
  # кадр кожного сегмента: медіанний бокс МОВЦЯ в межах сегмента; якщо мовця в кадрі нема — тримаємо
  # попередній кадр (той самий план) або беремо найбільше обличчя
  # v3.5.6 (16.09): збір боксів мовця в сегменті — винесено, щоб порахувати масштаб ДВІЧІ:
  # спершу по всіх сегментах (щоб задати один зум на план джерела), потім уже кадри.
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
  ZOOM_FLOOR = 0.62      # як авто-MIN_CROP для 4K у ОПУС_ФРЕЙМВОРК: апскейл 1.4x, кадр як в Opus
  def tch_for(fh_):
      t = min(H, max(MIN_CROP * H, fh_ / a.face_frac))
      if fh_ > 0 and fh_ / t < 0.17: t = min(H, max(ZOOM_FLOOR * H, fh_ / 0.17))
      return t
  # ОДИН масштаб на план джерела: медіана висоти обличчя по всьому плану (зважена довжиною сегмента).
  # Зум більше не «фліпає» між сусідніми сегментами всередині одного плану — саме це читалось як
  # «камера приближається/віддаляється» (правка Маріка 06.09) — і водночас у довгому загальному
  # плані спікер не лишається дрібним.
  _shot_fh = {}
  for (f0, f1) in segs:
      _b = seg_boxes(f0, f1)[2]
      if not _b: continue
      _shot_fh.setdefault(shot_of(f0), []).extend([float(np.median(_b))] * max(1, f1 - f0))
  SHOT_TCH = {}
  for _k, _v in _shot_fh.items():
      if not _v: continue
      _t = tch_for(float(np.percentile(_v, 10)))     # орієнтир — найширший момент плану, а не середній
      _t = max(_t, float(np.max(_v)) / 0.50)         # найбільше обличчя цього ж плану має влізти в кадр
      SHOT_TCH[_k] = float(min(H, _t))
  if _shot_fh:
      print('масштаб планів: ' + ' | '.join(
          f"{shots[k][0]/fps:.1f}-{shots[k][1]/fps:.1f}с p10={np.percentile(v,10)/H:.0%} max={np.max(v)/H:.0%} кроп={SHOT_TCH[k]:.0f}"
          for k, v in sorted(_shot_fh.items()) if k in SHOT_TCH), flush=True)
  CROPS = []; prev_crop = None; prev_punch = False; SEG_FH = []
  for (f0, f1) in segs:
      # v2.2: обираємо обличчя за оцінкою ASD усередині сегмента — це надійніше, ніж «персона-мовець»
      # (у загальному плані обличчя дрібне, ембеддинг може дати іншу персону, і кадр їхав не на того)
      bx, by, bh = seg_boxes(f0, f1)
      if not bx:
          CROPS.append(prev_crop if prev_crop else (W / 2, H / 2, H, 1.0)); SEG_FH.append(SEG_FH[-1] if SEG_FH else 1.0); continue
      fh_ = float(np.median(bh)); SEG_FH.append(fh_ / H)
      tch = SHOT_TCH.get(shot_of(f0)) or tch_for(fh_)   # v3.5.6: один масштаб на план джерела
      cx_ = float(np.median(bx)); cy_ = float(np.median(by)) + a.face_y * tch
      # v2.2: якщо кадр майже не змінився проти попереднього сегмента (склейка або власний різ у тому самому
      # плані) — чергуємо «звичайний ↔ підріз», щоб різ читався як монтажний, а не як глюк/сіпання
      same_as_prev = prev_crop and abs(cx_ - prev_crop[0]) < 0.06 * W and abs(tch - prev_crop[2]) < 0.08 * H
      # v3.5.7 (16.09): на спокійному профілі (--punch 1.0) чергування вимкнене повністю.
      # Раніше гілка «підрізу» все одно піднімала кроп до MIN_CROP*H*0.80 = 1728 px і
      # перебивала дозум на широкому плані — обличчя лишалось дрібним.
      if a.punch != 1.0 and same_as_prev and (f0 in SPL_F or f0 in extra):
          if prev_punch: tch = min(H, tch / a.punch)
          else: tch = max(MIN_CROP * H * 0.80, tch * a.punch)
          prev_punch = not prev_punch
          cy_ = float(np.median(by)) + a.face_y * tch
      elif not same_as_prev:
          prev_punch = False
      CROPS.append((cx_, cy_, tch, 1.0)); prev_crop = CROPS[-1]
  # v3.0: 1–3 виходи на кадр-смугу (розмитий верх/низ) — так робить Opus (10 % часу, 1.5–4.5 с за раз).
  # v3.1: в Opus смуга — це ШИРОКИЙ план джерела (обличчя дрібні, кроп не має сенсу), тож кандидати —
  # передусім широкі плани (обличчя мовця < band-wide висоти), серед них — де видима людина мовчить (слухає);
  # у фіналі кліпу — лише якщо фінальний план широкий (крупний план у смузі не читається як «вихід на загальний»).
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
      # v3.5.10 (16.09): сегмент, що впирається у склейку, у кандидати не беремо — смуга там
      # дає подвійний стик (зміна кадру + зміна сцени) і читається як стрибок.
      def _at_splice(k3):
          t0_, t1_ = segs[k3][0] / fps, segs[k3][1] / fps
          return any(abs(t1_ - sp) < 0.8 or abs(t0_ - sp) < 0.8 for sp in SPL)
      rest = [c for c in cand if c[0] not in chosen and 0 < c[0] < len(segs) - 1 and not _at_splice(c[0])]
      rest.sort(key=lambda c: (not c[3], c[2]))              # спершу широкі плани, серед них — найтихіші
      wide_rest = [c for c in rest if c[3]]
      pool = wide_rest if wide_rest else rest                # немає жодного широкого плану — беремо найтихіші
      for c in pool:                                         # v3.2: сусідні сегменти не беремо — інакше дві смуги зливаються в одну на 7–9 с
          if len(chosen) >= a.band: break
          if any(abs(c[0] - k3) <= 1 for k3 in chosen): continue
          chosen.append(c[0])
      if not last_wide and a.band_end: print('фінальний план крупний — смуги у фіналі не буде', flush=True)
      # смуга живе рівно band-len секунд усередині сегмента (референс Opus: 1.5–4.5 с), а не весь сегмент
      L2 = int(round(a.band_len * fps))
      for k2 in chosen:
          f0, f1 = segs[k2]
          if (f1 - f0) - L2 < a.min_seg * fps:
              BAND_SEGS.add((f0, f1))                                  # сегмент і так короткий — увесь на смугу
          else:
              BAND_SEGS.add((f1 - L2, f1))                             # смуга впирається в наступний різ, недогризків не лишає
      if BAND_SEGS: print('кадр-смуга (розмитий верх/низ): ' + ', '.join(f'{x/fps:.1f}-{y/fps:.1f} с' for x, y in sorted(BAND_SEGS)), flush=True)
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
  print(f'сегментів кадру: {len(segs)} ({len(segs)/(n/fps)*60:.1f}/хв) — межі: різи джерела + зміна мовця + склейки', flush=True)
  print(f'faces: tracks={next_id} frames={n} crop={CW}x{H} min-crop={MIN_CROP:.2f}', flush=True)
  if a.path_json: json.dump({'n': n, 'path': path, 'fps': fps, 'W': W, 'H': H, 'CW': CW, 'src_cuts': [round(c / fps, 2) for c in CUTS]}, open(a.path_json, 'w'))
  if a.path_only: sys.exit(0)

# ---------------- captions (same style as render_blurfit_tc.py) ----------------
WORDS = json.load(open(a.words, encoding='utf-8'))
# v1.5: чистка субтитрів — маркери спікера >>, теги [сміх]/[сопіння], чисті філери (е, ее, м), артефакти авто-субтитрів («до$» = доларів)
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
    print(f"!!! УВАГА: субтитри до {WORDS[-1]['e']:.1f} с, а відео лише {n/fps:.1f} с — кліп обрізано, кінець думки втрачено", flush=True)
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
        LEND.append(nt if nt - e_ < 1.0 else e_ + a.cap_hold)   # v3.1: у довгій паузі підпис зникає, а не висить до наступної фрази
    else: LEND.append(e_ + 0.35)
MAXW = 940; ROW = int(a.font_max * 1.20); FCACHE = {}
def gf(sz):
    if sz not in FCACHE:
        f = ImageFont.truetype(a.font, sz)
        for _v in (a.weight, 'Bold'):                      # v3.1: ExtraBold як в Opus; якщо в шрифті нема — Bold
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
WIDE_Y = 0.42     # центр смуги широкого плану (Opus ставить її трохи вище центру, підписи — під нею)
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
