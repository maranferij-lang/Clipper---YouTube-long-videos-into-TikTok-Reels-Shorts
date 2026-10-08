#!/usr/bin/env python3
"""
pipeline_v3.py (v3.1, 19.09.2026)  конвеєр «4K-діапазони -> raw -> Light-ASD -> render_opus_tc_v4 -> loudnorm -> QC» для пачки кліпів.
Резюмований: стан у <work>/state.json, стадії hq -> cut -> asd -> path -> render -> final -> (qc) -> done | qc_fail.
Виклик повторюється до "ALL DONE" (у пісочниці з лімітом на виклик <= 180 с, --budget 165; нативно на маку --budget 100000).

  python3 pipeline_v3.py jobs.json --work ~/ep5/w --budget 165 [--profile podcast|solo] [--no-asd] [--keep-raw] [--no-qc]

jobs.json: {"video":"<youtube id>","words":"<abs path yt_words.json>","out":"<preview dir>",
            "jobs":[{"name":..,"ranges":[[a,b],..],"title":"..", "fix":{"нениття":"не ниття"}, "plate":false}]}
  ranges: абсолютні секунди, точно по словах (старт = t першого слова, кінець = e останнього); кілька діапазонів =
          внутрішня склейка (лише вперед). Запас ±0.5 с для зсуву меж у тишу бере на себе pipeline (MARGIN), у відбір
          його не закладати: інакше в діапазон потрапляє чуже слово.
  fix:    орфографічні виправлення автосубтитрів для цього кліпу (слово -> слово). Змісту не міняє.
  plate:  плашка-заголовок угорі лише для цього кліпу (за замовчуванням її немає, як в Opus).

Що робить кожна стадія:
  hq     кожен діапазон качається окремо з запасом ±0.5 с (--force-keyframes-at-cuts) шматками по <= 28 с і склеюється
         без перекодування (довгі діапазони інакше не влазять у виклик).
  cut    огризки чужого плану на краях підрізаються до різу джерела (лише якщо там не починається слово); старт і
         кінець підтягуються в тишу по RMS (таймкоди YouTube відстають на ~0.08 с); частини з ALAC (без втрат) ->
         raw.mp4; words.json у таймлайні кліпу з виправленнями fix; meta.json (склейки, межі).
  asd    Light-ASD на 1440p-копії (run_asd_cpu_q4_v1.2.py, резюмований).
  path   траєкторія камери (render --path-only) -> path.json + path_info.json.
  render шматками по RENDER_CH кадрів (260) у r_*.mp4 з ALAC-звуком.
  final  склейка, highpass 85 + loudnorm I=-15 TP=-1.5 LRA=11 у два проходи, один AAC; qc/<name>.json і контакт-лист;
         потім qccheck_v2: чисто -> done (raw і 4K-вікна прибираються), інакше qc_fail (усе лишається для перерендеру).
Після ALL DONE пише <out>/НАЗВИ_для_описів.md із заголовків.
"""
import os, sys, json, time, subprocess, math, glob, shutil, argparse

ap = argparse.ArgumentParser()
ap.add_argument('jobs'); ap.add_argument('--budget', type=float, default=150)
ap.add_argument('--work', default=os.path.expanduser('~/ep3/pipe'))
ap.add_argument('--only', default='', help='лише ці кліпи (через кому)')
ap.add_argument('--asd-res', type=int, default=1440, help='висота копії для ASD (2160 = з 4K-кадрів, 1440 — швидше)')
ap.add_argument('--no-asd', action='store_true', help='не ганяти Light-ASD (соло-канал: одна людина в кадрі)')
ap.add_argument('--plate', action='store_true', help='плашка-заголовок угорі для ВСІХ кліпів (за замовчуванням лише там, де в jobs.json "plate": true)')
ap.add_argument('--renderer', default='render_opus_tc_v4.py')
ap.add_argument('--asd-script', default='run_asd_cpu_q4_v1.2.py')
ap.add_argument('--qc-script', default='qccheck_v2.py')
ap.add_argument('--profile', default='podcast', choices=['podcast', 'solo'])
ap.add_argument('--keep-raw', action='store_true', help='не прибирати raw.mp4 і 4K-вікна після done')
ap.add_argument('--no-qc', action='store_true', help='не запускати qccheck у final')
ap.add_argument('--chunk', type=int, default=int(os.environ.get('RENDER_CH', 260)), help='кадрів на шматок рендера')
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

# ---------- вікна 4K (спільні між кліпами) ----------
MARGIN = 0.5   # v3.1 (19.09): 4K-вікно качається з запасом ±0.5 с, щоб межу можна було посунути в тишу в обидва боки;
               # самі ranges у jobs.json лишаються точними (по словах), запас у відбір не закладається
def windows_for(ranges):
    return [(round(max(0.0, s - MARGIN), 2), round(e + MARGIN, 2)) for s, e in ranges]
def hq_path(s, e): return os.path.join(a.work, 'hq', f'{VID}_{s:.2f}_{e:.2f}.mp4')
def vdur(p):
    try: return float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', p], capture_output=True, text=True).stdout.strip())
    except Exception: return 0.0
HQ_CHUNK = 28.0   # с; 4K-діапазон довший за це не встигає завантажитись за один виклик
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
        if d and abs(d - (e - s)) < 0.6: os.replace(tmp, out); return True      # цілий файл потрібної довжини
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

EDGE = 1.6      # v2.1: у скільки секунд від краю діапазону шукаємо різ джерела
def edge_cuts(src, tail=False):
    """різи джерела в перших (або останніх, tail=True) EDGE секундах файлу: список часів від початку вікна"""
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
    """v2.1: якщо межа діапазону стоїть за 0.06–1.6 с від різу джерела і на цьому огризку немає слів —
    посуваємо межу до різу (у кадрі не буде 0.3–1.5 с чужого плану, який читається як стрибок). Повертає нові ranges."""
    out = []
    for i, (s, e) in enumerate(job['ranges']):
        src = hq_path(*windows_for([[s, e]])[0]); s2, e2 = s, e
        starts = [w['t'] for w in W if s <= w['t'] < e]
        cs, n1 = edge_cuts(src, tail=False)
        cs = [c for c in cs if 0.06 <= c <= EDGE]
        if cs:
            c = max(cs)                                     # останній різ у перших 1.6 с
            # без слів на огризку — зсуваємо; якщо слово «починається» на ньому, але огризок ≤ 0.4 с — теж зсуваємо
            # (межу діапазону я ставлю на початок слова за авто-субтитрами ±0.2 с, а монтажер ріже на реальному вступі голосу)
            if not any(s <= t < s + c + 0.05 for t in starts): s2 = round(s + c, 2)   # v2.2 (07.09): прибрано «or c <= 0.4» —
            # воно зсувало початок навіть коли на огризку починалось слово, і слово пропадало зі звуку (лишаючись у субтитрах)
        ce, n2 = edge_cuts(src, tail=True)
        ce = [EDGE - c for c in ce if 0.06 <= EDGE - c <= EDGE]    # відстань від кінця діапазону (вікно аналізу [e-EDGE, e])
        if ce:
            c = max(ce)                                     # перший різ в останніх 1.6 с
            cabs = e - c
            before = [t for t in starts if t < cabs]
            if not any(cabs - 0.05 <= t < e for t in starts) and (not before or before[-1] <= cabs - 0.4): e2 = round(cabs - 0.03, 2)
        if (s2, e2) != (s, e): log(f"  stub: r{i} {s:.2f}-{e:.2f} → {s2:.2f}-{e2:.2f} (початок +{s2-s:.2f}, кінець −{e-e2:.2f})")
        out.append([s2, e2])
    return out

def snap_silence(src, ranges0, ranges, W):
    """v3.1: старт і кінець у тишу в обидва боки в межах запасу MARGIN.
    Старт: якщо на межі вже є голос, дивимось, чи є тиша між межею і першим словом: є -> це хвіст чужої фрази, старт
    уперед до останньої тиші перед словом; немає -> слово почалося раніше за межу (таймкоди YouTube відстають), старт
    назад до останньої тиші перед межею (до 0.35 с). Кінець дзеркально: голос на межі і тиша після останнього слова ->
    назад до першої тиші після слова; голос неперервний від слова до межі -> уперед до першої тиші (до 0.4 с).
    Тиша = RMS вікна 20 мс < 12 % медіани RMS мовлення. Слова не чіпаємо: старт не заходить за перше слово."""
    import numpy as np
    out = []
    for i, ((s0, e0), (s, e)) in enumerate(zip(ranges0, ranges)):
        try:
            w0 = windows_for([[s0, e0]])[0][0]                       # файл починається з w0 = s0 - MARGIN
            raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', src[i], '-ac', '1', '-ar', '16000', '-f', 's16le', '-'], capture_output=True).stdout
            au = np.abs(np.frombuffer(raw, np.int16).astype(np.float32)); win = 320   # 20 мс
            nfr = len(au) // win
            if nfr < 30: out.append([s, e]); continue
            rms = np.sqrt((au[:nfr * win].reshape(nfr, win) ** 2).mean(axis=1))
            speech = np.median(rms[rms > np.percentile(rms, 40)]) if nfr else 0
            thr = 0.12 * max(speech, 1.0)
            K = lambda t: int(round((t - w0) / 0.02))               # час епізоду -> індекс вікна
            wt = [w['t'] for w in W if s0 <= w['t'] < e0]; we = [w.get('e', w['t'] + 0.3) for w in W if s0 <= w['t'] < e0]
            first_w = min(wt) if wt else s0; last_e = max(we) if we else e0
            s2, e2 = s, e
            def last_run(q):     # останній суцільний тихий відрізок зі списку індексів
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
            PAD = 4              # скільки тихих вікон (по 20 мс) лишати перед словом / після слова: 80 мс
            # старт
            if rms[K(s):K(s) + 3].mean() >= thr:
                q_fwd = [k for k in range(K(s), min(K(first_w - 0.03), nfr)) if rms[k] < thr]
                if q_fwd:                                                            # хвіст чужої фрази -> уперед, у тишу перед словом
                    r = last_run(q_fwd); s2 = round(w0 + max(r[0], r[-1] - PAD + 1) * 0.02, 2)
                else:
                    q_back = [k for k in range(max(0, K(s - 0.35)), K(s)) if rms[k] < thr]
                    if q_back:                                                       # слово почалось раніше -> назад, лишаючи тишу
                        r = last_run(q_back); s2 = round(w0 + max(r[0], r[-1] - PAD + 1) * 0.02, 2)
            # кінець
            if rms[max(0, K(e) - 3):K(e)].mean() >= thr:
                q_back2 = [k for k in range(max(0, K(last_e + 0.03)), K(e)) if rms[k] < thr]
                if q_back2:                                                          # чужий вступ -> назад, у тишу після слова
                    r = first_run(q_back2); e2 = round(w0 + (min(r[-1], r[0] + PAD - 1) + 1) * 0.02, 2)
                else:
                    q_fwd2 = [k for k in range(K(e), min(K(e + 0.40), nfr)) if rms[k] < thr]
                    if q_fwd2:                                                       # слово ще звучить -> уперед
                        r = first_run(q_fwd2); e2 = round(w0 + (min(r[-1], r[0] + PAD - 1) + 1) * 0.02, 2)
            if (s2, e2) != (s, e): log(f'  тиша: r{i} {s:.2f}-{e:.2f} -> {s2:.2f}-{e2:.2f} (старт {s2-s:+.2f}, кінець {e2-e:+.2f})')
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
        AF = f'aresample=async=1,afade=in:d=0.015,afade=out:st={max(0.0, (e - s) - 0.015):.3f}:d=0.015'   # fade на обох краях проти клацань
        # вікно ширше за діапазон на MARGIN, тому частина вирізається точно по межах з одним перекодуванням (ultrafast crf 14);
        # звук ALAC без втрат, AAC буде один раз у final
        sh(['ffmpeg', '-y', '-v', 'error', '-ss', f'{s - w0:.3f}', '-t', f'{e - s:.3f}', '-i', src, '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '14', '-threads', '2',
            '-pix_fmt', 'yuv420p', '-c:a', 'alac', '-ar', '48000', '-af', AF, p], check=True)
        parts.append(p)
    job['ranges'] = ranges
    open(os.path.join(d, 'list.txt'), 'w').write(''.join(f"file '{p}'\n" for p in parts))
    sh(['ffmpeg', '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', os.path.join(d, 'list.txt'), '-c', 'copy', raw], check=True)
    for p in parts: os.remove(p)
    rd = vdur(raw); want = sum(e - s for s, e in job['ranges'])
    if abs(rd - want) > 0.5: log(f'!!! cut: raw {rd:.2f} c, а має бути {want:.2f} c')
    # слова у таймлайні кліпу + склейки
    W = json.load(open(J['words'], encoding='utf-8')); out = []; off = 0.0; spl = []; lost = []
    for i, (s, e) in enumerate(job['ranges']):
        s0, e0 = ranges0[i]
        for k, w in enumerate(W):
            if min(s, s0) <= w['t'] < max(e, e0):
                if w['t'] >= e + 0.05 or w['t'] < s - 0.12: lost.append(w['w']); continue   # слово поза новими межами: у субтитри не йде
                we = w.get('e', min(W[k + 1]['t'], w['t'] + 1.0) if k + 1 < len(W) else w['t'] + 0.5)
                ww = w['w']
                for bad, good in job.get('fix', {}).items(): ww = ww.replace(bad, good)
                t0 = max(w['t'], s)
                out.append({'t': round(t0 - s + off, 2), 'e': round(max(min(we, e), t0 + 0.15) - s + off, 2), 'w': ww})
        off += e - s
        if i < len(job['ranges']) - 1: spl.append(round(off, 2))
    json.dump(out, open(os.path.join(d, 'words.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    multi = any(('&gt;&gt;' in w['w']) or ('>>' in w['w']) for w in out)
    if lost: log('   слова поза новими межами (не в субтитрах):', ' '.join(lost)[:120])
    json.dump({'splices': spl, 'dur': round(off, 2), 'multi_speaker': multi, 'ranges_used': job['ranges'], 'ranges_orig': ranges0, 'lost_words': lost}, open(os.path.join(d, 'meta.json'), 'w'), ensure_ascii=False)
    log('cut ok', name, f'{off:.1f} c, склейки {spl}')
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
        old = json.load(open(rj)) if os.path.exists(rj) else meta.get('ranges_orig')   # без ranges.json — старий ASD рахувався на вихідних межах
        if cur is not None and old != cur:      # v2.1: ASD рахувався на іншому таймлайні (до підрізання огризків) — перерахувати
            log('asd: таймлайн змінився (підрізані огризки) — перераховую', name); shutil.rmtree(ad, ignore_errors=True)
        else:
            s2 = os.path.join(d, f'raw_{a.asd_res}.mp4') if a.asd_res < 2160 else os.path.join(d, 'raw.mp4')
            if not os.path.exists(os.path.join(ad, 'dims.json')) and os.path.exists(s2): dims(s2)
            return True
    if a.no_asd:
        log('asd пропущено (--no-asd)', name); return True
    if left() < 45: return False
    src = os.path.join(d, 'raw.mp4')
    if a.asd_res < 2160:
        src2 = os.path.join(d, f'raw_{a.asd_res}.mp4')
        if not os.path.exists(src2):
            sh(['ffmpeg', '-y', '-v', 'error', '-i', src, '-vf', f'scale=-2:{a.asd_res}:flags=area', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '16', '-c:a', 'copy', src2], check=True)
        src = src2
    dur = json.load(open(os.path.join(d, 'meta.json')))['dur']
    log('asd', name, f'(бюджет {left():.0f} c)')
    try:
        sh([sys.executable, os.path.join(TOOLS, a.asd_script), '--src', src, '--start', '0', '--dur', str(math.ceil(dur) + 1), '--save', ad,
            '--weight', os.path.join(TOOLS, 'weight', 'pretrain_AVA_CVPR.model'), '--yunet', os.path.join(TOOLS, 'yunet.onnx')],
           timeout=max(20, left() - 8), cwd=os.path.join(HOME, 'Light-ASD'), capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        log('asd: бюджет вичерпано, продовжимо наступним викликом'); return False
    ok = os.path.exists(os.path.join(ad, 'pywork', 'scores.pckl'))
    if ok:
        dims(src)   # розміри копії, на якій рахувався ASD: pyavi/video.avi прибирається після рендеру, а рендеру потрібна ширина
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
    info = [l for l in r.stdout.splitlines() if l.startswith(('persons', 'ASD', 'speaker', 'різи', 'планів', 'сегментів', 'кадр-смуга', 'вставки', 'фінальн', 'власних', 'масштаб', '!!!'))]
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
            log('render: бюджет вичерпано на шматку', f'{A}:{B}'); return False
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
    # звук: highpass 85 Гц + loudnorm I=-15 TP=-1.5 LRA=11 (selection-standard), два проходи (виміряти → лінійна нормалізація без «дихання»)
    AF = 'highpass=f=85,loudnorm=I=-15:TP=-1.5:LRA=11'
    m = sh(['ffmpeg', '-v', 'info', '-i', cat, '-af', AF + ':print_format=json', '-f', 'null', '-'], capture_output=True, text=True).stderr
    try:
        js = json.loads(m[m.rindex('{'):m.rindex('}') + 1])
        AF2 = AF + f":measured_I={js['input_i']}:measured_TP={js['input_tp']}:measured_LRA={js['input_lra']}:measured_thresh={js['input_thresh']}:offset={js['target_offset']}:linear=true"
    except Exception as ex:
        log('loudnorm: вимір не вдався, один прохід', ex); AF2 = AF
    sh(['ffmpeg', '-y', '-v', 'error', '-i', cat, '-c:v', 'copy', '-af', AF2, '-ar', '48000', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', outp], check=True)
    os.remove(cat)
    # QC: тривалість відео vs субтитрів, розкладка
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
        if not qc['ok_len']: log(f"!!! {job['name']}: відео {vd:.1f} c коротше за субтитри {wend:.1f} c")
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
    # прибирання важкого
    for f in glob.glob(os.path.join(d, 'r_*.mp4')) + glob.glob(os.path.join(d, 'g_*.png')) + glob.glob(os.path.join(d, 'raw_*.mp4')): os.remove(f)
    shutil.rmtree(os.path.join(d, 'asd', 'pyframes'), ignore_errors=True); shutil.rmtree(os.path.join(d, 'asd', 'pycrop'), ignore_errors=True)
    shutil.rmtree(os.path.join(d, 'asd', 'pyavi'), ignore_errors=True)
    log('FINAL', outp, f'{os.path.getsize(outp)/1e6:.0f} MB')
    return True

def qc(job):
    """qccheck_v2 на готовому кліпі: чисто -> True (done), проблеми -> стан qc_fail, звіт у qc/<name>.qc.txt"""
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
        S[job['name']] = 'qc_fail'; save(); log('QC FAIL', job['name'], '- raw і 4K лишаються для перерендеру'); return 'fail'
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
        if ok == 'fail': break            # qc_fail: стан уже записано, до наступного кліпу
        if ok:
            st = STAGES[[s for s, _ in STAGES].index(st) + 1][0] if st != 'qc' else 'done'
            S[job['name']] = st; save()
    if ok == 'fail': continue
    if not ok:
        log('STOP на', job['name'], st, f'(лишилось {left():.0f} c)'); break
    # done з чистим QC: важке прибираємо (4K-вікна лише якщо їх не чекають інші кліпи)
    if not a.keep_raw:
        used = {hq_path(*w) for j2 in jobs if S.get(j2['name'], 'hq') not in ('done',) and j2 is not job for w in windows_for(j2['ranges'])}
        for w in windows_for(job['ranges']):
            p = hq_path(*w)
            if p not in used and os.path.exists(p): os.remove(p)
        for f in glob.glob(os.path.join(a.work, job['name'], 'raw*.mp4')) + glob.glob(os.path.join(a.work, job['name'], 'x.mp4')): os.remove(f)
if all(S.get(j['name']) in ('done', 'qc_fail') for j in jobs):
    names = [j['name'] for j in jobs if S.get(j['name']) == 'done']
    if names:
        open(os.path.join(J['out'], 'НАЗВИ_для_описів.md'), 'w', encoding='utf-8').write(
            '| Файл | Назва для опису шортса | Довжина, с |\n|---|---|---|\n' +
            '\n'.join(f"| {j['name']}.mp4 | {j.get('title', '')} | {sum(e - s for s, e in j['ranges']):.0f} |" for j in jobs if j['name'] in names) + '\n')
    fails = [j['name'] for j in jobs if S.get(j['name']) == 'qc_fail']
    print('ALL DONE' + (f'  (qc_fail: {", ".join(fails)})' if fails else ''))
print('стан:', {j['name']: S.get(j['name'], 'hq') for j in jobs})
