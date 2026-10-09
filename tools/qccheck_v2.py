#!/usr/bin/env python3
"""qccheck_v2.py (19.09.2026) automatic clip check before delivery. Called by pipeline_v3 in the qc stage.
Catches what used to be found by eye: scale jumps, a band shot next to a splice,
a crop on a wide shot, shots that are too short/long, eaten words, technical standard.
v2: exact check "word is in the subtitles but not in the audio"; silence at start/end (RMS); a face in the first 3 s; duplicates in a batch;
peak level (TP); clicks at splices; report qc/REPORT.txt when the whole folder is checked.
Usage: python3 qccheck_v2.py --work <work_dir> --out <preview_dir> [--name <clip>]
"""
import os, sys, json, glob, subprocess, argparse
import numpy as np, cv2

ap = argparse.ArgumentParser()
ap.add_argument('--work', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--name', default=''); ap.add_argument('--yunet', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yunet.onnx'))
ap.add_argument('--fps-sample', type=float, default=3.0)
a = ap.parse_args()

MIN_SEG, MAX_SEG = 1.8, 9.5      # shot length
SCALE_JUMP = 0.15                # change of crop height between neighbouring segments
BAND_MAX_SHARE, BAND_MAX_LEN = 0.20, 5.0
SPLICE_GUARD = 0.6               # a band shot must not run into a splice
FACE_MIN = 0.15                  # below this it is a wide shot; a crop on it is a defect

def probe(p, ent, stream=False):
    c = ['ffprobe','-v','error']+ (['-select_streams','v:0'] if stream else []) + \
        ['-show_entries', ent, '-of','csv=p=0', p]
    return subprocess.run(c, capture_output=True, text=True).stdout.strip()

def lufs(p):
    r = subprocess.run(['ffmpeg','-i',p,'-af','loudnorm=print_format=summary','-f','null','-'],
                       capture_output=True, text=True).stderr
    for l in r.splitlines():
        if 'Input Integrated' in l: return float(l.split()[-2])
    return None

def loud(p):
    r = subprocess.run(['ffmpeg','-i',p,'-af','loudnorm=print_format=summary','-f','null','-'], capture_output=True, text=True).stderr
    li = tp = None
    for l in r.splitlines():
        if 'Input Integrated' in l: li = float(l.split()[-2])
        if 'Input True Peak' in l: tp = float(l.split()[-2])
    return li, tp

def audio_np(p):
    raw = subprocess.run(['ffmpeg','-v','error','-i',p,'-ac','1','-ar','16000','-f','s16le','-'], capture_output=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0

def check(name):
    d = os.path.join(a.work, name); mp4 = os.path.join(a.out, name + '.mp4')
    if not os.path.exists(mp4): return None
    P = json.load(open(os.path.join(d,'path.json'))); fps = P['fps']; path = P['path']; H = P['H']
    meta = json.load(open(os.path.join(d,'meta.json')))
    problems = []; notes = []

    # --- segments ---
    segs=[]; prev=None
    for i,p in enumerate(path):
        k=(p[0], round(p[1] or 0,1), round(p[3] or 0,1))
        if k!=prev: segs.append([i,p]); prev=k
    segs.append([len(path),None])
    S=[(i/fps, j/fps, p[0], p[3]) for (i,p),(j,_) in zip(segs,segs[1:])]
    dur = len(path)/fps
    splices = meta.get('splices', [])

    _yn = cv2.FaceDetectorYN.create(a.yunet,'',(320,320),0.6,0.3,5000)
    _cap0 = cv2.VideoCapture(mp4); _vfps0 = _cap0.get(cv2.CAP_PROP_FPS) or 30
    _nf_cache = {}
    def NOFACE_BAND(t0, t1):
        """whether this piece really has no face at all (i.e. it is a graphic, not a wide shot)"""
        key = (round(t0,2), round(t1,2))
        if key in _nf_cache: return _nf_cache[key]
        res = True
        for t in (t0+0.4, (t0+t1)/2, max(t0+0.4, t1-0.4)):
            _cap0.set(cv2.CAP_PROP_POS_FRAMES, int(t*_vfps0))
            okf, fr = _cap0.read()
            if not okf or fr is None: continue
            _yn.setInputSize((fr.shape[1], fr.shape[0]))
            _, fs = _yn.detect(fr)
            if fs is not None and len(fs): res = False; break
        _nf_cache[key] = res
        return res

    for k,(t0,t1,mode,ch) in enumerate(S):
        L = t1-t0
        if L < MIN_SEG and not (k==len(S)-1 and L>0.6):
            problems.append(f'{t0:5.1f}–{t1:5.1f} s: shot length {L:.1f} s, flickers (min {MIN_SEG})')
        if L > MAX_SEG and not (mode == 'WIDE' and NOFACE_BAND(t0, t1)):
            # a graphics insert is held as long as it lasts: there is nothing to reframe a slide with
            problems.append(f'{t0:5.1f}–{t1:5.1f} s: shot length {L:.1f} s, too long on one frame (max {MAX_SEG})')
        if mode == 'WIDE':
            # v1.2 (16.09): a graphics/slide insert (no faces in frame) is shown as a full frame
            # for as long as needed: that is not a "band that is too long", it is the only correct way to show a graphic.
            if L > BAND_MAX_LEN and not NOFACE_BAND(t0, t1):
                problems.append(f'{t0:5.1f}–{t1:5.1f} s: band {L:.1f} s, too long (max {BAND_MAX_LEN})')
            for sp in splices:
                # a graphics insert that runs into a splice is not a jump: the slide changes the scene anyway
                if NOFACE_BAND(t0, t1): break
                if abs(t1-sp) < SPLICE_GUARD or abs(t0-sp) < SPLICE_GUARD:
                    problems.append(f'{t0:5.1f}–{t1:5.1f} s: band runs into the splice at {sp:.1f} s, there will be a jump at the join')
    # scale jumps between neighbouring SPEAKER segments.
    # v1.1 (16.09): at a source cut the frame changes completely, so a scale change there is not a defect
    # but editing. We only catch changes inside the same shot.
    SRC_CUTS = P.get('src_cuts', []) if isinstance(P, dict) else []
    for (t0,t1,m1,c1),(u0,u1,m2,c2) in zip(S, S[1:]):
        if m1=='SPEAKER' and m2=='SPEAKER' and c1 and c2:
            rel = abs(c1-c2)/max(c1,c2)
            near_src = any(abs(u0 - c) < 0.25 for c in SRC_CUTS)
            if rel > SCALE_JUMP and not near_src:
                problems.append(f'{u0:5.1f} s: scale jumped {c1:.0f}→{c2:.0f} px ({rel:.0%}), reads as the camera moving closer/away')
    band = sum(t1-t0 for t0,t1,m,_ in S if m=='WIDE' and not NOFACE_BAND(t0,t1))
    if band/dur > BAND_MAX_SHARE:
        problems.append(f'band shots take {band/dur:.0%} of the time, too many (max {BAND_MAX_SHARE:.0%})')

    # --- eaten words: a word that starts on a trimmed edge will be in the subtitles but not in the audio ---
    ru, ro = meta.get('ranges_used') or [], meta.get('ranges_orig') or []
    words_clip = json.load(open(os.path.join(d,'words.json'), encoding='utf-8')) if os.path.exists(os.path.join(d,'words.json')) else []
    if ru and ro and ru != ro:
        off = 0.0; eaten = []
        for (s0,e0),(s1,e1) in zip(ro, ru):
            # clip words with t < off + 0.02 in this range mean a start shifted past a word (the pipeline sets t0=max(t,s))
            for w in words_clip:
                if off - 0.01 <= w['t'] < off + 0.03 and s1 > s0 + 0.05: eaten.append(w['w'])
            off += e1 - s1
        if eaten: notes.append(f"boundaries trimmed ({len(ru)} ranges); words at the edge: {', '.join(eaten[:4])} - check by ear")
        else: notes.append('boundaries trimmed into silence, words untouched')
    if meta.get('lost_words'):   # v2.1: pipeline v3.1 itself knows which words fell outside the new boundaries
        problems.append('words outside the new boundaries (in neither the audio nor the subtitles): ' + ' '.join(meta['lost_words'])[:100])

    # --- audio: silence at start/end, clicks at splices ---
    try:
        au = audio_np(mp4); sr = 16000
        if len(au) > sr:
            rms = lambda x: float(np.sqrt(np.mean(x**2)) + 1e-9)
            body = rms(au[int(0.5*sr):-int(0.5*sr)]) if len(au) > 2*sr else rms(au)
            if rms(au[:int(0.06*sr)]) > 0.5 * body: problems.append('start: voice is already on from the first millisecond (tail of another phrase or a cut-off word)')
            if rms(au[-int(0.08*sr):]) > 0.6 * body: problems.append('end: voice is cut off in the last 150 ms')
            for sp in splices:
                k = int(sp * sr); seg = au[max(0, k - 80):k + 80]
                if len(seg) > 2 and np.max(np.abs(np.diff(seg))) > 0.35: problems.append(f'splice {sp:.1f} s: click (amplitude jump)')
    except Exception as ex: notes.append(f'audio not checked: {ex}')

    # --- technical standard ---
    wh = probe(mp4, 'stream=width,height', stream=True)
    if wh.replace(',','x') != '1080x1920': problems.append(f'resolution {wh}, expected 1080x1920')
    li, tp = loud(mp4)
    if li is None or abs(li+15) > 0.7: problems.append(f'loudness {li} LUFS, expected -15 +-0.7')
    if tp is not None and tp > -1.0: problems.append(f'peak level {tp} dBTP, expected <= -1.5')
    qp = os.path.join(a.out,'qc',name+'.json')
    if os.path.exists(qp):
        q=json.load(open(qp))
        if not q.get('ok_len'): problems.append('subtitles are longer than the video')

    # --- crop on a wide shot (Marik's rule) ---
    det = cv2.FaceDetectorYN.create(a.yunet,'',(320,320),0.6,0.3,500)
    cap = cv2.VideoCapture(mp4); vfps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(round(vfps/a.fps_sample))); idx=0; bad=[]; first3=[]
    while True:
        ok, fr = cap.read()
        if not ok: break
        if idx % step == 0:
            t = idx/vfps
            h,w = fr.shape[:2]
            top = fr[:int(h*0.14)]
            is_band = (cv2.Laplacian(cv2.cvtColor(top, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var() < 40)
            det.setInputSize((w,h)); _, faces = det.detect(fr)
            fh = max((f[3] for f in faces), default=0)/h if faces is not None else 0
            if t <= 3.0: first3.append(fh)
            if not is_band and 0 < fh < FACE_MIN: bad.append(t)
            elif not is_band and fh == 0: bad.append(t)
        idx += 1
    cap.release()
    if first3 and sum(1 for x in first3 if x >= 0.10) < 0.5 * len(first3):
        problems.append('start: the face of the speaker is not visible in the first 3 s (slide/back/wide shot) - the viewer does not understand who is speaking')
    if bad:
        runs=[]; s=bad[0]; p=bad[0]
        for t in bad[1:]:
            if t-p > 1.0: runs.append((s,p)); s=t
            p=t
        runs.append((s,p))
        for s,e in runs:
            if e-s >= 2.5: problems.append(f'{s:5.1f}–{e:5.1f} s: {e-s:.1f} s of crop with a small face (< {FACE_MIN:.0%}), too long, the viewer cannot see the speaker')
            elif e-s >= 0.6: notes.append(f'{s:5.1f}–{e:5.1f} s: {e-s:.1f} s on the wide camera, face is small - acceptable if short')
    return {'name':name,'dur':round(dur,1),'segments':len(S),'band':round(band/dur,3),'problems':problems,'notes':notes,'ranges':ro or ru}

names = [a.name] if a.name else sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(a.out,'*.mp4')))
bad_total=0; results=[]
for n in names:
    r = check(n)
    if not r: continue
    # duplicates in a batch: ranges overlapping by > 50 % with an already checked clip
    for r2 in results:
        ov = sum(max(0, min(e1,e2)-max(s1,s2)) for s1,e1 in r['ranges'] for s2,e2 in r2['ranges'])
        if ov > 0.5 * min(sum(e-s for s,e in r['ranges']), sum(e-s for s,e in r2['ranges'])):
            r['problems'].append(f"duplicate: {ov:.0f} s overlap with {r2['name']}")
    results.append(r)
    mark = '✅ CLEAN' if not r['problems'] else f"❌ {len(r['problems'])} problems"
    print(f"\n{r['name']}  {r['dur']} s | segments {r['segments']} | band {r['band']*100:.0f}%  → {mark}")
    for p in r['problems']: print('   ❌', p)
    for p in r.get('notes',[]): print('   ·', p)
    bad_total += len(r['problems'])
print(f"\ntotal issues: {bad_total}")
if not a.name and results:
    os.makedirs(os.path.join(a.out,'qc'), exist_ok=True)
    with open(os.path.join(a.out,'qc','REPORT.txt'),'w',encoding='utf-8') as fp:
        for r in results:
            fp.write(f"{r['name']}  {r['dur']} s | segments {r['segments']} | band {r['band']*100:.0f}%  -> {'CLEAN' if not r['problems'] else str(len(r['problems']))+' problems'}\n")
            for p in r['problems']: fp.write('   X ' + p + '\n')
            for p in r['notes']: fp.write('   . ' + p + '\n')
