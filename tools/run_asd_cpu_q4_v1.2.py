"""run_asd_cpu_q4_v1.2.py - ASD (active speaker detection, who is speaking) on CPU. Based on Light-ASD.
v1.2 (19.09): crop track by track, skipping finished ones, pad only past the edge; v1.1 (16.09): face-detection checkpoint.
Time budget per call: ASD_BUDGET (s), default 70.
Output in --save: pywork/{tracks.pckl,scores.pckl,faces.pckl}, pyavi/video.avi (25fps).
Run inside the cloned Light-ASD repository (so that import ASD works)."""
import sys, os, glob, pickle, subprocess, math, argparse
import numpy as np, cv2
from scipy import signal
from scipy.io import wavfile
from scipy.interpolate import interp1d
import python_speech_features
ap=argparse.ArgumentParser()
ap.add_argument('--src',required=True); ap.add_argument('--start',type=float,default=0)
ap.add_argument('--dur',type=float,default=0)  # 0 = whole file
ap.add_argument('--save',required=True)
ap.add_argument('--weight',default='weight/pretrain_AVA_CVPR.model'); ap.add_argument('--yunet',default='yunet.onnx')
ap.add_argument('--minTrack',type=int,default=10); ap.add_argument('--numFailedDet',type=int,default=10)
ap.add_argument('--minFaceSize',type=int,default=1); ap.add_argument('--cropScale',type=float,default=0.40)
a=ap.parse_args()
pyavi=os.path.join(a.save,'pyavi'); pyframes=os.path.join(a.save,'pyframes')
pywork=os.path.join(a.save,'pywork'); pycrop=os.path.join(a.save,'pycrop')
for d in (pyavi,pyframes,pywork,pycrop): os.makedirs(d,exist_ok=True)
videoF=os.path.join(pyavi,'video.avi'); audioF=os.path.join(pyavi,'audio.wav')
def sh(c): subprocess.call(c,shell=True,stdout=None,stderr=subprocess.DEVNULL)
dur=('' if a.dur<=0 else f'-t {a.dur}')
if not os.path.isfile(videoF):
    sh(f"ffmpeg -y -ss {a.start} {dur} -i {a.src} -qscale:v 2 -r 25 -async 1 {videoF} -loglevel panic")
if not os.path.isfile(audioF):
    sh(f"ffmpeg -y -i {videoF} -qscale:a 0 -ac 1 -vn -ar 16000 {audioF} -loglevel panic")
if len(glob.glob(os.path.join(pyframes,'*.jpg')))==0:
    sh(f"ffmpeg -y -i {videoF} -qscale:v 4 -f image2 {os.path.join(pyframes,'%06d.jpg')} -loglevel panic")
flist=sorted(glob.glob(os.path.join(pyframes,'*.jpg'))); print('frames:',len(flist),flush=True)
facesP=os.path.join(pywork,'faces.pckl')
if not os.path.isfile(facesP):
    # v1.1 (16.09): face-detection checkpoint: long clips do not finish in one call (limit 120 s)
    partP=os.path.join(pywork,'faces.part')
    dets=pickle.load(open(partP,'rb')) if os.path.isfile(partP) else []
    det=cv2.FaceDetectorYN.create(a.yunet,'',(320,320),0.6,0.3,5000)
    import time as _t; _t0=_t.time(); _lim=float(os.environ.get('ASD_BUDGET', os.environ.get('DET_BUDGET','70')))
    if dets: print('faces resume',len(dets),'/',len(flist),flush=True)
    for i,fn in enumerate(flist):
        if i<len(dets): continue
        if _t.time()-_t0>_lim:
            pickle.dump(dets,open(partP,'wb')); print('faces partial',len(dets),'/',len(flist),flush=True)
            print('DONE',a.save,flush=True); raise SystemExit
        img=cv2.imread(fn)
        if img is None:
            dets.append([]); continue
        h,w=img.shape[:2]; sm=cv2.resize(img,(w//2,h//2)); det.setInputSize((sm.shape[1],sm.shape[0]))
        _,fs=det.detect(sm); dets.append([])
        if fs is not None:
            for b in fs:
                x,y,wd,ht=b[0]*2,b[1]*2,b[2]*2,b[3]*2
                dets[-1].append({'frame':i,'bbox':[float(x),float(y),float(x+wd),float(y+ht)],'conf':float(b[-1])})
    pickle.dump(dets,open(facesP,'wb')); print('faces detected',flush=True)
    if os.path.isfile(os.path.join(pywork,'faces.part')): os.remove(os.path.join(pywork,'faces.part'))
else: dets=pickle.load(open(facesP,'rb'))
def iou(A,B):
    xA,yA=max(A[0],B[0]),max(A[1],B[1]); xB,yB=min(A[2],B[2]),min(A[3],B[3])
    inter=max(0,xB-xA)*max(0,yB-yA); aA=(A[2]-A[0])*(A[3]-A[1]); aB=(B[2]-B[0])*(B[3]-B[1])
    return inter/float(aA+aB-inter+1e-6)
def track_shot(sf):
    tracks=[]
    while True:
        track=[]
        for ff in sf:
            for f in ff:
                if track==[]: track.append(f); ff.remove(f)
                elif f['frame']-track[-1]['frame']<=a.numFailedDet:
                    if iou(f['bbox'],track[-1]['bbox'])>0.5: track.append(f); ff.remove(f); continue
                else: break
        if track==[]: break
        elif len(track)>a.minTrack:
            fn=np.array([f['frame'] for f in track]); bb=np.array([f['bbox'] for f in track])
            fi=np.arange(fn[0],fn[-1]+1); bi=[]
            for j in range(4): bi.append(interp1d(fn,bb[:,j])(fi))
            bi=np.stack(bi,axis=1)
            if max(np.mean(bi[:,2]-bi[:,0]),np.mean(bi[:,3]-bi[:,1]))>a.minFaceSize: tracks.append({'frame':fi,'bbox':bi})
    return tracks
tracksP=os.path.join(pywork,'tracks.pckl')
if not os.path.isfile(tracksP):
    # v1.2 (19.09): crop track by track, skipping finished ones (.avi), resumable between calls; frame pad once per track frame
    import time as _t; _t0=_t.time(); _lim=float(os.environ.get('ASD_BUDGET', os.environ.get('DET_BUDGET','70')))
    allT=track_shot([list(f) for f in dets]); print('tracks:',len(allT),flush=True); vid=[]
    procs=[]
    for ii,tr in enumerate(allT):
        d={'x':[],'y':[],'s':[]}
        for det in tr['bbox']:
            d['s'].append(max((det[3]-det[1]),(det[2]-det[0]))/2); d['y'].append((det[1]+det[3])/2); d['x'].append((det[0]+det[2])/2)
        d['s']=signal.medfilt(d['s'],13); d['x']=signal.medfilt(d['x'],13); d['y']=signal.medfilt(d['y'],13)
        procs.append(d)
    for ii,tr in enumerate(allT):
        cf=os.path.join(pycrop,'%05d'%ii); d=procs[ii]
        if os.path.isfile(cf+'.avi') and os.path.isfile(cf+'.wav'):
            vid.append({'track':tr,'proc_track':d}); continue
        if _t.time()-_t0>_lim:
            print('crop partial',ii,'/',len(allT),flush=True); print('DONE',a.save,flush=True); raise SystemExit
        vOut=cv2.VideoWriter(cf+'t.avi',cv2.VideoWriter_fourcc(*'XVID'),25,(224,224)); cs=a.cropScale
        for fidx,frame in enumerate(tr['frame']):
            bs=d['s'][fidx]; bsi=int(bs*(1+2*cs))
            img0=cv2.imread(flist[int(frame)]); h0,w0=img0.shape[:2]
            my=d['y'][fidx]; mx=d['x'][fidx]
            y0=int(my-bs); y1=int(my+bs*(1+2*cs)); x0=int(mx-bs*(1+cs)); x1=int(mx+bs*(1+cs))
            # pad only as much as extends past the edge (instead of np.pad on the whole 1440p frame)
            py0=max(0,-y0); px0=max(0,-x0); py1=max(0,y1-h0); px1=max(0,x1-w0)
            face=img0[max(0,y0):min(h0,y1), max(0,x0):min(w0,x1)]
            if py0 or px0 or py1 or px1: face=cv2.copyMakeBorder(face,py0,py1,px0,px1,cv2.BORDER_CONSTANT,value=(110,110,110))
            vOut.write(cv2.resize(face,(224,224)))
        vOut.release()
        aS=tr['frame'][0]/25; aE=(tr['frame'][-1]+1)/25
        sh(f"ffmpeg -y -i {audioF} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -ss {aS:.3f} -to {aE:.3f} {cf}.wav -loglevel panic")
        sh(f"ffmpeg -y -i {cf}t.avi -i {cf}.wav -c:v copy -c:a copy {cf}.avi -loglevel panic"); os.remove(cf+'t.avi')
        vid.append({'track':tr,'proc_track':d})
    pickle.dump(vid,open(tracksP,'wb')); print('cropped',flush=True)
else: vid=pickle.load(open(tracksP,'rb'))
scoresP=os.path.join(pywork,'scores.pckl')
if not os.path.isfile(scoresP):
    import time as _t; _t0=_t.time(); _lim=float(os.environ.get('ASD_BUDGET', os.environ.get('DET_BUDGET','70')))
    sys.argv=['x']; from ASD import ASD; import torch
    dev='cuda' if torch.cuda.is_available() else 'cpu'; print('ASD device:',dev,flush=True)
    s=ASD(); s.loadParameters(a.weight); s.eval()
    try: s.model.to(dev); s.lossAV.to(dev)
    except Exception as e: print('to(dev) warn',e,flush=True)
    files=sorted(glob.glob(os.path.join(pycrop,'*.avi'))); durationSet=[1,2,3]; done=0
    for f in files:
        fn=os.path.splitext(os.path.basename(f))[0]; cP=os.path.join(pywork,'score_'+fn+'.npy')
        if os.path.isfile(cP): done+=1; continue
        _,audio=wavfile.read(os.path.join(pycrop,fn+'.wav')); aF=python_speech_features.mfcc(audio,16000,numcep=13,winlen=0.025,winstep=0.010)
        capv=cv2.VideoCapture(os.path.join(pycrop,fn+'.avi')); vF=[]
        while capv.isOpened():
            r,fr=capv.read()
            if not r: break
            g=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY); g=cv2.resize(g,(224,224)); g=g[56:168,56:168]; vF.append(g)
        capv.release(); vF=np.array(vF)
        length=min((aF.shape[0]-aF.shape[0]%4)/100,vF.shape[0]); aF2=aF[:int(round(length*100)),:]; vFx=vF[:int(round(length*25)),:,:]
        allScore=[]
        for dur_ in durationSet:
            bs=int(math.ceil(length/dur_)); sc=[]
            with torch.no_grad():
                for i in range(bs):
                    ia=torch.FloatTensor(aF2[i*dur_*100:(i+1)*dur_*100,:]).unsqueeze(0).to(dev)
                    iv=torch.FloatTensor(vFx[i*dur_*25:(i+1)*dur_*25,:,:]).unsqueeze(0).to(dev)
                    ea=s.model.forward_audio_frontend(ia); ev=s.model.forward_visual_frontend(iv)
                    out=s.model.forward_audio_visual_backend(ea,ev); sc.extend(s.lossAV.forward(out))
            allScore.append(sc)
        ml=min(len(z) for z in allScore); sc=np.round(np.mean(np.array([z[:ml] for z in allScore]),axis=0),1).astype(float)
        np.save(cP,sc); done+=1; print('scored',fn,'len',len(sc),flush=True)
        if _t.time()-_t0>_lim and done<len(files): print('progress',done,'/',len(files),'(budget)',flush=True); print('DONE',a.save,flush=True); raise SystemExit
    if done==len(files):
        allScores=[np.load(os.path.join(pywork,'score_'+os.path.splitext(os.path.basename(f))[0]+'.npy')) for f in files]
        pickle.dump(allScores,open(scoresP,'wb')); print('ASD scores saved, tracks=',len(files),flush=True)
    else: print('progress',done,'/',len(files),flush=True)
else: print('scores cached',flush=True)
print('DONE',a.save,flush=True)
