import re,json,sys
raw=open(sys.argv[1],encoding='utf-8').read()
def to_s(t):
    h,m,s=t.split(':'); return int(h)*3600+int(m)*60+float(s)
blocks=re.split(r'\n\n+',raw); cues=[]
for b in blocks:
    m=re.search(r'(\d\d:\d\d:\d\d\.\d\d\d) --> (\d\d:\d\d:\d\d\.\d\d\d)',b)
    if not m: continue
    a,e=to_s(m.group(1)),to_s(m.group(2))
    if e-a<0.4: continue
    body=b[m.end():].split('\n',1)[-1]
    t=re.sub(r'<[^>]+>','',body)
    t=t.replace('&gt;',' ').replace('&lt;',' ').replace('&nbsp;',' ').replace('&amp;','&')
    t=re.sub(r'\[[^\]]*\]',' ',t).replace('>>',' ')
    t=re.sub(r'\s+',' ',t).strip()
    if t: cues.append((a,e,t))
words=[]; prev=[]
for a,e,txt in cues:
    w=txt.split(' '); k=0; mk=min(len(prev),len(w))
    for kk in range(mk,0,-1):
        if prev[-kk:]==w[:kk]: k=kk; break
    new=w[k:]
    for i,ww in enumerate(new): words.append([a+(e-a)*(i+0.5)/max(1,len(new)), ww])
    prev=w
tags=[(to_s(t),s.strip()) for t,s in re.findall(r'<(\d\d:\d\d:\d\d\.\d\d\d)><c>\s*([^<]+)</c>',raw) if s.strip()]
def nm(s): return re.sub(r"[^\w'-]",'',s.lower())
j=0; last=-1
for wd in words:
    n0=nm(wd[1]); f=None
    for k2 in range(j,min(len(tags),j+8)):
        if tags[k2][0]>last and nm(tags[k2][1])==n0: f=tags[k2][0]; j=k2+1; break
    wd[0]=f if f is not None else max(wd[0],last+0.05); last=wd[0]
json.dump([{'t':round(t,2),'w':w} for t,w in words],open(sys.argv[2],'w'),ensure_ascii=False)
print('слів',len(words),'до',round(words[-1][0]),'с')
