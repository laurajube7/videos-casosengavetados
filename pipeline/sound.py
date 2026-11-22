import json, numpy as np, soundfile as sf
from scipy import signal
D='WORK/'
SR=44100; DUR=89.8; N=int(SR*DUR)
LEAD=0.6; OFF=0.18           # tempo da animação -> tempo do vídeo final = t_anim - OFF
TM=json.load(open(D+'timing.json'))
BS={}
for w in TM['words']: BS.setdefault(w['s'],[]).append(w)
def wt(j,k): return BS[j][min(k,len(BS[j])-1)]['t0']+LEAD-OFF
def ss(j): return TM['sents'][j]['t0']+LEAD-OFF
def se(j): return TM['sents'][j]['t1']+LEAD-OFF
rng=np.random.default_rng(7)
def tt(d): return np.arange(int(SR*d))/SR
def put(buf,x,t,g=1.0,pan=0.0):
    i=int(t*SR)
    if i<0 or i>=len(buf[0]): return
    x=x[:len(buf[0])-i]
    buf[0][i:i+len(x)]+=x*g*(1-max(0,pan)); buf[1][i:i+len(x)]+=x*g*(1+min(0,pan))
def lp(x,fc,o=2): b,a=signal.butter(o,fc/(SR/2),'low'); return signal.lfilter(b,a,x)
def hp(x,fc,o=2): b,a=signal.butter(o,fc/(SR/2),'high'); return signal.lfilter(b,a,x)
def bp(x,f1,f2,o=2): b,a=signal.butter(o,[f1/(SR/2),f2/(SR/2)],'band'); return signal.lfilter(b,a,x)
def noise(d): return rng.standard_normal(int(SR*d))
def reverb(x,rt=1.6,wet=.35):
    n=int(SR*rt); t=np.arange(n)/SR
    ir=rng.standard_normal(n)*np.exp(-t*4.2/rt); ir=lp(ir,4500,1)
    ir/=np.sqrt((ir**2).sum())
    y=signal.fftconvolve(x,ir)[:len(x)]
    return x*(1-wet)+y*wet*1.4

# ---------- efeitos ----------
def thump(f0=70,d=.9):
    t=tt(d); f=f0*(1+2.2*np.exp(-t*18)); ph=2*np.pi*np.cumsum(f)/SR
    return np.sin(ph)*np.exp(-t*5.5)+lp(noise(d),600)*np.exp(-t*30)*.35
def stamp(d=1.1):
    x=thump(62,d)*1.0+bp(noise(d),1500,5000)*np.exp(-tt(d)*45)*.5
    return reverb(x,.9,.25)
def tick(f=1800,d=.05): t=tt(d); return np.sin(2*np.pi*f*t)*np.exp(-t*90)*.5+bp(noise(d),2000,6000)*np.exp(-t*120)*.25
def pop(f=520,d=.18): t=tt(d); return np.sin(2*np.pi*f*(1+.5*np.exp(-t*40))*t)*np.exp(-t*24)*.7
def whoosh(d=1.4,f1=300,f2=3500,peak=.5):
    n=noise(d); t=tt(d); env=np.sin(np.pi*np.clip(t/d,0,1))**2
    out=np.zeros_like(n)
    seg=40
    for k in range(seg):
        a=int(len(n)*k/seg); b=int(len(n)*(k+1)/seg)
        fc=f1*(f2/f1)**(np.sin(np.pi*(k/seg))**1.0)
        out[a:b]=bp(n[a:b+1][:b-a] if False else n[a:b],max(80,fc*.6),min(18000,fc*1.4),1)
    return out*env*peak*2.2
def paper(d=.35): t=tt(d); return bp(noise(d),1200,7000)*np.exp(-t*14)*np.sin(np.pi*np.clip(t/d,0,1))**.5*.8
def ping(f=1318,d=1.6,g=.5): t=tt(d); return (np.sin(2*np.pi*f*t)+.35*np.sin(2*np.pi*f*2.01*t))*np.exp(-t*3.2)*g
def sonar(d=1.8): t=tt(d); return np.sin(2*np.pi*1046*t)*np.exp(-t*2.6)*.55
def boom(d=2.6):
    t=tt(d); f=48*(1+1.5*np.exp(-t*6)); ph=2*np.pi*np.cumsum(f)/SR
    return reverb(np.sin(ph)*np.exp(-t*1.7)*.9+lp(noise(d),300)*np.exp(-t*3)*.4,1.6,.3)
def thunder(d=3.4):
    t=tt(d); n=lp(noise(d),240,2)*3; env=(1-np.exp(-t*25))*np.exp(-t*1.1)*(0.6+0.4*np.abs(np.sin(t*7)))
    crack=bp(noise(.25),800,6000)*np.exp(-tt(.25)*30)
    x=n*env; x[:len(crack)]+=crack*1.2
    return reverb(x,1.4,.25)*.8
def swell(d=2.4):
    t=tt(d); n=bp(noise(d),200,2500,2)*(t/d)**3*.7
    s=np.sin(2*np.pi*(110+90*(t/d))*t)*(t/d)**3*.4
    return n+s
def fall(d=1.9):
    t=tt(d); n=noise(d); out=np.zeros_like(n); seg=30
    for k in range(seg):
        a=int(len(n)*k/seg); b=int(len(n)*(k+1)/seg)
        fc=2600*(1-k/seg*.8)+200
        out[a:b]=bp(n[a:b],fc*.7,fc*1.3,1)
    return out*np.exp(-t*1.6)*1.6*(0.4+0.6*(1-t/d))
def slide(d=.6): t=tt(d); return lp(noise(d),1800,1)*np.sin(np.pi*np.clip(t/d,0,1))**2*.45
def ding(f=988,d=2.2): t=tt(d); return (np.sin(2*np.pi*f*t)+.4*np.sin(2*np.pi*f*3*t))*np.exp(-t*2.4)*.45

L=np.zeros(N);R=np.zeros(N);S=(L,R)
def ev(x,t,g=1.0,pan=0.0): put(S,x,t,g,pan)

ev(stamp(),0.15,1.0)                              # carimbo da capa
ev(whoosh(1.8,250,3000,.5),0.9-OFF+.2,.9)          # avião entra
ev(ping(1318,1.4,.5),wt(1,5),.8); ev(ping(1760,1.2,.4),wt(1,5)+.12,.6)   # dinheiro
ev(boom(),wt(2,2),1.0); ev(swell(1.6),wt(2,2)-1.0,.5)
for j,k in [(3,0),(3,4),(3,6)]: ev(stamp(.8),wt(j,k),.75)
ev(whoosh(1.5,300,2500,.4),wt(4,7),.7); ev(ping(880,1.2,.4),wt(4,8),.5); ev(ping(1175,1.2,.4),wt(4,10),.5)
ev(boom(1.8),wt(5,5),.8)
for j,k in [(6,0),(6,2),(6,4),(7,2)]: ev(slide(.4),wt(j,k),.6); ev(tick(1200,.07),wt(j,k)+.25,.8)
ev(whoosh(1.2,500,3000,.4),wt(8,1),.7); ev(paper(),wt(9,0),.9)
ev(boom(2.4),wt(9,5),1.0); ev(swell(1.8),wt(9,5)-1.2,.55)
# contador de dinheiro: tiques rápidos
a,b=wt(11,0),wt(11,2)+.2
for tt_ in np.arange(a,b,.045): ev(tick(1500+800*((tt_-a)/(b-a)),.04),tt_,.5)
ev(ping(1568,1.4,.5),b+.05,.9)
ev(slide(.5),wt(12,1)-.1,.6)
for i in range(4): ev(pop(420+i*90),wt(12,1)+.1+i*.16,.9)
ev(stamp(.9),wt(13,0),.7); ev(whoosh(1.4,200,2200,.4),ss(13)+.2,.6)
a=wt(14,0); b=se(14)
for i in range(36): ev(tick(1700+i*20,.04),a+(b-a)*(i/36*.9),.35)
ev(ping(1175,1.5,.5),wt(14,3),.7)
ev(slide(.6),ss(15),.7); ev(ping(1568,1.4,.5),wt(15,5),.7)
# tempestade
rainlen=se(18)-ss(16)
r=bp(noise(rainlen),2500,9000,1)*.16; r*=np.minimum(1,np.minimum(tt(rainlen)/1.2,(rainlen-tt(rainlen))/1.2)); ev(r,ss(16),1.0)
for tf in [wt(17,3)+.1,wt(17,7)+.2,wt(17,10)-.05,wt(18,2)]:
    ev(thunder(3.6),tf-OFF+OFF,1.0,pan=rng.uniform(-.4,.4))
ev(fall(2.0),wt(17,9),.9)
ev(ping(660,1.6,.4),wt(17,3),.5)
for i,tf in enumerate([wt(18,4),wt(18,6)]): ev(pop(500+i*120),tf,.8)
for i in range(3): ev(sonar(),ss(19)+i*1.8,.55-i*.1)
for k in [3,5,10]: ev(paper(.3),wt(20,k),.9); ev(tick(900,.06),wt(20,k)+.1,.5)
ev(stamp(.9),wt(21,1),.7); ev(slide(.4),wt(21,5),.6)
a,b=wt(21,13),wt(21,17)+.2
for tt_ in np.arange(a,b,.05): ev(tick(1400,.04),tt_,.4)
ev(ping(1318,1.4,.45),b+.02,.7)
ev(pop(600),wt(22,3),.8); ev(boom(2.2),wt(23,3),.9)
a,b=wt(25,5),wt(25,8)+.3
for tt_ in np.arange(a,b,.04): ev(tick(1300+900*((tt_-a)/(b-a)),.04),tt_,.45)
ev(ping(1046,1.4,.4),b,.6)
ev(stamp(1.1),ss(26)+.15,.8); ev(stamp(1.6),wt(26,8)-.05+.2,1.0)
ev(slide(.7),ss(27),.8); ev(boom(2.6),wt(27,4),1.0)
ev(swell(1.4),ss(28)-.9,.45); ev(ping(784,2.0,.4),ss(28),.6)
ev(pop(520),wt(28,2),.8); ev(pop(420),wt(28,2)+.12,.8)
ev(ding(988,2.8),TM['dur']+LEAD-OFF+.45,.8)
sfx=np.stack([L,R],1)

# ---------- música ----------
Lm=np.zeros(N);Rm=np.zeros(N)
t=np.arange(N)/SR
chords=[(0,[55,82.4,110,130.8,164.8]),(15,[43.65,65.4,87.3,110,130.8]),(30,[36.7,73.4,110,146.8,174.6]),
        (45,[55,82.4,110,130.8,164.8]),(60,[41.2,82.4,123.5,155.6,164.8]),(75,[55,82.4,110,130.8,164.8])]
pad=np.zeros(N)
for ci,(st,fs) in enumerate(chords):
    en=chords[ci+1][0] if ci+1<len(chords) else DUR+3
    env=np.clip((t-st)/3,0,1)*np.clip((en+3-t)/3,0,1)
    for f in fs:
        for dt in (-.4,.0,.4):
            pad+=np.sin(2*np.pi*(f+dt)*t+rng.uniform(0,6.28))*env*(0.5/len(fs))
pad+=0.0
pad=lp(pad,900,2)*(0.75+0.25*np.sin(2*np.pi*t/9))
# batida grave sutil, mais presente nas cenas de tensão
beat=np.zeros(N)
def amp_at(x): return .25+.75*(1 if (43<=x<=60 or 73<=x<=84) else 0)
x0=0.0
while x0<DUR-1:
    h=thump(58,.5)*.5; i=int(x0*SR); j=min(N,i+len(h)); beat[i:j]+=h[:j-i]*amp_at(x0)*.5
    h2=thump(52,.4)*.35; i2=int((x0+.28)*SR); j2=min(N,i2+len(h2)); beat[i2:j2]+=h2[:j2-i2]*amp_at(x0)*.5
    x0+=1.15
# sinos esparsos
bells=np.zeros(N); notes=[440,523.3,659.3,784,880,587.3]
x0=4.0
while x0<DUR-4:
    f=notes[rng.integers(len(notes))]; b=ping(f,3.0,.18); i=int(x0*SR); j=min(N,i+len(b)); bells[i:j]+=b[:j-i]
    x0+=rng.uniform(3.2,5.6)
music=pad*.9+beat*.8+reverb(bells,2.0,.6)
music=reverb(music,1.8,.25)
fin=np.clip((t)/3,0,1)*np.clip((DUR-t)/3.2,0,1)
music*=fin
music/=np.sqrt((music**2).mean())+1e-9

# ---------- voz ----------
v,_=sf.read(D+'voz.wav'); vo=np.zeros(N); i0=int((LEAD-OFF)*SR); vo[i0:i0+len(v)]=v[:N-i0]
envv=np.sqrt(np.maximum(signal.fftconvolve(vo**2,np.ones(int(.08*SR))/int(.08*SR))[:N],0))
envv=np.maximum(lp(envv,6,1),0); envv=np.clip(envv/ (np.percentile(envv,95)+1e-9),0,1)
duck=1-.55*envv
vrms=np.sqrt((vo**2).mean())
mus=music*vrms*10**(-18/20)*duck           # música ~ -20 dB abaixo da voz, e abaixa quando ela fala
sfxg=10**(-3.5/20)
sduck=(1-.45*envv)[:,None]
mix=np.stack([vo+mus,vo+mus],1)+sfx*sfxg*vrms*2.7*sduck
mix[:,0]+=0;mix[:,1]+=0
pk=np.abs(mix).max(); mix=mix/pk*.89
sf.write(D+'mix.wav',mix,SR)
print('ok',round(pk,2),'voz rms',round(vrms,4),'music rms',round(np.sqrt((mus**2).mean()),4))
