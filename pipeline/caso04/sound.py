import json, numpy as np, soundfile as sf
from scipy import signal
D='/tmp/claude-0/-home-claude/93433413-bce9-5cb9-9820-1a8dd907373e/scratchpad/caso4/'
SR=44100; DUR=100.1; N=int(SR*DUR)
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
def ticks(a,b,step,f0,f1,g=.45):
    for tt_ in np.arange(a,b,step): ev(tick(f0+(f1-f0)*((tt_-a)/max(1e-6,(b-a))),.04),tt_,g)
def scrape(d=.12): t=tt(d); return bp(noise(d),500,3500,1)*np.exp(-t*20)*.9
ev(stamp(),0.15,1.0)
ev(swell(1.8),wt(0,0)+.2,.4)
ev(pop(420),wt(1,1),.7); ev(pop(380),wt(1,1)+.15,.7); ev(thump(55,1.2),wt(1,1)+.05,.6)
# B
ev(paper(),wt(2,0),.8); ev(pop(520),wt(2,0)+.05,.6); ev(stamp(1.1),wt(2,5),.9); ev(pop(560),wt(3,0),.7); ev(ping(660,1.4,.35),wt(3,0)+.1,.5)
# C
ev(paper(),wt(4,0)-.05,.9); ev(thump(60,.7),wt(4,0),.7); ev(paper(),wt(4,4)-.05,.9); ev(thump(60,.7),wt(4,4),.7); ev(pop(600),wt(4,12),.6)
# D
ev(stamp(.9),wt(5,1),.8); ev(paper(),wt(5,8)-.05,.9); ev(thump(56,.7),wt(5,8),.6)
ev(whoosh(1.0,300,2000,.3),ss(6),.5)
for tt_ in np.arange(ss(6),se(6),.35): ev(tick(300,.05),tt_,.35); ev(scrape(.08),tt_+.17,.25)
ev(pop(520),wt(7,2),.7); ev(pop(560),wt(7,2)+.15,.7); ev(pop(640),wt(7,7),.7)
# E
ev(slide(.5),ss(8),.5); ev(pop(500),ss(8)+.1,.7)
ticks(wt(9,1),wt(9,6),.12,900,900,.45)
ev(thump(52,1.0),wt(9,0),.8); ev(stamp(1.2),wt(10,3),1.0)
# F1
ev(whoosh(1.0,200,1500,.3),ss(11),.5); ev(pop(520),ss(11)+.05,.6); ev(pop(480),wt(11,8),.7); ev(ping(440,1.8,.4),wt(11,8)+.05,.5)
# F2
ev(paper(),wt(12,0)-.05,.9); ev(thump(60,.6),wt(12,0),.6); ev(paper(),wt(12,1)-.05,.9); ev(paper(),wt(12,6)-.05,.9); ev(thump(70,.9),wt(12,6),.8); ev(pop(520),wt(12,12),.5)
ev(stamp(1.0),wt(13,0),.9)
# H
ev(slide(.6),wt(14,2)-.1,.7); ev(paper(.4),wt(14,2),.8)
for s_ in (15,16,17,18):
    for k in range(10): ev(scrape(.06),ss(s_)+.05+k*.05,.3)
ev(swell(2.0),ss(18)-.2,.5); ev(boom(1.8),wt(18,2),.7)
# I
ev(pop(560),ss(19),.8); ev(ping(880,1.6,.45),ss(19)+.1,.6)
ev(sonar(),ss(20)+.05,.7); ev(sonar(),ss(20)+.7,.5)
# J
ev(pop(520),wt(21,3),.7); ev(thump(60,.8),wt(22,3),.9); ev(stamp(.9),wt(22,3)+.05,.7); ev(pop(480),wt(22,8),.6)
# K
ev(stamp(1.4),wt(23,4),1.0); ev(boom(2.2),wt(23,4),.7)
# L
ev(pop(520),ss(24)+.1,.6)
for i in range(3): ev(pop(440+i*70),wt(24,8)+i*.15,.7)
ev(ping(784,2.2,.4),wt(24,12),.6)
# M
ev(swell(2.0),wt(25,4)-1.0,.5); ev(ping(1046,2.4,.5),wt(25,4),.8); ev(pop(520),wt(25,6),.7)
# N
ev(stamp(1.4),wt(26,2),1.0)
# P
ev(swell(1.6),wt(27,6)-.9,.5); ev(pop(520),wt(27,6),.8); ev(ding(784,2.6),wt(27,6)+.05,.5)
# Q
ev(pop(520),ss(28),.6)
for i in range(3): ev(pop(520+i*90),wt(28,5)+i*.45,.8)
ev(pop(600),wt(29,0)-.1,.8)
ev(ding(988,2.8),TM['dur']+LEAD-OFF+.45,.8)
sfx=np.stack([L,R],1)

# ---------- música ----------
Lm=np.zeros(N);Rm=np.zeros(N)
t=np.arange(N)/SR
chords=[(0,[55,82.4,110,130.8,164.8]),(12,[43.65,65.4,87.3,110,130.8]),(24,[36.7,73.4,110,146.8,174.6]),(36,[55,82.4,110,130.8,164.8]),(48,[41.2,82.4,123.5,155.6,164.8]),(60,[55,82.4,110,130.8,164.8]),(72,[43.65,65.4,87.3,110,130.8]),(84,[55,82.4,110,130.8,164.8]),(92,[41.2,82.4,123.5,155.6,164.8])]
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
def amp_at(x): return .25+.75*(1 if (ss(15)<=x<=se(20) or ss(23)<=x<=se(25)) else 0)
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
