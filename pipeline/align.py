import re, json, subprocess, math, sys
D = 'WORK/'
out = subprocess.run(['ffmpeg','-hide_banner','-i',D+'voz.mp3','-af','silencedetect=noise=-35dB:d=0.3','-f','null','-'],capture_output=True,text=True).stderr
starts=[float(x) for x in re.findall(r'silence_start: ([\d.]+)',out)]
ends=[float(x) for x in re.findall(r'silence_end: ([\d.]+)',out)]
TOTAL=86.7527
# speech segments
segs=[]; cur=0.0
gaps=[]
for i,e in enumerate(ends):
    s=starts[i]
    if s>cur+0.05: segs.append((cur,s))
    cur=e
if cur<TOTAL-0.05: segs.append((cur,TOTAL))
print('segs',len(segs),segs[:3],segs[-2:])

SENT = [
"Um homem entrou num avião com uma passagem comprada em dinheiro.",
"Saiu dele de paraquedas, com duzentos mil dólares.",
"E nunca mais foi visto.",
"Vinte e quatro de novembro de mil novecentos e setenta e um.",
"Voo trezentos e cinco da Northwest Orient, de Portland para Seattle.",
"O passageiro se identificou como Dan Cooper.",
"Terno escuro, gravata preta, camisa branca.",
"Pediu um bourbon com soda.",
"Então entregou um bilhete à comissária.",
"Dizia que ele tinha uma bomba na maleta.",
"As exigências eram duas.",
"Duzentos mil dólares em notas de vinte.",
"E quatro paraquedas.",
"Em Seattle, o avião pousou.",
"Trinta e seis passageiros foram liberados.",
"O dinheiro e os paraquedas, entregues.",
"Cooper mandou decolar de novo.",
"Por volta das oito da noite, em plena tempestade, ele saltou.",
"Em algum ponto entre Seattle e Reno.",
"Ninguém viu onde ele caiu.",
"Deixou para trás a gravata, bitucas de cigarro, e um único fio de cabelo.",
"Em mil novecentos e oitenta, um menino de oito anos, Brian Ingram, encontrou cinco mil e oitocentos dólares na margem do rio Columbia.",
"Eram notas do resgate.",
"E foi só isso que apareceu.",
"Nunca mais se achou nada.",
"Em cinco anos, o FBI investigou mais de oitocentos suspeitos.",
"Em dois mil e dezesseis, encerrou a investigação ativa.",
"O caso continua sem solução.",
"Você acha que ele sobreviveu?",
"Comenta aqui.",
]
def syl(w):
    w=w.lower()
    w=re.sub(r'[^a-zàáâãéêíóôõúç]','',w)
    n=len(re.findall(r'[aeiouàáâãéêíóôõú]+',w))
    return max(1,n)
words=[]  # (sent idx, word)
for i,s in enumerate(SENT):
    for w in s.split():
        words.append((i,w))
sw=[syl(w)+(0.6 if w[-1] in '.?!' else 0.0) for _,w in words]
# sentence syllable sums
ns=len(SENT)
ssyl=[sum(sw[k] for k,(i,_) in enumerate(words) if i==j) for j in range(ns)]
tot=sum(ssyl)
# speech-time axis
cum=[0.0]
for s,e in segs: cum.append(cum[-1]+e-s)
SP=cum[-1]
pace0=SP/tot
def sp_of(t):
    for k,(s,e) in enumerate(segs):
        if t<=s: return cum[k]
        if t<=e: return cum[k]+t-s
    return SP
def t_of(sp):
    for k,(s,e) in enumerate(segs):
        if sp<=cum[k+1]+1e-9: return s+max(0,sp-cum[k])
    return segs[-1][1]
# gap positions in speech-time: gap k between seg k and k+1
gp=[cum[k+1] for k in range(len(segs)-1)]
# DP over anchors: boundary j (after sentence j, j=0..ns-2) to gap g
# state (j,g) ; start=(-1,-1 as sp 0) end=(ns-1,len(gp) as sp SP)
csyl=[0]
for x in ssyl: csyl.append(csyl[-1]+x)
def cost_int(j0,sp0,j1,sp1):
    syls=csyl[j1+1]-csyl[j0+1]
    dur=sp1-sp0
    if syls<=0 or dur<=0.05: return 1e9
    r=math.log(dur/(syls*pace0))
    return (r*r)/(2*0.28**2)
REWARD=1.2
INF=1e18
G=len(gp)
best={}
best[(-1,-1)]=(0.0,None)
nodes=[(-1,-1)]
order=[(j,g) for j in range(ns-1) for g in range(G)]
sp_node=lambda n:(0.0 if n==(-1,-1) else gp[n[1]])
for n in order:
    j,g=n
    bc=INF;bp=None
    for (j0,g0),(c0,_) in list(best.items()):
        if j0<j and g0<g:
            c=c0+cost_int(j0,sp_node((j0,g0)),j,gp[g])-REWARD
            if c<bc: bc=c;bp=(j0,g0)
    if bp: best[n]=(bc,bp)
# end
bc=INF;bp=None
for (j0,g0),(c0,_) in best.items():
    c=c0+cost_int(j0,sp_node((j0,g0)),ns-1,SP)
    if c<bc: bc=c;bp=(j0,g0)
path=[]
n=bp
while n and n!=(-1,-1):
    path.append(n); n=best[n][1]
path=path[::-1]
print('anchors',[(j,round(t_of(gp[g]),2)) for j,g in path])
# boundaries (speech-time) per sentence: sentence j end
bsp=[None]*ns
for j,g in path: bsp[j]=gp[g]
bsp[ns-1]=SP
# fill: interpolate unanchored boundaries proportionally between anchors
last_j=-1;last_sp=0.0
for j in range(ns):
    if bsp[j] is not None:
        # fill between
        for k in range(last_j+1,j):
            frac=(csyl[k+1]-csyl[last_j+1])/(csyl[j+1]-csyl[last_j+1])
            bsp[k]=last_sp+frac*(bsp[j]-last_sp)
        last_j=j;last_sp=bsp[j]
# words
res=[];sentout=[]
start_sp=0.0
k=0
for j in range(ns):
    ws=[(w,sw[kk]) for kk,(i,w) in enumerate(words) if i==j]
    tot_s=sum(x for _,x in ws)
    a=start_sp;b=bsp[j]
    c=a
    for w,x in ws:
        d=(b-a)*x/tot_s
        t0=t_of(c); t1=t_of(c+d)
        res.append({'w':w,'s':j,'t0':round(t0,3),'t1':round(t1,3)})
        c+=d
    sentout.append({'t0':res[-len(ws)]['t0'],'t1':res[-1]['t1']})
    start_sp=b
json.dump({'words':res,'sents':sentout,'dur':TOTAL,'segs':segs},open(D+'timing.json','w'),ensure_ascii=False)
for j,s in enumerate(sentout): print(j,round(s['t0'],2),round(s['t1'],2),SENT[j][:40])
