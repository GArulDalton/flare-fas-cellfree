"""Overhead-aware port sounding: Table 5 and Fig. 10."""
import numpy as np, json
import sys, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flarefas.system import *
from flarefas import model as flare
ap = argparse.ArgumentParser(description='Port-subsampled sounding (Table 5) and overhead-aware effective SE (Fig. 10).')
ap.add_argument('--weights', default=str(ROOT / 'weights' / 'flare_params.npz'))
ap.add_argument('--batches', type=int, default=4, help='batches of 60 realizations per N_o')
args = ap.parse_args()
FIG = ROOT / 'figures'; RES = ROOT / 'results'; FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
P, _ = flare.load_params(args.weights)
rng=np.random.default_rng(21); L,K,N=16,6,16
res={}
for No in [1,2,4,8,16]:
    acc={m:[] for m in ['Max-gain','Greedy-CD','FLARE','FLARE-LS']}
    for b in range(args.batches):
        H,Hh,_=gen_batch_sub(rng,60,L,K,N,No)
        sels={'Max-gain':sel_maxgain(Hh),'Greedy-CD':sel_greedy(Hh),'FLARE':flare.sel_flare(P,Hh),'FLARE-LS':flare.sel_flare_ls(P,Hh)}
        for m,p in sels.items(): acc[m].append(rates(select(H,p),rzf(select(Hh,p))).sum(1))
    res[No]={m:float(np.concatenate(v).mean()) for m,v in acc.items()}
    print(No,{m:round(v,2) for m,v in res[No].items()},flush=True)
# conventional single fixed antenna (only the active port sounded)
acc=[]
for b in range(args.batches):
    H,Hh,_=gen_batch_sub(rng,60,L,K,1,1)
    z=np.zeros((60,L),int); acc.append(rates(select(H,z),rzf(select(Hh,z))).sum(1))
res['fixed']=float(np.concatenate(acc).mean()); print('fixed',res['fixed'])
json.dump(res, open(RES / 'overhead_sounding.json', 'w'), indent=1)

# ---------------- effective SE after pilot overhead (Eq. 14) ----------------
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Liberation Sans'],'font.size':8,'axes.grid':True,'grid.alpha':.3,'lines.linewidth':1.0,'axes.linewidth':0.6})
R = json.loads(json.dumps(res)); R = {str(k): v for k, v in R.items()}; K = 6
tc=np.array([100,150,200,300,500,750,1000,1500,2000])
fig,ax=plt.subplots(figsize=(3.3,2.6))
sty={'2':('v',':'),'4':('^','-.'),'8':('o','--'),'16':('D',(0,(5,1)))}
best=np.zeros(len(tc)); arg=[]
out={}
for No,(mk,ls) in sty.items():
    se=R[No]['FLARE']; eff=np.clip(1-int(No)*K/tc,0,None)*se
    out[No]=eff.round(2).tolist()
    ax.plot(tc,eff,marker=mk,ls=ls,color='k',ms=3.5,mfc='none',label=f'FLARE, $N_o$ = {No}')
allc=np.array([out[n] for n in sty]); best=allc.max(0); arg=[list(sty)[i] for i in allc.argmax(0)]
fx=(1-K/tc)*R['fixed']; out['fixed']=fx.round(2).tolist(); out['best']=best.round(2).tolist(); out['argbest']=arg
ax.plot(tc,best,color='tab:red',lw=2,alpha=.5,label='FLARE, adaptive $N_o$')
ax.plot(tc,fx,marker='s',color='tab:gray',ms=3.5,label='Conventional fixed antenna')
ax.set_xscale('log'); ax.set_xticks([100,200,500,1000,2000]); ax.set_xticklabels(['100','200','500','1000','2000'])
ax.set_ylim(0,50); ax.set_xlabel('Coherence block length $\\tau_c$ (symbols)'); ax.set_ylabel('Effective sum SE (bit/s/Hz)')
ax.legend(fontsize=6.5,loc='lower right'); fig.tight_layout(); fig.savefig(FIG / 'fig_overhead.png', dpi=600)
json.dump({'tc':tc.tolist(),**out},open(RES / 'overhead_effective_se.json', 'w'),indent=1)
for i,t in enumerate(tc): print(t, {k:out[k][i] for k in list(sty)+['fixed','best']}, arg[i])
