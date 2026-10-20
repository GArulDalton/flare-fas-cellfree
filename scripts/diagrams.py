"""Draw Fig. 1 (system model) and Fig. 2 (FLARE architecture)."""
import sys, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIG = ROOT / 'figures'; FIG.mkdir(exist_ok=True)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Liberation Sans'],'font.size':8})
# ---- system model ----
fig,ax=plt.subplots(figsize=(6.4,3.6)); ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis('off')
ax.add_patch(FancyBboxPatch((3.1,4.85),3.8,0.9,boxstyle='round,pad=0.05',fc='#dbe8f7',ec='k'))
ax.text(5,5.3,'Central Processing Unit (CPU)\nFLARE port selection + RZF precoding',ha='center',va='center',fontsize=8)
rng=np.random.default_rng(3)
aps=[(1,3.2),(3,3.6),(5,3.1),(7,3.6),(9,3.2)]
for i,(x,y) in enumerate(aps):
    ax.plot([x,5],[y+0.45,4.9],color='gray',lw=0.8,ls='--')
    ax.add_patch(Rectangle((x-0.45,y-0.12),0.9,0.24,fc='#fff2cc',ec='k'))
    sel=rng.integers(0,6)
    for n in range(6):
        ax.add_patch(Circle((x-0.37+n*0.148,y),0.05,fc='red' if n==sel else 'white',ec='k',lw=0.5))
    ax.plot([x,x],[y+0.12,y+0.45],'k',lw=1); ax.text(x+0.1,y+0.35,f'AP {i+1}' if i<4 else 'AP L',fontsize=7)
ues=[(1.8,1.0),(4.2,0.7),(6.1,1.3),(8.3,0.8)]
for j,(x,y) in enumerate(ues):
    ax.add_patch(FancyBboxPatch((x-0.15,y-0.25),0.3,0.5,boxstyle='round,pad=0.02',fc='#e2f0d9',ec='k'))
    ax.text(x,y-0.55,f'UE {j+1}' if j<3 else 'UE K',ha='center',fontsize=7)
    for (ax_,ay) in aps:
        if abs(ax_-x)<3.2: ax.plot([ax_,x],[ay-0.12,y+0.25],color='tab:blue',lw=0.5,alpha=0.5)
ax.add_patch(Circle((0.6,5.4),0.05,fc='red',ec='k')); ax.text(0.72,5.37,'activated port (filled)',fontsize=7)
ax.add_patch(Circle((0.6,5.05),0.05,fc='white',ec='k')); ax.text(0.72,5.02,'idle port (hollow), N per AP',fontsize=7)
ax.plot([0.45,0.85],[4.7,4.7],color='gray',ls='--',lw=0.8); ax.text(0.95,4.67,'fronthaul link',fontsize=7)
ax.plot([0.45,0.85],[4.4,4.4],color='tab:blue',lw=0.6); ax.text(0.95,4.37,'radio link',fontsize=7)
fig.tight_layout(); fig.savefig(FIG / 'fig_system.png',dpi=600,bbox_inches='tight')
# ---- architecture ----
fig,ax=plt.subplots(figsize=(6.8,2.6)); ax.set_xlim(0,14); ax.set_ylim(0,4); ax.axis('off')
blocks=[(0.2,'Estimated\nport channels\n$\\hat{\\mathbf{h}}_{l,k}\\in\\mathbb{C}^N$','#eeeeee'),
 (2.3,'Port-level\nfeatures\n$\\mathbf{f}_{l,k,n}\\in\\mathbb{R}^3$','#fff2cc'),
 (4.4,'Port encoder\n+ mean/max\nport pooling','#dbe8f7'),
 (6.5,'2× bipartite\nmessage\npassing\n(AP ↔ UE)','#dbe8f7'),
 (8.6,'Port scorer\n+ user\naggregation','#dbe8f7'),
 (10.7,'Logits $s_{l,n}$\n→ arg-max /\ntop-$C$ list','#fce4d6')]
for x,t,c in blocks:
    ax.add_patch(FancyBboxPatch((x,1.2),1.8,1.6,boxstyle='round,pad=0.05',fc=c,ec='k'))
    ax.text(x+0.9,2.0,t,ha='center',va='center',fontsize=6.6)
for i in range(len(blocks)-1):
    ax.annotate('',xy=(blocks[i+1][0]-0.05,2.0),xytext=(blocks[i][0]+1.85,2.0),arrowprops=dict(arrowstyle='->'))
ax.annotate('',xy=(13.9,2.0),xytext=(12.55,2.0),arrowprops=dict(arrowstyle='->'))
ax.text(13.3,2.35,'RZF\n(+LS)',ha='center',fontsize=7.5)
ax.text(7.4,0.6,'Shared weights across APs, UEs and ports → size-agnostic (any L, K, N); trained offline by imitation of Greedy-CD',ha='center',fontsize=7.5,style='italic')
fig.tight_layout(); fig.savefig(FIG / 'fig_arch.png',dpi=600,bbox_inches='tight')
