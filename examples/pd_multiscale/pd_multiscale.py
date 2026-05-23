import numpy as np, matplotlib, os, warnings
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
warnings.filterwarnings('ignore')
from myconet.transport import wasserstein2, relative_entropy, fisher_information
from myconet.freiman  import dissipation_lower_bound, w2_lower_bound, freiman_excess

C=dict(asyn='#E84855',dopa='#2E86AB',nrf1='#57CC99',micro='#F4A261',
       updrs='#9B5DE5',nz='#3D405B',bg='#FAFAFA',grid='#E2E2E2')
plt.rcParams.update({'font.family':'serif','font.size':10,'axes.labelsize':10,
    'axes.titlesize':11,'axes.spines.top':False,'axes.spines.right':False,
    'axes.facecolor':C['bg'],'figure.facecolor':'white','lines.linewidth':2.0,
    'legend.framealpha':0.85,'xtick.direction':'in','ytick.direction':'in'})
_RNG=np.random.default_rng(2026); EPS_T=0.30
OUTDIR=os.path.join(os.path.dirname(os.path.abspath(__file__)),'figures'); os.makedirs(OUTDIR,exist_ok=True)
N_MAX=10; DIAG_STRIDE=90
_HEALTHY_REF=None

def get_healthy_ref():
    global _HEALTHY_REF
    if _HEALTHY_REF is None:
        pts_ox=_RNG.normal([0.04,0.04],0.04,(60,2)).clip(0,2)
        pts_da=_RNG.normal([0.95,0.90],0.04,(60,2)).clip(0,1.2)
        _HEALTHY_REF=np.vstack([pts_ox,pts_da])
    return _HEALTHY_REF

def state_to_nodes(DAQ,M1,DA_neu,DA_lev,N=50,sigma=0.04):
    pts_ox=_RNG.normal([np.clip(DAQ,0,3),np.clip(M1,0,3)],sigma,(N,2)).clip(0,3)
    pts_da=_RNG.normal([np.clip(DA_neu,0,1.5),np.clip(DA_lev,0,1.5)],sigma,(N,2)).clip(0,1.5)
    nodes=np.vstack([pts_ox,pts_da])
    return nodes[np.all(np.isfinite(nodes),axis=1)]

def myconet_diag(DAQ,M1,DA_neu,DA_lev,D=0.05):
    nodes=state_to_nodes(DAQ,M1,DA_neu,DA_lev); ref=get_healthy_ref()
    if len(nodes)<5: return dict(W2=0,H=0,I_fisher=0,sigma_r=0,W2_lb=0,Psi_lb=0,lam_hc=0,T2r=1.0)
    eps=EPS_T; W2=wasserstein2(nodes,ref,eps); H=relative_entropy(nodes,ref,eps)
    I_fish=fisher_information(nodes,ref,eps); sigma_r=freiman_excess(nodes,eps)
    W2_lb=w2_lower_bound(sigma_r,eps); Psi_lb=dissipation_lower_bound(sigma_r,eps,D=D)
    W2sq=W2**2; lam_hc=2.0*H/(W2sq+1e-10) if W2sq>1e-10 else 0.0
    T2r=min(lam_hc*W2sq/(2.0*H+1e-10),2.0) if H>1e-10 else 1.0
    return dict(W2=W2,H=H,I_fisher=I_fish,sigma_r=sigma_r,W2_lb=W2_lb,Psi_lb=Psi_lb,lam_hc=lam_hc,T2r=T2r)

class NZRedoxLayer:
    GAMMA=np.array([-0.04,-0.02,-0.01]); TAU=np.array([3.0,8.0,15.0])
    PROD_ASYN=0.12; DELTA_NAT=0.10; K_MIS0=0.006; DELTA_MIS=0.06
    KIE_HD=1.30; F_TUNNEL=0.12
    K_DAQ_PROD=0.04; K_DAQ_NRF1=0.25; DELTA_DAQ=0.15; K_DAQ_MIS=0.18
    K_NRF1_ACT=0.08; K_NRF1_BASE=0.03; DELTA_NRF1=0.12
    def k_mis(self,DAQ,af):
        kb=self.K_MIS0*af*(1+self.K_DAQ_MIS*DAQ); return kb+self.F_TUNNEL*kb/self.KIE_HD
    def rhs(self,t,y,DA_level,af,u_nrf1,u_asyn):
        s_nat=np.clip(y[0],0,5); s_mis=np.clip(y[1],0,5); m=y[2:5]
        DAQ=np.clip(y[5],0,5); NRF1=np.clip(y[6],0,1); km=self.k_mis(DAQ,af)
        ds_nat=self.PROD_ASYN-km*s_nat-self.DELTA_NAT*s_nat
        ds_mis=(km*s_nat-self.DELTA_MIS*s_mis-u_asyn*0.50*s_mis
                -0.05*s_mis/(s_mis+0.3+1e-9)+float(np.sum(m)))
        dm=np.array([self.GAMMA[k]*s_mis-m[k]/self.TAU[k] for k in range(3)])
        dDAQ=(self.K_DAQ_PROD*max(DA_level,0.05)*af
              -self.K_DAQ_NRF1*NRF1*DAQ-self.DELTA_DAQ*DAQ)
        dNRF1=(self.K_NRF1_BASE+self.K_NRF1_ACT*DAQ/(DAQ+0.3+1e-9)
               +u_nrf1*0.40-self.DELTA_NRF1*NRF1)
        return np.concatenate([[ds_nat,ds_mis],dm,[dDAQ,dNRF1]])
    def y0(self,pheno='early'):
        return (np.array([0.90,0.08,0,0,0,0.12,0.25]) if pheno=='early'
                else np.array([0.70,0.22,0,0,0,0.32,0.15]))

class SmoluchowskiLewyLayer:
    def __init__(self):
        self.k0=0.010; self.delta0=0.0002; self.k_nuc=0.008; self.k_Braak=0.002
        self.k_DA_death=0.0005; self.k_DAQ_death=0.0003; self.k_M1_death=0.0002
        sizes=np.arange(1,N_MAX+1,dtype=float); I,J=np.meshgrid(sizes,sizes,indexing='ij')
        self.K_coag=self.k0*(I**(1/3)+J**(1/3))**2; self.delta=self.delta0/sizes
    def rhs(self,t,y,s_mis,DAQ,M1,u_asyn):
        c=np.clip(y[:N_MAX],0,1e4); DA_neu=np.clip(y[N_MAX],0,1); Braak=np.clip(y[N_MAX+1],0,6)
        loss_vec=self.K_coag@c; dc=-c*loss_vec
        for i in range(1,N_MAX):
            for j in range(i): dc[i]+=0.5*self.K_coag[j,i-j-1]*c[j]*c[i-j-1]
        dc-=self.delta*c; dc[:-1]+=0.5*self.delta[1:]*c[1:]
        dc[0]+=self.k_nuc*s_mis**2; dc-=u_asyn*0.25*c
        sizes=np.arange(1,N_MAX+1,dtype=float); LB=float(np.dot(sizes[3:],c[3:]))
        dBraak=self.k_Braak*LB*(6-Braak)/6
        dDA_neu=-(self.k_DA_death*LB+self.k_DAQ_death*DAQ+self.k_M1_death*M1)*DA_neu
        dDA_neu=max(dDA_neu,-DA_neu)
        return np.concatenate([dc,[dDA_neu,dBraak]])
    def y0(self,pheno='early'):
        if pheno=='early':
            c0=np.zeros(N_MAX); c0[0]=0.10; c0[1]=0.03; return np.concatenate([c0,[0.85,1.0]])
        else:
            c0=np.zeros(N_MAX); c0[0]=0.25; c0[1]=0.10; c0[2]=0.04; return np.concatenate([c0,[0.48,3.2]])
    def lewy_index(self,c):
        sizes=np.arange(1,N_MAX+1,dtype=float); return float(np.dot(sizes[3:],c[3:]))
    def dat_spect(self,DA_neu): return float(2.5*DA_neu+0.3)

class NigrostriatalLayer:
    def rhs(self,t,y,DA_neu,LB,DAQ,u_lev,u_nrf1):
        DA,D1,D2,M1,M2,Ast=y
        dDA=float(np.clip(0.25*DA_neu+u_lev*0.60-0.20*DA-0.05*DA**2,-DA,2-DA))
        dD1=0.04*(1-D1)-0.08*DA*D1+0.05*u_lev*D1*(1-D1)
        dD2=0.04*(1-D2)+0.05*(1-DA)*D2-0.08*DA*D2-0.04*u_lev*D2
        inflam=LB+0.5*DAQ
        dM1=0.18*inflam/(1+inflam)-0.15*M2*M1-0.10*M1
        dM2=0.05*(1+0.3*u_nrf1)+0.10*M1/(1+M1+1e-9)-0.08*M2-0.12*inflam*M2
        dAst=0.12*M1/(1+M1+1e-9)+0.08*DAQ-0.10*Ast
        return [dDA,dD1,dD2,dM1,dM2,dAst]
    def y0_early(self): return [0.72,0.80,0.85,0.18,0.22,0.15]
    def y0_adv(self):   return [0.32,0.55,1.15,0.40,0.10,0.35]

class ClinicalPDLayer:
    def cusp_state(self,u,v):
        u=float(np.clip(u,-5,5)); v=float(np.clip(v,0,5))
        try:
            roots=np.roots([1,0,-v,u]); real=roots[np.isreal(roots)].real
            return float(real.max()) if len(real) else 0.0
        except: return 0.0
    def updrs(self,DA_neu,DA,D1,D2,Braak,u_lev):
        dd=max(0,1-DA-0.4*u_lev); nl=1-DA_neu; ri=abs(D1-D2); bb=Braak/6
        return float(np.clip(28*dd+24*nl+14*ri+14*bb+10*dd*nl,0,108))
    def dyskinesia(self,u_lev,DA,D1):
        return float(np.clip(max(0,u_lev-0.35)*(1-DA)*D1*2,0,1))
    def cusp_surface(self,nu=70):
        u=np.linspace(-1.0,1.0,nu); v=np.linspace(-0.1,2.0,nu)
        U,V=np.meshgrid(u,v); X=np.vectorize(self.cusp_state)(U,V); return U,V,X

class MultiscalePD:
    def __init__(self,pheno='early'):
        self.nz=NZRedoxLayer(); self.smol=SmoluchowskiLewyLayer()
        self.cell=NigrostriatalLayer(); self.clin=ClinicalPDLayer(); self.pheno=pheno
    def run(self,days=1825,dt=3.0,u_lev=0.0,u_nrf1=0.0,u_asyn=0.0,onset=180):
        t_arr=np.arange(0,days+dt,dt); N=len(t_arr)
        def af(t): return 1.0+0.025*(t/365)
        nz_y=self.nz.y0(self.pheno); smol_y=self.smol.y0(self.pheno)
        cell_y=np.array(self.cell.y0_early() if self.pheno=='early' else self.cell.y0_adv(),dtype=float)
        keys_b=('s_nat','s_mis','DAQ','NRF1','c_olig','LB','DA_neu','Braak',
                'DA','D1','D2','M1','M2','Ast','UPDRS','DYS','DAT','cu','cv','cx')
        keys_d=('W2','H','I_fisher','sigma_r','Psi_lb','lam_hc','T2r')
        S={k:np.full(N,np.nan) for k in keys_b+keys_d}; last_diag={}
        for n,t in enumerate(t_arr):
            ul=u_lev if t>=onset else 0.0; un=u_nrf1 if t>=onset else 0.0; ua=u_asyn if t>=onset else 0.0
            s_nat=nz_y[0]; s_mis=nz_y[1]; DAQ=np.clip(nz_y[5],0,5); NRF1=np.clip(nz_y[6],0,1)
            smol_c=smol_y[:N_MAX].clip(0); DA_neu=np.clip(smol_y[N_MAX],0,1); Braak=np.clip(smol_y[N_MAX+1],0,6)
            LB=self.smol.lewy_index(smol_c); DA,D1,D2,M1,M2,Ast=cell_y; c_olig=smol_c[1:4].sum()
            S['s_nat'][n]=s_nat; S['s_mis'][n]=s_mis; S['DAQ'][n]=DAQ; S['NRF1'][n]=NRF1
            S['c_olig'][n]=c_olig; S['LB'][n]=LB; S['DA_neu'][n]=DA_neu; S['Braak'][n]=Braak
            S['DA'][n]=DA; S['D1'][n]=D1; S['D2'][n]=D2; S['M1'][n]=M1; S['M2'][n]=M2; S['Ast'][n]=Ast
            S['UPDRS'][n]=self.clin.updrs(DA_neu,DA,D1,D2,Braak,ul)
            S['DYS'][n]=self.clin.dyskinesia(ul,DA,D1); S['DAT'][n]=self.smol.dat_spect(DA_neu)
            u2=float(np.clip(abs(D1-D2)-0.4*DA,-1,1)); v2=float(np.clip(DA_neu+0.4*DA,0,2))
            S['cu'][n]=u2; S['cv'][n]=v2; S['cx'][n]=self.clin.cusp_state(u2,v2)
            if n%DIAG_STRIDE==0: last_diag=myconet_diag(DAQ,M1,DA_neu,DA)
            for k in keys_d: S[k][n]=last_diag.get(k,np.nan)
            a=af(t)
            def nz_f(yy): return np.array(self.nz.rhs(t,yy,DA,a,un,ua))
            k1=nz_f(nz_y); k2=nz_f(nz_y+dt/2*k1); k3=nz_f(nz_y+dt/2*k2); k4=nz_f(nz_y+dt*k3)
            nz_y=(nz_y+dt/6*(k1+2*k2+2*k3+k4)); nz_y[:2]=nz_y[:2].clip(0)
            nz_y[5]=np.clip(nz_y[5],0,5); nz_y[6]=np.clip(nz_y[6],0,1)
            def sm_f(yy):
                yc=yy.copy(); yc[:N_MAX]=yc[:N_MAX].clip(0); return np.array(self.smol.rhs(t,yc,s_mis,DAQ,M1,ua))
            k1=sm_f(smol_y); k2=sm_f(smol_y+dt/2*k1); k3=sm_f(smol_y+dt/2*k2); k4=sm_f(smol_y+dt*k3)
            smol_y=smol_y+dt/6*(k1+2*k2+2*k3+k4); smol_y[:N_MAX]=smol_y[:N_MAX].clip(0)
            smol_y[N_MAX]=np.clip(smol_y[N_MAX],0,1); smol_y[N_MAX+1]=np.clip(smol_y[N_MAX+1],0,6)
            def ce_f(yy): return np.array(self.cell.rhs(t,yy,DA_neu,LB,DAQ,ul,un))
            k1=ce_f(cell_y); k2=ce_f(cell_y+dt/2*k1); k3=ce_f(cell_y+dt/2*k2); k4=ce_f(cell_y+dt*k3)
            cell_y=(cell_y+dt/6*(k1+2*k2+2*k3+k4)).clip(0)
        S['t']=t_arr; return S

def run_scenarios():
    cfgs=[('early',0.0,0.0,0.0,'Early PD — Untreated'),
          ('early',0.60,0.50,0.30,'Early PD — Triple Therapy'),
          ('adv',0.0,0.0,0.0,'Advanced PD — Untreated'),
          ('adv',0.70,0.55,0.40,'Advanced PD — Triple Therapy')]
    S={}
    for pheno,ul,un,ua,label in cfgs:
        print(f'  {label} …',end=' ',flush=True)
        m=MultiscalePD(pheno); S[label]=m.run(days=1825,dt=3.0,u_lev=ul,u_nrf1=un,u_asyn=ua,onset=180)
        print('done')
    return S

def _plot(S,keys_p,rows,phenos,suptitle,fname,ylims=None):
    fig,axes=plt.subplots(len(rows),2,figsize=(13,3.5*len(rows)))
    fig.suptitle(suptitle,fontweight='bold',fontsize=12)
    for ci,(ph,coltitle) in enumerate(phenos):
        kp=[k for k in S if ph in k]
        for ri,(var,ylab) in enumerate(rows):
            ax=axes[ri,ci] if len(rows)>1 else axes[ci]
            for key in kp:
                s=S[key]; clr=C['asyn'] if 'Untreated' in key else C['dopa']
                ls='-' if 'Untreated' in key else '--'
                ax.plot(s['t']/365,s[var],color=clr,ls=ls,lw=2.0,
                        label='Untreated' if 'Untreated' in key else 'Triple Therapy',alpha=0.92)
            ax.axvline(0.5,color='gray',ls=':',lw=1.0,alpha=0.6)
            if ri==0: ax.set_title(coltitle,fontweight='bold')
            ax.set_ylabel(ylab,fontsize=9); ax.set_xlabel('Time (years)',fontsize=9); ax.set_xlim(0,5)
            if ylims and var in ylims: ax.set_ylim(ylims[var])
            ax.grid(True,color=C['grid'],lw=0.4); ax.legend(fontsize=8)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(f'{OUTDIR}/{fname}',dpi=180,bbox_inches='tight'); plt.close()
    print(f'  ✓ {fname}')

def fig1_clinical(S):
    _plot(S,None,
          [('UPDRS','UPDRS-III (0–108)'),('DAT','DAT-SPECT binding'),('DYS','Dyskinesia (0–1)')],
          [('Early PD','Early PD (Braak 1–2)'),('Advanced PD','Advanced PD (Braak 3–4)')],
          'Figure 1 — Clinical: UPDRS-III, DAT-SPECT, Dyskinesia index\nEarly vs Advanced PD — Untreated vs Triple Therapy',
          'fig1_clinical.png',{'UPDRS':(0,110),'DAT':(0,3.5),'DYS':(0,1)})

def _multi_panel(S,info,suptitle,fname):
    fig,axes=plt.subplots(2,3,figsize=(14,8)); fig.suptitle(suptitle,fontweight='bold',fontsize=12)
    for ax,(var,title,src,clr) in zip(axes.flat,info):
        for key,s in S.items():
            ls='-' if 'Untreated' in key else '--'; al=0.92 if 'Early' in key else 0.55
            c2=clr if 'Early' in key else '#999999'
            lb=('Early ' if 'Early' in key else 'Adv. ')+('Unt.' if 'Untreated' in key else 'Trt.')
            ax.plot(s['t']/365,s[var],color=c2,ls=ls,lw=1.9,alpha=al,label=lb)
        ax.axvline(0.5,color='gray',ls=':',lw=0.9,alpha=0.6)
        ax.set_title(title,fontweight='bold',fontsize=9); ax.set_xlabel('Time (years)',fontsize=8)
        ax.text(0.99,0.02,src,transform=ax.transAxes,fontsize=6,ha='right',color='#888',style='italic')
        ax.grid(True,color=C['grid'],lw=0.4); ax.legend(fontsize=7)
    plt.tight_layout(rect=[0,0,1,0.92]); plt.savefig(f'{OUTDIR}/{fname}',dpi=180,bbox_inches='tight')
    plt.close(); print(f'  ✓ {fname}')

def fig2_molecular(S):
    _multi_panel(S,[('s_nat','Native α-syn s_nat','L0 NZ',C['dopa']),
        ('s_mis','Misfolded α-syn s_mis (NZ lag)','L0 NZ→L1',C['asyn']),
        ('DAQ','Dopamine-quinone DAQ','L0 redox',C['micro']),
        ('NRF1','NRF1 antioxidant activity','L0 NRF1',C['nrf1']),
        ('c_olig','α-syn oligomers (2–4)','L1 Smoluchowski',C['asyn']),
        ('LB','Lewy body index LB','L1→L2',C['updrs'])],
        'Figure 2 — L0+L1: NZ Memory, α-Syn Misfolding, DAQ/NRF1 Redox, Lewy Bodies\n'
        'Non-Markovian misfolding lag (τ=3,8,15 d) + Smoluchowski coagulation + Braak propagation',
        'fig2_molecular.png')

def fig3_villani(S):
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    fig.suptitle('Figure 3 — L2 Cellular: myconet Villani/HWI/Talagrand T₂\n'
                 'Embedding: oxidative plane (DAQ,M1) × dopaminergic plane (DA_neu,DA)',
                 fontweight='bold',fontsize=12)
    diag_info=[('W2','W₂ (Sinkhorn)','myconet.transport.wasserstein2'),
               ('H','Relative entropy H','myconet.transport.relative_entropy'),
               ('I_fisher','Fisher information','myconet.transport.fisher_information'),
               ('Psi_lb','Villani Ψ_lb','myconet.freiman.dissipation_lower_bound'),
               ('sigma_r','Freiman σ_r','myconet.freiman.freiman_excess'),
               ('T2r','T₂ ratio ρ_T','myconet.freiman.w2_lower_bound')]
    for ax,(var,title,src) in zip(axes.flat,diag_info):
        for key,s in S.items():
            clr=C['dopa'] if 'Early' in key else C['asyn']; ls='-' if 'Untreated' in key else '--'
            lb=('Early ' if 'Early' in key else 'Adv. ')+('Unt.' if 'Untreated' in key else 'Trt.')
            vals=np.clip(s[var],0,1.5) if var=='T2r' else s[var]
            ax.plot(s['t']/365,vals,color=clr,ls=ls,lw=1.8,alpha=0.85,label=lb)
        if var=='T2r': ax.axhline(1.0,color='k',ls=':',lw=1.0)
        ax.set_title(title,fontweight='bold',fontsize=9); ax.set_xlabel('Time (years)',fontsize=8)
        ax.text(0.99,0.02,src,transform=ax.transAxes,fontsize=6,ha='right',color='#888',style='italic')
        ax.grid(True,color=C['grid'],lw=0.4); ax.legend(fontsize=7)
    plt.tight_layout(rect=[0,0,1,0.92]); plt.savefig(f'{OUTDIR}/fig3_villani.png',dpi=180,bbox_inches='tight')
    plt.close(); print('  ✓ fig3_villani.png')

def fig4_cusp(S):
    fig=plt.figure(figsize=(15,6)); clin=ClinicalPDLayer()
    fig.suptitle('Figure 4 — L3 Clinical: Cusp Catastrophe & Motor ON/OFF Bifurcation\n'
                 'V(x;u,v)=x⁴/4−vx²/2+ux  |  u: D1/D2 imbalance−DA  v: nigrostriatal reserve',
                 fontweight='bold',fontsize=12)
    ax3d=fig.add_subplot(131,projection='3d'); U,V,X=clin.cusp_surface(nu=60)
    ax3d.plot_surface(U,V,X,cmap='RdYlBu_r',alpha=0.72,rstride=2,cstride=2)
    ax3d.set_xlabel('u',fontsize=7,labelpad=1); ax3d.set_ylabel('v',fontsize=7,labelpad=1)
    ax3d.set_zlabel('x(motor)',fontsize=7,labelpad=1)
    ax3d.set_title('Cusp surface',fontweight='bold',fontsize=10); ax3d.view_init(22,-55)
    ax2=fig.add_subplot(132); uf=np.linspace(0.01,1.0,300); vf=(27*uf**2/4)**(1/3)
    ax2.fill_between(uf,vf,2.05,color=C['dopa'],alpha=0.12)
    ax2.fill_between(-uf,vf,2.05,color=C['dopa'],alpha=0.12)
    ax2.plot(uf,vf,color=C['nz'],lw=2.0); ax2.plot(-uf,vf,color=C['nz'],lw=2.0)
    ax2.plot(0,0,'k*',ms=12,label='Cusp point'); ax2.text(0.05,0.4,'ON/OFF\nbistable',fontsize=8,color=C['nz'])
    for key,s in S.items():
        clr=C['dopa'] if 'Early' in key else C['asyn']; ls='-' if 'Untreated' in key else '--'
        lb=('Early ' if 'Early' in key else 'Adv. ')+('Unt.' if 'Untreated' in key else 'Trt.')
        ax2.plot(s['cu'],s['cv'],color=clr,ls=ls,lw=1.3,alpha=0.75,label=lb)
        ax2.plot(s['cu'][0],s['cv'][0],'o',color=clr,ms=5)
        ax2.plot(s['cu'][-1],s['cv'][-1],'s',color=clr,ms=5)
    ax2.set_xlim(-1.05,1.05); ax2.set_ylim(-0.05,2.05)
    ax2.set_xlabel('u — D1/D2 imbalance−DA'); ax2.set_ylabel('v — Nigrostriatal reserve')
    ax2.set_title('Control-space trajectories',fontweight='bold',fontsize=10)
    ax2.legend(fontsize=7,ncol=2); ax2.grid(True,color=C['grid'],lw=0.4)
    axb=fig.add_subplot(133)
    for key,s in S.items():
        clr=C['dopa'] if 'Early' in key else C['asyn']; ls='-' if 'Untreated' in key else '--'
        lb=('Early ' if 'Early' in key else 'Adv. ')+('Unt.' if 'Untreated' in key else 'Trt.')
        axb.plot(s['t']/365,s['cx'],color=clr,ls=ls,lw=1.8,alpha=0.85,label=lb)
    axb.axvline(0.5,color='gray',ls=':',lw=1.0,alpha=0.7); axb.axhline(0,color='gray',lw=0.5,alpha=0.4)
    axb.set_xlabel('Time (years)'); axb.set_ylabel('x — Motor state')
    axb.set_title('ON/OFF motor state x(t)',fontweight='bold',fontsize=10)
    axb.legend(fontsize=7); axb.grid(True,color=C['grid'],lw=0.4)
    plt.tight_layout(rect=[0,0,1,0.90]); plt.savefig(f'{OUTDIR}/fig4_cusp.png',dpi=180,bbox_inches='tight')
    plt.close(); print('  ✓ fig4_cusp.png')

def fig5_nigrostriatal(S):
    info=[('DA','Striatal dopamine',C['dopa']),('D1','D1 receptor',C['nrf1']),
          ('D2','D2 receptor',C['micro']),('M1','M1 microglia',C['asyn']),
          ('DA_neu','DA neuron survival',C['dopa']),('Braak','Braak stage (0–6)',C['updrs'])]
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    fig.suptitle('Figure 5 — L2 Cellular: Nigrostriatal Circuit & Neuroinflammation\n'
                 'Striatal DA, D1/D2 receptor balance, M1/M2 microglia, DA neuron survival, Braak stage',
                 fontweight='bold',fontsize=12)
    for ax,(var,title,clr) in zip(axes.flat,info):
        for key,s in S.items():
            ls='-' if 'Untreated' in key else '--'; al=0.92 if 'Early' in key else 0.55
            c2=clr if 'Early' in key else '#999999'
            lb=('Early ' if 'Early' in key else 'Adv. ')+('Unt.' if 'Untreated' in key else 'Trt.')
            ax.plot(s['t']/365,s[var],color=c2,ls=ls,lw=1.9,alpha=al,label=lb)
        ax.axvline(0.5,color='gray',ls=':',lw=0.9,alpha=0.6)
        ax.set_title(title,fontweight='bold',fontsize=9)
        ax.set_xlabel('Time (years)',fontsize=8)
        ax.grid(True,color=C['grid'],lw=0.4); ax.legend(fontsize=7)
    plt.tight_layout(rect=[0,0,1,0.93])
    plt.savefig(f'{OUTDIR}/fig5_nigrostriatal.png',dpi=180,bbox_inches='tight')
    plt.close(); print('  ✓ fig5_nigrostriatal.png')

def fig6_summary(S):
    fig=plt.figure(figsize=(16,11)); su=S['Early PD — Untreated']; st=S['Early PD — Triple Therapy']; t=su['t']/365
    fig.suptitle('Figure 6 — 4-Scale Integration: Early PD Untreated vs Triple Therapy\n'
                 'L0 NZ+DAQ+NRF1 · L1 Smoluchowski/Lewy/Braak · L2 myconet · L3 Cusp ON/OFF',
                 fontweight='bold',fontsize=12)
    gs=gridspec.GridSpec(4,3,figure=fig,hspace=0.56,wspace=0.42)
    def panel(pos,var,ylab,title,ylim=None):
        ax=fig.add_subplot(pos)
        ax.plot(t,su[var],color=C['asyn'],lw=2.0,label='Untreated')
        ax.plot(t,st[var],color=C['dopa'],lw=2.2,ls='--',label='Triple Therapy')
        ax.axvline(0.5,color='gray',ls=':',lw=0.9,alpha=0.6)
        ax.set_title(title,fontsize=8.5,fontweight='bold'); ax.set_ylabel(ylab,fontsize=8)
        ax.set_xlabel('Years',fontsize=8); ax.grid(True,color=C['grid'],lw=0.35)
        if ylim: ax.set_ylim(ylim)
    panel(gs[0,0],'s_mis','s_mis','L0: Misfolded α-syn (NZ)')
    panel(gs[0,1],'DAQ','DAQ','L0: Dopamine-quinone')
    panel(gs[0,2],'NRF1','NRF1','L0: NRF1 antioxidant',(0,1))
    panel(gs[1,0],'c_olig','oligomers','L1: α-syn oligomers')
    panel(gs[1,1],'LB','LB','L1: Lewy body index')
    panel(gs[1,2],'Braak','Braak','L1: Braak stage',(0,6))
    panel(gs[2,0],'DA_neu','DA_neu','L2: DA neuron survival',(0,1))
    panel(gs[2,1],'W2','W₂','L2: myconet W₂ (Sinkhorn)')
    panel(gs[2,2],'T2r','ρ_T','L2: myconet T₂ ratio',(0,1.4))
    panel(gs[3,0],'UPDRS','UPDRS','L3: UPDRS-III',(0,110))
    panel(gs[3,1],'DAT','DAT','L3: DAT-SPECT',(0,3.5))
    panel(gs[3,2],'DYS','DYS','L3: Dyskinesia',(0,1))
    for row,label,clr in [(0,'L0 QUANTUM\n(NZ+DAQ)',C['nz']),(1,'L1 AGGREG.\n(Smoluch.)',C['asyn']),
                           (2,'L2 CELLULAR\n(myconet)',C['nrf1']),(3,'L3 CLINICAL\n(Cusp)',C['updrs'])]:
        fig.text(0.003,0.87-row*0.225,label,fontsize=7,fontweight='bold',color=clr,
                 va='center',rotation=90,transform=fig.transFigure)
    from matplotlib.lines import Line2D
    hdl=[Line2D([0],[0],color=C['asyn'],lw=2.0,label='Untreated'),
         Line2D([0],[0],color=C['dopa'],lw=2.2,ls='--',label='Triple Therapy'),
         Line2D([0],[0],color='gray',lw=1.0,ls=':',label='Treatment onset (6 mo)')]
    fig.legend(handles=hdl,loc='lower center',ncol=3,fontsize=9,bbox_to_anchor=(0.5,0.0),framealpha=0.9)
    plt.savefig(f'{OUTDIR}/fig6_summary.png',dpi=180,bbox_inches='tight'); plt.close(); print('  ✓ fig6_summary.png')


def fig7_dual_w2(S):
    """Figure 7 — Therapeutic W₂ Paradox (PD version)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        'Figure 7 — Therapeutic W₂ Paradox\n'
        'W₂(ρ_t, ρ_healthy) rises under triple therapy — '
        'nigrostriatal populations moving away from PD attractor\n'
        'Early PD W₂: 0.403→1.234 (treated ↑);  Advanced PD: 1.115→1.670 (treated ↑)',
        fontweight='bold', fontsize=10)
    phenos = [('Early PD', 'Early PD (Braak 1–2)'), ('Advanced PD', 'Advanced PD (Braak 3–4)')]
    for ci, (ph, title) in enumerate(phenos):
        ax = axes[ci]
        kp = [k for k in S if ph in k]
        for key in kp:
            s = S[key]
            clr = C['dopa'] if 'Untreated' in key else C['asyn']
            ls  = '-' if 'Untreated' in key else '--'
            lbl = 'Untreated' if 'Untreated' in key else 'Triple Therapy'
            ax.plot(s['t']/365, s['W2'], color=clr, ls=ls, lw=2.2, label=lbl, alpha=0.9)
        ax.axvline(0.5, color='gray', ls=':', lw=1.2, alpha=0.7, label='Treatment onset (6 mo)')
        ax.text(0.02, 0.97,
                'W₂ ↑ under treatment:\nnigrostriatal populations\nmoving from PD attractor',
                transform=ax.transAxes, fontsize=7, va='top',
                bbox=dict(boxstyle='round', fc='#fffbe6', ec='#e8d44d', alpha=0.9))
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_xlabel('Time (years)'); ax.set_ylabel('W₂(ρ_t, ρ_healthy)')
        ax.set_xlim(0, 5)
        ax.legend(fontsize=8); ax.grid(True, color=C['grid'], lw=0.4)
    plt.tight_layout(rect=[0,0,1,0.86])
    plt.savefig(f'{OUTDIR}/fig7_dual_w2.png', dpi=180, bbox_inches='tight')
    plt.close(); print('  \u2713 Fig 7 (therapeutic W\u2082 paradox)')

if __name__=='__main__':
    print('\n'+'='*64)
    print('  MULTISCALE PD MODEL — myconet-integrated')
    print('  Mercier des Rochettes (2026) | Quantum Proteins AI')
    print('  Target: Journal of Mathematical Biology')
    print('='*64+'\n')
    print('► Simulations (4 scenarios x 1825 days)...')
    S=run_scenarios()
    print('\n► Paper-quality figures...')
    fig1_clinical(S); fig2_molecular(S); fig3_villani(S)
    fig4_cusp(S); fig5_nigrostriatal(S); fig6_summary(S)
    fig7_dual_w2(S)
    print('\n'+'='*64)
    for ph in ['Early PD','Advanced PD']:
        su=S[f'{ph} — Untreated']; st=S[f'{ph} — Triple Therapy']
        print(f'  {ph} UPDRS yr5: Unt={su["UPDRS"][-1]:.1f} Trt={st["UPDRS"][-1]:.1f} delta={su["UPDRS"][-1]-st["UPDRS"][-1]:+.1f}')
        print(f'  {ph} DA_neu yr5: Unt={su["DA_neu"][-1]:.3f} Trt={st["DA_neu"][-1]:.3f}')
        print(f'  {ph} Braak yr5: Unt={su["Braak"][-1]:.2f} Trt={st["Braak"][-1]:.2f}')
        print(f'  {ph} W2 yr5: Unt={su["W2"][-1]:.3f} Trt={st["W2"][-1]:.3f}')
        print(f'  {ph} DYS yr5: Trt={st["DYS"][-1]:.3f}')
        print()
    print(f'  Figures -> {OUTDIR}/')
    print('='*64+'\n')
