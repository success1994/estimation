"""Full simulation: pilot design plus model-calibrated outliers (C4, C5) and data-driven gamma.

Writes results_full/sim_fits.csv (resumable). The pilot files in results/ are not touched.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import sys, json, time, hashlib, csv, argparse, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import beta as beta_dist
from tqdm import tqdm
from models import LogitQR,UnitWeibullQR,KumaraswamyQR,BetaReg
from dpd import fit,sandwich,pars,PENALTY

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results_full'
BETA=np.array([-.5,.8,-.6])
SETTINGS=[('UW',.6),('UW',1.5),('KW',.6),('KW',2.)]
GRID=[.1,.2,.3,.4,.5,.6,.7]
NAMES=(['MLE']+[f'LMDPDE_{g}' for g in GRID]+['LMDPDE_auto','MDPDE_0.3','LogitQR','BetaMLE','BetaLMDPDE_0.3'])
rng=np.random.default_rng(20261018)
TEST=np.c_[np.ones(1000),rng.normal(size=1000),rng.binomial(1,.5,1000)]
TRUE=expit(TEST@BETA)

def source_hash():
    h=hashlib.sha256()
    for path in sorted((ROOT/'code').glob('*.py')): h.update(path.name.encode()+path.read_bytes())
    return h.hexdigest()

def quantile(u,mu,family,shape):
    """Conditional quantile function of the true model at level u."""
    if family=='UW':
        alpha=np.log(2)/(-np.log(mu))**shape
        return np.exp(-(-np.log(u)/alpha)**(1/shape))
    b=np.log(.5)/np.log1p(-mu**shape)
    return (-np.expm1(np.log1p(-u)/b))**(1/shape)

def generate(si,n,ci,rep):
    # Same seeds and clean-data generation as the pilot (code/simulate.py) for C0-C3.
    rng=np.random.default_rng(np.random.SeedSequence([20261017,si,n,ci,rep]))
    X=np.c_[np.ones(n),rng.normal(size=n),rng.binomial(1,.5,n)];mu=expit(X@BETA);family,shape=SETTINGS[si]
    with np.errstate(all='ignore'):
        if family=='UW':
            alpha=np.log(2)/(-np.log(mu))**shape;y=np.exp(-(rng.exponential(size=n)/alpha)**(1/shape))
        else:
            b=np.log(.5)/np.log1p(-mu**shape); y=(-np.expm1(np.log1p(-rng.uniform(size=n))/b))**(1/shape)
    clipped=int(np.sum((y<1e-12)|(y>1-1e-12)))
    y=np.clip(y,1e-12,1-1e-12)
    out_clipped=0
    if ci:
        rate=.1 if ci in (2,5) else .05
        idx=rng.choice(n,size=int(np.ceil(rate*n)),replace=False)
        if ci==3:X[idx,1]=3.;y[idx]=rng.uniform(.005,.05,len(idx))
        elif ci in (1,2):y[idx]=np.where(mu[idx]<.5,rng.uniform(.95,.995,len(idx)),rng.uniform(.005,.05,len(idx)))
        else:
            # Model-calibrated outliers: beyond the true 99.9% (or below the 0.1%) conditional quantile,
            # on the side opposite to the median's position.
            up=mu[idx]<.5
            u=np.where(up,rng.uniform(.999,.9999,len(idx)),rng.uniform(1e-4,1e-3,len(idx)))
            with np.errstate(all='ignore'):yo=quantile(u,mu[idx],family,shape)
            out_clipped=int(np.sum((yo<1e-12)|(yo>1-1e-12)))
            y[idx]=np.clip(yo,1e-12,1-1e-12)
    return X,y,clipped,out_clipped

def select_gamma(fits,pilot=.5,max_iter=10):
    """Iterated Warwick-Jones: minimise sum of (beta_g - beta_pilot)^2 + trace Var(beta_g) over the grid."""
    usable={g:(v,se) for g,(v,se) in fits.items() if np.all(np.isfinite(se[:3]))}
    if not usable:return np.nan
    if pilot not in usable:pilot=min(usable,key=lambda g:abs(g-pilot))
    for _ in range(max_iter):
        vp=usable[pilot][0][:3]
        crit={g:float(np.sum((v[:3]-vp)**2)+np.sum(se[:3]**2)) for g,(v,se) in usable.items()}
        new=min(crit,key=crit.get)
        if new==pilot:break
        pilot=new
    return pilot

def row(sha,si,n,ci,rep,name,v,se,success,message,obj,runtime,boundary,clipped,out_clipped,family,shape,gamma_sel=np.nan):
    if name.startswith('Beta'):
        a,b=pars(v,TEST,'Beta');pred=beta_dist.ppf(.5,a,b)
    else:pred=expit(TEST@v[:3])
    se=np.r_[se,np.full(4-len(se),np.nan)] if len(se)<4 else se
    return dict(source_hash=sha,setting=si,family=family,shape=shape,n=n,scenario=f'C{ci}',rep=rep,estimator=name,
        converged=bool(success),message=message,estimate=json.dumps(np.asarray(v).tolist()),standard_errors=json.dumps(np.asarray(se).tolist()),
        objective=obj,runtime=runtime,boundary=boundary,gamma_selected=gamma_sel,
        slope_squared_error=float(np.sum((v[1:3]-BETA[1:3])**2)),beta1=float(v[1]),beta2=float(v[2]),
        hit1=bool(abs(v[1]-BETA[1])<=1.96*se[1]),hit2=bool(abs(v[2]-BETA[2])<=1.96*se[2]),
        se_valid=bool(np.all(np.isfinite(se[1:3]))),median_rmse=float(np.sqrt(np.mean((pred-TRUE)**2))),
        clipped=clipped,outlier_clipped=out_clipped)

def run_one(task):
    si,n,ci,rep,sha=task;family,shape=SETTINGS[si];X,y,clipped,oc=generate(si,n,ci,rep)
    rows=[];grid={}
    R=lambda *a,**k:rows.append(row(sha,si,n,ci,rep,*a,clipped=clipped,out_clipped=oc,family=family,shape=shape,**k))
    with warnings.catch_warnings(),np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        t=time.monotonic()
        try:qr=LogitQR(.5).fit(X,y,None);b0=qr.v;R('LogitQR',qr.v,np.asarray(qr.se),qr.success,qr.message,np.nan,time.monotonic()-t,False)
        except Exception as ex:
            b0=np.linalg.lstsq(X,np.log(y/(1-y)),rcond=None)[0];R('LogitQR',b0,np.full(3,np.nan),False,repr(ex),np.nan,time.monotonic()-t,False)
        mle={}
        for name,fam in (('MLE',family),('BetaMLE','Beta')):
            t=time.monotonic()
            try:
                m={'UW':UnitWeibullQR,'KW':KumaraswamyQR,'Beta':BetaReg}[fam](.5).fit(X,y,b0)
                se=sandwich(m.v,X,y,fam,0.,'logit');mle[fam]=m.v
                R(name,m.v,se,m.success,m.message,-m.loglik/n,time.monotonic()-t,False)
                if fam==family:grid[0.]=(m.v,se)
            except Exception as ex:R(name,np.r_[b0,0.],np.full(4,np.nan),False,repr(ex),np.nan,time.monotonic()-t,False)
        for name in [f'LMDPDE_{g}' for g in GRID]+['MDPDE_0.3','BetaLMDPDE_0.3']:
            fam='Beta' if name.startswith('Beta') else family;scale='original' if name.startswith('MDPDE') else 'logit'
            g=float(name.rsplit('_',1)[1]);t=time.monotonic();boundary=False
            try:
                f=fit(X,y,fam,g,scale,b0,mle.get(fam));v=f.x
                se=sandwich(v,X,y,fam,g,scale) if f.fun<PENALTY else np.full(4,np.nan)
                if name=='MDPDE_0.3':
                    if fam=='UW':boundary=bool(abs(v[-1])<.01)
                    else:
                        a,lb=pars(v,X,fam);th=g/(1+g)
                        boundary=bool(abs(v[-1]-np.log(th))<.01 or np.min(abs(lb-np.log(th)))<.01)
                R(name,v,se,f.success,str(f.message),f.fun,time.monotonic()-t,boundary)
                if scale=='logit' and fam==family:grid[g]=(v,se)
            except Exception as ex:R(name,np.r_[b0,0.],np.full(4,np.nan),False,repr(ex),np.nan,time.monotonic()-t,False)
        gs=select_gamma(grid)
        src=next((r for r in rows if r['estimator']==('MLE' if gs==0 else f'LMDPDE_{gs}')),None) if np.isfinite(gs) else None
        if src is not None:
            auto=dict(src);auto['estimator']='LMDPDE_auto';auto['gamma_selected']=gs;rows.append(auto)
        else:R('LMDPDE_auto',np.r_[b0,0.],np.full(4,np.nan),False,'no usable gamma',np.nan,0.,False,gamma_sel=np.nan)
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reps300',type=int,default=300);ap.add_argument('--reps100',type=int,default=200)
    ap.add_argument('--workers',type=int,default=9);ap.add_argument('--output',default='sim_fits.csv');args=ap.parse_args()
    OUT.mkdir(exist_ok=True);sha=source_hash();path=OUT/args.output;done=set()
    if path.exists():
        old=pd.read_csv(path)
        if set(old.source_hash)!={sha}:raise RuntimeError('Source hash mismatch: archive previous run before resuming.')
        for key,g in old.groupby(['setting','n','scenario','rep']):
            if len(g)!=len(NAMES):raise RuntimeError(f'Incomplete dataset block {key}')
            done.add((int(key[0]),int(key[1]),int(key[2][1:]),int(key[3])))
    tasks=[(si,n,ci,rep,sha) for n in (300,100) for si in range(4) for ci in range(6)
           for rep in range(args.reps300 if n==300 else args.reps100) if (si,n,ci,rep) not in done]
    # Interleave cells so the ETA is representative from the start.
    tasks.sort(key=lambda t:(t[3],t[1],t[0],t[2]))
    print(f'START reps300={args.reps300} reps100={args.reps100} workers={args.workers} hash={sha[:12]} remaining={len(tasks)}',flush=True);tick=time.monotonic()
    with path.open('a',newline='') as handle,ProcessPoolExecutor(max_workers=args.workers) as pool:
        writer=None;futures=[pool.submit(run_one,t) for t in tasks]
        for fut in tqdm(as_completed(futures),total=len(tasks),file=sys.stdout,mininterval=10,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
            rows=fut.result()
            if writer is None:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
                if handle.tell()==0:writer.writeheader()
            writer.writerows(rows);handle.flush();os.fsync(handle.fileno())
    el=time.monotonic()-tick
    (OUT/f'timing_{args.output}.json').write_text(json.dumps({'datasets':len(tasks),'seconds':el,'per_dataset_worker_s':el*args.workers/max(len(tasks),1)},indent=2))
    print(f'\nCOMPLETE datasets={len(tasks)} seconds={el:.0f}',flush=True)

if __name__=='__main__':main()
