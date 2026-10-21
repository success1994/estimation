import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import expit, betaln
from scipy.optimize._numdiff import approx_derivative
from pathlib import Path
from dpd import integral, terms, jac_terms, logh_nodes, pars
ROOT=Path(__file__).resolve().parents[1]

def check_integrals():
    rows=[]
    for family,shapes in [('UW',[.4,.6,1.,1.05,1.2,1.5,2.]),('KW',[.4,.6,1.,2.,3.]),('Beta',[.6,2.,10.])]:
        for shape in shapes:
            for mu in [.02,.1,.5,.9,.98]:
                X=np.ones((1,1)); v=np.array([np.log(mu/(1-mu)),np.log(shape)])
                for g in [0.,.1,.3,.5]:
                    if family=='Beta':
                        a,b=mu*shape,(1-mu)*shape
                        def f(z):
                            ly=-np.logaddexp(0.,-z); l1=-np.logaddexp(0.,z)
                            return np.exp((1+g)*(a*ly+b*l1-betaln(a,b)))
                        ref=quad(f,-np.inf,0,epsabs=1e-11,epsrel=1e-11)[0]+quad(f,0,np.inf,epsabs=1e-11,epsrel=1e-11)[0]
                    else:
                        # Independent adaptive integral in log(t), t=-log(y).
                        from dpd import log1m
                        aa,bb=pars(v,X,family); aa=float(aa); bb=float(bb[0])
                        def f(u):
                            t=np.exp(u)
                            with np.errstate(all='ignore'):
                                if family=='UW':lh=bb+np.log(aa)+(aa-1)*u-np.exp(bb+aa*u)+log1m(t)
                                else:lh=np.log(aa)+bb-aa*t+(np.exp(bb)-1)*log1m(aa*t)+log1m(t)
                                return float(np.exp((1+g)*lh+u-log1m(t)))
                        ref=quad(f,-700,20,epsabs=1e-11,epsrel=1e-11,points=[-100,-48,-24,-12,-6,0,3,5,10],limit=500)[0]
                    fast=integral(v,X,family,g)[0]; err=abs(fast-ref)/ref
                    rows.append(dict(family=family,shape=shape,median=mu,gamma=g,scale='logit',fast=fast,reference=ref,relative_error=err,pass_check=err<=1e-6 and (g!=0 or abs(ref-1)<1e-8)))
                if family=='UW' and shape>1:
                    alpha=np.log(2)/(-np.log(mu))**shape; g=.3
                    def f(t):
                        if t==0:return 0
                        return np.exp((1+g)*np.log(alpha*shape)+(shape-1)*(1+g)*np.log(t)+g*t-alpha*(1+g)*t**shape)
                    ref=quad(f,0,np.inf,epsabs=1e-10,epsrel=1e-10,limit=500)[0]; fast=integral(v,X,family,g,'original')[0];err=abs(fast-ref)/ref
                    rows.append(dict(family=family,shape=shape,median=mu,gamma=g,scale='original',fast=fast,reference=ref,relative_error=err,pass_check=err<=1e-6))
    # Original KW identity checked against an independent adaptive integral.
    for shape in [.4,.6,1.,2.,3.]:
        for mu in [.02,.1,.5,.9,.98]:
            X=np.ones((1,1));v=np.array([np.log(mu/(1-mu)),np.log(shape)]);g=.3
            aa,lb=pars(v,X,'KW');bb=np.exp(lb[0]);fast=integral(v,X,'KW',g,'original')[0]
            if aa<=g/(1+g) or bb<=g/(1+g):
                assert fast==1e100
                continue
            from dpd import log1m
            def f(u):
                t=np.exp(u)
                with np.errstate(all='ignore'):
                    lm=np.log(aa)+u-aa*t/2 if u < -36 else log1m(aa*t)
                    lf=np.log(aa)+np.log(bb)-(aa-1)*t+(bb-1)*lm
                    return float(np.exp((1+g)*lf+u-t))
            ref=quad(f,-5000,20,epsabs=1e-10,epsrel=1e-10,points=[-100,-48,-24,-12,-6,0,3,5,10],limit=500)[0];err=abs(fast-ref)/ref
            rows.append(dict(family='KW',shape=shape,median=mu,gamma=g,scale='original',fast=fast,reference=ref,relative_error=err,pass_check=err<=1e-6))
    for shape in [.4,.6]:
        assert integral(np.array([0.,np.log(shape)]),np.ones((1,1)),'UW',.3,'original')[0]==1e100
    df=pd.DataFrame(rows);df.to_csv(ROOT/'results/integral_check.csv',index=False)
    print(df.groupby(['family','scale']).relative_error.max().to_string(),flush=True)
    assert df.pass_check.all(),df[~df.pass_check].to_string()

def check_gradients():
    lines=['# Gradient checks','Complex-step gradients compared with independent three-point central differences on three seeded random datasets per family.','']
    for seed in range(3):
        rng=np.random.default_rng(702+seed);X=np.c_[np.ones(30),rng.normal(size=30),rng.binomial(1,.5,30)];y=rng.uniform(.02,.98,30)
        for family in ['UW','KW','Beta']:
            v=np.array([-.5,.8,-.6,np.log(1.5)])
            for g,scale in [(0.,'logit'),(.1,'logit'),(.3,'logit'),(.5,'logit'),(.3,'original')]:
                if family=='Beta' and scale=='original':continue
                a=jac_terms(v,X,y,family,g,scale)
                b=approx_derivative(lambda z:terms(z,X,y,family,g,scale),v,method='3-point')
                err=np.max(abs(a-b)/(1+abs(b))); assert err<1e-6,(family,g,scale,err)
                lines.append(f'- Dataset {seed}, {family}, gamma={g}, {scale}: scaled maximum error {err:.3g}.')
    (ROOT/'results/gradient_check.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
    check_integrals();check_gradients()
