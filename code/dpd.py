"""DPD objectives, fixed exponential-quantile quadrature, and empirical sandwiches."""
import numpy as np
from scipy.special import loggamma
from scipy.optimize import minimize
from numpy.polynomial.legendre import leggauss

L=np.log(2.)
PENALTY=1e100

def log1m(x):
    return np.where(np.real(x)>np.log(2.), np.log1p(-np.exp(-x)), np.log(-np.expm1(-x)))

def lbeta(a,b): return loggamma(a)+loggamma(b)-loggamma(a+b)

def nodes(order=24):
    z,w=leggauss(order); edges=[-48.,-24.,-12.,-6.,0.,3.,5.]
    return (np.concatenate([(a+b)/2+(b-a)/2*z for a,b in zip(edges[:-1],edges[1:])]),
            np.concatenate([(b-a)/2*w for a,b in zip(edges[:-1],edges[1:])]))
W,Q=nodes(); E=np.exp(W)
GLZ,GLW=leggauss(24)

def pars(v,X,family):
    eta=X@v[:-1]; mu=1/(1+np.exp(-eta)); s=np.exp(v[-1])
    neglogmu=np.where(eta.real>=0,np.log1p(np.exp(-eta)),-eta+np.log1p(np.exp(eta)))
    if family=='UW': return s, np.log(L)-s*np.log(neglogmu)
    if family=='KW':
        u=s*neglogmu
        logden=np.where(u.real>36,-u,np.log(-log1m(u)))
        return s,np.log(L)-logden
    return mu*s,(1-mu)*s

def logpdf(v,X,y,family):
    a,b=pars(v,X,family); t=-np.log(y)
    if family=='UW': return b+np.log(a)+t+(a-1)*np.log(t)-np.exp(b+a*np.log(t))
    if family=='KW': return np.log(a)+b-(a-1)*t+(np.exp(b)-1)*log1m(a*t)
    return (a-1)*np.log(y)+(b-1)*np.log1p(-y)-lbeta(a,b)

def logh_nodes(v,X,family,w=W):
    a,b=pars(v,X,family); e=np.exp(w)[None,:]
    if family=='UW':
        lt=(w[None,:]-b[:,None])/a; t=np.exp(lt)
        # log((1-exp(-t))/t), with its removable singularity handled.
        c=np.where(t.real<1e-5, -t/2+t*t/24, log1m(t)-lt)
        return np.log(a)+w[None,:]-e+c
    r=e/np.exp(b[:,None]); tiny=r.real<1e-10
    lm=np.where(tiny,np.log(r)-r/2+r*r/24,log1m(r))
    t=-lm/a; lt=np.where(r.real>36,-r-np.log(a),np.log(np.maximum(t.real,1e-300)) if not np.iscomplexobj(t) else np.log(t+1e-300))
    c=np.where(t.real<1e-5,-t/2+t*t/24,log1m(t)-lt)
    tail=np.where(r.real>36,0.,lt+np.log(a)+r)
    return b[:,None]-a*t+c-e+tail

def integral(v,X,family,gamma,scale='logit'):
    a,b=pars(v,X,family)
    if scale=='logit':
        if family=='Beta': return np.exp(lbeta(a*(1+gamma),b*(1+gamma))-(1+gamma)*lbeta(a,b))
        h=logh_nodes(v,X,family)
        return np.sum(np.exp(gamma*h+W-E)*Q,axis=1)
    threshold=gamma/(1+gamma)
    if family=='KW':
        bb=np.exp(b)
        if a.real<=threshold or np.any(bb.real<=threshold): return np.full(len(X),PENALTY)
        return np.exp((1+gamma)*(np.log(a)+b)-np.log(a)+lbeta(((a-1)*(1+gamma)+1)/a,(bb-1)*(1+gamma)+1))
    if a.real<1 or (a.real==1 and np.any(np.exp(b.real)*(1+gamma)<=gamma)): return np.full(len(X),PENALTY)
    # Centre original-scale UW quadrature at its log-t saddle. A fixed hazard
    # grid can miss the enormous finite tails just above k=1.
    ar=float(a.real); br=b.real; power=(ar-1)*(1+gamma)+1
    lo=np.full(len(X),-100.); hi=np.full(len(X),1000.)
    for _ in range(55):
        mid=(lo+hi)/2
        positive=np.log(1+gamma)+br+np.log(ar)+ar*mid > np.logaddexp(np.log(power),np.log(gamma)+mid)
        hi=np.where(positive,mid,hi);lo=np.where(positive,lo,mid)
    mode=(lo+hi)/2
    if np.any(mode>700):return np.full(len(X),PENALTY)
    tm=np.exp(mode);curvature=ar*power+(ar-1)*gamma*tm
    peak=(1+gamma)*(br+np.log(ar))+power*mode+gamma*tm-(1+gamma)*np.exp(br+ar*mode)
    if np.any(peak>225):return np.full(len(X),PENALTY)
    zz,ww=GLZ,GLW;edges=[-48,-24,-12,-6,0,3,6,12,24,48]
    z=np.concatenate([(l+r)/2+(r-l)*zz/2 for l,r in zip(edges[:-1],edges[1:])])
    weights=np.concatenate([(r-l)*ww/2 for l,r in zip(edges[:-1],edges[1:])])
    width=1/np.sqrt(curvature);u=mode[:,None]+width[:,None]*z
    t=np.exp(u)
    log_integrand=(1+gamma)*(b[:,None]+np.log(a))+((a-1)*(1+gamma)+1)*u+gamma*t-(1+gamma)*np.exp(b[:,None]+a*u)
    return np.sum(np.exp(log_integrand)*weights*width[:,None],axis=1)

def terms(v,X,y,family,gamma,scale='logit'):
    lp=logpdf(v,X,y,family)
    if gamma==0: return -lp
    integ=integral(v,X,family,gamma,scale)
    if np.any(integ.real>=PENALTY): return np.full(len(y),PENALTY)
    if scale=='logit': lp=lp+np.log(y)+np.log1p(-y)
    return integ-(1+1/gamma)*np.exp(gamma*lp)

def jac_terms(v,X,y,family,gamma,scale):
    # Observation i depends on beta only through eta_i. With an intercept,
    # one complex perturbation supplies every beta derivative by the chain rule.
    if np.all(X[:,0]==1):
        z=v.astype(complex);z[0]+=1e-20j
        deta=terms(z,X,y,family,gamma,scale).imag/1e-20
        z=v.astype(complex);z[-1]+=1e-20j
        dshape=terms(z,X,y,family,gamma,scale).imag/1e-20
        return np.c_[deta[:,None]*X,dshape]
    out=[]
    for j in range(len(v)):
        z=v.astype(complex); z[j]+=1e-20j
        out.append(terms(z,X,y,family,gamma,scale).imag/1e-20)
    return np.array(out).T

def fit(X,y,family,gamma,scale,b0,mle=None):
    p=X.shape[1]; bound=(-6,12) if family=='Beta' else (-6,6)
    starts=[np.r_[b0,s] for s in ((0.,2.,4.) if family=='Beta' else (-1.,0.,1.))]
    if mle is not None: starts.append(mle)
    if scale=='original' and family=='UW': bound=(0.,6.); starts=[np.r_[s[:-1],max(s[-1],.05)] for s in starts]
    def fun(v):
        with np.errstate(all='ignore'):
            z=np.mean(terms(v,X,y,family,gamma,scale))
            if not np.isfinite(z) or z>=PENALTY: return PENALTY,np.zeros(len(v))
            grad=jac_terms(v,X,y,family,gamma,scale).mean(axis=0)
            if not np.all(np.isfinite(grad)): return PENALTY,np.zeros(len(v))
            return float(z),grad
    results=[minimize(fun,s,jac=True,method='L-BFGS-B',bounds=[(-30,30)]*p+[bound],options={'maxiter':600,'maxls':100,'ftol':1e-10,'gtol':1e-6}) for s in starts]
    best=min(results,key=lambda r:r.fun if np.isfinite(r.fun) else PENALTY)
    grad=fun(best.x)[1]; bounds=[(-30,30)]*p+[bound]
    projected=np.array([0. if (x<=lo+1e-7 and g>0) or (x>=hi-1e-7 and g<0) else g for x,g,(lo,hi) in zip(best.x,grad,bounds)])
    best.success=bool(best.success and best.fun<PENALTY and np.linalg.norm(projected,ord=np.inf)<.01)
    if np.linalg.norm(projected,ord=np.inf)>=.01: best.message=str(best.message)+'; projected gradient exceeds 0.01'
    return best

def sandwich(v,X,y,family,gamma,scale):
    psi=jac_terms(v,X,y,family,gamma,scale); d=len(v); J=np.empty((d,d))
    for j in range(d):
        step=1e-4*max(1,abs(v[j])); dv=np.eye(d)[j]*step
        J[:,j]=(jac_terms(v+dv,X,y,family,gamma,scale).mean(0)-jac_terms(v-dv,X,y,family,gamma,scale).mean(0))/(2*step)
    J=(J+J.T)/2
    try:
        inv=np.linalg.inv(J); cov=inv@(psi.T@psi/len(y))@inv.T/len(y)
        return np.sqrt(np.maximum(np.diag(cov),0)) if np.all(np.isfinite(cov)) else np.full(d,np.nan)
    except np.linalg.LinAlgError: return np.full(d,np.nan)
