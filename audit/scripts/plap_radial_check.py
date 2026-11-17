import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
def shoot(lam, p, n, q, R=1.0, dense=False):
    # (r^{n-1}|u'|^{p-2}u')' = -lam r^{n-1} |u|^{q-2}u ; q=p homogeneous, q=2 code variant
    eps=1e-6
    def f(r,y):
        u,w=y
        s=w/r**(n-1)
        return [np.sign(s)*abs(s)**(1/(p-1)), -lam*r**(n-1)*np.sign(u)*abs(u)**(q-1)]
    return solve_ivp(f,[eps,R],[1.0,-lam*eps**n/n],rtol=1e-11,atol=1e-13,dense_output=True,max_step=1e-3)
def radial_eigs(p,n,q,k=2,lmax=120):
    g=lambda lam: shoot(lam,p,n,q).y[0,-1]
    ls=np.linspace(0.5,lmax,240); v=[g(l) for l in ls]; out=[]
    for i in range(len(ls)-1):
        if v[i]*v[i+1]<0: out.append(brentq(g,ls[i],ls[i+1],xtol=1e-10))
    return out[:k]
for (p,n) in [(2.0,2),(2.2,2),(2.0,3),(2.5,3),(2.2,3)]:
    ev=radial_eigs(p,n,p)
    print(f"p={p} n={n}: radial eigenvalues (homogeneous, correct eq.) {ev}")
    if p!=2:
        ev0=radial_eigs(p,n,2)
        res=[]
        for E0 in ev0:
            sol=shoot(E0,p,n,2); r=np.linspace(1e-6,1,20001); u=sol.sol(r)[0]
            m2=n*np.trapezoid(u*u*r**(n-1),r)       # mean over ball of u^2
            c=m2**-0.5
            res.append(E0*c**(p-2))
        print(f"     code variant  -Δ_p u = E u, mean(u²)=1 : E = {res}   (u(0)=1 gives E0={ev0})")
