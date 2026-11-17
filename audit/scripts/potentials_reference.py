import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.special import jv
def radial(m, omega, N):
    # v = sqrt(r) u : -v'' + ((m^2-1/4)/r^2 + w^2 r^2/2) v = E v, v(0)=v(1)=0 ; m=0 singular -> use direct FD in u on staggered grid
    h=1.0/N; r=(np.arange(N)+0.5)*h          # cell centers, symmetric FV scheme
    rp=r+h/2; rm=r-h/2
    d=(rp+rm)/(r*h*h)+m*m/r**2+0.5*omega**2*r**2
    d[-1]+= rp[-1]/(r[-1]*h*h)   # Dirichlet ghost u_{N}=-u_{N-1}
    e=-rp[:-1]/(np.sqrt(r[:-1]*r[1:])*h*h)
    return eigh_tridiagonal(d,e,select='i',select_range=(0,3))[0]
for om in (0,1,10):
    ev=[]
    for m in range(0,7):
        a=radial(m,om,2000); b=radial(m,om,4000); x=(4*b-a)/3
        ev+= [(x[k],m,k+1) for k in range(4)]
    ev.sort(); print("omega",om,[f"{e:.4f}(m={m},k={k},mult={1 if m==0 else 2})" for e,m,k in ev[:8]])
# first-order perturbation, 3D ball, ground state u=sin(pi r)/(r sqrt(2pi))
rng=np.random.default_rng(0); n=4_000_000
x=rng.normal(size=(n,3)); x/=np.linalg.norm(x,axis=1)[:,None]; x*=rng.random(n)[:,None]**(1/3)
r=np.linalg.norm(x,axis=1); u2=(np.sin(np.pi*r)/r)**2/(2*np.pi); vol=4*np.pi/3
def dw(x):
    s=np.array([1.,0,0]); return -(np.exp(-2*((x-s)**2).sum(1))+np.exp(-2*((x+s)**2).sum(1)))
def fw(x,a=.22,sg=.2):
    V=np.zeros(len(x))
    for c in [(a,a,0),(-a,a,0),(a,-a,0),(-a,-a,0)]:
        rr=np.sqrt(((x-np.array(c))**2).sum(1))/sg; m=rr<1; V[m]-=np.exp(-1/(1-rr[m]**2))
    return V
for nm,V in (("double-well",dw(x)),("four-well",fw(x))):
    print(nm,"V range",V.min(),V.max()," <u1,V u1>=",(V*u2).mean()*vol," => E1 ~",np.pi**2+(V*u2).mean()*vol)
