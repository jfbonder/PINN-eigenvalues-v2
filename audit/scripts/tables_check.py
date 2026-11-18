import numpy as np
from scipy.special import jn_zeros, jv
from scipy.optimize import brentq
from math import pi, sqrt
def jzeros(nu, k):
    xs=np.linspace(0.1,40,40000); f=jv(nu,xs); z=[]
    for i in range(len(xs)-1):
        if f[i]*f[i+1]<0: z.append(brentq(lambda x: jv(nu,x), xs[i],xs[i+1]))
    return z[:k]
def rep(name, rows):
    print("==",name)
    for lab, ex, comp, rel, res in rows:
        err=abs(comp-ex); r=err/ex
        print(f"{lab:8s} exact={ex:.6f} comp={comp:.6f} relerr={r:.3e} (paper {rel:.2e}) ratio={r/rel:.3f} |err|={err:.4f} sqrt(res)={sqrt(res):.4f} Weinstein {'OK' if err<=sqrt(res) else 'VIOLATED'}")
rep("T1 square",[("1,1",2*pi**2,19.739033,8.9e-6,6.72e-3),("1,2",5*pi**2,49.385078,7.5e-4,1.16e-1),("2,2",8*pi**2,78.962188,6.78e-5,1.26e-1),("1,3",10*pi**2,98.594123,1.03e-3,1.40e-3)])
print("ball3 zeros^2:")
allz=[]
for m in range(0,8):
    for n,z in enumerate(jzeros(m+0.5,4),1): allz.append((z*z,m,n,2*m+1))
for e in sorted(allz)[:12]: print("  %.4f m=%d n=%d mult=%d"%e)
d=dict(((m,n),e) for e,m,n,_ in allz)
rep("T2 ball3",[("0,1",d[0,1],9.885022,1.56e-3,8.61e-2),("1,1",d[1,1],20.182569,4.04e-4,4.59e-2),("2,1",d[2,1],33.243410,7.81e-4,1.00e-1),("0,2",d[0,2],39.494155,3.99e-4,1.81e-1),("3,1",d[3,1],48.826976,8.64e-5,2.24e-1),("1,2",d[1,2],59.734069,9.14e-4,2.04e-1),("4,1",d[4,1],66.991853,5.61e-4,4.82e-1)])
print("ball4 zeros^2:")
allz=[]
for m in range(0,8):
    for n,z in enumerate(jzeros(m+1,4),1): allz.append((z*z,m,n,(m+1)**2))
for e in sorted(allz)[:10]: print("  %.4f m=%d n=%d mult=%d"%e)
d=dict(((m,n),e) for e,m,n,_ in allz)
rep("T3 ball4",[("0,1",d[0,1],14.641796,2.73e-3,3.84e-3),("0,1 rev2",d[0,1],14.641796,2.73e-3,3.84e-2),("1,1",d[1,1],26.257056,4.47e-3,2.13e-1),("2,1",d[2,1],40.794700,2.16e-3,1.98e-1),("0,2",d[0,2],49.175395,8.74e-4,6.04e-1)])
rep("T4 tri",[("1,2",5*pi**2,49.369942,4.44e-4,2.42e-1),("1,3",10*pi**2,98.892161,1.99e-3,1.26),("2,3",13*pi**2,128.048509,2.00e-3,1.98),("1,4",17*pi**2,167.687185,5.73e-4,2.16)])
print("tri spectrum <=180:",sorted((pi**2*(m*m+n*n),m,n) for m in range(1,6) for n in range(1,m) if pi**2*(m*m+n*n)<185))
print("square spectrum <=105:",sorted((round(pi**2*(m*m+n*n),3),m,n) for m in range(1,5) for n in range(1,m+1) if pi**2*(m*m+n*n)<105))
rep("T5 plap disk",[("1",6.5320,6.679258,2.25e-2,1.81),("2",18.395,19.441806,5.69e-2,0.965)])
rep("T6 plap sq",[("1",5.4952,5.861383,6.66e-2,6.22e-2),("2",15.144,15.144490,3.24e-5,1.64)])
print("disk zeros^2:", sorted([(z*z,m,n) for m in range(0,5) for n,z in enumerate(jn_zeros(m,3),1)])[:8])
# annulus 0.5..1 first eig
from scipy.special import yv
f=lambda k: jv(0,k*0.5)*yv(0,k)-jv(0,k)*yv(0,k*0.5)
print("annulus E1=",brentq(f,5,7)**2)
