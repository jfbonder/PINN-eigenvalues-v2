import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
def end(E,p,d,r0,tol):
    def f(r,y):
        u,w=y
        return [np.sign(w)*abs(w/r**(d-1))**(1/(p-1)),-E*r**(d-1)*np.sign(u)*abs(u)**(p-1)]
    c=(E/d)**(1/(p-1))
    return solve_ivp(f,[r0,1],[1-(p-1)/p*c*r0**(p/(p-1)),-E*r0**d/d],rtol=tol,atol=tol*1e-2,method='DOP853').y[0,-1]
for p,d,a,b in [(2,3,8,11),(2.2,2,6,7),(2.2,3,11,12),(2.5,2,7,8.5),(2.5,3,13,15),(3,2,9,11)]:
    for r0,tol in [(1e-5,1e-10),(1e-8,1e-12)]:
        print(p,d,r0,brentq(lambda E:end(E,p,d,r0,tol),a,b,xtol=1e-12))
