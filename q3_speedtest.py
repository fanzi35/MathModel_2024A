import numpy as np, time
import dragon_data, utils

def point_distance_sq(theta_a, theta_b, b):
    ra=b*theta_a; rb=b*theta_b; d=theta_a-theta_b
    return ra*ra+rb*rb-2*ra*rb*np.cos(d)

def solve_fast(theta_prev, distance, b, tol=1e-12):
    # initial from local arc length
    x = theta_prev + distance/(b*np.sqrt(1+theta_prev*theta_prev))
    # damped Newton
    for _ in range(12):
        delta=x-theta_prev
        f=b*b*(theta_prev*theta_prev+x*x-2*theta_prev*x*np.cos(delta))-distance*distance
        if abs(f) < 1e-12:
            return x
        fp=b*b*(2*x-2*theta_prev*np.cos(delta)+2*theta_prev*x*np.sin(delta))
        if fp<=0 or not np.isfinite(fp):
            break
        xn=x-f/fp
        if xn <= theta_prev or not np.isfinite(xn):
            xn=0.5*(x+theta_prev)
        if abs(xn-x)<tol:
            return xn
        x=xn
    # fallback bracket
    low=theta_prev; high=max(x, theta_prev+1e-4)
    step=distance/(b*np.sqrt(1+theta_prev*theta_prev))
    while point_distance_sq(high, theta_prev,b)<distance*distance:
        high += max(step,1e-4)
        step *= 1.5
    for _ in range(50):
        mid=(low+high)/2
        if point_distance_sq(mid,theta_prev,b)<distance*distance: low=mid
        else: high=mid
    return (low+high)/2

def state(pitch,r,fast=True):
 b=utils.spiral_coefficient(pitch); th=np.zeros(dragon_data.POINT_COUNT); th[0]=r/b
 d=dragon_data.get_handle_distances()
 for i in range(1,dragon_data.POINT_COUNT):
    if fast: th[i]=solve_fast(th[i-1],d[i-1],b)
    else: th[i]=utils.solve_trailing_theta(th[i-1],d[i-1],b)
 return th
for fast in [False, True]:
 t=time.time()
 for k in range(100): state(0.45,4.5+0.01*k,fast)
 print(fast, time.time()-t)
# compare max error one state
th1=state(0.45,4.575,False); th2=state(0.45,4.575,True)
print(np.max(np.abs(th1-th2)))
