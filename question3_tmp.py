from __future__ import annotations
import numpy as np
import dragon_data, utils

def state(pitch, boundary_radius=4.5):
    b=utils.spiral_coefficient(pitch)
    theta=np.zeros(dragon_data.POINT_COUNT); theta_dot=np.zeros_like(theta); radius=np.zeros_like(theta); pos=np.zeros((dragon_data.POINT_COUNT,2)); speed=np.zeros_like(theta)
    theta[0]=boundary_radius/b
    distances=dragon_data.get_handle_distances()
    for i in range(1,dragon_data.POINT_COUNT):
        theta[i]=utils.solve_trailing_theta(theta[i-1], distances[i-1], b)
    radius=b*theta
    pos[:,0],pos[:,1]=utils.polar_to_cartesian(radius,theta)
    theta_dot[0] = -1.0/(b*np.sqrt(1+theta[0]**2))
    for i in range(1,dragon_data.POINT_COUNT):
        delta=theta[i]-theta[i-1]
        numerator=theta[i-1]-theta[i]*np.cos(delta)-theta[i]*theta[i-1]*np.sin(delta)
        denominator=theta[i]-theta[i-1]*np.cos(delta)+theta[i]*theta[i-1]*np.sin(delta)
        theta_dot[i]=-(numerator/denominator)*theta_dot[i-1]
    speed=utils.speed_from_theta(theta,theta_dot,b)
    return {'theta':theta,'theta_dot':theta_dot,'radius':radius,'position':pos,'speed':speed,'pitch':pitch}

def rect_arrays(position):
    front=position[:-1]; rear=position[1:]
    direction=front-rear
    norm=np.linalg.norm(direction,axis=1)
    direction=direction/norm[:,None]
    normal=np.column_stack([-direction[:,1],direction[:,0]])
    center=0.5*(front+rear)
    return {'center':center,'direction':direction,'normal':normal,'length':dragon_data.get_bench_lengths().astype(float),'width':float(dragon_data.BENCH_WIDTH)}

def all_pairs():
    pairs=[]
    n=dragon_data.BENCH_COUNT
    for i in range(n):
        for j in range(i+2,n):
            pairs.append((i,j))
    return np.array(pairs,dtype=int)
PAIRS=all_pairs()

def sat(rects,pairs=PAIRS):
    a=pairs[:,0]; b=pairs[:,1]
    ca=rects['center'][a]; cb=rects['center'][b]
    da=rects['direction'][a]; db=rects['direction'][b]
    na=rects['normal'][a]; nb=rects['normal'][b]
    la=rects['length'][a]; lb=rects['length'][b]; w=rects['width']
    axes=np.stack([da,na,db,nb],axis=1)
    delta=ca-cb
    proj=np.abs(np.einsum('mi,mki->mk',delta,axes))
    ha=0.5*la[:,None]*np.abs(np.einsum('mki,mi->mk',axes,da))+0.5*w*np.abs(np.einsum('mki,mi->mk',axes,na))
    hb=0.5*lb[:,None]*np.abs(np.einsum('mki,mi->mk',axes,db))+0.5*w*np.abs(np.einsum('mki,mi->mk',axes,nb))
    axis_gaps=proj-ha-hb
    gaps=np.max(axis_gaps,axis=1)
    return gaps

def gap(p):
    st=state(p)
    gaps=sat(rect_arrays(st['position']))
    k=int(np.argmin(gaps))
    return float(gaps[k]), tuple(PAIRS[k])

if __name__=='__main__':
    import time
    t=time.time()
    for p in [0.3,0.4,0.45,0.5,0.55,0.6,0.8]:
        print(p,gap(p))
    print('time',time.time()-t)
