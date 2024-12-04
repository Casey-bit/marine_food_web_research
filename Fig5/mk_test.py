import numpy as np
import math

def mk_test(data):
    length = len(data)
    s=0
    for m in range(0,length-1):
        for n in range(m+1,length):
            if data[n] > data[m]:
                s=s+1
            elif data[n] == data[m]:
                s=s+0
            else:
                s=s-1
    #计算vars
    vars=length*(length-1)*(2*length+5)/18
    #计算zc
    if s>0:
        zc=(s-1)/math.sqrt(vars)
    elif s==0:
        zc=0
    else:
        zc=(s+1)/math.sqrt(vars)
        
    #计算za    
    za=abs(zc)
        
    #计算倾斜度
    ndash=length*(length-1)/2
    slope1=np.zeros(int(ndash))
    m=0
    for k in range(0,length-1):
        for j  in range(k+1,length):
            slope1[m]=(data[j] - data[k])/(j-k)
            m=m+1
            
    slope=np.median(slope1)

    return slope, za