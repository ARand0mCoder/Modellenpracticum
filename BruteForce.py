#import time
import numpy as np
from itertools import chain,combinations

def powersetcompl(iterable, maxgroupsize):
    s = list(iterable)
    return chain.from_iterable(combinations(s,r) for r in range(len(s)-maxgroupsize,int(np.floor(len(s)/2))+1)) #Only generates "first half" of the power set, as the problem is symmetric

def Norm(array): 
    return np.amax(np.abs(array))

def BruteForce(input,Norm, maxgroupsize, m): #input of the form [ [],[],[],  ,[] ]
    #t0 = time.time()
    totalinput = sum(input)
    bestmax = np.amax(np.abs(totalinput))
    bestnorm = Norm(totalinput)
    if m == 2:
        for elem in powersetcompl(np.arange(0,len(input)), maxgroupsize):
            power1 = np.zeros(len(input[0]))
            for x in elem:
                power1 += input[x]
            if Norm(power1) < bestnorm: 
                power2 = totalinput - power1
                if Norm(power2) < bestnorm: 
                    bestnorm = max(Norm(power1),Norm(power2))
                    bestmax = max(np.amax(power1),np.amax(power2))
                    group1 = []
                    for x in elem:
                        group1.append(x)
                    #print(bestnorm,bestmax)
        group2 = np.setxor1d(np.arange(0,len(input)),group1)
        #t1 = time.time()
    return(group1, group2, bestmax)#,t1-t0)

##
print(BruteForce(power1,Norm,9,2)) #Takes ~0.19 sec for len(input)=10, ~9 sec for len(input)=15, NO restriction on group size; restricting to e.g. 9 --> ~6,8 sec
