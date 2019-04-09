#import time
import numpy as np
from itertools import chain,combinations

def powersetcompl(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s,r) for r in range(int(np.floor(len(s)/2))+1)) #Only generates "first half" of the power set, as the problem is symmetric

def BruteForce(input, m): #input of the form [ [],[],[],  ,[] ]
    #t0 = time.time()
    totalinput = sum(input)
    bestmax = max(totalinput)
    if m == 2:
        for elem in powersetcompl(np.arange(0,len(input))):
            power1 = np.zeros(len(input[0]))
            for x in elem:
                power1 += input[x]
            if max(power1) < bestmax: #Most time-consuming part
                power2 = totalinput - power1
                if max(power2) < bestmax: 
                    bestmax = max(max(power1),max(power2))
                    group1 = []
                    for x in elem:
                        group1.append(x)
                    #print(bestmax)
        group2 = np.setxor1d(np.arange(0,len(input)),group1)
        #t1 = time.time()
    return(group1, group2)#, t1-t0)

##
print(BruteForce(ZZZZZ,2)) #Takes ~2.1 sec for len(input)=10
