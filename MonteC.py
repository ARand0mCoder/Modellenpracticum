import numpy as np
import random
import copy

def value(list):
    max = 0
    for i in list:
        if max<sum(i):
            max = sum(i)
    return max
    
    
def probchange(old,new): #als functie van old/new
    ratio = (old/new)**30
    print(ratio)
    return ratio
#    return 0


def MonteC(algosol, n, m, iter): # m aantal stations, n aantal ontvangers, iter is #iteraties
    prevsol = algosol
    prevvalue = value(prevsol)
    newsol = copy.deepcopy(prevsol)  #bewaar de oude oplossing om te vergelijken
    
    for i in range (iter): #herhaal iter keer
        
        cng1,cng2=0,0
        while cng1 == cng2:
            cng1, cng2 = np.random.random_integers(0,m-1,2) #verwissel 2 elementen in verschillende groepen
            
        change1, change2 = random.choice(newsol[cng1]),random.choice(newsol[cng2])
        newsol[cng1].append(change2)
        newsol[cng2].append(change1)
        newsol[cng1].remove(change1)
        newsol[cng2].remove(change2)
        
        newvalue = value(newsol) #vind nieuwe waarde
        
        rate = probchange(prevvalue,newvalue)
        
        if newvalue <= prevvalue: #vervang als de nieuwe waarde kleiner is
            prevsol = copy.deepcopy(newsol)
            prevvalue = newvalue

        elif random.random()<rate: #kans op vervanging anders
            prevsol = copy.deepcopy(newsol)
            prevvalue = newvalue

        else: #als je niet vervangt
            newsol = copy.deepcopy(prevsol)

    return prevsol, prevvalue
    
sol = [[101,27,53,65],[15,51,88,99],[23,7,99,34],[60,55,78,64]]

print(MonteC(sol,12,4,10000)[1])
    
    
    