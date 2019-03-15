import numpy as np
import random

def Norm2(array): #Calculates the norm of an array (globally defined)
    return np.amax(np.abs(array))   
    
def MaxNorm(Data, Sol, Norm): # Calculates the maximum of the norms of a solution
    AllNorms = []
    
    for group in range(len(Sol)):
        array = np.zeros(len(Data[0]))
        for station in Sol[group]:
            array += Data[station]
        AllNorms.append(Norm(array))
    return max(AllNorms)

def MonteCarloTwist(groups, prob):  #groups given as indices of the stations
    m = len(groups)
    NewDist = [[] for i in range(m)];
    for group in range(m):
        for station in groups[group]:
            RandomNum = random.random() #probababilty prob it stays in the same box, probability (1-p)/(m-1) it goes to another box
            if RandomNum < prob: 
                # the case the it stays in his current group
                NewDist[group].append(station)
            else: 
                # the case where the groups changes
                RandomNum -= prob
                RandomNum *= (m-1)/(1-prob)
                RandomNum = int(RandomNum)
                if RandomNum >= group: #filtering out his own group (the "m-1")
                    NewDist[RandomNum + 1].append(station)
                else:
                    NewDist[RandomNum].append(station)
    return NewDist

def MonteCarloTwistAlg(algoSol, Data, Iterations, prob, RejectionRate, Norm):
    
    oldSol = algoSol[:]
    oldNorm = MaxNorm(Data, oldSol, Norm)
    currentSol = algoSol[:]
    currentNorm = oldNorm

    for iteration in range(Iterations):
        newSol = MonteCarloTwist(currentSol, prob)
        newNorm = MaxNorm(Data, newSol, Norm)
        
        if newNorm < currentNorm:  
            # take the new solution when the norm is smaller
            currentSol = newSol[:]
            currentNorm = newNorm
        elif newNorm == currentNorm:   
            # 50/50 change to change if the norms coincide
            if random.randrange(2) == 0:
                currentSol = newSol[:]
                currentNorm = newNorm
        else:  
            if random.random() > RejectionRate:
                currentSol = newSol[:]
                currentNorm = newNorm
    return currentNorm, currentSol #, oldSol, oldNorm
        
def MonteCarloSwap(groups):
    numberofgroups = sum(len(group) for group in groups)
    items = random.sample(list(range(numberofgroups)),2)
    item1 = items[0]
    item2 = items[1]
    newgroups = []
    for group in groups:
        if item1 in group and not item2 in group:
            group.remove(item1)
            group.append(item2)
        if item2 in group and not item1 in group:
            group.remove(item2)
            group.append(item1)
        newgroups.append(group)
    return newgroups
    
def MonteCarloSwapAlg(algoSol, Data, Iterations, RejectionRate, Norm):

    oldSol = algoSol[:]
    oldNorm = MaxNorm(Data, oldSol, Norm)
    currentSol = algoSol[:]
    currentNorm = oldNorm

    for iteration in range(Iterations):
        newSol = MonteCarloSwap(currentSol)
        newNorm = MaxNorm(Data, newSol, Norm)
        
        if newNorm < currentNorm:  
            # take the new solution when the norm is smaller
            currentSol = newSol[:]
            currentNorm = newNorm
        else:  
            if random.random() > RejectionRate:
                currentSol = newSol[:]
                currentNorm = newNorm
    return currentNorm, currentSol  #, oldSol, oldNorm
    
Test2 = []
for i in range(20):
 Test2.append(np.random.rand(100))

for i in range(180, 201):
    print(i/200, MonteCarloTwistAlg([[0,1,2,3,4,5,6,7,8,9],[10,11,12,13,14,15,16,17,18,19]], Test2, 20000, 0.9, i/200, Norm2)) 
    print(i/200, MonteCarloSwapAlg([[0,1,2,3,4,5,6,7,8,9],[10,11,12,13,14,15,16,17,18,19]], Test2,20000, i/200, Norm2))     
    
