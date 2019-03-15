import numpy as np
import random

def Norm(array): #Calculates the norm of an array (globally defined)
    return np.amax(np.abs(array))   
    
def MaxNorm(Data, Sol): # Calculates the maximum of the norms of a solution
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
    
def MonteCarlo(algoSol, Data, Iterations, prob):
    
    oldSol = algoSol[:]
    oldNorm = MaxNorm(Data, oldSol)
    
    currentSol = algoSol[:]
    currentNorm = oldNorm

    for iteration in range(Iterations):
        newSol = MonteCarloTwist(currentSol, prob)
        newNorm = MaxNorm(Data, newSol)
        
        if newNorm < currentNorm:  
            # take the new solution when the norm is smaller
            currentSol = newSol[:]
            currentNorm = newNorm
        elif newNorm == currentNorm:   
            # 50/50 change to change if the norms coincide
            if random.randrange(2) == 0:
                currentSol = newSol[:]
                currentNorm = newNorm
    return currentSol, currentNorm, oldSol, oldNorm
    
    
Test = [[2,5,3],[3,5,4],[4,1,3],[-2,-3,-1],[6,4,5],[4,5,6],[6,3,4],[5,4,3]] # Test case. Remove in final program
Test2 = []
for i in range(len(Test)):
 Test2.append(np.array(Test[i]))
print(MonteCarlo([[0,1,4,5],[2,3,6,7]], Test2, 10000, 0.6)) 
    