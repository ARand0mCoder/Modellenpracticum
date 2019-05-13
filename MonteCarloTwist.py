import numpy as np
import random

def AllowedDistr(AlgoSol, PlugsPerTrafo, PlugsPerField):
    for Trafo in range(len(AlgoSol)):
        SumOfPlugs = 0
        for Field in AlgoSol[Trafo]:
            SumOfPlugs += PlugsPerField[Field]
        if SumOfPlugs > PlugsPerTrafo[Trafo]:
            return False
    return True
    
def DistanceFromOldSol(OldSol, NewSol):
    NotTheSame = 0
    for Trafo in range(len(OldSol)):
        for i in OldSol[Trafo]:
            if i not in NewSol[Trafo]:
                NotTheSame += 1
    return NotTheSame

def MNorm(Data, Sol, Norm): # Calculates the maximum of the norms of a solution
    AllNorms = []
    
    for group in range(len(Sol)):
        array = np.zeros(len(Data[0]))
        for station in Sol[group]:
            array += Data[station]
        AllNorms.append(Norm(array))
    return max(AllNorms)
    
    
def MonteCarlo(algoSol, Data, PlugsPerTrafo, PlugsPerField, Iterations, RejectionRate, Norm, NumberToSave, type, Penalty, prob = 0.8):
    
    BestSols, BestNorms = [algoSol[:]], [MNorm(Data, algoSol, Norm)] 
    CurrentSol, CurrentNorm = BestSols[0], BestNorms[0]
        
    for iteration in range(Iterations):
        
        if type == "twist":
            NewSol = MonteCarloTwist(CurrentSol, prob)
        else:
            NewSol = MonteCarloSwap(CurrentSol)
            
        NewNorm = MNorm(Data, NewSol, Norm)
        
        Distance = DistanceFromOldSol(algoSol, NewSol)

        #check if distribution is allowed
        if AllowedDistr(NewSol, PlugsPerTrafo, PlugsPerField) and Distance < len(Penalty):
            NewNorm += Penalty[Distance]
            
            #If among the best, add it.
            NewSol = [sorted(array) for array in NewSol]
            if (iteration < NumberToSave or NewNorm < BestNorms[len(BestNorms)-1]) and NewSol not in BestSols:

                IsAdded = False
                for i in range(len(BestSols)):
                    if NewNorm < BestNorms[i]:
                        BestNorms.insert(i, NewNorm)
                        BestSols.insert(i, NewSol)
                        IsAdded = True
                        break
                        
                if not IsAdded:
                    BestNorms.append(NewNorm)
                    BestSols.append(NewSol)
                    
                if len(BestSols) > NumberToSave:
                    BestSols.pop()
                    BestNorms.pop()

            #If better then current replace
            if NewNorm < CurrentNorm:  
                # Take the new solution when the norm is smaller
                CurrentSol = NewSol[:]
                CurrentNorm = NewNorm
            elif random.random() < (float(CurrentNorm) / float(NewNorm))**RejectionRate:
                # With a small probability continue with the new choice 
                CurrentSol = NewSol[:]
                CurrentNorm = NewNorm

    return BestSols, BestNorms
      
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
        
def MonteCarloSwap(groups):
    numberofgroups = sum(len(group) for group in groups)
    items = random.sample(list(range(numberofgroups)),2)
    item1 = items[0]
    item2 = items[1]
    newgroups = []
    for grp in groups:
        group = grp[:]
        if item1 in group and not item2 in group:
            group.remove(item1)
            group.append(item2)
        elif item2 in group and not item1 in group:
            group.remove(item2)
            group.append(item1)
        newgroups.append(group)
    return newgroups
