import numpy as np
import random

'''
--------- CONTENT ---------
This file contains the functions needed to run a Monte Carlo simulation.
Functions:
    AllowedDistr    Checks if a solution found is allowed. The number of groups should be below the maximum allowed.
        Variables:  Algosol         The solution which needs to be checked.
                    PlugsPerTrafo   Number of plugs which are allowed for each trafo.
                    PlugsPerField   The number of plugs each group has (the weight).
                    
    MNorm           Calculates the maximum of the norm of each trafo of a solution.
        Variables:  Data     The data were the norm should be taken of.
                    Sol      The solution which is going to be checked.
                    Norm     The norm which is going to be used.
                    
    MonteCarlo      Runs a Monte Carlo simulation of the problem. Returns the best solututions and their norms.
        Variables:  algoSol         The current solution for the problem.
                    Data            The data which is going to be used.
                    PlugsPerTrafo   Number of plugs which are allowed for each trafo.
                    PlugsPerField   The number of plugs each group has (the weight).
                    Iterations      The number of times the Monte Carlo has to be tried.
                    RejectionRate   How common it should be that a worse result is rejected. Typically between 10 and 100.
                    Norm            The norm which is going to be used.
                    NumberToSave    The number of best solutions which need to be saved.
                    Type            If the twist or swap method needs to be used.
                    Penalty         The penalty which is taken for the distance with regard to the old.
                    Prob            The Probability which is needed for the twist algoritm.
                    
    MonteCarloTwist Given a solution and a prob, returns a random new solution.
        Variables:  groups  The old solution which is going to be changed.
                    prob    The probability that a group stays with the old trafo.
                    
    MonteCarloSwap  Given a solution, swaps two random groups.
        Variables:  groups  The old solution which is going to be changed.
----------------------------
'''

def AllowedDistr(AlgoSol, PlugsPerTrafo, PlugsPerField):
    for Trafo in range(len(AlgoSol)):
        
        # Sum the number of plugs for this trafo.
        SumOfPlugs = 0
        for Field in AlgoSol[Trafo]:
            SumOfPlugs += PlugsPerField[Field]
            
        # Too many groups. Solution is not allowed.
        if SumOfPlugs > PlugsPerTrafo[Trafo]:
            return False
    return True

def MNorm(Data, Sol, Norm): 
    AllNorms = []
    for group in range(len(Sol)):
        
        #Compute norm of this trafo
        array = np.zeros(len(Data[0]))
        for station in Sol[group]:
            array += Data[station]
        AllNorms.append(Norm(array))
        
    return max(AllNorms)
    
    
def MonteCarlo(algoSol, Data, PlugsPerTrafo, PlugsPerField, Iterations, RejectionRate, Norm, NumberToSave, Type, Penalty = [0 for i in range(0, 1000)], prob = 0.8):
    
    BestSols, BestNorms = [algoSol[:]], [MNorm(Data, algoSol, Norm)] 
    CurrentSol, CurrentNorm = BestSols[0], BestNorms[0]
        
    for iteration in range(Iterations):
        
        # Calculate the new random solution.
        if Type == "twist":
            NewSol = MonteCarloTwist(CurrentSol, prob)
        else:
            NewSol = MonteCarloSwap(CurrentSol)
        
        # Norm of the new solution.
        NewNorm = MNorm(Data, NewSol, Norm)
        
        Distance = DistanceFromOldSol(algoSol, NewSol, PlugsPerField)

        #Check if distribution is allowed
        if AllowedDistr(NewSol, PlugsPerTrafo, PlugsPerField) and Distance < len(Penalty):
            
            # Add penaly for the distance
            NewNorm += Penalty[Distance]
            
            #If among the best solutions, add it.
            NewSol = [sorted(array) for array in NewSol]
            
            if (iteration < NumberToSave or NewNorm < BestNorms[len(BestNorms)-1]) and NewSol not in BestSols:

                #Add it to the best solution
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

            #If better than the current best solution, replace it
            if NewNorm < CurrentNorm:  
                # Take the new solution when the norm is smaller
                CurrentSol = NewSol[:]
                CurrentNorm = NewNorm
                
            # With a small probability continue with the new choice 
            elif random.random() < (float(CurrentNorm) / float(NewNorm))**RejectionRate:
                CurrentSol = NewSol[:]
                CurrentNorm = NewNorm

    return BestSols, BestNorms
      
def MonteCarloTwist(groups, prob): 
    
    m = len(groups)
    NewDist = [[] for i in range(m)];
    
    for group in range(m):
        
        for station in groups[group]:
            
            RandomNum = random.random()
            
            #probababilty prob it stays in the same box, probability (1-p)/(m-1) it goes to another box
            if RandomNum < prob: 
                # the case the it stays in his current group
                NewDist[group].append(station)
            else: 
                # the case where the groups changes
                RandomNum -= prob
                RandomNum *= (m-1)/(1-prob)
                RandomNum = int(RandomNum)
                if RandomNum >= group: 
                    #filtering out his own group (the "m-1")
                    NewDist[RandomNum + 1].append(station)
                else:
                    NewDist[RandomNum].append(station)
    return NewDist
        
def MonteCarloSwap(groups):
    numberofgroups = sum(len(group) for group in groups)
    
    #The two groups which are going to be swapped
    items = random.sample(list(range(numberofgroups)),2)
    item1 = items[0]
    item2 = items[1]
    
    
    newgroups = []
    for grp in groups:
        group = grp[:]
        
        # Swap if necessary
        if item1 in group and not item2 in group:
            group.remove(item1)
            group.append(item2)
        elif item2 in group and not item1 in group:
            group.remove(item2)
            group.append(item1)
        newgroups.append(group)
    return newgroups
   
