import numpy as np


def Norm1(array): #Choose your own norm. Average of the highest n is also possible
    # Make sure that Norm greater or equal to 0
    return np.amax(array)
    
def Norm2(array):
    return np.amax(np.abs(array))
    
def FirstAlgorithm(data, m, Norm):
    # data is the measurements of the power at ceratain times
    # m is number of components (assume it to be at least 2 (but still small))
    
    n = len(data) # number of times power is measured (assume it to be big
    FinalDistribution = [] # the final distribution of the groups to ceratain components
    PotentialAdditions = []
    PotentialAdditionsNorm = []
    GroupsToDistribute = list(range(len(data))) #The groups which still have to be distributed 
    
    for i in range(m):
        FinalDistribution.append([])
        PotentialAdditions.append(data)
        PotentialAdditionsNorm.append([])
   
    for i in range(n):
        NewPotAddNorm = Norm(data[i])
        for j in range(m):
            PotentialAdditionsNorm[j].append(NewPotAddNorm)
    
    PotentialAdditionsNorm = np.transpose(np.array(PotentialAdditionsNorm))
    PotentialAdditions = np.swapaxes(np.array(PotentialAdditions), 0,1)

    
    for i in range(n):
        
        WorstAddition = np.amax(PotentialAdditionsNorm) # the biggest norm which can be achieved by adding 1 group to the solutions
        IndexWorstAddition = np.argwhere(PotentialAdditionsNorm == WorstAddition)[0][0] # the index where this happens
        
        BestToAdd = np.amin(PotentialAdditionsNorm[IndexWorstAddition]) # the index where the worst group keeps the norm the lowest
        IndexAddAddition = np.argwhere(PotentialAdditionsNorm[IndexWorstAddition] == BestToAdd)[0][0] 

        FinalDistribution[IndexAddAddition].append(IndexWorstAddition)
        GroupsToDistribute.remove(IndexWorstAddition)
        
        for j in range(m):
            PotentialAdditionsNorm[IndexWorstAddition][j] = np.array([0]) # Change the norms of the added group to 0. They won't be chosen anymore
            
        for j in GroupsToDistribute:
            PotentialAdditions[j][IndexAddAddition] += data[IndexWorstAddition] #Update the Potential additions and norms
            PotentialAdditionsNorm[j][IndexAddAddition] = Norm(PotentialAdditions[j][IndexAddAddition])
            
    return FinalDistribution

def SecondAlgorithm(data, m, Norm):
    # data is the measurements of the power at ceratain times
    # m is number of components (assume it to be at least 2 (but still small))
    
    n = len(data) # number of times power is measured (assume it to be big)
    FinalDistribution = [[] for i in range(m)] # the final distribution of the groups to certain components
    PotAdd = [data[:] for i in range(m)] #Potential additions to the transformators
    PotAddNorms = [[] for i in range(m)] 
    DistrNorms = [0 for i in range(m)] # keeps track of the norm of the current distribution
    GroupsToDistribute = list(range(n)) #The groups which still have to be distributed 
   
    for i in range(n):
        NewPotAddNorms = Norm(data[i])
        for j in range(m):
            PotAddNorms[j].append(NewPotAddNorms)
    
    PotAddNorms = np.transpose(np.array(PotAddNorms))
    PotAdd = np.swapaxes(np.array(PotAdd), 0,1)

    for i in range(n):
        NormDifferences = [[max([abs(PotAddNorms[j][k]-DistrNorms[l]) for l in range(m)]) for k in range(m)] for j in GroupsToDistribute]
        #Gives the difference between the norms if a group is potentially added. 
        
        WorstNormDifferences = [] # takes the maximum of these differences
        for j in range(n):
            if j in GroupsToDistribute:
                WorstNormDifferences.append(max(NormDifferences[GroupsToDistribute.index(j)]))
            else:
                WorstNormDifferences.append(0)
                
        WorstGroupToAdd = WorstNormDifferences.index(max(WorstNormDifferences))
        # WorstGroupToAdd is the index in the remaining groups of the group which causes the biggest difference
        
        NormDifWorstGroup = NormDifferences[GroupsToDistribute.index(WorstGroupToAdd)]
        # NormDifWorstGroup is the difference in these norms.
        
        BestTransformator = NormDifWorstGroup.index(min(NormDifWorstGroup))
        # The transformator where this group is added to
        
        FinalDistribution[BestTransformator].append(WorstGroupToAdd)
        GroupsToDistribute.remove(WorstGroupToAdd)
        
        DistrNorms[BestTransformator] = PotAddNorms[WorstGroupToAdd][BestTransformator]
            
        for j in GroupsToDistribute: #Update the Potential additions and norms
            PotAdd[j][BestTransformator] += data[WorstGroupToAdd] 
            PotAddNorms[j][BestTransformator] = Norm(PotAdd[j][BestTransformator])
            
    return FinalDistribution
