import numpy as np

Test = [[2,5],[3,5],[4,1],[-2,-3],[6,4]] # Test case. Remove in final program
Test2 = []
for i in range(len(Test)):
 Test2.append(np.array(Test[i]))
# run test case as "print(FirstAlgorithm(Test2,2))"

def Norm(array): #Choose your own norm. Average of the highest n is also possible
    # Make sure that Norm greater or equal to 0
    return np.amax(np.abs(array))

def FirstAlgorithm(data, m):
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
    
