import numpy as np


def Norm1(array): #Choose your own norm. Average of the highest n is also possible
    # Make sure that Norm greater or equal to 0
    return np.amax(array)
    
def Norm2(array):
    return np.amax(np.abs(array))
    
def Norm3(array):
    aamax = np.amax(array)
    aamin = np.amin(array)
    if abs(aamax) > abs(aamin):
        return aamax
    else:
        return aamin
        
def NormDistr(data, Distr, Norm):
    newNorms = []
    for j in range(len(Distr)):
        sumarray = np.zeros(len(data[0]))
        for k in Distr[j]:
            sumarray += data[k]
        newNorms.append(Norm(sumarray))
    return max(newNorms)
    
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
    
sab,sac,sbc,sba,sca,scb,sa,sb,sc = 0,0,0,0,0,0,0,0,0
m = 2 # number of transformators
for i in range(1000): # number of iterations
    Test2 = []
    for j in range(20): # number of groups to distribute
        Test2.append(4 * np.random.rand(100) - np.random.rand(100)) # change the lenght of the arrays or their distribution
    # Test2[0] = -2*Test2[0] # This can change individual values to add "solar parks"
    
    a = FirstAlgorithm(Test2, m ,Norm2)
    b = SecondAlgorithm(Test2, m, Norm1)
    c = SecondAlgorithm(Test2, m, Norm3)
    if a != b or a!=c or b!=c:
        Norma = NormDistr(Test2, a, Norm3)
        Normb = NormDistr(Test2, b, Norm3)
        Normc = NormDistr(Test2, c, Norm3)
        if Norma > Normb:
            sab+=1
            if Norma > Normc:
                sa+=1
        elif Normb > Norma:
            if Normb > Normc:
                sb+=1
            sba+=1
            
        if Norma > Normc:
            sac+=1
        elif Normc > Norma:
            if Normc > Normb:
                sc+=1
            sca+=1
            
        if Normb > Normc:
            sbc+=1
        elif Normc > Normb:
            scb+=1        
        #print(Norma,Normb,Normc)
        
print(sab,sba, "First to Second with max norm")
print(sac,sca,"First to Second with sign abs max norm")
print(sbc,scb, "Second max norm vs sign abs max norm")
print(sa, "times the first algorithm is alone the best")
print(sb, "times the second algorithm with max norm is alone the best")
print(sc, "times the second algorithm with sign abs max norm is alone the best")


