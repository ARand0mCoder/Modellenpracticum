'''
--------- CONTENT ---------
This file contains:
 - a couple of auxilary functions for the limitied brute force
 - a couple of functions to plot results
Functions:
    (??) Computes the distance between certain solutions.
    
    (??) plots the norm of multiple solutionsin a graph.
    DrawDistanceFunction plots the distance between different solutions.
----------------------------
'''


def DrawDistanceFunction(Solutions):
    NumOfGroups = sum([len(Solutions[1][i]) for i in range(0, len(Solutions[1]))])
    AllDist = [0 for i in range(1, 2*NumOfGroups)]
    for i in range(0, len(Solutions)):
        for j in range(i + 1, len(Solutions)):
            AllDist[DistanceFromOldSol(Solutions[i], Solutions[j])] += 1
    print(AllDist)
    AllDist = np.array(AllDist)
    print(AllDist)
    plt.bar([i for i in range(1, 2*NumOfGroups)], AllDist)
    plt.show()
   
