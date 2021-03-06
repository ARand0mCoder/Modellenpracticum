import numpy as np
import matplotlib.pyplot as plt
import datetime

'''
--------- CONTENT ---------
This file contains:
 - a couple of auxilary functions for the limitied brute force
 - a couple of functions to plot results
Functions:
    plot_data       plots the whole dataset, takes power and time as outputted
                    by the read_data function from DataParsing
    plot_specific_time  plots the whole dataset at one specific time throughout
                        the year, for the given hour and minute.
	DistanceFromOldSol Computes the distance between two solutions.
		Variables:	OldSol		The first solution
				NewSol		The second solution
				PlugsPerField	The weight of each group
		
    	PlotAllSolutions	 plots the norm of multiple solutions in a graph.
    		Variables:	Result		All norms (sorted)
		
   	 DrawDistanceFunction plots the distance between different solutions.
    		Variables:	Solutions	All solutions as arrays of arrays
----------------------------
'''


def plot_data(power, time):
    for group in range(len(power[0])):
        # plt.figure()
        data = [line[group] for line in power]
        plt.plot(time,data)
    plt.show()


def plot_specific_time(power, time, hour, minute):
    timeData = datetime.time(hour,minute)

    for group in range(len(power[0])):
        Xdata = []
        Ydata = []
        for i in range(len(power)):
            if time[i].time() == timeData:
                Xdata.append(time[i])
                Ydata.append(power[i][group])
        plt.plot(Xdata,Ydata)
    plt.show()


def DistanceFromOldSol(OldSol, NewSol, PlugsPerField):
    TotalDistance = 0
	#Sum the distance by checking if each group stays in the same trafo.
    for Trafo in range(len(OldSol)):
        for group in OldSol[Trafo]:
            if not group in NewSol[Trafo]:
                TotalDistance += PlugsPerField[group]
    return TotalDistance
   

def DrawDistanceFunction(Solutions, weights, OldSol):
    NumOfGroups = sum(weights)
    AllDist = [0 for i in range(1, 2*NumOfGroups)]
	
	# Compute the distance for each distinct pair of groups.
    for i in range(0, len(Solutions)):
        for j in range(i + 1, len(Solutions)):
            AllDist[DistanceFromOldSol(Solutions[i], Solutions[j], weights)] += 1
            
    FurthestBar = 0
    for i in range(len(AllDist)):
        if AllDist[i] >= 1:
            FurthestBar = i + 5
	
    AllDist = AllDist[:FurthestBar-1]
    AllDist = np.array(AllDist)
    
	#Plot the distance of solutions with respect to each other.
    plt.bar([i for i in range(0, FurthestBar - 1)], AllDist)
    plt.ylabel("Aantal keer dat dit voorkomt")
    plt.xlabel("Aantal veranderingen ten opzichte van elkaar")
    plt.show()
    
    AllDist = [0 for i in range(0, FurthestBar - 1)]
    for j in range(1, len(Solutions)):
        AllDist[DistanceFromOldSol(Solutions[0], Solutions[j], weights)] += 1
    
    AllDist = np.array(AllDist)
	
	#Plot the distance of solutions with respect to the old solution.
    plt.bar([i for i in range(0, FurthestBar - 1)], AllDist)
    plt.ylabel("Aantal keer dat dit voorkomt")
    plt.xlabel("Aantal veranderingen ten opzichte van huidige oplossing")
    plt.show()
    
    AllDist = [0 for i in range(0, FurthestBar - 1)]
    for j in range(1, len(Solutions)):
        AllDist[DistanceFromOldSol(OldSol, Solutions[j], weights)] += 1
    
    AllDist = np.array(AllDist)
	
	#Plot the distance of solutions with respect to the best solution found.
    plt.bar([i for i in range(0, FurthestBar - 1)], AllDist)
    plt.ylabel("Aantal keer dat dit voorkomt")
    plt.xlabel("Aantal veranderingen ten opzichte van beste oplossing")
    plt.show()
   
def PlotAllSolutions(Result):
    plt.ylabel("Norm (%)")
    plt.xlabel("Beste oplossing")
    plt.plot(Result, "ro")
    plt.show()
