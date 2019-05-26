import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime

# Enter directory of you files here, so that python knows where to look
file_dir = r'C:\Users\XpsBook\Documents\Radboud Universiteit Nijmegen\Jaar 3\Modellenpracticum'
sys.path.append(file_dir)

import DataProcessing as dpr
import DataParsing as dpa
import GenerateData as gd
import GreedyAlgorithm as GA
import Karmarkar_Karp_v1 as kkk
import MonteCarloTwist as MCT

#%%
def MaxNorm(array): #Choose your own norm. Average of the highest n is also possible
    # Make sure that Norm greater or equal to 0
    return np.amax(np.abs(array))

def Highest_k_Norm_sum(k, array):
    # k is assumed to be smaller than len(array)
    return sum(np.sort(np.abs(array))[-k :]) / k

def Highest_k_pow_Norm(p, k, array):
    return sum(np.power(np.sort(np.abs(array))[-k :], p))

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

#%%
#Hieronder volgt de lijst met capaciteiten, aantal stekkers, en nummer van de dubbele met weging
Ubase = 10500
#BZSV, stekker, dubbel  = [29.4,14.7], [15,5], [] #Bemmel
#BZSV, stekker, dubbel  = [52,22], [24,24], [] #Hoge Kadijk, in dit bestand kan ik niet netjes lezen of er dubbele kabels zijn...
#BZSV, stekker, dubbel  = [22,22],[12,12], [[112,114],[124,126]] #Uilenburg
#BZSV, stekker, dubbel  = [60,60],[17,20], [[107,108,111],[163,164]] #Zevenhuizen
#BZSV, stekker, dubbel  = [46,23.1],[21,19], [[22,24],[13,15,17],[7,9,11],[44,45],[3,4,5,6,7,8]] #Drachten
#BZSV, stekker, dubbel  = [], [], [[44,45,54],[43,50,52],[46,55],[48,56,57],[47,58],[32,34],[33,35],[2,6],[9,13]] #Emmeloord
#BZSV, stekker, dubbel  = [22,44], [12,18], [[103,104],[254,256],[255,259],[260,267],[263,270]], [[302,303],[402,403],[502,503],[602,603]] #Rijksuniversiteit, de derde verzameling heeft 2 kabels per groep nodig, de laatste maar 1 kabel per groep
#BZSV, stekker, dubbel  = [60.59845,29,31.5],[24,10,17],[[5,6],[3,15],[16,17],[18,19],[85,86],[87,93,94,95],[55,56,57,58,61],[72,75]] #WWG

path1 = r'C:\Users\XpsBook\Documents\Radboud Universiteit Nijmegen\Jaar 3\Modellenpracticum\Power npy\metingen_Uilenburg.npy'
path2 = r'C:\Users\XpsBook\Documents\Radboud Universiteit Nijmegen\Jaar 3\Modellenpracticum\Bestanden Alliander\Power Data\Uilenburg_2_2018_clean.csv'

rows_to_trim1 = [0, 1, 2, 3, 4, 5, 8, 9]
rows_to_trim2 = [0, 2, 3, 4, 7]

if 'metingen_' not in path1:
    power1, time1, time_diff1 = dpa.read_data(path1)
    power2, time2, time_diff2 = dpa.read_data(path2)

    power1 = np.transpose(np.array(power1))
    power2 = np.transpose(np.array(power2))

    power1 = dpr.auto_trim(power1, rows_to_trim1, time_diff1)
    power2 = dpr.auto_trim(power2, rows_to_trim2, time_diff2)
    #%%
    power = []
    for row in power1:
        power.append(row[0:34944])
    for row in power2:
        power.append(row[0:34944])
else:
    time = dpa.date_linspace(datetime.datetime(2016,1,1,0,0),datetime.datetime(2019,1,1,0,0),datetime.timedelta(minutes=5))
    power = np.load(path1)
    power,time = dpa.grab_year(power,time,2016)
    power = np.transpose(np.array(power))
#%%

def plot_solution(power, sol, m):
    for i in range(m):
        tot = 0
        for j in sol[i]:
            tot += power[j]
        plt.plot(tot)
        plt.show()
#%%

def sol_norms(power, sol, Norm):
    for group in sol:
        tot = np.zeros(len(power[0]))
        for val in group:
            tot += power[val]
        print(Norm(tot))

#%%

def solve(power, Norm):
    sol = GA.FirstAlgorithm(power, 2, Norm)
    plot_solution(power, sol, 2)
    sol_norms(power, sol, Norm)
    sol_norms(power, sol, MaxNorm)
    print(sol)


def solve2(power, Norm):
    sol = GA.SecondAlgorithm(power, 2, Norm)
    plot_solution(power, sol, 2)
    sol_norms(power, sol, Norm)
    sol_norms(power, sol, MaxNorm)
    print(sol)


'''
Data 'bootstrappen'
'''
solve(power,MaxNorm)
