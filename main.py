import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime

# Enter directory of you files here, so that python knows where to look
file_dir = r'C:\users\dionl\Documenten\Modellenpracticum'
sys.path.append(file_dir)

import DataProcessing as dpr
import DataParsing as dpa
import GenerateData as gd
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

# BZSV, stekker = [29.4,14.7], [15,5] # Bemmel
# BZSV, stekker = [52,22], [24,24] # Hoogte Kadijk
# BZSV, stekker = [22,22],[12,12] # Uilenburg
# BZSV, stekker = [60,60],[17,20] # Zevenhuizen
# BZSV, stekker = [46,23.1],[21,19] # Drachten
# BZSV, stekker = [44,44], [18,22] # Emmeloord
# BZSV, stekker = [22,44], [12,18] # Rijksuniversiteit
# BZSV, stekker = [60.59845,29,31.5],[24,10,17] # WWG
# Capacity condition: I x Ubase x sqrt(3) < BZSV

Ubase = 10500 # (V)
station = r'Rijksuniversiteit'

base_path = r'C:\Users\dionl\Documents\Modellenpracticum\Power npy\metingen_'
power_path = base_path + station + r'.npy'
groups_path = base_path + station + r'_groups.npy'

groups = np.load(groups_path)
power = np.load(power_path)

if station == r'Bemmel':
    BZSV, stekker = [29.4,14.7], [15,5]
    groups = groups[[0,1,2,3,4,5,6,7,8,9,10,11,12,23,24,25,26,27]]
    power = power[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,23,24,25,26,27]]
    
    rows_to_trim = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16]
    weights = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

elif station == r'HoogteKadijk':
    BZSV, stekker = [52,22], [24,24]
    groups = groups[[2,3,5,9,11,13,14,15,17,23,26,27,28,29,30,31,32,35,38,41,44,47,50,53,56,59,62,65,68,71,74,75,78,79,82,83,84,85,86,87,90,93,94,95,98,101,104,107]]
    power = power[:,[2,3,5,9,11,13,14,15,17,23,26,27,28,29,30,31,32,35,38,41,44,47,50,53,56,59,62,65,68,71,74,75,78,79,82,83,84,85,86,87,90,93,94,95,98,101,104,107]]
    
    rows_to_trim = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 31, 33, 34, 35, 36, 37, 40, 41, 42, 43, 44, 45, 47]
    weights = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

elif station == r'Uilenburg':
    BZSV, stekker = [22,22],[12,12]
    groups = groups[[0,1,4,5,6,9,12,15,18,19,20,23,26,29,30,31,34,37,40,41,43]]
    power = power[:,[0,1,4,5,6,9,12,15,18,19,20,23,26,29,30,31,34,37,40,41,43]]
    power[:,[0]] = power[:,[0]]+power[:,[2]] # V112 - V114
    power[:,[8]] = power[:,[8]]+power[:,[9]] # V124 - V126
    groups = np.delete(groups,[2,9])
    power = np.delete(power,[2,9],1)
    
    rows_to_trim = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    weights = [2,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1]
#elif station == r'Zevenhuizen':
#    BZSV, stekker = [60,60],[17,20]
#elif station == r'Drachten':
#    BZSV, stekker = [46,23.1],[21,19]
elif station == r'Emmeloord':
    BZSV, stekker = [44,44], [18,22]
    groups = groups[[0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,46,47,48,49,50,51,52,53,54,55,56,57]]
    power = power[:,[0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,46,47,48,49,50,51,52,53,54,55,56,57]]
    power[:,[0]] = power[:,[0]]+power[:,[6]]+power[:,[7]] # 43 - 50 - 52
    power[:,[1]] = power[:,[1]]+power[:,[2]]+power[:,[8]] # 44 - 45 - 54
    power[:,[3]] = power[:,[3]]+power[:,[9]] # 46 - 55
    power[:,[5]] = power[:,[5]]+power[:,[10]]+power[:,[11]] # 48 - 56 - 57
    power[:,[4]] = power[:,[4]]+power[:,[12]] # 47 - 58
    power[:,[15]] = power[:,[15]]+power[:,[16]] # 32 - 34
    power[:,[17]] = power[:,[17]]+power[:,[18]] # 33 - 35
    power[:,[19]] = power[:,[19]]+power[:,[22]] # 02 - 06
    power[:,[21]] = power[:,[21]]+power[:,[25]] # 13 - 09
    groups = np.delete(groups,[2,6,7,8,9,10,11,12,16,18,22,25])
    power = np.delete(power,[2,6,7,8,9,10,11,12,16,18,22,25],1)
    
    rows_to_trim = [2, 3, 5, 7, 8, 10, 11, 12, 13, 14, 15]
    weights = [3,3,2,2,3,1,1,2,2,2,1,2,1,1,1,1]

elif station == r'Rijksuniversiteit':
    BZSV, stekker = [22,44], [12,18]
    groups = groups[[0,1,2,3,4,5,6,7,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,72,75,78,81,84,87]]
    power = power[:,[0,1,2,3,4,5,6,7,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,72,75,78,81,84,87]]
    power[:,[6]] = power[:,[6]]+power[:,[7]] # V.103 - V.104
    power[:,[8]] = power[:,[8]]+power[:,[14]]+power[:,[16]] # 010V250 - 010V254 - 010V256
    power[:,[15]] = power[:,[15]]+power[:,[17]] # 010V255 - 010V259
    power[:,[18]] = power[:,[18]]+power[:,[21]] # 010V260 - 010V267
    power[:,[19]] = power[:,[19]]+power[:,[23]] # 010V263 - 010V270
    power[:,[12]] = power[:,[12]]+power[:,[13]] # 010V302 - 010V303, weight 1
    power[:,[24]] = power[:,[24]]+power[:,[25]] # 010V402 - 010V403, weight 1
    power[:,[26]] = power[:,[26]]+power[:,[27]] # 010V502 - 010V503, weight 1
    power[:,[28]] = power[:,[28]]+power[:,[29]] # 010V602 - 010V603, weight 1
    groups = np.delete(groups,[7,13,14,16,17,21,23,25,27,29])
    power = np.delete(power,[7,13,14,16,17,21,23,25,27,29],1)
    
    rows_to_trim = [0, 1, 2, 3, 4, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    weights = [1,1,1,1,1,1,2,3,1,1,1,1,2,2,2,1,1,1,1,1]


time = dpa.date_linspace(datetime.datetime(2016,1,1,0,0),datetime.datetime(2019,1,1,0,0),datetime.timedelta(minutes=5))
#power,time = dpa.grab_year(power,time,2016)
power = np.transpose(np.array(power))
dpr.auto_trim(power, rows_to_trim, 5)

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


