import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime
import functools

# Enter directory of you files here, so that python knows where to look
file_dir = r'C:\users\dionl\Documenten\Modellenpracticum'
sys.path.append(file_dir)

import DataProcessing as dpr
import DataParsing as dpa
import GenerateData as gd
import MonteCarloTwist as MCT
import LimitedBruteForce as LBF
import AuxiliaryAndPlottingFunctions as Aux



# Definition of several norms.

# Maximum of the absolute values
def MaxNorm(array):
    # Make sure that Norm greater or equal to 0
    return np.amax(np.abs(array))

# Average of the highest k absolute values
def Highest_k_Norm_sum(k, array):
    # k is assumed to be smaller than len(array)
    return sum(np.sort(np.abs(array))[-k :]) / k

# Sum of the p'th powers of the highest k absolute values
def Highest_k_pow_Norm(p, k, array):
    return sum(np.power(np.sort(np.abs(array))[-k :], p))

# Maximum of the values, not the 
def Norm1(array):
    return np.amax(array)

# Return value of element of highest absolute value. This norm can be negative
def Norm3(array):
    aamax = np.amax(array)
    aamin = np.amin(array)
    if abs(aamax) > abs(aamin):
        return aamax
    else:
        return aamin

# The norms in the code can only have on argument, the array on which they 
# must be evaluated. In order to convert a norm with extra parameters into one
# with only one parameter, use functools.partial(). For example, one uses the
# Highest_k_Norm_sum by not passing it directly as an argument to the algorithm,
# but as:   functools.partial(Highest_k_Norm_sum, 100), in this case k = 100.
       
# Plots a specific solution. For each transformator a separate plot is shown.
def plot_solution(power, sol):
    for i in range(len(sol)):
        tot = 0
        for j in sol[i]:
            tot += power[j]
        plt.plot(tot)
        plt.show()

# If the print_flag is True, this prints the norms for every transformator in
# the solution. If the print_flag is False, this returns the highest of those
# norms.
def sol_norms(power, sol, Norm, capacities, print_flag = True):
    maximum = 0
    for i in range(len(sol)):
        group = sol[i]
        tot = np.zeros(len(power[0]))
        for val in group:
            tot += power[val]
        norm = Norm(tot) / capacities[i]
        if print_flag:
            print(norm)
        if norm > maximum:
            maximum = norm
    if not print_flag:
        return maximum


# Here starts the main code. Change the base_path to the folder where the data
# has been saved, and change station to the station that has to be analysed.
Ubase = 10500 # (V)
station = r'Uilenburg'

base_path = r'C:\Users\dionl\Documents\Modellenpracticum\Power npy\metingen_'
power_path = base_path + station + r'.npy'
groups_path = base_path + station + r'_groups.npy'

# Load the data
groups = np.load(groups_path)
power = np.load(power_path)

# The following if-elif statement selects the specified station and combines 
# the power for stekkers that have to stay together, gives the correct weights
# and specifies the initial distribution. The rows that have to be trimmed are
# also given here.
if station == r'Bemmel':
    BZSV, stekker = [29.4,14.7], [15,5]
    groups = groups[[0,1,2,3,4,5,6,7,8,9,10,11,12,23,24,25,26,27]]
    power = power[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,23,24,25,26,27]]
    initial = [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],[15,16,17]]
    reserves = []
    
    rows_to_trim = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16]
    weights = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

elif station == r'HoogteKadijk':
    BZSV, stekker = [52,22], [24,24]
    groups = groups[[2,3,5,9,11,13,14,15,17,23,26,27,28,29,30,31,32,35,38,41,44,47,50,53,56,59,62,65,68,71,74,75,78,79,82,83,84,85,86,87,90,93,94,95,98,101,104,107]]
    power = power[:,[2,3,5,9,11,13,14,15,17,23,26,27,28,29,30,31,32,35,38,41,44,47,50,53,56,59,62,65,68,71,74,75,78,79,82,83,84,85,86,87,90,93,94,95,98,101,104,107]]
    initial = [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],[24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47]]
    reserves = []
    
    rows_to_trim = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 31, 33, 34, 35, 36, 37, 40, 41, 42, 43, 44, 45, 47]
    weights = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

elif station == r'Uilenburg':
    BZSV, stekker = [22,22],[12,12]
    groups = groups[[0,1,4,5,6,9,12,15,18,19,20,23,26,29,30,31,34,37,40,41,43]]
    power = power[:,[0,1,4,5,6,9,12,15,18,19,20,23,26,29,30,31,34,37,40,41,43]]
    initial = [[0,1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20]]
    
    power[:,[0]] = power[:,[0]]+power[:,[2]] # V112 - V114
    power[:,[8]] = power[:,[8]]+power[:,[9]] # V124 - V126
    groups = np.delete(groups,[2,9])
    power = np.delete(power,[2,9],1)

    initial = [[0,1,2,3,4,5,6,7,8], [9,10,11,12,13,14,15,16,17,18]]
    reserves = []
    
    rows_to_trim = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    weights = [2,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1]

elif station == r'Emmeloord':
    BZSV, stekker = [44,44], [18,22]
    groups = groups[[0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,46,47,48,49,50,51,52,53,54,55,56,57]]
    power = power[:,[0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,46,47,48,49,50,51,52,53,54,55,56,57]]
    initial = [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],[15,16,17,18,19,20,21,22,23,24,25,26,27]]
    
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
    initial = [[0,1,2,3,4,5,6], [7,8,9,10,11,12,13,14,15]]
    reserves = [5, 7, 8, 10, 11, 12, 13, 14, 15]
    
    rows_to_trim = [2, 3, 5, 7, 8, 10, 11, 12, 13, 14, 15]
    weights = [3,3,2,2,3,1,1,2,2,2,1,2,1,1,1,1]

elif station == r'Rijksuniversiteit':
    BZSV, stekker = [22,44], [12,18]
    groups = groups[[0,1,2,3,4,5,6,7,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,72,75,78,81,84,87]]
    power = power[:,[0,1,2,3,4,5,6,7,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,72,75,78,81,84,87]]
    initial = [[0,1,2,3,4,5,6,7],[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]]
    
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
    initial = [[0,1,2,3,4,5,6],[8,9,10,11,12,15,18,19,20,22,24,26,28]]
    initial = [[0,1,2,3,4,5,6], [7, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
    reserves = []
    
    rows_to_trim = [0, 1, 2, 3, 4, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    weights = [1,1,1,1,1,1,2,3,1,1,1,1,2,2,2,1,1,1,1,1]
    
elif station == r'Winselingseweg':
    BZSV, stekker = [60.5,29,31.5],[24,10,17]
    groups = groups[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91]]
    power = power[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91]]
    initial = [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],[21,22,23,24,25,26,27,28,29],[30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]]

    power[:,[0]] = power[:,[0]]+power[:,[1]]+power[:,[2]]+power[:,[3]]+power[:,[6]] # B-K55 - B-K56 - B-K57 - B-K58 - B-K61
    power[:,[14]] = power[:,[14]]+power[:,[17]] # B-K72 - B-K75
    power[:,[23]] = power[:,[23]]+power[:,[24]] # 2.85 - 2.86
    power[:,[25]] = power[:,[25]]+power[:,[27]]+power[:,[28]]+power[:,[29]] # 2.87 - 2.93 - 2.94 - 2.95
    power[:,[34]] = power[:,[34]]+power[:,[35]] # 2.05 - 2.06
    power[:,[32]] = power[:,[32]]+power[:,[40]] # 2.03 - 2.15
    power[:,[41]] = power[:,[41]]+power[:,[42]] # 2.16 - 2.17
    power[:,[44]] = power[:,[44]]+power[:,[45]] # 2.18 - 2.19
    groups = np.delete(groups,[1,2,3,6,17,24,27,28,29,35,40,42,45])
    power = np.delete(power,[1,2,3,6,17,24,27,28,29,35,40,42,45],1)
    initial = [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], [16,17,18,19,20], [21,22,23,24,25,26,27,28,29,30,31,32]]
    reserves = []
    
    rows_to_trim = [1, 3, 6, 7, 9, 10, 17, 22, 25, 29]
    weights = [5,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,2,4,1,1,1,2,1,2,1,1,1,1,2,1,2]

# Transpose the power array, now every power[i] is the dataset of one stekker
# or set of stekkers that has to stay together.
power = np.transpose(np.array(power))
time = dpa.date_linspace(datetime.datetime(2016,1,1,0,0),datetime.datetime(2019,1,1,0,0),datetime.timedelta(minutes=5))

# Trim the power array.
power = dpr.auto_trim(power, rows_to_trim, 5)

print("Trimmed")

# Not all rows have the same length after trimming, the following part ensures
# that they all have the same length again by forgetting the last part of the
# longer rows.
new_power = []
min_length = len(power[0])
for row in power:
    if len(row) < min_length:
        min_length = len(row)

for row in power:
    new_power.append(row[0:min_length])
power = new_power

# For Emmeloord, there are several rows in power which are reserves and do not 
# add anything to the distribution. They have to be removed.
new_initial = []
for group in initial:
    new_group = []
    for i in group:
        if i not in reserves:
            new_group.append(i)
    new_initial.append(new_group)
initial = new_initial

# Normalize the BZSV to show percentage of maximal capacity.
for i in range(len(BZSV)):
    BZSV[i] = BZSV[i] * 10**4 / (Ubase * np.sqrt(3))


def Bemmel2018():
    # This reads the dataset from Bemmel as given to us by Alliander. In order
    # to run this function, the station variable must have been set to Bemmel
    # in order to load the correct weights etc.
    # The data is split over two files, so it has to be combined after trimming.
    power1, time1, timeDiff1 = dpa.read_data(r'C:\Users\dionl\Documents\Modellenpracticum\Data\OS Bemmel data 2018 inst 1.csv')
    power2, time2, timeDiff2 = dpa.read_data(r'C:\Users\dionl\Documents\Modellenpracticum\Data\OS Bemmel data 2018 inst 2.csv')
    power1 = np.transpose(power1)
    power2 = np.transpose(power2)
    power = []
    rows_to_trim1 = [1, 3, 4, 5, 7, 9, 10, 14]
    rows_to_trim2 = [1]
    power1 = dpr.auto_trim(power1, rows_to_trim1, timeDiff1)
    power2 = dpr.auto_trim(power2, rows_to_trim2, timeDiff2)
    for row in power1:
        power.append(row)
    for row in power2:
        power.append(row)
    
    # Make sure all rows in power have the same length.
    new_power = []
    min_length = len(power[0])
    for row in power:
        if len(row) < min_length:
            min_length = len(row)
            
    for row in power:
        new_power.append(row[0:min_length])
    power = new_power
    
    # Plot the power usage of the initial distribution.
    tot1 = np.zeros(35040)
    tot2 = np.zeros(35040)
    for i in initial[0]:
        tot1 += power[i]
    for i in initial[1]:
        tot2 += power[i]
    tot1 = tot1 / BZSV[0]
    tot2 = tot2 / BZSV[1]
    plt.xlabel("Tijd")
    plt.ylabel("Percentage van maximumcapaciteit")
    plt.ylim(-10, 100)
    plt.plot(tot1)
    plt.show()
    plt.xlabel("Tijd")
    plt.ylabel("Percentage van maximumcapaciteit")
    plt.ylim(-10, 100)
    plt.plot(tot2)
    plt.show()
    return power
    

def limited_brute_force(power, initial, stekker, Norm, BZSV, weights, k):
    # Runs a limited brute force, plots the best solution and gives the norms
    # of the best solution.
    sol = LBF.k_step_brute_force(power, k, initial, stekker, Norm, BZSV, weights)
    plot_solution(power, sol)
    sol_norms(power, sol, MaxNorm, BZSV)

def MonteCarlo(power, initial, stekker, Norm, BZSV, weights, iterations, solutions_to_save, rejection_rate):
    # Standard Monte Carlo. Saves the best amount of solutions_to_save solutions,
    # and plots statistics about this. It plots the best solution, the norms
    # of the best solution, gives the distribution of the norms of the best 
    # solution and plots the distances among the solutions.
    solutions, norms = MCT.MonteCarlo(initial, power, stekker, weights, BZSV, iterations, rejection_rate, Norm, solutions_to_save, "twist")
    best_sol = solutions[0]
    plot_solution(power, best_sol)
    print("Norms of best distribution:")
    sol_norms(power, best_sol, Norm, BZSV)
    print("Norms of initial distribution:")
    sol_norms(power, initial, Norm, BZSV)
    Aux.DrawDistanceFunction(solutions, weights, initial)
    Aux.PlotAllSolutions(norms)


def Stability_and_quality(power, time, stekker, weights, BZSV, iterations, rejection_rate, Norm):
    # Generate solution for 2016 and plot its norms. Also find solutions for 2017
    # and 2018 and use this to find stability and qualities of solutions.
    power2016, time2016 = dpa.grab_year(np.transpose(power), time, 2016)
    power2016 = np.transpose(power2016)
    power2017 = np.transpose(dpa.grab_year(np.transpose(power), time, 2017)[0])
    power2018 = np.transpose(dpa.grab_year(np.transpose(power), time, 2018)[0])


    solutions2016, norms2016 = MCT.MonteCarlo(initial, power2016, stekker, weights, BZSV, iterations, rejection_rate, Norm, 20, "twist")
    sol_norms(power2016, solutions2016[0], Norm, BZSV)
    if False:
        for i in range(len(solutions2016)):
            sol = solutions[i]
            print("\nNorms of solution " + str(i) + " in year 2017")
            sol_norms(power2017, sol, Norm, BZSV)
            print("Year 2018")
            sol_norms(power2018, sol, Norm, BZSV)
    
    # Find best solutions for 2017 and 2018 using MonteCarlo
    solutions2017, norms2017 = MCT.MonteCarlo(initial, power2017, stekker, weights, BZSV, iterations, rejection_rate, Norm, 1, "twist")
    solutions2018, norms2018 = MCT.MonteCarlo(initial, power2018, stekker, weights, BZSV, iterations, rejection_rate, Norm, 1, "twist")

    qualities = []
    x = np.array([0, 1, 2])
    x_ticks = ["2016", "2017", "2018"]
    plt.ylim(-3,10)
    plt.xticks(x, x_ticks)
    
    # Calculate how bad solutions of 2016 are for 2017 and 2018. 
    for i in range(len(solutions2016)):
        sol = solutions2016[i]
        diff16 = sol_norms(power2016, sol, Norm, BZSV, False) - sol_norms(power2016, solutions2016[0], Norm, BZSV, False)
        diff17 = sol_norms(power2017, sol, Norm, BZSV, False) - sol_norms(power2017, solutions2017[0], Norm, BZSV, False)
        diff18 = sol_norms(power2018, sol, Norm, BZSV, False) - sol_norms(power2018, solutions2018[0], Norm, BZSV, False)
        plt.plot(x, [diff16, diff17, diff18])
        
        # Calculate quality of solution
        quality = np.sqrt(diff16**2 + diff17**2 + diff18**2)
        qualities.append(quality)
    
    # Plot the differences, this gives the stability plot.
    plt.xlabel("Jaar")
    plt.ylabel("Afstand tot beste oplossing")
    plt.show()
    
    # Find the best quality and its index.
    best_i = 0
    best_qual = qualities[0]
    for i in range(len(qualities)):
        qual = qualities[i]
        if qual < best_qual:
            best_qual = qual
            best_i = i
    qualities.sort()
    
    # Plot the sorted qualities.
    x_ticks = list(range(1,21))
    x = list(range(20))
    plt.xticks(x, x_ticks)
    plt.xlabel("Oplossingen")
    plt.ylabel("Kwaliteit")
    plt.plot(qualities, "ro")
    plt.show()
    
    # Print the solution with the lowest quality and print this solution.
    print(best_i, solutions2016[best_i], best_qual)
    
    return power2016, time2016, solutions2016, power2017, power2018


def RandomData(power2016, time2016, solutions2016, stekker, weights, BZSV, Norm, iterations, rejection_rate):
    # Generate two years of random data and compare to solution of 2016. Also
    # give the stability plot
    day_var = 30
    week_var = 15
    powerNoise = gd.generate_data(np.transpose(power2016),time2016,5, day_var, week_var)
    powerSmoothNoise = gd.smoothNoise(powerNoise,10, time2016) # rank is variable
    powerNoise2 = gd.generate_data(np.array(powerSmoothNoise), time2016, 5, day_var, week_var)
    powerSmoothNoise2 = gd.smoothNoise(powerNoise2, 10, time2016)
    
    powerSmoothNoise = np.transpose(powerSmoothNoise)
    powerSmoothNoise2 = np.transpose(powerSmoothNoise2)
    
    # Generate best solutions on the generated data.
    smoothsol1, norms1 = MCT.MonteCarlo(initial, powerSmoothNoise, stekker, weights, BZSV, iterations, rejection_rate, Norm, 1, "twist")
    smoothsol2, norms2 = MCT.MonteCarlo(initial, powerSmoothNoise2, stekker, weights, BZSV, iterations, rejection_rate, Norm, 1, "twist")
    
    # Make stability plot on the generated data.
    plt.ylim(-3,10)
    x = np.array([0, 1, 2])
    x_ticks = ["2016", "Random data 1", "Random data 2"]
    plt.xticks(x, x_ticks)
    
    for i in range(len(solutions2016)):
        sol = solutions2016[i]
        realdiff = sol_norms(power2016, sol, Norm, BZSV, False) - sol_norms(power2016, solutions2016[0], Norm, BZSV, False)
        randomdiff1 = sol_norms(powerSmoothNoise, sol, Norm, BZSV, False) - sol_norms(powerSmoothNoise, smoothsol1[0], Norm, BZSV, False)
        randomdiff2 = sol_norms(powerSmoothNoise2, sol, Norm, BZSV, False) - sol_norms(powerSmoothNoise2, smoothsol2[0], Norm, BZSV, False)
        plt.plot(x, [realdiff, randomdiff1, randomdiff2])
    plt.ylabel("Afstand tot beste oplossing")
    plt.show()


def less_data_points(power2016, solutions2016, initial, stekker, weights, iterations, BZSV, rejection_rate, Norm):    
    # Does it matter if we take a dataset with less points? The dataset gets
    # converted to one with data per 15 minutes instead of 5.
    reduced_power = dpr.to_quarter(power2016)
    
    # Optimize on the smaller dataset.
    quarter_sols, quarter_norms = MCT.MonteCarlo(initial, reduced_power, stekker, weights, BZSV, iterations, rejection_rate, Norm, 20, "twist")
    
    # Find out the differences between the two solution sets on the dataset 
    # with data for every 5 minutes.
    differences = []
    for i in range(len(quarter_sols)):
        differences.append(sol_norms(power2016, quarter_sols[i], Norm, BZSV, False) - sol_norms(power2016, solutions2016[i], Norm, BZSV, False))
    differences.sort()
    plt.ylabel("Verschil tussen data per 5 minuten\nen data per 15 minuten")
    x_ticks = list(range(1,21))
    x = list(range(20))
    plt.xticks(x, x_ticks)
    plt.plot(differences, "ro")
    plt.show()
