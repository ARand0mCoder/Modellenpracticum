import numpy as np
import matplotlib.pyplot as plt
import random
import datetime
import DataProcessing as dpr
import DataParsing as dpa
from calendar import isleap

def generate_data(power,time,timeDiff, day_variance, week_variance):
    """
    INPUTS: power, time for one year
    OUTPUTS: powerNew generated pseudorandomly from power
    Assumptions: weekdays and weekends the same dates as given data,
                 leap years generate leap years
    Generating new data:
    - Pseudorandom: Weekdays chosen from weekdays from last year from 2 weeks
            before to 2 weeks after same datetime, the same holds for weekends
    - Completely random: Days randomly chosen throughout the year
    - Normal: Nothing will be changed
    - Generate noise: Each day is lowered or increased based on a
        normal distribution as well as each 2 weeks with a larger deviation
        to account for large scale changes
    Most realistic for generating new data is pseudorandom + generate noise
    """
    totalTime = len(power)
    totalGroups = len(power[0])
    powerNew = np.zeros((totalTime,totalGroups))
    isLeapYear = isleap(time[0].year) # To account for a leap year
    for group in range(totalGroups):
        powerGroupNew = np.array([]) # powerNew for a given group
        for day in range(365 + int(isLeapYear)): # Generate new data per day
            ''' Pseudorandom '''
            currentDate = time[day*24*60//timeDiff]
            if currentDate.weekday() >= 0 and currentDate.weekday() <= 4: # Weekdays
                weekday = True
            else: # Weekends
                weekday = False
            randomDay = weekDayOrEnd(weekday,currentDate,timeDiff, time)
            powerDay = np.copy(power[randomDay*24*60//timeDiff:(randomDay+1)*24*60//timeDiff,group])

            ''' Completely random '''
            # randomDay = random.randint(0,364+int(isLeapYear))
            # powerDay = power[randomDay*24*60//timeDiff:(randomDay+1)*24*60//timeDiff,group]

            ''' Normal '''
            # powerDay = power[day*24*60//timeDiff:(day+1)*24*60//timeDiff,group]

            ''' Generate noise '''
            dayAverage = np.mean(powerDay)
            powerDay += random.normalvariate(dayAverage,dayAverage/day_variance)-dayAverage # Standard deviation is variable
            if day % 14 == 0: # Large scale period is variable
                weekRandom = random.normalvariate(dayAverage,dayAverage/week_variance)-dayAverage # Standard deviation is variable
            powerDay += weekRandom

            powerGroupNew = np.append(powerGroupNew,powerDay)
        powerNew[:,group] = powerGroupNew

    return powerNew

def weekDayOrEnd(weekday,currentDate,timeDiff, time):
    """ Returns a random day within 2 weeks of currentDate taking weekdays and
    weekends in consideration. The 14 day window is variable. """
    while True: # Trying to find a weekday within 2 weeks of currentDate
        lower = max(0,currentDate.timetuple().tm_yday-1-14)
        upper = min(364+int(isleap(currentDate.year)),currentDate.timetuple().tm_yday-1+14)
        randomDay = random.randint(lower,upper)
        newDate = time[randomDay*24*60//timeDiff]
        if (weekday and newDate.weekday() >= 0 and newDate.weekday() <= 4) or (not(weekday) and not(newDate.weekday() >= 0 and newDate.weekday() <= 4)):
            break
    return randomDay

def smoothNoise(power,rank,time):
    ''' Generates new dataset based on matrix approximations: Singular Value Decomposition (SVD)
    The higher the rank, the better the approximation of the data
    The lower the rank, the smoother the approximation of the data'''
    totalTime = len(power)
    totalGroups = len(power[0])
    isLeapYear = isleap(time[0].year)
    powerSmoothNoise = np.zeros((totalTime,totalGroups))
    for group in range(totalGroups):
        powerGroup = power[:,group]
        avg = np.mean(powerGroup)
        powerGroupMatrix = np.reshape(powerGroup,(365+int(isLeapYear),24*12))

        ''' Using full SVD by numpy '''
        u,s,v = np.linalg.svd(powerGroupMatrix,full_matrices=False) # Singular Value Decomposition

        # Low rank approximation
        smoothMatrix = np.zeros((len(u), len(v)))
        for i in range(rank):
            smoothMatrix += s[i] * np.outer(u.T[i], v[i])

        powerGroupMatrix, smoothMatrix = powerGroupMatrix.T, smoothMatrix.T

        ''' Image plot of original data and approximated data '''
        # plt.subplot(2,1,1)
        # plt.imshow(powerGroupMatrix)
        # plt.colorbar()
        # plt.subplot(2,1,2)
        # plt.imshow(smoothMatrix)
        # plt.colorbar()
        # plt.show()

        ''' Turn matrix to linear data array '''
        powerSmooth = np.array(np.reshape(smoothMatrix.T,(365+int(isLeapYear))*24*12))
        powerSmoothNoise[:,group] = powerSmooth

        ''' Graph plot of original data and approximated data '''
        # time = dpa.date_linspace(datetime.datetime(2016,1,1,0,0),datetime.datetime(2019,1,1,0,0),datetime.timedelta(minutes=5))
        # plt.plot(time,powerGroup)
        # plt.plot(time,powerSmooth)
        # plt.show()

    return powerSmoothNoise


''' Usage of functions '''
# time = dpa.date_linspace(datetime.datetime(2016,1,1,0,0),datetime.datetime(2019,1,1,0,0),datetime.timedelta(minutes=5))
# power = np.load(r'C:\Users\XpsBook\Documents\Radboud Universiteit Nijmegen\Jaar 3\Modellenpracticum\Power npy\metingen_Bemmel.npy')
# power = power[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,23,24,25,26,27]]
# power,time = dpa.grab_year(power,time,2016)
#
# powerNoise = generate_data(np.array(power),time,timeDiff=5)
# powerSmoothNoise = smoothNoise(powerNoise,rank=10) # rank is variable
# 
# # Plotting
# for group in range(len(power[0])): # Original data
#     # plt.figure()
#     data = [line[group] for line in power]
#     plt.plot(time,data)
# plt.figure()
# for group in range(len(power[0])): # Data with noise
#     # plt.figure()
#     data = [line[group] for line in powerNoise]
#     plt.plot(time,data)
# plt.figure()
# for group in range(len(power[0])): # Data with smooth noise
#     # plt.figure()
#     data = [line[group] for line in powerSmoothNoise]
#     plt.plot(time,data)
# plt.show()
