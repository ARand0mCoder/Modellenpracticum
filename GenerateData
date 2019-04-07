import numpy as np
import matplotlib.pyplot as plt
import random
import Dataverwerking as Dv

def generate_data(power,time,time_factor):
    """
     INPUTS: power, time for last year
     OUTPUTS: powerNew generated pseudorandomly from power
     Assumptions: no leap year, weekdays and weekends the same as last year
     Generating new data:
      - Weekdays chosen from weekdays from last year from 2 weeks before to 2 weeks after same datetime
      - Same holds for weekends
      - Power each data is lowered or increased based on a normal distribution with mean
    """
    totalTime = len(power)
    totalGroups = len(power[0])
    powerNew = np.zeros((totalTime,totalGroups))
    for group in range(totalGroups):
        powerGroupNew = np.array([])
        for day in range(365): # Does not account for leap years
            ''' Pseudorandom '''
            currentDate = time[day*24*60//timeDiff]
            if currentDate.weekday() >= 0 and currentDate.weekday() <= 4: #weekdays
                weekday = True
                randomDay = weekDayOrEnd(weekday,currentDate,timeDiff)
            else: #weekends
                weekday = False
                randomDay = weekDayOrEnd(weekday,currentDate,timeDiff)
            powerDay = power[randomDay*24*60//timeDiff:(randomDay+1)*24*60//timeDiff,group]

            ''' Completely random '''
            # randomDay = random.randint(0,364)
            # powerDay = power[randomDay*24*60//timeDiff:(randomDay+1)*24*60//timeDiff,group]

            ''' Normal '''
            # powerDay = power[day*24*60//timeDiff:(day+1)*24*60//timeDiff,group]

            ''' Decrease/increase with normal distribution '''
            powerDay += random.normalvariate(np.mean(powerDay),np.mean(powerDay)/15)-np.mean(powerDay)

            powerGroupNew = np.append(powerGroupNew,powerDay)
        powerNew[:,group] = powerGroupNew

    return powerNew

def weekDayOrEnd(weekday,currentDate,timeDiff):
    """ Calculates a random day within 2 weeks of currentDate taking weekdays and weekends in consideration """
    while True: # Trying to find a weekday within 2 weeks of currentDate
        lower = max(0,currentDate.timetuple().tm_yday-1-14)
        upper = min(364,currentDate.timetuple().tm_yday-1+14)
        randomDay = random.randint(lower,upper)
        newDate = time[randomDay*24*60//timeDiff]
        if (weekday and newDate.weekday() >= 0 and newDate.weekday() <= 4) or (not(weekday) and not(newDate.weekday() >= 0 and newDate.weekday() <= 4)):
            break
    return randomDay

path = r'C:\Users\XpsBook\Documents\Radboud Universiteit Nijmegen\Jaar 3\Modellenpracticum\Bestanden Alliander\Power Data\Uilenburg_2_2018_clean.csv'
# rows_to_trim1 = [0, 1, 2, 3, 4, 5, 8, 9]

power,time,timeDiff = Dv.read_data(path)
# print(len(power))
# Dv.plot_data(power,time)

# powerTranspose = np.transpose(np.array(power))
# power = Dv.auto_trim(powerTranspose, rows_to_trim1, timeDiff)
# print(len(power[0]))
# Dv.plot_data(power,time)

powerNew = generate_data(np.array(power),time,timeDiff)
Dv.plot_data(powerNew,time)
