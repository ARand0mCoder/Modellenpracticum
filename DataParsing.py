import numpy as np
import matplotlib.pyplot as plt
import csv
import datetime
from dateutil import parser
import DataProcessing as dp

'''
--------- CONTENT ---------
This file contains a couple of functions regarding parsing of the data as well as
all manual datahandling lines of consideration.
Functions:
    read_data and read_data2 both are parsing functions
        read_data is for csv files with several columns per rows
        read_data2 is for csv files with 2 columns (time and current) with all
            groups under each other
    date_linspace creates a linearly spaced vector of datetime elements
    grab_year takes a specific year of the full data
Manual data handling:
    After the functions are several lines of code for some of the stations
    The data from the csv file is stored as a numpy variable (.npy), after which
    the current has to be extracted for each group (so not data concerning P, Q, IL and IR)
    Secondly, groups that should remain together in a transformer are added
    and a weighting is added to those for the amount of spaces it occupies
----------------------------
'''

def read_data(path):
    time = []               # Time and day
    power = []              # Data of power from excel
    isDayFirst = False      # Used for date parsing
    checkDate = True        # When to stop checking date parsing
    previousTime = datetime.datetime(1,1,1,1,1)       # For checking double time values
    timeDiff = 1            # Difference in time between data points

    with open(path) as csv_file:
        csvReader = csv.reader(csv_file,delimiter=';') # Read excel file as .csv
        lineCount = 0
        for row in csvReader:
            if lineCount == 0:
                columns = row[2:]   # Names of the columns of power
                print(f'Columns: {", ".join(columns)}')
                print('\nProcessing...')
            else:
                # Checking if date is given by month-day or day-month
                if checkDate and row[0][0:3] == '1-2':
                    checkDate = False
                elif checkDate and row[0][0:3] == '2-1':
                    checkDate = False
                    isDayFirst = True

                # Stores power data
                tempLine = []
                for elem in row[2:]:
                    tempLine.append(float(elem))

                currentTime = parser.parse(row[0]+' '+row[1], dayfirst=isDayFirst)
                # Determining the time difference between data points
                if lineCount == 2:
                    timeDiff = int(currentTime.minute-previousTime.minute)

                # Checking whether time is the same or time is skipped (fill in with average)
                if currentTime == previousTime and lineCount != 1: # Same time occured one before
                    power[len(power)-1] = np.divide([power[len(power)-1][i]+tempLine[i] for i in range(len(tempLine))],2)
                elif currentTime != previousTime + datetime.timedelta(minutes=timeDiff) and currentTime != previousTime and lineCount != 1: # Time is skipped
                    previousTime += datetime.timedelta(minutes=timeDiff)
                    while currentTime != previousTime:
                        power.append(np.divide([power[len(power)-1][i]+tempLine[i] for i in range(len(tempLine))],2))
                        time.append(previousTime)
                        previousTime += datetime.timedelta(minutes=timeDiff)
                    power.append(tempLine)
                    time.append(currentTime)
                else:   # Normal time difference
                    power.append(tempLine)
                    time.append(currentTime)
                previousTime = currentTime

            if lineCount % 10000 == 0:  # Prints progress
                print(lineCount)
            lineCount += 1
        print(f'Processed {lineCount} lines.')

    # Delete single value for next year if it's present
    if time[len(time)-1].day == 1 and time[len(time)-1].month == 1 and time[len(time)-1].hour == 0 and time[len(time)-1].minute == 0:
        power = power[:-1]
        time = time[:-1]

    return power,time,timeDiff

def read_data2(path):
    time = []               # Time and day
    power = []              # Data of power from excel
    groups = []             # Names of the groups
    tempLine = []           # Stores data of one group at a time
    isDayFirst = False      # Used for date parsing
    checkDate = True        # When to stop checking date parsing
    buildTime = True        # Creates a time array during the first group
    previousTime = datetime.datetime(1,1,1,1,1)       # For checking double time values
    timeDiff = 1            # Difference in time between data points
    lineCount = 0           # Number of lines
    groupLineCount = 0      # Number of lines per group

    with open(path) as csv_file:
        csvReader = csv.reader(csv_file,delimiter=',') # Read excel file as .csv

        print('Processing...')
        for row in csvReader:
            if lineCount != 0:
                # # Checking if date is given by month-day or day-month (not needed for our given data)
                # if checkDate and row[1][5:10] == '01-02':
                #     checkDate = False
                # elif checkDate and row[1][5:10] == '02-01':
                #     checkDate = False
                #     isDayFirst = True

                if lineCount == 1: # On the first line
                    currentGroup = row[0]
                    groups.append(row[0])
                    group = 0

                if row[0] != currentGroup: # If a new group is started
                    currentGroup = row[0]
                    groups.append(row[0])
                    power.append(tempLine)
                    tempLine = []
                    previousTime = datetime.datetime(1,1,1,1,1)
                    buildTime = False
                    group += 1
                    groupLineCount = 1

                # Stores power data
                currentPower = float(row[2])

                currentTime = parser.parse(row[1], dayfirst=isDayFirst)
                # Determining the time difference between data points
                if lineCount == 2:
                    timeDiff = int(currentTime.minute-previousTime.minute)

                # Checking whether time is the same or time is skipped (fill in with average)
                if currentTime == previousTime and lineCount != 1: # Same time occured one before
                    tempLine[len(tempLine)-1] = (tempLine[len(tempLine)-1]+currentPower)/2
                elif currentTime != previousTime + datetime.timedelta(minutes=timeDiff) and currentTime != previousTime and groupLineCount != 1: # Time is skipped
                    previousTime += datetime.timedelta(minutes=timeDiff)
                    while currentTime != previousTime:
                        tempLine.append((tempLine[len(tempLine)-1]+currentPower)/2)
                        if buildTime:
                            time.append(previousTime)
                        previousTime += datetime.timedelta(minutes=timeDiff)
                    tempLine.append(currentPower)
                    if buildTime:
                        time.append(currentTime)
                else:   # Normal time difference
                    tempLine.append(currentPower)
                    if buildTime:
                        time.append(currentTime)
                previousTime = currentTime

            if lineCount % 1000000 == 0:  # Prints progress
                print(lineCount)
            lineCount += 1
            groupLineCount += 1
        power.append(tempLine)

        print(f'Processed {lineCount} lines.')

    power = np.transpose(np.array(power))
    # Delete single value for next year if it's present
    if time[len(time)-1].day == 1 and time[len(time)-1].month == 1 and time[len(time)-1].hour == 0 and time[len(time)-1].minute == 0:
        power = power[:-1]
        time = time[:-1]

    print(f'Groups: {", ".join(groups)}')
    print(f'Amount of groups: {len(power[0])}')
    print(f'Timesteps: {len(power)}')

    return power,time,timeDiff,groups

def date_linspace(start, end, delta):
    ''' Generates an array of linearly spaced datetime objects '''
    steps = int((end-start)/delta)
    increments = range(0, steps) * np.array([delta]*steps)
    return start + increments

def grab_year(power,time,year):
    ''' Returns only power from a specific year '''
    yearPower = []
    yearTime = []
    for i in range(len(time)):
        if time[i].year == year:
            yearPower.append(power[i])
            yearTime.append(time[i])
    return yearPower,yearTime

''' Generating .npy file with parsed data '''
# power,time,timeDiff,groups = read_data2(r'C:\Users\XpsBook\Documents\Radboud Universiteit Nijmegen\Jaar 3\Modellenpracticum\Bestanden Alliander\Power Data\test.csv')
# np.save(r'C:\Users\XpsBook\Documents\Radboud Universiteit Nijmegen\Jaar 3\Modellenpracticum\Power npy\power_test',power)
# np.save(r'C:\Users\XpsBook\Documents\Radboud Universiteit Nijmegen\Jaar 3\Modellenpracticum\Power npy\power_test_groups',groups)

''' Usage of functions '''

# groups = np.load(r'C:\Users\XpsBook\Documents\Radboud Universiteit Nijmegen\Jaar 3\Modellenpracticum\Power npy\metingen_Emmeloord_groups.npy')
# power = np.load(r'C:\Users\XpsBook\Documents\Radboud Universiteit Nijmegen\Jaar 3\Modellenpracticum\Power npy\metingen_Emmeloord.npy')

''' Example: Emmeloord (rest can be found in main.py) '''
# groups = groups[[0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,46,47,48,49,50,51,52,53,54,55,56,57]]
# power = power[:,[0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,46,47,48,49,50,51,52,53,54,55,56,57]]
#
# power[:,[0]] = power[:,[0]]+power[:,[6]]+power[:,[7]] # 43 - 50 - 52
# power[:,[1]] = power[:,[1]]+power[:,[2]]+power[:,[8]] # 44 - 45 - 54
# power[:,[3]] = power[:,[3]]+power[:,[9]] # 46 - 55
# power[:,[5]] = power[:,[5]]+power[:,[10]]+power[:,[11]] # 48 - 56 - 57
# power[:,[4]] = power[:,[4]]+power[:,[12]] # 47 - 58
# power[:,[15]] = power[:,[15]]+power[:,[16]] # 32 - 34
# power[:,[17]] = power[:,[17]]+power[:,[18]] # 33 - 35
# power[:,[19]] = power[:,[19]]+power[:,[22]] # 02 - 06
# power[:,[21]] = power[:,[21]]+power[:,[25]] # 13 - 09
# groups = np.delete(groups,[2,6,7,8,9,10,11,12,16,18,22,25])
# power = np.delete(power,[2,6,7,8,9,10,11,12,16,18,22,25],1)
#
# weights = [3,3,2,2,3,1,1,2,2,2,1,2,1,1,1,1]

# print(groups)
# print(power)

# time = date_linspace(datetime.datetime(2016,1,1,0,0),datetime.datetime(2019,1,1,0,0),datetime.timedelta(minutes=5))
# # power,time = grab_year(power,time,2016)
# dp.plot_data(power,time)
