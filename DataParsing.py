dpimport numpy as np
import matplotlib.pyplot as plt
import csv
import datetime
from dateutil import parser
import DataProcessing as dp

#%%

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
    tempLine = []
    isDayFirst = False      # Used for date parsing
    checkDate = True        # When to stop checking date parsing
    buildTime = True
    previousTime = datetime.datetime(1,1,1,1,1)       # For checking double time values
    timeDiff = 1            # Difference in time between data points
    lineCount = 0           # Number of lines
    groupLineCount = 0

    with open(path) as csv_file:
        csvReader = csv.reader(csv_file,delimiter=',') # Read excel file as .csv

        print('Processing...')
        for row in csvReader:
            if lineCount != 0:
                # # Checking if date is given by month-day or day-month
                # if checkDate and row[1][5:10] == '01-02':
                #     checkDate = False
                # elif checkDate and row[1][5:10] == '02-01':
                #     checkDate = False
                #     isDayFirst = True

                if lineCount == 1:
                    currentGroup = row[0]
                    groups.append(row[0])
                    group = 0

                if row[0] != currentGroup:
                    currentGroup = row[0]
                    groups.append(row[0])
                    power.append(tempLine)
                    tempLine = []
                    previousTime = datetime.datetime(1,1,1,1,1)
                    buildTime = False
                    group += 1
                    groupLineCount = 1

                # Stores power data
                # currentTime = parser.parse(row[1])
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
                        # print(len(tempLine)-1)
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

''' Generating .npy file with parsed data (Drachten & Zevenhuizen do not work for some reason)'''
# power,time,timeDiff,groups = read_data2(r'C:\Users\XpsBook\Documents\Radboud Universiteit Nijmegen\Jaar 3\Modellenpracticum\Bestanden Alliander\Power Data\test.csv')
# np.save(r'C:\Users\XpsBook\Documents\Radboud Universiteit Nijmegen\Jaar 3\Modellenpracticum\Power npy\power_test',power)
# np.save(r'C:\Users\XpsBook\Documents\Radboud Universiteit Nijmegen\Jaar 3\Modellenpracticum\Power npy\power_test_groups',groups)

''' Usage of functions '''
# time = date_linspace(datetime.datetime(2016,1,1,0,0),datetime.datetime(2019,1,1,0,0),datetime.timedelta(minutes=5))
# power = np.load(r'C:\Users\XpsBook\Documents\Radboud Universiteit Nijmegen\Jaar 3\Modellenpracticum\Power npy\metingen_Winselingseweg.npy')
# power,time = grab_year(power,time,2016)
# dp.plot_data(power,time)
