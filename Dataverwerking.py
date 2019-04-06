import numpy as np
import matplotlib.pyplot as plt
import csv
import datetime
from dateutil import parser

#%%

def read_data(path):
    time = []               # Time and day
    power = []              # Data of power from excel
    isDayFirst = False      # Used for date parsing
    checkDate = True        # When to stop checking date parsing
    previousTime = datetime.datetime(1,1,1,1,1)       # For checking double time values
    timeDiff = 1            # Difference in time between data points

    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=';') # Read excel file as .csv
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
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
                temp_line = []
                for elem in row[2:]:
                    temp_line.append(float(elem))

                currentTime = parser.parse(row[0]+' '+row[1], dayfirst=isDayFirst)
                # Determining the time difference between data points
                if line_count == 2:
                    timeDiff = int(currentTime.minute-previousTime.minute)

                # Checking whether time is the same or time is skipped (fill in with average)
                if currentTime == previousTime and line_count != 1: # Same time occured one before
                    power[len(power)-1] = np.divide([power[len(power)-1][i]+temp_line[i] for i in range(len(temp_line))],2)
                elif currentTime != previousTime + datetime.timedelta(minutes=timeDiff) and currentTime != previousTime and line_count != 1: # Time is skipped
                    previousTime += datetime.timedelta(minutes=timeDiff)
                    while currentTime != previousTime:
                        power.append(np.divide([power[len(power)-1][i]+temp_line[i] for i in range(len(temp_line))],2))
                        time.append(previousTime)
                        previousTime += datetime.timedelta(minutes=timeDiff)
                    power.append(temp_line)
                    time.append(currentTime)
                else:   # Normal time difference
                    power.append(temp_line)
                    time.append(currentTime)
                previousTime = currentTime

            if line_count % 10000 == 0:  # Prints progress
                print(line_count)
            line_count += 1
        print(f'Processed {line_count} lines.')

    # Delete single value for next year if it's present
    if time[len(time)-1].day == 1 and time[len(time)-1].month == 1 and time[len(time)-1].hour == 0 and time[len(time)-1].minute == 0:
        power = power[:-1]
        time = time[:-1]

    return power,time,timeDiff

#%%

''' All data throughout the year '''
def plot_data(power, time):
    for group in range(len(power[0])):
        # plt.figure()
        data = [line[group] for line in power]
        plt.plot(time,data)
    plt.show()

#%%

''' One specific time throughout the year '''
def plot_specific_time(power, time, hour, minute):
    data = []
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


#%%

# This function is used for manually trimming the data and gives the user the
# ability to agree on the trimmed data.
def manual_trim(power, timeDiff):
    rows_to_trim = []
    current_row = 0

    # For each dataset of one group we check if there are any outliers
    for row in power:
        row_length = len(row)
        maximums = []
        # We check for each day the maximum absolute capacity and store those
        i = 0
        while 24*60//timeDiff * (i + 1) < row_length:
            maximum = 0
            for j in range(24*60//timeDiff):
                if abs(row[24*60//timeDiff * i + j]) > maximum:
                    maximum = abs(row[24*60//timeDiff * i + j])
            maximums.append(maximum)
            i += 1
        plt.plot(list(range(len(maximums))), maximums)
        plt.show()

        avg = sum(maximums) / len(maximums)
        print(avg)

        suggested_row = np.array([])
        suggested_days = []
        i = 0
        while 24*60//timeDiff * (i + 1) < row_length:
            if maximums[i] <= 1.5 * avg:
                suggested_row = np.append(suggested_row, row[24*60//timeDiff * i : 24*60//timeDiff * (i + 1)])
                suggested_days.append(i)
            else:
                NaNs = np.zeros(24*60//timeDiff)
                suggested_row = np.append(suggested_row, NaNs)
            i += 1

        plt.plot(suggested_row)
        plt.show()
        plt.plot(row)
        plt.show()
        trim = input("Do you want to trim this? Enter 'y' or 'n' for yes or no.")
        if trim == 'y':
            rows_to_trim.append(current_row)
        current_row += 1

    print(rows_to_trim)

# This reads a txt file with the data on which rows to trim and returns the
# trimmed data
def auto_trim(power, rows_to_trim, timeDiff):
    suggested_power = []
    current_row = 0

    # For each dataset of one group we check if there are any outliers
    for row in power:
        if current_row in rows_to_trim:
            row_length = len(row)
            maximums = []
            # We check for each day the maximum absolute capacity and store those
            i = 0
            while 24*60//timeDiff * (i + 1) < row_length:
                maximum = 0
                for j in range(24*60//timeDiff):
                    if abs(row[24*60//timeDiff * i + j]) > maximum:
                        maximum = abs(row[24*60//timeDiff * i + j])
                maximums.append(maximum)
                i += 1

            avg = sum(maximums) / len(maximums)

            suggested_row = np.array([])
            suggested_days = []
            i = 0
            while 24*60//timeDiff * (i + 1) < row_length:
                if maximums[i] <= 1.5 * avg:
                    suggested_row = np.append(suggested_row, row[24*60//timeDiff * i : 24*60//timeDiff * (i + 1)])
                    suggested_days.append(i)
                else:
                    NaNs = np.zeros(24*60//timeDiff)
                    suggested_row = np.append(suggested_row, NaNs)
                i += 1
            suggested_power.append(suggested_row)
        else:
            suggested_power.append(row)

        current_row += 1

    return suggested_power
