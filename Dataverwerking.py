import numpy as np
import matplotlib.pyplot as plt
import csv
import datetime
from dateutil import parser

#%%

path = r'C:\Users\dion\Documents\Studie\Modellenpracticum\Uilenburg_1_2018_clean.csv'
time = []               # Time and day
power = []              # Data of power from excel
isDayFirst = False      # Used for date parsing
checkDate = True        # When to stop checking date parsing
previousTime = ''       # For checking double time values

with open(path) as csv_file:
    csv_reader = csv.reader(csv_file,delimiter=';') # Read excel file as .csv
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            columns = row[2:]   # Names of the columns of power
            print(f'Kolommen zijn: {", ".join(columns)}')
            print('\nAan het inlezen...')
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
            if row[1] == previousTime: # Same time occured one before
                power[len(power)-1] = np.divide(power[len(power)-1] + temp_line,2)
            else:
                power.append(temp_line)
                time.append(parser.parse(row[0]+' '+row[1], dayfirst=isDayFirst))
            previousTime = row[1]
        if line_count % 10000 == 0:  # Prints progress
            print(line_count)
        line_count += 1
    print(f'Well, that was easy. Totaal {line_count} regels ingelezen.')

#%%
                  
''' All data throughout the year '''
def plot_data():
    for i in range(len(row)-2):
        # plt.figure()
        data = [line[i] for line in power]
        plt.plot(time,data)
    plt.show()

#%%                  
                  
''' One specific time throughout the year '''
def plot_specific_time():
    data = []
    timeData = datetime.time(12,0)

    for j in range(len(row)-2):
        Xdata = []
        Ydata = []
        for i in range(len(power)):
            if time[i].time() == timeData:
                Xdata.append(time[i])
                Ydata.append(power[i][j])
        plt.plot(Xdata,Ydata)
    plt.show()

#%%

power = np.array(power)
power = np.transpose(power)

#%%

# This function is used for manually trimming the data and gives the user the
# ability to agree on the trimmed data. 
def manual_trim(power):
    # time_factor must be 1 if the data is per 15 mins, 3 if it is per 5 mins
    time_factor = 1
    rows_to_trim = []
    current_row = 0
    
    # For each dataset of one group we check if there are any outliers
    for row in power:
        row_length = len(row)
        maximums = []
        # We check for each day the maximum absolute capacity and store those
        i = 0
        while 96 * time_factor * (i + 1) < row_length:
            maximum = 0
            for j in range(96 * time_factor):
                if abs(row[96 * i * time_factor + j]) > maximum:
                    maximum = abs(row[96 * i * time_factor + j])
            maximums.append(maximum)
            i += 1
        plt.plot(list(range(len(maximums))), maximums)
        plt.show()
        
        avg = sum(maximums) / len(maximums)
        print(avg)
        
        suggested_row = np.array([])
        suggested_days = []
        i = 0
        while 96 * time_factor * (i + 1) < row_length:
            if maximums[i] <= 1.5 * avg:
                suggested_row = np.append(suggested_row, row[96 * time_factor * i : 96 * time_factor * (i + 1)])
                suggested_days.append(i)
            else:
                NaNs = np.zeros(96 * time_factor) 
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
def auto_trim(power, rows_to_trim):
    # time_factor must be 1 if the data is per 15 mins, 3 if it is per 5 mins
    time_factor = 1
    suggested_power = []
    current_row = 0
    
    # For each dataset of one group we check if there are any outliers
    for row in power:
        if current_row in rows_to_trim: 
            row_length = len(row)
            maximums = []
            # We check for each day the maximum absolute capacity and store those
            i = 0
            while 96 * time_factor * (i + 1) < row_length:
                maximum = 0
                for j in range(96 * time_factor):
                    if abs(row[96 * i * time_factor + j]) > maximum:
                        maximum = abs(row[96 * i * time_factor + j])
                maximums.append(maximum)
                i += 1
    
            avg = sum(maximums) / len(maximums)
            
            suggested_row = np.array([])
            suggested_days = []
            i = 0
            while 96 * time_factor * (i + 1) < row_length:
                if maximums[i] <= 1.5 * avg:
                    suggested_row = np.append(suggested_row, row[96 * time_factor * i : 96 * time_factor * (i + 1)])
                    suggested_days.append(i)
                else:
                    NaNs = np.zeros(96 * time_factor) 
                    suggested_row = np.append(suggested_row, NaNs)
                i += 1
            suggested_power.append(suggested_row)
        else:
            suggested_power.append(row)

        current_row += 1
    
    return suggested_power








