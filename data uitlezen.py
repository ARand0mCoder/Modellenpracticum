import numpy as np
import matplotlib.pyplot as plt
import csv
from dateutil import parser

time = []           # Time and day
power = []          # Data of power from excel
isDayFirst = False  # Used for date parsing
checkDate = True    # When to stop checking date parsing

path = r''          # Path to the file

with open(path) as csv_file:
    csv_reader = csv.reader(csv_file,delimiter=';') # Read excel file as .csv
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            columns = row[2:]   # Names of the columns of power
            print(f'Kolommen zijn: {", ".join(columns)}')
            print('\nAan het inlezen...')
        else:
            if checkDate and row[0][0:3] == '1-2': # Checking if date is given by month-day or day-month
                checkDate = False
            elif checkDate and row[0][0:3] == '2-1':
                checkDate = False
                isDayFirst = True
            time.append(parser.parse(row[0]+' '+row[1], dayfirst=isDayFirst)) # Parse date and time
            temp_line = []
            for elem in row[2:]:
                temp_line.append(float(elem))
            power.append(temp_line)  # Stores power data
        if line_count % 10000 == 0:  # Prints progress
            print(line_count)
        line_count += 1
    print(f'Well, that was easy. Totaal {line_count} regels ingelezen.')

for i in range(len(row)-2):
    # plt.figure()
    data = [line[i] for line in power]
    plt.plot(time,data)
plt.show()
