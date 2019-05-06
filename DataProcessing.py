import numpy as np
import matplotlib.pyplot as plt

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
