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

def trimmed_row(row, avg, row_length, timeDiff, maximums):
        suggested_row = np.array([])
        suggested_days = []
        non_suggested_days = []
        for j in range(len(maximums)):
            if maximums[j] > avg * 1.5:
                suggested_days.append(j)
            else:
                non_suggested_days.append(j)
        
        non_event_tot = 0
        for i in non_suggested_days:
            non_event_tot += maximums[i]
        non_event_avg = non_event_tot / max(1, len(non_suggested_days))
        
        events = []
        cur_event = []
        for j in suggested_days:
            if cur_event == [] or j - 1 in cur_event:
                cur_event.append(j)
            else:
                events.append(cur_event)
                cur_event = [j]
        if cur_event != []: 
            events.append(cur_event)
        
        event_avgs = []
        for event in events:
            event_avgs.append(sum(maximums[event[0] : event[0] + len(event)]) / len(event))
        print(event_avgs)
        
        i = 0
        while 24*60//timeDiff * (i + 1) < row_length:
            if i not in suggested_days:
                suggested_row = np.append(suggested_row, row[24*60//timeDiff * i : 24*60//timeDiff * (i + 1)])
            else:
                event_num = 0
                for k in range(len(events)):
                    if i in events[k]:
                        event_num = k
                        break
                        
                averaged_day = row[24*60//timeDiff * i: 24*60//timeDiff * (i + 1)] * (non_event_avg / event_avgs[event_num])
                suggested_row = np.append(suggested_row, averaged_day)
            i += 1

        return suggested_row


def day_maximums(row, row_length, timeDiff):
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
    return maximums, avg


# This function is used for manually trimming the data and gives the user the
# ability to agree on the trimmed data.
def manual_trim(power, timeDiff):
    rows_to_trim = []
    current_row = 0

    # For each dataset of one group we check if there are any outliers
    for row in power:
        row_length = len(row)
        maximums, avg = day_maximums(row, row_length, timeDiff)
        
        plt.plot(list(range(len(maximums))), maximums)
        plt.show()

        print(avg)

        suggested_row = trimmed_row(row, avg, row_length, timeDiff, maximums)
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
            
            maximums, avg = day_maximums(row, row_length, timeDiff)

            suggested_power.append(trimmed_row(row, avg, row_length, timeDiff, maximums))
        else:
            suggested_power.append(row)

        current_row += 1

    return suggested_power
