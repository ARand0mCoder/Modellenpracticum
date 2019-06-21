import numpy as np
import matplotlib.pyplot as plt
import datetime

'''
--------- CONTENT ---------
This file contains the functions that are needed for data processing. This file
contains some basic plotting functions, the trimming functions and a function
that can convert a dataset that contains data for every 5 minutes to a dataset
that contains data for every 15 minutes.

There are 4 functions for trimming. Every row of the data array has to be 
trimmed separately. The idea behind the trimming algorithm is  that it is 
approximately true that the data should be trimmed away if it is too high in 
comparison to the rest of the data. Therefore the maximum of the data of every 
day is calculated. If at some day the maximum of that day is higher than 1.5 
times the average of the day maximums, then it is assumed that it has to be 
lowered, otherwise not. Any consecutive days with a maximum that is too high 
are considered to be part of the same error, so they need to be scaled down by 
the same factor. These errors are rescaled to the average of the data that does
not have to be rescaled. Because this system is not perfect, there is a function
that shows all proposed trimmed data one by one, so that the user can decide if
it is realistic. All those values can then be stored and used as input for 
automatic trimming.  
Functions:
day_maximums: does what it says, gives the maximal value of each day in the 
    dataset
trimmed_row: Trims one row, by rescaling days with an average that is too high
    to the average of the maximums of days that do not have to be changed. 
    Local differences are preserved because consecutive days that have to be
    rescaled are rescaled by the same factor.
manual_trim: This takes the datasets and shows the user for each of them how
    they look before and after trimming. The user is then asked to enter y or n,
    depending on whether he agrees that the data should be trimmed. The data
    that should be trimmed according to the user is printed in the end.
auto_trim: This can automatically trim the data, but needs as input the rows 
    that it has to trim (as outputted by manual_trim). This allows for a fully
    automated trimming procedure after the trimming has been reviewed once by
    hand. The trim_data that has to be used is currently stored in main.py.

There is also the to_quarter function. This function takes a dataset with data
for every 5 minutes and converts it to data for every 15 minutes by grouping the
data into groups of three, and taking the average of every such group.
----------------------------
'''


# Trim one row, where the maximal value of every day is given.
def trimmed_row(row, avg, row_length, timeDiff, maximums):
    # The suggested row will eventually be the trimmed row, it is built day by
    # day. The suggested_days consists of those days of the dataset whose 
    # maximum is too high. The other days are in non_suggested_days.
    suggested_row = np.array([])
    suggested_days = []
    non_suggested_days = []
    for j in range(len(maximums)):
        if maximums[j] > avg * 1.5:
            suggested_days.append(j)
        else:
            non_suggested_days.append(j)
    
    # An event is a collection of consecutive days that has to be rescaled 
    # because it is too high. The average of the day-maximums of nonevents is 
    # calculated, because this is assumed to be the true average of the data.
    non_event_tot = 0
    for i in non_suggested_days:
        non_event_tot += maximums[i]
    non_event_avg = non_event_tot / max(1, len(non_suggested_days))
    
    # Consecutive days make one event, and the following code groups the days 
    # with a too high maximum into events.
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
    
    # For every event the average is calculated.
    event_avgs = []
    for event in events:
        event_avgs.append(sum(maximums[event[0] : event[0] + len(event)]) / len(event))
    
    # This loop does the true trimmming. It loops over all days in the code. If
    # a day is in some event, then it is rescaled to the non_event_avg, the 
    # average of days that are not in events, by the event_avg. This ensures
    # that differences between consecutive days in an event are preserved. Days
    # that are not in an event are simply appended to the suggested date
    # without any change.
    i = 0
    while 24*60//timeDiff * (i + 1) <= row_length:
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

# This automatically trims the rows of power that have to be trimmed
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


#%%

def to_quarter(data):
    # Converts data points given for each 5 minutes to data for each 15 minutes
    # by averaging every block of three data points.
    new_power = []
    for row in data:
        split_row = np.array_split(row, len(row)//3)
        new_row = np.mean(split_row, axis=1)
        new_power.append(new_row)
    return np.array(new_power)
    


