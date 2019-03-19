import numpy as np
import matplotlib.pyplot as plt
import sys

# Enter directory of you files here, so that python knows where to look
file_dir = r'C:\Users\dion\Documents\Studie\Modellenpracticum'
sys.path.append(file_dir)

import Dataverwerking as Dv
import GreedyAlgorithm as GA
import Karmarkar_karp_v1 as kkk
import MonteCarloTwist as MCT

#%%
def Norm(array): #Choose your own norm. Average of the highest n is also possible
    # Make sure that Norm greater or equal to 0
    return np.amax(np.abs(array))


#%%


path1 = r'C:\Users\dion\Documents\Studie\Modellenpracticum\Data\Uilenburg_1_2018_clean.csv'
path2 = r'C:\Users\dion\Documents\Studie\Modellenpracticum\Data\Uilenburg_2_2018_clean.csv'

rows_to_trim1 = [0, 1, 2, 3, 4, 5, 8, 9]
rows_to_trim2 = [0, 2, 3, 4, 7]

power1, time1 = Dv.read_data(path1)
power2, time2 = Dv.read_data(path2)

power1 = np.transpose(np.array(power1))
power2 = np.transpose(np.array(power2))

power1 = Dv.auto_trim(power1, rows_to_trim1)
power2 = Dv.auto_trim(power2, rows_to_trim2)
#%%
power = []
for row in power1:
    power.append(row[0:34944])
for row in power2:
    power.append(row[0:34944])

#%%
solution1 = GA.FirstAlgorithm(power, 2, Norm)

#%%

def plot_solution(power, sol, m):
    for i in range(m):
        tot = 0
        for j in sol[i]:
            tot += power[j]
        plt.plot(tot)
        plt.show()
#%%

def sol_norms(power, sol, Norm):
    for group in sol:
        tot = np.zeros(len(power[0]))
        for val in group:
            tot += power[val]
        print(Norm(tot))

