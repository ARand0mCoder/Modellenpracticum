##Karmarkar-Karp   #Still seems to be a lot worse than greedy algorithm
def Norm(array): #Choose your own norm. Average of the highest n is also possible
    # Make sure that Norm greater or equal to 0
    return np.amax(np.abs(array))
    
def Karmarkar_Karp(power): #only for m=2
    data0 = np.zeros(len(power))
    for i in range(len(power)):
        data0[i] = Norm(power[i])
    data1 = list(range(0,len(data0)))
    data = [data0,data1]    #data0 consists of the norms of the actual data, data1 is the index corresponding to the norm
    
    connectlines = [[] for i in range(len(data[0])-1)]  #If one considers the norms as points, the first step of K-K is to find the lines which connect these points
    
    for i in range(len(data[0])-1):
        data[0],data[1] = zip(*sorted(zip(data[0],data[1])))
        data = [np.array(data[0]),np.array(data[1])]
        data[0][-1] = Norm(power[1][-1] - data[1][-2]) #update norm
        #data[0][-1] = data[0][-1] - data[0][-2]    #don't update norm
        data[0][-2] = 0
        connectlines[i].append(data[1][-1])
        connectlines[i].append(data[1][-2])
        data = np.delete(data, -2, 1)
    
    station1 = []
    station2 = []
    
    station1.append(connectlines[0][0]) #The second step of K-K is to colour the obtained graph with two colours, s.t. points which are connected by a line have opposite colors. This will give the desired two groups.
    for i in range(len(connectlines)-1):    #Probably a faster way to do this, but this works.
        for elem in station1:
            for elem2 in connectlines:
                if elem in elem2:
                    elem2.remove(elem)            
                    station2 = np.append(station2,elem2)
                    connectlines.remove(elem2)
                    station2 = np.hstack(station2)
        for elem in station2:            
            for elem2 in connectlines:
                if elem in elem2:
                    elem2.remove(elem)            
                    station1 = np.append(station1,elem2)
                    connectlines.remove(elem2)
                    station1 = np.hstack(station1)
                    
    return(station1,station2)
    
station1, station2 = Karmarkar_Karp(power)  #find the actual max
powerstation1, powerstation2 = np.zeros(len(power[0])), np.zeros(len(power[0]))
for i in station1:
    powerstation1 += power[int(i)]
print(max(powerstation1))
for i in station2:
    powerstation2 += power[int(i)]
print(max(powerstation2))
    
