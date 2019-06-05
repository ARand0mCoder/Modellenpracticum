import numpy as np
import matplotlib.pyplot as plt
import random


a = np.random.random(100)
b = []
for i in range(100):
    b.append(0.90*(0.95*np.cos((i+10)/20)/2+0.65 + 0.1*np.sin(i/5)))
b = np.array(b)

plt.plot(a)
plt.show()

plt.plot(b)
plt.show()
