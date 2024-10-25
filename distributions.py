import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 9, 10)
y1 = np.random.poisson(100,1000) # first value is mean, second value is number of samples
y2 = np.random.normal(100, np.sqrt(100),1000) # mean, standard deviation, number of samples 
print(y1)
print(y2)

count, bins, ignored = plt.hist(y1, 20, density=True, histtype=u'step')
count2, bins2, ignored2 = plt.hist(y2, 20, density=True, histtype=u'step')

#plt.figure()
#plt.plot(x, y1)
#plt.plot(x, y2)

plt.show()