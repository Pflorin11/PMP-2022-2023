import numpy as np
from scipy import stats
from random import choices

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)
x=stats.expon.rvs(0,1/4,size=10000)
y=stats.expon.rvs(0,1/6,size=10000)

z=stats.expon.rvs(0,0,size=10000)

for client in range (10000):
    z[client]= np.float64(choices([x[client],y[client]], [0.4,0.6]))
az.plot_posterior({'M1':x,'M2':y,'X':z}) 
plt.show() 

