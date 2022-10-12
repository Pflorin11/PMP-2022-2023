import numpy as np
from scipy import stats
from random import choices

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

s1=stats.gamma.rvs(4,0,1/3,size=1000)
s2=stats.gamma.rvs(4,0,1/2,size=1000)
s3=stats.gamma.rvs(5,0,1/2,size=1000)
s4=stats.gamma.rvs(5,0,1/3,size=1000)

latency=stats.expon.rvs(0,1/4,size=1000)

x=stats.gamma.rvs(5,0,0,size=1000)
print(type(x))
for client in range (1000):
    x[client]= np.float64(choices([s1[client],s2[client],s3[client],s4[client]], [0.25, 0.25, 0.3, 0.2]))+latency[client]
az.plot_posterior({'S1':s1,'S2':s2,'S3':s3,'S4':s4,'LAT':latency,'X':x}) 
plt.show() 
print(np.count_nonzero(x > 3)/1000 ,"%")

