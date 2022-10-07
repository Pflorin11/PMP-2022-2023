import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)
#x = stats.norm.rvs(0, 1, size=10000) # Distributie normala cu media 0 si deviatie standard 1, 1000 samples
#y = stats.uniform.rvs(-1, 2, size=10000) # Distributie uniforma intre -1 si 1, 1000 samples . Primul parametru fiind limita inferioara a intervalului, al doilea parametru fiind "marimea" intervalului, aka [-1,-1+2] = [-1,1] 
x=stats.expon.rvs(0,1/4,size=1000)
y=stats.expon.rvs(0,1/6,size=1000)

z = x+y # Compunerea prin insumare a celor 2 distributii

az.plot_posterior({'M1':x,'M2':y,'X':z}) # Afisarea aproximarii densitatii probabilitatilor, mediei, intervalului etc. variabilelor x,y,z
plt.show() 