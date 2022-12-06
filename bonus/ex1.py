import pandas as pd
import numpy as np
import pymc3 as pm
from scipy import stats

alpha = 2
case=1
statii=1
A=0
while not A :
  model = pm.Model()
  with model:
    clienti = pm.Poisson('N', mu=20/60)
    t_casa = pm.Normal('T_c', mu=1, sd=0.5, shape=50)
    t_gatit = pm.Exponential('T_g', lam=1/alpha, shape=50)
    t_masa = pm.Normal('T_m', mu=10, sd=2, shape=50, initval=0)
    idx = np.arange(50)
    timp = pm.math.switch(clienti > idx, t_casa[idx] / case + t_gatit[idx] / statii + t_masa[idx], 0)
    succes = pm.Deterministic('S', pm.math.prod(pm.math.switch(timp < 15, 1, 0)))
    trace = pm.sample(1000)

  succese = trace['S']
  prob = len(succese[(succese == 1)]) / len(succese)
  if prob>0.95:
    A=1
    print(case, statii, prob)
    break
  else:
    print(prob)
    if case==statii:
      case+=0.05
    else:
      statii+=0.05