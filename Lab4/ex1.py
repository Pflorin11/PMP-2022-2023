import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az

def total_t(alfa):
  total_time= []
  model = pm.Model()
  with model:
    trafic = pm.Poisson('T',1/3)
    plata = pm.Normal('P', mu=1, sigma=0.5)
    gatit = pm.Exponential('G',1/alfa)
    trace = pm.sample(1000)

    
    while len(total_time) < 10000:
      val=plata.random(1)
      while val<0 :
        val=plata.random(1)
      val=val + gatit.random(1) 
      total_time.append(val)
    return total_time

def prob(alfa, smp):
  total_time = total_t(alfa)
  counter = 0
  for x in total_time:
    if x < 15:
      counter += 1
  return counter/smp


def mean(alpha,smp):
  return np.mean(total_t(alpha))

result = 0.1

scale = 0.1

while result > 0 and prob(result, 10000) > 0.95:
  result += scale
  print(result)
  print()
result-=scale

print(result, prob(result, 10000), mean(result,10000))