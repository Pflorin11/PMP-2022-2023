import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az

model = pm.Model()
alfa=10
with model:
    trafic = pm.Poisson('T',1/3)
    plata = pm.Normal('P', mu=1, sigma=0.5)
    gatit = pm.Exponential('G',alfa)
    trace = pm.sample(20000)

