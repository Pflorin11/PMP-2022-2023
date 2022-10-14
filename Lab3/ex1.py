import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az

model = pm.Model()

with model:
    cutremur = pm.Bernoulli('C', 0.0005)
    incendiu_p = pm.Deterministic('I_p', pm.math.switch(cutremur, 0.03,0.01))
    incendiu = pm.Bernoulli('I', p=incendiu_p)
    alarm_p = pm.Deterministic('A_p', pm.math.switch(incendiu, pm.math.switch(cutremur, 0.98, 0.95),pm.math.switch(cutremur, 0.02, 0.0001)))
    alarm = pm.Bernoulli('A', p=alarm_p)
    trace = pm.sample(20000)

dictionary = {
              'cutremur': trace['C'].tolist(),
              'incendiu': trace['I'].tolist(),
              'alarm': trace['A'].tolist()
              }
df = pd.DataFrame(dictionary)

#probabilitate cutremur
p_cutremur = df[((df['cutremur'] == 1) & (df['alarm'] == 1))].shape[0] / df[df['alarm'] == 1].shape[0]
print(p_cutremur)

az.plot_posterior(trace)
plt.show()