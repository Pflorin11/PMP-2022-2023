import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm

df = pd.read_csv("C:\\Users\\mihne\\Desktop\\PMP2022\\data.csv")

mage = df['momage']
res = df['ppvt']

res=res.values.reshape(len(res),1)
mage=mage.values.reshape(len(mage),1)
plt.scatter(res, mage)
plt.xticks(())
plt.yticks(())

with pm.Model() as model_g:
    α = pm.Normal('α', mu=0, sd=15)
    β = pm.Normal('β', mu=0, sd=5)
    ε = pm.HalfCauchy('ε', 5)
    μ = pm.Deterministic('μ', α + β * res)
    y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=mage)
    
    idata_g = pm.sample(100, tune=100, return_inferencedata=True)


posterior_g = idata_g.posterior.stack(samples={"momage", "ppvt"})

print(posterior_g)