from pandas import *
import matplotlib.pyplot as plt
import arviz as arv
import numpy as np

#data = files.upload()
data = read_csv("data.csv")

#x = np.array(data["momage"].tolist())
x = np.array(data["educ_cat"].tolist())

y = np.array(data["ppvt"].tolist())

with pm.Model() as model_g:
  a = pm.Normal('α', mu=0, sd=10)
  b = pm.Normal('β', mu=0, sd=1)
  e = pm.HalfCauchy('ε', 5)
  u = pm.Deterministic('μ', a + b * x)
  y_pred = pm.Normal('y_pred', mu=u, sd=e, observed=y)
  idata_g = pm.sample(2000, tune=2000, return_inferencedata=True)
  plt.plot(x, y, 'C0.')
  posterior_g = idata_g.posterior.stack()
  alpha_m = posterior_g['α'].mean().item()
  beta_m = posterior_g['β'].mean().item()
  plt.plot(x, posterior_g['α'][0].values + posterior_g['β'][0].values * x[:,None],c='gray', alpha=0.5)
  plt.plot(x, alpha_m + beta_m * x, c='k',label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
  plt.xlabel('x')
  plt.ylabel('y', rotation=0)
  plt.legend()