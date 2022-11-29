import arviz as az
import matplotlib.pyplot as plt

import numpy as np
import pymc3 as pm
import pandas as pd
import math

if __name__ == "__main__":
    data = pd.read_csv('Admission.csv')
    adm = data['Admission'].values
    gre = data['GRE'].values
    gpa = data['GPA'].values
    gre_c = gre - gre.mean()
    model = pm.Model()

    with model:
        alpha  = pm.Normal('Alpha',  mu = 0, sd = 10)
        beta_0 = pm.Normal('Beta_0', mu = 0, sd = 10)
        beta_1 = pm.Normal('Beta_1', mu = 0, sd = 10)
        beta_2 = pm.Normal('Beta_2', mu = 0, sd = 10)
        pi = pm.Deterministic('Pi', pm.math.sigmoid(beta_0 + beta_1 * gre + beta_2 * gpa))
        bd = pm.Deterministic('bd', -alpha / beta_0)
        idata  = pm.sample(1000, return_inferencedata = True, cores = 4, step = pm.Slice())

    posterior = idata.posterior.stack(samples = ("chain", "draw"))
    theta = posterior['Pi'].mean("samples")
    idx = np.argsort(gre_c)
    plt.plot(gre_c[idx], theta[idx], lw = 3)
    plt.vlines(posterior['bd'].mean(), 0, 1, color='k')
    bd_hpd = az.hdi(posterior['bd'].values)
    plt.fill_betweenx([0, 1], bd_hpd[0], bd_hpd[1], color = 'k', alpha = 0.5)
    plt.scatter(gre_c, np.random.normal(adm, 0.02), marker = '.', color = [f'C{x}'for x in adm])
    az.plot_hdi(gre_c, posterior['Pi'].T, smooth = False)
    plt.xlabel('GRE')
    plt.ylabel('Pi', rotation = 0)
    locs, _ = plt.xticks()
    plt.xticks(locs, np.round(locs + gre.mean(), 1))
    plt.show()