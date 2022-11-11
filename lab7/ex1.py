import arviz as az
import matplotlib.pyplot as plt

import numpy as np
import pymc3 as pm
import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv('Prices.csv')

    price = data['Price'].values
    speed = data['Speed'].values
    hard_drive = data['HardDrive'].values
    ram = data['Ram'].values
    premium = data['Premium'].values


    prices_model = pm.Model()
    with prices_model:
        alpha = pm.Normal('alpha', mu=0, sd=10)
        beta_1 = pm.Normal('beta_1', mu=0, sd=10)
        beta_2 = pm.Normal("beta_2", mu=0, sd=5)
        sigma = pm.HalfNormal('sigma', sd=10)
        mu = pm.Deterministic("mu", alpha + beta_1 * speed + beta_2 * hard_drive)
        price_like = pm.Normal('price_like', mu=mu, sigma=sigma, observed=price)
        trace = pm.sample(2000, tune=2000, cores=4)
        prm = pm.sample_posterior_predictive(trace, samples=100, model=prices_model)

    az.plot_trace(prm)
    plt.show()