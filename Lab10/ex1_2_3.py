import numpy as np
import arviz as az
import pymc3 as pm
import matplotlib.pyplot as plt
from theano import *
import theano.tensor as tt
clusters = 3
n_cluster = [130, 170, 200]
n_total = sum(n_cluster)
means = [-1,2,5]
std_devs = [2, 1, 2]
mix = np.random.normal(np.repeat(means, n_cluster),np.repeat(std_devs, n_cluster))
az.plot_kde(np.array(mix));

clusters = [2,3,4]
models = []
idatas = []
for cluster in clusters:
  with pm.Model() as model:
    p = pm.Dirichlet('p', a=np.ones(cluster))
    means = pm.Normal('means', mu=np.linspace(mix.min(), mix.max(), cluster), sd=10, shape=cluster, transform=pm.distributions.transforms.ordered)
    sd = pm.HalfNormal('sd', sd=10)
    order_means = pm.Potential('order_means',tt.switch(means[1]-means[0] < 0 ,-np.inf, 0))
    y = pm.NormalMixture('y', w=p, mu=means, sd=sd, observed=mix)
    idata = pm.sample(1000, tune=2000, target_accept=0.9, random_seed=123, return_inferencedata=True)
    idatas.append(idata)
    models.append(model)

waic_comp = az.compare(dict(zip([str(c) for c in clusters], idatas)), method='BB-pseudo-BMA', ic="waic", scale="deviance")
az.plot_compare(waic_comp)
print(waic_comp)
loo_comp = az.compare(dict(zip([str(c) for c in clusters], idatas)), method='BB-pseudo-BMA', ic="loo", scale="deviance")
az.plot_compare(loo_comp)
print(loo_comp)