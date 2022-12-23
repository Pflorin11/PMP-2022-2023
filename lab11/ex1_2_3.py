import numpy as np
import arviz as az
import random
import pymc3 as pm
import matplotlib.pyplot as plt
import scipy.stats as stats

def posterior_grid(grid_points=50, heads=6, tails=9):
    grid = np.linspace(0, 1, grid_points)
    prior = abs(grid - 0.5)
    likelihood = stats.binom.pmf(heads, heads + tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior


data = np.repeat([0, 1], (43, 30))
points = 50
h = data.sum()
t = len(data) - h
grid, posterior = posterior_grid(points, h, t)
plt.plot(grid, posterior, 'o-')
plt.title(f'heads = {h}, tails = {t}')
plt.yticks([])
plt.xlabel('θ')
plt.show()

def comp_pi(n):
    x, y = np.random.uniform(-1, 1, size=(2, n))
    inside = (x ** 2 + y ** 2) <= 1
    pi = inside.sum() * 4 / n
    error = abs((pi - np.pi) / pi) * 100
    return error


valuesN = [10,100, 1000, 10000,100000]
for N in valuesN:
    errors = []
    for i in range(1000):
        errors.append(comp_pi(N))
    std_dev_error = np.std(errors)
    mean_error = sum(errors) / len(errors)
    print(std_dev_error, mean_error)



def metropolis(func, draws=10000):
    trace = np.zeros(draws)
    old_x = 0.5 
    old_prob = func.pdf(old_x)
    delta = np.random.normal(0, 0.5, draws)
    for i in range(draws):
        new_x = old_x + delta[i]
        new_prob = func.pdf(new_x)
        acceptance = new_prob / old_prob
        if acceptance >= np.random.random():
            trace[i] = new_x
            old_x = new_x
            old_prob = new_prob
        else:
            trace[i] = old_x
    return trace


beta_params = [(1, 1), (20, 20), (1, 4)]
for params in beta_params:
    func = stats.beta(params[0], params[1])
    trace = metropolis(func=func)
    x = np.linspace(0.01, .99, 100)
    y = func.pdf(x)
    plt.xlim(0, 1)
    plt.plot(x, y, 'C1-', lw=3, label='True distribution')
    plt.hist(trace[trace > 0], bins=25, density=True, label='Estimated distribution')
    plt.xlabel('x')
    plt.ylabel('pdf(x)')
    plt.yticks([])
    plt.show()