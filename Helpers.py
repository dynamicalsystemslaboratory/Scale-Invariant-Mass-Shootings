import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import log, sqrt, exp
from scipy.stats import ks_2samp, pareto, kstest, linregress, genpareto, expon, kstest, rv_continuous, kstest
import statsmodels.api as sm
from scipy.optimize import minimize_scalar, minimize
from scipy.special import gamma, gammainc, gammaincc, zeta, erf, gamma


def expo_pmf(x, xmin, l):  ## From SIAM
    return np.exp(-x*l)*np.exp(xmin*l)*(1-np.exp(-l))

def lognorm_pmf(x, xmin, mu, sig):
    def F(k):
        return 0.5+0.5*erf((log(k)-mu)/(sig*sqrt(2)))
    return (F(x+1)-F(x))/(1-F(xmin))

def weibull_pmf(x, xmin, a, b):
    return exp((xmin/b)**a)*(exp(-(x/b)**a)-exp(-((x+1)/b)**a))

def powerlaw_pmf(x, xmin, a):
    return np.asarray(x, dtype=float)**(-a)/(zeta(a,xmin))

def yule_pdf(x, xmin, a):  ## From SIAM:
    x = np.asarray(x)
    return (gamma(x))/(gamma(x+a))*(a-1)*((gamma(xmin+a-1))/gamma(xmin))


def loglikhds(x, model, params):
    val = PMF(x, model, params)
    # Clip very small values to avoid log(0)
    val = np.clip(val, 1e-300, None)
    return np.log(val)

def loglikhd(x, model, params):
    return np.sum(loglikhds(x, model, params))

def MLE(x, model, xmin):
    if model == "expo": bounds = [(1e-6, None)]; x0 = 1
    if model == "powerlaw": bounds = [(1+1e-6, None)]; x0 = 2
    if model == "lognorm": bounds = [(-len(x)/5, None), (1, None)]; x0 = [0, 1.0]
    if model == "weibull": bounds = [(1e-6, 10), (0.1, 10)]; x0 = [1.0, 1.0]
    if model == "yule-simon": bounds = [(1+1e-6, None)]; x0 = 5

    def objective(params_tbo):
        return -loglikhd(x, model, [xmin] + list(params_tbo))

    result = minimize(objective, x0, bounds=bounds, tol= 1e-6)
    # if not result.success:
    #     result = minimize(objective, x0, bounds=bounds, method="Powell", options={"maxiter": 2000, "xtol": 1e-6}) ## this ruins the weibull fit

    params = result.x
    LogLikhd = -result.fun
    success = result.success
    return params, LogLikhd, success

def PMF(x, model, params):
    if model == "expo": return expo_pmf(x, *params)
    if model == "powerlaw": return powerlaw_pmf(x, *params)
    if model == "lognorm": return lognorm_pmf(x, *params)
    if model == "weibull": return weibull_pmf(x, *params)
    if model == "yule-simon": return yule_pdf(x, *params)


def CCDF(x, model, params):
    x = np.atleast_1d(x) 
    result = np.array([
        1 - np.sum(PMF(np.arange(params[0], xi, 1), model, params))
        for xi in x
    ])
    return result if len(result) > 1 else result[0]


def inverse_CCDF(r, model, params):
    x1 = int(params[0])
    x2 = int(params[0])

    while CCDF(x2, model, params) > 1 - r:
        x1 = x2
        x2 *= 2

    for i in range(x1, x2+1):
        if CCDF(i, model, params) < 1 - r: return i-1


def sample_discrete(size, model, params):
    sample = np.zeros([size])
    rs = np.random.uniform(0,1,size)
    for i in range(size):
        sample[i] = inverse_CCDF(rs[i], model, params)
    return rs, sample



def Relative_entropy(X, Px, Q_model, params):
    Qx = PMF(X, Q_model, params)
    D = np.sum(Px*log(Px/Qx))
    return D


def plug_in_pdf(Data):
    N = len(Data)
    X, counts = np.unique(Data, return_counts=True)
    Px = counts / N
    return X, Px

def plug_in_ccdf(Data):
    X, Px = plug_in_pdf(Data)
    Sx = np.cumsum(Px[::-1])[::-1]
    return X, Sx

