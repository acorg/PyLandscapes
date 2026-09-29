#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 13:53:32 2025

@author: avicenna
"""

import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import pytensor
import numpy as np
import sys
from scipy.stats import norm, t
from .landscape_utils import log2, _log2

sys.path.append("../")
_eps = np.finfo(pytensor.config.floatX).eps
default_prior_params = {"nu_mean":10, "nu_sd":5,
                        "sigma_mean":1.5, "sigma_sd":1,
                        "mu_mean":5, "mu_sd":5,
                        "nu_threshold":[1, 15],
                        "sigma_threshold":[0.1, 10]}


def t_discrete_censored(table, lld, uld, step,
                        use_nonparametric_estimates=False,
                        prior_params=None):
  return _discrete_censored(table, lld, uld, step, 't',
                            use_nonparametric_estimates, prior_params)

def n_discrete_censored(table, lld, uld, step,
                        use_nonparametric_estimates=False,
                        prior_params=None):
  return _discrete_censored(table, lld, uld, step, 'n',
                            use_nonparametric_estimates,
                            prior_params)

def t_continuous_censored(table, lld, uld,
                          use_nonparametric_estimates=False,
                          prior_params=None):
  return _continuous_censored(table, lld, uld, 't',
                              use_nonparametric_estimates,
                              prior_params)

def n_continuous_censored(table, lld, uld,
                          use_nonparametric_estimates=False,
                          prior_params=None):
  return _continuous_censored(table, lld, uld, 'n',
                              use_nonparametric_estimates,
                              prior_params)

def _discrete_censored(table, lld, uld, step, ltype,
                       use_nonparametric_estimates=True, prior_params=None):

  if prior_params is None:
    prior_params = {}

  prior_params = dict(default_prior_params, **prior_params)

  if ltype=='n':
    del prior_params["nu_sd"]

  observed, lower, upper, idx, coords = prep_data(table, lld, uld)

  model_meta = {}
  model_meta["observed"] = observed
  model_meta["lower"] = lower
  model_meta["upper"] = upper
  model_meta["idx"] = idx
  model_meta["coords"] = coords

  if use_nonparametric_estimates:
    sd, means = nonparametric_estimates(table, coords["pair"])
    prior_params["mu_mean"] = means
    prior_params["sigma_mean"] = sd
    prior_params["sigma_sd"] = np.min([sd, prior_params["sigma_sd"]])

    model_meta["nonparametric"] = {}
    model_meta["nonparametric"]["sd"] = sd
    model_meta["nonparametric"]["means"] = means


  with pm.Model(coords=coords) as model:

    idx_pair = pm.Data("idx_pair", idx["pair"])
    idx_exp = pm.Data("idx_exp", idx["experiment"])
    lower = pm.Data("lower", lower)
    upper = pm.Data("upper", upper)

    sigma = pm.InverseGamma.dist(mu=prior_params["sigma_mean"],
                                 sigma=prior_params["sigma_sd"])
    sigma = pm.Truncated("sigma", sigma, *prior_params["sigma_threshold"])

    mu = pm.Normal("mu", prior_params["mu_mean"], prior_params["mu_sd"],
                   dims="pair")

    if len(coords["experiment"])>1:
      exp_bias = pm.ZeroSumNormal("b", 0.5, dims="experiment")[idx_exp]
    else:
      exp_bias = 0

    if ltype=='t':
      nu = pm.InverseGamma.dist(mu=prior_params["nu_mean"],
                                   sigma=prior_params["nu_sd"])
      nu = pm.Truncated("nu", nu, *prior_params["nu_threshold"])

      pm.CustomDist("obs", mu[idx_pair] + exp_bias, sigma, nu, lower, upper,
                    step, logp=logp_t_discrete, random=random_t_discrete,
                    observed=observed, dtype="int64",
                    dims="obs_dim")

    elif ltype=='n':
      pm.CustomDist("obs", mu[idx_pair] + exp_bias, sigma, lower, upper, step,
                    logp=logp_n_discrete, random=random_n_discrete,
                    observed=observed, dtype="int64",
                    dims="obs_dim")
    else:
      raise ValueError(f"Unknown likelihood type (ltype) {ltype}")

  return model, prior_params, model_meta


def _continuous_censored(table, lld, uld, ltype,
                         use_nonparametric_estimates=False,
                         prior_params=None,):

  if prior_params is None:
    prior_params = {}

  prior_params = dict(default_prior_params, **prior_params)

  if ltype=='n':
    del prior_params["nu_sd"]

  observed, lower, upper, idx, coords = prep_data(table, lld, uld)

  model_meta = {}
  model_meta["observed"] = observed
  model_meta["lower"] = lower
  model_meta["upper"] = upper
  model_meta["idx_pair"] = idx
  model_meta["coords"] = coords

  if use_nonparametric_estimates:
    sd, means = nonparametric_estimates(table, coords["pair"])
    prior_params["mu_mean"] = means
    prior_params["sigma_mean"] = sd
    prior_params["sigma_sd"] = np.min([sd, prior_params["sigma_sd"]])

    model_meta["nonparametric"] = {}
    model_meta["nonparametric"]["sd"] = sd
    model_meta["nonparametric"]["means"] = means


  with pm.Model(coords=coords) as model:

    idx_pair = pm.Data("idx_pair", idx["pair"])
    idx_exp = pm.Data("idx_experiment", idx["experiment"])

    lower = pm.Data("lower", lower)
    upper = pm.Data("upper", upper)

    if len(coords["experiment"])>1:
      exp_bias = pm.ZeroSumNormal("b", 0.5, dims="experiment")[idx_exp]
    else:
      exp_bias = 0

    sigma = pm.InverseGamma.dist(mu=prior_params["sigma_mean"],
                                 sigma=prior_params["sigma_sd"])
    sigma = pm.Truncated("sigma", sigma, *prior_params["sigma_threshold"])
    mu = pm.Normal("mu", prior_params["mu_mean"], prior_params["mu_sd"],
                   dims="pair")

    if ltype=='t':
      nu = pm.InverseGamma.dist(mu=prior_params["nu_mean"],
                                sigma=prior_params["nu_sd"])
      nu = pm.Truncated("nu", nu, *prior_params["nu_threshold"])

      dist = pm.StudentT.dist(mu=mu[idx_pair] + exp_bias, sigma=sigma, nu=nu)
    elif ltype=='n':
      dist = pm.Normal.dist(mu=mu[idx_pair] + exp_bias, sigma=sigma)
    else:
      raise ValueError(f"Unknown likelihood type (ltype) {ltype}")

    pm.Censored("obs", dist, lower, upper, observed=observed,
                dims="obs_dim")

  return model, prior_params, model_meta


def nonparametric_estimates(table, pairs):



  table = table.copy()
  table["combined_index"] = table["ag_id"] + ";" + table["sr_id"]
  table["log_titer"] = table["titer"].map(lambda x: _log2(x, False, 1))

  means = []
  difs = []
  for pair in pairs:
    subtable = table[table.combined_index==pair]
    vals = subtable["log_titer"].values
    vals = vals[~(np.isnan(vals.astype(float)))]

    if vals.size>1:
      difs += list(vals-vals.mean())

    means.append(np.nanmean(vals))

  sd = np.nanstd(difs)

  return sd, means


def prep_data(table, lld, uld):


  if "experiment_id" not in table.columns:
    table["experiment_id"] = np.ones((table.shape[0],))

  table = table[~(table.titer.isin(['*','nan'])) & ~pd.isna(table.titer)
                & ~(table.titer.str.contains(r'\?', na=False))].copy()

  table["pair_id"] = table.loc[:, "ag_id"] + ';' + table.loc[:, "sr_id"]

  mid_values = [10*2**i for i in range(lld, uld+1)]
  mid_values = [str(int((x+y)/2)) for x,y in zip(mid_values[:-1], mid_values[1:])]
  table["titer"] = table["titer"].map(lambda x: _replace_mid_values(x, mid_values))

  coords = {"pair":list(pd.unique(table.loc[:,"pair_id"].values)),
            "experiment":list(pd.unique(table.loc[:,"experiment_id"].values))
            }

  idx = {}
  for key in ["pair", "experiment"]:
    idx[key] = [coords[key].index(x) for x in table.loc[:,f"{key}_id"]]

  data = [convert_data(x) for x in table.loc[:, "titer"].values]

  lower = np.array([x[0] for x in data])
  upper = np.array([x[-1] for x in data])
  observed = np.array([x[1] for x in data])

  _uld = uld+_eps
  _lld = lld-_eps

  if all(np.isinf(x) for x in lower):
    lower = _lld
    assert all(x>=_lld or np.isnan(x) for x in observed)
  else:
    lower = [l if not np.isinf(l) else _lld for x,l in zip(observed,lower)]
    assert all(x>=l or np.isnan(x) for x,l in zip(observed, lower))

  if all(np.isinf(x) for x in upper):
    upper = _uld
    assert all(x<=_uld or np.isnan(x) for x in observed)
  else:
    upper = [u if not np.isinf(u) else _uld for x,u in zip(observed,upper)]
    assert all(x<=u or np.isnan(x) for x,u in zip(observed, upper))

  obs_dim = [f"{table.loc[label,'ag_id']};{table.loc[label,'sr_id']};{label}"
             for label in table.index]
  coords["obs_dim"] = obs_dim

  return observed, lower, upper, idx, coords


def logp_t_discrete(value, mu, sd, nu, lower, upper, step):

  '''
  if value==lower, p = cdf(lower, params)
  if value==upper, p = 1 - cdf(upper, params) # not used because we dont have any
  else: p = cdf(val+1, params) - cdf(val, params)
  '''
  val = pt.switch(pt.eq(value, lower),
                  _log_lower('t', lower, mu, sd,  nu),
                  _log_interval('t', value, mu, sd, step, nu)
                  )

  return val

def logp_n_discrete(value, mu, sd, lower, upper, step):

  '''
  if value==lower, p = cdf(lower, params)
  if value==upper, p = 1 - cdf(upper, params) # not used because we dont have any
  else: p = cdf(val+1, params) - cdf(val, params)
  '''

  val = pt.switch(pt.eq(value, lower),
                  _log_lower('n', lower, mu, sd),
                  _log_interval('n', value, mu, sd, step)
                  )

  return val


def _log_interval(ltype, value, mu, sd, step, nu=None):

  if ltype == 't':
    return\
      pm.math.log(pm.math.exp(pm.StudentT.logcdf(value + step, nu, mu, sd)) -\
        pm.math.exp(pm.StudentT.logcdf(value, nu, mu, sd)))
  else:
    return\
      pm.math.log(pm.math.exp(pm.Normal.logcdf(value + step, mu, sd)) -\
        pm.math.exp(pm.Normal.logcdf(value, mu, sd)))


def _log_lower(ltype, lower, mu, sd, nu=None):
  if ltype=="t":
    return pm.StudentT.logcdf(lower, nu, mu, sd)
  else:
    return pm.Normal.logcdf(lower, mu, sd)


def _log_upper(ltype, upper, mu, sd, nu=None):
  if ltype=='t':
    return pm.math.log(1-pm.math.exp(pm.StudentT.logcdf(upper, nu, mu, sd)))
  else:
    return pm.math.log(1-pm.math.exp(pm.Normal.logcdf(upper, mu, sd)))


def random_t_discrete(mu, sigma, nu, lower, upper, step, rng=None, size=None):

    draws = t.rvs(loc=mu, scale=sigma, df=nu, size=size,
                  random_state=rng)
    draws = step*np.floor(draws/step).astype(int)

    if lower.size==1:
      lower = np.tile(lower, draws.size)
    if upper.size==1:
      upper = np.tile(upper, draws.size)

    draws[draws<lower] = lower[draws<lower]
    draws[draws>upper] = upper[draws>upper]

    return draws


def random_n_discrete(mu, sigma, lower, upper, step, rng=None, size=None):

    draws = norm.rvs(loc=mu, scale=sigma, size=size,
                     random_state=rng)
    draws = step*np.floor(draws/step).astype(int)

    if lower.size==1:
      lower = np.tile(lower, draws.size)
    if upper.size==1:
      upper = np.tile(upper, draws.size)

    draws[draws<lower] = lower[draws<lower]
    draws[draws>upper] = upper[draws>upper]

    return draws


def _replace_mid_values(x, mids):
  '''
  replace values like 30 with 10/20.
  '''

  dil_f = float(mids[1])/float(mids[0])

  if str(x) in mids:
    x = 10*2**np.floor(np.log2(float(x)/10)).astype(int)
    return f"{x}/{int(dil_f*x)}"

  else:
    return x


def convert_data(x, shift=0):
  '''
  we always use log2 and not log2 discrete. If a titer is for instance
  10/20 then its log2 value is 0.5 and the likelihood is
  cdf(1) - cdf(0.5) instead of cdf(1) - cdf(0). That is why we don't
  use log discrete here which would convert this to 1. The fact that
  we don't use discrete also means that values like 30 remain as they are

  '''

  lower = -np.inf
  upper = np.inf

  if x[0]=='?':
    return lower, np.nan, upper

  if '/' in x:

    lower1, _, upper1 = convert_data(x.split('/')[0], shift=0)
    lower2, _, upper2 = convert_data(x.split('/')[1], shift=0)

    val = _log2(x, False, shift=0)

    return max([lower1, lower2]), val, min([upper1, upper2])

  if x[0] == '<':
    val = log2(x) - shift
    lower = val
  elif x[0] == '>':
    val = log2(x) + shift
    upper = val
  else:
    val = log2(x)

  return lower, val, upper


def sample_observables(model, idata, pp, lld, uld):

  lower = np.array(idata["constant_data"]["lower"])
  upper = np.array(idata["constant_data"]["upper"])

  if upper.ndim==0 or upper.size==1:
    upper = uld
  else:
    upper = [uld if x>uld else x for x in
             np.array(idata["constant_data"]["upper"])]

  if lower.ndim==0 or lower.size==1:
    lower = lld
  else:
    lower = [lld if x<lld else x for x in
             np.array(idata["constant_data"]["lower"])]

  with model:
    pm.set_data({"lower":lower, "upper":upper}, model)
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)
    pm.compute_log_likelihood(idata, extend_inferencedata=True)

  var_names = ["sigma"]
  with model:
    pm.Deterministic("sigma", model.sigma_minus_offset + pp["sigma_offset"])

    if "nu_minus_offset" in dir(model):
      pm.Deterministic("nu", model.nu_minus_offset + pp["nu_offset"])
      var_names.append("nu")

    pm.sample_posterior_predictive(idata, extend_inferencedata=True,
                                   predictions=True, var_names=var_names)
