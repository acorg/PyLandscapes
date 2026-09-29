#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 16:27:16 2025

@author: avicenna
"""

import numpy as np
import tqdm
from multiprocessing import Pool
from scipy.spatial.distance import cdist
from scipy.stats import t

from .ffit import single_cone_MLE, is_LLD, is_ULD
from .landscape_utils import log2

class BootstrapResult(dict):

  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__

  def __getattr__(self, name):
      try:
          return self[name]
      except KeyError as e:
          raise AttributeError(name) from e

  def CI(self, confidence_level):

    assert 0<confidence_level<1
    assert "bootstrap_params" in self, ("p_MLE" in self or "p_MIN" in self)
    p = self["p_MLE"] if "p_MLE" in self else self["p_MIN"]
    s = 1 - confidence_level

    q0 = np.quantile(self["bootstrap_params"], s/2, axis=0)
    q1 = np.quantile(self["bootstrap_params"], 1-s/2, axis=0)

    CIs =  tuple(np.array([2*p - q1, 2*p - q0]).T)

    return CIs

  def sd(self):

    assert "bootstrap_params" in self, ("p_MLE" in self or "p_MIN" in self)

    p = self["p_MLE"] if "p_MLE" in self else self["p_MIN"]

    dif_params = self["bootstrap_params"] - p[None,:]

    return np.std(dif_params, axis=0)


def parametric(result, ag_coordinates, discrete_step, observable, bounds=None, N=500, seed=None, 
               mle_args=None, LLD=-np.inf, ULD=np.inf, display_progress=True, 
               num_cpus=1):

  '''
  This is a bayesian bootstrap routine to be used with results of
  single_cone_MLE which has been run with use_likelihood=True.

  Supply any argument other than method and priors which has been supplied
  to single_cone_MLE using mle_args.
  
  returns bootstrapped samples from fitted parameters.
  
  '''

  p_MLE = result.x

  if seed is None:
    seed = np.random.SeedSequence().spawn(1)[0]
  rng = np.random.default_rng(seed)

  if mle_args is None:
    mle_args = {}

  mle_args = dict({"p0":p_MLE, "bounds":bounds, "use_likelihood":True,
                   "method":result.method, "priors":result.priors,
                   "discrete_step":discrete_step},
                  **mle_args)


  samples = generate_samples(p_MLE, ag_coordinates, LLD, ULD,
                             log2, N, rng)
  print("not working yet. samples should somehow account for all sera.")
  output = BootstrapResult()
  output["p_MLE"] = p_MLE

  par_args = [[ag_coordinates, observable, mle_args]
              for sample in samples]


  if num_cpus>1:
    with Pool(num_cpus) as p:
      results = list(tqdm.tqdm(p.imap(_par_single_cone_MLE, par_args),
                               desc="Computing Parametric Bootstrap",
                               disable=not display_progress,
                               total=len(samples)))

  else:
    results = [_par_single_cone_MLE(arg) for arg in
               tqdm.tqdm(par_args, desc="Computing Parametric Bootstrap")]

  bootstrap_params = [result.x for result in results if result.success]
  bootstrap_params = np.array(bootstrap_params)
  output["bootstrap_params"] = bootstrap_params

  return output


def nonparametric(result, ag_coordinates, observables, 
                  use_likelihood, bounds=None, N=1000, seed=None,
                  mle_args=None, LLD=None, ULD=None, display_progress=True,
                  num_cpus=1):

  '''
  this is a non parametric bayesian resample routine, can be used both for
  likelihood or non likelihood based landscapes.
  '''

  if seed is None:
    seed = np.random.SeedSequence().spawn(1)[0]
  rng = np.random.default_rng(seed)

  if mle_args is None:
    mle_args = {}

  p_MIN = result.x

  s1,nrepeats = observables.shape
  nAg,ndims = ag_coordinates.shape
  if s1 != nAg:
    raise ValueError('Number of rows of the observables should '
                     'be the same as number of antigens')


  mle_args = dict({"p0":p_MIN, "bounds":bounds, "use_likelihood":use_likelihood,
                   "method":result.method, "priors":result.priors},
                  **mle_args)

  I_LLD = is_LLD(observables)
  I_ULD = is_ULD(observables)

  LLD, ULD = _get_LDM(observables, I_LLD, I_ULD, LLD, ULD)


  output = BootstrapResult()
  output["p_MIN"] = p_MIN

  weights = rng.dirichlet(np.ones(observables.size), size=(N))
  weights = np.reshape(weights, (N, *observables.shape)) * observables.size

  mle_args_par = [dict(mle_args,**{"weights":w}) for w in weights]

  par_args = [[ag_coordinates, observables, mlp]
              for mlp in mle_args_par]

  if num_cpus>1:
    with Pool(num_cpus) as p:
      results = list(tqdm.tqdm(p.imap(_par_single_cone_MLE, par_args),
                               desc="Computing Bayesian Bootstrap",
                               disable=not display_progress,
                               total=len(par_args)))

  else:
    results = [_par_single_cone_MLE(arg) for arg in
               tqdm.tqdm(par_args, desc="Computing Bayesian Bootstrap")]

  bootstrap_params = [result.x for result in results if result.success]
  bootstrap_params = np.array(bootstrap_params)

  output["bootstrap_params"] = bootstrap_params

  return output


def generate_samples(p, ag_coordinates, LLD, ULD,
                     log_fun, size, rng):

  '''
  generate samples from the single cone likelihood using MLE parameters p
  '''

  _,ndims = ag_coordinates.shape

  log_LLD = log_fun(LLD) if LLD>0 else -np.inf
  log_ULD = log_fun(ULD) if ULD>0 else np.inf

  apex_coordinates = np.array(p[0:ndims])[None,:]
  apex_val = p[ndims]
  cone_slope = p[ndims+1]
  biases = p[ndims+2:-2]
  
  if biases.size>3:
    b_sd = np.nanstd(biases)
    b_mean = np.nanmean(biases)
    b_gen = rng.normal(b_mean, b_sd, size)
  else:
    b_gen = rng.choice(biases, size)
    

  ag_distances = cdist(apex_coordinates, ag_coordinates).T.flatten()
  pred_vals = (apex_val - cone_slope*ag_distances)

  samples  =  t.rvs(p[-1], pred_vals, p[-2], random_state=rng,
                    size=[size]+list(pred_vals.shape)) + b_gen[:,None]

  samples = 10*2.0**samples

  samples[np.isinf(samples)] = np.nan # very rarerly outliers in t distribution can produce these

  samples_str = np.ones(samples.shape, dtype=object)*"*"
  samples_str[~np.isnan(samples)] = samples[~np.isnan(samples)]


  I_LLD = samples<log_LLD
  I_ULD = samples>log_ULD
  samples_str[I_LLD] = LLD
  samples_str[I_ULD] = ULD

  return samples_str


def _get_LDM(observables, I_LLD, I_ULD, LLD, ULD):
  '''
  Get upper and lower level of detection matrices
  '''

  if LLD is None:
    _llds = [float(x[1:]) for x in observables[I_LLD]]
    if len(_llds)>0:
      LLD = min(_llds)
    else:
      LLD = 10*2**-10 # beyond this is unrealistic

  if ULD is None:
    _ulds = [float(x[1:]) for x in observables[I_ULD]]
    if len(_ulds)>0:
      ULD = max(_ulds)
    else:
      ULD = 10*2**30 # beyond this is unrealistic

  LLD = np.ones(observables.shape)*LLD
  ULD = np.ones(observables.shape)*ULD

  if np.count_nonzero(I_LLD)>0:
    LLD[I_LLD] = list(map(lambda x: float(x[1:]), observables[I_LLD]))

  if np.count_nonzero(I_ULD)>0:
    ULD[I_ULD] = list(map(lambda x: float(x[1:]), observables[I_ULD]))

  return LLD, ULD


def _par_single_cone_MLE(args):
  '''
  convenience wrapper for parallelization of single_cone_MLE
  '''
  
  return single_cone_MLE(*args[:-1], **args[-1])
