#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 13:35:25 2022

@author: avicenna
"""
import numpy as np
import warnings
import pandas as pd
import tqdm
from multiprocessing import Pool
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from .landscape_utils import log2, is_LLD, is_ULD
from scipy.stats import t

K = 10


def single_cone_MLE(ag_coordinates, observables, discrete_step, p0, bounds=None,
                    use_likelihood=True, weights=None, min_options=None,
                    method=None, priors=None, display=False):

  '''
  This is a frequentist approach where either a distance based cost function
  or a t likelihood cost function is used to obtain point estimates for slope,
  and apex parameters. LLD and ULD values are handled with censored likelihoods
  in the latter case (likelihood) or a smooth decay function in the former (distance). 
  The likelihood also deals with discrete values though the classical cost function can't.

  For confidence intervals, use the functions inside bootstrap.py

  Observables should either be a table with dimensions Nag x Nobs or just Nag 
  or it should be a DataFrame with columns ag_id,sr_id,titer. Former is analogous
  to regular titer tables with single measurements where as latter would be case of 
  long format data with repeats. ag_id represent the order of the antigen given
  in the ag_coordinates so antigen with ag_id=0 has the first row of coordinates
  in ag_coordinates as its coordinates.

  p0 is the vector of initial starting parameters. it should have the following:
  apex coordinates (number of dimensions), apex height (1), slope (1),
  per repeat biases (nrepeats) + if likelihood is used sigma (1), df (1). you
  can create p0 using get_pb function in this module.

  Bounds should be a list of tuples of the form [(p0 lower lim, p0 upper lim),
  ...] with length equal to number of parameters. If they are not supplied,
  the defualt bounds is None for everything and (1, 100) for df, (0.1, 50)
  for sigma and (0.01,100) for slope. you can create bounds using the get_pb
  function in this moduke.

  Parameters fitted are: coordinates, cone height, cone slope, per serum bias,
  df and sigma if likelihood is used.

  If neut titers set discrete_step=0, if HI where titers of the form 30,60 are
  available set it to 0.5 if HI titers where titers are always 20,40,80 etc then
  set it to 1.

  Cost function for non likelihood based approach is mean squared difference
  between observed and predicted titers modulo smooth decay for LLD and ULD function

  If likelihood is used cost is based on a censored T distribution.

  Priors regularize the bias, sigma and df parameters (latter two exists
  only in the likelihood case). As these are given as the standard deviation
  of the normal distribution prior, higher the values the more flexible it is.

  Suggested pipeline:
    - Run the same data set with methods Nelder-Mead, L-BFGS-B, Powell with
      initial conditions and bounds from get_pb and also with more restricted
      bounds.
    - Take the one with the smallest likelihood = result.fun
    - Run with use_likelihood=False with same set of bounds, compare results.
    - Classical, Nelder-Mead and L-BFGS-B tend to give similar results unless
      in pathological cases (many thresholded titres etc) in which classical
      and Nelder-Mead might be more robust, hence the comparison.

  Notes:
    - Base maps are almost always optimized using classical MDS with a cost
    function like the one here. This means that non-likelihood version is
    actually more compatible with such maps. Indeed where as non likelihood
    version generally fits a cone of slope 1 for convalescent sera, likelihood
    seems to fit about 0-0.4 more depending on number of thresholded titers.
    This is probably because censoring will try to fit lower values for many
    thresholded titers hence making landscape decrease faster hence more slope.

  '''


  #some checks and balances
  if min_options is None:
    min_options = {}

  if priors is None:
    priors = {}
    
  priors = dict({"bias_sd":0.5, "bias_mean":0, "sigma_sd":0.25, "sigma_mean":0.5, 
                 "df_sd":3, "df_mean":5}, **priors)

  nAg,ndims = ag_coordinates.shape
  observables_f, I_ag, I_sr = _level_sets(observables, nAg)
  nSr = len(set(I_sr))

  nparams = ndims + nSr + 2
  if use_likelihood:
    nparams += 2

  if weights is None:
    weights = np.ones((nAg, nSr))
  elif weights is not None:
    assert weights.shape == (nAg, nSr), "weights should have shape nAg x nSr"
    assert np.all(weights>0), "weights should be positive"

  if bounds is None:
    bounds = [(None, None) for _ in range(ndims+1)] + [(0.01, 100)] +\
      [(-1e-6,1e-6)] + [(None, None) for p in range(nSr-1)] +\
        [(0.1, 50), (1, 100)]*int(use_likelihood)
    # unless supplied first serum basis is set to be restricted to 0 to
    # alleviate non-identifiability which can shift all means up and biases down

  assert p0.ndim==1 and p0.size == nparams, f"p0 should be a vector of size {nparams}"

  #preparation: transform observables, setup minimizer options
  if method is None:
    method= "L-BFGS-B"
    # if in discrete cases L-BFGS-B stops prematurely (i.e after one step) then
    # try higher eps values or Nelder-Mead or Powell. Note that scipy
    # implementation of Nelder-Mead is an improved version with adaptive steps
    # and in my experience generally gives very close results with L-BFGS-B
    # for this problem when both terminate succesfully.


  log_observables = log2(observables_f)

  if method in ["Nelder-Mead","Powell"]:
    if "maxiter" not in min_options:
      min_options["maxiter"] = nparams*3000
    if "maxfev" not in min_options:
      min_options["maxfev"] = nparams*3000

  # locations of lower than higher than limits of detection
  I_LLD = is_LLD(observables_f)
  I_ULD = is_ULD(observables_f)

  # run
  result = minimize(_cost_single_cone, p0, bounds=bounds, method=method,
                    args = (ag_coordinates, log_observables, I_LLD, I_ULD,
                            I_ag, I_sr, discrete_step, use_likelihood, weights, priors),
                    options=min_options)

  # collect some data into results and return

  # calculate mean error between predicted and target at best parameters
  # set priors to infinity sigma so they dont contribute
  mean_error=\
    _cost_single_cone(np.array(result.x)[:ndims + nSr + 2], ag_coordinates,
                      log_observables, I_LLD, I_ULD, I_ag, I_sr, discrete_step,
                      False, weights, priors={"bias_sd":np.inf, "bias_mean":0})


  result.mean_error = mean_error
  result.p0 = p0
  result.bounds = bounds
  result.priors = priors
  result.method = method
  result.observables = observables
  result.ag_coordinates = ag_coordinates
  result.discrete_step = discrete_step
  result.weights = weights
  result.use_likelihood = use_likelihood
  result.prediction = _predict(result.x, ag_coordinates, ndims, use_likelihood)


  if not result.success and display:
    print("Warning: minimizer did not converge. Try changing method, "
          "increasing maxiter, maxfev, ftol or xtol.")

  return result


class MultiStartResult(dict):

  '''
  dict subclass allowing attribute access, used as the return type of
  parallel_single_cone_MLE. Mirrors the BootstrapResult pattern in
  bootstrap.py.
  '''

  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__

  def __getattr__(self, name):
      try:
          return self[name]
      except KeyError as e:
          raise AttributeError(name) from e


def parallel_single_cone_MLE(ag_coordinates, observables, discrete_step, p0,
                             bounds=None, use_likelihood=True, weights=None,
                             min_options=None, method=None, priors=None,
                             nprocesses=1, ninitial=10, seed=None,
                             display_progress=True):

  '''
  Multi-start parallel wrapper around single_cone_MLE.

  Runs single_cone_MLE from ninitial different starting points, optionally
  distributed over nprocesses worker processes, and reports the best fit
  found. This automates the "Suggested pipeline" described in
  single_cone_MLE's docstring, which recommends comparing result.fun across
  multiple runs/initial conditions to guard against local minima.

  Accepts exactly the same arguments as single_cone_MLE (see its docstring
  for details on ag_coordinates, observables, discrete_step, p0, bounds,
  use_likelihood, weights, min_options, method, priors), plus:

    nprocesses: number of worker processes to use (via
      multiprocessing.Pool). If 1 (default), runs are computed sequentially
      in the current process.

    ninitial: number of starting points to optimize from (default 10). The
      supplied p0 is always included as one of them. The remaining
      ninitial-1 points are sampled per-parameter as follows:
        - if bounds are provided and both the lower and upper limit for
          that parameter are finite (e.g. (0.01, 100)), it is sampled
          uniformly within those bounds.
        - otherwise (bounds not provided at all, or a bound with one or
          both sides None, e.g. (0, None)) it is sampled from a normal
          distribution centered on p0 with standard deviation on roughly
          the same order of magnitude as p0. If a finite lower and/or
          upper limit is present (as in (0, None)) the sample is then
          clipped to enforce it.

    seed: seed for the random generator used to sample initial conditions.
      If None (default) a fresh one is drawn, see numpy.random.SeedSequence.

    display_progress: whether to show a tqdm progress bar (default True).

  Returns a MultiStartResult with:
    best: the OptimizeResult (as returned by single_cone_MLE) with the
      lowest .fun among runs with result.success True; if none succeeded,
      the lowest .fun among all runs.
    all_results: list of every OptimizeResult, one per starting point, in
      the same order as p0_samples.
    p0_samples: array of shape (ninitial, nparams) with the starting points
      that were used.
  '''

  assert ninitial>=1, "ninitial should be at least 1"
  assert nprocesses>=1, "nprocesses should be at least 1"

  if seed is None:
    seed = np.random.SeedSequence().spawn(1)[0]
  rng = np.random.default_rng(seed)

  p0_samples = _sample_initial_conditions(p0, bounds, ninitial, rng)

  par_args = [(ag_coordinates, observables, discrete_step, p0_i, bounds,
               use_likelihood, weights, min_options, method, priors, display_progress)
             for p0_i in p0_samples]

  if nprocesses>1:
    with Pool(nprocesses) as pool:
      results = list(tqdm.tqdm(pool.imap(_multistart_worker, par_args),
                               desc="Computing Multi-start MLE",
                               disable=not display_progress,
                               total=len(par_args)))
  else:
    results = [_multistart_worker(arg) for arg in
              tqdm.tqdm(par_args, desc="Computing Multi-start MLE",
                        disable=not display_progress)]

  converged = [r for r in results if r.success]
  candidates = converged if len(converged)>0 else results
  best = min(candidates, key=lambda r: r.fun)

  output = MultiStartResult()
  output["best"] = best
  output["all_results"] = results
  output["p0_samples"] = p0_samples

  return output


def get_pb(observables, ag_coordinates, use_likelihood,
           coord_flexibility=0.25, height_flexibility=0.5,
           bias_flexibility=1):

  '''
  A convenience function for getting bounds and initial p for a single
  cone based approach. places the cone apex at the antigen whose mean
  titre is highest and allows it to move as much as flexibility values in coord
  and height. Slope has initial value of 1 with boundary (0.05,100). per
  repeat biases are point estimated to get the initial value and then
  a boundary of (p-bias_flexibility,p+bias_flexibility) is set for these values.

  If likelihood is used, degrees of freedom has the boundary (2, 40) with
  initial value of 5 and sigma has the boundary (0.1, 10)

  Return p0 has the initial values in the following order:

  apex coordinates (number of dimensions), apex height (1), slope (1),
  per repeat biases (nrepeats) + if likelihood is used sigma (1), df (1).

  '''

  nAg,ndims = ag_coordinates.shape
  observables_f, I_ag, I_sr = _level_sets(observables, nAg)
  nrepeats = len(set(I_sr))

  log_observables = log2(observables_f)

  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    apex_val = [np.nanmax(log_observables[I_sr==i])
                for i in range(nrepeats)]

    # for each group find the index of antigen with highest
    # titer, note that this is index within group
    apex_ind = [np.nanargmax(log_observables[I_sr==i])
                for i in range(nrepeats)]

    # convert local index to global index
    apex_ind = [np.argwhere(I_sr==i1).flatten()[i0] for i0,i1 in
                zip(apex_ind,range(nrepeats))]

    # convert global index to antigen index
    ag_ind = [int(I_ag[i]) for i in apex_ind]


  apex_coords = list(ag_coordinates[ag_ind,:].mean(axis=0))
  apex_val = np.mean(apex_val)

  mean = np.nanmean(log_observables)
  bias =  [np.nanmean(log_observables[I_sr==i]) - mean
           for i in range(nrepeats)]

  p0 = apex_coords + [apex_val, 1] + bias

  bounds = [(p-coord_flexibility, p+coord_flexibility) for p in p0[:ndims]] +\
    [(apex_val-height_flexibility, apex_val+height_flexibility)] + [(0.1, 2)] +\
    [(p-bias_flexibility, p+bias_flexibility) for p in p0[ndims+2:ndims+2+nrepeats]]

  if use_likelihood:
    p0 += [1, 5]
    bounds += [(0.1, 5), (1, 20)]

  return np.array(p0), bounds


def _LLD_stress(x, y, discrete_step):

  '''
  x is target, y is predicted
  '''
  return (x-discrete_step-y)**2/(1+np.exp(K*(x-discrete_step-y)))

def _ULD_stress(x, y, discrete_step):
  '''
  x is target, y is predicted
  '''
  return (x+discrete_step-y)**2/(1+np.exp(-K*(x+discrete_step-y)))


def _stress_classical(predicted, target, I_LLD, I_ULD, I_ag, I_sr, discrete_step,
                      weights):

  '''
  unlikes classical antigenic cartography, if data is discrete
  a correction of +0.5 is added to reflect the fact that a
  measured log value of say 4 has equal possibility to be anywhere
  between 4-5 so the point estimate is 4.5

  similarly for LLD and ULD components a correction of -0.5 instead
  of the usual -1 in cartography is used.
  '''

  stress = []

  I_D = (~I_LLD) & (~I_ULD)

  predicted_f = predicted[I_ag, I_sr]
  weights_f = weights[I_ag, I_sr]

  stress += list(weights_f[I_D]*(target[I_D] - predicted_f[I_D])**2)

  if np.count_nonzero(I_LLD)>0:
    stress += list(weights_f[I_LLD]*_LLD_stress(target[I_LLD], predicted_f[I_LLD],
                                                discrete_step))

  if np.count_nonzero(I_ULD)>0:
    stress += list(weights_f[I_ULD]*_ULD_stress(target[I_ULD], predicted_f[I_ULD],
                                                discrete_step))

  return np.nanmean(stress)


def _stress_likelihood(predicted, target, I_LLD, I_ULD, I_ag, I_sr,
                       discrete_step, sigma, df, weights):

  l = []
  I_D = (~I_LLD) & (~I_ULD)
  w = []

  predicted_f = predicted[I_ag,I_sr]
  weights_f = weights[I_ag,I_sr]

  if discrete_step>0:
    l += list((t.cdf(target[I_D] + discrete_step, df, predicted_f[I_D], sigma) -\
               t.cdf(target[I_D], df, predicted_f[I_D], sigma)))
  else:
    l += list(t.pdf(target[I_D], df, predicted_f[I_D], sigma))

  w += list(weights_f[I_D])

  if np.count_nonzero(I_LLD)>0:
    l += list(t.cdf(target[I_LLD], df, predicted_f[I_LLD], sigma))
    w += list(weights_f[I_LLD])

  if np.count_nonzero(I_ULD)>0:
    l += list((1-t.cdf(target[I_ULD], df, predicted_f[I_ULD], sigma)))
    w += list(weights_f[I_ULD])


  l = np.array(l)
  
  if np.count_nonzero(l>0)>0:
    l[l<=0] = np.min(l[l>0])
  else:
    l[l<=0] = 1e-300

  l = -np.log(np.array(l))
  w = np.array(w)

  return np.nansum(w*l)


def _predict(p, ag_coordinates, ndims, use_likelihood):

  apex_coordinates = np.array(p[0:ndims])[None,:]
  apex_val = p[ndims]
  cone_slope = p[ndims+1]

  if use_likelihood:
    biases = p[ndims+2:-2]
  else:
    biases = p[ndims+2:]

  ag_distances = cdist(apex_coordinates, ag_coordinates).T.flatten()

  return (apex_val - cone_slope*ag_distances)[:,None] + biases[None,:]


def _cost_single_cone(p, ag_coordinates, observables, I_LLD, I_ULD,
                      I_ag, I_sr, discrete_step, use_likelihood, weights,
                      priors):

  _,ndims = ag_coordinates.shape

  if use_likelihood:
    biases = p[ndims+2:-2]
  else:
    biases = p[ndims+2:]

  pred_vals = _predict(p, ag_coordinates, ndims, use_likelihood)

  if use_likelihood:
    stress = _stress_likelihood(pred_vals, observables, I_LLD, I_ULD,
                                I_ag, I_sr, discrete_step, p[-2], p[-1],
                                weights)

    # prior on biases equivalent to normal with mean sigma given by priors dict
    for key,ind in zip(["sigma","df"], [-2,-1]):
      stress += (1/priors[f"{key}_sd"]**2)*(p[ind]-priors[f"{key}_mean"])**2 
    

  else:
    stress = _stress_classical(pred_vals, observables, I_LLD, I_ULD, I_ag, I_sr,
                               discrete_step, weights)

  stress += 1/priors["bias_sd"]**2*np.nansum((biases-priors["bias_mean"])**2)


  return stress


def _level_sets(observables, nag):

  if isinstance(observables, np.ndarray):

    if observables.ndim==1:
      observables = observables[:, None]
    elif observables.ndim>2:
      raise ValueError("Observables can only be two dimensional")

    s1,nrepeats = observables.shape

    if s1 != nag:
      raise ValueError('Number of rows of the observables should '
                       'be the same as number of antigens')

    nsr = observables.shape[1]

    I_ag = np.repeat(range(nag),nsr)
    I_sr = np.array([i for _ in range(nag) for i in range(nsr)])
    values = observables.flatten()

  elif isinstance(observables, pd.DataFrame):
    assert "ag_id" in observables
    assert all(isinstance(x,int) for x in observables["ag_id"])
    assert "sr_id" in observables
    assert all(isinstance(x,int) for x in observables["sr_id"])
    assert "titer" in observables

    I_ag = observables["ag_id"].values
    I_sr = observables["sr_id"].values
    values = observables["titer"].values

  else:
    raise ValueError("observables must be an array or DataFrame")

  return values, I_ag, I_sr


def generate_prediction(params, ag_coordinates, use_likelihood):

  vals = _predict(params, ag_coordinates, ag_coordinates.shape[1],
                  use_likelihood).mean(axis=1)

  return vals


def _sample_initial_conditions(p0, bounds, ninitial, rng):

  '''
  Generate ninitial starting points for parallel_single_cone_MLE's
  multi-start optimization.

  For each parameter: if bounds are supplied and both the lower and upper
  limit are finite, the corresponding column is sampled uniformly within
  those bounds. Otherwise (bounds not supplied, or a bound with one or
  both sides None) the column is sampled from a normal distribution
  centered on p0 with a standard deviation on roughly the same order of
  magnitude as the p0 value, then clipped to respect whichever side of
  the bound (if any) is finite.

  The first row is always set to p0 itself.
  '''

  p0 = np.asarray(p0, dtype=float)
  nparams = p0.size

  abs_p0 = np.abs(p0)
  with np.errstate(divide="ignore"):
    sd = np.where(abs_p0>0, 10.0**np.round(np.log10(abs_p0)), 0.1)
  sd = np.maximum(sd, 0.1)

  if bounds is not None:
    lo = np.array([-np.inf if b[0] is None else b[0] for b in bounds])
    hi = np.array([np.inf if b[1] is None else b[1] for b in bounds])
  else:
    lo = np.full(nparams, -np.inf)
    hi = np.full(nparams, np.inf)

  fully_bounded = np.isfinite(lo) & np.isfinite(hi)

  samples = rng.normal(p0, sd, size=(ninitial, nparams))

  if np.any(fully_bounded):
    samples[:, fully_bounded] =\
      rng.uniform(lo[fully_bounded], hi[fully_bounded],
                  size=(ninitial, np.count_nonzero(fully_bounded)))

  samples = np.clip(samples, lo, hi)
  samples[0, :] = p0

  return samples


def _multistart_worker(args):

  '''
  Picklable worker for parallel_single_cone_MLE; multiprocessing.Pool.imap
  requires a function taking a single argument, so the per-run arguments
  are packed into one tuple.

  min_options is shallow-copied since single_cone_MLE mutates it in place
  (e.g. adding "maxiter"/"maxfev" keys), and the same min_options object
  would otherwise be shared and mutated across all ninitial runs when
  nprocesses=1.
  '''

  (ag_coordinates, observables, discrete_step, p0, bounds, use_likelihood,
   weights, min_options, method, priors, display) = args

  min_options = None if min_options is None else dict(min_options)

  return single_cone_MLE(ag_coordinates, observables, discrete_step, p0,
                         bounds=bounds, use_likelihood=use_likelihood,
                         weights=weights, min_options=min_options,
                         method=method, priors=priors, display=display)
  
  
