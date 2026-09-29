#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Hessian-based (Laplace/asymptotic-normal) confidence intervals for
single_cone_MLE fits, and Monte Carlo propagation of the resulting
parameter covariance to the predicted cone.

This is a complementary alternative to the resampling-based CIs in
bootstrap.py, only defined for use_likelihood=True fits: only then is
_cost_single_cone an actual (regularized) negative log-likelihood, so
inverting its Hessian at the MLE is meaningful. 
For use_likelihood=False fits, use bootstrap.py instead.
'''

import numpy as np
import numdifftools as nd
import tqdm
from scipy.stats import norm
from scipy.spatial.distance import cdist

from .ffit import _level_sets, _cost_single_cone, generate_prediction
from .landscape_utils import log2, is_LLD, is_ULD

# Safety margin for the apex-coordinate step cap in _safe_coordinate_hessian
# (see _safe_coordinate_hessian docstring).
_COORD_STEP_SAFETY_FACTOR = 0.5


class ParameterCovarianceResult(dict):

  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__

  def __getattr__(self, name):
      try:
          return self[name]
      except KeyError as e:
          raise AttributeError(name) from e

  def CI(self, confidence_level):

    '''
    Wald normal confidence interval p +/- z*se. Pinned parameters (see
    parameter_covariance) have se=0, so their CI collapses to (p, p).
    '''

    assert 0<confidence_level<1
    z = norm.ppf(0.5 + confidence_level/2)

    return self["p"] - z*self["se"], self["p"] + z*self["se"]


class PredictionSampleResult(dict):

  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__

  def __getattr__(self, name):
      try:
          return self[name]
      except KeyError as e:
          raise AttributeError(name) from e

  def radius_CI(self, confidence_level, dims, center, add_med=False):
    '''
    CI for radius to center.
    '''
    assert 0<confidence_level<1
    s = 1 - confidence_level
    
    coords = self["param_samples"][:,:dims]
    
    ds = np.sqrt(np.sum((coords - center[None,:])**2, axis=1))
    
    if add_med:
      return (np.quantile(ds, s/2), np.quantile(ds, 0.5), np.quantile(ds, 1-s/2))
    else:
      return (np.quantile(ds, s/2), np.quantile(ds, 1-s/2))


  def cover_CI(self, confidence_level, lld, add_med=False):
    '''
    CI for proportion of antigens with non-thresholded titers
    '''
    
    assert 0<confidence_level<1
    s = 1 - confidence_level
    
    p=(self["samples"]>=lld).sum(axis=1)/self["samples"].shape[1]
    
    if add_med:
      return (np.quantile(p, s/2), np.quantile(p, 0.5), 
              np.quantile(p, 1-s/2))
    else:
      return (np.quantile(p, s/2), np.quantile(p, 1-s/2))
    

  def pred_CI(self, confidence_level, add_med=False):

    '''
    predicted titers CI
    '''

    assert 0<confidence_level<1
    s = 1 - confidence_level

    if add_med:
      return (np.quantile(self["samples"], s/2, axis=0),
              np.quantile(self["samples"], 0.5, axis=0),
              np.quantile(self["samples"], 1-s/2, axis=0))
    else:
      return (np.quantile(self["samples"], s/2, axis=0),
              np.quantile(self["samples"], 1-s/2, axis=0))

  def sd(self):
    return np.std(self["samples"], axis=0)


def _bound_distance(xi, lo, hi):

  ds = [abs(xi-b) for b in (lo, hi) if b is not None]

  return min(ds) if len(ds)>0 else np.inf


def _nd_finest_step(hessian_obj, xi):

  '''
  The smallest ("finest") step numdifftools.Hessian's own adaptive ladder
  would use for a single scalar coordinate value xi under hessian_obj's
  configuration (step=None). Used by _safe_coordinate_hessian as the step
  for free parameters that are NOT apex coordinates, so the recomputed
  cross terms use the same per-dimension magnitude numdifftools' own
  default would have used, rather than an arbitrary hand-picked constant.
  '''

  gen = hessian_obj.step.step_generator_function(
    np.asarray([xi]), method=hessian_obj.method, n=hessian_obj.n,
    order=hessian_obj.order)

  return float(np.abs(np.array(list(gen()))).min())


def _safe_coordinate_hessian(free_cost, x0_free, ndims_free_positions,
                             nearest_ag_distance, method):

  '''
  Recompute the free-parameter Hessian using an explicit per-dimension
  step: a capped, safe step for the apex-coordinate free dimensions, and
  numdifftools' own finest default-ladder step (see _nd_finest_step) for
  every other free dimension.

  Why this is needed: _predict's apex-to-antigen distance term has a
  genuine kink (discontinuous first derivative) wherever the
  apex coincides with an antigen. numdifftools.Hessian's default (step=
  None) adaptive ladder trials steps up to ~2.0 in magnitude and this large step 
  routinely crosses a nearby antigen regardless of exactly how close the fitted 
  apex happens to land.

  This function is only invoked for method='central' and only when there is
  at least one free apex-coordinate parameter; the safe step is always
  applied to those dimensions (not just when d looks suspiciously small)
  because the default ladder's ~2.0 max step is large enough to be a risk
  against ordinary antigenic-map-scale spacing generally, not only in
  close-call cases.

  Note: if the fitted apex coincides exactly with an antigen
  (nearest_ag_distance == 0), the second derivative genuinely does not
  exist in that direction; the capped step falls back to a floor (1e-8)
  rather than raising, but the resulting Hessian entry is not meaningful
  there. This is expected to be caught downstream by the positive-
  definiteness check in parameter_covariance rather than handled here.
  '''

  assert method=="central"

  n_free = x0_free.size
  default_hessian_obj = nd.Hessian(free_cost, step=None, method=method)

  step_array = np.array([_nd_finest_step(default_hessian_obj, x0_free[k])
                         for k in range(n_free)])

  step_nom = np.maximum(np.log(np.e + np.abs(x0_free[ndims_free_positions])), 1.0)
  step_coord = _COORD_STEP_SAFETY_FACTOR*nearest_ag_distance / (2.0*step_nom)
  step_array[ndims_free_positions] = np.maximum(step_coord, 1e-8)

  H_safe = nd.Hessian(free_cost, step=step_array, method=method)(x0_free)

  return 0.5*(H_safe + H_safe.T)


def _resolve_fit_arg(name, explicit, result):

  '''
  Resolve one of ag_coordinates/discrete_step/weights either from an
  explicit argument or from the corresponding attribute single_cone_MLE
  stores on its result (added specifically so parameter_covariance can
  detect a caller accidentally supplying something that doesn't match
  what the fit actually used).
  '''

  stored = getattr(result, name, None)

  if explicit is None and stored is None:
    raise ValueError(
      f"{name} was not found on result.{name} and was not supplied "
      f"explicitly. Either re-run the fit with the current single_cone_MLE "
      f"(which stores this automatically) or pass {name}= explicitly.")

  if explicit is None:
    return stored

  if stored is None:
    return explicit

  explicit_arr = np.asarray(explicit)
  stored_arr = np.asarray(stored)

  if explicit_arr.shape != stored_arr.shape or not np.array_equal(explicit_arr, stored_arr):
    detail = f"shapes {explicit_arr.shape} vs {stored_arr.shape}"
    if explicit_arr.shape == stored_arr.shape:
      detail += f", max abs diff {np.max(np.abs(explicit_arr-stored_arr)):.3g}"
    raise ValueError(
      f"Supplied {name} does not match result.{name} from the original fit "
      f"({detail}). parameter_covariance must be given the exact same "
      f"{name} that produced this result.")

  return explicit


def parameter_covariance(result, ag_coordinates=None, discrete_step=None,
                         weights=None, bound_tol=1e-4, force_pinned=None,
                         psd_rel_tol=1e-8, step=None, method="central"):

  '''
  covariance matrix for a single_cone_MLE result obtained with use_likelihood=True.

  Computes the Hessian (via numdifftools.Hessian, adaptive Richardson
  extrapolation) of the exact regularized negative-log-likelihood cost
  function actually minimized (_cost_single_cone, including the
  bias/sigma/df prior penalties) at result.x, and inverts it to obtain
  the asymptotic covariance matrix Cov ~= H^-1 of the MLE.

  Parameters whose fitted value sits within bound_tol of a finite bound
  edge (e.g. a bias pinned to (-1e-6, 1e-6) for identifiability, or a
  coordinate manually fixed via a tight bound) are "pinned". 
  These parameters are excluded from the block that gets inverted and 
  reported with exactly zero variance, fixed at their point
  estimate. force_pinned lets you additionally exclude parameter indices
  the automatic bound_tol check didn't catch (e.g. a parameter that
  converged suspiciously close to, but not within tolerance of, a bound
  -- see bound_distance in the returned result).

  Raises ValueError if result was not produced with use_likelihood=True
  -- the classical cost is not a (regularized) negative log-likelihood,
  so inverting its Hessian has no asymptotic-variance interpretation;
  use bootstrap.py for that case instead.

  ag_coordinates, discrete_step, weights default to the values
  single_cone_MLE stored on result; supplying them explicitly is only
  required for a result predating that (result.ag_coordinates etc.
  missing). If both are available they are cross-checked and a mismatch
  raises loudly, since a silently-wrong Hessian is otherwise
  undetectable.

  Raises ValueError if every parameter is pinned and raises 
  numpy.linalg.LinAlgError if the free-parameter Hessian is not
  (numerically) positive definite after symmetrization -- this can for 
  instance happen when the optimizer converges to a near-flat direction 

  Returns a ParameterCovarianceResult with:
    p: result.x
    cov: (nparams, nparams) covariance matrix, zero rows/cols for pinned
      params
    se: (nparams,) standard errors (sqrt of diagonal), 0 for pinned
      params
    hessian: the full (nparams, nparams) Hessian, embedded with zero
      rows/cols for pinned params
    hessian_free, eigenvalues_free: the (n_free, n_free) Hessian that was
      actually inverted, and its eigenvalues (used for diagnostics)
    pinned, free: boolean masks of length nparams
    bound_distance: (nparams,) distance from p to the nearest finite
      bound edge (inf where neither side is finite)
    bound_tol, force_pinned: as supplied
    ag_coordinates, discrete_step, weights, nparams, ndims, nSr: as
      resolved, kept so sample_predictions doesn't need to re-derive or
      re-validate them
    nearest_ag_distance: distance from the fitted apex to its nearest
      antigen, or None if there was no free apex-coordinate parameter to
      check (used for diagnostics).
    coordinate_step_override: True if the apex-coordinate rows/columns of
      the Hessian were recomputed with a capped step instead of
      numdifftools' default adaptive ladder (see _safe_coordinate_hessian
      for why -- the default ladder's steps are large enough to
      routinely cross the kink in the apex-to-antigen distance term at
      ordinary antigenic-map scales). False when step was given
      explicitly or method != "central", in which case that safeguard is
      not applied.
  '''

  if not getattr(result, "use_likelihood", False):
    if not hasattr(result, "use_likelihood"):
      raise ValueError(
        "result has no 'use_likelihood' attribute -- re-run the fit with "
        "the current single_cone_MLE, which stores this automatically.")
    raise ValueError(
      "parameter_covariance only supports fits produced with "
      "use_likelihood=True; for use_likelihood=False fits, use "
      "bootstrap.py instead.")

  ag_coordinates = _resolve_fit_arg("ag_coordinates", ag_coordinates, result)
  discrete_step = _resolve_fit_arg("discrete_step", discrete_step, result)
  weights = _resolve_fit_arg("weights", weights, result)

  ag_coordinates = np.asarray(ag_coordinates, dtype=float)
  nAg, ndims = ag_coordinates.shape

  observables_f, I_ag, I_sr = _level_sets(result.observables, nAg)
  nSr = len(set(I_sr))
  nparams = ndims + nSr + 4

  x0 = np.asarray(result.x, dtype=float)
  if x0.size != nparams:
    raise ValueError(
      f"result.x has size {x0.size}, expected {nparams} = ndims({ndims}) + "
      f"nSr({nSr}) + 4 for a use_likelihood=True fit. This usually means "
      f"the supplied ag_coordinates/observables don't match what produced "
      f"this result.")

  log_observables = log2(observables_f)
  I_LLD = is_LLD(observables_f)
  I_ULD = is_ULD(observables_f)

  bounds = result.bounds

  bound_distance = np.array([_bound_distance(x0[i], bounds[i][0], bounds[i][1])
                             for i in range(nparams)])

  pinned = bound_distance <= bound_tol
  if force_pinned is not None:
    pinned = pinned.copy()
    pinned[np.asarray(force_pinned, dtype=int)] = True
  free = ~pinned

  if not np.any(free):
    raise ValueError(
      "Every parameter is pinned (at a bound and/or via force_pinned); a "
      "covariance matrix that is zero everywhere is not a meaningful "
      "result. Check bound_tol/force_pinned.")

  # only non-pinned coordinates are allowed to move by setting the pinned
  # ones to be identical with x0.
  def _free_cost(x_free):
    x_full = x0.copy()
    x_full[free] = x_free
    return _cost_single_cone(x_full, ag_coordinates, log_observables, I_LLD,
                             I_ULD, I_ag, I_sr, discrete_step, True, weights,
                             result.priors)

  H = nd.Hessian(_free_cost, step=step, method=method)(x0[free])
  H = 0.5*(H + H.T)

  nearest_ag_distance = None
  coordinate_step_override = False
  coord_idx_full = np.arange(ndims)
  coord_free_full = coord_idx_full[free[coord_idx_full]]
  
  # hessian in coordinate direction may need to be recomputed because
  # of the singularity occuring at apex=antigen_coordinate conditions
  if step is None and method=="central" and coord_free_full.size>0:
    free_idx_full = np.flatnonzero(free)
    pos_of = {full_i: k for k, full_i in enumerate(free_idx_full)}
    coord_free_positions = np.array([pos_of[i] for i in coord_free_full])

    nearest_ag_distance = cdist(x0[None, :ndims], ag_coordinates).min()
    coordinate_step_override = True

    H_safe = _safe_coordinate_hessian(_free_cost, x0[free], coord_free_positions,
                                      nearest_ag_distance, method)
    H[coord_free_positions, :] = H_safe[coord_free_positions, :]
    H[:, coord_free_positions] = H_safe[:, coord_free_positions]

  eigvals = np.linalg.eigvalsh(H)
  if eigvals.min() <= psd_rel_tol*eigvals.max():
    ratio = eigvals.min()/eigvals.max() if eigvals.max()!=0 else float("nan")
    raise np.linalg.LinAlgError(
      f"Free-parameter Hessian is not (numerically) positive definite "
      f"(min/max eigenvalue ratio={ratio:.3g}); Inspect "
      f"bound_distance/free/pinned, consider re-fitting with "
      f"parallel_single_cone_MLE for a better optimum, or use "
      f"force_pinned= to exclude the bad direction.")

  Cov_free = np.linalg.inv(H)
  se_free = np.sqrt(np.diag(Cov_free))

  cov = np.zeros((nparams, nparams))
  cov[np.ix_(free, free)] = Cov_free
  se = np.zeros(nparams)
  se[free] = se_free
  hessian = np.zeros((nparams, nparams))
  hessian[np.ix_(free, free)] = H

  output = ParameterCovarianceResult()
  output["p"] = x0
  output["cov"] = cov
  output["se"] = se
  output["hessian"] = hessian
  output["hessian_free"] = H
  output["eigenvalues_free"] = eigvals
  output["pinned"] = pinned
  output["free"] = free
  output["bound_distance"] = bound_distance
  output["bound_tol"] = bound_tol
  output["force_pinned"] = force_pinned
  output["ag_coordinates"] = ag_coordinates
  output["discrete_step"] = discrete_step
  output["weights"] = weights
  output["nparams"] = nparams
  output["ndims"] = ndims
  output["nSr"] = nSr
  output["nearest_ag_distance"] = nearest_ag_distance
  output["coordinate_step_override"] = coordinate_step_override

  return output


def sample_predictions(result, param_cov, n_samples, query_coordinates=None,
                       seed=None, display_progress=True):

  '''
  Monte Carlo propagation of parameter uncertainty to the predicted cone.

  Draws n_samples parameter vectors from N(result.x, param_cov.cov) --
  pinned parameters (see parameter_covariance) are always held fixed at
  their point estimate -- and evaluates generate_prediction at each
  draw.

  query_coordinates defaults to the ag_coordinates the fit was run on
  (giving a CI band per antigen), but can be any (N, ndims)
  array of coordinates -- e.g. a fine grid -- to get a spatial
  confidence band anywhere in the landscape.

  Returns a PredictionSampleResult with:
    samples: (n_samples, N) predicted values at query_coordinates for
      every draw
    param_samples: (n_samples, nparams) the drawn parameter vectors
    query_coordinates, mean, median, seed, n_samples
  '''

  assert n_samples>=1, "n_samples should be at least 1"

  x0 = np.asarray(result.x, dtype=float)
  if not np.array_equal(x0, np.asarray(param_cov.p, dtype=float)):
    raise ValueError(
      "result.x does not match param_cov.p -- param_cov must come from "
      "parameter_covariance(result, ...) for this exact result.")

  free = param_cov.free
  ndims = param_cov.ndims

  if query_coordinates is None:
    qc = param_cov.ag_coordinates
  else:
    qc = np.asarray(query_coordinates, dtype=float)
    if qc.shape[1] != ndims:
      raise ValueError(f"query_coordinates has {qc.shape[1]} columns, "
                       f"expected {ndims}")

  if seed is None:
    seed = np.random.SeedSequence().spawn(1)[0]
  rng = np.random.default_rng(seed)

  samples_free = rng.multivariate_normal(
    x0[free], param_cov.cov[np.ix_(free, free)], size=n_samples,
    check_valid="raise")

  full_samples = np.tile(x0, (n_samples, 1))
  full_samples[:, free] = samples_free

  samples = np.empty((n_samples, qc.shape[0]))
  for i in tqdm.tqdm(range(n_samples), desc="Sampling predictions",
                     disable=not display_progress):
    samples[i] = generate_prediction(full_samples[i], qc, True)

  output = PredictionSampleResult()
  output["samples"] = samples
  output["param_samples"] = full_samples
  output["query_coordinates"] = qc
  output["mean"] = samples.mean(axis=0)
  output["median"] = np.median(samples, axis=0)
  output["seed"] = seed
  output["n_samples"] = n_samples

  return output
