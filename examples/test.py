#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 09:25:47 2025

@author: avicenna
"""

import PyRacmacs as pr
import numpy as np
import itertools as it
import warnings
import pickle
from scipy.stats import t
from PyLandscapes import fit


def info_print(result_lh, result_classical, real_slope, real_height):

  print(f"real slope: {real_slope:.2f}, real height: {real_height:.2f}")
  print(f"likelihood slope: {result_lh[4]:.2f}, likelihood height: {result_lh[3]:.2f}")
  print(f"classical slope: {result_classical[4]:.2f}, classical height: {result_classical[3]:.2f}")
  print("")

seed = 0
rng = np.random.default_rng(seed)
slopes = [1.5, 1, 0.5, 0.1]
heights = [7, 4, 2]
ag_inds = []
sigma = 1
df = 3

LLD = 0
ULD = 9

nrepeats = 4
acmap = pr.read_racmap("map_H5.ace")
ag_inds = [ind for ind, ag in enumerate(acmap.ag_names) if "/72/2004" in ag
           or "R60" in ag]
ag_coordinates = acmap.ag_coordinates
nag,ndims = ag_coordinates.shape

for ag_ind, height, slope in it.product(ag_inds, heights, slopes):

  center = ag_coordinates[ag_ind:ag_ind+1,:]
  dists = np.linalg.norm(center-ag_coordinates, axis=1)

  true = height - slope*dists

  log_observables = t.rvs(df, true, sigma, random_state=rng, size=(nrepeats, nag))
  biases = rng.normal(0, 1, size=nrepeats)[:,None]

  log_observables = np.floor(log_observables+biases).T



  observables = (10*2**log_observables).astype(int).astype(str)
  observables[log_observables<LLD] = f"<{int(10*2**LLD)}"
  observables[log_observables>ULD] = f">{int(10*2**ULD)}"

  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)

    p0,bounds = fit.get_pb(log_observables, ag_coordinates, True)

    result_lh1=\
      fit.single_cone_MLE(ag_coordinates, observables, discrete_step=1, p0=p0,
                          bounds=bounds, use_likelihood=True)

    result_lh2=\
      fit.single_cone_MLE(ag_coordinates, observables, discrete_step=1, p0=p0,
                          bounds=bounds, use_likelihood=True, method="Nelder-Mead")

    if result_lh1.fun<result_lh2.fun:
      result_lh = result_lh1
    else:
      result_lh = result_lh2

    p0,bounds = fit.get_pb(log_observables, ag_coordinates, False)

    result_classical=\
      fit.single_cone_MLE(ag_coordinates, observables, discrete_step=1, p0=p0,
                          bounds=bounds, use_likelihood=False)

    # if result_lh.x[4]>2 and slope==0.1:
    #   breakpoint()

    #   pathalogy = {}
    #   pathalogy["slope"] = slope
    #   pathalogy["biases"] = biases
    #   pathalogy["log_observables"] = log_observables
    #   pathalogy["observables"] = observables
    #   pathalogy["result_lh"] = result_lh
    #   pathalogy["result_classical"] = result_classical

    #   with open("pathalogy","wb") as fp:
    #     pickle.dump(pathalogy, fp)


    info_print(result_lh.x, result_classical.x, slope, height)
