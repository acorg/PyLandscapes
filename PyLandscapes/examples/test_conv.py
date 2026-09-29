#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 09:25:47 2025

@author: avicenna
"""

import PyRacmacs as pr
import warnings
from PyLandscapes import fit



def info_print(result_lh, result_classical, serum):

  print(f"{serum}")
  print(f"likelihood slope: {result_lh[4]:.2f}, likelihood height: {result_lh[3]:.2f}")
  print(f"classical slope: {result_classical[4]:.2f}, classical height: {result_classical[3]:.2f}")
  print("")

nrepeats = 4
acmap = pr.read_racmap("map_H5.ace")
ag_inds = [ind for ind, ag in enumerate(acmap.ag_names) if "/72/2004" in ag
           or "R60" in ag]
ag_coordinates = acmap.ag_coordinates
nag,ndims = ag_coordinates.shape

for inds,serum in enumerate(acmap.sr_names):

  observables = acmap.titer_table.loc[:, serum].values

  log_observables = fit.log2(observables)

  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)

    p0,bounds = fit.get_pb(log_observables, ag_coordinates, True,
                           height_flexibility=0.1, bias_flexibility=0.01)

    coords = acmap.sr_coordinates[inds,:]

    for i in range(2):
      p0[i] = coords[i]
      bounds[i] = (coords[i]-0.01, coords[i]+0.01)


    result_lh1=\
      fit.single_cone_MLE(ag_coordinates, observables,
                          discrete_step=1,
                          p0=p0,
                          bounds=bounds,
                          use_likelihood=True)

    result_lh2=\
      fit.single_cone_MLE(ag_coordinates,
                          observables,
                          discrete_step=1,
                          p0=p0,
                          bounds=bounds,
                          use_likelihood=True, method="Nelder-Mead")

    if result_lh1.fun<result_lh2.fun:
      result_lh = result_lh1
    else:
      result_lh = result_lh2

    p0,bounds = fit.get_pb(log_observables, ag_coordinates, False)

    result_classical=\
      fit.single_cone_MLE(ag_coordinates,
                          observables,
                          discrete_step=1,
                          p0=p0,
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


    info_print(result_lh.x, result_classical.x, serum)
    breakpoint()
