#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 18:47:31 2025

@author: avicenna
"""

import numpy as np

_ALLOWED_CHRS = ['<','>','/','*','.',',']

def _check_format(x):

  #try to get rid of stuff like 2e4 etc
  try:
    if x[0] not in ['<','>']:
      x = str(format(float(x), 'f'))
    else:
      x = x[0] + str(format(float(x[1:]), 'f'))
  except ValueError:
    pass

  assert all([char in _ALLOWED_CHRS or char.isnumeric() for char in str(x)]),\
    f"Titre {x} does not adhere to required format of numbers or {_ALLOWED_CHRS}."


def _log2(x, shift=0):

  if isinstance(x, float) and np.isinf(x):
    return x

  x = str(x)

  _check_format(x)

  if'/' in str(x):
    return np.max([_log2(x.split('/')[0], shift),
                   _log2(x.split('/')[1], shift)])

  try:
    if x[0] in ['<','>']:
      offset = (x[0]=='<')*(-shift) + (x[0]=='>')*(shift)
      return np.log2(float(x[1:])/10)+offset
    elif x=='*':
      return np.nan
    else:
      return np.log2(float(x)/10)
  except Exception as e:
    raise ValueError(f"Can't convert titre {x}: {e}")

def _is_LLD(x):
  return str(x)[0]=='<'

def _is_ULD(x):
  return str(x)[0]=='>'

is_LLD = np.vectorize(_is_LLD)
is_ULD = np.vectorize(_is_ULD)
log2 = np.vectorize(lambda x: _log2(x))
log2_shift = np.vectorize(lambda x: _log2(x, 1))