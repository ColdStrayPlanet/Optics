#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is for an OAP-based Gaussian mode laser transmission system
"""

import numpy as np

# effective focal length with lenses of f1 and f2, separted by distance d
def TwoLens(f1, f2, d):
   a = 1./f1 + 1./f2 - d/(f1*f2)
   return( 1./a )

#thin lens image distance. obd= object dist, f=focal length,
def ImageDist(obd, f):
   a = 1./f - 1./obd
   return( 1./a )

#f1, f2, d correspond to the small OAPs separated by d.  F is fl of the big OAP
def OAP_Feed(F,d):
   f1 = 16; f2 = 24; d_nom = 8;
   f_nom = TwoLens(f1, f2, d_nom)
   f_tru = TwoLens(f1, f2, d)
   obd = (f_nom - f_tru) + F
   return( ImageDist(obd, F) )
