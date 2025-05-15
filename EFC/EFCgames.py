#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 18:42:03 2025

@author: rfrazin
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import EFC

with open('Stuff05152025.pickle','rb') as fp: stuff = pickle.load(fp); del(fp)


A = EFC.EFC(EFC.B21, EFC.B33, EFC.DomMat, EFC.CroMat)





plt.figure(); plt.plot(np.log10(stuff['cost45'])-2.35,'ko-',np.log10(stuff['costXax'])-2.35,'bo-');
