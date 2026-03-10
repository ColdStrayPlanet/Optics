# -*- coding: utf-8 -*-
"""
This reads in a fits file and prints out an
ascii version.  Note that it assumes that the
original file contains only 1 2D array.
nad is the number of digits after the decimal 
(so there are nad + 1 significant digits)
"""

import numpy as np
import astropy.io.fits as fts

def WriteFitsSimple(inputfilename, outputfilename, nad=5, delim = ' ', newln = '\n'):
    hd = fts.open(inputfilename)
    fid = open(outputfilename, 'w')
    fmt = '%1.' + str(nad) + 'e'
    np.savetxt(outputfilename, hd[0].data, fmt=fmt, delimiter=delim, newline=newln)
    fid.close()
    return(None)

