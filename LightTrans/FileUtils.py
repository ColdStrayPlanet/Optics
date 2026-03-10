# -*- coding: utf-8 -*-
"""
File utilities for working with
LightTrans VirtualLab Fusion
"""

import numpy as np
from os.path import join, isfile
from os import listdir, system
import csv
import astropy.io.fits as fts
rightdblquote = '\u201D'
leftdblquote  = '\u201C'


"""
This reads .csv file output from LightTrans for numerical arrays.
It assumes:
    The comment lines begin with '#'
    Values are separated by commas
    Values are real (but read in as strings)
    All rows have the same length
    The final value in each row has a comma after it.
       This causes the reader to add an extra string with
       the value '' at the end.  So, if the rows have 512
       strings, the 513th string is ''
fn - the filename (incl. path and suffix)
finfile - for files with the suffix .fin, which are csv files created
    by the 'export detector' with my modification for a comma separator.
    the values between the commas are of the form:
        '3.70756134E-05 - i3.05492831E-05'
     values ss
ar - the array output
"""
def ReadCSVLightTrans(fn, finfile = False):  # also try the function ReadExportDet below!!
    ar = None
    fp = open(fn, 'r')
    reader = csv.reader(fp)
    #first, count the rows and columns containing data
    rowcount = 0
    for row in reader:
        if row[0][0] == '#':
            pass
        else:
            rowcount += 1
            if rowcount == 1:
                if row[-1] == '':
                    ncol = len(row) - 1
                else:
                    ncol = len(row)
    fp.close();
    ar = np.zeros((rowcount, ncol))
    if finfile:
        ar =  np.zeros((rowcount, ncol)).astype('complex')
    fp = open(fn, 'r')
    reader = csv.reader(fp);
    rowcount = -1
    for row in reader:
        if row[0][0] == '#':
            pass
        else:
            rowcount += 1
            for k in range(ncol):
                if not finfile:
                    ar[rowcount][k] = float(row[k])
                else:
                    if row[k] == '0':
                        ar[rowcount][k] = 0.0
                    else:
                        q = row[k].split(' ')  # string of the form '3.70756134E-05 - i3.05492831E-05'
                        sgn = 1.0
                        if q[1] == '-': sgn = -1.0  # get the sign of the imaginary part
                        ar[rowcount][k] = float(q[0]) + 1j*sgn*float(q[2][1:])
    fp.close();
    return(ar)

#This loops over the .csv files created by a VLF parameter run and creates .npy files
#The filenames are assumed to end in "_Re_n.csv" or "_Im_n.csv" where n is the file number.
#fnbase - (includes path) is the prefix to  "_Re_n.csv" or "_Im_n.csv"
#startind - file number of the first file in the loop
#endind   -                    last
def VLFcsv2npy(fnbase, startind, endind):
    for k in np.arange(startind, endind + 1):
        stk = str(k)
        fnr = fnbase + '_Re_' + stk +'.csv'
        fni = fnbase + '_Im_' + stk +'.csv'
        if not isfile(fnr):
            raise Exception(f"File Not Found: {fnr}")
        if not isfile(fni):
            raise Exception(f"File Not Found: {fni}")
        re = ReadCSVLightTrans(fnr, finfile=False)
        im = ReadCSVLightTrans(fni, finfile=False)
        fnnpy = fnbase + '_' + stk + '.npy'
        np.save(fnnpy, re + 1j*im)
    return None

"""
This loops over the .npy files (see the VLFcsv2npy function above) and creates a system matrix
The filenames are assumed to end in "_Ex_n.npy" or "_Ey_n.npy" where n is the file number
fnbase - (includes path) is the prefix to "_Ex_n.npy" or "_Ey_n.npy"
startind - file number of the first file in the loop
endind   -                    last
"""
def MakeSystemMatrix(fnbase,ExOrEy='Ex',startind=0,endind=1088):
    count = -1
    for k in np.arange(startind, endind + 1):
        stk = '_' + ExOrEy + '_' + str(k) + '.npy'
        fn = fnbase + stk
        if not isfile(fn):
            print("File Not Found:", fn)
            assert False
        field = np.load(fn)
        field = field.flatten()
        count += 1
        if k == startind:
            S = np.zeros((len(field), endind-startind+1)).astype('complex')
            lff = len(field)
        if len(field) != lff:
            print("Number of pixels is not consistent.  File =", fn)
            assert False
        S[:,count] = field
    fn = fnbase + '_SystemMatrix_' + ExOrEy + '.npy'
    np.save(fn, S)
    return None


"""
This reads the separate real and imaginary output files created by the LightTrans
  text export and then puts the complex field into an .npy file.  The complex
  field is returned.
fnbase is the base filename (incl. path).  Assumed filenames are of the form fnbase_real.txt.
  The output filename is of the form fnbase.npy
"""
def ReadReImWriteComplex(fnbase):  #file suffixes should be "_Re" (or "_Im") + ".csv" or ".txt"
    fcsvr = fnbase + "_Re.csv"
    fcsvi = fnbase + "_Im.csv"
    ftxtr = fnbase + "_Re.txt"
    ftxti = fnbase + "_Im.txt"
    if isfile(fcsvr):
        re = ReadCSVLightTrans(fcsvr)
    elif isfile (ftxtr):
        re = ReadCSVLightTrans(ftxtr)
    else:
        print("Did not find file:", fcsvr, "or", ftxtr)
        return(None)
    if isfile(fcsvi):
        im = ReadCSVLightTrans(fcsvi)
    elif isfile(ftxti):
        im = ReadCSVLightTrans(ftxti)
    else:
        print("Did not find file:", fcsvi, "or", ftxti)
        return(None)
    ss = re + 1j*im
    np.save(fnbase + ".npy", ss)
    return(ss)




#This a M^2-by-N system matrix (where the images are M-by-M) and averages
#  down to a (M/2)^2-by-N system matrix 
#S is the matrix
def ReBinSysMat(S):
    nn = int(np.sqrt(S.shape[0]))
    assert nn*nn == S.shape[0]
    assert np.mod(nn,2) == 0
    T = np.zeros((S.shape[0]//4,S.shape[1])).astype('complex')  #rebinned matrix
    for kx in range(nn//2): #loop over image pixels in the new matrix
        lx0 = 2*kx;  lx1 = 2*kx+1      
        for ky in range(nn//2):
            ly0 = 2*ky; ly1 = 2*ky+1
            ind0 = np.ravel_multi_index((ly0,lx0), (nn,nn))
            ind1 = np.ravel_multi_index((ly0,lx1), (nn,nn))
            ind2 = np.ravel_multi_index((ly1,lx0), (nn,nn))
            ind3 = np.ravel_multi_index((ly1,lx1), (nn,nn))
            indT = np.ravel_multi_index((ky,kx), (nn//2,nn//2))
            T[indT,:] = 0.25*( S[ind0,:] + S[ind1,:] + S[ind2,:] + S[ind3,:] )
    return T
def ReBinSquareImage(S):  #rebins a square image by a factor of 2 in each dimension
    sh = S.shape
    if np.iscomplexobj(S):
        T = np.zeros((sh[0]//2, sh[1]//2)).astype('complex')
    else: 
        T = np.zeros((sh[0]//2, sh[1]//2))
    for ky in range(sh[0]//2):
        ly0 = 2*ky; ly1 = ly0 + 1
        for kx in range(sh[1]//2): 
            lx0 = 2*kx; lx1 = lx0 + 1
            T[ky,kx] = 0.25*(S[ly0,lx0] + S[ly0,lx1] + S[ly1,lx0] + S[ly1,lx1] )
    return T


"""
This reads in a fits file and prints out an
ascii version.  Note that it assumes that the
original file contains only 1 2D array.
nad is the number of digits after the decimal 
(so there are nad + 1 significant digits)
"""
def ReadFitsWriteAscii(inputfilename, outputfilename, nad=5, delim = ' ', newln = '\n'):
    hd = fts.open(inputfilename)
    fid = open(outputfilename, 'w')
    fmt = '%1.' + str(nad) + 'e'
    np.savetxt(outputfilename, hd[0].data, fmt=fmt, delimiter=delim, newline=newln)
    fid.close()
    return(None)

#this edits filenames in a directory.  It's a script ... not really a function
def LoopOverFilesAndEditNames():
   #amplefilename = "2LensCgkn21x21_222.fin"
   for fn in listdir():
        nfn = fn[:14] + '_Ex' + fn[14:]
        cmd = 'rename ' + fn + ' ' + nfn  #'rename works on windows
        system(cmd)
   return None


#==============================================================================
#                      Scrap Yard                                             #
#==============================================================================


"""
This reads the B-spline output files (e.g., field_y-3x+4.fin) into a 
  large system matrix.  See the code for the specifics of the filenames.
FromNegativeMask - if True, it means that the system matrix is generated
  by subtracting the field values in the numbered files from the field
  contained in the 'Unobstrcted' file which corresponds to a plane wave
  with no basis function subtracted.
"""
def _Depricated_MakeBsplineSysMat(ExOrEy='Ex', FromNegativeMask=False):
    assert (ExOrEy == 'Ex') or (ExOrEy == 'Ey')
    pos = np.arange(21)  #index of lateral offsets for x and y  - note funny behavior of np.arange
    nc = len(pos)**2  #  number of columns in big matrix
#    loc = "E:\MagAOX\ScanData_VLF\PmaskNoBumpsLev2"  # location of .fin files
#    fn = 'MaskNoFoldNoBumpLev2Kn51x51_Kn172um_p256_1664um_l908nm_1492.fin'  # sample file name
#    q = ReadCSVLightTrans(join(loc,fn), finfile=True)  # note the double underscores
    loc = "E:/MyOpticalSetups/EFC Papers/ScanData"  # location of .fin files
    fn =  'TwoOAPCgkn21x21_91_' + leftdblquote + 'Ex-Component'+ rightdblquote + '.fin'  # sample file name
    q = ReadExportDet(join(loc,fn))
    nr = q.shape[0]*q.shape[1]  # number of rows in big matrix
    bm = np.zeros((nr, nc)).astype('complex')  # big matrix
    if FromNegativeMask:
        fn =  'TwoOAPCgkn21x21_Unobstructed_' + leftdblquote + ExOrEy + '-Component'+ rightdblquote + '.fin' 
        basefield = ReadExportDet(join(loc,fn))
    ccount = -1
    for nx in pos:  # my programmable parameter run (snippet) in VLF scans over the y direction for a fixed x position, only starting the next value of x when all of the ys are done
        for ny in pos:
            ccount += 1
            fn =  'TwoOAPCgkn21x21_' + str(ccount) + '_' + leftdblquote + ExOrEy + '-Component'+ rightdblquote + '.fin'
            print("reading", fn)
            #q = ReadCSVLightTrans(join(loc,fn))
            q = ReadExportDet(join(loc,fn))
            if FromNegativeMask:
                q = basefield - q
            bm[:, ccount] = q.reshape((nr, ), order='C')
    return(bm)


def _Depricated_MakeBsplineSysMat_Old():
    tts = ['-4', '-3', '-2', '-2', '+0', '+1', '+2', '+3', '+4']
    nc = len(tts)**2  # number of columns in big matrix
    # read a file to get the size
    q = ReadCSVLightTrans('field_y-4_x-2.fin', finfile=True)
    nr = q.shape[0]*q.shape[1]
    bm = np.zeros((nr, nc)).astype('complex') #  big output matrix
    ccount = -1
    for s1 in tts:
        for s2 in tts:
            ccount += 1
            fn = 'field_y' + s2 + '_x' + s1 + '.fin'  # filename
            q = ReadCSVLightTrans(fn, finfile=True)
            bm[:, ccount] = q.reshape((nr, ), order='F')
    return(bm)

#4/3/24 - As of VLF 2023.2 there export detector is an add-on and ReadCSVLightTrans() should be used.
#As of 1/2/24 this reads the new .fin files created by the export detector in a parameter run.
#If it does work try the utility above.  It might be necessary to change a filename to .txt
#  in order to inspect the contents.
#This is designed for files in which each line consists complex numbers of the form
#   real + 'i'complex, separated by tabs (\t).
#fn is the filename, incl. path
def _Depricated_ReadExportDet(fn, quiet=True):
    print("As of VLF 2023.2 there export detector is an add-on and ReadCSVLightTrans() should be used.")
    fp = open(fn, 'r')
    reader = csv.reader(fp)
    nrows = 0   #first, count the rows
    for line in reader:  #each line is a list object with 1 element
        if line[0][0] == '#': continue
        nrows += 1
    fp.close()
    row = line[0].split('\t')
    if len(row) == 1:
       print("This file does not seem to lines separated by tabs.")
       return(None)
    ncols = len(row)
    s = np.zeros((nrows, ncols)).astype('complex')
    
    fp = open(fn, 'r')  # fill out the s array
    reader = csv.reader(fp)
    rc = -1
    for line in reader:  #each line is a list object with 1 element
        if line[0][0] == '#': continue
        rc += 1
        row = line[0].split('\t')
        for cc in range(ncols):
            ni = row[cc].find('i')
            if ni == -1:  # the character 'i' is not in the string
                s[rc,cc] = float(row[cc])
                continue
            nn = row[cc].find(' - ')
            nq = row[cc].find(' + ')
            ns = -1
            sgnim = 1.0
            if nn + nq == -2:
                print("element", rc, ",", cc, ":", row[cc], "does not fit format requirements." )
                return(None)
            if nn > -1:
                sgnim = -1.0
                ns = nn 
            elif nq > -1:
                sgnim = 1.0
                ns = nq
            else:
                print("element", rc, ",", cc, ":", row[cc], "does not fit format requirements." )
                return(None)
            value = float(row[cc][:ns]) + 1j*sgnim*float(row[cc][ni+1:])
            s[rc,cc] = value 
    if np.max(np.abs(np.imag(s))) == 0.:
        s = np.real(s)
    return(s)

def _Depricated_ReadExportDet(fn, quiet=True):
    complexvalued = False
    fp = open(fn, 'r')
    reader = csv.reader(fp)
    nrows = 0   #first, count the rows
    for line in reader:  #each line is a list object with 1 element
        if line[0][0] == '#': continue
        nrows += 1
    row = line[0].split('\t')
    if len(row) == 1:
       print("This file does not seem to lines separated by tabs.")
       return(None)
    ncols = len(row)
    ni = row[0].find('i')
    if ni == -1:
        if not quiet:
            print("The quantities in this file appear to be real valued, with the last value being", row[-1])
        s = np.zeros((nrows, ncols))
    else:
        if not quiet:
            print("The quantities in this file appear to be complex valued, with the last value being", row[-1]) 
        complexvalued = True
        s = np.zeros((nrows, ncols)).astype('complex')
    fp.close()

    fp = open(fn, 'r')  # fill out the s array
    reader = csv.reader(fp)
    rc = -1
    for line in reader:  #each line is a list object with 1 element
        if line[0][0] == '#': continue
        rc += 1
        row = line[0].split('\t')
        for cc in range(ncols):
            if complexvalued:
                ni = row[cc].find('i')
                if ni == -1:
                    print("Error: Did not find the string 'i'.")
                    return(None)
                nn = row[cc].find(' - ')
                nq = row[cc].find(' + ')
                ns = -1
                sgnim = 1.0
                if nn + nq == -2:
                    print("element", rc, ",", cc, ":", row[cc], "does not fit format requirements." )
                    return(None)
                if nn > -1:
                    sgnim = -1.0
                    ns = nn 
                elif nq > -1:
                    sgnim = 1.0
                    ns = nq
                else:
                    print("element", rc, ",", cc, ":", row[cc], "does not fit format requirements." )
                    return(None)
                value = float(row[cc][:ns]) + 1j*sgnim*float(row[cc][ni+1:])
            else: # real valued
                 value = float(row[cc])
            s[rc,cc] = value 
    return(s)
