# -*- coding: utf-8 -*-

#cython: profile=True

"""
Created on Mon Sep 22 15:07:49 2014

@author: Fredrik Wahlberg <fredrik.wahlberg@it.uu.se>
"""

from __future__ import division

import numpy as np
cimport numpy as np
cimport cython


cdef extern from "math.h":
     int abs(int i)

cdef extern from "math.h":
     double fabs(double v)

cdef extern from "math.h":
    double floor(double x)

cdef extern from "math.h":
    double fmod(double x, double y)

def _getImageCoordinates(np.ndarray[np.int32_t, ndim=2] image):
    coordlist = []
    cdef int i = 0, j = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > 0:
                coordlist.append([i, j])
    return coordlist

#def _getShapeContext(int x, int y, double baseangle, np.ndarray[np.int32_t, ndim=2, mode="c"] image, 
#                     np.ndarray[np.int32_t, ndim=2, mode="c"] allowed, 
#                     np.ndarray[np.float64_t, ndim=2, mode="c"] anglemap, 
#                     int Nhalf, np.ndarray[np.int32_t, ndim=2, mode="c"] pFirstBin, 
#                     np.ndarray[np.float64_t, ndim=2, mode="c"] pFirstWeight, 
#                     np.ndarray[np.int32_t, ndim=2, mode="c"] pSecondBin, 
#                     np.ndarray[np.float64_t, ndim=2, mode="c"] pSecondWeight, 
#                     int p, int q):
#@cython.boundscheck(False)
#@cython.wraparound(False)
def _getShapeContext(int x, int y, double baseangle, np.ndarray[np.int32_t, ndim=2] image, 
                     np.ndarray[np.int32_t, ndim=2, mode="c"] allowed, 
                     np.ndarray[np.float64_t, ndim=2, mode="c"] anglemap, 
                     int Nhalf, np.ndarray[np.int32_t, ndim=2, mode="c"] pFirstBin, 
                     np.ndarray[np.float64_t, ndim=2, mode="c"] pFirstWeight, 
                     np.ndarray[np.int32_t, ndim=2] pSecondBin, 
                     np.ndarray[np.float64_t, ndim=2] pSecondWeight, 
                     int p, int q):
                            
    cdef np.ndarray[np.float64_t, ndim=2, mode="c"] H = np.zeros((p, q))

    cdef int i, j, u, v
    cdef int qFirstBin, qSecondbin
    cdef double phi, qDiff
    cdef double qFirstWeight, qSecondWeight    

    cdef int image_shape_0 = image.shape[0], image_shape_1 = image.shape[1]
    cdef double pi2 = 2*np.pi, bin_per_rad = q/(2*np.pi)
    
    for i in range(x-Nhalf, x+Nhalf+1):
        if i > 0 and i < image_shape_0:
            for j in range(y-Nhalf, y+Nhalf+1):
                if j > 0 and j < image_shape_1:
                    u = i - x + Nhalf
                    v = j - y + Nhalf
                    if allowed[u, v] and image[i, j] > 0:
                        # Load angle and conpensate for the grandient angle
                        phi = anglemap[u, v] - baseangle 
                        # Normalize angle
#                        phi %= 2*np.pi
                        phi = fmod(phi, pi2)
                        phi *= bin_per_rad
                        # Get first bin
                        qFirstBin = <int> floor(phi)
                        qDiff = phi - (qFirstBin + 0.5)
#                        assert fabs(qDiff) <= 0.5

        
                        qFirstWeight = 1-fabs(qDiff)
                        qSecondWeight = 1-qFirstWeight
                        if qDiff < 0:
                            qSecondbin = qFirstBin - 1
                        if qDiff >= 0:
                            qSecondbin = qFirstBin + 1
                        if qSecondbin < 0:
                            qSecondbin += q
                        if qSecondbin >= q:
                            qSecondbin -= q

                        H[pFirstBin[u, v], qFirstBin] += pFirstWeight[u, v]*qFirstWeight
                        H[pFirstBin[u, v], qSecondbin] += pFirstWeight[u, v]*qSecondWeight
                        H[pSecondBin[u, v], qFirstBin] += pSecondWeight[u, v]*qFirstWeight
                        H[pSecondBin[u, v], qSecondbin] += pSecondWeight[u, v]*qSecondWeight
#                            print logmap[u+m, v+m], phi
    # Normalize
    s = np.sum(H.ravel())
    if s > 0:
        H /= s
            
    return H
    
