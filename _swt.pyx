# -*- coding: utf-8 -*-
#cython: profile=True
"""
Cython code for inner loop part for the stroke width transform

@author: Fredrik Wahlberg <fredrik.wahlberg@it.uu.se>
"""

from __future__ import division

import numpy as np
cimport numpy as np
cimport cython
import random


cdef extern from "math.h":
     int abs(int i)

cdef extern from "math.h":
     double fabs(double v)

cdef extern from "math.h":
    double sqrt(double x)
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
def swt(np.ndarray[np.int32_t, ndim=2] edges, np.ndarray[np.float64_t, ndim=2] gradientangle, int x1, int y1, int x2, int y2, int maxsteps):
    cdef int steep = 0
    cdef int dx = abs(x2 - x1)
    cdef int dy = abs(y2 - y1)
    cdef int sx, sy, d, i
    cdef int steps = max(dx, dy)
    cdef int w = 0
    cdef int startx = x1, starty = y1
    cdef double stopgradient = gradientangle[x1, y1] - np.pi, localgrad = 0
    
    if maxsteps > 0:
        steps = min(steps, maxsteps)

    if (x2 - x1) > 0: 
        sx = 1
    else: 
        sx = -1
    if (y2 - y1) > 0: 
        sy = 1
    else: 
        sy = -1

    if dy > dx:
        steep = 1
        x1,y1 = y1,x1
        dx,dy = dy,dx
        sx,sy = sy,sx
    d = (2 * dy) - dx

    cdef int imgWidth, imgHeight
    if steep:
        imgWidth = edges.shape[1]
        imgHeight = edges.shape[0]
    else:
        imgWidth = edges.shape[0]
        imgHeight = edges.shape[1]

    for i in range(steps):
        if x1 < 0:
            break
        if y1 < 0:
            break
        if y1 >= imgHeight:
            break
        if x1 >= imgWidth:
            break
        # Find first 1 after first finding a 0
        if steep:
            if edges[y1, x1] > 0:
                localgrad = stopgradient - gradientangle[y1, x1]
                while localgrad > np.pi:
                    localgrad -= 2*np.pi
                while localgrad <= -np.pi:
                    localgrad += 2*np.pi
                if fabs(localgrad) <= np.pi/2:
                    return sqrt((startx-y1)*(startx-y1)+(starty-x1)*(starty-x1))
        else:
            if edges[x1, y1] > 0:
                localgrad = stopgradient - gradientangle[x1, y1]
                # TODO: See if this should be replaced by fmod
                while localgrad > np.pi:
                    localgrad -= 2*np.pi
                while localgrad <= -np.pi:
                    localgrad += 2*np.pi
                if fabs(localgrad) <= np.pi/2:
                    return sqrt((startx-x1)*(startx-x1)+(starty-y1)*(starty-y1))
        w += 1
        while d >= 0:
            y1 = y1 + sy
            d = d - (2 * dx)
        x1 = x1 + sx
        d = d + (2 * dy)
    return -1

