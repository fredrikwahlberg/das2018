# -*- coding: utf-8 -*-
"""
Function for the stroke with transform

Dependent on cython code in _swt.pyx

@author: Fredrik Wahlberg <fredrik.wahlberg@it.uu.se>
"""

from __future__ import division

import numpy as np
import cv2
from sklearn.mixture import GaussianMixture
import random
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True, inplace=True)
import _swt


def SWT(grayScaleImage, n_sobel=5, n_blur=0, n_gmmsamples=50000, n_gmm_iter=100, max_stroke_width=20):
    """
    Stroke Width Transfor using a GMM to estimate the Canny thresholds
    
    grayScaleImage, n_sobel=5, n_blur=0, n_gmmsamples=50000, n_gmm_iter=100, max_stroke_width=20
    
    returns
    swtedges1
    swtedges2
    anglemap
    gradientmagnitude

    """
    J = np.asarray(grayScaleImage.copy(), dtype=np.float)
    if n_blur > 0:
        J = np.asarray(cv2.GaussianBlur(J, (n_blur, n_blur), 0), dtype=np.uint8)
    sobelx = cv2.Sobel(J.copy(), cv2.CV_64F, 1, 0, ksize=n_sobel)
    sobely = cv2.Sobel(J.copy(), cv2.CV_64F, 0, 1, ksize=n_sobel)
    
    anglemap = np.arctan2(sobelx, sobely)
    gradientmagnitude = np.sqrt(np.abs(sobelx)**2 + np.abs(sobely)**2)
    
    # Sample from gradient magnitude data
    v = gradientmagnitude.ravel()
    vr = v[random.sample(range(len(v)), min(len(v), n_gmmsamples))]
    
    # Set up GMM
    gmm = GaussianMixture(n_components=2, covariance_type='diag', max_iter=n_gmm_iter)
    gmm.fit(np.vstack(vr))
    
    # Bisection search for the point of equal probability 
    i = np.argmin(gmm.means_)
    j = np.argmax(gmm.means_)
    hi = gmm.means_[j]
    lo = gmm.means_[i]
#    assert not np.isclose(lo - hi, 0.0), "In swt.py, hi and lo are equal"
    mid = (lo + hi) / 2
    while not np.isclose(lo - hi, 0.0):
        mid = (lo + hi) / 2
        pred = gmm.predict_proba([mid])[0, :]
    #    if np.isclose(g.predict_proba([mid])[0, j], 1.0):
        if pred[i] < pred[j]:
            hi = mid
        else:
            lo = mid
    thresholds = np.concatenate([mid, gmm.means_[j]])

    # Find edges (this uses canny)
    edges = np.asarray(cv2.Canny(np.asarray(J, dtype=np.uint8), thresholds[0], thresholds[1], apertureSize=n_sobel, L2gradient=True), dtype=np.int32)

    # SWT
    swtedges1 = np.zeros(edges.shape)
    swtedges2 = np.zeros(edges.shape)
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[i, j]:
                w1 = _swt.swt(edges, anglemap, i, j, int(i+np.cos(anglemap[i, j])*max_stroke_width*10),
                        int(j+np.sin(anglemap[i, j])*max_stroke_width*10), maxsteps=int(max_stroke_width*1.5))
                if w1 > 0 and w1 <= max_stroke_width:
                    swtedges1[i, j] = 255
                else:
                    w2 = _swt.swt(edges, anglemap, i, j, int(i+np.cos(anglemap[i, j]+np.pi)*max_stroke_width*10),
                        int(j+np.sin(anglemap[i, j]+np.pi)*max_stroke_width*10), maxsteps=int(max_stroke_width*1.5))
                    if w2 > 0 and w2 <= max_stroke_width:
                        swtedges2[i, j] = 255
    
    return swtedges1, swtedges2, anglemap, gradientmagnitude

