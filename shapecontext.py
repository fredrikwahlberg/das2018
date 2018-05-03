# -*- coding: utf-8 -*-
"""
@author: Fredrik Wahlberg <fredrik.wahlberg@it.uu.se>
"""

from __future__ import division

import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True, inplace=True)
import _shapecontext

__all__ = ['ShapeContext', 'getImageCoordinates']

#def getShapeContextDescriptors(image, N, p, q, rotationalinvariant=False, samples=None):
#    sc = ShapeContext(N, p, q, rotationalinvariant)
#    return None

def getImageCoordinates(image):
    assert image.dtype == np.dtype('int32')
    return _shapecontext._getImageCoordinates(image)

class ShapeContext:
    """Class for extracting shape context descriptors. Written as a 
    class for being able to hold precalc tables and debug information."""
    def __init__(self, N, p, q, rotationalinvariant=False):
        self.N = N
        if self.N % 2 == 0:
            self.N += 1
        Nhalf = int(np.floor(N/2))
        self.p = int(p)
        self.q = int(q)
        self.rotationalinvariant = rotationalinvariant
        distmap = np.zeros((self.N, self.N))
        self._anglemap = np.zeros((self.N, self.N), dtype=np.float64)

        for i in range(-Nhalf, Nhalf+1):
            for j in range(-Nhalf, Nhalf+1):
                u = i + Nhalf
                v = j + Nhalf
                distmap[u, v] = i**2 + j**2 # Square euclidean distance
                self._anglemap[u, v] = np.arctan2(i, j)
        self._anglemap += np.pi/self.q     # Moving centre of bin
        distmap = np.sqrt(distmap)      # Euclidean distance
        # Hack to avoid error messages on an element that is not used
        distmap[Nhalf, Nhalf] = 1
        # Make boolean map for values in range
        self._allowed = np.zeros(distmap.shape, dtype=np.int32)
        self._allowed[distmap < Nhalf] = 1
        self._allowed[Nhalf, Nhalf] = 0

        # Normalize distmap
        logmap = np.log(distmap)*p/np.log(Nhalf)

        self.pFirstBin = np.asarray(np.floor(logmap), dtype=np.int32)
        pDiff = logmap - (self.pFirstBin + 0.5)
        assert np.max(np.abs(pDiff)) <= 0.5
        self.pFirstWeight = np.asarray(1-np.abs(pDiff), dtype=np.float64)
        self.pSecondWeight = 1-self.pFirstWeight
        self.pSecondBin = self.pFirstBin.copy()
        self.pSecondBin[pDiff < 0] -= 1
        self.pSecondBin[pDiff >= 0] += 1
        self.pSecondBin[self.pSecondBin<0] = 0
        self.pSecondBin[self.pSecondBin>=p] = p-1
        self._anglemap = np.ascontiguousarray(self._anglemap)
        self._allowed = np.ascontiguousarray(self._allowed)
        self.pFirstBin = np.ascontiguousarray(self.pFirstBin)
        self.pFirstWeight = np.ascontiguousarray(self.pFirstWeight)
        self.pSecondBin = np.ascontiguousarray(self.pSecondBin)
        self.pSecondWeight = np.ascontiguousarray(self.pSecondWeight)
    

    def generate(self, edgemap, anglemap, samples=None):
#        edgemap, anglemap = self._edgemapFromGrayscale(grayscale_image)
        assert edgemap.ndim == 2
        assert anglemap.ndim == 2
        if self.rotationalinvariant:
            anglemap[:] = 0
        if samples is not None:
            assert samples > 0
            return self._sampleImageShapeContext(edgemap, anglemap, samples)
        else:
            return self._getImageShapeContext(edgemap, anglemap)

#    def _edgemapFromGrayscale(self, grayImage):
#        from quill import StrokeWidthTransformQuill
#        # Get gradient data from StrokeWidthTransformQuill
#        SWTQuill = StrokeWidthTransformQuill(grayImage, mask=np.ones(grayImage.shape))
#        sampleCoords = SWTQuill.getGoodCoordinates()
#        edgeMap = np.zeros(grayImage.shape)
#        for s in sampleCoords:
#            edgeMap[s[0], s[1]] = 1
#        return edgeMap, SWTQuill._ang.copy()

    def _getImageShapeContext(self, bwImage, grad):
        I = np.ascontiguousarray(bwImage, dtype=np.int32)
        vectors = np.zeros((np.sum(bwImage.ravel()>0), self.p*self.q))
        i = 0
        for x in range(I.shape[0]):
            for y in range(I.shape[1]):
                if I[x, y] > 0:
                    vectors[i, :] = self._getShapeContext(I, x, y, grad[x, y]).ravel()
                    i += 1
        return vectors

    def _sampleImageShapeContext(self, edges, grad, samples):
        edgemap = np.ascontiguousarray(edges, dtype=np.int32)
        assert samples > 0
        if samples > np.sum(edgemap.ravel()>0):
            samples = np.sum(edgemap.ravel()>0)
        vectors = np.zeros((samples, self.p*self.q))
        from random import sample
        for i, coords in enumerate(sample(getImageCoordinates(edgemap), samples)):
            vectors[i, :] = self._getShapeContext(edgemap, coords[0], coords[1], grad[coords[0], coords[1]]).ravel()
        return vectors

    def _getShapeContext(self, image, x, y, baseangle):
        return _shapecontext._getShapeContext(x, y, baseangle, image, self._allowed, 
                            self._anglemap, int(np.floor(self.N/2)), self.pFirstBin, 
                            self.pFirstWeight, self.pSecondBin, 
                            self.pSecondWeight, self.p, self.q)
                            
#    def getShapeContext(self, image, x, y, baseangle):
#        H = np.zeros((self.p, self.q))
#        for i in range(x-Nhalf, x+Nhalf+1):
#            if i > 0 and i < image.shape[0]:
#                for j in range(y-Nhalf, y+Nhalf+1):
#                    if j > 0 and j < image.shape[1]:
#                        u = i - x + Nhalf
#                        v = j - y + Nhalf
#                        if self._allowed[u, v] and image[i, j] > 0:
#                            # Load angle and conpensatew for the grandient angle
#                            phi = self._anglemap[u, v] - baseangle 
#                            # Normalize angle
#                            phi %= 2*np.pi
#                            phi *= self.q/(2*np.pi)
#                            # Get first bin
#                            qFirstBin = np.floor(phi)
#                            qDiff = phi - (qFirstBin + 0.5)
#                            assert np.abs(qDiff) <= 0.5
#            
#                            qFirstWeight = 1-np.abs(qDiff)
#                            qSecondWeight = 1-qFirstWeight
#                            if qDiff < 0:
#                                qSecondbin = qFirstBin - 1
#                            if qDiff >= 0:
#                                qSecondbin = qFirstBin + 1
#                            if qSecondbin < 0:
#                                qSecondbin += self.q
#                            if qSecondbin >= self.q:
#                                qSecondbin -= self.q
#
#                            H[self.pFirstBin[u, v], qFirstBin] += self.pFirstWeight[u, v]*qFirstWeight
#                            H[self.pFirstBin[u, v], qSecondbin] += self.pFirstWeight[u, v]*qSecondWeight
#                            H[self.pSecondBin[u, v], qFirstBin] += self.pSecondWeight[u, v]*qFirstWeight
#                            H[self.pSecondBin[u, v], qSecondbin] += self.pSecondWeight[u, v]*qSecondWeight
##                            print logmap[u+m, v+m], phi
#        # Normalize
#        s = np.sum(H.ravel())
#        if s > 0:
#            H /= s
#            
#        return H
