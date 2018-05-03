# -*- coding: utf-8 -*-
"""
@author: Fredrik Wahlberg <fredrik.wahlberg@it.uu.se>
"""

import os.path
import numpy as np
from model import SharedKernelClassifier


cache_path = os.path.expanduser("~/tmp/gp_multiclass.old")

# Parameter marginalization

filenames = [f for f in os.listdir(cache_path) if os.path.isfile(os.path.join(cache_path, f))]
filenames = [os.path.join(cache_path, f) for f in filenames if f[:6]=='result' and f[-3:]=='npz']
likelihoods = list()
for fn in filenames:
    try:
        data = np.load(fn)
        likelihoods.append(float(data['likelihood']))
    except:
        print("File corrupt %s ?" % (fn.split('/')[-1]))
likelihoods_sum = np.sum(np.exp(np.asarray(likelihoods)))
cs = np.cumsum(np.sort(np.exp(np.asarray(likelihoods))/likelihoods_sum)[::-1])

C = None
labels = None
for i, fn in enumerate(filenames):
    print("%i/%i %s" % (i, len(filenames), fn))
    try:
        data = np.load(fn)
        c = np.asarray(data['covariance_matrix'].tolist())
        c /= np.max(c)
        l = np.exp(float(data['likelihood']))/likelihoods_sum
        cl = c*l
        if np.all(np.isfinite(cl)):
            if C is None:
                C = cl
            else:
                C += cl
        else:
            print(" Some elements of covariance_matrix are not finite")
        if labels is None:
            labels = np.vstack(data['labels'].tolist())
    except:
        print(" File corrupt")

C /= np.max(C)
dummy_estimator = SharedKernelClassifier()
ntop = dummy_estimator._affinity_ntop(C, labels)

print("1-top after marginalization is %.1f%%" %(ntop[4]*100))

