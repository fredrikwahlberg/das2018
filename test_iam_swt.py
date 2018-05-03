# -*- coding: utf-8 -*-
"""
@author: Fredrik Wahlberg <fredrik.wahlberg@it.uu.se>
"""

import numpy as np
import cv2


#%% Load IAM
from dataloaders import load_iam_metadata
path = "/media/fredrik/UB Storage/Images/IAM"
keys, filenames, authors, bbxs = load_iam_metadata(path)
import os.path
full_filenames = [os.path.join(path, fn) for fn in filenames]
outpath = os.path.expanduser("~/tmp/iam_testing")
out_filenames = [os.path.join(outpath, fn.split("/")[-1]) for fn in filenames]

import random
samples = random.sample(list(zip(keys, authors, bbxs, full_filenames, out_filenames)), 150)

#%%
width = list()
height = list()
for x,y,w,h in bbxs:
    width.append(w)
    height.append(h)

print("Width min/max %i/%i" % (np.min(width), np.max(width)))
print("Height min/max %i/%i" % (np.min(height), np.max(height)))

#%%
import time
import swt
for key, author, bbx, infile, outfile in samples:
    print(infile.split("/")[-1])
    if not os.path.exists(outfile):
        t = time.time()
        I = cv2.imread(infile)
        I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
        assert len(np.unique(I)) > 2, "Image is binary"
        # x,y,w,h
        I = I[bbx[1]:bbx[1]+bbx[3], bbx[0]:bbx[0]+bbx[2]]
        swt_data = swt.SWT(I)
        J=swt_data[0]+swt_data[1]
        assert len(np.unique(J))==2
        cv2.imwrite(outfile, J)
        print(" Unique colours: %i" % len(np.unique(I)))
        print(" Processing time: %.1fs" % (time.time()-t))
    else:
        print(" Already processed")


