# -*- coding: utf-8 -*-
"""
@author: Fredrik Wahlberg <fredrik.wahlberg@it.uu.se>
"""
import swt
import shapecontext
import demoimage
import time

#%% Run feature extraction
print("Loading data...", end="")
I = demoimage.getGrayScaleImage()
print("done")

print("Running stroke width transform...", end="")
t = time.time()
swt_data = swt.SWT(I)
print("done (%.1fs)" % (time.time()-t))

N = 10
p = 5
q = 16
rotationalinvariant = False
samples = 100

print("Extracting shape context feature descriptors...", end="")
t = time.time()
sc = shapecontext.ShapeContext(N, p, q, rotationalinvariant)
v = sc.generate(swt_data[0], swt_data[2])
v_sampled = sc.generate(swt_data[0], swt_data[2], samples)
print("done (%.1fs)" % (time.time()-t))

#%% Plot
import matplotlib.pyplot as plt
plt.figure()
plt.subplot(2, 3, 1)
plt.title('Original')
plt.imshow(I, cmap='gray')
plt.subplot(2, 3, 2)
plt.title('SWT 1')
plt.imshow(swt_data[0], cmap='gray')
plt.subplot(2, 3, 3)
plt.title('SWT 2')
plt.imshow(swt_data[1], cmap='gray')
plt.subplot(2, 3, 4)
plt.title('Gradient angles')
plt.imshow(swt_data[2], cmap='gray')
plt.subplot(2, 3, 5)
plt.title('Gradient magnitudes')
plt.imshow(swt_data[3], cmap='gray')
plt.subplot(2, 3, 6)
plt.title('Shape context samples')
plt.imshow(v_sampled, cmap='winter')
plt.show()
