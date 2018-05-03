# -*- coding: utf-8 -*-
"""
Writer identification based of shape context codebook features and
a shared kernel GP classifier. This file covers feature extraction
given the parameter distribution below.

@author: Fredrik Wahlberg <fredrik.wahlberg@it.uu.se>
"""

from __future__ import print_function, division

import numpy as np
import os.path
import time

cache_path = os.path.expanduser("~/tmp/gp_multiclass")


def cache_filename_features(N, p, q):
        return os.path.join(cache_path,
                            "features_N%i_p%i_q%i.npz" % (N, p, q))


parameter_distribution = {'p': list(range(2, 6+1)),
                          'q': list(range(4, 9+1)),
                          'N': list(range(7, 21+1, 2))}

from sklearn.model_selection import ParameterGrid
valid_parameters = [p for p in ParameterGrid(parameter_distribution) if p['N']>p['p']*3]
print("%i parameter configurations" % len(valid_parameters))
import random
random.shuffle(valid_parameters)

# %% Load meta data
db_keys = ['iam', 'cvl']
basepaths = {'iam': ["~/Data"],
             'cvl': ["~/Data"]}


def find_working_path(pathlist):
    # Returns the first working path in pathlist
    import os.path
    for path in pathlist:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            return expanded_path
    return None


def flatten_path_dict(pathdict):
    # Flattens the path dictionary (for finding files on the cluster)
    l = list()
    for key in pathdict.keys():
        l.extend(pathdict[key])
    return l


basepathlist = flatten_path_dict(basepaths)

from dataloaders import load_iam_metadata
path = find_working_path(basepaths['iam'])
iam_keys, iam_filenames, iam_authors, iam_bbx = load_iam_metadata(path)

from dataloaders import load_cvl_metadata
path = find_working_path(basepaths['cvl'])
cvl_keys, cvl_authors, cvl_filenames, cvl_set = load_cvl_metadata(path)

cvl_iam_authors = list(map(lambda a:a+1000, cvl_authors))
cvl_iam_authors.extend(iam_authors)
cvl_iam_filenames = cvl_filenames.copy()
cvl_iam_filenames.extend(iam_filenames)
cvl_iam_bbx = [None]*len(cvl_authors)
cvl_iam_bbx.extend(iam_bbx)

authors_filenames_bbx = list(zip(cvl_iam_authors, cvl_iam_filenames, cvl_iam_bbx))

#authors_filenames_bbx = authors_filenames_bbx[-100:]

#%% Setting up parallelization using spark
print("Creating a spark session...", end="")
#try:
#    spark
#except NameError:
from pyspark.sql import SparkSession
spark = SparkSession\
        .builder\
        .appName("WriterIdentification")\
        .getOrCreate()
print("done")

print("Uploading modules...", end="")
pyfiles = ["shapecontext.py", "_shapecontext.pyx", "swt.py", "_swt.pyx"]
for pf in pyfiles:
    spark.sparkContext.addPyFile(pf)
print("done")


#%% Load images as RDD
sc_basepathlist = spark.sparkContext.broadcast(basepathlist)

def load_images(author_filename_bbx):
    # Finds a working path for each element in filename on the cluster
    # and load the image
    import os.path
    filename = author_filename_bbx[1]
    bbx = author_filename_bbx[2]
    for path in sc_basepathlist.value:
        expanded_path = os.path.join(os.path.expanduser(path), filename)
        if os.path.exists(expanded_path):
            import cv2
            I = cv2.imread(expanded_path)
            if I.ndim > 2:
                I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
            if bbx is not None:
                # x,y,w,h
                I = I[bbx[1]:bbx[1]+bbx[3], bbx[0]:bbx[0]+bbx[2]]
                pass
            return (author_filename_bbx[0], I)
    return (author_filename_bbx[0], None)

authors_filenames_bbx_rdd = spark.sparkContext.parallelize(authors_filenames_bbx)
authors_images_rdd = authors_filenames_bbx_rdd.map(load_images)

#a = authors_images_rdd.first()
#a = authors_images_rdd.collect()

#%% Feature extraction loop

for parameter_set in valid_parameters:
    t0 = time.time()

    #%% First pass with shape context feature extraction
    sc_N = parameter_set['N']
    sc_p = parameter_set['p']
    sc_q = parameter_set['q']
    sc_rotationalinvariant = False
    sc_samples = 100

    def shape_context_sampling_cvl_iam(author_image):
        I = author_image[1]
        import swt
        swt_data = swt.SWT(I, n_sobel=3)
        import shapecontext
        sc = shapecontext.ShapeContext(sc_N, sc_p, sc_q, sc_rotationalinvariant)
        v_sampled = sc.generate(swt_data[0]+swt_data[1], swt_data[2], sc_samples)
        return (author_image[0], v_sampled)

    def group_sampled_vectors(a, b):
        # Takes authors_scsamples, removes author index and 
        # concatenates the matrices
        if len(a) == 2:
            a = a[1]
        if len(b) == 2:
            b = b[1]
        return np.concatenate((a, b), axis=0)

    def shape_context_distribution_cvl_iam(author_image):
        I = author_image[1]
        import swt
        swt_data = swt.SWT(I, n_sobel=3)
        import shapecontext
        sc = shapecontext.ShapeContext(sc_N, sc_p, sc_q, sc_rotationalinvariant)
        v = sc.generate(swt_data[0]+swt_data[1], swt_data[2])
        def buildNearestCentroids(clusters):
            from sklearn.neighbors.nearest_centroid import NearestCentroid
            clf = NearestCentroid()
            clf.fit(clusters, range(clusters.shape[0]))
            return clf
        nearest_centroids = list(map(buildNearestCentroids, cluster_centers))
        normedHistogramFeature = [None]*len(nearest_centroids)
        for i in range(len(nearest_centroids)):
            normedHistogramFeature[i] = np.zeros(nearest_centroids[i].centroids_.shape[0], dtype=np.float32)
            for pred in nearest_centroids[i].predict(v):
                normedHistogramFeature[i][pred] += 1
            normedHistogramFeature[i] /= np.sum(normedHistogramFeature[i])
            normedHistogramFeature[i] = np.vstack(normedHistogramFeature[i]).T
        return (author_image[0], normedHistogramFeature)

    def reduce_feature_list(a, b):
        assert type(a[0])==int or type(a[0])==list
        assert type(b[0])==int or type(b[0])==list
        authorlist = list()
        if type(a[0])==int:
            authorlist.append(a[0])
        elif type(a[0])==list:
            authorlist.extend(a[0])
        if type(b[0])==int:
            authorlist.append(b[0])
        elif type(b[0])==list:
            authorlist.extend(b[0])
        assert len(a[1])==len(b[1])
        featurelist = list()
        for i in range(len(a[1])):
            featurelist.append(np.concatenate((a[1][i], b[1][i]), axis=0))
        return (authorlist, featurelist)

    print("Running feature extraction (N=%i, p=%i, q=%i)" % (sc_N, sc_p, sc_q))
    cache_fn = cache_filename_features(sc_N, sc_p, sc_q)
    if not (os.path.exists(cache_fn) or os.path.exists(cache_fn+".lock")):
        print(" Creating lock file...", end="")
        open(cache_fn+".lock", 'a').close()
        print("done")

        print(" Running")
        authors_scsamples_rdd = authors_images_rdd.map(shape_context_sampling_cvl_iam)
        sc_samples = authors_scsamples_rdd.reduce(group_sampled_vectors)
        
        #%% Cluster feature vectors
        from sklearn.cluster import MiniBatchKMeans
        codebooks = [MiniBatchKMeans(n_clusters=n_clusters,  max_iter=100, 
                                        batch_size=10000) for n_clusters in 
                                        range(200, 360+1, 40)]
        codebooks = list(map(lambda cb: cb.fit(sc_samples), codebooks))
        cluster_centers = list(map(lambda cb: cb.cluster_centers_.copy(), codebooks))
        
    
        #%% Second pass of shape context feature extraction, associating every vector to a cluster
        authors_features_rdd = authors_images_rdd.map(shape_context_distribution_cvl_iam)
        #d = authors_features_rdd.collect()
        authors, featurelist = authors_features_rdd.reduce(reduce_feature_list)
        print(" Finished feature extraction in %.1f minutes" % ((time.time()-t0)/60))
        #%% Store data
        print(" Storing data...", end="")
        featuredict = dict()
        for f in featurelist:
            featuredict['features'+str(f.shape[1])] = f
        np.savez_compressed(cache_fn, N=sc_N, p=sc_p, q=sc_q, authors=authors, 
                            cluster_centers=cluster_centers, codebooks=codebooks,
                            rotationalinvariant = sc_rotationalinvariant, **featuredict)
        print("done")
        print(" Removing lock file...", end="")
        os.remove(cache_fn+".lock")
        print("done")
    else:
        if os.path.exists(cache_fn):
            print(" Cached features found, %s" % cache_fn)
        if os.path.exists(cache_fn+".lock"):
            print(" Lock file found, %s" % (cache_fn+".lock"))
        
print("Shutting down sparkContext...", end="")
spark.stop()
print("done")
