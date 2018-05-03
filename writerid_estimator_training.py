# -*- coding: utf-8 -*-
"""
Writer identification based of shape context codebook features and 
a shared kernel GP classifier. This file covers the training and evaluation
of the estimators.

@author: Fredrik Wahlberg <fredrik.wahlberg@it.uu.se>
"""
from __future__ import print_function, division

import numpy as np
import os.path
from model import SharedKernelClassifier
import time


cache_path = os.path.expanduser("~/tmp/gp_multiclass")
def cache_filename_features(N, p, q):
    return os.path.join(cache_path, 
                        "features_N%i_p%i_q%i.npz" % (N, p, q))
def cache_filename_estimator(N, p, q):
    return os.path.join(cache_path, 
                        "estimator_N%i_p%i_q%i.npz" % (N, p, q))
def cache_filename_result_npz(N, p, q):
    return os.path.join(cache_path, 
                        "result_N%i_p%i_q%i.npz" % (N, p, q))
def cache_filename_result_csv(N, p, q):
    return os.path.join(cache_path, 
                        "result_N%i_p%i_q%i.csv" % (N, p, q))

parameter_distribution = {
        'p': list(range(2, 6+1)),
        'q': list(range(4, 9+1)),
        'N': list(range(7, 21+1, 2))}

from sklearn.model_selection import ParameterGrid
valid_parameters = [p for p in ParameterGrid(parameter_distribution) if p['N']>p['p']*2]
print("%i parameter configurations" % len(valid_parameters))

feature_files = [cache_filename_features(N=pa['N'], p=pa['p'], q=pa['q']) 
                 for pa in valid_parameters 
                 if os.path.exists(cache_filename_features(N=pa['N'], p=pa['p'], q=pa['q']))]
import random
random.shuffle(valid_parameters)


#%%
csv_categories = ['Dataset', 'Training authors', 'Training documents', 
                  'Testing authors', 'Testing documents', 'Kernel', 'ARD', 
                  'Codebook size', 'N', 'p', 'q', 'Hard 5-top', 'Hard 3-top', 
                  '1-top', 'Soft 3-top', 'Soft 5-top', 'Likelihood']
with open(os.path.join(cache_path, "00categories.csv"), 'w') as f:
    for i, cat in enumerate(csv_categories):
        f.write('"' + cat + '"')
        if i==len(csv_categories)-1:
            f.write("\n")
        else:
            f.write(";")

for feature_file in feature_files:
    # Read feature data file
    cache_file_data = np.load(feature_file)
    print(cache_file_data.keys())
    data = dict()
    for k in cache_file_data.keys():
        if k in ['cluster_centers', 'codebooks', 'N', 'p', 'q', 'rotationalinvariant', 'authors']:
            data[k] = cache_file_data[k].tolist()
        else:
            data[k] = cache_file_data[k]
    feature_keys = [k for k in data.keys() if k[:8]=='features']
    # Set up output and check for old results
    cvs_filename = cache_filename_result_csv(N=data['N'], p=data['p'], q=data['q'])
    if not os.path.exists(cvs_filename):
        csv_file = open(cvs_filename, 'w')
        # CVL authors are id>1000
        authors = data['authors']

        cvl_mask = [a>1000 for a in authors]
        cvl_train_mask = [a>1000 and a<=1050 for a in authors]
        cvl_test_mask = [a>1000 and a>1050 for a in authors]
        cvl_y_train = np.vstack([a for a,b in zip(authors, cvl_train_mask) if b])
        assert np.sum(cvl_train_mask)==len(cvl_y_train)
        cvl_y_test = np.vstack([a for a,b in zip(authors, cvl_test_mask) if b])
        assert np.sum(cvl_test_mask)==len(cvl_y_test)
        
        iam_mask = [a<1000 for a in authors]
        iam_train_mask = [a<1000 and a<=26 for a in authors]
        iam_test_mask = [a<1000 and a>26 for a in authors]
        iam_y_train = np.vstack([a for a,b in zip(authors, iam_train_mask) if b])
        assert np.sum(iam_train_mask)==len(iam_y_train)
        iam_y_test = np.vstack([a for a,b in zip(authors, iam_test_mask) if b])
        assert np.sum(iam_test_mask)==len(iam_y_test)

        cvl_iam_train_mask = [a or b for a, b in zip(cvl_train_mask, iam_train_mask)]
        cvl_iam_test_mask = [a or b for a, b in zip(cvl_test_mask, iam_test_mask)]
        assert np.sum(cvl_iam_train_mask) == np.sum(cvl_train_mask) + np.sum(iam_train_mask)
        assert np.sum(cvl_iam_test_mask) == np.sum(cvl_test_mask) + np.sum(iam_test_mask)
        cvl_iam_y_train = np.vstack([a for a,b in zip(authors, cvl_iam_train_mask) if b])
        assert np.sum(cvl_iam_train_mask)==len(cvl_iam_y_train)
        cvl_iam_y_test = np.vstack([a for a,b in zip(authors, cvl_iam_test_mask) if b])
        assert np.sum(cvl_iam_test_mask)==len(cvl_iam_y_test)
                
        # Model standard parameters
#        kernel='rbf'
        ard=False
        n_iter = 15
        n_restarts = 0
        
#        for feature_key in feature_keys[:1]:
        for feature_key in feature_keys:
    
            # Separate feature data sets
            cvl_X_train = data[feature_key][cvl_train_mask, :]
            cvl_X_test = data[feature_key][cvl_test_mask, :]
            assert cvl_X_train.shape[0] == np.sum(cvl_train_mask)
            assert cvl_X_test.shape[0] == np.sum(cvl_test_mask)
            iam_X_train = data[feature_key][iam_train_mask, :]
            iam_X_test = data[feature_key][iam_test_mask, :]
            assert iam_X_train.shape[0] == np.sum(iam_train_mask)
            assert iam_X_test.shape[0] == np.sum(iam_test_mask)
            cvl_iam_X_train = data[feature_key][cvl_iam_train_mask, :]
            cvl_iam_X_test = data[feature_key][cvl_iam_test_mask, :]
            assert cvl_iam_X_train.shape[0] == np.sum(cvl_iam_train_mask)
            assert cvl_iam_X_test.shape[0] == np.sum(cvl_iam_test_mask)

            #%% CVL Estimator
            for kernel in ['matern52', 'rbf']:
#                for ard in [True, False]:
                print("Kernel %s " % kernel, end="")
                if ard:
                    print("with ard")
                else:
                    print("without ard")
                cvl_estimator = SharedKernelClassifier(kernel=kernel, ard=ard, n_iter=n_iter, 
                                                   n_restarts=n_restarts, verbose=True)
                print("Training CVL estimator...", end="")
                t0 = time.time()
                cvl_estimator.fit(cvl_X_train, cvl_y_train)
                print("done (%.1fs)" % (time.time()-t0))
                
                print("CVL 1-top training set: %.1f%%" % (cvl_estimator.score_covar_ntop(cvl_X_train, cvl_y_train)[4]*100))
                ntop = cvl_estimator.score_covar_ntop(cvl_X_test, cvl_y_test)
                print("CVL 1-top test set: %.1f%%" % (ntop[4]*100))
                print("Likelihood %.2f" % cvl_estimator.log_likelihood_)
                ntop2 = cvl_estimator.score_covar_ntop(iam_X_test, iam_y_test)
                print("CVL on IAM 1-top test set: %.1f%%" % (ntop2[4]*100))
        
                dataset = 'CVL'
                d = (dataset, len(np.unique(cvl_y_train)), len(cvl_y_train), 
                     len(np.unique(cvl_y_test)), len(cvl_y_test), kernel, ard, 
                     cvl_X_train.shape[1], data['N'], data['p'], data['q'], ntop[0]*100, 
                     ntop[2]*100, ntop[4]*100, ntop[6]*100, ntop[8]*100, cvl_estimator.log_likelihood_)
                csv_file.write("\"%s\";%i;%i;%i;%i;\"%s\";\"%s\";%i;%i;%i;%i;%f;%f;%f;%f;%f;%f\n" % d)
    
                dataset = 'CVL on IAM'
                d = (dataset, len(np.unique(cvl_y_train)), len(cvl_y_train), 
                     len(np.unique(iam_y_test)), len(iam_y_test), kernel, ard, 
                     cvl_X_train.shape[1], data['N'], data['p'], data['q'], ntop2[0]*100, 
                     ntop2[2]*100, ntop2[4]*100, ntop2[6]*100, ntop2[8]*100, cvl_estimator.log_likelihood_)
                csv_file.write("\"%s\";%i;%i;%i;%i;\"%s\";\"%s\";%i;%i;%i;%i;%f;%f;%f;%f;%f;%f\n" % d)

            #%% IAM Estimator
            iam_estimator = SharedKernelClassifier(kernel=kernel, ard=ard, n_iter=n_iter, 
                                               n_restarts=n_restarts, verbose=True)
            print("Training IAM estimator...", end="")
            t0 = time.time()
            iam_estimator.fit(iam_X_train, iam_y_train)
            print("done (%.1fs)" % (time.time()-t0))
            
            print("IAM 1-top training set: %.1f%%" % (iam_estimator.score_covar_ntop(iam_X_train, iam_y_train)[4]*100))
            ntop = iam_estimator.score_covar_ntop(iam_X_test, iam_y_test)
            print("IAM 1-top test set: %.1f%%" % (ntop[4]*100))
            print("Likelihood %.2f" % iam_estimator.log_likelihood_)
            ntop2 = iam_estimator.score_covar_ntop(cvl_X_test, cvl_y_test)
            print("IAM on CVL 1-top test set: %.1f%%" % (ntop2[4]*100))
    
            dataset = 'IAM'
            d = (dataset, len(np.unique(iam_y_train)), len(iam_y_train), 
                 len(np.unique(iam_y_test)), len(iam_y_test), kernel, ard, 
                 iam_X_train.shape[1], data['N'], data['p'], data['q'], ntop[0]*100, 
                 ntop[2]*100, ntop[4]*100, ntop[6]*100, ntop[8]*100, iam_estimator.log_likelihood_)
            csv_file.write("\"%s\";%i;%i;%i;%i;\"%s\";\"%s\";%i;%i;%i;%i;%f;%f;%f;%f;%f;%f\n" % d)

            dataset = 'IAM on CVL'
            d = (dataset, len(np.unique(iam_y_train)), len(iam_y_train), 
                 len(np.unique(cvl_y_test)), len(cvl_y_test), kernel, ard, 
                 iam_X_train.shape[1], data['N'], data['p'], data['q'], ntop2[0]*100, 
                 ntop2[2]*100, ntop2[4]*100, ntop2[6]*100, ntop2[8]*100, iam_estimator.log_likelihood_)
            csv_file.write("\"%s\";%i;%i;%i;%i;\"%s\";\"%s\";%i;%i;%i;%i;%f;%f;%f;%f;%f;%f\n" % d)

    #        b = estimator.score_unseen_proba_metric(cvl_X_test, cvl_y_test)

            #%% CVL+IAM Estimator
            cvl_iam_estimator = SharedKernelClassifier(kernel=kernel, ard=ard, n_iter=n_iter, 
                                               n_restarts=n_restarts, verbose=True)
            print("Training CVL+IAM estimator...", end="")
            t0 = time.time()
            cvl_iam_estimator.fit(cvl_iam_X_train, cvl_iam_y_train)
            print("done (%.1fs)" % (time.time()-t0))
            
            print("CVL+IAM 1-top training set: %.1f%%" % (cvl_iam_estimator.score_covar_ntop(cvl_iam_X_train, cvl_iam_y_train)[4]*100))
            ntop = cvl_iam_estimator.score_covar_ntop(cvl_iam_X_test, cvl_iam_y_test)
            ntop2 = cvl_iam_estimator.score_covar_ntop(cvl_X_test, cvl_y_test)
            ntop3 = cvl_iam_estimator.score_covar_ntop(iam_X_test, iam_y_test)
            print("CVL+IAM 1-top test set: %.1f%%" % (ntop[4]*100))
            print("CVL+IAM on CVL 1-top test set: %.1f%%" % (ntop2[4]*100))
            print("CVL+IAM on IAM 1-top test set: %.1f%%" % (ntop3[4]*100))
            print("Likelihood %.2f" % cvl_iam_estimator.log_likelihood_)

            dataset = 'CVL+IAM'
            d = (dataset, len(np.unique(cvl_iam_y_train)), len(cvl_iam_y_train), 
                 len(np.unique(cvl_iam_y_test)), len(cvl_iam_y_test), kernel, ard, 
                 cvl_iam_X_train.shape[1], data['N'], data['p'], data['q'], ntop[0]*100, 
                 ntop[2]*100, ntop[4]*100, ntop[6]*100, ntop[8]*100, cvl_iam_estimator.log_likelihood_)
            csv_file.write(("\"%s\";%i;%i;%i;%i;\"%s\";\"%s\";%i;%i;%i;%i;%f;%f;%f;%f;%f;%f\n" % d).replace('.', ','))

            dataset = 'CVL+IAM on CVL'
            d = (dataset, len(np.unique(cvl_iam_y_train)), len(cvl_iam_y_train), 
                 len(np.unique(cvl_y_test)), len(cvl_y_test), kernel, ard, 
                 cvl_iam_X_train.shape[1], data['N'], data['p'], data['q'], ntop2[0]*100, 
                 ntop2[2]*100, ntop2[4]*100, ntop2[6]*100, ntop2[8]*100, cvl_iam_estimator.log_likelihood_)
            csv_file.write(("\"%s\";%i;%i;%i;%i;\"%s\";\"%s\";%i;%i;%i;%i;%f;%f;%f;%f;%f;%f\n" % d).replace('.', ','))

            dataset = 'CVL+IAM on IAM'
            d = (dataset, len(np.unique(cvl_iam_y_train)), len(cvl_iam_y_train), 
                 len(np.unique(iam_y_test)), len(iam_y_test), kernel, ard, 
                 cvl_iam_X_train.shape[1], data['N'], data['p'], data['q'], ntop3[0]*100, 
                 ntop3[2]*100, ntop3[4]*100, ntop3[6]*100, ntop3[8]*100, cvl_iam_estimator.log_likelihood_)
            csv_file.write(("\"%s\";%i;%i;%i;%i;\"%s\";\"%s\";%i;%i;%i;%i;%f;%f;%f;%f;%f;%f\n" % d).replace('.', ','))
#            csv_file.write("\"%s\";%i;%i;%i;%i;\"%s\";\"%s\";%i;%i;%i;%i;%f;%f;%f;%f;%f;%f\n" % d)

            if feature_key==feature_keys[-1]:
                np.savez_compressed(cache_filename_result_npz(N=data['N'], p=data['p'], q=data['q']), 
                                covariance_matrix = np.asarray(cvl_iam_estimator.get_kernel()(cvl_iam_X_test), dtype=np.float16),
                                labels = cvl_iam_y_test.ravel(),
                                likelihood = cvl_iam_estimator.log_likelihood_)
            
        csv_file.close()
    else:
        print("File already exists: %s" % cvs_filename)

