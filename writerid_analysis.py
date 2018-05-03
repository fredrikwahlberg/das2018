# -*- coding: utf-8 -*-

import os.path
import numpy as np
import time
import random
from model import SharedKernelClassifier

# Training requirements analysis
N = 19
p = 5
q = 8

cache_path = os.path.expanduser("~/tmp/gp_multiclass.old")
def cache_filename_features(N, p, q):
    return os.path.join(cache_path, 
                        "features_N%i_p%i_q%i.npz" % (N, p, q))

feature_file = cache_filename_features(N, p, q)
assert os.path.exists(feature_file)
cache_file_data = np.load(feature_file)
data = dict()
for k in cache_file_data.keys():
    if k in ['cluster_centers', 'codebooks', 'N', 'p', 'q', 'rotationalinvariant', 'authors']:
        data[k] = cache_file_data[k].tolist()
    else:
        data[k] = cache_file_data[k]
feature_keys = [k for k in data.keys() if k[:8]=='features']

authors = data['authors']

cvl_mask = [a>1000 for a in authors]
iam_mask = [a<1000 for a in authors]

# Separate feature data sets
feature_key = feature_keys[-1]
X = data[feature_key]
y = np.vstack(authors)

#%% Model standard parameters
n_iter = 30
n_restarts = 1
kernel='matern52'
ard=True

for datasource in ["cvl", "iam", "cvl-iam"]:
    filename = os.path.join(cache_path, "training_analaysis_" + datasource + ".npz")
    if not os.path.exists(filename + ".lock"):
        open(filename + ".lock", "w+").close()
        if datasource == "cvl":
            mask = cvl_mask
        elif datasource == "iam":
            mask = iam_mask
        elif datasource == "cvl-iam":
            mask = [a or b for a, b in zip(cvl_mask, iam_mask)]
            assert np.sum(mask) == len(authors)
        else:
            assert False, "name error"
        if os.path.exists(filename):
            data = np.load(filename)
            training_set = data['training_set'].tolist()
            test_set = data['test_set'].tolist()
            likelihoods = data['likelihoods'].tolist()
            ntops_test = data['ntops_test'].tolist()
            ntops_training = data['ntops_training'].tolist()
            separability = data['separability'].tolist()
            adjusted_mutual_info = data['adjusted_mutual_info'].tolist()
            adjusted_mutual_info2 = data['adjusted_mutual_info2'].tolist()
            thetas = data['thetas'].tolist()
            db_separation = data['db_separation'].tolist()
            assert len(training_set) == len(test_set)
            assert len(training_set) == len(likelihoods)
            assert len(training_set) == len(ntops_training)
            assert len(training_set) == len(ntops_test)
            assert len(training_set) == len(separability)
            assert len(training_set) == len(adjusted_mutual_info)
            assert len(training_set) == len(adjusted_mutual_info2)
            assert len(training_set) == len(thetas)
#            assert len(training_set) == len(db_separation)
        else:
            training_set = list()
            test_set = list()
            likelihoods = list()
            ntops_training = list()
            ntops_test = list()
            separability = list()
            adjusted_mutual_info = list()
            adjusted_mutual_info2 = list()
            thetas = list()
            db_separation = list()

        n_repetitions = 21
#        for n_training in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20]:
        for n_training in range(2, 20+1):
#        for n_training in [2, 3, 5, 8, 10, 14, 20]:
            if len(training_set) == 0:
                start = 0
            else:
                start = np.sum([len(ts)==n_training for ts in training_set])
            for reps in range(start, n_repetitions):
                training_authors = list(np.unique([a for i, a in enumerate(authors) if mask[i]]))
                training_authors = random.sample(training_authors, n_training)
                test_authors = list(np.unique([a for i, a in enumerate(authors) if mask[i] and a not in training_authors]))
                test_authors = random.sample(test_authors, (len(training_authors)+len(test_authors))//2)
                assert np.alltrue([a not in test_authors for a in training_authors]), "sets overlap"
                training_set.append(training_authors)
                test_set.append(test_authors)
    
                m = [a in training_authors for a in authors]
                X_train = X[m, :]
                y_train = y[m]
                assert len(np.unique(y_train)) == n_training
                m = [a in test_authors for a in authors]
                X_test = X[m, :]
                y_test = y[m]
                assert len(np.unique(y_test)) == len(test_authors)
                
                estimator = SharedKernelClassifier(kernel=kernel, ard=ard, 
                                                   n_iter=n_iter, n_restarts=n_restarts, 
                                                   verbose=False)
                print("Training estimator...", end="")
                t0 = time.time()
                estimator.fit(X_train, y_train)
                print("done (%.1fs)" % (time.time()-t0))
        
                print("Datasource is %s with %i authors (%i documents)" % (datasource, n_training, len(y_train)))
                print("1-top training set: %.1f%%" % (estimator.score_covar_ntop(X_train, y_train)[4]*100))
                ntops_training.append(estimator.score_covar_ntop(X_train, y_train))
                ntop = estimator.score_covar_ntop(X_test, y_test)
                ntops_test.append(ntop)
                print("1-top test set: %.1f%%" % (ntop[4]*100))
                print("Likelihood %.2f" % estimator.log_likelihood_)
                likelihoods.append(estimator.log_likelihood_)
                print("Theta: ", estimator.hyperparameters_)
                thetas.append(estimator.hyperparameters_)
#                assert False

                #%% Cluster metrics, max internal, min distance, homogeneity
                A = estimator.get_kernel()(X_test)
                A /= np.max(A)
                t = y_test.copy().ravel()
                assert len(t)==A.shape[0]
                classes = np.unique(t)
                min_affinity = np.zeros(len(classes))
                closest_other = np.zeros(len(classes))
                for i, cls in enumerate(classes):
                    # Min affinity within cluster
                    x1 = A[t==cls, :]
                    x1 = x1[:, t==cls]
                    min_affinity[i] = np.min(x1)
                    # Max affinity to any other cluster
                    x2 = A[t==cls, :]
                    x2 = x2[:, t!=cls]
                    closest_other[i] = np.max(x2)
                a = closest_other/min_affinity
                a = a[np.isfinite(a)]
                sep = np.sum(a<1)/len(closest_other)
                print("Separable: %.1f%%" % (sep*100))
                separability.append(sep)

                #%%
                from sklearn.cluster import AgglomerativeClustering
                clustering = AgglomerativeClustering(n_clusters=len(test_authors),
                                                     affinity='precomputed',
                                                     linkage='average')
                A = estimator.get_kernel()(X_test)
                A /= np.max(A)
                clustering.fit(1-A)
                from sklearn.metrics import adjusted_mutual_info_score
                amis = adjusted_mutual_info_score(y_test.ravel(), clustering.labels_)
                print("adjusted_mutual_info_score: %f" % amis)
                adjusted_mutual_info.append(amis)

                #%%
                clustering2 = AgglomerativeClustering(n_clusters=len(test_authors),
                                                     affinity='l1',
                                                     linkage='average')
                clustering2.fit(X_test)
                amis = adjusted_mutual_info_score(y_test.ravel(), clustering2.labels_)
                print("adjusted_mutual_info_score: %f" % amis)
                adjusted_mutual_info2.append(amis)

                #%% Try to tell CVL and IAM appart
                if datasource == "cvl-iam":
                    y_train_db_sep = np.vstack(np.asarray([a>1000 for a in y_train.ravel()], dtype=np.int))
                    y_test_db_sep = np.vstack(np.asarray([a>1000 for a in y_test.ravel()], dtype=np.int))
                    from sklearn.gaussian_process import GaussianProcessClassifier
                    if len(np.unique(y_train_db_sep))>1:
                        gpc = GaussianProcessClassifier()
                        gpc.fit(X_train, y_train_db_sep)
                        db_sep = gpc.score(X_test, y_test_db_sep)
                        print("Database separation: %.1f%%" % (db_sep*100))
                        db_separation.append(db_sep)
                
                #%% Save data
                np.savez_compressed(filename, training_set = training_set, 
                                    test_set = test_set,
                                    likelihoods = likelihoods, 
                                    ntops_training = ntops_training, 
                                    ntops_test = ntops_test, 
                                    separability=separability, 
                                    adjusted_mutual_info=adjusted_mutual_info,
                                    adjusted_mutual_info2=adjusted_mutual_info2,
                                    thetas=thetas,
                                    db_separation=db_separation)
        os.remove(filename + ".lock")



#%% Load the data for inspection
datasource = "cvl"
#datasource = "iam"
datasource = "cvl-iam"

filename = os.path.join(cache_path, "training_analaysis_" + datasource + ".npz")
if os.path.exists(filename):
    data = np.load(filename)
    training_set = data['training_set'].tolist()
    test_set = data['test_set'].tolist()
    likelihoods = data['likelihoods'].tolist()
    ntops_training = data['ntops_training'].tolist()
    ntops_test = data['ntops_test'].tolist()
    separability = data['separability'].tolist()
    adjusted_mutual_info = data['adjusted_mutual_info'].tolist()   
    adjusted_mutual_info2 = data['adjusted_mutual_info2'].tolist()   
    db_separation = data['db_separation'].tolist()   
    thetas = data['thetas'].tolist()   

[(len(ts), ntop[4]) for ts, ntop in zip(training_set, ntops_training)]
[(len(ts), ntop[4]) for ts, ntop in zip(training_set, ntops_test)]
[(len(ts), len(tes)) for ts, tes in zip(training_set, test_set)]
[(len(ts), ll) for ts, ll in zip(training_set, likelihoods)]
[(len(ts), sep) for ts, sep in zip(training_set, separability)]
[(len(ts), ami) for ts, ami in zip(training_set, adjusted_mutual_info)]
[(len(ts), ami) for ts, ami in zip(training_set, adjusted_mutual_info2)]
[(len(ts), ami) for ts, ami in zip(training_set, thetas)]
[(len(ts), ami) for ts, ami in zip(training_set, db_separation)]
ardmask = np.invert([np.sum(np.abs(theta[1] - np.asarray(theta)[1:])) < 1e-2 for ts, theta in zip(training_set, thetas)])

#%%

import matplotlib.pyplot as plt

var = [ntop[4] for ntop in ntops_test]
#var = adjusted_mutual_info
#var = separability

plt.figure()
x = np.asarray(np.unique([len(ts) for ts in training_set]), dtype=np.int)
y_med = np.zeros(x.shape)
y_hi = np.zeros(x.shape)
y_low = np.zeros(x.shape)
for i, e in enumerate(x):
    y_values = np.asarray([v for ts, v in zip(training_set, var) if len(ts)==e])
    y_med[i] = np.median(y_values)
    y_hi[i] = np.max(y_values)
    y_low[i] = np.min(y_values)
    
#plt.plot(x, y_med, 'b-', linewidth=1)
plt.plot(x, y_med, 'b.-', markersize=5, label="Median")
plt.fill(np.concatenate([x, x[::-1]]), np.concatenate([y_hi, y_low[::-1]]),
         alpha=.25, fc='b', ec='None', label='High/Low Interval')
plt.xlabel('# authors')
plt.ylabel('1-top accuracy')
#plt.ylabel('separability')
plt.xticks(list(range(np.min(x), np.max(x)+1)))
plt.legend(loc='lower right')
plt.show()
#plt.savefig('ntop_vs_n_authors.png')
#plt.savefig('sep_vs_n_authors.png')

#%%
x = np.asarray([ntop[4] for ntop in ntops_training])
y = np.exp(np.asarray(likelihoods))
plt.figure()
plt.plot(x, y, 'bx')
plt.xlabel('ntop')
plt.ylabel('lml')
plt.show()




#%% n-top with random features
A = np.random.uniform(0, 1, size=(len(authors), len(authors)))
for i in range(A.shape[0]):
    A[i, i] = 1
    for j in range(i, A.shape[0]):
        A[j, i] = A[i, j]
dummy_estimator = SharedKernelClassifier()
random_ntop = dummy_estimator._affinity_ntop(A, np.vstack(authors))
assert np.all(np.asarray(random_ntop) < 0.05)