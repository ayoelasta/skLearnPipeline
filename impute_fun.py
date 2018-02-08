import numpy as np
import random
from sklearn import datasets
from sklearn import neighbors

def impute(mat, learner, n_iter=3):
    mat = np.array(mat)
    mat_isnan = np.isnan(mat)        
    w = np.where(np.isnan(mat))
    ximp = mat.copy()
    for i in range(0, len(w[0])):
        n = w[0][i] # row where the nan is
        p = w[1][i] # column where the nan is
        col_isnan = mat_isnan[n, :] # empty columns in row n
        train = np.delete(mat, n, axis = 0) # remove row n to obtain a training set
        train_nonan = train[~np.apply_along_axis(np.any, 1, np.isnan(train)), :] # remove rows where there is a nan in the training set
        target = train_nonan[:, p] # vector to be predicted
        feature = train_nonan[:, ~col_isnan] # matrix of predictors
        learn = learner.fit(feature, target) # learner
        ximp[n, p] = learn.predict(mat[n, ~col_isnan].reshape(1, -1)) # predict and replace
    for iter in range(0, n_iter):
        for i in random.sample(range(0, len(w[0])), len(w[0])):
            n = w[0][i] # row where the nan is
            p = w[1][i] # column where the nan is
            train = np.delete(ximp, n, axis = 0) # remove row n to obtain a training set
            target = train[:, p] # vector to be predicted
            feature = np.delete(train, p, axis=1) # matrix of predictors
            learn = learner.fit(feature, target) # learner
            ximp[n, p] = learn.predict(np.delete(ximp[n,:], p).reshape(1, -1)) # predict and replace
    
    return ximp

# Impute with learner in the iris data set
iris = datasets.load_iris()
mat = iris.data.copy()

# throw some nans
mat[0,2] = np.NaN
mat[0,3] = np.NaN
mat[1,3] = np.NaN
mat[11,1] = np.NaN
mat = mat[range(30), :]

# impute
impute(mat=mat, learner=neighbors.KNeighborsRegressor(n_neighbors=3), n_iter=10)