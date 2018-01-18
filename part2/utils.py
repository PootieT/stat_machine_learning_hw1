from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from reg_linear_regressor_multi import RegularizedLinearReg_SquaredLoss
import plot_utils

#############################################################################
#  Normalize features of data matrix X so that every column has zero        #
#  mean and unit variance                                                   #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     Output: mu: D x 1 (mean of X)                                         #
#          sigma: D x 1 (std dev of X)                                      #
#         X_norm: N x D (normalized X)                                      #
#############################################################################

def feature_normalize(X):

    ########################################################################
    # TODO: modify the three lines below to return the correct values
    n = np.shape(X)[0]
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - np.tile(mu, (n,1))) / np.tile(sigma, (n,1))
  
    ########################################################################
    return X_norm, mu, sigma


#############################################################################
#  Plot the learning curve for training data (X,y) and validation set       #
# (Xval,yval) and regularization lambda reg.                                #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     y: vector of length N which are values corresponding to X             #
#     Xval: M x D where N is the number of rows and D is the number of      #
#           features                                                        #
#     yval: vector of length N which are values corresponding to Xval       #
#     reg: regularization strength (float)                                  #
#     Output: error_train: vector of length N-1 corresponding to training   #
#                          error on training set                            #
#             error_val: vector of length N-1 corresponding to error on     #
#                        validation set                                     #
#############################################################################

def learning_curve(X,y,Xval,yval,reg):
    num_examples,dim = X.shape
    error_train = np.zeros((num_examples,))
    error_val = np.zeros((num_examples,))
    
    ###########################################################################
    # TODO: compute error_train and error_val                                 #
    # 7 lines of code expected                                                #
    ###########################################################################
    for i in range(1,13):
        model = RegularizedLinearReg_SquaredLoss()
        theta = model.train(X[:i],y[:i],reg=reg,num_iters=1000)
        error_train[i-1] = model.loss(theta,X[:i],y[:i],0)
        error_val[i-1] = model.loss(theta,Xval,yval,0)
    ###########################################################################

    return error_train, error_val

#############################################################################
#  Plot the validation curve for training data (X,y) and validation set     #
# (Xval,yval)                                                               #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     y: vector of length N which are values corresponding to X             #
#     Xval: M x D where N is the number of rows and D is the number of      #
#           features                                                        #
#     yval: vector of length N which are values corresponding to Xval       #
#                                                                           #
#     Output: error_train: vector of length N-1 corresponding to training   #
#                          error on training set                            #
#             error_val: vector of length N-1 corresponding to error on     #
#                        validation set                                     #
#############################################################################

def validation_curve(X,y,Xval,yval):
  
    reg_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    error_train = np.zeros((len(reg_vec),))
    error_val = np.zeros((len(reg_vec),))

    ###########################################################################
    # TODO: compute error_train and error_val                                 #
    # 5 lines of code expected                                                #
    ###########################################################################
    for i, reg in enumerate(reg_vec):
        model = RegularizedLinearReg_SquaredLoss()
        theta = model.train(X,y,reg=reg,num_iters=1000)
        error_train[i] = model.loss(theta,X,y,0)
        error_val[i] = model.loss(theta,Xval,yval,0)

    return reg_vec, error_train, error_val

import random

#############################################################################
#  Plot the averaged learning curve for training data (X,y) and             #
#  validation set  (Xval,yval) and regularization lambda reg.               #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     y: vector of length N which are values corresponding to X             #
#     Xval: M x D where N is the number of rows and D is the number of      #
#           features                                                        #
#     yval: vector of length N which are values corresponding to Xval       #
#     reg: regularization strength (float)                                  #
#     Output: error_train: vector of length N-1 corresponding to training   #
#                          error on training set                            #
#             error_val: vector of length N-1 corresponding to error on     #
#                        validation set                                     #
#############################################################################

def averaged_learning_curve(X,y,Xval,yval,reg):
    num_examples,dim = X.shape
    error_train = np.zeros((num_examples,))
    error_val = np.zeros((num_examples,))

    ###########################################################################
    # TODO: compute error_train and error_val                                 #
    # 10-12 lines of code expected                                            #
    ###########################################################################
    for i in range(1,13):
        error_t = []
        error_v = []
        for _ in range(50):
            arr = np.arange(len(X))
            np.random.shuffle(arr)
            arr = arr[:i+1]
            Xi, yi, Xvali, yvali = np.squeeze(X[arr]), np.squeeze(y[arr]), np.squeeze(Xval[arr]), np.squeeze(yval[arr])
            print Xi.shape, yi.shape, Xvali.shape, yvali.shape
            print arr
            model = RegularizedLinearReg_SquaredLoss()
            theta = model.train(Xi,yi,reg=reg,num_iters=1000)
            error_t.append(model.loss(theta,np.array(Xi),np.array(yi),0))
            error_v.append(model.loss(theta,Xvali,yvali,0))
        print arr
        error_train[i-1] = np.mean(np.array(error_t))
        error_val[i-1] = np.mean(np.array(error_v))


    ###########################################################################
    return error_train, error_val


#############################################################################
# Utility functions
#############################################################################
    
def load_mat(fname):
  d = scipy.io.loadmat(fname)
  X = d['X']
  y = d['y']
  Xval = d['Xval']
  yval = d['yval']
  Xtest = d['Xtest']
  ytest = d['ytest']

  # need reshaping!

  X = np.reshape(X,(len(X),))
  y = np.reshape(y,(len(y),))
  Xtest = np.reshape(Xtest,(len(Xtest),))
  ytest = np.reshape(ytest,(len(ytest),))
  Xval = np.reshape(Xval,(len(Xval),))
  yval = np.reshape(yval,(len(yval),))

  return X, y, Xtest, ytest, Xval, yval









