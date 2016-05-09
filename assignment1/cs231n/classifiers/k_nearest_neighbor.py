# -*- coding: utf-8 -*-
import numpy as np
from collections import Counter

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y

  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      for j in xrange(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
        # pass
        # the other option would be to use linalg
        dists[i, j] = np.linalg.norm(self.X_train[j] - X[i])
        # dists[i,j] = np.sqrt(np.sum(np.square(self.X_train[i] - X[j])))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      #pass
      dists[i, :] = np.linalg.norm(self.X_train - X[i, :], axis=1)
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    # we have to expand the expression and calculate by parts
    # let us look in more details at how dists[i,j] looks like
    # for the sake of visual simplicity let us forget about sqrt for now
    # (X[i] - Y[j])^2 = (X[i]-Y[j])^t\star(X[i]-Y[j]) =
    # these are vectors! not matrices.
    # we must remember that i in [0,5000), j in [0,500)
    # and dists has dimentions (500, 5000)
    # for any arbitrary element dists[i,j] we have the following expantion
    # FOr now we forget about the sqrt for the sake of brevety
    #
    #
    # let us try to represnet this in a matrix notattion
    # meedle term is equal to
    # np.dot(X_j, Y_j.T)
    # effectively \forall i,j we can reperesent it as
    # np.dot(Y, X.T) this would have dimentions (500, 5000)
    #
    # Now how can we represent the other parts:
    # np.sum(X**2, axis = 1) has dimentions (5000,)
    # np.sum(Y**2, axis = 1) has dimentions (500, )
    # we can broadcast array of dim (5000,) during summing with (500,5000) but
    # we have to modify the dimentionality of the array (500,) to be abble to
    # add it to the np.dot(Y, X.T)
    # in order to do so we reshape it np.sum(Y**2, axis = 1).reshape(-1,1)
    # and get the array of dim (500,1)
    #
    dot_product = X.dot(self.X_train.T)
    X_train2 = np.sum(self.X_train ** 2, axis=1)
    X2 = np.sum(X ** 2, axis=1)
    dists = np.sqrt(-2*dot_product + X_train2 + X2.reshape(-1, 1))
    #
    #
    #
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      k_nearest = np.argsort(dists[i])[ :k]  # I am not sure of this
      # dimention is (500,2)
      # arsort returns the Indices that would sort the array
      # the way aroun without argsort would be
      # [i[0] for i in sorted(enumerate(myList), key=lambda x:x[1])]
      #
      closest_y = self.y_train[k_nearest]
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      y_pred[i] = Counter(closest_y).most_common(1)[0][0]
      #########################################################################
      #                           END OF YOUR CODE                            #
      #########################################################################

    return y_pred
