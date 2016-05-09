import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # first let us define the dimentions for the loops
  # number of classes, i.e. L_i, i \in [1,C]
  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores_matrix = np.dot(X,W) # dims are (N,C). compute once and for all
  # due to exponentials there is a huge risk of numerical blow-up
  # in order to avoid this we have to shift all the values so that the highest
  # number is 0
  stabilizer = np.amax(scores_matrix, axis=1)
  scores_matrix -= stabilizer.reshape(num_train, -1)
  exp_xv = np.exp(scores_matrix)
  exp_xw_sum = np.sum(exp_xv, axis = 1)

  for i in xrange(num_train):
    correct_class_score = scores_matrix[i,y[i]]
    loss += - correct_class_score + np.log(exp_xw_sum[i])

    for j in xrange(num_classes):
        dW[:,j] += 1.0/(exp_xw_sum[i])*exp_xv[i,j]*X[i,:]
        if j == y[i]:
            dW[:,j] -= X[i,:]

  #Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  # Same normalization must be applied to the gradient
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores_matrix = np.dot(X,W) # dims are (N,C). compute once and for all
  exp_xv = np.exp(scores_matrix)
  exp_xw_sum = np.sum(exp_xv, axis = 1)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

