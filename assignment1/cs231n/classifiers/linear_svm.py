import numpy as np

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin

        # print(X[i,:].shape)
        # print(dW[j,:].shape)
        # print(dW.shape)
        dW[:,y[i]] -= X[i,:]
        dW[:,j]    += X[i,:]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  # Same normalization must be applied to the gradient
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  num_train = X.shape[0]
  num_classes = W.shape[1]

  # print("inside svm_loss_vectorized. shpe of W " + str(W.shape))
  scores = X.dot(W)

  # this is funny Pyhon bididrectional indexing
  true_scores = scores[np.arange(num_train),y]

  # now we have to compute the difference, again we do it substracting vector
  # from matrix
  difference = scores - np.tile(true_scores.reshape(num_train,-1),num_classes) + 1
  # here we should not forget that we are counting one term that must be
  # excluded. We must also take max and sum in direction of axis = 1 (in reality
  # must be summed in both directions, i.e. in i and j

  # let us take the maximum first
  max_diff_matrix = np.maximum(difference, np.zeros((num_train,num_classes)))

  # now we have to sum and substract 1

  l_i_representation_with_one = np.sum(max_diff_matrix, axis = 1)
  l_i_representation = l_i_representation_with_one - 1

  loss = np.sum(l_i_representation)
  loss /= num_train

  # add regularization term

  loss +=  0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # here it is better (??) not to go with substracting one , but explicitely
  # remove the terms using the bidirectional indexing in 2d array
  difference[np.arange(num_train),y] = 0.

  # now we can take the maximum
  max_diff_matrix = np.maximum(difference, np.zeros((num_train,num_classes)))

  # now we have to create a boolean(binary) map to mark those positions in the
  # matrix that are greater than zero
  # we set all the values that are not zero (i.e grater than zero) to 1, rest
  # remains 0
  max_diff_matrix[max_diff_matrix > 0] = 1
  # print(max_diff_matrix.shape)
  # print(max_diff_matrix)


  # so now we have a matrix that has 1s in places where terms of L_{i} are
  # greater than zero and rest in all the other places
  # next step is to represent the desired Loss function /gothic{L} as a dot
  # product of this matrix and the X matrix
  # first let us take into account that we have explicitely "eliminated" the
  # "right class case" from the matrix. For the gradients with respect to any
  # w_j where j \neq y_i it would be a simple x_i. For the special case we have
  # to sum over all ones that we have (no exlusion is required here)
  # Let us do summing and store the result in a vector
  # Since the dim of our matrix is (500,10) where 10 is the number of classes we
  # have to do summation over the row.
  class_summ = np.sum(max_diff_matrix, axis = 1)
  # print(class_summ.shape)


  # now we have to put the sums with the negative sign in correct palces. Those
  # would be the ones indexed withour "correct classs vector" y (again
  # bidirectional indexing)
  max_diff_matrix[np.arange(num_train), y] = - class_summ[np.arange(num_train)]
  # print(max_diff_matrix)
  # print(max_diff_matrix.shape)
  # print(X.shape)

  dW =np.dot(X.T,max_diff_matrix)
  # print(dW.shape)


    # Divide
  dW /= num_train

  # Regularize
  dW += reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
