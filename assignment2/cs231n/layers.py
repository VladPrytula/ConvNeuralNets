import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)

  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  N = x.shape[0]
  out = np.dot(x.reshape(N,-1), w) + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  dx = np.dot(dout, w.T).reshape(x.shape)
  dw = np.dot(x.reshape(x.shape[0],-1).T, dout)
  db = np.dot(np.ones(dx.shape[0]),dout)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.maximum(0,x)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  mask = x>0
  dx = mask*dout
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.

  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    # Mini-batch mean and variance
    mu = np.mean(x, axis = 0)
    sigma2 = np.var(x, axis = 0)

    # normalize
    x_normalized = (x - mu)/np.sqrt(sigma2 + eps)

    # scale and shift
    out = gamma*x_normalized + beta

    running_mean = momentum * running_mean + (1.0 - momentum) * mu
    running_var = momentum * running_var + (1.0 - momentum) * sigma2

    # TODO: must store the cache also
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    # normalize with running statistics
    x_normalized = (x - running_mean)/np.sqrt(running_var+ eps)

    # scale and shift
    out = gamma*x_normalized + beta
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  # this is for a dummy backward pass. Ugly
  # Forward pass
  # Step 1 - shape of mu (D,)
  mu = 1 / float(N) * np.sum(x, axis=0)

  # Step 2 - shape of var (N,D)
  xmu = x - mu

  # Step 3 - shape of carre (N,D)
  carre = xmu**2

  # Step 4 - shape of var (D,)
  var = 1 / float(N) * np.sum(carre, axis=0)

  # Step 5 - Shape sqrtvar (D,)
  sqrtvar = np.sqrt(var + eps)

  # Step 6 - Shape invvar (D,)
  invvar = 1. / sqrtvar

  # Step 7 - Shape va2 (N,D)
  va2 = xmu * invvar

  # Step 8 - Shape va3 (N,D)
  va3 = gamma * va2

  # Step 9 - Shape out (N,D)
  # not needed
  cache = (mu, xmu, carre, var, sqrtvar, invvar,
           va2, va3, gamma, beta, x, bn_param)
  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.

  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.

  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.

  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  #unfold the variables stored in cache
  mu, xmu, carre, var, sqrtvar, invvar, va2, va3, gamma, beta, x, bn_param = cache
  eps = bn_param.get('eps', 1e-5)
  N, D = dout.shape

  # Backprop Step 9
  dva3 = dout
  dbeta = np.sum(dout, axis=0)

  # Backprop step 8
  dva2 = gamma * dva3
  dgamma = np.sum(va2 * dva3, axis=0)

  # Backprop step 7
  dxmu = invvar * dva2
  dinvvar = np.sum(xmu * dva2, axis=0)

  # Backprop step 6
  dsqrtvar = -1. / (sqrtvar**2) * dinvvar

  # Backprop step 5
  dvar = 0.5 * (var + eps)**(-0.5) * dsqrtvar

  # Backprop step 4
  dcarre = 1 / float(N) * np.ones((carre.shape)) * dvar

  # Backprop step 3
  dxmu += 2 * xmu * dcarre

  # Backprop step 2
  dx = dxmu
  dmu = - np.sum(dxmu, axis=0)

  # Basckprop step 1
  dx += 1 / float(N) * np.ones((dxmu.shape)) * dmu

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.

  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.

  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.

  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  mu, xmu, carre, var, sqrtvar, invvar, va2, va3, gamma, beta, x,\
      bn_param = cache
  eps = bn_param.get('eps', 1e-5)
  N, D = dout.shape

  dbeta = np.sum(dout, axis=0)
  dgamma = np.sum((x - mu) * (var + eps)**(-1. / 2.) * dout, axis=0)
  dx = (1. / N) * gamma * (var + eps)**(-1. / 2.) * \
      (N * dout - np.sum(dout, axis=0)\
       - (x - mu) * (var + eps)**(-1.0) * np.sum(dout * (x - mu), axis=0))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    # forward pass for example 3-layer neural network
    mask = (np.random.rand(*x.shape) < p) / p # first dropout mask. Notice /p!
    out = x*mask # drop!
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']

  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    dx = dout*mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  stride = conv_param['stride']
  pad = conv_param['pad']
  N, C, H, W = x.shape
  F, _, HH, WW = w.shape

  npad = ((0,0),(0,0),(pad,pad),(pad,pad)) # we are padding only H, and W
  padded_input = np.pad(x, pad_width=npad, mode='constant', constant_values=0)

  # define output height and width based on the formulas in docstring
  h_out =  1 + (H + 2 * pad - HH) / stride
  w_out =  1 + (W + 2 * pad - HH) / stride
  out = np.zeros((N,F,h_out,w_out))
  # print str(h_out) + "  h_out"
  # print str(out.shape) + "    shape"
  # print str(w_out) + "  w_out"
  # print str(stride) + "  stride"
  # print str(F) + "  F"

  # TODO: it looks like I have to itereate over the number of filters also. I do
  # not know how to make it looplsess without im2col implementation
  for n_step in np.arange(N):
    for f in np.arange(F):
      h_index=0
      for h_step in np.arange(H,step=stride):
        w_index=0
        for w_step in np.arange(W,step=stride):
          out[n_step,f,h_index,w_index]=\
              np.sum(padded_input[n_step,:,h_step:h_step+HH,\
                                  w_step:w_step+WW]*w[f,:,:,:])+b[f]
          w_index+=1
        h_index+=1
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)

  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
#  x, w, b, conv_param = cache
#  N, C, H, W = x.shape
#  F, _, HH, WW = w.shape
#  (_, _, H_out, W_out) = dout.shape
#
#  dx = np.zeros_like(x)
#  dw = np.zeros_like(w)
#  db = np.zeros_like(b)
#
#  stride = conv_param['stride']
#  pad = conv_param['pad']
#
#  # it is important to know padding here since in cache we have an original,
#  # non-padded version of x. So we have to padd it again, to be consistent with
#  # simple formulas
#
#  npad = ((0,0),(0,0),(pad,pad),(pad,pad)) # we are padding only H, and W
#  x_padded  = np.pad(x, pad_width=npad, mode='constant', constant_values=0)
#  dx_padded = np.pad(dx, pad_width=npad, mode='constant', constant_values=0)
#
#
#
#  for n_step in np.arange(N):
#    for f in np.arange(F):
#      for h_step in np.arange(H_out, step=stride):
#        for w_step in np.arange(W_out, step=stride):
#          print "w  " + str(w.shape)
#          print "dout    " + str(dout.shape)
#          dx_padded[n_step,:,h_step:h_step+HH,:w_step+WW]+=\
#              w[f,:,:,:]*dout[n_step,f,h_step,w_step]
#          dw[n_step,:,:,:]+=dout[n_step,f,h_step,w_step]*\
#              x_padded[n_step,:,h_step:h_step+HH, w_step:w_step+WW]
#          db[f] += dout[n_step,f,h_step,w_step]
#
#  dx = dx_padded[:,:,1:-1,1:-1]
#
  (x, w, b, conv_param) = cache
  (N, C, H, W) = x.shape
  (F, _, HH, WW) = w.shape
  (_, _, H_out, W_out) = dout.shape
  stride = conv_param['stride']
  pad = conv_param['pad']

  dx = np.zeros_like(x)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)
  for n in np.arange(N):
    dx_pad = np.pad(dx[n,:,:,:], ((0,0),(pad,pad),(pad,pad)), 'constant')
    x_pad = np.pad(x[n,:,:,:], ((0,0),(pad,pad),(pad,pad)), 'constant')
    for f in np.arange(F):
      for h_step in np.arange(H_out, step=stride):
        for w_step in np.arange(W_out,step=stride):
          h1 = h_step
          h2 = h_step + HH
          w1 = w_step
          w2 = w_step + WW
          dx_pad[:, h1:h2, w1:w2] += w[f,:,:,:] * dout[n,f,h_step,w_step]
          dw[f,:,:,:] += x_pad[:, h1:h2, w1:w2] * dout[n,f,h_step,w_step]
          db[f] += dout[n,f,h_step,w_step]
    dx[n,:,:,:] = dx_pad[:,1:-1,1:-1]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  (N, C, H, W) = x.shape
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']
  H_prime = 1 + (H - pool_height) / stride
  W_prime = 1 + (W - pool_width) / stride

  out = np.zeros((N, C, H_prime, W_prime))

  for n in xrange(N):
    for h in xrange(H_prime):
      for w in xrange(W_prime):
        h1 = h * stride
        h2 = h * stride + pool_height
        w1 = w * stride
        w2 = w * stride + pool_width
        window = x[n, :, h1:h2, w1:w2]
        out[n,:,h,w] = np.max(window.reshape((C, pool_height*pool_width)), axis=1)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x, pool_param = cache
  Hp = pool_param['pool_height']
  Wp = pool_param['pool_width']
  S = pool_param['stride']
  N, C, H, W = x.shape
  H1 = (H - Hp) / S + 1
  W1 = (W - Wp) / S + 1

  dx = np.zeros((N, C, H, W))
  for nprime in range(N):
      for cprime in range(C):
          for k in range(H1):
              for l in range(W1):
                  x_pooling = x[nprime, cprime, k *
                                S:k * S + Hp, l * S:l * S + Wp]
                  maxi = np.max(x_pooling)
                  x_mask = x_pooling == maxi
                  dx[nprime, cprime, k * S:k * S + Hp, l * S:l *
                      S + Wp] += dout[nprime, cprime, k, l] * x_mask
  return dx
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.

  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.

  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass

  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
