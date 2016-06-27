import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    # The RNN class has some internal state that it gets to update every time
    # step is called.
    # In the simplest case this state consists of a single hidden vector h.
    # Here is an implementation of the step function in a Vanilla RNN

    # update the hidden state
    next_h = np.tanh(np.dot(prev_h, Wh) + np.dot(x, Wx) + b)
    # compute the output vector
    # TODO : it looks like we do not need output vectore here since we might have
    # multiple layers?
    # y = np.dot(self.W_hy, self.h)
    cache = (x, Wx, Wh, prev_h, next_h)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache

def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN .

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (N, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    # unroll cached values
    x, Wx, Wh, prev_h, next_h = cache
    # compute the derivative of tanh
    Xi = (1 - next_h ** 2) * dnext_h
    # compute grads
    dx = np.dot(Xi, Wx.T)
    db = np.sum(Xi, axis=0)
    dWx = np.dot(x.T, Xi)
    dWh = np.dot(prev_h.T, Xi)
    dprev_h = np.dot(Xi, Wh.T)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above.                                                                     #
    ##############################################################################
    # IMPORTANT: effectively what is called h0 in the function signature is an
    # h_{-1}.we are using it for recurent formula to get h_0:h_{T-1}. This in
    # term means that we have to store that value in order to correctly compute
    # the last step for backprop!!!
    H = b.shape[0]
    N, T, D = x.shape
    prev_h = h0

    h = np.zeros((N, T, H))
    for i in np.arange(T):
        h[:,i,:], _ = rnn_step_forward(x[:, i, :], prev_h, Wx, Wh, b)
        prev_h = h[:,i,:]
    cache = (x, Wx, Wh, h,h0)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above.                                                             #
    ##############################################################################
    N, T, H = dh.shape
    D = cache[0].shape[2]
    # print "N " + str(N)
    # print "T " + str(T)
    # print "H " + str(H)
    # print "D " + str(D)
    # we have to define the D first
    dh0 = np.zeros((N, H))
    dWx = np.zeros((D, H))
    db = np.zeros(H)


    # unpack cache
    x, Wx, Wh, h, h0 = cache

    # let us combine h and h_0 into one bigger h_memory.shape = N, T+1, H
    h_memory = np.zeros((h.shape[0], h.shape[1]+1, h.shape[2]))
    h_memory[:,1:T+1,:] = h
    h_memory[:,0,:] = h0

    dx = np.zeros_like(x)

    dWh = np.zeros_like(Wh)

    for i in reversed(range(T)):
        # we will get cached values from dict of cache
        # TODO : here the funny thing is that for dh0 we have to return only the
        # final state, and not intermidiate dhs... This is due to the fact that
        # we
        dx[:,i,:], dh0, dWx_temp, dWh_t, db_temp = \
            rnn_step_backward(dh[:,i,:]+ dh0, \
                              (x[:,i,:], Wx, Wh, h_memory[:,i,:],h_memory[:,i+1,:]))
        # print dWx.shape
        # print dWx_temp.shape
        dWx += dWx_temp
        dWh += dWh_t
        db += db_temp
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx,dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This should be very simple.:                                         #
    ##############################################################################
    # N, T = x.shape
    # V, D = W.shape

    # out = np.zeros((N,T,D))

    # for t in np.arange(T):
    #     out[:,t,:] = W[x[:,t],:]
    out = W[x,:]
    cache = (x, W)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
     matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    x, W = cache
    dW = np.zeros_like(W)

    # effectively this is a linear backprop. We just have to modify those
    # elements of W that were used during the forward pass. (information comes
    # only from them)
    np.add.at(dW,x,dout)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    H = prev_h.shape[1]

    # Compute activation vector
    a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b

    # Split
    ai = a[:, :H]
    af = a[:, H:2 * H]
    ao = a[:, 2 * H:3 * H]
    ag = a[:, 3 * H:4 * H]

    # Compute the gates
    i = sigmoid(ai)
    f = sigmoid(af)
    o = sigmoid(ao)
    g = np.tanh(ag)

    # Use gate to compte the next cell state
    next_c = f * prev_c + i * g

    # Compute next hidden state
    next_h = o * np.tanh(next_c)

    # Save values for backward pass
    cache = i, f, o, g, a, ai, af, ao, ag, Wx, Wh, b, prev_h, prev_c, x, next_c, next_h
    return next_h, next_c, cache
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
        - dnext_h: Gradients of next hidden state, of shape (N, H)
        - dnext_c: Gradients of next cell state, of shape (N, H)
        - cache: Values from the forward pass

    Returns a tuple of:
        - dx: Gradient of input data, of shape (N, D)
        - dprev_h: Gradient of previous hidden state, of shape (N, H)
        - dprev_c: Gradient of previous cell state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
    dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################

    # Unroll
    i, f, o, g, a, ai, af, ao, ag, Wx, Wh, b, prev_h, prev_c, x, next_c, next_h \
        = cache

    # \to step 5

    do = np.tanh(next_c) * dnext_h
    dnext_c += o * (1 - np.tanh(next_c)**2) * dnext_h

    # \to step step 4
    df = prev_c * dnext_c
    dprev_c = f * dnext_c
    di = g * dnext_c
    dg = i * dnext_c

    # \tto the gate 3
    dag = (1 - np.tanh(ag)**2) * dg
    dao = sigmoid(ao) * (1 - sigmoid(ao)) * do
    daf = sigmoid(af) * (1 - sigmoid(af)) * df
    dai = sigmoid(ai) * (1 - sigmoid(ai)) * di

    # \tto 2
    da = np.hstack((dai, daf, dao, dag))

    # \to 1
    dx = np.dot(da, Wx.T)
    dWx = np.dot(x.T, da)
    dprev_h = np.dot(da, Wh.T)
    dWh = np.dot(prev_h.T, da)
    db = np.sum(da, axis=0)

    #############################################################################
    #                               END OF YOUR CODE                            #
    #############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    N, T, D = x.shape
    H = h0.shape[1]

    h = np.zeros((N,T,H))
    prev_h = h0
    prev_c = np.zeros((N,H))
    cache = []

    for i in np.arange(T):
        h[:,i,:], next_c, cache_temp =\
            lstm_step_forward(x[:,i,:], prev_h, prev_c, Wx, Wh, b)
        cache.append(cache_temp)
        prev_h = h[:,i,:]
        prev_c = next_c

    #############################################################################
    #                               END OF YOUR CODE                            #
    #############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    # Unpack cache:
    i, f, o, g, a, ai, af, ao, ag, Wx, Wh, b, prev_h, prev_c, x, next_c, next_h \
        = cache[-1]

    N,T,H = dh.shape
    D = x.shape[1]

    dx = np.zeros((N,T,D))
    dh0 = np.zeros((N,H))
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros_like(b)

    # TODO: Have to think about it.
    # The thing here is that it looks like we have think that the state of the
    # gates if the cell state is asuumed to be zero during backprop... Not
    # clearly understand why. At the same time we, of course, use implicitly,
    # from the cache, the state of the cells..

    # init prev_c, since it come from the future time series it will be zero
    prev_c = np.zeros_like(dh0)
    dnext_c = np.zeros_like(prev_c)

    # print dx.shape
    # print len(cache)

    for t in reversed(range(T)):
        dx[:,t,:], dh0, dprev_c_temp, dWx_temp, dWh_temp, db_temp = \
            lstm_step_backward(dh[:,t,:]+dh0, dnext_c, cache[t])
        dnext_c = dprev_c_temp
        dWx += dWx_temp
        dWh += dWh_temp
        db += db_temp
    #############################################################################
    #                               END OF YOUR CODE                            #
    #############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print 'dx_flat: ', dx_flat.shape

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
