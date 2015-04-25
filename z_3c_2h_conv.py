import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load import mnist
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d

srng = RandomStreams()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def model(X, w, w2, w3, w4, w5, w_o, b_h1, b_h2, b_o, p_drop_conv, p_drop_hidden):
    l1a = rectify(conv2d(X, w, border_mode='full')+b_c1.dimshuffle('x', 0, 'x', 'x') )
    l1 = max_pool_2d(l1a, (2, 2))
    l1 = dropout(l1, p_drop_conv)

    l2a = rectify(conv2d(l1, w2) + b_c2.dimshuffle('x', 0, 'x', 'x'))
    l2 = max_pool_2d(l2a, (2, 2))
    l2 = dropout(l2, p_drop_conv)

    l3a = rectify(conv2d(l2, w3) + b_c3.dimshuffle('x', 0, 'x', 'x'))
    l3b = max_pool_2d(l3a, (2, 2))
    l3 = T.flatten(l3b, outdim=2)
    l3 = dropout(l3, p_drop_conv)

    l4 = rectify(T.dot(l3, w4) + b_h1  )
    l4 = dropout(l4, p_drop_hidden)

    l5 = rectify(T.dot(l4, w5) + b_h2 )
    l5 = dropout(l5, p_drop_hidden)

    pyx = softmax(T.dot(l5, w_o) + b_o )
    return l1, l2, l3, l4, l5, pyx

trX, teX, trY, teY = mnist(onehot=True)

trX = trX.reshape(-1, 1, 28, 28)
teX = teX.reshape(-1, 1, 28, 28)

X = T.ftensor4()
Y = T.fmatrix()

w = init_weights((32, 1, 3, 3))
w2 = init_weights((64, 32, 3, 3))
w3 = init_weights((128, 64, 3, 3))
w4 = init_weights((128 * 3 * 3, 625))
w5 = init_weights((625, 625))
w_o = init_weights((625, 10))


b_c1 = theano.shared(floatX(np.zeros(32)))
b_c2 = theano.shared(floatX(np.zeros(64)))
b_c3 = theano.shared(floatX(np.zeros(128)))

b_h1 = theano.shared(floatX(np.zeros(625)))
b_h2 = theano.shared(floatX(np.zeros(625)))
b_o = theano.shared(floatX(np.zeros(10)))

alpha_c1 = theano.shared(floatX(np.asarray([.5,])), broadcastable=[True] )
alpha_c2 = theano.shared(floatX(np.asarray([.5,])), broadcastable=[True] )
alpha_c3 = theano.shared(floatX(np.asarray([.5,])), broadcastable=[True] )
alpha_h1 = theano.shared(floatX(np.asarray([.5,])), broadcastable=[True] )
alpha_h2 = theano.shared(floatX(np.asarray([.5,])), broadcastable=[True] )

noise_l1, noise_l2, noise_l3, noise_l4, noise_l5, noise_py_x = model(X, w, w2, w3, w4, w5, w_o, b_h1, b_h2, b_o, 0.2, 0.5)
l1, l2, l3, l4, l5, py_x = model(X, w, w2, w3, w4, w5, w_o, b_h1, b_h2, b_o, 0., 0.)
y_x = T.argmax(py_x, axis=1)


cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w, w2, w3, w4, w5, w_o,b_c1, b_c2, b_c3, b_h1, b_h2, b_o]
updates = RMSprop(cost, params, lr=0.001)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

for i in range(300):
    for start, end in zip(range(0, len(trX), 128), range(127, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
    print np.mean(np.argmax(teY, axis=1) == predict(teX))