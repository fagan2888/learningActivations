import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load import mnist, cifar
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d

srng = RandomStreams()
np.random.seed()


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
  
    l1_lin = conv2d(X, w, border_mode='full')+b_c1.dimshuffle('x', 0, 'x', 'x')
    l1a = alpha_c1 * rectify(l1_lin) + (1.-alpha_c1) * T.tanh(l1_lin)
    l1 = max_pool_2d(l1a, (2, 2))
    l1 = dropout(l1, p_drop_conv)

    l2_lin = conv2d(l1, w2) + b_c2.dimshuffle('x', 0, 'x', 'x')
    l2a = alpha_c2 * rectify(l2_lin) + (1.-alpha_c2) * T.tanh(l2_lin)
    l2 = max_pool_2d(l2a, (2, 2))
    l2 = dropout(l2, p_drop_conv)

    l3_lin = conv2d(l2, w3) + b_c3.dimshuffle('x', 0, 'x', 'x')
    l3a = alpha_c3 * rectify(l3_lin) + (1.-alpha_c3) * T.tanh(l3_lin)
    l3b = max_pool_2d(l3a, (2, 2))
    l3 = T.flatten(l3b, outdim=2)
    l3 = dropout(l3, p_drop_conv)

    l4_lin = T.dot(l3, w4) + b_h1 
    l4 = alpha_h1 * rectify(l4_lin) + (1.-alpha_h1) * T.tanh(l4_lin)
    l4 = dropout(l4, p_drop_hidden)

    l5_lin = T.dot(l4, w5) + b_h2
    l5 = alpha_h1 * rectify(l5_lin) + (1.-alpha_h2) * T.tanh(l5_lin)
    l5 = dropout(l5, p_drop_hidden)

    pyx = softmax(T.dot(l5, w_o) + b_o )
    return l1, l2, l3, l4, l5, pyx

# load mnist data

# trX, teX, trY, teY = mnist(onehot=True)
# trX = trX.reshape(-1, 1, 28, 28)
# teX = teX.reshape(-1, 1, 28, 28)


# load cifar data
trX, teX, trY, teY = cifar(onehot=True)
trX = trX.reshape((-1, 3, 32, 32))
teX = teX.reshape((-1, 3, 32, 32))

print "trX shape: "
print trX.shape 

X = T.ftensor4()
Y = T.fmatrix()

w = init_weights((32, 3, 3, 3))
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
alpha_h1 = theano.shared(floatX(np.zeros(625) + .5))
alpha_h2= theano.shared(floatX(np.zeros(625) + .5))

noise_l1, noise_l2, noise_l3, noise_l4, noise_l5, noise_py_x = model(X, w, w2, w3, w4, w5, w_o, b_h1, b_h2, b_o, 0.2, 0.5)
l1, l2, l3, l4, l5, py_x = model(X, w, w2, w3, w4, w5, w_o, b_h1, b_h2, b_o, 0., 0.)
y_x = T.argmax(py_x, axis=1)


cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w, w2, w3, w4, w5, w_o,
          b_c1, b_c2, b_c3, b_h1, b_h2, b_o, alpha_h1, alpha_h2,
        ]
updates = RMSprop(cost, params, lr=0.001)
updates.append([alpha_c1, alpha_c1 - .001 * T.grad(cost, alpha_c1)])
updates.append([alpha_c2, alpha_c2 - .001 * T.grad(cost, alpha_c2)])
updates.append([alpha_c3, alpha_c3 - .001 * T.grad(cost, alpha_c3)])


train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

def train_some_epochs(trX=trX, trY=trY, teX=teX, teY=teY, num_epochs=10):
  for i in range(num_epochs):
    ####################################
    # shuffle the rows before each epoch
    ####################################
    p = np.random.permutation(len(trX))
    trX = trX[p]
    trY = trY[p]
    
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
      cost = train(trX[start:end], trY[start:end])
    print "test acc: " + str(np.mean(np.argmax(teY, axis=1) == predict(teX))) + "\ttrain acc:" + str(np.mean(np.argmax(trY[1:10000], axis=1) == predict(trX[1:10000])))
    
    