import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load import mnist

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

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for param, grad in zip(params, grads):
        acc = theano.shared(param.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * grad ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        grad = grad / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((param, param - lr * grad))
    return updates

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def model(X, w_h, w_h2, w_o, alpha_h, alpha_h2, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)
    
    h_lin = T.dot(X, w_h)
    h =  rectify( h_lin ) * alpha_h + T.tanh( h_lin ) * (1-alpha_h)

    h =  dropout(h, p_drop_hidden)

    h2_lin = T.dot(h, w_h2)
    h2 = rectify( h2_lin ) * alpha_h2 + T.tanh( h2_lin ) * (1-alpha_h2)
    h2 = dropout(h2, p_drop_hidden)

    py_x = softmax(T.dot(h2, w_o))
    return h, h2, py_x

trX, teX, trY, teY = mnist(onehot=True)

X = T.fmatrix()
Y = T.fmatrix()

w_h = init_weights((784, 625))
w_h2 = init_weights((625, 625))
w_o = init_weights((625, 10))


alpha_h = theano.shared(floatX(np.asarray([.5,])), broadcastable=[True] )
alpha_h2 = theano.shared(floatX(np.asarray([.5,])), broadcastable=[True] )


noise_h, noise_h2, noise_py_x = model(X, w_h, w_h2, w_o, alpha_h, alpha_h2, p_drop_input=0.0, p_drop_hidden=0.5)
h, h2, py_x = model(X, w_h, w_h2, w_o, alpha_h, alpha_h2, 0., 0.)
y_x = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w_h, w_h2, w_o]
updates = RMSprop(cost, params, lr=0.0005)
updates.append([alpha_h, alpha_h - 0.0005 * T.grad(cost, alpha_h)])
updates.append([alpha_h2, alpha_h2 - 0.0005 * T.grad(cost, alpha_h2)])


train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

for i in range(300):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
    print np.mean(np.argmax(teY, axis=1) == predict(teX))

