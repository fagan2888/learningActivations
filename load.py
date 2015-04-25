import numpy as np
import os
import cPickle


datasets_dir = '../datasets/'

def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h


def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def cifar(ntrain=50000, ntest= 10000, onehot=True):
  data_dir = os.path.join(datasets_dir, 'cifar-10')
  
  
  trX = np.array([])
  trTempY = []
  trY = np.zeros(500000).reshape(50000,10)
  
  teX = np.array([])
  teTempY = []
  teY = np.zeros(100000).reshape(10000,10)
  
  for i in range(1,6):
    train_dict = unpickle(os.path.join(data_dir, 'data_batch_' + str(i)))
    trX = np.append( trX, train_dict['data'])
    trTempY = trTempY + train_dict['labels']

  for i in range(len(trTempY)):
    trY[i][trTempY[i]] = 1.

  test_dict = unpickle(os.path.join(data_dir, 'test_batch'))
  teX = test_dict['data']
  teTempY = test_dict['labels']
  
  for i in range(len(teTempY)):
    teY[i][teTempY[i]] = 1.
  
  return trX.reshape((50000,3072)), teX.reshape((10000,3072)), trY, teY
  #return tempY
  

def mnist(ntrain=60000,ntest=10000,onehot=True):
	data_dir = os.path.join(datasets_dir,'mnist/')
	fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trX = loaded[16:].reshape((60000,28*28)).astype(float)

	fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trY = loaded[8:].reshape((60000))

	fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teX = loaded[16:].reshape((10000,28*28)).astype(float)

	fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teY = loaded[8:].reshape((10000))

	trX = trX/255.
	teX = teX/255.

	trX = trX[:ntrain]
	trY = trY[:ntrain]

	teX = teX[:ntest]
	teY = teY[:ntest]

	if onehot:
		trY = one_hot(trY, 10)
		teY = one_hot(teY, 10)
	else:
		trY = np.asarray(trY)
		teY = np.asarray(teY)

	return trX,teX,trY,teY