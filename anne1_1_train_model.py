#!/bin/env python
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import time
import sys
import os
import re
import glob
import h5py
import pandas as pd

import sklearn.metrics
import scipy.stats
import scipy.stats.mstats_basic                                            

b_gpu=False
hostname=os.environ.get('HOSTNAME', '')
print('hostname is %s'%hostname)
hnparts=set(hostname.split('.'))

print('b_gpu set to: ', b_gpu)


if False:
  pass
else:
  os.environ['THEANO_FLAGS'] = 'device=gpu, floatX=float32, dnn.enabled=True, scan.allow_output_prealloc=False, scan.allow_gc=True, optimizer_excluding=more_mem, optimizer=fast_compile, exception_verbosity=high, optimizer_including=cudnn'


  cr=os.environ.get('CUDA_ROOT','')
  if cr=='':#error
    print('$CUDA_ROOT not set.')
    sys.exit()
  #end if
#end if 
import lib_anne

#np.random.seed(1337)  # for reproducibility



import keras.callbacks
n_epoch_to_report=0#50
n_epoch_to_save_model_earlyepochs=100#when epoch < 1000
n_epoch_to_save_model=1000# when epoch >=1000


y01=None
val_y01=None
X_train=None
Y_train=None
X_val=None
Y_val=None
#ypred_bybatch=None
#gbl_ypred=None

def get_activations_keras111(model, layer, X_batch):#keras 1.1.1, https://github.com/fchollet/keras/issues/41#issuecomment-219262860
  from keras import backend as K
  get_activations = K.function([model.layers[0].input, K.learning_phase()], model.layers[layer].output)
  activations = get_activations([X_batch,0])
  return activations

  
def get_activations_batchbybatch_keras111_ae(model, layer, X, batchsize=10):
  od=model.layers[layer].output_shape
  od=list(od)
  [nsample, ncol]=X.shape
  od[0]=nsample
  ret=np.zeros(shape=od)
  bs=batchsize#batchsize
  for i in range( int(np.ceil(1.0*nsample/bs)) ):
    act_i=get_activations_keras111(model, layer, X[i*bs:i*bs+bs,:])
    [d1,d2]=act_i.shape#in case the last portion of the sample doesnot have bs samples
    ret[i*bs:i*bs+d1,:]=act_i
    #print('i=',i, 'i*bs=',i*bs, act_i.shape)
  #end for i
  return ret


class my_callback_reg_ae(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if ( epoch+1+gbl_epoch_trained>=1000 and (epoch+1+gbl_epoch_trained) % n_epoch_to_save_model ==0 ) or \
       ( epoch+1+gbl_epoch_trained<1000 and (epoch+1+gbl_epoch_trained) % n_epoch_to_save_model_earlyepochs ==0 ) :
      #save model weights
      print('saving model...')
      [fn_ms, fn_mw]=gbl_cmi.save_model_and_weights(self.model, epoch+1+gbl_epoch_trained)
      print('model structure saved to file %s'%fn_ms)
      print('model weights saved to file %s'%fn_mw)
    #end if
    if n_epoch_to_report==0 or (epoch+1) % n_epoch_to_report !=0: return
    ts=time.time()
    print('training performance after epoch', epoch+1)
    print('calculating model predictions...')
    #ypred=lib_anne.get_activations_batchbybatch(self.model, len(self.model.layers)-1, X_train)#for plain nn, works with 1d input
    #ypred=get_activations_batchbybatch_keras111_ae(self.model, len(self.model.layers)-1, X_train)
    ypred=self.model.predict(X_train)
    #global gbl_ypred
    #gbl_ypred=ypred
    #print('stop to examine gbl_ypred');raw_input()
    #In [3]: gbl_ypred.shape
    #Out[3]: (99, 11840)
    #In [4]: X_train.shape
    #Out[4]: (99, 11840)
    #In [5]: Y_train.shape  
    #Out[5]: (99, 11840)
    #so, ypred has the same dimension with Y_train (and Y_train should be set to X_train)
    
    ypr=np.reshape(ypred, -1)#to 1-D array
    ytr=np.reshape(Y_train, -1)#to 1-D array
    print('time used for calculating predictions: %f'%(time.time()-ts))
    #lib_anne.output_performance(y01, ypred01)
    #print('shapes:')
    #print(Y_train.shape)
    #print(ypred.shape)
    #ypred=np.squeeze(ypred)
    #print(ypred.shape)
    #mse=sklearn.metrics.mean_squared_error(Y_train, ypred)
    mse=sklearn.metrics.mean_squared_error(ytr, ypr)
    print('training mse=%f'%mse)
    r, pv_r=scipy.stats.pearsonr(ytr, ypr)
    print('training r=%f, pv_r=%f'%(r, pv_r))

    #rho, pv_rho=scipy.stats.spearmanr(Y_train, ypred)#TODO: calculating spearman rho cause error in HMS cluster. need to find out why.
    rho, pv_rho=scipy.stats.mstats_basic.spearmanr(ytr, ypr)#temporarily solved by using mstats_basic.spearmanr
    if type(pv_rho)!=float: pv_rho=pv_rho.tolist()
    print('training rho=%f, pv_rho=%f'%(rho, pv_rho))

    #this is autoencoder, so X_train is X_val, no need to calculate test metrics
    #In [22]: np.array_equal(X_train, X_val)
    #Out[22]: True

    #print('test performance after epoch', epoch+1)
    #print('calculating model predictions...')
    ##ypred=lib_anne.get_activations_batchbybatch(self.model, len(self.model.layers)-1, X_val)
    #ypred=get_activations_batchbybatch_keras111_ae(self.model, len(self.model.layers)-1, X_val)
    #ypred=np.squeeze(ypred)
    #print('time used for calculating predictions: %f'%(time.time()-ts))
    ##lib_anne.output_performance(val_y01, ypred01)
    #mse=sklearn.metrics.mean_squared_error(Y_val, ypred)
    ##mse=my_mse(Y_val, ypred)
    #print('test mse=%f'%mse)
    #r, pv_r=scipy.stats.pearsonr(Y_val, ypred)
    #print('test r=%f, pv_r=%f'%(r, pv_r))
    #rho, pv_rho=scipy.stats.mstats_basic.spearmanr(Y_val, ypred)#TODO: same.
    #pv_rho=pv_rho.tolist()
    #print('test rho=%f, pv_rho=%f'%(rho, pv_rho))

    print('time used for on_epoch_end: %f'%(time.time()-ts))
    return
  #end def on_epoch_end 
#end class

#my own constraint function, placed in keras/constrains
#from keras import backend as K
#from keras.constraints import Constraint
#class Zero(Constraint):  #added by zc@20161114
#    '''Constrain the weights to be zero. to be used only with bias.
#    '''
#    def __call__(self, p):
#        p *= K.cast(K.equal(p, 0.), K.floatx())
#        return p

#def main():
print('works with keras 1.1.1 using model apis.')
b_logtofile=True
#fncfg='tempcfg.cfg'
#cmi=cnn_model_info(fncfg); cmi.output()
#cmi=cnn_model_info(); cmi.write_to_file(fncfg)
#sys.exit()
#cmi=cnn_model_info(); print(cmi.generate_summary_string()); sys.exit()

nargs=len(sys.argv)
if nargs!=2 and nargs!=3:
  print('usage: '+os.path.basename(__file__)+' <configuration file>')
  print('or   : '+os.path.basename(__file__)+' <template configuration file> --buildnew')
  sys.exit()
#end if
fpcfg=os.path.abspath(sys.argv[1])#'tempcfg.cfg'
cmi=lib_anne.cnn_model_info(fpcfg)
gbl_cmi=cmi


rs=1337
if 'random_seed' in cmi.d:
  rs=int(cmi.d['random_seed'])
#end if
#print('using random seed %d'%rs)
#np.random.seed(rs)

wd=os.path.dirname(fpcfg)
print('changing cwd to folder %s'%wd)
os.chdir(wd)
#cmi=cnn_model_info()
#cmi.write_to_file(fpcfg)

if nargs==3:
  if sys.argv[2]!='--buildnew':#
    print('unrecognized option: %s'%sys.argv[2])
    sys.exit()
  #end if
  cmi.d['timestamp']=time.strftime('%Y%m%d%H%M%S')
  fnnew=cmi.generate_config_filename()
  if fnnew != os.path.basename(fpcfg):
    fpcfg=os.path.join(wd, fnnew)
    cmi.write_to_file(fpcfg)
    print('config file renamed to ', fpcfg)
  #end if
  sys.exit()
#end if


orig_flags = os.environ.get('THEANO_FLAGS', '')
ompthrd=int(cmi.d['numthread'])
if ompthrd<1: ompthrd=1

if not b_gpu:
  os.environ['THEANO_FLAGS'] = 'openmp=true,exception_verbosity=high'
  os.environ['OMP_NUM_THREADS']=str(ompthrd)
  #print(cmi.d)
#end if not b_gpu
#sys.exit()  
ts=time.time()

#################################################################
#only import theano and keras after setting the environment variables
print('keras: using functional api to build model instead of graph')
import keras
import theano.tensor.shared_randomstreams #hidden imports
from keras.models import Model
from keras.layers import Input, Dense, merge 
from keras.layers.core import Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD 
from keras.utils.visualize_util import plot as kerasplot#this is my modified version on the source
#need these in path:    progs = {'dot': '', 'twopi': '', 'neato': '', 'circo': '', 'fdp': '', 'sfdp': ''}
#added the following lines to ~/.bash_profile
#graphviz_path=/opt/graphviz-2.38.0/bin/
#PATH=~/installed/bin:$LSF_BINDIR:$LSF_SERVERDIR:$_CUR_PATH_ENV:$graphviz_path


#from keras.utils import np_utils
#end import 

if not b_gpu:
  print('working with %d thread(s).'%ompthrd)
#end if not b_gpu
#sys.exit()

hostname='unknown'
if 'HOSTNAME' in os.environ: hostname=os.environ['HOSTNAME']
if '.' in hostname: hostname=hostname.split('.')[0]

#logfile=os.path.join(fpin, logfile)
#logfile=cmi.generate_log_filename()
#use a logfile that has the same main name as the input cfg file.
if fpcfg.endswith('.cfg'): 
  logfile=os.path.basename(fpcfg[:-4]+'.log')
else:#
  print('error: cannot generate log file name for config file %s. exit.'%fpcfg)
  sys.exit(1)
#end if
print('console output redirected to file %s'%logfile)
if b_logtofile:
  old_stdout=sys.stdout
  sys.stdout=open(logfile, 'ab', 0)
  sys.stderr=sys.stdout
#end if
lib_anne.dump_env()
print('config file: %s'%fpcfg)
timestr=time.strftime('%Y%m%d%H%M%S')# the start time
print('job started at: %s'%timestr)
print('using random seed %d'%rs)
np.random.seed(rs)


if not b_gpu:
  print('working on host %s with %d thread(s).'%(hostname, ompthrd))
#end if not b_gpu
#fnin='../dataset_ccle_2d_resize_100by100.h5'
fn_train=cmi.d['train_data_file']
fn_validate=cmi.d['validate_data_file']
fn_test=cmi.d['test_data_file']

if cmi.d.has_key('model_type') and cmi.d['model_type']=='autoencoder':
  pass
else:
  print('the config file should specify model_type as autoencoder')
  sys.exit()
#end if

print('loading traning samples from ', fn_train)
print('before loading: %f M'%lib_anne.memory_usage_resource())
if ';' not in fn_train and ( fn_train.endswith('.gz') or fn_train.endswith('.txt') ):
  pass
else:
  print('training data should be single tab-delimited text file.')
  sys.exit()
#end if
df=pd.read_csv(fn_train, sep='\t', header=0, index_col=0, compression='infer')
#df.index[1:3]
#Index([u'AKT3', u'MED6'], dtype='object', name=u'gene_symbol')
#df.columns[1:3]
#Index([u'SW1116_LARGE_INTESTINE', u'NCIH1694_LUNG'], dtype='object')
#df.shape
#(18140, 1036)
m=df.T.as_matrix()#nrow samples, ncol features
#m.shape
#(1036, 18140)
print('data loaded. time used: %.2f seconds.'%(time.time()-ts))
X=m
y=m
x_train=m
val_X=None
val_y=None
if fn_validate==fn_train:
  val_X=x_train
  val_y=x_train
else:
  pass
  df=pd.read_csv(fn_validate, sep='\t', header=0, index_col=0, compression='infer')
  m=df.T.as_matrix()#nrow samples, ncol features
  print('data loaded. time used: %.2f seconds.'%(time.time()-ts))
  val_X=m
  val_Y=m
#end

#end if
global X_train
X_train=X
global Y_train
Y_train=y
global X_val#validation
X_val=val_X
global Y_val
Y_val=val_y
#sys.exit()#debug
#nb_pool = int(cmi.d['poolsize'])
weight_init=cmi.d['weight_init']
batch_size = int(cmi.d['batch_size'])#1000#128
n_epoch_to_train = int(cmi.d['epoch_to_train'])#100#12

l_nndense=[]
if cmi.d['nndense']!='': l_nndense=[int(v) for v in cmi.d['nndense'].split('-')]
n_dense=len(l_nndense)
print('number dense layers: %d'%n_dense)
print(l_nndense)
loss_func=cmi.d['loss_func']
optmr=cmi.d['optimizer']
lr=float(cmi.d['lr'])
print('epoch to train: %d'%n_epoch_to_train)
print('loss function: %s'%loss_func)
print('optimizer: %s with lr=%f'%(optmr, lr))

#sys.exit()

#rebuild model each time and just load previous weights
#  because Graph model cannot be properly loaded (duplicate output node error)

model=None
fnmw=None
epoch_trained=0
#fnmw,  epoch_trained=cmi.try_find_weights() #do not look for files. save time.
gbl_epoch_trained=epoch_trained
print('before bulding model: %f M'%lib_anne.memory_usage_resource())
if b_gpu==True:
  from theano.sandbox.cuda import dnn_version as version
  print('cudnn version:', version())

if model==None:
  print('building model...')
  fun_optmr=optmr
  if optmr=='sgd':
    fun_optmr=keras.optimizers.sgd(lr=lr)
    print('with optimizer sgd, lr=%f'%(lr))
  elif optmr=='rmsprop':
    fun_optmr=keras.optimizers.RMSprop(lr=0.010, rho=0.9, epsilon=1e-6)
  #end if

  #now build an autoencoder model
  #code taken from https://blog.keras.io/building-autoencoders-in-keras.html
  
  from keras.layers import Input, Dense
  from keras.models import Model
  from keras.regularizers import l2, activity_l2, l1l2, l1  
  # this is the size of our encoded representations
  encoding_dim = l_nndense[0]
  print('single hidden layer autoencoder. encoding_dim=%d'%encoding_dim)  

  actfun='linear'
  from keras.constraints import Zero
  bcnst=None
  print('b_constraint set to None')
  breg=l1l2
  print('b_reg set to l1l2')
  print('all activation functions set to ', actfun)
  nsamp_in=m.shape[0]
  nfeat_in=m.shape[1]
  inputlyr = Input(shape=(nfeat_in,))# this is our input placeholder
  #encoded = Dense(encoding_dim, activation='relu')(inputlyr)# "encoded" is the encoded representation of the input
  #encoded = Dense(encoding_dim, activation=actfun)(inputlyr)# using relu results in decoder layer w all 0 and middle layer activation all 0 
  encoded = Dense(encoding_dim, activation=actfun, b_constraint=bcnst, b_regularizer=breg())(inputlyr)

  decoded = Dense(nfeat_in, activation=actfun, b_constraint=bcnst, b_regularizer=breg())(encoded)# "decoded" is the lossy reconstruction of the input
  autoencoder = Model(input=inputlyr, output=decoded)# this model maps an input to its reconstruction
  
  #use 2 separate models, encoder and decoder, to track the internal states of this autoencoder
  encoder = Model(input=inputlyr, output=encoded)# this model maps an input to its encoded representation
  
  encoded_input = Input(shape=(encoding_dim,))# create a placeholder for an encoded (32-dimensional) input
  decoder_layer = autoencoder.layers[-1]# retrieve the last layer of the autoencoder model
  decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))# create the decoder model
  
  autoencoder.compile(optimizer=fun_optmr, loss=loss_func)#
  model=autoencoder
  ##############no plotting to save time.
  ##fnmodelplot=fpcfg+'.png'
  ##print('plotting model structure to file %s'%fnmodelplot)
  ##kerasplot(model, to_file=fnmodelplot, show_shapes=True)#, recursive=True)#this is with keras 1.1 
  print('model compiled. time used so far: ', time.time()-ts, ' seconds.')
  print('after building model: %f M'%lib_anne.memory_usage_resource())

  if fnmw!=None:
    print('loading weights from file %s'%fnmw)
    model.load_weights(fnmw)
#end if

  
print('before model.fit')
#model.fit(x=X_train, y=Y_train , batch_size=batch_size, nb_epoch=n_epoch_to_train, verbose=2, validation_data=(val_X, val_y), callbacks=[cb])
cb=my_callback_reg_ae()
model.fit(x_train, x_train,
                  nb_epoch=n_epoch_to_train,
                  batch_size=batch_size,
                  shuffle=True, callbacks=[cb])#,
                  #validation_data=(x_train, x_train))#, callbacks=[cb]) #no validation_data, no callback to speed up the training

print('after model.fit: %f M'%lib_anne.memory_usage_resource())

#save model
print('saving model...')
[fn_ms, fn_mw]=cmi.save_model_and_weights(model, epoch_trained+n_epoch_to_train)
print('model structure saved to file %s'%fn_ms)
print('model weights saved to file %s'%fn_mw)

timeused=time.time()-ts
print('Time used: %.2f seconds.'%timeused)
timestr=time.strftime('%Y%m%d%H%M%S')# the start time
print('job finished at: %s'%timestr)
  
if b_logtofile:
  sys.stdout.close()
#  return
  
#end of main()
#if __name__ == '__main__':
#  main()

