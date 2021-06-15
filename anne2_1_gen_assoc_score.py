#!/bin/env python
from __future__ import absolute_import
from __future__ import print_function
import time
import sys
import os
import re
import glob
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt    


import lib_anne

np.random.seed(1337)  # for reproducibility
def append_to_mainfn(fnin, str_app):
  (strpath, strfn) = os.path.split(os.path.abspath(fnin))
  (strmainfn, strext) = os.path.splitext(strfn)

  (strmainfn2, strext2) = os.path.splitext(strmainfn)#secondary ext
  if strmainfn2=='' or strext2=='':#secondary split not needed
    pass
    #strmainfn=strmainfn; strext=strext#no operation
  else:#secondary splittinig needed
    strmainfn=strmainfn2
    strext=strext2+strext
  str_r = os.path.join(strpath, strmainfn+ str_app +strext)
  return str_r
#end def append_to_mainfn(fnin, str_app):


def float_to_str(v):
  #make a float value suitable for file name by removing its decimal dot.
  #
  s='%g'%v
  if '.' in s: s=s.replace('.','')
  return s

########################################################################################################
#def main():
  
nargs=len(sys.argv)
if nargs!=2:
  print('usage: '+os.path.basename(__file__)+' <configuration file>')
  sys.exit()
#end if
fpcfg=os.path.abspath(sys.argv[1])#'tempcfg.cfg'
fntrunc=os.path.splitext(fpcfg)[0]
cmi=lib_anne.cnn_model_info(fpcfg)
wd=os.path.dirname(fpcfg)
print('input file: %s'%fpcfg)
print('changing cwd to folder %s'%wd)
os.chdir(wd)
ts=time.time()
fn_train=cmi.d['train_data_file']
print('loading traning samples from ', fn_train)
print('before loading: %f M'%lib_anne.memory_usage_resource())
if ';' not in fn_train and ( fn_train.endswith('.gz') or fn_train.endswith('.txt') ):
  pass
else:
  print('training data should be single tab-delimited text file.')
  sys.exit()
#end if
df=pd.read_csv(fn_train, sep='\t', header=0, index_col=0, compression='infer')
#df.shape
#(11840, 99)
feature_names=list(df.index)

print('keras: using functional api to build model instead of graph')
import keras
from keras.models import Model
model=None
epoch_trained=0
[model, epoch_trained]=cmi.try_load_model_with_weights()
print('model loaded.')

print('time used so far: %.2f seconds.'%(time.time()-ts))
#sys.exit()
'''
In [5]: model.layers
Out[5]: 
[<keras.engine.topology.InputLayer at 0x7f98dee57e10>,
 <keras.layers.core.Dense at 0x7f98dee57fd0>,
 <keras.layers.core.Dense at 0x7f98dedf7050>]
'''
#InputLayer has no weight
'''
In [67]: model.layers[1].get_weights()[0].shape
Out[67]: (11840, 50)
In [68]: model.layers[1].get_weights()[1].shape                                
Out[68]: (50,)
'''
w1=model.layers[1].get_weights()[0]
print('w1.shape=', w1.shape)
w2=model.layers[2].get_weights()[0]
print('w2.shape=', w2.shape)
#

##################################################################
#feature selection using method 5.1.1 in review paper doi 10.1.1.54.4570
#
#  original paper for the method: 
#Yacoub,  M.  and  Bennani,  Y.  (1997).  
#HVS:  A  Heuristic  for  Variable  Selection  in  Multilayer Artificial Neural Network Classifier.
#in Proceedings of ANNIE'97. 527-532
#
#
#  for neural network I->H->O
#  where I is the input layer, O is the output layer, H is the hidden layer
#
#S[i] = sum[o in O] sum [j in H] abs( wn[o,j]*wn[j,i] )
#
#where S[i] is the importance score for input feature i (YB score)
#  wn[o,j] and wn[j,i] are normalized weights. 
#  defined as 
#  wn[o,j]=w[o,j]/sum(abs(w[o,.]))
#  w[o,j] is the original weight and w[o,.] are the weights of output neuron o to all neurons in H.
#
##  w1abs=np.abs(w1)
##  w2abs=np.abs(w2)
##  
##  w1abssum=np.apply_along_axis(func1d=lambda a: np.sum(a), axis=0, arr=w1abs)#w1 abs sum, the L1 norm of w1
##  print('w1abssum.shape=', w1abssum.shape)
##  
##  w2abssum=np.apply_along_axis(func1d=lambda a: np.sum(a), axis=0, arr=w2abs)#w2 abs sum, the L1 norm of w2
##  print('w2abssum.shape=', w2abssum.shape)
##  #sys.exit()
##  
##  nfeat=w1.shape[0]
##  nhidden=w2.shape[0]#also w1.shape[1]
##  if nhidden!=w1.shape[1]:#error
##    print('error, w1.shape[1]=%d but w2.shape[0]=%d, not equal. exit.'%(w1.shape[1], w2.shape[0]))
##    sys.exit()
##  #end if
##  noutput=w2.shape[1]
##  print('nfeat=%d, nhidden=%d, noutput=%d'%(nfeat, nhidden, noutput))
##  #normalize
##  w1absnorm=np.copy(w1abs)
##  w2absnorm=np.copy(w2abs)
##  
##  for j in range(nhidden):
##    w1absnorm[:,j]/=w1abssum[j]
##  for k in range(noutput):
##    w2absnorm[:,k]/=w2abssum[k]
##  
##  #sys.exit()
##  print('calculating YB score')
##  #s: YB scores for each input features
##  #try different way of calculation, s1, s2, s3
##  
##  ts0=time.time()
##  s0=np.dot(w1absnorm, w2absnorm)
##  print('s0 calculated. time used: %f'%(time.time()-ts0))
#
#for s0:
#dim 1: the from gene
#dim 2: the to gene


#pd_s=pd.DataFrame(s0, index=feature_names, columns=feature_names)
#
#print('check pd_s')
##sys.exit()
#fnh5=fntrunc+'.YBscore.h5'
#print('writing to file ', fnh5)
#store = pd.HDFStore(fnh5)
#store['default']=pd_s
#store.close()
#
#fnout=fntrunc+'.YBscore.csv'
#print('writing to file ', fnout)
#pd_s.to_csv(fnout, sep=',', header=False, index=False)

s0raw=np.dot(w1, w2)#no abs, no normalization
pd_s=pd.DataFrame(s0raw, index=feature_names, columns=feature_names)
#print('check pd_s')
#sys.exit()
fnh5=fntrunc+'.rawscore.h5'
print('writing to file ', fnh5)
store = pd.HDFStore(fnh5)
store['default']=pd_s
store.close()
#fnout=fntrunc+'.rawYBscore.csv'
#print('writing to file ', fnout)
#pd_s.to_csv(fnout, sep=',', header=False, index=False)




##only for models with a single output node
##ts1=time.time()#time start
##s1=np.asarray([0.0]*nfeat, dtype=float)
##for i in range(nfeat):
##  #if i%100==0: print('i=%d'%i)
##  tmpsum=0#temp sum
##  for k in range(noutput):
##    for j in range(nhidden):
##      tmpsum += w1absnorm[i,j]*w2absnorm[j,k]
##    #end for j
##  #end for k
##  s1[i]=tmpsum
###end for i
##print('s1 calculated. time used: %f'%(time.time()-ts1))
###s1 calculated. time used: 17.268697


###################
#this is correct but slow
#ts=time.time()#time start
#s2=np.asarray([0.0]*nfeat, dtype=float)#YB scores for input features
#for i in range(nfeat):
#  if i%1000==0: print('i=%d'%i)
#  js=0#sum for j
#  for j in range(nhidden):
#    ks=np.sum(w2absnorm[j,:])
#    js+=w1absnorm[i,j]*ks
#  #end for j
#  #js/=w1abssum[j]
#  s2[i]=js
##end for i
#print('s2 calculated. time used: %f'%(time.time()-ts))
##s2 calculated. time used: 147.870212
print('Done. total time used: %f'%(time.time()-ts))
#return
#end of main()

#if __name__=='__main__': main()

