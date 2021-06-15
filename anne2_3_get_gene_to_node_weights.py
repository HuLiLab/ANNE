#!/bin/env python
from __future__ import absolute_import
from __future__ import print_function
import time
import sys
import os
import re
import glob
#import numpy as np
import pandas as pd

#import matplotlib.pyplot as plt    


sys.path.append('/home/m143944/pharmacogenomics/gene_drug_2d/6_vis_weight')
import lib_anne

#np.random.seed(1337)  # for reproducibility
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
#w1.shape= (11840, 50)
#w2.shape= (50, 11840)

nodenames=['node_%d'%(i+1) for i in range(w1.shape[1])]

pd_s=pd.DataFrame(w1, index=feature_names, columns=nodenames)
print('check pd_s')
#sys.exit()
#fnh5=fntrunc+'.w1.h5'
#print('writing to file ', fnh5)
#store = pd.HDFStore(fnh5)
#store['default']=pd_s
#store.close()

fnout=fntrunc+'.w1.csv'
print('writing to file ', fnout)
pd_s.to_csv(fnout, sep=',', header=True, index=True)

print('Done. total time used: %f'%(time.time()-ts))
#writing to file  /data2/syspharm/projects/m143944/wd_autoencoder_geneexpr/model_for_pCR_RD/config_autoencoder_adadelta_linear_breg_pCR.w1.h5
#writing to file  /data2/syspharm/projects/m143944/wd_autoencoder_geneexpr/model_for_pCR_RD/config_autoencoder_adadelta_linear_breg_pCR.w1.csv
#Done. total time used: 2.756050

#return
#end of main()

#if __name__=='__main__': main()

