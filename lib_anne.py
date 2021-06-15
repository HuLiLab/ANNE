#!/bin/env python
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import time
import sys
import os
import re
import glob
#np.random.seed(1337)  # for reproducibility
gi_default_random_seed=1337
#
#model saving and loading are implemented with json and h5,
#  model structure saved in json format, while the weights saved in h5.
#  In this way the models will be saved as device independent compared to the cPickle method, which is device dependent.
#  (i.e. model build on GPU cannot be loaded for CPU.)

import h5py


class cnn_model_info:
  l_fields=['timestamp',
            'random_seed',
            'numthread',
            'poolsize',
            'conv_layers',
            'nndense',
            'weight_init',
            'n_class',
            'regression',
            'model_type',

            'loss_func',
            'optimizer',
            'lr',
            'batch_size',
            'epoch_to_train',
            'train_data_file',
            'validate_data_file',
            'test_data_file',
            'model_file',
            'log_file']

  #          'epoch_trained'
  #these fields are generated after 1st run
  l_fields_opt=[]
  
  def __init__(self, config_file=None):
    self.config_file=config_file
    #
    self.d=dict()
    if config_file!=None: self.d=self.load_from_file(config_file)
    else:#
      self.init_as_sample()
    #end if
    #check
    missing_fields=sorted(list(set(self.l_fields)-set(self.d.keys())))
    optional_fields=set(['regression', 'model_type'])
    if len(missing_fields)==0 \
      or set(missing_fields) <= optional_fields:
      # or (len(missing_fields)==1 and missing_fields[0]=='regression'):
      pass
    else:
      print('the following fields are missing in config file:')
      print(missing_fields)
      sys.exit()
    #end if
    #check lr format
    lr=float(self.d['lr'])
    lr_c2=float(self.str_add_dot(self.str_rm_dot(self.d['lr'])))
    if abs(lr-lr_c2)<1e-9:#passed
      pass 
    else:#
      print('error: invalid learning rate lr=%s'%self.d['lr'])
      sys.exit()
    #end if
    
  #end def __init__
  def init_as_sample(self):
    self.d=dict()
    self.d['timestamp']=time.strftime('%Y%m%d%H%M%S')
    self.d['batch_size']='10'
    self.d['random_seed']='1337'

    self.d['epoch_to_train']='10'
    self.d['poolsize']='2'
    self.d['numthread']='32'

    self.d['conv_layers']='25-3-3'
    self.d['nndense']='128' #can have the form of 1024-128 for multiple dense layers

    self.d['weight_init']='he_normal'
    self.d['n_class']='2'
    self.d['regression']='0'#0 or 1 only. if has key regression and value==1, this is a regression model

    self.d['loss_func']='categorical_crossentropy'
    self.d['optimizer']='sgd'#
    self.d['lr']='0.01'

    self.d['train_data_file']='dataset_ccle_2d_resize_200by200.h5'
    self.d['validate_data_file']='dataset_ccle_2d_resize_200by200.h5'
    self.d['test_data_file']=''

    self.d['model_file']=''
    self.d['log_file']=''
    #self.d['epoch_trained']='0'
  #end def 
  def str_rm_dot(self, float_in_str):
    #make a float value suitable for file name by removing its decimal dot.
    #
    s=str(float_in_str)
    if '.' in s: s=s.replace('.','')
    return s
  #end def str_rm_dot(self, float_in_str):
  def str_add_dot(self, float_in_str):
    #if s starts with 0, it is considered a fraction.
    #  otherwise s remains the same as input
    s=str(float_in_str)
    if s.startswith('0'):
      s='0.'+s[1:]
    #end if
    return s
  #end def str_add_dot(self, float_in_str):

  def generate_summary_string(self):
    lo=['',
        'poolsize%d'%int(self.d['poolsize']),
        'nndense%s'%('-'.join(self.d['nndense'].split('-'))),
        'convlayer%s'%('-'.join(self.d['conv_layers'].split('-'))),
        'optimizer%s'%self.d['optimizer'], 
        self.str_rm_dot('lr%s'%self.d['lr']),
        'rs%s'%self.d['random_seed']
       ]
    if self.d.has_key('regression') and self.d['regression']=='1':#regression
      lo[0]='reg'
    elif self.d.has_key('model_type') and self.d['model_type']=='autoencoder':#regression
      lo[0]='autoencoder' 
    else:
      lo[0]='class%d'%int(self.d['n_class'])
    #end if
    res='_'.join(lo)
    return res
  def generate_config_filename(self):
    ss=self.generate_summary_string()
    res='config_'+ss+'_'+self.d['timestamp']+'.cfg'
    return res
  #end def generate_config_filename(self):

  def generate_model_weight_filename_wildcard(self):
    #ss=self.generate_summary_string()
    #res='config_'+ss+'_'+self.d['timestamp']+'.model_weight_epoch*.h5'
    res=None
    if self.config_file.endswith('.cfg'): 
      res=os.path.basename(self.config_file[:-4]+'.model_weight_epoch*.h5')
    else:#
      print('error: cannot generate log file name for config file %s. exit.'%self.config_file)
      sys.exit(1)
    #end if

    return res
  #end def generate_model_weight_filename_wildcard(self):

  def generate_model_weight_filename(self, epoch_trained):
    #ss=self.generate_summary_string()
    #res='config_'+ss+'_'+self.d['timestamp']+'.model_weight_epoch%d.h5'%epoch_trained
    res=None
    if self.config_file.endswith('.cfg'): 
      res=os.path.basename(self.config_file[:-4]+'.model_weight_epoch%d.h5'%epoch_trained)
    else:#
      print('error: cannot generate log file name for config file %s. exit.'%self.config_file)
      sys.exit(1)
    #end if

    return res
  #end def generate_model_weight_filename(self, epoch_trained):

  def generate_model_structure_filename(self):
    #ss=self.generate_summary_string()
    #res='config_'+ss+'_'+self.d['timestamp']+'.model_structure.json'
    res=None
    if self.config_file.endswith('.cfg'): 
      res=os.path.basename(self.config_file[:-4]+'.model_structure.json')
    else:#
      print('error: cannot generate log file name for config file %s. exit.'%self.config_file)
      sys.exit(1)
    #end if

    return res
  #end def generate_model_structure_filename(self):

  def generate_log_filename(self):
    ss=self.generate_summary_string()
    res='config_'+ss+'_'+self.d['timestamp']+'.log'
    return res
  #end def generate_log_filename(self):
  def enum_weights(self):
    #search for previously trained models
    wd=os.path.dirname(os.path.abspath(self.config_file))
    old_wd=os.getcwd()
    os.chdir(wd)

    model=None
    epoch_trained=0

    mw_fn_wc=self.generate_model_weight_filename_wildcard()#model weight
    l_mw_fn=glob.glob(mw_fn_wc)
    l_epoch=[0]*len(l_mw_fn)
    fn_mw=None
    if len(l_mw_fn)!=0:#found model weights, select one of them
      print('found existing model weights')
      re_e=re.compile('epoch(\d+)')
      maxe=-1; maxp=None
      for i in range(len(l_mw_fn)):
        mdl=l_mw_fn[i]
        ms=re_e.findall(mdl)
        if ms==None or len(ms)!=1:#error
          print('model file name cannot be parsed: %s'%mdl)
          sys.exit()
        #end if
        #print(ms)
        e=int(ms[0])
        l_epoch[i]=e
        if e>maxe: maxe=e; maxp=i
        print('%dth model file %s with %d epochs trained.'%(i+1, mdl, l_epoch[i]))
      #end for i
      fn_mw=l_mw_fn[maxp]
      epoch_trained=maxe
    #end if
    os.chdir(old_wd)
    return [l_mw_fn, l_epoch]
  #end def enum_weights(self):

  def try_find_weights(self):#just find the files for model weights without loading it
    #because Graph model structures cannot be loaded (duplicate output node error)
    #  so build the model each time and load the weights
    from keras.models import model_from_json
    #search for previously trained models
    wd=os.path.dirname(os.path.abspath(self.config_file))
    old_wd=os.getcwd()
    os.chdir(wd)

    model=None
    epoch_trained=0

    mw_fn_wc=self.generate_model_weight_filename_wildcard()#model weight
    l_mw_fn=glob.glob(mw_fn_wc)
    l_epoch=[0]*len(l_mw_fn)
    fn_mw=None
    if len(l_mw_fn)!=0:#found model weights, select one of them
      print('found existing model weights')
      re_e=re.compile('epoch(\d+)')
      maxe=-1; maxp=None
      for i in range(len(l_mw_fn)):
        mdl=l_mw_fn[i]
        ms=re_e.findall(mdl)
        if ms==None or len(ms)!=1:#error
          print('model file name cannot be parsed: %s'%mdl)
          sys.exit()
        #end if
        #print(ms)
        e=int(ms[0])
        l_epoch[i]=e
        if e>maxe: maxe=e; maxp=i
        print('%dth model file %s with %d epochs trained.'%(i+1, mdl, l_epoch[i]))
      #end for i
      fn_mw=l_mw_fn[maxp]
      epoch_trained=maxe
    #end if
    os.chdir(old_wd)
    return [fn_mw, epoch_trained]
  #end def try_find_weights(self):

  def try_load_model_with_weights(self):
    from keras.models import model_from_json
    #search for previously trained models
    wd=os.path.dirname(os.path.abspath(self.config_file))
    old_wd=os.getcwd()
    os.chdir(wd)

    model=None
    epoch_trained=0

    fn_ms=self.generate_model_structure_filename()#file of model structure
    if os.path.exists(fn_ms):
      #model structure found, now try to find model weights
      mw_fn_wc=self.generate_model_weight_filename_wildcard()#model weight
      l_mw_fn=glob.glob(mw_fn_wc)
      l_epoch=[0]*len(l_mw_fn)
      fn_mw=None
      if len(l_mw_fn)!=0:#found model weights, select one of them
        print('found existing model weights')
        re_e=re.compile('epoch(\d+)')
        maxe=-1; maxp=None
        for i in range(len(l_mw_fn)):
          mdl=l_mw_fn[i]
          ms=re_e.findall(mdl)
          if ms==None or len(ms)!=1:#error
            print('model file name cannot be parsed: %s'%mdl)
            sys.exit()
          #end if
          #print(ms)
          e=int(ms[0])
          l_epoch[i]=e
          if e>maxe: maxe=e; maxp=i
          print('%dth model file %s with %d epochs trained.'%(i+1, mdl, l_epoch[i]))
        #end for i
        fn_mw=l_mw_fn[maxp]
        epoch_trained=maxe
      #end if
      if fn_mw!=None:
        print('loading latest model weights from %s trained for %d epochs...'%(fn_mw, epoch_trained))
     
        model=model_from_json(open(fn_ms, 'rb').read())
        model.load_weights(fn_mw)
      #end if
    else:#no model structure
      pass
    #end if
    os.chdir(old_wd)
    return [model, epoch_trained]
  #end def try_load_model_with_weights(self):

  def save_model_and_weights(self, model, epoch_trained):
    fn_ms=self.generate_model_structure_filename()
    fn_mw=self.generate_model_weight_filename(epoch_trained)
   
    json_string = model.to_json()
    open(fn_ms, 'wb').write(json_string)
    model.save_weights(fn_mw, overwrite=True)
    return [fn_ms, fn_mw]
  #end def save

  def output(self):
    for k in self.l_fields+self.l_fields_opt:
      print(k, self.d[k])
  #end def output
  def load_from_file(self, config_file):
    s_allfields=set(self.l_fields+self.l_fields_opt)
    re_eq=re.compile('^([_a-zA-Z0-9]+)=([${} ;_a-zA-Z0-9.-]*)$')
    d=dict()
    for l in open(config_file, 'rb'):
      sl=l.strip()
      if sl.startswith('#') or sl=='': continue#comments or empty line
      m=re_eq.match(sl)
      if m==None:
        print('error: line in config file not recognized.')
        print(sl)
        sys.exit()
      #print(m.group(1), m.group(2))
      k=m.group(1)
      v=m.group(2)

      if k in s_allfields: d[k]=v
      else:
        print('warning: unrecognized key discarded - %s'%k)
    #end for
    if 'random_seed' not in d:
      d['random_seed']=str(gi_default_random_seed)
    return d
  #end def load_from_file
  def write_to_file(self, config_file):
    print('writing config to file %s'%config_file)
    fout=open(config_file, 'wb')
    allfields=self.l_fields+self.l_fields_opt
    #print(allfields)
    #print(self.d.keys())
    for k in allfields:
      if k in self.d:
        fout.write('%s=%s\n'%(k, self.d[k]))
    #end for k
    fout.close()
    return
  #end def write_to_file

#end class cnn_model_info:
def memory_usage_resource():
  import resource
  import sys
  rusage_denom = 1024.
  if sys.platform == 'darwin':
    # ... it seems that in OSX the output is different units ...
    rusage_denom = rusage_denom * rusage_denom
  mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom
  return mem
#end def memory_usage_resource():
def dump_env():
  print('dumping environment variables:')
  for k in sorted(os.environ.keys()):
    print(k, os.environ[k])
#end def dump_env



