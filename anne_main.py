#!/bin/env python


import os
import sys
import time
import string
import subprocess
import shlex
import shutil



#list of executables:
fnscript_anne1_1='anne1_1_train_model.py'
fnscript_anne2_1='anne2_1_gen_assoc_score.py'
fnscript_anne2_2='anne2_2_gen_network_from_score.py'
fnscript_anne2_3='anne2_3_get_gene_to_node_weights.py'


#the master script, does the following:


ts=time.time()
#1. create config file
#2. build model
#3. gen association score
#4. gen network from score
#5. gen hidden layer weights to input genes.
d_s2desc={0:'whole workflow',
          1:'create config file',
          2:'train model',
          3:'generate gene-gene association score',
          4:'generate network from association score',
          5:'get hidden layer weight in input genes'
}

#use environment variable ANNE_STEP to determine which step to perform.
annestep=int(os.environ.get('ANNE_STEP', 0))

if annestep in [0,1,2,3,4,5]: pass
else:
  print 'ANNE_STEP can only take values 0,1,2,3,4,5. exit.'
  sys.exit()
#end if


global fncfg
fncfg=None

if annestep==0 or annestep==1:#create config file
  annestep_local=1
  print 'ANNE_STEP=%d, %s'%(annestep_local, d_s2desc[annestep_local])
  print time.strftime('%Y%m%d%H%M%S')
  print 'time used since job start: %.2f seconds.'%(time.time()-ts)
  tplcfg='''timestamp=${timestamp}
random_seed=1337
numthread=32
poolsize=
conv_layers=
nndense=50
weight_init=he_normal
n_class=2
regression=1
model_type=autoencoder
loss_func=mse
optimizer=adadelta
lr=1
batch_size=100
epoch_to_train=10000
train_data_file=${datafile}
validate_data_file=${datafile}
test_data_file=
model_file=
log_file=
'''
  if len(sys.argv)!=3:
    print 'usage: ANNE_STEP=%d ./anne_main <project_name> <input_data_file>'%annestep
    sys.exit()
  #end if
  projname=sys.argv[1]
  fndata=sys.argv[2]
  if os.path.exists(fndata) and os.path.isfile(fndata): pass
  else:
    print 'cannot access file %s'%fndata
    sys.exit()
  tpl=string.Template(tplcfg)
  tmstp=time.strftime('%Y%m%d%H%M%S')
  ret=tpl.substitute(timestamp=tmstp, datafile=fndata)
  fncfg='config_anne_%s.cfg'% projname 
  print 'creating config file %s'%fncfg
  open(fncfg, 'wb').write(ret)
  print 'created.'
if annestep==0 or annestep==2:#build model
  annestep_local=2
  print 'ANNE_STEP=%d, %s'%(annestep_local, d_s2desc[annestep_local])
  print time.strftime('%Y%m%d%H%M%S')
  print 'time used since job start: %.2f seconds.'%(time.time()-ts)
  #locate script to use
  fps=os.path.join(  os.path.dirname(os.path.abspath(sys.argv[0])), fnscript_anne1_1 )
  print 'executable to use:%s'%fps
  if fncfg==None:#not from 0, but just 2, so fncfg is not set

    if len(sys.argv)!=2:
      print 'usage: ANNE_STEP=%d ./anne_main <config.cfg>' %annestep_local
      sys.exit()
    #end if
    fncfg=sys.argv[1]
  #end if
  print 'config file: %s'%fncfg
  strcmd='%s %s'%(fps, fncfg)
  print 'running command %s'%strcmd
  pobj = subprocess.Popen(shlex.split(strcmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  cmdoutput = pobj.communicate()[0]
  print cmdoutput
  print 'done.'

if annestep==0 or annestep==3:#raw score
  annestep_local=3
  print 'ANNE_STEP=%d, %s'%(annestep_local, d_s2desc[annestep_local])
  print time.strftime('%Y%m%d%H%M%S')
  print 'time used since job start: %.2f seconds.'%(time.time()-ts)

  fps=os.path.join(  os.path.dirname(os.path.abspath(sys.argv[0])), fnscript_anne2_1 )
  print 'executable to use:%s'%fps
  if fncfg==None:#not from 0, but just 2, so fncfg is not set
    if len(sys.argv)!=2:
      print 'usage: ANNE_STEP=%d ./anne_main <config.cfg>'%annestep_local
      sys.exit()
    #end if
    fncfg=sys.argv[1]
  #end if
  print 'config file: %s'%fncfg
  strcmd='%s %s'%(fps, fncfg)
  print 'running command %s'%strcmd
  #sys.exit()
  pobj = subprocess.Popen(shlex.split(strcmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  cmdoutput = pobj.communicate()[0]
  print cmdoutput
  print 'done.'
if annestep==0 or annestep==4:#generate network
  annestep_local=4
  print 'ANNE_STEP=%d, %s'%(annestep_local, d_s2desc[annestep_local])
  print time.strftime('%Y%m%d%H%M%S')
  print 'time used since job start: %.2f seconds.'%(time.time()-ts)

  fps=os.path.join(  os.path.dirname(os.path.abspath(sys.argv[0])), fnscript_anne2_2 )
  print 'executable to use:%s'%fps
  if fncfg==None:#not from 0, but just 2, so fncfg is not set
    if len(sys.argv)!=2:
      print 'usage: ANNE_STEP=%d ./anne_main <config.cfg>'%annestep_local
      sys.exit()
    #end if
    fncfg=sys.argv[1]
  #end if
  print 'config file: %s'%fncfg
  strcmd='%s %s 200'%(fps, fncfg)
  print 'running command %s'%strcmd
  #sys.exit()
  pobj = subprocess.Popen(shlex.split(strcmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  cmdoutput = pobj.communicate()[0]
  print cmdoutput
  print 'done.'
  pass
if annestep==0 or annestep==5:#generate hidden layer weights
  annestep_local=5
  print 'ANNE_STEP=%d, %s'%(annestep_local, d_s2desc[annestep_local])
  print time.strftime('%Y%m%d%H%M%S')
  print 'time used since job start: %.2f seconds.'%(time.time()-ts)

  fps=os.path.join(  os.path.dirname(os.path.abspath(sys.argv[0])), fnscript_anne2_3 )
  print 'executable to use:%s'%fps
  if fncfg==None:#not from 0, but just 2, so fncfg is not set
    if len(sys.argv)!=2:
      print 'usage: ANNE_STEP=%d ./anne_main <config.cfg>'%annestep_local
      sys.exit()
    #end if
    fncfg=sys.argv[1]
  #end if
  print 'config file: %s'%fncfg
  strcmd='%s %s'%(fps, fncfg)
  print 'running command %s'%strcmd
  #sys.exit()
  pobj = subprocess.Popen(shlex.split(strcmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  cmdoutput = pobj.communicate()[0]
  print cmdoutput
  print 'done.'
  pass
print 'finished. time used since job start: %.2f seconds.'%(time.time()-ts)


