#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Python Mobules
import os, json, time
import numpy as np
from sys import argv
import tensorflow as tf
import tensorflow.keras as keras

# Reduce warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Basic information
n_epochs = 100
stop = 4000
batch_size = 128
learning_rate = 0.01
verbose = 0


# In[2]:


buildEnv = False

# Am I in a jupyter notebook?
try:
    get_ipython().__class__.__name__
    buildEnv = True
    argv = [ None, 'run-16-13']
    verbose = 1
    n_epochs = 2
    stop = 3
    print("In Notebook")

# Or am I in a python script?
except:
    if len( argv ) < 2:
        print("ERROR:  Need unique name" )
    
    

if buildEnv: 
    import matplotlib.pyplot as plt
    
runName = argv[1]

if 'test' in runName:
    verbose = 1
    n_epochs = 2
    stop = 3



# In[3]:


def correlation( v1, v2 ):
    corr = np.corrcoef( v1, v2 )[0,1]
    return corr


# In[4]:


# Read current devices
devices = tf.config.get_visible_devices()
if buildEnv:    print( 'Devices:', devices )

# If no GPU found, use CPU
if len(devices) == 1:
    strategy = tf.distribute.OneDeviceStrategy('CPU') # Use local GPU

# # if buildEnv, probably on hamilton. Use GPU 2
# elif buildEnv:
#     tf.config.set_visible_devices(devices[0:1]+devices[2:3])
#     tf.config.experimental.set_memory_growth(devices[1],True)
#     strategy = tf.distribute.OneDeviceStrategy('GPU:1') # Use local GPU

# Standard single GPU on backus
else:
    tf.config.set_visible_devices(devices[0:1]+devices[1:2])
    tf.config.experimental.set_memory_growth(devices[1],True)
    strategy = tf.distribute.OneDeviceStrategy('GPU:0') # Use local GPU


# In[5]:


def sliceKey( dictIn ):
    dictOut =  dictIn
    for k in dictOut:
        if type( dictOut[k] ) == type( 'string' ):
            if 'slice' in dictOut[k]:
                #print( k, dictOut[k] )
                tmp = dictOut[k].split('(')[1].split(')')[0].split(',')
                s = int( tmp[0] )
                e = int( tmp[1] )
                dictOut[k] = slice( s, e )
                print( dictOut[k] )
    
    return dictOut 


# In[6]:


with strategy.scope():
    
    # Prepare data
    with open( 'data/data-key.json' ) as keyFile:
        key = json.load( keyFile )
    
    key = sliceKey( key )
    
    print( key.keys() )


# In[7]:


with strategy.scope():
        
    # Load testing data
    testMin = np.load('data/test-data.npy' )[:,key['min']]
    testImg   = np.load('data/test-img.npy' )
    
    # Load training data
    trainMin = np.load('data/train-data.npy' )[:,key['min']]
    trainImg   = np.load('data/train-img.npy' )
        
    print( 'testScore: ', testMin.shape )
    print( 'testImg:   ', testImg.shape )

    print( 'trainScore:', trainMin.shape )
    print( 'trainImg:  ', trainImg.shape )


# In[8]:


with strategy.scope():
    
    # Build input layer
    x = keras.layers.Input(shape=trainImg.shape[1:])
    y = x    
        
    # Build resnet layer without top layer
    resnet = keras.applications.ResNet50V2(
        include_top = False,
        input_shape = y.shape[1:], 
    )
    
    y = resnet(y)
    
    # Flatten for final layer
    y = keras.layers.Flatten()(y)
    
    # Mid layer before final
    output = keras.layers.Dense( 64, activation= keras.activations.tanh )
    y = output(y)
    
    # Final layer.  Predicting a single value
    y = keras.layers.Dense( trainMin.shape[1] )(y)
    
    model = keras.Model( x, y )
    model.compile( 
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.mean_squared_error,
        )
    
    model.summary(expand_nested=False)
    
#keras.utils.plot_model(model,show_shapes=True,expand_nested=False)


# In[9]:


with strategy.scope():  
    # Quick prediction to test functionality
    if buildEnv: 
        print("Prediction: ", model.predict( testImg[:5] )[:,0] )


# In[ ]:


with strategy.scope():
    
    i = 0
        
    data_generator = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=180,
        zoom_range=0.1,
        horizontal_flip=True
    )

    dg_trainer = data_generator.flow( 
        trainImg, trainMin, 
        batch_size = batch_size
    )

    while i < stop:

        history = model.fit(
            dg_trainer,
            validation_data=(testImg,testMin),
            epochs=n_epochs,
            batch_size = batch_size,
            verbose=verbose)

        i += n_epochs
        timeid = int( time.time() )
        model.save( 'models/rmin-all/%s-%s-%s.h5'%(runName,str(timeid),str(i)), save_format='h5' )

        with open( 'results/results-rmin-all-%s.txt' % runName, 'a' ) as f: 

            print( 'Progress: %d - %d' % ( i, timeid ), file=f )
            print( "Validation accuracy:",*["%.8f"%(x) for x in history.history['loss']], file=f)    
            print( "Test accuracy:",*["%.8f"%(x) for x in history.history['val_loss']],file=f)


# In[11]:


print("Hi")


# In[ ]:




