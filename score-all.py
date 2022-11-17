#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Python Mobules
import os, json, time
import numpy as np
from sys import argv, stdout
import tensorflow as tf
import tensorflow.keras as keras

# Reduce warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Basic information
n_epochs = 100
stop = 2000
batch_size = 128
learning_rate = 0.01
verbose = 0


# In[2]:


buildEnv = False
freshStart = True

# Am I in a jupyter notebook?
try:
    get_ipython().__class__.__name__
    buildEnv = True
    argv = [ None, 'run-11-16' ]
    verbose = 1
    n_epochs = 2
    stop = 3
    print("In Notebook")

# Or am I in a python script?
except:
    if len( argv ) < 2:
        print("ERROR:  Need unique name" )
        
# Cmd line variables
runName = argv[1]

if len( argv ) >=3:
    freshStart = False
    modelLoc = argv[2]
    print("Model Loc: (%s) - %s" % (os.path.exists(modelLoc), modelLoc) )

# Misc initialization for build environment
if buildEnv: 
    import matplotlib.pyplot as plt
    
runName = argv[1]

if 'test' in runName:
    verbose = 1
    n_epochs = 2
    stop = 3



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

# # Standard single GPU on backus
# else:
#     tf.config.set_visible_devices(devices[0:1]+devices[1:2])
#     tf.config.experimental.set_memory_growth(devices[1],True)
#     strategy = tf.distribute.OneDeviceStrategy('GPU:0') # Use local GPU

else:
    tf.config.set_visible_devices(devices[0:1]+devices[2:3])
    tf.config.experimental.set_memory_growth(devices[1],True)
    strategy = tf.distribute.OneDeviceStrategy('GPU:1') # Use local GPU


# In[3]:


def correlation( v1, v2 ):
    corr = np.corrcoef( v1, v2 )[0,1]
    return corr

def buildModel( trainImg ):

    print("Building Model: ")

    # Build input layer
    x = keras.layers.Input(shape=trainImg.shape[1:])
    y = x    

    # Build resnet layer without top layer
    resnet = keras.applications.ResNet50V2(
        #weights=None,
        include_top = False,
        input_shape = y.shape[1:], 
    )

    y = resnet(y)

    # Flatten for final layer
    y = keras.layers.Flatten()(y)

    # Mid layer before final
    output = keras.layers.Dense( 16, activation= keras.activations.relu )
    y = output(y)

    # Final layer.  Predicting a single value
    #y = keras.layers.Dense( 1, activation='sigmoid' )(y)
    y = keras.layers.Dense( 1, )(y)

    model = keras.Model( x, y )
    model.compile( 
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.mean_squared_error,
        )


    return model
# End building model


# In[5]:


with strategy.scope():
    
    # Prepare data
    with open( 'data/data-key.json' ) as keyFile:
        key = json.load( keyFile )
        
    # Load testing data
    testScore = np.load('data/test-data.npy' )[:,key['score']]
    testImg   = np.load('data/test-img.npy' )
    
    # Load training data
    trainScore = np.load('data/train-data.npy' )[:,key['score']]
    trainImg   = np.load('data/train-img.npy' )
    
    # Reshape scores
    testScore  = np.reshape( testScore,  (  testScore.shape[0], 1 ) )
    trainScore = np.reshape( trainScore, ( trainScore.shape[0], 1 ) )
        
    print( 'testScore: ', testScore.shape )
    print( 'testImg:   ',   testImg.shape )

    print( 'trainScore:', trainScore.shape )
    print( 'trainImg:  ',   trainImg.shape )


# In[6]:


with strategy.scope():
    
    def readModel( modelLoc ):
        
        print("Reading Model: %s" % modelLoc )
        
        model = keras.models.load_model(modelLoc) 
        
        return model
        
        

    
    if freshStart:
        model = buildModel( trainImg )
        
    else:
        model = readModel(modelLoc)
    
    model.summary(expand_nested=False)

    
# keras.utils.plot_model(model, 
#                        to_file='score-all.png',
#                        show_shapes=True,
#                        expand_nested=True)


# In[7]:


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
        trainImg, trainScore, 
        batch_size = batch_size
    )
    
    std_out = stdout

    while i < stop:

        history = model.fit(
            dg_trainer,
            validation_data=(testImg,testScore),
            epochs=n_epochs,
            batch_size = batch_size,
            verbose=verbose)
        

        testCorrr  = correlation( testScore[:,0],  model.predict( testImg )[:,0] )
        trainCorr = correlation( trainScore[:,0], model.predict( trainImg )[:,0] )

        i += n_epochs
        timeid = int( time.time() )
        
        with open( 'results/results-score-all-%s.txt' % runName, 'a' ) as f: 
            print( 'Progress: %d %d %f %f' % 
                   ( i, timeid, testCorrr, trainCorr, ),
                  file = f
                 )

            print( "Validation accuracy:",*["%.8f"%(x) for x in history.history['loss']], file = f)    
            print( "Test accuracy:",*["%.8f"%(x) for x in history.history['val_loss']], file = f)
            

        #model.save( 'models/score-all/%s-%s-%s.h5'%(runName,str(timeid),str(i)), save_format='h5' )


# In[ ]:




