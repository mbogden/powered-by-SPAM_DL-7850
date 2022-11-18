#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Python Mobules
import os, json, time, argparse
import numpy as np
from sys import argv
import tensorflow as tf
import tensorflow.keras as keras
import keras_tuner as kt


# Reduce warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Basic information
print("Modules: Imported")


# In[2]:


# Command line arguments

buildEnv = False

parser = argparse.ArgumentParser()
parser.add_argument( '-runName', )
parser.add_argument( '-modelLoc', )
parser.add_argument( "-tid", default = '587722984435351614',  type=str )
parser.add_argument( "-start",   default = 0,    type=int, )
parser.add_argument( "-stop",    default = 3,  type=int, )
parser.add_argument( "-verbose", default = 0,    type=int, )
parser.add_argument( "-num_epochs",    default=2,   type=int )
parser.add_argument( "-learning_rate", default=0.01, type=float )
parser.add_argument( "-batch_size",    default=16,  type=int )
parser.add_argument( "-save_model",    default=True,  type=bool )

# Core Model types
parser.add_argument( "-model", default = 'efficientNetB1', )
parser.add_argument( "-pool",   default = None )
parser.add_argument( "-weights", default = 'imagenet',  type=str )

# Final layers
parser.add_argument( "-f_depth", default = 3,  type=int )
parser.add_argument( "-f_width", default = 32, type=int )
parser.add_argument( "-f_activation", default = 'tanh', type=str )
parser.add_argument( "-output_activation", default = None )

print("Args: Initialized")


# In[3]:


# Am I in a jupyter notebook?
try:
    get_ipython().__class__.__name__
    buildEnv = True
    print("In Notebook")
    cmdStr = ''
    cmdStr += ' -runName efficient-tuner'
    
    args = parser.parse_args(cmdStr.split())



# Or am I in a python script?
except:
    args = parser.parse_args()
    
    # Check for valid runName
    print( 'runName: ', args.runName )
    if args.runName == None:
        print("WARNING:  runName required")
        exit()
    
    runName = args.runName


# In[4]:


# Read current devices
devices = tf.config.get_visible_devices()

# If no GPU found, use CPU
if len(devices) == 1:
    strategy = tf.distribute.OneDeviceStrategy('CPU') # Use local GPU

# if buildEnv, probably on hamilton. Use GPU 2
elif buildEnv:
    tf.config.set_visible_devices(devices[0:1]+devices[1:2])
    tf.config.experimental.set_memory_growth(devices[1],True)
    strategy = tf.distribute.OneDeviceStrategy('GPU:0') # Use local GPU

# Standard single GPU on backus
else:
    tf.config.experimental.set_memory_growth(devices[1],True)
    strategy = tf.distribute.OneDeviceStrategy('GPU:1') # Use local GPU
if buildEnv:    print( 'Devices:', devices )


# In[5]:


def loadData( args ):
    
    # Prepare data
    with open( 'data/data-key.json' ) as keyFile:
        key = json.load( keyFile )
    
    print("Reading Data for: %s" % args.tid )
        
    data = np.load('data/norm-targets/data-%s.npy' % args.tid)
    img = np.load( 'data/raw/img-%s.npy' % args.tid )
    
    # Rescale image values based on core model being used.
    if 'efficientNet' in args.model :
        img *= 255

    if buildEnv:
        print( 'data: ', data.shape )
        print( 'img:  ', img.shape, np.amin(img), np.amax(img) )
        
        
    
    # Reserve every 5th for training
    n = data.shape[0]
    m = np.full( n, False )
    m[::5] = True
    
    # Seperate training from testing
    testData = data[m] 
    testImg  = img[m]
    
    trainData = data[~m]
    trainImg = img[~m]
    
    # Shuffle training data
    
    p = np.random.permutation( trainData.shape[0] )
    trainData = trainData[p]
    trainImg = trainImg[p]
    
    if buildEnv:
        print( "test:  ", testData.shape )
        print( "train: ", trainData.shape )
    
    # Only grab scores for prediction    
    trainScore = trainData[:,key['score']]
    testScore  =  testData[:,key['score']]
    
    # Reshape scores
    testScore  = np.reshape( testScore,  (  testScore.shape[0], 1 ) )
    trainScore = np.reshape( trainScore, ( trainScore.shape[0], 1 ) )
    
    # print scores shape
    
    if buildEnv:
        print('testScore:', testScore.shape, np.amin( testScore ), np.amax(testScore) )
        print("trainScore:", trainScore.shape )
    
    # Set standardized name
    
    return trainImg, trainScore, testImg, testScore


# In[6]:


with strategy.scope(): 

    # Load Data
    X, Y, Xval, Yval = loadData( args )


# In[7]:


def buildModel( args, X, Y ):
    
    print("Building Model: ")
        
    # Build input layer
    x = keras.layers.Input(shape=X.shape[1:])
    y = x    
    
    # What type of model
    
    if 'efficientNet' in args.model:
    
        if   args.model == 'efficientNetB0':  core_model = tf.keras.applications.EfficientNetB0
        elif args.model == 'efficientNetB1':  core_model = tf.keras.applications.EfficientNetB1
        elif args.model == 'efficientNetB2':  core_model = tf.keras.applications.EfficientNetB2
        elif args.model == 'efficientNetB3':  core_model = tf.keras.applications.EfficientNetB3
        elif args.model == 'efficientNetB4':  core_model = tf.keras.applications.EfficientNetB4
        elif args.model == 'efficientNetB5':  core_model = tf.keras.applications.EfficientNetB5
        elif args.model == 'efficientNetB6':  core_model = tf.keras.applications.EfficientNetB6
        elif args.model == 'efficientNetB7':  core_model = tf.keras.applications.EfficientNetB7
            
        core_model =  core_model(
                include_top=False,
                weights=args.weights,
                input_shape=y.shape[1:],
                pooling=args.pool,
            )
    
    elif args.model == 'resnet':
        # Build resnet layer without top layer
        core_model = keras.applications.ResNet50V2(
            include_top = False,
            weights = args.weights,
            input_shape = y.shape[1:], 
        )
    else:
        print("NO MODEL TYPE SELECTED")
        return None
    
    # Add core model
    y = core_model(y)
    
    # Flatten for final layers
    y = keras.layers.Flatten()(y)
    
    for i in range( args.f_depth ):
        
        if args.f_activation == None:
            y = keras.layers.Dense( args.f_width, activation= keras.activations.relu, name='act_relu_%d'%i )(y)
            
        if args.f_activation == 'relu':
            y = keras.layers.Dense( args.f_width, activation= keras.activations.relu, name='act_relu_%d'%i )(y)
            
        if args.f_activation == 'tanh':
            y = keras.layers.Dense( args.f_width, activation= keras.activations.tanh, name='act_tanh_%d'%i )(y)
    
    # Final layer.
    if args.output_activation == None or args.output_activation == 'linear':
        y = keras.layers.Dense( Y.shape[1], name='output_linear' )(y)
        
    elif args.output_activation == 'softmax':
        y = keras.layers.Dense( Y.shape[1] , activation='softmax', name='output_softmax' )(y)
        
    elif args.output_activation == 'sigmoid':
        y = keras.layers.Dense( Y.shape[1] , activation='sigmoid', name='output_sigmoid' )(y)


    # Compile
    model = keras.Model( x, y )
    model.compile( 
        optimizer=keras.optimizers.Adam(learning_rate= args.learning_rate ),
        loss=keras.losses.mean_squared_error,
        )
    
    return model
# end building model
    


# In[ ]:


def build( hp ):
    
    cmdStr = ''
    cmdStr += ' -runName test-efficient-tuner'
    cmdStr += ' -start 1'
    cmdStr += ' -stop 20'
    cmdStr += ' -num_epochs 20'
    cmdStr += ' -learning_rate %f' % hp.Float("lr", default=0.0001, min_value=1e-6, max_value=.01, sampling="log")
    # cmdStr += ' -pool None'
    # cmdStr += ' -verbose 1'
    # cmdStr += ' -save_model False'
    # cmdStr += ' -f_depth 3'
    # cmdStr += ' -f_width 32'
    # cmdStr += ' -f_activation tanh'
    
    cmdStr += ' -pool %s'         % hp.Choice("pool", [ 'None', 'avg', 'max' ], default=None,  )
    cmdStr += ' -f_depth %d'     % hp.Int( 'f_depth', default=3, min_value=1, max_value=8, step=1 )
    cmdStr += ' -f_width %d'     % hp.Int( 'f_width', default=32, min_value=8, max_value=64, step=8 )
    cmdStr += ' -f_activation %s' % hp.Choice( 'f_activation', default='tanh', values=[ 'tanh', 'relu' ] )
    cmdStr += ' -output_activation %s' % hp.Choice( 'output_activation', default='linear', values=[ 'linear', 'sigmoid', 'softmax' ] )
    cmdStr += ' -model %s' % hp.Choice( 'model', default='efficientNetB0', \
                                       values=[ 'efficientNetB0', 'efficientNetB1', 'efficientNetB2', 'efficientNetB3'] )

    
    print("Parsing Args")
    args = parser.parse_args(cmdStr.split())

    model = buildModel( args, X, Y )
    print('Model: ', model )
    
    return model
    
with strategy.scope():
    
    hp = kt.HyperParameters()
        
    # Build Data Generator
    data_generator = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=180,
        zoom_range=0.1,
        horizontal_flip=True,
    )

    dg_trainer = data_generator.flow( 
        X, Y, 
        batch_size = args.batch_size
    )
        
    print("Something: ")
    tuner = kt.Hyperband(
         build,
         objective=kt.Objective("val_loss", direction="min"),
         max_epochs=1000,
         factor=3,
         hyperband_iterations=3,
         directory='test_6',)

    print("Searching: ", )
    tuner.search( dg_trainer, epochs=1000, validation_data=(Xval, Yval))
    

