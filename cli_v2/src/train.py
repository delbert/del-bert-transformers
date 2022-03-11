import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

from azureml.core import Dataset , Run

import logging
logging.basicConfig ( level = logging.INFO )

from util.tokenization import *

#import mlflow.tensorflow
#mlflow.tensorflow.autolog()

module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer ( module_url , trainable = True )

run = Run.get_context()
workspace = run.experiment.workspace
train = Dataset.get_by_name ( workspace = workspace , name = 'bold_train' ).to_pandas_dataframe() #bold is a subset
test = Dataset.get_by_name ( workspace = workspace , name = 'bold_test' ).to_pandas_dataframe() #bold is a subset

#train = pd.read_json ( 'bold_train.json' ) . reset_index ( drop = True )  #bold is a subset
#test = pd.read_json ( 'bold_test.json' ) . reset_index ( drop = True )  #bold is a subset

train [ 'Review' ] = ( train [ 'title' ].map ( str ) + ' ' + train [ 'body' ] ).apply ( lambda row : row.strip() )
test [ 'Review' ] = ( test [ 'title' ].map ( str ) + ' ' + test [ 'body' ] ).apply ( lambda row : row.strip() )

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer ( vocab_file , do_lower_case )

def bert_encode ( texts , tokenizer , max_len = 512 ) :
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts :
        text = tokenizer.tokenize ( text )
            
        text = text [ :max_len-2 ]
        input_sequence = [ '[CLS]' ] + text + [ '[SEP]' ]
        pad_len = max_len - len ( input_sequence )
        
        tokens = tokenizer.convert_tokens_to_ids ( input_sequence ) + [ 0 ] * pad_len
        pad_masks = [ 1 ] * len ( input_sequence ) + [ 0 ] * pad_len
        segment_ids = [ 0 ] * max_len
        
        all_tokens.append ( tokens )
        all_masks.append ( pad_masks )
        all_segments.append ( segment_ids )
    
    return np.array ( all_tokens ) , np.array ( all_masks ) , np.array ( all_segments )

def build_model ( bert_layer , max_len = 512 ) :
    input_word_ids = tf.keras.Input ( shape = ( max_len , ) , dtype=tf.int32 , name = 'input_word_ids' )
    input_mask = tf.keras.Input ( shape = ( max_len , ) , dtype=tf.int32 , name = 'input_mask' )
    segment_ids = tf.keras.Input ( shape = ( max_len , ) , dtype=tf.int32 , name = 'segment_ids' )

    pooled_output , sequence_output = bert_layer ( [ input_word_ids , input_mask , segment_ids ] )
    clf_output = sequence_output [ : , 0 , : ]
    net = tf.keras.layers.Dense ( 64 , activation = 'relu' ) ( clf_output )
    net = tf.keras.layers.Dropout ( 0.2 ) ( net )
    net = tf.keras.layers.Dense( 32 , activation = 'relu' ) ( net )
    net = tf.keras.layers.Dropout ( 0.2 ) ( net )
    out = tf.keras.layers.Dense ( 3 , activation = 'softmax' ) ( net )
    
    model = tf.keras.models.Model ( inputs = [ input_word_ids , input_mask , segment_ids ] , outputs=out )
    model.compile ( tf.keras.optimizers.Adam ( lr = 1e-5 ) , loss = 'categorical_crossentropy' , metrics = [ 'accuracy' ] )
    
    return model

max_len = 150
train_input = bert_encode ( train.Review.values , tokenizer , max_len=max_len )
test_input = bert_encode ( test.Review.values , tokenizer , max_len=max_len )
train_labels = tf.keras.utils.to_categorical ( train.label.values , 3 )

model = build_model ( bert_layer , max_len = max_len )
model.summary ()

os.makedirs ( './outputs/model/weights' , exist_ok = True )
checkpoint = tf.keras.callbacks.ModelCheckpoint ( './outputs/model/model.h5' , monitor = 'val_accuracy' , save_best_only = True , verbose = 1 )
earlystopping = tf.keras.callbacks.EarlyStopping ( monitor = 'val_accuracy' , patience = 5 , verbose = 1 )

train_history = model.fit (
    train_input , train_labels , 
    validation_split = 0.2 ,
    epochs = 3 ,
    callbacks = [ checkpoint , earlystopping ] ,
    batch_size = 32 ,
    verbose = 1 )

model.load_weights ( './outputs/model/model.h5' )
test_pred = model.predict ( test_input )

print ( test_pred )

tf.saved_model.save ( model , './outputs/model/' )
