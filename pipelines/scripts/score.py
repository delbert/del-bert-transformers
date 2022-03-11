import os
import sys

import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

from azureml.core import Dataset , Datastore , Model , Run

import logging
logging.basicConfig ( level = logging.INFO )

from util.tokenization import *

#import mlflow.tensorflow
#mlflow.tensorflow.autolog()

module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer ( module_url , trainable = True )

run = Run.get_context()
workspace = run.experiment.workspace
#test = Dataset.get_by_name ( workspace = workspace , name = 'bold_test' ).to_pandas_dataframe() #bold is a subset

datastore = Datastore.get ( workspace , 'score_tf_bert' )

datastore_paths = [ ( datastore , sys.argv[ 1 ] ) ]

test_ds = Dataset.Tabular.from_delimited_files ( path = datastore_paths )

test_ds = test_ds.register ( workspace = workspace ,
                                name = 'bold_test' ,
                                description = 'latest bold_test scoring dataset' ,
                                create_new_version = True )

test = test_ds.to_pandas_dataframe()

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
test_input = bert_encode ( test.Review.values , tokenizer , max_len=max_len )

model = build_model ( bert_layer , max_len = max_len )
model.summary ()

aml_model = Model (  workspace = workspace , name = 'tf-bert-nlp' )
aml_model.download ( target_dir = './' )

#tf.saved_model.load ( model , './model/' )

model.load_weights ( './model/model.h5' )
test_pred = model.predict ( test_input )

print ( test_pred )

mounted_output_path = sys.argv[ 2 ]

fileName = str ( datetime.datetime.now() ).replace ( ' ' , '-' ).replace ( ':' , '-' ).replace ( '.' , '-')  + '-score.csv'

np.savetxt ( os.path.join ( mounted_output_path , fileName ) , test_pred , delimiter = ',' )
