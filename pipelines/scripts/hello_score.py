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


run = Run.get_context()
workspace = run.experiment.workspace

datastore = Datastore.get ( workspace , 'score_tf_bert' )

datastore_paths = [ ( datastore , sys.argv[ 1 ] ) ]

test_ds = Dataset.Tabular.from_delimited_files ( path = datastore_paths )

test_ds = test_ds.register ( workspace = workspace ,
                                name = 'bold_test' ,
                                description = 'latest bold_test scoring dataset' ,
                                create_new_version = True )

test = test_ds.to_pandas_dataframe()

print ( test )

mounted_output_path = sys.argv[ 2 ]

print ( 'mop=>' + mounted_output_path + '<' )

fileName = str ( datetime.datetime.now() ).replace ( ' ' , '-' ).replace ( ':' , '-' ).replace ( '.' , '-')  + '-hello.csv'

print ( 'fn=>' + fileName + '<' )

test.to_csv ( mounted_output_path + '/' + fileName )