from SiameseNeuralNetwork import siamese_model
from EmbeddingLayer.Word2VecLayer import create_word2vec_layer
from DataPreparation import prepare_data, vectorize_dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.utils import resample
from statistics import mean
from os import walk
import os

import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization,
    Dense,
    Embedding,
    LSTM,
    Bidirectional,
    Lambda,
    Conv1D,
    MaxPooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

#Threshold that determines whether two methods are clones or not for NiCad
nicad_threshold = 0.7

#Threshold that determines whether two methods are clones or not for the predictions
pred_threshold = 0.98

def get_np_preds(preds, actual):
    newarr = []
    for pred in preds:
        newarr.append(pred[0])
    arr1 = np.array(actual)
    arr2 = np.array(newarr)
    return [arr1, arr2]

def is_clone(arr, threshold):
    new_arr = []
    for elm in arr:
        if elm >= threshold:
            new_arr.append(True)
        else:
            new_arr.append(False)
    return new_arr

def print_scores(actual_arr, preds_arr):
  
  print(preds_arr[:20])
  print(actual_arr[:20])

  actual_is_clone = is_clone(actual_arr, nicad_threshold)
  preds_is_clone = is_clone(preds_arr, pred_threshold)
  
  print("accuracy: ", accuracy_score(actual_is_clone, preds_is_clone))
  print("precision: ", precision_score(actual_is_clone, preds_is_clone))
  print("recall: ", recall_score(actual_is_clone, preds_is_clone))
  print("f1_score", f1_score(actual_is_clone, preds_is_clone))


root_path = "/<your_path>/ByClone/"

train_path = "Train-Build/"
val_path = "Val-Build/"
test_path = "Test-Obf/"

[X_train, X_val, X_test, y_train, y_val, y_test] = prepare_data(root_path, train_path, val_path, test_path)

[train_source1_seq, train_source2_seq, val_source1_seq, val_source2_seq, test_source1_seq, test_source2_seq, t] = vectorize_dataset(X_train, X_val, X_test)

embedding_layer = create_instruction2vec_layer(t, train_source1_seq)

model = siamese_model(train_source1_seq, train_source2_seq, embedding_layer)

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])

#If you want to save a checkpoint
checkpoint_path = "<your_checkpoint_name>.ckpt"

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True, verbose=1
)

model.fit(
    [train_source1_seq, train_source2_seq],
    y_train.values.reshape(-1, 1),
    epochs=50,
    validation_data=([val_source1_seq, val_source2_seq], y_val.values.reshape(-1, 1)),
    batch_size=64,
    callbacks=[cp_callback]
)

preds = model.predict([test_source1_seq, test_source2_seq])
[actual_arr, preds_arr] = get_np_preds(preds, y_test)
print_scores(actual_arr, preds_arr)
