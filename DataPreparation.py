import pandas as pd
import numpy as np
import datetime
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
    concatenate,
    Bidirectional,
    Lambda,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def create_df(root_path, path):
  apps = []
  for (dirpath, dirnames, filenames) in walk(root_path + path):
      for filename in filenames:
          apps.append(
              pd.read_xml(
                  path_or_buffer=dirpath + "/" + filename,
                  iterparse={"clone": ["similarity", "source1", "source2", "file1", "file2", "startline1", "startline2", "endline1", "endline2", "pcid1", "pcid2"]},
              )
          )
  df = pd.concat(apps)
  df = df.sample(frac=1, random_state=1)
  df["is_clone"] = df["similarity"] >= 70
  df["similarity"] = df["similarity"] / 100
    
  return df

def prepare_data(root_path, train_path, val_path, test_path):
  train_df = create_df(root_path, train_path)
  val_df = create_df(root_path, val_path)
  test_df = create_df(root_path, test_path)

  train_df.dropna()
  val_df.dropna()
  test_df.dropna()
  
  df_resampled = resample_data(train_df.copy())
  
  X_train = df_resampled
  y_train = df_resampled["is_clone"]
  
  X_val = val_df[["source1", "source2"]].copy()
  y_val = val_df["is_clone"].copy()
  
  X_test = test_df[["source1", "source2"]].copy()
  y_test = test_df["is_clone"].copy()

  return [X_train, X_val, X_test, y_train, y_val, y_test]

def resample_data(df):
  df_majority = df[df["is_clone"] == 0]
  df_minority = df[df["is_clone"] == 1]
  
  df_majority_downsampled = resample(
      df_majority, replace=True, n_samples=int(df_minority.shape[0] * 10), random_state=1
  )
      
  return pd.concat([df_majority_downsampled, df_minority])

def tokenize_data(data, tokenizer):
  return tokenizer.texts_to_sequences(data.astype(str).values)

def vectorize_dataset(X_train, X_val, X_test):
  X_train["text"] = X_train[["source1", "source2"]].apply(
    lambda x: str(x[0]) + " " + str(x[1]), axis=1
  )
  t = Tokenizer(filters='!"#$%&()*+,-.;<=>?@[\\]^`|~\t\n')
  t.fit_on_texts(X_train["text"].values)
  
  train_source1_seq = tokenize_data(X_train["source1"], t)
  train_source2_seq = tokenize_data(X_train["source2"], t)
  val_source1_seq = tokenize_data(X_val["source1"], t)
  val_source2_seq = tokenize_data(X_val["source2"], t)
  test_source1_seq = tokenize_data(X_test["source1"], t)
  test_source2_seq = tokenize_data(X_test["source2"], t)

  train_source1_seq = pad_sequences(train_source1_seq, maxlen=1200, padding="pre")
  train_source2_seq = pad_sequences(train_source2_seq, maxlen=1200, padding="pre")
  val_source1_seq = pad_sequences(val_source1_seq, maxlen=1200, padding="pre")
  val_source2_seq = pad_sequences(val_source2_seq, maxlen=1200, padding="pre")
  test_source1_seq = pad_sequences(test_source1_seq, maxlen=1200, padding="pre")
  test_source2_seq = pad_sequences(test_source2_seq, maxlen=1200, padding="pre")

  return [train_source1_seq, train_source2_seq, val_source1_seq, val_source2_seq, test_source1_seq, test_source2_seq, t]
