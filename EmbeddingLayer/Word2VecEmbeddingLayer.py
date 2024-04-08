import numpy as np
from tensorflow.keras.layers import Embedding

def create_fasttext_layer(t, train_source1_seq):

  embeddings_index = {}
  f = open("bytecode_embeddings_word2vec.txt")

  for line in f:
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype="float32")
      embeddings_index[word] = coefs
  f.close()
  
  not_present_list = []
  vocab_size = len(t.word_index) + 1
  
  embedding_matrix = np.zeros((vocab_size, len(embeddings_index["return"])))
  for word, i in t.word_index.items():
    embedding_vector = None
    if word in embeddings_index.keys():
        embedding_vector = embeddings_index.get(word)
    else:
        not_present_list.append(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        embedding_matrix[i] = np.zeros(len(embeddings_index["return"]))
    
  return Embedding(
    input_dim=len(t.word_index) + 1,
    output_dim=len(embeddings_index["return"]),
    weights=[embedding_matrix],
    input_length=train_source1_seq.shape[1],
    trainable=False,
  )
