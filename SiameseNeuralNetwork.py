import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Bidirectional,
    Concatenate,
    Dense,
    Dot,
    Dropout,
    Dense,
    Embedding,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Input,
    LSTM,
    Lambda,
    Multiply,
    Permute,
    Subtract
)
from tensorflow.keras.models import Model

def attention_layer(inputs):
    attention = Dot(axes=-1)(inputs)
    attention_weights_1 = Lambda(lambda x: softmax(x, axis=1))(attention)
    attention_weights_2 = Permute((2,1))(Lambda(lambda x: softmax(x, axis=2))(attention))
    context_vector1 = Dot(axes=1)([attention_weights_1, inputs[0]])
    context_vector2 = Dot(axes=1)([attention_weights_2, inputs[1]])
    return context_vector1, context_vector2

def siamese_model(train_source1_seq, train_source2_seq, embedding_layer):
    bi_lstm_layer = Bidirectional(LSTM(300, return_sequences=True))
    batch_norm_layer = BatchNormalization(axis=2)
    
    sequence_1_input = Input(shape=(train_source1_seq.shape[1],), dtype="int32")
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    
    x1 = bi_lstm_layer(batch_norm_layer(embedded_sequences_1))
    
    sequence_2_input = Input(shape=(train_source2_seq.shape[1],), dtype="int32")
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    
    x2 = bi_lstm_layer(batch_norm_layer(embedded_sequences_2))
    
    attention1, attention2 = attention_layer([x1, x2])
    
    x1 = Concatenate()([x1, attention2, Concatenate()([Subtract()([x1, attention2]), Multiply()([x1, attention2])])])
    x2 = Concatenate()([x2, attention1, Concatenate()([Subtract()([x2, attention1]), Multiply()([x2, attention1])])]) 
       
    bi_lstm_layer_2 = Bidirectional(LSTM(300, return_sequences=True))
    x1 = bi_lstm_layer_2(x1)
    x2 = bi_lstm_layer_2(x2)

    x1 = Concatenate()([GlobalAveragePooling1D()(x1), GlobalMaxPooling1D()(x1)])
    x2 = Concatenate()([GlobalAveragePooling1D()(x2), GlobalMaxPooling1D()(x2)])

    merged = Concatenate()([x1, x2])
    
    merged = BatchNormalization()(merged)
    merged = Dense(300, activation='elu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.5)(merged)
    merged = Dense(300, activation='elu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.5)(merged)
    merged = Dense(1, activation='sigmoid')(merged)
    
    model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=merged)
    return model
