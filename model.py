import tensorflow as tf
from tensorflow.keras import layers, regularizers

def build_lstm_model(vocab_size, embedding_dim, lstm_units, regularizer_rate):
    inputs = tf.keras.Input(shape=(None,))
    x = layers.Embedding(vocab_size, embedding_dim)(inputs)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True,
                                         kernel_regularizer=regularizers.l2(regularizer_rate)))(x)
    # Attention requires [query, value] format
    query = x
    value = x
    attention_output = layers.Attention()([query, value])
    x = layers.GlobalAveragePooling1D()(attention_output)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(regularizer_rate))(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def build_gru_model(vocab_size, embedding_dim, gru_units, regularizer_rate):
    inputs = tf.keras.Input(shape=(None,))
    x = layers.Embedding(vocab_size, embedding_dim)(inputs)
    x = layers.Bidirectional(layers.GRU(gru_units, return_sequences=True,
                                        kernel_regularizer=regularizers.l2(regularizer_rate)))(x)
    # Attention requires [query, value] format
    query = x
    value = x
    attention_output = layers.Attention()([query, value])
    x = layers.GlobalAveragePooling1D()(attention_output)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(regularizer_rate))(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model