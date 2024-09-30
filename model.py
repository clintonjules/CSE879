import tensorflow as tf
from tensorflow.keras import layers, models, regularizers # type: ignore

def build_model(layers_units, activation, optimizer='adam', learning_rate=0.001, regularizer=None):
    model = models.Sequential()
    
    # Specify the input shape explicitly
    model.add(layers.Input(shape=(28, 28, 1)))
    model.add(layers.Flatten())

    # Add dense layers with optional L2 regularization
    for units in layers_units:
        if regularizer == 'l2':
            model.add(layers.Dense(units, activation=activation, kernel_regularizer=regularizers.l2(0.01)))
        else:
            model.add(layers.Dense(units, activation=activation))
    
    # Add the output layer
    model.add(layers.Dense(10, activation='softmax'))

    # Compile the model with the Adam optimizer and specified learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model