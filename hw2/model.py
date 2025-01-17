import tensorflow as tf

def build_model_1():
    """Builds a simple CNN with L2 regularization."""
    l2_reg = tf.keras.regularizers.l2(0.001)
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3),
                               kernel_regularizer=l2_reg),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2_reg),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2_reg),
        tf.keras.layers.Dense(100, activation='softmax')  # Output for 100 classes
    ])
    return model

def build_model_2():
    """Builds a deeper CNN with Dropout and L2 regularization."""
    l2_reg = tf.keras.regularizers.l2(0.001)
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3),
                               kernel_regularizer=l2_reg),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2_reg),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2_reg),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=l2_reg),
        tf.keras.layers.Dropout(0.5),  # Dropout for regularization
        tf.keras.layers.Dense(100, activation='softmax')
    ])
    return model