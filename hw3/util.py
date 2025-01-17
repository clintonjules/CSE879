import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

def load_data(buffer_size=10000, batch_size=64):
    # Load IMDB dataset from TensorFlow Datasets
    (train_data, test_data), info = tfds.load('imdb_reviews', 
                                              split=['train[:80%]', 'train[80%:]'],
                                              as_supervised=True,
                                              with_info=True)

    # Prepare data (tokenization, padding, etc.)
    tokenizer = tf.keras.layers.TextVectorization(max_tokens=10000, output_sequence_length=500)
    train_texts = train_data.map(lambda x, y: x)
    tokenizer.adapt(train_texts)

    train_data = train_data.map(lambda x, y: (tokenizer(x), y)).cache().shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_data = test_data.map(lambda x, y: (tokenizer(x), y)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Split 20% of the training data for validation
    validation_split = int(0.2 * len(train_data))
    val_data = train_data.take(validation_split)
    train_data = train_data.skip(validation_split)

    return train_data, val_data, test_data

def create_callbacks(monitor='val_loss', patience=5):
    """Create callbacks for training, including early stopping."""
    return [
        EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True)
    ]