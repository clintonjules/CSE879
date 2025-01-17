import tensorflow as tf
from tensorflow.keras import layers, models
import gc
import os

# Force CPU usage to avoid GPU memory issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def create_model(num_classes, img_size, learning_rate=0.001):
    # Clear any existing models/layers from memory
    tf.keras.backend.clear_session()
    gc.collect()
    
    try:
        # Create preprocessing layers
        inputs = layers.Input(shape=(img_size, img_size, 3))
        
        # Normalize and resize
        x = layers.Rescaling(1./255)(inputs)
        x = layers.Resizing(96, 96)(x)
        
        # MobileNetV2 preprocessing
        x = layers.Lambda(
            lambda x: tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(x, tf.float32))
        )(x)
        
        # Load pre-trained MobileNetV2 with specific input shape
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(96, 96, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Connect base model
        x = base_model(x)
        
        # Add custom top layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation='relu')(x)  # Reduced from 256
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='classifier')
        
        # Use legacy optimizer for M1/M2 Macs
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    except Exception as e:
        print(f"Error in create_model: {str(e)}")
        raise

def unfreeze_model(model, num_layers_to_unfreeze=15):  # Reduced from 30
    """
    Unfreeze the last n layers of the base model for fine-tuning
    """
    try:
        # Clear memory
        gc.collect()
        
        # Find MobileNetV2 layer
        base_model = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                base_model = layer
                break
        
        if base_model is None:
            raise ValueError("Could not find base model layer")
        
        # Unfreeze specific layers
        base_model.trainable = True
        
        # Freeze all layers except the last n
        for layer in base_model.layers[:-num_layers_to_unfreeze]:
            layer.trainable = False
            
        # Print summary of trainable layers
        print("\nTrainable layers after unfreezing:")
        trainable_count = 0
        for layer in base_model.layers[-num_layers_to_unfreeze:]:
            print(f"Unfrozen layer: {layer.name}")
            trainable_count += 1
        print(f"\nTotal number of unfrozen layers: {trainable_count}")
        
        # Use legacy optimizer for M1/M2 Macs with reduced learning rate
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=5e-6)  # Reduced from 1e-5
        
        # Recompile model
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    except Exception as e:
        print(f"Error in unfreeze_model: {str(e)}")
        raise