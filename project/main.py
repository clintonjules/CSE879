import tensorflow as tf
import pandas as pd
from model import create_model, unfreeze_model
from utils import prepare_datasets, plot_metrics, create_confusion_matrix
import os
import gc

# Force CPU usage to avoid GPU memory issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def main():
    try:
        # Clear any existing models/data from memory
        tf.keras.backend.clear_session()
        gc.collect()
        
        # Hyperparameters
        IMG_SIZE = 32
        BATCH_SIZE = 16
        INITIAL_EPOCHS = 20
        FINE_TUNING_EPOCHS = 30
        LEARNING_RATE = 0.001
        
        # Get current working directory
        data_dir = "data"
        
        # Prepare datasets
        print("Preparing datasets...")
        train_ds, val_ds, test_ds, num_classes = prepare_datasets(
            data_dir, 
            IMG_SIZE, 
            BATCH_SIZE
        )
        
        # Custom callback to save metrics to CSV
        class MetricsCSVCallback(tf.keras.callbacks.Callback):
            def __init__(self, phase="initial"):
                super().__init__()
                self.phase = phase
                
            def on_epoch_end(self, epoch, logs=None):
                metrics_df = pd.DataFrame({
                    'phase': [self.phase],
                    'epoch': [epoch],
                    'train_loss': [logs['loss']],
                    'train_accuracy': [logs['accuracy']],
                    'val_loss': [logs['val_loss']],
                    'val_accuracy': [logs['val_accuracy']]
                })
                metrics_df.to_csv('training_metrics.csv', 
                                mode='a', 
                                header=not os.path.exists('training_metrics.csv'),
                                index=False)
                del metrics_df
                gc.collect()

        class SimplifiedEarlyStopping(tf.keras.callbacks.EarlyStopping):
            def __init__(self, phase="initial", **kwargs):
                super().__init__(**kwargs)
                self.phase = phase
                
            def on_train_end(self, logs=None):
                # Only save when early stopping actually occurred
                if self.stopped_epoch > 0:
                    stopping_df = pd.DataFrame({
                        'phase': [self.phase],
                        'stopped_epoch': [self.stopped_epoch]
                    })
                    stopping_df.to_csv(f'early_stopping_{self.phase}.csv', index=False)
                    del stopping_df
                    gc.collect()

        # Initial training phase
        print("Initial training phase...")
        model = create_model(num_classes, IMG_SIZE, LEARNING_RATE)
        
        early_stopping = SimplifiedEarlyStopping(
            phase="initial",
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            mode='min'
        )
        
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'best_model_initial.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
        
        metrics_csv = MetricsCSVCallback(phase="initial")
        
        # Initial training
        history1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=INITIAL_EPOCHS,
            callbacks=[early_stopping, checkpoint, metrics_csv],
            verbose=1
        ).history
        
        # Fine-tuning phase
        print("\nFine-tuning phase...")
        model = unfreeze_model(model)
        
        early_stopping_ft = SimplifiedEarlyStopping(
            phase="fine_tuning",
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            mode='min'
        )
        
        checkpoint_ft = tf.keras.callbacks.ModelCheckpoint(
            'best_model_fine_tuned.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
        
        metrics_csv_ft = MetricsCSVCallback(phase="fine_tuning")
        
        # Fine-tuning training
        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=FINE_TUNING_EPOCHS,
            callbacks=[early_stopping_ft, checkpoint_ft, metrics_csv_ft],
            verbose=1
        ).history
        
        # Combine histories
        combined_history = {
            'accuracy': history1['accuracy'] + history2['accuracy'],
            'val_accuracy': history1['val_accuracy'] + history2['val_accuracy'],
            'loss': history1['loss'] + history2['loss'],
            'val_loss': history1['val_loss'] + history2['val_loss']
        }
        
        # Plot metrics
        plot_metrics(combined_history)
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
        print(f"\nTest accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        
        # Save test metrics
        test_metrics_df = pd.DataFrame({
            'metric': ['test_loss', 'test_accuracy'],
            'value': [test_loss, test_accuracy]
        })
        test_metrics_df.to_csv('test_metrics.csv', index=False)
        
        # Create confusion matrix
        create_confusion_matrix(model, test_ds)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
        
    finally:
        # Clean up
        tf.keras.backend.clear_session()
        gc.collect()

if __name__ == "__main__":
    main()