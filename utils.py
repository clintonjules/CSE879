import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import gc
import os
import pandas as pd

def prepare_datasets(data_dir, img_size, batch_size):
    try:
        # Clear any existing datasets from memory
        gc.collect()
        
        # Data augmentation for training
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.2),
        ])
        
        # Load and split datasets
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_size, img_size),
            batch_size=batch_size,
            label_mode='categorical'
        )
        
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_size, img_size),
            batch_size=batch_size,
            label_mode='categorical'
        )
        
        # Create a test dataset from validation set
        val_batches = tf.data.experimental.cardinality(val_ds)
        test_ds = val_ds.take(val_batches // 2)
        val_ds = val_ds.skip(val_batches // 2)
        
        # Get number of classes and class names
        num_classes = len(train_ds.class_names)
        
        # Save class labels and their encodings
        class_labels_df = pd.DataFrame({
            'class_name': train_ds.class_names,
            'encoding': range(num_classes)
        })
        class_labels_df.to_csv('class_labels.csv', index=False)
        
        # Configure datasets for performance
        AUTOTUNE = tf.data.AUTOTUNE
        
        def prepare_dataset(ds, augment=False):
            # Cache dataset for better performance
            ds = ds.cache()
            
            # Apply augmentation only to training set
            if augment:
                ds = ds.map(
                    lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=AUTOTUNE
                )
            
            # Prefetch for better performance
            return ds.prefetch(buffer_size=AUTOTUNE)
        
        # Prepare all datasets
        train_ds = prepare_dataset(train_ds, augment=True)
        val_ds = prepare_dataset(val_ds)
        test_ds = prepare_dataset(test_ds)
        
        return train_ds, val_ds, test_ds, num_classes
        
    except Exception as e:
        print(f"Error in prepare_datasets: {str(e)}")
        raise

def plot_metrics(history_dict):
    try:
        plt.close('all')
        
        # Create directory for plots if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        
        # Plot training and validation accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(history_dict['accuracy'], label='Training Accuracy')
        plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('plots/accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(history_dict['loss'], label='Training Loss')
        plt.plot(history_dict['val_loss'], label='Validation Loss')
        plt.title('Model Loss over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('plots/loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'epoch': range(1, len(history_dict['accuracy']) + 1),
            'train_accuracy': history_dict['accuracy'],
            'val_accuracy': history_dict['val_accuracy'],
            'train_loss': history_dict['loss'],
            'val_loss': history_dict['val_loss']
        })
        metrics_df.to_csv('plots/training_validation_metrics.csv', index=False)
        
    except Exception as e:
        print(f"Error in plot_metrics: {str(e)}")
        raise

def create_confusion_matrix(model, test_ds):
    try:
        plt.close('all')
        
        # Load class labels
        class_labels = pd.read_csv('class_labels.csv')['class_name'].tolist()
        
        # Get predictions and true labels
        y_pred = []
        y_true = []
        
        # Process test dataset in smaller batches
        for images, labels in test_ds:
            batch_pred = model.predict(images, verbose=0)
            y_pred.extend(np.argmax(batch_pred, axis=1))
            y_true.extend(np.argmax(labels.numpy(), axis=1))
            
            # Clear memory
            del batch_pred
            gc.collect()
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot confusion matrix with class labels
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels
        )
        plt.title('Normalized Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save confusion matrix to CSV
        cm_df = pd.DataFrame(
            cm_normalized,
            columns=class_labels,
            index=class_labels
        )
        cm_df.to_csv('plots/confusion_matrix.csv')
        
        # Calculate and save classification metrics per class
        precision = np.diag(cm) / np.sum(cm, axis=0)
        recall = np.diag(cm) / np.sum(cm, axis=1)
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        class_metrics_df = pd.DataFrame({
            'class': class_labels,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        })
        class_metrics_df.to_csv('plots/class_metrics.csv', index=False)
        
        # Clear memory
        del cm, cm_normalized, y_pred, y_true
        gc.collect()
        
    except Exception as e:
        print(f"Error in create_confusion_matrix: {str(e)}")
        raise