import csv
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Load the Fashion-MNIST dataset using TensorFlow Datasets
def load_data(data_dir="./tensorflow_datasets"):
    ds = tfds.load('fashion_mnist', data_dir=data_dir, shuffle_files=True, as_supervised=True)
    ds_train, ds_test = ds['train'], ds['test']
    return ds_train, ds_test

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    
    return image, label

# Normalize and preprocess the dataset, including the train-validation split
def preprocess_data(ds_train, val_split=0.1):
    ds_train = ds_train.map(preprocess).cache().shuffle(10000).batch(32)
    ds_train_size = int(60000 * (1 - val_split))
    
    ds_val = ds_train.skip(ds_train_size)
    ds_train = ds_train.take(ds_train_size)
    
    return ds_train, ds_val

# Function to plot loss vs. accuracy over epochs
def plot_accuracy(history, model_name):
    plt.figure(figsize=(10, 6))  # Create a new figure


    # Plot accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.title(f'Accuracy over Epochs for {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig(f'{model_name}_accuracy_plot.png')

    # Show the plots
    # plt.tight_layout()
    # plt.show()
    
    plt.close()  # Close the figure to free up memory
    
# Function to plot confusion matrix as heatmap
def plot_confusion_matrix_heatmap(model, ds_test, class_names, model_name):
    y_true = []
    y_pred = []

    # Gather true labels and predicted labels
    for images, labels in ds_test:
        predictions = model.predict(images)
        y_true.extend(labels.numpy())  # Actual labels
        y_pred.extend(np.argmax(predictions, axis=1))  # Predicted labels

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix Heatmap')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    
    # Save the confusion matrix heatmap
    plt.savefig(f'{model_name}_confusion_matrix_heatmap.png')
    
    # plt.show()
    
    plt.close()  # Close the figure to free up memory
    
# Function to save model accuracies to a CSV file
def save_model_accuracy_to_csv(model_name, history, test_acc, file_name='model_accuracies.csv'):
    train_acc = history.history['accuracy'][-1]  # Final training accuracy
    
    # Open (or create) CSV file and append results
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write header if file is new
        if file.tell() == 0:
            writer.writerow(["Model Name", "Training Accuracy", "Test Accuracy"])
        
        # Write model accuracy data
        writer.writerow([model_name, train_acc, test_acc])

    print(f"Accuracy data for {model_name} saved to {file_name}.")