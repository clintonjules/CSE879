import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import csv

def load_data():
    """Loads and preprocesses CIFAR-100 dataset."""
    dataset = tfds.load('cifar100', split=['train[:80%]', 'train[80%:]', 'test'], as_supervised=True)

    # Split into training, validation, and test sets
    train_ds = dataset[0]
    val_ds = dataset[1]
    test_ds = dataset[2]

    # Preprocess data (normalize and one-hot encode labels)
    train_ds = train_ds.map(preprocess).shuffle(10000)
    val_ds = val_ds.map(preprocess)
    test_ds = test_ds.map(preprocess)

    return train_ds, val_ds, test_ds

def preprocess(image, label):
    """Preprocess function: Normalizes the image and one-hot encodes the label."""
    image = tf.cast(image, tf.float32) / 255.0  # Normalize the image
    label = tf.one_hot(label, 100)  # Convert labels to one-hot encoding (100 classes)
    return image, label

def get_callbacks():
    """Returns callbacks for early stopping and other functionality."""
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    return [early_stopping]

def plot_training_history(history, acc_filename, loss_filename):
    """Plots and saves separate graphs for training/validation accuracy and training/validation loss."""
    
    # Extract training history
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Plot and save the accuracy graph
    plt.figure()
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(acc_filename)
    plt.close()

    # Plot and save the loss graph
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(loss_filename)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, filename):
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(np.argmax(y_true, axis=1), y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='g')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    plt.savefig(filename)
    plt.close()

def save_class_encodings():
    """Saves the CIFAR-100 class names and their number encodings to a CSV file."""
    # Load the CIFAR-100 dataset information
    cifar100_info = tfds.builder('cifar100').info

    # Retrieve the class labels (ClassLabel feature)
    class_labels = cifar100_info.features['label'].names

    # Write the class labels to a CSV file
    with open('cifar100_class_encodings.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Class Name", "Class Encoding"])

        for i, class_name in enumerate(class_labels):
            writer.writerow([class_name, i])

    print("Class encodings saved to 'cifar100_class_encodings.csv'.")