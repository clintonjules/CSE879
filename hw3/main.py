import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from util import load_data, create_callbacks
from model import build_lstm_model, build_gru_model
from tensorflow.keras.callbacks import EarlyStopping

# Load data
train_data, val_data, test_data = load_data()

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Training parameters
hyperparams = [
    {"model": "LSTM", "embedding_dim": 128, "units": 64, "regularizer_rate": 0.01},
    {"model": "GRU", "embedding_dim": 128, "units": 64, "regularizer_rate": 0.01}
]

# Training and evaluation
model_results = []

for params in hyperparams:
    print(f"Training {params['model']} model...")
    
    if params['model'] == 'LSTM':
        model = build_lstm_model(
            vocab_size=10000,
            embedding_dim=params['embedding_dim'],
            lstm_units=params['units'],
            regularizer_rate=params['regularizer_rate']
        )
    elif params['model'] == 'GRU':
        model = build_gru_model(
            vocab_size=10000,
            embedding_dim=params['embedding_dim'],
            gru_units=params['units'],
            regularizer_rate=params['regularizer_rate']
        )
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=30,
        callbacks=[early_stopping]
    )
    
    # Evaluate on test data
    test_loss, test_acc = model.evaluate(test_data)
    print(f"{params['model']} Test Accuracy: {test_acc:.4f}")

    # Record results
    model_results.append({
        "model": params['model'],
        "embedding_dim": params['embedding_dim'],
        "units": params['units'],
        "regularizer_rate": params['regularizer_rate'],
        "epochs_run": len(history.history['loss']),
        "test_accuracy": test_acc,
        "test_loss": test_loss
    })

    # Save epoch-wise metrics to CSV
    epoch_data = {
        'epoch': range(1, len(history.history['loss']) + 1),
        'train_loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'train_accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy']
    }
    epoch_df = pd.DataFrame(epoch_data)
    epoch_csv_filename = f"{params['model']}_epoch_metrics.csv"
    epoch_df.to_csv(epoch_csv_filename, index=False)
    print(f"Epoch metrics saved to '{epoch_csv_filename}'")

    # Generate predictions and confusion matrix
    y_true = np.concatenate([y.numpy() for _, y in test_data], axis=0)
    y_pred = (model.predict(test_data) > 0.5).astype("int32").flatten()

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    
    # Plot and save the confusion matrix
    plt.figure()
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"{params['model']} Confusion Matrix")
    plt.savefig(f"{params['model']}_confusion_matrix.png")
    plt.close()

    # Plot and save accuracy over epochs
    plt.figure()
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f"{params['model']} Accuracy Over Epochs")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"{params['model']}_accuracy.png")
    plt.close()

    # Plot and save loss over epochs
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f"{params['model']} Loss Over Epochs")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{params['model']}_loss.png")
    plt.close()

# Save final model results to a CSV file
results_df = pd.DataFrame(model_results)
results_df.to_csv('model_results.csv', index=False)
print("Model results saved to 'model_results.csv'")