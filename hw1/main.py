import tensorflow as tf
import matplotlib.pyplot as plt
from util import load_data, preprocess_data, preprocess, save_model_accuracy_to_csv
from model import build_model
from util import plot_accuracy, plot_confusion_matrix_heatmap

# Load and preprocess the Fashion-MNIST dataset
ds_train, ds_test = load_data()
ds_train, ds_val = preprocess_data(ds_train, val_split=0.1)
ds_test = ds_test.map(preprocess).batch(32)

# Number of training epochs
epochs = 50

# Callback for early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Architecture 1: Single hidden layer, 64 nodes
# Architecture 2: Two hidden layers, 64 and 32 nodes
# Hyperparameter sets: learning rates and regularization settings

# Architecture 1 with hyperparameter set 1 (learning rate 0.001, without regularization)
model_1a = build_model(layers_units=[64], activation='relu', optimizer='adam', learning_rate=0.001, regularizer=None)
history_1a = model_1a.fit(ds_train, epochs=epochs, validation_data=ds_val, callbacks=[early_stopping])

# Architecture 2 with hyperparameter set 1 (learning rate 0.001, without regularization)
model_1b = build_model(layers_units=[64, 32], activation='relu', optimizer='adam', learning_rate=0.001, regularizer=None)
history_1b = model_1b.fit(ds_train, epochs=epochs, validation_data=ds_val, callbacks=[early_stopping])

# Architecture 1 with hyperparameter set 2 (learning rate 0.001, with L2 regularization)
model_2a = build_model(layers_units=[64], activation='relu', optimizer='adam', learning_rate=0.001, regularizer='l2')
history_2a = model_2a.fit(ds_train, epochs=epochs, validation_data=ds_val, callbacks=[early_stopping])

# Architecture 2 with hyperparameter set 2 (learning rate 0.001, with L2 regularization)
model_2b = build_model(layers_units=[64, 32], activation='relu', optimizer='adam', learning_rate=0.001, regularizer='l2')
history_2b = model_2b.fit(ds_train, epochs=epochs, validation_data=ds_val, callbacks=[early_stopping])

# Architecture 1 with hyperparameter set 3 (learning rate 0.05, without regularization)
model_3a = build_model(layers_units=[64], activation='relu', optimizer='adam', learning_rate=0.05, regularizer=None)
history_3a = model_3a.fit(ds_train, epochs=epochs, validation_data=ds_val, callbacks=[early_stopping])

# Architecture 2 with hyperparameter set 3 (learning rate 0.05, without regularization)
model_3b = build_model(layers_units=[64, 32], activation='relu', optimizer='adam', learning_rate=0.05, regularizer=None)
history_3b = model_3b.fit(ds_train, epochs=epochs, validation_data=ds_val, callbacks=[early_stopping])

# Architecture 1 with hyperparameter set 4 (learning rate 0.05, with L2 regularization)
model_4a = build_model(layers_units=[64], activation='relu', optimizer='adam', learning_rate=0.05, regularizer='l2')
history_4a = model_4a.fit(ds_train, epochs=epochs, validation_data=ds_val, callbacks=[early_stopping])

# Architecture 2 with hyperparameter set 4 (learning rate 0.05, with L2 regularization)
model_4b = build_model(layers_units=[64, 32], activation='relu', optimizer='adam', learning_rate=0.05, regularizer='l2')
history_4b = model_4b.fit(ds_train, epochs=epochs, validation_data=ds_val, callbacks=[early_stopping])

# Evaluate all models on the test data
for i, model in enumerate([model_1a, model_1b, model_2a, model_2b, model_3a, model_3b, model_4a, model_4b], start=1):
    test_loss, test_acc = model.evaluate(ds_test)
    print(f"Model {i} Test Accuracy: {test_acc}")
    

# Class labels for Fashion-MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Evaluate all models on the test data and plot confusion matrix
for i, (model, history) in enumerate([(model_1a, history_1a), (model_1b, history_1b), 
                                      (model_2a, history_2a), (model_2b, history_2b), 
                                      (model_3a, history_3a), (model_3b, history_3b), 
                                      (model_4a, history_4a), (model_4b, history_4b)], start=1):
    test_loss, test_acc = model.evaluate(ds_test)
    print(f"Model {i} Test Accuracy: {test_acc}")
    
    model_name = f"Model {i}"
    
    # Plot loss vs. accuracy for each model
    plot_accuracy(history, model_name=model_name)
    
    # Plot confusion matrix heatmap for each model
    plot_confusion_matrix_heatmap(model, ds_test, class_names, model_name=model_name)
    
    # Save model accuracy to CSV
    save_model_accuracy_to_csv(model_name, history, test_acc)