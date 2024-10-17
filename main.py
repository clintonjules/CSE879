import tensorflow as tf
from model import build_model_1, build_model_2
from util import load_data, preprocess, get_callbacks, plot_training_history, plot_confusion_matrix, save_class_encodings
import numpy as np
import csv

# Save CIFAR-100 class encodings to a CSV file
save_class_encodings()

# Load and preprocess data
train_ds, val_ds, test_ds = load_data()

# Hyperparameter sets for each run
hyperparams_1 = {'learning_rate': 0.001, 'batch_size': 64}
hyperparams_2 = {'learning_rate': 0.0005, 'batch_size': 128}

# List of hyperparameter sets
hyperparam_sets = [hyperparams_1, hyperparams_2]

# Store test accuracy results
results = []

# Train and evaluate models for each architecture and hyperparameter set
for idx, hyperparams in enumerate(hyperparam_sets):
    print(f"\nTraining Model 1 with Hyperparameters Set {idx + 1}...")
    
    # Adjust batch size
    train_ds_adjusted = train_ds.batch(hyperparams['batch_size'])
    val_ds_adjusted = val_ds.batch(hyperparams['batch_size'])

    # Define Model 1
    model_1 = build_model_1()

    # Compile Model 1
    model_1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparams['learning_rate']),
                    loss='categorical_crossentropy', metrics=['accuracy'])

    # Train Model 1
    history_1 = model_1.fit(train_ds_adjusted, validation_data=val_ds_adjusted, epochs=50,
                            callbacks=get_callbacks())

    # Plot and save separate training history graphs for Model 1
    plot_training_history(history_1, 
                          f"model_1_hyperparams_{idx + 1}_accuracy.png", 
                          f"model_1_hyperparams_{idx + 1}_loss.png")

    # Evaluate Model 1 on the test set
    print(f"Evaluating Model 1 (Hyperparameters Set {idx + 1})...")
    test_loss_1, test_acc_1 = model_1.evaluate(test_ds.batch(hyperparams['batch_size']))
    print(f"Model 1 - Test Accuracy with Hyperparameters Set {idx + 1}: {test_acc_1}")

    # Save the results for Model 1
    results.append(["Model 1", idx + 1, hyperparams['learning_rate'], hyperparams['batch_size'], test_acc_1])

    # Generate confusion matrix for Model 1
    y_true = np.concatenate([y for x, y in test_ds.batch(hyperparams['batch_size'])], axis=0)
    y_pred = np.argmax(model_1.predict(test_ds.batch(hyperparams['batch_size'])), axis=1)
    plot_confusion_matrix(y_true, y_pred, f"model_1_hyperparams_{idx + 1}_confusion_matrix.png")

    print(f"\nTraining Model 2 with Hyperparameters Set {idx + 1}...")
    
    # Define Model 2
    model_2 = build_model_2()

    # Compile Model 2
    model_2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparams['learning_rate']),
                    loss='categorical_crossentropy', metrics=['accuracy'])

    # Train Model 2
    history_2 = model_2.fit(train_ds_adjusted, validation_data=val_ds_adjusted, epochs=50,
                            callbacks=get_callbacks())

    # Plot and save separate training history graphs for Model 2
    plot_training_history(history_2, 
                          f"model_2_hyperparams_{idx + 1}_accuracy.png", 
                          f"model_2_hyperparams_{idx + 1}_loss.png")

    # Evaluate Model 2 on the test set
    print(f"Evaluating Model 2 (Hyperparameters Set {idx + 1})...")
    test_loss_2, test_acc_2 = model_2.evaluate(test_ds.batch(hyperparams['batch_size']))
    print(f"Model 2 - Test Accuracy with Hyperparameters Set {idx + 1}: {test_acc_2}")

    # Save the results for Model 2
    results.append(["Model 2", idx + 1, hyperparams['learning_rate'], hyperparams['batch_size'], test_acc_2])

    # Generate confusion matrix for Model 2
    y_true = np.concatenate([y for x, y in test_ds.batch(hyperparams['batch_size'])], axis=0)
    y_pred = np.argmax(model_2.predict(test_ds.batch(hyperparams['batch_size'])), axis=1)
    plot_confusion_matrix(y_true, y_pred, f"model_2_hyperparams_{idx + 1}_confusion_matrix.png")

# Write results to CSV file
with open('model_test_accuracies.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(["Model", "Hyperparameter Set", "Learning Rate", "Batch Size", "Test Accuracy"])
    # Write the rows
    writer.writerows(results)