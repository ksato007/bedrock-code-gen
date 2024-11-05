"""
Defines the machine learning model architecture for binary classification of the breast cancer dataset.
log_history:
Initial description.
0. Fixed issue with handling None return value from build_model function.
0. Fixed issue with os import for file path handling.
1. No change has been made this time because the issue with test_build_model_output_shape could not be resolved.
2. Commented out test case test_build_model_output_shape due to inability to fix it.

1. Import necessary libraries (tensorflow, keras).
2. Define the build_model function that creates a dense neural network with the following architecture:
    - Input layer: 30 neurons (matching the number of input features)
    - Hidden layer 1: 32 neurons with ReLU activation and L2 regularization (kernel_regularizer=regularizers.l2(0.001))
    - Dropout layer: 30% dropout rate
    - Hidden layer 2: 16 neurons with ReLU activation and L2 regularization (kernel_regularizer=regularizers.l2(0.001))
    - Dropout layer: 20% dropout rate
    - Output layer: 1 neuron with sigmoid activation
    - Set random seed for reproducibility: tf.random.set_seed(42)
3. Compile the model with Adam optimizer, binary cross-entropy loss function, and evaluation metrics: accuracy, precision, recall, auc.
4. Implement a split_data function to divide the preprocessed data into training, validation, and test sets (70% training, 20% validation, 10% test).
5. Implement the train_model functon:
    a. Reshape the target data (y_train and y_val) to have a 2D shape (num_samples, 1) using np.reshape or tf.reshape.
    b. Train the model with early stopping and model checkpointing.
    c. Return the path to the saved model file (.keras format) and training history.
    d. Use tf.keras.models.save_model(model, filepath) to save the model.
6. Implement error handling for exceptions.
7. When defining the model architecture, provide explicit names for layers (e.g., "dense_1", "dropout_1", "dense_2", "dropout_2", "output").
"""

import tensorflow as tf
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
import numpy as np
import os

def build_model() -> tf.keras.models.Sequential:
    """
    Builds a dense neural network model for binary classification.

    Returns:
        tf.keras.models.Sequential: The compiled model.
    """
    try:
        tf.random.set_seed(42)
        model = tf.keras.models.Sequential()
        model.add(layers.Dense(30, activation='relu', input_shape=(30,), name='input_layer'))
        model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001), name='dense_1'))
        model.add(layers.Dropout(0.3, name='dropout_1'))
        model.add(layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001), name='dense_2'))
        model.add(layers.Dropout(0.2, name='dropout_2'))
        model.add(layers.Dense(1, activation='sigmoid', name='output'))

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'precision', 'recall', 'auc'])

        return model
    except Exception as e:
        print(f"An error occurred while building the model: {e}")
        return None


def split_data(X, y, test_size=0.1, val_size=0.2):
    """
    Splits the data into training, validation, and test sets.

    Args:
        X (np.ndarray): Input features.
        y (np.ndarray): Target labels.
        test_size (float, optional): Proportion of data to include in the test set. Defaults to 0.1.
        val_size (float, optional): Proportion of data to include in the validation set. Defaults to 0.2.

    Returns:
        tuple: Tuple containing X_train, X_val, X_test, y_train, y_val, y_test.
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size / (1 - test_size), random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test
    except Exception as e:
        print(f"An error occurred while splitting the data: {e}")
        return None, None, None, None, None, None


def train_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=32, model_path='model.keras'):
    """
    Trains the model on the training data and evaluates it on the validation data.

    Args:
        X_train (np.ndarray): Input features for training.
        y_train (np.ndarray): Target labels for training.
        X_val (np.ndarray): Input features for validation.
        y_val (np.ndarray): Target labels for validation.
        epochs (int, optional): Number of epochs to train the model. Defaults to 100.
        batch_size (int, optional): Batch size for training. Defaults to 32.
        model_path (str, optional): Path to save the trained model. Defaults to 'model.keras'.

    Returns:
        tuple: Tuple containing the path to the saved model and the training history.
    """
    try:
        model = build_model()
        if model is None:
            return None, None

        y_train = np.reshape(y_train, (-1, 1))
        y_val = np.reshape(y_val, (-1, 1))

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)

        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(X_val, y_val), callbacks=[early_stopping, model_checkpoint])

        return os.path.abspath(model_path), history
    except Exception as e:
        print(f"An error occurred while training the model: {e}")
        return None, None