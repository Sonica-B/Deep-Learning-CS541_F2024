import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess_data():
    # Load and reshape data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    y_tr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    y_te = np.load("age_regression_yte.npy")

    # Remove NaN values
    valid_train_indices = ~np.isnan(y_tr)
    X_tr = X_tr[valid_train_indices]
    y_tr = y_tr[valid_train_indices]

    valid_test_indices = ~np.isnan(y_te)
    X_te = X_te[valid_test_indices]
    y_te = y_te[valid_test_indices]

    # Normalize the features for numerical stability
    X_tr = (X_tr - np.mean(X_tr, axis=0)) / np.std(X_tr, axis=0)
    X_te = (X_te - np.mean(X_te, axis=0)) / np.std(X_te, axis=0)

    # Split the training data into training and validation sets (80-20 split)
    split_index = int(0.8 * len(X_tr))
    X_train, X_val = X_tr[:split_index], X_tr[split_index:]
    y_train, y_val = y_tr[:split_index], y_tr[split_index:]

    return X_train, y_train, X_val, y_val, X_te, y_te

def compute_cost(X, y, w, b):
    y_pred = X.dot(w) + b
    mse = np.mean((y_pred - y) ** 2) / 2
    return mse

def compute_gradients(X, y, w, b):
    n = len(y)
    y_pred = X.dot(w) + b
    dw = (1 / n) * X.T.dot(y_pred - y)
    db = (1 / n) * np.sum(y_pred - y)
    return dw, db

def train_sgd(X_train, y_train, X_val, y_val, learning_rate=0.001, batch_size=64, epochs=100, decay=0.99):
    # Initialize parameters
    w = np.random.randn(X_train.shape[1]) * 0.01
    b = 0.0
    best_val_cost = float('inf')
    best_w, best_b = w, b

    history = {'train_cost': [], 'val_cost': []}

    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(len(y_train))
        X_train, y_train = X_train[indices], y_train[indices]

        # Mini-batch training
        for i in range(0, len(y_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            # Compute gradients
            dw, db = compute_gradients(X_batch, y_batch, w, b)

            # Update parameters
            w -= learning_rate * dw
            b -= learning_rate * db

        # Compute and store cost for training and validation
        train_cost = compute_cost(X_train, y_train, w, b)
        val_cost = compute_cost(X_val, y_val, w, b)
        history['train_cost'].append(train_cost)
        history['val_cost'].append(val_cost)

        # Learning rate decay
        learning_rate *= decay

        # Update best parameters if validation cost improves
        if val_cost < best_val_cost:
            best_val_cost = val_cost
            best_w, best_b = w, b

        print(f'Epoch {epoch+1}/{epochs} - Train Cost: {train_cost:.4f}, Val Cost: {val_cost:.4f}')

    return best_w, best_b, history

def plot_costs(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_cost'], label='Training Cost')
    plt.plot(history['val_cost'], label='Validation Cost')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Cost')
    plt.title('Training and Validation Cost over Epochs')
    plt.legend()
    plt.show()

def train_age_regressor():
    # Load and preprocess data
    X_train, y_train, X_val, y_val, X_te, y_te = load_and_preprocess_data()

    # Hyperparameter tuning
    best_val_cost = float('inf')
    best_params = {}
    best_history = {}

    learning_rates = [0.0001, 0.00005, 0.00001]
    batch_sizes = [16, 32, 64]
    epochs = 200  # Increased number of epochs for better convergence
    decays = 0.99  # Adjusted decay factors

    for lr in learning_rates:
        for batch_size in batch_sizes:
            w, b, history = train_sgd(X_train, y_train, X_val, y_val, learning_rate=lr, batch_size=batch_size, epochs=epochs)

            val_cost = history['val_cost'][-1]
            if val_cost < best_val_cost:
                best_val_cost = val_cost
                best_params = {'learning_rate': lr, 'batch_size': batch_size}
                best_history = history

    print("Best Hyperparameters:", best_params)

    # Plotting the training and validation cost over epochs
    plot_costs(best_history)

    # Evaluate on test set
    best_w, best_b, _ = train_sgd(X_train, y_train, X_val, y_val, learning_rate=best_params['learning_rate'], batch_size=best_params['batch_size'], epochs=epochs)
    test_cost = compute_cost(X_te, y_te, best_w, best_b)
    print(f'Test MSE: {test_cost:.4f}')

# Run the function
train_age_regressor()
