import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data
X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48 * 48))
ytr = np.load("age_regression_ytr.npy")
X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48 * 48))
yte = np.load("age_regression_yte.npy")

# Standardize the data
X_train, X_val, y_train, y_val = train_test_split(X_tr, ytr, test_size=0.2, random_state=42)
X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
y_train = (y_train - np.mean(y_train)) / np.std(y_train)
X_val = (X_val - np.mean(X_val, axis=0)) / np.std(X_val, axis=0)
y_val = (y_val - np.mean(y_val)) / np.std(y_val)

# Initialize parameters
w = np.random.randn(X_train.shape[1]) * 0.01  # Smaller initial weights
b = 0.0  # Initialize bias as a scalar
learning_rate = 0.001  # Reduced learning rate
epochs = 100  # Number of iterations


def cost_function(X, y, w, b):
    n = len(y)
    y_hat = X.dot(w) + b
    cost = (1 / (2 * n)) * np.sum(np.square(y_hat - y))
    return cost


def gradient(X, y, w, b):
    n = len(y)
    y_hat = X.dot(w) + b  # y_hat shape (n_samples,)
    gw = (1 / n) * X.T.dot(y_hat - y)
    gb = (1 / n) * np.sum(y_hat - y)

    # Clip the gradients to prevent overflow
    gw = np.clip(gw, -1e3, 1e3)
    gb = np.clip(gb, -1e3, 1e3)

    return gw, gb


def stochastic_gradient_descent(X_train, y_train, X_val, y_val, learning_rates, mini_batch_sizes, epochs):
    best_val_cost = float('inf')
    best_w = None
    best_b = None
    best_hyperparams = None
    overall_cost_history = []

    for lr in learning_rates:
        for batch_size in mini_batch_sizes:
            w = np.random.randn(X_train.shape[1]) * 0.01
            b = 0.0
            cost_history = []

            for epoch in range(epochs):
                indices = np.random.permutation(len(y_train))
                X_shuffled = X_train[indices]
                y_shuffled = y_train[indices]

                for i in range(0, len(y_train), batch_size):
                    Xi = X_shuffled[i:i + batch_size]
                    Yi = y_shuffled[i:i + batch_size]

                    gw, gb = gradient(Xi, Yi, w, b)
                    w -= lr * gw
                    b -= lr * gb

                val_cost = cost_function(X_val, y_val, w, b)
                cost_history.append(val_cost)

                if val_cost < best_val_cost:
                    best_val_cost = val_cost
                    best_w = w
                    best_b = b
                    best_hyperparams = (lr, batch_size, epoch)

            overall_cost_history.extend(cost_history)

    return best_w, best_b, best_val_cost, best_hyperparams, overall_cost_history


# Define hyperparameters
learning_rates = [0.001, 0.0001]
mini_batch_sizes = [32, 64]
epochs = 100

best_w, best_b, best_val_cost, best_hyperparams, cost_history = stochastic_gradient_descent(
    X_train, y_train, X_val, y_val, learning_rates, mini_batch_sizes, epochs
)

print(
    f'Best Hyperparameters: Learning Rate = {best_hyperparams[0]}, Mini-batch Size = {best_hyperparams[1]}, Epochs = {best_hyperparams[2]}')
print(f'Best Validation Cost: {best_val_cost}')

# Evaluate on the test set
X_te = (X_te - np.mean(X_te, axis=0)) / np.std(X_te, axis=0)
yte = (yte - np.mean(yte)) / np.std(yte)
test_cost = cost_function(X_te, yte, best_w, best_b)
print(f'Test Set Cost: {test_cost}')

# Plotting cost history
plt.plot(cost_history)
plt.xlabel('Iterations')
plt.ylabel('Validation Cost')
plt.title('Validation Cost Over Time')
plt.show()

print('Final weights:', best_w)
print('Final bias:', best_b)
