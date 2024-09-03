import numpy as np
import matplotlib.pyplot as plt

# Load data
X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48 * 48))
ytr = np.load("age_regression_ytr.npy")
X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48 * 48))
yte = np.load("age_regression_yte.npy")

# #Check for NaN values
# if np.isnan(X_tr).any() or np.isnan(ytr).any() or np.isnan(X_te).any() or np.isnan(yte).any():
#     raise ValueError("Data contains NaN values.")


# Data Standardization
def standardize(X, y):
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    y = (y - np.mean(y)) / np.std(y)
    return X, y


# Split data
split_index = int(0.8 * len(X_tr))
X_train, X_val = X_tr[:split_index], X_tr[split_index:]
y_train, y_val = ytr[:split_index], ytr[split_index:]

# Standardize the data
X_train, y_train = standardize(X_train, y_train)
X_val, y_val = standardize(X_val, y_val)
X_te, yte = standardize(X_te, yte)



def cost_function(X, y, w, b):
    n = len(y)
    y_hat = X.dot(w) + b
    cost = (1 / (2 * n)) * np.sum(np.square(y_hat - y))
    return cost


def gradient(X, y, w, b):
    n = len(y)
    y_hat = X.dot(w) + b
    gw = (1 / n) * X.T.dot(y_hat - y)
    gb = (1 / n) * np.sum(y_hat - y)
    return gw, gb


def stochastic_gradient_descent(X_train, y_train, X_val, y_val, learning_rate, batch_size, epochs):

    w =np.zeros(X_train.shape[1])
    b = 0.0
    train_cost_history = []
    val_cost_history = []

    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(len(y_train))
        X_shuffled, y_shuffled = X_train[indices], y_train[indices]

        # Mini-batch training
        for i in range(0, len(y_train), batch_size):
            Xi = X_shuffled[i:i + batch_size]
            Yi = y_shuffled[i:i + batch_size]

            # Compute gradients
            gw, gb = gradient(Xi, Yi, w, b)

            # Update parameters
            w -= learning_rate * gw
            b -= learning_rate * gb

        # Compute costs for both training and validation
        train_cost = cost_function(X_train, y_train, w, b)
        val_cost = cost_function(X_val, y_val, w, b)

        train_cost_history.append(train_cost)
        val_cost_history.append(val_cost)

    return w, b, train_cost_history, val_cost_history


def hyperparameter_tuning(X_train, y_train, X_val, y_val, learning_rates, mini_batch_sizes, epochs):

    best_val_cost = float('inf')
    best_w = None
    best_b = None
    best_hyperparams = None
    best_train_cost_history = []
    best_val_cost_history = []

    for lr in learning_rates:
        for batch_size in mini_batch_sizes:
            w, b, train_cost_history, val_cost_history = stochastic_gradient_descent(
                X_train, y_train, X_val, y_val, lr, batch_size, epochs
            )

            val_cost = val_cost_history[-1]  # Use the last validation cost

            if val_cost < best_val_cost:
                best_val_cost = val_cost
                best_w = w
                best_b = b
                best_hyperparams = (lr, batch_size, epochs)
                best_train_cost_history = train_cost_history
                best_val_cost_history = val_cost_history

    return best_w, best_b, best_val_cost, best_hyperparams, best_train_cost_history, best_val_cost_history


# Define hyperparameters
learning_rates = [0.001, 0.0001, 0.00001]
mini_batch_sizes = [16, 32, 64, 128]
epochs = 200

# Hyperparameter tuning
best_w, best_b, best_val_cost, best_hyperparams, train_cost_history, val_cost_history = hyperparameter_tuning(
    X_train, y_train, X_val, y_val, learning_rates, mini_batch_sizes, epochs
)

print(
    f'Best Hyperparameters: Learning Rate = {best_hyperparams[0]}, Mini-batch Size = {best_hyperparams[1]}, Epochs = {best_hyperparams[2]}')
print(f'Best Validation Cost: {best_val_cost}')

# Evaluate on the test set
test_cost = cost_function(X_te, yte, best_w, best_b)
print(f'Test Set Cost: {test_cost}')

# Plotting training and validation cost history
plt.plot(train_cost_history, label='Training Cost')
plt.plot(val_cost_history, label='Validation Cost')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Training and Validation Cost Over Epochs')
plt.legend()
plt.show()

print('Final weights:', best_w)
print('Final bias:', best_b)
