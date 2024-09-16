import numpy as np
import matplotlib.pyplot as plt

def train_age_regressor ():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    y_tr = np.load("age_regression_ytr.npy")
    Xte = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    yte = np.load("age_regression_yte.npy")

    #Check for NaN values
    if np.isnan(X_tr).any() or np.isnan(y_tr).any() or np.isnan(Xte).any() or np.isnan(yte).any():
        raise ValueError("Data contains NaN values.")


    # #Data Standardization
    # def standardize(X, y):
    #     X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    #     y = (y - np.mean(y)) / np.std(y)
    #     return X, y


    # Split data
    split_index = int(0.8 * len(X_tr))
    X_train, X_val = X_tr[:split_index], X_tr[split_index:]
    y_train, y_val = y_tr[:split_index], y_tr[split_index:]

    # #Standardize the data
    # X_train, y_train = standardize(X_train, y_train)
    # X_val, y_val = standardize(X_val, y_val)
    # Xte, yte = standardize(Xte, yte)


    def cost_function(X,y,w,b):
        n = len(y)
        y_hat = X.dot(w) + b
        cost = (1/(2*n)) * np.sum(np.square(y_hat-y))
        return cost


    def gradient(X,y,w,b):
        n = len(y)
        y_hat = X.dot(w) + b
        gw = (1/n) * X.T.dot(y_hat-y)
        gb = (1/n) * np.sum(y_hat-y)
        return gw, gb


    def stochastic_gradient_descent(X_train, y_train, X_val, y_val, e, batch_size, epochs):

        w =np.zeros(X_train.shape[1])
        b = 0.0
        train_cost_history = []
        val_cost_history = []

        for epoch in range(epochs):
            # Shuffling data
            index = np.random.permutation(len(y_train))
            X_shuff, y_shuff = X_train[index], y_train[index]

            # Mini-batch training
            for i in range(0, len(y_train), batch_size):
                Xi = X_shuff[i:i + batch_size]
                Yi = y_shuff[i:i + batch_size]

                # Compute gradients
                gw, gb = gradient(Xi, Yi, w, b)

                # Update parameters
                w -= e * gw
                b -= e * gb

            # Compute cost
            train_cost = cost_function(X_train, y_train, w, b)
            val_cost = cost_function(X_val, y_val, w, b)

            train_cost_history.append(train_cost)
            val_cost_history.append(val_cost)

        return w, b, train_cost_history, val_cost_history


    def hyperparameter_tuning(X_train, y_train, X_val, y_val, e, mini_batches, epochs):

        opt_val_cost = float('inf')
        opt_w = None
        opt_b = None
        opt_hyperparams = None
        opt_train_cost_history = []
        opt_val_cost_history = []

        for lr in e:
            for batch_size in mini_batches:
                w, b, train_cost_history, val_cost_history = stochastic_gradient_descent(
                    X_train, y_train, X_val, y_val, lr, batch_size, epochs
                )

                val_cost = val_cost_history[-1]  # Use the last validation cost

                if val_cost < opt_val_cost:
                    opt_val_cost = val_cost
                    opt_w = w
                    opt_b = b
                    opt_hyperparams = (lr, batch_size, epochs)
                    opt_train_cost_history = train_cost_history
                    opt_val_cost_history = val_cost_history

        return opt_w, opt_b, opt_val_cost, opt_hyperparams, opt_train_cost_history, opt_val_cost_history


    # Define hyperparameters
    e = [0.001, 0.0001, 0.00001, 0.000001]
    mini_batches = [64, 128, 256]
    epochs = 300

    # Hyperparameter tuning
    opt_w, opt_b, opt_val_cost, opt_hyperparams, train_cost_history, val_cost_history = hyperparameter_tuning(
        X_train, y_train, X_val, y_val, e, mini_batches, epochs
    )

    print(f'Best Hyper Parameters Learning Rate = {opt_hyperparams[0]}, Mini-batch Size = {opt_hyperparams[1]}, Epochs = {opt_hyperparams[2]}')
    print(f'Best Validation Cost: {opt_val_cost}')

    # Test set
    test_cost = cost_function(Xte, yte, opt_w, opt_b)
    print(f'Test Set Cost: {test_cost}')

    # Plotting training and validation cost history
    plt.plot(train_cost_history, label='Training Cost')
    plt.plot(val_cost_history, label='Validation Cost')
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.title('Training and Validation Cost Over Epochs')
    plt.legend()
    plt.show()

    print('Final weights:', opt_w)
    print('Final bias:', opt_b)

    print(f'Cost values of Training Dataset of last 10 iterations: {train_cost_history[-10:]}')





train_age_regressor()
