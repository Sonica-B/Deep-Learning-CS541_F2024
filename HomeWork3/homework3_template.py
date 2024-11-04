import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

NUM_HIDDEN_LAYERS = 3
NUM_INPUT = 784
NUM_HIDDEN = NUM_HIDDEN_LAYERS * [ 72 ]
NUM_OUTPUT = 10

lr = 0.01
epochs = 100
batch_size = 128

def unpack (weights):
    # Unpack arguments
    Ws = []

    # Weight matrices
    start = 0
    end = NUM_INPUT*NUM_HIDDEN[0]
    W = weights[start:end]
    Ws.append(W)

    # Unpack the weight matrices as vectors
    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN[i]*NUM_HIDDEN[i+1]
        W = weights[start:end]
        Ws.append(W)

    start = end
    end = end + NUM_HIDDEN[-1]*NUM_OUTPUT
    W = weights[start:end]
    Ws.append(W)

    # Reshape the weight "vectors" into proper matrices
    Ws[0] = Ws[0].reshape(NUM_HIDDEN[0], NUM_INPUT)
    for i in range(1, NUM_HIDDEN_LAYERS):
        # Convert from vectors into matrices
        Ws[i] = Ws[i].reshape(NUM_HIDDEN[i], NUM_HIDDEN[i-1])
    Ws[-1] = Ws[-1].reshape(NUM_OUTPUT, NUM_HIDDEN[-1])

    # Bias terms
    bs = []
    start = end
    end = end + NUM_HIDDEN[0]
    b = weights[start:end]
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN[i+1]
        b = weights[start:end]
        bs.append(b)

    start = end
    end = end + NUM_OUTPUT
    b = weights[start:end]
    bs.append(b)

    return Ws, bs

def one_hot_encode(Y, num_classes):

    one_hot_Y = np.zeros((Y.size, num_classes))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y

def ReLU (Z): return np.maximum(0,Z)
def forward_pass(X, Ws, bs):
    activations = [X]
    pre_activations = []

    # Hidden layers forward pass
    for i in range(len(Ws) - 1):
        Z = np.matmul(activations[-1], Ws[i].T) + bs[i]
        pre_activations.append(Z)
        H = ReLU(Z)
        activations.append(H)

    # Output layer forward pass (softmax)
    Z_out = np.matmul(activations[-1], Ws[-1].T) + bs[-1]
    Y_hat = softmax(Z_out)

    return Y_hat, activations, pre_activations

def softmax(Z):
    exp_z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def backward_pass(X, Y, activations, pre_activations, Y_hat, Ws):
    n = X.shape[0]  # Number of samples in the batch
    grads_W = []
    grads_b = []

    # Convert Y to one-hot encoding
    Y = one_hot_encode(Y, num_classes=10)

    # Gradient for the output layer
    dZ = Y_hat - Y  # (batch_size, num_classes)
    dW = np.matmul(dZ.T,activations[-2]) / n  # (hidden_layer_size, num_classes)
    db = np.sum(dZ, axis=0) / n
    grads_W.append(dW)
    grads_b.append(db)

    # Backpropagate through hidden layers
    for i in range(len(Ws) - 2, -1, -1):
        dH = np.matmul(dZ, Ws[i + 1])  # (batch_size, hidden_layer_size)
        dH *= (pre_activations[i] > 0)  # Apply ReLU derivative
        dZ = dH

        # Calculate gradients for current layer
        dW = np.matmul(dZ.T, activations[i]) / n  # (input_size, hidden_layer_size)
        db = np.sum(dZ, axis=0) / n

        # Insert gradients at the beginning (to maintain layer order)
        grads_W.insert(0, dW)
        grads_b.insert(0, db)

    return grads_W, grads_b


def fCE(X, Y, weights, epsilon=1e-12):
    Ws, bs = unpack(weights)
    Y_hat, _, _ = forward_pass(X, Ws, bs)
    Y = one_hot_encode(Y, num_classes=10)

    # Clipping Y_hat to avoid log(0)
    Y_hat = np.clip(Y_hat, epsilon, 1. - epsilon)

    ce = -np.sum(Y * np.log(Y_hat)) / X.shape[0]
    return ce



def gradCE (X, Y, weights):
    Ws, bs = unpack(weights)
    # ...
    # Forward pass to get activations and pre-activations
    Y_hat, activations, pre_activations = forward_pass(X, Ws, bs)

    # Backward pass to calculate gradients
    grads_W, grads_b = backward_pass(X, Y, activations, pre_activations, Y_hat, Ws)

    allGradientsAsVector = np.hstack([gW.flatten() for gW in grads_W] + [gb.flatten() for gb in grads_b])
    return allGradientsAsVector


def initWeightsAndBiases ():
    Ws = []
    bs = []

    # Strategy:
    # Sample each weight using a variant of the Kaiming He Uniform technique.
    # Initialize biases to small positive number (0.01).

    np.random.seed(0)
    W = 2*(np.random.random(size=(NUM_HIDDEN[0], NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_HIDDEN[0])
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        W = 2*(np.random.random(size=(NUM_HIDDEN[i+1], NUM_HIDDEN[i]))/NUM_HIDDEN[i]**0.5) - 1./NUM_HIDDEN[i]**0.5
        Ws.append(W)
        b = 0.01 * np.ones(NUM_HIDDEN[i+1])
        bs.append(b)

    W = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN[-1]))/NUM_HIDDEN[-1]**0.5) - 1./NUM_HIDDEN[-1]**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_OUTPUT)
    bs.append(b)
    return Ws, bs
# Creates an image representing the first layer of weights (W0).
def show_W0 (W):
    Ws,bs = unpack(W)
    W = Ws[0]
    n = int(NUM_HIDDEN[0] ** 0.5)
    plt.imshow(np.vstack([
        np.hstack([ np.pad(np.reshape(W[idx1*n + idx2,:], [ 28, 28 ]), 2, mode='constant') for idx2 in range(n) ]) for idx1 in range(n)
    ]), cmap='gray'), plt.show()


def train(trainX, trainY, weights, testX, testY):
    for epoch in range(epochs):
        # Shuffle data
        perm = np.random.permutation(trainX.shape[0])
        trainX_shuffled = trainX[perm]
        trainY_shuffled = trainY[perm]

        # Split the data into minibatches
        num_batches = trainX.shape[0] // batch_size
        X_batches = np.array_split(trainX_shuffled, num_batches)
        Y_batches = np.array_split(trainY_shuffled, num_batches)

        for X_batch, Y_batch  in zip(X_batches, Y_batches):
            # X_batch = trainX_shuffled[batch: batch + batch_size]
            # Y_batch = trainY_shuffled[batch: batch + batch_size]

            gradVec = gradCE(X_batch, Y_batch, weights)
            weights -= lr * gradVec

        # Validate
        test_Y_hat, activations, pre_activations = forward_pass(testX, *unpack(weights))
        loss = fCE(testX, testY, weights)
        accuracy = np.mean(np.argmax(test_Y_hat, axis=1) == np.argmax(one_hot_encode(testY, num_classes=10), axis=1))
        print(f"Epoch {epoch + 1}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")

    return weights








if __name__ == "__main__":
    # Load training data.
    trainX = np.load('image/fashion_mnist_train_images.npy')
    trainY = np.load('image/fashion_mnist_train_labels.npy')
    testX = np.load('image/fashion_mnist_test_images.npy')
    testY = np.load('image/fashion_mnist_test_labels.npy')
    # Recommendation: divide the pixels by 255 (so that their range is [0-1]), and then subtract
    trainX = trainX / 255.0 - 0.5
    testX = testX / 255.0 - 0.5

    # 0.5 (so that the range is [-0.5,+0.5]).


    Ws, bs = initWeightsAndBiases()

    # "Pack" all the weight matrices and bias vectors into long one parameter "vector".
    weights = np.hstack([ W.flatten() for W in Ws ] + [ b.flatten() for b in bs ])
    # On just the first 5 training examlpes, do numeric gradient check.
    # Use just the first return value ([0]) of fCE, which is the cross-entropy.
    # The lambda expression is used so that, from the perspective of
    # check_grad and approx_fprime, the only parameter to fCE is the weights
    # themselves (not the training data).

    print(scipy.optimize.check_grad(lambda weights_: fCE(np.atleast_2d(trainX[:5]), np.atleast_2d(trainY[:5]), weights_), \
                                    lambda weights_: gradCE(np.atleast_2d(trainX[:5]), np.atleast_2d(trainY[:5]), weights_), \
                                    weights))
    print(scipy.optimize.approx_fprime(weights, lambda weights_: fCE(np.atleast_2d(trainX[:5]), np.atleast_2d(trainY[0:5]), weights_), 1e-6))
    weights = train(trainX, trainY, weights, testX, testY)

    # Hyperparameter tuning
    # Example values for tuning
    # batch_sizes = [64, 128, 256]
    # learning_rates = [0.0001, 0.00001,0.000001]

    # best_params, best_accuracy = hyperparameter_tuning(trainX, trainY, testX, testY, weights, EPOCHS, batch_sizes,
    #                                                    learning_rates)

    # print(f"Best parameters: {best_params}, Best accuracy: {best_accuracy:.4f}")
    show_W0(weights)
