
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

# Hyperparameters
NUM_HIDDEN_LAYERS = 1
NUM_INPUT = 784  # 28x28 pixels flattened
NUM_HIDDEN = NUM_HIDDEN_LAYERS * [72]  # 72 neurons in each hidden layer
NUM_OUTPUT = 10  # 10 classes (Fashion MNIST)

# Unpacking weights
def unpack(weights):
    Ws = []
    # Weight matrices
    start = 0
    end = NUM_INPUT * NUM_HIDDEN[0]
    W = weights[start:end]
    Ws.append(W)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN[i] * NUM_HIDDEN[i + 1]
        W = weights[start:end]
        Ws.append(W)

    start = end
    end = end + NUM_HIDDEN[-1] * NUM_OUTPUT
    W = weights[start:end]
    Ws.append(W)

    # Reshape the weight vectors into matrices
    Ws[0] = Ws[0].reshape(NUM_HIDDEN[0], NUM_INPUT)
    for i in range(1, NUM_HIDDEN_LAYERS):
        Ws[i] = Ws[i].reshape(NUM_HIDDEN[i], NUM_HIDDEN[i - 1])
    Ws[-1] = Ws[-1].reshape(NUM_OUTPUT, NUM_HIDDEN[-1])

    # Bias terms
    bs = []
    start = end
    end = end + NUM_HIDDEN[0]
    b = weights[start:end]
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN[i + 1]
        b = weights[start:end]
        bs.append(b)

    start = end
    end = end + NUM_OUTPUT
    b = weights[start:end]
    bs.append(b)

    return Ws, bs

#Cross-entropy loss
def fCE(X, Y, weights):
    Ws, bs = unpack(weights)
    a = X
    for i in range(len(Ws) - 1):
        z = np.dot(a, Ws[i].T) + bs[i]
        a = np.maximum(0, z)  #ReLU activation
    z_out = np.dot(a, Ws[-1].T) + bs[-1]
    exp_z = np.exp(z_out - np.max(z_out, axis=1, keepdims=True))
    y_hat = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    ce_loss = -np.sum(Y * np.log(y_hat)) / X.shape[0]
    return ce_loss

#Gradient of Cross-entropy loss
def gradCE(X, Y, weights):
    Ws, bs = unpack(weights)
    a = X
    activations = [a]
    z_list = []

    #Forward pass
    for i in range(len(Ws) - 1):
        z = np.dot(a, Ws[i].T) + bs[i]
        z_list.append(z)
        a = np.maximum(0, z)  #ReLU activation
        activations.append(a)

    z_out = np.dot(a, Ws[-1].T) + bs[-1]
    exp_z = np.exp(z_out - np.max(z_out, axis=1, keepdims=True))
    y_hat = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # Backward pass
    dz = (y_hat - Y) / X.shape[0]
    dWs = []
    dbs = []

    #Output layer gradients
    dW = np.dot(dz.T, activations[-1])
    dWs.append(dW)
    dbs.append(np.sum(dz, axis=0))

    #Backprop
    for i in range(NUM_HIDDEN_LAYERS - 1, -1, -1):
        dz = np.dot(dz, Ws[i + 1]) * (z_list[i] > 0)
        dW = np.dot(dz.T, activations[i])
        db = np.sum(dz, axis=0)
        dWs.append(dW)
        dbs.append(db)


    dWs.reverse()
    dbs.reverse()

    all_gradients = np.hstack([dW.flatten() for dW in dWs] + [db.flatten() for db in dbs])
    return all_gradients

#Forward pass
def forward(X, weights):
    Ws, bs = unpack(weights)
    a = X
    for i in range(len(Ws) - 1):
        z = np.dot(a, Ws[i].T) + bs[i]
        a = np.maximum(0, z)  # ReLU activation
    z_out = np.dot(a, Ws[-1].T) + bs[-1]
    exp_z = np.exp(z_out - np.max(z_out, axis=1, keepdims=True))
    y_hat = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    return y_hat


def show_W0(weights):
    Ws, bs = unpack(weights)
    W = Ws[0]
    n = int(NUM_HIDDEN[0] ** 0.5)
    plt.imshow(np.vstack([
        np.hstack([np.pad(np.reshape(W[idx1*n + idx2, :], [28, 28]), 2, mode='constant') for idx2 in range(n)]) for idx1 in range(n)
    ]), cmap='gray')
    plt.show()


def accuracy(y_true, y_pred):
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    return np.mean(y_true_labels == y_pred_labels)


def train(trainX, trainY, weights, testX, testY, lr, epochs, batch_size):
    for epoch in range(epochs):
        #Shuffle training data
        indices = np.random.permutation(trainX.shape[0])
        trainX_shuffled = trainX[indices]
        trainY_shuffled = trainY[indices]

        # Split the data into minibatches
        num_batches = trainX.shape[0] // batch_size
        X_batches = np.array_split(trainX_shuffled, num_batches)
        Y_batches = np.array_split(trainY_shuffled, num_batches)

        for X_batch, Y_batch  in zip(X_batches, Y_batches):
            # X_batch = trainX_shuffled[i:i + batch_size]
            # Y_batch = trainY_shuffled[i:i + batch_size]

            #Compute gradients and update weights
            gradVec = gradCE(X_batch, Y_batch, weights)
            weights -= lr * gradVec

        #Evaluate on the test set
        train_loss = fCE(trainX, trainY, weights)
        test_loss = fCE(testX, testY, weights)


        train_pred = forward(trainX, weights)
        test_pred = forward(testX, weights)

        train_acc = accuracy(trainY, train_pred)
        test_acc = accuracy(testY, test_pred)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

    return weights

# Weight and bias initialization
def initWeightsAndBiases():
    Ws = []
    bs = []
    # Kaiming He initialization
    np.random.seed(0)
    W = 2 * (np.random.random(size=(NUM_HIDDEN[0], NUM_INPUT)) / NUM_INPUT**0.5) - 1. / NUM_INPUT**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_HIDDEN[0])
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        W = 2 * (np.random.random(size=(NUM_HIDDEN[i + 1], NUM_HIDDEN[i])) / NUM_HIDDEN[i]**0.5) - 1. / NUM_HIDDEN[i]**0.5
        Ws.append(W)
        b = 0.01 * np.ones(NUM_HIDDEN[i + 1])
        bs.append(b)

    W = 2 * (np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN[-1])) / NUM_HIDDEN[-1]**0.5) - 1. / NUM_HIDDEN[-1]**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_OUTPUT)
    bs.append(b)

    return np.hstack([W.flatten() for W in Ws] + [b.flatten() for b in bs])

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

if __name__ == "__main__":
    # Load training data (replace these with actual .npy file paths)
    trainX = np.load('image/fashion_mnist_train_images.npy')
    trainY = np.load('image/fashion_mnist_train_labels.npy')
    testX = np.load('image/fashion_mnist_test_images.npy')
    testY = np.load('image/fashion_mnist_test_labels.npy')

    num_classes = 10  # Fashion MNIST has 10 classes
    trainY = one_hot_encode(trainY, num_classes)
    testY = one_hot_encode(testY, num_classes)
    # Recommendation: divide the pixels by 255 (so that their range is [0-1]), and then subtract
    # 0.5 (so that the range is [-0.5,+0.5]).
    trainX = trainX / 255.0 - 0.5
    testX = testX / 255.0 - 0.5


    weights = initWeightsAndBiases()

    # Perform numerical gradient check
    print(scipy.optimize.check_grad(
        lambda weights_: fCE(np.atleast_2d(trainX[0:5]), np.atleast_2d(trainY[0:5]), weights_), \
        lambda weights_: gradCE(np.atleast_2d(trainX[0:5]), np.atleast_2d(trainY[0:5]), weights_), \
        weights))
    print(scipy.optimize.approx_fprime(weights, lambda weights_: fCE(np.atleast_2d(trainX[0:5]), np.atleast_2d(trainY[0:5]), weights_), 1e-6))
    # Train the model
    weights = train(trainX, trainY, weights, testX, testY, lr=5e-2, epochs=100, batch_size=128)

    # Visualize the first layer of weights
    show_W0(weights)