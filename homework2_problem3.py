import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
train_images = np.load('image/fashion_mnist_train_images.npy')
train_labels = np.load('image/fashion_mnist_train_labels.npy')
test_images = np.load('image/fashion_mnist_test_images.npy')
test_labels = np.load('image/fashion_mnist_test_labels.npy')

# Normalize images
train_images = train_images / 255
test_images = test_images / 255

# Split data
def train_validation_split(X, y, validation_size=0.2):
    n_train = int((1 - validation_size) * X.shape[0])
    indices = np.random.permutation(X.shape[0])
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

train_labels = one_hot_encode(train_labels)
test_labels = one_hot_encode(test_labels)
X_train, y_train, X_val, y_val = train_validation_split(train_images, train_labels)



def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def CrossEntropy_loss(y_true, y_pred, W, a,regularized=True):
    n_samples = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(n_samples), np.argmax(y_true, axis=1)])
    unregularized_loss = np.sum(log_likelihood) / n_samples

    if regularized:
        regularization = (a / 2) * np.sum(W ** 2)
        return unregularized_loss + regularization, unregularized_loss
    else:
        return unregularized_loss


def predict(X, W, b):
    z = np.dot(X, W) + b
    return softmax(z)

def SGD(X, y_true, y_pred, W, a):
    n_samples = X.shape[0]
    gw = np.dot(X.T, (y_pred - y_true)) / n_samples + a * W
    gb = np.sum(y_pred - y_true, axis=0) / n_samples
    return gw, gb


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, e, a, batch_size, epochs):
    input_dim = X_train.shape[1]
    num_classes = y_train.shape[1]
    W = np.random.randn(input_dim, num_classes) * 0.01
    b = np.zeros((1, num_classes))


    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        # Shuffle the training data
        perm = np.random.permutation(X_train.shape[0])
        X_train, y_train = X_train[perm], y_train[perm]


        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]


            y_pred = predict(X_batch, W, b)     #Forward pass


            gw, gb = SGD(X_batch, y_batch, y_pred, W, a)


            W -= e * gw
            b -= e * gb


        y_val_pred = predict(X_val, W, b)       #Validation
        val_loss, unreg_val_loss = CrossEntropy_loss(y_val, y_val_pred, W, a, regularized=True)
        val_accuracy = np.mean(np.argmax(y_val_pred, axis=1) == np.argmax(y_val, axis=1))


        val_losses.append(val_loss)                 #Store loss and accuracy for plotting
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{epochs}: Validation Loss = {val_loss:.4f}, Validation Accuracy = {val_accuracy:.4f}")


    y_test_pred = predict(X_test, W, b)             #Evaluate test set performance
    test_loss, unreg_test_loss = CrossEntropy_loss(y_test, y_test_pred, W, a, regularized=True)
    test_accuracy = np.mean(np.argmax(y_test_pred, axis=1) == np.argmax(y_test, axis=1))

    print(
        f"Test Loss: {test_loss:.4f}, Unregularized Test Loss: {unreg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    return W, b, val_losses, val_accuracies, unreg_test_loss, test_accuracy


W, b, val_losses, val_accuracies, unreg_test_loss, test_accuracy = train_and_evaluate(
    X_train, y_train, X_val, y_val, test_images, test_labels, e=0.01, a=0.001, batch_size=64, epochs=100
)


def plot_results(val_losses, val_accuracies, unreg_test_loss, test_accuracy):
    epochs = len(val_losses)

    # Plot Validation Loss and Validation Accuracy over Epochs
    plt.figure(figsize=(14, 6))

    # Plot Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), val_losses, label="Validation Loss", color="red")
    plt.title("Validation Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), val_accuracies, label="Validation Accuracy", color="blue")
    plt.title("Validation Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


    print(f"Final Unregularized Test Loss: {unreg_test_loss:.4f}")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")


#Visualize
plot_results(val_losses, val_accuracies, unreg_test_loss, test_accuracy)





