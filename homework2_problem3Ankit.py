import numpy as np

def fashion_mnist_data():
    # Load data
    X_tr = np.load("fashion_mnist_train_images.npy")
    y_trlb = np.load("fashion_mnist_train_labels.npy")
    Xte = np.load("fashion_mnist_test_images.npy")
    y_telb = np.load("fashion_mnist_test_labels.npy")

    classes = np.max(y_trlb) + 1
    yh_train = np.eye(classes)[y_trlb]
    yhte_lb = np.eye(classes)[y_telb]


    # Splitting the data
    N = X_tr.shape[0]
    split = int(0.8 * N)
    Xtr = X_tr[:split]
    Xv = X_tr[split:]
    yhtr_lb = yh_train[:split]
    yhv_lb = yh_train[split:]

    return Xtr,yhtr_lb,Xv,yhv_lb,Xte,yhte_lb

def softmax(x,w,b):
    z = np.dot(x,w) + b
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z,axis=1,keepdims=True)

def ce_cost_function(x,y,w,b,alpha):
    n,f = np.shape(x)

    y_hat = softmax(x, w, b)
    reg = (alpha/2) * np.sum(w * w)
    loss = -np.sum(y * np.log(y_hat))/n
    ce_loss = loss + reg

    return ce_loss

def gradient(x,y,w,b,alpha):
    n, f = np.shape(x)
    y_hat = softmax(x, w, b)

    grad_w = (-(1/n) * np.dot(x.T,(y-y_hat))) + (alpha * w)
    grad_b = -(1/n) * np.sum((y-y_hat),axis=0)

    return grad_w,grad_b

def sgd_function(x,y,xv,yv,epsilon,mini_batch,epoch,alpha):
    n,f = np.shape(x)
    w = np.random.randn(f,10)
    b = np.random.randn(1,10)

    train_cost_list = []
    val_cost_list = []

    if mini_batch > n:
        mini_batch = n

    for i in range(epoch):
        # Shuffling the dataset
        ids = np.random.permutation(n)
        x_shuffle = x[ids]
        y_shuffle = y[ids]
        for j in range(0, n, mini_batch):

            xj = x_shuffle[j:j + mini_batch]
            yj = y_shuffle[j:j + mini_batch]

            # Gradient
            dldw,dldb = gradient(xj, yj, w, b, alpha)


            w -= (epsilon * dldw)
            b -= (epsilon * dldb)


        train_cost = ce_cost_function(x, y, w, b,alpha)
        train_cost_list.append(train_cost)

        val_cost = ce_cost_function(xv, yv, w, b,alpha)
        val_cost_list.append(val_cost)

        if i%10 == 0:
            print(f'For epoch:{i}, training cost:{train_cost} and validation cost:{val_cost}')

    return w, b, train_cost_list, val_cost_list

def accuracy(y,y_hat):
    predictions = np.argmax(y_hat,axis=1)
    true_labels = np.argmax(y,axis=1)

    a = np.mean(predictions == true_labels) * 100

    return a

X_train,y_train,X_val,y_val,X_test,y_test = fashion_mnist_data()
X_train = X_train/255.0
X_val = X_val/255.0
X_test = X_test/255.0

#learning_rates = np.random.choice(np.linspace(0.001,0.0025,100),size=3,replace=False)
learning_rates = [0.12,0.012]
mini_batch_size = [64]
epochs = [200,300,400]
alpha = [0.01,0.001]

# Recording the best values
best_cost_val = float('inf')
best_hparameters = {}
best_w, best_b = None, None

#hyperparameter values
for a in alpha:
    for epsilon in learning_rates:
        for batch in mini_batch_size:
            for e in epochs:
                print(f'Training set with learning rate:{epsilon}, mini batch size:{batch}, number of epochs:{e} and alpha:{a}')
                weight,bias,train_cost,val_cost = sgd_function(X_train,y_train,X_val,y_val,epsilon,batch,e,a)
                val_cost = val_cost[-1]

                if val_cost < best_cost_val:
                    best_cost_val = val_cost
                    best_hparameters = {'learning_rate':epsilon,'mini_batch_size':batch,'epochs':e}
                    best_w, best_b = weight,bias

                yte_hat = softmax(X_test, best_w, best_b)
                test_cost_value = ce_cost_function(X_test, y_test, best_w, best_b, a)

                unre_loss = -np.sum(y_test * np.log(yte_hat)) / np.shape(X_test)[0]
                a = accuracy(y_test, yte_hat)

print(f'Best Hyper Parameters: {best_hparameters}')
print(f'Test Cost Value: {test_cost_value}')
print(f'Best Validation cost value:{best_cost_val}')
print(f'Cost values of Training Dataset of last 10 iterations: {train_cost[-10:]}')
print(f'Unregularized Cross-Entropy Loss on Test Set: {unre_loss}')
print(f'Accuracy on Test data: {a}')

# Best Hyper Parameters: {'learning_rate': 0.012, 'mini_batch_size': 64, 'epochs': 400}
# Test Cost Value: 0.499340231523267
# Best Validation cost value:0.4740566966628365
# Cost values of Training Dataset of last 10 iterations: [np.float64(0.45424104094046597), np.float64(0.45407102116023457), np.float64(0.4548415949641077), np.float64(0.45294762459703625), np.float64(0.4537044665743292), np.float64(0.4527285956367639), np.float64(0.4536544309360167), np.float64(0.452777305301885), np.float64(0.4540003656099068), np.float64(0.45263964916231475)]
# Unregularized Cross-Entropy Loss on Test Set: 0.4509065411543971
# Accuracy on Test data: 84.1