import matplotlib.pyplot as plt
import numpy as np

# Function for f(x)
def f(x):
    if (x < 0.1):
        return -(x * x * x)
    elif ((x >= -0.1) & (x < 3)):
        return ((-3 * x / 100)) - (1/500)
    elif ((x >= 3) & (x < 5)):
        return - ((x - (31/10))*(x - (31/10))*(x - (31/10))) - (23/250)
    elif (x >= 5):
        return ((1083/200) * ((x - 6)*(x - 6))) - (6183/500)

# Function for gradient of f(x)
def gradient_f(x):
    if(x<0.1):
        return -3 * (x * x)
    elif((x>=-0.1) & (x<3)):
        return -3/100
    elif((x>=3) & (x<5)):
        return -3 * ((x-(31/10)) * (x-(31/10)))
    elif(x>=5):
        return (1083/100) * (x-6)

# Gradient descent function
def gradient_descent(x,iteration,epsilon):
    # Keeping start value of x in the array
    x_values = [x]

    for i in range(iteration):
        grad = gradient_f(x)

        # Checking NaN and Infinite values
        if np.isnan(grad) or np.isinf(grad):
            print(f'Found NaN or Inf value at x:{x}')
            break

        x -= (epsilon * grad)
        x_values.append(x)
    return np.array(x_values)

# Initializing the starting value of x while computing the gradient descent for each learning rate
start_value = -3

# Vectorize the f(x) and grad_f(x) function
f_vec = np.vectorize(f)
grad_vec = np.vectorize(gradient_f)

# Equally spacing values of x from start to end values
x = np.linspace(-4,10,1000)
y = f_vec(x)
y_grad = grad_vec(x)

# Assigning different learning rates and their corresponding scatter plot colors

# Added extra learning rate value where the gradient descent touches the global minimum and jumps out(observation)
epsilon = [1e-3, 1e-2, 1e-1, 1e0, 1e1,0.2268]
color = ['red','violet','cyan','indigo','black','orange']

# Creating a plot figure and a set of sub-plots within that figure
figure, multi_plot = plt.subplots(2,3)
multi_plot = multi_plot.ravel()

# Plotting Figures(f(x)/∇f(x) and values of x for different learning rates)
for i,(e,c) in enumerate(zip(epsilon,color)):
    x_scatter = gradient_descent(start_value, iteration=100, epsilon=e)
    y_scatter = f_vec(x_scatter)
    multi_plot[i].scatter(x_scatter, y_scatter, color=c, label=f'Epsilon:{e}', s=30)

    multi_plot[i].plot(x, y, label='f(x)', color='blue')
    multi_plot[i].plot(x, y_grad, label=' ∇f(x)', color='green')

    multi_plot[i].set_xlabel('x')
    multi_plot[i].set_ylabel('f(x)/ ∇f(x)')
    multi_plot[i].legend()
    multi_plot[i].grid(True)

plt.show()