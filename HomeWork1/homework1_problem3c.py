import matplotlib.pyplot as plt
import numpy as np

def f(x):
    if x >= 0:
        return 2/3 * (x**(3/2))
    else:
        return 2/3 * ((-x)**(3/2))
def gradient_f(x):
    if x >= 0:
        return x**(1/2)
    else:
        return -((-x)**(1/2))

def gradient_descent(x,iteration,epsilon):
    # Keeping start value of x in the array
    x_values = [x]

    for i in range(iteration):
        grad = gradient_f(x)
        if np.isnan(grad) or np.isinf(grad):
            print(f'Found NaN or Inf value at x:{x}')
            break
        x -= (epsilon * grad)
        x_values.append(x)

    return np.array(x_values)

# Vectorize the f(x) and grad_f(x) function
f_vec = np.vectorize(f)
grad_vec = np.vectorize(gradient_f)

# Equally spacing values of x from start to end values
x = np.linspace(-10,10,1000)
y = f_vec(x)
y_grad = grad_vec(x)

# Assigning different learning rates and their corresponding scatter plot colors
epsilon = [1e-3, 1e-2, 1e-1]
color = ['red','violet','black']

# Initializing the starting value of x while computing the gradient descent for each learning rate
start_value = [-3,-10,7]

# Creating a plot figure and a set of sub-plots within that figure
figure, multi_plot = plt.subplots(3,3,figsize=(12, 10))
multi_plot = multi_plot.ravel()

# Plotting Figures(f(x)/∇f(x) and values of x using different learning rates, for every start values of x)
i = 0
for x_start in start_value:
    for (e,c) in zip(epsilon,color):
        x_scatter = gradient_descent(x_start, iteration=100, epsilon=e)
        y_scatter = f_vec(x_scatter)
        multi_plot[i].scatter(x_scatter, y_scatter, color=c, label=f'Epsilon:{e}', s=30)

        multi_plot[i].plot(x, y, label='f(x)', color='blue')
        multi_plot[i].plot(x, y_grad, label=' ∇f(x)', color='green')

        multi_plot[i].set_xlabel('x')
        multi_plot[i].set_ylabel('f(x)/ ∇f(x)')
        multi_plot[i].set_title(f"Learning Rate: {e}, Start Value: {x_start}")
        multi_plot[i].legend()
        multi_plot[i].grid(True)

        i += 1

plt.tight_layout()
plt.show()