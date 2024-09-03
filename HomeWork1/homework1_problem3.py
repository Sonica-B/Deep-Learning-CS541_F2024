import numpy as np
import matplotlib.pyplot as plt

gradient_sequence = np.loadtxt('gradient_descent_sequence.txt')

x1 = gradient_sequence[:,0]
x2 = gradient_sequence[:,1]
a1 , a2 = 0.2, 6
c1, c2 = 0.6, 3
def zigzag(x1,x2, a1,a2,c1,c2):
    return (a1*(x1-c1)**2)+(a2*(x2-c2)**2)

# Create a meshgrid for plotting the contour
x1_grid, x2_grid = np.meshgrid(np.linspace(-4, 4, 400), np.linspace(-4, 4, 400))
z_grid = zigzag(x1_grid, x2_grid, a1, a2, c1, c2)

# Plot the contour of the function
plt.figure(figsize=(8, 6))
plt.contour(x1_grid, x2_grid, z_grid, levels=50, cmap='viridis')
plt.scatter(x1, x2, color='blue', label='Gradient Descent Sequence')
plt.plot(x1, x2, color='red', linestyle='-')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Gradient Descent on Convex Paraboloid')
plt.legend()
plt.axis('equal')
plt.show()







