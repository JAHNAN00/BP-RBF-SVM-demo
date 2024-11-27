import numpy as np
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = np.where(y <= 0, -1, 1)  # Ensure labels are -1 or 1
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) < 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

# Testing the implementation
if __name__ == "__main__":
    # Manually generate a larger dataset (more points)
    # Class -1 points (x1, x2)
    class_neg = np.array([
        [1, 2], [2, 3], [3, 3], [4, 5], [5, 6],
        [6, 7], [7, 8], [8, 9], [9, 10], [10, 11],
        [2, 5], [3, 4], [4, 6], [5, 7], [6, 8]
    ])
    # Class +1 points (x1, x2)
    class_pos = np.array([
        [7, 2], [8, 3], [9, 4], [10, 5], [11, 6],
        [12, 7], [13, 8], [14, 9], [15, 10], [16, 11],
        [7, 3], [8, 4], [9, 5], [10, 6], [11, 7]
    ])
    
    # Combine both classes
    X = np.vstack([class_neg, class_pos])
    y = np.array([-1] * len(class_neg) + [1] * len(class_pos))  # Labels
    
    # Normalize features manually
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_normalized = (X - X_mean) / X_std

    # Train the SVM
    svm = SVM(learning_rate=0.01, lambda_param=0.01, n_iters=1000)
    svm.fit(X_normalized, y)

    # Make predictions
    predictions = svm.predict(X_normalized)

    # Print the results
    print("Predictions:", predictions)
    print("Actual:", y)
    print("Accuracy:", np.mean(predictions == y))

    # Plotting the results
    plt.figure(figsize=(8, 6))

    # Plot class -1 (red) and class +1 (blue)
    plt.scatter(X_normalized[y == -1, 0], X_normalized[y == -1, 1], color='red', marker='o', label='Class -1')
    plt.scatter(X_normalized[y == 1, 0], X_normalized[y == 1, 1], color='blue', marker='x', label='Class +1')

    # Plot the decision boundary (SVM hyperplane)
    x_min, x_max = X_normalized[:, 0].min() - 1, X_normalized[:, 0].max() + 1
    y_min, y_max = X_normalized[:, 1].min() - 1, X_normalized[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    # Compute decision function
    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], svm.w) + svm.b
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and margins
    plt.contour(xx, yy, Z, levels=[0], linewidths=5, colors='black')  # Decision boundary (hyperplane)
    plt.contour(xx, yy, Z, levels=[1], linestyles='dashed', colors='black')  # Margin boundary (support vectors)
    plt.contour(xx, yy, Z, levels=[-1], linestyles='dashed', colors='black')  # Margin boundary (support vectors)

    # Labels and title
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary and Margins')
    plt.legend()

    # Show plot
    #plt.show()
    plt.savefig("sin_function_fitting.png")