import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_regression

# dataset:

np.random.seed(0)

(x , y) = make_regression(n_samples = 100, n_features = 1, noise = 10)
y = y + abs(y/0.2)
y = y.reshape(y.shape[0],1)
plt.scatter(x,y)
plt.show()

# setting the X matrix:

X = np.hstack((x,np.ones(x.shape)))
X = np.hstack((x**2,X))
X = np.hstack((x**3,X))
X = np.hstack((x**4,X))

# setting the teta vector:

teta = np.random.randn(5,1)

# setting the model: 

def regression_model(X, teta):
    return X.dot(teta)

# setting the cost function: 

def cost_function(X, y, teta):
    m = len(y)
    return 1/(2*m)*np.sum((regression_model(X, teta) - y)**2)

# setting the gradient: 

def gradient(X, y, teta):
    m = len(y)
    return 1/m*X.T.dot(regression_model(X,teta) - y)

# setting the gradient descent

def gradient_descent(X, y, teta, alpha, n_interation):
    cost_history = np.zeros(n_interation)
    for i in range(0, n_interation):
        teta = teta - alpha * gradient(X, y, teta)
        cost_history[i] = cost_function(X, y, teta)
        if (i%39 == 0):
            plt.scatter(x[:,0],regression_model(X,teta), c = 'g', s = 1)
    plt.show()        
    return teta, cost_history                       


final_teta, cost_history = gradient_descent(X, y, teta, alpha = 0.01, n_interation = 1000)
print("the final teta value: {}".format(final_teta))

# setting the prediction vector
prediction = regression_model(X,final_teta)
plt.scatter(x[:,0],y)
plt.scatter(x[:,0],prediction, c = 'r')
plt.show()
# learning rate 

plt.plot(range(1000),cost_history)
plt.show( )

# setting the correlation coefficient 

def correlation_coefficient(y,prediction):
    u = ((y - prediction)**2).sum()
    v = ((y - y.mean())**2).sum()
    return 1-u/v

print("correelation_coefficient: {}".format(correlation_coefficient(y,prediction)))