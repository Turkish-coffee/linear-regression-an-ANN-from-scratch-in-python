import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor


# Data set 
np.random.seed(0)    
x , y = make_regression(n_samples=100, n_features=1, noise=10)
y = y.reshape(y.shape[0], 1)
plt.scatter(x,y)
plt.show()


# seting the X matrix 
X = np.concatenate((x, np.ones(x.shape)), axis=1)

# setting the teta veector
teta = np.random.randn(2,1)

# setting the model
def regression_model(X,teta):
    return X.dot(teta)

# cost fonction 
def cost_function(X,y,teta):
    m = len(y) 
    return 1/(2*m)*np.sum((regression_model(X,teta) - y)**2)

# setting the gradient  
def gradient(X,y,teta):
    m = len(y)
    return 1/m * X.T.dot(regression_model(X, teta) - y)

# gradient_descent 
def gradient_descent(X, y, teta, alpha, n_iteration):
    cost_history = np.zeros(n_iteration)
    for i in range(0, n_iteration):
        teta = teta - alpha * gradient(X,y,teta)
        cost_history[i] = cost_function(X, y, teta)
        if(i%70 == 0):
            plt.plot(x, regression_model(X,teta), c = 'g')
            plt.pause(0.001)
    plt.show()
    return teta,cost_history

final_teta_value, cost_history = gradient_descent(X, y, teta, alpha = 0.01, n_iteration = 2000)

print("the final teta value is the following: {}".format(final_teta_value))

# setting the prediction vector 
prediction_vector = regression_model(X, final_teta_value)
plt.scatter(x,y)
plt.plot(x, prediction_vector,c = 'r')
plt.show()
# learning graph 
plt.plot(range(2000), cost_history)
plt.show()
# correlation_coefficient
def correlation_coefficient(y, prediction_vector):
    u = ((y - prediction_vector)**2).sum()
    v = ((y - y.mean())**2).sum()
    return 1-u/v
print("correlation value: {}".format(correlation_coefficient(y,prediction_vector)))
                

