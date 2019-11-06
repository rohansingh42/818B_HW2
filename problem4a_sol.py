
import numpy as np
import csv

from matplotlib import pyplot as plt

from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, ExpSineSquared

trainingInput = []
trainingOutput = []
testingInput = []
with open('training_data/problem4a_train.csv') as trainingDataFile:
	reader = csv.reader(trainingDataFile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
	for data in reader:
		trainingInput.append(data[0])
		trainingOutput.append(data[1])

with open('testing_data/problem4a_test.csv') as testingDataFile:
	reader = csv.reader(testingDataFile, quoting=csv.QUOTE_NONNUMERIC)
	for data in reader:
		testingInput.append(data)

# Creating X and Y vector
X = np.atleast_2d(trainingInput).T
Xstar = np.atleast_2d(testingInput)
Y = np.ravel(trainingOutput)
x = np.atleast_2d(np.linspace(0,5,100)).T

# Kernel definition
# kernel = ConstantKernel(1, (1e-6, 1e+6))*RBF(1, (1e-6, 1e+6)) + WhiteKernel(1, (1e-6, 1e+6))
kernel = ConstantKernel(1, (1e-6, 1e+6))*ExpSineSquared() + WhiteKernel(1, (1e-6, 1e+6))

# Start GP Regression Model
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=False)

# Fit data
gp.fit(X, Y)

# Predict Output
Ystar, sigma = gp.predict(Xstar, return_std=True)

# Print mean error
meanErr = np.linalg.norm((Ystar - np.ravel(np.sin(3*Xstar))))/np.sqrt(Ystar.shape[0])
print("The mean error is : ", meanErr)

# Print learned hyperparameters
hypParams = gp.kernel_;
print("The optimized kernel with learned values is : ", hypParams)

plt.figure()
plt.plot(x, np.sin(3*x), 'r:', label=r'$f(x) = sin(3x)$')
plt.plot(X, Y, 'r.', markersize=10, label='Observations')
plt.plot(Xstar, Ystar, 'b.', label='Prediction')
plt.fill(np.concatenate([Xstar, Xstar[::-1]]),
        np.concatenate([Ystar - 2 * sigma,
                        (Ystar + 2 * sigma)[::-1]]),
         	alpha=.5, fc='b', ec='None', label='2*sigma interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-2, 2)
plt.legend(loc='upper left')
plt.show()