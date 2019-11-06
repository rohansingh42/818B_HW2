
import numpy as np
import csv

from matplotlib import pyplot as plt

from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, DotProduct, ExpSineSquared, RationalQuadratic

trainingInput = []
trainingOutput = []
testingInput = []
testingGroundTruth = []
with open('training_data/problem4b_train.csv') as trainingDataFile:
	reader = csv.reader(trainingDataFile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
	for data in reader:
		trainingInput.append(data[0:4])
		trainingOutput.append(data[4])

with open('testing_data/problem4b_test.csv') as testingDataFile:
	reader = csv.reader(testingDataFile, quoting=csv.QUOTE_NONNUMERIC)
	for data in reader:
		testingInput.append(data[0:4])

with open('testing_data/problem4b_sol.csv') as testingGTDataFile:
	reader = csv.reader(testingGTDataFile, quoting=csv.QUOTE_NONNUMERIC)
	for data in reader:
		testingGroundTruth.append(data)

# Creating X and Y vector
X = np.atleast_2d(trainingInput)
Xstar = np.atleast_2d(testingInput)
Y = np.ravel(trainingOutput)
GT = np.ravel(testingGroundTruth)

# print(X, Xstar)
# Kernel definition
kernel = ConstantKernel(1, (1e-6, 1e+6))*RBF(1, (1e-6, 1e+6)) + WhiteKernel(1, (1e-6, 1e+6))
# kernel = ConstantKernel(1, (1e-6, 1e+6))*RBF(1, (1e-6, 1e+6))*ExpSineSquared() + WhiteKernel(1, (1e-6, 1e+6))
# kernel = ConstantKernel(1, (1e-6, 1e+6))*RBF(1, (1e-6, 1e+6)) + WhiteKernel(1, (1e-6, 1e+6))*DotProduct(1, (1e-6, 1e+6))
# kernel = ConstantKernel(1, (1e-6, 1e+6))*RationalQuadratic() + WhiteKernel(1, (1e-6, 1e+6))

# Start GP Regression Model
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)

noIter = 50
subsetSize = 500

for i in range(noIter):
	index = np.random.choice(X.shape[0], subsetSize, replace=False)

	# Fit data
	gp.fit(X[index], Y[index])

	print("Iteration : ", i)

# Predict Output
Ystar, sigma = gp.predict(Xstar, return_std=True)

# Print mean error
meanErr = np.linalg.norm(np.ravel(Ystar) - np.ravel(GT))/np.sqrt(Ystar.shape[0])
print("The mean error is : ", meanErr)

# Print learned hyperparameters
hypParams = gp.kernel_;
print("The optimized kernel with learned values is : ", hypParams)

with open('output/problem4b_output.csv', mode = 'w') as predictedData:
    fileWriter = csv.writer(predictedData, delimiter = ',', quoting=csv.QUOTE_NONNUMERIC)
    for i in range(sigma.shape[0]):
        fileWriter.writerow([Ystar[i], sigma[i]])
