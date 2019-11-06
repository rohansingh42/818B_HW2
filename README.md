# HW2 : SUBMODULARITY AND GAUSSIAN PROCESS REGRESSION

## Dependencies

Written on Python 3.5
- scikit-learn
- numpy

## Compilation 

- Output file is provided so not necessary. From base directory (818B_HW1_TSP)

```
g++ -std=c++11 src/p1.cpp -o output/p1
```
## Execution

Open Terminal from base directory (818B_HW1_TSP)

- For problem 4 (a),

```
python3 problem4a_sol.py
```
- For problem 4 (b),

```
python3 problem4b_sol.py
```
## Solution Description

- The training dataset files (.csv) are stored in _/training_data_ directory and the testing dataset files in _/testing_data_ directory. The generated output plots and predicted values are stored in _/output_ directory.
- On running the script for problem 4(a), the program will display the mean square error of the predicted values with the ground truth, kernel with optimized hyperparameter values and an output plot.
- On running the script for problem 4(b), the program will display the mean square error of the predicted values with the ground truth, kernel with optimized hyperparameter values and write the predicted values in the _/output_ directory. The name of the output file is _problem4b_output.csv_
