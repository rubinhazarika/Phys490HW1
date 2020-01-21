"""

PHYS 490 
Assignment 1
Rubin Hazarika (20607919)

"""
# use > conda activate base (in terminal)

import sys 
import json
import numpy as np
import random

# file paths to .in and .json files
inPath = sys.argv[1]
jsonPath = sys.argv[2]

# opening and reading .in file
inFile = open(inPath, 'r')
inData = inFile.readlines()

# opening and reading .json file and getting GD data
with open(jsonPath) as json_file:
    jsonData = json.load(json_file)
    lrate = jsonData["learning rate"]
    iters = jsonData["num iter"]

# closing files
inFile.close
json_file.close

# Solving using least square regression -------------------------------------------------------------------

# given input - stripping \n characters, converting to 2D array and casting as float
inData = np.array([list(map(float,i)) for i in [(x.rstrip("\n")).split() for x in inData]])

# isolate and pad X matrix with 1s
xMat = np.delete(inData, len(inData[0])-1, 1)    # deleting last column (y)
addCol = np.ones((len(inData),1))                # column of 1s to append (padding)
xMatPad = np.hstack((addCol,xMat))               # appending column of 1s

# train and calculate w vector
xMatT = np.transpose(xMatPad)      # transpose of x-matrix
a = np.dot(xMatT, xMatPad)         # finding X^T * X
y = inData[:,len(inData[0])-1]                    # isolating y vector
b = np.dot(xMatT, y)               # finding X^T * y

# solving for w
w = np.linalg.solve(a,b)                         # solving equation (X^T * X) w = X^T y  
w_an = ["{:.4f}".format(round(x,4)) for x in w]  # rounding and formatting


# solving using stochastic GD -----------------------------------------------------------------------------
w_old = np.ones(len(xMatPad[0]))

# iterating and updating w_new with formula: w <- w + alpha*(y-h(x))*x
for i in range(0,iters):
    randInd = random.randint(0,len(inData)-1)
    w_new = w_old + lrate*(y[randInd] - w_old.dot(xMatPad[randInd]))*(xMatPad[randInd])
    w_old = w_new

w_gd = ["{:.4f}".format(round(x,4)) for x in w_new]  # rounding and formatting

# outputing to file ---------------------------------------------------------------------------------------

# adding newline character for writing
w_an = [(a + "\n") for a in w_an]
w_gd = [(a + "\n") for a in w_gd[0:-1]] + [w_gd[-1]]

# concatenating into single string
w_write = np.concatenate((w_an, ["\n"],w_gd))

outF = open(jsonPath.replace("json", "out"), "w")
outF.writelines(w_write)

# close file
outF.close