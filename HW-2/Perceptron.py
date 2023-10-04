import numpy as np

def perceptron(x):
    for i in range(len(x)):
        x[i] = 1 if x[i] > 0 else 0


def sigmoid(x):
    for i in range(len(x)):
        x[i] = 1/(1 + np.exp(-x[i]))

def findOut(f, title):
    print(f"Function: {title}-------------------------------")
    inputs  =  [[0,0,0],  [0,0,1],  [0,1,0],  [0,1,1],  [1,0,0],  [1,0,1],  [1,1,0],  [1,1,1]]

    W_1 = np.array([[0.6, 0.5, -0.6], [-0.7, 0.4, 0.8]])

    B_1 = np.array([-.4, -.5]).T

    W_2 = np.array([1, 1])

    B_2 = np.array([-.5])

    for x in inputs:
        print(f"Input: {x}", end=" ")
        xi = np.array(x).T
        a_1 = W_1@xi + B_1
        print(f"Output_1: {a_1}", end=" ")
        f(a_1)
        a_2 = W_2@a_1 + B_2
        print(f"Output_2: {a_2}", end=" ")
        f(a_2)
        print(f"Final Output: {a_2}")



if __name__ == "__main__":

    findOut(perceptron, "Perceptron")
    findOut(sigmoid, "Sigmoid")
    
    
