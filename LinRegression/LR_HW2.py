import numpy as np 
import math
import random
from numpy import linalg as LA
import matplotlib.pyplot as plt
from numpy.linalg import inv

#bring in datasets
train = np.loadtxt('./LinRegression/data/train.csv', delimiter =',',usecols = range(8))
  
test = np.loadtxt('./LinRegression/data/test.csv', delimiter =',',usecols = range(8))





#define sgd cost function remove intial attempt
def cost_sgd(x, yvals, w):
	res = 0 
	for i in range(len(x)):
		temp = (yvals[i] - np.dot(w, x[i]))**2 
		res += temp 
	return 0.5*res

def sgd(x, yvals, r):

	w = np.zeros(x.shape[1])


	e = math.inf

	cost = [cost_sgd(x, yvals, w)]

	while e > 10e-10:
		i = random.randrange(len(x))

		grad_w = np.zeros(x.shape[1])
		for j in range(len(x[0])): 
			grad_w[j] = x[i][j] *(yvals[i] - np.dot(w, x[i]))

		new_W = w + r*grad_w
		w = new_W
		new_cost = cost_sgd(x, yvals, w) 
		e = abs(new_cost - cost[-1])
		cost.append(new_cost)

	return w, cost


#scg_inital
def scg_(xvalues, yvalues, lr: float = 1, epochs: int = 10, threshold = 1e-6):

    w = np.ones_like(xvalues[0])

    losses, lastloss, diff = [], 9999, 1
    for ep in range(epochs):
        if diff <= threshold: break
        # for each element, update weights
        for xi, yi in zip(xvalues, yvalues):
            for j in range(len(w)):
                w[j] += lr * (yi - np.dot(w, xi)) * xi[j]

            # compute loss
            loss = 0
            for xi, yi in zip(xvalues, yvalues):
                loss += (yi - np.dot(w, xi))**2
            loss /= 2
            
            diff = abs(loss - lastloss)
            lastloss = loss
            losses.append(loss)

    print(f"converged at epoch {ep} to {diff}")
    return lms_w(w), losses


def cost_f(x, yvals, w):
	res = 0 
	for i in range(len(x)):
		temp = (yvals[i] - np.dot(w, x[i]))**2 
		res += temp 
	return 0.5*res

#lms grad calc. get w and cost
def lms_grad(x, yvals, r):
	costs = []  

	W = np.zeros(x.shape[1])


	e = math.inf

	while e > 10e-6:
		grad_w = np.zeros(x.shape[1])
		
		for j in range(len(x[0])):
			temp = 0 
			for i in range(len(x)):
				temp += x[i][j] *(yvals[i] - np.dot(W, x[i]))
			grad_w[j] = temp 

		new_W = W + r*grad_w

		e = LA.norm(W - new_W)
		costs.append(cost_f(x, yvals, W))

		W = new_W

	costs.append(cost_f(x, yvals, W))
	return W, costs




X_train = train[:,:-1]
one_vect = np.ones(X_train.shape[0])
train_stk = np.column_stack((one_vect, X_train))
Y_train = train[:,-1]

X_test = test[:,:-1]
one_test = np.ones(X_test.shape[0])
D_test = np.column_stack((one_test, X_test))
Y_test = test[:,-1]

print()
# part a 
print("---4a---")
print("BGD")

r = 0.01
weghts, costs = lms_grad(train_stk, Y_train, r)
tst_cost = cost_f(D_test, Y_test, weghts)
print("r =  ", r)
print("w =  ", weghts)
print("Test cost f(x) =  ", tst_cost)
fig1 = plt.figure()
plt.plot(costs,c='purple')
fig1.suptitle('GD ', fontsize=20)
plt.xlabel('iteration', fontsize=18)
plt.ylabel('cost f(x)', fontsize=16)
plt.show()


#b
print()
print()
print("---4b---")
print("SGD")

r = 0.001
weghts, costs = sgd(train_stk, Y_train, r)
tst_cost = cost_f(D_test, Y_test, weghts)
print("r =  ", r)
print("w = ", weghts)
print("Test cost f(x) =  ", tst_cost)
fig2 = plt.figure()
plt.plot(costs,c='purple')
fig2.suptitle('SGD ', fontsize=20)
plt.xlabel('iteration', fontsize=18)
plt.ylabel('cost (fx)', fontsize=16)
plt.show()


#c
print()
print()
print("---4c---")
print("analytical form")

new_D_train = train_stk.T
temp = np.matmul(new_D_train, new_D_train.T)
invtemp = inv(temp)
final_w = np.matmul(np.matmul(invtemp, new_D_train), Y_train)
tst_cost = cost_f(D_test, Y_test, final_w)
print("w = ", final_w)
print("Test cost f(x) ", tst_cost)