import numpy as np
import time

#How to make an artificial neural network from scratch :)

#Step1 - generate random binary
#Step2 - Create a NN
#Step3 - Train NN
#Step4 - Use NN to make a prediction
#Step5 - Compare results for accuracy

#Our Variables
#n_hidden = number of hidden layers
n_hidden = 20 
#No. of variables being input
n_in = 20
#No. of variables being output
n_out = 20
#No. samples
n_sample = 500
	
#Hyper parameters
	
#When a neural network is initially presented with a pattern
#It makes a random 'guess' as to what it might be
#this 'guess' is then corrected via backwards and forwards propagation
#This method is known as 'backpropagational neural networks' (BPNNs)
#After multiple adjustments we form a solution surface
#However this surface is not 'smooth'
#The surface has pits and hills which form a error space
#In a NN we often cannot pre adjust for the nature of the error space
#So the neural network cycles to avoid local minima and find the global minimum
#Cycles are also known as Epochs
#Momentum controls the speed at which the NN forms a solution
#Momentum helps the network to overcome obstacles (local minima) in the error surface
#Learning rate is the rate of convergence between the current solution and the global minimum
learning_rate = 0.01
momentum = 0.9

#Seed makes sure you return the same random numbers each run
#We want to see the consistent return of our NN
np.random.seed(90)

#When data is trained by the NN its probabilities are shifted
#This 'shift' is determined by the activation function
def sigmoid(x):
	'''our sigmoid function turns numbers into probabilities'''		 	
	return 1.0/(1.0 + np.exp(-x))

def tanh_prime(x):	
	'''this is another activation function'''
	return 1 - np.tanh(x)**2

#x is the matrix input data
#t is our transpose which is used in matrix transformation
#V and W are the two layers of our NN
# bV and bW represent adjustments for bias
def train(x, t, V, W, bV, bW):
	#The NN works by first applying a set of weights going forward
	#Forward propagation
	#matrix multiply + bais
	A = np.dot(x, V) + bV
	Z = np.tanh(A)
	B = np.dot(Z, W) + bW
	Y= sigmoid(B)
	
	#Then creating a new set of weights in the opposite direction
	#Backward propagation
	#It is essentially our matrix of values flipped backwards Y-t
	Ew = Y - t
	Ev = tanh_prime(A) * np.dot(W, Ew)
	
	#predict loss
	dW = np.outer(Z, Ew)
	dV = np.outer(x, Ev)

	#Cross entropy function
	#there are a number of other loss functions such as mean squared error
	#However for classification the cross entry function offers the best performance
	loss = -np.mean((t * np.log(Y)) + ((1 - t) * np.log(1-Y)))

	return loss, (dV, dW, Ev, Ew)
	
	
#The next important part of the NN is the prediction function
#This is what allows the NN to make a prediction
def predict(x, V, W, bV,bW):
	A = np.dot(x, V) + bV
	B =np.dot(np.tanh(A), W) + bW
	return (sigmoid(B) > 0.5).astype(int)
	
#Creating Layers
V = np.random.normal(scale =0.1, size =(n_in, n_hidden))
W = np.random.normal(scale=0.1, size=(n_hidden, n_out))
	
bV = np.zeros(n_hidden)
bW = np.zeros(n_out)
	
params = [V, W, bV, bW]
	
#Generate data
X = np.random.binomial(1, 0.5, (n_sample, n_in))
T = X ^ 1
	
#Its Training Day!!!
for epoch in range(100):
	err = []
	upd = [0]*len(params)

	t0 = time.clock()
	
	#For each data point we want to update our weights
	for i in range(X.shape[0]):
		loss, grad = train(X[i], T[i], * params)

		#Update loss
		for j in range(len(params)):
			params[j] -= upd[j]
	
		for j in range(len(params)):
			upd[j] = learning_rate * grad[j] + momentum * upd[j]
			
			err.append(loss)
	
		print('Epoch: %d, loss: %.8f, Time: %.4fs'%(
			epoch, np.mean(err), time.clock()-t0))
	
#Lets print our predictions
x =np.random.binomial(1, 0.5, n_in)
print('Randomly generated binary')
print(x)
print('XOR (exclusive or) prediction')
endpredict = (predict(x, *params))
print(endpredict)
	
#How accurate is our prediction

p_a = 0
lst_x = np.ndarray.tolist(x)
lst_endpredict = np.ndarray.tolist(endpredict)
for i, item in enumerate(x):
	if (lst_x[i] == lst_endpredict[i]) is False:
		p_a += 1
print('Predictction Accuracy', (p_a/len(x))*100, '%')
