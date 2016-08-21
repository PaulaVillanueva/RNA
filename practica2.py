import numpy as np
import matplotlib.pyplot as plt
import random
import os, subprocess

def plot(XS, vec):
    
	#seteos del grafico
    fig = plt.figure(figsize=(5,5)) 
    plt.xlim(-1.5,1.5) 
    plt.ylim(-1.5,1.5)
    cols = {1: 'r', -1: 'b'} 
    for x,s in XS:
        plt.plot(x[1], x[2], cols[s]+'o')

	#magia del hiperplano
	l = np.linspace(-1.5,1.5) 
	a = -vec[1]/vec[2]
	b = -vec[0]/vec[2]
	plt.plot(l, a*l+b, 'g-')


def make_dataset(X,Z):
	D = []
	for i in range(4):
		D.append((X[i],Z[i]))
	return D

def perceptron(D, save=False):
	e = 1
	eps = 0.01
	t = 0
	T = 10
	coef = 0.01

	W = np.random.rand(3)/10
	E = np.array([0,0,0,0])

	while (e > eps) and (t < T):
		e = 0
		i = 0

		for en,s in D:
			Y = np.sign(np.inner(en,W))
			E[i] = s - Y
			dw = coef * np.inner(np.transpose(X),E)
			W += dw
			i += 1
			e += np.linalg.norm(E)**2
		t += 1

		if save:
			plot(D,W)
			plt.title('Iteration %s\n' \
				% (str(t)))
			plt.savefig('_it%s' % (str(t)), \
				dpi=200, bbox_inches='tight')

		print "Error",e,'\n'

	if not save:
		plot(D,W)
		plt.show()
	print "COEFICIENTES", W


#-------------------------------------------------------------#

X = np.array([[1, -1,-1],[1,1,-1],[1,-1,1],[1,1,1]]) #BIAS = 1

Z_OR = np.array([-1,1,1,1]) 
Z_XOR = np.array([-1,1,1,-1]) 
Z_AND = np.array([-1,-1,-1,1]) 

D = make_dataset(X,Z_XOR)
perceptron(D,save=True)
