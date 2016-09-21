import numpy as np

# Estamos poniendo beta=1, no tendriamos que poder variarlo? De ser asi tambien hay que cambiarlo en los demas .py

#def sigmoid_array(x):
#    return 1 / (1 + np.exp(-x))
#
#def sigmoid_gradient_array(x):
#	return sigmoid_array(x)*(1-sigmoid_array(x))
	
	# En la teorica de Segura la sigmoidea esta con un 2*b
	
def sigmoid_array(b,x): 
	return 1.0 / (1.0 + np.exp(-b*x))

def sigmoid_gradient_array(b,x):
	return b*sigmoid_array(b,x)*(1.0-sigmoid_array(b,x))
	
def sigmoid_tanh_array(b,x):
	return np.tanh(b*x)
	#return (np.exp(b*x) - np.exp(-b*x))/(np.exp(b*x)+ np.exp(-b*x))
	
def sigmoid_tanh_gradient_array(b,x):
	return b*(1-(np.power(sigmoid_tanh_array(b,x),2)))
