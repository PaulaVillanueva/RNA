import numpy as np

# Estamos poniendo beta=1, no tendríamos que poder variarlo? De ser así también hay que cambiarlo en los demás .py

def sigmoid_array(x): 
    return 1 / (1 + np.exp(-x))

def sigmoid_gradient_array(x):
	return sigmoid_array(x)*(1-sigmoid_array(x))
	
	# En la teórica de Segura la sigmoidea está con un 2*b
	
#def sigmoid_array(b,x): 
	#return 1 / (1 + np.exp(-2*b*x))

#def sigmoid_gradient_array(b,x):
	#return 2*b*sigmoid_array(b,x)*(1-sigmoid_array(b,x))
	
#def signoid_tanh_array(b,x):
	#return (np.exp(b*x) - np.exp(-b*x))/(np.exp(b*x)+ np.exp(-b*x))
	
#def sigmoid_tanh_gradient_array(b,x):
	#return b*(1-(signoid_tanh_array(b,x)))
