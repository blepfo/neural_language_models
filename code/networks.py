import numpy as np

def random_matrix(shape):
	# Creates matrix of values uniformly distributed on [-0.5, 0.5]
	return np.random.rand(shape[0], shape[1]) - 1

def reservoir_matrix(shape, alpha=0.98):
	# Creates matrix of values with spectral radius alpha 
	W_H0 = random_matrix(shape)
	eigenvalues = np.linalg.eigvals(W_H0)
	lambda_max = 0
	for eigenvalue in eigenvalues:
		current = np.linalg.norm(eigenvalue)
		if current > lambda_max:
			lambda_max = current
	return (alpha / lambda_max) * W_H0 

class RNN:
	""" Generic RNN with one hidden layer, optional input to output connections,
	optional backward output to hidden connections.
	If echo_state=True, learning only occurs in hidden to output connections and
	hidden to hidden weight matrix normalized to spectral radius 0.98
	"""
	def __init__(self, architecture, echo_state=False, alpha=0.98,
					direct=False, backward=False):
		if (len(architecture) != 3):
			raise Exception("Architecture must have 3 entries for I, H, O")
		self.architecture = architecture
		self.echo_state = echo_state
		self.direct = direct
		self.backward = backward
		# Initialize weight matrices
		self.W_I = random_matrix([architecture[1], architecture[0]])
		# If echo_state, create weight matrix with spectral radius alpha
		if echo_state:
			self.W_H = reservoir_matrix([architecture[1]] * 2)
		else:
			self.W_H = random_matrix([architecture[1]] * 2)
		# If direct connections, W_O needs to be large enough for [H;I]
		if direct:
			to_out_dim = architecture[0] + architecture[1]
		else:
			to_out_dim = architecture[1]
		self.W_O = random_matrix([architecture[2], to_out_dim])
		# If backward connections, create W_B matrix from output back to hidden
		if backward:
			self.W_B = random_matrix([architecture[1], architecture[2]])		
	
	def evaluate(self, sequence):
		""" Given sequence of vectors with dimensionality = input dimension, 
		return the hidden state and activations at each time step for the sequence
		"""
		W_I = self.W_I
		W_H = self.W_H
		W_O = self.W_O
		H = [np.zeros((self.architecture[1],))]
		if self.backward:
			O = [np.zeros((self.architecture[1],))]
			W_B = self.W_B
		else:
			O = []
		for	t, I in enumerate(sequence):
			if self.backward:
				H.append(np.tanh(
							np.matmul(W_I, I) +
							np.matmul(W_H, H[t]) +
							np.matmul(W_O, O[t])))
			else:
				H.append(np.tanh(
							np.matmul(W_I, I) +
							np.matmul(W_H, H[t])))
			if self.direct:
				O.append(np.tanh(
							np.matmul(W_O, np.append(I, H[t+1]))))
			else:
				O.append(np.tanh(
							np.matmul(W_O, H[t+1])))
		return H, O
			
	
	def train(training_data, batch_size, num_epochs):
		""" RNN takes sequences of training data and learns to predict the next
		element of a sequence given the previous """
		num_samples = len(training_data)
		batches_per_epoch = int(num_samples / batch_size)
		
	
# SRN Variations	
def SRN(architecture):
	return RNN(architecture, echo_state=False, direct=False, backward=False)
	
def SRN_D(architecture):
	return RNN(architecture, echo_state=False, direct=True, backward=False)

def SRN_B(architecture):
	return RNN(architecture, echo_state=False, direct=False, backward=True)
	
def SRN_DB(architecture):
	return RNN(architecture, echo_state=False, direct=True, backward=True)

# ESN Variations	
def ESN(architecture):
	return RNN(architecture, echo_state=True, direct=False, backward=False)
	
def ESN_D(architecture):
	return RNN(architecture, echo_state=True, direct=True, backward=False)

def ESN_DB(architecture):
	return RNN(architecture, echo_state=True, direct=False, backward=True)
	
def ESN_DB(architecture):
	return RNN(architecture, echo_state=True, direct=True, backward=True)