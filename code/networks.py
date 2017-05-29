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

def sigmoid(z):
	return 1 / (1 + np.exp(-z))
	
def sigmoid_prime(z):
	return np.multiply(sigmoid(z), (1 - sigmoid(z)))
	
def tanh_prime(z):
	return 1 - np.square(np.tanh(z))
	
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
		# If direct connections, Create W_OH and W_OI
		if direct:
			self.W_OI = random_matrix([architecture[2], architecture[0]])
		self.W_O = random_matrix([architecture[2], architecture[1]])
		# If backward connections, create W_B matrix from output back to hidden
		if backward:
			self.W_B = random_matrix([architecture[1], architecture[2]])		
	
	def evaluate(self, I, training=False):
		""" Given sequence of vectors with dimensionality = input dimension, 
		return the hidden state and activations at each time step for the sequence
		"""
		W_I = self.W_I
		W_H = self.W_H
		if self.direct:
			W_OI = self.W_OI
		else:
			W_O = self.W_O
		if self.backward:
			W_B = self.W_B
		# All activations at time 0 are 0
		Z_H = [np.zeros((self.architecture[1],))]
		H = [np.zeros((self.architecture[1],))]
		Z_O = [np.zeros((self.architecture[1],))]
		O = [np.zeros((self.architecture[1],))]
		for	t in range(len(I))
			# Since H and O start with 0 vector, H[t] and O[t] = activations from previous time step
			# Hidden Activations
			if self.backward:
				Z_H.append(np.matmul(W_I, I[t]) + np.matmul(W_H, H[t]) + np.matmul(W_B, O[t]))
			else:
				Z_H.append(np.matmul(W_I, I[t]) + np.matmul(W_H, H[t]))
			H.append(np.tanh(Z_H[t]))
			# Output Activations
			if self.direct:
				Z_O.append(np.matmul(W_O, H[t+1]) + np.matmul(W_OI, I[t]))
			else:
				Z_O.append(np.matmul(W_O, H[t+1]))
			O.append(np.tanh(Z_O[t]))
		if training:
			return Z_H, H, Z_O, O
		else:
			return H, O
			
	def train(self, training_data, batch_size, num_epochs, learning_rate=0.01):
		""" RNN takes sequences of training data and learns to predict the next
		element of a sequence given the previous """
		#batches_per_epoch = int(num_samples / batch_size)
		echo_state = self.echo_state
		direct = self.direct
		backward = self.backward
		for epoch in range(num_epochs):
			np.random.shuffle(training_data)
			for I in training_data:
				Z_H, H, Z_O, O = self.evaluate(I, training=True)
				# Initialize update matrices
				if direct:
					d_W_OI = np.zeros(self.W_OI.shape)
				d_W_O = np.zeros(self.W_O.shape)
				if not echo_state:
					if backward:
						d_W_B = np.zeros(self.W_B.shape)
					d_W_H = np.zeros(self.W_H.shape)
					d_W_I = np.zeros(self.W_I.shape)
				""" BACKPROPAGATION THROUGH TIME """
				# Network always starts with START, begins predicting element indexed at I[1]
				labels = I[1 : len(I)]
				# For Z_H, H, Z_O, and O, we only care about times [1 : len(I)] at t=0 -> 0 vector
				delta_O = [np.multiply((O[t] - I[t]), sigmoid_prime(Z_O[t])) for t in range(1, len(I) + 1)]					
				delta_H = [[np.multiply(np.matmul(np.transpose(W_O), delta_O[t]), tanh_prime(Z_H[t])] for t in range(1, len(I) + 1)]
				for t in range(1, len(I)):
					if not backward:
						d_W_O += np.outer(delta_O[t], H[t])
						if direct:
							d_W_OI += np.outer(delta_O[t], I[t])
					# Backpropagate through time
					for tau in range(t): 
		
	
	
	def MSE(self, testing_data, testing_labels):
		# Cumulative MSE for predicting elements in sequence
		num_tests = len(testing_data)
		cumulative_error = 0
		for test in testing_data:
			states, outputs = self.evaluate(test)
			error = 0
			for i, output in enumerate(outputs):
				error += np.sum(np.square(output[i] - testing_labels[i]))
			cumulative_error += error / num_tests
		return cumulative_error
		
	def bptt(self, x, y):
		T = len(y)
		# Perform forward propagation
		s, o = self.evaluate(x)
		# We accumulate the gradients in these variables
		dLdU = np.zeros(self.W_I.shape)
		dLdV = np.zeros(self.W_H.shape)
		dLdW = np.zeros(self.W_I.shape)
		delta_o = o
		delta_o[np.arange(len(y)), y] -= 1.
		# For each output backwards...
		for t in np.arange(T)[::-1]:
			dLdV += np.outer(delta_o[t], s[t].T)
			# Initial delta calculation
			delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
			# Backpropagation through time (for at most self.bptt_truncate steps)
			for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
				# print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
				dLdW += np.outer(delta_t, s[bptt_step-1])              
				dLdU[:,x[bptt_step]] += delta_t
				# Update delta for next step
				delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
		return [dLdU, dLdV, dLdW]
	
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