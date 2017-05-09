class RNN:
	""" Generic RNN with one hidden layer, optional input to output connections,
	optional backward output to hidden connections.
	If echo_state=True, learning only occurs in hidden to output connections and
	hidden to hidden weight matrix normalized to spectral radius 0.98
	"""
	def __init__(self, input_dim, hidden_dim, output_dim, echo_state=False
					direct=False, backward=False):
		pass
		
	def train(training_data, training_l)
	
	
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