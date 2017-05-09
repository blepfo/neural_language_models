""" Train and evaluate different RNN architectures using sentences 
generated by the CFG """

import networks

# Helper functions to load pickled data 
from load_help import load_encodings
from load_help import load_decodings
from load_help import load_training
from load_help import load_testing

def decode(encoding):
    return decodings[np.nonzero(encoding)[0][0]]

def encode_sentence(sentence):
    # Convert sentence to list of vector encodings
    return list(map(lambda word : encodings[word], sentence))

def decode_sentence(encoded_sentence):
    # Convert list of vector encodings to sentence
    return list(map(lambda encoding : decode(encoding), encoded_sentence))

	
decodings = load_decodings()
encodings = load_encodings()

training_data = load_training()
testing_data = load_testing()