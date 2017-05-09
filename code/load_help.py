""" Functions for loading dictionary/data from the data files """

import pickle

def load_data(path):
	# Generic function used to return unpickled data from a file 
	with open(path, 'rb') as file:
		return pickle.load(file)

def load_encodings():
	return load_data('../data/encodings.data')
	
def load_decodings():
	return load_data('../data/decodings.data')
		
def load_training():
	return load_data('../data/training.data')
	
def load_testing():
	return load_data('../data/testing.data')
