""" Generate sentences according to modified CFG specified in 
Learning Gramatical Structure with Echo State Networks
Tong et al, 2007"""

"""
S -> SNP SVP . | PNP PVP .
SNP -> PropN | SN | SN SRC
PNP -> PN | PN PRC
SRC -> who SNP SV | who SVP
PRC -> who PNP PV | who PVP
VP -> SVP | PVP
SVP -> SNDOV | SDOV SNP
PVP -> PNDOV | PDOV PNP
SN -> boy | girl | cat | dog
PN -> boys | girls | cats | dogs
propN -> john | mary
SNDOV -> walks | lives | sees | hears
SDOV -> chases | feeds | sees | hears
PNDOV -> walk | live | see | hear
PDOV -> chase | feed | see | hear
"""

import random
from math import floor
from itertools import chain

def plaintext(sentence_list):
	return " ".join(sentence_list[1:-1])

def choose_random(options):
    rand_num = floor(random.random() * len(options))
    return options[rand_num]

def flatten_list(start_list):
    flat = []
    for item in start_list:
        if type(item) == type([]):
            flat.extend(flatten_list(item))
        else:
            flat.append(item)
    return flat

def generate_S():
	rand_num = random.random()
	if rand_num < (1/2):
		return flatten_list(["START", generate_SNP(), generate_SVP(), "STOP"])
	else:
		return flatten_list(["START", generate_PNP(), generate_PVP(), "STOP"])

def generate_NP():
	rand_num = random.random()
	if rand_num < (1/2):
		return generate_SNP()
	else:
		return generate_PNP()

def generate_SNP():
	rand_num = random.random()
	if rand_num < (1/3):
		return generate_propN()
	elif (1/3) <= rand_num < (2/3):
		return generate_SN()
	else:
		return [generate_SN(), generate_SRC()]
	
def generate_PNP():
	rand_num = random.random()
	if rand_num < (1/2):
		return generate_PN()
	else:
		return [generate_PN(), generate_PRC()]

def generate_SVP():
	rand_num = random.random()
	if rand_num < (1/2):
		return generate_SNDOV()
	else:
		return [generate_SDOV(), generate_SNP()]
		
def generate_PVP():
	rand_num = random.random()
	if rand_num < (1/2):
		return generate_PNDOV()
	else:
		return [generate_PDOV(), generate_PNP()]
		
def generate_SRC():
	rand_num = random.random()
	if rand_num < (1/2):
		return ["who", generate_SNP(), generate_SDOV()]
	else:
		return ["who", generate_SVP()]
		
def generate_PRC():
	rand_num = random.random()
	if rand_num < (1/2):
		return ["who", generate_PNP(), generate_PDOV()]
	else:
		return ["who", generate_PVP()]
	
def generate_SN():
	SN = ['boy', 'girl', 'cat', 'dog']
	return choose_random(SN)
	
def generate_PN():
	PN = ['boys', 'girls', 'cats', 'dogs']
	return choose_random(PN)
	
def generate_propN():
	propN = ['john', 'mary']
	return choose_random(propN)

def generate_SNDOV():
	PNDOV = ['walks', 'lives', 'sees', 'hears']
	return choose_random(PNDOV)
	
def generate_SDOV():
	PDOV = ['chases', 'feeds', 'sees', 'hears']
	return choose_random(PDOV)
	
def generate_PNDOV():
	SNDOV = ['walk', 'live', 'see', 'hear']
	return choose_random(SNDOV)
	
def generate_PDOV():
	SDOV = ['chase', 'feed', 'see', 'hear']
	return choose_random(SDOV)
	
def create_corpus(num_sentences):
	corpus = [] 
	for _ in range(num_sentences):
		corpus.append(generate_S())
	return corpus
	
	
	
	
	
	
	
	
	
	
	