import sys
import os
import numpy as np
import struct

import ai
import config
import audio

model=config.model
rate=0.0001
epochs=1

phi=(5**0.5-1)/2

todo=sys.argv[1]

def isexotic(x):
	return np.isnan(x) or np.isinf(x)

while todo not in {"reuse","restart"}:
	todo=raw_input("Reuse or restart? ").lower()

if todo=="reuse":
	model.load(open("voice.rbm","rb"))
else:
	if os.path.isfile("voice.rbm"):
		if raw_input("Are you sure? ").lower()!="yes":
			print "Quitting"
			exit(0)
	model.randomize(0.00001)

trainer=model.CDTrainer(rate,*config.sizes)

data=np.empty(config.ndat,dtype='f')
avgerr=0.25
def train(fn):
	global data,avgerr
	
	counter=1
	
	f=audio.open(fn)
	
	print "Training on",fn
	
	audio.read(f,data,config.ndat)
	trainer.train(model,data,epochs)
	
	data[:-config.window]=data[config.window:]
	config.normalize(data)
	
	while audio.read(f,data[-config.window:],config.window):
		config.normalize(data[-config.window:])
		
		trainer.train(model,data,epochs)
		error=ai.mse(data,trainer.negv)
		
		if isexotic(error):
			print "Bad error",error,"after",counter,"samples, aborting"
			print data,trainer.negv
			exit(0)
		
		avgerr=(phi*error+(1-phi)*avgerr)
		
		if counter%5000==0:
			model.dump(open("voice.rbm","wb"))
			print "err = {:.8f}, rate = {}, epochs = {}".format(
				avgerr,rate,epochs
			)
		
		counter+=1
	
	print "Error = {:.8f}".format(avgerr)
	
	return f.getnframes()/f.getframerate()

print "Beginning training"
for root,dirs,files in os.walk("files"):
	for f in files:
		train(os.path.join(root,f))
