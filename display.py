import Image
import numpy as np
import struct
import sys

import rbm

if len(sys.argv)>1 and sys.argv[1]!="order":
	f=open(sys.argv[1],'rb')
else:
	f=open("voice.rbm",'rb')

model=rbm.load(f)
f.close()

nhid=model.nhid
nvis=model.nvis

print model.nvis,"x",model.nhid

def isexotic(f):
	return struct.unpack("L",struct.pack('d',f))[0]&(((1<<11)-1)<<52)==0x7ff<<52

inf=float('inf')
m=inf
M=-inf
mean=0
for y in model.w8s:
	for x in y:
		if not isexotic(x):
			if x<m:
				m=x
			elif x>M:
				M=x
			mean+=x

print "Min: %.6f Max: %.6f Mean: %.12f"%(m,M,mean/(nvis*nhid))

def normalize(x):
	if isexotic(x):
		return x
	
	if m!=M:
		x=10*(x-m)/(M-m)-5
	if x>40:
		return 255
	if x<-40:
		return 0
	
	return int(255/(1+np.exp(-x)))

normalize=np.frompyfunc(normalize,1,1)

model.w8s=normalize(model.w8s)

m=inf
M=-inf
mean=0
for x in model.bias:
	if not isexotic(x):
		if x<m:
			m=x
		elif x>M:
			M=x
		mean+=x
print "Bias Min: %.6f Max: %.6f Mean: %.6f"%(m,M,mean/nhid)
if np.isfinite(m) or np.isfinite(M):
	model.bias=normalize(model.bias)

if len(sys.argv)>1 and sys.argv[1]=="order":
	#Rearrange the data in order of column
	def energy(x):
		E=0
		for v in x:
			E+=v*v
		
		return float(E)
	
	p=np.argsort([energy(col) for col in np.hsplit(model.w8s,nhid)])
	
	cols=np.hsplit(model.w8s.T[p].T,nhid)
	
	cols=np.vsplit(np.hstack(cols),nvis)
	
	cols.insert(0,np.tile(255,nhid))
	cols.insert(0,model.bias[p])
	
	img=cols
else:
	img=np.vsplit(model.w8s,nvis)
	img.insert(0,np.tile(255,(1,nhid)))
	img.insert(0,model.bias)

img=np.vstack(img)

bad=[]
shape=img.shape
for y in xrange(shape[0]):
	for x in xrange(shape[1]):
		if isexotic(img[y,x]):
			bad.append((x,y))

if len(bad):
	NEG_INF_COLOR=(255,255,0)
	POS_INF_COLOR=(255,0,255)
	
	POS_NAN_COLOR=(0,255,0)
	POS_NAN_COLOR=(255,0,0)
	
	img=np.dstack((img,img,img))
	for x,y in bad:
		v=img[y,x,1]
		print v,"at",(x,y)
		if np.isinf(v):
			if v<0:
				img[y,x]=NEG_INF_COLOR
			else:
				img[y,x]=POS_INF_COLOR
		else:
			if v<0:
				img[y,x]=NEG_NAN_COLOR
			else:
				img[y,x]=POS_NAN_COLOR

Image.fromarray(img.astype('uint8')).save(open("w8s.bmp","wb"))
