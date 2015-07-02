#!/bin/python

'''
Extrapolates what an .rbm file would look like if it had more hidden nodes
by appending new weights per row based on the mean and standard deviation of
that row
'''

import sys
import struct
import numpy

if len(sys.argv)<=4:
	print "python extrapolate <in> <out> <nvis> <nhid>"
	exit(0)

src=open(sys.argv[1],'rb')
target=open(sys.argv[2],'wb')
newvis=int(sys.argv[3])
newhid=int(sys.argv[4])

nv=struct.unpack("L",src.read(8))[0]
nh=struct.unpack("L",src.read(8))[0]

if nh<newhid:
	def process_row(x):
		x=list(x)
		out=[]
		mean=0
		rowcount=0
		for v in x:
			rowcount+=1
			mean+=v
			out.append(v)
		mean/=rowcount
		
		var=0
		for v in x:
			d=v-mean
			var+=d*d
		sigma=numpy.sqrt(var/(rowcount-1))
		
		while len(out)<newhid:
			v=0
			while v==0:
				v=numpy.random.normal(mean,sigma)
			out.append(v)
		
		return out
	
	w8s=numpy.array([
		process_row(struct.unpack('d',src.read(8))[0] for x in xrange(nh))
			for y in xrange(nv)
	])
elif nh>newhid:
	w8s=numpy.array([
		[struct.unpack('d',src.read(8))[0] for x in xrange(nh)]
			for y in xrange(nv)
	])
	
	def energy(x):
		E=0
		for v in x:
			E+=v*v
		try:
			return E[0]
		except TypeError:
			return E
	
	cols=numpy.hsplit(w8s,newhid)
	cols.sort(lambda a,b:int(numpy.sign(energy(a)-energy(b))))
	E=0
	energies=[]
	for col in cols:
		e=energy(col)
		energies.append(e)
		E+=e
	
	En=energy(cols[-1])
	while len(cols)>newhid:
		col=cols[0]
		E0=energy(col)
		for i in xrange(nhid):
			Ei=energies[i]
			cols[i]+=(En+E0-Ei)/E
		cols=cols[1:]
	w8s=numpy.hstack(cols)
else:
	w8s=numpy.array([
		[struct.unpack('d',src.read(8))[0] for x in xrange(nh)]
			for y in xrange(nv)
	])

if nv!=newvis:
	#Scale vertically
	nw8s=[]
	scale=float(nv)/newvis
	maxoff=numpy.ceil(scale/2)
	for i in range(newvis):
		j=i*scale
		off=0
		v=numpy.zeros(newhid)
		while off<maxoff:
			lo=int(numpy.floor(j-off))
			hi=int(numpy.ceil(j-off))
			v+=w8s[lo]*abs(j-lo)+w8s[hi]*abs(j-hi)
			off+=1
		nw8s.append(v/maxoff)
else:
	nw8s=w8s

target.write(struct.pack("L",newvis))
target.write(struct.pack("L",newhid))

for y in nw8s:
	for x in y:
		target.write(struct.pack('d',x))
