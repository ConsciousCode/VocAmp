import struct
import numpy as np

class RBM:
	def __init__(self):
		self.w8s=None
		self.bias=None
	
	nvis=property(lambda self:self.w8s.shape[0])
	nhid=property(lambda self:self.w8s.shape[1])
	
	def dump(self,f):
		f.write(struct.pack("L",self.nvis))
		f.write(struct.pack("L",self.nhid))
		
		f.write(b''.join(struct.pack('d',v) for v in self.bias))
		
		f.write(b''.join(struct.pack('d',v) for v in self.w8s.flatten()))

def load(f):
	nv=struct.unpack("L",f.read(8))[0]
	nh=struct.unpack("L",f.read(8))[0]
	
	rbm=RBM()
	
	rbm.bias=np.array([struct.unpack('d',f.read(8))[0] for x in xrange(nh)])
	
	rbm.w8s=np.array([
		[struct.unpack('d',f.read(8))[0] for x in xrange(nh)] for x in xrange(nv)
	])
	
	return rbm

def dump(rbm,f):
	rbm.dump(f)
