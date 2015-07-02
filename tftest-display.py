import PIL.Image as Image
import numpy as np

orig=np.fromstring(open("orig.raw","rb").read(),dtype='f').reshape((16,16))
out=np.fromstring(open("out.raw","rb").read(),dtype='f').reshape((10,10))

print orig.max(),out.max()

orig=orig[3:-3,3:-3]

Image.fromarray((orig*255).astype('uint8')).save(open("orig.bmp","wb"))
Image.fromarray((out*255).astype('uint8')).save(open("out.bmp","wb"))
