
import numpy as np
import struct

def fspecialGauss(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()

def readFlowFile( filename ):
	# readFlowFile read a flow file FILENAME into 2-band image IMG

	TAG_FLOAT = 202021.25  # check for this when READING the file
	TAG_FLOAT = float(TAG_FLOAT)

	# sanity check
	if (len(filename) == 0):
		print 'Error: readFlowFile: empty filename'

	fid = open(filename, 'rb')
	buf = fid.read()

	index = 0
	tag = struct.unpack_from('<f' , buf , index)
	index += struct.calcsize('<f')

	width, height = struct.unpack_from('<ii' , buf , index)
	index += struct.calcsize('<ii')

	# sanity check

	if (not tag[0] == TAG_FLOAT):
		print 'Error: readFlowFile(%s): wrong tag (possibly due to big-endian machine?)' %(filename)
	if (width < 1 | width > 99999):
		print 'Error: readFlowFile(%s): illegal width %d' %(filename, width)
	if (height < 1 | height > 99999):
		print 'Error: readFlowFile(%s): illegal height %d' %(filename, height)

	nBands = 2

	tmp = struct.unpack_from('<%df'%(width*height*2) ,buf, index)
	index += struct.calcsize('<%df'%(width*height*2))
	tmp = np.array(tmp)

	tmp = tmp.reshape(height, width*nBands)
	im = np.zeros([height,width,2])
	im[:,:,0] = tmp[:,np.arange(width)*nBands]
	im[:,:,1] = tmp[:,np.arange(width)*nBands+1]

	return im

def writeFlowFile( img, filename ):
	TAG_STRING = 'PIEH'         # use this when WRITING the file

	# sanity check
	if (len(filename) == 0):
		print 'Error: writeFlowFile: empty filename'

	height, width, nBands = img.shape

	if (not nBands == 2):
		print 'Error: writeFlowFile: image must have two bands'

	fid = open(filename, 'wb')

	# write the header
	fid.write(TAG_STRING)
	fid.write(struct.pack('<ii',width,height))

	# arrange into matrix form
	tmp = np.zeros( [height, width*nBands, ], 'single' )

	tmp[:, np.arange(width)*nBands] = img[:,:,0]
	tmp[:, np.arange(width)*nBands+1] = img[:,:,1]

	fid.write(tmp.tobytes())

def flow_aae(f1, f2, mask=None):
	print f1.shape
	tmp = (np.sum(np.multiply(f1,f2), 2) + 1) / np.sqrt((np.sum(f1**2, 2) + 1) * (np.sum(f2**2, 2) + 1))
	tmp[tmp>1.0] = 1.0
	tmp[tmp<-1.0] = -1.0
	aae = np.arccos(tmp)
	try:
		aae = aae[mask]
	except:
		aae = aae

	sae = np.std(np.real(aae)) * (180 / np.pi)
	aae = np.mean(np.real(aae)) * (180 / np.pi)

	return aae, sae

def flowAngErr(tu, tv, u, v, bord):
	h, w = tu.shape
	smallflow = 0.0

	stu = tu[bord:h-bord,bord:w-bord]
	stv = tv[bord:h-bord,bord:w-bord]
	su = u[bord:h-bord,bord:w-bord]
	sv = v[bord:h-bord,bord:w-bord]

	# ignore a pixel if both u and v are zero
	# ind2=find((stu(:).*stv(:)|sv(:).*su(:))~=0);
	ind2 = (np.abs(stu)>smallflow) | (np.abs(stv>smallflow))

	n = 1.0 / np.sqrt(su[ind2]**2+sv[ind2]**2+1)
	un = su[ind2]*n
	vn = sv[ind2]*n
	tn = 1.0 / np.sqrt(stu[ind2]**2+stv[ind2]**2+1)
	tun = stu[ind2]*tn
	tvn = stv[ind2]*tn

	tmp = np.multiply(un,tun) + np.multiply(vn,tvn) + np.multiply(n,tn)
	tmp[tmp>1.0] = 1.0
	ang = np.arccos(tmp)
	mang = np.mean(ang)
	mang = mang*180/np.pi


	stdang = np.std(ang*180/np.pi)

	epe = np.sqrt((stu-su)**2 + (stv-sv)**2)
	epe = epe[ind2]
	mepe = np.mean(epe)
	return mang, stdang, mepe

def flowToColor(flow,maxFlow=-1):
	# flowToColor(flow, maxFlow) flowToColor color codes flow field, normalize
	# based on specified value,

	# flowToColor(flow) flowToColor color codes flow field, normalize
	# based on maximum flow present otherwise

	#  According to the c++ source code of Daniel Scharstein
	#  Contact: schar@middlebury.edu
	eps = 2.2204e-16
	UNKNOWN_FLOW_THRESH = 1.0e9
	UNKNOWN_FLOW = 1.0e10

	height, width, nBands = flow.shape

	if (not nBands==2):
		print 'Error: flowToColor: image must have two bands\n'

	u = flow[:,:,0]
	v = flow[:,:,1]

	maxu = -999.0
	maxv = -999.0

	minu = 999.0
	minv = 999.0
	maxrad = -1.0

	# fix unknown flow
	idx_unknown = (np.abs(u)>UNKNOWN_FLOW_THRESH) | (np.abs(v)>UNKNOWN_FLOW_THRESH)
	u[ idx_unknown ] = 0
	v[ idx_unknown ] = 0

	maxu = max(maxu, np.max(u))
	minu = min(minu, np.min(u))

	maxv = max(maxv, np.max(v))
	minv = min(minv, np.min(v))

	rad = np.sqrt(u**2+v**2)
	maxrad = max(maxrad, np.max(rad))

	print 'max flow: %.4f flow range: u = %.3f .. %.3f; v = %.3f .. %.3f\n' %(maxrad, minu, maxu, minv, maxv)

	if (not maxFlow==-1):
		if (maxFlow > 0):
			maxrad = maxFlow

	u = u/(maxrad+eps)
	v = v/(maxrad+eps)

	img = computeColor(u, v)
	idx_unknown = np.reshape(idx_unknown,[height,width,1])
	idx_unknown = np.tile(idx_unknown,[1,1,3])
	img[idx_unknown] = 0

	return img

def computeColor(u,v):
	nanIdx = (u==np.nan) | (v==np.nan)
	u[nanIdx] = 0
	v[nanIdx] = 0

	colorwheel = makeColorwheel()
	ncols = colorwheel.shape[0]

	rad = np.sqrt(u**2+v**2)

	a = np.arctan2(-v, -u)/np.pi

	fk = (a+1) /2 * (ncols-1) + 1  # -1~1 maped to 1~ncols

	k0 = np.floor(fk)                 # 1, 2, ..., ncols

	k1 = k0 + 1
	k1[k1==ncols+1] = 1

	f = fk - k0
	img = np.zeros([u.shape[0],u.shape[1],3])
	for i in range(colorwheel.shape[1]):
		tmp = colorwheel[:,i]
		col0 = tmp[k0.astype('int')-1]/255.0
		col1 = tmp[k1.astype('int')-1]/255.0
		col = (1-f)*col0 + f*col1

		idx = rad <= 1
		col[idx] = 1-rad[idx]*(1-col[idx])			# increase saturation with radius

		col[~idx] = col[~idx]*0.75			# out of range

		img[:,:,i] = np.floor(255.0*col).astype('uint8')

	return img

def makeColorwheel():
	RY = 15
	YG = 6
	GC = 4
	CB = 11
	BM = 13
	MR = 6

	ncols = RY + YG + GC + CB + BM + MR

	colorwheel = np.zeros([ncols, 3]) # r g b

	col = 0
	# RY
	colorwheel[0:RY, 0] = 255.0
	colorwheel[0:RY, 1] = np.floor(np.multiply(255.0/RY,range(RY)))
	col = col+RY

	# YG
	colorwheel[col+np.arange(0,YG), 0] = 255.0 - np.floor(np.multiply(255.0/YG,range(YG)))
	colorwheel[col+np.arange(0,YG), 1] = 255.0
	col = col+YG

	# GC
	colorwheel[col+np.arange(0,GC), 1] = 255.0
	colorwheel[col+np.arange(0,GC), 2] = np.floor(np.multiply(255.0/GC,range(GC)))
	col = col+GC

	# CB
	colorwheel[col+np.arange(0,CB), 1] = 255.0 - np.floor(np.multiply(255.0/CB,range(CB)))
	colorwheel[col+np.arange(0,CB), 2] = 255.0
	col = col+CB

	# BM
	colorwheel[col+np.arange(0,BM), 2] = 255.0
	colorwheel[col+np.arange(0,BM), 0] = np.floor(np.multiply(255.0/BM,range(BM)))
	col = col+BM

	# MR
	colorwheel[col+np.arange(0,MR), 2] = 255.0 - np.floor(np.multiply(255.0/MR,range(MR)))
	colorwheel[col+np.arange(0,MR), 0] = 255.0

	return colorwheel
