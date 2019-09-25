import mxnet as mx
from mxnet import ndarray as nd
import numpy as np
import nnvm.compiler
import nnvm.testing
import tvm
import cv2
from tvm.contrib import graph_runtime
from configs import configs

class FaceRecognition():
	dtype = None
	ctx = None 
	loaded_json = None
	loaded_lib = None
	loaded_params = None
	data_shape = None
	
	def __init__(self):
		print('[TVM] selection inference deivce.')
		self.ctx = tvm.gpu(0) 
		self.dtype = configs.tvm_dtype
		print('[TVM] loading graph.')
		self.loaded_json = open(configs.tvm_graph_json).read()
		print('[TVM] loading tvm lib.')
		self.loaded_lib = tvm.module.load(configs.tvm_lib)
		print('[TVM] loading param.')
		self.loaded_params = bytearray(open(configs.tvm_params, "rb").read())
		self.data_shape = configs.tvm_data_shape
		print('[TVM] creation runtime graph.')
		self.module = graph_runtime.create(self.loaded_json, self.loaded_lib, self.ctx)
		print('[TVM] apply to param.')
		self.module.load_params(self.loaded_params)
		print('[TVM] created.')
		
	def __del__(self): 
		print('[TVM] destoryed.')
		
	def get_face_feature(self,target_image,threshold=configs.tvm_face_similarity_threshold):
		im = cv2.resize(target_image, (self.data_shape[2], self.data_shape[3]))
		'''
		im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
		for i in range(3):
			im_tensor[0, i, :, :] = im[:, :, 2 - i]
		input_face1 = tvm.nd.array(im_tensor.astype(self.dtype))
		self.module.run(data=input_face1)
		v1 = self.module.get_output(0)
		if(v1!=None):
			v1=v1.asnumpy()[0]
			#print('[TVM] score:{}'.format(v1))
		'''
		'''
		rs_data = cv2.resize(target_image, (self.data_shape[2], self.data_shape[3]))
		
		data = np.zeros((1, 3, rs_data.shape[0], rs_data.shape[1]))
		for i in range(3):
			data[0, i, :, :] = rs_data[:, :, 2 - i]

		
		#face_1 = np.transpose(face_1, (2, 0, 1))
		#face_1 = np.expand_dims(face_1, axis=0)
		data = tvm.nd.array(rs_data.astype(self.dtype))
		#cv2.imshow('Face Detect1',data)
		#cv2.waitKey(5)	
		'''
		im_data = np.transpose(im, (2,0,1)) #maybe don't needed
		expand_data = np.expand_dims(im_data, axis=0)
		data = tvm.nd.array(expand_data.astype(self.dtype))
		
		
		self.module.run(data=data)
		v1 = self.module.get_output(0)
		if(v1!=None):
			#print('[TVM] score:{}'.format(v1.asnumpy()[1]))
			v1=v1.asnumpy()[0]
		
		return v1

	def compute_sim(self, img1, img2):
		emb1 = self.get_embedding(img1).flatten()
		emb2 = self.get_embedding(img2).flatten()
		from numpy.linalg import norm
		sim = np.dot(emb1, emb2)/(norm(emb1)*norm(emb2))
		return sim


