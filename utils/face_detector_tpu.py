import numpy as np
import re
from configs import configs
from edgetpu.detection.engine import DetectionEngine
#from edgetpu.utils import dataset_utils
import jetson.utils

#import face_recognition
#import cv2

class FaceDetector(object): 
	engine=None
	labels=None

	# Initializing 
	def __init__(self):
		print("Loading Modle %s."%(configs.tpu_model))
		self.engine = DetectionEngine(configs.tpu_model)
		if(configs.tpu_labels!=None):
			print("Loading Labels %s."%(configs.tpu_labels))
			self.labels = self.load_labels(configs.tpu_labels)
		print('[TPU] TPU Util created.')
		
	# Deleting (Calling destructor) 
	def __del__(self):
		print('[TPU] TPU Engine end of service')

	def load_labels(self, path):
		p = re.compile(r'\s*(\d+)(.+)')
		with open(path, 'r', encoding='utf-8') as f:
			lines = (p.match(line).groups() for line in f.readlines())
			return {int(num): text.strip() for num, text in lines}
		
	def get_faces(self,target_image,threshold,keep_aspect_ratio, relative_coord,top_k):
		_faces = self.engine.DetectWithImage(target_image, threshold=threshold,keep_aspect_ratio=keep_aspect_ratio, relative_coord=relative_coord,top_k=top_k)
		'''
		for face in _faces:
			x0, y0, x1, y1 = face.bounding_box.flatten().tolist()
			x, y, w, h = x0, y0, x1 - x0, y1 - y0
			#x, y, w, h = int(x * width), int(y * height), int(w * width), int(h * height)

			percent = int(100 * face.score)
			print('percent:{} x:{} y:{} w:{} h:{}'.format(percent,x,y,w,h))
		'''
		return _faces
