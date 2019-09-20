import cv2
import jetson.utils
import argparse
import time
import re
import imp
import os
import signal
import sys
import io
import numpy as np
import math
from PIL import Image 

from service import face_db #,face_track
from configs import configs
from service.camera_cv import Cam  
from utils.face_recognition import FaceRecognition as FR
from utils.face_detector import FaceDetector as FD

from utils import face_align

class App(Cam):
	_detector=None
	_recognition=None
	last_time = time.monotonic()
	scale=112
	
	def signal_handler(self,sig,args=None):
		self.set_stop()

	def __init__(self, *args, **kwargs):
		signal.signal(signal.SIGINT, self.signal_handler)
		signal.signal(signal.SIGTERM, self.signal_handler)
			
		self.face_db = face_db.Model()
		self.face_db.load_faces(configs.tvm_faces,configs.tvm_faces_descriptions)
		self._detector = FD()
		self._recognition = FR()
		super(App, self).__init__(configs.cam_csi_index,configs.cam_csi_width,configs.cam_csi_height)		
		
		#self.engine = DetectionEngine(opt.model)
		#self.labels = self.load_labels(opt.labels)

	def processs(self, images, width, height):
		#cuda_np_array = jetson.utils.cudaToNumpy(image, width, height, 4)
		rgb=images[0]
		bgr=images[1]
		#print(rgb.shape)
		x_scale=int(math.ceil(bgr.shape[1]/self.scale))
		y_scale=int(math.ceil(bgr.shape[0]/self.scale))
		img=cv2.resize(bgr, (112, 112))
		start_time = time.monotonic()
		#im = Image.fromarray(img)
		#im.save('{}.png'.format('test'))
		
		# Step1. Find and track face (frame ---> [Face_Tracker] ---> Faces Loactions)
		#self.face_tracker.process(frame)
		faces, landmarks  = self._detector.get_faces(img,0.1)

		end_time = time.monotonic()
		
		text_lines = [
		'Inference: %.2f ms' %((end_time - start_time) * 1000),
		'FPS: %.2f fps' %(1.0/(end_time - self.last_time)),
		]
		#print(' '.join(text_lines))
		if faces is not None:
			#print('find', faces.shape[0], 'faces')
			for i in range(faces.shape[0]):
# 				print('score', faces[i][4])
				box = faces[i].astype(np.int)
				if landmarks is not None:
					landmark5 = landmarks[i].astype(np.int)
					img = face_align.norm_crop(img,landmark5)
				
					_face_embedding = self._recognition.get_face_feature(img)
					_similar_faces = self.face_db.get_similar_faces(_face_embedding)
					#left:startX, top:startY, right:endX,bottom:endY			
					#(box[0]*x_scale, box[1]*y_scale), (box[2]*x_scale, box[3]*y_scale)
					startX,startY,endX,endY=box[0]*x_scale, box[1]*y_scale, box[2]*(x_scale), box[3]*(y_scale)
					cv2.rectangle(bgr, (startX, startY), (endX, endY),(255,0,0), 2)
					name='unknown'
					if 0 < len(_similar_faces):
						name=_similar_faces[0]
						
					cv2.rectangle(bgr, (startX, endY-15), (endX, endY),(255,0,0), cv2.FILLED)
					cv2.putText(bgr, name, (startX+3, endY-3), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255))
					
		return bgr


	def _viz_faces(self, faces_loc, frame):
		for _face_loc in faces_loc:
			x1 = int(_face_loc[0] * self.face_tracker.cam_w)
			y1 = int(_face_loc[1] * self.face_tracker.cam_h)
			x2 = int(_face_loc[2] * self.face_tracker.cam_w)
			y2 = int(_face_loc[3] * self.face_tracker.cam_h)
			cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
		cv2.imshow('faces', frame)
		cv2.waitKey(1)


if __name__ == '__main__':
	app = App()
	app.run()

