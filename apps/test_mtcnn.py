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
from PIL import Image 
import mxnet as mx

from service import face_db #,face_track
from configs import configs
from service.camera_cv import Cam  
from utils.face_recognition import FaceRecognition as FR
from utils.face_detector_tpu import FaceDetector as FD
from utils.face_detector_mtcnn import FaceDetector as FD_MTCNN

class App(Cam):
	_detector=None
	_align=None
	_recognition=None
	last_time = time.monotonic()
	
	def signal_handler(self,sig,args=None):
		self.set_stop()

	def __init__(self, *args, **kwargs):
		signal.signal(signal.SIGINT, self.signal_handler)
		signal.signal(signal.SIGTERM, self.signal_handler)
			
		self.face_db = face_db.Model()
		self.face_db.load_faces(configs.tvm_faces,configs.tvm_faces_descriptions)
		self._detector = FD()
		mtcnn_path = './model/mtcnn'
		ctx = mx.cpu()
		self._align = FD_MTCNN(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=[0.6, 0.7, 0.8])
		self._recognition = FR()
		super(App, self).__init__(configs.cam_csi_index,configs.cam_csi_width,configs.cam_csi_height)		

		#self.engine = DetectionEngine(opt.model)
		#self.labels = self.load_labels(opt.labels)

	def processs(self, images, width, height):
		#cuda_np_array = jetson.utils.cudaToNumpy(image, width, height, 4)
		rgb=images[0]
		bgr=images[1]		
		target_image = Image.frombytes('RGB', (width,height), rgb, 'raw')
		start_time = time.monotonic()

		
		# Step1. Find and track face (frame ---> [Face_Tracker] ---> Faces Loactions)
		#self.face_tracker.process(frame)
		_faces = self._detector.get_faces(target_image, threshold=configs.tpu_threshold,keep_aspect_ratio=configs.tpu_keep_aspect_ratio, relative_coord=configs.tpu_relative_coord,top_k=configs.tpu_top_k)

		end_time = time.monotonic()
		
		text_lines = [
		'Inference: %.2f ms' %((end_time - start_time) * 1000),
		'FPS: %.2f fps' %(1.0/(end_time - self.last_time)),
		]
		#print(' '.join(text_lines))

		_similar_faces = []
		if  _faces==None or len(_faces) == 0:
			return bgr
		for _face in _faces:
			box = _face.bounding_box.flatten().astype("int")
			(startX, startY, endX, endY) = box
			#print('{} {} {} {}'.format(startX, startY, endX, endY))
			#cv2.imshow('Face Detect1',bgr[startY:endY,startX:endX])
			#cv2.waitKey(5)
			
			_align_face = self._align.get_align_faces(bgr[startY:endY,startX:endX])
			if(0<len(_align_face)):
			
				cv2.imshow('Face Detect1',_align_face)
				cv2.waitKey(5)
				_face_embedding = self._recognition.get_face_feature(_align_face)
				#_face_descriptions.append(_face_description)

				# Step3. For each face, check whether there are similar faces and if not save it to db.
				# Below naive and verbose implementation is to tutor you how this work
				_similar_faces = self.face_db.get_similar_faces(_face_embedding)
				#left:startX, top:startY, right:endX,bottom:endY			
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

