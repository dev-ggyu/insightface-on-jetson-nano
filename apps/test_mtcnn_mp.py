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
import mxnet as mx
import numpy as np
from PIL import Image 

from service import face_db #,face_track
from configs import configs
from service.camera_cv import Cam  
from utils.face_recognition import FaceRecognition as FR
from utils.face_detector_tpu import FaceDetector as FD
from utils.face_detector_mtcnn import FaceDetector as FD_MTCNN

from multiprocessing import Process, Queue


class App(Cam):
	_detector=None
	recognition_faces=None
	concat_index=0;
	last_time = time.monotonic()
	def signal_handler(self,sig,args=None):
		self.set_stop()

	def __init__(self, *args, **kwargs):
		signal.signal(signal.SIGINT, self.signal_handler)
		signal.signal(signal.SIGTERM, self.signal_handler)
			
		self._detector = FD()
		self.recognition_faces= np.zeros((configs.cam_csi_height, 112, 3), np.uint8)
		super(App, self).__init__(configs.cam_csi_index,configs.cam_csi_width,configs.cam_csi_height)		
		
		#self.engine = DetectionEngine(opt.model)
		#self.labels = self.load_labels(opt.labels)

	def processs(self, images, width, height):
		rgb=images[0]
		bgr=images[1]		
		target_image = Image.frombytes('RGB', (width,height), rgb, 'raw')
		start_time = time.monotonic()
		# Step1. Find and track face (frame ---> [Face_Tracker] ---> Faces Loactions)
		#cv2.imshow('Face Detect1',rgb)
		#cv2.waitKey(5)
		_faces = self._detector.get_faces(target_image, threshold=configs.tpu_threshold,keep_aspect_ratio=configs.tpu_keep_aspect_ratio, relative_coord=configs.tpu_relative_coord,top_k=configs.tpu_top_k)
		
		
		_similar_faces = []
		_isInferenceTime=False
		if  _faces!=None and 0 < len(_faces):
			if(1000<(start_time-self.last_time) * 1000):
				_isInferenceTime=True
				self.last_time = time.monotonic()

			for _face in _faces:
				box = _face.bounding_box.flatten().astype("int")
				(startX, startY, endX, endY) = box
				if(_isInferenceTime==True):
					self.inQ.put((bgr,box))
				cv2.rectangle(bgr, (startX, startY), (endX, endY),(255,0,0), 2)
				

		if(0<outQ.qsize()):
			roc_image,roc_box,name =outQ.get()
			(left, top, right, bottom) = roc_box

			#top down merge #vertical
			if(self.concat_index%6==0): #122 =6  336,2
				self.concat_index=0
			
			h=(112*self.concat_index)
			#print('{}{}'.format(roc_image.shape,self.recognition_faces.shape))
			self.recognition_faces[h:h+112,:,:] = cv2.resize(roc_image[top:bottom,left:right], (112, 112))

			cv2.rectangle(self.recognition_faces, (0, h+112-15), (112, h+112),(255,0,0), cv2.FILLED)
			cv2.putText(self.recognition_faces, name, (0+3, h+112-3), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255))
			self.concat_index=self.concat_index+1
			#self.recognition_faces = np.vstack((self.recognition_faces,roc_image) )

			#left right merge #horizonal
			#bgr = np.concatenate((bgr,self.recognition_faces), axis=1)# np.hstack(bgr,self.recognition_faces)
			#cv2.imshow('Face Detect11111',self.recognition_faces)
			#cv2.waitKey(5)
			#print('end q data')
			
		bgr = np.concatenate((bgr,self.recognition_faces), axis=1)
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


def mp_recorgnition(inQ,outQ):
	from utils.face_detector_mtcnn import FaceDetector as FD_MTCNN
	_face_db = face_db.Model()
	_face_db.load_faces(configs.tvm_faces,configs.tvm_faces_descriptions)
	mtcnn_path = './model/mtcnn'
	_face_align = FD_MTCNN(model_folder=mtcnn_path, ctx=mx.cpu(), num_worker=1, accurate_landmark = True, threshold=[0.6, 0.7, 0.8])
	_recognition = FR()
	while True:
		data = inQ.get()
		bgr=data[0]
		box=data[1]
		(startX, startY, endX, endY) = box
		#cv2.imshow('Face Detect1',bgr[startY:endY,startX:endX])
		#cv2.waitKey(5)
		_align_img = _face_align.get_align_faces(bgr[startY:endY,startX:endX])
		if(0<len(_align_img)):
			_face_embedding = _recognition.get_face_feature(_align_img)

			_similar_faces = _face_db.get_similar_faces(_face_embedding)
			name='unknown'
			if 0 < len(_similar_faces):
				name=_similar_faces[0]
			outQ.put((bgr,box,name))

def mp_cam(inQ,outQ):
	app = App()
	app.run(inQ,outQ)



_running=True
def _signal_handler(self,sig):
	global _running
	_running =False

if __name__ == '__main__':
	#signal.signal(signal.SIGINT, _signal_handler)
	#signal.signal(signal.SIGTERM, _signal_handler)
	
	inQ = Queue()
	outQ = Queue()
	process_one = Process(target=mp_cam, args=(inQ, outQ))
	process_two = Process(target=mp_recorgnition, args=(inQ, outQ))
	process_one.start()
	process_two.start()
 	
	while _running:
		time.sleep(3)
		continue
		

	#q.close()
	#q.join_thread()
	print('wait stopping first process')
	process_one.terminate()
	print('wait stopping second process')
	process_two.terminate()
