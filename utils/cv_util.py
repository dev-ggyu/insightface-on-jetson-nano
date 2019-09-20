import numpy as np
import re
from configs import configs
from edgetpu.detection.engine import DetectionEngine
#from edgetpu.utils import dataset_utils
import jetson.utils

#import face_recognition
import cv2

class Cv(object): 
	face_cascade=None
	eye_cascade=None

	# Initializing 
	def __init__(self):
		print("Loading Modle %s."%(configs.tpu_model))
		self.face_cascade = cv2.CascadeClassifier('/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
		self.eye_cascade = cv2.CascadeClassifier('/usr/share/OpenCV/haarcascades/haarcascade_eye.xml')
		print('[CV] CV Util created.')
		
	# Deleting (Calling destructor) 
	def __del__(self):
		print('[CV] CV Engine end of service')

	def get_faces(self,target_image,threshold,keep_aspect_ratio, relative_coord,top_k):
		_faces = self.face_cascade.detectMultiScale(target_image, 1.3, 5)

		for (x,y,w,h) in _faces:
			cv2.rectangle(target_image,(x,y),(x+w,y+h),(255,0,0),2)
			roi_gray = target_image[y:y+h, x:x+w]
			roi_color = target_image[y:y+h, x:x+w]
			'''
			eyes = self.eye_cascade.detectMultiScale(roi_gray)
			for (ex,ey,ew,eh) in eyes:
				cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
			'''
		return target_image
