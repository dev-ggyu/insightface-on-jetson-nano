#!/usr/bin/python
import time
import jetson.utils
import cv2
from configs import configs
from service import face_db

class Cam(object): 
	display = None
	camera = None
	isStop = False
	camera_idx = None
	width=None
	height=None
	inQ=None
	outQ=None
	# Initializing 
	def __init__(self,camera_idx=configs.cam_csi_index,width=configs.cam_csi_width,height=configs.cam_csi_height):
		self.camera_idx=camera_idx;
		self.width=width
		self.height=height
		print('[Camera] camera created... index:{} width:{} height:{}'.format(self.camera_idx,self.width,self.height)) 
  
	# Deleting (Calling destructor) 
	def __del__(self): 
		if(self.camera!=None):
			self.camera.release()
			cv2.destroyAllWindows()

	# Must be overridden
	def processs(self, image, width, height):
		raise NotImplementedError
		
	def set_stop(self):
		self.isStop=True
	
	def gstreamer_pipeline (self, cam_index=0, capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=30, flip_method=2):   
		return ('nvarguscamerasrc sensor-id={} ! ' 
		'video/x-raw(memory:NVMM), '
		'width=(int){}, height=(int){}, '
		'format=(string)NV12, framerate=(fraction){}/1 ! '
		'nvvidconv flip-method={} ! '
		'video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! '
		'videoconvert ! '
		'video/x-raw, format=(string)BGR ! appsink'.format(cam_index,capture_width,capture_height,framerate,flip_method,display_width,display_height))
	'''
	nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, framerate=30/1, format=(string)NV12 ! nvvidconv flip-method=2 ! video/x-raw ! appsink name=mysink
'''
	def run(self,_inQ=None,_outQ=None):
		try:
			if(_inQ!=None and _outQ!=None):
				self.inQ=_inQ
				self.outQ=_outQ
			pipeline=self.gstreamer_pipeline(cam_index=self.camera_idx,capture_width=self.width,capture_height=self.height,display_width=self.width,display_height=self.height)
			print('[Camera] pipeline:{}'.format(pipeline))
			self.camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
		except Exception as e:
			print("[Camera] cam-open failed :{}".format(str(e)))
			return
			
		try:
			if self.camera.isOpened():
				cv2.namedWindow('Face Detect', cv2.WINDOW_AUTOSIZE)
				prevTime = 0 
				while cv2.getWindowProperty('Face Detect',0) >= 0 and self.isStop == False:
					ret, img = self.camera.read()
					curTime = time.time()
					#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
					rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
					image=self.processs((rgb,img), self.width,self.height)
					
					'''
					for face in _faces:
						x0, y0, x1, y1 = face.bounding_box.flatten().tolist()
						x, y, w, h = x0, y0, x1 - x0, y1 - y0
						x, y, w, h = int(x * 1280), int(y * 720), int(w * 1280), int(h * 720)
						cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
					'''
					sec = curTime - prevTime
					prevTime = curTime
					fps = 1/(sec)
					str_fps = "FPS : %0.1f" % fps
					cv2.putText(image, str_fps, (5, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255),1)
					cv2.imshow('Face Detect',image)
					keyCode = cv2.waitKey(30) & 0xff
					# Stop the program on the ESC key
					if keyCode == 27:
						break
		except Exception as e:
			print("[Camera] cam error :{}".format(str(e)))

		finally:
			pass

