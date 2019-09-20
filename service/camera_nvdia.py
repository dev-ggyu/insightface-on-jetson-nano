#!/usr/bin/python
import jetson.utils
from configs import configs

class Cam(object): 
	display = None
	camera = None
	isStop = False
	camera_idx = None
	width=None
	height=None
	# Initializing 
	def __init__(self,camera_idx=configs.cam_csi_index,width=configs.cam_csi_width,height=configs.cam_csi_height):
		self.camera_idx=camera_idx;
		self.width=width
		self.height=height
		print('[Camera] camera created...') 
  
	# Deleting (Calling destructor) 
	def __del__(self): 
		if(self.camera!=None):
			self.camera.Close()

	# Must be overridden
	def processs(self, image, width, height):
		raise NotImplementedError
		
	def set_stop(self):
		self.isStop=True
	
	def run(self):
		try:
			# create display window and camera
			self.display = jetson.utils.glDisplay()
			# create camera device
			print("[Camera] cam_idx:{} width:{} height:{}".format(self.camera_idx, self.width, self.height ))
			self.camera = jetson.utils.gstCamera(self.width, self.height, self.camera_idx)
			# open the camera for streaming
			self.camera.Open()
		except:
			print("[Camera] cam-open failed")
			return
			
		try:
			# capture frames until user exits
			while self.display.IsOpen() and self.isStop == False :
				image, width, height = self.camera.CaptureRGBA(zeroCopy=1)
				if(image!=None):
					ret=self.processs(image,width,height)
					self.display.RenderOnce(ret, width, height)
					self.display.SetTitle("{:s} | {:d}x{:d} | {:.0f} FPS".format("Camera Viewer", width, height, self.display.GetFPS()))
		finally:
			pass

