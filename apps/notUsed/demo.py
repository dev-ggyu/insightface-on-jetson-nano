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

from service import face_db #,face_track
from configs import configs
from service.camera_nvdia import Cam 
from utils.face_recognition import FaceRecogntion as FR
from utils.face_detector import FaceDetector as FD

class App(Cam):
	tpu=None
	Tvm=None
	last_time = time.monotonic()
	
	def signal_handler(self,sig):
		super.set_stop()

	def __init__(self, *args, **kwargs):
		signal.signal(signal.SIGINT, self.signal_handler)
		signal.signal(signal.SIGTERM, self.signal_handler)
			
		self.face_db = face_db.Model()
		self.face_db.load_faces(configs.tvm_faces,configs.tvm_faces_descriptions)
		self.tpu = Tpu()
		self.tvm = Tvm()
		super(App, self).__init__(configs.cam_csi_index,configs.cam_csi_width,configs.cam_csi_height)		
		'''
		#self.face_tracker = face_track_server.FaceTrackServer()
		self.face_describer = face_describer_server.FDServer(
			model_fp=configs.face_describer_model_fp,
			input_tensor_names=configs.face_describer_input_tensor_names,
			output_tensor_names=configs.face_describer_output_tensor_names,
			device=configs.face_describer_device)
		'''	
		
		#self.engine = DetectionEngine(opt.model)
		#self.labels = self.load_labels(opt.labels)

	def processs(self, images, width, height):
		cuda_np_array = jetson.utils.cudaToNumpy(images, width, height, 4)		
		target_image = Image.frombytes('RGB', (width,height), cuda_np_array, 'raw')#Image.fromarray(cuda_np_array)#self.get_image(image,width,height,4) #self.array2PIL(cuda_np_array,(width,height))#Image.frombytes('RGB', (width,height), cuda_np_array, 'RGB')
		start_time = time.monotonic()

		

		
		# Step1. Find and track face (frame ---> [Face_Tracker] ---> Faces Loactions)
		#self.face_tracker.process(frame)
		_faces = self.tpu.get_faces(target_image, threshold=configs.tpu_threshold,keep_aspect_ratio=configs.tpu_keep_aspect_ratio, relative_coord=configs.tpu_relative_coord,top_k=configs.tpu_top_k)

		# Uncomment below to visualize face
		# _faces_loc = self.face_tracker.get_faces_loc()
		# self._viz_faces(_faces_loc, frame)
		
		end_time = time.monotonic()
		
		text_lines = [
		'Inference: %.2f ms' %((end_time - start_time) * 1000),
		'FPS: %.2f fps' %(1.0/(end_time - self.last_time)),
		]
		#print(' '.join(text_lines))

		_similar_faces = []
		if  _faces!=None or 0 < len(_faces):
			for _face in _faces:
				box = _face.bounding_box.flatten().astype("int")
				(startX, startY, endX, endY) = box
				#print('{} {} {} {}'.format(startX, startY, endX, endY))
				#cv2.imshow('Face Detect',image[startY:endY,startX:endX])
				#cv2.waitKey(5)
				_face_embedding = self.tvm.get_face_feature(cuda_np_array[startY:endY,startX:endX])
				#_face_descriptions.append(_face_description)

				# Step3. For each face, check whether there are similar faces and if not save it to db.
				# Below naive and verbose implementation is to tutor you how this work
				_similar_faces = self.face_db.get_similar_faces(_face_embedding)
				cv2.rectangle(cuda_np_array, (startX, endY), (startY, endX),(255,0,0), 2)
				name='unknown'
				if 0 < len(_similar_faces):
					name=_similar_faces[0]
				cv2.rectangle(cuda_np_array, (startX, endY-15), (startY, endX),(255,0,0), cv2.FILLED)
				cv2.putText(cuda_np_array, name, (startX+3, endY-3), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255))
		cuda_mem = jetson.utils.cudaFromNumpy(cuda_np_array)
		#del cuda_mem
		del cuda_np_array
		return cuda_mem
		'''
		self.last_time = end_time
		return _faces
		# Step2. For each face, get the cropped face area, feeding it to face describer (insightface) to get 512-D Feature Embedding
                _face_descriptions = []
		_num_faces = len(_faces)
		if _num_faces == 0:
			return
		for _face in _faces:
			#_face_resize = cv2.resize(_face, configs.face_describer_tensor_shape)
			#_data_feed = [np.expand_dims(_face_resize, axis=0), configs.face_describer_drop_out_rate]
			_face_description = ''#self.face_describer.inference(_data_feed)[0][0]
			_face_descriptions.append(_face_description)

			# Step3. For each face, check whether there are similar faces and if not save it to db.
			# Below naive and verbose implementation is to tutor you how this work
			_similar_faces = self.face_db.get_similar_faces(_face_description)
			if len(_similar_faces) == 0 or len(self.face_db.faces) == 0:
				self.face_db.add_face(face_img=_face, face_description=_face_description)
		print('[Demo] -----------------------------------------------------------')
		
		#image = np.array(Image.open(io.BytesIO(target_image))) 
		#image = np.frombuffer(target_image, dtype=target_image.dtype)
		image = np.array(target_image) 
		cuda_mem = jetson.utils.cudaFromNumpy(cuda_np_array)
		del cuda_mem
		del cuda_np_array
		'''	

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

