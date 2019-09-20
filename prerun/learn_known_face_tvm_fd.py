import glob
import os
import cv2
from PIL import Image 
import numpy as np
import math

from utils import face_align
from configs import configs
from service.face_db import Model
from utils.face_recognition import FaceRecognition as FR
from utils.face_detector import FaceDetector as FD

_tpu=FD()
_tvm = FR()
_model=Model()
_label=""
_before_label = ""
_index=0
scales = [112, 112]
scale=112
folders = [f for f in glob.glob(configs.tvm_faces_dir + "**/*", recursive=True)]


def get_label_from_path(path):
    return path.split('/')[-2]

for f in folders:
	if(not os.path.isdir(f)):
		print("label:{}".format(get_label_from_path(f)))
		_label=get_label_from_path(f)
		if(_before_label!=_label):
			_index=0
			
		print("fullpath :{} ".format(f))
		img=cv2.imread(f)
		print('getting faces....')
		img=cv2.resize(img, (112, 112))
# 		target_image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		faces, landmarks  = _tpu.get_faces(img,0.5,1)
		
		if faces is not None:
			print('found', faces.shape[0], 'faces')
			for i in range(faces.shape[0]):
				print('score', faces[i][4])
				box = faces[i].astype(np.int)
				color = (0,0,255)
				#cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
				if landmarks is not None:
					print('found landmarks')
					landmark5 = landmarks[i].astype(np.int)
					#print(landmark.shape)
					#left:startX, top:startY, right:endX,bottom:endY			
					#img= img[box[1]:box[3],box[0]:box[2]]
					img = face_align.norm_crop(img,landmark5)
					for l in range(landmark5.shape[0]):
						color = (0,0,255)
						if l==0 or l==3:
							color = (0,255,0)
						#print('{} {}'.format(landmark5[l][0], landmark5[l][1]))
						#cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)
				else:
					print('not found landmarks')
				#print('==========write name{}, index{}'.format(_label,_index))
				img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
				im = Image.fromarray(img)
				name='{}.{}'.format(_label,_index)
				_model.add_face(name,_tvm.get_face_feature(img))
				im.save('{}.png'.format(name))
				_index=_index+1
		_before_label=_label
		'''
		print(img.shape)
		im_shape = img.shape
		target_size = scales[0]
		max_size = scales[1]
		im_size_min = np.min(im_shape[0:2])
		im_size_max = np.max(im_shape[0:2])
		#im_scale = 1.0
		im_scale = float(target_size) / float(im_size_min)
		if np.round(im_scale * im_size_max) > max_size:
			im_scale = float(max_size) / float(im_size_max)
		print('im_scale', im_scale)
		scale = [im_scale]
		'''
		'''
		print(img.shape)
		x_scale=int((img.shape[1]/scale,0))
		y_scale=int((img.shape[0]/scale,0))
		print('{} {}'.format(x_scale,y_scale))
		faces, landmarks  = _tpu.get_faces(target_image,0.5)
		
		if faces is not None:
			print('find', faces.shape[0], 'faces')
			for i in range(faces.shape[0]):
				print('score', faces[i][4])
				box = faces[i].astype(np.int)
				#color = (255,0,0)
				color = (0,0,255)
				#print('{} {} {} {}'.format(box[0],box[1],box[2],box[3]))
				#left:startX, top:startY, right:endX,bottom:endY			
				cv2.rectangle(img, (box[0]*x_scale, box[1]*y_scale), (box[2]*x_scale, box[3]*y_scale), color, 2)
				if landmarks is not None:
					landmark5 = landmarks[i].astype(np.int)
					#print(landmark.shape)
					for l in range(landmark5.shape[0]):
						color = (0,0,255)
						if l==0 or l==3:
							color = (0,255,0)
						#print('{} {}'.format(landmark5[l][0], landmark5[l][1]))
						cv2.circle(img, (landmark5[l][0]*x_scale, landmark5[l][1]*y_scale), 1, color, 2)
			im = Image.fromarray(img)
			name='{}.{}'.format(_label,_index)
			_model.add_face(name,_tvm.get_face_feature(img))
			im.save('{}.png'.format(name))
			_index=_index+1
		'''

		'''
		if  _faces==None or len(_faces) == 0:
			print('not found!!!')
			continue
		for _face in _faces:
			print('getting faces embedding data...')
			box = _face.bounding_box.flatten().astype("int")
			(startX, startY, endX, endY) = box
			#(start_x, start_y, start_x + width, start_y + height) 
			crop_img=(image[startY:endY,startX:endX])#image.crop((startX,startY,endX-startX,endY-startY))
			name='{}.{}'.format(_label,_index)
			_model.add_face(name,_tvm.get_face_feature(crop_img))
# 			cv2.imwrite(name,crop_img)
			im = Image.fromarray(crop_img)
			im.save('{}.png'.format(name))
			_index=_index+1
		'''


_model.save_faces(configs.tvm_faces,configs.tvm_faces_descriptions)
