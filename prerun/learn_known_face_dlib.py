import glob
import os
import cv2
from PIL import Image 

from configs import configs
from service.face_db import Model
from utils.tpu_util import Tpu
from utils.dlib_util import Dlib

_tpu=Tpu()
_tvm = Dlib()
_model=Model()
_label=""
_before_label = ""
_index=0
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
		image=cv2.imread(f)
		print('getting faces....')
		target_image = Image.frombytes('RGB', (image.shape[1],image.shape[0]), image, 'raw')
		_faces = _tpu.get_faces(target_image, threshold=configs.tpu_threshold,keep_aspect_ratio=configs.tpu_keep_aspect_ratio, relative_coord=configs.tpu_relative_coord,top_k=configs.tpu_top_k)

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
			crop_img = cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB)
			im = Image.fromarray(crop_img)
			im.save('{}.png'.format(name))
			_index=_index+1
		_before_label=_label
_model.save_faces(configs.tvm_faces,configs.tvm_faces_descriptions)
'''
		image=cv2.imread(f)
		print('getting faces....')
		target_image = Image.frombytes('RGB', (image.shape[1],image.shape[0]), image, 'raw')
		_faces = _tpu.get_faces(target_image, threshold=configs.tpu_threshold,keep_aspect_ratio=configs.tpu_keep_aspect_ratio, relative_coord=configs.tpu_relative_coord,top_k=configs.tpu_top_k)

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

_model.save_faces(configs.tvm_faces,configs.tvm_faces_descriptions)
exit'''