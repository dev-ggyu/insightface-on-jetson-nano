import face_recognition
import cv2
import numpy as np
from configs import configs


class Dlib():
	data_shape = None
	
	def __init__(self):
		self.data_shape = configs.tvm_data_shape
		print('[DLIB] created.')
		
	def __del__(self): 
		print('[DLIB] destoryed.')
		
	def get_face_feature(self,face):
		try:
			img=cv2.resize(face, (self.data_shape[2], self.data_shape[3]))
			#face_locations = face_recognition.face_locations(face)
			#print('face locations')			
			#print(face_locations)
			face_encodings = face_recognition.face_encodings(face=)
			#print('embedding data')
			#print(face_encodings)
		except Exception as e:
			print("[DLIB] cam error :{}".format(str(e)))
		return face_encodings

'''
def 
for face_encoding in self.face_encodings:
                # See if the face is a match for the known face(s)
                distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                min_value = min(distances)

                # tolerance: How much distance between faces to consider it a match. Lower is more strict.
                # 0.6 is typical best performance.
                name = "Unknown"
                if min_value < 0.6:
                    index = np.argmin(distances)
                    name = self.known_face_names[index]

                self.face_names.append(name)

'''
