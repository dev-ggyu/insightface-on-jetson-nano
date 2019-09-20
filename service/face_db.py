from utils import face_util
import numpy as np
from configs import configs

class Model(object):
	faces = []
	faces_descriptions = []
	face_util=None
	def __init__(self):
		#self.load_faces(configs.tvm_faces,configs.tvm_faces_descriptions)
		self.face_util=face_util
	
	def add_face(self, name, face_description):
		self.faces.append(name)
		self.faces_descriptions.append(face_description)

	def save_faces(self,faces_file='faces.npy',faces_descriptions_file='faces_descriptions.npy'):
		np.save(faces_file, self.faces) # name
		print('[Model] Saved {} faces.'.format(len(self.faces)))
		np.save(faces_descriptions_file, self.faces_descriptions) #embedding data
		print('[Model] Saved {} faces_descriptions.'.format(len(self.faces_descriptions)))

	def load_faces(self,faces_file='faces.npy',faces_descriptions_file='faces_descriptions.npy'):
		self.faces = np.load(faces_file)
		print('[Model] Loadded {} faces.'.format(len(self.faces)))
		for s in self.faces:
			print('[Model] Faces Name:{}'.format(s))
		self.faces_descriptions = np.load(faces_descriptions_file)
		print('[Model] Loadded {} faces_descriptions.'.format(len(self.faces_descriptions)))

	def drop_all(self):
		self.faces = []
		self.faces_descriptions = []

	def get_all(self):
		return self.faces, self.faces_descriptions
	
	'''
	def face_distance(self,face_encodings, face_to_compare):
		if len(face_encodings) == 0:
			return np.empty((0))
		face_dist_value = np.linalg.norm(face_encodings - face_to_compare, axis=1)
		print('[Face Services | face_distance] Distance between two faces is {}'.format(face_dist_value))
		return face_dist_value

	def compare_faces(self,known_face_encodings, face_encoding_to_check, tolerance=configs.tvm_face_similarity_threshold):
		true_list = list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)
		similar_indx = list(np.where(true_list)[0])
		return similar_indx
	'''
	def get_similar_faces(self, target_face_description):
		print('[Face DB] Looking for similar faces in a DataBase of {} {} faces...'.format(len(self.faces),len(self.faces_descriptions)))
		if len(self.faces) == 0 or len(self.faces_descriptions)==0: 
			return []
		# Use items in Python 3*, below is by default for Python 2*

		similar_face_idx = self.face_util.compare_faces(self.faces_descriptions, target_face_description)
		similar_faces = np.array(self.faces)[similar_face_idx]
		num_similar_faces = len(similar_faces)
		print('[Face DB] Found {} similar faces in a DataBase of {} faces...'.format(num_similar_faces, len(self.faces)))
		return similar_faces
