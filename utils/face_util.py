import numpy as np
from configs import configs

def face_dis(face_encodings, face_to_compare):
	if len(face_encodings) == 0:
		return np.empty((0))
	face_dist_value = np.linalg.norm(face_encodings - face_to_compare, axis=1)
	
	print('[Face Services | face_distance] Distance between two faces is {}'.format(face_dist_value))
	return face_dist_value

def face_cos(face_encodings, face_to_compare):
	if len(face_encodings) == 0:
		return np.empty((0))
	#face_dist_value = np.linalg.norm(face_encodings - face_to_compare, axis=1)
	res=[]
	emb2 = face_to_compare.flatten()
	for e in face_encodings:
		emb1 = e.flatten()
		sim = np.dot(emb1, emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))	
		res.append(sim)
		#print(cos)
	face_dist_value = np.array(res)
	'''
        emb1 = self.get_embedding(img1).flatten()
        emb2 = face_to_compare.flatten()
        
        sim = np.dot(emb1, emb2)/(norm(emb1)*norm(emb2))
        return sim
	'''
	'''
	res=[]
	for e in face_encodings:
		num=float(np.sum(e*face_to_compare))
		#print(num)
		denom=np.linalg.norm(e)*np.linalg.norm(face_to_compare)
		cos = num / denom
		res.append(cos)
		#print(cos)
	face_dist_value = np.array(res)
	'''
	print('[Face Services | face_distance] Distance between two faces is {}'.format(face_dist_value))
	return face_dist_value

def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=configs.tvm_face_similarity_threshold):
	if(configs.loss_function=='cos'):
		print('cos')
		index = compare_faces_cos(known_face_encodings, face_encoding_to_check, tolerance)
	else:
		print('distances')
		index = compare_faces_distance(known_face_encodings, face_encoding_to_check, tolerance)
	return index

def compare_faces_cos(known_face_encodings, face_encoding_to_check, tolerance):
	face_dist_list=face_cos(known_face_encodings, face_encoding_to_check)
	max_face_dist= np.max(face_dist_list);
	similar_indx = list(np.where(max_face_dist==face_dist_list)[0]) if max_face_dist>=tolerance else list([])
	print('--------------------{}:{}-----------------'.format(max_face_dist,similar_indx))
	#true_list = list( max_face_dist <= tolerance))
	#similar_indx = list(np.where(true_list)[0])
	return similar_indx

def compare_faces_distance(known_face_encodings, face_encoding_to_check, tolerance):
	 # See if the face is a match for the known face(s)
	distances = face_dis(known_face_encodings, face_encoding_to_check)
	min_value = min(distances)
	print(min_value)
	similar_indx = list(np.where(min_value==distances)[0]) if min_value<=tolerance else list([])
	print("similar index:",similar_indx)
	return similar_indx
	'''
	min_value = min(distances)

	# tolerance: How much distance between faces to consider it a match. Lower is more strict.
	# 0.6 is typical best performance.
	if min_value < 0.6:
		index = np.argmin(distances)
	'''
	
	
