import os

#base config
#BASE_PATH = '/'.join(os.getcwd().split('/')[:-1]) # Using ubuntu machine may require removing this -1
BASE_PATH = '/'.join(os.getcwd().split('/')) # Using ubuntu machine may require removing this -1
face_describer_input_tensor_names = ['img_inputs:0', 'dropout_rate:0']
face_describer_output_tensor_names = ['resnet_v1_50/E_BN2/Identity:0']
face_describer_device = '/cpu:0'
face_describer_model_fp = '{}/pretrained/insightface.pb'.format(BASE_PATH)
face_describer_tensor_shape = (112, 112)
face_describer_drop_out_rate = 0.1


#loss_function="cos"
loss_function="distances"
#tvm config
tvm_dtype="float32"
tvm_model_dim='aid512'
#tvm_model_dim='512'
tvm_model_dir="{}/model/insightface_{}dim/".format(BASE_PATH,tvm_model_dim)
#tvm_model_dir="{}/model/arcface_r100_v1/".format(BASE_PATH)
tvm_graph_json = "{}/deploy_graph.json".format(tvm_model_dir)
tvm_lib = "{}/deploy_lib.so".format(tvm_model_dir)
tvm_params = "{}/deploy_param.params".format(tvm_model_dir)
tvm_data_shape = (1, 3, 112, 112)
#512dim
#tvm_face_similarity_threshold = 24
#128dim
#tvm_face_similarity_threshold = 10
#512dim
#tvm_face_similarity_threshold = 24
#128dim
tvm_face_similarity_threshold = 21
#tvm_face_similarity_threshold = 1.8
#tvm known data
tvm_faces_dir="{}/images/".format(BASE_PATH)
tvm_faces_feature_dir="{}/model/feature_{}dim/".format(BASE_PATH,tvm_model_dim)
tvm_faces_descriptions="{}/faces_descriptions.npy".format(tvm_faces_feature_dir)
tvm_faces="{}/faces.npy".format(tvm_faces_feature_dir)

tvm_fd_dtype="float32"
tvm_fd_model_name='retinaface_mnet025_v1'
tvm_fd_model_dir="{}/model/{}/".format(BASE_PATH,tvm_fd_model_name)
tvm_fd_graph_json = "{}/deploy_graph.json".format(tvm_fd_model_dir)
tvm_fd_lib = "{}/deploy_lib.so".format(tvm_fd_model_dir)
tvm_fd_params = "{}/deploy_param.params".format(tvm_fd_model_dir)
tvm_fd_data_shape = (1, 3, 112, 112)
tvm_fd_nms = 0.4
tvm_fd_rac = 'net3'

tvm_mtcnn_model_name="mtcnn/tvm"
tvm_mtcnn_dir="{}/model/{}/".format(BASE_PATH,tvm_mtcnn_model_name)
tvm_mtcnn_graph_json = tvm_mtcnn_dir+"{}_deploy_graph.json"
tvm_mtcnn_lib = tvm_mtcnn_dir+"{}_deploy_lib.so"
tvm_mtcnn_params = tvm_mtcnn_dir+"{}_deploy_param.params"
tvm_mtcnn_data_shape = (1, 3, 112, 112)

#
#tpu config
tpu_model_dir = '{}/../all_models/'.format(BASE_PATH)
#/home/crsarang/works/ml/tpu/project/../all_models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite
#tpu_model = '{}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'.format(tpu_model_dir)
tpu_model = '{}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite'.format(tpu_model_dir)
#tpu_labels = '{}/coco_labels.txt'.format(tpu_model_dir)
tpu_labels = None
tpu_threshold=0.1
tpu_keep_aspect_ratio=False
tpu_relative_coord=False
tpu_top_k=10

#cam config
cam_csi_width=1280
cam_csi_height=720
#cam_csi_width=1280#1920#3280#1280
#cam_csi_height=720#1080#2464#720
cam_csi_index="0"


