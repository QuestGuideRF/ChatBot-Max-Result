camera_index = 0
frame_width = 1280
frame_height = 720

frame_skip = 25
detection_frame_size = (320, 320)

model_det_path = "./models/face_detection_yunet_2023mar_int8bq.onnx"
model_rec_path = "models/backbone_ir50_ms1m_epoch120.pth"
data_root = "data/"
trusted_faces_path = "trusted_faces/"
foreign_faces_path = "foreign/"

cos_sim_threshold = 0.4