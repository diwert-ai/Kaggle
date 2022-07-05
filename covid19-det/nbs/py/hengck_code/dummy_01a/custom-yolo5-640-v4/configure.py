import numpy as np


num_class = 1

# yolo net
num_head = 3
image_size = 640
feature_size = [
    80, 40, 20
]
feature_stride = [
    8, 16, 32
]

anchor_size = [
    [[ 10,13], [ 16, 30], [ 33, 23]],  # P3/8
    [[ 30,61], [ 62, 45], [ 59,119]],  # P4/16
    [[116,90], [156,198], [373,326]],  # P5/32
]
num_anchor = 3 #per head

def make_norm_anchor_size():
    norm_anchor_size = np.array(anchor_size)/np.array(feature_stride).reshape(3,1,1)
    norm_anchor_size = norm_anchor_size.tolist()
    return norm_anchor_size


#----------------------------------------------

# yolo loss
anchor_match_ratio_threshold = 4
loss_level_balance = [4.0, 1.0, 0.4]
loss_obj_balance = 100
loss_cls_balance = 1
loss_box_balance = 0.05

#nms
nms_objectness_threshold = 0.001
nms_iou_threshold = 0.5
nms_pre_max_num   = 3000
nms_post_max_num  = 100
