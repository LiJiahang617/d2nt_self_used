import os
import numpy as np
import cv2

# 定义路径
dataset_base = "/media/ljh/Kobe24/KITTI_Semantic"
calib_path = os.path.join(dataset_base, "training", "calib_cam_to_cam")
disp_path = os.path.join(dataset_base, "training", "disp_occ_0")
depth_path = os.path.join(dataset_base, "training", "depth_0")  # 存放深度图的目录

if not os.path.exists(depth_path):
    os.makedirs(depth_path)

# 解析校准文件以获取焦距和基线
def parse_calibration(filename):
    with open(filename, 'r') as file:
        f, B = None, None
        for line in file:
            if 'P_rect_02:' in line:
                values = line.split()
                f = float(values[1])  # 焦距
            if 'T_03:' in line:
                values = line.split()
                B = -float(values[1])  # 基线长度
    return f, B

# 将视差转换为深度，跳过空洞点
def disparity_to_depth(disp_image, f, B):
    depth_image = np.zeros(disp_image.shape, dtype=np.float32)
    valid = disp_image > 0
    depth_image[valid] = f * B / disp_image[valid]
    return depth_image

# 遍历并处理每个文件
for filename in os.listdir(calib_path):
    if filename.endswith(".txt"):
        base_name = filename.split('.')[0]
        f, B = parse_calibration(os.path.join(calib_path, filename))
        disp_filename = base_name + "_10.png"
        disp_image_path = os.path.join(disp_path, disp_filename)
        if os.path.exists(disp_image_path):
            disp_image = cv2.imread(disp_image_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 256.0
            depth_image = disparity_to_depth(disp_image, f, B)
            depth_filename = os.path.join(depth_path, disp_filename)
            cv2.imwrite(depth_filename, (depth_image*255.0).astype(np.uint16))
            print(f"Processed and saved depth image for {disp_filename}")
