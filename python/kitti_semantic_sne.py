import os
import numpy as np
import cv2
from utils import *

def get_cam_params_from_file(calib_file_path):
    with open(calib_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if 'P_rect_02:' in line:
                values = line.split(' ')[1:]
                fx, cx, fy, cy = float(values[0]), float(values[2]), float(values[5]), float(values[6])
                break
    return fx, fy, cx, cy

def depth2normal(depth_data, cam_fx, cam_fy, u0, v0):
    h, w = depth_data.shape[0:2]

    u_map = np.ones((h, 1)) * np.arange(1, w + 1) - u0
    v_map = np.arange(1, h + 1).reshape(h, 1) * np.ones((1, w)) - v0
    Gu, Gv = get_DAG_filter(depth_data)
    est_nx = Gu * cam_fx
    est_ny = Gv * cam_fy
    est_nz = -(depth_data + v_map * Gv + u_map * Gu)
    est_normal = cv2.merge((est_nx, est_ny, est_nz))
    est_normal = vector_normalization(est_normal)
    est_normal = MRF_optim(depth_data, est_normal)
    n_vis = (1 - est_normal) / 2

    return n_vis

def process_folder(root_folder, calib_folder):
    for root, dirs, files in os.walk(root_folder):
        if os.path.basename(root) == 'vitas_depth_0':
            print("Processing files in: " + root)
            for file_name in files:
                if file_name.endswith('.png'):
                    depth_file_path = os.path.join(root, file_name)
                    calib_file_path = os.path.join(calib_folder, file_name.replace('_10.png', '.txt'))
                    cam_fx, cam_fy, u0, v0 = get_cam_params_from_file(calib_file_path)

                    depth_data = cv2.imread(depth_file_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)/256.0
                    # depth_data = depth_data[:, :, 0]
                    n_vis = depth2normal(depth_data, cam_fx, cam_fy, u0, v0)
                    n_vis_255 = np.round(n_vis * 255).astype(np.uint8)

                    output_file_path = depth_file_path.replace('depth_0', 'sne_0')
                    output_dir = os.path.dirname(output_file_path)
                    os.makedirs(output_dir, exist_ok=True)
                    cv2.imwrite(output_file_path, n_vis_255)
                    print("Processed file: " + depth_file_path)

root_folder = '/media/ljh/Kobe24/KITTI_Semantic/training/vitas_depth_0'  # 设定为新的根目录路径
calib_folder = '/media/ljh/Kobe24/KITTI_Semantic/training/calib_cam_to_cam'  # 标定文件路径
process_folder(root_folder, calib_folder)
print('All finished!')
