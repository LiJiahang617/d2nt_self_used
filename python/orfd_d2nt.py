import os.path
import torchvision.transforms as transforms
import torch
import cv2
import numpy as np
import glob
from utils import *


class orfdCalibInfo():
    """
    Read calibration files in the ORFD dataset,
    we need to use the intrinsic parameter
    """
    def __init__(self, filepath):
        """
        Args:
            filepath ([str]): calibration file path (AAA.txt)
        """
        self.data = self._load_calib(filepath)

    def get_cam_param(self):
        """
        Returns:
            [numpy.array]: intrinsic parameter
        """
        return self.data['K']

    def _load_calib(self, filepath):
        rawdata = self._read_calib_file(filepath)
        data = {}
        K = np.reshape(rawdata['cam_K'], (3,3))
        data['K'] = K
        return data

    def _read_calib_file(self, filepath):
        """Read in a calibration file and parse into a dictionary."""
        data = {}

        with open(filepath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data


def depth2normal(depth_data, cam_fx, cam_fy, u0, v0):
    h, w = (720, 1280)

    u_map = np.ones((h, 1)) * np.arange(1, w + 1) - u0
    v_map = np.arange(1, h + 1).reshape(h, 1) * np.ones((1, w)) - v0
    Gu, Gv = get_DAG_filter(depth_data)
    est_nx = Gu * cam_fx
    est_ny = Gv * cam_fy
    est_nz = -(depth_data + v_map * Gv + u_map * Gu)
    est_normal = cv2.merge((est_nx, est_ny, est_nz))
    est_normal = vector_normalization(est_normal)
    est_normal = MRF_optim(depth_data, est_normal)
    n_vis = (est_normal + 1) / 2

    return n_vis


# 递归遍历文件夹
def process_folder(root_folder):

    cam_fx, cam_fy, u0, v0 = get_cam_params('/home/ljh/Desktop/Workspace/depth-to-normal-translator/python/orfd.txt')
    for root, dirs, files in os.walk(root_folder):
        if 'dense_depth' in root:
            print("Processing files in: " + root)
            for file_name in files:
                if file_name.endswith('.png'):
                    # 读取深度图
                    depth_file_path = os.path.join(root, file_name)
                    depth_data = cv2.imread(depth_file_path, cv2.IMREAD_UNCHANGED)
                    # 计算真实深度
                    depth_real = depth_data.astype(np.float32)/256
                    # 转换为法向量
                    n_vis = depth2normal(depth_real, cam_fx, cam_fy, u0, v0)
                    # 转换为uint16
                    n_vis_65535 = np.round(n_vis * 65535).astype(np.uint16)
                    n_vis_65535 = cv2.cvtColor(n_vis_65535, cv2.COLOR_RGB2BGR)
                    # 保存为png
                    output_file_path = depth_file_path.replace('dense_depth', 'sne')
                    output_dir = os.path.dirname(output_file_path)
                    os.makedirs(output_dir, exist_ok=True)
                    cv2.imwrite(output_file_path, n_vis_65535)
                    print("Processed files: " + depth_file_path)
root_folder = '/home/ljh/Desktop/Workspace/depth-to-normal-translator/ORFD_sequence'  # 设置数据集根目录路径
process_folder(root_folder)
print('all finished!')