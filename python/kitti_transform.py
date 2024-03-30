import cv2
import numpy as np
from tqdm import tqdm
import os
import torch
from sne_model import SNE
import imageio
import sys




class kittiCalibInfo():
    """
    Read calibration files in the kitti dataset,
    we need to use the intrinsic parameter of the cam2
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
            [numpy.array]: intrinsic parameter of the cam2
        """
        return self.data['P2']

    def _load_calib(self, filepath):
        rawdata = self._read_calib_file(filepath)
        data = {}
        P0 = np.reshape(rawdata['P0'], (3, 4))
        P1 = np.reshape(rawdata['P1'], (3, 4))
        P2 = np.reshape(rawdata['P2'], (3, 4))
        P3 = np.reshape(rawdata['P3'], (3, 4))
        R0_rect = np.reshape(rawdata['R0_rect'], (3, 3))
        Tr_velo_to_cam = np.reshape(rawdata['Tr_velo_to_cam'], (3, 4))

        data['P0'] = P0
        data['P1'] = P1
        data['P2'] = P2
        data['P3'] = P3
        data['R0_rect'] = R0_rect
        data['Tr_velo_to_cam'] = Tr_velo_to_cam

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

sne_model = SNE()
root_path = '/home/ljh/Desktop/Workspace/custom_dataset/kitti/kitti_road/testing'
for root, dirs, files in os.walk(root_path):
    for name in tqdm(files):
        if 'lidar_depth_2' in root and name.endswith('.png'):
            depth_image = cv2.imread(os.path.join(root, name), cv2.IMREAD_ANYDEPTH)
            calib_path = os.path.join(root, name).replace('lidar_depth_2', 'calib')
            calib = kittiCalibInfo(calib_path.replace('png', 'txt'))
            camParam = torch.tensor(calib.get_cam_param(), dtype=torch.float32)
            normal = sne_model(torch.tensor(depth_image.astype(np.float32) / 1000), camParam)
            another_image = normal.cpu().numpy()
            another_image = np.transpose(another_image, [1, 2, 0])
            another_image = (another_image+1)/2
            sne_store = (another_image * 65535).astype(np.uint16)
            sne_save_path = os.path.join(root, name).replace('lidar_depth_2', 'sne')
            os.makedirs(root.replace('lidar_depth_2', 'sne'),exist_ok=True)
            imageio.imwrite(sne_save_path, sne_store, format='png-FI')
            print(f'finished processing {sne_save_path}')
print('all finished!')


