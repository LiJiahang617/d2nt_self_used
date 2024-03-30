import os
import sys
import numpy as np
import tifffile as tiff
from tqdm import tqdm
from pathlib import Path
import cv2
import imageio
imageio.plugins.freeimage.download()


root_path = '/media/ljh/data/cityscapesxxx/left_disp'
for root, dirs, files in os.walk(root_path):
    for name in tqdm(files):
        if name.endswith('.tiff'):  # Only process .npy files in directories.
            disp_path = os.path.join(root, name)
            disp = tiff.imread(disp_path)
            disp_store = (disp * 255).astype(np.uint16)
            disp_save_path = Path(disp_path.replace('left_disp', 'disparity'))
            disp_save_path.parent.mkdir(parents=True, exist_ok=True)
            disp_save_path = Path(str(disp_save_path).replace('.tiff', '.png'))
            # 使用 imageio 保存 sne_store 为 PNG 图像
            imageio.imwrite(disp_save_path, disp_store, format='png-FI')
            print(f'finished processing {disp_save_path}')
print('all finished!')