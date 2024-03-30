import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *




depth_data1 = np.load('/home/ljh/Desktop/disp/000000_10.npy')
print(depth_data1.shape, depth_data1.dtype)
print(np.min(depth_data1), np.max(depth_data1))
# depth_data2 = cv2.imread('/home/ljh/Desktop/SUNRGBD/kv1/b3dodata/img_0063/depth_bfx/img_0063_abs.png', cv2.IMREAD_UNCHANGED)
# print(depth_data2.shape, depth_data2.dtype)
# print(np.min(depth_data2), np.max(depth_data2))
# cv2.waitKey()
# cv2.destroyAllWindows()