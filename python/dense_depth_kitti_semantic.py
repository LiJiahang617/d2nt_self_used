import cv2
import numpy as np
import os
#
# def interpolate_depth(depth_image, top_rows=125):
#     # 确保深度图像是单通道
#     if len(depth_image.shape) == 3 and depth_image.shape[2] == 3:
#         depth_image = depth_image[:, :, 0]
#
#     # 确保深度图像是float32格式
#     depth_image = depth_image.astype(np.float32)
#
#     # 创建一个副本用于插值，初始化为0
#     interpolated_image = np.zeros_like(depth_image)
#
#     # 将除了最上方0-125行以外的区域用于插值
#     valid_depth_part = depth_image[top_rows:, :]
#     interpolated_image[top_rows:, :] = cv2.resize(valid_depth_part, (depth_image.shape[1], depth_image.shape[0] - top_rows), interpolation=cv2.INTER_CUBIC)
#
#     return interpolated_image
#
# # 用于读取和保存图像的路径
# disp_path = '/media/ljh/Kobe24/KITTI_Semantic/training/depth'
# depth_path = '/media/ljh/Kobe24/KITTI_Semantic/training/dense_depth'
#
# # 创建输出目录
# if not os.path.exists(depth_path):
#     os.makedirs(depth_path)
#
# for filename in os.listdir(disp_path):
#     if filename.endswith(".png"):
#         # 读取深度图
#         depth_image = cv2.imread(os.path.join(disp_path, filename), cv2.IMREAD_UNCHANGED)
#         if depth_image is None:
#             continue
#         # 插值深度图
#         interpolated_depth_image = interpolate_depth(depth_image)
#         # 保存插值后的深度图
#         interpolated_depth_filename = os.path.join(depth_path, filename)
#         cv2.imwrite(interpolated_depth_filename, interpolated_depth_image)
#
# print("快速深度图插值完成")


def fill_depth_holes(depth_image, top_rows=125):
    # 确保深度图像是单通道
    if len(depth_image.shape) == 3 and depth_image.shape[2] == 3:
        depth_image = depth_image[:, :, 0]

    # 保留原始的顶部mask
    original_top_mask = np.zeros_like(depth_image)
    original_top_mask[:top_rows, :] = 1

    # 首先应用形态学闭操作尝试填补小空洞
    kernel = np.ones((5, 5), np.uint8)
    depth_closed = cv2.morphologyEx(depth_image, cv2.MORPH_CLOSE, kernel)

    # 然后使用非零像素的双三次插值进行大面积空洞填补
    mask = depth_image > 0
    depth_filled = cv2.inpaint(depth_closed, (1 - mask).astype(np.uint8), 3, cv2.INPAINT_TELEA)

    # 还原顶部区域
    depth_filled[original_top_mask == 1] = 0

    return depth_filled


# 用于读取和保存图像的路径
disp_path = '/media/ljh/Kobe24/KITTI_Semantic/training/depth'
depth_path = '/media/ljh/Kobe24/KITTI_Semantic/training/dense_depth'

# 创建输出目录
if not os.path.exists(depth_path):
    os.makedirs(depth_path)

filename = '000006_10.png'  # 假定文件名为上传文件的名称
# 读取深度图
depth_image = cv2.imread(os.path.join(disp_path, filename), cv2.IMREAD_UNCHANGED)
if depth_image is None:
    print("无法读取图像文件")
else:
    # 填补深度图中的空洞
    depth_filled_image = fill_depth_holes(depth_image)
    # 保存填补后的深度图
    filled_depth_filename = os.path.join(depth_path, filename)
    cv2.imwrite(filled_depth_filename, depth_filled_image)

    print("深度图空洞填补完成，文件保存于: {}".format(filled_depth_filename))