import os
import cv2

# 替换为你的文件夹路径
folder_path = '/home/ljh/Desktop/my_little_script/vis_image_0810_4'

for filename in os.listdir(folder_path):
    if filename.endswith(".png"):
        ori_img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_UNCHANGED)
        new_img = ori_img[:,:,0]
        cv2.imwrite(filename, new_img)
        print(f"successfully translating {filename} to 1 channel")
