import os
from PIL import Image


"""
It's a script for transforming cityscapes normal images (jpg) to png format.
"""
def convert_images(root_folder, output_folder):
    # 遍历目录
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            # 检查文件是否为jpg格式
            if file.endswith('.jpg'):
                # 打开jpg图片
                img = Image.open(os.path.join(root, file))
                # 创建新的文件路径，保持与原目录结构相同
                new_root = root.replace(root_folder, output_folder)
                os.makedirs(new_root, exist_ok=True)  # 确保新目录存在
                # 删除.jpg后缀，添加.png后缀
                new_filename = os.path.splitext(os.path.join(new_root, file))[0] + '.png'
                # 保存为png图片
                img.save(new_filename)

# 调用函数，传入原始目录和新的目录
convert_images('/media/ljh/data/cityscapes/left_normal', '/media/ljh/data/cityscapes/left_normal_png')
