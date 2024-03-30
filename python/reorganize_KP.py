import os
import shutil

# 定义原始数据集和目标数据集的路径
src_dataset_path = '/media/ljh/Kobe24/KPdataset_concise'
dst_dataset_path = '/media/ljh/Kobe24/KPdataset1'

# 定义子集名称
sets = ['train', 'val', 'test']

# 创建目标数据集的文件夹结构
for set_name in sets:
    os.makedirs(os.path.join(dst_dataset_path, 'images', set_name), exist_ok=True)
    os.makedirs(os.path.join(dst_dataset_path, 'labels', set_name), exist_ok=True)
    os.makedirs(os.path.join(dst_dataset_path, 'thermal', set_name), exist_ok=True)

# 复制和重命名文件的函数
def copy_and_rename(src_file, dst_folder, new_filename, is_label=False):
    if is_label:
        new_filename = new_filename.replace('.jpg', '.png')
    dst_file = os.path.join(dst_folder, new_filename)
    shutil.copy2(src_file, dst_file)
    print(f"文件 {src_file} 已复制到 {dst_file}")

# 处理每个子集
for set_name in sets:
    set_file_path = os.path.join(src_dataset_path, f'{set_name}.txt')
    with open(set_file_path, 'r') as file:
        for line in file:
            filename = line.strip()
            base_filename = filename.split('.')[0]  # 去掉扩展名
            parts = base_filename.split('_')
            new_image_name = '_'.join(parts[-3:]) + '.jpg'  # 包括set, V和I的部分

            # 构造原始文件路径
            src_label_path = os.path.join(src_dataset_path, 'labels', filename)
            src_thermal_path = os.path.join(src_dataset_path, 'images', parts[0], parts[1], 'lwir', parts[-1] + '.jpg')
            src_visible_path = os.path.join(src_dataset_path, 'images', parts[0], parts[1], 'visible', parts[-1] + '.jpg')

            # 目标文件夹
            dst_image_folder = os.path.join(dst_dataset_path, 'images', set_name)
            dst_label_folder = os.path.join(dst_dataset_path, 'labels', set_name)
            dst_thermal_folder = os.path.join(dst_dataset_path, 'thermal', set_name)

            # 复制和重命名文件
            copy_and_rename(src_label_path, dst_label_folder, new_image_name, is_label=True)
            copy_and_rename(src_thermal_path, dst_thermal_folder, new_image_name)
            copy_and_rename(src_visible_path, dst_image_folder, new_image_name)