import os
import shutil

# 指定原始数据集和新数据集的根目录
dataset_root = '/media/ljh/data/cityscapesxxx'  # 需要修改为你的数据集路径
new_dataset_root = '/media/ljh/data/Cityscapes'  # 新的数据集路径

# 创建新的文件夹结构
for dataset_type in ['images', 'sne', 'annotations', 'disp']:
    for split_type in ['train', 'val', 'test']:
        os.makedirs(os.path.join(new_dataset_root, dataset_type, split_type), exist_ok=True)

# 定义原始和新数据集的子目录名
sub_dirs = {'images': 'leftImg8bit', 'sne': 'left_normal', 'annotations': 'gtFine', 'disp': 'disparity'}

# 需要复制的文件关键字或后缀名
# keywords = ['leftImg8bit.png', 'normal.png', '_gtFine_labelTrainIds.png', '_gtFine_color.png', '_gtFine_instanceIds.png']
keywords = ['_disp.png']
# 将文件从原始文件夹复制到新的文件夹
for split_type in ['train', 'val', 'test']:
    for dataset_type, sub_dir in sub_dirs.items():
        source_folder = os.path.join(dataset_root, sub_dir, split_type)
        dest_folder = os.path.join(new_dataset_root, dataset_type, split_type)
        for city in os.listdir(source_folder):
            city_folder = os.path.join(source_folder, city)
            for filename in os.listdir(city_folder):
                # 检查文件名是否包含关键字或后缀名
                if any(keyword in filename for keyword in keywords):
                    shutil.copy2(os.path.join(city_folder, filename),
                                 os.path.join(dest_folder, filename))
            print(f'finished processing {city}')
    print(f'finished processing {split_type}')
print('all finished!')
