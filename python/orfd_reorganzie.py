import os
import shutil


def reorganize_data(old_root_dir, new_root_dir, phase):
    sub_dirs = ['image_data', 'gt_image', 'dense_depth', 'sne']
    new_sub_dirs = ['images', 'annotations', 'depth', 'normal']

    for sub_dir, new_sub_dir in zip(sub_dirs, new_sub_dirs):
        for root, dirs, files in os.walk(os.path.join(old_root_dir, phase)):
            if root.endswith(sub_dir):
                for file in files:
                    old_file_path = os.path.join(root, file)
                    new_file_dir = os.path.join(new_root_dir, new_sub_dir, phase)
                    new_file_path = os.path.join(new_file_dir, file)

                    os.makedirs(new_file_dir, exist_ok=True)
                    shutil.copy2(old_file_path, new_file_path)


old_root_dir = '/home/ljh/Desktop/Workspace/Inter_Attention-Network/datasets/ORFD'
new_root_dir = '/home/ljh/Desktop/Workspace/Inter_Attention-Network/datasets/orfd'
phases = ['training', 'validation', 'testing']

for phase in phases:
    reorganize_data(old_root_dir, new_root_dir, phase)
