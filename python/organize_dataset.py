import os
import shutil
import random
from pathlib import Path


# divide kitti dataset
def move_files_to_validation(modality_dict, file_to_move):
    for modality, ext in modality_dict.items():
        source_file = file_to_move.replace('image_2', modality).replace('.png', ext).replace('validation', 'training')
        target_file = file_to_move.replace('image_2', modality).replace('.png', ext)
        target_dir = os.path.dirname(target_file)

        # Create the target directory if it doesn't exist.
        Path(target_dir).mkdir(parents=True, exist_ok=True)

        # Move the file.
        shutil.move(source_file, target_file)
        print(f'successfully transform to {target_file}')

# change num of samples
def sample_files_for_validation(root_path):
    # Dictionary of all modalities and their respective file extensions
    modalities = {'sne': '.png', 'lidar_depth_2': '.png', 'disp_2': '.tiff'}
    val_rgb_path = os.path.join(root_path, 'validation', 'image_2')
    # Get a list of all the 'rgb_front_left' .png files
    all_files = [os.path.join(val_rgb_path, filename) for filename in os.listdir(val_rgb_path)]
    for selected_file in all_files:
        move_files_to_validation(modalities, selected_file)

root_path = '/home/ljh/Desktop/Workspace/custom_dataset/kitti/kitti_road'  # Update with your dataset path
sample_files_for_validation(root_path)
print('finished!')