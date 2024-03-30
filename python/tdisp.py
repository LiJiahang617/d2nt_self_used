import math
import os
import sys
import numpy as np
import tifffile
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path

plt.ion()


def pt_param_estimation(disp, mask):
    height, width = disp.shape
    length = width * height
    umap = (np.ones((height, 1)) * np.arange(1, width +
            1).reshape(1, width)).astype(np.float64)
    vmap = (np.arange(1, height + 1).reshape(height, 1)
            * np.ones((1, width))).astype(np.float64)
    u_1d = umap.reshape(length, 1)
    v_1d = vmap.reshape(length, 1)
    # if images have the same shape, the above lines can be moved outside.

    d_1d = disp.reshape(length, 1)
    m_1d = mask.reshape(length, 1)
    d_1d = np.delete(d_1d, np.where(m_1d == 1), 0)
    u_1d = np.delete(u_1d, np.where(m_1d == 1), 0)
    v_1d = np.delete(v_1d, np.where(m_1d == 1), 0)
    m_1d = np.delete(m_1d, np.where(m_1d == 1), 0)
    # axis arg (0) should be considered to keep the vector shape.

    su = np.sum(u_1d)
    sv = np.sum(v_1d)
    sd = np.sum(d_1d)
    su2 = np.sum(np.square(u_1d))
    sv2 = np.sum(np.square(v_1d))
    suv = np.sum(u_1d * v_1d)
    sdu = np.sum(d_1d * u_1d)
    sdv = np.sum(d_1d * v_1d)

    n = len(d_1d)

    beta0 = (sd ** 2 * (sv2 + su2) - 2 * sd *
             (sv * sdv + su * sdu) + n * (sdv ** 2 + sdu ** 2)) / 2
    beta1 = (sd ** 2 * (sv2 - su2) + 2 * sd *
             (su * sdu - sv * sdv) + n * (sdv ** 2 - sdu ** 2)) / 2
    beta2 = -sd ** 2 * suv + sd * (sv * sdu + su * sdv) - n * sdv * sdu
    gamma0 = (n * sv2 + n * su2 - sv ** 2 - su ** 2) / 2
    gamma1 = (n * sv2 - n * su2 - sv ** 2 + su ** 2) / 2
    gamma2 = sv * su - n * suv

    del su, sv, sd, su2, sv2, suv, sdu, sdv, n

    A = (beta1 * gamma0 - beta0 * gamma1)
    B = (beta0 * gamma2 - beta2 * gamma0)
    C = (beta1 * gamma2 - beta2 * gamma1)

    delta = A ** 2 + B ** 2 - C ** 2
    theta1 = math.atan((A + math.sqrt(delta)) / (B - C))
    theta2 = math.atan((A - math.sqrt(delta)) / (B - C))

    del A, B, C, beta0, beta1, beta2, gamma0, gamma1, gamma2, delta

    t1_1d = v_1d * math.cos(theta1) - u_1d * math.sin(theta1)
    t2_1d = v_1d * math.cos(theta2) - u_1d * math.sin(theta2)

    o_1d = np.ones((len(t1_1d), 1)).astype(np.float64)

    T1 = np.concatenate((o_1d, t1_1d), axis=1)
    T2 = np.concatenate((o_1d, t2_1d), axis=1)

    del t1_1d, t2_1d

    f1 = np.dot(np.dot(np.dot(np.transpose(d_1d), T1), np.linalg.inv(np.dot(np.transpose(T1), T1))),
                np.dot(np.transpose(T1), d_1d))
    f2 = np.dot(np.dot(np.dot(np.transpose(d_1d), T2), np.linalg.inv(np.dot(np.transpose(T2), T2))),
                np.dot(np.transpose(T2), d_1d))

    if f1 < f2:
        theta = theta2
    else:
        theta = theta1

    del T1, T2, theta1, theta2

    t_1d = v_1d * np.cos(theta) - u_1d * np.sin(theta)
    T = np.concatenate((o_1d, t_1d), axis=1)
    a = np.dot(np.dot(np.linalg.inv(
        np.dot(np.transpose(T), T)), np.transpose(T)), d_1d)
    # first convert a to an array, then remove axes of length one.
    a = np.squeeze(a)
    tmap = vmap * np.cos(theta) - umap * np.sin(theta)
    tdisp = (disp - a[0] - a[1] * tmap)

    return a, theta, tdisp

root_path = Path('/media/ljh/data/carla_v2')  # Update with your dataset path.
# root_disp_path = root_path / 'carla_v2'
# root_tdisp_path = root_disp_path  # Modify if your tdisp path is different.

if __name__ == '__main__':
    for root, dirs, files in os.walk(root_path):
        for name in tqdm(files):
            if 'disparity_left' in root and name.endswith('.npy'):  # Only process .npy files in disparity_left directories.
                disp_path = os.path.join(root, name)
                seg_gt_path = disp_path.replace('disparity_left', 'gt_seg').replace(
                    '.npy', '.png')

                disp_2 = np.load(disp_path)
                seg_gt = Image.open(seg_gt_path).convert('L')

                disp_2_np = disp_2.astype(np.float32)
                disp_store = (disp_2 * 255).astype(np.uint16)
                seg_gt_np = np.array(seg_gt).astype(np.float32)

                mask = np.ones(disp_2_np.shape).astype(np.float32)
                """
                NumPy在处理逻辑运算时并不使用Python的标准 and, or, not 运算符，
                它使用的是位运算符 &, |, ~。这允许NumPy在处理大规模数组时进行高效的向量化操作。
                """
                mask[(seg_gt_np == 90)] = 0
                # sys.exit(0)
                if mask.all():  # mask 全为1，未检测到路面
                    with open(root_path / 'error_gt.txt', 'a') as f:
                        f.write(seg_gt_path+'\n')
                    continue

                a, theta, tdisp = pt_param_estimation(disp_2_np, mask)
                tdisp = abs(tdisp)
                tdisp_store = (tdisp * 255).astype(np.uint16)
                # tdisp float64 single channel
                disp_save_path = Path(disp_path.replace('disparity_left', 'disp_left'))
                tdisp_save_path = Path(disp_path.replace('disparity_left', 'tdisp2_left'))
                # save_path_jpg = Path(disp_path.replace('disparity_left', 'tdisp_left_jpg').replace(
                #     '.npy', '_tdisp.jpg'))
                disp_save_path.parent.mkdir(parents=True, exist_ok=True)
                tdisp_save_path.parent.mkdir(parents=True, exist_ok=True)
                # save_path_jpg.parent.mkdir(parents=True, exist_ok=True)

                # np.save(save_path, tdisp)
                # 保存 disp_store 为 PNG 图像
                disp_store_image = Image.fromarray(disp_store)
                disp_store_image.save(disp_save_path.with_name(disp_save_path.stem + ".png"))

                # 保存 tdisp_store 为 PNG 图像
                tdisp_store_image = Image.fromarray(tdisp_store)
                tdisp_store_image.save(tdisp_save_path.with_name(tdisp_save_path.stem + ".png"))
                # fig = plt.figure('disp')
                # plt.imshow(disp_store)
                # plt.imshow(tdisp_store)
                # plt.axis('off')
                # plt.show()
                # plt.savefig(save_path_jpg, bbox_inches='tight', pad_inches=0)
                # plt.clf()
                print(f'finished processing {tdisp_save_path}')
    print('all finished!')



