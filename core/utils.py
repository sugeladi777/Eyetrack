import numpy as np
import os
import torch
import cv2
from PIL import Image


def pad_bbox(bbox, img_wh, padding_ratio=0.2):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    size_bb = int(max(width, height) * (1+padding_ratio))
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(img_wh[0] - x1, size_bb)
    size_bb = min(img_wh[1] - y1, size_bb)

    return [x1, y1, x1+size_bb, y1+size_bb]


def get_pose_lm_weights(device):
    w = torch.ones(68).to(device)
    w[0:25]=2
    norm_w = w / w.sum()
    return norm_w


def get_exp_lm_weights(device):
    w = torch.ones(68).to(device)
    w[26:]=2
    norm_w = w / w.sum()
    return norm_w


def save_obj(path, v, f, colors=None, point_size=10):
    with open(path, 'w') as file:
        for i in range(len(v)):
            if colors is not None and i < len(colors):
                file.write('v %f %f %f %f %f %f\n' %
                           (v[i, 0], v[i, 1], v[i, 2], colors[i, 0], colors[i, 1], colors[i, 2]))
            else:
                file.write('v %f %f %f\n' %
                           (v[i, 0], v[i, 1], v[i, 2]))

        file.write('\n')

        for i in range(len(f)):
            file.write('f %d %d %d\n' % (f[i, 0], f[i, 1], f[i, 2]))

        if colors is not None:
            for i in range(len(v)):
                if np.array_equal(colors[i], [1, 0, 0]):
                    file.write('p %d\n' % (i + 1))
                    file.write('ps %d\n' % point_size)

    file.close()


def mark_landmarks_on_image(lms_np, resized_face_img, res_folder):
    # 将关键点标记在图像上
    for (x, y) in lms_np:
        cv2.circle(resized_face_img, (int(x), int(y)), 2, (0, 0, 255), -1)

    # 保存标记后的图像
    output_image_path = os.path.join(res_folder, 'landmarks_detected.jpg')
    cv2.imwrite(output_image_path, resized_face_img)
    print(f'Landmarks image saved to {output_image_path}')

def save_resized_face_image(resized_face_img, res_folder):
    resized_face_img_path = os.path.join(res_folder, 'resized_face_img.jpg')
    Image.fromarray(resized_face_img).save(resized_face_img_path)
    print(f'Resized face image saved to {resized_face_img_path}')