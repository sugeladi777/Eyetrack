import os
import sys
import cv2
import time
import argparse
import numpy as np
from tqdm import tqdm
from detector import LandmarkDetector, LandmarkDetectorV2, LandmarkDetectorV3
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils import *

def read_SenseTime_lmk(file):
    face_lmk_all = []
    face_lmk = []
    eyelid_lmk = []
    iris_lmk = []
    with open(file, 'r') as f:
        lines = f.readlines()
        if len(lines) == 0:
            return face_lmk, eyelid_lmk, iris_lmk
        face_lmk_num = int(lines[0])
        face_strs = lines[1].split()
        for i in range(face_lmk_num):
            x = float(face_strs[2 * i])
            y = float(face_strs[2 * i + 1])
            face_lmk_all.append([x, y])

        eyelid_lmk_num = int(lines[2])
        eyelid_strs = lines[3].split()
        for i in range(eyelid_lmk_num):
            x = float(eyelid_strs[2 * i])
            y = float(eyelid_strs[2 * i + 1])
            eyelid_lmk.append([x, y])

        iris_lmk_num = int(lines[4])
        iris_strs = lines[5].split()
        for i in range(iris_lmk_num):
            x = float(iris_strs[2 * i])
            y = float(iris_strs[2 * i + 1])
            iris_lmk.append([x, y])

        face_lmk += face_lmk_all[0:33:2]
        face_lmk += face_lmk_all[33:64]
        face_lmk += face_lmk_all[84:104]
    return np.array(face_lmk + iris_lmk + eyelid_lmk)

def save_lmk(save_path, lmks):
    face_lmks = lmks[:68]
    iris_lmks = lmks[68:68+38]
    eyelid_lmks = lmks[68+38:]

    with open(save_path, 'w') as f:
        # Save face landmarks
        f.write('%d\n' % 68)
        for i in range(68):
            f.write('%.3f %.3f ' % (face_lmks[i, 0], face_lmks[i, 1]))
        f.write('\n')
        # Save eyelid landmarks
        f.write('%d\n' % 48)
        for i in range(48):
            f.write('%.3f %.3f ' % (eyelid_lmks[i, 0], eyelid_lmks[i, 1]))
        f.write('\n')
        # Save iris landmarks
        f.write('%d\n' % 38)
        for i in range(38):
            f.write('%.3f %.3f ' % (iris_lmks[i, 0], iris_lmks[i, 1]))
        f.write('\n')

def cal_pixel_err(pred, gt):
    return np.mean(np.linalg.norm(pred - gt, axis=1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='pretrained/mergefaneye_d2_ep40_sym_DM_ce_gray_noblend.onnx')
    
    parser.add_argument('--data', type=str, default='/home/cloud/StrabismusScripts/fan_eye-main/seqdata/image')
    parser.add_argument('--out', type=str, default='/home/cloud/StrabismusScripts/fan_eye-main/seqdata/result')
    parser.add_argument('--savelmk', action='store_true', default=True)
    parser.add_argument('--usefilter', action='store_true', default=True)
    args = parser.parse_args()

    gtlmk_dir = '../seqdata/landmark2'
    
    save_img_dir = os.path.join(args.out, 'overlap')
    os.makedirs(save_img_dir, exist_ok=True)
    if args.savelmk:
        save_lmk_dir = os.path.join(args.out, 'landmark')
        os.makedirs(save_lmk_dir, exist_ok=True)

    use_onnx = args.ckpt[-4:] == 'onnx'
    lmk_detector = LandmarkDetectorV3(args.ckpt, use_onnx=use_onnx, use_filter=args.usefilter)

    img_list = os.listdir(args.data)
    img_list.sort()

    err_arr = []
    time_arr = []
    for i, img_name in enumerate(tqdm(img_list)):
        img_path = os.path.join(args.data, img_name)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        st = time.time()
        lmk, interval_det = lmk_detector(img)
        interval = (time.time() - st) * 1000    # ms

        img_draw = draw_all_lmk(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), lmk)
        cv2.putText(img_draw, "Total: %.2f ms" % interval, (50, 75), 
                    cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
        cv2.putText(img_draw, "Detect: %.2f ms" % interval_det, (50, 150), 
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
        cv2.putText(img_draw, "Landmark: %.2f ms" % (interval - interval_det), (50, 225), 
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        cv2.putText(img_draw, "Frame: %d" % i, (50, 300), 
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 128, 128), 2)
        
        cv2.imwrite(os.path.join(save_img_dir, img_name), img_draw)
        if args.savelmk:
            save_lmk(os.path.join(save_lmk_dir, 'landmark_%d.txt' % i), lmk)
