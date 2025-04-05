import os
import sys
import cv2
import time
import numpy as np
import torch
import mediapipe as mp
import onnxruntime
# import dlib
# from ibug.face_detection import RetinaFacePredictor
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from models import *
from utils import *
from one_euro_filter import OneEuroFilter

def center_crop(img):
    H, W = img.shape[:2]
    size = min(H, W)
    H_st = (H - size) // 2
    W_st = (W - size) // 2

    return img[H_st:H_st+size, W_st:W_st+size], H_st, W_st


class MediaPipeFaceDetector():
    def __init__(self, model_select=0) -> None:
        self.face_detector = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.5, model_selection=model_select
        )
    
    def __call__(self, img, ds=1, ret_kp=False):
        H, W = img.shape[:2]
        if ds != 1:
            H_ds, W_ds = H // ds, W // ds
            img_ds = cv2.resize(img, (W_ds, H_ds))
        else:
            img_ds = img
        res = self.face_detector.process(img_ds)
        if res.detections is None:
            return None, None if ret_kp else None
        rel_bbox = res.detections[0].location_data.relative_bounding_box
        face_box = np.array([[(rel_bbox.xmin) * W, (rel_bbox.ymin) * H],
                             [(rel_bbox.xmin + rel_bbox.width) * W, (rel_bbox.ymin + rel_bbox.height) * H]])
        if ret_kp:
            key_points = np.array([[res.detections[0].location_data.relative_keypoints[i].x * W, res.detections[0].location_data.relative_keypoints[i].y * H] for i in range(6)])
            return face_box, key_points # [leye, reye, nose, mouth, lear, rear]
        
        return face_box

class LandmarkDetector():
    def __init__(self, ckpt_path, use_onnx=False, device='cuda:0'):
        # self.face_detector = iBugFaceDetector()
        # self.face_detector = DlibFaceDetector()
        self.face_detector = MediaPipeFaceDetector()

        if use_onnx:
            if device == 'cpu':
                providers = ['CPUExecutionProvider']
            else:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

            self.net = onnxruntime.InferenceSession(ckpt_path, providers=providers)

            # warmup
            onnx_input = {self.net.get_inputs()[0].name: np.random.rand(1, 3, 256, 256).astype(np.float32)}
            heatmap = self.net.run(None, onnx_input)
        else:
            self.net = FAN(FAN.create_config(num_modules=1, 
                                            num_landmarks=86, 
                                            hg_depth=4, 
                                            hg_num_features=256, 
                                            use_avg_pool=False)).to(device)
            self.net.load_state_dict(torch.load(ckpt_path, map_location=device))
            self.net.eval()

            # warmup
            self.net(torch.zeros([1, 3, 256, 256]).to(device))

        self.use_onnx = use_onnx

    @torch.no_grad()
    def __call__(self, img):
        # Face Detection
        # assume the first face box is what we want
        st = time.time()
        face_box = self.face_detector(img)
        interval_det = (time.time() - st) * 1000
        
        # Crop Face Region
        crop_lt, crop_rb = get_crop_from_box(face_box)
        img_crop = pad_and_crop(img, crop_lt, crop_rb)
        img_resize = cv2.resize(img_crop, (256, 256))

        # Landmark Detection
        if self.use_onnx:
            img_onnx = img_resize.astype(np.float32)[np.newaxis, :].transpose((0, 3, 1, 2)) / 255
            onnx_input = {self.net.get_inputs()[0].name: img_onnx}
            heatmap = self.net.run(None, onnx_input)
            lmk, scores = decode_landmarks(torch.from_numpy(heatmap[-1]), gamma=1.0, radius=0.1)
            lmk = lmk[0].detach().numpy() * (img_crop.shape[-2] / heatmap[-1].shape[-1])
        else:
            img_torch = torch.from_numpy(img_resize.astype(np.float32)[np.newaxis, :].transpose(
                    (0, 3, 1, 2))).to('cuda:0') / 255
            heatmap = self.net(img_torch)
            lmk, scores = decode_landmarks(heatmap[-1], gamma=1.0, radius=0.1)
            lmk = lmk[0].detach().cpu().numpy() * (img_crop.shape[-2] / heatmap[-1].shape[-1])
        
        # Map Back to Original Image
        lmk = lmk + crop_lt

        return lmk, interval_det


# V2 use local eye-region image to predict eye landmarks
# V2 only support onnx for simplification
class LandmarkDetectorV2():
    def __init__(self, face_ckpt, eye_ckpt, use_onnx=True, device='cuda:0'):
        self.face_detector = MediaPipeFaceDetector()

        self.leye_idx = [36, 37, 38, 39, 40, 41]    # left eye in image (OD)
        self.reye_idx = [42, 43, 44, 45, 46, 47]    # right eye in image (OS)

        if device == 'cpu':
            providers = ['CPUExecutionProvider']
        else:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self.face_net = onnxruntime.InferenceSession(face_ckpt, providers=providers)

        self.eye_net = onnxruntime.InferenceSession(eye_ckpt, providers=providers)

        # warmup
        onnx_face_input = {self.face_net.get_inputs()[0].name: np.random.rand(1, 3, 256, 256).astype(np.float32)}
        _ = self.face_net.run(None, onnx_face_input)
        onnx_eye_input = {self.eye_net.get_inputs()[0].name: np.random.rand(2, 3, 128, 128).astype(np.float32)}
        _ = self.eye_net.run(None, onnx_eye_input)

        self.use_onnx = use_onnx
        assert use_onnx, "Only Support onnx!"

    def crop_face_region(self, img, face_box):
        crop_lt, crop_rb = get_crop_from_box(face_box)
        img_crop = pad_and_crop(img, crop_lt, crop_rb)
        img_resize = cv2.resize(img_crop, (256, 256))

        return img_resize, img_crop, crop_lt, crop_rb

    def crop_eye_region(self, img, eye_box):
        crop_lt, crop_rb = get_crop_from_box(eye_box)
        img_crop = pad_and_crop(img, crop_lt, crop_rb)
        img_resize = cv2.resize(img_crop, (128, 128))

        return img_resize, img_crop, crop_lt, crop_rb

    @torch.no_grad()
    def __call__(self, img):
        # Face Detection
        # assume the first face box is what we want
        st = time.time()
        face_box = self.face_detector(img)
        interval_det = (time.time() - st) * 1000
        
        # Crop Face Region
        face_img_resize, face_img_crop, face_crop_lt, _ = self.crop_face_region(img, face_box)

        # Landmark Detection (Face)
        face_img_onnx = face_img_resize.astype(np.float32)[np.newaxis, :].transpose((0, 3, 1, 2)) / 255
        face_onnx_input = {self.face_net.get_inputs()[0].name: face_img_onnx}
        face_heatmap = self.face_net.run(None, face_onnx_input)
        face_lmk, face_scores = decode_landmarks(torch.from_numpy(face_heatmap[-1]), gamma=1.0, radius=0.1)
        face_lmk = face_lmk[0].detach().numpy() * (face_img_crop.shape[-2] / face_heatmap[-1].shape[-1])    # get face_img_crop landmarks

        # Crop Eye Region
        leye_box = get_box_from_landmarks(face_lmk[self.leye_idx], face_img_crop.shape[0], face_img_crop.shape[1], pad_scale=0.4)
        reye_box = get_box_from_landmarks(face_lmk[self.reye_idx], face_img_crop.shape[0], face_img_crop.shape[1], pad_scale=0.4)
        leye_img_resize, leye_img_crop, leye_crop_lt, _ = self.crop_eye_region(face_img_crop, leye_box)
        reye_img_resize, reye_img_crop, reye_crop_lt, _ = self.crop_eye_region(face_img_crop, reye_box)
        
        # Landmark Detection (Eye)
        # eye_img_resize.shape: [2, 3, 128, 128]
        eye_img_onnx = np.concatenate([cv2.flip(leye_img_resize, flipCode=1).astype(np.float32)[np.newaxis, :], 
                                       reye_img_resize.astype(np.float32)[np.newaxis, :]], axis=0).transpose((0, 3, 1, 2)) / 255
        eye_onnx_input = {self.eye_net.get_inputs()[0].name: eye_img_onnx}
        eye_heatmaps = self.eye_net.run(None, eye_onnx_input)
        eye_lmks, eye_scores = decode_landmarks(torch.from_numpy(eye_heatmaps[-1]), gamma=1.0, radius=0.1)  # [2, 43, 2]
        leye_lmk = eye_lmks[0].detach().numpy() * (leye_img_crop.shape[-2] / eye_heatmaps[-1].shape[-1])
        reye_lmk = eye_lmks[1].detach().numpy() * (reye_img_crop.shape[-2] / eye_heatmaps[-1].shape[-1])

        # Map Back to Original Face Image
        leye_lmk[:, 0] = leye_img_crop.shape[-2] - leye_lmk[:, 0]   # flip back
        leye_lmk = leye_lmk + leye_crop_lt
        reye_lmk = reye_lmk + reye_crop_lt
        lmk = np.concatenate([face_lmk, leye_lmk[:19], reye_lmk[:19], leye_lmk[19:], reye_lmk[19:]], axis=0)

        # Map Back to Original Image
        lmk = lmk + face_crop_lt

        return lmk, interval_det


# V3 use local eye-region image to predict eye landmarks
# V3 only support onnx for simplification
# V3 use face_detector to crop eye-region image
class LandmarkDetectorV3():
    def __init__(self, ckpt_path, use_onnx=True, use_filter=False, device='cuda:0'):
        self.face_detector = MediaPipeFaceDetector(model_select=1)

        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 4
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3

        if device == 'cpu':
            providers = ['CPUExecutionProvider']
        else:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self.net = onnxruntime.InferenceSession(ckpt_path, providers=providers, sess_options=options)

        if use_filter:
            # decrease min_cutoff, more stable
            # increase beta, follow faster
            self.filter_face = OneEuroFilter(min_cutoff=0.3, beta=0.01, d_cutoff=1.0)
            self.filter_leye = OneEuroFilter(min_cutoff=0.3, beta=0.02, d_cutoff=1.0)
            self.filter_reye = OneEuroFilter(min_cutoff=0.3, beta=0.02, d_cutoff=1.0)
            self.t_count = 0

        # warmup
        onnx_input = {
            self.net.get_inputs()[0].name: np.random.rand(1, 3, 256, 256).astype(np.float32),
            self.net.get_inputs()[1].name: np.random.rand(1, 3, 128, 128).astype(np.float32),
            self.net.get_inputs()[2].name: np.random.rand(1, 3, 128, 128).astype(np.float32)
        }
        _ = self.net.run(None, onnx_input)

        self.use_filter = use_filter
        self.use_onnx = use_onnx
        assert use_onnx, "Only Support onnx!"

    def reset(self):
        if self.use_filter:
            self.filter_face.reset()
            self.filter_leye.reset()
            self.filter_reye.reset()
            self.t_count = 0

    def crop_eye_region(self, img, lcenter, rcenter):
        lr_distance = np.linalg.norm(rcenter - lcenter)
        size = np.round(lr_distance * 0.33).astype(np.int32)

        lcenter = np.round(lcenter).astype(np.int32)
        lcrop_lt = lcenter - size
        lcrop_rb = lcenter + size
        leye_img_crop = pad_and_crop(img, lcrop_lt, lcrop_rb)
        leye_img_resize = cv2.resize(leye_img_crop, (128, 128))

        rcenter = np.round(rcenter).astype(np.int32)
        rcrop_lt = rcenter - size
        rcrop_rb = rcenter + size
        reye_img_crop = pad_and_crop(img, rcrop_lt, rcrop_rb)
        reye_img_resize = cv2.resize(reye_img_crop, (128, 128))

        return leye_img_resize, leye_img_crop, lcrop_lt, lcrop_rb, \
                reye_img_resize, reye_img_crop, rcrop_lt, rcrop_rb

    @torch.no_grad()
    def __call__(self, img):
        # Face Detection
        # assume the first face box is what we want
        st = time.time()
        face_box, key_points = self.face_detector(img, ret_kp=True)
        if face_box is None:
            return np.zeros([68+48+38, 2], np.float32), 0
        key_points[3, 1] += 50  # empirically finetune
        key_points[0, 1] -= 10  # empirically finetune
        key_points[1, 1] -= 10  # empirically finetune
        face_box = get_box_from_landmarks(key_points[:4], img.shape[0], img.shape[1], pad_scale=0.5)
        interval_det = (time.time() - st) * 1000

        # Crop Face Region
        face_crop_lt, face_crop_rb = get_crop_from_box(face_box)
        face_img_crop = pad_and_crop(img, face_crop_lt, face_crop_rb)
        face_img_resize = cv2.resize(face_img_crop, (256, 256))

        # Crop Eye Region
        leye_img_resize, leye_img_crop, leye_crop_lt, leye_crop_rb, \
        reye_img_resize, reye_img_crop, reye_crop_lt, reye_crop_rb \
            = self.crop_eye_region(img, key_points[0], key_points[1])
        
        # for i in range(6):
        #     x = int(key_points[i, 0] + 0.5) - face_crop_lt[0]
        #     y = int(key_points[i, 1] + 0.5) - face_crop_lt[1]
        #     cv2.circle(face_img_crop, (x, y), 2, (0, 255, 0), -1)
        # cv2.imshow("face_img_crop", cv2.cvtColor(face_img_crop, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(30)
        # cv2.imwrite("face_img_crop.png", cv2.cvtColor(face_img_crop, cv2.COLOR_RGB2BGR))
        # cv2.imwrite("leye_img_crop.png", cv2.cvtColor(leye_img_crop, cv2.COLOR_RGB2BGR))
        # cv2.imwrite("reye_img_crop.png", cv2.cvtColor(reye_img_crop, cv2.COLOR_RGB2BGR))
        # exit(0)

        face_img_onnx = face_img_resize.astype(np.float32)[np.newaxis, :].transpose((0, 3, 1, 2)) / 255
        leye_img_onnx = cv2.flip(leye_img_resize, flipCode=1).astype(np.float32)[np.newaxis, :].transpose((0, 3, 1, 2)) / 255
        reye_img_onnx = reye_img_resize.astype(np.float32)[np.newaxis, :].transpose((0, 3, 1, 2)) / 255

        onnx_input = {
            self.net.get_inputs()[0].name: face_img_onnx,
            self.net.get_inputs()[1].name: leye_img_onnx,
            self.net.get_inputs()[2].name: reye_img_onnx,
        }
        face_heatmaps, eye_heatmaps = self.net.run(None, onnx_input)
        face_lmk, face_scores = decode_landmarks(torch.from_numpy(face_heatmaps), gamma=1.0, radius=0.1)
        eye_lmks, eye_scores = decode_landmarks(torch.from_numpy(eye_heatmaps), gamma=1.0, radius=0.1)

        # xxx_crop      [H, W, C]
        # xxx_heatmaps  [B, C, H, W]
        face_lmk = face_lmk[0].detach().numpy() * np.array([face_img_crop.shape[-2] / face_heatmaps.shape[-1], face_img_crop.shape[-3] / face_heatmaps.shape[-2]])
        leye_lmk = eye_lmks[0].detach().numpy() * np.array([leye_img_crop.shape[-2] / eye_heatmaps.shape[-1], leye_img_crop.shape[-3] / eye_heatmaps.shape[-2]])
        reye_lmk = eye_lmks[1].detach().numpy() * np.array([reye_img_crop.shape[-2] / eye_heatmaps.shape[-1], reye_img_crop.shape[-3] / eye_heatmaps.shape[-2]])

        # Map Back to Original Face Image
        face_lmk = face_lmk + face_crop_lt
        leye_lmk[:, 0] = leye_img_crop.shape[-2] - 1 - leye_lmk[:, 0]   # flip back
        leye_lmk = leye_lmk + leye_crop_lt
        reye_lmk = reye_lmk + reye_crop_lt

        if self.use_filter:
            norm_factor = np.array([img.shape[1], img.shape[0]])
            face_lmk = self.filter_face(self.t_count, face_lmk / norm_factor) * norm_factor
            leye_lmk = self.filter_leye(self.t_count, leye_lmk / norm_factor) * norm_factor
            reye_lmk = self.filter_reye(self.t_count, reye_lmk / norm_factor) * norm_factor
            self.t_count += 1

        lmk = np.concatenate([face_lmk, leye_lmk[:19], reye_lmk[:19], leye_lmk[19:], reye_lmk[19:]], axis=0)

        return lmk, interval_det