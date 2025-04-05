import torch
import torch.nn as nn
import numpy as np
import cv2
import face_alignment
from facenet_pytorch import MTCNN
from tqdm import tqdm
import os
from core.Face import utils
from core.Face import losses
from pytorch3d.structures import Meshes
from core.Face.BaseModel import BaseReconModel
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
)


class FaceWarehouseReconModel(BaseReconModel):
    def __init__(self, model_array, **kwargs):
        super(FaceWarehouseReconModel, self).__init__(**kwargs)
        self.model = model_array.to(self.device)
        # 加载三角面片索引
        tri_raw = []
        with open('FaceWareHouse/face_tri.obj', 'r') as file:
            for line in file:
                if line.startswith('f '):
                    _, x, y, z = line.strip().split()
                    tri_raw.append([int(x)-1, int(y)-1, int(z)-1])
        self.tri = torch.tensor(tri_raw, dtype=torch.int64, device=self.device)

        # 加载关键点索引
        points_raw = []
        with open('FaceWareHouse/68_ldm.txt', 'r') as file:
            for line in file:
                point = line.strip().split()
                points_raw.append([int(point[0])])
        self.kp_inds = torch.tensor(
            points_raw, dtype=torch.int64, device=self.device)
        self.kp_inds = self.kp_inds.squeeze()

    def get_lms(self, vs):
        lms = vs[:, self.kp_inds, :]
        return lms

    def split_coeffs(self, coeffs):
        id_coeff = coeffs[:, :50]        # 身份系数，维度 50
        exp_coeff = coeffs[:, 50:97]     # 表情系数，维度 47
        angles = coeffs[:, 97:100]      # 旋转角度，维度 3
        translation = coeffs[:, 100:]    # 平移向量，维度 3
        return id_coeff, exp_coeff, angles, translation

    def merge_coeffs(self, id_coeff, exp_coeff, angles, translation):
        coeffs = torch.cat([id_coeff, exp_coeff,
                            angles, translation], dim=1)
        return coeffs

    def forward(self, coeffs):
        batch_num = coeffs.shape[0]
        id_coeff, exp_coeff, angles, translation = self.split_coeffs(
            coeffs)
        vs = self.get_vs(id_coeff, exp_coeff)
        rotation = self.compute_rotation_matrix(angles)
        vs_t = self.rigid_transform(vs, rotation, translation)
        lms_t = self.get_lms(vs_t)
        lms_proj = self.project_vs(lms_t)
        lms_proj = torch.stack(
            [lms_proj[:, :, 0], self.img_size - lms_proj[:, :, 1]], dim=2)
        return {'lms_proj': lms_proj,
                    'vs': vs_t,
                    'tri': self.tri}

    def get_vs(self, id_coeff, exp_coeff):
        n_b = id_coeff.size(0)
        bs = torch.einsum('bi,ijk->bjk', id_coeff, self.model)  # [batch_size, exp_dims, num_vertices * 3]
        face_shape = torch.einsum('bjk,bj->bk', bs, exp_coeff)  # [batch_size, num_vertices * 3]
        face_shape = face_shape.view(n_b, -1, 3)
        face_shape = face_shape - face_shape.mean(dim=1, keepdim=True)  # 去中心化
        return face_shape

    def init_coeff_dims(self):
        self.id_dims = 50
        self.exp_dims = 47
        self.rot_dims = 3
        self.trans_dims = 3

    def fit(args):
        # init face detection and lms detection models
        print('loading models')
        mtcnn = MTCNN(device=args.device, select_largest=False)
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D, flip_input=False)
        recon_model = get_recon_model(model=args.recon_model,
                                    device=args.device,
                                    batch_size=1,
                                    img_size=args.tar_size)

        print('loading images')
        img_arr = cv2.imread(args.img_path)[:, :, ::-1]
        orig_h, orig_w = img_arr.shape[:2]
        print('image is loaded. width: %d, height: %d' % (orig_w, orig_h))

        # detect the face using MTCNN
        bboxes, probs = mtcnn.detect(img_arr)
        if bboxes is None:
            print('no face detected')
        else:
            bbox = utils.pad_bbox(bboxes[0], (orig_w, orig_h), args.padding_ratio)
            face_w = bbox[2] - bbox[0]
            face_h = bbox[3] - bbox[1]
            assert face_w == face_h
        print('A face is detected. l: %d, t: %d, r: %d, b: %d'
            % (bbox[0], bbox[1], bbox[2], bbox[3]))
        face_img = img_arr[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        resized_face_img = cv2.resize(face_img, (args.tar_size, args.tar_size))
        utils.save_resized_face_image(resized_face_img, args.res_folder)
        lms = fa.get_landmarks_from_image(resized_face_img)[0]
        lms = lms[:, :2][None, ...]
        lms = torch.tensor(lms, dtype=torch.float32, device=args.device)
        print('landmarks detected.')

        lm_pose_weights = utils.get_pose_lm_weights(args.device)
        lm_exp_weights = utils.get_exp_lm_weights(args.device)

        print('start rigid fitting')
        rigid_optimizer = torch.optim.LBFGS(
            [recon_model.get_rot_tensor(), recon_model.get_trans_tensor()],
            lr=args.rf_lr,
            line_search_fn="strong_wolfe"
        )

        def rigid_closure():
            rigid_optimizer.zero_grad()
            pred_dict = recon_model(recon_model.get_packed_tensors())
            lm_loss_val = losses.lm_loss(
                pred_dict['lms_proj'], lms, lm_pose_weights, img_size=args.tar_size)
            total_loss = args.lm_loss_w * lm_loss_val
            total_loss.backward()
            return total_loss

        for _ in tqdm(range(args.first_rf_iters)):
            rigid_optimizer.step(rigid_closure)

        with torch.no_grad():
            pred_dict = recon_model(recon_model.get_packed_tensors())
            lm_loss_val = losses.lm_loss(
                pred_dict['lms_proj'], lms, lm_pose_weights, img_size=args.tar_size)
        print('done rigid fitting. lm_loss: %f' %
            lm_loss_val.detach().cpu().numpy())

        print('start non-rigid fitting')
        nonrigid_optimizer = torch.optim.LBFGS(
            [recon_model.get_id_tensor(),
            recon_model.get_exp_tensor(),
            recon_model.get_rot_tensor(),
            recon_model.get_trans_tensor()],
            lr=args.nrf_lr,
            line_search_fn="strong_wolfe"
        )

        def nonrigid_closure():
            nonrigid_optimizer.zero_grad()
            pred_dict = recon_model(recon_model.get_packed_tensors())
            lm_loss_val = losses.lm_loss(pred_dict['lms_proj'], lms,
                                        lm_exp_weights, img_size=args.tar_size)
            id_reg_loss = losses.get_l2(recon_model.get_id_tensor(),
                                        recon_model.get_id_init_tensor())
            exp_reg_loss = losses.get_l2(recon_model.get_exp_tensor(),
                                        recon_model.get_exp_init_tensor())
            loss = lm_loss_val*args.lm_loss_w + \
                id_reg_loss*args.id_reg_w + \
                exp_reg_loss*args.exp_reg_w
            loss.backward()
            return loss

        for _ in tqdm(range(args.first_nrf_iters)):
            nonrigid_optimizer.step(nonrigid_closure)

        with torch.no_grad():
            pred_dict = recon_model(recon_model.get_packed_tensors())
            lm_loss_val = losses.lm_loss(pred_dict['lms_proj'], lms,
                                        lm_exp_weights, img_size=args.tar_size)
            id_reg_loss = losses.get_l2(
                recon_model.get_id_tensor(), recon_model.get_id_init_tensor())
            exp_reg_loss = losses.get_l2(
                recon_model.get_exp_tensor(), recon_model.get_exp_init_tensor())

        loss_str = 'lm_loss: %f\tid_reg_loss: %f\texp_reg_loss: %f\t' % (
            lm_loss_val.detach().cpu().numpy(),
            id_reg_loss.detach().cpu().numpy(),
            exp_reg_loss.detach().cpu().numpy()
        )
        print('done non rigid fitting.', loss_str)

        with torch.no_grad():
            coeffs = recon_model.get_packed_tensors()
            pred_dict = recon_model(coeffs)
            basename = os.path.basename(args.img_path)[:-4]
            out_obj_path = os.path.join(
                args.res_folder, basename+'_mesh.obj')
            vs = pred_dict['vs'].cpu().numpy().squeeze()
            tri = pred_dict['tri'].cpu().numpy().squeeze()
            utils.save_obj(out_obj_path, vs, tri+1)

        lms_np = lms.cpu().numpy().squeeze()
        utils.mark_landmarks_on_image(lms_np, resized_face_img, args.res_folder)
