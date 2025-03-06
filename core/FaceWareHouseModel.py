import torch
import torch.nn as nn
import numpy as np
from pytorch3d.structures import Meshes
from core.BaseModel import BaseReconModel
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
