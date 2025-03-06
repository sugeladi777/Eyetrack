import torch
import torch.nn as nn
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    blending
)


class BaseReconModel(nn.Module):
    def __init__(self, batch_size=1,
                 focal=1015, img_size=512, device='cuda:0'):
        super(BaseReconModel, self).__init__()

        self.batch_size = batch_size
        self.img_size = img_size
        self.device = torch.device(device)

        self.focal = focal
        self.p_mat = self._get_p_mat(device)
        self.reverse_z = self._get_reverse_z(device)
        self.camera_pos = self._get_camera_pose(device)

        self.rot_tensor = None
        self.trans_tensor = None
        self.id_init_tensor = torch.tensor(
            [0.0816002, -0.00200936, 0.00144346, -0.00122638, 3.41109e-006, 0.000488255, -0.000264426, 0.000311428, -0.000101785, 7.52266e-005,
             -8.91395e-005, -3.3526e-005, 4.18179e-005, 4.89653e-005, 5.29148e-006, 1.70476e-005, 3.64553e-005, -
             1.57865e-005, 2.2001e-007, 1.7283e-005,
             3.26759e-005, 1.42219e-005, 1.05463e-005, -2.50615e-005, 1.27146e-005, 1.56672e-006, -
             8.5737e-006, 2.91094e-005, -1.94864e-005, 4.18188e-006,
             1.0258e-005, 8.23654e-006, 2.98166e-005, 1.68243e-006, 2.10065e-005, 7.28352e-006, -
             6.17266e-006, -2.22083e-006, -1.78076e-005, -1.50372e-005,
             -8.59465e-007, 1.5521e-005, -3.57433e-006, -9.40569e-006, 1.3817e-005, 7.26787e-006, -1.84304e-005, -4.06662e-006, -1.37241e-006, -1.06965e-005],
            dtype=torch.float32, requires_grad=True, device=self.device).reshape(1, -1)
        self.id_tensor = self.id_init_tensor

        self.exp_init_tensor = torch.tensor(
            [0.145831, -0.00119506, 0.00248977, -0.00555244, -0.0636425, 0.00551691, 0.0483311, -0.031428, -0.00887051, 0.0125705,
             0.0661306, 0.0104712, -0.00596938, -0.00125942, 0.071329, -
             0.00245361, 0.0310085, 0.00739647, 0.0184184, 0.00647163,
             -0.00596638, 0.0110645, 0.0591841, -0.0477235, 0.0048617, -
             0.0104558, 0.0108776, 0.0222797, -0.0137854, 0.00955756,
             0.00425438, -0.00370248, -0.0238861, -0.000651003, 0.000525417, -
             0.121418, -0.000577637, -0.0414738, 0.0422691, 0.029649,
             -0.107663, 0.02445, 0.0152015, 0.0935797, -0.790362, -0.0118333, -0.536054],
            dtype=torch.float32, requires_grad=True, device=self.device).reshape(1, -1)
        self.exp_tensor = self.exp_init_tensor


        self.init_coeff_dims()

        self.init_coeff_tensors(id_coeff=self.id_tensor, exp_coeff=self.exp_tensor,
                                rot_tensor=self.rot_tensor, trans_tensor=self.trans_tensor)

    def _get_camera_pose(self, device):
        camera_pos = torch.tensor(
            [0.0, 0.0, 10.0], device=device).reshape(1, 1, 3)
        return camera_pos

    def _get_p_mat(self, device):
        half_image_width = self.img_size // 2
        p_matrix = np.array([self.focal, 0.0, half_image_width,
                             0.0, self.focal, half_image_width,
                             0.0, 0.0, 1.0], dtype=np.float32).reshape(1, 3, 3)
        return torch.tensor(p_matrix, device=device)

    def _get_reverse_z(self, device):
        reverse_z = np.reshape(
            np.array([1.0, 0, 0, 0, 1, 0, 0, 0, -1.0], dtype=np.float32), [1, 3, 3])
        return torch.tensor(reverse_z, device=device)

    def compute_norm(self, vs, tri, point_buf):

        face_id = tri
        point_id = point_buf
        v1 = vs[:, face_id[:, 0], :]
        v2 = vs[:, face_id[:, 1], :]
        v3 = vs[:, face_id[:, 2], :]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = e1.cross(e2)
        empty = torch.zeros((face_norm.size(0), 1, 3),
                            dtype=face_norm.dtype, device=face_norm.device)
        face_norm = torch.cat((face_norm, empty), 1)
        v_norm = face_norm[:, point_id, :].sum(2)
        v_norm = v_norm / v_norm.norm(dim=2).unsqueeze(2)

        return v_norm

    def project_vs(self, vs):
        batchsize = vs.shape[0]

        vs = torch.matmul(vs, self.reverse_z.repeat(
            (batchsize, 1, 1))) + self.camera_pos

        aug_projection = torch.matmul(
            vs, self.p_mat.repeat((batchsize, 1, 1)).permute((0, 2, 1)))

        face_projection = aug_projection[:, :, :2] / \
            torch.reshape(aug_projection[:, :, 2], [batchsize, -1, 1])
        return face_projection

    def compute_rotation_matrix(self, angles):
        n_b = angles.size(0)
        sinx = torch.sin(angles[:, 0])
        siny = torch.sin(angles[:, 1])
        sinz = torch.sin(angles[:, 2])
        cosx = torch.cos(angles[:, 0])
        cosy = torch.cos(angles[:, 1])
        cosz = torch.cos(angles[:, 2])

        rotXYZ = torch.eye(3).view(1, 3, 3).repeat(
            n_b * 3, 1, 1).view(3, n_b, 3, 3).to(angles.device)

        rotXYZ[0, :, 1, 1] = cosx
        rotXYZ[0, :, 1, 2] = -sinx
        rotXYZ[0, :, 2, 1] = sinx
        rotXYZ[0, :, 2, 2] = cosx
        rotXYZ[1, :, 0, 0] = cosy
        rotXYZ[1, :, 0, 2] = siny
        rotXYZ[1, :, 2, 0] = -siny
        rotXYZ[1, :, 2, 2] = cosy
        rotXYZ[2, :, 0, 0] = cosz
        rotXYZ[2, :, 0, 1] = -sinz
        rotXYZ[2, :, 1, 0] = sinz
        rotXYZ[2, :, 1, 1] = cosz

        rotation = rotXYZ[2].bmm(rotXYZ[1]).bmm(rotXYZ[0])

        return rotation.permute(0, 2, 1)

    def rigid_transform(self, vs, rot, trans):

        vs_r = torch.matmul(vs, rot)
        vs_t = vs_r + trans.view(-1, 1, 3)

        return vs_t

    def get_lms(self, vs):
        raise NotImplementedError()

    def forward(self, coeffs, render=True):
        raise NotImplementedError()

    def get_vs(self, id_coeff, exp_coeff):
        raise NotImplementedError()

    def get_rot_tensor(self):
        return self.rot_tensor

    def get_trans_tensor(self):
        return self.trans_tensor

    def get_exp_tensor(self):
        return self.exp_tensor

    def get_id_tensor(self):
        return self.id_tensor

    def init_coeff_dims(self):
        raise NotImplementedError()

    def get_id_init_tensor(self):
        return self.id_init_tensor

    def get_exp_init_tensor(self):
        return self.exp_init_tensor

    def get_packed_tensors(self):
        return self.merge_coeffs(self.id_tensor.repeat(self.batch_size, 1),
                                 self.exp_tensor,
                                 self.rot_tensor,
                                 self.trans_tensor)

    def init_coeff_tensors(self, id_coeff,
                           exp_coeff, rot_tensor, trans_tensor):
        if id_coeff is None:
            self.id_tensor = torch.zeros(
                (1, self.id_dims), dtype=torch.float32,
                requires_grad=True, device=self.device)
        else:
            assert id_coeff.shape == (1, self.id_dims)
            self.id_tensor = id_coeff.clone().detach().to(self.device).requires_grad_(True)

        if exp_coeff is None:
            self.exp_tensor = torch.zeros(
                (self.batch_size, self.exp_dims), dtype=torch.float32,
                requires_grad=True, device=self.device)
        else:
            assert exp_coeff.shape == (self.batch_size, self.exp_dims)
            self.exp_tensor = exp_coeff.clone().detach().to(self.device).requires_grad_(True)
        if rot_tensor is None:
            self.rot_tensor = torch.zeros(
                (self.batch_size, 3), dtype=torch.float32,
                requires_grad=True, device=self.device)
        else:
            assert rot_tensor.shape == (self.batch_size, 3)
            self.rot_tensor = rot_tensor.clone().detach().to(self.device).requires_grad_(True)
        if trans_tensor is None:
            self.trans_tensor = torch.zeros(
                (self.batch_size, 3), dtype=torch.float32,
                requires_grad=True, device=self.device)
        else:
            assert trans_tensor.shape == (self.batch_size, 3)
            self.trans_tensor = trans_tensor.clone().detach().to(self.device).requires_grad_(True)