from core.FaceWareHouseModel import FaceWarehouseReconModel
from scipy.io import loadmat
import torch
import struct
import numpy as np


def get_recon_model(model='model_i50_e47', **kargs):
    if model == 'model_i50_e47':
        model_path = 'FaceWareHouse/model_i50_e47.bin'
        with open(model_path, 'rb') as fptr:
            ud = struct.unpack('3i', fptr.read(12))
            model_array = np.zeros((ud[0], ud[1], ud[2]), dtype=np.float32)
            model_array = np.frombuffer(
                fptr.read(ud[0] * ud[1] * ud[2] * 4), dtype=np.float32)
            model_array = np.copy(model_array)
            model_array = torch.from_numpy(model_array).reshape(
                ud[2], ud[1], ud[0])  # 50, 47, 34530
        recon_model = FaceWarehouseReconModel(model_array, **kargs)
        return recon_model
    else:
        raise NotImplementedError()
