from SenseTimeLandmark import SenseTimeLandmark
import trimesh
import torch
import cv2
import numpy as np
import math
from PIL import Image
import gc
from scipy.spatial.transform import Rotation as R
import os
import sys


class TextureUnit:
    def __init__(self):
        """
        初始化 TextureUnit 类的成员变量。
        """
        self.texImgW = 0          # 纹理图像的宽度
        self.texImgH = 0          # 纹理图像的高度
        self.texImgBPP = 0        # 每像素的位数（Bits Per Pixel）
        self.texImgData = None    # 保存纹理图像的 numpy 数组


class EyeballModel:
    def __init__(self):
        # 成员变量初始化
        self.eyeballMesh = None
        self.sphere = None
        self.irisVertexNum = 0
        self.irisFaceNum = 0
        self.irisTex = TextureUnit()
        self.corneaTex = TextureUnit()
        self.useTexture = False
        self.irisPhi = 0.25 * (math.pi / 2)

        # 使用两个 3D 向量表示左右眼位置
        self.eyePos = [np.zeros(3, dtype=np.float32),
                       np.zeros(3, dtype=np.float32)]
        self.pupilPos = [np.zeros(3, dtype=np.float32),
                         np.zeros(3, dtype=np.float32)]

    def initialize(self, fileName, irisTexName, corneaTexName):
        self.loadObject(fileName)
        self.loadTexture(self.irisTex, irisTexName)
        self.loadTexture(self.corneaTex, corneaTexName)
        if self.irisTex is not None and self.corneaTex is not None:
            self.useTexture = True

    def loadObject(self, fileName):
        """
        使用 trimesh 加载 3D 模型文件。
        :param fileName: 3D 模型文件路径
        """
        if not fileName:
            print("Load eyeball object failed.")
            return
        try:
            # 使用 trimesh 加载模型
            self.eyeballMesh = trimesh.load(fileName, force='mesh')
            print(
                f"Loaded mesh with {len(self.eyeballMesh.vertices)} vertices and {len(self.eyeballMesh.faces)} faces.")
        except Exception as e:
            print(f"Failed to load mesh: {e}")
            self.eyeballMesh = None

    def loadTexture(self, textUnit: TextureUnit, texImgName: str):
        """
        加载纹理图像，并初始化 TextureUnit 对象的属性。
        :param textUnit: TextureUnit 对象
        :param texImgName: 纹理图像文件路径
        """
        if not texImgName:
            print("No texture for eyeball.")
            return

        try:
            # 使用 Pillow 加载图像
            img = Image.open(texImgName)
            img = img.convert("RGBA")  # 转换为 RGBA 格式，确保有 4 通道
            textUnit.texImgW, textUnit.texImgH = img.size
            textUnit.texImgBPP = 32  # 每像素 32 位（RGBA 格式）
            textUnit.texImgData = np.array(img)  # 转换为 numpy 数组
        except Exception as e:
            print(f"Cannot load image {texImgName}: {e}")
            textUnit.texImgW = 0
            textUnit.texImgH = 0
            textUnit.texImgBPP = 0
            textUnit.texImgData = None

    def updateTextureCordinates(self, irisP: float):
        """
        更新纹理坐标。
        :param irisP: 新的虹膜角度
        """
        if self.eyeballMesh is None or self.eyeballMesh.visual.uv is None:
            print("Eyeball mesh or texture coordinates not loaded.")
            return

        print(f"Set iris size: {irisP}")
        rate = self.irisPhi / irisP
        center = np.array([0.5, 0.5], dtype=np.float32)

        # 更新纹理坐标
        uv_coords = self.eyeballMesh.visual.uv
        uv_coords = center + (uv_coords - center) * rate
        uv_coords = np.clip(uv_coords, 0, 1)
        self.eyeballMesh.visual.uv = uv_coords
        self.irisPhi = irisP
        print("Updated texture coordinates.")

    def updateEyeballSize(self, radius: float):
        """
        更新眼球大小。
        :param radius: 新的眼球半径
        """
        if self.eyeballMesh is None:
            print("Eyeball mesh not loaded.")
            return

        # 计算当前顶点的平均半径
        vertices = self.eyeballMesh.vertices
        current_radius = np.linalg.norm(vertices, axis=1).mean()

        # 缩放顶点以匹配新的半径
        scale_factor = radius / current_radius
        self.eyeballMesh.vertices *= scale_factor
        print(f"Updated eyeball size to radius {radius}.")

    def setEyeballPos(self, et: list):
        """
        设置左右眼的位置。
        :param et: 包含左右眼位置的列表或数组，长度为 6。
                   et[0:3] 表示左眼位置，et[3:6] 表示右眼位置。
        """
        if len(et) != 6:
            raise ValueError("Input 'et' must have exactly 6 elements.")

        # 设置左眼位置
        self.eyePos[0] = np.array(et[0:3], dtype=np.float32)

        # 设置右眼位置
        self.eyePos[1] = np.array(et[3:6], dtype=np.float32)

    def phiTheta2Rotation(self, phi: float, theta: float) -> np.ndarray:
        """
        根据 phi 和 theta 计算旋转矩阵。
        :param phi: 上下角度（绕 X 轴旋转）
        :param theta: 左右角度（绕 Y 轴旋转）
        :return: 旋转矩阵（3x3 的 numpy 数组）
        """
        # 绕 Y 轴的旋转矩阵
        matY = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ], dtype=np.float32)

        # 绕 X 轴的旋转矩阵
        matX = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ], dtype=np.float32)

        # 组合旋转矩阵
        ret = matY @ matX
        return ret


class EyeballFitter:
    def __init__(self):
        # --------------------
        # 图像及几何参数
        # --------------------
        self.face_Rvec_ = None
        self.face_Tvec_ = None
        self.camera_flength_ = None
        self.img_width_ = 0
        self.img_height_ = 0

        # --------------------
        # 眼球参数
        # --------------------
        self.eyeRadius_ = 0.12
        self.eyeIrisPhi_ = 0.42
        self.eyePhi_ = [0.0, 0.0]    # 左右眼上下角度
        self.eyeTheta_ = [0.0, 0.0]  # 左右眼左右角度

        # --------------------
        # 采样、平移及标志点信息
        # --------------------
        self.eyeSample_ = []           # 球面采样点
        self.eyeTrans_ = []            # 眼睛平移向量
        self.eyeTrans_backup_ = []   # 眼睛平移向量备份
        self.eyeTrans_delta_backup_ = []  # 平移向量增量备份
        self.eye_lmk_id_ = []          # 眼部关键点索引
        self.eyeAABB_ = []             # 眼部的轴对齐边界框
        self.irisLandMark_ = []        # 虹膜关键点数据
        self.landmark_ellipse_ = []       # 眼部椭圆拟合数据
        self.eyeIrisBoundary_ = []      # 虹膜边界数据
        self.EYE_NUM = 2
        self.LONGI_NUM2 = 19
        self.eyeball_ = EyeballModel()  # 眼球模型对象

        # --------------------
        # 眼部图像及裁剪信息
        # --------------------
        self.face_img_ = None
        self.eye_img_ = []             # 存储单个眼部图像
        self.eyeCrop_ = []             # 存储眼部图像裁剪区域
        self.eye_center_2d_ = []       # 眼部中心点（2D）

        # --------------------
        # 其它参数
        # --------------------
        self.eye_center_dis_ = 0.0     # 两眼中心点距离

        # 构造时初始化成员变量
        self.init()

    def init(self):
        """初始化各个参数以及默认眼部标志点和平移向量"""
        self.eye_lmk_id_.clear()
        self.eyeTrans_.clear()
        self.eyeTrans_backup_.clear()
        self.eyePhi_.clear()
        self.eyeTheta_.clear()
        self.irisLandMark_.clear()
        self.landmark_ellipse_.clear()

        if self.eyeball_ is not None:
            self.eyeball_.initialize("core/Eyeball/EyeballModel/eyeball_128_128.obj",
                                     "core/Eyeball/EyeballModel/eye_iris_.png",
                                     "core/Eyeball/EyeballModel/eye_iris_.png")

        # 定义左右眼的关键点索引
        left_lmk_id = [36, 37, 38, 39, 40, 41]
        right_lmk_id = [42, 43, 44, 45, 46, 47]
        self.eye_lmk_id_.append(left_lmk_id)
        self.eye_lmk_id_.append(right_lmk_id)

        # 设置默认的眼球平移向量
        self.eyeRadius_ = 0.12
        self.eyeIrisPhi_ = 0.42
        left_trans = np.array(
            [[-0.283510238], [0.471765947], [0.195450214]], dtype=np.float32)
        right_trans = np.array([[0.283510238], [0.471765947], [
                               0.195450214]], dtype=np.float32)
        self.eyeTrans_.append(left_trans)
        self.eyeTrans_.append(right_trans)
        self.eyeTrans_backup_.append(left_trans.copy())
        self.eyeTrans_backup_.append(right_trans.copy())
        self.eyeTrans_delta_backup_ = [left_trans.copy(), right_trans.copy()]
        self.irisLandMark_ = []
        self.eyePhi_ = [0.0, 0.0]
        self.eyeTheta_ = [0.0, 0.0]
        self.landmark_ellipse_ = [None, None]

    def setIrisLandmark(self, lmk):
        """
        设置虹膜关键点数据。
        :param lmk: 包含虹膜关键点的列表，每个关键点是一个 2x1 的 numpy 数组。
        """
        self.irisLandMark_ = lmk

    def getEyeAABB(self, lmk2d: list):
        """
        根据2D关键点计算每只眼的轴对齐边界框和中心点，
        同时计算两只眼中心点之间的距离。
        """
        self.eyeAABB_.clear()
        self.eye_center_2d_.clear()
        for e in range(len(self.eye_lmk_id_)):
            pts = np.array([lmk2d[i].reshape(2)
                           for i in self.eye_lmk_id_[e]], dtype=np.float32)
            x_coords, y_coords = pts[:, 0], pts[:, 1]
            left, top = int(min(x_coords)), int(min(y_coords))
            right, bottom = int(max(x_coords)), int(max(y_coords))
            self.eyeAABB_.append((left, top, right - left, bottom - top))
            center_2d = np.mean(pts, axis=0).reshape((2, 1))
            self.eye_center_2d_.append(center_2d)

        if len(self.eye_center_2d_) == 2:
            dx = self.eye_center_2d_[0][0, 0] - self.eye_center_2d_[1][0, 0]
            dy = self.eye_center_2d_[0][1, 0] - self.eye_center_2d_[1][1, 0]
            self.eye_center_dis_ = float(np.sqrt(dx * dx + dy * dy))

    def sampleEye2(self, iris_phi: float):
        """
        根据虹膜角度 iris_phi 生成左右眼球面采样点。
        采用球面坐标公式计算采样点。
        """
        self.eyeSample_.clear()
        self.eyeIrisBoundary_.clear()

        phi = iris_phi
        deltaTheta = 2 * np.pi / self.LONGI_NUM2

        # 左眼采样：theta 从 2π 递减
        for j in range(self.LONGI_NUM2):
            theta = 2 * np.pi - j * deltaTheta
            sample = np.array([
                self.eyeRadius_ * np.sin(phi) * np.cos(theta),
                self.eyeRadius_ * np.sin(phi) * np.sin(theta),
                self.eyeRadius_ * np.cos(phi)
            ], dtype=np.float32)
            self.eyeSample_.append(sample)

        # 右眼采样：theta 从 π 递增
        for j in range(self.LONGI_NUM2):
            theta = np.pi + j * deltaTheta
            sample = np.array([
                self.eyeRadius_ * np.sin(phi) * np.cos(theta),
                self.eyeRadius_ * np.sin(phi) * np.sin(theta),
                self.eyeRadius_ * np.cos(phi)
            ], dtype=np.float32)
            self.eyeSample_.append(sample)

    def phiTheta2Rotation(self, phi: float, theta: float) -> np.ndarray:
        """
        根据 phi 和 theta 计算旋转矩阵。
        :param phi: 上下角度（绕 X 轴旋转）
        :param theta: 左右角度（绕 Y 轴旋转）
        :return: 旋转矩阵（3x3 的 numpy 数组）
        """
        # 绕 Y 轴的旋转矩阵
        matY = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ], dtype=np.float32)

        # 绕 X 轴的旋转矩阵
        matX = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ], dtype=np.float32)

        # 组合旋转矩阵
        ret = matY @ matX
        return ret

    def iris_phi_eye_trans_obj(self, params,eye_mode):
        """
        根据输入参数计算虹膜角度和眼睛平移参数，生成眼球采样点的 3D 坐标，并投影到 2D 图像平面。
        :param params: 包含虹膜角度和眼睛平移参数的 numpy 数组，形状为 (3,)。
        :return: 投影后的 2D 坐标矩阵。
        """
        phi = self.eyePhi_[eye_mode]
        theta = self.eyeTheta_[eye_mode]

        # 计算旋转矩阵
        rmat = self.phiTheta2Rotation(phi, theta)
        rmatF = R.from_rotvec(self.face_Rvec_.flatten()).as_matrix()

        # 更新虹膜角度并生成采样点
        self.sampleEye2(params[0])  # iris_phi

        # 构造平移向量
        trans = np.zeros((3, 1), dtype=np.float32)
        trans[0, 0] = params[1]
        trans[1, 0] = params[2]
        trans[2, 0] = self.eyeTrans_[eye_mode][2, 0]

        # 计算采样点的 3D 坐标
        sample_transformed = []
        start = 0 if eye_mode == 0 else len(self.eyeSample_) // self.EYE_NUM
        end = len(self.eyeSample_) // self.EYE_NUM if eye_mode == 0 else len(self.eyeSample_)
        for s in range(start, end):
            sample = self.eyeSample_[s]
            if sample.shape == (3,):  # 如果是 1D 数组，转换为 2D 列向量
                sample = sample.reshape(3, 1)
            temp = rmatF @ (rmat @ sample + trans) + self.face_Tvec_
            temp = temp.reshape(3, 1)  # 确保 temp 的形状为 (3, 1)
            sample_transformed.append(temp)

        # 投影到 2D 图像平面
        proj_key_vertices = self.project_3d_to_2d(
            sample_transformed, self.camera_flength_, self.img_width_, self.img_height_)

        # 返回结果
        return torch.tensor(np.vstack(proj_key_vertices), dtype=torch.float32, requires_grad=True)

    def project_3d_to_2d(self, points_3d, flength, img_width, img_height):
        """
        将 3D 点投影到 2D 图像平面。
        :param points_3d: 3D 点列表，每个点是一个 numpy 数组，形状为 (3, 1)。
        :param flength: 相机焦距。
        :param img_width: 图像宽度。
        :param img_height: 图像高度。
        :return: 投影后的 2D 点列表。
        """
        # 构造投影矩阵
        pro_mat = np.array([
            [flength, 0, img_width / 2],
            [0, flength, img_height / 2],
            [0, 0, 1]
        ], dtype=np.float32)

        points_2d = []
        for point in points_3d:
            # 确保点是 (3, 1) 的形状
            if point.shape != (3, 1):
                raise ValueError(f"Invalid point shape: {point.shape}, expected (3, 1)")

            # 投影到 2D
            p2d = pro_mat @ point
            x, y, z = p2d.flatten()
            if z != 0:
                u = img_width - (x / z)
                v = img_height - (y / z)
                points_2d.append(np.array([[u], [v]], dtype=np.float32))
        return points_2d

    def fitBothEyes(self, phi_left: float, theta_left: float, phi_right: float, theta_right: float, iter: int = 200):
        """
        使用 LBFGS 优化器同时拟合左右眼的虹膜角度（eyeIrisPhi_）和眼睛平移参数（eyeTrans_）。
        """
        # 初始化左右眼的角度
        self.eyePhi_[0] = phi_left
        self.eyeTheta_[0] = theta_left
        self.eyePhi_[1] = phi_right
        self.eyeTheta_[1] = theta_right

        # 初始化优化参数：左右眼的虹膜角度和眼睛平移（x, y）
        params = torch.tensor([
            self.eyeIrisPhi_,  # 左眼虹膜角度
            self.eyeTrans_[0][0, 0],  # 左眼平移 x
            self.eyeTrans_[0][1, 0],  # 左眼平移 y
            self.eyeIrisPhi_,  # 右眼虹膜角度
            self.eyeTrans_[1][0, 0],  # 右眼平移 x
            self.eyeTrans_[1][1, 0]   # 右眼平移 y
        ], dtype=torch.float32, requires_grad=True)

        # 准备目标输出（假设 irisLandMark_ 存储了目标数据）
        outputs = np.vstack(self.irisLandMark_)
        outputs = torch.tensor(outputs, dtype=torch.float32, requires_grad=False)

        # 定义目标函数
        def objective_function(params):
            # 左眼参数
            iris_phi_left = params[0]
            trans_x_left = params[1]
            trans_y_left = params[2]

            # 右眼参数
            iris_phi_right = params[3]
            trans_x_right = params[4]
            trans_y_right = params[5]

            # 计算预测值
            predicted_left = self.iris_phi_eye_trans_obj(
                torch.tensor([iris_phi_left, trans_x_left, trans_y_left]), eye_mode=0)
            predicted_right = self.iris_phi_eye_trans_obj(
                torch.tensor([iris_phi_right, trans_x_right, trans_y_right]), eye_mode=1)

            predicted = torch.cat([predicted_left, predicted_right], dim=0)

            # 计算损失
            loss = torch.sum((predicted - outputs) ** 2)
            return loss

        # 使用 LBFGS 优化器
        optimizer = torch.optim.LBFGS(
            [params], lr=0.01, line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            loss = objective_function(params)
            loss.backward()
            return loss

        # 优化迭代
        for _ in range(iter):
            optimizer.step(closure)

        # 更新结果
        self.eyeIrisPhi_ = params[0].item()
        self.eyeTrans_[0][0, 0] = params[1].item()
        self.eyeTrans_[0][1, 0] = params[2].item()
        self.eyeIrisPhi_ = params[3].item()
        self.eyeTrans_[1][0, 0] = params[4].item()
        self.eyeTrans_[1][1, 0] = params[5].item()

        # 打印结果
        print(
            f"Left Eye - Iris Phi: {params[0].item()}, Position: ({params[1].item()}, {params[2].item()})")
        print(
            f"Right Eye - Iris Phi: {params[3].item()}, Position: ({params[4].item()}, {params[5].item()})")

    def getCropRect(self, rect):
        """
        根据输入的边界框 rect = (left, top, width, height) 计算扩展后的裁剪区域。
        返回值：(left, top, crop_w, crop_h)
        """
        left, top, w, h = rect
        right, bottom = left + w, top + h
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        scale_base = max(h // 2, w // 3)
        scale_base = int(1.8 * scale_base)
        top = center_y - scale_base
        left = center_x - (scale_base * 3) // 2
        crop_w, crop_h = 3 * scale_base, 2 * scale_base
        return (left, top, crop_w, crop_h)

    def setEyeImg(self, cv_img, lmk2d):
        """
        根据输入图像和关键点，获取眼部图像区域。
        同时计算眼睛的边界框和裁剪区域，并存储到 eye_img_ 与 eyeCrop_。
        """
        self.eye_img_.clear()
        self.eyeCrop_.clear()
        self.getEyeAABB(lmk2d)

        for e in range(len(self.eyeAABB_)):
            (lx, ty, w, h) = self.getCropRect(self.eyeAABB_[e])
            lx = max(0, lx)
            ty = max(0, ty)
            rx = min(cv_img.shape[1], lx + w)
            by = min(cv_img.shape[0], ty + h)
            self.eye_img_.append(cv_img[ty:by, lx:rx].copy())
            self.eyeCrop_.append((lx, ty, rx - lx, by - ty))
        self.face_img_ = cv_img.copy()

    def fit2(self, cv_img, lmk2d, rvec, tvec, f: float):
        self.face_Rvec_ = rvec
        self.face_Tvec_ = tvec
        self.camera_flength_ = f
        self.img_width_ = cv_img.shape[1]
        self.img_height_ = cv_img.shape[0]

        self.setEyeImg(cv_img, lmk2d)

        self.fitBothEyes(0.0, 0.0, 0.0, 0.0, 3)

        sample_params = torch.tensor([
            self.eyePhi_[0], self.eyePhi_[1],
            self.eyeTheta_[0], self.eyeTheta_[1]
        ], dtype=torch.float32, requires_grad=True)

        optimizer = torch.optim.LBFGS(
            [sample_params], lr=0.01, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = (sample_params ** 2).sum()
            loss.backward()
            return loss

        max_iter = 200
        for _ in range(max_iter):
            optimizer.step(closure)

        self.eyePhi_[0] = sample_params[0].item()
        self.eyePhi_[1] = sample_params[1].item()
        self.eyeTheta_[0] = sample_params[2].item()
        self.eyeTheta_[1] = sample_params[3].item()

        print(f"LP: {self.eyePhi_[0]}")
        print(f"LT: {self.eyeTheta_[0]}")
        print(f"RP: {self.eyePhi_[1]}")
        print(f"RT: {self.eyeTheta_[1]}")

    def save(self, out_dir: str, filename: str):
        """
        使用 trimesh 保存带颜色的眼球模型。
        :param out_dir: 输出目录
        :param filename: 文件名
        """
        # 将旋转向量转换为旋转矩阵
        rmatF = R.from_rotvec(self.face_Rvec_.flatten()).as_matrix()

        for e in range(2):  # 遍历左右眼
            # 计算眼球的旋转矩阵
            eye_rot = self.eyeball_.phiTheta2Rotation(
                self.eyePhi_[e], self.eyeTheta_[e])

            # 获取顶点和纹理坐标
            vertices = self.eyeball_.eyeballMesh.vertices.copy()
            tex_coords = self.eyeball_.eyeballMesh.visual.uv

            # 更新顶点位置
            for i in range(len(vertices)):
                vertices[i] = (
                    rmatF @ (eye_rot @ vertices[i] +
                             self.eyeball_.eyePos[e]) + self.face_Tvec_.flatten()
                )

            # 创建颜色数组
            vertex_colors = np.zeros(
                (len(vertices), 4), dtype=np.uint8)  # RGBA 格式
            for i in range(len(vertices)):
                # 获取纹理坐标
                u = int((self.eyeball_.irisTex.texImgW - 1)
                        * tex_coords[i][0] + 0.5)
                v = int((self.eyeball_.irisTex.texImgH - 1)
                        * tex_coords[i][1] + 0.5)

                # 确保纹理坐标在有效范围内
                u = np.clip(u, 0, self.eyeball_.irisTex.texImgW - 1)
                v = np.clip(v, 0, self.eyeball_.irisTex.texImgH - 1)

                # 从纹理图像中获取颜色
                color = self.eyeball_.irisTex.texImgData[v, u]
                vertex_colors[i] = [color[0], color[1],
                                    color[2], 255]  # 添加 alpha 通道

            # 创建 trimesh 对象
            mesh = trimesh.Trimesh(
                vertices=vertices, faces=self.eyeball_.eyeballMesh.faces, vertex_colors=vertex_colors)

            # 保存 mesh 到文件
            output_path = os.path.join(out_dir, f"{filename}_eyeball_{e}.ply")
            mesh.export(output_path)
            print(f"Colored mesh saved to {output_path}")


def loadFit2Inputs(file_path):
    with open(file_path, "rb") as file:

        # 加载旋转向量
        rvec_rows = int(np.frombuffer(file.read(4), dtype=np.int32))
        rvec_cols = int(np.frombuffer(file.read(4), dtype=np.int32))
        rvec_type = int(np.frombuffer(file.read(4), dtype=np.int32))
        rvec_data = np.frombuffer(
            file.read(rvec_rows * rvec_cols * 4), dtype=np.float32)  # 假设是 float32 类型
        rvec = rvec_data.reshape((rvec_rows, rvec_cols))

        # 加载平移向量
        tvec_rows = int(np.frombuffer(file.read(4), dtype=np.int32))
        tvec_cols = int(np.frombuffer(file.read(4), dtype=np.int32))
        tvec_type = int(np.frombuffer(file.read(4), dtype=np.int32))
        tvec_data = np.frombuffer(
            file.read(tvec_rows * tvec_cols * 4), dtype=np.float32)  # 假设是 float32 类型
        tvec = tvec_data.reshape((tvec_rows, tvec_cols))

    return rvec, tvec


def loadLandmark2D(file_path):
    """
    从文件中读取 2D 关键点数据。
    文件格式：
    第一行：关键点数量
    接下来的行：每个关键点的 (x, y) 坐标，可能多个点的坐标在同一行
    :param file_path: 关键点文件路径
    :return: 关键点列表，每个关键点是一个 2x1 的 numpy 数组
    """
    lmk2d = []
    with open(file_path, "r") as file:
        while True:
            # 读取点的数量
            line = file.readline().strip()
            if not line:  # 文件结束
                break
            num_points = int(line)  # 当前块的点数量

            # 读取点的坐标
            coords = []
            while len(coords) < num_points * 2:  # 每个点有两个值 (x, y)
                line = file.readline().strip()
                coords.extend(map(float, line.split()))

            # 将坐标转换为 2x1 的 numpy 数组
            for i in range(num_points):
                x, y = coords[2 * i], coords[2 * i + 1]
                lmk2d.append(np.array([x, y], dtype=np.float32).reshape(2, 1))
    return lmk2d


rvec, tvec = loadFit2Inputs(
    "/home/lichengkai/Eyetrack/core/Eyeball/parameters")

cv_img = cv2.imread("image/frame_00001.jpg")
lmk2d = loadLandmark2D(
    "/home/lichengkai/Eyetrack/landmark/landmark/landmark_1.txt")
f = 2000

eyeball_fitter = EyeballFitter()
# 初始化 EyeballFitter 对象
eyeball_fitter.init()
# 进行拟合
st_lmk = SenseTimeLandmark()
st_lmk.loadSenseTimeLandmark("/home/lichengkai/Eyetrack/landmark", 1, 1)
eyeball_fitter.setIrisLandmark(st_lmk.irisLandMark_)
eyeball_fitter.fit2(cv_img, lmk2d, rvec, tvec, f)
# 保存结果
eyeball_fitter.save("/home/lichengkai/Eyetrack/landmark",
                    "1")
