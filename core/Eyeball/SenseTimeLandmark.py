import os
import cv2
import numpy as np


class OneEuroFilter:
    """
    OneEuroFilter 类，用于平滑关键点数据。
    """
    def __init__(self):
        self.last_time = None
        self.last_value = None
        self.dx = None
        self.min_cutoff = 0.04
        self.beta = 0.003
        self.d_cutoff = 1.0

    def set_parameters(self, time, value, dx, min_cutoff, beta, d_cutoff):
        self.last_time = time
        self.last_value = value
        self.dx = dx
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

    def filter(self, time, value):
        if self.last_time is None:
            self.last_time = time
            self.last_value = value
            return value

        dt = time - self.last_time
        alpha = self.compute_alpha(dt, self.d_cutoff)
        dx = (value - self.last_value) / dt if dt > 0 else 0
        self.dx = self.dx + alpha * (dx - self.dx)

        cutoff = self.min_cutoff + self.beta * abs(self.dx)
        alpha = self.compute_alpha(dt, cutoff)
        smoothed_value = self.last_value + alpha * (value - self.last_value)

        self.last_time = time
        self.last_value = smoothed_value
        return smoothed_value

    def compute_alpha(self, dt, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)


class SenseTimeLandmark:
    def __init__(self):
        """
        初始化 SenseTimeLandmark 类。
        """
        self.face_filter_x_ = [OneEuroFilter() for _ in range(68)]
        self.face_filter_y_ = [OneEuroFilter() for _ in range(68)]
        self.faceLandMark_ = []
        self.eyelidLandMark_ = []
        self.irisLandMark_ = []
        self.status_ = []

    def loadSenseTimeLandmark(self, pre_path, frame, frame_type):
        """
        加载 SenseTime 的关键点数据。
        :param pre_path: 关键点文件的路径前缀。
        :param frame: 当前帧编号。
        :param frame_type: 帧类型。
        :return: 当前帧的状态。
        """
        self.faceLandMark_.clear()
        self.eyelidLandMark_.clear()
        self.irisLandMark_.clear()

        filename = os.path.join(pre_path, f"landmark/landmark_{frame}.txt")
        if frame_type == 0:
            self.status_.clear()
            status_file = os.path.join(pre_path, "status.txt")
            if os.path.exists(status_file):
                with open(status_file, "r") as fin:
                    self.status_ = [int(line.strip()) for line in fin]
            else:
                print("[ERROR] No Status File!")
                return -1

        if not os.path.exists(filename):
            print(f"[ERROR] Landmark file not found: {filename}")
            return -1

        with open(filename, "r") as fin:
            # 加载面部关键点
            lmk_num = int(fin.readline().strip())  # 读取关键点数量
            face_line = fin.readline().strip()  # 读取包含所有关键点的行
            face_coords = list(map(float, face_line.split()))  # 将所有坐标解析为浮点数
            for i in range(lmk_num):
                x, y = face_coords[2 * i], face_coords[2 * i + 1]
                self.faceLandMark_.append(np.array([[x], [y]], dtype=np.float32))

            # 加载眼睑关键点
            lmk_num = int(fin.readline().strip())  # 读取关键点数量
            eyelid_line = fin.readline().strip()  # 读取包含所有关键点的行
            eyelid_coords = list(map(float, eyelid_line.split()))  # 将所有坐标解析为浮点数
            for i in range(lmk_num):
                x, y = eyelid_coords[2 * i], eyelid_coords[2 * i + 1]
                self.eyelidLandMark_.append(np.array([[x], [y]], dtype=np.float32))

            # 加载虹膜关键点
            lmk_num = int(fin.readline().strip())  # 读取关键点数量
            iris_line = fin.readline().strip()  # 读取包含所有关键点的行
            iris_coords = list(map(float, iris_line.split()))  # 将所有坐标解析为浮点数
            for i in range(lmk_num):
                x, y = iris_coords[2 * i], iris_coords[2 * i + 1]
                self.irisLandMark_.append(np.array([[x], [y]], dtype=np.float32))

        print(f"Face landmark: {len(self.faceLandMark_)}")
        print(f"Eyelid landmark: {len(self.eyelidLandMark_)}")
        print(f"Iris landmark: {len(self.irisLandMark_)}")

        # 应用 OneEuroFilter
        if frame_type >= 3:
            if frame == 0:
                for i in range(68):
                    self.face_filter_x_[i].set_parameters(
                        frame, self.faceLandMark_[i][0, 0], 0.0, 0.04, 0.003, 1.0
                    )
                    self.face_filter_y_[i].set_parameters(
                        frame, self.faceLandMark_[i][1, 0], 0.0, 0.04, 0.003, 1.0
                    )
            elif frame > 0:
                for i in range(68):
                    self.faceLandMark_[i][0, 0] = self.face_filter_x_[i].filter(
                        frame, self.faceLandMark_[i][0, 0]
                    )
                    self.faceLandMark_[i][1, 0] = self.face_filter_y_[i].filter(
                        frame, self.faceLandMark_[i][1, 0]
                    )

        return self.status_[frame] if frame < len(self.status_) else -1
    def getFaceLandmark(self):
        """
        获取面部关键点。
        :return: 面部关键点列表。
        """
        return self.faceLandMark_

    def getEyelidLandmark(self):
        """
        获取眼睑关键点。
        :return: 眼睑关键点列表。
        """
        return self.eyelidLandMark_

    def getIrisLandmark(self):
        """
        获取虹膜关键点。
        :return: 虹膜关键点列表。
        """
        return self.irisLandMark_