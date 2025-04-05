import numpy as np
import cv2

class MyEllipse:
    def __init__(self):
        # 椭圆参数
        self.x_off_ = 0.0  # x 偏移
        self.y_off_ = 0.0  # y 偏移
        self.a_ = 1.0      # 长轴
        self.b_ = 1.0      # 短轴
        self.theta_ = 0.0  # 旋转角度

        # 其他参数
        self.para_reg_weight_ = 1.0
        self.center_reg_weight_ = 1.0

        # 样本点
        self.sample_ = []
        self.sample_num_ = 0

    def getA(self):
        return abs(self.a_)

    def getB(self):
        return abs(self.b_)

    def getXOff(self):
        return self.x_off_

    def getYOff(self):
        return self.y_off_

    def getTheta(self):
        return self.theta_

    def getABT(self):
        """
        获取椭圆的参数：长短轴比、旋转角度、x 偏移、y 偏移。
        """
        adb = self.a_ / self.b_
        return [adb, self.theta_, self.x_off_, self.y_off_]

    def ObjFunc(self, params):
        """
        椭圆拟合的目标函数。
        :param params: 椭圆参数 [a, b, theta, x_off, y_off]
        :return: 误差矩阵
        """
        a, b, theta, x_off, y_off = params.flatten()
        residuals = []
        for point in self.sample_:
            x, y = point[0, 0], point[1, 0]
            x_rot = (x - x_off) * np.cos(theta) + (y - y_off) * np.sin(theta)
            y_rot = -(x - x_off) * np.sin(theta) + (y - y_off) * np.cos(theta)
            residual = (x_rot / a) ** 2 + (y_rot / b) ** 2 - 1
            residuals.append(residual)
        return np.array(residuals).reshape(-1, 1)

    def RotObjFunc(self, params):
        """
        旋转相关的目标函数。
        """
        # 示例实现，具体逻辑根据需求调整
        return self.ObjFunc(params)

    def RotABObjFunc(self, params):
        """
        长短轴和旋转相关的目标函数。
        """
        # 示例实现，具体逻辑根据需求调整
        return self.ObjFunc(params)

    def fit(self, sample):
        """
        拟合椭圆参数。
        :param sample: 样本点列表，每个点为 2D 坐标。
        """
        self.sample_ = sample
        self.sample_num_ = len(sample)

        # 初始化参数
        params = np.array([self.a_, self.b_, self.theta_, self.x_off_, self.y_off_], dtype=np.float32)

        # 使用优化器（如高斯牛顿法）拟合
        def objective_function(params):
            return self.ObjFunc(params).flatten()

        from scipy.optimize import least_squares
        result = least_squares(objective_function, params)
        self.a_, self.b_, self.theta_, self.x_off_, self.y_off_ = result.x

    def getPoints(self, x_arr):
        """
        根据 x 坐标生成椭圆上的点。
        :param x_arr: x 坐标数组
        :return: 椭圆上的点列表
        """
        points = []
        for x in x_arr:
            y = self.b_ * np.sqrt(1 - (x / self.a_) ** 2)
            points.append(np.array([x, y]))
        return points

    def draw(self, cv_img, center, id):
        """
        在图像上绘制椭圆。
        :param cv_img: OpenCV 图像
        :param center: 椭圆中心
        :param id: 椭圆 ID
        """
        center = (int(center[0]), int(center[1]))
        axes = (int(self.a_), int(self.b_))
        angle = np.degrees(self.theta_)
        cv2.ellipse(cv_img, center, axes, angle, 0, 360, (0, 255, 0), 2)

    def reset(self):
        """
        重置椭圆参数。
        """
        self.x_off_ = 0.0
        self.y_off_ = 0.0
        self.a_ = 1.0
        self.b_ = 1.0
        self.theta_ = 0.0

# 适配器类
class EllipseAdapter:
    def __init__(self, ellipse):
        self.ellipse = ellipse

    def __call__(self, params):
        return self.ellipse.ObjFunc(params)

class EllipseRotAdapter:
    def __init__(self, ellipse):
        self.ellipse = ellipse

    def __call__(self, params):
        return self.ellipse.RotObjFunc(params)

class EllipseRotABAdapter:
    def __init__(self, ellipse):
        self.ellipse = ellipse

    def __call__(self, params):
        return self.ellipse.RotABObjFunc(params)