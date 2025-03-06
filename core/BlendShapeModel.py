import yaml
import glob
import os
import numpy as np
import cv2

EYE_NUM = 2
EYELID_NUM = 4
CURVE_SAMPLE_NUM = 12
CURVE_PARA_NUM = 4
EYELID_WID_NUM = 20
EYELID_WEXP_NUM = 2
EYELID_WEXPLUS_NUM = 17


class EyelidCurve:
    def __init__(self, params=None, xl=0, xr=1, energy=0, valid=False):
        if params is None:
            self.params = np.zeros(CURVE_PARA_NUM, dtype=np.float32)
        else:
            self.params = params
        self.xl = xl
        self.xr = xr
        self.energy = energy
        self.valid = valid

    def copyFrom(self, ec):
        self.params = np.copy(ec.params)
        self.valid = ec.valid
        self.xl = ec.xl
        self.xr = ec.xr
        self.energy = ec.energy

    def getY(self, tx):
        ty = 0
        temp = 1
        for j in range(self.params.shape[0]):
            ty += temp * self.params[j]
            temp *= tx
        return ty

    def getPoint(self, tx, imgW, imgH):
        ty = self.getY(tx)
        ret = np.array([tx * imgW, ty * imgH], dtype=np.float32)
        return ret


class TriMesh:
    def __init__(self):
        self.vertices = None
        self.faces = None
        self.normals = None

    def test(self, filename):
        vertices = []
        with open(filename, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 3:
                    try:
                        vertices.append(
                            [float(parts[0]), float(parts[1]), float(parts[2])])
                    except ValueError:
                        print(
                            f"Warning: Skipping line with non-float values: {line.strip()}")
                else:
                    print(f"Warning: Skipping malformed line: {line.strip()}")
        self.setVertices_(vertices)

    def loadMesh(self, path):

        vertices = []
        normals = []
        faces = []
        with open(path, 'r') as file:
            for line in file:
                if line.startswith('v '):
                    parts = line.split()
                    vertices.append(
                        [float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('vn '):
                    parts = line.split()
                    normals.append(
                        [float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('f '):
                    parts = line.split()
                    face = []
                    for token in parts[1:]:
                        face.append(int(token.split('/')[0]) - 1)
                    faces.append(face)
        self.vertices = np.array(vertices, dtype=np.float32)
        self.normals = np.array(normals, dtype=np.float32)
        self.faces = np.array(faces, dtype=np.int32)
        print(
            f"Loaded mesh from {path} with {len(vertices)} vertices and {len(faces)} faces.")

    def copyFrom(self, other):
        self.vertices = np.copy(other.vertices)
        self.faces = np.copy(other.faces)
        self.normals = np.copy(other.normals)

    def getAllVertices_(self):
        return self.vertices

    def getAllVertexNormals_(self):
        return self.normals

    def setVertices_(self, vertices):
        self.vertices = vertices

    def setVertexNormals_(self, normals):
        self.normals = normals

    def getVertexNum(self):
        return self.vertices.shape[0]

    def saveMesh(self, path):
        with open(path, 'w') as f:
            for v in self.vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in self.faces:
                face_str = " ".join(str(idx + 1) for idx in face)
                f.write(f"f {face_str}\n")
        print(f"Saved mesh to {path}")


class EyelidBlender:
    def __init__(self):
        self.n_id_ = 0
        self.n_exp_ = 0
        self.core_ = []
        self.core_backup_ = None
        self.deform_to_face_id_ = []
        self.mask_ = []
        self.uMatrix = None
        self.eyelid_lr_id_ = []
        self.face_eyelid_corr_ = {}
        self.eyelid_face_corr_ = {}
        self.face_Rvec_ = np.zeros((3, 1), dtype=np.float32)
        self.face_Tvec_ = np.zeros((3, 1), dtype=np.float32)
        self.camera_flength_ = 0
        self.img_width_ = 0
        self.img_height_ = 0
        self.eye_img_ = []
        self.eyeCrop_ = []
        self.eyeAABB_ = []
        self.face_img_ = None
        self.eye_curve_ = [[], []]
        self.eye_lmk_id_ = [[36, 37, 38, 39, 40, 41],[42, 43, 44, 45, 46, 47]]
        self.eyeLandMark_ = []

    def init(self):
        self.loadBlendshape("Blendshape")
        self.loadMask("Blendshape/eyelid_mask_modified.txt")
        self.loadDeformToFace("Blendshape/eyelid_deform.txt")
        self.loadUMatrix("Blendshape/UMatrix.txt")
        self.load_eyelid_lr_id("Blendshape/eyelid_mask_left.txt",
                               "Blendshape/eyelid_mask_right.txt")
        self.load_eyelid_border_corr("Blendshape/eyelid_border_corr.txt")

    def load_eyelid_border_corr(self, pathCor):
        with open(pathCor, 'r') as finCor:
            for line in finCor:
                face_vid, eyelid_vid = map(int, line.strip().split())
                self.face_eyelid_corr_[face_vid] = eyelid_vid
                self.eyelid_face_corr_[eyelid_vid] = face_vid

    def load_eyelid_lr_id(self, pathl, pathr):
        with open(pathl, 'r') as finL:
            vidL = [int(line.strip()) for line in finL]
            self.eyelid_lr_id_.append(vidL)
        with open(pathr, 'r') as finR:
            vidR = [int(line.strip()) for line in finR]
            self.eyelid_lr_id_.append(vidR)

    def loadUMatrix(self, path):
        with open(path, 'r') as fin:
            lines = fin.readlines()
            matrix = []
            for line in lines:
                values = list(map(float, line.strip().split()))
                matrix.append(values)
            self.uMatrix = np.array(matrix)
            self.uMatrix.reshape((len(matrix), len(matrix[0])))
            print(f"Loaded UMatrix with shape {self.uMatrix.shape}")

    def loadBlendshape(self, path):
        filesExp, filesId = [], []
        basePath = os.path.join(path, "base")

        self.getFileNames(basePath, filesExp)
        filesExp.sort()
        self.n_exp_ = len(filesExp)
        print(f"Total: {self.n_exp_} exp blendshapes")

        self.getFileNames(path, filesId)
        filesId.sort()
        self.n_id_ = len(filesId) // self.n_exp_ + 1
        print(f"Total: {self.n_id_} id blendshapes")
        self.core_ = [[TriMesh() for _ in range(self.n_exp_)]
                      for _ in range(self.n_id_)]
        for i in range(self.n_exp_):
            print(f"loading base: {filesExp[i]}")
            self.core_[0][i].loadMesh(os.path.join(basePath, filesExp[i]))

        for i in range(1, self.n_id_):
            for j in range(self.n_exp_):
                index = (i - 1) * self.n_exp_ + j
                print(f"loading {filesId[index]}")
                self.core_[i][j].loadMesh(os.path.join(path, filesId[index]))
                self.applyBlendshapeDiff(self.core_[0][j], self.core_[i][j])

        for j in range(1, self.n_exp_):
            for i in range(self.n_id_):
                self.applyBlendshapeDiff(self.core_[i][0], self.core_[i][j])

        self.core_backup_ = TriMesh()
        self.core_backup_.copyFrom(self.core_[0][0])

    def loadMask(self, path):
        mask = []
        with open(path, 'r') as file:
            for line in file:
                parts = line.split()
                mask.append(int(parts[0]))
        self.mask_ = mask

    def loadDeformToFace(self, pathId, dimId=150):
        print("loadDeformToFace")

        self.deform_to_face_id_.clear()

        vtNum = self.core_[0][0].getVertexNum()
        maskVtNum = len(self.mask_)

        with open(pathId, 'r') as finId:
            line = finId.readline().strip().split()
            values = list(map(float, line))
        data_index = 0
        while len(self.deform_to_face_id_) < dimId:
            if len(self.deform_to_face_id_) % 10 == 0:
                print(f"Processing {len(self.deform_to_face_id_)}", end='\r')

            maskVt = np.zeros((vtNum, 3), dtype=np.float32)
            for i in range(maskVtNum):
                maskVt[self.mask_[i]] = [values[data_index],
                                         values[data_index + 1], values[data_index + 2]]
                data_index += 3
            self.deform_to_face_id_.append([maskVt])
        assert len(self.deform_to_face_id_) == dimId

    def getFileNames(self, path, files):
        for ext in ['*.obj', '*.OBJ']:
            for filename in glob.glob(os.path.join(path, ext)):
                if os.path.isfile(filename):
                    files.append(os.path.basename(filename))

    def applyBlendshapeDiff(self, base, dst):
        vtBase = base.getAllVertices_()
        vtDst = dst.getAllVertices_()
        vtDiff = vtDst - vtBase

        vtBaseNorm = base.getAllVertexNormals_()
        vtDstNorm = dst.getAllVertexNormals_()
        vtDiffNorm = vtDstNorm - vtBaseNorm

        dst.setVertices_(vtDiff)
        dst.setVertexNormals_(vtDiffNorm)

    def applyDeformToFace(self, mesh, widFace):
        vtNum = self.core_backup_.getAllVertices_().shape[0]
        vtBuf = np.zeros((vtNum, 3), dtype=np.float32)
        deform = np.zeros((vtNum, 3), dtype=np.float32)
        vtMesh = self.core_backup_.getAllVertices_()

        for i in range(widFace.shape[0]):
            deform = self.deform_to_face_id_[i][0]
            vtBuf += widFace[i] * deform

        mesh.setVertices_(vtBuf)
        mesh.saveMesh("output.obj")
    def setEyelidLandmark(self, lmk):
        self.eyeLandMark_ = lmk
        
    def fit(self, cv_img, lmk2d, rvec, tvec, f):
        self.face_Rvec_ = rvec
        self.face_Tvec_ = tvec
        self.camera_flength_ = f
        self.img_width_ = cv_img.shape[1]
        self.img_height_ = cv_img.shape[0]

        self.setEyeImg(cv_img, lmk2d)
        for e in range(EYE_NUM):
            self.fitCurve(e)
            self.calcIntersection(e)
            self.generateLandmarkXpos(e)

        if self.frame_ == 0:
            self.subfit_wid()
            self.drawResult(cv_img)
        self.subfit_wexp()

        self.calcCloseRate()

        self.frame_ += 1

    def setEyeImg(self, cv_img, lmk2d):
        self.eye_img_ = []
        self.eyeCrop_ = []
        self.getEyeAABB(lmk2d)

        crop_L = self.getCropRect(self.eyeAABB_[0])
        crop_R = self.getCropRect(self.eyeAABB_[1])

        self.eye_img_.append(
            cv_img[crop_L[1]:crop_L[1]+crop_L[3], crop_L[0]:crop_L[0]+crop_L[2]])
        self.eye_img_.append(
            cv_img[crop_R[1]:crop_R[1]+crop_R[3], crop_R[0]:crop_R[0]+crop_R[2]])
        self.eyeCrop_.append(crop_L)
        self.eyeCrop_.append(crop_R)
        self.face_img_ = cv_img.copy()

    def getEyeAABB(self, lmk2d):
        self.eyeAABB_ = []
        eye_center_2d = []
        for e in range(EYE_NUM):
            center_2d = np.zeros((2, 1), dtype=np.float32)
            top = np.iinfo(np.int32).max
            left = np.iinfo(np.int32).max
            bottom = 0
            right = 0
            for l in range(len(self.eye_lmk_id_[e])):
                lmk_id = self.eye_lmk_id_[e][l]
                center_2d += lmk2d[lmk_id]
                x = int(lmk2d[lmk_id][0, 0] + 0.5)
                y = int(lmk2d[lmk_id][1, 0] + 0.5)
                left = min(left, x)
                top = min(top, y)
                right = max(right, x)
                bottom = max(bottom, y)
            center_2d /= len(self.eye_lmk_id_[e])
            eye_center_2d.append(center_2d)
            self.eyeAABB_.append((left, top, right - left, bottom - top))

        self.eye_center_dis_ = np.sqrt(
            (eye_center_2d[0][0, 0] - eye_center_2d[1][0, 0]) ** 2 +
            (eye_center_2d[0][1, 0] - eye_center_2d[1][1, 0]) ** 2
        )
        self.eye_center_dis_ = np.sqrt(
            (eye_center_2d[0][0, 0] - eye_center_2d[1][0, 0]) ** 2 +
            (eye_center_2d[0][1, 0] - eye_center_2d[1][1, 0]) ** 2
        )

    def getCropRect(self, rec):
        left = rec[0]
        top = rec[1]
        right = left + rec[2]
        bottom = top + rec[3]
        centerX = (left + right) // 2
        centerY = (top + bottom) // 2
        h = rec[3]
        w = rec[2]
        extentionScale = 1.8
        scaleBase = int(h / 2.0 if h / 2.0 > w /
                        3.0 else w / 3.0 * extentionScale)
        top = centerY - scaleBase
        left = centerX - int(1.5 * scaleBase)
        h = 2 * scaleBase
        w = int(3 * scaleBase)

        cropRect = (left, top, w, h)
        return cropRect

    def fitCurve(self, eye):
        self.eye_curve_[eye] = []
        start = 0 if eye == 0 else len(self.eyeLandMark_) // 2
        end = len(self.eyeLandMark_) // 2 if eye == 0 else len(self.eyeLandMark_)
        mid = (start + end) // 2

        offset = np.array(
            [self.eyeCrop_[eye][0], self.eyeCrop_[eye][1]], dtype=np.float32)
        imgW = self.eye_img_[eye].shape[1]
        imgH = self.eye_img_[eye].shape[0]

        A = np.zeros((CURVE_SAMPLE_NUM, CURVE_PARA_NUM), dtype=np.float32)
        Y = np.zeros(CURVE_SAMPLE_NUM, dtype=np.float32)
        P = np.zeros(CURVE_PARA_NUM, dtype=np.float32)
        ctr = 0
        leftmost = 1.0
        rightmost = 0.0

        # Top Edge
        for i in range(mid, end):
            x = (self.eyeLandMark_[i][0, 0] - offset[0]) / imgW
            y = (self.eyeLandMark_[i][1, 0] - offset[1]) / imgH
            leftmost = min(leftmost, x)
            rightmost = max(rightmost, x)
            temp = 1.0
            for j in range(CURVE_PARA_NUM):
                A[ctr, j] = temp
                temp *= x
            Y[ctr] = y
            ctr += 1

        ATA = np.dot(A.T, A)
        ATY = np.dot(A.T, Y)
        P = np.linalg.solve(ATA, ATY)
        self.eye_curve_[eye].append(EyelidCurve(
            P, leftmost, rightmost, 1.0))  # lid 0
        self.eye_curve_[eye].append(EyelidCurve(
            P, leftmost, rightmost, 1.0))  # lid 1

        # Bottom Edge
        ctr = 0
        leftmost = 1.0
        rightmost = 0.0
        for i in range(start, mid):
            x = (self.eyeLandMark_[i][0, 0] - offset[0]) / imgW
            y = (self.eyeLandMark_[i][1, 0] - offset[1]) / imgH
            leftmost = min(leftmost, x)
            rightmost = max(rightmost, x)
            temp = 1.0
            for j in range(CURVE_PARA_NUM):
                A[ctr, j] = temp
                temp *= x
            Y[ctr] = y
            ctr += 1

        ATA = np.dot(A.T, A)
        ATY = np.dot(A.T, Y)
        P = np.linalg.solve(ATA, ATY)
        self.eye_curve_[eye].append(EyelidCurve(
            P, leftmost, rightmost, 1.0))  # lid 2
        self.eye_curve_[eye].append(EyelidCurve(False))  # lid 3

    def calcIntersection(self, eye):
        lm, rm = 0.0, 0.0
        offset = np.array(
            [self.eyeCrop_[eye][0], self.eyeCrop_[eye][1]], dtype=np.float32)
        imgW = self.eye_img_[eye].shape[1]
        imgH = self.eye_img_[eye].shape[0]

        if eye == 0:
            lmp = (self.eyeLandMark_[0] + self.eyeLandMark_[12]) / 2 - offset
            rmp = (self.eyeLandMark_[11] + self.eyeLandMark_[23]) / 2 - offset
            lm = lmp[0] / imgW
            rm = rmp[0] / imgW
        else:
            lmp = (self.eyeLandMark_[35] + self.eyeLandMark_[47]) / 2 - offset
            rmp = (self.eyeLandMark_[24] + self.eyeLandMark_[36]) / 2 - offset
            lm = lmp[0] / imgW
            rm = rmp[0] / imgW

        self.eye_curve_[eye][1].xl = lm
        self.eye_curve_[eye][2].xl = lm

        self.eye_curve_[eye][1].xr = rm
        self.eye_curve_[eye][2].xr = rm

    def generateLandmarkXpos(self, eye):
        eye_img = self.eye_img_[eye]
        offset = np.array(
            [self.eyeCrop_[eye][0], self.eyeCrop_[eye][1]], dtype=np.float32)
        imgW = eye_img.shape[1]
        imgH = eye_img.shape[0]
        vertexNum = self.core_[0][0].getVertexNum()
        rmatF, _ = cv2.Rodrigues(self.face_Rvec_)

        vtx = np.zeros((vertexNum, 3), dtype=np.float32)
        self.synthesis(vtx, self.eyeWids_[eye], self.eyeWexps_[eye])

        self.eye_landmark_[eye] = [[] for _ in range(EYELID_NUM)]

        for lid in range(EYELID_NUM):  # only Top, Bottom curve
            if not self.eye_curve_[eye][lid].valid:
                continue
            self.eye_landmark_[eye][lid] = []
            landmark_num = len(self.eyelid_vertex_id_[eye][lid])
            p3d = []
            for i in range(landmark_num):
                vid = self.eyelid_vertex_id_[eye][lid][i]
                temp = np.dot(rmatF, vtx[vid]) + self.face_Tvec_.flatten()
                p3d.append(temp)
            p2d = self.project3Dto2D(
                p3d, self.camera_flength_, self.img_width_, self.img_height_)

            for i in range(landmark_num):
                tx = (p2d[i][0] - offset[0]) / imgW
                temp = self.eye_curve_[eye][lid].getPoint(tx, imgW, imgH)
                self.eye_landmark_[eye][lid].append(temp + offset)

    def subfit_wid(self):
        # 实现 subfit_wid 方法
        pass

    def drawResult(self, cv_img):
        # 实现 drawResult 方法
        pass

    def subfit_wexp(self):
        # 实现 subfit_wexp 方法
        pass

    def calcCloseRate(self):
        # 实现 calcCloseRate 方法
        pass

    def saveWholeFaceEyelidFace(self, out_dir, filename, faceMesh):
        eyelid_mesh = TriMesh()
        eyelid_mesh.copyFrom(self.core_[0][0])
        eyelid_vert = eyelid_mesh.vertices.copy()
        vertexNum = eyelid_mesh.getVertexNum()

        rmatF, _ = cv2.Rodrigues(self.face_Rvec_)
        for i in range(vertexNum):
            p3d = eyelid_vert[i].reshape(3, 1)
            p3d = np.dot(rmatF, p3d).flatten() + self.face_Tvec_.flatten()
            eyelid_vert[i] = p3d

        merged_eyelid = TriMesh()
        merged_eyelid.copyFrom(self.core_[0][0])
        merged_eyelid.setVertices_(eyelid_vert)

        eyelid_vert = merged_eyelid.getAllVertices_().copy()
        face_vert = faceMesh.getAllVertices_().copy()

        for eyelid_vid, face_vid in self.eyelid_face_corr_.items():
            eyelid_vert[eyelid_vid] = (
                face_vert[face_vid] + eyelid_vert[eyelid_vid]) / 2.0

        final_vert = np.vstack([eyelid_vert, face_vert])

        eyelid_faces = merged_eyelid.faces.copy()
        face_faces = faceMesh.faces.copy()

        offset = eyelid_vert.shape[0]
        for f in range(face_faces.shape[0]):
            for c in range(3):
                vid = face_faces[f, c]
                if vid in self.face_eyelid_corr_:
                    face_faces[f, c] = self.face_eyelid_corr_[vid]
                else:
                    face_faces[f, c] = vid + offset

        final_faces = np.vstack([eyelid_faces, face_faces])

        save_path = os.path.join(out_dir, f"{filename}_eyelidface.obj")
        with open(save_path, 'w') as fout:
            for v in final_vert:
                fout.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in final_faces:
                fout.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

        print(f"Saved merged eyelid-face mesh to {save_path}")



def opencv_matrix_constructor(loader, node):
    """自定义 YAML 构造器，用于解析 OpenCV 矩阵"""
    mapping = loader.construct_mapping(node, deep=True)
    rows = mapping['rows']
    cols = mapping['cols']
    dt = mapping['dt']
    data = mapping['data']
    
    # 根据数据类型转换数组
    if dt == 'f':
        mat = np.array(data, dtype=np.float32)
    elif dt == 'd':
        mat = np.array(data, dtype=np.float64)
    elif dt == 'u':
        mat = np.array(data, dtype=np.uint8)
    else:
        mat = np.array(data)
    
    # 重塑矩阵
    mat = mat.reshape((rows, cols))
    return mat

def load_parameters(yaml_file):
    """加载参数文件"""
    # 注册自定义标签
    yaml.SafeLoader.add_constructor('!!opencv-matrix', opencv_matrix_constructor)
    yaml.SafeLoader.add_constructor('tag:yaml.org,2002:opencv-matrix', opencv_matrix_constructor)
    
    with open(yaml_file, 'r') as f:
        params = yaml.safe_load(f)
    return params

params = load_parameters('parameters.yml')

# 提取数据
lmk2d = [np.array(lmk) for lmk in params['lmk2d']]
rvec = params['rvec']
tvec = params['tvec']
f = params['f']
eyeLandMark = [np.array(lmk) for lmk in params['eyeLandMark']]

# 读取图像
cv_img = cv2.imread("002.jpg")

# 初始化并调用 fit
test = EyelidBlender()
test.setEyelidLandmark(eyeLandMark)
test.init()
test.fit(cv_img, lmk2d, rvec, tvec, f)
