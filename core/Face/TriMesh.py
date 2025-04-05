import numpy as np


class TriMesh:
    def __init__(self):
        self.vertices = np.empty((0, 3), dtype=np.float32)  # 初始化为空的 numpy 数组
        self.faces = np.empty((0, 3), dtype=np.int32)  # 初始化为空的 numpy 数组
        self.normals = np.empty((0, 3), dtype=np.float32)  # 初始化为空的 numpy 数组
        self.adjacent_faces = None  # 每个顶点的邻接面
        self.adjacent_vertices = None  # 每个顶点的邻接顶点
        self.vertex_texture_ = []  # 存储顶点纹理坐标的列表
        self.vertex_colors = []  # 存储每个顶点的颜色，格式为 [R, G, B]

    def initTopology(self):
        """
        初始化网格的拓扑结构，包括邻接面和邻接顶点。
        """
        if self.faces is None:
            print("No faces available to initialize topology.")
            return

        # 初始化邻接面和邻接顶点的存储结构
        num_vertices = self.vertices.shape[0]
        self.adjacent_faces = [[] for _ in range(num_vertices)]
        self.adjacent_vertices = [[] for _ in range(num_vertices)]

        # 遍历每个面，构建邻接关系
        for face_idx, face in enumerate(self.faces):
            for i in range(len(face)):
                v0 = face[i]  # 当前顶点
                v1 = face[(i + 1) % len(face)]  # 下一个顶点（循环）
                v2 = face[(i + 2) % len(face)]  # 另一个顶点

                # 添加邻接面
                self.adjacent_faces[v0].append(face_idx)

                # 添加邻接顶点（避免重复）
                if v1 not in self.adjacent_vertices[v0]:
                    self.adjacent_vertices[v0].append(v1)
                if v2 not in self.adjacent_vertices[v0]:
                    self.adjacent_vertices[v0].append(v2)

        print("Topology initialized: adjacent_faces and adjacent_vertices computed.")

    def getAdjacentFaces(self, vertex_idx):
        """
        获取指定顶点的邻接面索引。
        """
        if self.adjacent_faces is None:
            print("Topology not initialized. Call initTopology() first.")
            return []
        return self.adjacent_faces[vertex_idx]

    def getAdjacentVertices(self, vertex_idx):
        """
        获取指定顶点的邻接顶点索引。
        """
        if self.adjacent_vertices is None:
            print("Topology not initialized. Call initTopology() first.")
            return []
        return self.adjacent_vertices[vertex_idx]

    def setVertexTexCoords_(self, vertex_tex: list):
        """
        设置顶点的纹理坐标。
        :param vertex_tex: 包含纹理坐标的列表，每个元素是一个形状为 (2,) 的 numpy 数组。
        """
        self.vertex_texture_ = vertex_tex

    def getAllTexCoords_(self):
        """
        获取所有顶点的纹理坐标。
        :param vertex_tex: 用于存储纹理坐标的列表。
        """
        return self.vertex_texture_

    def setVertexColor_(self, index: int, r: int, g: int, b: int):
        """
        设置指定顶点的颜色。
        :param index: 顶点索引
        :param r: 红色分量（0-255）
        :param g: 绿色分量（0-255）
        :param b: 蓝色分量（0-255）
        """
        # 确保顶点索引有效
        if index < 0 or index >= len(self.vertex_colors):
            raise IndexError(f"Vertex index {index} is out of range.")

        # 设置顶点颜色
        self.vertex_colors[index] = [r, g, b]

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

    def copy(self):
        """
        创建当前 TriMesh 对象的深拷贝。
        :return: 一个新的 TriMesh 对象，其内容与当前对象相同。
        """
        new_mesh = TriMesh()
        new_mesh.vertices = self.vertices.copy()  # 深拷贝顶点
        new_mesh.faces = self.faces.copy()  # 深拷贝面
        new_mesh.normals = self.normals.copy()  # 深拷贝法线
        new_mesh.vertex_texture_ = [vt.copy() for vt in self.vertex_texture_]  # 深拷贝纹理坐标
        new_mesh.vertex_colors = [color.copy() for color in self.vertex_colors]  # 深拷贝顶点颜色
        return new_mesh

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

    def saveColorMesh(self, file_path: str):
        """
        保存带颜色的网格到文件。
        :param file_path: 输出文件路径
        """
        try:
            with open(file_path, 'w') as file:
                # 写入顶点信息
                file.write(f"OFF\n")
                file.write(f"{len(self.vertices)} {len(self.faces)} 0\n")
                for i, vertex in enumerate(self.vertices):
                    r, g, b = self.vertex_colors[i]
                    file.write(
                        f"{vertex[0]} {vertex[1]} {vertex[2]} {r / 255.0} {g / 255.0} {b / 255.0}\n")

                # 写入面信息
                for face in self.faces:
                    face_str = " ".join(map(str, face))
                    file.write(f"3 {face_str}\n")  # 假设每个面是三角形

            print(f"Mesh saved to {file_path}")
        except Exception as e:
            print(f"Failed to save mesh to {file_path}: {e}")
