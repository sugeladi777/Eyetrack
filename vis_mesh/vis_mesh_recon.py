import trimesh
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from mesh_renderer import MeshRenderer


# mesh_path = "VID_20240125_092850-1.png_eyelidface.obj"
mesh_path = "vis_mesh/002_mesh.obj"
img_path = "vis_mesh/resized_face_img.jpg"

device = "cuda"
mesh_renderer = MeshRenderer(device=device)


f = 1015
intrinsic = torch.tensor([
    [f, 0, 512/2],
    [0, f, 512/2],
    [0, 0, 1],
]).float().to(device)

mesh = trimesh.load_mesh(mesh_path)  # y to up, z to back, x to left
cam_verts = torch.from_numpy(mesh.vertices).float()
cam_verts[:,-1] -= 10
cam_verts[:,-1] = -cam_verts[:,-1]
cam_verts[:,0] = -cam_verts[:,0]


faces = torch.from_numpy(mesh.faces)
img = transforms.ToTensor()(Image.open(img_path)).to(device)[None, ...]

cam_verts = cam_verts.to(device)
faces = faces.to(device)

# compute the shading
normal = torch.from_numpy(mesh.vertex_normals).float()
light_dir = torch.tensor([0., 0., 1.])
color = torch.sum(normal * light_dir, dim=-1, keepdim=True).clamp(min=0.).to(device)
mask = torch.ones_like(cam_verts[..., :1])
attr = torch.cat([color, mask], dim=-1)

# render the mesh to image space
mesh_dict = {
    "vertice": cam_verts[None, ...],
    "faces": faces[None, ...],
    "attributes": attr[None, ...],
    "size": (512, 512),
}

cam_int = torch.clone(intrinsic)  # [3,3]
cam_int[0] /= 512
cam_int[1] /= 512
cam_int = cam_int[None, ...]  # [1,3,3]

cam_ext = torch.eye(4)[None, :3].to(device)

output, pix_to_face = mesh_renderer.render_mesh(mesh_dict, cam_int=cam_int, cam_ext=cam_ext)
output = torch.flip(output, dims=(-1, -2))

render = output[:, :1]
mask = output[:, 1:2] * 0.5
vis = mask * render + (1 - mask) * img

vis = torch.cat([img, vis], dim=-1)
save_image(vis, "vis_mesh/1_result.jpg")
