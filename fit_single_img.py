from facenet_pytorch import MTCNN
from core.options import ImageFittingOptions
import cv2
import face_alignment
import numpy as np
from core import get_recon_model
import os
import torch
import core.utils as utils
from tqdm import tqdm
import core.losses as losses


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


if __name__ == '__main__':
    args = ImageFittingOptions()
    args = args.parse()
    args.device = 'cuda:%d' % args.gpu
    fit(args)
