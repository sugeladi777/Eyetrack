import argparse
import os
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--tar_size', type=int, default=512,
                                 help='size for rendering window. We use a square window.')
        self.parser.add_argument('--padding_ratio', type=float, default=0.3,
                                 help='enlarge the face detection bbox by a margin.')
        self.parser.add_argument('--recon_model', type=str, default='model_i50_e47',
                                 help='choose a 3dmm model, default: model_i50_e47')
        self.parser.add_argument('--first_rf_iters', type=int, default=2000,
                                 help='iteration number of rigid fitting for the first frame in video fitting.')
        self.parser.add_argument('--first_nrf_iters', type=int, default=2000,
                                 help='iteration number of non-rigid fitting for the first frame in video fitting.')
        self.parser.add_argument('--rf_lr', type=float, default=1e-4,
                                 help='learning rate for rigid fitting')
        self.parser.add_argument('--nrf_lr', type=float, default=1e-4,
                                 help='learning rate for non-rigid fitting')
        self.parser.add_argument('--lm_loss_w', type=float, default=100,
                                 help='weight for landmark loss')
        self.parser.add_argument('--id_reg_w', type=float, default=1,
                                 help='weight for id coefficient regularizer')
        self.parser.add_argument('--exp_reg_w', type=float, default=1,
                                 help='weight for expression coefficient regularizer')
        self.parser.add_argument('--res_folder', type=str, default='vis_mesh',
                                 help='output path for the image')
        self.initialized = True

    def parse(self, parse_args=False):
        if not self.initialized:
            self.initialize()
        if parse_args:
            self.opt = self.parser.parse_args()
        else:
            self.opt = self.parser.parse_args(['--img_path', 'data/002.jpg'])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt


class ImageFittingOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--img_path', type=str, required=True,
                                 help='path for the image')
        self.parser.add_argument('--gpu', type=int, default=0,
                                 help='gpu device')
