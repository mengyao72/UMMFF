# Author: JianJun Liu
# Date: 2022-1-13
import numpy as np
import scipy.io as sio
import os
import torch
import torch.nn.functional as fun
import torch.utils.data as data
from data_loader import build_datasets
import args_parser
import random
import torch.nn.functional as F

args = args_parser.args_parser()
class toolkits(object):
    @staticmethod
    def compute_psnr(img1: np.ndarray, img2: np.ndarray, channel=False):
        assert img1.ndim == 3 and img2.ndim == 3
        img_h, img_w, img_c = img1.shape
        ref = img1.reshape(-1, img_c)
        tar = img2.reshape(-1, img_c)
        msr = np.mean((ref - tar) ** 2, 0)
        if channel is False:
            max2 = np.max(ref) ** 2
        else:
            max2 = np.max(ref, axis=0) ** 2
        psnrall = 10 * np.log10(max2 / msr)
        out_mean = np.mean(psnrall)
        return out_mean

    @staticmethod
    def compute_sam(label: np.ndarray, output: np.ndarray):
        h, w, c = label.shape
        x_norm = np.sqrt(np.sum(np.square(label), axis=-1))
        y_norm = np.sqrt(np.sum(np.square(output), axis=-1))
        xy_norm = np.multiply(x_norm, y_norm)
        xy = np.sum(np.multiply(label, output), axis=-1)
        dist = np.mean(np.arccos(np.minimum(np.divide(xy, xy_norm + 1e-8), 1.0 - 1.0e-9)))
        dist = np.multiply(180.0 / np.pi, dist)
        return dist

    @staticmethod
    def check_dir(path: str):
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def channel_last(input_tensor: np.ndarray, squeeze=True):
        if squeeze is True:
            input_tensor = np.squeeze(input_tensor)
        input_tensor = np.transpose(input_tensor, axes=(1, 2, 0))
        return input_tensor

    @staticmethod
    def channel_first(input_tensor: np.ndarray, expand=True):
        input_tensor = np.transpose(input_tensor, axes=(2, 0, 1))
        if expand is True:#是否进行数组扩展
            input_tensor = np.expand_dims(input_tensor, axis=0)
        return input_tensor




class torchkits(object):
    @staticmethod
    def extract_patches(input_tensor: torch.Tensor, kernel=3, stride=1, pad_num=0):
        # input_tensor: N x C x H x W, patches: N * H' * W', C, h, w
        if pad_num != 0:
            input_tensor = torch.nn.ReflectionPad2d(pad_num)(input_tensor)
        all_patches = input_tensor.unfold(2, kernel, stride).unfold(3, kernel, stride)
        N, C, H, W, h, w = all_patches.shape
        all_patches = all_patches.permute(0, 2, 3, 1, 4, 5)
        all_patches = torch.reshape(all_patches, shape=(N * H * W, C, h, w))
        return all_patches

    @staticmethod
    def torch_norm(input_tensor: torch.Tensor, mode=1):
        if mode == 1:
            loss = torch.sum(torch.abs(input_tensor))
            return loss
        return None

    @staticmethod
    def get_param_num(model):#计算参数
        num = sum(x.numel() for x in model.parameters())
        print("model has {} parameters in total".format(num))
        return num

    @staticmethod
    def to_numpy(val: torch.Tensor):
        return val.cpu().detach().numpy()


class BlurDown(object):
    def __init__(self, shift_h=0, shift_w=0, stride=0):
        self.shift_h = shift_h
        self.shift_w = shift_w
        self.stride = stride
        pass

    def __call__(self, input_tensor: torch.Tensor, psf, pad, groups, ratio):#输出张量
        if psf.shape[0] == 1:
            psf = psf.repeat(groups, 1, 1, 1)
        if self.stride == 0:
            input_tensor = input_tensor.type(torch.cuda.FloatTensor)
            output_tensor = fun.conv2d(input_tensor, psf, None, (1, 1), (pad, pad), groups=groups)
            output_tensor = output_tensor[:, :, self.shift_h:: ratio, self.shift_h:: ratio]
        else:
            output_tensor = fun.conv2d(input_tensor, psf, None, (ratio, ratio), (pad, pad), groups=groups)
        return output_tensor


class DataInfo(object):
    """
        file structure
        ./data/
        ../data/pavia/
        ../data/moffett/
        ../data/dc/
        .../data/pavia/XXX/
        .../data/pavia/BlindTest/
        .../data/pavia/pavia_data_r?_?_?.mat
        ..../data/pavia/BlindTest/r?_?_?/
        ..../data/pavia/BlindTest/model/
        ..../data/pavia/BlindTest/BR.mat
    """
    def __init__(self, ndata=0, nratio=8, nsnr=0):
        name = self.__class__.__name__
        print('%s is running' % name)
        self.gen_path = 'UMMFF-main/UMMFF/data'
        self.folder_names = ['pavia/', 'ksc/', 'dc/', 'uh/']
        # self.data_names = ['pavia_data_r', 'ksc_data_r', 'dc_data_r', 'UH_test_r']
        self.noise = ['_20_30', '_25_35', '_30_40', '_50_60', '']
        image_size = 128
        scale_ratio = 8
        train_list, test_list = build_datasets(args.root,
                                               args.dataset,
                                               args.image_size,
                                               args.n_select_bands,
                                               args.scale_ratio)
        train_ref, train_lr, train_hr = train_list

        h, w = train_ref.size(2), train_ref.size(3)
        h_str = 0
        w_str = 0

        train_ref = train_ref[:, :, h_str:h_str + image_size, w_str:w_str + image_size]
        train_lr = F.interpolate(train_ref, scale_factor=1 / (scale_ratio * 1.0))
        train_hr = train_hr[:, :, h_str:h_str + image_size, w_str:w_str + image_size]
        self.hsi = train_lr
        self.msi = train_hr
        self.ref = train_ref
        self.ref = torchkits.to_numpy(self.ref)
        self.ref = toolkits.channel_last(self.ref)
        self.psf = np.ones(shape=(self.msi.shape[2] // self.hsi.shape[2], self.msi.shape[3] // self.hsi.shape[3]))
        self.srf = np.ones(shape=(self.msi.shape[1], self.hsi.shape[1]))
        self.save_path = self.gen_path + self.folder_names[ndata] + name + '/r' + str(nratio) + self.noise[
            nsnr] + '/'
        self.model_save_path = self.save_path + 'model/'
        self.hs_bands, self.ms_bands = self.hsi.shape[1], self.msi.shape[1]
        self.ratio = int(self.msi.shape[-1] / self.hsi.shape[-1])
        self.height, self.width = self.msi.shape[2], self.msi.shape[3]
        pass


class PatchDataset(data.Dataset):
    def __init__(self, hsi: torch.Tensor, msi: torch.Tensor, hsi_up: torch.Tensor, kernel, stride, ratio=1):
        super(PatchDataset, self).__init__()
        self.hsi = hsi
        self.msi = msi
        self.hsi_up = hsi_up
        self.num = self.msi.shape[0]
        assert self.hsi.shape[0] == self.num


    def __getitem__(self, item):
        hsi = self.hsi[item, :, :, :]
        msi = self.msi[item, :, :, :]
        hsi_up = self.hsi_up[item, :, :, :]
        return hsi, msi, hsi_up, item

    def __len__(self):
        return self.num
