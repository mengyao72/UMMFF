

import scipy.io as sio
import os
import time
import torch.cuda
import torch.optim as optim
import torch.nn.functional as fun
import torch.utils.data as data
from utils import toolkits, torchkits, DataInfo, BlurDown, PatchDataset
from blind import Blind
from einops import rearrange
from PositionEncoding import *
import retention
import torch
import torch.nn as nn
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CrossAttention(nn.Module):
    def __init__(self, input_dim):
        super(CrossAttention, self).__init__()
        self.input_dim = input_dim
        self.softmax = nn.Softmax(dim=-1)
        self.W = nn.Linear(input_dim, input_dim)

    def forward(self, input1, input2):

        attention = torch.matmul(self.W(input1), input2.transpose(1, 2))
        attention = self.softmax(attention)
        output1 = torch.matmul(attention, input2)
        output2 = torch.matmul(attention.transpose(1, 2), input1)

        return output1, output2


class AENet(nn.Module):

    def __init__(self, hs_bands, ms_bands, edm_num):
        super().__init__()
        self.module_list = nn.ModuleList([])
        self.hs_bands = hs_bands
        self.ms_bands = ms_bands
        self.edm_num = edm_num
        self.num_featureT = 3 * self.edm_num
        edm = torch.ones([self.hs_bands, self.edm_num, 1, 1]) * (1.0 / self.edm_num)
        self.edm = nn.Parameter(edm)
        self.S0_net = nn.Sequential(
            nn.Conv2d(2 * self.edm_num, self.edm_num, (1, 1)),
            nn.LeakyReLU(0.2)
        )
        self.S1_net = nn.Sequential(
            nn.Conv2d(3 * self.edm_num, self.edm_num, (1, 1)),
            nn.LeakyReLU(0.2)
        )
        self.T_E = Transformer_E(self.edm_num)
        self.Embedding1 = nn.Sequential(
            nn.Linear(self.hs_bands, self.edm_num),
        )
        self.Embedding2 = nn.Sequential(
            nn.Linear(self.ms_bands, self.edm_num),
        )
        self.ca1 = ChannelAttention(self.edm_num)
        self.sa1 = SpatialAttention()
        self.Y_net = nn.Sequential(
            nn.Conv2d(self.edm_num, self.edm_num, (1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.edm_num, self.edm_num, (1, 1)),
            nn.LeakyReLU(0.2)
        )
        self.Z_net = nn.Sequential(
            nn.Conv2d(self.edm_num, self.edm_num, (1, 1)),
            nn.LeakyReLU(0.2)
        )
        self._init_weights(self)


    def forward(self, Yu, Z):

        N, B, H, W = Yu.shape
        N, b, H, W = Z.shape
        Yu = Yu.type(torch.cuda.FloatTensor)
        Z = Z.type(torch.cuda.FloatTensor)
        sz = Yu.size(2)
        E1 = rearrange(Yu, 'B c H W -> B (H W) c', H=sz)
        E2 = rearrange(Z, 'B c H W -> B (H W) c', H=sz)
        E1 = self.Embedding1(E1)
        E2 = self.Embedding2(E2)
        Code1 = self.T_E(E1,E2)
        Code2 = self.T_E(E2,E1)
        Code1 = Code1.real
        Code1 = Code1.type(torch.cuda.FloatTensor)
        Code2 = Code2.real
        Code2 = Code2.type(torch.cuda.FloatTensor)
        E3 = rearrange(Code1, 'B (H W) C -> B C H W', H=sz)
        E4 = rearrange(Code2, 'B (H W) C -> B C H W', H=sz)
        Code3 = self.Y_net(E3)
        Code5 = E3 * self.ca1(Code3)
        Code4 = self.Z_net(E4)
        Code6 = E4 * self.sa1(Code4)
        S = torch.cat([Code5, Code6], dim=1)
        S = self.S0_net(S)
        S3 = torch.cat([S, Code5, Code6], dim=1)
        S4 = self.S1_net(S3)
        Highpass = S4.real
        Highpass = Highpass.type(torch.cuda.FloatTensor)
        Highpass = fun.conv2d(Highpass, self.edm, None)
        Highpass = Highpass.clamp_(0, 1)
        return Highpass


    def _init_weights(model, init_type='normal'):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                num_inputs = m.weight.data.shape[1]
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                if init_type == 'normal':
                    nn.init.trunc_normal_(m.weight.data, mean=0.0, std=np.sqrt(1.0 / num_inputs))
                elif init_type == 'constant':
                    nn.init.constant_(m.weight.data, 1.0 / num_inputs)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        pass


# -----------------Transformer-----------------

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x,z, **kwargs):
        Z = self.fn(x,z, **kwargs) + x
        return Z

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, z, **kwargs):
        x = x.real
        x = x.type(torch.cuda.FloatTensor)
        z = z.real
        z = z.type(torch.cuda.FloatTensor)
        return self.fn(self.norm(x), self.norm(z), **kwargs)

class Residual1(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.fn(x, **kwargs) + x
        return x

class PreNorm1(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = x.real
        x = x.type(torch.cuda.FloatTensor)
        x = self.fn(self.norm(x), **kwargs)
        return x




class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)

        )

    def forward(self, x):
        x = self.net(x)
        return x


class Transformer_E(nn.Module):
    def __init__(self, dim, depth=1, heads=1,  mlp_dim=48, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, retention.MultiScaleRetention(dim, heads))),
                Residual1(PreNorm1(dim,FeedForward(dim, mlp_dim, dropout=dropout)))]))

    def forward(self, x, y):
        for attn, ff in self.layers:
            x = attn(x,y)
            x = ff(x)

        return x








class UMMFF(DataInfo):
    def __init__(self, ndata, nratio, psf=None, srf=None, edm_num=48):
        super().__init__(ndata, nratio)
        self.strX = 'X.mat'  # 创建mat文件
        if psf is not None:  # 判断是否有psf
            self.psf = psf
        if srf is not None:
            self.srf = srf
        # set
        self.lr = 0.001
        self.edm_num = edm_num
        self.ker_size = self.psf.shape[0]
        self.patch_size = self.ratio
        self.patch_stride = self.ratio
        self.batch_size = self.set_batch_size()
        self.lam_A, self.lam_B, self.lam_C = 1, 1, 1e-3
        self.lr_fun = lambda epoch: (1.0 - max(0, epoch + 1 - 100) / 2900)
        # define
        self.psf = np.reshape(self.psf, newshape=(1, 1, self.ker_size, self.ker_size))
        self.psf = torch.tensor(self.psf).cuda()
        self.psf_hs = self.psf.repeat(self.hs_bands, 1, 1, 1)
        self.srf = np.reshape(self.srf, newshape=(self.ms_bands, self.hs_bands, 1, 1))
        self.srf = torch.tensor(self.srf).cuda()
        self.__hsi = self.hsi
        self.__msi = self.msi
        self.__hsi_up = nn.Upsample(scale_factor=self.ratio, mode='bilinear', align_corners=False)(
            self.__hsi)
        self.test__hsi = self.test_hsi
        self.test__msi = self.test_msi
        self.test__hsi_up = nn.Upsample(scale_factor=self.ratio, mode='bilinear', align_corners=False)(
            self.test__hsi)
        self.model = AENet(self.hs_bands, self.ms_bands, self.edm_num).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.lam_C)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, self.lr_fun)
        toolkits.check_dir(self.model_save_path)
        torchkits.get_param_num(self.model)
        self.hs_border = math.ceil(
            (self.ker_size - 1) / 2 / self.ratio)
        self.ms_border = self.hs_border * self.ratio
        self.dataset = PatchDataset(self.__hsi, self.__msi, self.__hsi_up,
                                    self.patch_size, self.patch_stride, self.ratio)
        self.test_loader = PatchDataset(self.test__hsi, self.__msi, self.test__hsi_up,
                                    self.patch_size, self.patch_stride, self.ratio)
        self.blur_down = BlurDown()
        self.model_save_path = self.save_path + 'model/'
        pass


    def set_batch_size(self):
        batch_size = 1
        return batch_size

    def cpt_target(self, X):
        Y = self.blur_down(X, self.psf_hs, int((self.ker_size - 1) / 2), self.hs_bands, self.ratio)
        Z = fun.conv2d(X, self.srf, None)
        return Y, Z


    def build_loss(self, Y, Z, hsi, msi):
        dY = Y - hsi
        dZ = Z - msi
        loss = self.lam_A * torch.sum(torch.mean(torch.abs(dY), dim=(2, 3))) * (self.height / self.ratio) * (
                self.width / self.ratio) + self.lam_B * torch.sum(torch.mean(torch.abs(dZ), dim=(2, 3))) * self.width * self.height
        return loss


    def train(self, max_iter=3000, verb=True, is_save=True):
        # train ...
        loader = data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, drop_last=True)
        iteration, epoch = 0, 0
        time_start = time.perf_counter()
        self.model.train()
        while True:
            # train
            for i, (hsi, msi, hsi_up, item) in enumerate(loader):
                hsi, msi, hsi_up = hsi.cuda(), msi.cuda(), hsi_up.cuda()
                X = self.model(hsi_up, msi)
                Yhat, Zhat = self.cpt_target(X)
                loss = self.build_loss(Yhat, Zhat, hsi, msi)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                iteration += 1
                if verb is True:
                    if iteration % 10 == 0:
                        self.evaluation(iteration)
                        self.model.train()
                if iteration >= max_iter:
                    break
                self.scheduler.step()
                pass
            epoch += 1
            if iteration >= max_iter:
                if verb is True:
                    self.evaluation(epoch)
                break
            pass
        time_end = time.perf_counter()
        train_time = time_end - time_start
        print('running time %ss' % train_time)
        X, test_time = self.evaluation(epoch)
        if is_save is True:
            torch.save(self.model.state_dict(), self.model_save_path + 'parameter.pkl')
            sio.savemat(self.save_path + self.strX, {'X': X})
        pass

    def evaluation(self, iteration):
        self.model.eval()
        lr = self.optimizer.param_groups[0]['lr']
        t0 = time.perf_counter()
        X = self.model(self.__hsi_up.cuda(), self.__msi.cuda())
        test_time = time.perf_counter() - t0
        Yhat, Zhat = self.cpt_target(X)
        loss = self.build_loss(Yhat, Zhat, self.__hsi.cuda(), self.__msi.cuda())
        Xh = torchkits.to_numpy(X)
        Xh = toolkits.channel_last(Xh)
        if self.ref is not None:
            psnr = toolkits.compute_psnr(self.ref, Xh)
            sam = toolkits.compute_sam(self.ref, Xh)
            rmse = toolkits.calc_rmse(self.ref, Xh)
            ergas = toolkits.calc_ergas(self.ref, Xh)
            print('iter/epoch: %s, lr: %s, psnr: %s, sam: %s, rmse: %s, ergas: %s' % (iteration, lr, psnr, sam, rmse, ergas))
        else:
            print('iter/epoch: %s, lr: %s, loss: %s' % (iteration, lr, loss))
        return Xh, test_time







if __name__ == '__main__':
    ndata, nratio = 1, 8
    edm_num = 48 # pavia: 3, 80; ksc: 3, 80; dc: 3, 30; UH: 3, 30
    blind = Blind(ndata=ndata, nratio=nratio,  blind=True)
    blind.train()
    blind.get_save_result()
    net = UMMFF(ndata=ndata, nratio=nratio, psf=blind.psf, srf=blind.srf, edm_num=edm_num)
    net.train()

    pass
