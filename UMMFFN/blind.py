# BlindTest estimation network
# Author: JianJun Liu
# Date: 2022-1-13
import torch
import scipy.io as sio
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as fun
from utils import toolkits, torchkits, DataInfo, BlurDown
from PositionEncoding import *
from tools import device

class SSFFcn(torch.nn.Module):
    def __init__(self, L, out_dim):
        super(SSFFcn, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=2 * L + 1, out_features=2 * L),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=2*L, out_features=2*L),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=2*L, out_features=2*L),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=2*L, out_features=2*L),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=2*L, out_features=out_dim)#,
            #nn.ReLU(inplace=True)
        )

    def forward(self, x):

        y = self.layers(x)

        return y


class ConvFcn(torch.nn.Module):
    def __init__(self, L, out_dim):

        super(ConvFcn, self).__init__()

        self.layers = nn.Sequential(
            torch.nn.Conv2d(4 * L + 2, 4 * L, (1, 1), padding=0),
            nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(4 * L, 2 * L, (1, 1), padding=0),
            nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(2 * L, 1 * L, (1, 1), padding=0),
            nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(1 * L, 2, (1, 1), padding=0),
            nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(2, out_dim, (1, 1), padding=0)#,
            #nn.ReLU(inplace=True)
        )

    def forward(self, x):

        y = self.layers(x)

        return y


class BlindNet(nn.Module):
    def __init__(self, hs_bands, ms_bands, ker_size, ratio):
        super().__init__()
        self.hs_bands = hs_bands
        self.ms_bands = ms_bands
        self.ker_size = ker_size
        self.ratio = ratio
        self.pad_num = int((self.ker_size - 1) / 2)
        self.psf = torch.ones([1, 1, self.ker_size, self.ker_size]) * (1.0 / (self.ker_size ** 2))

        self.srf = torch.ones([self.ms_bands, self.hs_bands, 1, 1]) * (1.0 / self.hs_bands)

        self.blur_down = BlurDown()



        self.input_1D = torch.from_numpy(positionencoding1D(self.hs_bands, 5)).float().to(device)
        self.ssffcn = SSFFcn(5, 5)
        self.input_2D = torch.from_numpy(positionencoding2D(self.ker_size, self.ker_size, 1, petype)).float().to(
            device).permute(2, 0, 1).unsqueeze(0)
        self.convfcn = ConvFcn(1, 1)

    def forward(self, Y, Z):

        Phi = self.ssffcn(self.input_1D) ** 2  # self.Phi*self.Phi
        Conv = self.convfcn(self.input_2D) ** 2
        Conv = torch.nn.Softmax(dim=0)(Conv.reshape(self.ker_size ** 2)).reshape(self.ker_size, self.ker_size)

        self.srf = Phi.view(self.ms_bands, self.hs_bands, 1, 1)
        self.psf = Conv.view(1, 1, self.ker_size, self.ker_size)

        srf_div = torch.sum(self.srf, dim=1, keepdim=True)
        srf_div = torch.div(1.0, srf_div)#点除
        srf_div = torch.transpose(srf_div, 0, 1)
        Y = Y.type(torch.cuda.FloatTensor)
        Ylow = fun.conv2d(Y, self.srf, None)
        Ylow = torch.mul(Ylow, srf_div)
        Ylow = torch.clamp(Ylow, 0.0, 1.0)
        Zlow = self.blur_down(Z, self.psf, self.pad_num, self.ms_bands, self.ratio)
        Zlow = torch.clamp(Zlow, 0.0, 1.0)#

        return Ylow, Zlow


class Blind(DataInfo):
    def __init__(self, ndata, nratio, nsnr=0, blind=True):
        super().__init__(ndata, nratio, nsnr)
        self.strBR = 'BR.mat'
        self.blind = blind
        if self.blind is False:
            # self.psf, self.srf
            print('using true psf and srf!')
            return
        print('estimate psf and srf ...')
        # set
        self.lr = 5e-5  # learning rate
        self.ker_size = 2 * self.ratio - 1
        # variable, graph and etc.
        self.__hsi = self.hsi
        self.__msi = self.msi
        self.model = BlindNet(self.hs_bands, self.ms_bands, self.ker_size, self.ratio).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        toolkits.check_dir(self.model_save_path)

    def train(self, max_iter=30000, verb=True):
        if self.blind is False:
            return
        hsi, msi = self.__hsi.cuda(), self.__msi.cuda()
        for epoch in range(0, max_iter):
            Ylow, Zlow = self.model(hsi, msi)
            loss = torchkits.torch_norm(Ylow - Zlow)
            if verb is True:
                if (epoch + 1) % 2000 == 0:
                    print('epoch: %s, lr: %s, loss: %s' % (epoch + 1, self.lr, loss))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.apply(self.check_weight)
        torch.save(self.model.state_dict(), self.model_save_path + 'parameter.pkl')
        self.psf = torch.tensor(torchkits.to_numpy(self.model.psf.data))
        self.srf = torch.tensor(torchkits.to_numpy(self.model.srf.data))

    def get_save_result(self, is_save=True):
        if self.blind is False:
            return
        print('save psf and srf ...')
        self.model.load_state_dict(torch.load(self.model_save_path + 'parameter.pkl'))
        psf = torchkits.to_numpy(self.model.psf.data)
        srf = torchkits.to_numpy(self.model.srf.data)
        psf = np.squeeze(psf)
        srf = np.squeeze(srf)  # b x B
        self.psf, self.srf = psf, srf
        if is_save is True:
            sio.savemat(self.save_path + self.strBR, {'B': psf, 'R': srf})
        return

    @staticmethod
    def check_weight(model):
        if hasattr(model, 'psf'):
            w = model.psf.data
            w.clamp_(0.0, 1.0)
            psf_div = torch.sum(w)
            psf_div = torch.div(1.0, psf_div)
            w.mul_(psf_div)
        if hasattr(model, 'srf'):
            w = model.srf.data
            w.clamp_(0.0, 10.0)
            srf_div = torch.sum(w, dim=1, keepdim=True)
            srf_div = torch.div(1.0, srf_div)#点除
            w.mul_(srf_div)
