import argparse
import logging
import math
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)
from models.Detect.MuitlHead3 import *
from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr
from utils.plots import *

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    export_cat = False  # onnx export cat output

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        # self.no = nc + 5  # number of outputs per anchor
        # --------------------------------------
        self.no = nc + 5 + 12  # number of outputs per anchor

        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        # ---sly热力图可视化
        logits_ = []
        if self.export_cat:
            for i in range(self.nl):
                x[i] = self.m[i](x[i])  # conv
                bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    # self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                    self.grid[i], self.anchor_grid[i] = self._make_grid_new(nx, ny, i)

                y = torch.full_like(x[i], 0)
                # ---------------------------------------------------------
                y = y + torch.cat((x[i][:, :, :, :, 0:5].sigmoid(),
                                   torch.cat((x[i][:, :, :, :, 5:17], x[i][:, :, :, :, 17:17 + self.nc].sigmoid()), 4)),
                                  4)

                box_xy = (y[:, :, :, :, 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                box_wh = (y[:, :, :, :, 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                # box_conf = torch.cat((box_xy, torch.cat((box_wh, y[:, :, :, :, 4:5]), 4)), 4)

                landm1 = y[:, :, :, :, 5:7] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
                    i]  # landmark x1 y1
                landm2 = y[:, :, :, :, 7:9] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
                    i]  # landmark x2 y2
                landm3 = y[:, :, :, :, 9:11] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
                    i]  # landmark x3 y3
                landm4 = y[:, :, :, :, 11:13] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
                    i]  # landmark x4 y4
                landm5 = y[:, :, :, :, 13:15] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
                    i]  # landmark x5 y5
                # ---------------------------------------------------------
                landm6 = y[:, :, :, :, 15:17] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
                    i]  # landmark x5 y5
                # landm = torch.cat((landm1, torch.cat((landm2, torch.cat((landm3, torch.cat((landm4, landm5), 4)), 4)), 4)), 4)
                # y = torch.cat((box_conf, torch.cat((landm, y[:, :, :, :, 15:15+self.nc]), 4)), 4)
                # ---------------------------------------------------------
                y = torch.cat([box_xy, box_wh, y[:, :, :, :, 4:5], landm1, landm2, landm3, landm4, landm5, landm6,
                               y[:, :, :, :, 17:17 + self.nc]], -1)

                z.append(y.view(bs, -1, self.no))
            return torch.cat(z, 1)

        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                # ---sly热力图可视化
                logits = x[i][..., 5:]

                y = torch.full_like(x[i], 0)
                # --------------------------------------
                class_range = list(range(5)) + list(range(17, 17 + self.nc))
                y[..., class_range] = x[i][..., class_range].sigmoid()
                # --------------------------------------
                y[..., 5:17] = x[i][..., 5:17]
                # y = x[i].sigmoid()

                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                # y[..., 5:15] = y[..., 5:15] * 8 - 4
                y[..., 5:7] = y[..., 5:7] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
                    i]  # landmark x1 y1
                y[..., 7:9] = y[..., 7:9] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
                    i]  # landmark x2 y2
                y[..., 9:11] = y[..., 9:11] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
                    i]  # landmark x3 y3
                y[..., 11:13] = y[..., 11:13] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
                    i]  # landmark x4 y4
                y[..., 13:15] = y[..., 13:15] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
                    i]  # landmark x5 y5
                # --------------------------------------
                y[..., 15:17] = y[..., 15:17] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
                    i]  # landmark x6 y6
                # y[..., 5:7] = (y[..., 5:7] * 2 -1) * self.anchor_grid[i]  # landmark x1 y1
                # y[..., 7:9] = (y[..., 7:9] * 2 -1) * self.anchor_grid[i]  # landmark x2 y2
                # y[..., 9:11] = (y[..., 9:11] * 2 -1) * self.anchor_grid[i]  # landmark x3 y3
                # y[..., 11:13] = (y[..., 11:13] * 2 -1) * self.anchor_grid[i]  # landmark x4 y4
                # y[..., 13:15] = (y[..., 13:15] * 2 -1) * self.anchor_grid[i]  # landmark x5 y5

                z.append(y.view(bs, -1, self.no))
                # ---sly热力图可视化
                logits_.append(logits.view(bs, -1, self.no - 5))
        return x if self.training else (torch.cat(z, 1), x)
        # return x if self.training else (torch.cat(z, 1), torch.cat(logits_, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def _make_grid_new(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if '1.10.0' in torch.__version__:  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(
            (1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info('Overriding model.yaml nc=%g with nc=%g' % (self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))]) ---sly
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 128  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())
        # ---------sly
        # if isinstance(m, (DetectX, DetectYoloX)):
        #     m.inplace = self.inplace
        #     self.stride = torch.tensor(m.stride)
        #     m.initialize_biases()  # only run once
        #     self.model_type = 'yolox'
        #     self.loss_category = ComputeXLoss  # use ComputeXLoss
        elif isinstance(m, ASFF_Detect) or isinstance(m, ASFF_Detect4):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)
            # print()
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)  # must be in pixel-space (not grid-space)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # s = 256 # 2x min stride
            # m.inplace = self.inplace
            # m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            # m.anchors /= m.stride.view(-1, 1, 1)
            # check_anchor_order(m)
            # self.stride = m.stride
            # try:
            #     self._initialize_biases()  # only run once
            #     LOGGER.info('initialize_biases done')
            # except:
            #     LOGGER.info('decoupled no biase ')
        elif isinstance(m, Decoupled_Detect):
            s = 128  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.empty(1, ch, s, s))])  # forward
            check_anchor_order(m)  # must be in pixel-space (not grid-space)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_dh_biases()  # only run once
        elif isinstance(m, IDetect):
            s = 128 # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
        elif isinstance(m, IAuxDetect):
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))[:4]])  # forward
            # print(m.stride)
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_aux_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
    # def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si)
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite('img%g.jpg' % s, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            y = self._clip_augmented(y)  # clip augmented tails----sly
            return torch.cat(y, 1), None  # augmented inference, train
        # else: ---sly
        #     # return self.forward_once(x, profile)  # single-scale inference, train  -------sly
        #     return self.forward_once(x, profile, visualize)

        return self.forward_once(x, profile, visualize)

    def forward_once(self, x, profile=False, visualize=False):
    # def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                # ----sly
                self._profile_one_layer(m, x, dt)
                # o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                # t = time_synchronized()
                # for _ in range(10):
                #     _ = m(x)
                # dt.append((time_synchronized() - t) * 100)
                # print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

            # -----sly 特征图可视化 注意路径和层数的修改 该函数在plots.py中
            #             无论是在detect.py还是在train.py中都会进行可视化特征图。
            #             然而训练的过程中并不一定需要一直可视化特征图
            # feature_vis参数是用来控制是否保存可视化特征图的，保存的特征图会存在features文件夹中。
            # 如果想看其它层的特征只需要修改m.type或是用m.i来进行判断是否可视化特征图。
            # m.type对应的是yaml文件中的module，即yolov5的基础模块，例如c3，conv，spp等等，而m.i则更好理解，即是模块的id，通常就是顺序
            #             visualize = Path('runs/detect/exp/featurevisualization')
            # if m.type == 'models.common.C3' and visualize:
            #     feature_visualization(x, m.type, m.i, save_dir=visualize)
            # return x 害死我了 不过还好逮到你
            # feature_vis = True
            # if m.type == 'models.common.C3' and feature_vis:
            #     print(m.type, m.i)
            #     feature_visualization(x, m.type, m.i)

        # if profile:
        #     print('%.1fms total' % sum(dt))
        return x

    # def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
    #     # https://arxiv.org/abs/1708.02002 section 3.3
    #     # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
    #     m = self.model[-1]  # Detect() module
    #     for mi, s in zip(m.m, m.stride):  # from
    #         b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
    #         b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
    #         b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
    #         mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1).detach()  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _initialize_dh_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            # reg_bias = mi.reg_preds.bias.view(m.na, -1).detach()
            # reg_bias += math.log(8 / (640 / s) ** 2)
            # mi.reg_preds.bias = torch.nn.Parameter(reg_bias.view(-1), requires_grad=True)

            # cls_bias = mi.cls_preds.bias.view(m.na, -1).detach()
            # cls_bias += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            # mi.cls_preds.bias = torch.nn.Parameter(cls_bias.view(-1), requires_grad=True)
            b = mi.b3.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            mi.b3.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            b = mi.c3.bias.data
            b += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.c3.bias = torch.nn.Parameter(b, requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
            elif type(m) is nn.Upsample:
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    # -----sly
    # def _profile_one_layer(self, m, x, dt):
    #     # c = isinstance(m, Detect)  # update is final layer, copy input as inplace fix
    #     # c = isinstance(m, Detect) or isinstance(m, ASFF_Detect) or isinstance(m,
    #     #                                                                       Decoupled_Detect)
    #     c = isinstance(m, ASFF_Detect) or isinstance(m, Decoupled_Detect)  # copy input as inplace fix
    #     o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
    #     t = time_sync()
    #     for _ in range(10):
    #         m(x.copy() if c else x)
    #     dt.append((time_sync() - t) * 100)
    #     if m == self.model[0]:
    #         LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
    #     LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
    #     if c:
    #         LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _profile_one_layer(self, m, x, dt):
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        # if isinstance(m, Detect) or isinstance(m, ASFF_Detect) or isinstance(m, Decoupled_Detect):
        if isinstance(m, Detect) or isinstance(m, ASFF_Detect) or isinstance(m, Decoupled_Detect) or isinstance(m, ASFF_Detect4):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y


# def parse_model(d, ch):  # model_dict, input_channels(3)
#     logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
#     anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
#     na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
#     no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
#
#     layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
#     for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
#         m = eval(m) if isinstance(m, str) else m  # eval strings
#         for j, a in enumerate(args):
#             try:
#                 args[j] = eval(a) if isinstance(a, str) else a  # eval strings
#             except:
#                 pass
#
#         n = max(round(n * gd), 1) if n > 1 else n  # depth gain
#         if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
#                  BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x,
#                  C3HB, CBH, ES_Bottleneck, CoT3, BoT3, SEAM, RFEM, C3RFEM, ConvMixer, MultiSEAM, C3STR, MobileOneBlock,
#                  StemBlock, CoordAtt, CTR3]:
#             c1, c2 = ch[f], args[0]
#             # Normal
#             # if i > 0 and args[0] != no:  # channel expansion factor
#             #     ex = 1.75  # exponential (default 2.0)
#             #     e = math.log(c2 / ch[1]) / math.log(2)
#             #     c2 = int(ch[1] * ex ** e)
#             # if m != Focus:
#
#             c2 = make_divisible(c2 * gw, 8) if c2 != no else c2
#
#             # Experimental
#             # if i > 0 and args[0] != no:  # channel expansion factor
#             #     ex = 1 + gw  # exponential (default 2.0)
#             #     ch1 = 32  # ch[1]
#             #     e = math.log(c2 / ch1) / math.log(2)  # level 1-n
#             #     c2 = int(ch1 * ex ** e)
#             # if m != Focus:
#             #     c2 = make_divisible(c2, 8) if c2 != no else c2
#
#             args = [c1, c2, *args[1:]]
#             if m in [BottleneckCSP, C3, C3TR, CTR3, C3Ghost, C3x, CoT3, BoT3, C3HB, C3RFEM, C3STR]:
#                 args.insert(2, n)
#                 n = 1
#
#         elif m is nn.BatchNorm2d:
#             args = [ch[f]]
#             # add module research
#         # GAMAttention,
#         elif m in [CARAFE, SPPCSPC, RepConv, BoT3, CA, CBAM, NAMAttention, GAMAttention, Involution, Stem,
#                    ResCSPC, ResCSPB, \
#                    ResXCSPB, ResXCSPC, BottleneckCSPB, BottleneckCSPC,
#                    ASPP, BasicRFB, SPPCSPC_group, HorBlock, CNeB, nn.ConvTranspose2d]:
#             c1, c2 = ch[f], args[0]
#             if c2 != no:  # if not output
#                 c2 = make_divisible(c2 * gw, 8)
#
#             args = [c1, c2, *args[1:]]
#             if m in [C3RFEM, SPPCSPC, BoT3, ResCSPC, ResCSPB, ResXCSPB, ResXCSPC, BottleneckCSPB, BottleneckCSPC, \
#                      HorBlock, CNeB]:
#                 args.insert(2, n)  # number of repeats
#                 n = 1
#             elif m is nn.ConvTranspose2d:
#                 if len(args) >= 7:
#                     args[6] = make_divisible(args[6] * gw, 8)
#         elif m in [CBH, ES_Bottleneck, DWConvblock, RepVGGBlock, LC_Block, Dense, conv_bn_relu_maxpool, \
#                    Shuffle_Block, stem, mobilev3_bneck, conv_bn_hswish, MobileNetV3_InvertedResidual, DepthSepConv, \
#                    ShuffleNetV2_Model, Conv_maxpool, CoT3, ConvNextBlock, RepBlock]:
#             c1, c2 = ch[f], args[0]
#             if c2 != no:  # if not output
#                 c2 = make_divisible(c2 * gw, 8)
#
#             args = [c1, c2, *args[1:]]
#             if m in [CoT3, ConvNextBlock]:
#                 args.insert(2, n)  # number of repeats
#                 n = 1
#
#         # yolov4, r
#         elif m in [SPPCSP, BottleneckCSP2, DownC, BottleneckCSPF, RepVGGBlockv6]:
#             c1, c2 = ch[f], args[0]
#             if c2 != no:  # if not output
#                 c2 = make_divisible(c2 * gw, 8)
#
#             args = [c1, c2, *args[1:]]
#             if m in [SPPCSP, BottleneckCSP2, DownC, BottleneckCSPF]:
#                 args.insert(2, n)  # number of repeats
#                 n = 1
#         elif m in [ReOrg, DWT]:
#             c2 = ch[f] * 4
#         # ACmix,
#         elif m in [S2Attention, SimSPPF, CrissCrossAttention, SOCA, ShuffleAttention, SEAttention, SimAM,
#                    SKAttention]:
#             c1, c2 = ch[f], args[0]
#             if c2 != no:  # if not output
#                 c2 = make_divisible(c2 * gw, 8)
#             args = [c1, *args[1:]]
#
#         # ------------sly
#         elif m is MobileOne:
#             c1, c2 = ch[f], args[0]
#             c2 = make_divisible(c2 * gw, 8)
#             args = [c1, c2, n, *args[1:]]
#
#         elif m is HorBlock:
#             c1, c2 = ch[f], args[0]
#             if c2 != no:
#                 c2 = make_divisible(c2 * gw, 8)
#
#             args = [c1, c2, *args[1:]]
#             if m is HorBlock:
#                 args.insert(2, n)
#                 n = 1
#
#         elif m is HorNet:
#             c2 = args[0]
#             args = args[1:]
#
#         elif m is C3HB:
#             c1, c2 = ch[f], args[0]
#             if c2 != no:
#                 c2 = make_divisible(c2 * gw, 8)
#
#             args = [c1, c2, *args[1:]]
#             if m is C3HB:
#                 args.insert(2, n)
#                 n = 1
#
#         elif m is CNeB:
#             c1, c2 = ch[f], args[0]
#             if c2 != no:
#                 c2 = make_divisible(c2 * gw, 8)
#
#             args = [c1, c2, *args[1:]]
#             if m is CNeB:
#                 args.insert(2, n)
#                 n = 1
#
#         elif m is ConvNextBlock:
#             c1, c2 = ch[f], args[0]
#             if c2 != no:
#                 c2 = make_divisible(c2 * gw, 8)
#
#             args = [c1, c2, *args[1:]]
#             if m is ConvNextBlock:
#                 args.insert(2, n)
#                 n = 1
#
#
#
#         elif m is Concat:
#             # c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
#             c2 = sum([ch[x] for x in f])
#         elif m is ConvNeXt:
#             c2 = args[0]
#             args = args[1:]
#         elif m in [RepLKNet_Stem, RepLKNet_stage1, RepLKNet_stage2, RepLKNet_stage3, RepLKNet_stage4]:
#             c2 = args[0]
#             args = args[1:]
#         elif m is ADD:
#             c2 = sum([ch[x] for x in f]) // 2
#         elif m is Concat_bifpn:
#             c2 = max([ch[x] for x in f])
#         elif m is RepBlock:
#             args.insert(2, n)
#             n = 1
#
#         elif m is Detect:
#             args.append([ch[x + 1] for x in f])
#             if isinstance(args[1], int):  # number of anchors
#                 args[1] = [list(range(args[1] * 2))] * len(f)
#         elif m is space_to_depth:
#             c2 = 4 * ch[f]
#         elif m is ASFF_Detect:
#             args.append([ch[x] for x in f])
#             if isinstance(args[1], int):  # number of anchors
#                 args[1] = [list(range(args[1] * 2))] * len(f)
#         elif m is ASFF_Detect4:
#             args.append([ch[x] for x in f])
#             if isinstance(args[1], int):  # number of anchors
#                 args[1] = [list(range(args[1] * 2))] * len(f)
#         elif m is Decoupled_Detect:
#             args.append([ch[x] for x in f])
#             if isinstance(args[1], int):  # number of anchors
#                 args[1] = [list(range(args[1] * 2))] * len(f)
#         elif m in [IDetect, IAuxDetect]:
#             args.append([ch[x] for x in f])
#             if isinstance(args[1], int):  # number of anchors
#                 args[1] = [list(range(args[1] * 2))] * len(f)
#
#         elif m is Contract:
#             c2 = ch[f] * args[0] ** 2
#         # torchvision
#         elif m is RegNet1 or m is RegNet2 or m is RegNet3:
#             c2 = args[0]
#         elif m is Efficient1 or m is Efficient2 or m is Efficient3:
#             c2 = args[0]
#         elif m is MobileNet1 or m is MobileNet2 or m is MobileNet3:
#             c2 = args[0]
#         elif m is Expand:
#             c2 = ch[f] // args[0] ** 2
#         else:
#             c2 = ch[f]
#
#         m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
#         t = str(m)[8:-2].replace('__main__.', '')  # module type
#         np = sum([x.numel() for x in m_.parameters()])  # number params
#         m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
#         logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
#         save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
#         layers.append(m_)
#         ch.append(c2)
#     return nn.Sequential(*layers), sorted(save)
def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, DWConv, GhostConv, RepConv, RepConv_OREPA, DownC,
                 SPP, SPPF, SPPCSPC, GhostSPPCSPC, MixConv2d, Focus, Stem, CrossConv,
                 Bottleneck, BottleneckCSPA, BottleneckCSPB, BottleneckCSPC,
                 RepBottleneck, RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC,
                 Res, ResCSPA, ResCSPB, ResCSPC,
                 RepRes, RepResCSPA, RepResCSPB, RepResCSPC,
                 ResX, ResXCSPA, ResXCSPB, ResXCSPC,
                 RepResX, RepResXCSPA, RepResXCSPB, RepResXCSPC,
                 Ghost, GhostCSPA, GhostCSPB, GhostCSPC,
                 SwinTransformerBlock, STCSPA, STCSPB, STCSPC,
                 SwinTransformer2Block, ST2CSPA, ST2CSPB, ST2CSPC,C3]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [DownC, SPPCSPC, GhostSPPCSPC,
                     BottleneckCSPA, BottleneckCSPB, BottleneckCSPC,
                     RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC,
                     ResCSPA, ResCSPB, ResCSPC,
                     RepResCSPA, RepResCSPB, RepResCSPC,
                     ResXCSPA, ResXCSPB, ResXCSPC,
                     RepResXCSPA, RepResXCSPB, RepResXCSPC,
                     GhostCSPA, GhostCSPB, GhostCSPC,
                     STCSPA, STCSPB, STCSPC,
                     ST2CSPA, ST2CSPB, ST2CSPC,C3]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Chuncat:
            c2 = sum([ch[x] for x in f])
        elif m is Shortcut:
            c2 = ch[f[0]]
        elif m is Foldcut:
            c2 = ch[f] // 2
        elif m in [Detect, IDetect, IAuxDetect]:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is ReOrg:
            c2 = ch[f] * 4
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

# yolov7
class IDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False
    export_cat = False  # onnx export cat output
    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(IDetect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5 +12 # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        if self.export_cat:
            for i in range(self.nl):
                x[i] = self.m[i](self.ia[i](x[i]))  # conv
                x[i] = self.im[i](x[i])
                bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    # self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                    self.grid[i], self.anchor_grid[i] = self._make_grid_new(nx, ny, i)

                y = torch.full_like(x[i], 0)
                # ---------------------------------------------------------
                y = y + torch.cat((x[i][:, :, :, :, 0:5].sigmoid(),
                                   torch.cat((x[i][:, :, :, :, 5:17], x[i][:, :, :, :, 17:17 + self.nc].sigmoid()), 4)),
                                  4)

                box_xy = (y[:, :, :, :, 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                box_wh = (y[:, :, :, :, 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                # box_conf = torch.cat((box_xy, torch.cat((box_wh, y[:, :, :, :, 4:5]), 4)), 4)

                landm1 = y[:, :, :, :, 5:7] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
                    i]  # landmark x1 y1
                landm2 = y[:, :, :, :, 7:9] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
                    i]  # landmark x2 y2
                landm3 = y[:, :, :, :, 9:11] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
                    i]  # landmark x3 y3
                landm4 = y[:, :, :, :, 11:13] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
                    i]  # landmark x4 y4
                landm5 = y[:, :, :, :, 13:15] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
                    i]  # landmark x5 y5
                # ---------------------------------------------------------
                landm6 = y[:, :, :, :, 15:17] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
                    i]  # landmark x5 y5
                # landm = torch.cat((landm1, torch.cat((landm2, torch.cat((landm3, torch.cat((landm4, landm5), 4)), 4)), 4)), 4)
                # y = torch.cat((box_conf, torch.cat((landm, y[:, :, :, :, 15:15+self.nc]), 4)), 4)
                # ---------------------------------------------------------
                y = torch.cat([box_xy, box_wh, y[:, :, :, :, 4:5], landm1, landm2, landm3, landm4, landm5, landm6,
                               y[:, :, :, :, 17:17 + self.nc]], -1)

                z.append(y.view(bs, -1, self.no))
            return torch.cat(z, 1)

        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def fuseforward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                    wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z,)
        elif self.concat:
            out = torch.cat(z, 1)
        else:
            out = (torch.cat(z, 1), x)

        return out

    def fuse(self):
        print("IDetect.fuse")
        # fuse ImplicitA and Convolution
        for i in range(len(self.m)):
            c1, c2, _, _ = self.m[i].weight.shape
            c1_, c2_, _, _ = self.ia[i].implicit.shape
            self.m[i].bias += torch.matmul(self.m[i].weight.reshape(c1, c2),
                                           self.ia[i].implicit.reshape(c2_, c1_)).squeeze(1)

        # fuse ImplicitM and Convolution
        for i in range(len(self.m)):
            c1, c2, _, _ = self.im[i].implicit.shape
            self.m[i].bias *= self.im[i].implicit.reshape(c2)
            self.m[i].weight *= self.im[i].implicit.transpose(0, 1)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                      dtype=torch.float32,
                                      device=z.device)
        box @= convert_matrix
        return (box, score)

    def _make_grid_new(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if '1.10.0' in torch.__version__:  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(
            (1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid

from thop import profile
from thop import clever_format

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    stride = model.stride.max()
    if stride == 32:
        input = torch.Tensor(1, 3, 480, 640).to(device)
    else:
        input = torch.Tensor(1, 3, 512, 640).to(device)
    model.train()
    print(model)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print('Flops:', flops, ',Params:', params)
