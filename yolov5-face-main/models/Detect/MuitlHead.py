import argparse
import contextlib
import os
import sys
from copy import deepcopy
import platform
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.Models.research import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)
# from utils.loss import SigmoidBin


# class DecoupledHead(nn.Module):
#     def __init__(self, ch=256, nc=80, anchors=()):
#         super().__init__()
#         self.nc = nc  # number of classes
#         self.nl = len(anchors)  # number of detection layers
#         self.na = len(anchors[0]) // 2  # number of anchors
#         self.merge = Conv(ch, 256, 1, 1)
#         self.cls_convs1 = Conv(256, 256, 3, 1, 1)
#         self.cls_convs2 = Conv(256, 256, 3, 1, 1)
#         self.reg_convs1 = Conv(256, 256, 3, 1, 1)
#         self.reg_convs2 = Conv(256, 256, 3, 1, 1)
#         self.cls_preds = nn.Conv2d(256, self.nc * self.na, 1)
#         self.reg_preds = nn.Conv2d(256, 4 * self.na, 1)
#         self.obj_preds = nn.Conv2d(256, 1 * self.na, 1)
#
#     def forward(self, x):
#         x = self.merge(x)
#         x1 = self.cls_convs1(x)
#         x1 = self.cls_convs2(x1)
#         x1 = self.cls_preds(x1)
#         x2 = self.reg_convs1(x)
#         x2 = self.reg_convs2(x2)
#         x21 = self.reg_preds(x2)
#         x22 = self.obj_preds(x2)
#         out = torch.cat([x21, x22, x1], 1)
#         return out
#
#
# class ASFFV5(nn.Module):
#     def __init__(self, level, multiplier=1, rfb=False, vis=False, act_cfg=True):
#         """
#         ASFF version for YoloV5 .
#         different than YoloV3
#         multiplier should be 1, 0.5
#         which means, the channel of ASFF can be
#         512, 256, 128 -> multiplier=1
#         256, 128, 64 -> multiplier=0.5
#         For even smaller, you need change code manually.
#         """
#         super(ASFFV5, self).__init__()
#         self.level = level
#         self.dim = [int(1024 * multiplier), int(512 * multiplier),
#                     int(256 * multiplier)]
#         # print(self.dim)
#
#         self.inter_dim = self.dim[self.level]
#         if level == 0:
#             self.stride_level_1 = Conv(int(512 * multiplier), self.inter_dim, 3, 2)
#
#             self.stride_level_2 = Conv(int(256 * multiplier), self.inter_dim, 3, 2)
#
#             self.expand = Conv(self.inter_dim, int(
#                 1024 * multiplier), 3, 1)
#         elif level == 1:
#             self.compress_level_0 = Conv(
#                 int(1024 * multiplier), self.inter_dim, 1, 1)
#             self.stride_level_2 = Conv(
#                 int(256 * multiplier), self.inter_dim, 3, 2)
#             self.expand = Conv(self.inter_dim, int(512 * multiplier), 3, 1)
#         elif level == 2:
#             self.compress_level_0 = Conv(
#                 int(1024 * multiplier), self.inter_dim, 1, 1)
#             self.compress_level_1 = Conv(
#                 int(512 * multiplier), self.inter_dim, 1, 1)
#             self.expand = Conv(self.inter_dim, int(
#                 256 * multiplier), 3, 1)
#
#         # when adding rfb, we use half number of channels to save memory
#         compress_c = 8 if rfb else 16
#         self.weight_level_0 = Conv(
#             self.inter_dim, compress_c, 1, 1)
#         self.weight_level_1 = Conv(
#             self.inter_dim, compress_c, 1, 1)
#         self.weight_level_2 = Conv(
#             self.inter_dim, compress_c, 1, 1)
#
#         self.weight_levels = Conv(
#             compress_c * 3, 3, 1, 1)
#         self.vis = vis
#
#     def forward(self, x):  # l,m,s
#         """
#         # 128, 256, 512
#         512, 256, 128
#         from small -> large
#         """
#         x_level_0 = x[2]  # l
#         x_level_1 = x[1]  # m
#         x_level_2 = x[0]  # s
#         # print('x_level_0: ', x_level_0.shape)
#         # print('x_level_1: ', x_level_1.shape)
#         # print('x_level_2: ', x_level_2.shape)
#         if self.level == 0:
#             level_0_resized = x_level_0
#             level_1_resized = self.stride_level_1(x_level_1)
#             level_2_downsampled_inter = F.max_pool2d(
#                 x_level_2, 3, stride=2, padding=1)
#             level_2_resized = self.stride_level_2(level_2_downsampled_inter)
#         elif self.level == 1:
#             level_0_compressed = self.compress_level_0(x_level_0)
#             level_0_resized = F.interpolate(
#                 level_0_compressed, scale_factor=2, mode='nearest')
#             level_1_resized = x_level_1
#             level_2_resized = self.stride_level_2(x_level_2)
#         elif self.level == 2:
#             level_0_compressed = self.compress_level_0(x_level_0)
#             level_0_resized = F.interpolate(
#                 level_0_compressed, scale_factor=4, mode='nearest')
#             x_level_1_compressed = self.compress_level_1(x_level_1)
#             level_1_resized = F.interpolate(
#                 x_level_1_compressed, scale_factor=2, mode='nearest')
#             level_2_resized = x_level_2
#
#         # print('level: {}, l1_resized: {}, l2_resized: {}'.format(self.level,
#         #      level_1_resized.shape, level_2_resized.shape))
#         level_0_weight_v = self.weight_level_0(level_0_resized)
#         level_1_weight_v = self.weight_level_1(level_1_resized)
#         level_2_weight_v = self.weight_level_2(level_2_resized)
#         # print('level_0_weight_v: ', level_0_weight_v.shape)
#         # print('level_1_weight_v: ', level_1_weight_v.shape)
#         # print('level_2_weight_v: ', level_2_weight_v.shape)
#
#         levels_weight_v = torch.cat(
#             (level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
#         levels_weight = self.weight_levels(levels_weight_v)
#         levels_weight = F.softmax(levels_weight, dim=1)
#
#         fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
#                             level_1_resized * levels_weight[:, 1:2, :, :] + \
#                             level_2_resized * levels_weight[:, 2:, :, :]
#
#         out = self.expand(fused_out_reduced)
#
#         if self.vis:
#             return out, levels_weight, fused_out_reduced.sum(dim=1)
#         else:
#             return out
#
#
# class Decoupled_Detect(nn.Module):
#     stride = None  # strides computed during build
#     onnx_dynamic = False  # ONNX export parameter
#     export = False  # export mode
#
#     def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
#         super().__init__()
#
#         self.nc = nc  # number of classes
#         self.no = nc + 5  # number of outputs per anchor
#         self.nl = len(anchors)  # number of detection layers
#         self.na = len(anchors[0]) // 2  # number of anchors
#         self.grid = [torch.zeros(1)] * self.nl  # init grid
#         self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
#         self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
#         self.m=nn.ModuleList(DecoupledHead(x,nc,anchors) for x in ch)
#         self.inplace = inplace  # use in-place ops (e.g. slice assignment)
#
#
#     def forward(self, x):
#         z = []  # inference output
#         for i in range(self.nl):
#             x[i] = self.m[i](x[i])  # conv
#             bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
#             x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
#
#             if not self.training:  # inference
#                 if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
#                     self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
#
#                 y = x[i].sigmoid()
#                 if self.inplace:
#                     y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]  # xy
#                     y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
#                 else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
#                     xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
#                     xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
#                     wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
#                     y = torch.cat((xy, wh, conf), 4)
#                 z.append(y.view(bs, -1, self.no))
#
#         return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)
#
#     def _make_grid(self, nx=20, ny=20, i=0):
#         d = self.anchors[i].device
#         t = self.anchors[i].dtype
#         shape = 1, self.na, ny, nx, 2  # grid shape
#         y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
#         if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
#             yv, xv = torch.meshgrid(y, x, indexing='ij')
#         else:
#             yv, xv = torch.meshgrid(y, x)
#         grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
#         anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
#         return grid, anchor_grid
#
# class ASFF_Detect(nn.Module):   #add ASFFV5 layer and Rfb
#     stride = None  # strides computed during build
#     onnx_dynamic = False  # ONNX export parameter
#     export = False  # export mode
#
#     def __init__(self, nc=80, anchors=(), ch=(), multiplier=0.5,rfb=False,inplace=True):  # detection layer
#         super().__init__()
#         self.nc = nc  # number of classes
#         self.no = nc + 5  # number of outputs per anchor
#         self.nl = len(anchors)  # number of detection layers
#         self.na = len(anchors[0]) // 2  # number of anchors
#         self.grid = [torch.zeros(1)] * self.nl  # init grid
#         self.l0_fusion = ASFFV5(level=0, multiplier=multiplier,rfb=rfb)
#         self.l1_fusion = ASFFV5(level=1, multiplier=multiplier,rfb=rfb)
#         self.l2_fusion = ASFFV5(level=2, multiplier=multiplier,rfb=rfb)
#         self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
#         self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
#         self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
#         self.inplace = inplace  # use in-place ops (e.g. slice assignment)
#
#     def forward(self, x):
#         z = []  # inference output
#         result=[]
#
#         result.append(self.l2_fusion(x))
#         result.append(self.l1_fusion(x))
#         result.append(self.l0_fusion(x))
#         x=result
#         for i in range(self.nl):
#             x[i] = self.m[i](x[i])  # conv
#             bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
#             x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
#
#             if not self.training:  # inference
#                 if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
#                     self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
#
#                 y = x[i].sigmoid() # https://github.com/iscyy/yoloair
#                 if self.inplace:
#                     y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]  # xy
#                     y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
#                 else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
#                     xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
#                     xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
#                     wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
#                     y = torch.cat((xy, wh, conf), 4)
#                 z.append(y.view(bs, -1, self.no))
#
#         return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)
#
#     def _make_grid(self, nx=20, ny=20, i=0):
#         d = self.anchors[i].device
#         t = self.anchors[i].dtype
#         shape = 1, self.na, ny, nx, 2  # grid shape
#         y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
#         if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
#             yv, xv = torch.meshgrid(y, x, indexing='ij')
#         else:
#             yv, xv = torch.meshgrid(y, x)
#         grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
#         anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
#         #print(anchor_grid)
#         return grid, anchor_grid

class IDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(IDetect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5+12  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch) # https://github.com/iscyy/yoloair
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)
    
    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
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

                # y = x[i].sigmoid()
                # if not torch.onnx.is_in_onnx_export():
                #     y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                #     y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                # else:
                #     xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                #     wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].data  # wh
                #     y = torch.cat((xy, wh, y[..., 4:]), -1)
                # z.append(y.view(bs, -1, self.no))

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


        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z, )
        else:
            out = (torch.cat(z, 1), x)

        # return out
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
        return x if self.training else (torch.cat(z, 1), x)
    
    def fuse(self):
        print("IDetect.fuse")
        # fuse ImplicitA and Convolution
        for i in range(len(self.m)):
            c1,c2,_,_ = self.m[i].weight.shape
            c1_,c2_, _,_ = self.ia[i].implicit.shape
            self.m[i].bias += torch.matmul(self.m[i].weight.reshape(c1,c2),self.ia[i].implicit.reshape(c2_,c1_)).squeeze(1)

        # fuse ImplicitM and Convolution
        for i in range(len(self.m)):
            c1,c2, _,_ = self.im[i].implicit.shape
            self.m[i].bias *= self.im[i].implicit.reshape(c2)
            self.m[i].weight *= self.im[i].implicit.transpose(0,1)
            
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

class IAuxDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(IAuxDetect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch[:self.nl])  # output conv
        self.m2 = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch[self.nl:])  # output conv
        
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch[:self.nl])
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch[:self.nl])

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            x[i+self.nl] = self.m2[i](x[i+self.nl])
            x[i+self.nl] = x[i+self.nl].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].data  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x[:self.nl])
        
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
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].data  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z, )
        else:
            out = (torch.cat(z, 1), x)

        return out

    def fuse(self):
        print("IAuxDetect.fuse")
        # fuse ImplicitA and Convolution
        for i in range(len(self.m)):
            c1,c2,_,_ = self.m[i].weight.shape
            c1_,c2_, _,_ = self.ia[i].implicit.shape
            self.m[i].bias += torch.matmul(self.m[i].weight.reshape(c1,c2),self.ia[i].implicit.reshape(c2_,c1_)).squeeze(1)

        # fuse ImplicitM and Convolution
        for i in range(len(self.m)):
            c1,c2, _,_ = self.im[i].implicit.shape
            self.m[i].bias *= self.im[i].implicit.reshape(c2)
            self.m[i].weight *= self.im[i].implicit.transpose(0,1)
            
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