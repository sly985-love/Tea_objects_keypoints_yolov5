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


# class IDetect(nn.Module):
#     stride = None  # strides computed during build
#     export = False  # onnx export
#     end2end = False
#     include_nms = False
#     concat = False
#
#     def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
#         super(IDetect, self).__init__()
#         self.nc = nc  # number of classes
#         self.no = nc + 5 +12 # number of outputs per anchor
#         self.nl = len(anchors)  # number of detection layers
#         self.na = len(anchors[0]) // 2  # number of anchors
#         self.grid = [torch.zeros(1)] * self.nl  # init grid
#         a = torch.tensor(anchors).float().view(self.nl, -1, 2)
#         self.register_buffer('anchors', a)  # shape(nl,na,2)
#         self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
#         self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
#
#         self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
#         self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)
#
#     def forward(self, x):
#         # x = x.copy()  # for profiling
#         z = []  # inference output
#         self.training |= self.export
#         if self.export_cat:
#             for i in range(self.nl):
#                 x[i] = self.m[i](self.ia[i](x[i]))  # conv
#                 x[i] = self.im[i](x[i])
#                 bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
#                 x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
#
#                 if self.grid[i].shape[2:4] != x[i].shape[2:4]:
#                     # self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
#                     self.grid[i], self.anchor_grid[i] = self._make_grid_new(nx, ny, i)
#
#                 y = torch.full_like(x[i], 0)
#                 # ---------------------------------------------------------
#                 y = y + torch.cat((x[i][:, :, :, :, 0:5].sigmoid(),
#                                    torch.cat((x[i][:, :, :, :, 5:17], x[i][:, :, :, :, 17:17 + self.nc].sigmoid()), 4)),
#                                   4)
#
#                 box_xy = (y[:, :, :, :, 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
#                 box_wh = (y[:, :, :, :, 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
#                 # box_conf = torch.cat((box_xy, torch.cat((box_wh, y[:, :, :, :, 4:5]), 4)), 4)
#
#                 landm1 = y[:, :, :, :, 5:7] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
#                     i]  # landmark x1 y1
#                 landm2 = y[:, :, :, :, 7:9] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
#                     i]  # landmark x2 y2
#                 landm3 = y[:, :, :, :, 9:11] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
#                     i]  # landmark x3 y3
#                 landm4 = y[:, :, :, :, 11:13] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
#                     i]  # landmark x4 y4
#                 landm5 = y[:, :, :, :, 13:15] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
#                     i]  # landmark x5 y5
#                 # ---------------------------------------------------------
#                 landm6 = y[:, :, :, :, 15:17] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
#                     i]  # landmark x5 y5
#                 # landm = torch.cat((landm1, torch.cat((landm2, torch.cat((landm3, torch.cat((landm4, landm5), 4)), 4)), 4)), 4)
#                 # y = torch.cat((box_conf, torch.cat((landm, y[:, :, :, :, 15:15+self.nc]), 4)), 4)
#                 # ---------------------------------------------------------
#                 y = torch.cat([box_xy, box_wh, y[:, :, :, :, 4:5], landm1, landm2, landm3, landm4, landm5, landm6,
#                                y[:, :, :, :, 17:17 + self.nc]], -1)
#
#                 z.append(y.view(bs, -1, self.no))
#             return torch.cat(z, 1)
#         for i in range(self.nl):
#             x[i] = self.m[i](self.ia[i](x[i]))  # conv
#             x[i] = self.im[i](x[i])
#             bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
#             x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
#
#             if not self.training:  # inference
#                 if self.grid[i].shape[2:4] != x[i].shape[2:4]:
#                     self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
#
#                 # ---sly热力图可视化
#                 logits = x[i][..., 5:]
#
#                 y = torch.full_like(x[i], 0)
#                 # --------------------------------------
#                 class_range = list(range(5)) + list(range(17, 17 + self.nc))
#                 y[..., class_range] = x[i][..., class_range].sigmoid()
#                 # --------------------------------------
#                 y[..., 5:17] = x[i][..., 5:17]
#                 # y = x[i].sigmoid()
#
#                 y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
#                 y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
#
#                 # y[..., 5:15] = y[..., 5:15] * 8 - 4
#                 y[..., 5:7] = y[..., 5:7] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
#                     i]  # landmark x1 y1
#                 y[..., 7:9] = y[..., 7:9] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
#                     i]  # landmark x2 y2
#                 y[..., 9:11] = y[..., 9:11] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
#                     i]  # landmark x3 y3
#                 y[..., 11:13] = y[..., 11:13] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
#                     i]  # landmark x4 y4
#                 y[..., 13:15] = y[..., 13:15] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
#                     i]  # landmark x5 y5
#                 # --------------------------------------
#                 y[..., 15:17] = y[..., 15:17] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[
#                     i]
#
#         # for i in range(self.nl):
#         #     x[i] = self.m[i](self.ia[i](x[i]))  # conv
#         #     x[i] = self.im[i](x[i])
#         #     bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
#         #     x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
#         #
#         #     if not self.training:  # inference
#         #         if self.grid[i].shape[2:4] != x[i].shape[2:4]:
#         #             self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
#         #
#         #         y = x[i].sigmoid()
#         #         y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
#         #         y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
#         #         z.append(y.view(bs, -1, self.no))
#
#         return x if self.training else (torch.cat(z, 1), x)
#
#     def fuseforward(self, x):
#         # x = x.copy()  # for profiling
#         z = []  # inference output
#         self.training |= self.export
#         for i in range(self.nl):
#             x[i] = self.m[i](x[i])  # conv
#             bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
#             x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
#
#             if not self.training:  # inference
#                 if self.grid[i].shape[2:4] != x[i].shape[2:4]:
#                     self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
#
#                 y = x[i].sigmoid()
#                 if not torch.onnx.is_in_onnx_export():
#                     y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
#                     y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
#                 else:
#                     xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
#                     xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
#                     wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
#                     y = torch.cat((xy, wh, conf), 4)
#                 z.append(y.view(bs, -1, self.no))
#
#         if self.training:
#             out = x
#         elif self.end2end:
#             out = torch.cat(z, 1)
#         elif self.include_nms:
#             z = self.convert(z)
#             out = (z,)
#         elif self.concat:
#             out = torch.cat(z, 1)
#         else:
#             out = (torch.cat(z, 1), x)
#
#         return out
#
#     def fuse(self):
#         print("IDetect.fuse")
#         # fuse ImplicitA and Convolution
#         for i in range(len(self.m)):
#             c1, c2, _, _ = self.m[i].weight.shape
#             c1_, c2_, _, _ = self.ia[i].implicit.shape
#             self.m[i].bias += torch.matmul(self.m[i].weight.reshape(c1, c2),
#                                            self.ia[i].implicit.reshape(c2_, c1_)).squeeze(1)
#
#         # fuse ImplicitM and Convolution
#         for i in range(len(self.m)):
#             c1, c2, _, _ = self.im[i].implicit.shape
#             self.m[i].bias *= self.im[i].implicit.reshape(c2)
#             self.m[i].weight *= self.im[i].implicit.transpose(0, 1)
#
#     @staticmethod
#     def _make_grid(nx=20, ny=20):
#         yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
#         return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
#
#     def convert(self, z):
#         z = torch.cat(z, 1)
#         box = z[:, :, :4]
#         conf = z[:, :, 4:5]
#         score = z[:, :, 5:]
#         score *= conf
#         convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
#                                       dtype=torch.float32,
#                                       device=z.device)
#         box @= convert_matrix
#         return (box, score)
#
#     def _make_grid_new(self, nx=20, ny=20, i=0):
#         d = self.anchors[i].device
#         if '1.10.0' in torch.__version__:  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
#             yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
#         else:
#             yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
#         grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
#         anchor_grid = (self.anchors[i].clone() * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(
#             (1, self.na, ny, nx, 2)).float()
#         return grid, anchor_grid


class IDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(IDetect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5 + 12 # number of outputs per anchor
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


# 对YoloX代码训练速度优化
class DetectX(nn.Module):
    stride = [8, 16, 32]
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self,
                 num_classes,
                 anchors=1,
                 in_channels=(128, 128, 128, 128, 128, 128),
                 inplace=True,
                 prior_prob=1e-2,):  # detection layer
        super().__init__()
        if isinstance(anchors, (list, tuple)):
            self.n_anchors = len(anchors)
        else:
            self.n_anchors = anchors
        self.num_classes = num_classes

        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()

        cls_in_channels = in_channels[0::2]
        reg_in_channels = in_channels[1::2]
        for cls_in_channel, reg_in_channel in zip(cls_in_channels, reg_in_channels):
            cls_pred = nn.Conv2d(
                    in_channels=cls_in_channel,
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            reg_pred = nn.Conv2d(
                    in_channels=reg_in_channel,
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            obj_pred = nn.Conv2d(
                    in_channels=reg_in_channel,
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            self.cls_preds.append(cls_pred)
            self.reg_preds.append(reg_pred)
            self.obj_preds.append(obj_pred)

        self.nc = self.num_classes  # number of classes
        # self.no = self.num_classes + 5  # number of outputs per anchor
        self.nl = len(cls_in_channels)  # number of detection layers
        self.na = self.n_anchors  # number of anchors

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.grids = [torch.zeros(1)] * len(in_channels)    # 用于保存每层的每个网格的坐标
        self.xy_shifts = [torch.zeros(1)] * len(in_channels)
        self.org_grids = [torch.zeros(1)] * len(in_channels)
        self.grid_sizes = [[0, 0, 0] for _ in range(len(in_channels))]
        self.expanded_strides = [torch.zeros(1)] * len(in_channels)
        self.center_ltrbes = [torch.zeros(1)] * len(in_channels)
        # gt框中心点的2.5个网格半径的矩形框内的anchor
        self.center_radius = 2.5

        self.prior_prob = prior_prob
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def initialize_biases(self):
        prior_prob = self.prior_prob
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _forward(self, xin):
        outputs = []
        cls_preds = []
        bbox_preds = []
        obj_preds = []
        origin_preds = []
        org_xy_shifts = []
        xy_shifts = []
        expanded_strides = []
        center_ltrbes = []

        cls_xs = xin[0::2]
        reg_xs = xin[1::2]
        in_type = xin[0].type()
        h, w = reg_xs[0].shape[2:4]
        h *= self.stride[0]
        w *= self.stride[0]
        for k, (stride_this_level, cls_x, reg_x) in enumerate(zip(self.stride, cls_xs, reg_xs)):
            cls_output = self.cls_preds[k](cls_x)  # [batch_size, num_classes, hsize, wsize]

            reg_output = self.reg_preds[k](reg_x)  # [batch_size, 4, hsize, wsize]
            obj_output = self.obj_preds[k](reg_x)  # [batch_size, 1, hsize, wsize]

            if self.training:
                batch_size = cls_output.shape[0]
                hsize, wsize = cls_output.shape[-2:]
                size = hsize * wsize
                cls_output = cls_output.view(batch_size, -1, size).permute(0, 2, 1).contiguous()  # [batch_size, num_classes, hsize*wsize] -> [batch_size, hsize*wsize, num_classes]
                reg_output = reg_output.view(batch_size, 4, size).permute(0, 2, 1).contiguous()  # [batch_size, 4, hsize*wsize] -> [batch_size, hsize*wsize, 4]
                obj_output = obj_output.view(batch_size, 1, size).permute(0, 2, 1).contiguous()  # [batch_size, 1, hsize*wsize] -> [batch_size, hsize*wsize, 1]
                if self.use_l1:
                    origin_preds.append(reg_output.clone())
                output, grid, xy_shift, expanded_stride, center_ltrb = self.get_output_and_grid(reg_output, hsize, wsize, k, stride_this_level, in_type)

                org_xy_shifts.append(grid)  # 网格x, y坐标, [1, 1*hsize*wsize, 2]
                xy_shifts.append(xy_shift)  # 网格x, y坐标, [1, 1*hsize*wsize, 2]
                expanded_strides.append(expanded_stride)   # dims: [1, hsize*wsize]
                center_ltrbes.append(center_ltrb)  # [1, 1*hsize*wsize, 4]
                cls_preds.append(cls_output)
                bbox_preds.append(output)
                obj_preds.append(obj_output)
            else:
                output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
                outputs.append(output)

        if self.training:
            bbox_preds = torch.cat(bbox_preds, 1)  # [batch, n_anchors_all, 4]
            obj_preds = torch.cat(obj_preds, 1)  # [batch, n_anchors_all, 1]
            cls_preds = torch.cat(cls_preds, 1)  # [batch, n_anchors_all, n_cls]

            org_xy_shifts = torch.cat(org_xy_shifts, 1)  # [1, n_anchors_all, 2]
            xy_shifts = torch.cat(xy_shifts, 1)  # [1, n_anchors_all, 2]
            expanded_strides = torch.cat(expanded_strides, 1)
            center_ltrbes = torch.cat(center_ltrbes, 1)  # [1, n_anchors_all, 4]

            if self.use_l1:
                origin_preds = torch.cat(origin_preds, 1)  # dims: [n, n_anchors_all, 4]
            else:
                origin_preds = bbox_preds.new_zeros(1)

            whwh = torch.Tensor([[w, h, w, h]]).type_as(bbox_preds)

            return (bbox_preds,
                    cls_preds,
                    obj_preds,
                    origin_preds,
                    org_xy_shifts,
                    xy_shifts,
                    expanded_strides,
                    center_ltrbes,
                    whwh,)
        else:
            return outputs

    def forward(self, x):
        outputs = self._forward(x)

        if self.training:
            return outputs
        else:
            self.hw = [out.shape[-2:] for out in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [out.flatten(start_dim=2) for out in outputs], dim=2
            ).permute(0, 2, 1)
            outputs = self.decode_outputs(outputs, dtype=x[0].type())
            return (outputs, )

    def forward_export(self, x):
        cls_xs = x[0::2]
        reg_xs = x[1::2]
        outputs = []
        for k, (stride_this_level, cls_x, reg_x) in enumerate(zip(self.stride, cls_xs, reg_xs)):
            cls_output = self.cls_preds[k](cls_x)  # [batch_size, num_classes, hsize, wsize]

            reg_output = self.reg_preds[k](reg_x)  # [batch_size, 4, hsize, wsize]
            obj_output = self.obj_preds[k](reg_x)  # [batch_size, 1, hsize, wsize]

            output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
            outputs.append(output)
        outputs = torch.cat(
            [out.flatten(start_dim=2) for out in outputs], dim=2
        ).permute(0, 2, 1)
        return outputs

    def get_output_and_grid(self, reg_box, hsize, wsize, k, stride, dtype):
        grid_size = self.grid_sizes[k]
        if (grid_size[0] != hsize) or (grid_size[1] != wsize) or (grid_size[2] != stride):
            grid_size[0] = hsize
            grid_size[1] = wsize
            grid_size[2] = stride

            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2).type(dtype).contiguous()  # [1, 1*hsize*wsize, 2]
            self.grids[k] = grid
            xy_shift = (grid + 0.5)*stride
            self.xy_shifts[k] = xy_shift
            expanded_stride = torch.full((1, grid.shape[1], 1), stride, dtype=grid.dtype, device=grid.device)
            self.expanded_strides[k] = expanded_stride
            center_radius = self.center_radius*expanded_stride
            center_radius = center_radius.expand_as(xy_shift)
            center_lt = center_radius + xy_shift
            center_rb = center_radius - xy_shift
            center_ltrb = torch.cat([center_lt, center_rb], dim=-1)
            self.center_ltrbes[k] = center_ltrb

        xy_shift = self.xy_shifts[k]
        grid = self.grids[k]
        expanded_stride = self.expanded_strides[k]
        center_ltrb = self.center_ltrbes[k]

        # l, t, r, b
        half_wh = torch.exp(reg_box[..., 2:4]) * (stride/2)  # （第k层）预测物体的半宽高
        reg_box[..., :2] = (reg_box[..., :2]+grid)*stride  # （第k层）预测物体的中心坐标
        reg_box[..., 2:4] = reg_box[..., :2] + half_wh  # （第k层）预测物体的右下坐标
        reg_box[..., :2] = reg_box[..., :2] - half_wh  # （第k层）预测物体的左上坐标

        return reg_box, grid, xy_shift, expanded_stride, center_ltrb

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.stride):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def get_losses(
        self,
        bbox_preds,
        cls_preds,
        obj_preds,
        origin_preds,
        org_xy_shifts,
        xy_shifts,
        expanded_strides,
        center_ltrbes,
        whwh,
        labels,
        dtype,
    ):
        # calculate targets
        nlabel = labels[:, 0].long().bincount(minlength=cls_preds.shape[0]).tolist()
        batch_gt_classes = labels[:, 1].type_as(cls_preds).contiguous()  # [num_gt, 1]
        batch_org_gt_bboxes = labels[:, 2:6].contiguous()  # [num_gt, 4]  bbox: cx, cy, w, h
        batch_org_gt_bboxes.mul_(whwh)
        batch_gt_bboxes = torch.empty_like(batch_org_gt_bboxes)  # [num_gt, 4]  bbox: l, t, r, b
        batch_gt_half_wh = batch_org_gt_bboxes[:, 2:]/2
        batch_gt_bboxes[:, :2] = batch_org_gt_bboxes[:, :2] - batch_gt_half_wh
        batch_gt_bboxes[:, 2:] = batch_org_gt_bboxes[:, :2] + batch_gt_half_wh
        batch_org_gt_bboxes = batch_org_gt_bboxes.type_as(bbox_preds)
        batch_gt_bboxes = batch_gt_bboxes.type_as(bbox_preds)
        del batch_gt_half_wh

        total_num_anchors = bbox_preds.shape[1]

        cls_targets = []
        reg_targets = []
        l1_targets = []
        fg_mask_inds = []

        num_fg = 0.0
        num_gts = 0
        index_offset = 0
        batch_size = bbox_preds.shape[0]
        for batch_idx in range(batch_size):
            num_gt = int(nlabel[batch_idx])
            if num_gt == 0:
                cls_target = bbox_preds.new_zeros((0, self.num_classes))
                reg_target = bbox_preds.new_zeros((0, 4))
                l1_target = bbox_preds.new_zeros((0, 4))
            else:
                _num_gts = num_gts + num_gt
                org_gt_bboxes_per_image = batch_org_gt_bboxes[num_gts:_num_gts]  # [num_gt, 4]  bbox: cx, cy, w, h
                gt_bboxes_per_image = batch_gt_bboxes[num_gts:_num_gts]  # [num_gt, 4]  bbox: l, t, r, b
                gt_classes = batch_gt_classes[num_gts:_num_gts]  # [num_gt]
                num_gts = _num_gts
                bboxes_preds_per_image = bbox_preds[batch_idx]  # [n_anchors_all, 4]
                cls_preds_per_image = cls_preds[batch_idx]  # [n_anchors_all, n_cls]
                obj_preds_per_image = obj_preds[batch_idx]  # [n_anchors_all, 1]

                try:
                    (
                        gt_matched_classes,
                        fg_mask_ind,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        num_gt,
                        total_num_anchors,
                        org_gt_bboxes_per_image,
                        gt_bboxes_per_image,
                        gt_classes,
                        self.num_classes,
                        bboxes_preds_per_image,
                        cls_preds_per_image,
                        obj_preds_per_image,
                        center_ltrbes,
                        xy_shifts,
                    )
                except RuntimeError:
                    LOGGER.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    print("------------CPU Mode for This Batch-------------")
                    _org_gt_bboxes_per_image = org_gt_bboxes_per_image.cpu().float()
                    _gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
                    _bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
                    _cls_preds_per_image = cls_preds_per_image.cpu().float()
                    _obj_preds_per_image = obj_preds_per_image.cpu().float()
                    _gt_classes = gt_classes.cpu().float()
                    _center_ltrbes = center_ltrbes.cpu().float()
                    _xy_shifts = xy_shifts.cpu()

                    (
                        gt_matched_classes,
                        fg_mask_ind,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        num_gt,
                        total_num_anchors,
                        _org_gt_bboxes_per_image,
                        _gt_bboxes_per_image,
                        _gt_classes,
                        self.num_classes,
                        _bboxes_preds_per_image,
                        _cls_preds_per_image,
                        _obj_preds_per_image,
                        _center_ltrbes,
                        _xy_shifts
                    )

                    gt_matched_classes = gt_matched_classes.cuda()
                    fg_mask_ind = fg_mask_ind.cuda()
                    pred_ious_this_matching = pred_ious_this_matching.cuda()
                    matched_gt_inds = matched_gt_inds.cuda()

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.view(-1, 1)  # [num_gt, num_classes]
                reg_target = gt_bboxes_per_image[matched_gt_inds]  # [num_gt, 4]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        bbox_preds.new_empty((num_fg_img, 4)),
                        org_gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask_ind],
                        xy_shifts=org_xy_shifts[0][fg_mask_ind],
                    )
                if index_offset > 0:
                    fg_mask_ind.add_(index_offset)
                fg_mask_inds.append(fg_mask_ind)
            index_offset += total_num_anchors

            cls_targets.append(cls_target)  # [num_gt, num_classes]
            reg_targets.append(reg_target)  # [num_gt, 4]
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)  # [batch_size*num_gt, num_classes]
        reg_targets = torch.cat(reg_targets, 0)  # [batch_size*num_gt, 4]
        fg_mask_inds = torch.cat(fg_mask_inds, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_mask_inds], reg_targets, True)
        ).sum() / num_fg
        obj_preds = obj_preds.view(-1, 1)
        obj_targets = torch.zeros_like(obj_preds).index_fill_(0, fg_mask_inds, 1)
        loss_obj = (
            self.bcewithlog_loss(obj_preds, obj_targets)
        ).sum() / num_fg
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_mask_inds], cls_targets
            )
        ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_mask_inds], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = torch.zeros_like(loss_iou)

        reg_weight = 5.0
        loss_iou = reg_weight * loss_iou
        loss = loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    @staticmethod
    def get_l1_target(l1_target, gt, stride, xy_shifts, eps=1e-8):
        l1_target[:, 0:2] = gt[:, 0:2] / stride - xy_shifts
        l1_target[:, 2:4] = torch.log(gt[:, 2:4] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
            self,
            num_gt,
            total_num_anchors,
            org_gt_bboxes_per_image,  # [num_gt, 4]
            gt_bboxes_per_image,  # [num_gt, 4]
            gt_classes,  # [num_gt]
            num_classes,
            bboxes_preds_per_image,  # [n_anchors_all, 4]
            cls_preds_per_image,  # [n_anchors_all, n_cls]
            obj_preds_per_image,  # [n_anchors_all, 1]
            center_ltrbes,  # [1, n_anchors_all, 4]
            xy_shifts,  # [1, n_anchors_all, 2]
    ):
        fg_mask_inds, is_in_boxes_and_center = self.get_in_boxes_info(
            org_gt_bboxes_per_image,  # [num_gt, 4]
            gt_bboxes_per_image,  # [num_gt, 4]
            center_ltrbes,  # [1, n_anchors_all, 4]
            xy_shifts,  # [1, n_anchors_all, 2]
            total_num_anchors,
            num_gt,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask_inds]  # [fg_count, 4]
        cls_preds_ = cls_preds_per_image[fg_mask_inds]  # [fg_count, num_classes]
        obj_preds_ = obj_preds_per_image[fg_mask_inds]  # [fg_count, 1]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]  # num_in_boxes_anchor == fg_count

        pair_wise_ious = self.bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, True, inplace=True)
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)  # [num_gt, fg_count]

        cls_preds_ = cls_preds_.float().sigmoid_().unsqueeze(0).expand(num_gt, num_in_boxes_anchor, num_classes)
        obj_preds_ = obj_preds_.float().sigmoid_().unsqueeze(0).expand(num_gt, num_in_boxes_anchor, 1)
        cls_preds_ = (cls_preds_ * obj_preds_).sqrt_()  # [num_gt, fg_count, num_classes]

        del obj_preds_

        gt_cls_per_image = F.one_hot(gt_classes.to(torch.int64), num_classes).float()  # [num_gt, num_classes]
        gt_cls_per_image = gt_cls_per_image[:, None, :].expand(num_gt, num_in_boxes_anchor, num_classes)

        with autocast(enabled=False):
            pair_wise_cls_loss = F.binary_cross_entropy(cls_preds_, gt_cls_per_image, reduction="none").sum(-1)  # [num_gt, fg_count]
        del cls_preds_, gt_cls_per_image

        # 负例给非常大的cost（100000.0及以上）
        cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_ious_loss
                + 100000.0 * (~is_in_boxes_and_center)
        )  # [num_gt, fg_count]
        del pair_wise_cls_loss, pair_wise_ious_loss, is_in_boxes_and_center

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
            fg_mask_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask_inds)
        del cost, pair_wise_ious

        return (
            gt_matched_classes,
            fg_mask_inds,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    @staticmethod
    def get_in_boxes_info(
            org_gt_bboxes_per_image,  # [num_gt, 4]
            gt_bboxes_per_image,  # [num_gt, 4]
            center_ltrbes,  # [1, n_anchors_all, 4]
            xy_shifts,  # [1, n_anchors_all, 2]
            total_num_anchors,
            num_gt,
    ):
        xy_centers_per_image = xy_shifts.expand(num_gt, total_num_anchors, 2)
        gt_bboxes_per_image = gt_bboxes_per_image[:, None, :].expand(num_gt, total_num_anchors, 4)

        b_lt = xy_centers_per_image - gt_bboxes_per_image[..., :2]
        b_rb = gt_bboxes_per_image[..., 2:] - xy_centers_per_image
        bbox_deltas = torch.cat([b_lt, b_rb], 2)  # [n_gt, n_anchor, 4]
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0  # [_n_gt, _n_anchor]
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        center_ltrbes = center_ltrbes.expand(num_gt, total_num_anchors, 4)
        org_gt_xy_center = org_gt_bboxes_per_image[:, 0:2]
        org_gt_xy_center = torch.cat([-org_gt_xy_center, org_gt_xy_center], dim=-1)
        org_gt_xy_center = org_gt_xy_center[:, None, :].expand(num_gt, total_num_anchors, 4)
        center_deltas = org_gt_xy_center + center_ltrbes
        is_in_centers = center_deltas.min(dim=-1).values > 0.0  # [_n_gt, _n_anchor]
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all  # fg_mask [n_anchors_all]

        is_in_boxes_and_center = (
                is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return torch.nonzero(is_in_boxes_anchor)[..., 0], is_in_boxes_and_center

    @staticmethod
    def dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask_inds):
        # Dynamic K
        # ---------------------------------------------------------------
        device = cost.device
        matching_matrix = torch.zeros(cost.shape, dtype=torch.uint8, device=device)  # [num_gt, fg_count]

        ious_in_boxes_matrix = pair_wise_ious  # [num_gt, fg_count]
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = topk_ious.sum(1).int().clamp_min_(1)
        if num_gt > 3:
            min_k, max_k = torch._aminmax(dynamic_ks)
            min_k, max_k = min_k.item(), max_k.item()
            if min_k != max_k:
                offsets = torch.arange(0, matching_matrix.shape[0] * matching_matrix.shape[1],
                                       step=matching_matrix.shape[1], dtype=torch.int, device=device)[:, None]
                masks = (torch.arange(0, max_k, dtype=dynamic_ks.dtype, device=device)[None, :].expand(num_gt, max_k) < dynamic_ks[:, None])
                _, pos_idxes = torch.topk(cost, k=max_k, dim=1, largest=False)
                pos_idxes.add_(offsets)
                pos_idxes = torch.masked_select(pos_idxes, masks)
                matching_matrix.view(-1).index_fill_(0, pos_idxes, 1)
                del topk_ious, dynamic_ks, pos_idxes, offsets, masks
            else:
                _, pos_idxes = torch.topk(cost, k=max_k, dim=1, largest=False)
                matching_matrix.scatter_(1, pos_idxes, 1)
                del topk_ious, dynamic_ks
        else:
            ks = dynamic_ks.tolist()
            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(cost[gt_idx], k=ks[gt_idx], largest=False)
                matching_matrix[gt_idx][pos_idx] = 1
            del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        anchor_matching_one_more_gt_mask = anchor_matching_gt > 1

        anchor_matching_one_more_gt_inds = torch.nonzero(anchor_matching_one_more_gt_mask)
        if anchor_matching_one_more_gt_inds.shape[0] > 0:
            anchor_matching_one_more_gt_inds = anchor_matching_one_more_gt_inds[..., 0]
            # _, cost_argmin = torch.min(cost[:, anchor_matching_one_more_gt_inds], dim=0)
            _, cost_argmin = torch.min(cost.index_select(1, anchor_matching_one_more_gt_inds), dim=0)
            # matching_matrix[:, anchor_matching_one_more_gt_inds] = 0
            matching_matrix.index_fill_(1, anchor_matching_one_more_gt_inds, 0)
            matching_matrix[cost_argmin, anchor_matching_one_more_gt_inds] = 1
            # fg_mask_inboxes = matching_matrix.sum(0) > 0
            fg_mask_inboxes = matching_matrix.any(dim=0)
            fg_mask_inboxes_inds = torch.nonzero(fg_mask_inboxes)[..., 0]
        else:
            fg_mask_inboxes_inds = torch.nonzero(anchor_matching_gt)[..., 0]
        num_fg = fg_mask_inboxes_inds.shape[0]

        matched_gt_inds = matching_matrix.index_select(1, fg_mask_inboxes_inds).argmax(0)
        fg_mask_inds = fg_mask_inds[fg_mask_inboxes_inds]
        gt_matched_classes = gt_classes[matched_gt_inds]

        # pred_ious_this_matching = pair_wise_ious[:, fg_mask_inboxes_inds][matched_gt_inds, torch.arange(0, matched_gt_inds.shape[0])]  # [matched_gt_inds_count]
        pred_ious_this_matching = pair_wise_ious.index_select(1, fg_mask_inboxes_inds).gather(dim=0, index=matched_gt_inds[None, :])  # [1, matched_gt_inds_count]

        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds, fg_mask_inds

    @staticmethod
    def bboxes_iou(bboxes_a, bboxes_b, xyxy=True, inplace=False):
        # if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        #     raise IndexError

        if inplace:
            if xyxy:
                tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
                br_hw = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
                br_hw.sub_(tl)  # hw
                br_hw.clamp_min_(0)  # [rows, 2]
                del tl
                area_ious = torch.prod(br_hw, 2)  # area
                del br_hw
                area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
                area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
            else:
                tl = torch.max(
                    (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                    (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
                )
                br_hw = torch.min(
                    (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                    (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
                )
                br_hw.sub_(tl)  # hw
                br_hw.clamp_min_(0)  # [rows, 2]
                del tl
                area_ious = torch.prod(br_hw, 2)  # area
                del br_hw
                area_a = torch.prod(bboxes_a[:, 2:], 1)
                area_b = torch.prod(bboxes_b[:, 2:], 1)

            union = (area_a[:, None] + area_b - area_ious)
            area_ious.div_(union)  # ious

            return area_ious
        else:
            if xyxy:
                tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
                br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
                area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
                area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
            else:
                tl = torch.max(
                    (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                    (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
                )
                br = torch.min(
                    (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                    (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
                )

                area_a = torch.prod(bboxes_a[:, 2:], 1)
                area_b = torch.prod(bboxes_b[:, 2:], 1)

            hw = (br - tl).clamp(min=0)  # [rows, 2]
            area_i = torch.prod(hw, 2)

            ious = area_i / (area_a[:, None] + area_b - area_i)
            return ious

