import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


class SNE(nn.Module):
    """Our SNE takes depth and camera intrinsic parameters as input,
    and outputs normal estimations.
    """

    def __init__(self):
        super(SNE, self).__init__()
        self.kernel_Gx = torch.tensor([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=torch.float32)
        self.kernel_Gy = torch.tensor([[0, -1, 0], [0, 0, 0], [0, 1, 0]], dtype=torch.float32)

    def forward(self, depth, camParam):
        cam_fx, cam_fy, u0, v0 = camParam[0, 0], camParam[1, 1], camParam[0, 2], camParam[1, 2]
        h, w = depth.size()
        vMax, uMax = depth.size()
        # v_map, u_map = torch.meshgrid(torch.arange(h), torch.arange(w))

        u_map = torch.ones((vMax, 1)) * torch.arange(1, uMax + 1) - u0  # u-u0
        v_map = torch.arange(1, vMax + 1).reshape(vMax, 1) * torch.ones((1, uMax)) - v0  # v-v0
        v_map = v_map.type(torch.float32)
        u_map = u_map.type(torch.float32)

        # get partial Z
        Z = depth  # h, w

        Gu = F.conv2d(Z.view(1, 1, h, w), self.kernel_Gx.view(1, 1, 3, 3), padding=1)
        Gv = F.conv2d(Z.view(1, 1, h, w), self.kernel_Gy.view(1, 1, 3, 3), padding=1)

        Nx_t = Gu * cam_fx
        Ny_t = Gv * cam_fy
        Nz_t = -(Z + v_map * Gv + u_map * Gu)

        N_t = torch.stack([Nx_t.squeeze(), Ny_t.squeeze(), Nz_t.squeeze()], dim=0)
        N_t = F.normalize(N_t, p=2, dim=0)
        return N_t
