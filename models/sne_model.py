import torch
import torch.nn as nn
import torch.nn.functional as F


class SNE(nn.Module):
    """Our SNE takes depth and camera intrinsic parameters as input,
    and outputs normal estimations.
    """
    def __init__(self):
        super(SNE, self).__init__()

    def forward(self, depth, camParam):
        h,w = depth.size()
        v_map, u_map = torch.meshgrid(torch.arange(h), torch.arange(w))
        v_map = v_map.type(torch.float32)
        u_map = u_map.type(torch.float32)

        Z = depth   # h, w
        Y = Z.mul((v_map - camParam[1,2])) / camParam[0,0]  # h, w
        X = Z.mul((u_map - camParam[0,2])) / camParam[0,0]  # h, w
        Z[Y <= 0] = 0
        Y[Y <= 0] = 0
        Z[torch.isnan(Z)] = 0
        D = torch.div(torch.ones(h, w), Z)  # h, w

        Gx = torch.tensor([[0,0,0],[-1,0,1],[0,0,0]], dtype=torch.float32)
        Gy = torch.tensor([[0,-1,0],[0,0,0],[0,1,0]], dtype=torch.float32)

        Gu = F.conv2d(D.view(1,1,h,w), Gx.view(1,1,3,3), padding=1)
        Gv = F.conv2d(D.view(1,1,h,w), Gy.view(1,1,3,3), padding=1)

        nx_t = Gu * camParam[0,0]   # 1, 1, h, w
        ny_t = Gv * camParam[1,1]   # 1, 1, h, w

        phi = torch.atan(torch.div(ny_t, nx_t)) + torch.ones([1,1,h,w])*3.141592657
        a = torch.cos(phi)
        b = torch.sin(phi)

        diffKernelArray = torch.tensor([[-1, 0, 0, 0, 1, 0, 0, 0, 0],
                                        [ 0,-1, 0, 0, 1, 0, 0, 0, 0],
                                        [ 0, 0,-1, 0, 1, 0, 0, 0, 0],
                                        [ 0, 0, 0,-1, 1, 0, 0, 0, 0],
                                        [ 0, 0, 0, 0, 1,-1, 0, 0, 0],
                                        [ 0, 0, 0, 0, 1, 0,-1, 0, 0],
                                        [ 0, 0, 0, 0, 1, 0, 0,-1, 0],
                                        [ 0, 0, 0, 0, 1, 0, 0, 0,-1]], dtype=torch.float32)

        sum_nx = torch.zeros((1,1,h,w), dtype=torch.float32)
        sum_ny = torch.zeros((1,1,h,w), dtype=torch.float32)
        sum_nz = torch.zeros((1,1,h,w), dtype=torch.float32)

        for i in range(8):
            diffKernel = diffKernelArray[i].view(1,1,3,3)
            X_d = F.conv2d(X.view(1,1,h,w), diffKernel, padding=1)
            Y_d = F.conv2d(Y.view(1,1,h,w), diffKernel, padding=1)
            Z_d = F.conv2d(Z.view(1,1,h,w), diffKernel, padding=1)

            nz_i = torch.div((torch.mul(nx_t, X_d) + torch.mul(ny_t, Y_d)), Z_d)
            norm = torch.sqrt(torch.mul(nx_t, nx_t) + torch.mul(ny_t, ny_t) + torch.mul(nz_i, nz_i))
            nx_t_i = torch.div(nx_t, norm)
            ny_t_i = torch.div(ny_t, norm)
            nz_t_i = torch.div(nz_i, norm)

            nx_t_i[torch.isnan(nx_t_i)] = 0
            ny_t_i[torch.isnan(ny_t_i)] = 0
            nz_t_i[torch.isnan(nz_t_i)] = 0

            sum_nx = sum_nx + nx_t_i
            sum_ny = sum_ny + ny_t_i
            sum_nz = sum_nz + nz_t_i

        theta = -torch.atan(torch.div((torch.mul(sum_nx, a) + torch.mul(sum_ny, b)), sum_nz))
        nx = torch.mul(torch.sin(theta), torch.cos(phi))
        ny = torch.mul(torch.sin(theta), torch.sin(phi))
        nz = torch.cos(theta)

        nx[torch.isnan(nz)] = 0
        ny[torch.isnan(nz)] = 0
        nz[torch.isnan(nz)] = -1

        sign = torch.ones((1,1,h,w), dtype=torch.float32)
        sign[ny > 0] = -1

        nx = torch.mul(nx, sign).squeeze(dim=0)
        ny = torch.mul(ny, sign).squeeze(dim=0)
        nz = torch.mul(nz, sign).squeeze(dim=0)

        return torch.cat([nx, ny, nz], dim=0)
