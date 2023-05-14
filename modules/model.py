

import torch
import torch.nn.functional as F
import numpy as np
from pytorch3d.ops import knn_points
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor
)

from kornia.filters import canny

from modules.sh import eval_sh
from modules.utils import device, load_mem_data, \
    get_rays, remove_outlier, calc_splatting_r, calc_data_r


class CoreModel(torch.nn.Module):
    def __init__(self, args):
        super(CoreModel, self).__init__()
        self.raster_n = args.raster_n
        self.img_h = args.img_h
        self.img_w = args.img_w
        self.dataname = args.dataname
        pointcloud, imagesgt, K, R, T, poses,masks = load_mem_data(args.memitem)

        self.R, self.T, self.K = R, T, K
        #print("self.R: ", self.R.shape)
        #print("self.T: ", self.T.shape)
        #print("self.K: ", self.K.shape)

        #print("Does k work? ", self.K[0][0] / self.K[0][2])

        self.poses = torch.tensor(poses).to(device)
        #self.imagesgt = imagesgt
        self.imagesgt = torch.tensor(imagesgt).float().to(device)
        print("imagegt shape: ", self.imagesgt.shape) # (400, 400, 400, 3) # images, image dimensions, RGB
        
        # in adaptive setting, splatting_r = f_net(images_gt[0-99]))
        if args.adaptive_splattingr == "True":
            edges = canny(self.imagesgt[0].permute(2,0,1).unsqueeze(dim=0))[1]
            self.splatting_r = calc_splatting_r(edges).item()
        else:
            self.splatting_r = args.splatting_r
        
        if args.adaptive_datar == "True":
            self.data_r = calc_data_r(self.splatting_r)
        else:
            self.data_r = args.data_r
        
        print("Using splatting_r: ", self.splatting_r)
        print("Using data_r: ", self.data_r)

        self.masks = masks

        # get indices of N random points among pointcloud
        N = int(pointcloud.shape[0] * self.data_r)
        ids = np.random.permutation(pointcloud.shape[0])[:N]
        pointcloud = pointcloud[ids][:, :6]
        print('Initialized point number:{}'.format(pointcloud.shape[0]))

        self.vertsparam = torch.nn.Parameter(torch.Tensor(pointcloud[:, :3]))
        self.sh_n, sh_param = 2, [torch.Tensor(pointcloud[:, 3:])]
        for i in range((self.sh_n + 1) ** 2):
            sh_param.append(torch.rand((pointcloud.shape[0], 3)))
        sh_param = torch.cat(sh_param, -1)
        self.sh_param = torch.nn.Parameter(sh_param)
        self.viewdir = []
        for i in range(self.poses.shape[0]):
            rays_o, rays_d = get_rays(self.img_h, self.img_w, torch.tensor(K).to(device), self.poses[i])
            rays_d = torch.nn.functional.normalize(rays_d, dim=2)
            self.viewdir.append(rays_d)
        
        if args.dataset_type == "blender":
            self.raster_settings = PointsRasterizationSettings(
                bin_size=23,
                image_size=(self.img_h, self.img_w),
                radius=self.splatting_r,
                points_per_pixel=self.raster_n,
            )
        elif args.dataset_type == "llff":
            self.raster_settings = PointsRasterizationSettings(
                bin_size=None,
                image_size=(self.img_h, self.img_w),
                radius=self.splatting_r,
                points_per_pixel=self.raster_n,
            )
        self.onlybase = False

    def repeat_pts(self):
        self.vertsparam.data = self.vertsparam.data.repeat(2,1)
        self.sh_param.data = self.sh_param.data.repeat(2, 1)
        if self.vertsparam.grad is not None:
            self.vertsparam.grad = self.vertsparam.grad.repeat(2,1)
        if self.sh_param.grad is not None:
            self.sh_param.grad = self.sh_param.grad.repeat(2, 1)

    def remove_out(self):
        pts_all = self.vertsparam.data
        pts_in = remove_outlier(pts_all.cpu().data.numpy())
        pts_in = torch.tensor(pts_in).to(device).float()
        idx = knn_points(pts_in[None,...], pts_all[None,...], None, None, 1).idx[0,:,0]
        self.vertsparam.data = self.vertsparam.data[idx].detach()
        self.sh_param.data = self.sh_param.data[idx].detach()
        if self.vertsparam.grad is not None:
            self.vertsparam.grad = self.vertsparam.grad[idx].detach()
        if self.sh_param.grad is not None:
            self.sh_param.grad = self.sh_param.grad[idx].detach()

    def forward(self, id):
        cameras = PerspectiveCameras(focal_length=self.K[0][0] / self.K[0][2],
                                     device=device, R=-self.R[id:id + 1], T=-self.T[id:id + 1])
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor()
        )
        point_cloud = Pointclouds(points=[self.vertsparam], features=[self.sh_param])
        # print("point_cloud._N: ", point_cloud._N)
        # print("point_cloud._P: ", point_cloud._P)
        feat = renderer(point_cloud).flip(1)
        base, shfeat = feat[..., :3], feat[..., 3:]
        shfeat = torch.stack(shfeat.split(3, 3), -1)
        if self.onlybase:
            image = base
        else:
            image = base + eval_sh(self.sh_n, shfeat, self.viewdir[id])
        return image.clamp(min=0, max=1)
