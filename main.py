

import os
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image

from modules.model import CoreModel
from modules.utils import device, mse2psnr, \
    grad_loss, safe_path, set_seed
from modules.config import config_parser
from dataloader.dataset import Data
from pytorch_msssim import MS_SSIM
from DISTS_pytorch import DISTS
import torchvision.transforms as transforms
from kornia.filters import canny
from torchvision.utils import save_image

class Trainder(object):
    def __init__(self, args):
        
        self.args = args
        self.dataname = args.dataname
        self.logpath = args.basedir
        self.outpath = safe_path(os.path.join(self.logpath, 'output'))
        self.weightpath = safe_path(os.path.join(self.logpath, 'weight'))
        self.imgpath = safe_path(os.path.join(self.outpath, 'images'))
        self.imgpath = safe_path(os.path.join(self.imgpath, '{}'.format(self.dataname)))
        self.logfile = os.path.join(self.outpath, 'log_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.txt'.format(self.dataname,
            "msssim" + self.args.use_msssim, "msssim_coeff" + str(self.args.lr_msssim), 
            "edges" + self.args.use_edges, "edges_coeff" + str(self.args.lr_edges),
            "canny_sigma" + str(self.args.canny_sigma),
            "canny_low" + str(self.args.canny_low), 
            "canny_high" + str(self.args.canny_high),
            "new_edge_loss" + self.args.new_edge_loss,
            "lr_mae+" + str(self.args.lr_mae),
            "use_custom_edgecombo" + self.args.use_custom_edgecombo))
        self.logfile = open(self.logfile, 'w')
        self.model = CoreModel(args).to(device)
        self.loss_fn = torch.nn.MSELoss()
        self.lr1, self.lr2  = args.lr1, args.lr2
        self.lrexp, self.lr_s = args.lrexp, args.lr_s
        self.set_optimizer(self.lr1, self.lr2)
        #self.imagesgt = torch.tensor(self.model.imagesgt).float().to(device)
        self.imagesgt = self.model.imagesgt
        self.masks = torch.tensor(self.model.masks).float().to(device)
        self.imagesgt_train = self.imagesgt
        self.imgout_path = safe_path(os.path.join(self.imgpath,
                        'v2_{:.3f}_{:.3f}'.format(args.data_r, args.splatting_r), 
                        "msssim" + self.args.use_msssim + \
                            "msssim_coeff" + str(self.args.lr_msssim) + \
                                "edges" + self.args.use_edges + \
                                     "edges_coeff" + str(self.args.lr_edges) + \
                                        "canny_sigma" + str(self.args.canny_sigma) + \
                                            "canny_low" + str(self.args.canny_low) + \
                                                   "canny_high" + str(self.args.canny_high) + \
                                                    "new_edge_loss" + self.args.new_edge_loss + \
                                                        "lr_mae+" + str(self.args.lr_mae) + \
                                                            "use_custom_edgecombo" + self.args.use_custom_edgecombo))

        # create dirs for test, train, edges if needed
        safe_path(os.path.join(self.imgpath,
                        'v2_{:.3f}_{:.3f}'.format(args.data_r, args.splatting_r), 
                        "msssim" + self.args.use_msssim + \
                            "msssim_coeff" + str(self.args.lr_msssim) + \
                                "edges" + self.args.use_edges + \
                                     "edges_coeff" + str(self.args.lr_edges) + \
                                        "canny_sigma" + str(self.args.canny_sigma) + \
                                            "canny_low" + str(self.args.canny_low) + \
                                                   "canny_high" + str(self.args.canny_high) + \
                                                    "new_edge_loss" + self.args.new_edge_loss + \
                                                        "lr_mae+" + str(self.args.lr_mae) + \
                                                            "use_custom_edgecombo" + self.args.use_custom_edgecombo,
                                                    "train"))
        
        safe_path(os.path.join(self.imgpath,
                'v2_{:.3f}_{:.3f}'.format(args.data_r, args.splatting_r), 
                "msssim" + self.args.use_msssim + \
                    "msssim_coeff" + str(self.args.lr_msssim) + \
                        "edges" + self.args.use_edges + \
                                "edges_coeff" + str(self.args.lr_edges) + \
                                "canny_sigma" + str(self.args.canny_sigma) + \
                                    "canny_low" + str(self.args.canny_low) + \
                                            "canny_high" + str(self.args.canny_high) + \
                                            "new_edge_loss" + self.args.new_edge_loss + \
                                                "lr_mae+" + str(self.args.lr_mae) + \
                                                    "use_custom_edgecombo" + self.args.use_custom_edgecombo,
                                            "test"))

        if args.use_edges == "True":
            safe_path(os.path.join(self.imgpath,
                        'v2_{:.3f}_{:.3f}'.format(args.data_r, args.splatting_r), 
                        "msssim" + self.args.use_msssim + \
                            "msssim_coeff" + str(self.args.lr_msssim) + \
                                "edges" + self.args.use_edges + \
                                     "edges_coeff" + str(self.args.lr_edges) + \
                                        "canny_sigma" + str(self.args.canny_sigma) + \
                                            "canny_low" + str(self.args.canny_low) + \
                                                   "canny_high" + str(self.args.canny_high) + \
                                                    "new_edge_loss" + self.args.new_edge_loss + \
                                                        "lr_mae+" + str(self.args.lr_mae) + \
                                                            "use_custom_edgecombo" + self.args.use_custom_edgecombo,
                                        "edges"))
        self.training_time = 0
        print(self.imgout_path)

        self.ms_ssim = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1.0, size_average=True)
        #self.normalize = transforms.Normalize(mean=[0.406, 0.456, 0.485],
        #                         std=[0.225, 0.224, 0.229])
        self.D = DISTS().to(device)
        self.coeff_edgeloss = args.lr_edges
        self.coeff_msssimloss = args.lr_msssim
        self.train_cycle = 0

        self.l1_sum = torch.nn.L1Loss(reduction="sum")
        self.l1 = torch.nn.L1Loss(reduction="none")

        self.coeff_alpha = args.lr_mae

    def canny_custom(self, img):
        return canny(img, low_threshold=self.args.canny_low, high_threshold=self.args.canny_high, sigma=(self.args.canny_sigma, self.args.canny_sigma))[1] # 1, 1, H, W
    
    def set_onlybase(self):
        self.model.onlybase = True
        self.set_optimizer(3e-3,self.lr2)

    def remove_onlybase(self):
        self.model.onlybase = False
        self.set_optimizer(self.lr1,self.lr2)

    def set_optimizer(self, lr1=3e-3, lr2=8e-4):
        sh_list = [name for name, params in self.model.named_parameters() if 'sh' in name]
        sh_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in sh_list,
                                self.model.named_parameters()))))
        other_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in sh_list,
                                self.model.named_parameters()))))
        optimizer = torch.optim.Adam([
            {'params': sh_params, 'lr': lr1},
            {'params': other_params, 'lr': lr2}])
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.lrexp, -1)
        self.optimizer, self.lr_scheduler = optimizer, lr_scheduler
        return None

    def train(self,epoch_n=30):
        self.logfile.write('-----------Stage Segmentation Line-----------')
        self.logfile.flush()
        max_psnr = 0.
        start_time = time.time()

        for epoch in range(epoch_n):
            loss_all, psnr_all, posedgeloss_all, negedgeloss_all = [], [], [], []
            
            if args.dataset_type == "llff":
                ids = np.random.permutation(dataset.i_split[0])
            else:
                ids = np.random.permutation(100)
            
            #for id in tqdm(ids):
            for id in ids:
                images = self.model(id)
                if self.args.new_edge_loss == "True":
                    pixel_loss = self.l1_sum(images[0], self.imagesgt_train[id])
                    tot_loss = self.coeff_alpha * pixel_loss + self.lr_s * grad_loss(images[0], self.imagesgt_train[id])
                else:
                    pixel_loss = self.loss_fn(images[0], self.imagesgt_train[id])
                    tot_loss = pixel_loss + self.lr_s * grad_loss(images[0], self.imagesgt_train[id])
                #####################################
                # ADDED
                if self.args.use_msssim == "True":
                    #print("We are using MS-SSIM loss term.")
                    ms_ssim_loss = 1 - self.ms_ssim(images[0].permute(2,0,1).unsqueeze(dim=0), self.imagesgt_train[id].permute(2,0,1).unsqueeze(dim=0))
                    tot_loss = tot_loss + ms_ssim_loss

                if self.args.use_dists == "True":
                    # probably something to do with how the images are normalized...check this
                    #normalized_predicted = self.normalize(images[0].permute(2,0,1).unsqueeze(dim=0))
                    #normalized_gt = self.normalize(self.imagesgt_train[id].permute(2,0,1).unsqueeze(dim=0))
                    dists_loss = self.D(images[0].permute(2,0,1).unsqueeze(dim=0), self.imagesgt_train[id].permute(2,0,1).unsqueeze(dim=0), require_grad=True, batch_average=True) 
                    tot_loss = tot_loss + 0.01 * dists_loss

                if self.args.use_edges == "True":
                    canny_predicted = self.canny_custom(images[0].permute(2,0,1).unsqueeze(dim=0)) # 1, 1, H, W
                    canny_gt = self.canny_custom(self.imagesgt_train[id].permute(2,0,1).unsqueeze(dim=0)) # 1, 1, H, W
                    
                    if self.args.new_edge_loss == "True":
                        edge_loss_tot = torch.mean(canny_gt * self.l1(images[0].permute(2,0,1).unsqueeze(dim=0), self.imagesgt_train[id].permute(2,0,1).unsqueeze(dim=0)))
                        if self.args.use_custom_edgecombo == "True":
                            tot_loss = tot_loss + (20. * self.coeff_alpha) * edge_loss_tot # CUSTOM COMBO OF WEIGHT LOSS - ADJUST COEFF AS NECESSARY
                        else:
                            tot_loss = tot_loss + (1. - self.coeff_alpha) * edge_loss_tot

                            ###################
                            """canny_predicted_filtered_pos = canny_predicted[canny_predicted > 0]
                            canny_gt_filtered_pos = canny_gt[canny_predicted > 0]

                            canny_predicted_filtered_neg = canny_predicted[canny_predicted <= 0]
                            canny_gt_filtered_neg = canny_gt[canny_predicted <= 0]

                            edge_loss_pos = torch.sum(canny_predicted_filtered_pos - canny_gt_filtered_pos)
                            edge_loss_neg = torch.sum(canny_gt_filtered_neg - canny_predicted_filtered_neg)
                            edge_loss_tot = edge_loss_pos + edge_loss_neg
                            tot_loss = tot_loss + self.coeff_edgeloss * edge_loss_tot"""
                            ###################
                    else:
                        canny_predicted_filtered_pos = canny_predicted[canny_predicted > 0]
                        canny_gt_filtered_pos = canny_gt[canny_predicted > 0]

                        canny_predicted_filtered_neg = canny_predicted[canny_predicted <= 0]
                        canny_gt_filtered_neg = canny_gt[canny_predicted <= 0]

                        edge_loss_pos = torch.sum(canny_predicted_filtered_pos - canny_gt_filtered_pos)
                        edge_loss_neg = torch.sum(canny_gt_filtered_neg - canny_predicted_filtered_neg)
                        edge_loss_tot = edge_loss_pos + edge_loss_neg
                        tot_loss = tot_loss + self.coeff_edgeloss * edge_loss_tot

                    # save canny edge comparisions
                    if epoch % 9 == 0:
                        canny_predicted = canny_predicted.squeeze(dim=0)
                        canny_gt = canny_gt.squeeze(dim=0)
                        canny_edges = torch.concatenate((canny_predicted,canny_gt),1)
                        #canny_edges = Image.fromarray(canny_edges.astype(np.uint8))
                        #canny_edges.save(os.path.join(self.imgout_path, "train",
                        #    'img_{}_{}_{:.2f}_edges.png'.format(self.dataname, id, mse2psnr(mse_loss).item())))
                        save_image(canny_edges, os.path.join(self.imgout_path, "edges",
                            'img_{}_{}_{}_{}_edges.png'.format(id, self.train_cycle, epoch, edge_loss_tot)))
                
                # if visual
                if epoch % 9 == 0:
                    pred = images[0, ..., :3].detach().cpu().data.numpy()
                    gt = self.imagesgt[id].detach().cpu().data.numpy()
                    # set background as white for visualization
                    mask = self.masks[id].cpu().data.numpy()
                    pred = pred*mask+1-mask
                    gt = gt*mask+1-mask

                    img_gt = np.concatenate((pred,gt),1)
                    img_gt = Image.fromarray((img_gt*255).astype(np.uint8))
                    img_gt.save(os.path.join(self.imgout_path, "train",
                            'img_{}_{}_{}_{:.2f}.png'.format(id, self.train_cycle, epoch, mse2psnr(pixel_loss).item())))
                    
                #####################################
                self.optimizer.zero_grad()
                tot_loss.backward()
                self.optimizer.step()
                if self.args.use_edges == "True" and self.args.new_edge_loss == "False":
                    posedgeloss_all.append(edge_loss_pos)
                    negedgeloss_all.append(edge_loss_neg)
                loss_all.append(tot_loss)
                psnr_all.append(mse2psnr(pixel_loss))
            self.lr_scheduler.step()

            if self.args.use_edges == "True" and self.args.new_edge_loss == "False":
                posedgeloss_e = torch.stack(posedgeloss_all).mean().item()
                negedgeloss_e = torch.stack(negedgeloss_all).mean().item()
            loss_e = torch.stack(loss_all).mean().item()
            psnr_e = torch.stack(psnr_all).mean().item()
            if self.args.use_edges == "True" and self.args.new_edge_loss == "False":
                info = '-----train-----  epoch:{} posedge_loss:{:.3f} negedge_loss:{:.3f} psnr:{:.3f}'.format(epoch, posedgeloss_e, negedgeloss_e, psnr_e)
            elif self.args.use_edges == "True" and self.args.new_edge_loss == "True":
                info = '-----train-----  epoch:{} edge_loss_tot:{:.3f} psnr:{:.3f}'.format(epoch, edge_loss_tot, psnr_e)
            else:
                info = '-----train-----  epoch:{} loss:{:.3f} psnr:{:.3f}'.format(epoch, loss_e, psnr_e)

            print(info)
            self.logfile.write(info + '\n')
            self.logfile.flush()
            psnr_val = self.test(115, 138, False)
            if psnr_val > max_psnr:
                max_psnr = psnr_val
        self.training_time += time.time()-start_time
        self.train_cycle += 1
        torch.save(self.model.state_dict(), os.path.join(
                 self.weightpath,'model_{}.pth'.format(self.dataname)))

    def test(self, start=100, end=115, visual=False):
        plt.cla()
        plt.clf()

        if args.dataset_type == "llff":
            ids = dataset.i_split[2]
        else:
            ids = range(start, end)

        with torch.no_grad():
            loss_all, psnr_all, edgeloss_all = [], [], []
            for id in ids:
                images = self.model(id)

                if self.args.new_edge_loss == "True":
                    pixel_loss = self.l1_sum(images[0], self.imagesgt[id])
                    tot_loss = self.coeff_alpha * pixel_loss
               
                mse_loss = self.loss_fn(images[0], self.imagesgt[id])
                #####################################
                # ADDED
                if self.args.use_msssim == "True":
                    #print("We are using MS-SSIM loss term.")
                    ms_ssim_loss = 1 - self.ms_ssim(images[0].permute(2,0,1).unsqueeze(dim=0), self.imagesgt[id].permute(2,0,1).unsqueeze(dim=0))
                    tot_loss = tot_loss + ms_ssim_loss

                if self.args.use_dists == "True":
                    # probably something to do with how the images are normalized...check this
                    #normalized_predicted = self.normalize(images[0].permute(2,0,1).unsqueeze(dim=0))
                    #normalized_gt = self.normalize(self.imagesgt[id].permute(2,0,1).unsqueeze(dim=0))
                    dists_loss = self.D(images[0].permute(2,0,1).unsqueeze(dim=0), self.imagesgt[id].permute(2,0,1).unsqueeze(dim=0), require_grad=True, batch_average=True) 
                    tot_loss = tot_loss + 0.01 * dists_loss

                if self.args.use_edges == "True":
                    canny_predicted = self.canny_custom(images[0].permute(2,0,1).unsqueeze(dim=0))
                    canny_gt = self.canny_custom(self.imagesgt[id].permute(2,0,1).unsqueeze(dim=0))

                    if self.args.new_edge_loss == "True":
                        edge_loss_tot = torch.mean(canny_gt * self.l1(images[0].permute(2,0,1).unsqueeze(dim=0), self.imagesgt[id].permute(2,0,1).unsqueeze(dim=0)))
                        if self.args.use_custom_edgecombo == "True":
                            tot_loss = tot_loss + (20. * self.coeff_alpha) * edge_loss_tot # CUSTOM COMBO OF WEIGHT LOSS - ADJUST COEFF AS NECESSARY
                        else:
                            tot_loss = tot_loss + (1. - self.coeff_alpha) * edge_loss_tot
                    else:
                        canny_predicted_filtered_pos = canny_predicted[canny_predicted > 0]
                        canny_gt_filtered_pos = canny_gt[canny_predicted > 0]

                        canny_predicted_filtered_neg = canny_predicted[canny_predicted <= 0]
                        canny_gt_filtered_neg = canny_gt[canny_predicted <= 0]

                        edge_loss_pos = torch.sum(canny_predicted_filtered_pos - canny_gt_filtered_pos)
                        edge_loss_neg = torch.sum(canny_gt_filtered_neg - canny_predicted_filtered_neg)
                        edge_loss_tot = edge_loss_pos + edge_loss_neg
                        tot_loss = tot_loss + self.coeff_edgeloss * edge_loss_tot

                #####################################
                if self.args.use_edges == "True":                
                    edgeloss_all.append(edge_loss_tot)
                loss_all.append(tot_loss)
                psnr_all.append(mse2psnr(mse_loss))
                if visual:
                    pred = images[0, ..., :3].detach().cpu().data.numpy()
                    gt = self.imagesgt[id].detach().cpu().data.numpy()
                    # set background as white for visualization
                    mask = self.masks[id].cpu().data.numpy()
                    pred = pred*mask+1-mask
                    gt = gt*mask+1-mask
                    #####################################
                    # ADDED
                    # save canny edge comparisions
                    if self.args.use_edges == "True":
                        canny_predicted = canny_predicted.squeeze(dim=0)
                        canny_gt = canny_gt.squeeze(dim=0)
                        canny_edges = torch.concatenate((canny_predicted,canny_gt),1)
                        save_image(canny_edges, os.path.join(self.imgout_path, "test",
                            'img_{}_{}_{:.2f}_edges.png'.format(self.dataname, id, mse2psnr(mse_loss).item())))
                    #####################################
                    img_gt = np.concatenate((pred,gt),1)
                    img_gt = Image.fromarray((img_gt*255).astype(np.uint8))
                    img_gt.save(os.path.join(self.imgout_path, "test",
                            'img_{}_{}_{:.2f}.png'.format(self.dataname, id, mse2psnr(mse_loss).item())))
            if self.args.use_edges == "True":
                edgeloss_e = torch.stack(edgeloss_all).mean().item()
            loss_e = torch.stack(loss_all).mean().item()
            psnr_e = torch.stack(psnr_all).mean().item()
            if self.args.use_edges == "True":
                info = '-----eval-----  edge_loss:{:.3f} psnr:{:.3f}'.format(edgeloss_e, psnr_e)
            else:
                info = '-----eval-----  loss:{:.3f} psnr:{:.3f}'.format(loss_e, psnr_e)
            print(info)
            self.logfile.write(info + '\n')
            self.logfile.flush()
            return psnr_e

    def get_fps_modelsize(self):
        start_time = time.time()
        if self.args.dataset_type == "blender":
            ids = range(0, 138)
        elif self.args.dataset_type == "llff":
            ids = range(0, 20)

        for id in ids:
            images = self.model(id)
        end_time = time.time()
        fps = 138 / (end_time - start_time)
        model_path = os.path.join(
                 self.weightpath,'model_{}.pth'.format(self.dataname))
        model_size = os.path.getsize(model_path)
        model_size = model_size / float(1024 * 1024)
        model_size = round(model_size, 2)
        return fps,model_size


def solve(args):
    trainer = Trainder(args)
    trainer.set_onlybase()
    trainer.train(epoch_n=20)
    trainer.remove_onlybase()
    trainer.train()

    # course to fine
    for i in range(args.refine_n):
        if args.remove_outlier == "True":
            trainer.model.remove_out()
        trainer.model.repeat_pts()
        trainer.set_optimizer(args.lr1, args.lr2)
        trainer.train()
    trainer.logfile.write('Total Training Time: '
                  '{:.2f}s\n'.format(trainer.training_time))
    trainer.logfile.flush()
    psnr_e = trainer.test(115, 138, True)
    fps,model_size = trainer.get_fps_modelsize()
    print('Training time: {:.2f} s'.format(trainer.training_time))
    print('Rendering quality: {:.2f} dB'.format(psnr_e))
    print('Rendering speed: {:.2f} fps'.format(fps))
    print('Model size: {:.2f} MB'.format(model_size))

    trainer.logfile.write('Rendering quality: {:.2f} dB'.format(psnr_e))
    trainer.logfile.flush()
    trainer.logfile.write('Rendering speed: {:.2f} fps'.format(fps))
    trainer.logfile.flush()
    trainer.logfile.write('Model size: {:.2f} MB'.format(model_size))
    trainer.logfile.flush()


if __name__ == '__main__':
    print("Using ", device)
    set_seed(0)
    parser = config_parser()
    args = parser.parse_args()

    dataset = Data(args, device)
    args.memitem = dataset.genpc()
    solve(args)
