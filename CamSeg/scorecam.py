import torch
import torch.optim as optim
import torch.nn.functional as F
import os
from datetime import datetime
import numpy as np

from models import *
from tqdm import tqdm
from skimage import io
from skimage import img_as_ubyte
import cv2
import torch

from evaluation import *
from utils import *
from postprocess import *

class ScoreCAM(object):
    def __init__(self, project_path, classifier_path, exp_name, encoder_model_type):
        encoder_path = os.path.join(project_path, "record", classifier_path, "model", "encoder.pth")
        decoder_path = os.path.join(project_path, "record", classifier_path, "model", "decoder.pth")
        eval_enet_path = os.path.join(project_path, "record", classifier_path, "model", "encoder.pth")
        eval_dnet_path = os.path.join(project_path, "record", classifier_path, "model", "decoder.pth")
        self.encoder_model_type = encoder_model_type
        if encoder_model_type == 'UNet':
            self.encoder = UNetModel(encoder_path).cuda()
            self.eval_enet = UNetModel(eval_enet_path).cuda()
        elif encoder_model_type == 'SSL':
            self.encoder = SSLModel(encoder_path).cuda()
            self.eval_enet = SSLModel(eval_enet_path).cuda()
        elif encoder_model_type == 'DLab':
            self.encoder = DeepLabModel(encoder_path).cuda()
            self.eval_enet = DeepLabModel(eval_enet_path).cuda()
        elif encoder_model_type == 'Res50':
            self.encoder = Res50(encoder_path).cuda()
            self.eval_enet = Res50(eval_enet_path).cuda()

        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.eval_enet.parameters():
            param.requires_grad = False
        self.decoder = CModel(512, decoder_path).cuda()  
        self.eval_dnet = CModel(512, eval_dnet_path).cuda()  
        for param in self.decoder.parameters():
            param.requires_grad = False  
        for param in self.eval_dnet.parameters():
            param.requires_grad = False   
        self.result_path = os.path.join(project_path, "test_results", exp_name)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

    def step(self, data_batch):
        feat_maps, representation = self.encoder(data_batch.cuda())
        logit = self.decoder(representation)
        pred = torch.sigmoid(torch.flatten(logit))
        # print(torch.sigmoid(pred))
        return feat_maps.detach().cpu(), pred.detach().cpu(), logit.detach().cpu(), representation.detach().cpu()

    def eval_step(self, data_batch):
        feat_maps, representation = self.eval_enet(data_batch.cuda())

        logit = self.eval_dnet(representation)
        pred = torch.sigmoid(torch.flatten(logit))
        # print(torch.sigmoid(pred))
        return pred.detach().cpu()

    def get_cam_weight(self):
        weight = None
        for (k, v) in self.decoder.state_dict().items():
            if "weight" in k:
                weight = v.detach().cpu()
        return weight.squeeze(0).numpy()

    def run(self, loader):
        self.encoder.eval()
        self.decoder.eval()
        self.eval_enet.eval()
        self.eval_dnet.eval()
        
        all_dice = []
        all_iou = []
        conf_drop = []
        log_path = os.path.join(self.result_path, "result.log")
        log_file = open(log_path, "w+")
        log_file.writelines(str(datetime.now())+"\n")
        log_file.close()
        test_bar = tqdm(loader)

        with torch.no_grad():
            for img_name, case_batch, seg_batch in test_bar:
                img_name = img_name[0][:-4]
                seg_image = seg_batch[0].permute(1, 2, 0).squeeze(2).numpy()
                feat_maps, confidence, _, rep = self.step(case_batch)
                origin_conf = self.eval_step(case_batch)
                if confidence[0] > 0.5:
                    norm_feat_map = []
                    featured_img = []
                    input_image = case_batch[0].permute(1, 2, 0)
                    img_size = case_batch.shape[2], case_batch.shape[3]
                    _, _, baseline_logit, _ = self.step(torch.zeros_like(case_batch))
                    featured_logit = []
                    norm_feat_map = []

                    for idx, feat_map in enumerate(feat_maps[0]):
                        feat_map = F.interpolate(feat_map.unsqueeze(0).unsqueeze(1), size=img_size, mode='bilinear', align_corners=False)
                        if (feat_map.max() - feat_map.min()) > 0:
                            feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min())
                        feat_map = feat_map.permute(0, 2, 3, 1).squeeze(0)
                        f = feat_map.repeat(1, 1, 3)
                        featured_img = input_image * f
                        featured_tensor = featured_img.unsqueeze(0).permute(0, 3, 1, 2)
                        _, _, logit, _ = self.step(featured_tensor.cuda())
                        featured_logit.append(logit-baseline_logit)
                        norm_feat_map.append(feat_map.squeeze(2).numpy())

                    norm_feat_map = np.asarray(norm_feat_map)
                    featured_logit = torch.cat(featured_logit).squeeze(1)
                    cam_weight = F.softmax(featured_logit, dim=0).numpy()

                    final_map = np.sum(norm_feat_map*cam_weight[:, np.newaxis, np.newaxis], axis=0)
                    final_map = np.maximum(final_map, 0)

                    f = final_map
                    first_seg, final_seg = gen_seg_mask(input_image, f)
                    
                    seg_gt = (seg_image*4).astype(np.uint8)
                    whole_gt = np.where(seg_gt!=0, 1, 0)
                    
                    first_dice = compute_dice(whole_gt, first_seg)
                    first_iou = compute_mIOU(whole_gt, first_seg)

                    dice = compute_dice(whole_gt, final_seg)
                    iou = compute_mIOU(whole_gt, final_seg)

                    tumor_seg = torch.Tensor(final_seg).unsqueeze(2).repeat(1,1,3)
                    input_image = input_image
                    tumor_erase = input_image * (1-tumor_seg)
                    tumor_erase = tumor_erase.unsqueeze(0).permute(0, 3, 1, 2)
                    tumor_erase_conf = self.eval_step(tumor_erase.cuda())
                    conf_drop_rate = np.maximum(0, ((origin_conf[0] - tumor_erase_conf[0])/origin_conf[0]).numpy())
                    print("Img Name:", img_name, ", Tumor Confidence:", origin_conf[0].numpy(), "->", tumor_erase_conf[0].numpy())
                    print("Dice Score:", first_dice, "->", dice)
                    print("IOU Score:", first_iou, "->", iou)
                    print("Conf Drop:", conf_drop_rate)
                    log_file = open(log_path, "a")
                    log_file.writelines("Img Name: {}, Tumor Confidence: {} -> {}\n".format(img_name, origin_conf[0].numpy(), tumor_erase_conf[0].numpy()))
                    log_file.writelines("Dice Score: {:.4f} -> {:.4f}\n".format(first_dice, dice))
                    log_file.writelines("IOU Score:  {:.4f} -> {:.4f}\n".format(first_iou, iou))
                    log_file.writelines("Conf Drop:  {:.4f}\n".format(conf_drop_rate))
                    log_file.close()
                
                    all_dice.append(dice)
                    all_iou.append(iou)
                    conf_drop.append(conf_drop_rate)

        all_dice = np.asarray(all_dice)
        all_iou = np.asarray(all_iou)
        conf_drop = np.asarray(conf_drop)
        print("Average Test Dice Score: {:.4f}".format(np.mean(all_dice)))
        print("Average Test IOU Score:  {:.4f}".format(np.mean(all_iou)))
        print("Average Test Confidence Drop:  {:.4f}%".format(np.mean(conf_drop)*100))
        log_file = open(log_path, "a")
        log_file.writelines("Average Test Dice Score: {:.4f}\n".format(np.mean(all_dice)))
        log_file.writelines("Average Test IOU Score:  {:.4f}\n".format(np.mean(all_iou)))
        log_file.writelines("Average Test Confidence Drop:  {:.4f}%\n".format(np.mean(conf_drop)*100))
        log_file.writelines(str(datetime.now())+"\n")
        log_file.close()