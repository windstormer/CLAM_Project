import torch
import torch.optim as optim
import os
from datetime import datetime
import numpy as np

from models import *
from utils import *
from tqdm import tqdm
import torch
# from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score

class SNet(object):
    def __init__(self, epochs, batch_size, learning_rate, train_loader, val_loader, log_path,
    project_path, record_path, model_name, gpuid):
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log_path = log_path
        self.lr = learning_rate
        # self.encoder = UNetModel()
        self.encoder = UNet(n_channels=1, n_classes=1, bilinear=True)
        
        
        if len(gpuid) > 1:
            self.encoder = torch.nn.DataParallel(self.encoder, device_ids=gpuid)
        self.encoder = self.encoder.cuda()
        # self.optimizer = optim.Adam(self.encoder.parameters(), lr=self.lr, weight_decay=1e-5)
        self.optimizer = optim.SGD(self.encoder.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-6)
        # self.optimizer = optim.SGD(self.encoder.parameters(), lr=self.lr)
        self.model_name = model_name
        self.project_path = project_path
        self.record_path = record_path
        # self.gamma = 2
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=0.000005)
        self.loss = nn.BCEWithLogitsLoss()
        # self.loss = nn.CrossEntropyLoss()
    
    # def optimal_thresh(self, fpr, tpr, thresholds, p=0):
    #     loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    #     idx = np.argmin(loss, axis=0)
    #     return fpr[idx], tpr[idx], thresholds[idx]

    def run(self):
        self.encoder.train()
        # log_file = open(self.log_path, "a")
        # log_file.writelines(str(datetime.now())+"\n")
        # log_file.close()
        train_record = {'dice':[], 'loss':[]}
        val_record = {'dice':[], 'loss':[]}

        best_score = 0.0
        for epoch in range(1, self.epochs + 1):
            train_loss, dice, iou = self.train(epoch)
            train_record['loss'].append(train_loss)
            train_record['dice'].append(dice)
            # if epoch >= 10:
            self.scheduler.step()
            
            log_file = open(self.log_path, "a")
            # log_file.writelines(
            #     "Epoch {:4d}/{:4d} | Train Loss: {}, Train Acc: {}, AUC: {}, Sensitivity: {}, Specificity: {}\n"
            #     .format(epoch, self.epochs, train_loss, train_acc, auc, sensitivity, specificity))
            log_file.writelines(
                "Epoch {:4d}/{:4d} | Train Loss: {}, Train Dice: {}, IOU: {}\n"
                .format(epoch, self.epochs, train_loss, dice, iou))
            log_file.close()

            val_loss, dice, iou = self.val(epoch)
            val_record['loss'].append(val_loss)
            val_record['dice'].append(dice)
            log_file = open(self.log_path, "a")
            log_file.writelines(
                "Epoch {:4d}/{:4d} | Val Loss: {}, Val Dice: {}, IOU: {}\n"
                .format(epoch, self.epochs, val_loss, dice, iou))
            
            cur_score = dice
            if cur_score > best_score:
                best_score = cur_score
                log_file.writelines("Save model at Epoch {:4d}/{:4d} | Val Loss: {}, Val Dice: {}, IOU: {}\n".format(epoch, self.epochs, val_loss, dice, iou))
                encoder_path = os.path.join(self.project_path, self.record_path, self.model_name, "model", "encoder.pth")
                torch.save(self.encoder.state_dict(), encoder_path)
            log_file.close()
        log_file = open(self.log_path, "a")
        log_file.writelines(str(datetime.now())+"\n")
        log_file.close()
        save_chart(self.epochs, train_record['dice'], val_record['dice'], os.path.join(self.project_path, self.record_path, self.model_name, "dice.png"), name='dice')
        save_chart(self.epochs, train_record['loss'], val_record['loss'], os.path.join(self.project_path, self.record_path, self.model_name, "loss.png"), name='loss')

    def train(self, epoch):
        train_bar = tqdm(self.train_loader)
        total_loss, total_num = 0.0, 0
        dice_batch = []
        iou_batch = []
        # out_results = []

        for case_batch, label_batch in train_bar:
            self.optimizer.zero_grad()
            loss, pred_batch, out_batch = self.step(case_batch, label_batch)
            loss.backward()
            self.optimizer.step()
            total_num += case_batch.shape[0]
            total_loss += loss.item() * case_batch.shape[0]
            train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, self.epochs, total_loss / total_num))


            # pred_batch = pred_batch.squeeze(1)
            # label_batch = label_batch.squeeze(1)
            # out_batch = out_batch.squeeze(1)
            # print(torch.sigmoid(pred_batch[0, 70:80, 130:140]))
            # print(label_batch[0, 70:80, 130:140])
            # print(out_batch[0, 70:80, 130:140])
            # if not os.path.exists(os.path.join(self.project_path, self.record_path, self.model_name, "val_result", "epoch_{}".format(epoch))):
            #     os.makedirs(os.path.join(self.project_path, self.record_path, self.model_name, "val_result", "epoch_{}".format(epoch)))
            # fig, axes = plt.subplots(1, 3, figsize=(8, 8))
            # ax = axes.flatten()
            # ax[0].imshow(case_batch[0].permute(1,2,0), cmap="gray")
            # # ax[0].set_axis_off()
            # ax[1].imshow(label_batch[0], cmap="gray")
            # # ax[1].set_axis_off()
            # ax[2].imshow(out_batch[0], cmap="gray")
            # # ax[2].set_axis_off()
            # plt.savefig(os.path.join(self.project_path, self.record_path, self.model_name, "val_result", "epoch_{}".format(epoch), "img_train.png"))
            # plt.close()

            # exit(0)

            dice_score, iou_score = self.evaluate(label_batch.numpy(), out_batch.numpy())
            dice_batch.append(dice_score)
            iou_batch.append(iou_score)
        # print(pred_results.shape)
        # print(train_labels.shape)
        dice_batch = np.asarray(dice_batch)
        iou_batch = np.asarray(iou_batch)
        return total_loss / total_num, dice_batch.mean(), iou_batch.mean()

    def step(self, data_batch, label_batch):
        pred, out = self.encoder(data_batch.cuda())
        # loss = self.loss(pred, label_batch.cuda()) + dice_loss(torch.sigmoid(pred).float(), label_batch.cuda())
        loss = dice_loss(torch.sigmoid(pred).float(), label_batch.cuda())
        # pred = torch.clamp(pred, min=1e-5, max=1. - 1e-5)
        # pt = (target * pred + (1. - target) * (1. - pred))
        # logpt = torch.log(pt)
        # weight = target * alpha + (1. - target) * (1 - alpha)
        # loss = -(weight * (1 - pt) ** self.gamma * logpt).mean()
        return loss, pred.detach().cpu(), out.detach().cpu()
    
    def val(self, epoch):
        self.encoder.eval()

        val_bar = tqdm(self.val_loader)
        total_loss, total_num = 0.0, 0
        val_labels = []
        pred_results = []
        out_results = []
        with torch.no_grad():
            for case_batch, label_batch in val_bar:
                loss, pred_batch, out_batch = self.step(case_batch, label_batch)

                total_num += case_batch.shape[0]
                total_loss += loss.item() * case_batch.shape[0]
                val_bar.set_description('Val Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, self.epochs, total_loss / total_num))
                pred_results.append(pred_batch)
                val_labels.append(label_batch)
                out_results.append(out_batch)
            
            label_batch = label_batch.squeeze(1)
            out_batch = out_batch.squeeze(1)
            # if not os.path.exists(os.path.join(self.project_path, self.record_path, self.model_name, "val_result", "epoch_{}".format(epoch))):
            #     os.makedirs(os.path.join(self.project_path, self.record_path, self.model_name, "val_result", "epoch_{}".format(epoch)))
            # for i in range(case_batch.shape[0]):
            #     fig, axes = plt.subplots(1, 3, figsize=(8, 8))
            #     ax = axes.flatten()
            #     ax[0].imshow(case_batch[i].permute(1,2,0), cmap="gray")
            #     ax[0].set_axis_off()
            #     ax[1].imshow(label_batch[i], cmap="gray")
            #     ax[1].set_axis_off()
            #     ax[2].imshow(out_batch[i], cmap="gray")
            #     ax[2].set_axis_off()
            #     plt.savefig(os.path.join(self.project_path, self.record_path, self.model_name, "val_result", "epoch_{}".format(epoch), "img_{}.png".format(i)))
            #     plt.close()

        pred_results = torch.cat(pred_results, dim=0).numpy()
        val_labels = torch.cat(val_labels, dim=0).numpy()
        out_results = torch.cat(out_results, dim=0).numpy()
        # print(pred_results.shape)
        # print(val_labels.shape)
        dice_score, iou_score = self.evaluate(val_labels, out_results)
        self.encoder.train()
        return total_loss / total_num, dice_score, iou_score

    def test(self, loader, load_model=None):
        self.encoder.eval()
        if load_model == None:
            encoder_path = os.path.join(self.project_path, self.record_path, self.model_name, "model", "encoder.pth")
        else:
            encoder_path = os.path.join(self.project_path, self.record_path, load_model, "model", "encoder.pth")

        state_dict_weights = torch.load(encoder_path)
        state_dict_init = self.encoder.state_dict()
        new_state_dict = OrderedDict()
        for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
            name = k_0
            new_state_dict[name] = v
        self.encoder.load_state_dict(new_state_dict, strict=False)
        test_bar = tqdm(loader)
        total_loss, total_num = 0.0, 0
        test_labels = []
        pred_results = []
        out_results = []
        with torch.no_grad():
            for case_batch, label_batch in test_bar:
                loss, pred_batch, out_batch = self.step(case_batch, label_batch)

                total_num += self.batch_size
                total_loss += loss.item() * self.batch_size
                pred_results.append(pred_batch)
                test_labels.append(label_batch)
                out_results.append(out_batch)
        pred_results = torch.cat(pred_results, dim=0).numpy()
        test_labels = torch.cat(test_labels, dim=0).numpy()
        out_results = torch.cat(out_results, dim=0).numpy()
        # print(pred_results.shape)
        # print(test_labels.shape)
        dice_score, iou_score = self.evaluate(test_labels, out_results)
        return total_loss / total_num, dice_score, iou_score

    def evaluate(self, labels, pred):
        dice_batch = []
        iou_batch = []
        for i in range(labels.shape[0]):
            dice_score = self.compute_dice(labels[i], pred[i])
            dice_batch.append(dice_score)
            iou_score = self.compute_mIOU(labels[i], pred[i])
            iou_batch.append(iou_score)
        dice_batch = np.asarray(dice_batch)
        iou_batch = np.asarray(iou_batch)
        return dice_batch.mean(), iou_batch.mean()

    def compute_dice(self, gt, gen):
        # print(gen.dtype, gt.dtype)
        gt = gt.astype(np.uint8)
        gen = gen.astype(np.uint8)
        # print(gt.shape, gen.shape)
        inse = np.logical_and(gt, gen).sum()
        dice = (2. * inse + 1e-5) / (np.sum(gt) + np.sum(gen) + 1e-5)
        return dice

    def compute_mIOU(self, gt, gen):
        gt = gt.astype(np.uint8)
        gen = gen.astype(np.uint8)
        intersection = np.logical_and(gt, gen)
        # print(intersection)
        union = np.logical_or(gt, gen)
        if np.sum(union) == 0:
            return 0
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score
    
def dice_coeff(input, target, reduce_batch_first = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]

def dice_loss(input, target):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    return 1 - dice_coeff(input, target, reduce_batch_first=True)