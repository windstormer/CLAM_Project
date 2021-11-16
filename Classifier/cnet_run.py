import torch
import torch.optim as optim
import os
from datetime import datetime
import numpy as np

from models import *
from utils import *
from tqdm import tqdm
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score

class CNet(object):
    def __init__(self, epochs, batch_size, learning_rate, train_loader, val_loader, log_path,
    project_path, record_path, model_name, encoder_model_path, encoder_mode, encoder_model_type, gpuid):
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log_path = log_path
        self.lr = learning_rate
        self.decoder = CModel(512).cuda()
        self.encoder_mode = encoder_mode
        if self.encoder_mode != "fixed":
            if encoder_model_type == 'UNet':
                self.encoder = UNetModel(encoder_model_path)
            elif encoder_model_type == 'SSL':
                self.encoder = SSLModel(encoder_model_path)
            elif encoder_model_type == 'DLab':
                self.encoder = DeepLabModel(encoder_model_path)
            elif encoder_model_type == 'Res50':
                self.encoder = Res50(encoder_model_path)
            self.optimizer = optim.Adam(list(self.encoder.parameters())+list(self.decoder.parameters()), lr=self.lr, weight_decay=1e-5)
        else:
            if encoder_model_type == 'UNet':
                self.encoder = UNetModel(encoder_model_path)
            elif encoder_model_type == 'SSL':
                self.encoder = SSLModel(encoder_model_path)
            elif encoder_model_type == 'DLab':
                self.encoder = DeepLabModel(encoder_model_path)
            elif encoder_model_type == 'Res50':
                self.encoder = Res50(encoder_model_path)
            for param in self.encoder.parameters():
                param.requires_grad = False  
            self.optimizer = optim.Adam(self.decoder.parameters(), lr=self.lr, weight_decay=1e-5)
        
        if len(gpuid) > 1:
            self.encoder = torch.nn.DataParallel(self.encoder, device_ids=gpuid)
        self.encoder = self.encoder.cuda()
        self.model_name = model_name
        self.project_path = project_path
        self.record_path = record_path
        # self.gamma = 2
        
        # self.optimizer = optim.Adam(self.decoder.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=0.000005)
        self.loss = nn.BCEWithLogitsLoss()
        # self.loss = nn.CrossEntropyLoss()
    
    def optimal_thresh(self, fpr, tpr, thresholds, p=0):
        loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
        idx = np.argmin(loss, axis=0)
        return fpr[idx], tpr[idx], thresholds[idx]

    def run(self):
        if self.encoder_mode != "fixed":
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.train()
        # log_file = open(self.log_path, "a")
        # log_file.writelines(str(datetime.now())+"\n")
        # log_file.close()
        train_record = {'auc':[], 'loss':[]}
        val_record = {'auc':[], 'loss':[]}

        best_score = 0.0
        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc, sensitivity, specificity, auc = self.train(epoch)
            train_record['loss'].append(train_loss)
            train_record['auc'].append(auc)
            # if epoch >= 10:
            self.scheduler.step()
            
            log_file = open(self.log_path, "a")
            # log_file.writelines(
            #     "Epoch {:4d}/{:4d} | Train Loss: {}, Train Acc: {}, AUC: {}, Sensitivity: {}, Specificity: {}\n"
            #     .format(epoch, self.epochs, train_loss, train_acc, auc, sensitivity, specificity))
            log_file.writelines(
                "Epoch {:4d}/{:4d} | Train Loss: {}, Train Acc: {}, AUC: {}\n"
                .format(epoch, self.epochs, train_loss, train_acc, auc))
            log_file.close()

            val_loss, val_acc, sensitivity, specificity, val_auc = self.val(epoch)
            val_record['loss'].append(val_loss)
            val_record['auc'].append(val_auc)
            log_file = open(self.log_path, "a")
            log_file.writelines(
                "Epoch {:4d}/{:4d} | Val Loss: {}, Val Acc: {}, AUC: {}, Sensitivity: {}, Specificity: {}\n"
                .format(epoch, self.epochs, val_loss, val_acc, val_auc, sensitivity, specificity))
            
            cur_score = val_auc
            if cur_score > best_score:
                best_score = cur_score
                log_file.writelines("Save model at Epoch {:4d}/{:4d} | Val Loss: {}, Val Acc: {}, AUC: {}\n".format(epoch, self.epochs, val_loss, val_acc, val_auc))
                encoder_path = os.path.join(self.project_path, self.record_path, self.model_name, "model", "encoder.pth")
                decoder_path = os.path.join(self.project_path, self.record_path, self.model_name, "model", "decoder.pth")
                torch.save(self.encoder.state_dict(), encoder_path)
                torch.save(self.decoder.state_dict(), decoder_path)
            log_file.close()
        log_file = open(self.log_path, "a")
        log_file.writelines(str(datetime.now())+"\n")
        log_file.close()
        save_chart(self.epochs, train_record['auc'], val_record['auc'], os.path.join(self.project_path, self.record_path, self.model_name, "auc.png"), name='auc')
        save_chart(self.epochs, train_record['loss'], val_record['loss'], os.path.join(self.project_path, self.record_path, self.model_name, "loss.png"), name='loss')

    def train(self, epoch):
        train_bar = tqdm(self.train_loader)
        total_loss, total_num = 0.0, 0
        train_labels = []
        pred_results = []
        # out_results = []

        for case_batch, label_batch in train_bar:
            self.optimizer.zero_grad()
            loss, pred_batch = self.step(case_batch, label_batch)
            loss.backward()
            self.optimizer.step()
            total_num += self.batch_size
            total_loss += loss.item() * self.batch_size
            train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, self.epochs, total_loss / total_num))
            pred_results.append(pred_batch)
            train_labels.append(label_batch)
        pred_results = torch.cat(pred_results, dim=0).numpy()
        train_labels = torch.cat(train_labels, dim=0).numpy()
        # print(pred_results.shape)
        # print(train_labels.shape)
        acc, sensitivity, specificity, auc_score = self.evaluate(train_labels, pred_results)
        return total_loss / total_num, acc, sensitivity, specificity, auc_score

    def step(self, data_batch, label_batch):
        representation = self.encoder(data_batch.cuda())
        pred = self.decoder(representation)
        loss = self.loss(pred, label_batch.cuda())
        pred = pred.detach().cpu()

        # pred = torch.clamp(pred, min=1e-5, max=1. - 1e-5)
        # pt = (target * pred + (1. - target) * (1. - pred))
        # logpt = torch.log(pt)
        # weight = target * alpha + (1. - target) * (1 - alpha)
        # loss = -(weight * (1 - pt) ** self.gamma * logpt).mean()
        return loss, pred
    
    def val(self, epoch):
        self.encoder.eval()
        self.decoder.eval()

        val_bar = tqdm(self.val_loader)
        total_loss, total_num = 0.0, 0
        val_labels = []
        pred_results = []
        out_results = []
        with torch.no_grad():
            for case_batch, label_batch in val_bar:
                loss, pred_batch = self.step(case_batch, label_batch)

                total_num += self.batch_size
                total_loss += loss.item() * self.batch_size
                val_bar.set_description('Val Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, self.epochs, total_loss / total_num))
                pred_results.append(pred_batch)
                val_labels.append(label_batch)
        pred_results = torch.cat(pred_results, dim=0).numpy()
        val_labels = torch.cat(val_labels, dim=0).numpy()
        # print(pred_results.shape)
        # print(val_labels.shape)
        acc, sensitivity, specificity, auc_score = self.evaluate(val_labels, pred_results)
        if self.encoder_mode != "fixed":
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.train()
        return total_loss / total_num, acc, sensitivity, specificity, auc_score

    def test(self, loader, load_model=None):
        self.encoder.eval()
        self.decoder.eval()
        if load_model == None:
            decoder_path = os.path.join(self.project_path, self.record_path, self.model_name, "model", "decoder.pth")
        else:
            decoder_path = os.path.join(self.project_path, self.record_path, load_model, "model", "decoder.pth")
        # decoder_path = os.path.join(self.project_path, self.record_path, self.model_name, "model", "decoder.pth")
        # decoder_path = os.path.join(self.project_path, self.record_path, "weights", "aggregator.pth")
        state_dict_weights = torch.load(decoder_path)
        state_dict_init = self.decoder.state_dict()
        new_state_dict = OrderedDict()
        for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
            name = k_0
            new_state_dict[name] = v
        self.decoder.load_state_dict(new_state_dict, strict=False)
        test_bar = tqdm(loader)
        total_loss, total_num = 0.0, 0
        test_labels = []
        pred_results = []
        with torch.no_grad():
            for case_batch, label_batch in test_bar:
                loss, pred_batch = self.step(case_batch, label_batch)

                total_num += self.batch_size
                total_loss += loss.item() * self.batch_size
                pred_results.append(pred_batch)
                test_labels.append(label_batch)
        pred_results = torch.cat(pred_results, dim=0).numpy()
        test_labels = torch.cat(test_labels, dim=0).numpy()
        # print(pred_results.shape)
        # print(test_labels.shape)
        acc, sensitivity, specificity, auc_score = self.evaluate(test_labels, pred_results)
        return total_loss / total_num, acc, sensitivity, specificity, auc_score

    def evaluate(self, labels, pred):
        fpr, tpr, threshold = roc_curve(labels, pred, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = self.optimal_thresh(fpr, tpr, threshold)
        out_results = [pred > threshold_optimal for pred in pred]
        # for i in range(len(labels)):
        #     print(labels[i], pred[i])
        # print(labels, pred)
        auc_score = roc_auc_score(labels, pred)

        tn, fp, fn, tp = confusion_matrix(labels, out_results, labels=[0,1]).ravel()
        acc = (tp+tn) / (tn+fp+fn+tp)
        specificity = tn / (tn+fp)
        sensitivity = tp / (tp+fn)
        return acc, sensitivity, specificity, auc_score