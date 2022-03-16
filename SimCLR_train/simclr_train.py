import torch
import torch.optim as optim
import os
from datetime import datetime
from collections import OrderedDict

from models import *
from loss import *
from tqdm import tqdm
from utils import *

class SimCLR(object):
    def __init__(self, epochs, batch_size, learning_rate, train_loader, val_loader, log_path,
    project_path, record_path, model_name, gpuid, unsupervised, model_type):
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log_path = log_path
        
        if model_type == "UNet":
            model = UNetModel()
        elif model_type == "Res18":
            model = Res18()
        elif model_type == "DLab":
            model = DeepLabModel()
        elif model_type == "Res34":
            model = Res34()
        elif model_type == "CNN":
            model = CNNModel()
        # model = SSLModel(None)

        if len(gpuid) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpuid)
        # pretrain_model_path = os.path.join(project_path, record_path, "weights", "embedder.pth")
        self.load_pretrain_model(model, None)

        self.model = model.to('cuda')
        self.lr = learning_rate
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=0, last_epoch=-1)
        self.loss = SupConLoss()
        self.model_name = model_name
        self.project_path = project_path
        self.record_path = record_path
        self.unsupervised = unsupervised

    def load_pretrain_model(self, model, pretrain_model_path=None):
        if pretrain_model_path == None:
            print("Training from scratch.")
        else:
            print("Model restore from", pretrain_model_path)
            state_dict_weights = torch.load(pretrain_model_path)
            state_dict_init = model.state_dict()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
                print(k, k_0)
            model.load_state_dict(new_state_dict, strict=False)

    def run(self):
        log_file = open(self.log_path, "w+")
        log_file.writelines(str(datetime.now())+"\n")
        log_file.close()
        record = {'train_loss':[], 'val_loss':[]}
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train(epoch)
            record['train_loss'].append(train_loss)
            
            if epoch >= 10:
                self.scheduler.step()
            
            log_file = open(self.log_path, "a")
            if epoch > 1:
                log_file.writelines("Epoch {:4d}/{:4d} | Cur lr: {} | Train Loss: {}\n".format(epoch, self.epochs, self.scheduler.get_last_lr()[0], train_loss))
            else:
                log_file.writelines("Epoch {:4d}/{:4d} | Cur lr: {} | Train Loss: {}\n".format(epoch, self.epochs, self.lr, train_loss))
            

            val_loss = self.val(epoch)
            record['val_loss'].append(val_loss)
            log_file = open(self.log_path, "a")
            log_file.writelines("Epoch {:4d}/{:4d} | Val Loss: {}\n".format(epoch, self.epochs, val_loss))
            log_file.close()

            if epoch % 1 == 0:
                parameter_path = os.path.join(self.project_path, self.record_path, self.model_name, "model", "self_model_{}.pth".format(epoch))
                torch.save(self.model.state_dict(), parameter_path)
            log_file = open(self.log_path, "a")
            log_file.writelines(str(datetime.now())+"\n")
            log_file.close()
        save_chart(self.epochs, record['train_loss'], record['val_loss'], os.path.join(self.project_path, self.record_path, self.model_name, "loss.png"), name='loss')

    def train(self, epoch):
        self.model.train()
        train_bar = tqdm(self.train_loader)
        total_loss, total_num = 0.0, 0
        for aug1, aug2, label in train_bar:
            self.optimizer.zero_grad()
            loss = self.step(aug1, aug2, label)
            loss.backward()
            self.optimizer.step()

            total_num += self.batch_size
            total_loss += loss.item() * self.batch_size
            train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, self.epochs, total_loss / total_num))

        return total_loss / total_num

    def step(self, aug1, aug2, label):
        aug1 = aug1.cuda()
        aug2 = aug2.cuda()
        feature_1, out_1 = self.model(aug1)
        feature_2, out_2 = self.model(aug2)

        # bag_feature1 = []
        # bag_feature2 = []
        # for idx in range(self.batch_size):
        #     bag1 = bag_batch1[idx].cuda()
        #     bag2 = bag_batch2[idx].cuda()
        #     # print("bag1.shape", bag1.shape)
        #     # print("bag2.shape", bag2.shape)
        #     feature1 = self.model(bag1/255.0)
        #     feature2 = self.model(bag2/255.0)
        #     # print("feature1.shape", feature1.shape)
        #     # print("feature2.shape", feature2.shape)
        #     bag_feature1.append(feature1)
        #     bag_feature2.append(feature2)
        # bag_feature1 = torch.cat(bag_feature1, dim=0)
        # bag_feature2 = torch.cat(bag_feature2, dim=0)
        # print("bag_feature1.shape", bag_feature1.shape)
        # print("bag_feature2.shape", bag_feature2.shape)
        out_1 = F.normalize(out_1, dim=1)
        out_2 = F.normalize(out_2, dim=1)
        if self.unsupervised:
            return self.loss(out_1, out_2, None)
        else:
            return self.loss(out_1, out_2, label)
            
        # loss = contrastive_loss(bag_feature1, bag_feature2, batch_size, device=bag_feature1.device)
            
    def val(self, epoch):
        self.model.eval()
        val_bar = tqdm(self.val_loader)
        total_loss, total_num = 0.0, 0
        with torch.no_grad():
            for aug1, aug2, label in val_bar:
                loss = self.step(aug1, aug2, label)

                total_num += self.batch_size
                total_loss += loss.item() * self.batch_size
                val_bar.set_description('Val Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, self.epochs, total_loss / total_num))

        self.model.train()
        return total_loss / total_num
