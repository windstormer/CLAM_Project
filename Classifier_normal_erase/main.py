import argparse
import os, glob
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import KFold
from datetime import datetime

from dataset import *
from cnet_run import CNet
from sklearn.model_selection import train_test_split
import csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--project_path",
                        type=str, 
                        default="../../",
                        help="path of project")
    
    parser.add_argument("--input_path",
                        type=str, 
                        default="BraTS_patch",
                        help="path of dataset")

    parser.add_argument("--encoder_model",
                        type=str,
                        default=None,
                        help="pretrained encoder model")

    parser.add_argument("--gpu_id",
                        type=str,
                        default ='',
                        help="gpu id number")

    parser.add_argument("--epochs",
                        "-e",
                        type=int,
                        default=30,
                        help="number of epoch")

    parser.add_argument("--batch_size",
                        "-b",
                        type=int,
                        default=512,
                        help="batch size") 

    parser.add_argument("--learning_rate",
                        "-lr",
                        type=float,
                        default=1e-4,
                        help="learning rate")          

    parser.add_argument("--suffix",
                        '-s',
                        type=str,
                        default=None,
                        help="suffix")

    parser.add_argument("--inference_model",
                        type=str, 
                        default=None,
                        help="model name of test model")
    
    parser.add_argument('-m',
                        '--modality',
                        type=str,
                        default='t1',
                        help='Modality select [flair, t1, t1ce, t2]')

    parser.add_argument("--encoder_mode",
                        type=str, 
                        default="fixed",
                        help="fixed or finetuned")

    parser.add_argument("--encoder_model_type",
                        type=str,
                        default="SSL",
                        help="Type of Model [SSL, UNet, DLab]")          

    # args parse
    args = parser.parse_args()
    gpu_id = args.gpu_id
    if gpu_id == '':
        gpu_id = ",".join([str(g) for g in np.arange(torch.cuda.device_count())])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    gpuid = range(len(gpu_id.split(",")))
    print("Use {} GPU".format(len(gpu_id.split(","))))
    
    

    batch_size = args.batch_size
    epochs = args.epochs
    project_path = args.project_path
    suffix = args.suffix
    learning_rate = args.learning_rate
    modality = args.modality
    encoder_mode = args.encoder_mode
    record_path = "record"

    dataset_path = "/work/vincent18/"
    input_path = os.path.join(dataset_path, args.input_path)

    assert modality in ['flair', 't1', 't1ce', 't2'], 'error modality given'

    if args.encoder_model != None:
        sections = args.encoder_model.split("_")
        pretrain_epoch = int(sections[2][2:])
        encoder_model_type = sections[0]
        encoder_model = args.encoder_model
        encoder_model_path = os.path.join(args.project_path, record_path, args.encoder_model, "model", "self_model_{}.pth".format(pretrain_epoch))
    else:
        encoder_model_path = None
        encoder_model = args.encoder_model_type + "_None"
        encoder_model_type = args.encoder_model_type
        encoder_mode = 'scratch'

    
    model_name = "CNet_normal_erase_{}_ep{}_b{}".format(modality, epochs, batch_size)
    model_name += ".{{{}}}".format(encoder_model)
    if encoder_mode == 'finetuned':
        model_name += ".finetuned"

    if suffix != None:
        model_name = model_name + ".{}".format(suffix)

    print("============== Create Record Directory ===============")
    if not os.path.exists(os.path.join(project_path, record_path, model_name)):
        os.mkdir(os.path.join(project_path, record_path, model_name))

    if not os.path.exists(os.path.join(project_path, record_path, model_name, "model")):
        os.mkdir(os.path.join(project_path, record_path, model_name, "model"))

    full_log_path = os.path.join(project_path, record_path, model_name, "log.log")

    if args.inference_model == None:
        normal_path = glob.glob(os.path.join(input_path, modality, "training", "normal", "*.jpg"))
        tumor_path = glob.glob(os.path.join(input_path, modality, "training", "tumor", "*.jpg"))
        print("Num of Tumor Data:", len(tumor_path))
        print("Num of Normal Data:", len(normal_path))
        data_list = normal_path + tumor_path
        print("Data length", len(data_list))
        label_list = []
        for d in data_list:
            sec = d.split(os.path.sep)
            lab = sec[-1]
            if sec[-2] == "tumor":
                label_list.append(1)
            else:
                label_list.append(0)

        label_list = np.asarray(label_list)
        total_cases = len(data_list)

        log_file = open(full_log_path, "w+")
        log_file.writelines(str(datetime.now())+"\n")
        log_file.close()
        
        print("============== Dataset Setup ===============")
        train_index, val_index = train_test_split(range(total_cases), test_size=0.125)
        train_sampler = SubsetRandomSampler(train_index)
        test_sampler = SubsetRandomSampler(val_index)
        train_dataset = CDataset(data_list, label_list)
        val_dataset = VDataset(data_list, label_list)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=16,
                                pin_memory=True, drop_last=False, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16,
                                pin_memory=True, drop_last=False, sampler=test_sampler)
        print("============== Model Setup ===============")
        classifier = CNet(epochs, batch_size, learning_rate, train_loader, val_loader, full_log_path,
        project_path, record_path, model_name, encoder_model_path, encoder_mode, encoder_model_type, gpuid)
        print("============== Start Training ===============")
        classifier.run()

    print("============== Start Testing ===============")
    normal_path = glob.glob(os.path.join(input_path, modality, "validate", "normal", "*.jpg"))
    tumor_path = glob.glob(os.path.join(input_path, modality, "validate", "tumor", "*.jpg"))
    print("Num of Tumor Data:", len(tumor_path))
    print("Num of Normal Data:", len(normal_path))
    data_list = normal_path + tumor_path
    print("Data length", len(data_list))
    label_list = []
    for d in data_list:
        sec = d.split(os.path.sep)
        lab = sec[-1]
        if sec[-2] == "tumor":
            label_list.append(1)
        else:
            label_list.append(0)

    label_list = np.asarray(label_list)
    total_cases = len(data_list)
    test_dataset = VDataset(data_list, label_list)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16,
                                pin_memory=True, drop_last=False)

    if args.inference_model != None:
        classifier = CNet(epochs, batch_size, learning_rate, None, None, None, project_path, record_path, model_name, encoder_model_path, encoder_mode, encoder_model_type, gpuid)
        test_loss, test_acc, sensitivity, specificity, auc = classifier.test(test_loader, args.inference_model)
        log_file = open(os.path.join(project_path, record_path, args.inference_model, "log.log"), "a")
        log_file.writelines(
            "Final !! Test Loss: {}, Test Acc: {}, AUC: {}, Sensitivity: {}, Specificity: {}\n"
            .format(test_loss, test_acc, auc, sensitivity, specificity))
        log_file.writelines(str(datetime.now())+"\n")
        log_file.close()
    else:
        test_loss, test_acc, sensitivity, specificity, auc = classifier.test(test_loader, args.inference_model)
        log_file = open(full_log_path, "a")
        log_file.writelines(
            "Final !! Test Loss: {}, Test Acc: {}, AUC: {}, Sensitivity: {}, Specificity: {}\n"
            .format(test_loss, test_acc, auc, sensitivity, specificity))
        log_file.writelines(str(datetime.now())+"\n")
        log_file.close()
