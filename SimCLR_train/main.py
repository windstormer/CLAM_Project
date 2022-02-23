import argparse
import os
import torch
from torch.utils.data import DataLoader

from dataset import *
from simclr_train import SimCLR
import glob
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler

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

    parser.add_argument("--gpu_id",
                        type=str,
                        default ='',
                        help="gpu id number")

    parser.add_argument("--patch_size",
                        "-p",
                        type=int,
                        default=240,
                        help="image size")

    parser.add_argument("--epochs",
                        "-e",
                        type=int,
                        default=200,
                        help="number of epoch")

    parser.add_argument("--batch_size",
                        "-b",
                        type=int,
                        default=128,
                        help="batch size") 

    parser.add_argument("--learning_rate",
                        "-lr",
                        type=float,
                        default=1e-3,
                        help="learning rate")          

    parser.add_argument("--suffix",
                        '-s',
                        type=str,
                        default=None,
                        help="suffix")

    parser.add_argument('-m',
                        '--modality',
                        type=str,
                        default='t1',
                        help='Modality select [flair, t1, t1ce, t2]')
    
    parser.add_argument('--unsupervised',
                        action='store_true',
                        help='Use unsupervised loss or not')

    parser.add_argument("--model_type",
                        type=str,
                        default="Res18",
                        help="Type of Model [Res18, UNet, DLab, Res50, CNN]")

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
    patch_size = args.patch_size
    suffix = args.suffix
    learning_rate = args.learning_rate
    record_path = "record/SimCLR"
    modality = args.modality
    unsupervised = args.unsupervised
    model_type = args.model_type

    dataset_path = "/work/vincent18/"
    input_path = os.path.join(dataset_path, args.input_path)

    assert modality in ['flair', 't1', 't1ce', 't2'], 'error modality given'
    assert model_type in ["Res18", "UNet", "DLab", "Res50", "CNN"], 'error model type given'

    model_name = "{}_{}_ep{}_b{}".format(model_type, modality, epochs, batch_size)
    if unsupervised:
        model_name += ".unsupervised"

    if suffix != None:
        model_name = model_name + ".{}".format(suffix)
        
    if not os.path.exists(os.path.join(project_path, record_path, model_name)):
        os.mkdir(os.path.join(project_path, record_path, model_name))

    if not os.path.exists(os.path.join(project_path, record_path, model_name, "model")):
        os.mkdir(os.path.join(project_path, record_path, model_name, "model"))

    full_log_path = os.path.join(project_path, record_path, model_name, "log.log")
    print("============== Load Dataset ===============")
    normal_path = glob.glob(os.path.join(input_path, modality, "training", "normal", "*.jpg"))
    tumor_path = glob.glob(os.path.join(input_path, modality, "training", "tumor", "*.jpg"))
    print("Num of Tumor Data:", len(tumor_path))
    print("Num of Normal Data:", len(normal_path))
    data = normal_path + tumor_path
    print("data length", len(data))
    total_cases = len(data)

    print("============== Model Setup ===============")
    # train_index, _ = train_test_split(range(total_cases), test_size=0.9)
    # train_index, val_index = train_test_split(train_index, test_size=0.1)

    train_index, val_index = train_test_split(range(total_cases), test_size=0.1)
    train_sampler = SubsetRandomSampler(train_index)
    test_sampler = SubsetRandomSampler(val_index)

    train_dataset = TrainDataset(data, patch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=16,
                                pin_memory=True, drop_last=True, sampler=train_sampler)
    val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=16,
                            pin_memory=True, drop_last=True, sampler=test_sampler)

    ssl = SimCLR(epochs, batch_size, learning_rate, train_loader, val_loader, full_log_path,
    project_path, record_path, model_name, gpuid, unsupervised, model_type)

    print("============== Start Training ===============")
    
    ssl.run()
        