import argparse
import os, glob
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import KFold
from datetime import datetime

from dataset import *
from cam import CAM
from scorecam import ScoreCAM
from clam import CLAM
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
    
    parser.add_argument("--classifier_path",
                        type=str, 
                        default="",
                        help="pretrained classifier path")

    parser.add_argument("--gpu_id",
                        type=str,
                        default ='',
                        help="gpu id number")   

    parser.add_argument("--suffix",
                        '-s',
                        type=str,
                        default=None,
                        help="suffix")
    
    parser.add_argument('-m',
                        '--modality',
                        type=str,
                        default='t1',
                        choices=['flair', 't1', 't1ce', 't2'],
                        help='Modality select')

    parser.add_argument('--cam',
                        '-c',
                        type=str, 
                        default="CLAM",
                        choices=['CAM', 'ScoreCAM', 'CLAM'],
                        help="CAM technology")

    # args parse
    args = parser.parse_args()
    if args.gpu_id != '':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    project_path = args.project_path
    suffix = args.suffix
    modality = args.modality
    classifier_path = args.classifier_path

    dataset_path = "/work/vincent18/"
    input_path = os.path.join(dataset_path, args.input_path)

    assert classifier_path != '', 'classifier_path not given'
    
    encoder_model_type = classifier_path.split('{')[1].split("_")[0]
    print("Encoder Model Type:", encoder_model_type)
    exp_name = "{}_test_{}_{{{}}}".format(args.cam, modality, classifier_path)
    if suffix != None:
        exp_name += ".{}".format(suffix)
    print(exp_name)


    print("============== Load Image ===============")
    normal_path = glob.glob(os.path.join(input_path, modality, "validate", "normal", "*.jpg"))
    tumor_path = glob.glob(os.path.join(input_path, modality, "validate", "tumor", "*.jpg"))
    seg_path = glob.glob(os.path.join(input_path, modality, "validate", "seg", "*.png"))
    test_path = glob.glob(os.path.join("test", "*.jpg"))
    print("Num of Tumor Data:", len(tumor_path))
    print("Num of Normal Data:", len(normal_path))
    print("Num of Seg Data:", len(seg_path))

    print("============== Load Model ===============")

    if args.cam == 'CAM':
        cam_object = CAM(project_path, classifier_path, exp_name, encoder_model_type)
    elif args.cam == 'ScoreCAM':
        cam_object = ScoreCAM(project_path, classifier_path, exp_name, encoder_model_type)
    elif args.cam == 'CLAM':
        cam_object = CLAM(project_path, classifier_path, exp_name, encoder_model_type)
    print("============== Start Testing Tumor ===============")
    test_id = ['00183-66', '00300-98', '00459-66', '00472-52', '01119-42', '01232-108', '01284-113', '01388-118', '01468-50', '01573-101',
     '01612-100', '01560-97', '01486-92', '01303-97', '01189-72']
    normal_id = ['01651-100', "01582-121", "00028-51", "00024-104"]
    tumor_new_path = []
    for path in tumor_path:
        if path.split(os.path.sep)[-1][:-4] in test_id:
            tumor_new_path.append(path)
    for path in normal_path:
        if path.split(os.path.sep)[-1][:-4] in normal_id:
            tumor_new_path.append(path)
    test_dataset = FeatureDataset(tumor_new_path, seg_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                                pin_memory=True, drop_last=False)
    cam_object.run(test_loader)
