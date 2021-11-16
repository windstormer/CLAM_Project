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
    exp_name = "{}_{}_{{{}}}".format(args.cam, modality, classifier_path)
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
    tumor_path.sort()
    # discard_list = []
    if modality == 't1':
        discard_list = ['00016-54', '00016-55', '00016-56', '00016-57', '00016-58', '00016-59', '00016-60', '00016-61', '00016-62', '00016-63', '00016-65', '00016-66', '00016-67', '00016-68', '00016-69', '00016-70', '00016-90', '00016-92', '00016-93', '00016-94', '00016-95', '00016-96', '00016-97', '00024-29', '00024-30', '00024-31', '00024-32', '00024-33', '00024-34', '00028-75', '00028-76', '00028-77', '00028-78', '00028-79', '00054-118', '00054-119', '00054-120', '00054-121', '00054-122', '00054-123', '00054-124', '00054-76', '00054-77', '00054-78', '00054-79', '00054-80', '00054-81', '00054-82', '00054-83', '00054-84', '00054-85', '00054-86', '00054-87', '00061-44', '00061-45', '00061-46', '00061-47', '00061-48', '00061-49', '00061-50', '00061-51', '00061-52', '00061-53', '00061-54', '00061-55', '00061-56', '00061-58', '00061-59', '00061-60', '00077-101', '00077-102', '00077-103', '00077-104', '00077-105', '00077-106', '00077-107', '00077-108', '00077-109', '00077-110', '00077-111', '00077-112', '00077-113', '00077-114', '00077-115', '00077-116', '00077-117', '00077-118', '00077-119', '00077-120', '00077-121', '00077-123', '00077-124', '00077-125', '00077-64', '00077-65', '00077-66', '00077-67', '00077-68', '00077-69', '00077-70', '00077-71', '00077-72', '00077-73', '00077-74', '00077-75', '00077-76', '00077-84', '00077-85', '00077-86', '00077-87', '00077-88', '00077-94', '00077-95', '00077-96', '00077-97', '00077-98', '00159-63', '00159-64', '00159-65', '00159-66', '00159-67', '00159-68', '00159-69', '00159-70', '00159-71', '00159-72', '00159-73', '00183-122', '00183-58', '00183-59', '00183-60', '00183-61', '00183-62', '00188-115', '00188-76', '00188-77', '00188-79', '00195-120', '00195-124', '00195-83', '00207-115', '00207-116', '00207-117', '00207-87', '00220-68', '00220-72', '00220-73', '00220-77', '00220-78', '00220-82', '00220-96', '00220-97', '00220-98', '00220-99', '00238-115', '00238-116', '00238-117', '00238-74', '00239-43', '00239-44', '00239-45', '00239-46', '00239-47', '00239-48', '00239-49', '00239-79', '00239-80', '00239-81', '00261-36', '00261-37', '00261-38', '00261-39', '00261-40', '00261-41', '00282-57', '00282-64', '00282-67', '00282-68', '00282-69', '00282-70', '00282-71', '00282-80', '00292-43', '00292-44', '00292-45', '00292-46', '00292-47', '00292-48', '00292-49', '00292-50', '00292-52', '00292-58', '00292-59', '00292-60', '00292-63', '00292-64', '00292-65', '00292-66', '00292-67', '00292-68', '00297-57', '00297-58', '00297-59', '00297-60', '00297-61', '00297-62', '00300-40', '00300-41', '00313-100', '00313-102', '00313-104', '00313-105', '00313-106', '00313-107', '00313-108', '00313-84', '00313-85', '00313-93', '00313-94', '00313-97', '00339-130', '00339-131', '00339-132', '00344-43', '00344-44', '00344-53', '00344-54', '00344-55', '00344-56', '00344-57', '00344-58', '00344-59', '00344-60', '00344-61', '00344-62', '00344-63', '00344-64', '00344-65', '00344-66', '00344-67', '00344-68', '00349-41', '00349-43', '00349-44']
    elif modality == 't2':
        discard_list = ['00016-54', '00016-55', '00016-56', '00016-57', '00016-97', '00024-29', '00024-30', '00024-31', '00028-78', '00028-79', '00054-117', '00054-118', '00054-119', '00054-120', '00054-121', '00054-122', '00054-123', '00054-124', '00054-76', '00054-77', '00061-47', '00061-48', '00061-49', '00077-64', '00077-65', '00159-63', '00159-64', '00159-65', '00159-66', '00159-67', '00159-68', '00159-69', '00159-70', '00159-71', '00159-72', '00159-73', '00159-74', '00159-75', '00188-107', '00207-49', '00207-50', '00207-51', '00207-52', '00207-53', '00207-54', '00207-55', '00207-56', '00207-57', '00207-58', '00207-59', '00207-60', '00207-61', '00207-62', '00207-63', '00207-64', '00207-65', '00207-69', '00207-70', '00207-71', '00239-43', '00239-45', '00239-46', '00239-53', '00239-54', '00239-73', '00239-74', '00239-75', '00239-76', '00239-77', '00239-78', '00239-79', '00261-36', '00261-37', '00282-71', '00292-43', '00292-66', '00292-67', '00297-57', '00297-58', '00297-59', '00344-62', '00344-63', '00344-64', '00344-65', '00344-66', '00344-67', '00344-68']
    tumor_new_path = []
    for path in tumor_path[:1000]:
        img_name = path.split(os.path.sep)[-1][:-4]
        if not img_name in discard_list:
            tumor_new_path.append(path)
    # test_dataset = FeatureDataset(test_path, seg_path)
    test_dataset = FeatureDataset(tumor_new_path, seg_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                                pin_memory=True, drop_last=False)
    cam_object.run(test_loader)
