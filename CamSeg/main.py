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
from clam_acc import CLAMACC
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

    dataset_path = "/hdd1/vincent18/"
    input_path = os.path.join(dataset_path, args.input_path)

    assert classifier_path != '', 'classifier_path not given'
    
    encoder_model_type = classifier_path.split('{')[1].split("_")[0]
    print("Encoder Model Type:", encoder_model_type)
    exp_name = "Acc_Exp_{}_{}_{{{}}}".format(args.cam, modality, classifier_path)
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
    elif args.cam == 'CLAMACC':
        cam_object = CLAMACC(project_path, classifier_path, exp_name, encoder_model_type)
    print("============== Start Testing Tumor ===============")
    # discard_list = ["01119-42", "01284-113", "01500-128", "00077-124", "01492-99", "01547-55", "01166-86", "00016-58", '00220-73', '00412-111', '00459-43', '01486-120', '01522-74', '01354-68', '00220-72', '01529-53', '01157-65', '01560-109', '01157-113', '00313-112', '01560-96', '01522-83', '00459-83', '01560-106', '01284-53', '00477-52', '00432-125', '01277-45', '01160-46', '01172-56', '01354-71', '00016-55', '01024-110', '01044-125', '00054-117', '01327-65', '01571-85', '01486-119', '01110-95', '00459-81', '01342-109', '01187-111', '01359-53', '00386-80', '01342-100', '01115-43', '01110-86', '00459-49', '00477-50', '01141-67', '00339-131', '01342-69', '01119-27', '01010-35', '00459-82', '01547-56', '00313-104', '00638-66', '01486-121', '01110-91', '01342-75', '01110-90', '00220-100', '01571-84', '01342-105', '00638-122', '00344-57', '00344-63', '01187-108', '00261-92', '01154-57', '01560-115', '01154-62', '01522-72', '01416-132', '01405-110', '01486-114', '00344-49', '00444-54', '01612-57', '01560-99', '00598-31', '01404-69', '01119-24', '00159-68', '01492-115', '01486-117', '00016-92', '01522-85', '00024-32', '01529-43', '01110-63', '01187-104', '01172-59', '01612-95', '01342-114', '01404-47', '01560-104', '01110-85', '01347-77', '01560-107', '00432-129', '01166-88', '01612-99', '01560-114', '01529-54', '01571-88', '00403-52', '01262-70', '01262-69', '00775-26', '01119-32', '01450-43', '01172-58', '01044-126', '01529-58', '00459-84', '01277-46', '01187-80', '01529-38', '01119-47', '01612-58', '01450-42', '00444-53', '00054-82', '01416-130', '00313-114', '01119-15', '01529-60', '00638-71', '00339-132', '00016-96', '01347-79', '01529-39', '00016-97', '00477-47', '01187-79', '00638-72', '01450-44', '01560-95', '00840-129', '01359-88', '01486-112', '01404-68', '01547-57', '01347-78', '01529-50', '01187-96', '01405-108', '01522-78', '00638-69', '00459-57', '01154-60', '01492-107', '01122-132', '01492-113', '01172-60', '01327-62', '00054-118', '01492-112', '01154-54', '01486-118', '01154-61', '00061-47', '01327-58', '01187-97', '00444-58', '01359-84', '00159-70', '00810-98', '01529-56', '00598-83', '00459-48', '00292-66', '00054-123', '00220-70', '01529-37', '01342-113', '01265-51', '01119-26', '01012-54', '01154-58', '00292-68', '01053-40', '00544-57', '00638-67', '01359-89', '00061-45', '00369-127', '00188-116', '01304-55', '01416-131', '00313-107', '00412-109', '01122-131', '01492-96', '01119-39', '00344-68', '01359-86', '01115-41', '01573-122', '01327-60', '00207-117', '01119-35', '00638-140', '01450-51', '01359-90', '00054-78', '01110-60', '01110-84', '01571-83', '01110-93', '00344-58', '01405-111', '00344-65', '00028-75', '00344-47', '01486-123', '00638-139', '00292-49', '01303-125', '01115-40', '00292-45', '01404-63', '01522-67', '00638-68', '00412-110', '00477-51', '01157-114', '00282-101', '01119-17', '01262-68', '00344-60', '01187-99', '00220-99', '00775-24', '01354-66', '01110-87', '00282-58', '01404-64', '01612-90', '00054-81', '00344-50', '00313-113', '01187-81', '01388-126', '01303-124', '01119-20', '00444-57', '01422-114', '01560-110', '00292-59', '00292-61', '01450-89', '00386-81', '01500-105', '00477-45', '01119-40', '01450-50', '00344-67', '00638-121', '01529-49', '01119-30', '00313-105', '00238-82', '00459-85', '00598-32', '01571-80', '01154-59', '00028-76', '00028-78', '01342-70', '01529-59', '01187-106', '01110-64', '00472-30', '01259-38', '01187-114', '00016-59', '01529-57', '00220-98', '01053-97', '01160-47', '01187-113', '01522-84', '01187-98', '00220-74', '01342-106', '00459-44', '00292-44', '01416-127', '00292-67', '01529-46', '01119-16', '00292-63', '01612-66', '00292-64', '01416-128', '01612-96', '01560-97', '01141-120', '00412-113', '00598-88', '00477-70', '01342-76', '01612-62', '00775-25', '01187-78', '01571-82', '00444-59', '00344-55', '01529-52', '01522-80', '01422-111', '00445-100', '01141-66', '00313-109', '00220-75', '01529-85', '01141-121', '01522-79', '01327-57', '00159-74', '01262-62', '01327-59', '01187-109', '01522-71', '01262-67', '01486-109', '00313-115', '01154-66', '00344-62', '01486-110', '00292-47', '01552-54', '01571-81', '00016-95', '01547-112', '01450-90', '01547-58', '00054-80', '00598-82', '00344-56', '01612-82', '01321-63', '01347-62', '01422-113', '00016-94', '01119-41', '01119-36', '00532-130', '01187-105', '01187-112', '01450-46', '01044-123', '00054-76', '00024-29', '01141-65', '00344-45', '00638-63', '00344-61', '00016-54', '00638-65', '01141-123', '01450-54', '00459-59', '00598-86', '00445-102', '01342-104', '00459-58', '01119-48', '01422-109', '01571-86', '01110-66', '01547-61', '00459-41', '01110-92', '01522-77', '00054-77', '01110-94', '01262-65', '01122-130', '01154-63', '01342-68', '00061-48', '00477-62', '00459-45', '01612-59', '00054-124', '01327-64', '01529-51', '01529-55', '01571-87', '00598-29', '00459-42', '01404-65', '01486-115', '00816-35', '01404-48', '00016-93', '01529-45', '01359-51', '01187-100', '00459-60', '01342-112', '01571-89', '01119-29', '01327-61', '01522-69', '00238-87', '01404-67', '01486-106', '01327-63', '00618-88', '00292-65', '00477-61', '00238-81', '00432-130', '01529-40', '00459-64', '01119-21', '01354-67', '00459-86', '01321-118', '01486-107', '00412-112', '00028-77', '01529-41', '01522-75', '00344-66', '00016-57', '01044-122', '01404-49', '01242-100', '01119-28', '01342-72', '01450-41', '01110-58', '01187-102', '01321-119', '01141-92', '01172-57', '01141-124', '00477-46', '01612-100', '01522-73', '00477-60', '00292-48', '01189-114', '01560-116', '00344-48', '01522-86', '00532-129', '00598-87', '01422-96', '00477-56', '01416-129', '00477-55', '01141-119', '00292-50', '01166-89', '01492-100', '01486-122', '00016-56', '00024-30', '01612-98', '01262-66', '01492-111', '00477-72', '01492-97', '01110-65', '01529-48', '01342-103', '00313-103', '01110-89', '01119-37', '01119-23', '01110-59', '00477-71', '00369-128', '01560-103', '01486-113', '01450-53', '00159-65', '01529-42', '01450-52', '00054-83', '01422-97', '01560-113', '00292-51', '00195-126', '01342-73', '01461-87', '00344-46', '00207-116', '01450-45', '00598-85', '01492-110', '01612-93', '00444-56', '00159-64', '01153-43', '01119-19', '00188-117', '00344-64', '01119-44', '01492-98', '00061-49', '00054-116', '01119-43', '01522-68', '00638-70', '01189-115', '01571-90', '00775-62', '00220-68', '01119-38', '01342-71', '01110-96', '00344-44', '01187-103', '01612-56', '01560-108', '00292-46', '01119-46', '01486-111', '01154-64', '01386-43', '01560-111', '01119-33', '01342-102', '01110-61', '00054-120', '01522-70', '01262-71', '01529-44', '00444-55', '01342-101', '00544-55', '01612-97', '01354-65', '00816-36', '01422-110', '01612-92', '01405-109', '00054-79', '01612-91', '00386-82', '01303-123', '01522-76', '01119-31', '01612-83', '01115-44', '01187-101', '01522-82', '00292-43', '01119-25', '00459-63', '00220-69', '00432-124', '01119-18', '01166-63', '01115-42', '01262-64', '00313-106', '01612-94', '01422-112', '00188-77', '01450-85', '00638-64', '01492-114', '01486-108', '01119-34', '01522-66', '01342-107', '00195-125', '01560-112', '00292-58', '01153-41', '00220-71', '01327-56', '01461-88', '00477-48', '01153-42', '01262-63', '00238-88', '01259-39', '01110-67', '01115-81', '00445-101', '00412-108', '01529-47', '00444-60', '00159-69', '00159-72', '00238-89', '00638-138', '00618-89', '01547-54', '01187-110', '01154-65', '01486-116', '01386-44', '00159-73', '01342-108', '00544-56', '01010-34', '00292-60', '01119-22', '01110-62', '01141-122', '00238-76', '01386-45', '01347-76', '01044-124', '01404-66', '01450-47', '00532-131', '01359-52'
    # , '00061-46', '01552-48', '00432-66', '01347-63', '00077-64', '01461-85', '01347-71', '01110-88', '01212-74', '01552-109', '00238-98', '01393-123', '01347-66', '01482-85', '00532-74', '01347-72', '01303-63', '01482-89', '01547-59', '00840-78', '01265-89', '00239-44', '00367-109', '01547-64', '01461-47', '01552-51', '01461-44', '01482-84', '01537-127', '01321-65', '00432-128', '01482-67', '01461-82', '00349-44', '01461-80', '01347-65', '01262-89', '01265-90', '01416-65', '01416-66', '01262-83', '01262-73', '00477-65', '01404-61', '00238-96', '01053-95', '01187-86', '01223-80', '01153-122', '00810-40', '00477-53', '00477-64', '00077-125', '01537-66', '01089-38', '01554-50', '00638-133', '00477-54', '01157-66', '01461-86', '00238-97', '00477-49', '00412-95', '00367-108', '01552-108', '00238-106', '01422-100', '01461-83', '00440-118', '01482-80', '00297-107', '00238-102', '01404-62', '01537-65', '01321-64', '01262-84', '01386-116', '01242-48', '01053-44', '01347-68', '00477-63', '01265-88', '01537-67', '00432-64', '01422-99', '01166-65', '01242-98', '00292-56', '01110-81', '01257-78', '00386-42', '01560-105', '01552-46', '00840-76', '00638-137', '01141-68', '01486-102', '01404-60', '00367-107', '01560-98', '01321-68', '01242-97', '01404-59', '01321-70', '01187-95', '01284-115', '01119-45', '01582-57', '01461-45', '00432-61', '00183-60', '00544-53', '01404-50', '01242-99', '01153-45', '01044-62', '01166-64', '01340-83', '00292-62', '00159-71', '01573-120', '01354-70', '00238-86', '01482-83', '01486-100', '01157-117', '01482-79', '01347-69', '00444-67', '01265-87', '01461-89', '01450-48', '01560-102', '00159-63', '01537-61', '01166-66', '01482-72', '01388-125', '01212-75', '01573-121', '01522-81', '00054-122', '01537-125', '00432-65', '00412-107', '01257-75', '00188-76', '01482-74', '01223-79', '01157-64', '00432-122', '00477-57', '01347-73', '01354-69', '01405-107', '00183-61', '01482-73', '01354-106', '01347-70', '00638-76', '00440-119', '00432-63', '00367-117', '00238-105', '01202-38', '01262-90', '00188-79', '01257-76', '00386-41', '01157-116', '01612-65', '01537-63', '01257-77', '01262-75', '01547-60', '00188-80', '00440-61', '01053-94', '00188-81', '01262-78', '01537-124', '00238-85', '01242-49', '01154-55', '01482-78', '01537-126', '00444-66', '00638-73', '01347-67', '01612-73', '00349-43', '01024-109', '00477-59', '01347-75', '00432-62', '00292-54', '00077-123', '01405-106', '01354-63', '00444-65', '01612-72', '01404-52', '00638-75', '00574-39', '01110-82', '01262-74', '01202-40', '00159-66', '00444-64', '01552-50', '01612-87', '01262-82', '00386-84', '01450-55', '01321-69', '00638-134', '01262-80', '00459-80', '01212-77', '01212-78', '01486-99', '01089-104', '01187-107', '01157-63', '00440-62', '00386-83', '01110-83', '00220-97', '00810-39', '00349-41', '01547-63', '01092-120', '01284-114', '00077-76', '01547-62', '01262-77', '01537-62', '01486-98', '01461-81', '01482-87', '00344-59', '00239-81', '01053-96', '00183-59', '01262-72', '01100-90', '01461-46', '00292-57', '00638-135', '01153-44', '00532-75', '00638-81', '01154-56', '01354-101', '01265-91', '01386-115', '01482-71', '01303-64', '01262-85', '01347-64', '01347-74', '01166-87', '00432-67', '00840-77', '01342-115', '00444-68', '00477-58', '00440-117', '00638-136', '01189-68', '01354-64', '01115-80', '00077-75', '00183-58', '01187-82', '01303-126', '00472-31', '01187-88', '01612-84', '01422-98', '00440-57', '00077-65', '00297-57', '01482-86', '01552-47', '00440-59', '01404-51', '01187-94', '01552-110', '01560-101', '01582-58', '01262-76', '01422-101', '00292-53', '01212-76', '01482-82', '00238-107', '01552-49', '00349-85', '01560-100', '00598-30', '00061-44', '01187-87', '01321-67', '00445-43', '01612-86', '01321-66', '01089-39', '01212-73', '01450-49', '01537-64', '00239-45', '01461-84', '01482-88'
    # , '00077-113', '01354-104', '01157-68', '00313-101', '00220-95', '01340-109', '01223-81', '01359-87', '00188-78', '00077-116', '00810-38', '00472-32', '01342-92', '01651-110', '00349-84', '01212-55', '00445-103', '01092-59', '01651-117', '01262-79', '01092-58', '00349-42', '01571-118', '00840-79', '01342-78', '00810-42', '01571-119', '01492-86', '00239-49', '01500-127', '01354-103', '01342-93', '01304-56', '00810-41', '01450-40', '01342-74', '00775-61', '00239-48', '01537-69', '01284-112', '00077-114', '00459-46', '00412-106', '00024-31', '00339-130', '01141-64', '01115-84', '01187-85', '01115-83', '01212-58', '01342-77', '01141-61', '00781-29', '00412-81', '00367-82', '01115-87', '01157-69', '01303-65', '00016-91', '00459-47', '00282-57', '00452-66', '00188-115', '00544-34', '00367-71', '00445-42', '01187-83', '01187-84', '00618-50', '01492-85', '01492-84', '01141-62', '00239-79', '00412-80', '00239-80', '01537-68', '00544-33', '00261-36', '00028-79', '00339-129', '00077-122', '01157-115', '00220-96', '01141-63', '01354-105', '01153-121', '00810-43', '01044-121', '00077-115']
    # tumor_new_path = []
    # for path in tumor_path:
    #     img_name = path.split(os.path.sep)[-1][:-4]
    #     if not img_name in discard_list:
    #         tumor_new_path.append(path)
    # test_dataset = FeatureDataset(test_path, seg_path)
    test_dataset = FeatureDataset(tumor_path, seg_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                                pin_memory=True, drop_last=False)
    cam_object.run(test_loader)
