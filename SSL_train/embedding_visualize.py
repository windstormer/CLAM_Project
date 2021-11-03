import argparse
import os, glob
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import *
from models import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_path",
                        type=str, 
                        default="../../",
                        help="path of project")

    parser.add_argument("--dataset_path",
                        type=str, 
                        default="/hdd1/vincent18/BraTS_patch/",
                        help="path of dataset")

    parser.add_argument("--gpu_id",
                        type=str,
                        default ='',
                        help="gpu id number")


    parser.add_argument("--pretrain_model",
                        type=str,
                        default=None,
                        help="pretrain model")
    
    parser.add_argument("--visualize",
                        "-v",
                        type=str,
                        default="PCA",
                        help="visualization method [PCA, TSNE]")

    parser.add_argument('--batch_size', '-b', default=512, type=int, help='Batch size of dataloader')

    # args parse
    args = parser.parse_args()
    if args.gpu_id != '':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    record_path = "record"
    dataset_path = args.dataset_path
    assert args.visualize in ['PCA', 'TSNE'], 'undefined visualize method'
    assert args.pretrain_model != None, 'pretrain model is None'

    sections = args.pretrain_model.split("_")
    for section in sections:
        if "ep" in section:
            pretrain_epoch = int(section[2:])
    modality = sections[1]
    pretrain_model_path = os.path.join(args.project_path, record_path, args.pretrain_model, "model", "self_model_{}.pth".format(pretrain_epoch))
    
    # pretrain_model_path = os.path.join(args.project_path, record_path, "weights", "embedder.pth")

    if not os.path.exists("temp_data/{}".format(args.pretrain_model)):
        os.mkdir("temp_data/{}".format(args.pretrain_model))
        print("============== Load Dataset ===============")
        label_dict = {}
        normal_path = glob.glob(os.path.join(dataset_path, modality, "training", "normal", "*.jpg"))
        tumor_path = glob.glob(os.path.join(dataset_path, modality, "training", "tumor", "*.jpg"))

        print("============== Model Setup ===============")
        
        model = SSLModel_Inference(pretrain_model_path).cuda()
        for param in model.parameters():
            param.requires_grad = False
        print("============== Start Convert ===============")
        print("========= Convert Normal Feature ===========")
        dataset = ImageDataset(normal_path)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
        normal_feature_list = []
        with torch.no_grad():
            for data_batch in tqdm(loader):
                # print(data_batch.shape)
                bag = data_batch.cuda()
                feature = model(bag)
                normal_feature_list.append(feature.cpu())
        normal_feature_list = torch.cat(normal_feature_list, dim=0)
        normal_feature_list = normal_feature_list.numpy()
        print("normal_feature_list.shape", normal_feature_list.shape)
        np.save("temp_data/{}/normal_feature.npy".format(args.pretrain_model), normal_feature_list)

        print("========= Convert Tumor Feature ===========")
        dataset = ImageDataset(tumor_path)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
        tumor_feature_list = []
        with torch.no_grad():
            for data_batch in tqdm(loader):
                # print(data_batch.shape)
                bag = data_batch.cuda()
                feature = model(bag)
                tumor_feature_list.append(feature.cpu())
        tumor_feature_list = torch.cat(tumor_feature_list, dim=0)
        tumor_feature_list = tumor_feature_list.numpy()
        print("tumor_feature_list.shape", tumor_feature_list.shape)
        np.save("temp_data/{}/tumor_feature.npy".format(args.pretrain_model), tumor_feature_list)

    else:
        tumor_feature_list = np.load("temp_data/{}/tumor_feature.npy".format(args.pretrain_model))
        normal_feature_list = np.load("temp_data/{}/normal_feature.npy".format(args.pretrain_model))
        print("tumor_feature_list.shape", tumor_feature_list.shape)
        print("normal_feature_list.shape", normal_feature_list.shape)
    feature_list = np.concatenate([tumor_feature_list, normal_feature_list])
    tumor_num = len(tumor_feature_list)
    normal_num = len(normal_feature_list)
    print("feature_list.shape", feature_list.shape)
    print("============== Feature Reduction ===============")
    if args.visualize == 'PCA':
        X = PCA(n_components=2).fit_transform(feature_list)
    elif args.visualize == 'TSNE':
        X = TSNE(n_components=2).fit_transform(feature_list)
    print("reduced feature_list.shape", X.shape)
    X_norm = (X - X.min()) / (X.max() - X.min())
    plt.figure(figsize=(12, 12))
    plt.scatter(X_norm[:tumor_num, 0], X_norm[:tumor_num, 1], c='g', label='tumor')
    plt.scatter(X_norm[tumor_num:normal_num, 0], X_norm[tumor_num:normal_num, 1], c='b', label='normal')
    plt.legend(prop={'size': 12})
    # plt.xlim(0,0.3) 
    plt.savefig("image/{}_{}.png".format(args.pretrain_model, args.visualize))