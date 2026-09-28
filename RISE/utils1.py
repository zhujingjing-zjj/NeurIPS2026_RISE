import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import random
from sklearn.metrics import confusion_matrix
import torch.optim as optim
# from model import *
from datasets1 import *
import ipdb
import copy
from loss import *
import argparse
from classes_names import name_classes
import clip 
import random

# # 设置随机种子确保一致性
# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)

# # 调用 set_seed 来保证翻转的一致性
# set_seed(42)

def log_model_info(model, verbose=False):
    """Logs model info"""
    if verbose:
        logger.info(f"Classification Model:\n{model}")
    model_total_params = sum(p.numel() for p in model.parameters())  #统计模型“总参数量”

    model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  #统计“参与反向传播的参数量”
    logger.info("Total Parameters: {0}\t Gradient Parameters: {1}".format(model_total_params, model_grad_params))  #Total Parameters: 151005184    Gradient Parameters: 5120
    logger.info("tuned percent:%.3f"%(model_grad_params/model_total_params*100))   #tuned percent:0.003
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--device', type=str, default='cuda:2', help='The device to run the program')
    args = parser.parse_args()
    return args

args = get_args()
device = torch.device(args.device)

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def get_updateModel(model, cal_dnn):
    model_dict = model.state_dict()
    dnn_dict = cal_dnn.state_dict()
#     for k, v in dnn_dict.items():
#         if k in dnn_dict:
#             print(k)
#     print('********************************')   
#     import ipdb; ipdb.set_trace()    
    shared_dict = {k: v for k, v in dnn_dict.items() if (k in model_dict)}

    model_dict.update(shared_dict)
    model.load_state_dict(model_dict)
    return model


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def load_mnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)

def load_fmnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    fmnist_train_ds = FashionMNIST_truncated(datadir, train=True, download=True, transform=transform)
    fmnist_test_ds = FashionMNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = fmnist_train_ds.data, fmnist_train_ds.target
    X_test, y_test = fmnist_test_ds.data, fmnist_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_cifar10_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_tinyimagenet_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    xray_train_ds = TinyImageNet_load('../datasets/tiny-imagenet-200/', train=True, transform=transform)
    xray_test_ds = TinyImageNet_load('../datasets/tiny-imagenet-200/', train=False, transform=transform)

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples])

    return (X_train, y_train, X_test, y_test)

def load_vireo_data():
    transform = transforms.Compose([transforms.ToTensor()])

    vireo_train_ds = Vireo172_truncated(transform=transform, mode='train')
    vireo_test_ds = Vireo172_truncated(transform=transform, mode='test')

    X_train, y_train = vireo_train_ds.path_to_images, vireo_train_ds.labels
    X_test, y_test = vireo_test_ds.path_to_images, vireo_test_ds.labels

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)

def load_food_data():
    transform = transforms.Compose([transforms.ToTensor()])

    vireo_train_ds = Food101_truncated(transform=transform, mode='train')
    vireo_test_ds = Food101_truncated(transform=transform, mode='test')

    X_train, y_train = vireo_train_ds.path_to_images, vireo_train_ds.labels
    X_test, y_test = vireo_test_ds.path_to_images, vireo_test_ds.labels

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_dtd_data():
    transform = transforms.Compose([transforms.ToTensor()])

    dtd_train_ds = DTD(transform=transform, mode='train')
    dtd_test_ds = DTD(transform=transform, mode='test')

    X_train, y_train = dtd_train_ds.path_to_images, dtd_train_ds.labels
    X_test, y_test = dtd_test_ds.path_to_images, dtd_test_ds.labels

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_cal101_data():
    transform = transforms.Compose([transforms.ToTensor()])

    cal101_train_ds = Caltech101(transform=transform, mode='train')
    cal101_test_ds = Caltech101(transform=transform, mode='test')

    X_train, y_train = cal101_train_ds.path_to_images, cal101_train_ds.labels
    X_test, y_test = cal101_test_ds.path_to_images, cal101_test_ds.labels

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_aircraft_data():
    transform = transforms.Compose([transforms.ToTensor()])

    air_train_ds = Air100(transform=transform, mode='train')
    air_test_ds = Air100(transform=transform, mode='test')

    X_train, y_train = air_train_ds.path_to_images, air_train_ds.labels
    X_test, y_test = air_test_ds.path_to_images, air_test_ds.labels

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)

def load_cars_data():
    transform = transforms.Compose([transforms.ToTensor()])

    cars_train_ds = Cars196(transform=transform, mode='train')
    cars_test_ds = Cars196(transform=transform, mode='test')
    
    X_train, y_train = cars_train_ds.path_to_images, cars_train_ds.labels
    X_test, y_test = cars_test_ds.path_to_images, cars_test_ds.labels
    
    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_flowers_data():
    transform = transforms.Compose([transforms.ToTensor()])

    flowers_train_ds = Flowers102(transform=transform, mode='train')
    flowers_test_ds = Flowers102(transform=transform, mode='test')

    X_train, y_train = flowers_train_ds.path_to_images, flowers_train_ds.labels
    X_test, y_test = flowers_test_ds.path_to_images, flowers_test_ds.labels

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_ucf_data():
    transform = transforms.Compose([transforms.ToTensor()])

    ucf_train_ds = UCF101(transform=transform, mode='train')
    ucf_test_ds = UCF101(transform=transform, mode='test')

    X_train, y_train = ucf_train_ds.path_to_images, ucf_train_ds.labels
    X_test, y_test = ucf_test_ds.path_to_images, ucf_test_ds.labels

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)

def record_net_data_stats(y_train, net_dataidx_map, logdir):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list=[]
    for net_id, data in net_cls_counts.items():
        n_total=0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def partition_data(dataset, datadir, logdir, partition, n_parties, name_classes,beta=0.4):
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    elif dataset == 'fmnist':
        X_train, y_train, X_test, y_test = load_fmnist_data(datadir)          
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    elif dataset == 'tinyimagenet':
        X_train, y_train, X_test, y_test = load_tinyimagenet_data(datadir)
    elif dataset == 'vireo172':
        X_train, y_train, X_test, y_test = load_vireo_data()
    elif dataset == 'food101':
        X_train, y_train, X_test, y_test = load_food_data()
    elif dataset == 'dtd':
        X_train, y_train, X_test, y_test = load_dtd_data() 
    elif dataset == 'cal101':
        X_train, y_train, X_test, y_test = load_cal101_data()   
    elif dataset == 'air100':
        X_train, y_train, X_test, y_test = load_aircraft_data() 
    elif dataset == 'cars196':
        X_train, y_train, X_test, y_test = load_cars_data()   
    elif dataset == 'flowers102':
        X_train, y_train, X_test, y_test = load_flowers_data()   
    elif dataset == 'ucf101':
        X_train, y_train, X_test, y_test = load_ucf_data()   
        
    n_train = y_train.shape[0]
    net_classname_map = {}
    if partition == "homo" or partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}


    elif partition == "noniid-labeldir" or partition == "noniid":
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset == 'cifar100' or dataset == 'air100':
            K = 100
        elif dataset == 'tinyimagenet':
            K = 200
        elif dataset == 'vireo172':
            K = 172
        elif dataset == 'food101':
            K = 101 
        elif dataset == 'dtd':
            K = 47   
        elif dataset == 'cal101':
            K = 101   
        elif dataset == 'cars196':
            K = 196  
        elif dataset == 'flowers102':
            K = 102 
        elif dataset == 'ucf101':
            K = 101  
            
        N = y_train.shape[0]
        print('mmm', np.unique(y_train)) 
        net_dataidx_map = {}
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # if K == 2 and n_parties <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])   #打乱当前客户端的样本顺序
            net_dataidx_map[j] = idx_batch[j]  #保存该客户端的“样本索引映射”
            unq, unq_cnt = np.unique(y_train[net_dataidx_map[j]], return_counts=True)
            #unq:客户端j拥有哪些“类别 ID”      unq_cnt:每个类别在客户端中的样本数量
            # ipdb.set_trace()
            # net_classname_map[j] = np.array(name_classes)[unq.astype(int)]
            # net_unq_map[j] = unq

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    # return (X_train, y_train, X_test, y_test, net_dataidx_map, net_classname_map,traindata_cls_counts)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)


def get_trainable_parameters(net, device='cpu'):
    'return trainable parameter values as a vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    # print("net.parameter.data:", list(net.parameters()))
    paramlist = list(trainable)
    #print("paramlist:", paramlist)
    N = 0
    for params in paramlist:
        N += params.numel()
        # print("params.data:", params.data)
    X = torch.empty(N, dtype=torch.float64, device=device)
    X.fill_(0.0)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            X[offset:offset + numel].copy_(params.data.view_as(X[offset:offset + numel].data))
        offset += numel
    # print("get trainable x:", X)
    return X


def put_trainable_parameters(net, X):
    'replace trainable parameter values by the given vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    paramlist = list(trainable)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            params.data.copy_(X[offset:offset + numel].data.view_as(params.data))
        offset += numel


def compute_accuracy1(model, dnn, dataloader, get_confusion_matrix=False, device="cpu", multiloader=False):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    if device == 'cpu':
        criterion = nn.CrossEntropyLoss()
    elif "cuda" in device.type:
        criterion = nn.CrossEntropyLoss().to(device)
    loss_collector = []

    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            #print("x:",x)
            if device != 'cpu':
                x, target = x.to(device), target.to(dtype=torch.int64).to(device)
            _,feats, _ = model(x)
            _, out = dnn(feats)

            loss = criterion(out, target)
            _, pred_label = torch.max(out.data, 1)
            loss_collector.append(loss.item())
            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()

            if device == "cpu":
                pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                true_labels_list = np.append(true_labels_list, target.data.numpy())                             
            else:
                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
        avg_loss = sum(loss_collector) / len(loss_collector)

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct / float(total), conf_matrix, avg_loss

    return correct / float(total), avg_loss

def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu", multiloader=False):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    if device == 'cpu':
        criterion = nn.CrossEntropyLoss()
    elif "cuda" in device.type:
        criterion = nn.CrossEntropyLoss().to(device)
    loss_collector = []

    #---------------------------------------------------计算文本-----------------------------------------------------------#
    # # ipdb.set_trace()
    # classes = name_classes(args)  
    # prompts = [f"a photo of {c.replace('_', ' ')}" for c in classes]
    # with torch.no_grad():
    #     text_tokens = clip.tokenize(prompts).to(device)
    #     cls_txt = model.clip.encode_text(text_tokens).float()  #10 768
    #---------------------------------------------------计算文本-----------------------------------------------------------#

    with torch.no_grad():
        # for batch_idx, (x, target) in enumerate(dataloader):
        for batch_idx, (x, target, _) in enumerate(dataloader):
            if device != 'cpu':
                x, target = x.to(device), target.to(dtype=torch.int64).to(device)
                # ipdb.set_trace()
            #------------------------resnet18和clip(PromptFL)----------------------------#
            # _, feats, out = model(x)
            #-------------------------vit和clip(使用分类头的情况)--------------------------#
            _,_,out = model(x)
            #--------------clip使用文本和图像计算相似度(FedCLIP)--------------------#
            # cls_img = model(x)   #64 768
            # if cls_txt.dim() == 2:
            #     cls_txt=cls_txt.unsqueeze(0).repeat((cls_img.size(0)), 1, 1)   #64 10 768 
            # elif cls_txt.dim() == 3:
            #     cls_txt_base = cls_txt[0]
            #     cls_txt = cls_txt_base.unsqueeze(0).expand(cls_img.size(0), -1, -1)            
            # cls_txt = cls_txt/cls_txt.norm(dim=-1, keepdim=True) 
            # cls_img = cls_img/cls_img.norm(dim=-1, keepdim=True) 
            # tem = model.clip.logit_scale.exp().to(device) 
            # out= (tem * torch.bmm(cls_img.unsqueeze(1), cls_txt.transpose(1, 2)).squeeze(1))  #64 10
            #------------------------------------------------------------#

            loss = criterion(out, target)
            _, pred_label = torch.max(out.data, 1)

            # 计算 batch 正确数
            batch_correct = (pred_label == target).sum().item()
            batch_total = target.size(0)
            batch_acc = batch_correct / batch_total
            print(
                f"[Batch {batch_idx}] "
                f"Pred: {pred_label[:10].detach().cpu().numpy()} | "
                f"GT: {target[:10].detach().cpu().numpy()} | "
                f"Batch Acc: {batch_acc:.4f}"
            )        

            loss_collector.append(loss.item())
            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()

            if device == "cpu":
                pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                true_labels_list = np.append(true_labels_list, target.data.numpy())                             
            else:
                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
        avg_loss = sum(loss_collector) / len(loss_collector)

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct / float(total), conf_matrix, avg_loss

    return correct / float(total), avg_loss


def compute_accuracy_features(model, dataloader, get_confusion_matrix=False, device="cpu"):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    if isinstance(device, str):
        device = torch.device(device)

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    criterion = nn.CrossEntropyLoss().to(device)
    loss_collector = []

    model.to(device)

    with torch.no_grad():
        for batch_idx, (feats, target, _) in enumerate(dataloader):
            feats = feats.to(device)
            target = target.to(dtype=torch.int64).to(device)

            # feats 是已经保存好的冻结编码器特征
            _, _, out = model(feats)

            # out 必须是 [B, num_classes]
            loss = criterion(out, target)
            _, pred_label = torch.max(out.data, 1)

            batch_correct = (pred_label == target).sum().item()
            batch_total = target.size(0)
            batch_acc = batch_correct / batch_total

            print(
                f"[Batch {batch_idx}] "
                f"Pred: {pred_label[:10].detach().cpu().numpy()} | "
                f"GT: {target[:10].detach().cpu().numpy()} | "
                f"Batch Acc: {batch_acc:.4f}"
            )

            loss_collector.append(loss.item())
            total += target.size(0)
            correct += (pred_label == target).sum().item()

            pred_labels_list = np.append(
                pred_labels_list,
                pred_label.detach().cpu().numpy()
            )
            true_labels_list = np.append(
                true_labels_list,
                target.detach().cpu().numpy()
            )

    avg_loss = sum(loss_collector) / len(loss_collector)

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)
    else:
        conf_matrix = None

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct / float(total), conf_matrix, avg_loss

    return correct / float(total), avg_loss

def compute_accuracy_etf(model, dataloader, get_confusion_matrix=False, device="cpu", multiloader=False):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    if device == 'cpu':
        criterion = nn.CrossEntropyLoss()
    elif "cuda" in device.type:
        criterion = nn.CrossEntropyLoss().to(device)
    loss_collector = []

    #---------------------------------------------------计算文本-----------------------------------------------------------#
    # # ipdb.set_trace()
    # classes = name_classes(args)  
    # prompts = [f"a photo of {c.replace('_', ' ')}" for c in classes]
    # with torch.no_grad():
    #     text_tokens = clip.tokenize(prompts).to(device)
    #     cls_txt = model.clip.encode_text(text_tokens).float()  #10 768
    #---------------------------------------------------计算文本-----------------------------------------------------------#

    with torch.no_grad():
        # for batch_idx, (x, target) in enumerate(dataloader):
        for batch_idx, (x, target, _) in enumerate(dataloader):
            if device != 'cpu':
                x, target = x.to(device), target.to(dtype=torch.int64).to(device)
                # ipdb.set_trace()
            #------------------------resnet18和clip(PromptFL)----------------------------#
            # _, feats, out = model(x)
            #-------------------------vit和clip(使用分类头的情况)--------------------------#
            _,_,feature = model(x)
            feature=feature.to(device)
            model.proto_classifier.proto = model.proto_classifier.proto.to(device)
            out = torch.matmul(feature, model.proto_classifier.proto)
            _, pred_label = torch.max(out.data, 1)
            #--------------clip使用文本和图像计算相似度(FedCLIP)--------------------#
            # cls_img = model(x)   #64 768
            # if cls_txt.dim() == 2:
            #     cls_txt=cls_txt.unsqueeze(0).repeat((cls_img.size(0)), 1, 1)   #64 10 768 
            # elif cls_txt.dim() == 3:
            #     cls_txt_base = cls_txt[0]
            #     cls_txt = cls_txt_base.unsqueeze(0).expand(cls_img.size(0), -1, -1)            
            # cls_txt = cls_txt/cls_txt.norm(dim=-1, keepdim=True) 
            # cls_img = cls_img/cls_img.norm(dim=-1, keepdim=True) 
            # tem = model.clip.logit_scale.exp().to(device) 
            # out= (tem * torch.bmm(cls_img.unsqueeze(1), cls_txt.transpose(1, 2)).squeeze(1))  #64 10
            #------------------------------------------------------------#


            # 计算 batch 正确数
            batch_correct = (pred_label == target).sum().item()
            batch_total = target.size(0)
            batch_acc = batch_correct / batch_total
            print(
                f"[Batch {batch_idx}] "
                f"Pred: {pred_label[:10].detach().cpu().numpy()} | "
                f"GT: {target[:10].detach().cpu().numpy()} | "
                f"Batch Acc: {batch_acc:.4f}"
            )        

            # loss_collector.append(loss.item())
            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()

            if device == "cpu":
                pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                true_labels_list = np.append(true_labels_list, target.data.numpy())                             
            else:
                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
        # avg_loss = sum(loss_collector) / len(loss_collector)

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct / float(total), conf_matrix, None

    return correct / float(total), None


def compute_accuracy_etf_features(model,dataloader,get_confusion_matrix=False,device="cpu"):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    if isinstance(device, str):
        device = torch.device(device)

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    criterion = nn.CrossEntropyLoss().to(device)
    loss_collector = []

    model.to(device)

    with torch.no_grad():
        for batch_idx, (feats, target, _) in enumerate(dataloader):
            feats = feats.to(device)
            target = target.to(dtype=torch.int64).to(device)

            # feats 已经是 CLIP 冻结特征，所以这里只经过 MLP
            _, _, feature = model(feats)

            model.proto_classifier.proto = model.proto_classifier.proto.to(device)

            out = torch.matmul(feature, model.proto_classifier.proto)
            out = model.scaling_train * out

            loss = criterion(out, target)
            loss_collector.append(loss.item())

            _, pred_label = torch.max(out.data, 1)

            batch_correct = (pred_label == target).sum().item()
            batch_total = target.size(0)
            batch_acc = batch_correct / batch_total

            print(
                f"[Batch {batch_idx}] "
                f"Pred: {pred_label[:10].detach().cpu().numpy()} | "
                f"GT: {target[:10].detach().cpu().numpy()} | "
                f"Batch Acc: {batch_acc:.4f}"
            )

            total += target.size(0)
            correct += (pred_label == target).sum().item()

            pred_labels_list = np.append(
                pred_labels_list,
                pred_label.detach().cpu().numpy()
            )
            true_labels_list = np.append(
                true_labels_list,
                target.detach().cpu().numpy()
            )

    avg_loss = sum(loss_collector) / len(loss_collector)

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)
    else:
        conf_matrix = None

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct / float(total), conf_matrix, avg_loss

    return correct / float(total), avg_loss


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0):
    if dataset in ('cifar10', 'cifar100'):
        if dataset == 'cifar10':
            dl_obj = CIFAR10_truncated

            # normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            # transform_train = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Lambda(lambda x: F.pad(
            #         Variable(x.unsqueeze(0), requires_grad=False),
            #         (4, 4, 4, 4), mode='reflect').data.squeeze()),
            #     transforms.ToPILImage(),
            #     transforms.ColorJitter(brightness=noise_level),
            #     transforms.RandomCrop(32),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     normalize
            # ])
            # transform_test = transforms.Compose([
            #     transforms.ToTensor(),
            #     normalize])
            
            normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Lambda(lambda x: F.pad(Variable(x.unsqueeze(0), requires_grad=False),(4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.Resize([224,224]),
                # transforms.ColorJitter(brightness=noise_level),
                # transforms.RandomCrop(224),
                # transforms.RandomCrop(32),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),      
                transforms.ToPILImage(), 
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                normalize])

        elif dataset == 'cifar100':
            dl_obj = CIFAR100_truncated

            # normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
            #                                  std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
            # # transform_train = transforms.Compose([
            # #     transforms.RandomCrop(32),
            # #     transforms.RandomHorizontalFlip(),
            # #     transforms.ToTensor(),
            # #     normalize
            # # ])
            # transform_train = transforms.Compose([
            #     # transforms.ToPILImage(),
            #     transforms.RandomCrop(32, padding=4),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.RandomRotation(15),
            #     transforms.ToTensor(),
            #     normalize
            # ])
            # # data prep for test set
            # transform_test = transforms.Compose([
            #     transforms.ToTensor(),
            #     normalize])

            normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
            # transform_train = transforms.Compose([
            #     transforms.RandomCrop(32),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     normalize
            # ])
            transform_train = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.Resize(224),
                # transforms.RandomCrop(224, padding=4),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.ToTensor(),
                normalize])

        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=False, num_workers=4, pin_memory=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True)

    elif dataset == 'tinyimagenet':
        dl_obj = TinyImageNet_load
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(
                Variable(x.unsqueeze(0), requires_grad=False),
                (4, 4, 4, 4), mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=noise_level),
            transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        train_ds = dl_obj('../datasets/tiny-imagenet-200/', train=True, dataidxs=dataidxs, transform=transform_train)
        test_ds = dl_obj('../datasets/tiny-imagenet-200/', train=False, transform=transform_test)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True, num_workers=2, pin_memory=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=2, pin_memory=True)
    elif dataset == 'mnist':
        dl_obj = MNIST_truncated

        normalize = transforms.Normalize((0.1307,), (0.3081,))

        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()])

        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)
    
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True, num_workers=4, pin_memory=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True)
    elif dataset == 'fmnist':
        dl_obj = FashionMNIST_truncated

        normalize = transforms.Normalize((0.1307,), (0.3081,))

        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()])
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True, num_workers=4, pin_memory=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True)

        
    elif dataset == 'vireo172':
        dl_obj = Vireo172_truncated
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
#         transform_train = transforms.Compose([
#         transforms.ToTensor(),
#         normalize,
#          ])       
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        normalize,
         ])

        
        train_ds = dl_obj(dataidxs, transform_train, mode='train')
        test_ds = dl_obj(None, transform_test, mode='test')
        
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True, num_workers=4, pin_memory=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True)          
        
    elif dataset == 'food101':
        dl_obj = Food101_truncated
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
#         transform_train = transforms.Compose([
#         transforms.ToTensor(),
#         normalize,
#          ])       
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
         ])

        
        train_ds = dl_obj(dataidxs, transform_train, mode='train')
        test_ds = dl_obj(None, transform_test, mode='test')
        
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True, num_workers=2, pin_memory=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=2, pin_memory=True)   

    elif dataset == 'dtd':
        dl_obj = DTD
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
#         transform_train = transforms.Compose([
#         transforms.ToTensor(),
#         normalize,
#          ])       
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize,
         ])
        
        train_ds = dl_obj(dataidxs, transform_train, mode='train')
        test_ds = dl_obj(None, transform_test, mode='test')
        
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=False, num_workers=4, pin_memory=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True)   


    elif dataset == 'cal101':
        dl_obj = Caltech101
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
#         transform_train = transforms.Compose([
#         transforms.ToTensor(),
#         normalize,
#          ])       
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize,
         ])

        
        train_ds = dl_obj(dataidxs, transform_train, mode='train')
        test_ds = dl_obj(None, transform_test, mode='test')
        
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=False, num_workers=4, pin_memory=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True)   

    elif dataset == 'air100':
        dl_obj = Air100
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
#         transform_train = transforms.Compose([
#         transforms.ToTensor(),
#         normalize,
#          ])       
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize,
         ])

        train_ds = dl_obj(dataidxs, transform_train, mode='train')
        test_ds = dl_obj(None, transform_test, mode='test')
        
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=False, num_workers=4, pin_memory=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True)  

    elif dataset == 'cars196':
        dl_obj = Cars196
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
#         transform_train = transforms.Compose([
#         transforms.ToTensor(),
#         normalize,
#          ])       
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize,
         ])

        train_ds = dl_obj(dataidxs, transform_train, mode='train')
        test_ds = dl_obj(None, transform_test, mode='test')
        
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=False, num_workers=4, pin_memory=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True)  

    elif dataset == 'flowers102':
        dl_obj = Flowers102
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
#         transform_train = transforms.Compose([
#         transforms.ToTensor(),
#         normalize,
#          ])       
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize,
         ])

        train_ds = dl_obj(dataidxs, transform_train, mode='train')
        test_ds = dl_obj(None, transform_test, mode='test')
        
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=False, num_workers=4, pin_memory=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True)  

    elif dataset == 'ucf101':
        dl_obj = UCF101
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
#         transform_train = transforms.Compose([
#         transforms.ToTensor(),
#         normalize,
#          ])       
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize,
         ])

        train_ds = dl_obj(dataidxs, transform_train, mode='train')
        test_ds = dl_obj(None, transform_test, mode='test')
        
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=False, num_workers=4, pin_memory=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True)  
                
    return train_dl, test_dl, train_ds, test_ds

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval() 

# def gen_proto_local(net, dataloader, n_class=10):
#     feats = []
#     labels = []
#     net.eval()
#     net.apply(fix_bn)
#     net.cpu()
#     with torch.no_grad():
#         for batch_idx, (x, target) in enumerate(dataloader):
#             _, feat, _ = net(x)

#             feats.append(feat.cpu().numpy())
#             labels.append(target.numpy())

#     feats = np.concatenate(feats, 0)
#     labels = np.concatenate(labels, 0)
    
#     prototype = []
#     class_label = []
#     for i in range(n_class):
#         index = np.where(labels == i)[0]
#         if len(index) > 0:
#             class_label.append(int(i))
#             feature_classwise = feats[index]
#             prototype.append(torch.tensor(np.mean(feature_classwise, axis=0).reshape((1, -1))))
# #     ipdb.set_trace()
#     return torch.cat(prototype, dim=0).cuda(), torch.tensor(class_label).cuda()

def gen_proto_local(net, dataloader, n_class=10, device='cuda:0'):
    feats = []
    labels = []
    net.eval()
    net.apply(fix_bn)
    net.to(device)
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.to(device), target.to(device)
            _, feat, _ = net(x)

            feats.append(feat)
            labels.extend(target)

    feats = torch.cat(feats)
    labels = torch.tensor(labels)
#     ipdb.set_trace()
    prototype = []
    class_label = []
    for i in range(n_class):
        index = torch.nonzero(labels == i).reshape(-1)
        if len(index) > 0:
            class_label.append(int(i))
            feature_classwise = feats[index]
            prototype.append(torch.mean(feature_classwise, axis=0).reshape((1, -1)))
#     ipdb.set_trace()
    return torch.cat(prototype, dim=0), torch.tensor(class_label)


def dropout_proto_local(net, dataloader, args, n_class=10, device='cuda:0'):
    feats = []
    labels = []
    net.eval()
    net.apply(fix_bn)
    net.to(device)
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.to(device), target.to(device)
            _, feat, _ = net(x)

            feats.append(feat)
            labels.extend(target)

    feats = torch.cat(feats)
    labels = torch.tensor(labels)
#     ipdb.set_trace()
    prototype = []
    proto_label = []
    class_label = []
    class_idx = []
    for i in range(n_class):
        index = torch.nonzero(labels == i).reshape(-1)
        if len(index) > 0:
            class_idx.append(index)
            class_label.append(int(i))
        else:  
            class_idx.append([-1])
            class_label.append(-1)
            
    for i in range(n_class):
        if i in class_label: 
            if len(class_idx[i])>=10:
                for j in range(args.number):
                    idx = np.random.choice(np.arange(len(class_idx[i])), int(len(class_idx[i])*args.ratio))
                    feature_classwise = feats[class_idx[i][idx]]
                    prototype.append(torch.mean(feature_classwise, axis=0).reshape((1, -1)))
                    proto_label.append(int(i))
            else:
                proto_label.append(int(i))
                feature_classwise = feats[class_idx[i]]
                prototype.append(torch.mean(feature_classwise, axis=0).reshape((1, -1)))            
#     ipdb.set_trace()
    return torch.cat(prototype, dim=0), torch.tensor(proto_label)


def dropout_proto_local_v2(net, dataloader, args, n_class=10, device='cuda:0'):
    feats = []
    labels = []
    net.eval()
    net.apply(fix_bn)
    net.to(device)
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.to(device), target.to(device)
            _, feat, _ = net(x)

            feats.append(feat)
            labels.extend(target)

    feats = torch.cat(feats)
    labels = torch.tensor(labels)
#     ipdb.set_trace()
    prototype = []
    proto_label = []
    class_label = []
    class_idx = []
    for i in range(n_class):
        index = torch.nonzero(labels == i).reshape(-1)
        if len(index) > 0:
            class_idx.append(index)
            class_label.append(int(i))
        else:  
            class_idx.append([-1])
            class_label.append(-1)
            
    for i in range(n_class):
        if i in class_label: 
            for j in range(len(class_idx[i])):
                idx = np.random.choice(np.arange(len(class_idx[i])), int(len(class_idx[i])*args.ratio))
                feature_classwise = feats[class_idx[i][idx]]
                prototype.append(torch.mean(feature_classwise, axis=0).reshape((1, -1)))
                proto_label.append(int(i))
           
    return torch.cat(prototype, dim=0), torch.tensor(proto_label)


def noisy_proto_local(net, dataloader, args, n_class=10, device='cuda:0'):
    feats = []
    labels = []
    net.eval()
    net.apply(fix_bn)
    net.to(device)
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.to(device), target.to(device)
            _, feat, _ = net(x)

            feats.append(feat)
            labels.extend(target)

    feats = torch.cat(feats)
    labels = torch.tensor(labels)
#     ipdb.set_trace()
    prototype = []
    proto_label = []
    class_label = []
    class_idx = []
    var_set = []
    for i in range(n_class):
        index = torch.nonzero(labels == i).reshape(-1)
        if len(index) > 0:
            class_label.append(int(i))
            feature_classwise = feats[index]
            
            v_i = torch.var(feature_classwise, axis=0)
            var_set.append(v_i) 
                     
            prototype.append(torch.mean(feature_classwise, axis=0).reshape((1, -1)))
            
            
    for i in range(n_class):
        if i in class_label: 
            for j in range(len(class_idx[i])):
                idx = np.random.choice(np.arange(len(class_idx[i])), int(len(class_idx[i])*args.ratio))
                feature_classwise = feats[class_idx[i][idx]]
                prototype.append(torch.mean(feature_classwise, axis=0).reshape((1, -1)))
                proto_label.append(int(i))
           
    return torch.cat(prototype, dim=0), torch.tensor(proto_label)






def gen_proto_global(feats, labels, n_classes):
    local_proto = []
    local_labels = []
    for i in range(n_classes):
#         ipdb.set_trace()
        c_i = torch.nonzero(labels == i).reshape(-1)
        proto_i = torch.sum(feats[c_i], dim=0) / len(c_i)
        local_proto.append(proto_i.reshape(1, -1))
        local_labels.append(i)
    
    return torch.cat(local_proto, dim=0), torch.tensor(local_labels)




def detached_statedict(state_dict):
    """
    Returns copied state_dict, with clone-detached tensors
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k] = v.clone().detach()
        new_state_dict[k].requires_grad = v.requires_grad

    return new_state_dict

def set_client_from_params(mdl, params):
    dict_param = copy.deepcopy(dict(mdl.named_parameters()))
    ipdb.set_trace()
    idx = 0
    for name, param in mdl.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(torch.tensor(params[idx:idx+length].reshape(weights.shape)))
        idx += length
    
    mdl.load_state_dict(dict_param)    
    return mdl

def get_mdl_params(model_list, n_par=None):
    
    if n_par==None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))
    
    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)


def compute_cluster_center_and_average_distance(cluster_features):
    """
    计算簇中心和簇内样本的平均距离。
    
    参数：
    cluster_features (np.ndarray): 包含簇内特征的NumPy数组,每行代表一个样本，每列代表一个特征。
    
    返回：
    cluster_center (np.ndarray): 簇中心，即特征的平均值。
    average_distance_to_center (float): 簇内样本的平均距离。
    """
    if cluster_features.shape[0] > 1:
    # 计算簇中心：计算特征的平均值
        cluster_center = np.mean(cluster_features, axis=0)
        # 计算簇内样本的平均距离：计算每个样本与簇中心的距离，并取平均值
        distances_to_center = np.linalg.norm(cluster_features - cluster_center, axis=1)
        average_distance_to_center = np.mean(distances_to_center)
    else:
        cluster_center = cluster_features
        average_distance_to_center = 1e-5

    return cluster_center, average_distance_to_center


# def compute_cluster_center_and_average_distance(cluster_features):
#     """
#     计算簇中心和簇内样本的平均距离（全 PyTorch 版本）。
    
#     参数：
#     cluster_features (torch.Tensor): [n_samples, D]，每行代表一个样本，每列代表特征。
    
#     返回：
#     cluster_center (torch.Tensor): [D] 簇中心。
#     average_distance_to_center (float): 簇内样本到中心的平均欧氏距离。
#     """
#     if cluster_features.shape[0] > 1:
#         # 计算簇中心
#         cluster_center = cluster_features.mean(dim=0)  # PyTorch 的 mean
#         # 计算每个样本到簇中心的距离
#         distances_to_center = torch.norm(cluster_features - cluster_center, dim=1)
#         average_distance_to_center = distances_to_center.mean().item()  # 转成 float
#     else:
#         cluster_center = cluster_features[0]  # 单样本
#         average_distance_to_center = 1e-5

#     return cluster_center, average_distance_to_center

    



    