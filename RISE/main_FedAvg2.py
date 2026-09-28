import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import os
import copy
import datetime
import random
# os.environ['CUDA_VISIBLE_DEVICES']='0'
from RGA import *
from resnet_base import *
from utils1 import *
from utils1 import get_dataloader
from utils1 import mkdirs
from utils1 import partition_data
from loss import *
import torch.nn.functional as F
import ipdb
import clip 
from peft import LoraConfig, get_peft_model
from vit import VisionTransformer
from Resnet8_256 import ResNet_cifar
from classes_names import name_classes
from model1 import *
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("tifffile").setLevel(logging.WARNING)
from collections import defaultdict
import math
from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='simple-cnn', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='domain_pacs', help='dataset used for training')
    parser.add_argument('--device', type=str, default='cuda:3', help='The device to run the program')
    parser.add_argument('--domain_id', type=int, default=[3])

    parser.add_argument('--beta', type=float, default=0.1,help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--n_parties', type=int, default=15, help='number of workers in a distributed cluster')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--comm_round', type=int, default=30, help='number of maximum communication roun')
    
    parser.add_argument('--use_mlp',default=True,help='whether to enable MLP projection head')
    parser.add_argument('--use_residual',default=False,help='whether to enable MLP projection head')
    parser.add_argument('--use_spectral_loss',default=False,help='')

    parser.add_argument('--method', type=int, default=3, help='')
    parser.add_argument('--mu', type=int, default=0.4, help='方法一的损失系数')
    parser.add_argument('--nu', type=int, default=2, help='方法二的损失系数')
    parser.add_argument('--lu', type=int, default=1, help='分类损失系数')
    parser.add_argument('--te', type=int, default=100, help='方法二MSE除以的系数')

    parser.add_argument('--use_prototype_pool',type=bool,default=False,help='是否使用原型池 (True/False)')
    parser.add_argument('--prototype_pool_method',type=int,default=9,help='原型池构建方法编号:1/2/3/4/.....')
    parser.add_argument('--num_samples_per_proto',type=int, default=10,help='每个原型生成的伪样本数量')
    parser.add_argument('--num_samples_per_pair',type=int, default=2,help='每两个原型之间生成的伪样本数量')
    parser.add_argument('--num_mixup_samples_per_class',type=int, default=25,help='每个类别生成的伪样本数量')

    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.1)')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--logdir', type=str, required=False, default="./logs_new_new_新", help='Log directory path')
    parser.add_argument('--feat_logdir',type=str,required=False,default="./feat_logs_new_new_experiment9",help='Feature save directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models_new_new_新", help='Model directory path')
    parser.add_argument('--train_first_transformer_block', type=bool, default=False, help='是否只训练第一个 Transformer block (True/False)')

    # parser.add_argument('--dataset', type=str, default='vireo172', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='noniid', help='the data partitioning strategy')
    parser.add_argument('--alg', type=str, default='fedavg',help='communication strategy: fedavg/fedprox')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    # parser.add_argument('--logdir', type=str, required=False, default="./logs/cifar10/clip加图像适配器并且使用其输出当做注意力权重_和文本进行交叉熵损失", help='Log directory path')
    # parser.add_argument('--logdir', type=str, required=False, default="./logs/cifar10/clip加图像适配器并且使用其输出当做特征_和文本进行交叉熵损失", help='Log directory path')
    # parser.add_argument('--logdir', type=str, required=False, default="./logs/cifar10/vit加图像适配器并且使用其输出当做特征", help='Log directory path')

    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    # parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    # parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    # parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    # parser.add_argument('--local_max_epoch', type=int, default=100,help='the number of epoch for local optimal training')
    # parser.add_argument('--model_buffer_size', type=int, default=1,help='store how many previous models for contrastive loss')
    # parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    # parser.add_argument('--sample_fraction', type=float, default=1, help='how many clients are sampled in each round')
    parser.add_argument('--sample_fraction', type=float, default=1, help='how many clients are sampled in each round')
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    parser.add_argument('--load_model_round', type=int, default=None, help='how many rounds have executed for the loaded model')
    parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')
    parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model')
    parser.add_argument('--loss', type=str, default='contrastive')
    parser.add_argument('--save_model', type=int, default=0)
    parser.add_argument('--use_project_head', type=int, default=1)
    parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')
    parser.add_argument('--ckpt_root',type=str,default='./models/cifar10/simple-cnn',help='Root directory to save model checkpoints')
    parser.add_argument('--save_rounds',type=int,nargs='+',default=[1],help='Communication rounds to save client and global models, e.g., --save_rounds 50 100 200')

    args = parser.parse_args()
    return args

args = get_args()
device = torch.device(args.device)

import torch
import torch.nn.functional as F
from collections import deque

class ClassMemoryQueue:
    def __init__(self, num_classes, feat_dim, queue_size=256, device="cuda"):
        self.num_classes = num_classes
        self.queue_size = queue_size
        self.device = device 
        self.queues = {c: deque(maxlen=queue_size) for c in range(num_classes)}

    def enqueue(self, features, labels):
        """
        features: [B, D]  (x_pre)
        labels:   [B]
        """
        features = features.detach().cpu()
        labels = labels.detach().cpu()

        for feat, label in zip(features, labels):
            self.queues[int(label.item())].append(feat)

    def get_class_feats(self, cls):
        if len(self.queues[cls]) == 0:
            return None
        feats = torch.stack(list(self.queues[cls]), dim=0)
        return feats.to(self.device)


    def get_class_proto(self, cls):
        """
        返回某个类别的原型: [D]
        """
        feats = self.get_class_feats(cls)
        if feats is None:
            return None
        proto = feats.mean(dim=0)   # [D]
        return proto

    def get_all_prototypes(self, min_samples=1):
        """
        返回所有类别的原型字典
        return:
            proto_dict[class_id] = [D]
        """
        # {
        #     0: tensor([512]),
        #     1: tensor([512]),
        #     3: tensor([512]),
        #     7: tensor([512]),
        #     ...
        # }
        proto_dict = {}
        for cls in range(self.num_classes):
            if len(self.queues[cls]) < min_samples:
                continue
            feats = self.get_class_feats(cls)   # [N_c, D]
            proto = feats.mean(dim=0)           # [D]
            proto_dict[cls] = proto

        return proto_dict

def cosine_sim(a, b, eps=1e-8):
    a = a / (a.norm() + eps)
    b = b / (b.norm() + eps)
    return torch.dot(a, b)

def weighted_mean(vectors, weights):
    w = torch.tensor(weights, device=vectors[0].device, dtype=torch.float)
    w = w / (w.sum() + 1e-8)
    return sum(w[i] * vectors[i] for i in range(len(vectors)))

class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, feats, labels):
        self.feats = feats
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.feats[idx], self.labels[idx], -1

def spectral_loss(feat_pre, feat_mlp):
    """
    feat_pre : [B, D] 预训练特征
    feat_mlp : [B, D] 适配后特征
    """

    # ---- 计算协方差 ----#
    feat_pre = feat_pre - feat_pre.mean(dim=0, keepdim=True)
    feat_mlp = feat_mlp - feat_mlp.mean(dim=0, keepdim=True)

    cov_pre = torch.matmul(feat_pre.T, feat_pre) / feat_pre.shape[0]
    cov_mlp = torch.matmul(feat_mlp.T, feat_mlp) / feat_mlp.shape[0]

    # ---- 计算特征值谱 ----#
    # 使用 torch.linalg.eigvalsh 计算协方差矩阵的特征值
    # 特征值描述了数据在每个特征方向上的变异程度，较大的特征值表示该方向上的变化较大，较小的特征值表示变化较小
    eig_pre = torch.linalg.eigvalsh(cov_pre)
    eig_mlp = torch.linalg.eigvalsh(cov_mlp)

    # ---- 只保留主能量谱（更稳定）----
    # 主能量谱：通过 torch.topk 函数保留协方差特征值中的 前 k 个最大特征值，k 取值为 64 或者特征值的数量（取较小的值）
    k = min(128, len(eig_pre))
    eig_pre = torch.topk(eig_pre, k).values
    eig_mlp = torch.topk(eig_mlp, k).values

    # 预训练特征和适配后特征在主成分方向上的方差差异的指标
    # 这意味着我们在训练时会关注特征值谱的差异，尽量保持预训练特征和适配后特征的主要变化方向一致，从而避免适配层（MLP）对特征的过度修改
    return torch.mean((eig_pre - eig_mlp) ** 2)

#--------------------------------------------------------不加文本-----------------------------------------------------#
class VisualNet(nn.Module):
    def __init__(self, clip, num_cls, dim_feature,args):
        super(VisualNet, self).__init__()
        self.classifier = nn.Linear(dim_feature, num_cls)
        self._initialize_weights()
        self.clip = clip
        self.norm = nn.LayerNorm(dim_feature)
        self.relu = nn.ReLU()
        if args.use_mlp:
            self.mlp=nn.Sequential(nn.Linear(512, 256),nn.ReLU(),nn.Linear(256, 512)).to(device)

    def forward(self, x):
        # image_embeds = self.clip.encode_image(x)
        # ipdb.set_trace()
        # image_embeds.float()
        # cls_img=image_embeds[:,0,:]

        # cls_img = self.clip.encode_image(x)
        # output_img = self.classifier(cls_img) 
        # return None,output_img
        # ipdb.set_trace()
        with torch.no_grad():
            cls_img = self.clip.encode_image(x)
        if args.use_mlp:
            if args.use_residual:
                x=self.norm(cls_img)
                residual = self.mlp(x)
                cls_img = x + residual
            else:
                cls_img=self.norm(cls_img)
                cls_img=self.mlp(cls_img)

        output_img = self.classifier(cls_img) 
        return None,None,output_img

        # cls_img = self.clip.encode_image(x)
        # cls_img_att= self.img_adap(cls_img)  
        # cls_img=torch.mul(cls_img_att, cls_img)
        # output_img = self.classifier(cls_img) 
        # return None,cls_img
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

class VisualNet_mlp(nn.Module):
    def __init__(self, clip, num_cls, dim_feature,args):
        super(VisualNet_mlp, self).__init__()
        self.classifier = nn.Linear(dim_feature, num_cls)
        self._initialize_weights()
        self.norm = nn.LayerNorm(dim_feature)
        # self.clip = clip
        self.relu = nn.ReLU()
        if args.use_mlp:
            self.mlp=nn.Sequential(nn.Linear(512, 256),nn.ReLU(),nn.Linear(256, 512)).to(device)

    def forward(self, cls_img):
        # image_embeds = self.clip.encode_image(x)
        # ipdb.set_trace()
        # image_embeds.float()
        # cls_img=image_embeds[:,0,:]

        # cls_img = self.clip.encode_image(x)
        # output_img = self.classifier(cls_img) 
        # return None,output_img
        # ipdb.set_trace()
        pre_img=cls_img
        if args.use_mlp:
            if args.use_residual:
                x=self.norm(cls_img)
                residual = self.mlp(x)
                cls_img = x + residual
            else:
                cls_img=self.norm(cls_img)
                cls_img=self.mlp(cls_img)

        output_img = self.classifier(cls_img) 
        return pre_img,cls_img,output_img

        # cls_img = self.clip.encode_image(x)
        # cls_img_att= self.img_adap(cls_img)  
        # cls_img=torch.mul(cls_img_att, cls_img)
        # output_img = self.classifier(cls_img) 
        # return None,cls_img
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

clip_model, preprocess = clip.load('ViT-B/16',device="cuda",jit=False)
clip_model = clip_model.float()

#----------------------------------------------------------------联邦学习给每个客户端创建一个模型-----------------------------------------------------------------------#
def init_nets(net_configs, n_parties, args, n_classes, device='cuda:0'):
    nets = {net_i: None for net_i in range(n_parties)}

    for net_i in range(n_parties):
        # ipdb.set_trace()
        net = VisualNet(clip_model, n_classes, 512,args)
        #--------------------打印参数------------------------#
        for name, param in net.named_parameters():
            param.requires_grad = False

        # for name, param in net.named_parameters():
        #     if "classifier" in name:
        #         param.requires_grad = True

        for name, param in net.named_parameters():
            if name.startswith(("classifier", "img_adap", "mlp")):
                param.requires_grad = True

        # for name, param in net.named_parameters():
        #     if args.train_first_transformer_block:
        #         if name.startswith(("classifier", "img_adap", "mlp")) or "clip.visual.transformer.resblocks.0" in name:
        #             param.requires_grad = True
        #         else:
        #             param.requires_grad = False
        #     else:
        #         if name.startswith(("classifier", "img_adap", "mlp")) or "clip.visual.transformer.resblocks.0" in name or "clip.visual.transformer.resblocks.1" in name:
        #             param.requires_grad = True
        #         else:
        #             param.requires_grad = False
        
        # for name, param in net.named_parameters():
        #     if "clip" in name:
        #         param.requires_grad = False

        nets[net_i] = net.to(device)

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type
    #nets：所有客户端模型
    #model_meta_data：每一层参数形状
    #layer_type：每一层的名字

def init_nets_mlp(net_configs, n_parties, args, n_classes, device='cuda:0'):
    nets = {net_i: None for net_i in range(n_parties)}

    for net_i in range(n_parties):
        # ipdb.set_trace()
        net = VisualNet_mlp(clip_model, n_classes, 512,args)        
        #--------------------打印参数------------------------#
        # for name, param in net.named_parameters():
        #     param.requires_grad = False

        # for name, param in net.named_parameters():
        #     if "classifier" in name:
        #         param.requires_grad = True

        for name, param in net.named_parameters():
            if name.startswith(("classifier", "img_adap", "mlp")):
                param.requires_grad = True

        # for name, param in net.named_parameters():
        #     if args.train_first_transformer_block:
        #         if name.startswith(("classifier", "img_adap", "mlp")) or "clip.visual.transformer.resblocks.0" in name:
        #             param.requires_grad = True
        #         else:
        #             param.requires_grad = False
        #     else:
        #         if name.startswith(("classifier", "img_adap", "mlp")) or "clip.visual.transformer.resblocks.0" in name or "clip.visual.transformer.resblocks.1" in name:
        #             param.requires_grad = True
        #         else:
        #             param.requires_grad = False
        
        # for name, param in net.named_parameters():
        #     if "clip" in name:
        #         param.requires_grad = False
        
        nets[net_i] = net.to(device)

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type
    #nets：所有客户端模型
    #model_meta_data：每一层参数形状
    #layer_type：每一层的名字
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# def exp_map_to_poincare(x, scale=0.05):
#     """
#     x: [N, D] Euclidean features
#     scale: 控制投影到球内的距离
#     """
#     norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
#     scaled_norm = scale * norm_x
#     factor = torch.tanh(scaled_norm) / (norm_x)
#     return factor * x

    
# def poincare_distance(u, v, eps=1e-5):
#     """
#     计算 Poincare 球上两个点的测地距离
#     u, v: [B, D] tensor
#     返回: [B] 每个样本的测地距离
#     """
#     u=u.to(device)
#     v=v.to(device)
#     norm_u = u.norm(p=2, dim=-1)**2
#     norm_v = v.norm(p=2, dim=-1)**2
#     diff = (u - v).norm(p=2, dim=-1)**2
    
#     denom = (1 - norm_u) * (1 - norm_v) + eps  # 防止除零
#     arg = 1 + 2 * diff / denom
#     arg = torch.clamp(arg, min=1 + eps)        # arcosh定义域: [1, +∞)
    
#     return torch.acosh(arg)


# def get_fused_prototypes(x_mlp, target, memory, alpha=0.5, min_memory_samples=1):
#     """
#     x_mlp: [B, D] 当前 batch 特征
#     target: [B] 当前 batch label
#     memory: ClassMemoryQueue 对象
#     alpha: batch 原型权重
#     min_memory_samples: memory 中类别至少有多少样本才算有效
#     return:
#         fused_proto_dict[class_id] = tensor[D]
#     """
#     device = x_mlp.device
#     batch_proto_dict = {}

#     # --- 先计算 batch 中每个类别的原型 ---
#     unique_classes = target.unique().tolist()
#     for cls in unique_classes:
#         mask = (target == cls)
#         cls_feats = x_mlp[mask]           # [N_c, D]
#         batch_proto = cls_feats.mean(dim=0)  # [D]
#         batch_proto_dict[cls] = batch_proto

#     # --- 获取 memory 中各类别原型 ---
#     memory_proto_dict = memory.get_all_prototypes(min_samples=min_memory_samples)

#     # --- 融合 batch + memory 原型 ---
#     fused_proto_dict = {}
#     all_classes = set(unique_classes) | set(memory_proto_dict.keys())

#     for cls in all_classes:
#         batch_proto = batch_proto_dict.get(cls, None)
#         memory_proto = memory_proto_dict.get(cls, None)

#         if batch_proto is not None and memory_proto is not None:
#             fused_proto = alpha * batch_proto + (1 - alpha) * memory_proto
#         elif batch_proto is not None:
#             fused_proto = batch_proto
#         elif memory_proto is not None:
#             fused_proto = memory_proto
#         else:
#             continue  # 这个类别既不在batch也不在memory，理论上不会发生

#         fused_proto_dict[cls] = fused_proto.to(device)

#     return fused_proto_dict

# def hyperbolic_distance_preservation_loss(fused_proto_dict, pre_global_proto_by_class, scale=0.05, device="cuda"):
#     """
#     Args:
#         fused_proto_dict: dict[class_id] -> [D] 训练融合原型
#         pre_global_proto_by_class: dict[class_id] -> [D] 预训练原型
#     Returns:
#         loss: scalar tensor, 保持类别间双曲测地距离
#     """
#     # 找出交集类别，保证矩阵对应
#     classes = sorted(list(set(fused_proto_dict.keys()) & set(pre_global_proto_by_class.keys())))
#     if len(classes) == 0:
#         return torch.tensor(0.0, device=device, requires_grad=True)

#     # 堆叠为矩阵 [C, D]
#     fused_mat = torch.stack([fused_proto_dict[c] for c in classes], dim=0).to(device)   # [C, D]
#     pre_mat = torch.stack([pre_global_proto_by_class[c].to(device) for c in classes], dim=0)  # [C, D]

#     # 映射到双曲空间
#     fused_hyp = exp_map_to_poincare(fused_mat, scale=scale)   # [C, D]
#     pre_hyp  = exp_map_to_poincare(pre_mat, scale=scale)     # [C, D]

#     # 计算测地距离矩阵
#     def poincare_dist_matrix(proto_hyp, eps=1e-5):
#         u = proto_hyp.unsqueeze(1)  # [C,1,D]
#         v = proto_hyp.unsqueeze(0)  # [1,C,D]
#         norm_u = (u ** 2).sum(dim=-1)  # [C,1]
#         norm_v = (v ** 2).sum(dim=-1)  # [1,C]
#         diff = ((u - v) ** 2).sum(dim=-1)  # [C,C]
#         denom = (1 - norm_u) * (1 - norm_v) + eps
#         arg = 1 + 2 * diff / denom
#         arg = torch.clamp(arg, min=1+eps)
#         dist_mat = torch.acosh(arg)
#         dist_mat.fill_diagonal_(0.0)
#         return dist_mat

#     dist_fused = poincare_dist_matrix(fused_hyp)
#     dist_pre   = poincare_dist_matrix(pre_hyp)

#     # MSE loss 保持训练原型间距离与预训练原型一致
#     loss = ((dist_fused - dist_pre) ** 2).mean()
#     return loss


def get_fused_prototypes(x_mlp, target, memory, alpha=0.5, min_memory_samples=1):
    """
    x_mlp: [B, D] 当前 batch 特征
    target: [B] 当前 batch label
    memory: ClassMemoryQueue 对象
    alpha: batch 原型权重
    min_memory_samples: memory 中类别至少有多少样本才算有效
    return:
        fused_proto_dict[class_id] = tensor[D]
    """
    device = x_mlp.device
    batch_proto_dict = {}

    # --- 先计算 batch 中每个类别的原型 ---
    unique_classes = target.unique().tolist()
    for cls in unique_classes:
        mask = (target == cls)
        cls_feats = x_mlp[mask]           # [N_c, D]
        batch_proto = cls_feats.mean(dim=0)  # [D]
        batch_proto_dict[cls] = batch_proto

    # --- 获取 memory 中各类别原型 ---
    memory_proto_dict = memory.get_all_prototypes(min_samples=min_memory_samples)

    # --- 融合 batch + memory 原型 ---
    fused_proto_dict = {}
    all_classes = set(unique_classes) | set(memory_proto_dict.keys())

    for cls in all_classes:
        batch_proto = batch_proto_dict.get(cls, None)
        memory_proto = memory_proto_dict.get(cls, None)

        if batch_proto is not None and memory_proto is not None:
            fused_proto = alpha * batch_proto + (1 - alpha) * memory_proto
        elif batch_proto is not None:
            fused_proto = batch_proto
        elif memory_proto is not None:
            fused_proto = memory_proto
        else:
            continue  # 这个类别既不在batch也不在memory，理论上不会发生

        fused_proto_dict[cls] = fused_proto.to(device)

    return fused_proto_dict


def exp_map_to_lorentz_space(x, c1=1.0, eps=1e-8):
    # ipdb.set_trace()
    # x = x
    # norm_x = torch.norm(x, p=2, dim=-1, keepdim=True).clamp_min(eps)
    # sqrt_c = torch.sqrt(torch.as_tensor(c, device=x.device, dtype=x.dtype))
    # factor = torch.sinh(sqrt_c * norm_x) / (sqrt_c * norm_x)

    # ipdb.set_trace()
    rc_xnorm = c1**0.5 * torch.norm(x, dim=-1, keepdim=True)
    # Ensure numerical stability in sinh by clamping input.
    sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2**15))
    _output = torch.sinh(sinh_input) * x / torch.clamp(rc_xnorm, min=eps)
    return _output

def lorentz_pairwise_inner(x, y, c1=1.0, eps=1e-8):
    """
    x: [N, D],Lorentz模型中的 space part
    y: [M, D],Lorentz模型中的 space part
    c: 正数，实际双曲曲率为 -c
    """
    c1 = torch.as_tensor(c1, device=x.device, dtype=x.dtype)

    x_time = torch.sqrt(1.0 / c1 + torch.sum(x ** 2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1.0 / c1 + torch.sum(y ** 2, dim=-1, keepdim=True))

    # Lorentz inner product: <x,y>_L = x_space^T y_space - x_time * y_time
    return x @ y.transpose(0, 1) - x_time @ y_time.transpose(0, 1)


def lorentz_dist_matrix(proto_hyp, c1=1.0, eps=1e-6):
    """
    proto_hyp: [C, D]，这里是 exp_map_to_lorentz_space 的输出（space part）
    返回: [C, C] Lorentz 双曲面上的测地距离矩阵
    """
    c1 = torch.as_tensor(c1, device=proto_hyp.device, dtype=proto_hyp.dtype)

    inner = lorentz_pairwise_inner(proto_hyp, proto_hyp, c1=c1)   # [C, C]
    arg = -c1 * inner
    arg = torch.clamp(arg, min=1.0 + eps)

    dist_mat = torch.acosh(arg) / torch.sqrt(c1)
    dist_mat.fill_diagonal_(0.0)
    return dist_mat

def hyperbolic_distance_preservation_loss(fused_proto_dict, pre_global_proto_by_class, c1=1.0,device="cuda"):
    """
    Args:
        fused_proto_dict: dict[class_id] -> [D] 训练融合原型
        pre_global_proto_by_class: dict[class_id] -> [D] 预训练原型
    Returns:
        loss: scalar tensor, 保持类别间双曲测地距离
    """
    # 找出交集类别，保证矩阵对应
    classes = sorted(list(set(fused_proto_dict.keys()) & set(pre_global_proto_by_class.keys())))
    if len(classes) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 堆叠为矩阵 [C, D]
    fused_mat = torch.stack([fused_proto_dict[c] for c in classes], dim=0).to(device)   # [C, D]
    pre_mat = torch.stack([pre_global_proto_by_class[c].to(device) for c in classes], dim=0)  # [C, D]

    # 映射到双曲空间
    pre_hyp  = exp_map_to_lorentz_space(pre_mat,c1=c1)     # [C, D]
    fused_hyp = exp_map_to_lorentz_space(fused_mat,c1=c1)   # [C, D]
    # print(fused_hyp.norm(dim=1)) 
    # print(pre_hyp.norm(dim=1))  

    dist_fused = lorentz_dist_matrix(fused_hyp,c1=c1)
    dist_pre   = lorentz_dist_matrix(pre_hyp,c1=c1)

    # print("适配训练原型的 Lorentz 测地距离矩阵:")
    # print(dist_fused.detach().cpu())

    # print("\n预训练原型的 Lorentz 测地距离矩阵:")
    # print(dist_pre.detach().cpu())

    # MSE loss 保持训练原型间距离与预训练原型一致
    loss = ((dist_fused - dist_pre) ** 2).mean()/args.te

    # loss = F.smooth_l1_loss(dist_fused, dist_pre)
    return loss


def train_net_fedavg(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args,device="cuda:0", pre_global_proto_by_class=None):
    global_net.to(device)
    # net = nn.DataParallel(net)
    net.to(device)

    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    # logger.info('n_test: %d' % len(test_dataloader))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0

    # logger.info(f"=== Net {net_id} Trainable Parameters ===")
    # for name, param in net.named_parameters():
    #     logger.info(f"{name} -> requires_grad: {param.requires_grad}")
    # logger.info("=========================================")

    # ipdb.set_trace()

    memory = ClassMemoryQueue(num_classes=n_classes,feat_dim=512,queue_size=30,device=device)

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target,_) in enumerate(train_dataloader):
            # ipdb.set_trace()
            x, target = x.to(device), target.to(device)
            if args.dataset == 'pmnist':
                target = target.reshape(-1)
            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()
            # ipdb.set_trace()
            # out = net(x)
            # out4, out_feat,out = net(x)
            #out4.shape=64 512     out_feat.shape=64 512     out.shape=64 10
            x_pre,x_mlp,out = net(x)

            #========================双曲空间的类别结构对齐====================================#
            # fused_proto_dict = get_fused_prototypes(x_mlp, target, memory, alpha=0.5)
            # # # fused_proto_dict 里每个类别都是 [D] 的 tensor，可直接用来计算测地距离 loss
            # # for cls, proto in fused_proto_dict.items():
            # #     print(f"Class {cls} fused prototype shape: {proto.shape}")
            # loss_proto = hyperbolic_distance_preservation_loss(fused_proto_dict, pre_global_proto_by_class, scale=0.05)
            # # loss_proto = euclidean_distance_preservation_loss(fused_proto_dict, pre_global_proto_by_class,device=device)
            # # loss_proto = relative_direction_loss_with_near_neighbors(fused_proto_dict, pre_global_proto_by_class, scale=0.05, device=device)

            # loss_cls = criterion(out, target)
            # loss_cls=loss_cls.to(device)
            # loss_proto=loss_proto.to(device)
            # loss=loss_cls+2*loss_proto

            # print(f"Batch {batch_idx}: loss_proto={loss_proto.item():.6f}")
            #================================================================================#

            #===================================最初的方法==============================================#
            if args.method==0:
                loss_cls = criterion(out, target)
                loss_cls=loss_cls.to(device)
                loss=loss_cls
            #===============================================================================#

            #========================方法一：+方法二：====================================#
            if args.method==3:
                fused_proto_dict = get_fused_prototypes(x_mlp, target, memory, alpha=0.5)
                curv1=0.5
                loss_proto = hyperbolic_distance_preservation_loss(fused_proto_dict, pre_global_proto_by_class,c1=curv1,device=device)
                loss_proto=loss_proto.to(device)

                # print_euclidean_distances(fused_proto_dict, pre_global_proto_by_class, device=device)

                loss_cls = criterion(out, target)
                loss_cls=loss_cls.to(device)

                loss_local = spectral_loss(x_pre, x_mlp)
                loss_local=loss_local.to(device)
            
                loss=args.lu*loss_cls+args.mu*loss_local+args.nu*loss_proto
                # print(f"Batch {batch_idx}: curv1={curv1.item():.6f}")
                print(f"Batch {batch_idx}: loss_proto={loss_proto.item():.6f}")
                print(f"Batch {batch_idx}: loss_local={loss_local.item():.6f}")
            #================================================================================#



            # # === 新增：计算预测类别 ===
            # pred_label = torch.argmax(out, dim=1)
            # # === 新增：计算 batch 准确率 ===
            # correct = (pred_label == target).sum().item()
            # batch_size = target.size(0)
            # batch_acc = correct / batch_size
            # # 打印（建议只打前几个样本，避免刷屏）
            # print(f"[Net {net_id} | Epoch {epoch} | Batch {batch_idx}] "
            #     f"Acc: {batch_acc:.4f} ({correct}/{batch_size}) | "
            #     f"Pred: {pred_label[:10].detach().cpu().numpy()} | "
            #     f"GT: {target[:10].detach().cpu().numpy()}")

            #=============================================#
            # if args.use_spectral_loss:
            #     loss_local = spectral_loss(x_pre, x_mlp)
            # else:
            #     loss_local= torch.tensor(0.0, dtype=x_pre.dtype, device=x_pre.device)
            # loss_cls = criterion(out, target)
            # loss=loss_cls+0.4*loss_local
            #=============================================#

            # # --- 打印每个 batch 的 loss ---#
            # print(f"Batch {batch_idx}: loss_cls={loss_cls.item():.6f}, loss_local={loss_local.item():.6f},loss={loss.item():.6f}")

            loss.backward()
            optimizer.step()
            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    net.to('cpu')
    logger.info(' ** Training complete **')
    return 0, 0

#------------------------------------------------------------------让每个客户端在自己的本地数据上训练-------------------------------------------------------------------------#
def local_train_net(nets, args, net_dataidx_map, train_dl=None, 
    test_dl=None, global_model = None, prev_model_pool = None, 
    prev_protos_pool=None, prev_protos_label_pool=None, server_c = None, 
    clients_c = None, round=None, device="cuda:0",proto_bank_by_class=None,
    proto_bank_by_client=None,missing_classes_per_client=None,pseudo_pool_by_client=None,
    pre_global_proto_by_class=None,
):
    avg_acc = 0.0
    acc_list = []
    # ipdb.set_trace()
    if global_model:
        global_model.cuda()
    if server_c:
        server_c.cuda()
        server_c_collector = list(server_c.cuda().parameters())
        new_server_c_collector = copy.deepcopy(server_c_collector)
        

    for net_id, net in nets.items():
        # ipdb.set_trace()
        real_feats  = proto_bank_by_client[net_id]["feats"]
        real_labels = proto_bank_by_client[net_id]["labels"]
        logger.info("Training network %s. n_training: %d" % (str(net_id), len(real_labels)))

        if pseudo_pool_by_client is not None and net_id in pseudo_pool_by_client:
            pseudo_feats  = pseudo_pool_by_client[net_id]["feats"]
            pseudo_labels = pseudo_pool_by_client[net_id]["labels"]
        else:
            pseudo_feats, pseudo_labels = None, None

        real_feats=real_feats.to(device)
        if pseudo_feats is not None:
            pseudo_feats=pseudo_feats.to(device)
        real_labels=real_labels.to(device)
        if pseudo_labels is not None:
            pseudo_labels=pseudo_labels.to(device)

        if args.use_prototype_pool and pseudo_feats is not None and pseudo_feats.numel() > 0:
            all_feats  = torch.cat([real_feats, pseudo_feats], dim=0)
            all_labels = torch.cat([real_labels, pseudo_labels], dim=0)
        else:
            all_feats=real_feats
            all_labels=real_labels
        feature_dataset = FeatureDataset(all_feats, all_labels)
        train_dl_local = torch.utils.data.DataLoader(feature_dataset,batch_size=args.batch_size,shuffle=True)

        # dataidxs = net_dataidx_map[net_id]
        # ipdb.set_trace()
        # train_dl_local, test_dl_global, train_ds_local, test_ds_global = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
        # len(train_dl_local)=5307/64=82   len(test_dl_local)=10000/64=313
        #len(train_ds_local)=5307    len(test_ds_global)=10000

        n_epoch = args.epochs
        for name, param in net.named_parameters():
            if name.startswith(("classifier", "img_adap", "mlp")):
                param.requires_grad = True

        # for name, param in net.named_parameters():
        #     if args.train_first_transformer_block:
        #         if name.startswith(("classifier", "img_adap", "mlp")) or "clip.visual.transformer.resblocks.0" in name:
        #             param.requires_grad = True
        #         else:
        #             param.requires_grad = False
        #     else:
        #         if name.startswith(("classifier", "img_adap", "mlp")) or "clip.visual.transformer.resblocks.0" in name or "clip.visual.transformer.resblocks.1" in name:
        #             param.requires_grad = True
        #         else:
        #             param.requires_grad = False

        trainacc, testacc = train_net_fedavg(net_id, net, global_model, train_dl_local, None, 
        n_epoch, args.lr,args.optimizer, args, device=device, pre_global_proto_by_class=pre_global_proto_by_class,)      

    if global_model:
        global_model.to('cpu')
    if server_c:
        for param_index, param in enumerate(server_c.parameters()):
            server_c_collector[param_index] = new_server_c_collector[param_index]
        server_c.to('cpu')
    return nets

#--------------------------------------------------------------------------clip计算各个客户端的原型--------------------------------------------------------------------------#
def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval() 

def column_variance(arr):
    # 计算每一列的方差，沿着第0维计算
    column_vars = np.var(arr, axis=0, ddof=0)
    return column_vars     

def dropout_proto_local_clustering(net, dataloader, args, n_class=10, device='cuda:0',client_id=None):
    feats = []
    labels = []
    net.eval()
    net.apply(fix_bn) 
    net.to(device)
    with torch.no_grad():
        for batch_idx, (x, target,_) in enumerate(dataloader):
            # ipdb.set_trace()
            x, target = x.to(device), target.to(device)
            feat= net.encode_image(x)
            if batch_idx == 0:
                feats = feat
                labels = target
            else:
                feats = torch.cat([feats, feat])
                labels = torch.cat([labels, target])

        #--------------------------保存特征---------------------------------#
        # epoch=0
        # save_root = os.path.join(args.feat_logdir, "frozen_encoder_feats", f"client_{client_id}")
        # os.makedirs(save_root, exist_ok=True)
        # save_path = os.path.join(save_root, f"epoch_{epoch}.pt")
        # torch.save({
        #     "feat": feats,
        #     "label": labels,
        # }, save_path)
        # print(f"[Client {client_id}] Features saved to {save_path}")

    prototype = []
    proto_label = []
    var = []
    proto_size = [] 
    
    # if args.dataset == 'tinyimagenet':
    #     #k:做聚类的最小样本数阈值  cs:看起来是cluster size
    #     k, cs = 50, 15  
    # elif args.dataset == 'cifar100' or args.dataset == 'tinyimagenet':
    #     k, cs = 40, 10 
    k=10000

    sample_feats = feats
    sample_labels = labels
    feats = feats.detach().cpu().numpy()   #5248 84  len(dataloder)=82  82*batch_size(64)=5248
    labels = labels.detach().cpu().numpy() 

    # kmeans = KMEANS(n_clusters=2, max_iter=20)
    # kmeans = KMEANS(n_clusters=2, max_iter=20)
    # cluster_size_max = 0
    # ipdb.set_trace()

    for i in range(n_class):
        idx_i = np.where(labels == i)[0]  #类别i在这个客户端上的所有样本索引  idx_i.shape=939
        
        # if len(idx_i) < k:
        #     continue

        if len(idx_i) >= k:
            predict_labels_i = kmeans.fit(feats[idx_i])  
            #predict_labels_i全为0，是因为n_clusters = 1  len(predict_labels_i)=939
            cluster_i_set, unq_cluster_i_size = np.unique(predict_labels_i, return_counts=True)
            #cluster_i_set=0  当前类别的cluster id集合  这个类别只被分成了1个cluster，编号是0
            #unq_cluster_i_size=939 每个cluster里有多少个样本  cluster0里有939个样本
            
            for cluster_id, cluster_size in zip(cluster_i_set, unq_cluster_i_size):
                #从“当前类别的所有样本中”，找出“属于某一个cluster的那些样本的下标” 
                data_cluster_id = np.where(predict_labels_i == cluster_id)[0] #len=939
                #取出该cluster的特征   
                feature_classwise = feats[idx_i][data_cluster_id]  #feature_classwise.shape=939 84
                #计算“当前类别i的某一个cluster内，特征在每一个维度上的方差”   这个cluster在特征空间里“有多分散/有多不稳定”
                clusters_var = column_variance(feats[idx_i][data_cluster_id]) #计算该cluster的方差
                #记录“最大cluster”的方差
                if cluster_size > cluster_size_max:
                    cluster_size_max = cluster_size
                    max_var = clusters_var  
                #proto是所有样本特征的均值向量  average_distance_to_proto是cluster整体的紧凑程度
                proto, average_distance_to_proto = compute_cluster_center_and_average_distance(feature_classwise)  #proto.shape=84     average_distance_to_proto=4.7828197
                prototype.append(proto)                        
                proto_label.append(int(i))
                #给刚刚算出来的这个prototype记录它“属于哪一个类别”
                var.append(clusters_var)
                proto_size.append(cluster_size)

        # 假设：类 0 → 1 个 proto  类 1 → 2 个 proto（类内聚类）  类 2 → 1 个 proto  
        # prototype = [
        #     proto_0_0,     # 类0
        #     proto_1_0,     # 类1
        #     proto_1_1,     # 类1
        #     proto_2_0      # 类2
        # ]   
        # proto_label = [
        #     0,
        #     1,
        #     1,
        #     2
        # ]     

        elif len(idx_i) > 1 and len(idx_i) < k: 
            #样本太少 → 不做聚类，直接把整个类别当成一个cluster 
            clusters_var = column_variance(feats[idx_i])   
            feature_classwise = feats[idx_i]
            proto, average_distance_to_proto = compute_cluster_center_and_average_distance(feature_classwise)    
            prototype.append(proto)                        
            proto_label.append(int(i))
            var.append(clusters_var) 
            proto_size.append(len(idx_i))   

                         
    # ipdb.set_trace()
    prototype = np.vstack(prototype) 
    proto_label = np.array(proto_label) 
    #len(prototype)=len(proto_label)=len(var)=7 应该是7个类别
    #prototype[0].shape=84  var[0].shape=84  proto_label=[0, 1, 2, 3, 4, 5, 7]

    #max_var是“所有cluster中，样本数最多的那个cluster的方差向量”，也就是数据最多、统计最稳定、最“可靠”的方差估计
    # var = 0.9 * np.vstack(var) + 0.1 * max_var
    # var = 0.5 * np.vstack(var) + 0.5 * max_var
    #这句代码的目的就是防止小样本cluster 的方差“乱飞”,用最大cluster的方差做“正则化 / 平滑”
    var = np.vstack(var)
    proto_size = np.array(proto_size)

    return (
        torch.tensor(prototype).to(device),
        torch.tensor(proto_label).to(device),
        torch.tensor(var).to(device),
        torch.tensor(proto_size).to(device),
        sample_feats.to(device),      
        sample_labels.to(device)      
    )

def compute_all_clients_prototypes(nets, net_dataidx_map, args, n_classes, device,train_dl_per_client):
    """
    返回：
      proto_bank_by_client: dict[client_id] -> {"proto": [Mi,d], "label":[Mi], "var":[Mi,d]}
      proto_bank_by_class:  dict[class_id]  -> list of {"client_id":id, "proto":[d], "var":[d]}
    """
    proto_bank_by_client = {}
    proto_bank_by_class = defaultdict(list)

    for client_id, net in nets.items():
        # ipdb.set_trace()
        if not 'domain' in args.dataset:
            dataidxs = net_dataidx_map[client_id]  #675
            train_dl_local, test_dl_global, train_ds_local, test_ds_global = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
            # len(train_dl_local)=675/64=10  len(test_dl_global)=3333/64=105
            #len(train_ds_local)=675    len(test_ds_global)=3333
            proto, proto_label, var, proto_size,feats, labels = dropout_proto_local_clustering(net=clip_model,dataloader=train_dl_local,args=args,n_class=n_classes,device=device,client_id=client_id)
        else:
            train_dl_local=train_dl_per_client[client_id]
            proto, proto_label, var, proto_size,feats, labels = dropout_proto_local_clustering(net=clip_model,dataloader=train_dl_local,args=args,n_class=n_classes,device=device,client_id=client_id)



        # 假设：类 0 → 1 个 proto  类 1 → 2 个 proto（类内聚类）  类 2 → 1 个 proto  
        # prototype = [
        #     proto_0_0,     # 类0
        #     proto_1_0,     # 类1
        #     proto_1_1,     # 类1
        #     proto_2_0      # 类2
        # ]   
        # proto_label = [
        #     0,
        #     1,
        #     1,
        #     2
        # ]  

        proto_cpu = proto.detach().cpu()
        proto_label_cpu = proto_label.detach().cpu()
        var_cpu = var.detach().cpu()
        proto_size_cpu = proto_size.detach().cpu()

        proto_bank_by_client[client_id] = {
            "proto": proto_cpu,
            "label": proto_label_cpu,
            "var": var_cpu,
            "size": proto_size_cpu,
            "feats": feats.detach().cpu(),    
            "labels": labels.detach().cpu()    
        }

        # 按类别展开存一份，方便服务端聚合
        for j in range(proto_label_cpu.shape[0]):
            c = int(proto_label_cpu[j].item())
            proto_bank_by_class[c].append({
                "client_id": client_id,
                "proto": proto_cpu[j],  # [d]
                "var": var_cpu[j],       # [d]
                "size": proto_size_cpu[j]
            })

    # ipdb.set_trace()
    return proto_bank_by_client, dict(proto_bank_by_class)
    #len(proto_bank_by_client)=10
    #len(proto_bank_by_client[0]["proto"])=7  也就是7个类别
    #proto_bank_by_client[0]["label"]=tensor([0, 1, 2, 3, 4, 5, 7])

   # proto_bank_by_client[1] = {
    #     "proto": [
    #         proto_0_0,     # 类 0
    #         proto_2_0,     # 类 2
    #     ],
    #     "label": [
    #         0,
    #         2,
    #     ],
    #     "var": [
    #         var_0_0,
    #         var_2_0,
    #     ]
    # }
 
    # proto_bank_by_class[c] = [
    #     {
    #         "client_id": i,
    #         "proto": [d],
    #         "var":   [d],
    #         "size":  scalar
    #     },
    #     {
    #         "client_id": j,
    #         "proto": [d],
    #         "var":   [d],
    #         "size":  scalar
    #     },
    #     ...
    # ]

# def aggregate_one_proto_per_class(proto_bank_by_class, eps=1e-6, device="cuda:0"):
#     """
#     输入:
#       proto_bank_by_class[c] = [
#           {"client_id":id, "proto":[d], "var":[d], "size":scalar}, ...
#       ]
#     输出:
#       global_proto[c] = tensor[d]
#     """
#     global_proto = {}
#     # ipdb.set_trace()
#     for c, items in proto_bank_by_class.items():
#         P = torch.stack([it["proto"] for it in items]).to(device)  # [m,d]
#         V = torch.stack([it["var"]  for it in items]).to(device)   # [m,d]
#         S = torch.tensor([it["size"] for it in items],dtype=P.dtype, device=device) 
#         S = S.unsqueeze(1)
#         # W = S/V                                       # [m,d] 逐维权重(precision) 
#         W = 1/V  
#         g = (W * P).sum(dim=0) / W.sum(dim=0)
#         global_proto[c] = g   #g.shape=512
#     return global_proto

def extract_frozen_domain_test_features(clip_model, dataloader, device, save_path=None):
    """
    对 domain 测试集提取冻结 CLIP 特征。
    输入是原始图像 dataloader。
    输出是 feats: [N, 512], labels: [N]
    """
    clip_model.eval()
    clip_model.to(device)

    all_feats = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (x, target, _) in enumerate(dataloader):
            x = x.to(device)
            target = target.to(device)

            feats = clip_model.encode_image(x).float()

            all_feats.append(feats.detach().cpu())
            all_labels.append(target.detach().cpu())

    all_feats = torch.cat(all_feats, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(
            {
                "feats": all_feats,
                "labels": all_labels,
            },
            save_path
        )
        print("Frozen domain test features saved to:", save_path)

    return all_feats, all_labels

def build_domain_test_feature_loader(clip_model, test_dl, args, device, save_path, batch_size=64):
    """
    如果测试域特征已经存在，就直接加载；
    如果不存在，就先提取并保存。
    """
    if os.path.exists(save_path):
        cached = torch.load(save_path, map_location="cpu")
        test_feats = cached["feats"]
        test_labels = cached["labels"]
        print("Loaded cached domain test features from:", save_path)
    else:
        test_feats, test_labels = extract_frozen_domain_test_features(
            clip_model,
            test_dl,
            device,
            save_path=save_path
        )

    test_feature_dataset = FeatureDataset(test_feats, test_labels)

    test_feature_loader = torch.utils.data.DataLoader(
        test_feature_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return test_feature_loader

if __name__ == '__main__':
    args = get_args()
    if args.use_mlp:
        sub_dir = 'clip冻结加上mlp和分类器'
    else:
        sub_dir = 'clip冻结加上分类器'
    # args.logdir = args.logdir + '/' + str(args.beta) + '/clients_' + str(args.n_parties)
    args.logdir = os.path.join(args.logdir, args.dataset, sub_dir)
    args.logdir = args.logdir + '/' + str(args.beta) + '/clients_' + str(args.n_parties)
    mkdirs(args.logdir)
    args.feat_logdir = os.path.join(args.feat_logdir, args.dataset, sub_dir)
    args.feat_logdir = args.feat_logdir + '/' + str(args.beta) + '/clients_' + str(args.n_parties)
    mkdirs(args.feat_logdir)
    mkdirs(args.modeldir)

    # ipdb.set_trace()
    if args.log_file_name is None:
        # argument_path = 'resnet18-experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")   #experiment_arguments-2025-11-18-1621-30.json
        argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")   #experiment_arguments-2025-11-18-1621-30.json
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if args.log_file_name is None:
        # args.log_file_name = 'resnet18-experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
        # args.log_file_name = 'clip-experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
    log_path = args.log_file_name + '.log'   #experiment_log-2025-11-18-1622-27.log
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)


    if args.dataset == "domain_office": 
        # name_classes = ['backpack', 'bike', 'calculator', 'headphones', 'keyboard', 'laptop_computer', 'monitor', 'mouse', 'mug', 'projector']
        name_classes = ["Alarm_Clock", "Backpack", "Batteries", "Bed", "Bike", "Bottle", "Bucket", "Calculator", "Calendar", "Candles",
    "Chair", "Clipboards", "Computer", "Couch", "Curtains", "Desk_Lamp", "Drill", "Eraser", "Exit_Sign", "Fan", "File_Cabinet", "Flipflops", "Flowers",
    "Folder",    "Fork",    "Glasses",    "Hammer",    "Helmet",    "Kettle",    "Keyboard",    "Knives",    "Lamp_Shade",    "Laptop",    "Marker",    
    "Monitor",    "Mop",    "Mouse",    "Mug",    "Notebook",    "Oven",    "Pan",    "Paper_Clip",    "Pen",    "Pencil",    "Postit_Notes",    "Printer",    
    "Push_Pin",    "Radio",    "Refrigerator",    "Ruler",    "Scissors",    "Screwdriver",    "Shelf",    "Sink",    "Sneakers",    "Soda",    "Speaker",
    "Spoon",    "Table",    "Telephone",    "ToothBrush",    "Toys",    "Trash_Can",    "TV",    "Webcam"]
    elif args.dataset == "domain_vlcs":
        name_classes = ["bird", "car", "chair", "dog", "person"]
    elif args.dataset == "domain_pacs":
        name_classes = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
    elif args.dataset == "domain_net":
        name_classes = ['The_Eiffel_Tower', 'The_Great_Wall_of_China', 'The_Mona_Lisa', 'aircraft_carrier', 'airplane', 'alarm_clock', 'ambulance', 'angel', 'animal_migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball']#, 'baseball_bat', 'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday_cake', 'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel', 'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling_fan', 'cell_phone', 'cello', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock', 'cloud', 'coffee_cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise_ship', 'cup', 'diamond', 'dishwasher', 'diving_board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser', 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'fence', 'finger', 'fire_hydrant', 'fireplace', 'firetruck', 'fish', 'flamingo', 'flashlight', 'flip_flops', 'floor_lamp', 'flower', 'flying_saucer', 'foot', 'fork', 'frog', 'frying_pan', 'garden', 'garden_hose', 'giraffe', 'goatee', 'golf_club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey_puck', 'hockey_stick', 'horse', 'hospital', 'hot_air_balloon', 'hot_dog', 'hot_tub', 'hourglass', 'house', 'house_plant', 'hurricane', 'ice_cream', 'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop', 'leaf', 'leg', 'light_bulb', 'lighter', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', 'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail', 'necklace', 'nose', 'ocean', 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paint_can', 'paintbrush', 'palm_tree', 'panda', 'pants', 'paper_clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano', 'pickup_truck', 'picture_frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police_car', 'pond', 'pool', 'popsicle', 'postcard', 'potato', 'power_outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain', 'rainbow', 'rake', 'remote_control', 'rhinoceros', 'rifle', 'river', 'roller_coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'school_bus', 'scissors', 'scorpion', 'screwdriver', 'sea_turtle', 'see_saw', 'shark', 'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping_bag', 'smiley_face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman', 'soccer_ball', 'sock', 'speedboat', 'spider', 'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo', 'stethoscope', 'stitches', 'stop_sign', 'stove', 'strawberry', 'streetlight', 'string_bean', 'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing_set', 'sword', 'syringe', 't-shirt', 'table', 'teapot', 'teddy-bear', 'telephone', 'television', 'tennis_racquet', 'tent', 'tiger', 'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado', 'tractor', 'traffic_light', 'train', 'tree', 'triangle', 'trombone', 'truck', 'trumpet', 'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing_machine', 'watermelon', 'waterslide', 'whale', 'wheel', 'windmill', 'wine_bottle', 'wine_glass', 'wristwatch', 'yoga', 'zebra', 'zigzag']

    logger.info("Partitioning data")
    if 'domain' not in args.dataset:
        X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, name_classes=None,beta=args.beta)
        # ipdb.set_trace()
        #X_train.shape=(50000,32,32,3)  y_train.shape=(50000,) X_test.shape=(10000,32,32,3)  y_test.shape=(10000,)
        #net_dataidx_map：dict: {client_id: 样本索引数组}
        #traindata_cls_counts：dict: {client → 各类别样本数}

        #打印各个客户端的样本数
        clients_indices = [net_dataidx_map[i] for i in range(args.n_parties)]
        # 打印每个客户端的样本数
        for client_id, client_data in enumerate(clients_indices):
            print(f"Client {client_id} has {len(client_data)} samples.")
        for client_id, client_data in enumerate(clients_indices):
            logger.info(f"Client {client_id} has {len(client_data)} samples.")
        # ipdb.set_trace()

        n_classes = len(np.unique(y_train))
        if args.dataset == 'cifar10':
            n_classes=10        
        elif args.dataset == 'cifar100':
            n_classes=100
        elif args.dataset == 'dtd':
            n_classes=47 
        elif args.dataset == 'cal101':
            n_classes=101  
        elif args.dataset == 'air100':
            n_classes=100   
        elif args.dataset == 'cars196':
            n_classes=196    

        missing_classes_per_client = {}
        for cid, cls_count in traindata_cls_counts.items():
            present_classes = set(cls_count.keys())      # 该客户端实际拥有的类
            all_classes = set(range(n_classes))           # 全部类
            missing_classes = sorted(all_classes - present_classes)
            missing_classes_per_client[cid] = missing_classes

        # ipdb.set_trace()
        train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset, args.datadir,args.batch_size, 32)
        #train_dl_global:整个训练集   test_dl_global:整个测试集   train_ds_global:完整训练数据集对象  test_ds_global:完整测试数据集对象
        #len(train_ds_global)=50000   len(test_ds_global)=10000  
        #len(train_dl_global)=50000 / 64 = 781个batch   len(test_dl_global)=10000/32=313个batch

        print("len train_dl_global:", len(train_ds_global))
        copy_domains_list = None
    else:
        if 'domain_office' == args.dataset:
            n_classes = 65
            domain_lists = ['Art', 'Clipart', 'Product', 'RealWorld']
        elif 'domain_vlcs' == args.dataset:
            n_classes = 5
            domain_lists = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
        elif 'domain_pacs' == args.dataset:
            n_classes = 7
            domain_lists = ['art_painting', 'cartoon', 'photo', 'sketch']
        elif 'domain_net' == args.dataset:
            n_classes = 20
            domain_lists = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        copy_domains_list = copy.deepcopy(domain_lists)
        copy_domains_list = [v for i, v in enumerate(copy_domains_list) if i not in args.domain_id]
        net_classname_map = [name_classes for i in range(args.n_parties)]
        net_dataidx_map = None
        test_dl_global = None
        missing_classes_per_client = {}

    # ipdb.set_trace()
    n_party_per_round = int(args.n_parties * args.sample_fraction)  #10：选择的每轮运行的客户端的数量
    #------------------------------------两者有什么区别-------------------------------------#
    party_list = [i for i in range(args.n_parties)]  #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]客户端集合
    # global_party_list = [i for i in range(args.n_parties)]   #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    party_list_rounds = []
    if n_party_per_round != args.n_parties:  #如果每轮不是选择全部客户端而是选择部分客户端则运行下面这个
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)
    #party_list_rounds里面一共多少行就是代表客户端运行多少轮，每一行里面的就是选择哪些客户端进行运行

    train_dl = None
    test_dl = None
    train_dl_per_client = None

    #------------------------------------------------domain情况下-------------------------------------------------#
    # ipdb.set_trace()
    if 'domain' in args.dataset:
        train_dl_per_client = {}
        n_domain = len(copy_domains_list)
        n_i = args.n_parties // n_domain  # 每个 domain 分多少 client
        for client_id in range(args.n_parties):
            domain_id = client_id // n_i
            domain_name = copy_domains_list[domain_id]

            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize
            ])

            #构造domain数据集
            train_ds = Domain_G(args.dataset, domain_name, transform_train)
            #fold切分（模拟多个client）
            fold_id = client_id % n_i
            train_data, train_labels, m_labels = train_ds.folds[fold_id]
            #DataLoader
            train_dataset = CustomDataset(train_data, train_labels, transform=transform_train)
            train_dl = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=4)
            train_dl_per_client[client_id] = train_dl

        domain_test_feature_loaders = {}
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])

        for idd in args.domain_id:
            test_domain_name = domain_lists[idd]
            test_ds = Domain_G(args.dataset,test_domain_name,transform_test)
            test_dl = DataLoader(test_ds,batch_size=32,shuffle=False,num_workers=4)
            test_feat_path = os.path.join(args.feat_logdir,"cached_domain_test_features",f"{args.dataset}_test_domain_{test_domain_name}_clip_feats.pt")
            test_feature_loader = build_domain_test_feature_loader(
                clip_model,
                test_dl,
                args,
                device,
                save_path=test_feat_path,
                batch_size=args.batch_size
            )
            domain_test_feature_loaders[idd] = {
                "loader": test_feature_loader,
                "domain_name": test_domain_name,
            }

    # ipdb.set_trace()
    logger.info("Initializing nets")
    nets, local_model_meta_data, layer_type = init_nets_mlp(args.net_config, args.n_parties, args, n_classes,device=device)
    #nets是关于客户端的字典
    #local_model_meta_data是客户端和服务端模型的网络的形状
    #layer_type是客户端和服务端模型的网络的名称

    global_models, global_model_meta_data, global_layer_type = init_nets_mlp(args.net_config, 1, args, n_classes, device=device)
    global_model = global_models[0]
    n_comm_rounds = args.comm_round

    # ipdb.set_trace()
    if args.load_model_file and args.alg != 'plot_visual':
        global_model.load_state_dict(torch.load(args.load_model_file))
        n_comm_rounds -= args.load_model_round

    if args.server_momentum:
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0

    if not 'domain' in args.dataset:
        proto_bank_by_client, proto_bank_by_class=compute_all_clients_prototypes(nets, net_dataidx_map, args, n_classes, device,train_dl_per_client)
        # ipdb.set_trace()

        pseudo_pool_by_client = defaultdict(dict)
        for net_id in range(args.n_parties):
            real_labels = proto_bank_by_client[net_id]["labels"]
            num_real_samples = len(real_labels)
            num_present_classes = len(torch.unique(real_labels))
            avg_samples_per_class = num_real_samples // num_present_classes

            # save_root = "./pseudo_npy"
            # save_dir = os.path.join(save_root, args.dataset, f"method_{args.prototype_pool_method}")
            # os.makedirs(save_dir, exist_ok=True)
            pseudo_feats=None
            if pseudo_feats is not None:
                pseudo_pool_by_client[net_id]["feats"] = pseudo_feats.detach().cpu()
                pseudo_pool_by_client[net_id]["labels"] = pseudo_labels.detach().cpu()
                # pseudo_feats_np = pseudo_feats.detach().cpu().numpy()
                # pseudo_labels_np = pseudo_labels.detach().cpu().numpy()
                # np.save(os.path.join(save_dir, f"client_{net_id}_pseudo_feats.npy"), pseudo_feats_np)
                # np.save(os.path.join(save_dir, f"client_{net_id}_pseudo_labels.npy"), pseudo_labels_np)
            else:
                pseudo_pool_by_client[net_id]["feats"] = None
                pseudo_pool_by_client[net_id]["labels"] = None
    else:
        proto_bank_by_client, proto_bank_by_class=compute_all_clients_prototypes(nets, net_dataidx_map, args, n_classes, device,train_dl_per_client)
        # ipdb.set_trace()
        pseudo_pool_by_client = defaultdict(dict)
        for net_id in range(args.n_parties):
            # save_root = "./pseudo_npy"
            # save_dir = os.path.join(save_root, args.dataset, f"method_{args.prototype_pool_method}")
            # os.makedirs(save_dir, exist_ok=True)
            pseudo_feats=None
            if pseudo_feats is not None:
                pseudo_pool_by_client[net_id]["feats"] = pseudo_feats.detach().cpu()
                pseudo_pool_by_client[net_id]["labels"] = pseudo_labels.detach().cpu()
                # pseudo_feats_np = pseudo_feats.detach().cpu().numpy()
                # pseudo_labels_np = pseudo_labels.detach().cpu().numpy()
                # np.save(os.path.join(save_dir, f"client_{net_id}_pseudo_feats.npy"), pseudo_feats_np)
                # np.save(os.path.join(save_dir, f"client_{net_id}_pseudo_labels.npy"), pseudo_labels_np)
            else:
                pseudo_pool_by_client[net_id]["feats"] = None
                pseudo_pool_by_client[net_id]["labels"] = None


        pre_global_proto_by_class = {}
        pre_global_proto_var_by_class={}
        for c, items in proto_bank_by_class.items():
            total_size = 0.0
            weighted_sum = None
            proto_list = []
            size_list = []
            for item in items:
                proto = item["proto"].float()   # [D]
                size = float(item["size"])
                proto_list.append(proto)
                size_list.append(size)
                if weighted_sum is None:
                    weighted_sum = size * proto
                else:
                    weighted_sum += size * proto
                total_size += size
            # 全局原型
            global_proto = weighted_sum / total_size
            pre_global_proto_by_class[c] = global_proto
            # 计算加权方差
            proto_stack = torch.stack(proto_list)       # [num_clients, D]
            size_tensor = torch.tensor(size_list).view(-1,1)  # [num_clients,1]
            var_c = ((proto_stack - global_proto)**2 * size_tensor).sum(dim=0) / total_size  # [D]
            pre_global_proto_var_by_class[c] = var_c

        
    for round in range(n_comm_rounds):
        logger.info("in comm round:" + str(round))
        party_list_this_round = party_list_rounds[round]  #party_list_this_round指的是当前有哪些客户端参与了训练
        global_w = global_model.state_dict()
        nets_this_round = {k: nets[k] for k in party_list_this_round}  #取到当前需要训练的客户端的模型
        
        # logger.info(f"===== Global model parameters =====")
        # for name, param in global_model.named_parameters():
        #     logger.info(f"{name} -> requires_grad: {param.requires_grad}")

        #给每个客户端的模型加载全局模型的权重
        for net in nets_this_round.values():
            # logger.info(f"===== Client model parameters =====")
            # for name, param in net.named_parameters():
            #     logger.info(f"{name} -> requires_grad: {param.requires_grad}")
            net.load_state_dict(global_w, strict=False)

        # ipdb.set_trace()
        nets_this_round = local_train_net(nets_this_round, args, net_dataidx_map, 
        train_dl=train_dl, test_dl=test_dl_global,global_model=global_model, 
        round=round, device=device,proto_bank_by_class=proto_bank_by_class,
        proto_bank_by_client=proto_bank_by_client,missing_classes_per_client=missing_classes_per_client,
        pseudo_pool_by_client=pseudo_pool_by_client,
        pre_global_proto_by_class=pre_global_proto_by_class,
        )

        # len(nets_this_round)=10  输出是训练后更新过的客户端模型
        global_model.to('cpu')

        # update global model
        # global_nets_this_round = {k: nets[k] for k in global_party_list} 
        # total_data_points = sum([len(net_dataidx_map[r]) for r in global_party_list])
        # fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in global_party_list]

        # ipdb.set_trace()
        # global_nets_this_round = {k: nets[k] for k in party_list_this_round} 


        if 'domain' not in args.dataset:
            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])  #当前训练的客户端的总的数据量：50000
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round] #各个客户端数据占总数据量的比例
        else:
            fed_avg_freqs = [1 / args.n_parties] * args.n_parties
            

        #给全局模型聚合参数
        # ipdb.set_trace()
        for net_id, net in enumerate(nets_this_round.values()):
            net.to('cpu')
            net_para = net.state_dict()
            if net_id == 0:
                for key in net_para:
                    global_w[key] = net_para[key] * fed_avg_freqs[net_id]
            else:
                for key in net_para:
                    global_w[key] += net_para[key] * fed_avg_freqs[net_id]
                    
        global_model.load_state_dict(global_w,strict=True)

        #---------------------------------指定轮次保存全局模型权重和参与训练的客户端模型权重---------------------------------#
        # if round in args.save_rounds:

        #     ckpt_root = os.path.join(args.ckpt_root,f"beta_{args.beta}",f"clients_{args.n_parties}")
        #     os.makedirs(ckpt_root, exist_ok=True)
        #     round_dir = os.path.join(ckpt_root, f"round_{round:03d}")
        #     os.makedirs(round_dir, exist_ok=True)

        #     #保存全局模型
        #     torch.save(
        #         global_model.state_dict(),
        #         os.path.join(round_dir, "global_model.pt")
        #     )

        #     #保存本轮参与训练的客户端模型
        #     for client_id in party_list_this_round:
        #         client_model = nets[client_id]
        #         torch.save(
        #             client_model.state_dict(),
        #             os.path.join(round_dir, f"client_{client_id}.pt")
        #         )

        #     logger.info(f"Saved models at communication round {round}")
        #--------------------------------------------------------------------------------------------------------------#

        if 'domain' not in args.dataset:
            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            # ipdb.set_trace()
            global_model.to(device)
            #评估全局模型在训练集上的表现
            # train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            #评估全局模型在测试集上的表现
            test_acc, _,test_loss = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)

            # logger.info('>> Global Model Train accuracy: %f' % train_acc)
            # logger.info('>> Global Model Train loss: %f' % train_loss)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Test loss: %f' % test_loss)   

            
            # if round % 5 == 4:
            # if round == 0 or (round >= 9 and round % 10 == 9):
            #     save_root = os.path.join(args.modeldir,'FedAvg',sub_dir,args.dataset,argument_path,str(round))
            #     mkdirs(save_root)
            #     global_model.to('cpu')
            #     torch.save(global_model.state_dict(),os.path.join(save_root, 'global_model.pth'))
            #     for i in range(10):
            #         torch.save(nets[i].state_dict(), os.path.join(save_root, f'local_{i}.pth'))

        else:
            # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
            # transform_test = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.ToPILImage(),
            #     transforms.Resize((224, 224)),
            #     transforms.ToTensor(),
            #     normalize
            # ])
            # global_model.to(device)
            # for idd in args.domain_id:
            #     test_ds = Domain_G(args.dataset, domain_lists[idd], transform_test)
            #     test_dl = DataLoader(test_ds,batch_size=args.batch_size,shuffle=False,num_workers=4)
            #     test_acc, conf_matrix, test_loss = compute_accuracy(global_model,test_dl,get_confusion_matrix=True,device=device)
            #     logger.info('>> Global Model LODO Test accuracy: %f, Domain ID: %s'% (test_acc, str(idd)))
            #     logger.info('>> Global Model LODO Test loss: %f, Domain ID: %s'% (test_loss, str(idd)))
            global_model.to(device)

            for idd in args.domain_id:
                test_feature_loader = domain_test_feature_loaders[idd]["loader"]
                test_domain_name = domain_test_feature_loaders[idd]["domain_name"]

                test_acc, conf_matrix, test_loss = compute_accuracy_features(
                    global_model,
                    test_feature_loader,
                    get_confusion_matrix=True,
                    device=device
                )

                logger.info(
                    '>> Global Model LODO Test accuracy: %f, Domain ID: %s, Domain: %s'
                    % (test_acc, str(idd), test_domain_name)
                )

                logger.info(
                    '>> Global Model LODO Test loss: %f, Domain ID: %s, Domain: %s'
                    % (test_loss, str(idd), test_domain_name)
                )



