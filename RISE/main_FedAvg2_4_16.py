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
from itertools import combinations
import time
import csv

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='simple-cnn', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='ucf101', help='dataset used for training')
    parser.add_argument('--use_mlp',default=True,help='whether to enable MLP projection head')
    parser.add_argument('--device', type=str, default='cuda:3', help='The device to run the program')
    parser.add_argument('--beta', type=float, default=0.5,help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--comm_round', type=int, default=30, help='number of maximum communication roun')
    parser.add_argument('--sample_fraction', type=float, default=1, help='how many clients are sampled in each round')

    parser.add_argument('--method', type=int, default=2, help='')
    parser.add_argument('--mu', type=float, default=0.2, help='方法一的损失系数')
    parser.add_argument('--nu', type=int, default=2, help='方法二的损失系数')
    parser.add_argument('--lu', type=int, default=1, help='分类损失系数')
    parser.add_argument('--te', type=int, default=100, help='方法二MSE除以的系数50/100')
    parser.add_argument('--K', type=int, default=128, help='特征值个数')

    parser.add_argument('--use_gate',default=False,help='whether to enable to use gate')
    parser.add_argument('--use_residual',default=False,help='whether to enable MLP projection head')
    parser.add_argument('--use_spectral_loss',default=False,help='')
    parser.add_argument('--use_subspace_rotation',default=False,help='')
    parser.add_argument('--num_clusters', type=int, default=5, help='客户端内簇的个数')

    parser.add_argument('--use_prototype_pool',type=bool,default=False,help='是否使用原型池 (True/False)')
    parser.add_argument('--prototype_pool_method',type=int,default=9,help='原型池构建方法编号:1/2/3/4/.....')
    parser.add_argument('--num_samples_per_proto',type=int, default=10,help='每个原型生成的伪样本数量')
    parser.add_argument('--num_samples_per_pair',type=int, default=2,help='每两个原型之间生成的伪样本数量')
    parser.add_argument('--num_mixup_samples_per_class',type=int, default=25,help='每个类别生成的伪样本数量')

    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.1)')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--logdir', type=str, required=False, default="./logs_new_new_experiment10", help='Log directory path')
    parser.add_argument('--feat_logdir',type=str,required=False,default="./feat_logs_new_new_experiment9",help='Feature save directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models_new_new_experiment9", help='Model directory path')
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

def sync_cuda(device):
    if torch.cuda.is_available() and "cuda" in str(device):
        torch.cuda.synchronize()

def cosine_sim(a, b, eps=1e-8):
    a = a / (a.norm() + eps)
    b = b / (b.norm() + eps)
    return torch.dot(a, b)

def cosine_sim_matrix(x):
    """
    x: [B, D]
    return: [B, B] cosine similarity matrix
    """
    x = F.normalize(x, dim=1)   # 单位化，避免尺度影响
    return x @ x.T

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

#来控制各个主方向的贡献程度，而不是直接控制主方向的方向
#损失函数并不会约束哪些特征向量（主方向）会被选择，它只是约束了不同主方向上的方差分布应当尽可能一致
#因此，主方向的方向可能会发生变化，但它们的贡献程度不会变得极端

# →限制协方差谱的偏离
# → 防止单方向能量主导
# → 限制 effective rank collapse
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

    # ---- 打印特征值及数量 ----#
    print(f"Pretrained features eigenvalues ({len(eig_pre)}): {eig_pre}")
    print(f"Adapted features eigenvalues ({len(eig_mlp)}): {eig_mlp}")

    # ---- 只保留主能量谱（更稳定）----
    # 主能量谱：通过 torch.topk 函数保留协方差特征值中的 前 k 个最大特征值，k 取值为 64 或者特征值的数量（取较小的值）
    k = min(args.K, len(eig_pre))
    eig_pre = torch.topk(eig_pre, k).values
    eig_mlp = torch.topk(eig_mlp, k).values

    # 预训练特征和适配后特征在主成分方向上的方差差异的指标
    # 这意味着我们在训练时会关注特征值谱的差异，尽量保持预训练特征和适配后特征的主要变化方向一致，从而避免适配层（MLP）对特征的过度修改
    return torch.mean((eig_pre - eig_mlp) ** 2)

# def spectral_loss(feat_pre, feat_mlp):
#     feat_pre = feat_pre - feat_pre.mean(dim=0, keepdim=True)
#     feat_mlp = feat_mlp - feat_mlp.mean(dim=0, keepdim=True)

#     cov_pre = feat_pre.T @ feat_pre / feat_pre.shape[0]
#     cov_mlp = feat_mlp.T @ feat_mlp / feat_mlp.shape[0]

#     eig_pre = torch.linalg.eigvalsh(cov_pre)
#     eig_mlp = torch.linalg.eigvalsh(cov_mlp)

#     k = min(128, eig_pre.numel())
#     eig_pre = torch.topk(eig_pre, k).values
#     eig_mlp = torch.topk(eig_mlp, k).values

#     eig_pre = eig_pre / (eig_pre.sum() + 1e-8)
#     eig_mlp = eig_mlp / (eig_mlp.sum() + 1e-8)

#     return F.mse_loss(eig_mlp, eig_pre)

# def spectral_loss_svd(feat_pre, feat_mlp):
#     """
#     feat_pre : [B, D] 预训练特征
#     feat_mlp : [B, D] 适配后特征
#     使用 SVD 计算协方差特征值谱，稳定性更好
#     """
#     # ---- 零均值化 ----
#     feat_pre = feat_pre - feat_pre.mean(dim=0, keepdim=True)
#     feat_mlp = feat_mlp - feat_mlp.mean(dim=0, keepdim=True)

#     # ---- SVD 计算特征值 ----
#     # 协方差矩阵的特征值 = (奇异值)^2 / (B-1)
#     _, s_pre, _ = torch.linalg.svd(feat_pre, full_matrices=False)
#     _, s_mlp, _ = torch.linalg.svd(feat_mlp, full_matrices=False)

#     eig_pre = s_pre**2 / (feat_pre.shape[0] - 1)
#     eig_mlp = s_mlp**2 / (feat_mlp.shape[0] - 1)

#     # ---- 只保留主能量谱（稳定） ----
#     k = min(128, len(eig_pre))
#     eig_pre = torch.topk(eig_pre, k).values
#     eig_mlp = torch.topk(eig_mlp, k).values

#     # ---- 计算谱差损失 ----
#     loss = torch.mean((eig_pre - eig_mlp) ** 2)
#     return loss

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
        if args.use_gate:
            self.gate = nn.Sequential(nn.Linear(514, 512),nn.Sigmoid()).to(device)
            # # 让初始 gate 偏小：一开始尽量少改预训练特征
            # nn.init.zeros_(self.gate[0].weight)
            # nn.init.constant_(self.gate[0].bias, -2.0)
            # # sigmoid(-2) ≈ 0.12

        # ===== 新增：共享可学习曲率（学 log c）=====#
        # curv_init1 = 0.5
        # self.log_curv1 = nn.Parameter(torch.tensor(curv_init1).log())

        # curv_init2 = 1.0
        # self.log_curv2 = nn.Parameter(torch.tensor(curv_init2).log())

        # scale_init = 0.1
        # self.log_scale = nn.Parameter(torch.tensor(scale_init).log())

    # def get_curv1(self):
    #     log_curv1 = torch.clamp(self.log_curv1, min=math.log(0.05), max=math.log(5.0))
    #     return log_curv1.exp()

    # def get_curv2(self):
    #     log_curv2 = torch.clamp(self.log_curv2, min=math.log(0.05), max=math.log(5.0))
    #     return log_curv2.exp()

    # def get_scale(self):
    #     log_scale = torch.clamp(self.log_scale, min=math.log(0.05), max=math.log(1.0))
    #     return log_scale.exp()

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
            # feature_mean = cls_img.mean(dim=1, keepdim=True)
            # feature_std = cls_img.std(dim=1, keepdim=True)
        if args.use_mlp:
            if args.use_residual:
                x=self.norm(cls_img)
                residual = self.mlp(x)
                if args.use_gate:
                    gate_input = torch.cat([x, feature_mean, feature_std], dim=1)
                    gate = self.gate(gate_input)
                    cls_img = x + gate * residual
                else:
                    cls_img = x + residual
            else:
                cls_img=self.norm(cls_img)
                cls_img=self.mlp(cls_img)
                # cls_img = cls_img / cls_img.norm(dim=1, keepdim=True)
        
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
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             # 跳过已经手动初始化的 gate 层
    #             if hasattr(self, "gate") and m is self.gate[0]:
    #                 continue
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)

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
        if args.use_gate:
            self.gate = nn.Sequential(nn.Linear(514, 512),nn.Sigmoid()).to(device)
            # nn.init.zeros_(self.gate[0].weight)
            # nn.init.normal_(self.gate[0].weight, mean=0.0, std=0.01)
            # nn.init.constant_(self.gate[0].bias, -2.0)

        # ===== 新增：共享可学习曲率（学 log c）=====#
        # curv_init1 = 0.5
        # self.log_curv1 = nn.Parameter(torch.tensor(curv_init1).log())

        # curv_init2 = 1.0
        # self.log_curv2 = nn.Parameter(torch.tensor(curv_init2).log())

        # scale_init = 0.1
        # self.log_scale = nn.Parameter(torch.tensor(scale_init).log())

    # def get_curv1(self):
    #     log_curv1 = torch.clamp(self.log_curv1, min=math.log(0.05), max=math.log(5.0))
    #     return log_curv1.exp()

    # def get_curv2(self):
    #     log_curv2 = torch.clamp(self.log_curv2, min=math.log(0.05), max=math.log(5.0))
    #     return log_curv2.exp()

    # def get_scale(self):
    #     log_scale = torch.clamp(self.log_scale, min=math.log(0.05), max=math.log(1.0))
    #     return log_scale.exp()


    def forward(self, cls_img):
        # image_embeds = self.clip.encode_image(x)
        # ipdb.set_trace()
        # image_embeds.float()
        # cls_img=image_embeds[:,0,:]

        # cls_img = self.clip.encode_image(x)
        # output_img = self.classifier(cls_img) 
        # return None,output_img
        # ipdb.set_trace()

        # feature_mean = cls_img.mean(dim=1, keepdim=True)
        # feature_std = cls_img.std(dim=1, keepdim=True)

        pre_img=cls_img
        if args.use_mlp:
            if args.use_residual:
                x=self.norm(cls_img)
                residual = self.mlp(x)
                if args.use_gate:
                    gate_input = torch.cat([x, feature_mean, feature_std], dim=1)
                    gate = self.gate(gate_input)
                    cls_img = x + gate * residual
                    print("Gate mean value:", gate.mean().item())
                else:
                    cls_img = x + residual
            else:
                cls_img=self.norm(cls_img)
                cls_img=self.mlp(cls_img)
                # cls_img = cls_img / cls_img.norm(dim=1, keepdim=True)
        
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
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             if hasattr(self, "gate") and m is self.gate[0]:
    #                 continue
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)

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
            if name.startswith(("classifier", "img_adap", "mlp")) or name == "log_curv1" or name == "log_curv2":
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
            if name.startswith(("classifier", "img_adap", "mlp")) or name == "log_curv1" or name == "log_curv2":
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

# def exp_map_to_poincare(x):
#     # ipdb.set_trace()
#     # 计算欧几里得空间中x的L2范数
#     norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
#     # 使用指数映射公式将特征映射到Poincaré球
#     return torch.tanh(norm_x) * x / norm_x

# def exp_map_to_poincare(x, scale=0.05):
#     """
#     x: [N, D] Euclidean features
#     scale: 控制投影到球内的距离
#     """
#     norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
#     scaled_norm = scale * norm_x
#     factor = torch.tanh(scaled_norm) / (norm_x)
#     return factor * x

# def exp_map_to_poincare(x, c=1.0, scale=0.05, eps=1e-8):
#     """
#     x: [N, D] Euclidean tangent vectors at origin
#     c: positive curvature parameter, hyperbolic curvature is -c
#     scale: input scaling factor before exp map
#     """
#     x = x * scale
#     norm_x = torch.norm(x, p=2, dim=-1, keepdim=True).clamp_min(eps)
#     sqrt_c = torch.sqrt(torch.as_tensor(c, device=x.device, dtype=x.dtype))
#     factor = torch.tanh(sqrt_c * norm_x) / (sqrt_c * norm_x)
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

def hyperbolic_distance_preservation_loss_cached(fused_proto_dict,pre_dist_full,class_to_idx,c1=0.5,device="cuda"):
    """
    使用预先计算好的预训练类间距离矩阵，避免每个 batch 重复计算 dist_pre。
    """
    classes = sorted([cls for cls in fused_proto_dict.keys() if cls in class_to_idx])

    if len(classes) < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 当前 batch / memory 融合原型，需要保留梯度
    fused_mat = torch.stack(
        [fused_proto_dict[c] for c in classes],
        dim=0
    ).to(device)  # [C_cur, D]

    fused_hyp = exp_map_to_lorentz_space(fused_mat, c1=c1)
    dist_fused = lorentz_dist_matrix(fused_hyp, c1=c1)

    # 从完整预训练关系矩阵中取出当前 batch 类别对应的子矩阵
    idx = torch.tensor(
        [class_to_idx[c] for c in classes],
        dtype=torch.long,
        device=device
    )

    pre_dist_full = pre_dist_full.to(device)
    dist_pre = pre_dist_full.index_select(0, idx).index_select(1, idx).detach()

    loss = ((dist_fused - dist_pre) ** 2).mean() / args.te
    return loss

# def euclidean_dist_matrix(proto_mat):
#     """
#     计算欧式距离矩阵 [C, C]，proto_mat: [C, D] 是原型矩阵
#     """
#     dist_mat = torch.cdist(proto_mat, proto_mat, p=2)   # 计算欧式距离矩阵
#     return dist_mat

# def print_euclidean_distances(fused_proto_dict, pre_global_proto_by_class, device="cuda"):
#     """
#     打印 fused_proto_dict 和 pre_global_proto_by_class 的欧式空间距离矩阵
#     fused_proto_dict: 训练融合原型
#     pre_global_proto_by_class: 预训练原型
#     """
#     # 获取交集类别
#     classes = sorted(list(set(fused_proto_dict.keys()) & set(pre_global_proto_by_class.keys())))
#     if len(classes) == 0:
#         print("没有交集的类别")
#         return

#     # 堆叠为矩阵 [C, D]
#     fused_mat = torch.stack([fused_proto_dict[c] for c in classes], dim=0).to(device)   # [C, D]
#     pre_mat = torch.stack([pre_global_proto_by_class[c].to(device) for c in classes], dim=0)  # [C, D]

#     # 计算欧式距离矩阵
#     euc_dist_fused = euclidean_dist_matrix(fused_mat)  # [C, C]
#     euc_dist_pre = euclidean_dist_matrix(pre_mat)      # [C, C]

#     # 打印欧式距离矩阵
#     print("适配训练原型的欧式距离矩阵:")
#     print(euc_dist_fused.detach().cpu())

#     print("\n预训练原型的欧式距离矩阵:")
#     print(euc_dist_pre.detach().cpu())


def euclidean_distance_preservation_loss(fused_proto_dict, pre_global_proto_by_class, device="cuda"):
    """
    Args:
        fused_proto_dict: dict[class_id] -> [D] 训练融合原型
        pre_global_proto_by_class: dict[class_id] -> [D] 预训练原型
    Returns:
        loss: scalar tensor, 保持类别间欧式距离
    """
    # 找出交集类别，保证矩阵对应
    classes = sorted(list(set(fused_proto_dict.keys()) & set(pre_global_proto_by_class.keys())))
    if len(classes) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 堆叠为矩阵 [C, D]
    fused_mat = torch.stack([fused_proto_dict[c] for c in classes], dim=0).to(device)   # [C, D]
    pre_mat = torch.stack([pre_global_proto_by_class[c].to(device) for c in classes], dim=0)  # [C, D]


    # 计算欧式距离矩阵
    def euclidean_dist_matrix(proto_mat, eps=1e-6):
        u = proto_mat.unsqueeze(1)  # [C,1,D]
        v = proto_mat.unsqueeze(0)  # [1,C,D]
        dist_mat = torch.norm(u - v + eps, dim=-1)  # [C,C] 欧式距离
        dist_mat = dist_mat - torch.diag(dist_mat.diagonal())
        return dist_mat

    dist_fused = euclidean_dist_matrix(fused_mat)
    dist_pre   = euclidean_dist_matrix(pre_mat)

    # MSE loss 保持训练原型间距离与预训练原型一致
    loss = ((dist_fused - dist_pre) ** 2).mean()
    return loss


# def log_map_to_tangent(p, q, c=1.0, eps=1e-5):
#     """
#     Log map of point q at base point p on Poincare ball with curvature c
#     p, q: [D] or [B, D] in Poincare ball
#     Returns: vector in tangent space T_p
#     """
#     c = torch.tensor(c, device=p.device, dtype=p.dtype)
    
#     p_norm2 = torch.sum(p**2, dim=-1, keepdim=True)
#     q_norm2 = torch.sum(q**2, dim=-1, keepdim=True)
#     inner_pq = torch.sum(p * q, dim=-1, keepdim=True)
    
#     # 正确的 Möbius 减法: (-p) ⊕_c q
#     num = (1.0 - c * p_norm2) * q - (1.0 - 2.0 * c * inner_pq + c * q_norm2) * p
#     den = 1.0 - 2.0 * c * inner_pq + c**2 * p_norm2 * q_norm2
#     den = torch.clamp(den, min=eps)
#     v = num / den  # 已平移至原点切空间

#     v_norm = torch.norm(v, dim=-1, keepdim=True)
#     v_norm_safe = torch.clamp(v_norm, min=eps)

#     sqrt_c = torch.sqrt(c)
#     atanh_arg = torch.clamp(sqrt_c * v_norm, max=1.0 - eps)  # 防 NaN
#     factor = (1.0 / sqrt_c) * torch.atanh(atanh_arg) / v_norm_safe

#     return factor * v

# def relative_direction_loss(fused_proto_dict, pre_proto_dict, scale=0.05, device="cuda"):
#     """
#     计算不同类别原型之间在双曲空间切向量上的相对方向损失
#     fused_proto_dict: 当前融合后的原型字典 {class_id: tensor}
#     pre_proto_dict: 预训练原型字典 {class_id: tensor}
#     scale: 指数映射到 Poincare 球的缩放
#     """
#     classes = sorted(list(set(fused_proto_dict.keys()) & set(pre_proto_dict.keys())))
#     if len(classes) < 2:
#         return torch.tensor(0.0, device=device, requires_grad=True)

#     loss = 0.0
#     count = 0

#     for i in range(len(classes)):
#         ci = classes[i]
#         pi = fused_proto_dict[ci].to(device)
#         pi_pre = pre_proto_dict[ci].to(device)

#         for j in range(len(classes)):
#             if i == j:
#                 continue
#             cj = classes[j]
#             pj = fused_proto_dict[cj].to(device)
#             pj_pre = pre_proto_dict[cj].to(device)

#             # 映射到双曲空间
#             pi_hyp     = exp_map_to_poincare(pi, scale=scale)
#             pj_hyp     = exp_map_to_poincare(pj, scale=scale)
#             pi_pre_hyp = exp_map_to_poincare(pi_pre, scale=scale)
#             pj_pre_hyp = exp_map_to_poincare(pj_pre, scale=scale)

#             # 切空间映射（相对于参考类别 i）
#             v_ij     = log_map_to_tangent(pi_hyp, pj_hyp)
#             v_ij_pre = log_map_to_tangent(pi_pre_hyp, pj_pre_hyp)

#             # 方向归一化
#             v_ij     = v_ij / (v_ij.norm() + 1e-6)
#             v_ij_pre = v_ij_pre / (v_ij_pre.norm() + 1e-6)

#             # 相对方向损失
#             loss += 1 - (v_ij * v_ij_pre).sum()
#             count += 1

#     return loss / count


# def relative_direction_loss_with_near_neighbors(fused_proto_dict, pre_proto_dict, scale=0.0, k=5, device="cuda"):
#     """
#     计算近距离类别原型之间的方向损失，并返回方向损失。
    
#     Args:
#         fused_proto_dict: dict[class_id] -> [D] 当前训练的类别原型
#         pre_proto_dict: dict[class_id] -> [D] 预训练的类别原型
#         scale: 指数映射到 Poincare 球的缩放
#         k: 选择每个类别最近的 k 个邻居来计算方向损失
#     Returns:
#         loss: scalar tensor, 方向损失
#     """
#     # 找出交集类别
#     classes = sorted(list(set(fused_proto_dict.keys()) & set(pre_proto_dict.keys())))
#     if len(classes) == 0:
#         return torch.tensor(0.0, device=device, requires_grad=True)

#     # 堆叠为矩阵 [C, D]
#     fused_mat = torch.stack([fused_proto_dict[c] for c in classes], dim=0).to(device)   # [C, D]
#     pre_mat = torch.stack([pre_proto_dict[c].to(device) for c in classes], dim=0)  # [C, D]

#     # 映射到双曲空间
#     fused_hyp = exp_map_to_poincare(fused_mat)   # [C, D]

#     # 计算测地距离矩阵
#     def poincare_dist_matrix(proto_hyp, eps=1e-5):
#         u = proto_hyp.unsqueeze(1)  # [C,1,D]
#         v = proto_hyp.unsqueeze(0)  # [1,C,D]
#         norm_u = (u ** 2).sum(dim=-1)  # [C,1]
#         norm_v = (v ** 2).sum(dim=-1)  # [1,C]
#         diff = ((u - v) ** 2).sum(dim=-1)  # [C,C]
#         denom = (1 - norm_u) * (1 - norm_v) + eps
#         arg = 1 + 2 * diff / denom
#         arg = torch.clamp(arg, min=1 + eps)
#         dist_mat = torch.acosh(arg)
#         dist_mat.fill_diagonal_(0.0)
#         return dist_mat

#     dist_fused = poincare_dist_matrix(fused_hyp)

#     # 对每个类别，选择最近的 k 个类别
#     loss = 0.0
#     count = 0
#     for i in range(len(classes)):
#         ci = classes[i]
#         pi = fused_proto_dict[ci].to(device)
#         pi_pre = pre_proto_dict[ci].to(device)

#         # 找到类别 i 的最近 k 个类别
#         _, nearest_indices = torch.topk(dist_fused[i], k, largest=False)

#         for idx in nearest_indices:
#             cj = classes[idx]
#             if ci == cj:
#                 continue

#             # 获取当前类别和邻居类别的原型
#             pi_hyp = exp_map_to_poincare(pi, scale=scale)
#             pj_hyp = exp_map_to_poincare(fused_proto_dict[cj].to(device), scale=scale)

#             pi_pre_hyp = exp_map_to_poincare(pi_pre, scale=scale)
#             pj_pre_hyp = exp_map_to_poincare(pre_proto_dict[cj].to(device), scale=scale)

#             # 计算切空间映射（相对于参考类别 i）
#             v_ij     = log_map_to_tangent(pi_hyp, pj_hyp)
#             v_ij_pre = log_map_to_tangent(pi_pre_hyp, pj_pre_hyp)

#             # 方向归一化
#             v_ij     = v_ij / (v_ij.norm() + 1e-6)
#             v_ij_pre = v_ij_pre / (v_ij_pre.norm() + 1e-6)

#             # 相对方向损失
#             loss += (1 - (v_ij * v_ij_pre).sum())
#             count += 1

#     return loss / count if count > 0 else torch.tensor(0.0, device=device, requires_grad=True)

# def hyperbolic_angle_preservation_loss(fused_proto_dict,neighbor_dict,pre_angle_dict,c=1.0,scale=0.1,device="cuda",reduction="mean",):
#     """
#     保持当前客户端融合原型与预训练定义好的近邻角度一致

#     Args:
#         fused_proto_dict: dict[class_id] -> tensor[D]
#             当前客户端 batch/memory 融合后的类别原型
#         neighbor_dict: dict[class_id] -> list[neighbor_class_id]
#             预训练阶段固定好的近邻类别
#         pre_angle_dict: dict[(class_id, neighbor_class_id)] -> scalar tensor/float
#             预训练阶段保存好的 angle(O, class_i, neighbor_j)
#         c: Lorentz 曲率参数（实际曲率为 -c）
#         scale: 映射到 Lorentz 双曲空间前的缩放
#         device: 设备
#         reduction: "mean" 或 "sum"

#     Returns:
#         loss: scalar tensor
#     """
#     angle_losses = []

#     # 当前客户端里有哪些类别
#     current_classes = set(fused_proto_dict.keys())

#     for cid in current_classes:
#         # 如果这个类别没有预定义近邻，跳过
#         if cid not in neighbor_dict:
#             continue

#         # 当前类别原型 -> Lorentz 双曲空间
#         proto_i = fused_proto_dict[cid].to(device)          # [D]
#         proto_i_hyp = exp_map_to_lorentz_space(
#             proto_i.unsqueeze(0), c=c, scale=scale
#         ).squeeze(0)                                        # [D]

#         for nid in neighbor_dict[cid]:
#             # 只有当近邻类别也在当前客户端出现时，才算这个角度
#             if nid not in current_classes:
#                 continue

#             # 如果预训练里没保存这个角度，跳过
#             if (cid, nid) not in pre_angle_dict:
#                 continue

#             proto_j = fused_proto_dict[nid].to(device)
#             proto_j_hyp = exp_map_to_lorentz_space(
#                 proto_j.unsqueeze(0), c=c, scale=scale
#             ).squeeze(0)

#             # 当前客户端的角度
#             angle_cur = oxy_angle(proto_i_hyp, proto_j_hyp, c=c)   # scalar

#             # 预训练参考角度
#             angle_pre = pre_angle_dict[(cid, nid)]
#             if not torch.is_tensor(angle_pre):
#                 angle_pre = torch.tensor(angle_pre, device=device, dtype=proto_i.dtype)
#             else:
#                 angle_pre = angle_pre.to(device=device, dtype=proto_i.dtype)

#             # MSE
#             angle_losses.append((angle_cur - angle_pre) ** 2)
#             # angle_losses.append(1.0 - torch.cos(angle_cur - angle_pre))

#             # # 带容忍范围的角度损失
#             # delta=0.1
#             # diff = torch.abs(angle_cur - angle_pre)
#             # angle_losses.append(torch.clamp(diff - delta, min=0.0) ** 2)

#     if len(angle_losses) == 0:
#         return torch.tensor(0.0, device=device, requires_grad=True)

#     angle_losses = torch.stack(angle_losses)

#     if reduction == "sum":
#         return angle_losses.sum()
#     else:
#         return angle_losses.mean()


# def hyperbolic_angle_preservation_loss_sampled(fused_proto_dict,neighbor_dict,pre_angle_dict,c=1.0,scale=0.05,device="cuda",reduction="mean",
#     sample_ratio=1.0,     # 采样比例，例如 0.5 表示只取一半有效对
#     max_pairs=None,       # 最多取多少对，例如 8
# ):
#     """
#     随机采样一部分有效 (cid, nid) 角度对来计算角度保持损失

#     Args:
#         fused_proto_dict: dict[class_id] -> tensor[D]
#         neighbor_dict: dict[class_id] -> list[neighbor_class_id]
#         pre_angle_dict: dict[(cid, nid)] -> scalar tensor/float
#         c: Lorentz 曲率参数
#         scale: 映射到 Lorentz 双曲空间前缩放
#         device: 设备
#         reduction: "mean" 或 "sum"
#         sample_ratio: 从所有有效类对中随机采样的比例
#         max_pairs: 最多采样多少对,None 表示不限制

#     Returns:
#         loss: scalar tensor
#     """
#     current_classes = set(fused_proto_dict.keys())

#     # 1. 收集所有当前可用的 (cid, nid) 对
#     valid_pairs = []
#     for cid in current_classes:
#         if cid not in neighbor_dict:
#             continue
#         for nid in neighbor_dict[cid]:
#             if nid not in current_classes:
#                 continue
#             if (cid, nid) not in pre_angle_dict:
#                 continue
#             valid_pairs.append((cid, nid))

#     # 没有可用对
#     if len(valid_pairs) == 0:
#         return torch.tensor(0.0, device=device, requires_grad=True)

#     # 2. 随机采样
#     num_pairs = len(valid_pairs)

#     if sample_ratio < 1.0:
#         sampled_num = max(1, int(num_pairs * sample_ratio))
#     else:
#         sampled_num = num_pairs

#     if max_pairs is not None:
#         sampled_num = min(sampled_num, max_pairs)

#     if sampled_num < num_pairs:
#         sampled_pairs = random.sample(valid_pairs, sampled_num)
#     else:
#         sampled_pairs = valid_pairs

#     # 3. 预先把采样到的类别原型映射到 Lorentz，避免重复映射
#     needed_classes = set()
#     for cid, nid in sampled_pairs:
#         needed_classes.add(cid)
#         needed_classes.add(nid)

#     hyp_proto_dict = {}
#     for cls in needed_classes:
#         proto = fused_proto_dict[cls].to(device)
#         proto_hyp = exp_map_to_lorentz_space(
#             proto.unsqueeze(0), c=c, scale=scale
#         ).squeeze(0)
#         hyp_proto_dict[cls] = proto_hyp

#     # 4. 计算采样对的角度 loss
#     angle_losses = []
#     for cid, nid in sampled_pairs:
#         proto_i_hyp = hyp_proto_dict[cid]
#         proto_j_hyp = hyp_proto_dict[nid]

#         angle_cur = oxy_angle(proto_i_hyp, proto_j_hyp, c=c)

#         angle_pre = pre_angle_dict[(cid, nid)]
#         if not torch.is_tensor(angle_pre):
#             angle_pre = torch.tensor(angle_pre, device=device, dtype=proto_i_hyp.dtype)
#         else:
#             angle_pre = angle_pre.to(device=device, dtype=proto_i_hyp.dtype)

#         angle_losses.append((angle_cur - angle_pre) ** 2)

#     angle_losses = torch.stack(angle_losses)

#     if reduction == "sum":
#         return angle_losses.sum()
#     else:
#         return angle_losses.mean()

def calculate_sensitivity(D_orig, D_perturbed):
    """
    计算扰动前后距离变化的灵敏度度量
    D_orig: 原始距离矩阵
    D_perturbed: 扰动后的距离矩阵
    返回: Frobenius范数的变化量
    """
    return torch.norm(D_perturbed - D_orig, p='fro')

def calculate_relative_sensitivity(mat_original, mat_perturbed):
    delta = torch.norm(mat_perturbed - mat_original, p='fro')
    scale = torch.norm(mat_original, p='fro') + 1e-12  # 避免除0
    return delta / scale

def normalize_distance_matrix_by_offdiag_mean(D, eps=1e-12):
    """
    使用非对角线距离均值对距离矩阵做尺度归一化。
    D: [C, C] 距离矩阵
    """
    C = D.size(0)
    mask = ~torch.eye(C, dtype=torch.bool, device=D.device)

    offdiag_values = D[mask]
    scale = offdiag_values.mean().clamp_min(eps)

    return D / scale


def calculate_scale_normalized_fro_error(D_global, D_client, eps=1e-12):
    """
    计算尺度归一化后的 Frobenius 相对误差。
    先分别对 global 和 client 的距离矩阵做尺度归一化，
    再计算二者之间的 Frobenius 相对误差。
    """
    D_global_norm = normalize_distance_matrix_by_offdiag_mean(D_global, eps)
    D_client_norm = normalize_distance_matrix_by_offdiag_mean(D_client, eps)

    delta = torch.norm(D_client_norm - D_global_norm, p='fro')
    scale = torch.norm(D_global_norm, p='fro').clamp_min(eps)

    return delta / scale

def print_proto_distance_matrices_with_subset(pre_global_proto_by_class, fused_proto_dict, c1=0.5, device="cuda"):
    """
    计算fused_proto_dict和pre_global_proto_by_class两个原型子集之间的欧式距离矩阵和双曲测地距离矩阵，并计算它们的Frobenius范数变化量
    """
    # 设置torch的打印选项，避免科学计数法，显示更多的精度
    torch.set_printoptions(precision=6, sci_mode=False, threshold=10000)

    # 提取fused_proto_dict中类别的子集，确保从pre_global_proto_by_class中获取对应的原型
    classes_fused = sorted(fused_proto_dict.keys())  # 从fused_proto_dict中获取类别
    classes_full = sorted(pre_global_proto_by_class.keys())  # 从pre_global_proto_by_class中获取所有类别

    # 确保fused_proto_dict中的类别都在pre_global_proto_by_class中
    fused_proto_subset = {cls: pre_global_proto_by_class[cls].to(device) for cls in classes_fused if cls in pre_global_proto_by_class}
    
    # 将fused_proto_dict对应的原型组合为一个tensor [C, D]
    fused_proto_mat = torch.stack([fused_proto_dict[cls].to(device) for cls in classes_fused], dim=0)

    # 计算fused_proto_dict子集的欧式距离矩阵和双曲测地距离矩阵
    euc_dist_mat_fused = torch.cdist(fused_proto_mat, fused_proto_mat, p=2)   # [C, C]
    proto_hyp_fused = exp_map_to_lorentz_space(fused_proto_mat, c1=c1)   # [C, D]
    hyp_dist_mat_fused = lorentz_dist_matrix(proto_hyp_fused, c1=c1)     # [C, C]

    # 计算原始的欧式距离矩阵和双曲距离矩阵（对于pre_global_proto_by_class子集）
    proto_mat_full = torch.stack([pre_global_proto_by_class[cls].to(device) for cls in classes_fused], dim=0)
    euc_dist_mat_full = torch.cdist(proto_mat_full, proto_mat_full, p=2)   # [C, C]
    proto_hyp_full = exp_map_to_lorentz_space(proto_mat_full, c1=c1)   # [C, D]
    hyp_dist_mat_full = lorentz_dist_matrix(proto_hyp_full, c1=c1)     # [C, C]

    # =========================
    # 3. 计算Frobenius范数变化量
    # =========================
    # euc_sensitivity_fused = calculate_relative_sensitivity(euc_dist_mat_full, euc_dist_mat_fused)
    # hyp_sensitivity_fused = calculate_relative_sensitivity(hyp_dist_mat_full, hyp_dist_mat_fused)

    euc_scale_norm_error = calculate_scale_normalized_fro_error(euc_dist_mat_full, euc_dist_mat_fused)

    hyp_scale_norm_error = calculate_scale_normalized_fro_error(hyp_dist_mat_full, hyp_dist_mat_fused)

    # 输出结果
    # print("类别顺序（fused_proto_dict）:", classes_fused)
    # print("\nfused_proto_dict的原始欧式距离矩阵:")
    # print(euc_dist_mat_fused.detach().cpu())  # 打印fused_proto_dict的欧式距离矩阵

    # print("\nfused_proto_dict的原始Lorentz测地距离矩阵:")
    # print(hyp_dist_mat_fused.detach().cpu())  # 打印fused_proto_dict的双曲距离矩阵

    # print("\npre_global_proto_by_class的原始欧式距离矩阵:")
    # print(euc_dist_mat_full.detach().cpu())  # 打印pre_global_proto_by_class的欧式距离矩阵

    # print("\npre_global_proto_by_class的原始Lorentz测地距离矩阵:")
    # print(hyp_dist_mat_full.detach().cpu())  # 打印pre_global_proto_by_class的双曲距离矩阵

    # print(f"\nfused_proto_dict的欧式距离矩阵灵敏度: {euc_sensitivity_fused.item():.6f}")
    # print(f"fused_proto_dict的双曲测地距离矩阵灵敏度: {hyp_sensitivity_fused.item():.6f}")

    # # 对比敏感性
    # if hyp_sensitivity_fused > euc_sensitivity_fused:
    #     print("\n说明双曲流形空间对类间距离变化更为敏感")
    # else:
    #     print("\n说明欧式空间对类间距离变化更为敏感")

    # return classes_fused, euc_dist_mat_fused, hyp_dist_mat_fused, euc_sensitivity_fused, hyp_sensitivity_fused
    # return euc_sensitivity_fused, hyp_sensitivity_fused
    return euc_scale_norm_error, hyp_scale_norm_error


def train_net_fedavg(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args,device="cuda:0",memory=None,
missing_classes_per_client=None,round=None,
global_proto_by_class=None,
pre_global_proto_by_class=None,
# neighbor_dict=None, pre_angle_dict=None,
pre_dist_full=None,
pre_class_to_idx=None,
):
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

    last_epoch_feats = []
    last_epoch_labels = []
    memory = ClassMemoryQueue(num_classes=n_classes,feat_dim=512,queue_size=30,device=device)


    for epoch in range(epochs):
        euc_sensitivities = []
        hyp_sensitivities = []
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
            # x_mlp_norm = x_mlp / x_mlp.norm(dim=1, keepdim=True)
            # x_pre_norm = x_pre / x_pre.norm(dim=1, keepdim=True)

            # 先把当前batch的适配特征存进memory
            if memory is not None:
                memory.enqueue(x_mlp.detach(), target.detach())

            #===================================最初的方法==============================================#
            if args.method==0:
                fused_proto_dict = get_fused_prototypes(x_mlp, target, memory, alpha=0.5)
                loss_cls = criterion(out, target)
                loss_cls=loss_cls.to(device)
                loss=loss_cls
                # euc_sensitivity, hyp_sensitivity = print_proto_distance_matrices_with_subset(pre_global_proto_by_class, fused_proto_dict, c1=0.5, device=device)
                # euc_sensitivities.append(euc_sensitivity.item())
                # hyp_sensitivities.append(hyp_sensitivity.item())
            #===============================================================================#
            

            #===================================方法一：==============================================#
            if args.method==1:
                loss_local = spectral_loss(x_mlp,x_pre)
                loss_local=loss_local.to(device)
                loss_cls = criterion(out, target)
                loss_cls=loss_cls.to(device)
                loss=loss_cls+args.mu*loss_local
                print(f"Batch {batch_idx}: loss_local={loss_local.item():.6f}")
            #===============================================================================#



            #========================方法二：双曲空间的类别结构对齐====================================#
            if args.method==2:
                fused_proto_dict = get_fused_prototypes(x_mlp, target, memory, alpha=0.5)
                # # fused_proto_dict 里每个类别都是 [D] 的 tensor，可直接用来计算测地距离 loss
                # for cls, proto in fused_proto_dict.items():
                #     print(f"Class {cls} fused prototype shape: {proto.shape}")

                # loss_proto = euclidean_distance_preservation_loss(fused_proto_dict, pre_global_proto_by_class,device=device)
                # loss_proto_dir = relative_direction_loss_with_near_neighbors(fused_proto_dict, pre_global_proto_by_class, scale=0.05, device=device)
                # curv1 = net.get_curv1()
                # curv2 = net.get_curv2()
                # curv1 = net.get_curv1()

                curv1=0.5
                loss_proto = hyperbolic_distance_preservation_loss(fused_proto_dict, pre_global_proto_by_class,c1=curv1,device=device)
                loss_proto=loss_proto.to(device)

                loss_cls = criterion(out, target)
                loss_cls=loss_cls.to(device)

                # loss_proto_dir=loss_proto_dir.to(device)

                loss=loss_cls+args.nu*loss_proto

                # print(f"Batch {batch_idx}: curv1={curv1.item():.6f}")
                print(f"Batch {batch_idx}: loss_proto={loss_proto.item():.6f}")
                # print(f"Batch {batch_idx}: loss_proto_dir={loss_proto_dir.item():.6f}")
                # euc_sensitivity, hyp_sensitivity = print_proto_distance_matrices_with_subset(pre_global_proto_by_class, fused_proto_dict, c1=0.5, device=device)
                # euc_sensitivities.append(euc_sensitivity.item())
                # hyp_sensitivities.append(hyp_sensitivity.item())
            #================================================================================#


            #========================方法一：+方法二：====================================#
            if args.method==3:
                fused_proto_dict = get_fused_prototypes(x_mlp, target, memory, alpha=1)
                curv1=0.5
                # loss_proto = hyperbolic_distance_preservation_loss(fused_proto_dict, pre_global_proto_by_class,c1=curv1,device=device)
                loss_proto = hyperbolic_distance_preservation_loss_cached(
                    fused_proto_dict=fused_proto_dict,
                    pre_dist_full=pre_dist_full,
                    class_to_idx=pre_class_to_idx,
                    c1=curv1,
                    device=device
                )
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


            #==============================方法四============================================#
            # if args.method==4:
            #     fused_proto_dict = get_fused_prototypes(x_mlp, target, memory, alpha=0.5)
            #     loss_angle = hyperbolic_angle_preservation_loss(
            #         fused_proto_dict=fused_proto_dict,
            #         neighbor_dict=neighbor_dict,
            #         pre_angle_dict=pre_angle_dict,
            #         c=1.0,
            #         scale=0.05,
            #         device=device,
            #         reduction="mean",
            #     )
            #     loss_angle = loss_angle.to(device)
            #     loss_cls = criterion(out, target)
            #     loss_cls = loss_cls.to(device)
            #     loss = loss_cls + 0.5 * loss_angle
            #     print(f"Batch {batch_idx}: loss_angle={loss_angle.item():.6f}")
            #=================================================================================#


            
            #====================================方法五========================================#
            # if args.method==5:
            #     fused_proto_dict = get_fused_prototypes(x_mlp, target, memory, alpha=0.5)
            #     loss_proto = hyperbolic_distance_preservation_loss(fused_proto_dict, pre_global_proto_by_class, c=1.0,scale=0.1,device=device)
            #     loss_proto=loss_proto.to(device)

            #     loss_cls = criterion(out, target)
            #     loss_cls=loss_cls.to(device)

            #     # if round<10:
            #     #     loss_angle = hyperbolic_angle_preservation_loss(
            #     #         fused_proto_dict=fused_proto_dict,
            #     #         neighbor_dict=neighbor_dict,
            #     #         pre_angle_dict=pre_angle_dict,
            #     #         c=1.0,
            #     #         scale=0.1,
            #     #         device=device,
            #     #         reduction="mean",
            #     #     )
            #     #     # loss_angle = hyperbolic_angle_preservation_loss_sampled(
            #     #     #     fused_proto_dict=fused_proto_dict,
            #     #     #     neighbor_dict=neighbor_dict,
            #     #     #     pre_angle_dict=pre_angle_dict,
            #     #     #     c=1.0,
            #     #     #     scale=0.05,
            #     #     #     device=device,
            #     #     #     reduction="mean",
            #     #     #     sample_ratio=0.5,   # 只随机取 50% 的有效角度对
            #     #     #     max_pairs=8,        # 最多只算 8 对
            #     #     # )
            #     #     loss_angle = loss_angle.to(device)
            #     # else:
            #     #     loss_angle=torch.tensor(0.0, dtype=x_pre.dtype, device=x_pre.device)
            #     #     loss_angle = loss_angle.to(device)

            #     loss_angle = hyperbolic_angle_preservation_loss(
            #         fused_proto_dict=fused_proto_dict,
            #         neighbor_dict=neighbor_dict,
            #         pre_angle_dict=pre_angle_dict,
            #         c=1.0,
            #         scale=0.1,
            #         device=device,
            #         reduction="mean",
            #     )
            #     loss_angle = loss_angle.to(device)

            #     loss = loss_cls + 0.1 * loss_angle + 2*loss_proto
            #================================================================================#        


            #========================双曲空间的类别原型对齐====================================#
            # if global_proto_by_class is not None:
            #     x_hyperbolic_mlp = exp_map_to_poincare(x_mlp)
            #     x_hyperbolic_pre = exp_map_to_poincare(x_pre)

            #     batch_prototypes = torch.stack([global_proto_by_class[int(t.item())] for t in target])
            #     pre_batch_prototypes = torch.stack([pre_global_proto_by_class[int(t.item())] for t in target])

            #     # batch_prototypes_norm=batch_prototypes/batch_prototypes.norm(dim=1, keepdim=True)
            #     # pre_batch_prototypes_norm=pre_batch_prototypes/pre_batch_prototypes.norm(dim=1, keepdim=True)

            #     pre_batch_prototypes_hyperbolic=exp_map_to_poincare(pre_batch_prototypes)
            #     batch_prototypes_hyperbolic=exp_map_to_poincare(batch_prototypes)

            #     loss_proto = poincare_distance(x_hyperbolic_mlp, batch_prototypes_hyperbolic).mean()
            #     pre_loss_proto = poincare_distance(x_hyperbolic_mlp, pre_batch_prototypes_hyperbolic).mean()

            #     # print(x_hyperbolic_pre.norm(dim=1))          # 样本距离球心
            #     # print(x_hyperbolic_mlp.norm(dim=1))          # 样本距离球心
            #     # print(pre_batch_prototypes_hyperbolic.norm(dim=1)) # 预训练原型距离球心
            #     # print(batch_prototypes_hyperbolic.norm(dim=1))  # 本地原型距离球心
            # else:
            #     loss_proto=torch.tensor(0.0, dtype=x_pre.dtype, device=x_pre.device)
            #     pre_loss_proto=torch.tensor(0.0, dtype=x_pre.dtype, device=x_pre.device)

            # loss_cls = criterion(out, target)
            # loss=loss_cls+loss_proto

            # # print(f"Batch {batch_idx}: loss_proto={loss_proto.item():.6f}")
            # # print(f"Batch {batch_idx}: pre_loss_proto={pre_loss_proto.item():.6f}")
            #================================================================================#

            

            #===========================================================#
            # x_hyperbolic_mlp = exp_map_to_poincare(x_mlp)
            # x_hyperbolic_pre = exp_map_to_poincare(x_pre)

            # loss_proto = poincare_distance(x_hyperbolic_pre, x_hyperbolic_mlp).mean() 
            # loss_proto = F.mse_loss(x_pre, x_mlp)

            # loss_cls = criterion(out, target)
            # loss=loss_cls+loss_proto

            # print(f"Batch {batch_idx}: loss_proto={loss_proto.item():.6f}")
            #===========================================================#


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

            loss.backward()
            optimizer.step()
            # memory.enqueue(x_pre, target)
            cnt += 1
            epoch_loss_collector.append(loss.item())
            # --- 只保存最后一个 epoch 的特征 ---
            if epoch == epochs - 1:
                last_epoch_feats.append(x_mlp.detach().cpu())
                last_epoch_labels.append(target.detach().cpu())

        # avg_euc_sensitivity = sum(euc_sensitivities) / len(euc_sensitivities) if euc_sensitivities else 0
        # avg_hyp_sensitivity = sum(hyp_sensitivities) / len(hyp_sensitivities) if hyp_sensitivities else 0
        # print(f"Epoch {epoch + 1} - 平均欧式灵敏度: {avg_euc_sensitivity:.6f}, 平均双曲灵敏度: {avg_hyp_sensitivity:.6f}")
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    # if last_epoch_feats:  # 防止为空
    #     all_feats = torch.cat(last_epoch_feats, dim=0)   # shape: [num_samples, feat_dim]
    #     all_labels = torch.cat(last_epoch_labels, dim=0) # shape: [num_samples]
    # else:
    #     all_feats = torch.empty((0, net.feature_dim))   # 根据你的 net.feature_dim 设置
    #     all_labels = torch.empty((0,), dtype=torch.long)
    all_feats=None
    all_labels=None
    net.to('cpu')
    logger.info(' ** Training complete **')
    # return 0, 0
    return all_feats, all_labels

#------------------------------------------------------------------让每个客户端在自己的本地数据上训练-------------------------------------------------------------------------#
def compute_class_stats(feats, labels, device):
    """
    feats: [N, D] 训练后的残差特征
    labels: [N] 对应类别标签
    return: dict[class_id] -> {"mean": [D], "num": int, "var": [D]}
    """
    class_stats = {}
    unique_labels = torch.unique(labels)
    for cls in unique_labels:
        cls_mask = (labels == cls)
        cls_feats = feats[cls_mask]  # [num_cls_samples, D]
        cls_num = cls_feats.size(0)
        cls_mean = cls_feats.mean(dim=0)  # [D]
        cls_var = cls_feats.var(dim=0, unbiased=False)  # [D]，可以改为unbiased=True
        class_stats[int(cls.item())] = {
            "proto": cls_mean.detach().cpu(),
            "num": cls_num,
            "var": cls_var.detach().cpu()
        }
    return class_stats


# def log_map_zero(x, eps=1e-5):
#     """
#     Poincare Ball 的 log map, 参考点 o=0
#     x: [N, D] 双曲空间特征
#     返回: 切空间向量 [N, D]
#     """
#     norm_x = torch.norm(x, dim=-1, keepdim=True).clamp(min=eps)
#     return 0.5 * torch.log((1 + norm_x) / (1 - norm_x)) * x / norm_x

# def hyper_angle_loss(global_proto_by_class, pre_global_proto_by_class, classes, scale=0.05):
#     """
#     classes: list of class indices
#     返回: 每个类的方向漂移角度（弧度）
#     """
#     angle_list = []
#     for c in classes:
#         # 1. 取出预训练和适配后的原型
#         pc_pre = pre_global_proto_by_class[c].to(device)   # [D] 或 [1,D]
#         pc_mlp = global_proto_by_class[c].to(device)       # [D] 或 [1,D]

#         # 2. 投影到 Poincare Ball
#         pc_pre_hyp = exp_map_to_poincare(pc_pre, scale=scale)
#         pc_mlp_hyp = exp_map_to_poincare(pc_mlp, scale=scale)

#         # 3. 映射到球心切空间
#         vc_pre = log_map_zero(pc_pre_hyp)   # [D]
#         vc_mlp = log_map_zero(pc_mlp_hyp)   # [D]

#         # 4. 计算夹角
#         cos_theta = torch.sum(vc_pre * vc_mlp) / (torch.norm(vc_pre) * torch.norm(vc_mlp) + 1e-8)
#         # 防止数值误差导致 arccos 出错
#         cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
#         theta = torch.acos(cos_theta)  # 单位: 弧度

#         angle_list.append(theta)

#     # 转为 tensor
#     angle_tensor = torch.stack(angle_list)  # [num_classes]
#     return angle_tensor


# def hyper_radius_change(global_proto_by_class, pre_global_proto_by_class, classes, scale=0.05):
#     """
#     返回:
#         radius_pre_tensor: 每个类预训练双曲深度
#         radius_mlp_tensor: 每个类适配后双曲深度
#         delta_tensor:      每个类深度变化 (mlp - pre)
#         abs_delta_tensor:  每个类深度绝对变化
#     """
#     radius_pre_list = []
#     radius_mlp_list = []
#     delta_list = []
#     abs_delta_list = []

#     for c in classes:
#         pc_pre = pre_global_proto_by_class[c].to(device)
#         pc_mlp = global_proto_by_class[c].to(device)

#         # 映射到 Poincare Ball
#         pc_pre_hyp = exp_map_to_poincare(pc_pre, scale=scale)
#         pc_mlp_hyp = exp_map_to_poincare(pc_mlp, scale=scale)

#         # 映射到球心切空间
#         vc_pre = log_map_zero(pc_pre_hyp)
#         vc_mlp = log_map_zero(pc_mlp_hyp)

#         # 双曲深度 = log map 后向量长度
#         r_pre = torch.norm(vc_pre)
#         r_mlp = torch.norm(vc_mlp)

#         delta = r_mlp - r_pre
#         abs_delta = torch.abs(delta)

#         radius_pre_list.append(r_pre)
#         radius_mlp_list.append(r_mlp)
#         delta_list.append(delta)
#         abs_delta_list.append(abs_delta)

#     radius_pre_tensor = torch.stack(radius_pre_list)
#     radius_mlp_tensor = torch.stack(radius_mlp_list)
#     delta_tensor = torch.stack(delta_list)
#     abs_delta_tensor = torch.stack(abs_delta_list)

#     return radius_pre_tensor, radius_mlp_tensor, delta_tensor, abs_delta_tensor


def local_train_net(nets, args, net_dataidx_map, train_dl=None, test_dl=None, global_model = None, prev_model_pool = None, 
    prev_protos_pool=None, prev_protos_label_pool=None, server_c = None, clients_c = None, round=None, device="cuda:0",proto_bank_by_class=None,
    proto_bank_by_client=None,missing_classes_per_client=None,n_classes=None,
    global_proto_by_class=None,
    pre_global_proto_by_class=None,
    # neighbor_dict=None, pre_angle_dict=None,
    pre_dist_full=None,
    pre_class_to_idx=None,
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

    # if global_proto_by_class is not None:
    #     classes = list(global_proto_by_class.keys())  # 所有类别
    #     angle_tensor = hyper_angle_loss(global_proto_by_class, pre_global_proto_by_class, classes, scale=0.05)

    #     # 角向变化
    #     print("每个类的方向漂移角度（弧度）：", angle_tensor)
    #     print("平均方向漂移角度:", angle_tensor.mean())

    #     # 径向/层级变化
    #     r_pre, r_mlp, delta_r, abs_delta_r = hyper_radius_change(global_proto_by_class, pre_global_proto_by_class, classes, scale=0.05)
    #     print("每个类预训练层级深度:", r_pre)
    #     print("每个类适配后层级深度:", r_mlp)
    #     print("每个类层级变化 (mlp - pre):", delta_r)
    #     print("每个类层级绝对变化:", abs_delta_r)
    #     print("平均层级绝对变化:", abs_delta_r.mean())
        
    class_stats_by_client = {}
    for net_id, net in nets.items():
        # ipdb.set_trace()
        memory = ClassMemoryQueue(num_classes=n_classes,feat_dim=512,queue_size=256,device=device)

        real_feats  = proto_bank_by_client[net_id]["feats"]
        real_labels = proto_bank_by_client[net_id]["labels"]

        # num_real_samples = len(real_labels)
        # num_present_classes = len(torch.unique(real_labels))
        # avg_samples_per_class = num_real_samples // num_present_classes
        logger.info("Training network %s. n_training: %d" % (str(net_id), len(real_labels)))

        pseudo_feats=None
        pseudo_labels=None
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
        # for name, param in net.named_parameters():
        #     if name.startswith(("classifier", "img_adap", "mlp")):
        #         param.requires_grad = True

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

        feats, labels = train_net_fedavg(net_id, net, global_model, train_dl_local, None, n_epoch, args.lr,args.optimizer, args, 
                        device=device,memory=memory,
                        missing_classes_per_client=missing_classes_per_client,round=round,
                        global_proto_by_class=global_proto_by_class,
                        pre_global_proto_by_class=pre_global_proto_by_class,
                        # neighbor_dict=neighbor_dict, pre_angle_dict=pre_angle_dict,
                        pre_dist_full=pre_dist_full,
                        pre_class_to_idx=pre_class_to_idx,
                        )    
        # class_stats = compute_class_stats(feats, labels, device)
        # class_stats_by_client[net_id] = class_stats

    if global_model:
        global_model.to('cpu')
    if server_c:
        for param_index, param in enumerate(server_c.parameters()):
            server_c_collector[param_index] = new_server_c_collector[param_index]
        server_c.to('cpu')
    return nets,class_stats_by_client

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

def compute_all_clients_prototypes(nets, net_dataidx_map, args, n_classes, device):
    """
    返回：
      proto_bank_by_client: dict[client_id] -> {"proto": [Mi,d], "label":[Mi], "var":[Mi,d]}
      proto_bank_by_class:  dict[class_id]  -> list of {"client_id":id, "proto":[d], "var":[d]}
    """
    proto_bank_by_client = {}
    proto_bank_by_class = defaultdict(list)
    
    all_client_stats = []
    for client_id, net in nets.items():
        # ipdb.set_trace()
        dataidxs = net_dataidx_map[client_id]  #675
        train_dl_local, test_dl_global, train_ds_local, test_ds_global = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
        # len(train_dl_local)=675/64=10  len(test_dl_global)=3333/64=105
        #len(train_ds_local)=675    len(test_ds_global)=3333

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


# def build_pretrained_hyper_structure(pre_global_proto_by_class, scale=0.05, device="cuda"):
#     """
#     return:
#         pre_global_proto_hyp_by_class: 每个类别的双曲原型
#         pre_shared_center_euc: 欧式共享中心
#         pre_shared_center_hyp: 双曲共享中心
#         pre_depth_by_class: 每个类别相对于共享中心的深度
#         pre_branch_dir_by_class: 每个类别相对于共享中心的方向（欧式近似版）
#     """
#     # 1. 所有全局预训练类别原型堆起来
#     classes = sorted(list(pre_global_proto_by_class.keys()))
#     proto_list = []
#     for c in classes:
#         proto_list.append(pre_global_proto_by_class[c].to(device).float())
#     proto_stack = torch.stack(proto_list, dim=0)   # [C, D]

#     # 2. 欧式共享中心
#     pre_shared_center_euc = proto_stack.mean(dim=0)   # [D]

#     # 3. 双曲共享中心（第一版近似）
#     pre_shared_center_hyp = exp_map_to_poincare(pre_shared_center_euc.unsqueeze(0), scale=scale).squeeze(0)   # [D]

#     # 4. 每个类别映射到双曲空间
#     pre_global_proto_hyp_by_class = {}
#     pre_depth_by_class = {}
#     pre_branch_dir_by_class = {}

#     for c in classes:
#         p_euc = pre_global_proto_by_class[c].to(device).float()
#         p_hyp = exp_map_to_poincare(p_euc.unsqueeze(0), scale=scale).squeeze(0)

#         pre_global_proto_hyp_by_class[c] = p_hyp

#         # 5. 到共享中心的双曲深度
#         depth = poincare_distance(pre_shared_center_hyp, p_hyp)
#         pre_depth_by_class[c] = depth

#         # 6. 相对于共享中心的方向（第一版先用欧式近似）
#         branch_vec = p_euc - pre_shared_center_euc
#         branch_dir = branch_vec / (torch.norm(branch_vec) + 1e-8)
#         pre_branch_dir_by_class[c] = branch_dir

#     return (
#         pre_global_proto_hyp_by_class,
#         pre_shared_center_euc,
#         pre_shared_center_hyp,
#         pre_depth_by_class,
#         pre_branch_dir_by_class
#     )


# def build_parent_dict(pre_global_proto_hyp_by_class, pre_depth_by_class):
#     classes = sorted(list(pre_global_proto_hyp_by_class.keys()))
#     parent_dict = {}

#     for c in classes:
#         z_c = pre_global_proto_hyp_by_class[c]
#         r_c = pre_depth_by_class[c]

#         candidates = [j for j in classes if pre_depth_by_class[j] < r_c]
#         if len(candidates) == 0:
#             parent_dict[c] = None
#             continue

#         best_parent = None
#         best_dist = 1e9
#         for j in candidates:
#             z_j = pre_global_proto_hyp_by_class[j]
#             d = poincare_distance(z_c, z_j).item()
#             if d < best_dist:
#                 best_dist = d
#                 best_parent = j

#         parent_dict[c] = best_parent

#     return parent_dict


# def compute_angle_matrix(x):
#     """
#     x: [num_vectors, D]
#     返回: [num_vectors, num_vectors] 夹角矩阵，单位弧度
#     """
#     x_norm = x / x.norm(dim=-1, keepdim=True)  # 先归一化
#     cos_matrix = x_norm @ x_norm.t()           # 余弦相似度矩阵
#     # 防止浮点数误差导致 arccos 出现 nan
#     cos_matrix = torch.clamp(cos_matrix, -1.0, 1.0)
#     angle_matrix = torch.acos(cos_matrix)     # 弧度
#     return angle_matrix


# def oxy_angle(x, y, c=1.0, eps=1e-6):
#     """
#     计算双曲三角形 Oxy 在 x 点的角度
#     x: [N, D] 或 [D],Lorentz 双曲面上的 space part
#     y: [N, D] 或 [D],Lorentz 双曲面上的 space part
#     c: 正数，实际双曲曲率为 -c
#     return: [N] 或标量
#     """
#     if x.dim() == 1:
#         x = x.unsqueeze(0)
#     if y.dim() == 1:
#         y = y.unsqueeze(0)

#     c = torch.as_tensor(c, device=x.device, dtype=x.dtype)

#     x_time = torch.sqrt(1.0 / c + torch.sum(x ** 2, dim=-1))
#     y_time = torch.sqrt(1.0 / c + torch.sum(y ** 2, dim=-1))

#     # c_xyl = c * <x, y>_L
#     c_xyl = c * (torch.sum(x * y, dim=-1) - x_time * y_time)

#     acos_numer = y_time + c_xyl * x_time
#     acos_denom = torch.sqrt(torch.clamp(c_xyl ** 2 - 1.0, min=eps))

#     acos_input = acos_numer / (torch.norm(x, dim=-1) * acos_denom + eps)
#     angle = torch.acos(torch.clamp(acos_input, min=-1.0 + eps, max=1.0 - eps))

#     return angle.squeeze()


# def build_pretrained_neighbors_and_angles(pre_global_proto_by_class,k=3,c=1.0,scale=0.05,device="cuda"):
#     """
#     用预训练类别原型：
#     1) 定义每个类别的 top-k 近邻类别
#     2) 计算并保存每个 (class_i, neighbor_j) 的预训练双曲角度

#     Args:
#         pre_global_proto_by_class: dict[class_id] -> tensor[D]
#         k: 每个类别保留几个最近邻
#         c: Lorentz 曲率参数（实际曲率为 -c）
#         scale: exp map 前缩放
#         device: 设备

#     Returns:
#         neighbor_dict: dict[class_id] -> list[neighbor_class_id]
#         pre_angle_dict: dict[(class_i, class_j)] -> scalar tensor
#         pre_hyp_proto_dict: dict[class_id] -> tensor[D]  # 已映射到 Lorentz 的 space part
#         pre_dist_mat: [C, C]
#         classes: list[class_id]  # 矩阵下标到类 id 的映射
#     """
#     # ---- 1. 固定类别顺序 ----
#     classes = sorted(list(pre_global_proto_by_class.keys()))
#     if len(classes) <= 1:
#         return {}, {}, {}, None, classes

#     # ---- 2. 堆叠原型 ----
#     pre_mat = torch.stack(
#         [pre_global_proto_by_class[cid].to(device) for cid in classes],
#         dim=0
#     )  # [C, D]

#     # ---- 3. 映射到 Lorentz 双曲空间 ----
#     pre_hyp = exp_map_to_lorentz_space(pre_mat, c=c, scale=scale)  # [C, D]

#     # ---- 4. 计算测地距离矩阵 ----
#     pre_dist_mat = lorentz_dist_matrix(pre_hyp, c=c)  # [C, C]

#     # ---- 5. 保存每个类别的双曲原型 ----
#     pre_hyp_proto_dict = {
#         cid: pre_hyp[i].detach().clone()
#         for i, cid in enumerate(classes)
#     }

#     # ---- 6. 定义近邻 ----
#     neighbor_dict = {}
#     pre_angle_dict = {}

#     for i, cid in enumerate(classes):
#         row = pre_dist_mat[i].clone()

#         # 排除自己
#         row[i] = float("inf")

#         # 取 top-k 最近邻
#         nn_idx = torch.topk(row, k=min(k, len(classes)-1), largest=False).indices.tolist()
#         nn_classes = [classes[j] for j in nn_idx]
#         neighbor_dict[cid] = nn_classes

#         # ---- 7. 保存预训练角度 ----
#         x = pre_hyp[i]  # 类别 cid 的双曲原型
#         for j in nn_idx:
#             nid = classes[j]
#             y = pre_hyp[j]
#             angle_ij = oxy_angle(x, y, c=c)   # angle(O, x, y)
#             pre_angle_dict[(cid, nid)] = angle_ij.detach().clone()

#     return neighbor_dict, pre_angle_dict, pre_hyp_proto_dict, pre_dist_mat, classes


# def print_proto_distance_matrices(pre_global_proto_by_class, c1=0.5, device="cuda"):
#     """
#     pre_global_proto_by_class: dict[class_id] -> [D]
#     打印：
#         1. 类别顺序
#         2. 欧式距离矩阵
#         3. Lorentz 双曲测地距离矩阵
#     """
#     classes = sorted(pre_global_proto_by_class.keys())
#     if len(classes) == 0:
#         print("没有类别原型")
#         return

#     # [C, D]
#     proto_mat = torch.stack(
#         [pre_global_proto_by_class[cls].to(device) for cls in classes],
#         dim=0
#     )

#     # =========================
#     # 1. 欧式距离矩阵
#     # =========================
#     # 方法1：直接用 torch.cdist
#     euc_dist_mat = torch.cdist(proto_mat, proto_mat, p=2)   # [C, C]

#     # =========================
#     # 2. Lorentz 测地距离矩阵
#     # =========================
#     proto_hyp = exp_map_to_lorentz_space(proto_mat, c1=c1)   # [C, D]
#     hyp_dist_mat = lorentz_dist_matrix(proto_hyp, c1=c1)                  # [C, C]

#     print("类别顺序:", classes)
#     print("\n欧式距离矩阵:")
#     print(euc_dist_mat.detach().cpu())

#     print("\nLorentz 模型测地距离矩阵:")
#     print(hyp_dist_mat.detach().cpu())

#     return classes, euc_dist_mat, hyp_dist_mat

# def print_proto_distance_matrices(pre_global_proto_by_class, c1=0.5, device="cuda"):
#     """
#     pre_global_proto_by_class: dict[class_id] -> [D]
#     打印：
#         1. 类别顺序
#         2. 欧式距离矩阵
#         3. Lorentz 双曲测地距离矩阵
#     """
#     # 设置torch的打印选项，避免科学计数法，显示更多的精度
#     torch.set_printoptions(precision=6, sci_mode=False, threshold=10000)

#     classes = sorted(pre_global_proto_by_class.keys())
#     if len(classes) == 0:
#         print("没有类别原型")
#         return

#     # [C, D]
#     proto_mat = torch.stack(
#         [pre_global_proto_by_class[cls].to(device) for cls in classes],
#         dim=0
#     )

#     # =========================
#     # 1. 欧式距离矩阵
#     # =========================
#     # 方法1：直接用 torch.cdist
#     euc_dist_mat = torch.cdist(proto_mat, proto_mat, p=2)   # [C, C]

#     # =========================
#     # 2. Lorentz 测地距离矩阵
#     # =========================
#     proto_hyp = exp_map_to_lorentz_space(proto_mat, c1=c1)   # [C, D]
#     hyp_dist_mat = lorentz_dist_matrix(proto_hyp, c1=c1)                  # [C, C]

#     print("类别顺序:", classes)
#     print("\n欧式距离矩阵:")
#     print(euc_dist_mat.detach().cpu())  # 直接打印矩阵

#     print("\nLorentz 模型测地距离矩阵:")
#     print(hyp_dist_mat.detach().cpu())  # 直接打印矩阵

#     return classes, euc_dist_mat, hyp_dist_mat

def print_proto_distance_matrices(
    pre_global_proto_by_class, 
    c1=0.5, 
    device="cuda",
    save_path="./proto_distance_matrices.pt"
):
    """
    pre_global_proto_by_class: dict[class_id] -> [D]

    保存：
        classes: 类别顺序
        euc_dist_mat: 欧式距离矩阵
        hyp_dist_mat: 双曲测地距离矩阵
    """
    torch.set_printoptions(precision=6, sci_mode=False, threshold=10000)

    classes = sorted(pre_global_proto_by_class.keys())
    if len(classes) == 0:
        print("没有类别原型")
        return

    # [C, D]
    proto_mat = torch.stack(
        [pre_global_proto_by_class[cls].to(device) for cls in classes],
        dim=0
    )

    # 欧式距离矩阵
    euc_dist_mat = torch.cdist(proto_mat, proto_mat, p=2)

    # 双曲测地距离矩阵
    proto_hyp = exp_map_to_lorentz_space(proto_mat, c1=c1)
    hyp_dist_mat = lorentz_dist_matrix(proto_hyp, c1=c1)

    print("类别顺序:", classes)

    print("\n欧式距离矩阵:")
    print(euc_dist_mat.detach().cpu())

    print("\nLorentz 模型测地距离矩阵:")
    print(hyp_dist_mat.detach().cpu())

    # 保存结果
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(
        {
            "classes": classes,
            "euc_dist_mat": euc_dist_mat.detach().cpu(),
            "hyp_dist_mat": hyp_dist_mat.detach().cpu(),
        },
        save_path
    )

    print(f"\n距离矩阵已保存到: {save_path}")

    return classes, euc_dist_mat, hyp_dist_mat


def add_random_perturbation(proto_mat, sigma=0.1, device="cuda"):
    """
    对每个类别原型添加随机扰动
    proto_mat: 原始类别原型矩阵 [C, D]
    sigma: 高斯扰动的标准差
    返回：带有扰动的类别原型矩阵
    """
    perturbation = torch.normal(mean=0.0, std=sigma, size=proto_mat.size()).to(device)
    return proto_mat + perturbation

def calculate_sensitivity(D_orig, D_perturbed):
    """
    计算扰动前后距离变化的灵敏度度量
    D_orig: 原始距离矩阵
    D_perturbed: 扰动后的距离矩阵
    返回:Frobenius范数的变化量
    """
    return torch.norm(D_perturbed - D_orig, p='fro')

def calculate_relative_sensitivity(mat_original, mat_perturbed):
    delta = torch.norm(mat_perturbed - mat_original, p='fro')
    scale = torch.norm(mat_original, p='fro') + 1e-12  # 避免除0
    return delta / scale

# def generate_random_rotation_matrix(dim):
#     """
#     生成一个随机的正交矩阵作为旋转矩阵
#     dim: 旋转矩阵的维度，即类别原型的特征维度
#     返回：一个随机旋转矩阵
#     """
#     # 使用QR分解生成随机正交矩阵
#     random_matrix = torch.randn(dim, dim)
#     q, r = torch.qr(random_matrix)  # q 是一个正交矩阵，r 是上三角矩阵
#     # 对r进行符号调整，确保旋转矩阵是正交的
#     d = torch.diag(r.sign())
#     rotation_matrix = torch.matmul(q, d)  # 得到旋转矩阵
#     return rotation_matrix


# def apply_rotation_to_prototypes(pre_global_proto_by_class, rotation_matrix):
#     """
#     将旋转矩阵应用于pre_global_proto_by_class中的所有类别原型
#     pre_global_proto_by_class: 字典，键为类别ID，值为类别原型向量
#     rotation_matrix: 用于旋转的旋转矩阵
#     返回：旋转后的类别原型字典
#     """
#     #rotation_matrix.shape=512
#     rotated_prototypes = {}
#     # ipdb.set_trace()
#     for cls, prototype in pre_global_proto_by_class.items():
#         # 对类别原型应用旋转
#         #prototype.shape=512
#         rotated_prototype = torch.matmul(prototype, rotation_matrix)
#         rotated_prototypes[cls] = rotated_prototype
    
#     return rotated_prototypes

# def rotate_class_prototypes(pre_global_proto_by_class, dim):
#     """
#     对pre_global_proto_by_class中的类别原型应用旋转
#     pre_global_proto_by_class: 字典，键为类别ID，值为类别原型向量
#     dim: 旋转矩阵的维度，即类别原型的特征维度
#     返回：旋转后的类别原型字典
#     """
#     # 生成随机旋转矩阵
#     rotation_matrix = generate_random_rotation_matrix(dim)
    
#     # 对类别原型应用旋转
#     rotated_prototypes = apply_rotation_to_prototypes(pre_global_proto_by_class, rotation_matrix)
#     # ipdb.set_trace()
#     return rotated_prototypes

def print_proto_distance_matrices_with_perturbation(pre_global_proto_by_class, c1=0.5, sigma=0.1, device="cuda"):
    """
    在原始类别原型上添加扰动，计算扰动前后的欧式距离矩阵和双曲测地距离矩阵的变化量，并计算灵敏度
    """
    # 设置torch的打印选项，避免科学计数法，显示更多的精度
    torch.set_printoptions(precision=6, sci_mode=False, threshold=10000)

    classes = sorted(pre_global_proto_by_class.keys())
    if len(classes) == 0:
        print("没有类别原型")
        return

    # [C, D]
    proto_mat = torch.stack(
        [pre_global_proto_by_class[cls].to(device) for cls in classes],
        dim=0
    )

    # =========================
    # 1. 计算原始的欧式距离矩阵和双曲距离矩阵
    # =========================
    euc_dist_mat = torch.cdist(proto_mat, proto_mat, p=2)   # [C, C]
    proto_hyp = exp_map_to_lorentz_space(proto_mat, c1=c1)   # [C, D]
    hyp_dist_mat = lorentz_dist_matrix(proto_hyp, c1=c1)     # [C, C]

    # =========================
    # 2. 添加扰动并计算扰动后的距离矩阵
    # =========================
    proto_mat_perturbed = add_random_perturbation(proto_mat, sigma=sigma, device=device)
    # proto_mat_perturbed = rotate_class_prototypes(pre_global_proto_by_class, dim=proto_mat.size(1))
    # proto_mat_perturbed = torch.stack([proto_mat_perturbed[cls].to(device) for cls in classes], dim=0)

    # 计算扰动后的欧式距离矩阵和双曲测地距离矩阵
    euc_dist_mat_perturbed = torch.cdist(proto_mat_perturbed, proto_mat_perturbed, p=2)
    proto_hyp_perturbed = exp_map_to_lorentz_space(proto_mat_perturbed, c1=c1)
    hyp_dist_mat_perturbed = lorentz_dist_matrix(proto_hyp_perturbed, c1=c1)

    # # =========================
    # # 3. 计算扰动前后的灵敏度度量（Frobenius范数变化量）
    # # =========================
    # euc_sensitivity = calculate_sensitivity(euc_dist_mat, euc_dist_mat_perturbed)
    # hyp_sensitivity = calculate_sensitivity(hyp_dist_mat, hyp_dist_mat_perturbed)

    # =========================
    # 3. 计算扰动前后的灵敏度度量（相对Frobenius范数变化量）
    # =========================
    euc_sensitivity = calculate_relative_sensitivity(euc_dist_mat, euc_dist_mat_perturbed)
    hyp_sensitivity = calculate_relative_sensitivity(hyp_dist_mat, hyp_dist_mat_perturbed)

    # # 输出结果
    # print("类别顺序:", classes)
    # print("\n原始欧式距离矩阵:")
    # print(euc_dist_mat.detach().cpu())  # 打印原始欧式距离矩阵

    # print("\n原始Lorentz测地距离矩阵:")
    # print(hyp_dist_mat.detach().cpu())  # 打印原始Lorentz测地距离矩阵

    # print("\n扰动后的欧式距离矩阵:")
    # print(euc_dist_mat_perturbed.detach().cpu())  # 打印扰动后的欧式距离矩阵

    # print("\n扰动后的Lorentz测地距离矩阵:")
    # print(hyp_dist_mat_perturbed.detach().cpu())  # 打印扰动后的Lorentz测地距离矩阵

    print(f"\n欧式距离矩阵的灵敏度: {euc_sensitivity.item():.6f}")
    print(f"双曲测地距离矩阵的灵敏度: {hyp_sensitivity.item():.6f}")

    # 对比敏感性
    if hyp_sensitivity > euc_sensitivity:
        print("\n说明双曲流形空间对类间距离变化更为敏感")
    else:
        print("\n说明欧式空间对类间距离变化更为敏感")

    # return classes, euc_dist_mat, hyp_dist_mat, euc_sensitivity, hyp_sensitivity



def normalize_distance_matrix_by_offdiag_mean(D, eps=1e-12):
    """
    用非对角线距离均值归一化距离矩阵，避免欧式距离和双曲距离尺度不同。
    """
    C = D.size(0)
    mask = ~torch.eye(C, dtype=torch.bool, device=D.device)
    mean_dist = D[mask].mean().clamp_min(eps)
    return D / mean_dist


def sample_gromov_delta(D, num_samples=10000, seed=0):
    """
    近似计算距离矩阵的 Gromov delta-hyperbolicity。
    D: [C, C] 距离矩阵，建议输入归一化后的距离矩阵。
    """
    random.seed(seed)
    C = D.size(0)

    if C < 4:
        raise ValueError("至少需要4个类别才能计算 Gromov delta.")

    D_cpu = D.detach().cpu()
    deltas = []

    for _ in range(num_samples):
        a, b, c, d = random.sample(range(C), 4)

        s1 = D_cpu[a, b] + D_cpu[c, d]
        s2 = D_cpu[a, c] + D_cpu[b, d]
        s3 = D_cpu[a, d] + D_cpu[b, c]

        values = sorted([s1.item(), s2.item(), s3.item()])
        delta = (values[2] - values[1]) / 2.0

        deltas.append(delta)

    deltas = np.array(deltas)

    return {
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "p90_delta": float(np.percentile(deltas, 90)),
        "p95_delta": float(np.percentile(deltas, 95)),
        "max_delta": float(deltas.max()),
    }


def compare_tree_likeness(pre_global_proto_by_class, c1=0.5, device="cuda", num_samples=10000):
    """
    比较预训练全局类别原型在欧式空间和双曲空间中的树状/层级结构程度。
    """
    classes = sorted(pre_global_proto_by_class.keys())
    proto_mat = torch.stack([pre_global_proto_by_class[cls].to(device) for cls in classes], dim=0)
    with torch.no_grad():
        # 欧式距离矩阵
        D_euc = torch.cdist(proto_mat, proto_mat, p=2)

        # 双曲测地距离矩阵
        proto_hyp = exp_map_to_lorentz_space(proto_mat, c1=c1)
        D_hyp = lorentz_dist_matrix(proto_hyp, c1=c1)

        # 归一化，去除尺度影响
        D_euc_norm = normalize_distance_matrix_by_offdiag_mean(D_euc)
        D_hyp_norm = normalize_distance_matrix_by_offdiag_mean(D_hyp)

        # 计算 Gromov delta
        euc_delta = sample_gromov_delta(D_euc_norm, num_samples=num_samples, seed=0)
        hyp_delta = sample_gromov_delta(D_hyp_norm, num_samples=num_samples, seed=0)

    print("Euclidean delta:", euc_delta)
    print("Hyperbolic delta:", hyp_delta)

    return euc_delta, hyp_delta



def analyze_manual_four_classes(pre_global_proto_by_class,selected_classes,split_pair_1,split_pair_2,c1=0.5,device="cuda"):
    """
    手动选择四个类别，分析它们在欧式空间和双曲空间下的树状分组关系。
    selected_classes: 例如 [0, 1, 27, 28]
    split_pair_1: 例如 (0, 1)
    split_pair_2: 例如 (27, 28)
    """

    # 1. 提取四个类别的原型
    proto_mat = torch.stack([pre_global_proto_by_class[cls].to(device) for cls in selected_classes],dim=0)

    # 建立类别id到矩阵index的映射
    cls_to_idx = {cls: i for i, cls in enumerate(selected_classes)}

    a, b = split_pair_1
    c, d = split_pair_2

    ia, ib = cls_to_idx[a], cls_to_idx[b]
    ic, id_ = cls_to_idx[c], cls_to_idx[d]

    with torch.no_grad():
        # 2. 欧式距离矩阵
        D_euc = torch.cdist(proto_mat, proto_mat, p=2)

        # 3. 双曲测地距离矩阵
        proto_hyp = exp_map_to_lorentz_space(proto_mat, c1=c1)
        D_hyp = lorentz_dist_matrix(proto_hyp, c1=c1)

        # 4. 做尺度归一化，便于比较
        D_euc_norm = normalize_distance_matrix_by_offdiag_mean(D_euc)
        D_hyp_norm = normalize_distance_matrix_by_offdiag_mean(D_hyp)

    def analyze_one_space(D, space_name):
        dab = D[ia, ib].item()
        dcd = D[ic, id_].item()

        dac = D[ia, ic].item()
        dad = D[ia, id_].item()
        dbc = D[ib, ic].item()
        dbd = D[ib, id_].item()

        intra = (dab + dcd) / 2.0
        inter = (dac + dad + dbc + dbd) / 4.0
        inter_intra_ratio = inter / (intra + 1e-12)

        # 四点条件
        s1 = dab + dcd          # 组内配对距离和
        s2 = dac + dbd          # 跨组配对1
        s3 = dad + dbc          # 跨组配对2

        sorted_vals = sorted([s1, s2, s3])
        delta = (sorted_vals[2] - sorted_vals[1]) / 2.0

        # 如果假设 (a,b)|(c,d) 是正确树状分裂，那么希望 s1 最小，s2 和 s3 接近
        tree_gap = min(s2, s3) - s1

        result = {
            "space": space_name,
            "split": f"({a}, {b}) | ({c}, {d})",
            f"d({a},{b})": dab,
            f"d({c},{d})": dcd,
            f"d({a},{c})": dac,
            f"d({a},{d})": dad,
            f"d({b},{c})": dbc,
            f"d({b},{d})": dbd,
            "intra_mean": intra,
            "inter_mean": inter,
            "inter/intra_ratio": inter_intra_ratio,
            "s1_within": s1,
            "s2_cross": s2,
            "s3_cross": s3,
            "tree_gap": tree_gap,
            "delta": delta,
        }

        return result

    euc_result = analyze_one_space(D_euc_norm, "Euclidean")
    hyp_result = analyze_one_space(D_hyp_norm, "Hyperbolic")

    print("\n===== Euclidean Case =====")
    for k, v in euc_result.items():
        print(f"{k}: {v}")

    print("\n===== Hyperbolic Case =====")
    for k, v in hyp_result.items():
        print(f"{k}: {v}")

    return euc_result, hyp_result, D_euc_norm, D_hyp_norm


def extract_frozen_test_features_clip(clip_model, dataloader, device, save_path=None):
    """
    用冻结 CLIP 提取测试特征。
    输入 dataloader 仍然是原始图像 dataloader。
    输出 feats: [N, 512], labels: [N]
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
        print("Frozen test features saved to:", save_path)

    return all_feats, all_labels


def build_test_feature_loader_clip(clip_model, raw_test_loader, args, device, save_path, batch_size=32):
    """
    如果已经保存过测试特征，就直接加载；
    否则先用 CLIP 提取并保存。
    """
    if os.path.exists(save_path):
        cached = torch.load(save_path, map_location="cpu")
        test_feats = cached["feats"]
        test_labels = cached["labels"]
        print("Loaded cached test features from:", save_path)
    else:
        test_feats, test_labels = extract_frozen_test_features_clip(
            clip_model,
            raw_test_loader,
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


def build_pre_relation_cache(pre_global_proto_by_class, c1=0.5, device="cuda", save_path=None):
    """
    预先计算所有类别的预训练双曲类间距离矩阵。
    
    返回：
        pre_dist_full: [C, C] 所有类别的预训练距离矩阵
        class_to_idx: dict, 类别 id -> 矩阵下标
    """
    classes = sorted(list(pre_global_proto_by_class.keys()))
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    # 如果已经保存过，直接加载
    if save_path is not None and os.path.exists(save_path):
        cache = torch.load(save_path, map_location="cpu")

        if cache["classes"] == classes and abs(cache["c1"] - float(c1)) < 1e-12:
            pre_dist_full = cache["pre_dist_full"].to(device)
            return pre_dist_full, class_to_idx
        else:
            print("Cached pre relation does not match current classes or curvature. Recomputing...")

    with torch.no_grad():
        pre_mat = torch.stack(
            [pre_global_proto_by_class[cls].to(device) for cls in classes],
            dim=0
        )  # [C, D]

        pre_hyp = exp_map_to_lorentz_space(pre_mat, c1=c1)
        pre_dist_full = lorentz_dist_matrix(pre_hyp, c1=c1).detach()  # [C, C]

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(
            {
                "classes": classes,
                "class_to_idx": class_to_idx,
                "c1": float(c1),
                "pre_dist_full": pre_dist_full.detach().cpu(),
            },
            save_path
        )
        print("Pretrained relation matrix saved to:", save_path)

    return pre_dist_full, class_to_idx


def save_pairwise_proto_distances(
    pre_global_proto_by_class,
    c1=0.5,
    device="cuda",
    save_path="./relation_analysis/pairwise_distances.pt",
    save_csv=True
):
    """
    pre_global_proto_by_class: dict[class_id] -> Tensor [D]

    保存两两类别之间的距离：
        class_i
        class_j
        euclidean_distance
        hyperbolic_distance
    """

    torch.set_printoptions(precision=6, sci_mode=False, threshold=10000)

    classes = sorted(pre_global_proto_by_class.keys())

    if len(classes) == 0:
        print("没有类别原型")
        return None

    # [C, D]
    proto_mat = torch.stack(
        [pre_global_proto_by_class[cls].to(device) for cls in classes],
        dim=0
    )

    # =========================
    # 1. 欧式距离矩阵
    # =========================
    euc_dist_mat = torch.cdist(proto_mat, proto_mat, p=2)  # [C, C]

    # =========================
    # 2. 双曲测地距离矩阵
    # =========================
    proto_hyp = exp_map_to_lorentz_space(proto_mat, c1=c1)
    hyp_dist_mat = lorentz_dist_matrix(proto_hyp, c1=c1)  # [C, C]

    # =========================
    # 3. 保存两两类别距离
    # =========================
    pairwise_results = []

    C = len(classes)

    for i in range(C):
        for j in range(i + 1, C):
            pairwise_results.append({
                "class_i": classes[i],
                "class_j": classes[j],
                "euclidean_distance": euc_dist_mat[i, j].detach().cpu().item(),
                "hyperbolic_distance": hyp_dist_mat[i, j].detach().cpu().item(),
            })

    # =========================
    # 4. 保存为 .pt 文件
    # =========================
    save_dir = os.path.dirname(save_path)
    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)

    torch.save(pairwise_results, save_path)

    print(f"两两类别距离已保存到: {save_path}")

    # =========================
    # 5. 同时保存 csv，方便后面画图或查看
    # =========================
    if save_csv:
        csv_path = save_path.replace(".pt", ".csv")

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "class_i",
                "class_j",
                "euclidean_distance",
                "hyperbolic_distance"
            ])

            for item in pairwise_results:
                writer.writerow([
                    item["class_i"],
                    item["class_j"],
                    item["euclidean_distance"],
                    item["hyperbolic_distance"]
                ])

        print(f"CSV 文件已保存到: {csv_path}")

    return pairwise_results
    
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

    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, name_classes=None,beta=args.beta)
    # ipdb.set_trace()
    #X_train.shape=(50000,32,32,3)  y_train.shape=(50000,) X_test.shape=(10000,32,32,3)  y_test.shape=(10000,)
    #net_dataidx_map：dict: {client_id: 样本索引数组}
    #traindata_cls_counts：dict: {client → 各类别样本数}

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
    elif args.dataset == 'flowers102':
        n_classes=102
    elif args.dataset == 'ucf101':
        n_classes=101           

    missing_classes_per_client = {}
    for cid, cls_count in traindata_cls_counts.items():
        present_classes = set(cls_count.keys())      # 该客户端实际拥有的类
        all_classes = set(range(n_classes))           # 全部类
        missing_classes = sorted(all_classes - present_classes)
        missing_classes_per_client[cid] = missing_classes

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

    # ipdb.set_trace()
    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset, args.datadir,args.batch_size, 32)
    #train_dl_global:整个训练集   test_dl_global:整个测试集   train_ds_global:完整训练数据集对象  test_ds_global:完整测试数据集对象
    #len(train_ds_global)=50000   len(test_ds_global)=10000  
    #len(train_dl_global)=50000 / 64 = 781个batch   len(test_dl_global)=10000/32=313个batch

    test_feat_path = os.path.join(
        args.feat_logdir,
        "cached_test_features",
        f"{args.dataset}_test_clip_frozen_feats.pt"
    )

    test_feature_loader = build_test_feature_loader_clip(
        clip_model,
        test_dl_global,
        args,
        device,
        save_path=test_feat_path,
        batch_size=32
    )

    print("len train_dl_global:", len(train_ds_global))
    train_dl = None
    data_size = len(test_ds_global)

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

    proto_bank_by_client, proto_bank_by_class=compute_all_clients_prototypes(nets, net_dataidx_map, args, n_classes, device)
    # ipdb.set_trace()

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

        # curv1 = 0.5
        # pre_relation_path = os.path.join(
        #     args.logdir,
        #     "cached_pre_relations",
        #     f"{args.dataset}_pre_hyperbolic_relation_c{curv1}.pt"
        # )
        # pre_dist_full, pre_class_to_idx = build_pre_relation_cache(
        #     pre_global_proto_by_class=pre_global_proto_by_class,
        #     c1=curv1,
        #     device=device,
        #     save_path=pre_relation_path
        # )

        # pre_dist_full = pre_dist_full.to(device)
        # pre_dist_full = pre_dist_full.detach()
        
    round_time_list = []
    sync_cuda(device)
    round_start_time = time.perf_counter()

        # neighbor_dict, pre_angle_dict, pre_hyp_proto_dict, pre_dist_mat, classes = build_pretrained_neighbors_and_angles(
        #         pre_global_proto_by_class,
        #         k=3,
        #         c=1.0,
        #         scale=0.1,
        #         device=device
        #     )
        #每个类别的固定近邻:neighbor_dict[1] = [2, 3]
        #每个类别和它近邻的预训练角度:pre_angle_dict[(1, 2)] = tensor(...)  pre_angle_dict[(1, 3)] = tensor(...)

    #========================================类间角度=============================================#
    # class_ids = list(pre_global_proto_by_class.keys())
    # protos = torch.stack([pre_global_proto_by_class[c] for c in class_ids])  # [num_classes, D]
    # angles_euclidean = compute_angle_matrix(protos)
    # print("欧式空间类别原型两两夹角矩阵（弧度）:")
    # print(angles_euclidean)
    #=============================================================================================#


    #=========================================类间方差=================================================#
    # # global_protos: [num_classes, D] 
    # global_protos = torch.stack(list(pre_global_proto_by_class.values()))  # [num_classes, D]
    
    # scale = 0.5
    # scaled_protos = scale * global_protos               # 欧式空间缩放
    # # 欧式空间距离原点
    # euclidean_norms = scaled_protos.norm(p=2, dim=-1)  # [num_classes]
    # # 欧式空间径向方差
    # euclidean_radial_var = euclidean_norms.var(unbiased=False)
    # print(f"欧式空间全局原型到原点的径向方差: {euclidean_radial_var.item():.6f}")

    # # 先将全局原型投影到 Poincare 球
    # scale = 0.5
    # global_protos_h = exp_map_to_poincare(global_protos, scale=scale)
    # # 球心原点
    # origin = torch.zeros_like(global_protos_h)
    # # 计算每个全局原型到球心的测地距离
    # distances_h = poincare_distance(global_protos_h, origin)  # [num_classes]
    # # 双曲空间径向方差
    # hyperbolic_radial_var = distances_h.var(unbiased=False)
    # print(f"双曲空间全局原型到球心的测地距离方差: {hyperbolic_radial_var.item():.6f}")
    #===============================================================================================#


    #===============================================类内方差===========================================#
    # euclidean_var_by_class = {}
    # hyperbolic_var_by_class = {}

    # for c, items in proto_bank_by_class.items():
    #     proto_list = [item["proto"].float() for item in items]  # [num_clients, D]
    #     proto_stack = torch.stack(proto_list)  # [num_clients, D]
    
    #     scale = 0.05
    #     scaled_proto_stack = scale * proto_stack
    #     # 欧式模长方差
    #     norms = scaled_proto_stack.norm(p=2, dim=-1)
    #     euclidean_var_by_class[c] = norms.var(unbiased=False)

    #     # 投影到双曲
    #     proto_hyper = exp_map_to_poincare(proto_stack, scale=0.05)

    #     # 球心测地距离方差
    #     origin = torch.zeros_like(proto_hyper)
    #     distances = poincare_distance(proto_hyper, origin)
    #     hyperbolic_var_by_class[c] = distances.var(unbiased=False)


    # print("类别 | 欧式模长方差 | 双曲空间测地距离方差")
    # print("-"*40)

    # for c in euclidean_var_by_class.keys():
    #     e_var = euclidean_var_by_class[c].item()  # 转成 float
    #     h_var = hyperbolic_var_by_class[c].item()
    #     print(f"{c:5} | {e_var:10.6f} | {h_var:10.6f}")
    #=================================================================================================#


    #==================================================================================================#
    # classes, proto_euc, proto_hyp, dist_mat, graph = build_pretrained_hyper_complete_graph(pre_global_proto_by_class,scale=0.05,device=device)

    # (
    #     pre_global_proto_hyp_by_class,
    #     pre_shared_center_euc,
    #     pre_shared_center_hyp,
    #     pre_depth_by_class,
    #     pre_branch_dir_by_class
    # ) = build_pretrained_hyper_structure(pre_global_proto_by_class,scale=0.05,device=device)

    # # 打印每个类别深度
    # for c in sorted(pre_depth_by_class.keys()):
    #     print(f"class {c} depth = {pre_depth_by_class[c].item():.4f}")

    # parent_dict = build_parent_dict(pre_global_proto_hyp_by_class,pre_depth_by_class)
    # print("parent_dict =", parent_dict)
    # for c in sorted(parent_dict.keys()):
    #     print(f"class {c} -> parent {parent_dict[c]}")
    #================================================================================================#


    #========================================类间距离==============================================#
    # print_proto_distance_matrices(pre_global_proto_by_class,c1=0.5,device="cuda")
    # print_proto_distance_matrices_with_perturbation(pre_global_proto_by_class, c1=0.5, sigma=0.5, device="cuda")

    # classes, euc_dist_mat, hyp_dist_mat = print_proto_distance_matrices(
    #     pre_global_proto_by_class,
    #     c1=0.5,
    #     device="cuda",
    #     save_path="./relation_analysis/fgvcaircraft_dist_matrices.pt"
    # )

    # pairwise_results = save_pairwise_proto_distances(
    #     pre_global_proto_by_class,
    #     c1=0.5,
    #     device="cuda",
    #     save_path="./relation_analysis/fgvcaircraft_pairwise_distances.pt",
    #     save_csv=True
    # )

    #===================================================================================================#

    # euc_delta, hyp_delta = compare_tree_likeness(pre_global_proto_by_class,c1=0.5,device=device,num_samples=10000)
    # selected_classes = [49,50,69,70]
    # euc_result, hyp_result, D_euc_norm, D_hyp_norm = analyze_manual_four_classes(
    #     pre_global_proto_by_class=pre_global_proto_by_class,
    #     selected_classes=selected_classes,
    #     split_pair_1=(49,50),
    #     split_pair_2=(69,70),
    #     c1=0.5,
    #     device=device
    # )

    # ipdb.set_trace()
    global_proto_by_class=None

    # os.makedirs(args.logdir, exist_ok=True)
    # time_log_path = os.path.join(args.logdir, "communication_time_log.csv")
    # with open(time_log_path, "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow([
    #         "round",
    #         "download_time",
    #         "local_train_time",
    #         "aggregation_time",
    #         "round_total_time"
    #     ])


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


        # nets_this_round,class_stats_by_client = local_train_net(
        #     nets_this_round, args, net_dataidx_map,
        #     train_dl=train_dl, test_dl=test_dl_global,
        #     global_model=global_model, round=round, device=device,
        #     proto_bank_by_class=proto_bank_by_class, proto_bank_by_client=proto_bank_by_client,
        #     missing_classes_per_client=missing_classes_per_client,
        #     n_classes=n_classes,
        #     global_proto_by_class=global_proto_by_class,
        #     pre_global_proto_by_class=pre_global_proto_by_class,
        #     # neighbor_dict=neighbor_dict, pre_angle_dict=pre_angle_dict,
        # )

        nets_this_round, class_stats_by_client = local_train_net(
            nets_this_round, args, net_dataidx_map,
            train_dl=train_dl, test_dl=test_dl_global,
            global_model=global_model, round=round, device=device,
            proto_bank_by_class=proto_bank_by_class,
            proto_bank_by_client=proto_bank_by_client,
            missing_classes_per_client=missing_classes_per_client,
            n_classes=n_classes,
            global_proto_by_class=global_proto_by_class,
            pre_global_proto_by_class=pre_global_proto_by_class,
            pre_dist_full=None,
            pre_class_to_idx=None,
        )

        # global_proto_by_class = {}
        # # 先拿到所有出现过的类别
        # all_classes = set()
        # for client_id, class_stats in class_stats_by_client.items():
        #     all_classes.update(class_stats.keys())
        # # 对每个类别做加权平均
        # for cls in all_classes:
        #     total_num = 0
        #     weighted_sum = None
        #     for client_id, class_stats in class_stats_by_client.items():
        #         if cls not in class_stats:
        #             continue
        #         cls_mean = class_stats[cls]["proto"].to(device)   # [D]
        #         cls_num = class_stats[cls]["num"]
        #         if weighted_sum is None:
        #             weighted_sum = cls_mean * cls_num
        #         else:
        #             weighted_sum += cls_mean * cls_num
        #         total_num += cls_num
        #         global_proto_by_class[cls] = weighted_sum / total_num


        global_model.to('cpu')

        # update global model
        # global_nets_this_round = {k: nets[k] for k in global_party_list} 
        # total_data_points = sum([len(net_dataidx_map[r]) for r in global_party_list])
        # fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in global_party_list]

        # ipdb.set_trace()
        # global_nets_this_round = {k: nets[k] for k in party_list_this_round} 
        total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])  #当前训练的客户端的总的数据量：50000
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round] #各个客户端数据占总数据量的比例

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
        # ipdb.set_trace()
        global_model.to(device)


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

        logger.info('global n_training: %d' % len(train_dl_global))
        logger.info('global n_test: %d' % len(test_dl_global))

        #评估全局模型在训练集上的表现
        # train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
        #评估全局模型在测试集上的表现
        test_acc, _,test_loss = compute_accuracy_features(global_model, test_feature_loader, get_confusion_matrix=True, device=device)

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

        sync_cuda(device)
        round_time = time.perf_counter() - round_start_time
        round_time_list.append(round_time)
        logger.info(f"Round {round} communication training time: {round_time:.4f} seconds")



