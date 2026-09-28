import torch.utils.data as data
from PIL import Image
import numpy as np
import torchvision
from torchvision.datasets import MNIST, EMNIST, CIFAR10, CIFAR100, SVHN, FashionMNIST, ImageFolder, DatasetFolder, utils
from torch.utils.data import Dataset
import os
import os.path
import logging
import sys
import torch
import io
import scipy.io as matio
import ipdb
import pandas as pd
import random
from sklearn.model_selection import KFold


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def map_list(lst):
    unique_values = list(set(lst))  # 找出列表中的唯一值
    unique_values.sort()  # 对唯一值进行排序
    value_to_index = {value: index for index, value in enumerate(unique_values)}
    mapped_list = [value_to_index[value] for value in lst]
    return mapped_list

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def default_loader(image_path):
    return Image.open(image_path).convert('RGB')    
    
    
class MNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download


        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        mnist_dataobj = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = mnist_dataobj.train_data, np.array(mnist_dataobj.train_labels)
            else:
                data, target = mnist_dataobj.test_data, np.array(mnist_dataobj.test_labels)
        else:
            data = mnist_dataobj.data
            target = np.array(mnist_dataobj.targets)


        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0
                   
            

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        img = img.reshape(28,28,1).cpu().numpy()


        if self.transform is not None:
            img = self.transform(img)


        return img, target

    def __len__(self):
        return len(self.data)




class FashionMNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        fmnist_dataobj = FashionMNIST(self.root, self.train, self.transform, self.target_transform, self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = fmnist_dataobj.train_data, np.array(fmnist_dataobj.train_labels)
            else:
                data, target = fmnist_dataobj.test_data, np.array(fmnist_dataobj.test_labels)
        else:
            data = fmnist_dataobj.data
            target = np.array(fmnist_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class CIFAR10_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        # self.data, self.target = self.__build_truncated_dataset__()
        self.data, self.target, self.m_target = self.__build_truncated_dataset__()
        
    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        m_target = map_list(target)

        # return data, target
        return data, target, m_target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        m_target = self.m_target[index]
        # img = Image.fromarray(img)
        # print("cifar10 img:", img)
        # print("cifar10 target:", target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target
        return img, target, m_target

    def __len__(self):
        return len(self.data)


class CIFAR100_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        # self.data, self.target = self.__build_truncated_dataset__()
        self.data, self.target, self.m_target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR100(self.root, self.train, self.transform, self.target_transform, self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        # return data, target
        m_target = map_list(target)

        return data, target, m_target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        m_target = self.m_target[index]
        img = Image.fromarray(img)
        # print("cifar10 img:", img)
        # print("cifar10 target:", target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target
        return img, target, m_target

    def __len__(self):
        return len(self.data)

class TinyImageNet_load(Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")
        self.dataidxs = dataidxs
        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        
        if self.dataidxs is not None:
            self.samples = self.images[dataidxs]
        else:
            self.samples = self.images

#         print('samples.shape', self.samples.shape)
        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
#         print(self.tgt_idx_to_class)
    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(val_image_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)
        self.images = np.array(self.images)
#         print('dataset.shape', self.images.shape)
    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]
    

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        img_path, tgt = self.samples[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        tgt = int(tgt)
        return sample, tgt     


class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)

        
class Food101_truncated(torch.utils.data.Dataset):
    def __init__(self, dataidxs=None, transform=None, loader=default_loader, mode = None):

        image_path = '/Food101_Image/images/'
        data_path = '/Food101_Text/'
        if mode == 'train':
            with io.open(data_path + 'train_images.txt', encoding='utf-8') as file:
                path_to_images = file.read().split('\n')[:-1] #list-len:68175
            with io.open(data_path + 'train_labels.txt', encoding='utf-8') as file:
                labels = file.read().split('\n')[:-1] #list-len:68175

        elif mode == 'test':

            with io.open(data_path + 'test_images.txt', encoding='utf-8') as file:
                path_to_images = file.read().split('\n')[:-1] #list-len:25250
            with io.open(data_path + 'test_labels.txt', encoding='utf-8') as file:
                labels = file.read().split('\n')[:-1] #list-len:25250

        elif mode == 'val':
            with io.open(data_path + 'val_images.txt', encoding='utf-8') as file:
                path_to_images = file.read().split('\n')[:-1]
            with io.open(data_path + 'val_labels.txt', encoding='utf-8') as file:
                labels = file.read().split('\n')[:-1]

        else:
            assert 1<0, 'Please fill mode with any of train/val/test to facilitate dataset creation'

        #import ipdb; ipdb.set_trace()
        if mode == 'train' and dataidxs != None:
#             print('xxxxxxxxx', path_to_images)
            self.image_path = image_path
#             ipdb.set_trace()
            self.path_to_images = np.array(path_to_images)[dataidxs]
            self.labels = np.array(labels, dtype=int)[dataidxs]
            print('mode:', mode, 'len(path_to_images):', len(self.path_to_images))
            
        else:
            self.image_path = image_path
            self.path_to_images = path_to_images
            self.labels = np.array(labels, dtype=int)
            
        self.transform = transform
        self.loader = loader
        self.mode = mode
        

    def __getitem__(self, index):
        # get image matrix and transform to tensor
        path = self.path_to_images[index]
        img = self.loader(self.image_path + path + '.jpg')

        if self.transform is not None:
            img = self.transform(img)
        
        # get label
        label = self.labels[index]
            
        return img, label


    def __len__(self):
        return len(self.path_to_images)    

    
class Vireo172_truncated(torch.utils.data.Dataset):
    def __init__(self, dataidxs=None, transform=None, loader=default_loader, mode = None):

        image_path = '/Vireo172_Image/ready_chinese_food/'
        data_path =  '/Vireo172_Text/SplitAndIngreLabel/'
        
        if mode == 'train':
            
            with io.open(data_path + 'TR.txt', encoding='utf-8') as file:
                path_to_images = file.read().split('\n')[:-1]
            labels = matio.loadmat(data_path + 'train_label.mat')['train_label'][0]

        elif mode == 'test':
            
            with io.open(data_path + 'TE.txt', encoding='utf-8') as file:
                path_to_images = file.read().split('\n')[:-1]
            labels = matio.loadmat(data_path + 'test_label.mat')['test_label'][0]

        elif mode == 'val':

            with io.open(data_path + 'VAL.txt', encoding='utf-8') as file:
                path_to_images = file.read().split('\n')[:-1]
            labels = matio.loadmat(data_path + 'val_label.mat')['validation_label'][0]

        else:
            assert 1<0, 'Please fill mode with any of train/val/test to facilitate dataset creation'

        #import ipdb; ipdb.set_trace()
      
        if mode == 'train' and dataidxs != None:
#             print('xxxxxxxxx', path_to_images)
            self.image_path = image_path
            self.path_to_images = np.array(path_to_images)[dataidxs]
            self.labels = np.array(labels, dtype=int)[dataidxs]-1
            print('mode:', mode, 'len(path_to_images):', len(self.path_to_images))
            
        else:
            self.image_path = image_path
            self.path_to_images = path_to_images
            self.labels = np.array(labels, dtype=int)-1
        

        self.transform = transform
        self.loader = loader

        
    def __getitem__(self, index):
        # get image matrix and transform to tensor
        path = self.path_to_images[index]

        img = self.loader(self.image_path + path)

        if self.transform is not None:
            img = self.transform(img)
        
        # get label
        label = self.labels[index]

        #change vireo labels from 1-indexed to 0-indexed values

            
        return [img, label]

    def __len__(self):
        return len(self.path_to_images)  


class DTD(torch.utils.data.Dataset):
    def __init__(self, dataidxs=None, transform=None, mode='train'):
        """
        Make a dataset from the csv.
        :param csv_dir: directory of csv of the img(train/valid/test) fold.
        :param transform: transform for img.
        """
        
        if mode == 'train':
            csv_dir = '../datasets/DTD/train.csv'
        elif mode == 'test':
            csv_dir = '../datasets/DTD/test.csv'
            
        self.csv_info = pd.read_csv(csv_dir).values
        print(self.csv_info)
        self.path_to_images = self.csv_info[:, 0]
        self.labels = self.csv_info[:, 1]
        self.m_labels = self.csv_info[:, 1]
        self.transform = transform
        
        if mode == 'train' and dataidxs != None:
            self.path_to_images = self.path_to_images[dataidxs]
            self.labels = self.labels[dataidxs]
            self.m_labels = map_list(self.labels)
            print('mode:', mode, 'len(path_to_images):', len(self.path_to_images))
        

    def __getitem__(self, index):
        img_dir, label, m_label = self.path_to_images[index], self.labels[index], self.m_labels[index]
        img = Image.open(img_dir).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            # print(img.shape)
        return img, label, m_label

    def __len__(self):
        return len(self.path_to_images)       


class Caltech101(torch.utils.data.Dataset):
    def __init__(self, dataidxs=None, transform=None, mode='train'):
        """
        Make a dataset from the csv.
        :param csv_dir: directory of csv of the img(train/valid/test) fold.
        :param transform: transform for img.
        """
        
        if mode == 'train':
            csv_dir = '../datasets/Caltech101/train.csv'
        elif mode == 'test':
            csv_dir = '../datasets/Caltech101/test.csv'
         
        self.csv_info = pd.read_csv(csv_dir).values
        print(self.csv_info)
        self.path_to_images = self.csv_info[:, 0]
        self.labels = self.csv_info[:, 1]
        self.transform = transform
        self.m_labels = self.csv_info[:, 1]
        
        if mode == 'train' and dataidxs != None:
            self.path_to_images = self.path_to_images[dataidxs]
            self.labels = self.labels[dataidxs]
            self.m_labels = map_list(self.labels)
            print('mode:', mode, 'len(path_to_images):', len(self.path_to_images))
        

    def __getitem__(self, index):
        img_dir, label, m_label = self.path_to_images[index], self.labels[index], self.m_labels[index]
        
        img = Image.open(img_dir).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
            # print(img.shape)
        return img, label, m_label

    def __len__(self):
        return len(self.path_to_images)      
        

class Air100(torch.utils.data.Dataset):
    def __init__(self, dataidxs=None, transform=None, mode='train'):
        """
        Make a dataset from the csv.
        :param csv_dir: directory of csv of the img(train/valid/test) fold.
        :param transform: transform for img.
        """
        if mode == 'train':
            csv_dir = '../datasets/FGVC_Aircraft/train.csv'
        elif mode == 'test':
            csv_dir = '../datasets/FGVC_Aircraft/test.csv'
            
        self.csv_info = pd.read_csv(csv_dir).values
        self.path_to_images = self.csv_info[:, 0]
        self.labels = self.csv_info[:, 1]
        self.m_labels = self.csv_info[:, 1]
        self.transform = transform
        
        if mode == 'train' and dataidxs != None:
            self.path_to_images = self.path_to_images[dataidxs]
            self.labels = self.labels[dataidxs]
            self.m_labels = map_list(self.labels)
            print('mode:', mode, 'len(path_to_images):', len(self.path_to_images))
        

    def __getitem__(self, index):
        img_dir, label, m_label = self.path_to_images[index], self.labels[index], self.m_labels[index]
        
        img = Image.open(img_dir).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
            # print(img.shape)
        return img, label, m_label

    def __len__(self):
        return len(self.path_to_images)   


class Cars196(torch.utils.data.Dataset):
    def __init__(self, dataidxs=None, transform=None, mode='train'):
        """
        Make a dataset from the csv.
        :param csv_dir: directory of csv of the img(train/valid/test) fold.
        :param transform: transform for img.
        """
        
        if mode == 'train':
            csv_dir = '../datasets/Stanford_Cars/train.csv'
        elif mode == 'test':
            csv_dir = '../datasets/Stanford_Cars/test.csv'           
        self.csv_info = pd.read_csv(csv_dir).values
        # print(self.csv_info)
        self.path_to_images = self.csv_info[:, 0]
        self.labels = self.csv_info[:, 1]
        self.m_labels = self.csv_info[:, 1]
        self.transform = transform
        
        if mode == 'train' and dataidxs != None:
            self.path_to_images = self.path_to_images[dataidxs]
            self.labels = self.labels[dataidxs]
            self.m_labels = map_list(self.labels)
            print('mode:', mode, 'len(path_to_images):', len(self.path_to_images))
        

    def __getitem__(self, index):
        img_dir, label, m_label = self.path_to_images[index], self.labels[index], self.m_labels[index]
        
        img = Image.open(img_dir).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
            # print(img.shape)
        return img, label, m_label

    def __len__(self):
        return len(self.path_to_images)  


class Flowers102(torch.utils.data.Dataset):
    def __init__(self, dataidxs=None, transform=None, mode='train'):
        """
        Make a dataset from the csv.
        :param csv_dir: directory of csv of the img(train/valid/test) fold.
        :param transform: transform for img.
        """
            
        if mode == 'train':
            csv_dir = '../datasets/Flowers102/train.csv'
        elif mode == 'test':
            csv_dir = '../datasets/Flowers102/test.csv'        
            
            
        self.csv_info = pd.read_csv(csv_dir).values
        self.path_to_images = self.csv_info[:, 0]
        self.labels = self.csv_info[:, 1] - 1
        self.m_labels = self.csv_info[:, 1] - 1
        self.transform = transform
        
        if mode == 'train' and dataidxs != None:
            self.path_to_images = self.path_to_images[dataidxs]
            self.labels = self.labels[dataidxs]
            self.m_labels = map_list(self.labels)
            print('mode:', mode, 'len(path_to_images):', len(self.path_to_images))
        

    def __getitem__(self, index):
        img_dir, label, m_label = self.path_to_images[index], self.labels[index], self.m_labels[index]
        img = Image.open(img_dir).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label, m_label

    def __len__(self):
        return len(self.path_to_images)    



class UCF101(torch.utils.data.Dataset):
    def __init__(self, dataidxs=None, transform=None, mode='train'):
        """
        Make a dataset from the csv.
        :param csv_dir: directory of csv of the img(train/valid/test) fold.
        :param transform: transform for img.
        """
        
        if mode == 'train':
            csv_dir = '../datasets/UCF101/train.csv'
        elif mode == 'test':
            csv_dir = '../datasets/UCF101/test.csv'
          
        self.csv_info = pd.read_csv(csv_dir).values
        print(self.csv_info)
        self.path_to_images = self.csv_info[:, 0]

        self.labels = self.csv_info[:, 1]
        self.m_labels = self.csv_info[:, 1]
        self.transform = transform
        
        if mode == 'train' and dataidxs != None:
            self.path_to_images = self.path_to_images[dataidxs]
            self.labels = self.labels[dataidxs]
            self.m_labels = map_list(self.labels)
            print('mode:', mode, 'len(path_to_images):', len(self.path_to_images))
        

    def __getitem__(self, index):
        img_dir, label, m_label = self.path_to_images[index], self.labels[index], self.m_labels[index]
        img = Image.open(img_dir).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label, m_label

    def __len__(self):
        return len(self.path_to_images)   


class Domain_G(torch.utils.data.Dataset):
    def __init__(self, dataset, domain='amazon', transform=None, dataidxs=None):
        """
        初始化Office数据集类,加载数据集路径和域、类别信息
        :param dataset_path: 数据集的根目录路径
        """
        self.dataset = dataset
        if dataset == "domain_office":
            self.dataset_path = '../datasets/Office_Home/'
        elif dataset == "domain_vlcs":
            self.dataset_path = '../datasets/VLCS/'
        elif dataset == "domain_pacs":
            self.dataset_path = '../datasets/PACS/'
        elif dataset == 'domain_net':
            self.dataset_path = '../datasets/DomainNet/'
        self.domain = domain  # 存储域名称
        self.dataidxs = dataidxs
        self.classes = []  # 存储类别名称
        self.class_to_idx = {}  # 类别名到标签的映射
        self.data = []  # 存储图像数据
        self.labels = []  # 存储对应标签
        self.transform = transform
        # 加载数据集
        self._load_data()


    def _filter_bad_images(self):
        clean_data = []
        clean_labels = []

        for image_path, label in zip(self.data, self.labels):
            try:
                # img = Image.open(image_path)
                # img.verify()   # 验证图片完整性
                # clean_data.append(image_path)
                # clean_labels.append(label)
                img = Image.open(image_path).convert("RGB")
                clean_data.append(image_path)
                clean_labels.append(label)
            except Exception as e:
                print(f"[坏图跳过] {image_path}")
                print(f"[坏图标签] label={label}, error={e}")

        self.data = clean_data
        self.labels = clean_labels

    def _load_data(self):
        """
        加载数据集的图像和标签，保存到数据列表中
        """
        # 获取所有域的文件夹（即子目录）,先排序
        domain_folder = os.path.join(self.dataset_path, self.domain)
        # 遍历域中的所有类别
        if self.dataset == "domain_office":
            class_names = sorted([x for x in os.listdir(domain_folder) if x != '.ipynb_checkpoints'])  #所有类别
        elif self.dataset == 'domain_net':
            class_names = sorted([x for x in os.listdir(domain_folder) if x != '.ipynb_checkpoints'])[:20]  #前20个类别
        elif self.dataset == "domain_vlcs":
            class_names = sorted([x for x in os.listdir(domain_folder) if x != '.ipynb_checkpoints'])  #所有类别
        elif self.dataset == "domain_pacs":
            class_names = sorted([x for x in os.listdir(domain_folder) if x != '.ipynb_checkpoints'])  #所有类别
        # print(class_names)
        for class_name in class_names:
            class_folder = os.path.join(domain_folder, class_name)
            if os.path.isdir(class_folder):
                # 为每个类别分配标签
                self.classes.append(class_name)
                self.class_to_idx[class_name] = len(self.classes) - 1
                # 遍历类别中的所有图像文件
                for image_name in os.listdir(class_folder):
                    image_path = os.path.join(class_folder, image_name)
                    if image_path.endswith(('.jpg', '.png', '.jpeg')):  # 只考虑图像文件
                        self.data.append(image_path)
                        self.labels.append(self.class_to_idx[class_name])
                        
        indices = list(range(len(self.data)))
        random.seed(2023)
        random.shuffle(indices)
        
        self.data = [self.data[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        # ipdb.set_trace()
        self._filter_bad_images()
        # 将数据划分为 k 份
        if self.dataset == 'domain_net':
            kk = 3
        if self.dataset == 'domain_office':
            kk = 5
        if self.dataset == 'domain_vlcs':
            kk = 5
        if self.dataset == 'domain_pacs':
            kk = 5

        kf = KFold(n_splits=kk, shuffle=True, random_state=42)  # KFold 进行划分
        self.folds = []  # 存储 k 份数据
        for _, test_idx in kf.split(self.data):
            fold_data = [self.data[i] for i in test_idx]
            fold_labels = [self.labels[i] for i in test_idx]
            m_labels = map_list(fold_labels)
            self.folds.append((fold_data, fold_labels, m_labels))

    def __getitem__(self, index):
        """
        获取指定索引的图像、类别标签和域标签
        :param index: 索引
        :return: 图像、类别标签和域标签
        """
        image_path = self.data[index]
        label = self.labels[index]
        img = Image.open(image_path).convert('RGB')
        # try:
        #     image = Image.open(img_path).convert("RGB")
        # except Exception as e:
        #     print(f"[Error] failed to load image: {image_path}")
        #     print(f"[Error] idx: {index}, label: {label}, error: {e}")
        #     raise e

        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]

        return img, label, label        

    def __len__(self):
        """
        返回数据集的大小
        """
        return len(self.data)    

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        自定义数据集
        :param data: 图像路径列表
        :param labels: 对应的标签列表
        :param transform: 图像变换（可选）
        """
        self.data = data
        self.labels = labels
        self.transform = transform
        self._filter_bad_images()

    def _filter_bad_images(self):
        clean_data = []
        clean_labels = []

        for img_path, label in zip(self.data, self.labels):
            try:
                # img = Image.open(img_path)
                # img.verify()
                # clean_data.append(img_path)
                # clean_labels.append(label)
                img = Image.open(img_path).convert("RGB")
                clean_data.append(img_path)
                clean_labels.append(label)
            except Exception as e:
                print(f"[坏图跳过] {img_path}")
                print(f"[坏图标签] label={label}, error={e}")

        self.data = clean_data
        self.labels = clean_labels
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        # 读取图像
        image = Image.open(img_path).convert("RGB")  # 确保是RGB格式
        # try:
        #     image = Image.open(img_path).convert("RGB")
        # except Exception as e:
        #     print(f"[Error] failed to load image: {img_path}")
        #     print(f"[Error] idx: {idx}, label: {label}, error: {e}")
        #     raise e
        
        if self.transform:
            image = self.transform(image)  # 进行变换
        
        return image, label, label   