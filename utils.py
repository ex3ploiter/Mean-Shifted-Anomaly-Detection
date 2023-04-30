import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import faiss
import torchvision.models as models
import torch.nn.functional as F
from PIL import ImageFilter
import random
from torchvision.transforms import InterpolationMode
from torchvision.datasets import  ImageFolder
from tqdm import tqdm

BICUBIC = InterpolationMode.BICUBIC

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


transform_color = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),])
                                    #   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform_resnet18 = transforms.Compose([
    transforms.Resize(224, interpolation=BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


moco_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),])
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class Transform:
    def __init__(self):
        self.moco_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),])
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __call__(self, x):
        x_1 = self.moco_transform(x)
        x_2 = self.moco_transform(x)
        return x_1, x_2


class Model(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        mu = torch.tensor(mean).view(3,1,1).cuda()
        std = torch.tensor(std).view(3,1,1).cuda()        
        self.norm = lambda x: ( x - mu ) / std

        if backbone == 152:
            self.backbone = models.resnet152(pretrained=True)
        else:
            self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = torch.nn.Identity()
        freeze_parameters(self.backbone, backbone, train_fc=False)

    def forward(self, x):
        x = self.norm(x)
        z1 = self.backbone(x)
        z_n = F.normalize(z1, dim=-1)
        return z_n

def freeze_parameters(model, backbone, train_fc=False):
    if not train_fc:
        for p in model.fc.parameters():
            p.requires_grad = False
    if backbone == 152:
        for p in model.conv1.parameters():
            p.requires_grad = False
        for p in model.bn1.parameters():
            p.requires_grad = False
        for p in model.layer1.parameters():
            p.requires_grad = False
        for p in model.layer2.parameters():
            p.requires_grad = False



def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)


def get_loaders_normal(dataset, label_class, batch_size, backbone):
    if dataset == "cifar10":
        ds = torchvision.datasets.CIFAR10
        transform = transform_color if backbone == 152 else transform_resnet18
        coarse = {}
        trainset = ds(root='../data', train=True, download=True, transform=transform, **coarse)
        testset = ds(root='../data', train=False, download=True, transform=transform, **coarse)
        trainset_1 = ds(root='../data', train=True, download=True, transform=Transform(), **coarse)
        
        
        idx = np.array(trainset.targets) == label_class
        testset.targets = [int(t != label_class) for t in testset.targets]
        trainset.data = trainset.data[idx]
        trainset.targets = [trainset.targets[i] for i, flag in enumerate(idx, 0) if flag]
        trainset_1.data = trainset_1.data[idx]
        trainset_1.targets = [trainset_1.targets[i] for i, flag in enumerate(idx, 0) if flag]
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2,
                                                   drop_last=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2,
                                                  drop_last=False)
        return train_loader, test_loader, torch.utils.data.DataLoader(trainset_1, batch_size=batch_size,
                                                                      shuffle=True, num_workers=2, drop_last=False)
    
    
    
    elif dataset == "BrainMRI" or dataset == "X-ray" or dataset == "Head-CT":    
        
        transform = transform_color if backbone == 152 else transform_resnet18
        coarse = {}

        if dataset == "BrainMRI" : # 2
            path1='/mnt/new_drive/Masoud_WorkDir/MeanShift_Tests/Training'
            path2='/mnt/new_drive/Masoud_WorkDir/MeanShift_Tests/Testing'
        elif dataset == "X-ray" : # 0
            path1='/mnt/new_drive/Sepehr/chest_xray/train'
            path2='/mnt/new_drive/Sepehr/chest_xray/test'

        elif dataset == "Head-CT" :# 1
            path1='/mnt/new_drive/Masoud_WorkDir/Test_Hamid/HEAD_CT/Train'
            path2='/mnt/new_drive/Masoud_WorkDir/Test_Hamid/HEAD_CT/Test'




        trainset = ImageFolder(root=path1, transform=transform)
        testset = ImageFolder(root=path2, transform=transform)
        trainset_1 = ImageFolder(root=path1, transform=Transform())

        indices = [i for i, val in enumerate(trainset.targets) if val==label_class]
        trainset = torch.utils.data.Subset(trainset, indices)        

        indices = [i for i, val in enumerate(trainset_1.targets) if val==label_class]
        trainset_1 = torch.utils.data.Subset(trainset_1, indices)        

        
        
        trainset.samples=[(pth,label_class) for (pth,target) in testset.samples ]
        trainset_1.samples=[(pth,label_class) for (pth,target) in testset.samples ]
        testset.samples=[(pth,int(target!=label_class)) for (pth,target) in testset.samples ]

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2,
                                                   drop_last=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=30, shuffle=False, num_workers=2,
                                                  drop_last=False)
        
        
        print(f'Len of trainset : {len(trainset)}')
        print(f'Len of testset : {len(testset)}')
        
        
        
        return train_loader, test_loader, torch.utils.data.DataLoader(trainset_1, batch_size=batch_size,
                                                                      shuffle=True, num_workers=2, drop_last=False)
    


    else:
        print('Unsupported Dataset')
        exit()


def get_loaders_blackbox(dataset, label_class, batch_size, backbone):
    if dataset == "cifar10":
        
        trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_resnet18)
        trainset.targets = [int(t!=label_class) for t in trainset.targets]

        testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_resnet18)
        testset.targets  = [int(t!=label_class) for t in testset.targets]

        ds=torch.utils.data.ConcatDataset([trainset, testset])
        train_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)

        return train_loader
        
    
    
    elif dataset == "BrainMRI" or dataset == "X-ray" or dataset == "Head-CT":    
        
        # transform = transform_color if backbone == 152 else transform_resnet18
        if dataset == "BrainMRI" : # 2
            path1='/mnt/new_drive/Masoud_WorkDir/MeanShift_Tests/Training'
            path2='/mnt/new_drive/Masoud_WorkDir/MeanShift_Tests/Testing'
        elif dataset == "X-ray" : # 0
            path1='/mnt/new_drive/Sepehr/chest_xray/train'
            path2='/mnt/new_drive/Sepehr/chest_xray/test'

        elif dataset == "Head-CT" :# 1
            path1='/mnt/new_drive/Masoud_WorkDir/Test_Hamid/HEAD_CT/Train'
            path2='/mnt/new_drive/Masoud_WorkDir/Test_Hamid/HEAD_CT/Test'
        
        
        trainset = ImageFolder(root=path1, transform=transform_resnet18)
        testset = ImageFolder(root=path2, transform=transform_resnet18)

        
        trainset.samples=[(pth,int(target!=label_class)) for (pth,target) in trainset.samples]
        testset.samples=[(pth,int(target!=label_class)) for (pth,target) in testset.samples ]




        ds=torch.utils.data.ConcatDataset([trainset, testset])
        train_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)

        return train_loader
        

    
    else:
        print('Unsupported Dataset')
        exit()        


class Wrap_Model(torch.nn.Module):
    def __init__(self, model, train_loader):
        super().__init__()

        self.model = model

        self.train_feature_space = []
        with torch.no_grad():
            for (imgs, _) in tqdm(train_loader, desc='Train set feature extracting'):
                imgs = imgs.to('cuda')
                features = self.model(imgs)
                self.train_feature_space.append(features.detach().cpu())
            self.train_feature_space = torch.cat(self.train_feature_space, dim=0).contiguous().cpu().numpy()

        self.mean_train = torch.mean(torch.Tensor(self.train_feature_space), axis=0)


    def forward(self, x):
        test_adversarial_feature_space = []
        features = self.model(x)
        test_adversarial_feature_space.append(features.detach().cpu())
        test_adversarial_feature_space = torch.cat(test_adversarial_feature_space).detach().cpu().numpy()
        distances = knn_score(self.train_feature_space, test_adversarial_feature_space)
        
        return distances        