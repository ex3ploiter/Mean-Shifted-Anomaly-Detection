import torch
from sklearn.metrics import roc_auc_score,accuracy_score
import torch.optim as optim
import argparse
import utils
from tqdm import tqdm
import torch.nn.functional as F
import torchvision
import torch.nn as nn
import torchattacks

def contrastive_loss(out_1, out_2):
    out_1 = F.normalize(out_1, dim=-1)
    out_2 = F.normalize(out_2, dim=-1)
    bs = out_1.size(0)
    temp = 0.25
    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temp)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * bs, device=sim_matrix.device)).bool()
    # [2B, 2B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * bs, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temp)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss


def train_model(model, train_loader, test_loader, train_loader_1, device, args):
    model.eval()
    auc, feature_space = get_score(model, device, train_loader, test_loader)
    print('Epoch: {}, AUROC is: {}'.format(0, auc))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.00005)
    center = torch.FloatTensor(feature_space).mean(dim=0)
    if args.angular:
        center = F.normalize(center, dim=-1)
    center = center.to(device)
    for epoch in range(args.epochs):
        running_loss = run_epoch(model, train_loader_1, optimizer, center, device, args.angular)
        print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
        auc, _ = get_score(model, device, train_loader, test_loader)
        print('Epoch: {}, AUROC is: {}'.format(epoch + 1, auc))

    return model

def run_epoch(model, train_loader, optimizer, center, device, is_angular):
    total_loss, total_num = 0.0, 0
    for ((img1, img2), _) in tqdm(train_loader, desc='Train...'):

        img1, img2 = img1.to(device), img2.to(device)

        optimizer.zero_grad()

        out_1 = model(img1)
        out_2 = model(img2)
        out_1 = out_1 - center
        out_2 = out_2 - center

        loss = contrastive_loss(out_1, out_2)

        if is_angular:
            loss += ((out_1 ** 2).sum(dim=1).mean() + (out_2 ** 2).sum(dim=1).mean())

        loss.backward()

        optimizer.step()

        total_num += img1.size(0)
        total_loss += loss.item() * img1.size(0)

    return total_loss / (total_num)


def get_score(model, device, train_loader, test_loader):
    train_feature_space = []
    with torch.no_grad():
        for (imgs, _) in tqdm(train_loader, desc='Train set feature extracting'):
            imgs = imgs.to(device)
            features = model(imgs)
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
    test_feature_space = []
    test_labels = []
    with torch.no_grad():
        for (imgs, labels) in tqdm(test_loader, desc='Test set feature extracting'):
            imgs = imgs.to(device)
            features = model(imgs)
            test_feature_space.append(features)
            test_labels.append(labels)
        test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
        test_labels = torch.cat(test_labels, dim=0).cpu().numpy()

    distances = utils.knn_score(train_feature_space, test_feature_space)

    
    
    auc = roc_auc_score(test_labels, distances)
    print("CLEAN AUC: ",auc)

    return auc, train_feature_space


class Model(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        mu = torch.tensor(mean).view(3,1,1).cuda()
        std = torch.tensor(std).view(3,1,1).cuda()        
        self.norm = lambda x: ( x - mu ) / std
        if backbone == 152:
            self.backbone = torchvision.models.resnet152(pretrained=True)
        else:
            self.backbone = torchvision.models.resnet18(pretrained=True)
        self.fc1=nn.Linear(1000,2)        
    def forward(self, x):
        x = self.norm(x)
        z1 = self.backbone(x)
        z1=self.fc1(z1)
        
        return z1



def get_score_adv(model_normal,model_blackbox, device, train_loader, test_loader):

    steps=10
    eps=1/255
    attack=torchattacks.PGD(model_blackbox, eps=eps, steps=steps, alpha=2.5 * eps / steps)

    train_feature_space = []
    with torch.no_grad():
        for (imgs, _) in tqdm(train_loader, desc='Train set feature extracting'):
            imgs = imgs.to(device)
            features = model_normal(imgs)
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
        torch.cuda.empty_cache()
    test_feature_space = []
    test_labels = []
    # with torch.no_grad():
    for (imgs, labels) in tqdm(test_loader, desc='Test set feature extracting'):
        imgs = imgs.to(device)
        labels = labels.to(device)
        imgs_adv=attack(imgs,labels)
        features = model_normal(imgs_adv)
        test_feature_space.append(features.detach().cpu())
        test_labels.append(labels.detach().cpu())

    # gc.collect()
        torch.cuda.empty_cache()
        del imgs,labels,imgs_adv,features

    test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
    test_labels = torch.cat(test_labels, dim=0).cpu().numpy()

    distances = utils.knn_score(train_feature_space, test_feature_space)

    auc = roc_auc_score(test_labels, distances)

    print("ADV AUC: ",auc)

    return auc, train_feature_space



def train_model_blackbox(epoch, model, trainloader, device): 
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    soft = torch.nn.Softmax(dim=1)

    preds = []
    anomaly_scores = []
    true_labels = []
    running_loss = 0
    accuracy = 0

    
    with tqdm(trainloader, unit="batch") as tepoch:
        torch.cuda.empty_cache()
        for i, (data, targets) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch + 1}")
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            true_labels += targets.detach().cpu().numpy().tolist()

            predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
            preds += predictions.detach().cpu().numpy().tolist()
            correct = (torch.tensor(preds) == torch.tensor(true_labels)).sum().item()
            accuracy = correct / len(preds)

            probs = soft(outputs).squeeze()
            anomaly_scores += probs[:, 1].detach().cpu().numpy().tolist()

            running_loss += loss.item() * data.size(0)

            tepoch.set_postfix(loss=running_loss / len(preds), accuracy=100. * accuracy)

        print("AUC : ",roc_auc_score(true_labels, anomaly_scores) )
        print("accuracy_score : ",accuracy_score(true_labels, preds, normalize=True) )

    return  model




def load_pretrain_robust(model):
    checkpoint = torch.load("../resnet18_linf_eps8.0.ckpt")
    state_dict_path = 'model'
    sd = checkpoint[state_dict_path]
    sd = {k[len('module.'):]:v for k,v in sd.items()}
    sd_t = {k[len('attacker.model.'):]:v for k,v in sd.items() if k.split('.')[0]=='attacker' and k.split('.')[1]!='normalize'}
    model.load_state_dict(sd_t)
    return model
    
def main(args):
    print('Dataset: {}, Normal Label: {}, LR: {}'.format(args.dataset, args.label, args.lr))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    model_blackbox=Model(18)
    model_blackbox = model_blackbox.to(device)
    train_loader_blackbox = utils.get_loaders_blackbox(dataset=args.dataset, label_class=args.label, batch_size=args.batch_size, backbone=args.backbone)

    model_blackbox=load_pretrain_robust(model_blackbox)
    
    for epoch in range(10):
        model_blackbox=train_model_blackbox(epoch,model_blackbox, train_loader_blackbox, device)


    
    model_main = utils.Model(args.backbone)
    model_main = model_main.to(device)
    train_loader, test_loader, train_loader_1 = utils.get_loaders_normal(dataset=args.dataset, label_class=args.label, batch_size=args.batch_size, backbone=args.backbone)
    model_main=train_model(model_main, train_loader, test_loader, train_loader_1, device, args)


    get_score(model_main, device, train_loader, test_loader)
    get_score_adv(model_main,model_blackbox, device, train_loader, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--epochs', default=20, type=int, metavar='epochs', help='number of epochs')
    parser.add_argument('--label', default=0, type=int, help='The normal class')
    parser.add_argument('--lr', type=float, default=1e-5, help='The initial learning rate.')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--backbone', default=152, type=int, help='ResNet 18/152')
    parser.add_argument('--angular', action='store_true', help='Train with angular center loss')
    args = parser.parse_args()
    main(args)
