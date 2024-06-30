import torchvision,torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CIFAR100
import torch.nn as nn
from tqdm import tqdm
from utils.logger import Logger

"""
freeze paras of model, train on cifar-100 dataset
"""
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
data_dir = "/data1/zy/DATA620004/FinalPJ/data/cifar-100-python"
# 加载 CIFAR-100 数据集

trainset = CIFAR100(dataset_dir="./data/cifar-100-python",subset="train")
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = CIFAR100(dataset_dir="./data/cifar-100-python",subset="test")
# testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=False, transform=transform)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
    
class LinearEvaluation(nn.Module):
    def __init__(self, model, classes):
        super().__init__()
        simclr = model
        simclr.linear_eval=True
        simclr.projection = Identity()
        self.simclr = simclr
        for param in self.simclr.parameters():
            param.requires_grad = False
        self.linear = nn.Linear(200, classes)
    def forward(self, x):
        encoding = self.simclr(x)
        pred = self.linear(encoding)
        return pred

def linear_protocol(model,args,num_epochs=50,lr=1e-4):
    logger = Logger(args)
    eval_model = LinearEvaluation(model, 100).to(DEVICE)
    correct, total = 0, 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(eval_model.linear.parameters(),lr=lr)
    for epoch in tqdm(range(num_epochs)):
        eval_model.train()
        running_loss = 0.0
        for img,gt in trainloader:
            image = img.to(DEVICE).float()
            label = gt.to(DEVICE)
            optimizer.zero_grad()
            outputs = eval_model(image)
            loss = criterion(outputs,label)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        print("Epoch:{:02d},loss:{:.4f}".format(epoch,running_loss/len(trainloader)))
    
        eval_model.eval()
        correct,total=0,0
        with torch.no_grad():
            for img, gt in trainloader:
                image = img.to(DEVICE).float()
                label = gt.to(DEVICE)
                pred = eval_model(image)
                _, pred = torch.max(pred.data, 1)
                total += label.size(0)
                correct += (pred == label).float().sum().item()
        acc = correct/total
        print("Accuracy of the network on the {} Train images: {:.4f} %".format(
                    total, 100 * acc))
        logger.record_scalar('eval_acc',acc,epoch)
        
if __name__=='__main__':
    for img,label in trainloader:
        print()