import torch
import torch.optim
import torch.utils.data
from torch.autograd import Variable
import copy
from tqdm import tqdm

# from data.data_entry import select_train_loader, select_eval_loader
from dataset import CLRDataset
from loss import nt_xent_loss
from models import ResNet,SimCLR
from options import get_train_args
from utils.logger import Logger
from evaluate import linear_protocol
MODELS = {"ResNet":ResNet,"SimCLR":SimCLR}

class Trainer:
    def __init__(self):
        args = get_train_args()
        if  torch.cuda.is_available():
            args.device = torch.device("cuda")
        else:
            args.device = torch.device("cpu")
        self.args = args
        torch.manual_seed(args.seed)
        self.logger = Logger(args)
        dataset = CLRDataset()
        train_dataset = dataset.get_dataset(
            args.dataset_name+'-train',args.n_views,device=args.device
        )
        self.datasize = len(train_dataset)
        
        # train_dataset=train_dataset[:self.datasize]
        print(self.datasize)
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,batch_size=args.batch_size,shuffle=True,drop_last=True
        )

        self.model = MODELS[args.model_name](out_dim=args.out_dim)
        print("Model {} is loaded!".format(args.model_name))
        # self.criterion = self.compute_loss()
        self.criterion = torch.nn.CrossEntropyLoss()
        # if args.load_model_path != '':
        #     print("=> using pre-trained weights for {}".format(args.model_name))
            # if args.load_not_strict:
            #     load_match_dict(self.model, args.load_model_path)
            # else:
            #     self.model.load_state_dict(torch.load(args.load_model_path).state_dict())

        self.model = torch.nn.DataParallel(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.lr,weight_decay=self.args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=len(self. train_loader), eta_min=0, last_epoch=-1
    )
    def train(self):
        for epoch in tqdm(range(self.args.epochs+1)):
            # train for one epoch
            if self.args.trainer=='Supervised':
                self.train_per_epoch(epoch)
            elif self.args.trainer=='SelfSupervised':
                self.SimCLR_train_per_epoch(epoch)
            self.logger.save_curves(epoch)
            if epoch>=10:
                self.scheduler.step()        
        # save the best:
        self.logger.save_check_point(self.best_model_wts, self.best_epoch)

    def compute_loss(self):
        if self.args.trainer =='Supervised':
            return torch.nn.CrossEntropyLoss
        elif self.args.trainer == 'SelfSupervised':
            return nt_xent_loss

    def train_per_epoch(self, epoch):
        # switch to train mode
        self.model.train()
        running_loss,running_corrects=0,0
        self.best_acc = 0.0
        
        for images, labels in self.train_loader:
            # for supervised learning, we set n-views to 1 
            img = images.to(device=self.args.device, dtype=torch.float)
            labels = labels.to(device=self.args.device)
            logits = self.model(img)
            # loss = self.compute_loss(logits, labels)
            # self.criterion = torch.nn.CrossEntropyLoss()
            loss = self.criterion(logits,labels)
            _, preds = torch.max(logits, 1)

        # for i, data in enumerate(self.train_loader):
        #     img, pred, label = self.step(data)
            
        #     # # compute loss
        #     # metrics = self.compute_metrics(pred, label, is_train=True)

        #     # get the item for backward
        #     loss = self.compute_loss(pred,label)

            # compute gradient and do Adam step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # running_loss += loss.item()
            running_loss += loss.item() * img.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / self.datasize
        epoch_acc = running_corrects.double() / self.datasize
        print('Epoch:{} Loss: {:.4f} Acc: {:.4f}'.format(
            epoch, epoch_loss, epoch_acc))
        self.logger.record_scalar('loss',epoch_loss,epoch)
        self.logger.record_scalar('acc',epoch_acc,epoch)
        
        if epoch_acc > self.best_acc:
            self.best_epoch = epoch
            self.best_acc = epoch_acc
            self.best_model_wts = copy.deepcopy(self.model.state_dict())
            
    def SimCLR_train_per_epoch(self, epoch):
        # switch to train mode
        self.model.train()
        running_loss=0
        self.best_loss=float('inf')
        for imgs, _ in self.train_loader:# label is not needed
            # for selfsupervised learning, we set n-views to 2(use nt-xent-loss) 
            img1,img2 = imgs
            img1 = img1.to(device=self.args.device, dtype=torch.float)
            img2 = img2.to(device=self.args.device, dtype=torch.float)
            self.optimizer.zero_grad()
            z_i,z_j = self.model(img1),self.model(img2)
            loss= nt_xent_loss(z_i,z_j,temperature=self.args.temperature)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * img1.size(0)
        epoch_loss = running_loss / self.datasize
        print('Epoch:{} Loss: {:.4f}'.format(
            epoch, epoch_loss))
        self.logger.record_scalar('loss',epoch_loss,epoch)
        if self.best_loss>epoch_loss:
            self.best_loss=epoch_loss
            self.best_epoch=epoch
            self.best_model_wts = copy.deepcopy(self.model.state_dict())
        

    # def val_per_epoch(self, epoch):
    #     self.model.eval()
    #     for i, data in enumerate(self.val_loader):
    #         img, pred, label = self.step(data)
    #         metrics = self.compute_metrics(pred, label, is_train=False)

    #         for key in metrics.keys():
    #             self.logger.record_scalar(key, metrics[key])

    #         if i == len(self.val_loader) - 1:
    #             self.logger.save_imgs(self.gen_imgs_to_write(img, pred, label, False), epoch)

    def step(self, data):
        img, label = data
        # warp input
        img = Variable(img).cuda()
        label = Variable(label).cuda()

        # compute output
        pred = self.model(img)
        return img, pred, label

    # def compute_metrics(self, pred, gt, is_train):
    #     # you can call functions in metrics.py
    #     l1 = (pred - gt).abs().mean()
    #     prefix = 'train/' if is_train else 'val/'
    #     metrics = {
    #         prefix + 'l1': l1
    #     }
    #     return metrics

    # def gen_imgs_to_write(self, img, pred, label, is_train):
    #     # override this method according to your visualization
    #     prefix = 'train/' if is_train else 'val/'
    #     return {
    #         prefix + 'img': img[0],
    #         prefix + 'pred': pred[0],
    #         prefix + 'label': label[0]
    #     }
    def evaluate(self):
        self.model = linear_protocol(self.model,self.args)



def main():
    trainer = Trainer()
    trainer.train()
    trainer.evaluate()

if __name__ == '__main__':
    main()