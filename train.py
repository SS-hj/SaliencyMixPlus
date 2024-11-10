import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import datasets, transforms
from utils import CSVLogger
from data_augmentations import CutMix, SaliencyMix, SaliencyMixPlus
import os
import timm

parser = argparse.ArgumentParser(description='SaliencyMixPlus')
parser.add_argument('--dataset', '-d', default='oxford_pet', help='dataset (options: cifar10, cifar100, and oxford_pet)')
parser.add_argument('--model', '-a', default='resnet18')
parser.add_argument('--batch_size', type=int, default=256,
                    help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 200)')
parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='learning rate')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 0)')
parser.add_argument('--mix_prob', default=0.5, type=float,
                    help='probability of the mix augmentation')
parser.add_argument('--mix_aug', default='saliencymix_plus', type=str,
                    help='type of the mix augmentation')
parser.add_argument('--using_timm', type=bool, default=True,
                    help='Using timm library to load models')
parser.add_argument('--start_mixaugmentation', type=int, default=0)

def main():
    args = parser.parse_args()
    test_id = f'{args.dataset}_{args.model}_{args.mix_aug}{args.mix_prob}_{args.epochs}'
    
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    
    if args.dataset.startswith('cifar'):
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        
        if args.dataset == 'cifar10':
            num_classes = 10
            train_dataset = datasets.CIFAR10(root='data/',
                                            train=True,
                                            transform=train_transform,
                                            download=True)
            test_dataset = datasets.CIFAR10(root='data/',
                                            train=False,
                                            transform=test_transform,
                                            download=True)
        elif args.dataset == 'cifar100':
            num_classes = 100
            train_dataset = datasets.CIFAR100(root='data/',
                                            train=True,
                                            transform=train_transform,
                                            download=True)
            test_dataset = datasets.CIFAR100(root='data/',
                                            train=False,
                                            transform=test_transform,
                                            download=True)
        else:
            raise Exception('unknown dataset: {}'.format(args.dataset))
        
    elif args.dataset == "oxford_pet":
        class OxfordIIITPetDataset(Dataset):
            def __init__(self, img_dir, annotations_file, transform=None):
                self.img_dir = img_dir
                self.transform = transform
                self.img_labels = []

                with open(annotations_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split(' ')
                        img_name = parts[0] + '.jpg'
                        label = int(parts[1]) - 1
                        self.img_labels.append((img_name, label))

            def __len__(self):
                return len(self.img_labels)

            def __getitem__(self, idx):
                img_name, label = self.img_labels[idx]
                img_path = os.path.join(self.img_dir, img_name)
                image = Image.open(img_path).convert("RGB")

                if self.transform:
                    image = self.transform(image)

                return image, label
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.RandomResizedCrop((224,224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        test_transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
            ])
        
        img_dir = './dataset/images'
        train_annotations_file = './dataset/annotations/trainval.txt'
        test_annotations_file = './dataset/annotations/test.txt'

        num_classes = 37
        train_dataset = OxfordIIITPetDataset(img_dir=img_dir, annotations_file=train_annotations_file, transform=train_transform)
        test_dataset = OxfordIIITPetDataset(img_dir=img_dir, annotations_file=test_annotations_file, transform=test_transform)
    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))

    train_loader = DataLoader(dataset=train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=2)
    test_loader = DataLoader(dataset=test_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=2)
        
    cnn = timm.create_model(args.model, pretrained=False, num_classes=num_classes)

    if args.dataset.startswith('cifar'):
        conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        bn1 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        act1 = torch.nn.ReLU(inplace=True)

        cnn = torch.nn.Sequential(conv1, bn1, act1, *list(cnn.children())[4:])

    cnn = nn.DataParallel(cnn).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True, weight_decay=5e-4)
    scheduler =  MultiStepLR(cnn_optimizer, milestones=[75, 150, 225], gamma=0.1) 
    cutmix = CutMix()
    saliencymix = SaliencyMix()
    saliencymix_plus = SaliencyMixPlus()
    
    if not os.path.exists('logs'):
        os.mkdir('logs')
    filename = f'logs/{test_id}.csv'
    csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train Top 1-err', 'test Top 1-err'], filename=filename)

    best_err1 = 100
    best_err5 = 100
    best_epoch = 0

    for epoch in range(args.epochs):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        progress_bar = tqdm(train_loader)
        for images, labels in progress_bar:
            progress_bar.set_description(f'Epoch {str(epoch)}')

            images = images.cuda()
            labels = labels.cuda()
            
            cnn.zero_grad()

            r = np.random.rand(1)
            if r < args.mix_prob:
                # generate mixed sample
                if args.mix_aug == 'cutmix':
                    lam, images, label_a, label_b = cutmix.forward(images, labels)
                    pred = cnn(images)
                    loss = criterion(pred, label_a)*lam + criterion(pred, label_b) * (1. - lam)
                elif args.mix_aug == 'saliencymix':
                    lam, images, label_a, label_b = saliencymix.forward(images, labels)
                    pred = cnn(images)
                    loss = criterion(pred, label_a)*lam + criterion(pred, label_b) * (1. - lam)
                else:
                    lam, images, label_a, label_b = saliencymix_plus.forward(images, labels, cnn, criterion)
                    pred = cnn(images)
                    loss = criterion(pred, label_a)*lam + criterion(pred, label_b) * (1. - lam)
                    loss = torch.mean(loss)
            else:
                pred = cnn(images)
                loss = criterion(pred, labels)

            # measure accuracy and record loss
            err1, err5 = accuracy(pred.data, labels[:len(pred)], topk=(1, 5))

            losses.update(loss.item(), images.size(0))
            top1.update(err1.item(), images.size(0))
            top5.update(err5.item(), images.size(0))

            loss.backward()
            cnn_optimizer.step()
            
            progress_bar.set_postfix(
                {'Loss':round(losses.avg,4), 'Top 1-err':round(top1.avg,4), 'Top 5-err':round(top5.avg,4)})

        test_err1, test_err5 = test(test_loader, cnn, criterion)
        tqdm.write(f'test Top 1-err={test_err1:.4f} test Top 5-err={test_err5:.4f}')

        scheduler.step()

        row = {'epoch': str(epoch), 'train Top 1-err': str(top1.avg), 'test Top 1-err': str(test_err1)}
        csv_logger.writerow(row)

        if(test_err1 <= best_err1):
            best_err1 = test_err1
            best_err5 = test_err5
            best_epoch = epoch
    
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    torch.save(cnn.state_dict(), f'checkpoints/{test_id}.pt')
    csv_logger.close()

    with open(f"{test_id} best_err1.txt", "a+") as f:
        f .write('best Top 1-err: %.3f and Top 5-err: %.3f at iteration: %d \r\n' % (best_err1, best_err5, best_epoch))

def test(loader, cnn, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    cnn.eval()
    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            pred = cnn(images)
            
        loss = criterion(pred, labels)
        
        # measure accuracy and record loss
        err1, err5 = accuracy(pred.data, labels, topk=(1, 5))

        losses.update(loss.item(), images.size(0))
        top1.update(err1.item(), images.size(0))
        top5.update(err5.item(), images.size(0))
    
    cnn.train()
    return top1.avg, top5.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res

if __name__ == '__main__':
    main()