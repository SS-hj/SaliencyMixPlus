import numpy as np
import torch
import cv2

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class CutMix(object):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, images, labels):
        lam = np.random.beta(1, 1)
        rand_index = torch.randperm(images.size()[0]).cuda()
        label_a = labels
        label_b = labels[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
        
        lam = 1 - ((bbx2-bbx1)*(bby2-bby1)/(images.size()[-1]*images.size()[-2]))

        return lam, images, label_a, label_b
    
def saliency_bbox(img, lam):
    _, W, H = img.shape
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # initialize OpenCV's static fine grained saliency detector and
    # compute the saliency map
    temp_img = img.cpu().numpy().transpose(1, 2, 0)
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(temp_img)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    maximum_indices = np.unravel_index(np.argmax(saliencyMap, axis=None), saliencyMap.shape)

    x = maximum_indices[0]
    y = maximum_indices[1]

    bbx1 = np.clip(x - cut_w // 2, 0, W)
    bby1 = np.clip(y - cut_h // 2, 0, H)
    bbx2 = np.clip(x + cut_w // 2, 0, W)
    bby2 = np.clip(y + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class SaliencyMix(object):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, images, labels):
        lam = np.random.beta(1, 1)
        rand_index = torch.randperm(images.size()[0]).cuda()
        labels_a = labels
        labels_b = labels[rand_index]
        bbx1, bby1, bbx2, bby2 = saliency_bbox(images[rand_index[0]], lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))

        return lam, images, labels_a, labels_b
    
def saliency_plus_bbox(img, lam):
    W, H = img.shape
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    x, y = np.unravel_index(np.argmax(img, axis=None), img.shape)

    bbx1 = np.clip(x - cut_w // 2, 0, W)
    bby1 = np.clip(y - cut_h // 2, 0, H)
    bbx2 = np.clip(x + cut_w // 2, 0, W)
    bby2 = np.clip(y + cut_h // 2, 0, H)

    # calculate the bbox importance
    bbox_importance = np.sum(img[bbx1:bbx2, bby1:bby2])

    # calculate the total importance
    total_importance = np.sum(img)

    return bbx1, bby1, bbx2, bby2, bbox_importance, total_importance

class SaliencyMixPlus(object):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, images, labels, model, criterion):
        lam = np.random.beta(1, 1)
        
        half_size = len(images)//2
        r_idx = torch.randperm(half_size).cuda()
        labels_a = labels[:half_size]
        labels_b = labels[half_size:][r_idx]
        a_importance_list = []
        b_importance_list = []

        # compute the saliency map using the training model
        images.requires_grad = True

        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        
        saliency_map = torch.abs(images.grad).cpu().numpy().sum(axis=1)
        
        images.requires_grad = False
        
        for i in range(half_size):
            bbx1, bby1, bbx2, bby2, b_bbox_importance, b_total_importance = saliency_plus_bbox(saliency_map[r_idx[i]+half_size], lam)

            # calculate the bbox importance
            a_bbox_importance = np.sum(saliency_map[i][bbx1:bbx2, bby1:bby2])
            # calculate the total importance
            a_total_importance = np.sum(saliency_map[i])

            images[i, :, bbx1:bbx2, bby1:bby2] = images[r_idx[i]+half_size, :, bbx1:bbx2, bby1:bby2]

            # calculate the importance of each label in the image
            a_importance = (a_total_importance - a_bbox_importance)/a_total_importance
            b_importance = (b_bbox_importance)/b_total_importance

            a_importance_list.append(a_importance)
            b_importance_list.append(b_importance)

        a_importance_list = torch.tensor(a_importance_list).cuda()
        b_importance_list = torch.tensor(b_importance_list).cuda()
        
        # calculate the percentage of importance of a in the image
        lam = a_importance_list/(a_importance_list+b_importance_list)

        return lam, images[:half_size], labels_a, labels_b