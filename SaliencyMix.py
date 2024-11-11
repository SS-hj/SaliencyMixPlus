import numpy as np
import torch
import cv2

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