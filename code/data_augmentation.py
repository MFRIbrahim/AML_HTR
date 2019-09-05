import torchvision
import torch


class DataAugmenter(object):
    def __init__(self, p_erase=1, p_jitter=1, p_translate=1, p_perspective=1):
        erasing = torchvision.transforms.RandomErasing(p=p_erase, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=255) #Größe Testen
        toPIL = torchvision.transforms.ToPILImage()
        translate =  torchvision.transforms.RandomApply([torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=p_translate)
        jitter = torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter()], p=p_jitter) #Erwartet evtl. 3 Channel Bilder
        perspective = torchvision.transforms.RandomPerspective(distortion_scale=0.1, p=p_perspective) #distortion scale testen
        toTensor = torchvision.transforms.ToTensor()
        self.transform = torchvision.transforms.Compose([erasing, toPIL, jitter, translate, perspective, toTensor])

    def __call__(self, sample):
        reduce_dim = False
        reduce_channels = False
        if sample.ndim == 2:
            reduce_dim = True
            sample = sample.unsqueeze(-1)
        if sample.shape[-1] == 1:
            reduce_channels = True
            sample = sample.repeat(1, 1, 3)
        transformed = self.transform(sample)
        if reduce_channels:
            transformed = torch.mean(transformed, -1, keepdim=True)
        if reduce_dim:
            transformed = transformed.squeeze(-1)
        return transformed

    def add_transform(self, transform, p=1):
        RandTransform = torchvision.transform.RandomApply([transform], p=p)
        self.transform = torchvision.transforms.Compose([self.transform, RandTransform])
