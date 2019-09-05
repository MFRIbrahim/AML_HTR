import torchvision
import torch
from PIL import Image
import numpy





class DataAugmenter(object):
    def __init__(self, p_erase=0.1, p_jitter=0.1, p_translate=0.1, p_perspective=0.1):
        self.p_perspective = p_perspective
        erasing = torchvision.transforms.RandomErasing(p=p_erase, scale=(0.02, 0.04), ratio=(0.3, 3.3), value=1) #Größe Testen
        toPIL = torchvision.transforms.ToPILImage("L")
        translate =  torchvision.transforms.RandomApply([torchvision.transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), fillcolor=255)], p=p_translate)
        jitter = torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter()], p=p_jitter) #Erwartet evtl. 3 Channel Bilder
        perspective = torchvision.transforms.Lambda(self.RandomPerspective) #distortion scale testen
        self.toTensor = torchvision.transforms.ToTensor()
        self.transform = torchvision.transforms.Compose([erasing, toPIL, jitter, translate, perspective])

    def __call__(self, sample):
        reduce_dim = False
        reduce_channels = False
        if sample.ndim == 2:
            reduce_dim = True
            sample = sample.unsqueeze(0)

        transformed = self.transform(sample)
        transformed = self.toTensor(transformed)


        if reduce_dim:
            transformed = transformed.squeeze(0)
        return transformed

    def add_transform(self, transform, needs_PIL=True, p=1):
        RandTransform = torchvision.transform.RandomApply([transform], p=p)
        if needs_PIL:
            self.transform = torchvision.transforms.Compose([self.transform, RandTransform])
        else:
            self.transform = torchvision.transforms.Compose([RandTransform, self.transform])

    def RandomPerspective(self, img):
        if numpy.random.rand() > self.p_perspective:
            return img
        pa = torch.Tensor([[0,0], [0,1], [1,0], [1,1]])
        rand_range = 0.0004
        pb = pa + torch.randn(4,2)*rand_range

        img = img.transform(img.size, Image.PERSPECTIVE, self.find_coeffs(pa,pb), Image.BICUBIC, fillcolor=255)
        return img

    def find_coeffs(self, pa, pb):
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

        A = numpy.matrix(matrix, dtype=numpy.float)
        B = numpy.array(pb).reshape(8)

        res = numpy.dot(numpy.linalg.inv(A.T * A) * A.T, B)
        return numpy.array(res).reshape(8)
