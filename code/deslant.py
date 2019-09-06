#-------------------------------------------------------------------#
#---From https://github.com/SiddhantKapil/deslant_cursive_images----#
#-------------------------------------------------------------------#
import cv2
import numpy as np
from collections import deque
import math

# data structure for internal usage
class Result:
    def  __init__(self):
        self.sum_alpha = 0.0
        self.transform = None
        self.size = 0

    def __ge__(self, other):
        return self.sum_alpha >= other.sum_alpha

# deslant image by calculating its slope and then rotating it overcome the effect of that shift
def deslant_image(img, bgcolor=255):
    TransformToFloat = False
    if np.max(img) <= 1:
        img = (img*255).astype(np.uint8)
        TransformToFloat = True

    _, imgBW = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    alphaVals = [ -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0 ]
    sum_alpha = [0]*len(alphaVals)

    results = []
    for i in range(len(alphaVals)):
        result = Result()
        alpha = alphaVals[i]
        shiftX = np.max([-alpha * imgBW.shape[1], 0])
        result.size = (imgBW.shape[1] + math.ceil(abs(alpha*imgBW.shape[1])), imgBW.shape[0])
        result.transform = np.zeros((2,3), dtype=np.float)
        result.transform[0,0] = 1
        result.transform[0,1] = alpha
        result.transform[0,2] = shiftX
        result.transform[1,0] = 0
        result.transform[1,1] = 1
        result.transform[1,2] = 0

        imgSheared = cv2.warpAffine(imgBW, result.transform, result.size, cv2.INTER_NEAREST)

        for x in range(imgSheared.shape[1]):
            fgIndices = []
            for y in range(imgSheared.shape[0]):
                if imgSheared[y,x]:
                    fgIndices.append(y)
            if len(fgIndices) == 0:
                continue

            h_alpha = len(fgIndices)
            delta_y_alpha = fgIndices[-1] - fgIndices[0] + 1

            if h_alpha == delta_y_alpha:
                result.sum_alpha += h_alpha * h_alpha

        results.append(result)


    bestResult = np.max(results)
    imgDeslanted = cv2.warpAffine(img, bestResult.transform, bestResult.size, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=bgcolor)

    if TransformToFloat:
        imgDeslanted = imgDeslanted.astype(np.float)/255

    return imgDeslanted
