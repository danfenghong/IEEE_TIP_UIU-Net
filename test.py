import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# import torch.optim as optim
from model.metrics import *
import numpy as np
from PIL import Image
import glob
import time
import cv2
from tqdm import tqdm
import torch.utils.data as Data
from utils.data import SirstDataset

from thop import profile

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import UIUNET

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')

def main():

    # --------- 1. get image path and name ---------
    model_name = 'uiunet'


    image_dir = os.path.join(os.getcwd(), 'test_data', 'Quantification_Results', 'test_images')
    label_dir = os.path.join(os.getcwd(), 'test_data', 'Quantification_Results', 'test_labels')
    prediction_dir = os.path.join(os.getcwd(), 'test_data',  model_name + '_results' + os.sep)
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')

    img_name_list = glob.glob(image_dir + os.sep + '*')
    print(img_name_list)

    label_name_list = glob.glob(label_dir + os.sep + '*')

    # --------- 2. dataloader ---------
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = label_name_list,
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    net = UIUNET(3,1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # --------- 4. inference for each image ---------
    iou_metric = SigmoidMetric()
    nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.55)
    iou_metric.reset()
    nIoU_metric.reset()
    best_iou = 0
    best_nIoU = 0
    total_iou = 0
    total_niou = 0
    # t0 = 0.0
    # seen = 0
    #####################
    for i_test, data_test in enumerate(test_salobj_dataloader):
        # seen += 1
        print("inferencing:", img_name_list[i_test].split(os.sep)[-1])
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)
        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        # iou/niou
        labels = data_test['label'].cpu()
        output = pred.unsqueeze(0).cpu()
        iou_metric.update(output, labels)
        nIoU_metric.update(output, labels)
        _, IoU = iou_metric.get()
        _, nIoU = nIoU_metric.get()


    if IoU > best_iou:
        best_iou = IoU
    if nIoU > best_nIoU:
        best_nIoU = nIoU

    total_iou = total_iou + IoU
    total_niou = total_niou + nIoU

    del d1,d2,d3,d4,d5,d6,d7

    IoU=total_iou/20
    nIoU=total_niou/20
    print(IoU, nIoU)
    print(best_iou, best_nIoU)


if __name__ == "__main__":
    main()
