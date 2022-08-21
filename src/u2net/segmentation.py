import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from src.u2net.data_loader import RescaleT
from src.u2net.data_loader import ToTensor
from src.u2net.data_loader import ToTensorLab
from src.u2net.data_loader import SalObjDataset

from src.u2net.model import U2NET # full size version 173.6 MB

model_name = 'u2net'
model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'u2net', 'saved_models', model_name, model_name + '.pth')

if (model_name == 'u2net'):
    print("...load U2NET---173.6 MB")
    net = U2NET(3, 1)

if torch.cuda.is_available():
    net.load_state_dict(torch.load(model_dir))
    net.cuda()
else:
    net.load_state_dict(torch.load(model_dir, map_location='cpu'))
net.eval()
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

def segment(image):

    # --------- 1. get image path and name ---------




    inputs_test = image
    inputs_test = inputs_test.type(torch.FloatTensor)

    if torch.cuda.is_available():
        inputs_test = Variable(inputs_test.cuda())
    else:
        inputs_test = Variable(inputs_test)

    d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

    # normalization
    pred = d1[:,0,:,:]
    pred = normPRED(pred)
    pred = pred.squeeze()
    pred = pred.cpu().data.numpy()

    del d1,d2,d3,d4,d5,d6,d7

    return np.array(pred)

#
# if __name__ == "__main__":
#     main()
