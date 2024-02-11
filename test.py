import argparse
from dataset import FlameSet
from torch.utils.data import DataLoader
from dataset import FlameSet,DataPrefetcher
import torch
from torch.autograd import Variable
import numpy as np
import os
import cv2
import time
from PIL import Image


parser=argparse.ArgumentParser(description='')
parser.add_argument('--dataset_test', type=str, default='../datasets/QB/test')
parser.add_argument('--checkpoint', type=str,default='model_best_epoch.pth')
parser.add_argument("--net", type=str, default='TransDenConvQB')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--cuda', default=True, help='use cuda?')
parser.add_argument('--gpu_id', default=3, type=int,help='GPU ID to use')
opt = parser.parse_args()

test_set = FlameSet(opt.dataset_test)
test_data_loader = DataLoader(dataset=test_set, batch_size=opt.testBatchSize, shuffle=False)

bestmodel = torch.load('log/202401130914_batchSize16_lr0.000100_SSdatasetz_c/bestmodel_dir/%s'%opt.checkpoint)

image_path = '../DMF_Net/log/202401130914_batchSize16_lr0.000100_SSdatasetz_c/test_image'
start_time = time.time()
def test(test_data_loader, model):
    with torch.no_grad():
        model.eval()
        for i,(pan,ms,gt,filename) in enumerate(test_data_loader):
            if opt.cuda:
                input_pan = pan.cuda(opt.gpu_id)
                input_ms = ms.cuda(opt.gpu_id)
                start_time = time.time()
                output = model(input_pan, input_ms)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Total elapsed time: {elapsed_time} seconds:")
                imgf = output.squeeze().cpu().numpy()  # (WV-2,256,256)
                imgf = np.clip(imgf,-1,1)
                img_gt = gt.squeeze().cpu().numpy()  # (WV-2,256,256)
                imgf_gt = np.clip(img_gt, -1, 1)
                two_img = np.concatenate((imgf, imgf_gt), axis=2)
                two_img = ((two_img * 0.5 + 0.5) * 255).astype('uint8')
                only_test_image = ((imgf * 0.5 + 0.5) * 255).astype('uint8')
                image = cv2.merge(only_test_image)
                cv2.imwrite(os.path.join(image_path, '%s' % (filename)), image)


if not os.path.exists(image_path):
    os.makedirs(image_path)


test(test_data_loader, bestmodel['model'])

