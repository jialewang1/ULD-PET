"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config,get_all_data_loaders
from trainer import MUNIT_Trainer
import matplotlib.pyplot as plt
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image
import SimpleITK as sitk
# from skimage.measure import compare_psnr, compare_ssim
import numpy as np
from collections import defaultdict
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser() 
parser.add_argument('--config', type=str, default='/data3/jiale/body_pet/configs/synthia2cityscape_folder.yaml', help="net configuration")
parser.add_argument('--input', type=str, default = '/data3/jiale/data/brain_pet/val', help="input image path")

parser.add_argument('--output_folder', type=str, default='/data3/jiale/body_pet/outputs/model2/submit_100x_54', help="output image path")
parser.add_argument('--checkpoint', type=str, default='/data3/jiale/body_pet/outputs/model2/checkpoints/gen_00540001.pt',
                    help="checkpoint of autoencoders") 
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--psnr', action="store_false", help='is used to compare psnr')
opts = parser.parse_args()


# Load experiment setting
config = get_config(opts.config)
style_dim = config['gen']['style_dim']
config['vgg_model_path'] = "/data3/jiale/body_pet"
trainer = MUNIT_Trainer(config)
# Setup model and data loaderopts.trainer == 'UNIT':
state_dict = torch.load(opts.checkpoint, map_location='cpu')
trainer.gen_a.load_state_dict(state_dict['a'])
trainer.cuda()
trainer.eval()

##test

test_loader_a= get_all_data_loaders(config)

logs=[]
for it, batch in enumerate(test_loader_a):
    dataB,fname=batch[0],batch[3]
    # dataA = next(train_loader_a_iter)
    # x_a =dataA.cuda().detach()
    x_b =dataB.cuda().detach()
    x_ba=[]
    for i in range(x_b.size(0)):
        with torch.no_grad():
####test
        outputs= trainer.gen_a(x_b[i].unsqueeze(0))   
 
        for output in outputs.items(): 
            output=sitk.GetImageFromArray(output) 
            image=sitk.ReadImage(fname)
            spacing = image.GetSpacing()
            direction = image.GetDirection()
            origin = image.GetOrigin()
            output.SetOrigin(origin)
            output.SetSpacing(spacing)
            output.SetDirection(direction)
            sitk.WriteImage(output, opts.output_folder+'/output_'+fname.split('/')[-1])

