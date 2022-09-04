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
# trainer = UNIT_Trainer(config)
state_dict = torch.load(opts.checkpoint, map_location='cpu')
trainer.gen_a.load_state_dict(state_dict['a'])
trainer.cuda()
trainer.eval()

##test

train_loader_a,test_loader_a= get_all_data_loaders(config)

logs=[]
for it, batch in enumerate(test_loader_a):
    dataB,mean,std,fname,slice_num=batch[0],batch[1],batch[2],batch[3],batch[4]
    # dataA = next(train_loader_a_iter)
    # x_a =dataA.cuda().detach()
    x_b =dataB.cuda().detach()
    x_ba=[]
    for i in range(x_b.size(0)):
        with torch.no_grad():
####test
            outputs= trainer.gen_a(x_b[i].unsqueeze(0))   
            x_ba.append(torch.clamp(outputs*30*std[i]+mean[i],min=0))
    image_outputs = torch.cat(x_ba)
    log={'output':image_outputs.cpu(),'fname':fname,'slice_num':slice_num}
    logs.append(log)
    print(it)   
    #if slice_num[-1]%359==0 and slice_num[-1]!=0 :
    #if slice_num[-1]%672==0 and slice_num[-1]!=0 :
    #if slice_num[-1]%439==0 and slice_num[-1]!=0 :
    if slice_num[-1]%643==0 and slice_num[-1]!=0 : 
        outputs = defaultdict(dict)
        # targets = defaultdict(dict)
        # means  = defaultdict(dict)
        # stds  = defaultdict(dict)
        for log in logs:
            for i, (fname, slice_num) in enumerate(zip(log["fname"], log["slice_num"])):
                outputs[fname][int(slice_num)] = log["output"][i][0]
                # targets[fname][int(slice_num)] = log["target"][i]
                # means[fname][int(slice_num)] = log["mean"]
                # stds[fname][int(slice_num)] = log["std"]
        logs=[]
        # stack all the slices for each file
        for fname in outputs:
            outputs[fname] = np.stack(
                [out for _, out in sorted(outputs[fname].items())]
            )
        for fname, output in outputs.items():
            #output=sitk.GetImageFromArray(output.transpose(1,0,2))   
            #output=sitk.GetImageFromArray(output.transpose(1,2,0))   
            output=sitk.GetImageFromArray(output)  ##axial
            image=sitk.ReadImage(fname)
            spacing = image.GetSpacing()
            direction = image.GetDirection()
            origin = image.GetOrigin()
            output.SetOrigin(origin)
            output.SetSpacing(spacing)
            output.SetDirection(direction)
            sitk.WriteImage(output, opts.output_folder+'/output_'+fname.split('/')[-1])

