import torch
import yaml
from common.options import args
import os
import numpy as np
from train.tasks.segmented.modules.SalsaNext import *
from train.tasks.segmented.modules.Discriminator import PixelDiscriminator
from train.tasks.segmented.dataset.kitti.parser import Parser
from train.tasks.segmented.modules.FutureFrame import FutureHead
import torch.backends.cudnn as cudnn
##TODO
"""
Create networks, load weights, change everything but generator/discriminator do eval mode
call trainer
"""

if __name__ == "__main__":
    meanL = []
    stdL = []
    ARCH = yaml.safe_load(open(args.arch_cfg, 'r'))
    DATA = yaml.safe_load(open(args.data_cfg, 'r'))
    parser = Parser(root=args.dataset,
                      train_sequences=DATA["split"]["train"],
                      valid_sequences=DATA["split"]["valid"],
                      test_sequences=DATA["split"]["test"],
                      labels=DATA["labels"],
                      color_map=DATA["color_map"],
                      learning_map=DATA["learning_map"],
                      learning_map_inv=DATA["learning_map_inv"],
                      sensor=ARCH["dataset"]["sensor"],
                      max_points=ARCH["dataset"]["max_points"],
                      batch_size=1,
                      workers=ARCH["train"]["workers"],
                      gt=True,
                      shuffle_train=False,
                      transform=False)

    trainDataLoader = parser.trainloader


    for i,(rgb, rgb_labels, proj_labels, proj_mask, proj, rgb_labels_one_hot, proj_labels_one_hot,depth) in enumerate(trainDataLoader):
        print("\r{}/{}".format(i,len(trainDataLoader)))
        depth_1 = transforms.Normalize((1.6719), (6.189))(depth)
        meanL.append(depth.mean().item())
        stdL.append(depth.std().item())
    print("MEAN:{}\tSTD:{}".format(np.mean(meanL),np.mean(stdL)))