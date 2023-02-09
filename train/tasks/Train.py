import yaml

from train.tasks.segmented.trainer import Trainer
from common.options import args
import torch.optim as optim
import os
import numpy as np
from train.tasks.segmented.modules.SalsaNext import *
from train.tasks.segmented.modules.Discriminator import PixelDiscriminator
from train.tasks.segmented.dataset.kitti.parser import Parser
from train.tasks.segmented.modules.FutureFrame import FutureHead
import torch.backends.cudnn as cudnn
from common.warmupLR import warmupLR
from train.tasks.segmented.modules.Full_scan import Full_scan
from fvcore.nn import FlopCountAnalysis
##TODO
"""
Create networks, load weights, change everything but generator/discriminator do eval mode
call trainer
"""

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    #print(args.cuda)
    #print(os.environ["CUDA_VISIBLE_DEVICES"])
    list_gpus = list(range(0,len(list(map(int,args.cuda.split(','))))))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    cudnn.fastest = True

    discriminator =PixelDiscriminator(49,1)#.to('cuda:1') #SPADEDiscriminator().to('cuda:1')
    discriminator = nn.DataParallel(discriminator,device_ids=[1,2,3,4,5,6,7]).to('cuda:1')#.to('cuda:1')
    # full_scan = Full_scan()
    # full_scan = nn.DataParallel(full_scan,device_ids=[1,2,3,4,5,6,7]).to('cuda:1')#.to('cuda:1')
    generator = SalsaNext()#.to('cuda:1')
    generator = nn.DataParallel(generator,device_ids=[1,2,3,4,5,6,7]).to('cuda:1')#.to('cuda:1')

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
                      batch_size=ARCH["train"]["batch_size"],
                      workers=ARCH["train"]["workers"],
                      gt=True,
                      shuffle_train=False,
                      transform=False)



    optimizers = {
        'generator': optim.Adam(list(generator.parameters()), lr=ARCH['train']['lr'], betas=(0.5, 0.999),weight_decay=ARCH['train']['w_decay']),
         'discriminator': optim.Adam(discriminator.parameters(), lr=ARCH['train']['lr'], betas=(0.5, 0.999),weight_decay=ARCH['train']['w_decay']),
         #'futureHead': optim.Adam(list(futurehead.parameters()), lr=ARCH['train']['lr'], betas=(0.5, 0.999),weight_decay=ARCH['train']['w_decay'])
    }

    schedulerG = warmupLR(optimizer=optimizers['generator'],
                          lr=ARCH["train"]["lr"],
                          warmup_steps=int(ARCH["train"]["wup_epochs"]*parser.get_train_size()),
                          momentum=ARCH["train"]["momentum"],
                          decay=ARCH["train"]["lr_decay"] ** (1 / int(ARCH["train"]["wup_epochs"]*parser.get_train_size()))
                                   )
    schedulerD = warmupLR(optimizer=optimizers['discriminator'],
                          lr=ARCH["train"]["lr"],
                          warmup_steps=int(ARCH["train"]["wup_epochs"]*parser.get_train_size()),
                          momentum=ARCH["train"]["momentum"],
                          decay=ARCH["train"]["lr_decay"] ** (1 / int(ARCH["train"]["wup_epochs"]*parser.get_train_size()))
                                   )

    # schedulerF = warmupLR(optimizer=optimizers['futureHead'],
    #                       lr=ARCH["train"]["lr"],
    #                       warmup_steps=int(ARCH["train"]["wup_epochs"] * parser.get_train_size()),
    #                       momentum=ARCH["train"]["momentum"],
    #                       decay=ARCH["train"]["lr_decay"] ** (
    #                                   1 / int(ARCH["train"]["wup_epochs"] * parser.get_train_size())))
    schedulers = {'generator':schedulerG,'discriminator':schedulerD,#'futureHead':schedulerF
                  }
    #rgbsegmentor,lidarsegmenter,generator,discriminator,optimizer,parser,args
    train = Trainer(generator,discriminator,None,optimizers,parser,args,schedulers)
    train.trainer().run(parser.trainloader,args.epochs)


