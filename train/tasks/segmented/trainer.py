from PIL import Image
from matplotlib import pyplot as plt
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events
from ignite.engine import Engine
import numpy as np
from ignite.metrics import RunningAverage
from common.checkpoint import ModelCheckpoint
from ignite.handlers import Timer
import os
import cv2
from torch import nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from train.tasks.segmented.modules.Lovasz_Softmax import Lovasz_softmax
from train.tasks.segmented.modules.RecallCrossEntropy import RecallCrossEntropy
#RecallCrossEntropy
from torchvision.utils import save_image
from train.tasks.segmented.modules.ioueval import iouEval
from kornia.losses import InverseDepthSmoothnessLoss,SSIMLoss
from train.tasks.segmented.modules.rmi import RMILoss
from train.tasks.segmented.modules.EntropyLoss import Entropy
TAG_CHAR = np.array([202021.25], np.float32)


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b

def imgrad(img):
    img = torch.mean(img, 1, True)
    fx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.to('cuda:1')#cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)

    fy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.to('cuda:1')
    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)

    #     grad = torch.sqrt(torch.pow(grad_x,2) + torch.pow(grad_y,2))

    return grad_y, grad_x


def imgrad_yx(img):
    N, C, _, _ = img.size()
    grad_y, grad_x = imgrad(img)
    return torch.cat((grad_y.view(N, C, -1), grad_x.view(N, C, -1)), dim=1)

palette = [0, 0, 0, #0 outlier
           100, 150, 245,   #1 car
           100,230,245,   #2 bicycle
           30, 60, 150,  #3 motorcycle
           80,30,180,  #4 truck
           0,0,250,    #5 other-vehichle
               255,30,30,     #6 person
           255,0,255,#7 road
           75,0,75, #8 sidewalk
           255,200,0,#9 building
           255,120,50, #10 fence
           0,175,0,   #11 vegetation
           150,240,80, #12 terrain
           255,240,150,#13 pole
           255,0,0 #14 traffic-sign
           ]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


# This file is covered by the LICENSE file in the root of this project.


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def psnr_(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


def compute_gradient_penalty(D, real_images, fake_images, rgb):
    eta = torch.Tensor(1, 1, 1, 1).normal_(0, 1)
    eta = eta.expand(1, real_images.size(1), real_images.size(2), real_images.size(3))
    eta = eta.to('cuda:1')#.to('cuda:0')

    interpolated = eta * real_images + ((1 - eta) * fake_images)
    interpolated = interpolated.to('cuda:1')

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = D(interpolated, rgb)

    # calculate gradients of probabilities with respect to examples
    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                    grad_outputs=torch.ones(
                                        prob_interpolated.size()).to('cuda:1'),#.to('cuda:0'),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    return ((gradients_norm - 1) ** 2).mean()


def colorize_mask(mask):
    """
    Colorize a segmentation mask.
    """
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


class Trainer():
    def __init__(self, generator, discriminator,futurehead, optimizer, parser, args, schedulers):

        self.generator = generator
        self.discriminator = discriminator
        self.futurehead = futurehead
        self.optimizer = optimizer
        self.parser = parser
        self.args = args
        self.timer = Timer(average=True)
        self.old_lr = 0
        self.entropy = Entropy()
        #self.rmi = RMILoss(15).to('cuda:1')
        self.schedulers = schedulers
        self.validDataLoader = self.parser.validloader
        self.trainDataLoader = self.parser.trainloader
        self.testDataLoader = self.parser.testloader
        self.epoch_c = args.epoch_c
        self.ls = Lovasz_softmax(ignore=0).to('cuda:1')
        self.wc = torch.nn.NLLLoss(weight=torch.FloatTensor([1e-7, 1.79017374e+01, 3.00196513e+03, 2.21811080e+03,
       3.02741845e+02, 3.02711045e+03, 1.72900678e+03, 6.05093770e+00,
       1.21582491e+01, 6.15016074e+00, 1.75401216e+01, 2.91377291e+00,
       2.14266522e+01, 1.48283054e+02, 4.99487622e+02])).to('cuda:1')
        self.depth_loss = InverseDepthSmoothnessLoss().to('cuda:1')
        self.ssim = SSIMLoss(5).to('cuda:1')
        self.MAE = nn.L1Loss().to('cuda:1')
        self.evaluator = iouEval(15,"cuda:1",[0])
        self.info = {"g_loss": 0,
                     "d_loss": 0,
                     "ls": 0,
                     "wdistance": 0,
                     "lr": 0,
                     "ssim": 0,
                     "psnr": 0,
                     'miou':0,
                     'acc': 0}

        if self.epoch_c != 0:
            self.generator.load_state_dict(
                torch.load(args.output_dir+"/checkpoints/training_generator_{}.pth".format(
                    self.epoch_c)))
            # self.discriminator.load_state_dict(
            #     torch.load(
            #         args.output_dir+"/checkpoints/training_discriminator_{}.pth".format(
            #             self.epoch_c)))
            self.optimizer['discriminator'].load_state_dict(torch.load(
                args.output_dir+"/checkpoints/training_optimizer_D_{}.pth".format(
                    self.epoch_c)))
            self.optimizer['generator'].load_state_dict(torch.load(
                args.output_dir+"/checkpoints/training_optimizer_G_{}.pth".format(
                    self.epoch_c)))
            self.schedulers['generator'].load_state_dict(torch.load(
                args.output_dir + "/checkpoints/training_scheduler_G_{}.pth".format(
                    self.epoch_c)))
            self.schedulers['discriminator'].load_state_dict(torch.load(
                args.output_dir + "/checkpoints/training_scheduler_D_{}.pth".format(
                    self.epoch_c)))
        mone = torch.tensor(-1, dtype=torch.float)
        mone = mone.to('cuda:1')
        def step(engine, batch):
            generator.train()

            rgb, rgb_labels, proj_labels, proj_mask, proj, rgb_labels_one_hot, proj_labels_one_hot, depth = batch

            #noise = torch.randn((rgb.size(0),256,4,4),requires_grad=True)+1e-7
            #noise = noise.to('cuda:1')
            # try:
            rgb = rgb.to('cuda:1')
            proj = proj.to('cuda:1')  # [b].to('cuda:1')
            rgb_labels_one_hot = rgb_labels_one_hot.to('cuda:1') # [b].to('cuda:1')#.to('cuda:0')
            proj_labels_one_hot = proj_labels_one_hot.to('cuda:1')  # [b].to('cuda:1')#.to('cuda:0')
            rgb_labels = rgb_labels.unsqueeze(1).to('cuda:1')  # [b].unsqueeze(1).to('cuda:1')#.to('cuda:0')
            proj_labels = proj_labels.to('cuda:1')  # [b].to('cuda:1')#.to('cuda:0')
            depth = depth.unsqueeze(1).to('cuda:1')


            for p in self.discriminator.module.parameters():
                p.requires_grad = True

            self.discriminator.zero_grad()
            fake,fakedepth,bn = self.generator(proj, proj_labels_one_hot)
            fake = nn.Softmax(dim=1)(fake)
            fake_detach = fake.detach()
            fakedepth_detach = fakedepth.detach()
            inp = torch.cat((rgb_labels_one_hot,depth),dim=1)
            cond = proj_labels_one_hot#torch.cat((proj_labels_one_hot,proj),dim=1)
            dloss_real = self.discriminator(inp, cond)
            dloss_real = dloss_real.mean()
            dloss_real.backward(mone, retain_graph=True)
            fake_inp = torch.cat((fake_detach,fakedepth_detach),dim=1)
            dloss_fake = self.discriminator(fake_inp, cond)

            dloss_fake = dloss_fake.mean()
            dloss_fake.backward(-1 * mone, retain_graph=True)
            real_inp = torch.cat((rgb_labels_one_hot,depth),dim=1)
            gp = 10*compute_gradient_penalty(self.discriminator, real_inp.data, fake_inp.data,
                                          cond.data)

            gp.backward(retain_graph=True)

            self.optimizer['discriminator'].step()

            d_loss = dloss_fake - dloss_real + gp

            wdistance = dloss_real - dloss_fake
            for p in self.discriminator.module.parameters():
                p.requires_grad = False

            self.generator.zero_grad()
            fake,fakedepth,bn = self.generator(proj, proj_labels_one_hot)
            fake = nn.Softmax(dim=1)(fake)


            inp = torch.cat((fake,fakedepth),dim=1)
            g_loss = self.discriminator(inp, cond)
            g_loss = g_loss.mean()
            g_loss.backward( mone, retain_graph=True)
            ls = 0.1*self.ls(fake, rgb_labels.squeeze(1))

            depth_loss = 0.1*self.depth_loss(fakedepth,rgb)

            ssim = 0.1*self.ssim(fakedepth,depth)

            mae = 0.1*self.MAE(fakedepth, depth)

            weighted_cross = 0.1*self.wc(torch.log(fake.clamp(min=1e-8)),rgb_labels.squeeze(1))
            entropy_loss = self.entropy(fake)
            #print(rgb_labels_one_hot.shape)
            #rmi = 0.1*self.rmi(fake,rgb_labels_one_hot)

            loss = ls + ssim + depth_loss + mae + weighted_cross + entropy_loss #+ rmi
            loss.backward()

            self.optimizer['generator'].step()
            self.schedulers['generator'].step()
            self.schedulers['discriminator'].step()
            self.generator.zero_grad()
            if engine.state.iteration % self.args.print_freq == 0:
                directory = os.path.dirname(
                    self.args.output_dir + "/examples/epoch_{}/".format(engine.state.epoch))
                if not os.path.exists(directory):
                    os.makedirs(directory)

                gpc_fake_argmax = fake.argmax(dim=1, keepdim=True).cpu().numpy()
                entropy = Entropy(reduce=False)(fake).detach().cpu().numpy()
                #fake_ = fake__.argmax(dim=1, keepdim=True).detach().cpu().numpy()
                lidar = parser.to_color(proj_labels[0].cpu().numpy().astype(np.int32).astype(np.uint8))
                save_image(rgb[0], self.args.output_dir + "/examples/epoch_{}/rgb.png".format(engine.state.epoch))
                save_image(rgb[-1], self.args.output_dir + "/examples/epoch_{}/rgb_last.png".format(engine.state.epoch))
                cv2.imwrite(
                    self.args.output_dir + "/examples/epoch_{}/generated_lidar.png".format(engine.state.epoch),
                    lidar)

                colorize_mask(rgb_labels[0][0].cpu().numpy()).save(
                    self.args.output_dir + "/examples/epoch_{}/cam_sem.png".format(engine.state.epoch))
                colorize_mask(gpc_fake_argmax[0][0]).save(
                    self.args.output_dir + "/examples/epoch_{}/generated_camseg.png".format(engine.state.epoch))
                # colorize_mask(fake_[0][0]   ).save(
                #     self.args.output_dir + "/examples/epoch_{}/generated_camseg_full.png".format(engine.state.epoch))


                plt.imsave(self.args.output_dir + "/examples/epoch_{}/depth.png".format(engine.state.epoch),depth[0][0].detach().cpu().numpy()*255)

                plt.imsave(self.args.output_dir + "/examples/epoch_{}/depth_fake.png".format(engine.state.epoch), fakedepth[0][0].detach().cpu().numpy()*255)



            return {'g_loss': -g_loss.mean().item(),
                    'd_loss': d_loss.mean().item(),
                    'ls': ls.mean().item(),
                    'wdistance': wdistance.mean().item(),
                    'lrD': self.optimizer['discriminator'].param_groups[0]['lr'],
                    'lrG':self.optimizer['generator'].param_groups[0]['lr'],
                    #'kd_loss': 0,
                    'depth': depth_loss.mean().item(),
                    'ssim': ssim.mean().item(),
                    'mae': mae.mean().item(),
                    'weighted_cross': weighted_cross.mean().item(),
                    }

        trainer = Engine(step)
        self.t = trainer

        checkpoint_handler = ModelCheckpoint(self.args.output_dir + '/checkpoints/', 'training',
                                             save_interval=self.args.checkpoint_interval,
                                             n_saved=self.args.epochs, require_empty=False, iteration=self.args.epoch_c)

        monitoring_metrics = ['g_loss',
                              'd_loss',
                              'wdistance',
                              'ls',
                              'lrD',
                              'lrG',
                              #'kd_loss',
                              'depth',
                              'mae',
                              'FocalLoss',
                              'ssim',
                              'weighted_cross'
                              ]
        RunningAverage(alpha=0.98, output_transform=lambda x: x['g_loss']).attach(trainer, 'g_loss')
        RunningAverage(alpha=0.98, output_transform=lambda x: x['d_loss']).attach(trainer, 'd_loss')
        RunningAverage(alpha=0.98, output_transform=lambda x: x['wdistance']).attach(trainer, 'wdistance')
        RunningAverage(alpha=0.98, output_transform=lambda x: x['ls']).attach(trainer, 'ls')
        RunningAverage(alpha=0.01, output_transform=lambda x: x['lrD']).attach(trainer, 'lrD')
        RunningAverage(alpha=0.01, output_transform=lambda x: x['lrG']).attach(trainer, 'lrG')
        #RunningAverage(alpha=0.01, output_transform=lambda x: x['kd_loss']).attach(trainer, 'kd_loss')
        RunningAverage(alpha=0.98, output_transform=lambda x: x['depth']).attach(trainer, 'depth')
        RunningAverage(alpha=0.98, output_transform=lambda x: x['ssim']).attach(trainer, 'ssim')
        RunningAverage(alpha=0.98, output_transform=lambda x: x['mae']).attach(trainer, 'mae')
        RunningAverage(alpha=0.98, output_transform=lambda x: x['weighted_cross']).attach(trainer, 'weighted_cross')


        self.pbar = ProgressBar()
        self.pbar.attach(trainer, metric_names=monitoring_metrics)

        trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler,
                                  to_save={"discriminator": self.discriminator,
                                           "generator": self.generator,
                                           "optimizer_D": self.optimizer['discriminator'],
                                           "optimizer_G": self.optimizer['generator'],
                                           "scheduler_G": self.schedulers['generator'],
                                           "scheduler_D": self.schedulers['discriminator'],
                                           })

        @trainer.on(Events.EPOCH_STARTED(every=1))
        def evaluator(engine):
            acc = AverageMeter()
            iou = AverageMeter()
            self.generator.eval()
            self.discriminator.eval()
            self.evaluator.reset()
            directory = os.path.dirname(self.args.output_dir+"/examples/epoch_{}/".format(engine.state.epoch))
            if not os.path.exists(directory):
                os.makedirs(directory)
            directory = os.path.dirname(self.args.output_dir+"/examples/epoch_{}/validation/".format(engine.state.epoch))
            if not os.path.exists(directory):
                os.makedirs(directory)
            directory_depth = os.path.dirname(
                self.args.output_dir + "/examples/epoch_{}/depth/".format(engine.state.epoch))
            if not os.path.exists(directory_depth):
                os.makedirs(directory_depth)
            for i, (_, _, _, _, proj, _, proj_labels_one_hot, _,path_seq_1,path_name_1) in enumerate(parser.validloader):
                directory = os.path.dirname(
                    args.output_dir + "/examples/epoch_{}/depth/{:02}/".format(engine.state.epoch,int(path_seq_1[0])))
                if not os.path.exists(directory):
                    os.makedirs(directory)

                directory = os.path.dirname(
                    args.output_dir + "/examples/epoch_{}/depth_fake/{:02}/".format(engine.state.epoch, int(path_seq_1[0])))
                if not os.path.exists(directory):
                    os.makedirs(directory)


                print("\r Validation Step {}/{}".format(i,len(parser.validloader)),end='')
                proj = proj.to('cuda:1')
                proj_labels_one_hot = proj_labels_one_hot.to('cuda:1')
                rgb_labels = rgb_labels.to('cuda:1').long()
                generated_img, fakedepth, bn= self.generator(proj, proj_labels_one_hot)
                generated_img = generated_img
                generated_img = nn.Softmax(dim=1)(generated_img)
                generated_argmax = generated_img.argmax(dim=1, keepdim=True)

                #self.evaluator.addBatch(generated_argmax,rgb_labels)
                generated_argmax = colorize_mask(generated_argmax[0][0].cpu().numpy()).convert(mode='RGB')
                convertion_dict = {0: 0,
                                   1: 26,
                                   2: 33,
                                   3: 32,
                                   4: 27,
                                   5: 29,
                                   6: 24,
                                   7: 7,
                                   8: 8,
                                   9: 11,
                                   10: 13,
                                   11: 21,
                                   12: 22,
                                   13: 17,
                                   14: 20}
                #fakedepth = nn.Upsample(size=(376, 1241), mode='bilinear', align_corners=True)(fakedepth)
                #print(generated_argmax[0][0].shape)
                # newArray2 = np.copy(generated_argmax[0][0].cpu().numpy())  # .astype('uint8')
                # for k, v in convertion_dict.items():
                #     newArray2[generated_ar    gmax[0][0].cpu().numpy() == k] = v

                # np.save(self.args.output_dir+"/examples/epoch_{}/depth/{:02}/360_{:06}".format(engine.state.epoch,  int(path_seq_1[0]), int(
                #         path_name_1[0].split(".")[0])),fakedepth[0].detach().cpu().numpy())
                # cv2.imwrite(self.args.output_dir+"/examples/epoch_{}/depth_fake/{:02}/360_{:06}.png".format(engine.state.epoch,  int(path_seq_1[0]), int(
                #         path_name_1[0].split(".")[0])), newArray2)
                generated_argmax.save(
                    self.args.output_dir+"/examples/epoch_{}/validation/seq{:02}_semantic_360_{:06}.png".format(engine.state.epoch, int(path_seq_1[0]), path_name_1[0].split(".")[0]))

            accuracy = self.evaluator.getacc()
            jaccard, class_jaccard = self.evaluator.getIoU()
            acc.update(accuracy.item(), rgb.size(0))
            iou.update(jaccard.item(), rgb.size(0))
            del proj
            del proj_labels_one_hot
            del rgb
            del rgb_labels
            del rgb_labels_one_hot
            print("mIoU:", iou.avg)\

            print("Acc:", accuracy)
            for i, jacc in enumerate(class_jaccard):
                print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                    i=i, class_str=i, jacc=jacc))
            self.info['miou'] = iou.avg
            self.info['acc'] = acc.avg

            # save_to_log(logdir=self.args.output_dir,
            #             logger=self.tb_logger,
            #             info=self.info,
            #             epoch=engine.state.epoch)
            generator.train()
            discriminator.train()



        @trainer.on(Events.EPOCH_COMPLETED)
        def print_times(engine):
            self.pbar.log_message(
                'Epoch {} done. Time per batch: {:.3f}[s]'.format(engine.state.epoch, self.timer.value()))
            self.timer.reset()

        @trainer.on(Events.ITERATION_COMPLETED(every=args.print_freq))
        def print_logs(engine):
            columns = engine.state.metrics.keys()
            values = [str(round(value, 5)) for value in engine.state.metrics.values()]
            self.info['g_loss'] = engine.state.output['g_loss']
            self.info['d_loss'] = engine.state.output['d_loss']
            self.info['ls'] = engine.state.output['ls']
            self.info['wdistance'] = engine.state.output['wdistance']
            self.info['lrD'] = engine.state.output['lrD']
            self.info['lrG'] = engine.state.output['lrG']
            self.info['weighted_cross'] = engine.state.output['weighted_cross']

            # save_to_log(logdir=self.args.output_dir,
            #             logger=self.tb_logger,
            #             info=self.info,
            #             epoch=engine.state.epoch)

            i = (engine.state.iteration % len(self.trainDataLoader))
            message = '[{epoch}/{max_epoch}][{i}/{max_i}]'.format(epoch=engine.state.epoch,
                                                                  max_epoch=self.args.epochs,
                                                                  i=i,
                                                                  max_i=len(self.trainDataLoader))
            for name, value in zip(columns, values):
                message += ' | {name}: {value}'.format(name=name, value=value)

            self.pbar.log_message(message)

        @trainer.on(Events.EPOCH_COMPLETED)
        def print_times(engine):
            self.pbar.log_message(
                'Epoch {} done. Time per batch: {:.3f}[s]'.format(engine.state.epoch, self.timer.value()))
            self.timer.reset()

        @trainer.on(Events.STARTED)
        def loaded(engine):
            if self.epoch_c != 0:
                engine.state.epoch = self.epoch_c
                engine.state.iteration = self.args.epoch_c * len(self.trainDataLoader)

    def update_learning_rate(self, epoch):
        if epoch > 50:
            lrd = 0.001  / 0
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:

            new_lr_G = new_lr / 2
            new_lr_D = new_lr * 2

            for param_group in self.optimizer['discriminator'].param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer['generator'].param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr

    def trainer(self):
        return self.t
