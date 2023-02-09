import torch
import torch.nn as nn
from train.tasks.segmented.modules.ConvLSTM import ConvLSTM
from train.tasks.segmented.modules.SelfAttention import AxialAttentionBlock

class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3), stride=1,
                 pooling=True, drop_out=True):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.InstanceNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3, 3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.InstanceNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1)
        self.act4 = nn.LeakyReLU()
        self.bn3 = nn.InstanceNorm2d(out_filters)

        self.conv5 = nn.Conv2d(out_filters * 3, out_filters, kernel_size=(1, 1))
        self.act5 = nn.LeakyReLU()
        self.bn4 = nn.InstanceNorm2d(out_filters)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        resA = self.conv4(resA2)
        resA = self.act4(resA)
        resA3 = self.bn3(resA)

        concat = torch.cat((resA1, resA2, resA3), dim=1)
        resA = self.conv5(concat)
        resA = self.act5(resA)
        resA = self.bn4(resA)
        resA = shortcut + resA

        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            resB = self.pool(resB)

            return resB, resA
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB



class FutureHead(nn.Module):

    def __init__(self):

        super(FutureHead,self).__init__()
        self.attn = AxialAttentionBlock(768,32)
        self.conv3dS1 = nn.Conv3d(256,256, kernel_size=(1, 3, 3), padding=(0, 1, 1),stride=(2,1,1))
        self.conv3dS2 = nn.Conv3d(256, 256,kernel_size=(1, 3, 3), padding=(0, 1, 1),stride=(2,1,1))
        self.conv3dS3 = nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1),stride=(2,1,1))
        self.transpose1 = nn.ConvTranspose3d(256, 128, (1, 2, 2), (1, 2, 2))
        self.transpose2 = nn.ConvTranspose3d(128, 64, (1, 2, 2), (1, 2, 2))
        self.transpose3 = nn.ConvTranspose3d(64, 32, (1, 2, 2), (1, 2, 2))
        self.transpose4 = nn.ConvTranspose3d(32, 32, (1, 2, 2), (1, 2, 2))
        self.transpose5 = nn.ConvTranspose3d(32, 32, (1, 2, 2), (1, 2, 2))

        self.BNT1 = nn.BatchNorm3d(128)
        self.BNT2 = nn.BatchNorm3d(64)
        self.BNT3 = nn.BatchNorm3d(32)
        self.BNT4 = nn.BatchNorm3d(32)
        self.BNT5 = nn.BatchNorm3d(32)

        self.BN3dS1 = nn.BatchNorm3d(256)
        self.BN3dS2 = nn.BatchNorm3d(256)
        self.BN3dS3 = nn.BatchNorm3d(256)

        self.refine1 = ResBlock(32,32,pooling=False,dropout_rate=0.2)
        self.refine2 = ResBlock(32,32,pooling=False,dropout_rate=0.2)
        self.refine3 = ResBlock(32,32,pooling=False,dropout_rate=0.2)
        self.logit = nn.Conv2d(32,15,kernel_size=(1,1))
        self.upsample = nn.Upsample(size=(1,376,1241), mode='trilinear', align_corners=False)
        self.temporal_pooling = nn.AdaptiveMaxPool3d((1, 376, 1241))

    def forward(self, seg):
        #B,T,C,H,W
        #Soutput, Slast_state = self.semantic(seg)
        #Doutput, Dlast_state = self.depth(depth)
        #print(Soutput[-1].shape)
        seg = self.attn(nn.Upsample(size=(32,32),mode='bilinear',align_corners=True)(seg))
        seg = nn.Upsample(size=(4,32),mode='bilinear',align_corners=True)(seg).view(3, 256,4,32).unsqueeze(0)
        seg = seg.permute(0, 2, 1, 3, 4)

        Soutput = self.conv3dS1(seg)
        Soutput = self.BN3dS1(Soutput)
        Soutput = nn.functional.relu(Soutput)

        Soutput = self.conv3dS2(Soutput)
        Soutput = self.BN3dS2(Soutput)
        Soutput = nn.functional.relu(Soutput)

        Soutput = self.conv3dS3(Soutput)
        Soutput = self.BN3dS3(Soutput)
        Soutput = nn.functional.relu(Soutput)
        bn = Soutput.permute(0, 2, 1, 3, 4)[0]

        Soutput = self.transpose1(Soutput)
        Soutput = self.BNT1(Soutput)
        Soutput = nn.functional.relu(Soutput)

        Soutput = nn.Upsample(size=(1,24, 77), mode='trilinear', align_corners=True)(Soutput)

        Soutput = self.transpose2(Soutput)
        Soutput = self.BNT2(Soutput)

        Soutput = self.transpose3(Soutput)
        Soutput = self.BNT3(Soutput)
        Soutput = nn.functional.relu(Soutput)

        Soutput = self.transpose4(Soutput)
        Soutput = self.BNT4(Soutput)
        Soutput = nn.functional.relu(Soutput)

        Soutput = self.transpose5(Soutput)
        Soutput = self.BNT5(Soutput)
        Soutput = nn.functional.relu(Soutput)


        seg = self.temporal_pooling(self.upsample(Soutput)).permute(0, 2, 1, 3, 4)[0]
        seg = self.refine1(seg)
        seg = self.refine2(seg)
        seg = self.refine3(seg)

        seg = self.logit(seg)
        #seg = nn.functional.softmax(seg,dim=1)


        #depth = self.conv3dD1(depth.permute(0, 2, 1, 3, 4))
        #depth = self.BN3dD1(depth)
        #depth = nn.functional.relu(depth)

        #depth = self.conv3dD2(depth)
        #depth = self.BN3dD2(depth)
        #depth = nn.functional.relu(depth)

        #depth = self.conv3dD3(depth)
        #depth = self.BN3dD3(depth)
        #depth = nn.functional.relu(depth)

        #depth = self.temporal_pooling(depth).permute(0, 2, 1, 3, 4)[0]

        #depth = self.depth_final(depth)
        #depth = nn.functional.relu(depth)
        #next = torch.cat((seg,depth),dim=1)
        return seg,bn#, depth, next

