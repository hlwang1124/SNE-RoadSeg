import torch
import torch.nn as nn
from torch.nn import init
import torchvision
from torch.optim import lr_scheduler
import torch.nn.functional as F


### help functions ###
def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        lambda_rule = lambda epoch: opt.lr_gamma ** ((epoch+1) // opt.lr_decay_epochs)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,step_size=opt.lr_decay_iters, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net, init_type='normal', gain=0.02):
        net = net
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'pretrained':
                    pass
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None and init_type != 'pretrained':
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)
        print('initialize network with %s' % init_type)
        net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)

    for root_child in net.children():
        for children in root_child.children():
            if children in root_child.need_initialization:
                init_weights(children, init_type, gain=init_gain)
            else:
                init_weights(children, "pretrained", gain=init_gain)
    return net

def define_RoadSeg(num_labels, use_sne=True, init_type='xavier', init_gain=0.02, gpu_ids=[]):

    net = RoadSeg(num_labels, use_sne)
    return init_net(net, init_type, init_gain, gpu_ids)


### network ###
class conv_block_nested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)
        return output

class upsample_layer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(upsample_layer, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.conv1(x)
        x = self.bn1(x)
        output = self.activation(x)
        return output


class RoadSeg(nn.Module):
    """Our RoadSeg takes rgb and another (depth or normal) as input,
    and outputs freespace predictions.
    """
    def __init__(self, num_labels, use_sne):
        super(RoadSeg, self).__init__()

        self.num_resnet_layers = 152

        if self.num_resnet_layers == 18:
            resnet_raw_model1 = torchvision.models.resnet18(pretrained=True)
            resnet_raw_model2 = torchvision.models.resnet18(pretrained=True)
            filters = [64, 64, 128, 256, 512]
        elif self.num_resnet_layers == 34:
            resnet_raw_model1 = torchvision.models.resnet34(pretrained=True)
            resnet_raw_model2 = torchvision.models.resnet34(pretrained=True)
            filters = [64, 64, 128, 256, 512]
        elif self.num_resnet_layers == 50:
            resnet_raw_model1 = torchvision.models.resnet50(pretrained=True)
            resnet_raw_model2 = torchvision.models.resnet50(pretrained=True)
            filters = [64, 256, 512, 1024, 2048]
        elif self.num_resnet_layers == 101:
            resnet_raw_model1 = torchvision.models.resnet101(pretrained=True)
            resnet_raw_model2 = torchvision.models.resnet101(pretrained=True)
            filters = [64, 256, 512, 1024, 2048]
        elif self.num_resnet_layers == 152:
            resnet_raw_model1 = torchvision.models.resnet152(pretrained=True)
            resnet_raw_model2 = torchvision.models.resnet152(pretrained=True)
            filters = [64, 256, 512, 1024, 2048]
        else:
            raise NotImplementedError('num_resnet_layers should be 18, 34, 50, 101 or 152')

        ### encoder for another image ###
        if use_sne:
            self.encoder_another_conv1 = resnet_raw_model1.conv1
        else:
            # if another image is depth, initialize the weights of the first layer
            self.encoder_another_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.encoder_another_conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model1.conv1.weight.data, dim=1), dim=1)

        self.encoder_another_bn1 = resnet_raw_model1.bn1
        self.encoder_another_relu = resnet_raw_model1.relu
        self.encoder_another_maxpool = resnet_raw_model1.maxpool
        self.encoder_another_layer1 = resnet_raw_model1.layer1
        self.encoder_another_layer2 = resnet_raw_model1.layer2
        self.encoder_another_layer3 = resnet_raw_model1.layer3
        self.encoder_another_layer4 = resnet_raw_model1.layer4

        ###  encoder for rgb image  ###
        self.encoder_rgb_conv1 = resnet_raw_model2.conv1
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer4 = resnet_raw_model2.layer4

        ###  decoder  ###
        self.conv1_1 = conv_block_nested(filters[0]*2, filters[0], filters[0])
        self.conv2_1 = conv_block_nested(filters[1]*2, filters[1], filters[1])
        self.conv3_1 = conv_block_nested(filters[2]*2, filters[2], filters[2])
        self.conv4_1 = conv_block_nested(filters[3]*2, filters[3], filters[3])

        self.conv1_2 = conv_block_nested(filters[0]*3, filters[0], filters[0])
        self.conv2_2 = conv_block_nested(filters[1]*3, filters[1], filters[1])
        self.conv3_2 = conv_block_nested(filters[2]*3, filters[2], filters[2])

        self.conv1_3 = conv_block_nested(filters[0]*4, filters[0], filters[0])
        self.conv2_3 = conv_block_nested(filters[1]*4, filters[1], filters[1])

        self.conv1_4 = conv_block_nested(filters[0]*5, filters[0], filters[0])

        self.up2_0 = upsample_layer(filters[1], filters[0])
        self.up2_1 = upsample_layer(filters[1], filters[0])
        self.up2_2 = upsample_layer(filters[1], filters[0])
        self.up2_3 = upsample_layer(filters[1], filters[0])

        self.up3_0 = upsample_layer(filters[2], filters[1])
        self.up3_1 = upsample_layer(filters[2], filters[1])
        self.up3_2 = upsample_layer(filters[2], filters[1])

        self.up4_0 = upsample_layer(filters[3], filters[2])
        self.up4_1 = upsample_layer(filters[3], filters[2])

        self.up5_0 = upsample_layer(filters[4], filters[3])

        self.final = upsample_layer(filters[0], num_labels)

        ### layers without pretrained model need to be initialized ###
        self.need_initialization = [self.conv1_1, self.conv2_1, self.conv3_1, self.conv4_1, self.conv1_2,
                                    self.conv2_2, self.conv3_2, self.conv1_3, self.conv2_3, self.conv1_4,
                                    self.up2_0, self.up2_1, self.up2_2, self.up2_3, self.up3_0, self.up3_1,
                                    self.up3_2, self.up4_0, self.up4_1, self.up5_0, self.final]

    def forward(self, rgb, another):
        # encoder
        rgb = self.encoder_rgb_conv1(rgb)
        rgb = self.encoder_rgb_bn1(rgb)
        rgb = self.encoder_rgb_relu(rgb)
        another = self.encoder_another_conv1(another)
        another = self.encoder_another_bn1(another)
        another = self.encoder_another_relu(another)
        rgb = rgb + another
        x1_0 = rgb

        rgb = self.encoder_rgb_maxpool(rgb)
        another = self.encoder_another_maxpool(another)
        rgb = self.encoder_rgb_layer1(rgb)
        another = self.encoder_another_layer1(another)
        rgb = rgb + another
        x2_0 = rgb

        rgb = self.encoder_rgb_layer2(rgb)
        another = self.encoder_another_layer2(another)
        rgb = rgb + another
        x3_0 = rgb

        rgb = self.encoder_rgb_layer3(rgb)
        another = self.encoder_another_layer3(another)
        rgb = rgb + another
        x4_0 = rgb

        rgb = self.encoder_rgb_layer4(rgb)
        another = self.encoder_another_layer4(another)
        x5_0 = rgb + another

        # decoder
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], dim=1))
        x4_1 = self.conv4_1(torch.cat([x4_0, self.up5_0(x5_0)], dim=1))

        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_1(x3_1)], dim=1))
        x3_2 = self.conv3_2(torch.cat([x3_0, x3_1, self.up4_1(x4_1)], dim=1))

        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_2(x2_2)], dim=1))
        x2_3 = self.conv2_3(torch.cat([x2_0, x2_1, x2_2, self.up3_2(x3_2)], dim=1))

        x1_4 = self.conv1_4(torch.cat([x1_0, x1_1, x1_2, x1_3, self.up2_3(x2_3)], dim=1))
        out = self.final(x1_4)
        return out


class SegmantationLoss(nn.Module):
    def __init__(self, class_weights=None):
        super(SegmantationLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
    def __call__(self, output, target, pixel_average=True):
        if pixel_average:
            return self.loss(output, target) #/ target.data.sum()
        else:
            return self.loss(output, target)
