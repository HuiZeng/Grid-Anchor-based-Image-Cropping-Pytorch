import torch
import torch.nn as nn
import torchvision.models as models
from roi_align.modules.roi_align import RoIAlignAvg, RoIAlign
from rod_align.modules.rod_align import RoDAlignAvg, RoDAlign
import torch.nn.init as init
from ShuffleNetV2 import shufflenetv2
from mobilenetv2 import MobileNetV2
from thop import profile


class vgg_base(nn.Module):

    def __init__(self, loadweights=True, downsample=4):
        super(vgg_base, self).__init__()

        vgg = models.vgg16(pretrained=True)

        if downsample == 4:
            self.feature = nn.Sequential(vgg.features[:-1])
        elif downsample == 5:
            self.feature = nn.Sequential(vgg.features)

        self.feature3 = nn.Sequential(vgg.features[:23])
        self.feature4 = nn.Sequential(vgg.features[23:30])
        self.feature5 = nn.Sequential(vgg.features[30:])

        #flops, params = profile(self.feature, input_size=(1, 3, 256,256))

    def forward(self, x):
        #return self.feature(x)
        f3 = self.feature3(x)
        f4 = self.feature4(f3)
        f5 = self.feature5(f4)
        return f3, f4, f5

class resnet50_base(nn.Module):

    def __init__(self, loadweights=True, downsample=4):
        super(resnet50_base, self).__init__()

        resnet50 = models.resnet50(pretrained=True)

        self.feature3 = nn.Sequential(resnet50.conv1,resnet50.bn1,resnet50.relu,resnet50.maxpool,resnet50.layer1,resnet50.layer2)
        self.feature4 = nn.Sequential(resnet50.layer3)
        self.feature5 = nn.Sequential(resnet50.layer4)

        #flops, params = profile(self.feature, input_size=(1, 3, 256,256))

    def forward(self, x):
        #return self.feature(x)
        f3 = self.feature3(x)
        f4 = self.feature4(f3)
        f5 = self.feature5(f4)
        return f3, f4, f5


class mobilenetv2_base(nn.Module):

    def __init__(self, loadweights=True, downsample=4, model_path='pretrained_model/mobilenetv2_1.0-0c6065bc.pth'):
        super(mobilenetv2_base, self).__init__()

        model = MobileNetV2(width_mult=1.0)

        if loadweights:
            model.load_state_dict(torch.load(model_path))

        #if downsample == 4:
        #    self.feature = nn.Sequential(model.features[:14])
        #elif downsample == 5:
        #    self.feature = nn.Sequential(model.features)

        self.feature3 = nn.Sequential(model.features[:7])
        self.feature4 = nn.Sequential(model.features[7:14])
        self.feature5 = nn.Sequential(model.features[14:])

        #flops, params = profile(self.feature, input_size=(1, 3, 256,256))

    def forward(self, x):
        #return self.feature(x)
        f3 = self.feature3(x)
        f4 = self.feature4(f3)
        f5 = self.feature5(f4)
        return f3, f4, f5


class shufflenetv2_base(nn.Module):

    def __init__(self, loadweights=True, downsample=4, model_path='pretrained_model/shufflenetv2_x1_69.402_88.374.pth.tar'):
        super(shufflenetv2_base, self).__init__()

        model = shufflenetv2(width_mult=1.0)

        if loadweights:
            model.load_state_dict(torch.load(model_path))

        self.feature3 = nn.Sequential(model.conv1, model.maxpool, model.features[:4])
        self.feature4 = nn.Sequential(model.features[4:12])
        self.feature5 = nn.Sequential(model.features[12:])

        #if downsample == 4:
        #    self.feature = nn.Sequential(model.conv1, model.maxpool, model.features[:12])
        #elif downsample == 5:
        #    self.feature = nn.Sequential(model.conv1, model.maxpool, model.features)

        #flops, params = profile(self.feature, input_size=(1, 3, 256,256))

    def forward(self, x):
        #return self.feature(x)
        f3 = self.feature3(x)
        f4 = self.feature4(f3)
        f5 = self.feature5(f4)
        return f3, f4, f5


def fc_layers(reddim = 32, alignsize = 8):
    conv1 = nn.Sequential(nn.Conv2d(reddim, 768, kernel_size=alignsize, padding=0),nn.BatchNorm2d(768),nn.ReLU(inplace=True))
    #conv1 = nn.Sequential(nn.Conv2d(reddim, 768, kernel_size=3, padding=1, stride=2),nn.BatchNorm2d(768),nn.ReLU(inplace=True),
    #                      nn.Conv2d(768, reddim, kernel_size=1, padding=0),nn.BatchNorm2d(reddim),nn.ReLU(inplace=True),
    #                      nn.Conv2d(reddim, 768, kernel_size=3, padding=1,stride=2),nn.BatchNorm2d(768),nn.ReLU(inplace=True),
    #                      nn.Conv2d(768, reddim, kernel_size=1, padding=0),nn.BatchNorm2d(reddim),nn.ReLU(inplace=True),
    #                      nn.Conv2d(reddim, 768, kernel_size=3, padding=0,stride=1),nn.BatchNorm2d(768),nn.ReLU(inplace=True))
    #conv1 = nn.Sequential(nn.Conv2d(reddim, 768, kernel_size=5, padding=2, stride=2),nn.BatchNorm2d(768),nn.ReLU(inplace=True),
    #                      nn.Conv2d(768, reddim, kernel_size=1, padding=0),nn.BatchNorm2d(reddim),nn.ReLU(inplace=True),
    #                      nn.Conv2d(reddim, 768, kernel_size=5, padding=0,stride=1),nn.BatchNorm2d(768),nn.ReLU(inplace=True))
    conv2 = nn.Sequential(nn.Conv2d(768, 128, kernel_size=1),nn.BatchNorm2d(128),nn.ReLU(inplace=True))
    dropout = nn.Dropout(p=0.5)
    conv3 = nn.Conv2d(128, 1, kernel_size=1)
    layers = nn.Sequential(conv1, conv2, dropout, conv3)
    return layers


class crop_model_single_scale(nn.Module):

    def __init__(self, alignsize = 8, reddim = 8, loadweight = True, model = None, downsample=4):
        super(crop_model_single_scale, self).__init__()

        if model == 'shufflenetv2':
            self.Feat_ext = shufflenetv2_base(loadweight,downsample)
            if downsample == 4:
                self.DimRed = nn.Conv2d(232, reddim, kernel_size=1, padding=0)
            else:
                self.DimRed = nn.Conv2d(464, reddim, kernel_size=1, padding=0)
        elif model == 'mobilenetv2':
            self.Feat_ext = mobilenetv2_base(loadweight,downsample)
            if downsample == 4:
                self.DimRed = nn.Conv2d(96, reddim, kernel_size=1, padding=0)
            else:
                self.DimRed = nn.Conv2d(320, reddim, kernel_size=1, padding=0)
        elif model == 'vgg16':
            self.Feat_ext = vgg_base(loadweight,downsample)
            self.DimRed = nn.Conv2d(512, reddim, kernel_size=1, padding=0)
        elif model == 'resnet50':
            self.Feat_ext = resnet50_base(loadweight,downsample)
            self.DimRed = nn.Conv2d(1024, reddim, kernel_size=1, padding=0)

        self.RoIAlign = RoIAlignAvg(alignsize, alignsize, 1.0/2**downsample)
        self.RoDAlign = RoDAlignAvg(alignsize, alignsize, 1.0/2**downsample)
        self.FC_layers = fc_layers(reddim*2, alignsize)

        #flops, params = profile(self.FC_layers, input_size=(1,reddim*2,9,9))

    def forward(self, im_data, boxes):

        f3,base_feat,f5 = self.Feat_ext(im_data)
        red_feat = self.DimRed(base_feat)
        RoI_feat = self.RoIAlign(red_feat, boxes)
        RoD_feat = self.RoDAlign(red_feat, boxes)
        final_feat = torch.cat((RoI_feat, RoD_feat), 1)
        prediction = self.FC_layers(final_feat)
        return prediction

    def _init_weights(self):
        print('Initializing weights...')
        self.DimRed.apply(weights_init)
        self.FC_layers.apply(weights_init)


class crop_model_multi_scale_individual(nn.Module):

    def __init__(self, alignsize = 8, reddim = 32, loadweight = True, model = None, downsample = 4):
        super(crop_model_multi_scale_individual, self).__init__()

        if model == 'shufflenetv2':
            self.Feat_ext1 = shufflenetv2_base(loadweight,downsample)
            self.Feat_ext2 = shufflenetv2_base(loadweight,downsample)
            self.Feat_ext3 = shufflenetv2_base(loadweight,downsample)
            self.DimRed = nn.Conv2d(232, reddim, kernel_size=1, padding=0)
        elif model == 'mobilenetv2':
            self.Feat_ext1 = mobilenetv2_base(loadweight,downsample)
            self.Feat_ext2 = mobilenetv2_base(loadweight,downsample)
            self.Feat_ext3 = mobilenetv2_base(loadweight,downsample)
            self.DimRed = nn.Conv2d(96, reddim, kernel_size=1, padding=0)
        elif model == 'vgg16':
            self.Feat_ext1 = vgg_base(loadweight,downsample)
            self.Feat_ext2 = vgg_base(loadweight,downsample)
            self.Feat_ext3 = vgg_base(loadweight,downsample)
            self.DimRed = nn.Conv2d(512, reddim, kernel_size=1, padding=0)

        self.downsample2 = nn.UpsamplingBilinear2d(scale_factor=1.0/2.0)
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2.0)
        self.RoIAlign = RoIAlignAvg(alignsize, alignsize, 1.0/2**downsample)
        self.RoDAlign = RoDAlignAvg(alignsize, alignsize, 1.0/2**downsample)
        self.FC_layers = fc_layers(reddim*2, alignsize)

    def forward(self, im_data, boxes):

        base_feat = self.Feat_ext1(im_data)

        up_im = self.upsample2(im_data)
        up_feat = self.Feat_ext2(up_im)
        up_feat = self.downsample2(up_feat)

        down_im = self.downsample2(im_data)
        down_feat = self.Feat_ext3(down_im)
        down_feat = self.upsample2(down_feat)

        #cat_feat = torch.cat((base_feat,up_feat,down_feat),1)
        cat_feat = 0.5*base_feat + 0.35*up_feat + 0.15*down_feat
        red_feat = self.DimRed(cat_feat)
        RoI_feat = self.RoIAlign(red_feat, boxes)
        RoD_feat = self.RoDAlign(red_feat, boxes)
        final_feat = torch.cat((RoI_feat, RoD_feat), 1)
        prediction = self.FC_layers(final_feat)
        return prediction

    def _init_weights(self):
        print('Initializing weights...')
        self.DimRed.apply(weights_init)
        self.FC_layers.apply(weights_init)

class crop_model_multi_scale_shared(nn.Module):

    def __init__(self, alignsize = 8, reddim = 32, loadweight = True, model = None, downsample = 4):
        super(crop_model_multi_scale_shared, self).__init__()

        if model == 'shufflenetv2':
            self.Feat_ext = shufflenetv2_base(loadweight,downsample)
            self.DimRed = nn.Conv2d(812, reddim, kernel_size=1, padding=0)
        elif model == 'mobilenetv2':
            self.Feat_ext = mobilenetv2_base(loadweight,downsample)
            self.DimRed = nn.Conv2d(448, reddim, kernel_size=1, padding=0)
        elif model == 'vgg16':
            self.Feat_ext = vgg_base(loadweight,downsample)
            self.DimRed = nn.Conv2d(1536, reddim, kernel_size=1, padding=0)
        elif model == 'resnet50':
            self.Feat_ext = resnet50_base(loadweight,downsample)
            self.DimRed = nn.Conv2d(3584, reddim, kernel_size=1, padding=0)

        self.downsample2 = nn.UpsamplingBilinear2d(scale_factor=1.0/2.0)
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2.0)
        self.RoIAlign = RoIAlignAvg(alignsize, alignsize, 1.0/2**downsample)
        self.RoDAlign = RoDAlignAvg(alignsize, alignsize, 1.0/2**downsample)
        self.FC_layers = fc_layers(reddim*2, alignsize)


    def forward(self, im_data, boxes):

        #base_feat = self.Feat_ext(im_data)

        #up_im = self.upsample2(im_data)
        #up_feat = self.Feat_ext(up_im)
        #up_feat = self.downsample2(up_feat)

        #down_im = self.downsample2(im_data)
        #down_feat = self.Feat_ext(down_im)
        #down_feat = self.upsample2(down_feat)

        f3,f4,f5 = self.Feat_ext(im_data)
        cat_feat = torch.cat((self.downsample2(f3),f4,0.5*self.upsample2(f5)),1)

        #cat_feat = torch.cat((base_feat,up_feat,down_feat),1)
        #cat_feat = base_feat + 0.35*up_feat + 0.15*down_feat
        red_feat = self.DimRed(cat_feat)
        RoI_feat = self.RoIAlign(red_feat, boxes)
        RoD_feat = self.RoDAlign(red_feat, boxes)
        final_feat = torch.cat((RoI_feat, RoD_feat), 1)
        prediction = self.FC_layers(final_feat)
        return prediction

    def _init_weights(self):
        print('Initializing weights...')
        self.DimRed.apply(weights_init)
        self.FC_layers.apply(weights_init)

def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def build_crop_model(scale='single', alignsize=8, reddim=32, loadweight=True, model=None, downsample=4):

    if scale=='single':
        return crop_model_single_scale(alignsize, reddim, loadweight, model, downsample)
    elif scale=='multi':
        return crop_model_multi_scale_shared(alignsize, reddim, loadweight, model, downsample)



