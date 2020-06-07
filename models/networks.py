# encoding: utf-8
import copy
from torchvision.models.resnet import resnet50, Bottleneck

from models.attention import *
from models.split_attention.resnest import resnest50


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class BFE(nn.Module):
    def __init__(self, num_classes, width_ratio=0.5, height_ratio=0.5):
        super(BFE, self).__init__()
        # resnet = resnet50()
        # resnet.load_state_dict(torch.load('./resnet50-19c8e357.pth'))
        # print("load ResNet50 parameters")
        print("load ResNeSt50 and ResNet50 parameters")
        model = resnet50()
        resnet = resnest50()
        model.load_state_dict(torch.load('../resnet50-19c8e357.pth'))
        resnet.load_state_dict(torch.load('../resnest50-528c19ca.pth'))

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            CAM_Module(256),
            resnet.layer2,
            CAM_Module(512),
        )
        self.res_part_head = nn.Sequential(
            resnet.layer3,
            CAM_Module(1024)
        )
        self.res_part = nn.Sequential(
            Bottleneck(1024, 512, stride=1, downsample=nn.Sequential(
                nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(2048),
            )),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512),
        )
        # self.res_part.load_state_dict(resnet.layer4.state_dict())
        self.res_part.load_state_dict(model.layer4.state_dict())
        self.c4 = CAM_Module(2048)

        self.part_pool = nn.AdaptiveMaxPool2d((1, 1))
        # self.part_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.res_part2 = Bottleneck(2048, 512)
        self.batch_crop = BatchDrop(height_ratio, width_ratio)

        self.reduction = nn.Sequential(
            nn.Linear(2048, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.reduction.apply(weights_init_kaiming)
        self.softmax = nn.Linear(1024, num_classes)
        self.softmax.apply(weights_init_kaiming)

        # Auxiliary branch
        ab_vector_size = 256
        reduction = nn.Sequential(
            nn.Conv2d(2048, ab_vector_size, 1),
            nn.BatchNorm2d(ab_vector_size),
            nn.ReLU(inplace=True)
        )
        self.auxiliary_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.auxiliary_softmax = nn.Linear(ab_vector_size, num_classes)
        self.auxiliary_softmax.apply(weights_init_kaiming)
        self.auxiliary_reduction = copy.deepcopy(reduction)
        self.auxiliary_reduction.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.backbone(x)
        x = self.res_part_head(x)
        x = self.res_part(x)
        x = self.c4(x)
        predict = []
        triplet_features = []
        softmax_features = []

        # auxiliary branch
        auxiliary = self.auxiliary_avgpool(x)
        auxiliary_triplet_feature = self.auxiliary_reduction(auxiliary).squeeze()
        auxiliary_softmax_class = self.auxiliary_softmax(auxiliary_triplet_feature)
        softmax_features.append(auxiliary_softmax_class)
        triplet_features.append(auxiliary_triplet_feature)
        predict.append(auxiliary_triplet_feature)

        # main branch
        x = self.res_part2(x)
        x = self.batch_crop(x)
        triplet_feature = self.part_pool(x).squeeze()
        feature = self.reduction(triplet_feature)
        softmax_feature = self.softmax(feature)
        triplet_features.append(feature)
        softmax_features.append(softmax_feature)
        predict.append(feature)

        if self.training:
            return triplet_features, softmax_features
        else:
            return torch.cat(predict, 1)

    def get_optim_policy(self):
        params = [
            {'params': self.backbone.parameters()},
            {'params': self.res_part_head.parameters()},
            {'params': self.res_part.parameters()},
            {'params': self.c4.parameters()},
            {'params': self.res_part2.parameters()},
            {'params': self.reduction.parameters()},
            {'params': self.softmax.parameters()},
            {'params': self.auxiliary_reduction.parameters()},
            {'params': self.auxiliary_softmax.parameters()},

        ]
        return params


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total : {:.2f}M, Trainable : {:.2f}M".format(total_num / 1000000, trainable_num / 1000000))


if __name__ == "__main__":
    get_parameter_number(BFE(751))
    data = torch.Tensor(torch.randn(2, 3, 384, 128))
    model = BFE(751)
    out = model(data)
    print(model.get_optim_policy())
