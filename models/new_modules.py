import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import models.pointnet2_sem_seg_msg as PointNet
from models.AdaptiveWingLoss.core.models import FAN
from renderers import *
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    TexturesVertex,
)

class W_Discriminator(nn.Module):
    def __init__(self,input_channels = 512):
        super(W_Discriminator,self).__init__()
        slope = 0.2
        # self.linear1 = layers.Dense(512, kernel_initializer=get_weights(slope), input_shape=(512,))
        self.linear2 = nn.Linear(input_channels , 256)
        self.linear3 = nn.Linear(256 , 128)
        self.linear4 = nn.Linear(128 , 64)
        self.linear5 = nn.Linear(64 , 1)
        self.relu = nn.LeakyReLU(slope)

    def call(self, x):
        # x = self.linear1(x)
        # x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.linear5(x)
        return x



class GeometryEncoder_Projection(nn.Module):
    def __init__(self, projection_image_size=128 , point_out_channel = 24,out_code_channel = 1000):
        super(GeometryEncoder_Projection, self).__init__()
        self.image_size = projection_image_size
        self.pointnet = PointNet.get_model(num_classes=point_out_channel)
        self.encoder = torchvision.models.resnet18()
        self.encoder.conv1 = nn.Conv2d(point_out_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # layers = [
        #
        #     nn.conv2d(in_channel=point_out_channel,out_channel= 32,kernel_size=3,stride = 2,padding=1),
        #     nn.LeakyReLU(),
        #     # projection_image_size/2
        #     nn.conv2d(in_channel=32,out_channel= 64,kernel_size=3,stride = 2,padding=1),
        #     nn.LeakyReLU(),
        #     # projection_image_size/4
        #     nn.conv2d(in_channel=64, out_channel=64, kernel_size=3, stride=2, padding=1),
        #     nn.LeakyReLU(),
        #     # projection_image_size/8
        #     nn.conv2d(in_channel=64, out_channel=64, kernel_size=3, stride=2, padding=1),
        #     nn.LeakyReLU(),
        #     # projection_image_size/16  8*8*64 in size
        #
        # ]
        # self.conv_out = nn.Sequential(*layers)
        #
        # mlp = [
        #     nn.Linear(projection_image_size / 4 * projection_image_size / 4 *32 , out_code_channel),
        #     nn.LeakyReLU(),
        # ]
        # self.last_fc = nn.Sequential(*mlp)
    def forward(self, source_verts,source_faces,eye_tensor, at_tensor):
        # N,C,P = x.shape
        renderer = create_renderer(eye_tensor, at_tensor, image_size=self.image_size)
        source_geo_code,_ = self.pointnet(source_verts.permute(0, 2, 1))

        source_render_texture = TexturesVertex(verts_features=source_geo_code)
        source_train_meshes = Meshes(verts=source_verts,
                                     faces=source_faces,
                                     textures=source_render_texture)

        source_geo_feature_images = renderer(source_train_meshes)

        source_geo_feature_images = source_geo_feature_images.permute(0, 3, 1, 2)
        source_encoded_geo_features = self.encoder(source_geo_feature_images).contiguous()
        return source_encoded_geo_features #len 1000

# Convert into code directly.
class GeometryEncoder(nn.Module):
    def __init__(self,point_count=8192, point_out_channel = 24,out_code_channel = 2048):
        super(GeometryEncoder, self).__init__()
        self.pointnet = PointNet.get_model(num_classes=point_out_channel)

        layers = [
            nn.Linear(point_count*point_out_channel,out_code_channel),
            nn.ReLU(),
            nn.Linear(out_code_channel,out_code_channel),
            nn.ReLU(),
            nn.Linear(out_code_channel,out_code_channel),
            nn.ReLU()
        ]

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # N,C,P = x.shape
        x,_ = self.pointnet(x)
        x = torch.flatten(x,start_dim =1)
        x = self.mlp(x)
        return x

class AppearEncoder(nn.Module):
    def __init__(self, out_code_channel = 2048):
        super(AppearEncoder, self).__init__()
        self.inception_v3 = torchvision.models.inception_v3(pretrained=True)
        self.inception_v3.fc = nn.Identity() # (N,2048)
        self.inception_v3_preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # self.last_fc = nn.Linear(2048,out_code_channel)

    def forward(self, x):
        # N,C,P = x.shape
        x = self.inception_v3_preprocess(x)
        x = self.inception_v3(x)
        # x = F.leaky_relu(self.last_fc(x),negative_slope=0.2)
        if self.inception_v3.training:
            return x[0]
        else:
            return x




class LandmarkExtractor(nn.Module):
    # [ICCV 2019] Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression - Official Implementation
    def __init__(self, pretrained = True):
        super(LandmarkExtractor, self).__init__()
        self.landmarknet = FAN(4, False, False, 98)

        # self.last_fc = nn.Linear(point_count,out_code_channel)
        PRETRAINED_WEIGHTS = 'models/AdaptiveWingLoss/ckpt/WFLW_4HG.pth'
        if PRETRAINED_WEIGHTS != "None" and pretrained:
            checkpoint = torch.load(PRETRAINED_WEIGHTS)
            if 'state_dict' not in checkpoint:
                self.landmarknet.load_state_dict(checkpoint)
            else:
                pretrained_weights = checkpoint['state_dict']
                model_weights = self.landmarknet.state_dict()
                pretrained_weights = {k: v for k, v in pretrained_weights.items() \
                                      if k in model_weights}
                model_weights.update(pretrained_weights)
                self.landmarknet.load_state_dict(model_weights)

    def forward(self, x):
        # N,C,P = x.shape
        outputs, boundary_channels = self.landmarknet(x)
        return outputs[-1][:, :-1, :, :]


    def get_preds_fromhm(self, hm, center=None, scale=None, rot=None):
        max, idx = torch.max(
            hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
        idx += 1
        preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
        preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1)
        preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)

        for i in range(preds.size(0)):
            for j in range(preds.size(1)):
                hm_ = hm[i, j, :]
                pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
                if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                    diff = torch.FloatTensor(
                        [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                         hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                    preds[i, j].add_(diff.sign_().mul_(.25))

        preds.add_(-0.5)

        preds_orig = torch.zeros(preds.size())
        if center is not None and scale is not None:
            for i in range(hm.size(0)):
                for j in range(hm.size(1)):
                    preds_orig[i, j] = transform(
                        preds[i, j], center, scale, hm.size(2), rot, True)

        return preds
    def get_landmarks(self,x):
        outputs, boundary_channels = self.landmarknet(x)
        pred_heatmap = outputs[-1][:, :-1, :, :].detach().cpu()
        pred_landmarks = self.get_preds_fromhm(pred_heatmap)
        return pred_landmarks



class LatentEmbeddingMLP(nn.Module):
    #module for Latent Embedding
    def __init__(self,in_channel,mlp_width,out_channel):
        super(LatentEmbeddingMLP,self).__init__()
        self.f1 = nn.Linear(in_channel,mlp_width)
        self.f2 = nn.Linear(mlp_width,mlp_width)
        self.f3 = nn.Linear(mlp_width, mlp_width)
        self.f4 = nn.Linear(mlp_width,out_channel)
    def forward(self,x):
        x = F.leaky_relu(self.f1(x),negative_slope=0.2)
        x = F.leaky_relu(self.f2(x),negative_slope=0.2)
        x = F.leaky_relu(self.f3(x),negative_slope=0.2)
        x = F.leaky_relu(self.f4(x),negative_slope=0.2)
        return x
