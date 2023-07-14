from renderers import *
from pytorch3d.io import IO
from pytorch3d.io import load_objs_as_meshes, load_obj
from Utils import *

import models.pointnet2_sem_seg_msg as PointNet
from models.encoders.psp_encoders import GradualStyleEncoder as StyleGANEncoder
from models.modules import PNet,PNet_NoRes
from models.stylegan2.model import StyleGAN2Generator,Discriminator
from models.new_modules import LandmarkExtractor,GeometryEncoder_Projection
from PIL import Image, ImageFont, ImageDraw
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    blending
)
from Metrics import cal_similarities
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import pyfqmr

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

def render_with_phong_renderer(input_mesh, eye_tensor, at_tensor, image_size=256):
    with torch.no_grad():
        renderer = create_phong_renderer(eye_tensor, at_tensor, image_size)
        gray_image = renderer(input_mesh)
        gray_image_np = (255 * gray_image[0, ..., :3].cpu().detach().numpy()).astype('uint8')
    return gray_image_np


def render_with_albedo_renderer(gt_mesh, eye_tensor, at_tensor, image_size=256):
    with torch.no_grad():
        renderer = create_renderer(eye_tensor, at_tensor, image_size)
        gt_image = renderer(gt_mesh)
        gt_image_np = (255 * gt_image[0, ..., :3].cpu().detach().numpy()).astype('uint8')
    return gt_image_np

def render_blank_image(image_width = 256,image_height = 256):
    blank_image = Image.new(mode="RGB",
                            size=(image_width, image_height))
    blank_image_np = np.array(blank_image)
    return blank_image_np


def build_generator(load_model_path,load_model_step,image_size = 256):
    def get_keys(d, name):
        if 'state_dict' in d:
            d = d['state_dict']
        d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
        return d_filt

    geo_feature_channels = 24

    GeoCodeExtractor = PointNet.get_model(num_classes=geo_feature_channels).cuda()
    AppearCodeExtractor = StyleGANEncoder(50, 'ir_se').cuda()
    # pSp_checkpoint_path = "./pretrained_modules/psp_ffhq_encode.pt"
    # ckpt = torch.load(pSp_checkpoint_path)
    # AppearCodeExtractor.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
    #
    # for i in AppearCodeExtractor.parameters():
    #     i.requires_grad = False
    # AppearCodeExtractor.eval()

    # Pnet = PNet(geo_feature_channels,512).cuda()
    Pnet = PNet_NoRes(geo_feature_channels, 512).cuda()
    Generator = StyleGAN2Generator(
        image_size, 512, channel_multiplier=2
    ).to(device)
    if load_model_path != '' and os.path.exists(load_model_path):
        print(timelog_str() + ' Loading pretrained weight from: ' + load_model_path + ', step: ' + str(load_model_step))
        AppearCodeExtractor.load_state_dict(
            torch.load(os.path.join(load_model_path, 'AppearCodeExtractor_' + str(load_model_step) + '.pth')))
        GeoCodeExtractor.load_state_dict(
            torch.load(os.path.join(load_model_path, 'GeoCodeExtractor_' + str(load_model_step) + '.pth')))
        Pnet.load_state_dict(
            torch.load(os.path.join(load_model_path, 'Pnet_' + str(load_model_step) + '.pth')))
        Generator.load_state_dict(
            torch.load(os.path.join(load_model_path, 'Generator_' + str(load_model_step) + '.pth')))
    else:
        print("No model found in :"+load_model_path)
    GeoCodeExtractor.eval()
    Pnet.eval()
    Generator.eval()
    AppearCodeExtractor.eval()
    class ComposedGenerator(nn.Module):
        def __init__(self,GeoEncoder,AppearEncoder,PNet,StyleGAN2Generator):
            super(ComposedGenerator,self).__init__()
            self.GE = GeoEncoder
            self.AE = AppearEncoder
            self.PN = PNet
            self.SG = StyleGAN2Generator

            self.GE.eval()
            self.AE.eval()
            self.PN.eval()
            self.SG.eval()

        def forward(self,ref_image,mesh,eye_tensor,at_tensor):
            source_verts = mesh.verts_padded().float().to(device)
            source_faces = mesh.faces_padded().to(device)
            input_verts = source_verts.permute(0, 2, 1)
            source_geo_code, _ = self.GE(input_verts)

            target_appear_code = self.AE(ref_image)
            source_render_texture = TexturesVertex(verts_features=source_geo_code)
            source_train_meshes = Meshes(verts=source_verts,
                                  faces=source_faces,
                                  textures=source_render_texture)

            renderer = create_renderer(eye_tensor, at_tensor, image_size=128)
            source_geo_feature_images = renderer(source_train_meshes)

            source_geo_feature_images = source_geo_feature_images.permute(0, 3, 1, 2)
            source_encoded_geo_features = self.PN(source_geo_feature_images).contiguous()

            fake_images, _ = self.SG(init_features=source_encoded_geo_features, styles=[target_appear_code], input_is_latent=True)
            return fake_images

    gen = ComposedGenerator(GeoCodeExtractor,AppearCodeExtractor,Pnet,Generator)
    return gen

def render_fusion_image_tensor(ref_image_tensor,mesh,eye_tensor,at_tensor,generator):
    img = generator(ref_image_tensor,mesh,eye_tensor,at_tensor).permute(0, 2, 3, 1)
    return img

def load_meshes(mesh_paths,use_vertices_reduction=True):
    mesh_objs = []
    for path in mesh_paths:
        meshes = IO().load_mesh(path, device=device)

        # copy meshes
        max_vertices = 20000
        verts = meshes.verts_list()[0].cpu().numpy().astype(np.double)
        faces_idx = meshes.faces_list()[0].cpu().numpy().astype(np.uint32)
        if meshes.verts_list()[0].shape[0] > max_vertices and use_vertices_reduction:
            # verts = meshes.verts_list()[0].cpu().numpy().astype(np.double)
            # faces_idx = meshes.faces_list()[0].cpu().numpy().astype(np.uint32)
            mesh_simplifier = pyfqmr.Simplify()
            mesh_simplifier.setMesh(verts, faces_idx)
            mesh_simplifier.simplify_mesh(target_count=max_vertices, aggressiveness=4, preserve_border=True,
                                          verbose=10)
            new_verts, new_faces_idx, _ = mesh_simplifier.getMesh()
            # new_verts = new_verts / np.max(new_verts)

            tex = TexturesVertex(verts_features=torch.full((1, new_verts.shape[0], 3), 0.5).to(device))
            input_mesh = Meshes(verts=[torch.from_numpy(new_verts.astype(np.float32))], faces=[torch.from_numpy(new_faces_idx.astype(np.int32))], textures=tex).to(device)

        else:

            tex = TexturesVertex(verts_features=torch.full((1, verts.shape[0], 3), 0.5).to(device))
            input_mesh = Meshes(verts=[torch.from_numpy(verts.astype(np.float32))], faces=[torch.from_numpy(faces_idx.astype(np.int32))], textures=tex).to(device)

        mesh_objs.append(input_mesh)
    return mesh_objs


def load_image_tensors(ref_paths):
    ref_tensors = []
    for path in ref_paths:
        with Image.open(path).resize((256, 256)) as ref_image:
            ref_image_tensor = torch.from_numpy(1.0 / 255 * np.array([np.array(ref_image)], dtype='float32')).permute(0, 3,1, 2).to(device)
            ref_tensors.append(ref_image_tensor)
    return ref_tensors

def load_ref_images(ref_paths,image_size):
    ref_imgs = []
    for path in ref_paths:
        with Image.open(path).resize((image_size, image_size)) as ref_image:
            ref_image = np.array(ref_image)
            ref_imgs.append(ref_image)
    return ref_imgs

def create_cross_fusion_images(mesh_paths, ref_paths, load_model_path,load_model_step, save_path, image_size):
    mesh_objs = load_meshes(mesh_paths)
    ref_tensors = load_image_tensors(ref_paths)
    ref_images = load_ref_images(ref_paths,image_size)
    eye_tensor = torch.Tensor(
        [[0, 0, 1]]).to(device)
    at_tensor = torch.Tensor(
        [[0, 0, 0]]).to(device)

    fig = plt.figure(figsize=(40.,40.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(len(mesh_objs)+1, len(ref_tensors)+1),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    generator = build_generator(load_model_path,load_model_step,image_size)

    #create header
    all_images = []
    blank_image = render_blank_image(image_size,image_size)
    all_images.append(blank_image)

    for mesh in mesh_objs:
        gray_mesh_img_np = render_with_phong_renderer(mesh, eye_tensor, at_tensor, image_size=image_size)
        all_images.append(gray_mesh_img_np)

    #create lines
    for idx,ref_tensor in enumerate(ref_tensors):
        all_images.append(ref_images[idx])
        for mesh_id,mesh in enumerate(mesh_objs):
            fusion_img_tensor = render_fusion_image_tensor(ref_tensor,mesh,eye_tensor,at_tensor,generator)
            np_img = np.clip(fusion_img_tensor[0, ..., :3].cpu().detach().numpy(),0,1)

            all_images.append(np_img)

            im = Image.fromarray((255 * np_img).astype('uint8'))
            im.save(str(idx) + '_' + str(mesh_id) + '.jpg')
            # all_images.append(render_blank_image(image_size, image_size))

    for ax, im in zip(grid, all_images):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)

    plt.savefig(save_path)


if __name__ == '__main__':

    common_path = "FFHQCrossTest"
    mesh_paths = [
        os.path.join(common_path, "sample_0_marching_cubes_mesh_azim5_elev9.obj"),
        os.path.join(common_path, "sample_1_marching_cubes_mesh_azim-24_elev-5.obj"),
        os.path.join(common_path, "sample_2_marching_cubes_mesh_azim-6_elev-5.obj")
    ]
    ref_image_paths = [
        os.path.join(common_path, "0000000.png"),
        os.path.join(common_path, "0000001.png"),
        os.path.join(common_path, "0000002.png")
    ]
    # mesh_paths, ref_paths, load_model_path, save_path, image_size

    # create_cross_fusion_images(mesh_paths,ref_image_paths,load_model_path="data/tmp/train_2022_08_10_23_20_43_v10_4_Unpaired+FFHQ_Small_256_G_Pretrain_Notargetloss/models",
    #                            load_model_step=40, save_path="DemoImageTest6.jpg",image_size=256)
    create_cross_fusion_images(mesh_paths,ref_image_paths,load_model_path="data/tmp/train_2022_08_16_11_56_51_v10_4_Unpaired+FFHQ_Small_256_G_Pretrain_NoLNDLoss_0.1transfer/models",
                               load_model_step=20, save_path="DemoImageTest6_4.jpg",image_size=256)