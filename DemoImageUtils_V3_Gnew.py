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
from models.new_modules import LandmarkExtractor
#from models.Landmark_Encoder.Landmark_Encoder import Encoder_Landmarks_FULL as LandmarkExtractor

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

from torchvision import transforms
import lpips

def render_with_phong_renderer(input_mesh, eye_tensor, at_tensor, image_size=256):
    with torch.no_grad():
        renderer = create_phong_renderer(eye_tensor, at_tensor, image_size,device)
        gray_image = renderer(input_mesh)
        gray_image_np = (255 * gray_image[0, ..., :3].cpu().detach().numpy()).astype('uint8')
    return gray_image_np


def render_with_albedo_renderer(gt_mesh, eye_tensor, at_tensor, image_size=256):
    with torch.no_grad():
        renderer = create_renderer(eye_tensor, at_tensor, image_size,device)
        gt_image = renderer(gt_mesh)
        gt_image_np = (255 * gt_image[0, ..., :3].cpu().detach().numpy()).astype('uint8')
    return gt_image_np

def render_blank_image(image_width = 256,image_height = 256):
    blank_image = Image.new(mode="RGB",
                            size=(image_width, image_height))
    blank_image_np = np.array(blank_image)
    return blank_image_np

def build_appear_encoder(load_model_path,load_model_step,image_size = 256):
    AppearCodeExtractor = StyleGANEncoder(50, 'ir_se').cuda()
    if load_model_path != '' and os.path.exists(load_model_path):
        print(timelog_str() + ' Loading pretrained weight from: ' + load_model_path + ', step: ' + str(load_model_step))
        AppearCodeExtractor.load_state_dict(
            torch.load(os.path.join(load_model_path, 'AppearCodeExtractor_' + str(load_model_step) + '.pth')))
    else:
        print("No model found in :"+load_model_path)
    AppearCodeExtractor.eval()
    return AppearCodeExtractor

def cal_appear_loss(a_image,b_image,AppearCodeExtractor):
    ResizeImage = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    a_ref_image = ResizeImage(a_image)
    a_appear_code = AppearCodeExtractor(a_ref_image)

    b_ref_image = ResizeImage(b_image)
    b_appear_code = AppearCodeExtractor(b_ref_image)
    L1Loss = nn.L1Loss()
    loss = L1Loss(a_appear_code, b_appear_code)
    return loss.item()


def mark_landmarks_all(head_image,LndExtractor):
    # head_lnd_features = LndExtractor(head_image)
    # lnd_np = head_lnd_features.cpu().detach().numpy()
    # lnd_np = np.squeeze(lnd_np,axis = 0)

    # lst = []
    # for i in range(lnd_np.shape[0]):
    #     lst.append((lnd_np[i,0]/112 *256,lnd_np[i,1]/112 *256))
    # return lst

    head_lnd_features = LndExtractor.get_landmarks(head_image)
    lnd_np = head_lnd_features.cpu().detach().numpy()
    lnd_np = np.squeeze(lnd_np, axis=0)

    lst = []
    for i in range(lnd_np.shape[0]):
        lst.append((lnd_np[i,0]*4,lnd_np[i,1]*4))
    return lst

def build_LndExtractor():
    LndExtractor = LandmarkExtractor(pretrained=True).cuda()
    #LndExtractor = LandmarkExtractor(model_dir='models/pretrained_data/mobilefacenet_model_best.pth.tar').cuda()
    return LndExtractor

def cal_lnd_loss(a_image,b_image,LndExtractor):
    MSELoss = nn.MSELoss()
    a_lnd_features = LndExtractor(a_image)
    b_lnd_features = LndExtractor(b_image)
    lnd_loss = MSELoss(a_lnd_features, b_lnd_features)
    return lnd_loss.item()

VGGLoss = lpips.LPIPS(net='alex').to(device)
def cal_recon_loss(a_image,b_image):
    L1Loss = nn.L1Loss()
    recon_L1Loss = L1Loss(a_image, b_image)
    recon_VGGLoss = torch.mean(VGGLoss((a_image*2)-1, (b_image * 2) - 1))
    return recon_L1Loss.item(),recon_VGGLoss.item()

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

            ResizeImage = transforms.Compose(
                [transforms.Resize((256, 256)), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            target_appear_code = self.AE(ResizeImage(ref_image))
            source_render_texture = TexturesVertex(verts_features=source_geo_code)
            source_train_meshes = Meshes(verts=source_verts,
                                  faces=source_faces,
                                  textures=source_render_texture)

            renderer = create_renderer(eye_tensor, at_tensor, 128,device)
            source_geo_feature_images = renderer(source_train_meshes)

            source_geo_feature_images = source_geo_feature_images.permute(0, 3, 1, 2)
            source_encoded_geo_features = self.PN(source_geo_feature_images).contiguous()

            fake_images, _ = self.SG(init_features=source_encoded_geo_features, styles=[target_appear_code], input_is_latent=True)
            return (fake_images+1)/2

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

def create_cross_fusion_images(mesh_paths, ref_paths, load_model_path,load_model_step, save_path, image_size,verbose = True,make_landmarks = True):
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
    appear_encoder = build_appear_encoder(load_model_path,load_model_step,image_size)
    landmark_extractor = build_LndExtractor()

    #create header
    all_images = []
    blank_image = render_blank_image(image_size,image_size)
    blank_image = Image.fromarray(np.uint8(blank_image * 255)).convert('RGB')
    draw = ImageDraw.Draw(blank_image)
    font = ImageFont.truetype(r'consola.ttf', 24)
    stroke_color = (0, 0, 0)
    draw.text((140, 35), 'Mesh\nInput', font=font, align="left", stroke_width=1,
              stroke_fill=stroke_color)
    draw.text((15, 170), 'Reference\nImage Input', font=font, align="left", stroke_width=1,
              stroke_fill=stroke_color)
    w, h = 256, 256
    shape = [(20, 20), (w - 10, h - 10)]
    draw.line(shape, fill =(255,255,255), width = 0)

    blank_image = np.clip(np.array(blank_image) / 255.0, 0, 1)
    all_images.append(blank_image)

    for mesh in mesh_objs:
        gray_mesh_img_np = render_with_phong_renderer(mesh, eye_tensor, at_tensor, image_size=image_size)
        all_images.append(gray_mesh_img_np)

    #create lines
    for idx,ref_tensor in enumerate(ref_tensors):
        all_images.append(ref_images[idx])
        for mesh_id,mesh in enumerate(mesh_objs):
            fusion_img_tensor = render_fusion_image_tensor(ref_tensor,mesh,eye_tensor,at_tensor,generator)

            appear_loss = cal_appear_loss(fusion_img_tensor.permute(0, 3, 1, 2),ref_tensor,appear_encoder)
            landmark_loss = cal_lnd_loss(fusion_img_tensor.permute(0, 3, 1, 2),ref_tensors[mesh_id],landmark_extractor)


            np_img = np.clip(fusion_img_tensor[0, ..., :3].cpu().detach().numpy(),0,1)

            # NP img point drawing


            blank_image =  Image.fromarray(np.uint8(np_img*255)).convert('RGB')
            draw = ImageDraw.Draw(blank_image)

            if make_landmarks:

                lnd_this = mark_landmarks_all(fusion_img_tensor.permute(0, 3, 1, 2), landmark_extractor)
                for point in lnd_this:
                    draw.ellipse((point[0] - 1, point[1] - 1, point[0] + 1, point[1] + 1), fill=(255, 0, 0),
                                 outline=(255, 0, 0))

                # draw.point(lnd_this,fill=(255, 0, 0))

                lnd_this = mark_landmarks_all(ref_tensors[mesh_id], landmark_extractor)
                for point in lnd_this:
                    draw.ellipse((point[0] - 1, point[1] - 1, point[0] + 1, point[1] + 1), fill=(0, 255, 0),
                                 outline=(0, 255, 0))
                # draw.point(lnd_this,fill=(0, 255, 0))

            if verbose:
                font = ImageFont.truetype(r'consola.ttf', 20)
                stroke_color = (0, 0, 0)
                draw.text((5, 235), 'Appear:'+str(appear_loss)[0:6], font=font, align="left",stroke_width=1, stroke_fill=stroke_color)
                draw.text((5, 215), 'LandMark:'+str(landmark_loss)[0:6], font=font, align="left",stroke_width=1, stroke_fill=stroke_color)

                if(idx == mesh_id):
                    L1,VGG = cal_recon_loss(fusion_img_tensor.permute(0, 3, 1, 2),ref_tensor)
                    font = ImageFont.truetype(r'consola.ttf', 18)
                    draw.text((5, 5), 'Recon:\nL1:' + str(L1)[0:6]+ '\nLPIPS:'+str(VGG)[0:6], font=font, align="left", stroke_width=1,
                              stroke_fill=stroke_color)

            np_img = np.clip(np.array(blank_image)/255.0, 0, 1)

            all_images.append(np_img)

            #im = Image.fromarray((255 * np_img).astype('uint8'))
            #im.save(str(idx) + '_' + str(mesh_id) + '.jpg')
            # all_images.append(render_blank_image(image_size, image_size))

    for ax, im in zip(grid, all_images):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)

    plt.savefig(save_path)

def create_multiview_images(mesh_paths, ref_paths, load_model_path,load_model_step, save_path, image_size,verbose = True,make_landmarks = True):
    from models.ID_Encoder import ID_Encoder

    import models.id_loss as id_loss
    E_ID_LOSS_PATH = './models/pretrained_data/model_ir_se50.pth'
    id_encoder = id_loss.IDLoss(E_ID_LOSS_PATH)
    id_encoder = id_encoder.to(device)
    id_encoder = id_encoder.eval()
    def cal_arcface_loss(a_image,b_image):
        loss = nn.MSELoss()
        with torch.no_grad():
            a_id_vec = id_encoder.extract_feats((a_image * 2) - 1)
            b_id_vec = id_encoder.extract_feats((b_image * 2) - 1)
            return loss(a_id_vec,b_id_vec).item()


    mesh_objs = load_meshes(mesh_paths)
    mesh = mesh_objs[0]
    mesh_id = 0
    #assert len(mesh_objs) == 1
    ref_tensors = load_image_tensors(ref_paths)
    ref_images = load_ref_images(ref_paths,image_size)
    eye_tensors = [
        torch.Tensor([[0.5, 0, 0.86603]]).to(device),
        torch.Tensor([[0.2588, 0, 0.966]]).to(device),
        torch.Tensor([[0, 0, 1]]).to(device),
        torch.Tensor([[-0.2588, 0, 0.966]]).to(device),
        torch.Tensor([[-0.5, 0, 0.86603]]).to(device),
    ]
    # eye_tensor = torch.Tensor(
    #     [[0, 0, 1]]).to(device)
    at_tensor = torch.Tensor(
        [[0, 0, 0]]).to(device)

    fig = plt.figure(figsize=(40.,40.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(len(ref_tensors)+1, len(eye_tensors)+1),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    generator = build_generator(load_model_path,load_model_step,image_size)
    appear_encoder = build_appear_encoder(load_model_path,load_model_step,image_size)
    landmark_extractor = build_LndExtractor()

    #create header
    all_images = []
    blank_image = render_blank_image(image_size,image_size)
    blank_image = Image.fromarray(np.uint8(blank_image * 255)).convert('RGB')
    draw = ImageDraw.Draw(blank_image)
    font = ImageFont.truetype(r'consola.ttf', 24)
    stroke_color = (0, 0, 0)
    draw.text((140, 35), 'Mesh\nInput', font=font, align="left", stroke_width=1,
              stroke_fill=stroke_color)
    draw.text((15, 170), 'Reference\nImage Input', font=font, align="left", stroke_width=1,
              stroke_fill=stroke_color)
    w, h = 256, 256
    shape = [(20, 20), (w - 10, h - 10)]
    draw.line(shape, fill =(255,255,255), width = 0)

    blank_image = np.clip(np.array(blank_image) / 255.0, 0, 1)
    all_images.append(blank_image)

    for eye_tensor in eye_tensors:
        gray_mesh_img_np = render_with_phong_renderer(mesh, eye_tensor, at_tensor, image_size=image_size)
        all_images.append(gray_mesh_img_np)

    #create lines
    for idx,ref_tensor in enumerate(ref_tensors):
        all_images.append(ref_images[idx])
        # for mesh_id,mesh in enumerate(mesh_objs):
        base_image_tensor = None
        for eye_tensor in eye_tensors:
            fusion_img_tensor = render_fusion_image_tensor(ref_tensor,mesh,eye_tensor,at_tensor,generator)

            if base_image_tensor == None:
                base_image_tensor = fusion_img_tensor

            arcface_loss = cal_arcface_loss(fusion_img_tensor.permute(0, 3, 1, 2),base_image_tensor.permute(0, 3, 1, 2))
            appear_loss = cal_appear_loss(fusion_img_tensor.permute(0, 3, 1, 2),ref_tensor,appear_encoder)
            landmark_loss = cal_lnd_loss(fusion_img_tensor.permute(0, 3, 1, 2),ref_tensors[mesh_id],landmark_extractor)


            np_img = np.clip(fusion_img_tensor[0, ..., :3].cpu().detach().numpy(),0,1)

            # NP img point drawing


            blank_image =  Image.fromarray(np.uint8(np_img*255)).convert('RGB')
            draw = ImageDraw.Draw(blank_image)

            if make_landmarks:
                lnd_this = mark_landmarks_all(fusion_img_tensor.permute(0, 3, 1, 2), landmark_extractor)
                for point in lnd_this:
                    draw.ellipse((point[0] - 1, point[1] - 1, point[0] + 1, point[1] + 1), fill=(255, 0, 0),
                                 outline=(255, 0, 0))

                # draw.point(lnd_this,fill=(255, 0, 0))

                lnd_this = mark_landmarks_all(ref_tensors[mesh_id], landmark_extractor)
                for point in lnd_this:
                    draw.ellipse((point[0] - 1, point[1] - 1, point[0] + 1, point[1] + 1), fill=(0, 255, 0),
                                 outline=(0, 255, 0))
                # draw.point(lnd_this,fill=(0, 255, 0))

            if verbose:
                font = ImageFont.truetype(r'consola.ttf', 32)
                stroke_color = (0, 0, 0)

                draw.text((5, 215), '' + str(arcface_loss)[0:6], font=font, align="left", stroke_width=1,
                          stroke_fill=stroke_color)
                # draw.text((5, 235), 'Appear:'+str(appear_loss)[0:6], font=font, align="left",stroke_width=1, stroke_fill=stroke_color)
                # draw.text((5, 235), 'LandMark:'+str(landmark_loss)[0:6], font=font, align="left",stroke_width=1, stroke_fill=stroke_color)

                # if(idx == mesh_id):
                #     L1,VGG = cal_recon_loss(fusion_img_tensor.permute(0, 3, 1, 2),ref_tensor)
                #     font = ImageFont.truetype(r'consola.ttf', 18)
                #     draw.text((5, 5), 'Recon:\nL1:' + str(L1)[0:6]+ '\nVGG:'+str(VGG)[0:6], font=font, align="left", stroke_width=1,
                #               stroke_fill=stroke_color)

            np_img = np.clip(np.array(blank_image)/255.0, 0, 1)

            all_images.append(np_img)

            #im = Image.fromarray((255 * np_img).astype('uint8'))
            #im.save(str(idx) + '_' + str(mesh_id) + '.jpg')
            # all_images.append(render_blank_image(image_size, image_size))

    for ax, im in zip(grid, all_images):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)

    plt.savefig(save_path)

def create_interpolated_images(mesh_paths, ref_paths, load_model_path,load_model_step, save_path, image_size,verbose = True,make_landmarks = True):
    mesh_objs = load_meshes(mesh_paths)
    ref_tensors = load_image_tensors(ref_paths)
    ref_images = load_ref_images(ref_paths,image_size)
    eye_tensor = torch.Tensor(
        [[0, 0, 1]]).to(device)
    at_tensor = torch.Tensor(
        [[0, 0, 0]]).to(device)

    fig = plt.figure(figsize=(32.,4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(1, 8),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    generator = build_generator(load_model_path,load_model_step,image_size)
    appear_encoder = build_appear_encoder(load_model_path,load_model_step,image_size)
    landmark_extractor = build_LndExtractor()

    #create header
    all_images = []
    # blank_image = render_blank_image(image_size,image_size)
    # blank_image = Image.fromarray(np.uint8(blank_image * 255)).convert('RGB')
    # draw = ImageDraw.Draw(blank_image)
    # font = ImageFont.truetype(r'consola.ttf', 24)
    # stroke_color = (0, 0, 0)
    # draw.text((140, 35), 'Mesh\nInput', font=font, align="left", stroke_width=1,
    #           stroke_fill=stroke_color)
    # draw.text((15, 170), 'Reference\nImage Input', font=font, align="left", stroke_width=1,
    #           stroke_fill=stroke_color)
    # w, h = 256, 256
    # shape = [(20, 20), (w - 10, h - 10)]
    # draw.line(shape, fill =(255,255,255), width = 0)
    #
    # blank_image = np.clip(np.array(blank_image) / 255.0, 0, 1)
    # all_images.append(blank_image)
    #
    # for mesh in mesh_objs:
    #     gray_mesh_img_np = render_with_phong_renderer(mesh, eye_tensor, at_tensor, image_size=image_size)
    #     all_images.append(gray_mesh_img_np)

    #create lines
    # for t in range(3):
    #     all_images.append(ref_images[1+t])
    #     for i in range(6):
    #         fusion_img_tensor = render_fusion_image_tensor(ref_tensors[0+t]*i*1.0/6.0 +ref_tensors[1+t]*(6-i)*1.0/6.0, mesh_objs[3], eye_tensor, at_tensor, generator)
    #         np_img = np.clip(fusion_img_tensor[0, ..., :3].cpu().detach().numpy(), 0, 1)
    #         all_images.append(np_img)
    #
    #     all_images.append(ref_images[0+t])

    all_images.append(ref_images[1])
    for i in range(6):
        fusion_img_tensor = render_fusion_image_tensor(ref_tensors[0]*i*1.0/6.0 +ref_tensors[1]*(6-i)*1.0/6.0, mesh_objs[1], eye_tensor, at_tensor, generator)
        np_img = np.clip(fusion_img_tensor[0, ..., :3].cpu().detach().numpy(), 0, 1)
        all_images.append(np_img)

    all_images.append(ref_images[0])

    # for idx,ref_tensor in enumerate(ref_tensors):
    #     all_images.append(ref_images[idx])
    #     for mesh_id,mesh in enumerate(mesh_objs):
    #         fusion_img_tensor = render_fusion_image_tensor(ref_tensor,mesh,eye_tensor,at_tensor,generator)
    #
    #         appear_loss = cal_appear_loss(fusion_img_tensor.permute(0, 3, 1, 2),ref_tensor,appear_encoder)
    #         landmark_loss = cal_lnd_loss(fusion_img_tensor.permute(0, 3, 1, 2),ref_tensors[mesh_id],landmark_extractor)
    #
    #
    #         np_img = np.clip(fusion_img_tensor[0, ..., :3].cpu().detach().numpy(),0,1)
    #
    #         # NP img point drawing
    #
    #
    #         blank_image =  Image.fromarray(np.uint8(np_img*255)).convert('RGB')
    #         draw = ImageDraw.Draw(blank_image)
    #
    #         if make_landmarks:
    #
    #             lnd_this = mark_landmarks_all(fusion_img_tensor.permute(0, 3, 1, 2), landmark_extractor)
    #             for point in lnd_this:
    #                 draw.ellipse((point[0] - 1, point[1] - 1, point[0] + 1, point[1] + 1), fill=(255, 0, 0),
    #                              outline=(255, 0, 0))
    #
    #             # draw.point(lnd_this,fill=(255, 0, 0))
    #
    #             lnd_this = mark_landmarks_all(ref_tensors[mesh_id], landmark_extractor)
    #             for point in lnd_this:
    #                 draw.ellipse((point[0] - 1, point[1] - 1, point[0] + 1, point[1] + 1), fill=(0, 255, 0),
    #                              outline=(0, 255, 0))
    #             # draw.point(lnd_this,fill=(0, 255, 0))
    #
    #         if verbose:
    #             font = ImageFont.truetype(r'consola.ttf', 20)
    #             stroke_color = (0, 0, 0)
    #             draw.text((5, 235), 'Appear:'+str(appear_loss)[0:6], font=font, align="left",stroke_width=1, stroke_fill=stroke_color)
    #             draw.text((5, 215), 'LandMark:'+str(landmark_loss)[0:6], font=font, align="left",stroke_width=1, stroke_fill=stroke_color)
    #
    #             if(idx == mesh_id):
    #                 L1,VGG = cal_recon_loss(fusion_img_tensor.permute(0, 3, 1, 2),ref_tensor)
    #                 font = ImageFont.truetype(r'consola.ttf', 18)
    #                 draw.text((5, 5), 'Recon:\nL1:' + str(L1)[0:6]+ '\nLPIPS:'+str(VGG)[0:6], font=font, align="left", stroke_width=1,
    #                           stroke_fill=stroke_color)
    #
    #         np_img = np.clip(np.array(blank_image)/255.0, 0, 1)
    #
    #         all_images.append(np_img)
    #
    #         #im = Image.fromarray((255 * np_img).astype('uint8'))
    #         #im.save(str(idx) + '_' + str(mesh_id) + '.jpg')
    #         # all_images.append(render_blank_image(image_size, image_size))

    for ax, im in zip(grid, all_images):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)

    plt.savefig(save_path)

if __name__ == '__main__':

    common_path = "FFHQCrossTest"
    mesh_paths = [
        # os.path.join(common_path, "sample_0_marching_cubes_mesh_azim5_elev9.obj"),
        # os.path.join(common_path, "sample_1_marching_cubes_mesh_azim-24_elev-5.obj"),
        # os.path.join(common_path, "sample_2_marching_cubes_mesh_azim-6_elev-5.obj")
        # os.path.join(common_path, "david.obj"),
        os.path.join(common_path, "sample_0_marching_cubes_mesh_azim1_elev-8.obj"),
        os.path.join(common_path, "sample_1_marching_cubes_mesh_azim-4_elev12.obj"),
        os.path.join(common_path, "sample_2_marching_cubes_mesh_azim-4_elev6.obj"),
        os.path.join(common_path, "sample_2531_marching_cubes_mesh_azim-18_elev-3.obj"),
        os.path.join(common_path, "sample_3168_marching_cubes_mesh_azim-23_elev-1.obj")
    ]
    ref_image_paths = [
        os.path.join(common_path,"tmp", "0000000.png"),
        os.path.join(common_path, "0000001.png"),
        os.path.join(common_path, "0000002.png"),
        os.path.join(common_path, "0002531.png"),
        os.path.join(common_path, "0003168.png")
    ]

    # mesh_paths = [
    #     os.path.join(common_path, "old/sample_0_marching_cubes_mesh_azim5_elev9.obj"),
    #     os.path.join(common_path, "old/sample_1_marching_cubes_mesh_azim-24_elev-5.obj"),
    #     os.path.join(common_path, "old/sample_2_marching_cubes_mesh_azim-6_elev-5.obj")
    # ]
    # ref_image_paths = [
    #     os.path.join(common_path, "old/0000000.png"),
    #     os.path.join(common_path, "old/0000001.png"),
    #     os.path.join(common_path, "old/0000002.png"),
    # ]

    # hindbrain_difficiency = [852,]
    # extreme_angle_difficiency = [497,31]
    #
    # mesh_paths = []
    # ref_image_paths = []
    # dataset_path = './datasets/FFHQ_SDF_Small_1000'
    # if (os.path.exists(os.path.join(dataset_path, 'index.pkl'))):
    #     with open(os.path.join(dataset_path, 'index.pkl'), 'rb') as pkl_file:
    #         index_dict = pickle.load(pkl_file)
    #         print("Dataset length:", len(index_dict))
    #         #idxs = [918, 51, 851, 391, 195, 499, 403, 190, 869, 853]
    #         #idxs = [918, 851, 391, 195, 403, 190, 869, 853]
    #         #idxs = [18,110,15]
    #         #idxs = [653, 844, 451]
    #         idxs = [743, 134,451,653, 844,107,217]
    #         #idxs = [random.randint(0, len(index_dict)-1) for i in range(10)]
    #         print(idxs)
    #         for idx in idxs:
    #             ref_image_paths.append(os.path.join(dataset_path,index_dict[idx]['image_path']))
    #             mesh_paths.append(os.path.join(dataset_path,index_dict[idx]['marching_cubes_meshes_path']))

    # mesh_paths = ["./david.obj"]
    # mesh_paths, ref_paths, load_model_path, save_path, image_size

    # create_cross_fusion_images(mesh_paths,ref_image_paths,load_model_path="data/tmp/train_2022_08_10_23_20_43_v10_4_Unpaired+FFHQ_Small_256_G_Pretrain_Notargetloss/models",
    #                            load_model_step=40, save_path="DemoImageTest6.jpg",image_size=256)

    # create_cross_fusion_images(mesh_paths,ref_image_paths,load_model_path="data/tmp/train_2022_10_15_13_35_51_test_16_fixed_G_w5/models",
    #                            load_model_step=20, save_path="test_16_landmarks_20v2_noV.jpg",image_size=256)

    # create_cross_fusion_images(mesh_paths,ref_image_paths,load_model_path="data/tmp/train_2022_10_27_00_43_28_test_17_label_landmark_2/models",
    #                            load_model_step=20, save_path="test_17_landmarks_20v2_neo.jpg",image_size=256,verbose = False)

    #Last
    create_cross_fusion_images(mesh_paths, ref_image_paths,
                               load_model_path="data/tmp/train_2022_11_08_23_23_05_test_24_wzhwjaw_NOflip_CONTEX/models",
                               load_model_step=4, save_path="crossgeneration.jpg", image_size=256,
                               verbose=False,make_landmarks=False)

    create_interpolated_images(mesh_paths, ref_image_paths,
                               load_model_path="data/tmp/train_2022_11_08_23_23_05_test_24_wzhwjaw_NOflip_CONTEX/models",
                               load_model_step=4, save_path="crossfusion.jpg", image_size=256,
                               verbose=False,make_landmarks=False)

    #
    # create_cross_fusion_images(mesh_paths, ref_image_paths,
    #                            load_model_path="data/tmp/train_2022_11_08_15_52_25_test_23_wzhwjaw_latentlock_flip/models",
    #                            load_model_step=2, save_path="train_24_10x10_wzhwjaw_latentlock_flip.jpg", image_size=256,
    #                            verbose=False,make_landmarks=False)
    #
    # create_cross_fusion_images(mesh_paths, ref_image_paths,
    #                            load_model_path="data/tmp/train_2022_11_07_21_56_40_test_22_wzhwjaw_latent_lock_more_recon/models",
    #                            load_model_step=6, save_path="train_24_10x10_wzhwjaw_latent_lock_more_recon.jpg", image_size=256,
    #                            verbose=False,make_landmarks=False)
    #
    # create_cross_fusion_images(mesh_paths, ref_image_paths,
    #                            load_model_path="data/tmp/train_2022_10_27_13_30_48_test_17_label_landmark_3/models",
    #                            load_model_step=16, save_path="train_24_10x10_7_label_landmark.jpg", image_size=256,
    #                            verbose=False,make_landmarks=False)
    #
    # create_cross_fusion_images(mesh_paths, ref_image_paths,
    #                            load_model_path="data/tmp/train_2022_11_06_21_48_09_test_21_wzlmjaw_latent_lock/models",
    #                            load_model_step=16, save_path="train_24_10x10_7_wzlmjaw_latent_lock.jpg", image_size=256,
    #                            verbose=False,make_landmarks=False)

    # create_multiview_images(mesh_paths, ref_image_paths,
    #                            load_model_path="data/tmp/train_2022_11_08_23_23_05_test_24_wzhwjaw_NOflip_CONTEX/models",
    #                            load_model_step=4, save_path="train_24_10x10_mw3.jpg", image_size=256,
    #                            verbose=True,make_landmarks=False)

