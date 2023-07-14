import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import autograd
import lpips
import numpy as np
import wandb
from PIL import Image
from renderers import *
from config import BaseOptions
from Utils import *
#from Metrics import cal_similarities
from Dataloader import create_dataloader_splitted_v3,get_dataloader_ffhqv3
# from ModuleLoader import build_models
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.renderer import (
    TexturesVertex,
)
import models.pointnet2_sem_seg_msg as PointNet
from models.encoders.psp_encoders import GradualStyleEncoder as StyleGANEncoder
from models.modules import PNet,PNet_NoRes,ANet_NoRes,Dummy_Discriminator
from models.stylegan2.model import StyleGAN2Generator,Discriminator
from models.stylegan2.op import conv2d_gradfix
#from models.new_modules import LandmarkExtractor #,GeometryEncoder_Projection
from models.Landmark_Encoder.Landmark_Encoder import Encoder_Landmarks as LandmarkExtractor


# scp -P 22701 -r  chenruizhao@202.120.38.4:/home/chenruizhao/RemoteWorks/GeoFace/dataset_1000id_V2 /home/wangyue/crz_work/GeoFace

from gpu_mem_track import MemTracker
from sklearn.decomposition import PCA
from torchvision import transforms
from pytorch_msssim import MS_SSIM
#import contextual_loss as cl
def split_latent(latent: torch.Tensor, coarse_latent_size: int):
    latent_coarse = latent[:, :coarse_latent_size]
    #latent_coarse.requires_grad = False
    latent_fine = latent[:, coarse_latent_size:]
    #latent_fine.requires_grad = True
    return latent_coarse, latent_fine

def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def train(opt):
    print(timelog_str() + " Starting task.")
    if opt.training.wandb:
        print("Using wandb.")
        wandb.login()

    if opt.training.wandb:
        wandb.init(
            project="GeoFace_V2",
            entity="geofacedev",
            group="Main training",
            name=opt.experiment.expname,
            config=opt,
            tags=["debug"]
        )

    random.seed(10)

    loss_record = []
    loss_epoch = []
    Dloss_record = []

    val_loss_record = []
    val_loss_epoch = []

    debug_gpu_mem_log = opt.experiment.debug_gpu_mem_log
    if debug_gpu_mem_log:
        gpu_tracker = MemTracker()

    mean_path_length = 0
    mean_path_length_avg = 0

    L1Loss = nn.L1Loss()
    MSELoss = nn.MSELoss()
    VGGLoss = lpips.LPIPS(net='alex').to(device)

    use_contextual_loss = False
    if use_contextual_loss:
        ContextualLoss = cl.ContextualLoss(use_vgg=True, vgg_layer='relu4_4').to(device)

    RECON_L1_WEIGHT = opt.training.RECON_L1_WEIGHT
    RECON_VGG_WEIGHT = opt.training.RECON_VGG_WEIGHT
    RECON_GAN_WEIGHT = opt.training.RECON_GAN_WEIGHT
    RECON_LND_WEIGHT = opt.training.RECON_LND_WEIGHT
    RECON_APP_WEIGHT = opt.training.RECON_APP_WEIGHT

    TRANSFER_GAN_WEIGHT = opt.training.TRANSFER_GAN_WEIGHT
    TRANSFER_LND_WEIGHT = opt.training.TRANSFER_LND_WEIGHT
    TRANSFER_APP_WEIGHT = opt.training.TRANSFER_APP_WEIGHT

    TRANSFER_CONTEXTUAL_WEIGHT = 0.1
    RECON_CONTEXTUAL_WEIGHT = 0.1

    use_landmark_loss = True
    use_appear_loss = True

    # temp folder prep
    tmp_path, tmp_path_images, tmp_path_models = create_temp_dir(opt.experiment.expname)
    timelog(' Created temp directorys, models will be saved to: ' + tmp_path_models)

    # Creating Dataloader
    timelog('Creating Dataloaders.')
    # train_dataloader = get_dataloader_ffhqv3(opt.dataset.dataset_path,
    #                                          opt.model.image_size,
    #                                        opt.training.batch_size,
    #                                        opt.training.dataloader_number_workers)

    train_dataloader, val_dataloader =create_dataloader_splitted_v3(opt.dataset.dataset_path,
                                             opt.model.image_size,
                                           opt.training.batch_size,
                                           opt.training.dataloader_number_workers,opt.training.val_length)
    # val_dataloader = get_dataloader_ffhqv3(opt.dataset.dataset_path,
    #                                          opt.model.image_size,
    #                                        1,
    #                                        opt.training.dataloader_number_workers,shuffle = False)
    # train_dataloader, val_dataloader, test_dataloader = create_dataloader_splitted(args['dataset_path'],
    #                                                                                args['render_image_size'],
    #                                                                                args['batch_size'],
    #                                                                                args['dataloader_numworkers'])
    timelog('Created Dataloaders.')

    # build network models
    timelog('Building network models...')

    if debug_gpu_mem_log:
        gpu_tracker.track()

    GeoCodeExtractor = PointNet.get_model(num_classes=opt.model.geo_feature_channels).cuda()

    if debug_gpu_mem_log:
        gpu_tracker.track()
    #AppearCodeExtractor =ANet_NoRes(in_channel =3, style_dim =opt.model.style_dim).cuda() #StyleGANEncoder(50, 'ir').cuda()
    AppearCodeExtractor =StyleGANEncoder(50, 'ir_se').cuda()

    if debug_gpu_mem_log:
        gpu_tracker.track()

    # Pnet = PNet(geo_feature_channels,512).cuda()
    Pnet = PNet_NoRes(opt.model.geo_feature_channels, 512).cuda()
    if debug_gpu_mem_log:
        gpu_tracker.track()
    Generator = StyleGAN2Generator(
        opt.model.image_size, opt.model.style_dim, channel_multiplier=opt.model.channel_multiplier
    ).to(device)

    if opt.training.pretrained_stylegan2:
        checkpoint = torch.load("./models/pretrained_data/stylegan_ffhq_256_550000.pt")
        Generator.load_state_dict(checkpoint["g"], strict=False)

    latent_avg = None
    pretrained_pspencoder = False
    if pretrained_pspencoder:
        def get_keys(d, name):
            if 'state_dict' in d:
                d = d['state_dict']
            d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
            return d_filt

        pSp_checkpoint_path = "./models/pretrained_data/best_model.pt"
        ckpt = torch.load(pSp_checkpoint_path)

        AppearCodeExtractor.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
        Generator.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
        latent_avg = ckpt['latent_avg'].to(device)
    else:
        latent_avg = torch.zeros([1,14,512]).to(device)


    # checkpoint = torch.load("stylegan2-ffhq-config-f.pt")
    # checkpoint = torch.load("ffhq-256-config-e-003810.pt")

    if debug_gpu_mem_log:
        gpu_tracker.track()

    #Discrim = Dummy_Discriminator().to(device)
    Discrim = Discriminator(opt.model.image_size, channel_multiplier=opt.model.channel_multiplier).to(device)

    if use_landmark_loss :
        #LndExtractor = LandmarkExtractor(pretrained = True).cuda()
        LndExtractor = LandmarkExtractor(model_dir='models/pretrained_data/mobilefacenet_model_best.pth.tar').cuda()
        LndExtractor.eval()
    if debug_gpu_mem_log:
        gpu_tracker.track()

    # pretrained weight load.
    if opt.experiment.continue_training:
        load_model_path = os.path.join(DATA_DIR, 'tmp', opt.training.checkpoints_dir, 'models')
        load_model_step = opt.training.checkpoints_epoch
        if os.path.exists(load_model_path):
            print(timelog_str() + ' Loading pretrained weight from: ' + load_model_path + ', step: ' + str(load_model_step))

            if not os.path.exists(os.path.join(load_model_path, 'AppearCodeExtractor_' + str(load_model_step) + '.pth')):
                print("File doesn't exist: "+os.path.join(load_model_path, 'AppearCodeExtractor_' + str(load_model_step) + '.pth'))
                return
            else:
                AppearCodeExtractor.load_state_dict(
                    torch.load(os.path.join(load_model_path, 'AppearCodeExtractor_' + str(load_model_step) + '.pth')))
                GeoCodeExtractor.load_state_dict(
                    torch.load(os.path.join(load_model_path, 'GeoCodeExtractor_' + str(load_model_step) + '.pth')))
                Pnet.load_state_dict(
                    torch.load(os.path.join(load_model_path, 'Pnet_' + str(load_model_step) + '.pth')))
                Generator.load_state_dict(
                    torch.load(os.path.join(load_model_path, 'Generator_' + str(load_model_step) + '.pth')))
                Discrim.load_state_dict(
                    torch.load(os.path.join(load_model_path, 'Discrim_' + str(load_model_step) + '.pth')))


    timelog('Network models built.')

    G_params = \
        list(GeoCodeExtractor.parameters()) + \
        list(Pnet.parameters()) + \
        list(AppearCodeExtractor.parameters()) \
        #list(Generator.parameters())

    g_reg_ratio = opt.training.g_reg_every / (opt.training.g_reg_every + 1)
    optimizerG = torch.optim.Adam(G_params,
        lr=opt.training.g_lr *g_reg_ratio, betas=(0.9**g_reg_ratio, 0.99**g_reg_ratio), eps=1e-8)

    # optimizerG = torch.optim.Adam(G_params,
    #     lr=opt.training.g_lr, betas=(0, 0.99), eps=1e-8)

    # TODO : Try RMSProp
    # optimizer = torch.optim.Adam([
    #     {'params': GeoCodeExtractor.parameters()},
    #     {'params': Pnet.parameters()},
    #     {'params': AppearCodeExtractor.parameters()},
    #     {'params': Generator.parameters()}
    # ],
    #     lr=opt.training.lr, betas=(0.9, 0.999), eps=1e-8)

    d_reg_ratio = opt.training.d_reg_every / (opt.training.d_reg_every + 1)
    optimizerD = torch.optim.Adam(Discrim.parameters(), lr=opt.training.d_lr *  d_reg_ratio , betas=(0.9 **  d_reg_ratio , 0.99 **  d_reg_ratio ), eps=1e-8)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,100], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,1.05)

    cur_batches = 0
    total_time_consume = 0

    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    #training loop
    ResizeImage = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        #transforms.RandomHorizontalFlip(p=0.5),
        #transforms.RandomVerticalFlip(p=0.5)
    ])

    # source_geo_features = torch.randn(1, 512, 4, 4).cuda()
    # checkpoint = torch.load("./models/pretrained_data/stylegan_ffhq_256_550000.pt")
    # for name, param in checkpoint['g'].items():
    #     if (name == 'input.input'):
    #         source_geo_features = param.cuda()
    #         print(source_geo_features.shape)
    #         print(source_geo_features)
    #         print("Loaded Constant from checkpoint")
    # source_geo_features.requires_grad = False

    use_latent_avg = True
    # if use_latent_avg :
    #     latent_avg = Generator.mean_latent(int(1e5))[0].detach()


    for epoch in range(1, opt.training.epoch + 1):
        AppearCodeExtractor.train()
        GeoCodeExtractor.train()
        Pnet.train()
        Generator.train()
        epoch_start_time = time.time()

        #training
        for (idx, batch) in enumerate(train_dataloader):
            cur_batches = cur_batches + 1

            loss_dict = {}
            '''
            batch:
            meshes
            ref_image:torch.Size([1, 3, 256, 256])
            gt_image:torch.Size([1, 3, 512, 512])
            eye_tensor:torch.Size([1, 3])
            at_tensor:torch.Size([1, 3])
            '''
            if debug_gpu_mem_log:
                gpu_tracker.track()
            source_verts = batch['source_meshes'].verts_padded().float().to(device)
            source_faces = batch['source_meshes'].faces_padded().to(device)
            # source_verts = batch['source_verts'].to(device)
            # source_faces = batch['source_faces'].to(device)
            source_images = batch['source_image'].to(device)

            # target_verts = batch['target_meshes'].verts_padded().float().to(device)
            # target_faces = batch['target_meshes'].faces_padded().to(device)
            target_images = batch['target_image'].to(device)

            # eye_tensor = batch['eye_tensor'].to(device)
            # at_tensor = batch['at_tensor'].to(device)
            eye_tensor = torch.Tensor(
                [[0, 0, 1]]).to(device)
            at_tensor = torch.Tensor(
                [[0, 0, 0]]).to(device)

            # create cameras for this batch
            renderer = create_renderer(eye_tensor, at_tensor, image_size=128 , device= device)
            phong_renderer = create_phong_renderer(eye_tensor, at_tensor, image_size=128,device= device)

            if debug_gpu_mem_log:
                gpu_tracker.track()

            # Discriminator training------------------------------------------------------------------------------------
            optimizerD.zero_grad()

            # AppearCodeExtractor.eval()
            # GeoCodeExtractor.eval()
            # Pnet.eval()
            # Generator.eval()
            # Discrim.train()

            requires_grad(AppearCodeExtractor, False)
            requires_grad(GeoCodeExtractor, False)
            requires_grad(Pnet, False)
            requires_grad(Generator, False)
            requires_grad(Discrim, True)

            #Get faked image
            input_verts = source_verts.permute(0, 2, 1)
            source_geo_code, _ = GeoCodeExtractor(input_verts)
            source_render_texture = TexturesVertex(verts_features=source_geo_code)
            source_train_meshes = Meshes(verts=source_verts,
                                  faces=source_faces,
                                  textures=source_render_texture)

            source_geo_features = renderer(source_train_meshes)
            source_geo_features = source_geo_features.permute(0, 3, 1, 2)
            source_geo_features = Pnet(source_geo_features).contiguous()

            target_ref_images = ResizeImage(target_images)
            target_appear_code = AppearCodeExtractor(target_ref_images)
            #target_appear_code = target_appear_code + latent_avg.repeat(target_appear_code.shape[0], 1, 1)
            # target_appear_code = AppearCodeExtractor(target_images)
            d_fake_images, _ = Generator(init_features=source_geo_features, styles=[target_appear_code], input_is_latent=True)

            if debug_gpu_mem_log:
                gpu_tracker.track()

            #reconstruct original image

            source_ref_images = ResizeImage(source_images)
            source_appear_code = AppearCodeExtractor(source_ref_images)
            #source_appear_code = source_appear_code + latent_avg.repeat(source_appear_code.shape[0], 1, 1)
            # source_appear_code = AppearCodeExtractor(source_images)
            d_recon_images, _ = Generator(init_features=source_geo_features, styles=[source_appear_code],
                                       input_is_latent=True)

            if debug_gpu_mem_log:
                gpu_tracker.track()

            #loss calc images range are from -1 1
            source_features = Discrim(source_images*2-1)
            recon_features = Discrim(d_recon_images)
            fake_features = Discrim(d_fake_images)
            target_features = Discrim(target_images*2-1)

            d_source_loss = F.softplus(-source_features)
            d_recon_loss = F.softplus(recon_features)
            d_target_loss = F.softplus(-target_features)
            d_fake_loss = F.softplus(fake_features)

            DLoss = d_fake_loss.mean() + d_target_loss.mean() + d_recon_loss.mean() + d_source_loss.mean()

            loss_dict['Discriminator/source_loss'] = d_source_loss.mean().item()
            loss_dict['Discriminator/recon_loss'] = d_recon_loss.mean().item()
            loss_dict['Discriminator/target_loss'] = d_target_loss.mean().item()
            loss_dict['Discriminator/fake_loss'] = d_fake_loss.mean().item()
            loss_dict['DLoss'] = DLoss.item()

            Discrim.zero_grad()
            DLoss.backward()

            if debug_gpu_mem_log:
                get_model_size(AppearCodeExtractor,"AppearCodeExtractor")
                get_model_size(GeoCodeExtractor, "GeoCodeExtractor")
                get_model_size(Pnet, "Pnet")
                get_model_size(Generator, "Generator")
                get_model_size(Discrim, "Discrim")

            optimizerD.step()

            d_regularize = cur_batches % opt.training.d_reg_every == 0

            if d_regularize:
                source_images.requires_grad = True
                real_img_aug = source_images

                real_pred = Discrim(real_img_aug)
                r1_loss = d_r1_loss(real_pred, source_images)

                Discrim.zero_grad()
                (opt.training.r1 / 2 * r1_loss * opt.training.d_reg_every + 0 * real_pred[0]).backward()

                optimizerD.step()


            if debug_gpu_mem_log:
                gpu_tracker.track()

            del d_fake_images
            del d_recon_images
            torch.cuda.empty_cache()

            if debug_gpu_mem_log:
                gpu_tracker.track()

            # Generator training----------------------------------------------------------------------------------------
            optimizerG.zero_grad()
            #optimizerD.zero_grad()
            # AppearCodeExtractor.train()
            # GeoCodeExtractor.train()
            # Pnet.train()
            # Generator.train()
            # Discrim.eval()

            requires_grad(AppearCodeExtractor, True)
            requires_grad(GeoCodeExtractor,True)
            requires_grad(Pnet, True)
            requires_grad(Generator, False)
            requires_grad(Discrim, False)

            train_log_image = {}
            if cur_batches % opt.training.image_log_batch_interval == 0:
                images_np = []
                for img_index in range(opt.training.image_log_batch_size):
                    im = Image.fromarray(
                        (255 * source_images.permute(0, 2, 3, 1)[img_index, ..., :3].cpu().detach().numpy()).astype(
                            'uint8'))
                    images_np.append(im)
                    im.save(
                        os.path.join(tmp_path_images, str(cur_batches) + '_source_images' + str(img_index) + '.jpg'))
                train_log_image['train/source_images'] = [wandb.Image(image, caption="Source Images") for image in
                                                          images_np]

                images_np = []
                for img_index in range(opt.training.image_log_batch_size):
                    im = Image.fromarray(
                        (255 * target_images.permute(0, 2, 3, 1)[img_index, ..., :3].cpu().detach().numpy()).astype(
                            'uint8'))
                    images_np.append(im)
                    im.save(
                        os.path.join(tmp_path_images, str(cur_batches) + '_target_images' + str(img_index) + '.jpg'))
                train_log_image['train/target_images'] = [wandb.Image(image, caption="Target Images") for image in
                                                          images_np]

            #Get faked image
            input_verts = source_verts.permute(0, 2, 1)
            source_geo_code, _ = GeoCodeExtractor(input_verts)
            source_render_texture = TexturesVertex(verts_features=source_geo_code)
            source_train_meshes = Meshes(verts=source_verts,
                                  faces=source_faces,
                                  textures=source_render_texture)
            source_geo_features = renderer(source_train_meshes)

            # with torch.no_grad():
            #     source_original_tex = TexturesVertex(verts_features=torch.full((source_geo_code.shape[0], source_geo_code.shape[1], 3), 0.5).to(device))
            #     source_original_meshes = Meshes(verts=source_verts.detach(),
            #                           faces=source_faces.detach(),
            #                           textures=source_original_tex)
            #     source_geo_original = phong_renderer(source_original_meshes)
            #
            #     if cur_batches % opt.training.image_log_batch_interval == 0:
            #         images_np = []
            #         for img_index in range(opt.training.image_log_batch_size):
            #             np_img = source_geo_original[img_index, :, :, :].cpu().detach().numpy()
            #             im = Image.fromarray(
            #                 (255 * np_img).astype('uint8'))
            #             images_np.append(im)
            #             im.save(
            #                 os.path.join(tmp_path_images, str(cur_batches) + '_original_mesh' + str(img_index) + '.png'))
            #
            #         train_log_image['train/original_mesh'] = [wandb.Image(image, caption="original_mesh")
            #                                                 for image in images_np]


            if cur_batches % opt.training.image_log_batch_interval == 0:
                images_np = []
                for img_index in range(opt.training.image_log_batch_size):
                    np_img = source_geo_features[img_index, :, :, :].cpu().detach().numpy()
                    original_shape = np_img.shape
                    np_img = np_img.reshape((-1, source_geo_code.shape[-1]))
                    pca = PCA(n_components=3, svd_solver='full')
                    np_img_pca = pca.fit_transform(np_img)
                    np_img_pca = np_img_pca.reshape((original_shape[0], original_shape[1], 3))
                    np_img_pca = (np_img_pca - np.min(np_img_pca)) / (
                            np.max(np_img_pca) - np.min(np_img_pca))  # scale to 0~1
                    im = Image.fromarray(
                        (255 * np_img_pca).astype('uint8'))
                    images_np.append(im)
                    im.save(os.path.join(tmp_path_images, str(cur_batches) + '_beforeRenderer' + str(img_index) + '.jpg'))


                train_log_image['train/before_renderer'] = [wandb.Image(image,caption = "Before renderer, PCA result") for image in images_np]

            source_geo_features = source_geo_features.permute(0, 3, 1, 2)
            source_geo_features = Pnet(source_geo_features).contiguous()

            target_ref_images = ResizeImage(target_images)
            target_appear_code = AppearCodeExtractor(target_ref_images)
            #target_appear_code = target_appear_code + latent_avg.repeat(target_appear_code.shape[0], 1, 1)
            # target_appear_code = AppearCodeExtractor(target_images)

            fake_images, _ = Generator(init_features=source_geo_features, styles=[target_appear_code], input_is_latent=True)

            normalized_fake_images = (fake_images +1)/2

            if cur_batches % opt.training.image_log_batch_interval == 0:
                images_np = []
                for img_index in range(opt.training.image_log_batch_size):
                    np_img = 255 * normalized_fake_images.permute(0, 2, 3, 1)[img_index, ..., :3].cpu().detach().numpy()
                    np_img = np.clip(np_img,0,255)
                    im = Image.fromarray(np_img.astype('uint8'))
                    images_np.append(im)
                    im.save(os.path.join(tmp_path_images, str(cur_batches) + '_afterRenderer' + str(img_index) + '.jpg'))
                train_log_image['train/after_renderer'] = [wandb.Image(image, caption="Source Geo + Target Appear") for image in images_np]

            if debug_gpu_mem_log:
                gpu_tracker.track()

            #reconstruct original image

            source_ref_images = ResizeImage(source_images)
            source_appear_code = AppearCodeExtractor(source_ref_images)
            #source_appear_code = source_appear_code + latent_avg.repeat(source_appear_code.shape[0], 1, 1)
            # source_appear_code = AppearCodeExtractor(source_images)
            recon_images, _ = Generator(init_features=source_geo_features, styles=[source_appear_code],input_is_latent=True)

            normalized_recon_images = (recon_images + 1) / 2

            if cur_batches % opt.training.image_log_batch_interval == 0:
                images_np = []
                for img_index in range(opt.training.image_log_batch_size):
                    np_img = 255 * normalized_recon_images.permute(0, 2, 3, 1)[img_index, ..., :3].cpu().detach().numpy()
                    np_img = np.clip(np_img,0,255)
                    im = Image.fromarray(np_img.astype('uint8'))
                    images_np.append(im)
                    im.save(os.path.join(tmp_path_images, str(cur_batches) + '_afterRenderer_recon' + str(img_index) + '.jpg'))
                train_log_image['train/recon_image'] = [wandb.Image(image, caption="Reconstruction Output") for image in images_np]

            if debug_gpu_mem_log:
                gpu_tracker.track()

            #appear code consistency
            # target_input_verts = target_verts.permute(0, 2, 1)
            # target_geo_code, _ = GeoCodeExtractor(target_input_verts)
            # target_render_texture = TexturesVertex(verts_features=target_geo_code)
            # target_train_meshes = Meshes(verts=target_verts,
            #                       faces=target_faces,
            #                       textures=target_render_texture)
            #
            # target_geo_feature_images = renderer(target_train_meshes)
            #
            # target_geo_feature_images = target_geo_feature_images.permute(0, 3, 1, 2)
            # target_encoded_geo_features = Pnet(target_geo_feature_images).contiguous()

            recon_ref_images = ResizeImage(normalized_recon_images)
            recon_appear_code = AppearCodeExtractor(recon_ref_images)
            #recon_appear_code = recon_appear_code + latent_avg.repeat(recon_appear_code.shape[0], 1, 1)

            # recon_appear_code = AppearCodeExtractor(normalized_recon_images)

            fake_ref_images = ResizeImage(normalized_fake_images)
            fake_appear_code = AppearCodeExtractor(fake_ref_images)
            #fake_appear_code = fake_appear_code + latent_avg.repeat(fake_appear_code.shape[0], 1, 1)
            # fake_appear_code = AppearCodeExtractor(normalized_fake_images)
            # consistency_images, _ = Generator(init_features=target_encoded_geo_features, styles=[fake_appear_code],
            #                            input_is_latent=True)
            if debug_gpu_mem_log:
                gpu_tracker.track()
            # source and reconstructed source
            recon_L1Loss = RECON_L1_WEIGHT * L1Loss(normalized_recon_images, source_images)

            recon_VGGLoss = RECON_VGG_WEIGHT * torch.mean(VGGLoss(recon_images, (source_images * 2) - 1))

            #recon_MSSSIMLoss = (1 - ms_ssim_module(normalized_recon_images, source_images))
            #recon_VGGLoss = RECON_VGG_WEIGHT *  (1 - ms_ssim_module(normalized_recon_images, source_images))

            if debug_gpu_mem_log:
                gpu_tracker.track()

            recon_features = Discrim(recon_images)
            recon_Gloss = RECON_GAN_WEIGHT * F.softplus(-recon_features).mean()

            #recon_appear_loss = RECON_APP_WEIGHT * L1Loss(recon_appear_code, source_appear_code)


            recon_loss = recon_L1Loss  +  recon_VGGLoss + recon_Gloss #+ recon_appear_loss

            #log losses
            loss_dict['recon_L1Loss'] = recon_L1Loss.item()
            loss_dict['recon_VGGLoss'] = recon_VGGLoss.item()
            loss_dict['recon_AdvLoss'] = recon_Gloss.item()
            #loss_dict['recon_AppearLoss'] = recon_appear_loss.item()

            if use_contextual_loss:
                recon_contextual_loss = RECON_CONTEXTUAL_WEIGHT * ContextualLoss(normalized_recon_images,source_images)
                recon_loss += recon_contextual_loss
                loss_dict['recon_ContextualLoss'] = recon_contextual_loss.item()

            # if use_landmark_loss and (epoch > opt.training.use_landmark_loss_after_epoch):
            #     recon_lnd_features = LndExtractor(normalized_recon_images)
            #     source_lnd_features = LndExtractor(source_images)
            #     recon_lnd_loss = RECON_LND_WEIGHT * MSELoss(recon_lnd_features, source_lnd_features )
            #     recon_loss += recon_lnd_loss
            #     loss_dict['recon_LndLoss'] = recon_lnd_loss.item()
            # else:
            #     recon_lnd_loss = 0
            #     loss_dict['recon_LndLoss'] = 0

            loss_dict['recon_Loss'] = recon_loss.item()
            #style transferred image and transfer target

            #transfer_L1Loss = TRANSFER_L1_WEIGHT * L1Loss(normalized_fake_images, source_images)
            #loss_dict['transfer_L1Loss'] = transfer_L1Loss.item()

            #transfer_VGGLoss = TRANSFER_VGG_WEIGHT * torch.mean(VGGLoss(fake_images, (source_images * 2) - 1))
            #transfer_VGGLoss = TRANSFER_VGG_WEIGHT *  (1 - ms_ssim_module(normalized_fake_images, source_images))
            #loss_dict['transfer_VGGLoss'] = transfer_VGGLoss.item()
            transfer_features = Discrim(fake_images)
            transfer_Gloss = TRANSFER_GAN_WEIGHT * F.softplus(-transfer_features).mean()

            coarse_size = 64
            l_size = int(math.log(coarse_size, 2)) *2 -2
            fake_latent_coarse, fake_latent_fine = split_latent(fake_appear_code, l_size)
            target_latent_coarse, target_latent_fine = split_latent(target_appear_code, l_size)

            # supervise fine latent part of W+
            transfer_appear_loss = TRANSFER_APP_WEIGHT * L1Loss(fake_latent_fine, target_latent_fine)
            #transfer_appear_loss = TRANSFER_APP_WEIGHT * L1Loss(fake_appear_code,target_appear_code)

            loss_dict['transfer_AdvLoss'] = transfer_Gloss.item()
            loss_dict['transfer_AppearLoss'] = transfer_appear_loss.item()

            transfer_loss = transfer_appear_loss + transfer_Gloss # +transfer_L1Loss + transfer_VGGLoss

            if use_contextual_loss:
                transfer_contextual_loss = TRANSFER_CONTEXTUAL_WEIGHT * ContextualLoss(normalized_fake_images,target_images)
                transfer_loss += transfer_contextual_loss
                loss_dict['transfer_ContextualLoss'] = transfer_contextual_loss.item()

            if use_landmark_loss and (epoch > opt.training.use_landmark_loss_after_epoch):
                transfer_lnd_features = LndExtractor(normalized_fake_images)
                source_lnd_features = LndExtractor(source_images)
                transfer_lnd_loss = TRANSFER_LND_WEIGHT * MSELoss(transfer_lnd_features, source_lnd_features)
                transfer_loss += transfer_lnd_loss
                loss_dict['transfer_LndLoss'] = transfer_lnd_loss.item()
            else:
                transfer_lnd_loss = 0
                loss_dict['transfer_LndLoss'] = 0

            loss_dict['transfer_Loss'] = transfer_loss.item()

            GLoss = recon_loss + transfer_loss
            loss_dict['GLoss'] = GLoss.item()

            # if cur_batches % opt.training.interval_between_transfer == 0:
            #     GLoss = transfer_loss
            # else:
            #     GLoss = recon_loss

            # ---------------Patch NCE Loss---------------
            # nce_geo_features = Pnet(feature_images, output_medium_features = True)
            # nce_output_features = Pnet(gt_images.repeat(1, int(geo_feature_channels / 3), 1, 1),
            #                            output_medium_features=True)
            #
            # feat_geo_pool, sample_ids = Fnet(nce_geo_features, 32, None)
            # feat_output_pool, _ =  Fnet(nce_output_features, 32, None)
            #
            # total_nce_loss = 0.0
            # for f_g, f_o, crit, nce_layer in zip(feat_geo_pool,feat_output_pool, criterionNCE,nce_layers ):
            #     loss = crit(f_g, f_o) * NCE_WEIGHT
            #     total_nce_loss += loss.mean()
            # total_nce_loss /= len(nce_layers)
            # print(total_nce_loss)

            # calculate total variation regularization (anisotropic version)
            # https://www.wikiwand.com/en/Total_variation_denoising
            # diff_i = torch.sum(torch.abs(pred_images[:, :, :, 1:] - pred_images[:, :, :, :-1]))
            # diff_j = torch.sum(torch.abs(pred_images[:, :, 1:, :] - pred_images[:, :, :-1, :]))
            # tv_loss = TV_WEIGHT * (diff_i + diff_j)

            AppearCodeExtractor.zero_grad()
            GeoCodeExtractor.zero_grad()
            Pnet.zero_grad()
            Generator.zero_grad()

            GLoss.backward()

            if debug_gpu_mem_log:
                get_model_size(AppearCodeExtractor,"AppearCodeExtractor")
                get_model_size(GeoCodeExtractor, "GeoCodeExtractor")
                get_model_size(Pnet, "Pnet")
                get_model_size(Generator, "Generator")
                get_model_size(Discrim, "Discrim")

            optimizerG.step()

            g_regularize = cur_batches % opt.training.g_reg_every == 0

            # if g_regularize:
            #     path_batch_size = max(1, opt.training.batch_size // opt.training.path_batch_shrink)
            #     #noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
            #     #fake_img, latents = generator(noise, return_latents=True)
            #
            #
            #     input_verts = source_verts.permute(0, 2, 1)
            #     source_geo_code, _ = GeoCodeExtractor(input_verts)
            #     source_render_texture = TexturesVertex(verts_features=source_geo_code)
            #     source_train_meshes = Meshes(verts=source_verts,
            #                                  faces=source_faces,
            #                                  textures=source_render_texture)
            #     source_geo_features = renderer(source_train_meshes)
            #     source_geo_features = source_geo_features.permute(0, 3, 1, 2)
            #     source_geo_features = Pnet(source_geo_features).contiguous()
            #
            #     target_appear_code = AppearCodeExtractor(target_images)
            #     fake_images, latent = Generator(init_features=source_geo_features, styles=[target_appear_code],
            #                                input_is_latent=True,return_latents=True)
            #     path_loss, mean_path_length, path_lengths = g_path_regularize(
            #         fake_images, latent, mean_path_length
            #     )
            #     # fake_images, _ = Generator(init_features=source_geo_features, styles=[target_appear_code],input_is_latent=True)
            #     Generator.zero_grad()
            #     weighted_path_loss = opt.training.path_regularize * opt.training.g_reg_every * path_loss
            #
            #     #if opt.training.path_batch_shrink:
            #     #    weighted_path_loss += 0 * fake_images[0, 0, 0, 0]
            #
            #     weighted_path_loss.backward()
            #
            #     optimizerG.step()

                # mean_path_length_avg = (
                #         reduce_sum(mean_path_length).item() / get_world_size()
                # )

            if debug_gpu_mem_log:
                gpu_tracker.track()

            del fake_images
            del recon_images
            torch.cuda.empty_cache()

            if debug_gpu_mem_log:
                gpu_tracker.track()
            # LOG
            # timelog('Epoch-{}/D Loss at batch {}: {}'.format(epoch, cur_batches, DLoss.item()))
            # LOG
            timelog('Epoch-{}/Batch-{} G Loss: {} / D Loss: {} '.format(epoch, cur_batches, GLoss.item(), DLoss.item()))
            # timelog('   --ReconstructLoss: L1:{}, VGGLoss: {}, GLoss: {}, Landmark Loss: {} '.format(recon_L1Loss.item(),recon_VGGLoss.item(),recon_Gloss.item(),recon_lnd_loss.item()))
            # timelog(
            #     '   --Transfer Loss: L1:{}, VGGLoss: {}, GLoss: {}, Landmark Loss: {}, appear Loss: {} '.format(transfer_L1Loss.item(),
            #                                                                                      transfer_VGGLoss.item(),
            #                                                                                      transfer_Gloss.item(),
            #                                                                                      transfer_lnd_loss.item(),
            #                                                                                    transfer_appear_loss.item()))
            timelog('\n   --G Losses:')
            for name, loss in loss_dict.items():
                print(name +": "+ str(loss))


            # if use_contextual_loss:
            #     timelog('   --Contextual Loss: Recon: {}, Transfer Loss: {} '.format(recon_contextual_loss.item(),
            #                                                                    transfer_contextual_loss.item()))
            # if use_landmark_loss and (epoch > opt.training.use_landmark_loss_after_epoch):
            #     timelog('   --ReconstructLoss: L1:{}, VGG Loss: {}, GANLoss: {}, Landmark Loss: {}, appear Loss: {} '.format(recon_L1Loss.item(),recon_VGGLoss.item(),recon_Gloss.item(),recon_lnd_loss.item(),recon_appear_loss.item()))
            #     timelog(
            #         '   --Transfer Loss: GANLoss: {}, Landmark Loss: {}, appear Loss: {} '.format(transfer_Gloss.item(),
            #                                                                                          transfer_lnd_loss.item(),
            #                                                                                        transfer_appear_loss.item()))
            # else:
            #     timelog('   --ReconstructLoss: L1:{}, VGG Loss: {}, GANLoss: {}, appear Loss: {} '.format(recon_L1Loss.item(),recon_VGGLoss.item(),recon_Gloss.item(),recon_appear_loss.item()))
            #     timelog('   --Transfer Loss: GANLoss: {}, appear Loss: {} '.format(transfer_Gloss.item(),transfer_appear_loss.item()))

            # save loss
            # if cur_batches % args['loss_record_batch_interval'] == 0:
            #log to wandb
            metrics = {"train/G_loss": GLoss,
                       "train/D_loss": DLoss,
                       "train/recon_L1Loss": recon_L1Loss,
                       "train/recon_VGGLoss": recon_VGGLoss,
                       "train/recon_GANloss": recon_Gloss,
                       "train/recon_appear_loss": recon_appear_loss,
                       "train/recon_lnd_loss": recon_lnd_loss,
                       "train/transfer_GANloss": transfer_Gloss,
                       "train/transfer_appear_loss": transfer_appear_loss,
                       "train/transfer_lnd_loss": transfer_lnd_loss,
                       # "train/transfer_L1Loss": transfer_L1Loss,
                       # "train/transfer_VGGLoss": transfer_VGGLoss,
                       # "train/recon_contextual_loss": recon_contextual_loss,
                       # "train/transfer_contextual_loss": transfer_contextual_loss,

                       "train/epoch": epoch,
                       "train/example_count": cur_batches * opt.training.batch_size,


                       #"train/learning_rate": scheduler.get_last_lr()[0],

                       "Discriminator/recon_loss":d_recon_loss.mean(),
                       "Discriminator/source_loss": d_source_loss.mean(),
                       "Discriminator/fake_loss": d_fake_loss.mean(),
                       "Discriminator/target_loss": d_target_loss.mean(),
                       }
            if opt.training.wandb:
                if cur_batches % opt.training.image_log_batch_interval == 0:
                    wandb.log({**train_log_image,**metrics})
                else:
                    wandb.log(metrics)
            #Dloss_record.append(DLoss.item())
            #loss_record.append(GLoss.item())
            #loss_epoch.append(cur_batches)
            # plt.clf()
            # plt.plot(loss_epoch, loss_record, 'r-', label="Training GLoss", linewidth=1)
            # plt.plot(loss_epoch, Dloss_record, 'g-', label="Training DLoss", linewidth=1)
            # # plt.plot(val_loss_epoch, val_loss_record, 'b-', label="Validation Loss", linewidth=1)
            # plt.legend()
            # plt.xlabel("batches")
            # plt.ylabel("loss")
            # plt.savefig(os.path.join(tmp_path, 'Loss.jpg'))

        epoch_end_time = time.time()
        epoch_seconds = epoch_end_time - epoch_start_time
        total_time_consume = total_time_consume + epoch_seconds
        print("Finished epoch with: " + str(epoch_seconds))
        # scheduler.step()

        if epoch % opt.training.model_save_epoch_interval == 0:
            with torch.no_grad():
                AppearCodeExtractor.eval()
                GeoCodeExtractor.eval()
                Pnet.eval()
                Generator.eval()
                Discrim.eval()

                torch.save(AppearCodeExtractor.state_dict(),
                           os.path.join(tmp_path_models, 'AppearCodeExtractor_' + str(epoch) + '.pth'))
                torch.save(GeoCodeExtractor.state_dict(),
                           os.path.join(tmp_path_models, 'GeoCodeExtractor_' + str(epoch) + '.pth'))
                torch.save(Pnet.state_dict(), os.path.join(tmp_path_models, 'Pnet_' + str(epoch) + '.pth'))
                torch.save(Generator.state_dict(),
                           os.path.join(tmp_path_models, 'Generator_' + str(epoch) + '.pth'))
                torch.save(Discrim.state_dict(),
                           os.path.join(tmp_path_models, 'Discrim_' + str(epoch) + '.pth'))

                AppearCodeExtractor.train()
                GeoCodeExtractor.train()
                Pnet.train()
                Discrim.train()
                Generator.train()

        if (opt.training.use_validation) and epoch % opt.training.validation_epoch_interval==0:
            print('Validating...')
            with torch.no_grad():
                AppearCodeExtractor.eval()
                GeoCodeExtractor.eval()
                Pnet.eval()
                Generator.eval()
                Discrim.eval()

                val_batch_count = 0
                val_loss = []
                all_losses = []

                val_image_outputs = []
                val_image_gts = []
                for (idx, batch) in enumerate(val_dataloader):
                    source_verts = batch['source_meshes'].verts_padded().float().to(device)
                    source_faces = batch['source_meshes'].faces_padded().to(device)
                    source_images = batch['source_image'].to(device)


                    eye_tensor = torch.Tensor(
                        [[0, 0, 1]]).to(device)
                    at_tensor = torch.Tensor(
                        [[0, 0, 0]]).to(device)

                    # mesh image
                    renderer = create_phong_renderer(eye_tensor, at_tensor, opt.model.image_size, device= device)
                    gray_image = renderer(batch['source_meshes'].to(device))
                    gray_image_np = (255 * gray_image[0, ..., :3].cpu().detach().numpy()).astype('uint8')

                    # reconstruct original image
                    renderer = create_renderer(eye_tensor, at_tensor, image_size=128, device= device)

                    input_verts = source_verts.permute(0, 2, 1)
                    source_geo_code, _ = GeoCodeExtractor(input_verts)
                    source_render_texture = TexturesVertex(verts_features=source_geo_code)
                    source_train_meshes = Meshes(verts=source_verts,
                                                 faces=source_faces,
                                                 textures=source_render_texture)

                    source_geo_features = renderer(source_train_meshes)
                    source_geo_features = source_geo_features.permute(0, 3, 1, 2)
                    source_geo_features = Pnet(source_geo_features).contiguous()

                    source_appear_code = AppearCodeExtractor(source_images)
                    recon_images, _ = Generator(init_features=source_geo_features, styles=[source_appear_code],
                                                input_is_latent=True)


                    pred_images_np = (255 * recon_images.permute(0, 2, 3, 1)[0, ..., :3].cpu().detach().numpy()).astype(
                        'uint8')
                    gt_images_np = (255 * source_images.permute(0, 2, 3, 1)[0, ..., :3].cpu().detach().numpy()).astype('uint8')


                    gt_image = Image.fromarray(gt_images_np)
                    pred_image = Image.fromarray(pred_images_np)
                    all_losses.append(cal_similarities(gt_image, pred_image))


                    val_batch_count += 1
                    if (val_batch_count <=5):
                        composed_img_np = np.concatenate([gray_image_np,gt_images_np,pred_images_np],0)
                        composed_img = Image.fromarray(composed_img_np)
                        val_image_outputs.append(composed_img)

                    if (val_batch_count >= 50):
                        break

                # val_loss_np = np.array(val_loss)
                # val_loss_mean = np.mean(val_loss_np)
                # val_loss_record.append(val_loss_mean)
                # val_loss_epoch.append(cur_batches)
                val_metrics = {}
                if opt.training.wandb:
                    val_metrics["val/val_output"] = [wandb.Image(image,caption = "Top: Mesh reference Medium: Reference image for reconstruction, Bottom: Output") for image in val_image_outputs]

                metrics_header = ['mse', 'ssim', 'lpips']
                for k in metrics_header:
                    temp_array = []
                    for i in range(len(all_losses)):
                        temp_array.append(all_losses[i][k])
                    temp_array = np.array(temp_array)
                    val_metrics["val/"+k] =np.mean(temp_array)
                    print(k+":" +str(np.mean(temp_array)) )

                if opt.training.wandb:
                    wandb.log({**val_metrics})


    print("Finished training, time consumption: " + str(total_time_consume))
    if opt.training.wandb:
        wandb.finish()

if __name__ == '__main__':
    opt = BaseOptions().parse()
    train(opt)