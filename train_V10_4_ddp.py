import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import lpips
import numpy as np
import wandb
from PIL import Image
from renderers import *
from config import BaseOptions
from Utils import *

from Dataloader_ddp import create_dataloader_splitted_v3,get_dataloader_ffhqv3
# from ModuleLoader import build_models


# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.renderer import (
    TexturesVertex,
)
import models.pointnet2_sem_seg_msg as PointNet
from models.encoders.psp_encoders import GradualStyleEncoder as StyleGANEncoder
from models.modules import PNet,PNet_NoRes
from models.stylegan2.model import StyleGAN2Generator,Discriminator
from models.new_modules import LandmarkExtractor,GeometryEncoder_Projection

# scp -P 22701 -r  chenruizhao@202.120.38.4:/home/chenruizhao/RemoteWorks/GeoFace/dataset_1000id_V2 /home/wangyue/crz_work/GeoFace

from gpu_mem_track import MemTracker
from sklearn.decomposition import PCA
from torchvision import transforms

# from pytorch_msssim import MS_SSIM

def train(opt):

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    opt.distributed = n_gpu > 1
    #cuda Init
    dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端
    local_rank = dist.get_rank()
    print(f"Running DDP training on rank {local_rank}.")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # else:
    #     if torch.cuda.is_available():
    #         device = torch.device("cuda:0")
    #         torch.cuda.set_device(device)
    #     else:
    #         device = torch.device("cpu")

    # torch.autograd.set_detect_anomaly(True)

    if opt.training.use_validation:
        from Metrics import cal_similarities

    print(timelog_str() + " Starting task.")
    if opt.training.wandb:
        print("Using wandb.")
        wandb.login()
        wandb.init(
            project="GeoFace_V2",
            entity="geofacedev",
            group="DDP_"+opt.experiment.expname,
            name=opt.experiment.expname,
            config=opt)

    random.seed(10)

    loss_record = []
    loss_epoch = []
    Dloss_record = []

    val_loss_record = []
    val_loss_epoch = []

    debug_gpu_mem_log = opt.experiment.debug_gpu_mem_log
    if debug_gpu_mem_log:
        gpu_tracker = MemTracker()

    L1Loss = nn.L1Loss()
    MSELoss = nn.MSELoss()
    VGGLoss = lpips.LPIPS(net='vgg').to(device)

    # ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=3)

    RECON_L1_WEIGHT = opt.training.RECON_L1_WEIGHT
    RECON_VGG_WEIGHT = opt.training.RECON_VGG_WEIGHT
    RECON_GAN_WEIGHT = opt.training.RECON_GAN_WEIGHT
    RECON_LND_WEIGHT = opt.training.RECON_LND_WEIGHT
    RECON_APP_WEIGHT = opt.training.RECON_APP_WEIGHT

    TRANSFER_GAN_WEIGHT = opt.training.TRANSFER_GAN_WEIGHT
    TRANSFER_LND_WEIGHT = opt.training.TRANSFER_LND_WEIGHT
    TRANSFER_APP_WEIGHT = opt.training.TRANSFER_APP_WEIGHT

    use_landmark_loss = False

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
    AppearCodeExtractor = StyleGANEncoder(50, 'ir_se').cuda()

    # def get_keys(d, name):
    #     if 'state_dict' in d:
    #         d = d['state_dict']
    #     d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    #     return d_filt
    # pSp_checkpoint_path = "./pretrained_modules/psp_ffhq_encode.pt"
    # ckpt = torch.load(pSp_checkpoint_path)
    # AppearCodeExtractor.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)

    # for i in AppearCodeExtractor.parameters():
    #     i.requires_grad = False
    # AppearCodeExtractor.eval()

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
        checkpoint = torch.load("./models/pretrained_data/stylegan_ffhq_256_550000.pt",map_location=device)
        Generator.load_state_dict(checkpoint["g"], strict=False)
    # checkpoint = torch.load("stylegan2-ffhq-config-f.pt")
    # checkpoint = torch.load("ffhq-256-config-e-003810.pt")

    # Generator.load_state_dict(checkpoint["g_ema"])
    if debug_gpu_mem_log:
        gpu_tracker.track()

    Discrim = Discriminator(opt.model.image_size, channel_multiplier=opt.model.channel_multiplier).to(device)

    if use_landmark_loss:
        LndExtractor = LandmarkExtractor(pretrained = True).cuda()
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
                    torch.load(os.path.join(load_model_path, 'AppearCodeExtractor_' + str(load_model_step) + '.pth'),map_location=device))
                GeoCodeExtractor.load_state_dict(
                    torch.load(os.path.join(load_model_path, 'GeoCodeExtractor_' + str(load_model_step) + '.pth'),map_location=device))
                Pnet.load_state_dict(
                    torch.load(os.path.join(load_model_path, 'Pnet_' + str(load_model_step) + '.pth'),map_location=device))
                Generator.load_state_dict(
                    torch.load(os.path.join(load_model_path, 'Generator_' + str(load_model_step) + '.pth'),map_location=device))
                Discrim.load_state_dict(
                    torch.load(os.path.join(load_model_path, 'Discrim_' + str(load_model_step) + '.pth'),map_location=device))

    if(opt.distributed):
        print("Converting models to DDP models...")
        AppearCodeExtractor = DDP(AppearCodeExtractor, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False,)
        GeoCodeExtractor = DDP(GeoCodeExtractor, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False,)
        Pnet = DDP(Pnet, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False,)
        Generator = DDP(Generator, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True,
            broadcast_buffers=False,)
        Discrim = DDP(Discrim, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False,)
        print("Converting model")

    timelog('Network models built.')

    # TODO : Try RMSProp
    optimizer = torch.optim.Adam([
        {'params': AppearCodeExtractor.parameters()},
        {'params': GeoCodeExtractor.parameters()},
        {'params': Pnet.parameters()},
        {'params': Generator.parameters()}
    ],
        lr=opt.training.lr, betas=(0.9, 0.999), eps=1e-8)
    optimizerD = torch.optim.Adam(Discrim.parameters(), lr=opt.training.lr, betas=(0.9, 0.999), eps=1e-8)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,100], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,1.05)

    cur_batches = 0
    total_time_consume = 0

    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    #training loop
    ResizeImage = transforms.Resize((256,256))

    for epoch in range(1, opt.training.epoch + 1):
        AppearCodeExtractor.train()
        GeoCodeExtractor.train()
        Pnet.train()
        Generator.train()
        epoch_start_time = time.time()

        #training
        for (idx, batch) in enumerate(train_dataloader):
            cur_batches = cur_batches + 1
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
            renderer = create_renderer(eye_tensor, at_tensor, image_size=128,device=device)
            if debug_gpu_mem_log:
                gpu_tracker.track()
            # Discriminator training------------------------------------------------------------------------------------
            optimizerD.zero_grad()
            AppearCodeExtractor.eval()
            GeoCodeExtractor.eval()
            Pnet.eval()
            Generator.eval()
            Discrim.train()


            # Get faked image
            input_verts = source_verts.permute(0, 2, 1)
            source_geo_code, _ = GeoCodeExtractor(input_verts)

            # target_ref_images = ResizeImage(target_images)
            # target_appear_code = AppearCodeExtractor(target_ref_images)
            target_appear_code = AppearCodeExtractor(target_images)
            source_render_texture = TexturesVertex(verts_features=source_geo_code)
            source_train_meshes = Meshes(verts=source_verts,
                                  faces=source_faces,
                                  textures=source_render_texture)

            source_geo_features = renderer(source_train_meshes)

            source_geo_features = source_geo_features.permute(0, 3, 1, 2)
            source_geo_features = Pnet(source_geo_features).contiguous()
            fake_images, _ = Generator(init_features=source_geo_features, styles=[target_appear_code], input_is_latent=True)

            if debug_gpu_mem_log:
                gpu_tracker.track()

            #reconstruct original image

            # source_ref_images = ResizeImage(source_images)
            # source_appear_code = AppearCodeExtractor(source_ref_images)
            source_appear_code = AppearCodeExtractor(source_images)
            recon_images, _ = Generator(init_features=source_geo_features, styles=[source_appear_code],
                                       input_is_latent=True)

            if debug_gpu_mem_log:
                gpu_tracker.track()

            #loss calc
            source_features = Discrim(source_images)
            recon_features = Discrim(recon_images)
            fake_features = Discrim(fake_images)
            target_features = Discrim(target_images)

            source_loss = F.softplus(-source_features)
            recon_loss = F.softplus(recon_features)
            target_loss = F.softplus(-target_features)
            fake_loss = F.softplus(fake_features)

            DLoss = recon_loss.mean() + source_loss.mean() + fake_loss.mean() + target_loss.mean()

            DLoss.backward()
            # for name, param in Generator.named_parameters():
            #     if param.grad is None:
            #         print("Not Used Param:",name)
            optimizerD.step()

            if debug_gpu_mem_log:
                gpu_tracker.track()

            del fake_images
            del recon_images
            torch.cuda.empty_cache()

            if debug_gpu_mem_log:
                gpu_tracker.track()

            # Generator training----------------------------------------------------------------------------------------
            optimizer.zero_grad()
            AppearCodeExtractor.train()
            GeoCodeExtractor.train()
            Pnet.train()
            Generator.train()
            Discrim.eval()

            train_log_image = {}

            # Get faked image
            source_geo_code, _ = GeoCodeExtractor(input_verts)
            target_ref_images = ResizeImage(target_images)
            target_appear_code = AppearCodeExtractor(target_ref_images)
            source_render_texture = TexturesVertex(verts_features=source_geo_code)
            source_train_meshes = Meshes(verts=source_verts,
                                  faces=source_faces,
                                  textures=source_render_texture)
            source_geo_features = renderer(source_train_meshes)

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

            fake_images, _ = Generator(init_features=source_geo_features, styles=[target_appear_code], input_is_latent=True)

            if cur_batches % opt.training.image_log_batch_interval == 0:
                images_np = []
                for img_index in range(opt.training.image_log_batch_size):
                    im = Image.fromarray(
                        (255 * fake_images.permute(0, 2, 3, 1)[img_index, ..., :3].cpu().detach().numpy()).astype(
                            'uint8'))
                    images_np.append(im)
                    im.save(os.path.join(tmp_path_images, str(cur_batches) + '_afterRenderer' + str(img_index) + '.jpg'))
                train_log_image['train/after_renderer'] = [wandb.Image(image, caption="Output After Renderer") for image in images_np]

            if debug_gpu_mem_log:
                gpu_tracker.track()

            #reconstruct original image

            source_ref_images = ResizeImage(source_images)
            source_appear_code = AppearCodeExtractor(source_ref_images)
            recon_images, _ = Generator(init_features=source_geo_features, styles=[source_appear_code],
                                       input_is_latent=True)

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
            #
            #
            # fake_ref_images = ResizeImage(fake_images)
            recon_appear_code = AppearCodeExtractor(recon_images)

            fake_ref_images = ResizeImage(fake_images)
            fake_appear_code = AppearCodeExtractor(fake_ref_images)
            # consistency_images, _ = Generator(init_features=target_encoded_geo_features, styles=[fake_appear_code],
            #                            input_is_latent=True)
            if debug_gpu_mem_log:
                gpu_tracker.track()
            # source and reconstructed source
            recon_L1Loss = RECON_L1_WEIGHT * L1Loss(recon_images, source_images)

            recon_VGGLoss = RECON_VGG_WEIGHT * torch.mean(VGGLoss(recon_images * 2 - 1, source_images * 2 - 1))
            if debug_gpu_mem_log:
                gpu_tracker.track()

            # recon_MSSSIMLoss = 1 - ms_ssim_module(recon_images, source_images)

            if debug_gpu_mem_log:
                gpu_tracker.track()

            recon_features = Discrim(recon_images)
            recon_Gloss = RECON_GAN_WEIGHT * F.softplus(-recon_features).mean()

            recon_appear_loss = RECON_APP_WEIGHT * L1Loss(recon_appear_code, source_appear_code)
            recon_loss = recon_L1Loss + recon_VGGLoss + recon_Gloss +  recon_appear_loss

            if use_landmark_loss:
                recon_lnd_features = LndExtractor(recon_images)
                source_lnd_features = LndExtractor(source_images)
                recon_lnd_loss = RECON_LND_WEIGHT * MSELoss(recon_lnd_features, source_lnd_features )
                recon_loss += recon_lnd_loss
            else:
                recon_lnd_loss = 0

            #style transferred image and transfer target

            # transfer_L1Loss = L1_WEIGHT * L1Loss(fake_images, target_images)

            # transfer_VGGLoss = VGG_WEIGHT * torch.mean(VGGLoss(fake_images * 2 - 1, target_images * 2 - 1))

            transfer_features = Discrim(fake_images)
            transfer_Gloss = TRANSFER_GAN_WEIGHT * F.softplus(-transfer_features).mean()
            transfer_appear_loss = TRANSFER_APP_WEIGHT * L1Loss(fake_appear_code,target_appear_code)

            transfer_loss = transfer_Gloss +  transfer_appear_loss  #transfer_L1Loss + transfer_VGGLoss +

            if use_landmark_loss:
                transfer_lnd_features = LndExtractor(fake_images)
                transfer_lnd_loss = TRANSFER_LND_WEIGHT * MSELoss(transfer_lnd_features, source_lnd_features)
                transfer_loss += transfer_lnd_loss
            else:
                transfer_lnd_loss = 0

            GLoss = recon_loss + transfer_loss   #+consistency_loss

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

            GLoss.backward()
            optimizer.step()

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
            timelog('Epoch-{}/Batch-{} G Loss: {} / D Loss: {} '.format(epoch, cur_batches, GLoss.item(),DLoss.item()))
            # timelog('   --ReconstructLoss: L1:{}, VGGLoss: {}, GLoss: {}, Landmark Loss: {} '.format(recon_L1Loss.item(),recon_VGGLoss.item(),recon_Gloss.item(),recon_lnd_loss.item()))
            # timelog(
            #     '   --Transfer Loss: L1:{}, VGGLoss: {}, GLoss: {}, Landmark Loss: {}, appear Loss: {} '.format(transfer_L1Loss.item(),
            #                                                                                      transfer_VGGLoss.item(),
            #                                                                                      transfer_Gloss.item(),
            #                                                                                      transfer_lnd_loss.item(),
            #                                                                                    transfer_appear_loss.item()))
            if use_landmark_loss:
                timelog('   --ReconstructLoss: L1:{}, VGG Loss: {}, GANLoss: {}, Landmark Loss: {}, appear Loss: {} '.format(recon_L1Loss.item(),recon_VGGLoss.item(),recon_Gloss.item(),recon_lnd_loss.item(),recon_appear_loss.item()))
                timelog(
                    '   --Transfer Loss: GANLoss: {}, Landmark Loss: {}, appear Loss: {} '.format(transfer_Gloss.item(),
                                                                                                     transfer_lnd_loss.item(),
                                                                                                   transfer_appear_loss.item()))
            else:
                timelog('   --ReconstructLoss: L1:{}, VGG Loss: {}, GANLoss: {}, appear Loss: {} '.format(recon_L1Loss.item(),recon_VGGLoss.item(),recon_Gloss.item(),recon_appear_loss.item()))
                timelog('   --Transfer Loss: GANLoss: {}, appear Loss: {} '.format(transfer_Gloss.item(),transfer_appear_loss.item()))

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
                       "train/epoch": epoch,
                       "train/example_count": cur_batches * opt.training.batch_size,
                       #"train/learning_rate": scheduler.get_last_lr()[0],
                       }

            if opt.training.wandb:
                if cur_batches % opt.training.image_log_batch_interval == 0:
                    wandb.log({**train_log_image,**metrics})
                else:
                    wandb.log(metrics)
            Dloss_record.append(DLoss.item())
            loss_record.append(GLoss.item())
            loss_epoch.append(cur_batches)
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

        if epoch % opt.training.model_save_epoch_interval == 0 and dist.get_rank() == 0:
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
                    renderer = create_phong_renderer(eye_tensor, at_tensor, opt.model.image_size,device=device)
                    gray_image = renderer(batch['source_meshes'].to(device))
                    gray_image_np = (255 * gray_image[0, ..., :3].cpu().detach().numpy()).astype('uint8')

                    # reconstruct original image
                    renderer = create_renderer(eye_tensor, at_tensor, image_size=128,device=device)

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

    dist.destroy_process_group()

if __name__ == '__main__':
    opt = BaseOptions().parse()
    train(opt)