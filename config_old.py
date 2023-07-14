import configargparse
from munch import *

class BaseOptions():
    def __init__(self):
        self.parser = configargparse.ArgumentParser()
        self.initialized = False

    def initialize(self):

        # Dataset options
        dataset = self.parser.add_argument_group('dataset')
        dataset.add_argument("--dataset_path", type=str, default='./datasets/FFHQ_SDF_Test_5_fixed_angles', help="path to the dataset")

        # Experiment Options
        experiment = self.parser.add_argument_group('experiment')
        experiment.add_argument('--config', is_config_file=True, help='config file path')
        experiment.add_argument("--expname", type=str, default='debug', help='experiment name')
        experiment.add_argument("--continue_training", action="store_true", help="continue training the model")
        experiment.add_argument("--debug_gpu_mem_log", action="store_true", help="print GPU memory debug log")
        experiment.add_argument("--local_rank", type=int,default=-1)

        # Training loop options
        training = self.parser.add_argument_group('training')
        training.add_argument("--checkpoints_dir", type=str, default='./checkpoint', help='checkpoints directory name')
        training.add_argument("--checkpoints_epoch", type=int, default=20, help='which epoch\'s weight of checkpoints should be load')

        training.add_argument("--model_save_epoch_interval", type=int, default=2,help='interval between saving epoches')
        training.add_argument("--image_log_batch_interval", type=int, default=100,
                              help='interval between logging')
        training.add_argument("--image_log_batch_size", type=int, default=1,
                              help='log how many images into wandb')

        training.add_argument("--epoch", type=int, default=20, help="total number of training iterations")
        training.add_argument("--batch_size", type=int, default=4, help="batch sizes for each GPU. A single RTX2080 can fit batch=4, chunck=1 into memory.")
        training.add_argument("--chunk", type=int, default=4, help='number of samples within a batch to processed in parallel, decrease if running out of memory')
        training.add_argument("--dataloader_number_workers", type=int, default=10, help="number of workers used by the dataloader")
        training.add_argument("--val_length", type=int, default=1,
                              help="how many samples for validation.")

        # training.add_argument("--val_n_sample", type=int, default=8, help="number of test samples generated during training")
        # training.add_argument("--d_reg_every", type=int, default=16, help="interval for applying r1 regularization to the StyleGAN generator")
        # training.add_argument("--g_reg_every", type=int, default=4, help="interval for applying path length regularization to the StyleGAN generator")
        # training.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
        # training.add_argument("--mixing", type=float, default=0.9, help="probability of latent code mixing")

        training.add_argument("--lr", type=float, default=0.0001, help="learning rate")

        # training.add_argument("--path_regularize", type=float, default=2, help="weight of the path length regularization")
        # training.add_argument("--path_batch_shrink", type=int, default=2, help="batch size reducing factor for the path length regularization (reduce memory consumption)")

        training.add_argument("--RECON_L1_WEIGHT", type=float, default=1, help="weight of the RECON_L1_WEIGHT")
        training.add_argument("--RECON_VGG_WEIGHT", type=float, default=1, help="weight of the RECON_VGG_WEIGHT")
        training.add_argument("--RECON_GAN_WEIGHT", type=float, default=1, help="weight of the RECON_GAN_WEIGHT")
        training.add_argument("--RECON_LND_WEIGHT", type=float, default=1, help="weight of the RECON_LND_WEIGHT")
        training.add_argument("--RECON_APP_WEIGHT", type=float, default=1, help="weight of the RECON_APP_WEIGHT")

        training.add_argument("--TRANSFER_LND_WEIGHT", type=float, default=1, help="weight of the TRANSFER_LND_WEIGHT")
        training.add_argument("--TRANSFER_APP_WEIGHT", type=float, default=1, help="weight of the TRANSFER_APP_WEIGHT")
        training.add_argument("--TRANSFER_GAN_WEIGHT", type=float, default=1, help="weight of the TRANSFER_GAN_WEIGHT")

        training.add_argument("--wandb", action="store_true", help="use weights and biases logging")
        training.add_argument("--use_validation", action="store_true", help="use validation step")
        training.add_argument("--validation_epoch_interval", type=int, default=5,
                              help='interval between validations')
        training.add_argument("--pretrained_stylegan2", action="store_true", help="use pretrained stylegan2")
        # Inference Options
        inference = self.parser.add_argument_group('inference')
        inference.add_argument("--results_dir", type=str, default='./evaluations', help='results/evaluations directory name')
        inference.add_argument("--truncation_ratio", type=float, default=0.5, help="truncation ratio, controls the diversity vs. quality tradeoff. Higher truncation ratio would generate more diverse results")
        inference.add_argument("--truncation_mean", type=int, default=10000, help="number of vectors to calculate mean for the truncation")
        inference.add_argument("--identities", type=int, default=16, help="number of identities to be generated")
        inference.add_argument("--num_views_per_id", type=int, default=1, help="number of viewpoints generated per identity")
        inference.add_argument("--no_surface_renderings", action="store_true", help="when true, only RGB outputs will be generated. otherwise, both RGB and depth videos/renderings will be generated. this cuts the processing time per video")
        inference.add_argument("--fixed_camera_angles", action="store_true", help="when true, the generator will render indentities from a fixed set of camera angles.")
        inference.add_argument("--azim_video", action="store_true", help="when true, the camera trajectory will travel along the azimuth direction. Otherwise, the camera will travel along an ellipsoid trajectory.")

        # Generator options
        model = self.parser.add_argument_group('model')
        model.add_argument("--image_size", type=int, default=256, help="image sizes for the model")
        model.add_argument("--style_dim", type=int, default=512, help="number of style input dimensions")
        model.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier factor for the StyleGAN decoder. config-f = 2, else = 1")
        model.add_argument("--geo_feature_channels", type=int, default=24, help="number of GeoNet created feature map channels")

        model.add_argument("--n_mlp", type=int, default=8, help="number of mlp layers in stylegan's mapping network")
        model.add_argument("--lr_mapping", type=float, default=0.01, help='learning rate reduction for mapping network MLP layers')
        model.add_argument("--renderer_spatial_output_dim", type=int, default=64, help='spatial resolution of the StyleGAN decoder inputs')
        model.add_argument("--project_noise", action='store_true', help='when true, use geometry-aware noise projection to reduce flickering effects (see supplementary section C.1 in the paper). warning: processing time significantly increases with this flag to ~20 minutes per video.')

        # Camera options
        # camera = self.parser.add_argument_group('camera')
        # camera.add_argument("--uniform", action="store_true", help="when true, the camera position is sampled from uniform distribution. Gaussian distribution is the default")
        # camera.add_argument("--azim", type=float, default=0.3, help="camera azimuth angle std/range in Radians")
        # camera.add_argument("--elev", type=float, default=0.15, help="camera elevation angle std/range in Radians")
        # camera.add_argument("--fov", type=float, default=6, help="camera field of view half angle in Degrees")
        # camera.add_argument("--dist_radius", type=float, default=0.12, help="radius of points sampling distance from the origin. determines the near and far fields")

        self.initialized = True

    def parse(self):
        self.opt = Munch()
        if not self.initialized:
            self.initialize()
        try:
            args = self.parser.parse_args()
        except: # solves argparse error in google colab
            args = self.parser.parse_args(args=[])

        for group in self.parser._action_groups[2:]:
            title = group.title
            self.opt[title] = Munch()
            for action in group._group_actions:
                dest = action.dest
                self.opt[title][dest] = args.__getattribute__(dest)

        return self.opt