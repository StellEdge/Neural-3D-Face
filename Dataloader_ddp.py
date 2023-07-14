import os
import random

import torch
import numpy as np
from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from pytorch3d.io import load_objs_as_meshes, load_obj
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
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.io import IO
from renderers import *
from PIL import Image
# import pandas as pd
# from quad_mesh_simplify import simplify_mesh
import pickle
import pyfqmr

class GeoFaceFFHQDatasetV3(Dataset):


    def __init__(self,imagesize, dataset_path,use_mesh_simplify=True,max_vertices = 16000):
        self.image_size = imagesize
        self.dataset_path =dataset_path
        # load all objs previously.
        if (os.path.exists(os.path.join(dataset_path, 'index.pkl'))):
            with open(os.path.join(dataset_path, 'index.pkl'), 'rb') as pkl_file:
                self.index_dict = pickle.load(pkl_file)
                print("Dataset length:",len(self.index_dict))
        else:
            raise RuntimeError("No index.pkl file found.")

        self.use_mesh_simplify = use_mesh_simplify
        self.max_vertices = max_vertices
        self.mesh_simplifier = pyfqmr.Simplify()
    def __len__(self):
        return len(self.index_dict)

    def get_simplified_mesh(self,meshes,max_vertices):
        verts = meshes.verts_list()[0].cpu().numpy().astype(np.double)
        faces_idx = meshes.faces_list()[0].cpu().numpy().astype(np.uint32)

        self.mesh_simplifier.setMesh(verts, faces_idx)
        self.mesh_simplifier.simplify_mesh(target_count=max_vertices, aggressiveness=4, preserve_border=True,
                                      verbose=0)
        new_verts, new_faces_idx, _ = self.mesh_simplifier.getMesh()
        # new_verts = new_verts / np.max(new_verts)

        tex = TexturesVertex(verts_features=torch.full((1, new_verts.shape[0], 3), 0.5))
        input_mesh = Meshes(verts=[torch.from_numpy(new_verts.astype(np.float32))],
                            faces=[torch.from_numpy(new_faces_idx.astype(np.int32))], textures=tex)
        return input_mesh

    def __getitem__(self, idx):

        paired_idx = random.randint(0,len(self.index_dict)-1)

        # source_params = self.index_dict[idx]['g_params']['cam_extrinsics']
        # target_params = self.index_dict[paired_idx]['g_params']['cam_extrinsics']
        source_params = self.index_dict[idx]['g_params']['sample_z']
        target_params = self.index_dict[paired_idx]['g_params']['sample_z']

        source_R = self.index_dict[idx]['g_params']['R']
        target_R = self.index_dict[paired_idx]['g_params']['R']
        source_T = self.index_dict[idx]['g_params']['T']
        target_T = self.index_dict[paired_idx]['g_params']['T']

        source_mesh = IO().load_mesh(os.path.join(self.dataset_path,self.index_dict[idx]['marching_cubes_meshes_path']),include_textures = False)
        target_mesh = IO().load_mesh(os.path.join(self.dataset_path,self.index_dict[paired_idx]['marching_cubes_meshes_path']),include_textures = False)

        if self.use_mesh_simplify:
            source_mesh = self.get_simplified_mesh(source_mesh, self.max_vertices)
            target_mesh = self.get_simplified_mesh(target_mesh, self.max_vertices)
        # source_mesh = load_obj(os.path.join(self.dataset_path,self.index_dict[idx]['marching_cubes_mesh_path']))
        # target_mesh = load_obj(os.path.join(self.dataset_path,self.index_dict[paired_idx]['marching_cubes_mesh_path']))


        source_image = None
        target_image = None

        with Image.open(os.path.join(self.dataset_path,self.index_dict[idx]['image_path'])) as s_image:
            source_image = torch.from_numpy(1.0 / 255 * np.array([np.array(s_image)], dtype='float32')).permute(0, 3,1, 2)

        with Image.open(os.path.join(self.dataset_path,self.index_dict[paired_idx]['image_path'])) as t_image:
            target_image = torch.from_numpy(1.0 / 255 * np.array([np.array(t_image)], dtype='float32')).permute(0, 3,1, 2)

        r_size = transforms.Resize((self.image_size,self.image_size))
        source_image = r_size(source_image)
        target_image = r_size(target_image)

        sample = {
            'source_mesh': source_mesh,
            # 'source_verts': source_verts,
            # 'source_faces': source_faces,
            'source_image': source_image,
            'source_params': source_params,
            'target_mesh': target_mesh,
            # 'target_verts': target_verts,
            # 'target_faces': target_faces,
            'target_image': target_image,
            'target_params': target_params,
            'source_R': source_R,
            'source_T': source_T,
            'target_R': target_R,
            'target_T': target_T,
        }
        return sample


def collate_fn_ffhqv3(data):
    source_meshes = []
    for sample in data:
        source_meshes.append(sample['source_mesh'])
        # sample.pop('meshes')
    source_meshes = join_meshes_as_batch(source_meshes)

    target_meshes = []
    for sample in data:
        target_meshes.append(sample['target_mesh'])
        # sample.pop('meshes')
    target_meshes = join_meshes_as_batch(target_meshes)

    batch = {}
    # batch['meshes'] = meshes
    batch['source_meshes'] = source_meshes
    # batch['source_verts'] = torch.cat([sample['source_verts'] for sample in data], 0)
    # batch['source_faces'] = torch.cat([sample['source_faces'] for sample in data], 0)
    batch['source_image'] = torch.cat([sample['source_image'] for sample in data], 0)
    batch['source_params'] = torch.cat([sample['source_params'] for sample in data], 0)
    batch['target_meshes'] = target_meshes
    # batch['target_verts'] = torch.cat([sample['target_verts'] for sample in data], 0)
    # batch['target_faces'] = torch.cat([sample['target_faces'] for sample in data], 0)
    batch['target_image'] = torch.cat([sample['target_image'] for sample in data], 0)
    batch['target_params'] = torch.cat([sample['target_params'] for sample in data], 0)

    batch['target_R'] = torch.cat([sample['target_R'] for sample in data], 0)
    batch['target_T'] = torch.cat([sample['target_T'] for sample in data], 0)
    batch['source_R'] = torch.cat([sample['source_R'] for sample in data], 0)
    batch['source_T'] = torch.cat([sample['source_T'] for sample in data], 0)

     # batch['ref_vector'] = torch.cat([sample['ref_vector'] for sample in data], 0)
    return batch

def get_dataloader_ffhqv3(dataset_path,image_size, batch_size, num_workers,shuffle=True):
    trainset = GeoFaceFFHQDatasetV3(image_size,dataset_path)
    train_sampler = DistributedSampler(trainset)
    train_dataloader = DataLoader(trainset, batch_size=batch_size,num_workers=num_workers,sampler=train_sampler,
                                  collate_fn=collate_fn_ffhqv3)
    return train_dataloader

def create_dataloader_splitted_v3(dataset_path,image_size, batch_size,num_workers,test_len):
    training_data = GeoFaceFFHQDatasetV3(image_size,dataset_path)

    # train_len = int(len(training_data) * 0.8)
    # test_len = len(training_data) - train_len

    train_len = len(training_data) - test_len
    trainset, testset = random_split(training_data, [train_len, test_len], generator=torch.Generator().manual_seed(42))

    train_sampler = DistributedSampler(trainset)
    train_dataloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers,sampler=train_sampler,
                                  collate_fn=collate_fn_ffhqv3)

    test_sampler = DistributedSampler(testset)
    test_dataloader = DataLoader(testset, batch_size=1, num_workers=num_workers,sampler=test_sampler,
                                 collate_fn=collate_fn_ffhqv3)
    return train_dataloader, test_dataloader


def create_dataloader(dataset_path, batch_size, num_workers):
    training_data = GeoFaceDataset(dataset_path)

    # train_len = int(len(training_data) * 0.8)
    # test_len = len(training_data) - train_len

    train_len = len(training_data) - batch_size
    test_len = batch_size

    trainset, testset = random_split(training_data, [train_len, test_len], generator=torch.Generator().manual_seed(42))
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  collate_fn=collate_fn)
    test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 collate_fn=collate_fn)
    return train_dataloader, test_dataloader


def get_dataloader_sub(dataset_path, imagesize, batch_size, num_workers):
    trainset = GeoFaceDataset(dataset_path, imagesize)
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  collate_fn=collate_fn)
    return train_dataloader


def create_dataloader_splitted(dataset_path, imagesize, batch_size, num_workers):
    return get_dataloader_sub(os.path.join(dataset_path, "train"), imagesize, batch_size, num_workers), \
           get_dataloader_sub(os.path.join(dataset_path, "val"), imagesize, 1, num_workers), \
           get_dataloader_sub(os.path.join(dataset_path, "test"), imagesize, batch_size, num_workers)


def test_dataloader_speed(dataset_path):
    from time import time
    import multiprocessing as mp
    for num_workers in range(2, mp.cpu_count(), 2):
        print("Evaluating :", num_workers)
        start = time()
        train_loader = get_dataloader_sub(dataset_path, 16, num_workers)
        for epoch in range(1, 3):
            for i, batch in enumerate(train_loader, 0):
                # meshes = batch['meshes'].to(device)
                verts = batch['verts'].to(device)
                faces = batch['faces'].to(device)

                ref_images = batch['ref_image'].to(device)
                gt_images = batch['gt_image'].to(device)
                eye_tensor = batch['eye_tensor'].to(device)
                at_tensor = batch['at_tensor'].to(device)

        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))


# with open('datasetBFMXV/mesh_data.pkl','rb') as pkl_file:
#     dict_mesh = pickle.load(pkl_file)
#     for k_path in dict_mesh:
#         print(k_path)
#         for content in dict_mesh[k_path]:
#             print(content)
#             print(dict_mesh[k_path][content])

# train_dataloader = create_dataloader(1)
# sample = next(iter(train_dataloader))
# for x in sample.keys():
#   print(x)
#   if not x=='meshes':
#       print(sample[x].shape)
if __name__ == "__main__":
    test_dataloader_speed("datasetBFMZIP_20_simple_withmesh/train")
