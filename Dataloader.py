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

class GeoFaceDataset(Dataset):
    def __init__(self, dataset_path, imagesize):
        self.image_size = imagesize
        self.csv_labels = pd.read_csv(os.path.join(dataset_path, 'index.csv'))
        header = ["mesh_obj_path", "ref_image_path", "gt_image_path", "eye_tensor0", "eye_tensor1", "eye_tensor2",
                  "original0", "original1", "original2"]
        for i in range(3, len(header)):
            self.csv_labels[header[i]] = self.csv_labels[header[i]].astype("float32")
        '''
            index.csv contents:
            line idx: mesh_path, ref_image_path, groundtruth_image_path, camera_vectors(6 dim,eyes_vector and at_vector), 
        '''

        # load all objs previously.
        self.mesh_objs = {}
        if (os.path.exists(os.path.join(dataset_path, 'mesh_data.pkl'))):
            with open(os.path.join(dataset_path, 'mesh_data.pkl'), 'rb') as pkl_file:
                self.mesh_objs = pickle.load(pkl_file)
        else:
            raise RuntimeError("No mesh_data.pkl file found.")

        self.ref_images = {}
        if (os.path.exists(os.path.join(dataset_path, 'ref_image_data.pkl'))):
            with open(os.path.join(dataset_path, 'ref_image_data.pkl'), 'rb') as pkl_file:
                self.ref_images = pickle.load(pkl_file)
        else:
            raise RuntimeError("No ref_image_data.pkl file found.")

        self.gt_images = {}
        if (os.path.exists(os.path.join(dataset_path, 'gt_image_data.pkl'))):
            with open(os.path.join(dataset_path, 'gt_image_data.pkl'), 'rb') as pkl_file:
                self.gt_images = pickle.load(pkl_file)
        else:
            raise RuntimeError("No gt_image_data.pkl file found.")

        # for i in range(len(self.csv_labels)):
        #     mesh_obj_path = self.csv_labels.iloc[i, 0]
        #     if not (mesh_obj_path in self.mesh_objs):
        #         verts, faces, _ = load_obj(mesh_obj_path)
        #         mesh = Meshes(verts=[verts], faces=[faces.verts_idx])
        #         self.mesh_objs[mesh_obj_path] = mesh

    def __len__(self):
        return len(self.csv_labels)

    def __getitem__(self, idx):
        # mesh_obj_path = self.csv_labels.iloc[idx, 0]
        mesh_obj_name = self.csv_labels.iloc[idx, 0]
        ref_image_path = self.csv_labels.iloc[idx, 1]
        gt_image_path = self.csv_labels.iloc[idx, 2]

        eye_tensor = torch.Tensor(
            [[float(self.csv_labels.iloc[idx, 3]), self.csv_labels.iloc[idx, 4], self.csv_labels.iloc[idx, 5]]])
        # .to(device)
        at_tensor = torch.Tensor(
            [[float(self.csv_labels.iloc[idx, 6]), self.csv_labels.iloc[idx, 7], self.csv_labels.iloc[idx, 8]]])
        # .to(device)


        ref_image_tensor = torch.from_numpy(1.0 / 255 * np.array([self.ref_images[ref_image_path]['image']], dtype='float32')).permute(0, 3, 1, 2)
        gt_image_tensor = torch.from_numpy(1.0 / 255 * np.array([self.gt_images[gt_image_path]['image']], dtype='float32')).permute(0, 3, 1, 2)

        # if (self.image_size != 224):
        r_size = transforms.Resize((self.image_size,self.image_size))
        ref_image_tensor = r_size(ref_image_tensor)

        r_size_gt = transforms.Resize((1024, 1024))
        gt_image_tensor = r_size_gt(gt_image_tensor)


        ref_vector = torch.reshape( torch.from_numpy(self.ref_images[ref_image_path]['vectors']['beta'].astype('float32')),(1,-1))
        # with Image.open(ref_image_path).resize((self.image_size, self.image_size)) as ref_image:
        #     ref_image_tensor = torch.from_numpy(1.0 / 255 * np.array([np.array(ref_image)], dtype='float32')).permute(0, 3,1, 2)
        # with Image.open(gt_image_path).resize((self.image_size, self.image_size)) as gt_image:
        #     gt_image_tensor = torch.from_numpy(1.0 / 255 * np.array([np.array(gt_image)], dtype='float32')).permute(0, 3, 1,2)


        # verts, faces, _ = load_obj(mesh_obj_path)
        # mesh = Meshes(verts=[verts.to(device)], faces=[faces.verts_idx.to(device)])

        verts = torch.from_numpy(self.mesh_objs[mesh_obj_name]['verts'].astype(np.float32))  # .to(device)
        faces = torch.from_numpy(self.mesh_objs[mesh_obj_name]['faces'])

        verts = torch.unsqueeze(verts, 0)
        faces = torch.unsqueeze(faces, 0)
        sample = {
            'verts': verts,
            'faces': faces,
            # 'meshes': mesh,
            'ref_image': ref_image_tensor,
            'gt_image': gt_image_tensor,
            'eye_tensor': eye_tensor,
            'at_tensor': at_tensor,
            'ref_vector':ref_vector
        }
        return sample


def collate_fn(data):
    # meshes = []
    # for sample in data:
    #     meshes.append(sample['meshes'])
    #     sample.pop('meshes')
    # meshes = join_meshes_as_batch(meshes)

    batch = {}
    # batch['meshes'] = meshes

    batch['verts'] = torch.cat([sample['verts'] for sample in data], 0)
    batch['faces'] = torch.cat([sample['faces'] for sample in data], 0)
    batch['ref_image'] = torch.cat([sample['ref_image'] for sample in data], 0)
    batch['gt_image'] = torch.cat([sample['gt_image'] for sample in data], 0)
    batch['eye_tensor'] = torch.cat([sample['eye_tensor'] for sample in data], 0)
    batch['at_tensor'] = torch.cat([sample['at_tensor'] for sample in data], 0)
    batch['ref_vector'] = torch.cat([sample['ref_vector'] for sample in data], 0)
    return batch
    # for key in batch[0].keys():
    # res = torch.stack(batch, 0, out=out)
    # res.

class GeoFaceFFHQDataset(Dataset):
    def __init__(self, dataset_path):

        self.csv_labels = pd.read_csv(os.path.join(dataset_path, 'index.csv'))
        header = ["mesh_obj_path", "ref_image_path", "gt_image_path", "eye_tensor0", "eye_tensor1", "eye_tensor2",
                  "original0", "original1", "original2"]
        for i in range(3, len(header)):
            self.csv_labels[header[i]] = self.csv_labels[header[i]].astype("float32")
        '''
            index.csv contents:
            line idx: mesh_path, ref_image_path, groundtruth_image_path, camera_vectors(6 dim,eyes_vector and at_vector), 
        '''
        self.ffhq_path = "FFHQ"
        # load all objs previously.
        self.mesh_objs = {}
        if (os.path.exists(os.path.join(dataset_path, 'mesh_data.pkl'))):
            with open(os.path.join(dataset_path, 'mesh_data.pkl'), 'rb') as pkl_file:
                self.mesh_objs = pickle.load(pkl_file)
        else:
            raise RuntimeError("No mesh_data.pkl file found.")

        # self.ref_images = {}
        # if (os.path.exists(os.path.join(dataset_path, 'ref_image_data.pkl'))):
        #     with open(os.path.join(dataset_path, 'ref_image_data.pkl'), 'rb') as pkl_file:
        #         self.ref_images = pickle.load(pkl_file)
        # else:
        #     raise RuntimeError("No ref_image_data.pkl file found.")
        #
        # self.gt_images = {}
        # if (os.path.exists(os.path.join(dataset_path, 'gt_image_data.pkl'))):
        #     with open(os.path.join(dataset_path, 'gt_image_data.pkl'), 'rb') as pkl_file:
        #         self.gt_images = pickle.load(pkl_file)
        # else:
        #     raise RuntimeError("No gt_image_data.pkl file found.")

        # for i in range(len(self.csv_labels)):
        #     mesh_obj_path = self.csv_labels.iloc[i, 0]
        #     if not (mesh_obj_path in self.mesh_objs):
        #         verts, faces, _ = load_obj(mesh_obj_path)
        #         mesh = Meshes(verts=[verts], faces=[faces.verts_idx])
        #         self.mesh_objs[mesh_obj_path] = mesh

    def __len__(self):
        return len(self.csv_labels)

    def __getitem__(self, idx):
        # mesh_obj_path = self.csv_labels.iloc[idx, 0]
        mesh_obj_name = self.csv_labels.iloc[idx, 0]
        # ref_image_path = self.csv_labels.iloc[idx, 1]
        # gt_image_path = self.csv_labels.iloc[idx, 2]

        ref_image_path = os.path.join(self.ffhq_path,str(idx+10000)+".png")
        real_image_path = os.path.join(self.ffhq_path, str(idx+10010)+".png")

        eye_tensor = torch.Tensor(
            [[float(self.csv_labels.iloc[idx, 3]), self.csv_labels.iloc[idx, 4], self.csv_labels.iloc[idx, 5]]])
        # .to(device)
        at_tensor = torch.Tensor(
            [[float(self.csv_labels.iloc[idx, 6]), self.csv_labels.iloc[idx, 7], self.csv_labels.iloc[idx, 8]]])
        # .to(device)

        with Image.open(ref_image_path).resize((256, 256)) as ref_image:
            ref_image_tensor = torch.from_numpy(1.0 / 255 * np.array([np.array(ref_image)], dtype='float32')).permute(0, 3,1, 2)
        with Image.open(real_image_path).resize((1024, 1024)) as real_image:
            real_image_tensor = torch.from_numpy(1.0 / 255 * np.array([np.array(real_image)], dtype='float32')).permute(0, 3, 1,2)
        # ref_image_tensor = torch.from_numpy(1.0 / 255 * np.array([self.ref_images[ref_image_path]['image']], dtype='float32')).permute(0, 3, 1, 2)
        # gt_image_tensor = torch.from_numpy(1.0 / 255 * np.array([self.gt_images[gt_image_path]['image']], dtype='float32')).permute(0, 3, 1, 2)

        # if (self.image_size != 224):
        # r_size = transforms.Resize((self.image_size,self.image_size))
        # ref_image_tensor = r_size(ref_image_tensor)
        #
        # r_size_gt = transforms.Resize((1024, 1024))
        # gt_image_tensor = r_size_gt(gt_image_tensor)


        # ref_vector = torch.reshape( torch.from_numpy(self.ref_images[ref_image_path]['vectors']['beta'].astype('float32')),(1,-1))



        # verts, faces, _ = load_obj(mesh_obj_path)
        # mesh = Meshes(verts=[verts.to(device)], faces=[faces.verts_idx.to(device)])

        verts = torch.from_numpy(self.mesh_objs[mesh_obj_name]['verts'].astype(np.float32))  # .to(device)
        faces = torch.from_numpy(self.mesh_objs[mesh_obj_name]['faces'])

        verts = torch.unsqueeze(verts, 0)
        faces = torch.unsqueeze(faces, 0)
        sample = {
            'verts': verts,
            'faces': faces,
            # 'meshes': mesh,
            'ref_image': ref_image_tensor,
            'real_image': real_image_tensor,
            'eye_tensor': eye_tensor,
            'at_tensor': at_tensor,
            # 'ref_vector':ref_vector
        }
        return sample


def collate_fn_ffhq(data):
    # meshes = []
    # for sample in data:
    #     meshes.append(sample['meshes'])
    #     sample.pop('meshes')
    # meshes = join_meshes_as_batch(meshes)

    batch = {}
    # batch['meshes'] = meshes

    batch['verts'] = torch.cat([sample['verts'] for sample in data], 0)
    batch['faces'] = torch.cat([sample['faces'] for sample in data], 0)
    batch['ref_image'] = torch.cat([sample['ref_image'] for sample in data], 0)
    batch['real_image'] = torch.cat([sample['real_image'] for sample in data], 0)
    batch['eye_tensor'] = torch.cat([sample['eye_tensor'] for sample in data], 0)
    batch['at_tensor'] = torch.cat([sample['at_tensor'] for sample in data], 0)
     # batch['ref_vector'] = torch.cat([sample['ref_vector'] for sample in data], 0)
    return batch

def get_dataloader_ffhq(dataset_path, imagesize, batch_size, num_workers):
    trainset = GeoFaceFFHQDataset(dataset_path)
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  collate_fn=collate_fn_ffhq)
    return train_dataloader

class GeoFaceFFHQDatasetV2(Dataset):
    def __init__(self,imagesize, dataset_path):
        self.image_size = imagesize
        # load all objs previously.
        self.mesh_objs = {}
        if (os.path.exists(os.path.join(dataset_path, 'mesh_data.pkl'))):
            with open(os.path.join(dataset_path, 'mesh_data.pkl'), 'rb') as pkl_file:
                self.mesh_objs = pickle.load(pkl_file)
        else:
            raise RuntimeError("No mesh_data.pkl file found.")

        self.images = {}
        if (os.path.exists(os.path.join(dataset_path, 'image_data.pkl'))):
            with open(os.path.join(dataset_path, 'image_data.pkl'), 'rb') as pkl_file:
                self.images = pickle.load(pkl_file)
        else:
            raise RuntimeError("No image_data.pkl file found.")

    def __len__(self):
        return len(self.mesh_objs)

    def __getitem__(self, idx):

        paired_idx = random.randint(0,len(self.mesh_objs)-1)

        eye_tensor = torch.Tensor(
            [[0, 0, 0.2]])
        # .to(device)
        at_tensor = torch.Tensor(
            [[0, 0, 0]])
        # .to(device)

        #source image and mesh
        # source_verts = torch.unsqueeze(torch.from_numpy(self.mesh_objs[idx]['verts'].astype(np.float32)),0)
        # source_faces = torch.unsqueeze(torch.from_numpy(self.mesh_objs[idx]['faces']),0)
        source_image = torch.from_numpy(1.0 / 255 * np.array([self.images[idx]], dtype='float32')).permute(0, 3,1, 2)

        # target_verts = torch.unsqueeze(torch.from_numpy(self.mesh_objs[paired_idx]['verts'].astype(np.float32)),0)
        # target_faces = torch.unsqueeze(torch.from_numpy(self.mesh_objs[paired_idx]['faces']),0)
        target_image = torch.from_numpy(1.0 / 255 * np.array([self.images[paired_idx]], dtype='float32')).permute(0, 3,1, 2)


        if (self.image_size != 1024):
            r_size = transforms.Resize((self.image_size,self.image_size))
            source_image = r_size(source_image)
            target_image = r_size(target_image)
        #
        # r_size_gt = transforms.Resize((1024, 1024))
        # gt_image_tensor = r_size_gt(gt_image_tensor)

        sample = {
            'source_mesh':self.mesh_objs[idx]['meshes'],
            # 'source_verts': source_verts,
            # 'source_faces': source_faces,
            'source_image': source_image,

            'target_mesh': self.mesh_objs[paired_idx]['meshes'],
            # 'target_verts': target_verts,
            # 'target_faces': target_faces,
            'target_image': target_image,

            'eye_tensor': eye_tensor,
            'at_tensor': at_tensor,
        }
        return sample


def collate_fn_ffhqv2(data):
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

    batch['target_meshes'] = target_meshes
    # batch['target_verts'] = torch.cat([sample['target_verts'] for sample in data], 0)
    # batch['target_faces'] = torch.cat([sample['target_faces'] for sample in data], 0)
    batch['target_image'] = torch.cat([sample['target_image'] for sample in data], 0)

    batch['eye_tensor'] = torch.cat([sample['eye_tensor'] for sample in data], 0)
    batch['at_tensor'] = torch.cat([sample['at_tensor'] for sample in data], 0)
     # batch['ref_vector'] = torch.cat([sample['ref_vector'] for sample in data], 0)
    return batch

def get_dataloader_ffhqv2(dataset_path,image_size, batch_size, num_workers):
    trainset = GeoFaceFFHQDatasetV2(image_size,dataset_path)
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                  collate_fn=collate_fn_ffhqv2)
    return train_dataloader

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

        #normalized to -1 1
        #source_image = source_image*2-1
        #target_image = target_image*2-1
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
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                  collate_fn=collate_fn_ffhqv3)
    return train_dataloader

def create_dataloader_splitted_v3(dataset_path,image_size, batch_size,num_workers,test_len):
    training_data = GeoFaceFFHQDatasetV3(image_size,dataset_path)

    # train_len = int(len(training_data) * 0.8)
    # test_len = len(training_data) - train_len

    train_len = len(training_data) - test_len
    trainset, testset = random_split(training_data, [train_len, test_len], generator=torch.Generator().manual_seed(42))
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  collate_fn=collate_fn_ffhqv3)
    test_dataloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=num_workers,
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
