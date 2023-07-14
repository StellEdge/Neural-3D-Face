import math
import random
import torch
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

from pytorch3d.transforms import (
    axis_angle_to_matrix,
    euler_angles_to_matrix
)

from shaders import myShader
from pytorch3d.renderer.blending import BlendParams

from matplotlib import pyplot as plt

def generate_random_eyes(camera_num, original_vectors, distance, distance_random_range=0,x_rot_range = math.pi / 12, y_rot_range =math.pi / 12, debug=False):
    """
    Generate random positions for camera
    :param camera_num:
    :param original_vectors:
    :param distance:
    :param distance_random_range:
    :param debug:
    :return:
    """
    # print(timelog_str()+" Generate cameras: eye tensors.")

    # original_vectors = original.to(device)
    # original_vectors = torch.unsqueeze(original_vectors,0)
    # original_vectors = original_vectors.repeat(camera_num,1)

    base_dist_z = torch.ones([camera_num], device=device) * distance
    if not distance_random_range == 0:
        base_dist_z = base_dist_z + (torch.rand([camera_num], device=device) - 0.5) * distance_random_range

    zero_vectors = torch.zeros([camera_num], device=device)
    base_vector = torch.vstack([zero_vectors, zero_vectors, base_dist_z])
    # rand_dist_diff_y = torch.zeros([camera_num],device= device)
    # rand_dist_diff_z = torch.zeros([camera_num],device= device)
    # base_vector = torch.vstack([base_dist_x,rand_dist_diff_y,rand_dist_diff_z])
    base_vector = torch.transpose(base_vector, 0, 1)

    # pitch rot
    # x_axis_rot_angles = torch.zeros([camera_num],device= device)
    # z_axis_rot_angles = (torch.rand([camera_num]) - 0.3) * (-math.pi / 2)
    x_axis_rot_angles = torch.clamp(torch.normal(mean=zero_vectors, std=0.1) * x_rot_range,-x_rot_range,x_rot_range)
    # y_axis_rot_angles = torch.zeros([camera_num],device= device)
    # rot_angles_tensor = torch.vstack([x_axis_rot_angles,y_axis_rot_angles,z_axis_rot_angles])

    rot_angles_tensor = torch.vstack([x_axis_rot_angles, zero_vectors, zero_vectors])

    rot_angles_tensor = torch.transpose(rot_angles_tensor, 0, 1)

    rot_mat_tensor_1 = axis_angle_to_matrix(rot_angles_tensor)
    # print(rot_mat_tensor_1.shape)

    # yaw rotation
    # x_axis_rot_angles = torch.zeros([camera_num],device= device)
    # z_axis_rot_angles = torch.zeros([camera_num],device= device)
    y_axis_rot_angles = torch.clamp(torch.normal(mean=zero_vectors, std=0.1) * y_rot_range,-y_rot_range,y_rot_range)
    # rot_angles_tensor = torch.vstack([x_axis_rot_angles,y_axis_rot_angles,z_axis_rot_angles])

    rot_angles_tensor = torch.vstack([zero_vectors, y_axis_rot_angles, zero_vectors])
    rot_angles_tensor = torch.transpose(rot_angles_tensor, 0, 1)

    rot_mat_tensor_2 = axis_angle_to_matrix(rot_angles_tensor)
    # print(rot_mat_tensor_2.shape)

    base_vector = torch.unsqueeze(base_vector, -1)
    out_vector = torch.bmm(rot_mat_tensor_1, base_vector)
    # out_vector = torch.unsqueeze(out_vector,-1)
    out_vector = torch.bmm(rot_mat_tensor_2, out_vector)
    out_vector = torch.squeeze(out_vector, -1)

    out_vector = out_vector + original_vectors
    if debug:

        print(out_vector)
        out_vector = torch.transpose(out_vector, 0, 1)
        fig = plt.figure()

        ax = plt.axes(projection='3d')
        # ax.set_xlim3d(-0.5, 0.5)
        # ax.set_ylim3d(-0.5, 0.5)
        # ax.set_zlim3d(original[-1]-0.5, original[-1]+0.5)

        # Coordinates fixed to be correspond to Pytorch 3D coordinates.
        xdata = out_vector[0].cpu().numpy()
        zdata = out_vector[1].cpu().numpy()
        ydata = out_vector[2].cpu().numpy()

        ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        plt.savefig("3dTEST.JPG")
    else:
        # print(timelog_str() + " eye_tensors ready.")
        return out_vector



def generate_random_ats(camera_num, original):
    at_vectors = torch.unsqueeze(original, 0)
    at_vectors = at_vectors.repeat(camera_num, 1)
    return at_vectors

def generate_front_eyes(camera_num, original_vectors, distance):
    dist_vectors = torch.tensor([0, 0, distance]).to(device)
    dist_vectors = torch.unsqueeze(dist_vectors, 0)
    dist_vectors = dist_vectors.repeat(camera_num, 1)
    return original_vectors + dist_vectors


def generate_y_rot_eyes(camera_num, original_vectors, distance, y_axis_rot_euler):
    y_axis_rot = y_axis_rot_euler/360 * 2 * math.pi
    dist_vectors = torch.tensor([distance * math.sin(y_axis_rot), 0, distance * math.cos(y_axis_rot)]).to(device)
    dist_vectors = torch.unsqueeze(dist_vectors, 0)
    dist_vectors = dist_vectors.repeat(camera_num, 1)
    return original_vectors + dist_vectors

def create_renderer(eye_tensor, at_tensor, image_size , device):

    R, T = look_at_view_transform(eye=eye_tensor, at=at_tensor)
    cameras = FoVPerspectiveCameras(device=device, fov=12, znear=0.01, R=R, T=T)

    # sigma = 1e-4
    raster_settings_soft = RasterizationSettings(
        image_size=image_size,
        # blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Place a point light in front of the object. As mentioned above, the front of the cow is facing the
    # -z direction.
    # lights = PointLights(device=device, location=[[0.0, 1.75, -3.0]])

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings_soft
        ),
        shader=myShader(
            device=device
        )
    )
    return renderer

def create_phong_renderer(eye_tensor, at_tensor, image_size,device):
    R, T = look_at_view_transform(eye=eye_tensor, at=at_tensor)
    cameras = FoVPerspectiveCameras(device=device, fov=12, znear=0.01, R=R, T=T)

    # sigma = 1e-4
    raster_settings_soft = RasterizationSettings(
        image_size=image_size,
        # blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Place a point light in front of the object. As mentioned above, the front of the cow is facing the
    # -z direction.
    lights = PointLights(device=device, location=[[0.0, 1.75, 4.0]])

    bg = BlendParams(1e-4,1e-4,(0,0,0))

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings_soft
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            blend_params=bg,
        )
    )
    return renderer


# def create_renderer(R, T, image_size=512):
#
#     R, T = look_at_view_transform(eye=eye_tensor, at=at_tensor)
#     cameras = FoVPerspectiveCameras(device=device, fov=45, znear=0.1, R=R, T=T)
#
#     # sigma = 1e-4
#     raster_settings_soft = RasterizationSettings(
#         image_size=image_size,
#         # blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
#         blur_radius=0.0,
#         faces_per_pixel=1,
#     )
#
#     # Place a point light in front of the object. As mentioned above, the front of the cow is facing the
#     # -z direction.
#     # lights = PointLights(device=device, location=[[0.0, 1.75, -3.0]])
#
#     renderer = MeshRenderer(
#         rasterizer=MeshRasterizer(
#             cameras=cameras,
#             raster_settings=raster_settings_soft
#         ),
#         shader=myShader(
#             device=device
#         )
#     )
#     return renderer
#
# def create_phong_renderer(eye_tensor, at_tensor, image_size=512):
#     R, T = look_at_view_transform(eye=eye_tensor, at=at_tensor)
#     cameras = FoVPerspectiveCameras(device=device, fov=45, znear=0.1, R=R, T=T)
#
#     # sigma = 1e-4
#     raster_settings_soft = RasterizationSettings(
#         image_size=image_size,
#         # blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
#         blur_radius=0.0,
#         faces_per_pixel=1,
#     )
#
#     # Place a point light in front of the object. As mentioned above, the front of the cow is facing the
#     # -z direction.
#     lights = PointLights(device=device, location=[[0.0, 1.75, 4.0]])
#
#     bg = BlendParams(1e-4,1e-4,(0,0,0))
#
#     renderer = MeshRenderer(
#         rasterizer=MeshRasterizer(
#             cameras=cameras,
#             raster_settings=raster_settings_soft
#         ),
#         shader=SoftPhongShader(
#             device=device,
#             cameras=cameras,
#             lights=lights,
#             blend_params=bg,
#         )
#     )
#     return renderer

