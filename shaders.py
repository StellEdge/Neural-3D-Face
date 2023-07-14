from typing import Optional

import torch
from pytorch3d.renderer.blending import BlendParams, softmax_rgb_blend,hard_rgb_blend
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.rasterizer import Fragments

class myShader(torch.nn.Module):

    def __init__(
        self,
        device="cpu",
        blend_params: Optional[BlendParams] = None,
    ):
        super().__init__()
        #self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        #cameras = kwargs.get("cameras", self.cameras)
        # if cameras is None:
        #     msg = "Cameras must be specified either at initialization \
        #         or in the forward pass of TexturedSoftPhongShader"
        #     raise ValueError(msg)
        # get renderer output
        #blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)

        is_background = fragments.pix_to_face[..., 0] < 0  # (N, H, W)
        background_color = torch.zeros(texels.shape[-1]).to('cuda')

        # Find out how much background_color needs to be expanded to be used for masked_scatter.
        num_background_pixels = is_background.sum()

        # Set background color.
        pixel_colors = texels[..., 0, :].masked_scatter(
            is_background[..., None],
            background_color[None, :].expand(num_background_pixels, -1),
        )  # (N, H, W, 3)


        #images = hard_rgb_blend(texels, fragments, blend_params)
        #images = texels[:,:,:,0,:].squeeze(-2)
        #print('images min ', torch.min(pixel_colors))
        return pixel_colors