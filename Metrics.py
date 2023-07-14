import lpips
from skimage.metrics import structural_similarity as ssim


#loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# lpips
loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores

def cal_similarities(img1,img2,verbose = False):

    img1_np, img2_np = np.array(img1), np.array(img2)
    #ssim
    img1_gray = img1.convert('L')
    img2_gray = img2.convert('L')
    img1_gray, img2_gray = np.array(img1_gray), np.array(img2_gray)
    # 此处因为转换为灰度值之后的图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
    ssim_score = ssim(img1_gray, img2_gray, data_range=255)

    img1_tensor = torch.from_numpy(2.0*(img1_np/255.0)-1.0).unsqueeze(0).permute(0,3,1,2)  # image should be RGB, IMPORTANT: normalized to [-1,1]

    img2_tensor = torch.from_numpy(2.0*(img2_np/255)-1.0).unsqueeze(0).permute(0,3,1,2)
    lpips_score = loss_fn_alex(img1_tensor.float(), img2_tensor.float()).item()

    loss_l1 = nn.L1Loss()
    img1_tensor = torch.from_numpy(img1_np/255.0).unsqueeze(0).permute(0,3,1,2).float()  # image should be RGB, IMPORTANT: normalized to [-1,1]
    img2_tensor = torch.from_numpy(img2_np/255.0).unsqueeze(0).permute(0,3,1,2).float()
    L1_score = loss_l1(img1_tensor,img2_tensor).item()
    loss_mse = nn.MSELoss()
    MSE_score = loss_mse(img1_tensor, img2_tensor).item()
    #fid
    if(verbose):
        print('L1 score: ', str(L1_score))
        print('MSE score: ', str(MSE_score))
        print('SSIM score: ', str(ssim_score))
        print('LPIPS score: ', str(lpips_score))
    return {'l1':L1_score,'mse':MSE_score,'ssim':ssim_score,'lpips':lpips_score}


def cal_similarities_multiple_paths(paths_a,paths_b):
    ssim_list = []
    lpips_list = []
    for a,b in zip(paths_a,paths_b):
        img1 = Image.open(a)
        img2 = Image.open(b)
        ssim_score,lpips_score = cal_similarities(img1,img2)
        ssim_list.append(ssim_score)
        lpips_list.append(lpips_score)

    ssim_list = np.array(ssim_list)
    lpips_list = np.array(lpips_list)

    print("SSIM: ",ssim_list.mean(),'+-',ssim_list.std())
    print("LPIPS: ", lpips_list.mean(), '+-', lpips_list.std())

# def build_paths(img_num):
#     paths_a = []
#     paths_b = []
#     for i in range(img_num):
#         paths_a.append("50000_groundtruth" +str(i)+ ".jpg")
#         paths_b.append("50000_afterUNet_otheraspects" + str(i) + ".jpg")
#     return paths_a,paths_b

# pa,pb =build_paths(10)
# cal_similarities_multiple(pa,pb)