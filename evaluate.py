from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
import numpy as np
import torch
import piq
from sewar.full_ref import uqi, vifp

def psnr(source, ir, integral ,  fused):
     return (     piq.psnr( torch.from_numpy(np.atleast_3d(ir) / 255).permute(2,0,1).unsqueeze(0), torch.from_numpy(fused / 255).permute(2,0,1).unsqueeze(0)) \
                + piq.psnr( torch.from_numpy(source / 255).permute(2,0,1).unsqueeze(0), torch.from_numpy(fused / 255).permute(2,0,1).unsqueeze(0)) \
                + piq.psnr( torch.from_numpy(integral / 255).permute(2,0,1).unsqueeze(0), torch.from_numpy(fused / 255).permute(2,0,1).unsqueeze(0))) / 3

def mutual_information(image1, image2):
    # Convert images to grayscale if needed
    # if len(image1.shape) > 2:
    #     image1 = np.mean(image1, axis=2)
    # if len(image2.shape) > 2:
    #     image2 = np.mean(image2, axis=2)

    # Calculate histogram
    hist_2d, x_edges, y_edges = np.histogram2d(
        image1.ravel(),
        image2.ravel(),
        bins=20  # You can adjust the number of bins for accuracy
    )

    # Calculate probabilities
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)  # Marginal for x over y
    py = np.sum(pxy, axis=0)  # Marginal for y over x
    px_py = px[:, None] * py[None, :]

    # Avoid log(0) by adding a small epsilon
    eps = np.finfo(float).eps

    # Calculate mutual information
    mi = np.sum(pxy * np.log((pxy + eps) / (px_py + eps)))
    return mi


#CMT
# image_path = r'D:\Research\Image Fusion\Codes\Survey\CMTFusion\fusion_outputs\4.png'

#MetaFusion
# image_path = r'D:\Research\Image Fusion\Codes\Survey\MetaFusion\results\6.jpg'

# COCO
# image_path = r'D:\Research\Image Fusion\Codes\Survey\CoCoNet-main\results\6.bmp'

# Swin
# image_path = r'D:\Research\Image Fusion\Codes\Survey\SwinFusion\results\SwinFusion_MSRS\6.png'

# alpha
image_path = r'results\alpha_results\5.png'

# Old Paper
# image_path = r'D:\Research\Image Fusion\Codes\Fusion-Pytorch\results\F10.png'

# Ours
# image_path = r'results\Fusion\5.png'

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (512,512))


source_path = r'data\Evaluate\RGB\5.jpg'
ir_path = r'data\Evaluate\Thermal_Integral\5.png'
in_path = r'data\Evaluate\RGB_Integral\5.png'


source = cv2.imread(source_path)
source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
source = cv2.resize(source, (512,512))

integral = cv2.imread(in_path)
integral = cv2.cvtColor(integral, cv2.COLOR_BGR2RGB)

ir = cv2.imread(ir_path)
ir = cv2.cvtColor(ir, cv2.COLOR_BGR2RGB)


####################################

print((vifp( (np.atleast_3d(ir)), (image))
                + vifp( (source), (image))
                + vifp( (integral), (image))
                ))

print(mutual_information(ir, image) + mutual_information(source, image)+ mutual_information(integral, image))

print(psnr(source, ir, integral, image))
