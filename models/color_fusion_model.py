import numpy as np
from sporco.signal import tikhonov_filter
import torch
from torchvision.models.vgg import vgg19
from torchvision.models import VGG19_Weights
import cv2
import matplotlib.pyplot as plt
from .utils import *

class ColorFusion:
    def __init__(self, vis, ir, color_map_image, color_map_method,  Image):

        # Visible image, thermal image, integral image, color_map image, name of color map image, name of image in original data ex: F5, F10
        self._output = self.fuse_best_color_code(vis, ir, color_map_image, color_map_method, Image)

       
    def getFusion(self):
        '''
            This method just return the fused image
        '''
        return self._output
    
    def fuse_best_color_code(self,vis, ir, color_image,name,ImageName,  model = None):
        '''
            This is the main method that control all the method where the fusion happens.
            Parameter:
                vis(image): Single RGB image
                ir(image): Themral Integral image
                model(DL model): VGG in our case
            
            Return:
                Final Fused image with color code
        '''

        ir_image = ir.astype(np.float32)/255
        _, ir_high = lowpass(ir_image)
        pathDir = 'results/'

        # cv2.imwrite(pathDir+ImageName+'/ir_low.png', (ir_low * 255).astype(np.int32))
        # cv2.imwrite(pathDir+ImageName+'/ir_high.png', (ir_high * 255).astype(np.int32))  
        # cv2.imwrite(pathDir+ImageName+'/in_low.png', (in_low * 255).astype(np.int32))
        # cv2.imwrite(pathDir+ImageName+'/in_high.png', (in_high * 255).astype(np.int32)) 
        if model is None:
            model = vgg19(weights=VGG19_Weights.DEFAULT)
        model.cuda().eval()

        relus = [2, 7]
        unit_relus = [1, 2]
        
        ir_in = torch.from_numpy(c3(ir_image)).cuda()

        relus_ir = get_activation(model, relus, ir_in, ImageName, 'Thermal')
        
        ir_feats = [l1_features(out) for out in relus_ir]
        
        check = True
        feature_summation = None
        for idx in range(len(relus)):
            all_features_combined_thermal = fusion_strategy_New(
                    ir_feats[idx], ir_high, unit_relus[idx]
                    )

            if check:
                feature_summation = all_features_combined_thermal
                check = False
            else:
                feature_summation += all_features_combined_thermal
        all_features_combined_thermal = feature_summation

        # all_features_combined_thermal = (all_features_combined_thermal - all_features_combined_thermal.min()) / (all_features_combined_thermal.max() - all_features_combined_thermal.min())
        
        # cv2.imwrite(ImageName+'.png', (((np.atleast_3d(all_features_combined_thermal) * (color_image.astype(np.float32) / 255))) * 255).astype(np.int32)) 

        return  np.atleast_3d(vis.astype(np.float32) / 255) + (np.atleast_3d(all_features_combined_thermal) * (color_image.astype(np.float32) / 255)) 