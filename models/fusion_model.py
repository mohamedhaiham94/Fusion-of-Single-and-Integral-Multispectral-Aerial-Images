import numpy as np
from sporco.signal import tikhonov_filter
import torch
from torchvision.models.vgg import vgg19
from torchvision.models.resnet import resnet50
from torchvision.models import VGG19_Weights
import cv2
import matplotlib.pyplot as plt
import time
from .utils import *
import torch.nn as nn


class Fusion(nn.Module):
    def __init__(self, vis, ir, Integral,  Image):
        super().__init__()

        #First
        # self._output = self.fuse_no_vgg(vis, ir,Integral, Image)

        #Second
        # self._output = self.fuse_new(vis, ir,Integral, Image)
        self.outputs = {}

        self.net = resnet50(pretrained = True).eval().cuda()
        self._output = self.fuse_best(vis, ir,Integral, Image)
        
    def getFusion(self):
        '''
            This method just return the fused image
        '''
        return self._output
    def activation(self, name):
        def hook(module, input, output):
            self.outputs[name] = output.detach()
        return hook
    
    def get_activation(self):
        return self.outputs
    
    def forward(self, x):
        return self.net(x)
    
    def fuse_best(self,vis, ir, Integral,ImageName, model = None):
        '''
            This is the main method that control all the method where the fusion happens.
            Parameter:
                vis(image): Single RGB image
                ir(image): Themral Integral image
                Integral(image): Integral RGB image
                model(DL model): VGG in our case
            
            Return:
                Final Fused image with color code
        '''
        Gray_Integral = cv2.cvtColor(Integral, cv2.COLOR_BGR2GRAY)

        ir_image = ir.astype(np.float32)/255

        integral_image = Gray_Integral.astype(np.float32)/255

        ir_low, ir_high = lowpass(ir_image)
        in_low, in_high = lowpass(integral_image)
        pathDir = 'results/'

        # cv2.imwrite(pathDir+ImageName+'/ir_low.png', (ir_low * 255).astype(np.int32))
        # cv2.imwrite(pathDir+ImageName+'/ir_high.png', (ir_high * 255).astype(np.int32))  
        # cv2.imwrite(pathDir+ImageName+'/in_low.png', (in_low * 255).astype(np.int32))
        # cv2.imwrite(pathDir+ImageName+'/in_high.png', (in_high * 255).astype(np.int32)) 
        
        in_in = torch.from_numpy(c3(integral_image)).cuda()
        ir_in = torch.from_numpy(c3(ir_image)).cuda()
        
        if model is None:
            model = vgg19(weights=VGG19_Weights.DEFAULT)
        else:
    
            layer1 = self.net.layer1.register_forward_hook(self.activation('layer1'))
            layer2 = self.net.layer2.register_forward_hook(self.activation('layer2'))            
            vis_ = self.net(ir_in)

            activation_visible = self.get_activation()
            # visualizeFeatures(activation_visible, 'Mona', 'd')
            
        model.cuda().eval()

        relus = [2, 7]
        unit_relus = [1, 2]


        start = time.time()
        relus_in = get_activation(model, relus, in_in, ImageName, 'IRGB')
        in_feats = [l1_features(out) for out in relus_in]
        check = True
        feature_summation = None
        for idx in range(len(relus)):
            all_features_combined_integral= fusion_strategy_New(
                    in_feats[idx], in_high, unit_relus[idx]
                    )
            if check:
                feature_summation = all_features_combined_integral
                check = False
            else:
                feature_summation += all_features_combined_integral

        all_features_combined_integral = feature_summation
        end = time.time()
        print(f"Integral time {end - start}")

        start = time.time()

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
        end = time.time()
        print(f"IR time {end - start}")


        all_features_combined_thermal = (all_features_combined_thermal - all_features_combined_thermal.min()) / (all_features_combined_thermal.max() - all_features_combined_thermal.min())
        all_features_combined_integral= (all_features_combined_integral- all_features_combined_integral.min()) / (all_features_combined_integral.max() - all_features_combined_integral.min())
        start = time.time()

        x = np.atleast_3d(vis.astype(np.float32) / 255) + np.atleast_3d((all_features_combined_thermal * ir_image) + ir_high)    
        end = time.time()
        print(f"IR time {end - start}")
        return  (np.atleast_3d(all_features_combined_integral) * (Integral.astype(np.float32) / 255)) + np.atleast_3d(vis.astype(np.float32) / 255) + np.atleast_3d((all_features_combined_thermal * ir_image) + ir_high)    


    def fuse_no_vgg(self,vis, ir, Integral, ImageName, model = None):
        '''
            This is the main method that control all the method where the fusion happens.

            Parameter:
                vis(image): Single RGB image
                ir(image): Themral Integral image
                Integral(image): Integral RGB image
                model(DL model): VGG in our case
            
            Return:
                Final Fused image
        '''
        _, ir_high = lowpass(ir.astype(np.float32)/255)
        _, in_high = lowpass(Integral.astype(np.float32)/255)

        ir_high = (ir_high - ir_high.min()) / (ir_high.max() - ir_high.min())
        in_high = (in_high - in_high.min()) / (in_high.max() - in_high.min())

        return  (vis.astype(np.float32) / 255)  + ((in_high * Integral.astype(np.float32)/255)) + np.atleast_3d((( ir_high * ir.astype(np.float32)/255)))


  
    def fuse_new(self,vis, ir, Integral, ImageName, model = None):
        '''
            This is the main method that control all the method where the fusion happens.
            Parameter:
                vis(image): Single RGB image
                ir(image): Themral Integral image
                Integral(image): Integral RGB image
                model(DL model): VGG in our case
            
            Return:
                Final Fused image
        '''

        pathDir = 'results/'
        Gray_Integral = cv2.cvtColor(Integral, cv2.COLOR_BGR2GRAY)
        integral_image = Gray_Integral.astype(np.float32)/255

        if model is None:
            model = vgg19(weights=VGG19_Weights.DEFAULT)
        model.cuda().eval()
        relus = [2, 7]
        unit_relus = [1, 2]

        
        ir_in = torch.from_numpy(self.c3(ir.astype(np.float32)/255)).cuda()
        in_in = torch.from_numpy(self.c3(integral_image)).cuda()

        # model, layer_numbers, input_image, ImageName, ttype
        relus_ir = get_activation(model, relus_ir, ir_in, ImageName, 'Thermal')
        relus_in = get_activation(model, relus_in, in_in, ImageName, 'IRGB')
        
        ir_feats = [self.l1_features(out) for out in relus_ir]
        in_feats = [self.l1_features(out) for out in relus_in]
        
        check = True
        feature_summation = None
        for idx in range(len(relus)):
            all_features_combined_thermal = fusion_strategy_New(
                    ir_feats[idx], ir.astype(np.float32)/255, unit_relus[idx]
                    )
            if check:
                feature_summation = all_features_combined_thermal
                check = False
            else:
                feature_summation += all_features_combined_thermal
        all_features_combined_thermal = feature_summation

        
        check = True
        feature_summation = None
        for idx in range(len(relus)):
            all_features_combined_integral = fusion_strategy_New(
                    in_feats[idx], Integral.astype(np.float32)/255, unit_relus[idx]
                    )
            if check:
                feature_summation = all_features_combined_integral
                check = False
            else:
                feature_summation += all_features_combined_integral
        all_features_combined_integral= feature_summation

        all_features_combined_thermal = (all_features_combined_thermal - all_features_combined_thermal.min()) / (all_features_combined_thermal.max() - all_features_combined_thermal.min())
        all_features_combined_integral= (all_features_combined_integral- all_features_combined_integral.min()) / (all_features_combined_integral.max() - all_features_combined_integral.min())


        return  (vis.astype(np.float32) / 255)  + (np.atleast_3d(all_features_combined_integral) * Integral.astype(np.float32) / 255) + np.atleast_3d((( all_features_combined_thermal * ir.astype(np.float32)/255)))