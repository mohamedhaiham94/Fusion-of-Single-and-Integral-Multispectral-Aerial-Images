import numpy as np
from sporco.signal import tikhonov_filter
import torch
import cv2
import matplotlib.pyplot as plt

def lowpass(s, lda = 5, npad = 16):
    '''
        This method is taking image as input and apply unified filter then return base and detail part of the input image

        Parameter:
            s(image): input image
            lda(int): Regularization parameter controlling unified filtering.
            npad(int): Number of samples to pad at image boundaries.
        
        Return:
            two images base and detail
    '''
    return tikhonov_filter(s, lda, npad)

def c3(s):
    '''
        This image takes image as input and check if it has less than 3 channels then it stack another channel to make it 3
        and return the new image with rollaxis its axis like permute method

        Parameter:
            s(image) : input image
        
        Return:
            permuted image to be used in the next step
    '''
    if s.ndim == 2:
        s3 = np.dstack([s, s, s])
    else:
        s3 = s
    return np.rollaxis(s3, 2, 0)[None, :, :, :]

def l1_features(out):
    '''
        This method computes the l1 features from a matrix.
        for example the input matrix is 512x512x64 the method will calculate the l1 feature along with axis 2
        and return a new matrix of the following size 512x512

        Parameter:
            out(matrix): feature matrix
        
        Return:
            The method returns the l1 features
    '''
    h, w, _ = out.shape
    A_temp = np.zeros((h, w))
    l1_norm = np.sum(np.abs(out), axis=2)     
    A_temp[:h, :w] = l1_norm

    A_temp = (A_temp - A_temp.min()) / (A_temp.max() - A_temp.min())

    return A_temp

def visualizeFeatures(features, ImageName, ttype):
    '''
        This method visualize the feature map extracted from VGG

        Parameter:
            features(list) : list of features from fiddrenet layers
            ImageName(str): name of the dataset folder 
            ttype(str): define the type of the image wheter it is Single RGB, Thermal, or Integral RGB
    '''
    processed = []
    for feature_map in features:
        # feature_map = features[feature_map].squeeze(0)
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map,0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())
    pathDir = 'results/'
    
    fig = plt.figure(figsize=(30, 50))
    for i in range(len(processed)):
        a = fig.add_subplot(5, 4, i+1)
        plt.imshow(processed[i])
        a.axis("off")
    plt.savefig(str(pathDir+'/feature_maps_'+ttype+'.jpg'), bbox_inches='tight')

def get_activation(model, layer_numbers, input_image, ImageName, ttype):
    '''
        This method extract the features from the desired VGG layers.

        Parameter:
            model(DL model): in this case will be VGG model
            layer_number(list): list contains number of desired layers
            input_image(image): image feeds as input to the VGG
            ImageName(str): name of the dataset folder
            ttype(str): define the type of the image wheter it is Single RGB, Thermal, or Integral RGB
            Return:
                The method will return the features from each layer.
    '''
    outs = []
    feats = []
    out = input_image
    maxi_ = max(layer_numbers)
    for i in range(maxi_+1):
        with torch.no_grad():
            out = model.features[i](out)
        if i in layer_numbers:
            outs.append(np.rollaxis(out.detach().cpu().numpy()[0], 0, 3))
            feats.append(out.detach().cpu())
    # visualizeFeatures(feats, ImageName, 'vgg')
    return outs

def fusion_strategy_New(feat_a, source_a, unit):
    '''
        This method is resposible for the fusion approach by calculatring the weighted matrix extracted from softmax 
        then it multplied the weighted matrix with the source image.

        Parameter:
            feat_a (matrix): features extracted from the details Single RGB image.
            source_a (image): Input image.
            unit (int): upscaling parameter.
        Return:
            Fused Image
    '''
    m1, n1 = source_a.shape[:2]
    m, n = feat_a.shape

    weight_ave_temp1 = np.zeros((m1, n1))
    for i in range(1, m): 
        for j in range(1, n):
            A1 = feat_a[i-1:i+2, j-1:j+2].sum() / 9
            weight_ave_temp1[(i-2)*unit+1:(i-1)*unit+1, (j-2)*unit+1:(j-1)*unit+1] = A1


    weight_ave_temp1 = (weight_ave_temp1 - weight_ave_temp1.min()) / (weight_ave_temp1.max() - weight_ave_temp1.min())

    source_a = (source_a - source_a.min()) / (source_a.max() - source_a.min())

    if source_a.ndim == 3:
        weight_ave_temp1 = weight_ave_temp1[:, :, None]

    source_a_fuse = ((source_a) * (weight_ave_temp1))
    
    if source_a.ndim == 3:
        gen = np.atleast_3d(source_a_fuse)
    else:
        gen = source_a_fuse
    
    return gen

def alpha_blending(vis, thermal, integral):
    '''
        This method takes 3 images and merge them using alpha-blending.
        Parameter:
            vis(image): Single RGB image
            thermal(image): Themral integral image
            integral(image): Integral RGB image        
        Return:
            Final Fused image
    '''
    vis = cv2.resize(vis, (512,512))
    integral = cv2.resize(integral, (512,512))
    if len(thermal.shape) == 3 and thermal.shape[-1] > 3:
        thermal = thermal[:,:,:3]
    else:
        thermal = np.dstack([thermal, thermal, thermal])
    
    return (.333*thermal + .333 * integral + .333 * vis)

def color_space_conversion(thermal):
    '''
        This method generate color space for any input image but to be more specific thermal integral image.
        Parameter:
            thermal(image): Themral integral image
    '''
    cmap = ['autumn', 'jet']
    fig = plt.figure(figsize=(15, 10)) 
    i = 1
    for idx, color_map in enumerate(cmap):
        print(color_map)
        fig.add_subplot(3, 5, idx+1) 
        plt.imshow(thermal, cmap=color_map)
        plt.imsave("results/color_map/"+color_map+".png", thermal, cmap=color_map)
        plt.colorbar(label='Temperature')  # Add a colorbar with temperature scale
        plt.title(color_map)
        # plt.savefig('results/color_map/colored_thermal_image.png')
