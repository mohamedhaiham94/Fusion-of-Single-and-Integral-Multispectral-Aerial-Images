from models import ColorFusion
from imageio.v2 import imread
import cv2
import numpy as np
import os
import time
from models.utils import color_space_conversion

def test_color(data):
    start = time.time()
    print('--Running--')
    image_path = f'data/Thermal_Integral/{data[0]}.png'
    img = cv2.imread(image_path)

    alpha = 2 
    beta = 0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    color_space_conversion(adjusted)

    image_files = [f for f in os.listdir('results/color_map/') if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'))]
    print(image_files)

    for image in data:
        vis = imread('data/Evaluate/RGB/'+image+'.jpg')
        ir = imread('data/Thermal_Integral/'+image+'.png')
        ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)

        vis = cv2.resize(vis, (512,512))
        vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

        for img in image_files:
            print(img)
            rgb_thermal = imread('results/color_map/'+img)
            rgb_thermal = cv2.cvtColor(rgb_thermal, cv2.COLOR_BGR2RGB)
            rgb_thermal = cv2.resize(rgb_thermal, (512,512))
            
            adjusted = cv2.resize(adjusted, (512,512))

            model = ColorFusion(vis, adjusted, rgb_thermal, img, image)
            image = '70'
            cv2.imwrite('results/color_map/fusion/'+image+'/'+img, (model.getFusion() * 255).astype(np.int32))
    print('--Finished--')

if __name__=="__main__":
  data = ['70']
  test_color(data)
