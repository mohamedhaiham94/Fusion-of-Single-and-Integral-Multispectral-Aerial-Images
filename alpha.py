from imageio.v2 import imread
import cv2
import numpy as np
from models.utils import alpha_blending

def test_alpha(data):   
   print('--Running--')
   for image in data:
      vis = imread('data/Evaluate/RGB/'+image+'.jpg')
      thermal = imread('data/Evaluate/Thermal_Integral/'+image+'.png')
      Integral = imread('data/Evaluate/RGB_Integral/'+image+'.png')
      vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
      # thermal = cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)

      blended = alpha_blending(vis, thermal, Integral)
      try:
         cv2.imwrite('results/alpha_results/'+image+'.png', blended)
      except:
         print('Folder Already Exists')
      print(image)
   print('--Finished--')


if __name__=="__main__":
   data = ['2']
   test_alpha(data)