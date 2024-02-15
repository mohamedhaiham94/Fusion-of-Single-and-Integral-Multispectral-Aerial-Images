from models import Fusion
from imageio.v2 import imread
import cv2
import numpy as np
import time

def test_fusion(data):
  start = time.time()
  print('--Running--')
  for image in data:
      vis = imread('data/Evaluate/RGB/'+image+'.jpg')
      thermal = imread('data/Evaluate/Thermal_Integral/'+image+'.png')
      Integral = imread('data/Evaluate/RGB_Integral/'+image+'.png')
      vis = cv2.resize(vis, (512,512))
      vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
      thermal = cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)
      
      Integral = cv2.resize(Integral, (512,512))
      model = Fusion(vis, thermal, Integral, image)
      try:
        cv2.imwrite('results/Fusion2/'+image+'.png', (model.getFusion() * 255).astype(np.int32))
      except:
        print('Folder Already Exists')
      print(image)
  print('--Finished--')
  end = time.time()
  print(end - start)


if __name__=="__main__":
  data = ['4']
  test_fusion(data)