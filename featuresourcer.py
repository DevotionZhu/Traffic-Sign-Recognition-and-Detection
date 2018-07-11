from skimage.feature import hog
import numpy as np
from helpers import convert 

class FeatureSourcer:
  def __init__(self, p, start_frame):
    
    self.color_model = p['color_model']
    self.s = p['bounding_box_size']
    
    self.ori = p['number_of_orientations']
    self.ppc = (p['pixels_per_cell'], p['pixels_per_cell'])
    self.cpb = (p['cells_per_block'], p['cells_per_block']) 
    self.do_sqrt = p['do_transform_sqrt']

    self.ABC_img = None
    self.dims = (None, None, None)
    self.hogA, self.hogB, self.HogC = None, None, None
    self.hogA_img, self.hogB_img, self.hogC = None, None, None
    
    self.RGB_img = start_frame
    self.new_frame(self.RGB_img)

  def hog(self, channel):
    features, hog_img = hog(channel, 
                            orientations = self.ori, 
                            pixels_per_cell = self.ppc,
                            cells_per_block = self.cpb, 
                            transform_sqrt = self.do_sqrt, 
                            visualise = True, 
                            feature_vector = False)
    return features, hog_img

  def new_frame(self, frame):
    
    self.RGB_img = frame 
    self.ABC_img = convert(frame, src_model= 'rgb', dest_model = self.color_model)
    self.dims = self.RGB_img.shape
    
    self.hogA, self.hogA_img = self.hog(self.ABC_img[:, :, 0])
   # self.hogB, self.hogB_img = self.hog(self.ABC_img[:, :, 1])
   # self.hogC, self.hogC_img = self.hog(self.ABC_img[:, :, 2])
    
  def slice(self, x_pix, y_pix, w_pix = None, h_pix = None):
    x_start, x_end, y_start, y_end = self.pix_to_hog(x_pix, y_pix, h_pix, w_pix)
    
    hogA = self.hogA[y_start: y_end, x_start: x_end].ravel()
    #hogB = self.hogB[y_start: y_end, x_start: x_end].ravel()
    #hogC = self.hogC[y_start: y_end, x_start: x_end].ravel()
    #hog = np.hstack((hogA, hogB, hogC))
    return hogA 
    
  def color_features(self):
    Y, U, V = [], [], []
    frame0 = self.ABC_img[:,:,0]
    frame1 = self.ABC_img[:,:,1]
    frame2 = self.ABC_img[:,:,2]
    ppc = self.ppc[0]//2
    x = (frame1.shape[1]//(2*ppc)) * 2*ppc;
    y = (frame1.shape[0]//ppc) * ppc;
    for j in range(0, x, 2*ppc):
        for i in range(0, y, ppc):
            Y.append(np.mean(frame0[i:i+ppc, j:j+2*ppc]))
            U.append(np.mean(frame1[i:i+ppc, j:j+2*ppc]))
            V.append(np.mean(frame2[i:i+ppc, j:j+2*ppc]))
    return np.hstack((Y,U,V))
    

  def features(self, frame):
    self.new_frame(frame)
    return np.hstack((self.slice(0, 0, frame.shape[1], frame.shape[0]), self.color_features())) 

  def visualize(self):
    return self.RGB_img, self.hogA_img, self.hogB_img, self.hogC_img

  def pix_to_hog(self, x_pix, y_pix, h_pix, w_pix):

    if h_pix is None and w_pix is None: 
      h_pix, w_pix = self.s, self.s
    
    h = h_pix // self.ppc[0]
    w = w_pix // self.ppc[0]
    y_start = y_pix // self.ppc[0]
    x_start = x_pix // self.ppc[0]
    y_end = y_start + h - 1
    x_end = x_start + w - 1
    
    return x_start, x_end, y_start, y_end