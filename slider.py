import numpy as np 
import cv2
from featuresourcer import FeatureSourcer
from binaryclassifier import BinaryClassifier

class Slider:
  
  def __init__(self, sourcer, classifier, increment):
    self.sourcer = sourcer
    self.classifier = classifier
    self.i = increment
    self.h = sourcer.s
    self.current_strip = None 
    
  def prepare(self, frame, y, ws):
    
    scaler = ws / self. h
    y_end = y + ws 
    w = np.int(frame.shape[1] / scaler)
    
    strip = frame[y: y_end, :, :]
    strip = cv2.resize(strip, (w, self.h))
    self.current_strip = strip 
    
    return scaler, strip

  def strip(self):
    return self.current_strip

  def locate(self, frame, window_size, window_position):
    
    y, ws = window_position, window_size 
    scaler, strip = self.prepare(frame, y, ws)
    
    boxes = []
    y_scores = []
    self.sourcer.new_frame(strip)
    f_color = self.sourcer.color_features()
    f_color_YUV = f_color.reshape(3,len(f_color)//3)
    f_color_Y = f_color_YUV[0].reshape(8, len(f_color_YUV[0])//8)
    f_color_U = f_color_YUV[1].reshape(8, len(f_color_YUV[1])//8) #Hardcoded for 8 * 16  for color features
    f_color_V = f_color_YUV[2].reshape(8, len(f_color_YUV[2])//8)
    
    x_end = (strip.shape[1] // self.h - 1) * self.h
        
    for resized_x in range(0, x_end, self.i):

        features = self.sourcer.slice(resized_x, 0, self.h, self.h)  
       # features = np.hstack((features, self.sourcer.color_features(self.sourcer.ABC_img[0:self.h, resized_x:(resized_x+self.h)])))
        features = np.hstack((features, f_color_Y[0:8, resized_x//16:(resized_x+self.h)//16].ravel(), f_color_U[0:8, resized_x//16:(resized_x+self.h)//16].ravel(), f_color_V[0:8, resized_x//16:(resized_x+self.h)//16].ravel()))          
        if self.classifier.predict(features): 
            x = np.int(scaler * resized_x)
            boxes.append((x, y, ws))
        y_scores.append(self.classifier.decision_function(features))
    return boxes, y_scores

  def locate_test(self, frame, window_size, window_position, area):
    
    y, ws = window_position, window_size 
    scaler, strip = self.prepare_test(frame, y, ws)   
    
    y_test = []
    per_test = []
    x_end = (strip.shape[1] // self.h - 1) * self.h
    
    for resized_x in range(0, x_end, self.i):
        if area < (ws*ws) and area > ws*ws*0.65:
            area = ws*ws
        elif area < ws*ws*0.65:
            area = ws*ws*0.65
        percent = np.sum(strip[0:self.h, resized_x:resized_x+self.h])*scaler*scaler/255/(area)
        if percent > 0.6:
            y_test.append(1)
        else:
            y_test.append(0)
        per_test.append(percent)
    return y_test, per_test

  def prepare_test(self, frame, y, ws):
    
    scaler = ws / self. h
    y_end = y + ws 
    w = np.int(frame.shape[1] / scaler)
    
    strip = frame[y: y_end, :]
    strip = cv2.resize(strip, (w, self.h))
    self.current_strip = strip 
    
    return scaler, strip