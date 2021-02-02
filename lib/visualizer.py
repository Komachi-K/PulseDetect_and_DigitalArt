import numpy as np
from time import time
from cv2 import TickMeter, Canny, FONT_HERSHEY_SIMPLEX, COLOR_GRAY2BGR, cvtColor, resize, putText, addWeighted, imshow
import os
import sys
from copy import deepcopy



def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)



class lightFace(object):

    def __init__(self, bpm_limits=[], data_spike_limit=250,
                 face_detector_smoothness=10):

        self.bpm = 0
        self.frame_in = np.zeros((10, 10))
        self.frame_out = np.zeros((10, 10))
        self.kernel = np.ones((3,3),np.uint8)

        self.scale_size_x = 0.1
        self.scale_size_y = 0.03
        self.now_sec = 1
        self.light_scale_max = 0.6
        self.light_scale1 = 0.6
        self.light_scale2 = 1 - self.light_scale_max
        self.col_gray = (211, 211, 211) # ライトグレー

        self.res_show_down = 1
        self.tm = TickMeter()
        self.tm.start()
        


    #scheduler(1, worker, False)

    def reset(self):
        return self.tm.reset()
    
    def stop(self):
        return self.tm.stop()

    def start(self):
        return self.tm.start()
    
    def getTimeSec(self):
        return self.tm.getTimeSec()


    def run(self, cam):
        self.start()
        self.frame_out = self.frame_in

        img = self.frame_in
        canny_img = Canny(img, 170, 200)
        
        
        # オブジェクト情報を項目別に抽出
        canny_img = cvtColor(canny_img, COLOR_GRAY2BGR)
        
        
        sub_time = time() - self.t0
        if sub_time > self.now_sec:
            self.now_sec += 60/self.bpm
            self.scale_size_x = 0.1
            self.scale_size_y = 0.03
            self.light_scale1 = deepcopy(self.light_scale_max)
        
        # scheduler(1, worker, False)

        canny_img = resize(canny_img,None, fx= 1+self.scale_size_x, fy= 1+self.scale_size_y)
        if self.scale_size_y > 0 and self.scale_size_x > 0:
            # self.scale_size_y = self.scale_size_y-0.001
            # self.scale_size_x = self.scale_size_x-0.001
            self.scale_size_y = self.scale_size_y/1.03
            self.scale_size_x = self.scale_size_x/1.03        

        canny_img = canny_img[0:img.shape[0],0:img.shape[1]]
        effect_img = deepcopy(canny_img)
        effect_pixels = (effect_img > (10,10,10)).all(axis = -1)
        effect_img[effect_pixels] = (50*(self.light_scale2+self.light_scale1),50*(self.light_scale2+self.light_scale1),250*(self.light_scale2+self.light_scale1))


        effect_pixels = (effect_img <= (10,10,10)).all(axis = -1)
        effect_img[effect_pixels] = (20,20,60)
        #cv2.imshow("effect1",effect_img)
        self.light_scale1 = self.light_scale1 / 1.05
        Dst = addWeighted(src1 = img,alpha = 0.5, src2 = effect_img,beta = 0.5,gamma = 0)

        #cv2.imshow("effect",effect_img)


        if self.now_sec < 10:
            putText(Dst, "Your Heartrate BPM:{:2d}".format(int(self.bpm)),
                       (int(Dst.shape[0]*0.5), int(Dst.shape[1]*0.1)), FONT_HERSHEY_SIMPLEX,0.6, (211-self.res_show_down, 211-self.res_show_down, 211-self.res_show_down), 2)
        self.res_show_down += 0.5
        # オブジェクト情報を利用してラベリング結果を画面に表示
        # for i in range(n):
            
        #     # if (max_S//2) < data[i][4]:
        #     #     cv2.dilate(labelimage[i+1],kernel,2)
        #     # if (max_S//2) < data[i][4]:
        #     #     cv2.dilate(labelimage[labelimage == i],kernel,1)
                
        #     # 各オブジェクトの外接矩形を赤枠で表示
        #     x0 = data[i][0]
        #     y0 = data[i][1]
        #     x1 = data[i][0] + data[i][2]
        #     y1 = data[i][1] + data[i][3]
        #     cv2.rectangle(canny_img, (x0, y0), (x1, y1), (0, 0, 255))
        #     cv2.rectangle(canny_img, (x0, y0), (x1, y1), (0, 0, 255))

        #     # 各オブジェクトのラベル番号と面積に黄文字で表示
        #     cv2.putText(canny_img, "ID: " +str(i + 1), (x1 - 20, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
        #     cv2.putText(canny_img, "S: " +str(data[i][4]), (x1 - 20, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))

        #     # 各オブジェクトの重心座標をに黄文字で表示
        #     cv2.putText(canny_img, "X: " + str(int(center[i][0])), (x1 - 30, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
        #     cv2.putText(canny_img, "Y: " + str(int(center[i][1])), (x1 - 30, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
        
        # cv2.imshow("canny",canny_img)
        imshow("Processed",Dst)

