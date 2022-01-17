#-------------------  讀取標籤用圖片 把圖片處理成樣本標籤集 --------------------------------------------------------------------
import numpy as np
import cv2

def READ_LABEL_IMG():
    print("============================ 做成樣本標籤集 START ============================")
    img_target = cv2.imread("input_img/full_duck_label.jpg") #讀取圖片
    print("標籤圖片 SIZE   ------ ", img_target.shape)
    
    img_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)#轉灰階降維
    ret,img_target = cv2.threshold(img_target,127,255,cv2.THRESH_BINARY)#確保target二元化
    img_target_flatten = img_target.flatten()#把圖片攤平成像素標籤
    img_target_flatten = (img_target_flatten/255).astype('int32') #正規化標籤到 0 or 1 
    print("樣本標籤集 SIZE ------ ", img_target_flatten.shape)

    

    # img_target_list = img_target.tolist()#轉list才能印出來
    # print(img_target_list[62100000:62110000])#印出其中一區段檢查
    # #np.set_printoptions(threshold=np.inf)#強迫完整印出numpy aarray 但失敗了
    print("=========================== 做成樣本標籤集 SUCCESS ===========================\n\n")
    return img_target_flatten
