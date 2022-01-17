#------------------- 讀取原始圖片 把圖片處理成樣本數據集 --------------------------------------------------------------------
import numpy as np
import cv2


def READ_IMAGE():
    print("\n\n============================ 做成樣本數據集 START ============================")
    img = cv2.imread("input_img/full_duck.jpg")
    print("原始圖片 SIZE   ------ "  , img.shape)
    # cv2.imshow('My Image', img)# 顯示圖片
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()# 按下任意鍵則關閉所有視窗
    img_flatten = img.transpose(2,0,1).reshape(3,-1)#把圖片攤平成RGB的3*N三維數組 (維度調換再reshape攤平)
    print("樣本數據集 SIZE ------ ", img_flatten.shape)
    img_flatten_T = img_flatten.transpose(1,0)#將資料取轉置，符合分類器需要

    print("=========================== 做成樣本數據集 SUCCESS ===========================\n\n")
    return img_flatten , img_flatten_T , img