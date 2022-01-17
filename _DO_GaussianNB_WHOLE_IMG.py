#-------------------  完整圖片的分析結果 --------------------------------------------------------------------
import numpy as np
import cv2
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

def DO_GAUSSIAN_NB_WHOLE_IMG(img_flatten_T,img_target_flatten , model):
    img_T_test_anwser = model.predict(img_flatten_T)#分析完整圖片
    # print("整個圖片個別像素判定結果 = \n", img_T_test_anwser)#整個圖片個別像素判定結果

    train_score = model.score(img_flatten_T , img_target_flatten)#圖片像素正確率
    print("============================================================================== ")
    print("================== 完整圖片辨識正確率 = ", train_score , "================== ")#輸出圖片像素正確率
    print("============================================================================== \n")

    # print("type , shape -------", type(img_flatten_T_teat_anwser), img_flatten_T_teat_anwser.shape)
    # img_flatten_T_teat_anwser = img_flatten_T_teat_anwser.tolist()#轉list才能印出來
    # print(img_flatten_T_teat_anwser[62100000:62110000])#印出其中一區段檢查

    return img_T_test_anwser