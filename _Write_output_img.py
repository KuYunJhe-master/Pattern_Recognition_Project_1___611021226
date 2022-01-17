#-------------------  寫出結果圖片 --------------------------------------------------------------------
import numpy as np
import cv2


def WEITE_OUTPUT_IMG(img_T_test_anwser , img_target_flatten , img):
    img_different = abs(img_T_test_anwser - img_target_flatten) #計算與正確結果的差異圖
    img_different_out = img_different*255 #恢復成圖片的數據0~255格式
    img_different_out = np.reshape(img_different_out, (img.shape[0],img.shape[1]))#將圖片恢復長寬
    cv2.imwrite("output_img/different.jpg", img_different_out)#寫出圖片
    print("================== 【different.jpg】 write SUCCESS =================")


    out_img = out_img = img_T_test_anwser*255 #恢復成圖片的數據0~255格式
    out_img = np.reshape(out_img, (img.shape[0],img.shape[1]))#將圖片恢復長寬
    cv2.imwrite("output_img/output.jpg", out_img)#寫出圖片
    print("================== 【output.jpg】    write SUCCESS =================")