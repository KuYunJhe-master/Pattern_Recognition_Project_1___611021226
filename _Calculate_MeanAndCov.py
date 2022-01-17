#------------------- 計算 平均向量、共變異矩陣 --------------------------------------------------------------------
import numpy as np
import cv2

def _CALCU_M_And_C(img_flatten):
    print("============================ 計算平均向量、共變異矩陣 START ============================")

    mean_R, std = cv2.meanStdDev(img_flatten[0])#R頻道平均值
    mean_G, std = cv2.meanStdDev(img_flatten[1])#B頻道平均值
    mean_B, std = cv2.meanStdDev(img_flatten[2])#G頻道平均值
    mean_R = float(mean_R[0,0])#從array取出
    mean_G = float(mean_G[0,0])#從array取出
    mean_B = float(mean_B[0,0])#從array取出



    print("---------------------------------- 平均向量 ----------------------------------")
    print("R 方向的平均向量 -----------------  ", mean_R)
    print("G 方向的平均向量 -----------------  ", mean_G)
    print("B 方向的平均向量 -----------------  ", mean_B)
    mean = np.array([mean_R, mean_G, mean_B]) # 組合三頻道平均值
    print("平均向量  ------------------------  mean =",mean)
    print("-----------------------------------------------------------------------------\n")


    cov = np.cov(img_flatten)  # 計算共變異矩陣

    print("----------------- 共變異矩陣 -----------------\n",cov,"\n---------------------------------------------")


    print("========================== 計算平均向量、共變異矩陣 FINISH ============================\n\n")
    return mean , cov
