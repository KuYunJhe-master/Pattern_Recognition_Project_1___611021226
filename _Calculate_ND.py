#------------------- 計算 正態分布 --------------------------------------------------------------------
import numpy as np
import cv2
import matplotlib.pyplot as plt

def _CALCU_N_D(mean, cov, img_flatten):
    print("=========================== 計算正態分布 ==============================")
    print("....分布圖輸出在新視窗....")
    print("....需要花一點時間........")
    print("..........................")


    x = np.random.multivariate_normal(mean, cov, img_flatten.shape, 'warn')   # 計算正態分布
    # print(x.shape)

    # plt.subplot(2, 1, 2)

    print("...八千多萬個點...........")
    print("...需要一點時間跑.........")

    plt.scatter(x[:,:,0], x[:,:,2], #畫出分布圖
                s = 1,
                c ="#FFDEAD", 
                #linewidths = 2, 
                marker =".",  
                #edgecolor ="red"
                )


    plt.xlim(-500, 500)
    plt.ylim(-500, 500)
    # plt.show()
    plt.ion()
    plt.pause(500)
    plt.close()

    print("..........關閉分布圖視窗或等待以繼續.............")
    print("========================== 計算正態分布 FINISH =========================\n\n")