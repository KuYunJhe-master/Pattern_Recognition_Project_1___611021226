#-------------------  訓練貝氏分類器 --------------------------------------------------------------------
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

def DO_GAUSSIAN_NB(img_flatten_T,img_target_flatten , test_set_size):
    print("=========================== 計算貝氏分類器辨識率 ==============================")
    
    Data_train , Data_test , Label_train , Label_test = train_test_split(img_flatten_T,img_target_flatten,test_size = test_set_size , stratify = img_target_flatten ) #分割訓練集和測試集

    model = GaussianNB()#建立高斯貝氏分類器模型
    model.fit(Data_train , Label_train)#訓練
    teat_anwser = model.predict(Data_test)#測試
    # print(teat_anwser)#個別判定結果

    teat_proba_anwser = model.predict_proba(Data_test)#test集個別判定機率
    train_score = model.score(Data_train , Label_train)#訓練集正確率
    test_score = model.score(Data_test , Label_test)#測試集正確率

    print("訓練集正確率 = ",  train_score)#輸出訓練集正確率
    print("測試集正確率 = ",  test_score)#輸測試集正確率

    print("=============================== 辨識FINISH =================================\n\n")
    return model
