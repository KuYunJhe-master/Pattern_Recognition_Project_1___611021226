import _Read_image
import _Read_Lable_Img
import _Calculate_MeanAndCov
import _Calculate_ND
import _Do_GaussianNB
import _DO_GaussianNB_WHOLE_IMG
import _Write_output_img


test_set_size = 0.3#------------------- 指定訓練集與測試集的比例，0.3表示測試集是占所有資料的30%

img_Flatten , img_DATA, img = _Read_image.READ_IMAGE()#------------------- 讀取原始圖片 把圖片處理成樣本數據集

img_Label = _Read_Lable_Img.READ_LABEL_IMG()#-------------------  讀取標籤用圖片 把圖片處理成樣本標籤集

mean , cov = _Calculate_MeanAndCov._CALCU_M_And_C(img_Flatten)#------------------- 計算 平均向量、共變異矩陣

_Calculate_ND._CALCU_N_D(mean, cov, img_Flatten)#------------------- 計算 正態分布

model_Trained = _Do_GaussianNB.DO_GAUSSIAN_NB(img_DATA,img_Label , test_set_size)#-------------------  訓練貝氏分類器

img_recognized = _DO_GaussianNB_WHOLE_IMG.DO_GAUSSIAN_NB_WHOLE_IMG(img_DATA,img_Label , model_Trained)#-------------------  完整圖片的分析結果

_Write_output_img.WEITE_OUTPUT_IMG(img_recognized , img_Label , img)#-------------------  寫出結果圖片
