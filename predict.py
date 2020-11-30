# -*- coding: utf-8 -*-
# Machine learning API
import tensorflow as tf

# Image array processing
import cv2
import matplotlib.pyplot as plt

# my library
from data import get_data_list


#%%  參數設置
print('\nload model .....')
model = tf.keras.models.load_model('best-model')

DIRECTORY_AM = r'D:\YJ\ioplab\glass-defect\data\gd_T01\*\*_A.png'
DIRECTORY_PH = r'D:\YJ\ioplab\glass-defect\data\gd_T01\*\*_P.png'

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64


#%%  讀取資料
print('\nget data list ')

data_Am, data_Ph, data_Lb = get_data_list(
    DIRECTORY_AM, DIRECTORY_PH, shuffle=True, seed=5)

print('\nload data ')
_idx = 8002
Lb = data_Lb[_idx]
Am = tf.io.read_file(data_Am[_idx])
Ph = tf.io.read_file(data_Ph[_idx])

Am = tf.io.decode_png(Am)
Ph = tf.io.decode_png(Ph)
Am = tf.image.resize(Am, (IMAGE_HEIGHT,IMAGE_WIDTH)).numpy()
Ph = tf.image.resize(Ph, (IMAGE_HEIGHT,IMAGE_WIDTH)).numpy()

plt.imshow(Am); plt.colorbar(); plt.show()
plt.imshow(Ph); plt.colorbar(); plt.show()
cv2.imwrite('0 Am.bmp', Am)
cv2.imwrite('0 Ph.bmp', Ph)

Am_tensor = Am.reshape((1,IMAGE_HEIGHT,IMAGE_WIDTH,1))
Ph_tensor = Ph.reshape((1,IMAGE_HEIGHT,IMAGE_WIDTH,1))


#%%  預測結果
print('\npredict model')
pre = model.predict({'XA': Am_tensor, 'XP': Ph_tensor}, verbose=1)
Pb = pre['YC'][0]    # probability
Pr = Pb.argmax(axis=-1)    # max position of probability
print('\nshow predicted result')
print('ground true:', Lb)
print(' predcit Pr:', Pr)
print(' predcit Pb:', Pb)