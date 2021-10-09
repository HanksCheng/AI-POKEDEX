# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 21:42:46 2020

@author: Tyler
"""

from keras.models import load_model
import matplotlib.pyplot as plt
import os
import cv2 as cv
import numpy as np
 

path = 'D:\\AIpython\\test_pokemon' 
classes = os.listdir(path)

counts = {}
for c in classes:
    counts[c] = len(os.listdir(os.path.join(path, c)))

imbalanced = sorted(counts.items(), key = lambda x: x[1], reverse = True)[:5]
imbalanced = [i[0] for i in imbalanced]
print(imbalanced)

X = [] # List for images
Y = [] # List for labels



# Loop through all classes 
for c in classes:
    # We take only classes that we defined in 'imbalanced' list
    if c in imbalanced:
        print(c)
        #確定c(pokemon name)有在被篩選的5個類別裡的話，把該資料夾的位置寫進dir_path 並且把label標記為該pokemon的名字
        dir_path = os.path.join(path, c)
        label = imbalanced.index(c) # Our label is an index of class in 'imbalanced' list
        #將pokemon的label轉成數字(根據資料多寡排成0~4)
        print(label)
        #0、1、2、3、4 (排序是亂的)
        # Reading, resizing and adding image and label to lists
        for i in os.listdir(dir_path):
            image = cv.imread(os.path.join(dir_path, i))
            #使用cv2進行讀黨
            try:
                resized = cv.resize(image, (96, 96)) # Resizing images to (96, 96)
                #把圖片縮放為96*96
                X.append(resized)
                Y.append(label)
                #把資料放入X與Y
            # If we can't read image - we skip it
            except:
                print(os.path.join(dir_path, i), '[ERROR] can\'t read the file')
                continue       
            
            
X = np.array(X).reshape(-1, 96, 96, 3)
count=[]
for i in range(0,len(X)):
    count.append(i)
    
X = X / 255.0

model=load_model('modelpokemon.hdf5')

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

ans=model.predict_classes(X)
#['Mewtwo', 'Pikachu', 'Charmander', 'Bulbasaur', 'Squirtle'] 原本訓練的編號0~4
print("Y=",Y)
print("ans:",ans)

plt.plot(count,Y,'g.')
plt.plot(count,ans,'r.')

