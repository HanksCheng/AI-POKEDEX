# Importing all necessary libraries
import numpy as np
#陣列工具,負責做資料處理
import pandas as pd
#讀寫檔案用(DEBUG時使用)
import cv2 as cv
#開啟圖片
import os
#資料預處理用
import matplotlib.pyplot as plt
import seaborn as sns
#製作圖表工具



from collections import Counter
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical


path = 'D:\AIpython\dataset1\dataset' # Path contains classes
classes = os.listdir(path) # List of all classes
#print(classes)
#['Blastoise', 'Bulbasaur', 'Butterfree', 'Charizard', 'Charmander', 'Charmeleon', 'Farfetchd', 'Gyarados', 'Ivysaur', 'Machamp', 'Machoke', 'Magikarp', 'Meowth', 'Ninetales', 'Pikachu', 'Psyduck', 'Squirtle', 'Venusaur', 'Vulpix', 'Wartortle']
#print(f'Total number of categories: {len(classes)}')
#Total number of categories: 20

counts = {}
for c in classes:
    counts[c] = len(os.listdir(os.path.join(path, c)))
    #列出Key value 對應 c=pokemon name
    #print(c,":",counts[c])
    
# Number of images in each clsss plot
fig = plt.figure(figsize = (25, 15))
#指定圖片大小，單位為英吋，且影響sns所畫出的圖案
sns.lineplot(x = list(counts.keys()), y = list(counts.values())).set_title('Number of images in each class')
#折線圖
plt.xticks(rotation = 90)
#x軸轉幾度
plt.margins(x=0)
#設定x軸與y軸之間的兼具
plt.show()

imbalanced = sorted(counts.items(), key = lambda x: x[1], reverse = True)[:5]
#排列並取得列表中包含的圖片最多的5個
#print(imbalanced)
#[('Mewtwo', 307), ('Pikachu', 298), ('Charmander', 296), ('Bulbasaur', 289), ('Squirtle', 280)]
imbalanced = [i[0] for i in imbalanced]
#print(imbalanced)
#['Mewtwo', 'Pikachu', 'Charmander', 'Bulbasaur', 'Squirtle']
X = [] # List for images
Y = [] # List for labels

# Loop through all classes
for c in classes:
    # We take only classes that we defined in 'imbalanced' list
    if c in imbalanced:
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
            
print('DONE')

# Counting appearances of each label in labels list
obj = Counter(Y)
#print(obj)
#Counter({0: 306, 1: 298, 2: 296, 3: 289, 4: 280})

# Plotting number of images in each class
fig = plt.figure(figsize = (15, 5))
sns.barplot(x = [imbalanced[i] for i in obj.keys()], y = list(obj.values())).set_title('Number of images in each class')
plt.margins(x=0)
plt.show()
#劃出被篩選出的pokemon數量及類別

# Convert list with images to numpy array and reshape it 
X = np.array(X).reshape(-1, 96, 96, 3)
#將照片存進 numpy array，-1為自動判斷列數

# Scaling data in array
X = X / 255.0
#將np.array裡的資料換成0~1之間
# Convert labels to categorical format
y = to_categorical(Y, num_classes = len(imbalanced))
#one-hot encoding 把資料與照片進行對應
# Splitting data to train and test datasets
# I'll use these datasets only for training, for final predictions I'll use random pictures from internet
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, stratify = y, shuffle = True, random_state = 666)
#將資料分成train跟test，stratify則是根據label平均分配測試資料，達到平均分配,shuffle:打亂資料順序
# Defining ImageDataGenerator Iinstance
datagen = ImageDataGenerator(rotation_range = 45, # Degree range for random rotations
                            zoom_range = 0.2, # Range for random zoom 
                            horizontal_flip = True, # Randomly flip inputs horizontally
                            width_shift_range = 0.15, # Range for horizontal shift 
                            height_shift_range = 0.15, # Range for vertical shift 
                            shear_range = 0.2) # Shear Intensity

datagen.fit(X_train)
#以上皆為數據處，特徵化處理
#print(len(X_train))
#######################################################################
model = Sequential()
#順序式模型,就是一種簡單的模型，單一輸入、單一輸出，按順序一層(Dense)一層的由上往下執行
model.add(Conv2D(32, 3, padding = 'same', activation = 'relu', input_shape =(96, 96, 3), kernel_initializer = 'he_normal'))
#二維卷積層，padding“same”代表保留邊界處的卷積结果，通常会導致输出shape與输入shape相同。激活函数:激活函數
model.add(BatchNormalization(axis = -1))
#標準化，通常每層都會加
model.add(MaxPooling2D((2, 2)))
#池化層，圖片資料量減少並保留重要資訊的方法，把原本的資料做一個最大化或是平均化的降維計算。
model.add(Dropout(0.25))
#防止過度擬和,會有一定機率把下層神經元丟棄,減少過度依賴下層神經元
####以上都是輸入層#######################
model.add(Conv2D(64, 3, padding = 'same', kernel_initializer  = 'he_normal', activation = 'relu'))
#二維卷積層,kernel_initializer:權值初始化方法(He正態分布初始化方法，参數由0均值，標准差為sqrt(2 / fan_in) 的正態分布產生，其中fan_in權重張量的扇入))
model.add(BatchNormalization(axis = -1))
model.add(Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', activation = 'relu'))
model.add(BatchNormalization(axis = -1))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal', activation = 'relu'))
model.add(BatchNormalization(axis = -1))
model.add(Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal', activation = 'relu'))
model.add(BatchNormalization(axis = -1))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal', activation = 'relu'))
model.add(BatchNormalization(axis = -1))
model.add(Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal', activation = 'relu'))
model.add(BatchNormalization(axis = -1))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
#卷積池化層
model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
#核心網路層，輸出為(*,512),可直接用全連接層做解釋
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(len(imbalanced), activation = 'softmax'))
#全連接層,通常全連接層在卷積神經網絡尾部。當前面卷積層抓取到足以用來識別圖片的特徵後，接下來的就是如何進行分類。

model.summary()
#顯示modle概況
checkpoint = ModelCheckpoint('best_modelP.h5', verbose = 1, monitor = 'val_acc', save_best_only = True)
#確認模型是否為最佳,根據Val_acc去確定,verbose為確認日誌狀況(1為顯示)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#配秩序練模型,優化器為'adam',loss會是categorical_crossentropy這個參數產生(我不清楚內部算法),metrics為評估直

###################以上完成model建設

history = model.fit_generator(datagen.flow(X_train, y_train, batch_size = 32), epochs = 50, validation_data = [X_test, y_test],
                             steps_per_epoch=len(X_train) // 5, callbacks = [checkpoint])
#訓練,生成器與模型並行運行，以提高效率。 
#例如，这可以让你在 CPU 上對圖像进行實時數據增強，以在 GPU 上訓練模型。
#datagen.flow(X_train, y_train, batch_size = 32)訓練資料處理
# epochs = 50 代數為幾次
#validation_data:每次訓練後的測事
#在宣告一個 epoch 完成并开始下一个 epoch 之前從 generator 產生的總步数（批次樣本）。
#callbacks,每次訓練後,去確認178行的條件是否達成(val_acc是否進步)決定是否要把權重進行存檔

fig = plt.figure(figsize = (17, 4))
plt.subplot(121)
plt.plot(history.history['acc'], label = 'acc')
plt.plot(history.history['val_acc'], label = 'val_acc')
plt.legend()
plt.grid()
plt.title(f'accuracy')
#劃出acc與val_acc曲線

plt.subplot(122)
plt.plot(history.history['loss'], label = 'loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.legend()
plt.grid()
plt.title(f'loss')
#劃出loss與val_loss曲線

# Loading weights(權重) from best model
model.load_weights('best_modelP.h5')

# Saving all model
model.save('modelpokemon.hdf5')
###英文你看得懂我就不解釋了
