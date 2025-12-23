import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

Dataset_path="C:/Users/rabin/OneDrive/Desktop/Projects/Final/Clustered_Dataset"
img_size=(32,32)

data=[]
labels=[]

for label in os.listdir(Dataset_path):
    class_path=os.path.join(Dataset_path,label)
    if not os.path.isdir(class_path):
        continue

    for file in os.listdir(class_path):
        img_path=os.path.join(class_path,file)
        img=cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,img_size)
        img=img.flatten()/255.0

        data.append(img)
        labels.append(label)

x=np.array(data)
y=np.array(labels)


df=pd.DataFrame(x)
df['label']=y
print(df.describe())