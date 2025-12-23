import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
from tqdm import tqdm
import shutil


Dataset_path="C:/Users/rabin/OneDrive/Desktop/Projects/Final/Dataset_All"
Output_path="C:/Users/rabin/OneDrive/Desktop/Projects/Final/Clustered_Dataset"
img_size=(32,32)

#Create output directory
os.makedirs(Output_path, exist_ok=True)

image_paths=[]
features=[]

#Load and preprocess images
for file in tqdm(os.listdir(Dataset_path)):
    img_path= os.path.join(Dataset_path,file)

    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        img=cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img=cv2.resize(img,img_size)
        img=img.flatten()/255.0  
 
        features.append(img)
        image_paths.append(img_path)

features=np.array(features)

#Clustering
kmeans=KMeans(n_clusters=12, random_state=42, n_init=10)
labels=kmeans.fit_predict(features)
print("Clustering completed.")

#Save Results
for idx, label in enumerate(labels):
    group_folder= os.path.join(Output_path, f"Group_{label}")
    os.makedirs(group_folder, exist_ok=True)

    shutil.copy(
        image_paths[idx],
        os.path.join(group_folder, os.path.basename(image_paths[idx]))
    )
print("Images  saved.")