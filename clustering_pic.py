#%%
from glob import glob
import shutil
import cv2
import os
from sklearn.cluster import KMeans
import numpy as np

#%%画像をnumpy配列で読み込み、変形
IMAGE_DIR = './pic_clustering/design_picture/*'
pathlist = glob(IMAGE_DIR)
features = np.array([cv2.resize(cv2.imread(p), (120, 60), cv2.IMREAD_COLOR) for p in pathlist])
features = features.reshape(len(features), -1).astype(np.float64) #二次元配列に変換

#%%
OUTPUT_DIR = './output_pic'
model = KMeans(n_clusters=10).fit(features)

#%%
for i in range(model.n_clusters):
    cluster_dir = OUTPUT_DIR + "/cluster{}".format(i)
    if os.path.exists(cluster_dir):
        shutil.rmtree(cluster_dir)
    os.makedirs(cluster_dir)

for label, p in zip(model.labels_, pathlist):
    shutil.copyfile(p, OUTPUT_DIR + '/cluster{}/{}'.format(label, p.split('/')[-1]))
# %%
