import numpy as np
import scipy.spatial.distance as distance
import os
from tqdm import tqdm
import sys
from scipy import linalg, mat, dot
import json


features_name = sys.argv[1]
source_split = sys.argv[2]
target_split = sys.argv[3]

print(f"Processing {features_name} ...")
sys.stdout.flush()

source_features_dir = f"./pascal-5i/VOC2012/{features_name}_{source_split}_all_detection"
target_features_dir = f"./pascal-5i/VOC2012/{features_name}_{target_split}_all_detection"

print(source_features_dir)
print(target_features_dir)

feature_file = 'folder' + '.npz'
print(f"Processing {feature_file} ...")
sys.stdout.flush()
source_path = os.path.join(source_features_dir, feature_file)
target_path = os.path.join(target_features_dir, feature_file)
try:
    source_file_npz = np.load(source_path)
    target_file_npz = np.load(target_path)
except:
    print(f"no folder {feature_file} ...")
    sys.stdout.flush()

source_examples = source_file_npz["examples"].tolist()
target_examples = target_file_npz["examples"].tolist()
source_features = source_file_npz["features"].astype(np.float32)
target_features = target_file_npz["features"].astype(np.float32)

source_features = source_features.reshape(source_features.shape[0], -1)
target_features = target_features.reshape(target_features.shape[0], -1)

print('source_features shape: ', source_features.shape)
print('target_features shape: ', target_features.shape)

target_sample_idx = np.random.choice(target_features.shape[0], size=int(target_features.shape[0]), replace=False)
target_sample_feature = target_features[target_sample_idx, :]
similarity = dot(source_features, target_sample_feature.T)/(linalg.norm(source_features, axis=1, keepdims=True) * linalg.norm(target_sample_feature, axis=1, keepdims=True).T)

# The 200 examples with the greatest similarity were selected as the prompt pair obtained from the train for the val dataset.
similarity_idx = np.argsort(similarity, axis=1)[:, -200:]
print("similarity_idx shape: ", similarity_idx.shape)

similarity_idx_dict = {}
for _, (cur_example, cur_similarity) in enumerate(zip(source_examples, similarity_idx)):
    # print("cur_example: ", cur_example)
    img_name = cur_example.strip().split('/')[-1][:-4]
    # print("img_name: ", img_name)

    cur_similar_name = list(target_examples[target_sample_idx[idx]].strip().split('/')[-1][:-4] for idx in cur_similarity[::-1])
    cur_similar_name = list(dict.fromkeys(cur_similar_name))

    assert len(cur_similar_name) >= 50, "num of cur_similar_name is too small, please enlarge the similarity_idx size"

    if source_split == 'val' and target_split == 'train':
        # select top50 prompt pairs for each sample.
        if img_name not in similarity_idx_dict:
            similarity_idx_dict[img_name] = cur_similar_name[:50]
    elif source_split == 'train' and target_split == 'train':
        # select top50 prompt pairs for each sample.
        if img_name not in similarity_idx_dict:
            similarity_idx_dict[img_name] = cur_similar_name[1:51]  # to avoid the sample itself.

with open(f"{source_features_dir}/top_50-similarity.json", "w") as outfile:
    json.dump(similarity_idx_dict, outfile)
        
