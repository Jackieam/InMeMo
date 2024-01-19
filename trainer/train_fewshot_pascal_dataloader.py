"""Based on https://github.com/Seokju-Cho/Volumetric-Aggregation-Transformer/blob/main/data/pascal.py
"""
import os
from PIL import Image
from scipy.io import loadmat
import numpy as np
import torch
from torch.utils.data import Dataset
from evaluate.mae_utils import PURPLE, YELLOW
import json
import sys
import random


def create_grid_from_images_old(canvas, support_img, support_mask, query_img, query_mask):
    canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
    canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
    canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
    canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask
    return canvas


class DatasetPASCAL(Dataset):
    def __init__(self, datapath, fold, split, image_transform, mask_transform, padding: bool = 1,
                 use_original_imgsize: bool = False, flipped_order: bool = False,
                 reverse_support_and_query: bool = False, random: bool = False, ensemble: bool = False,
                 purple: bool = False, cluster: bool = False, feature_name: str = 'features_vit-laion2b_no_cls_trn',
                 percentage: str = '', seed: int = 0, mode: str = '', arr: bool = False, n_shot: int = 16):
        self.fold = fold
        self.split = split
        self.nfolds = 4
        self.flipped_order = flipped_order
        self.nclass = 20  # 20
        self.ncluster = 200
        self.padding = padding
        self.random = random
        self.ensemble = ensemble
        self.purple = purple
        self.cluster = cluster
        self.use_original_imgsize = use_original_imgsize

        self.img_path = os.path.join(datapath, 'VOC2012/JPEGImages/')
        self.ann_path = os.path.join(datapath, 'VOC2012/SegmentationClassAug/')
        self.image_transform = image_transform
        self.reverse_support_and_query = reverse_support_and_query
        self.mask_transform = mask_transform
        self.n_shot = n_shot

        self.class_ids = self.build_class_ids()
        self.img_metadata_trn = self.build_img_metadata('trn')
        self.img_few_shot_metadata_trn = self.build_few_shot_metadata('trn', n_shot=self.n_shot)

        self.feature_name = feature_name
        self.seed = seed
        self.percentage = percentage
        self.images_top50_for_training = self.get_top50_images_for_training()
        self.mode = mode
        self.arr = arr

    def __len__(self):
        return len(self.img_few_shot_metadata_trn) if self.split == 'trn' else 1000

    def get_top50_images_for_training(self):
        with open(f"./pascal-5i/VOC2012/{self.feature_name}/folder{self.fold}_top50-similarity.json") as f:
            images_top50 = json.load(f)

        images_top50_new = {}
        for img_name, img_class in self.img_metadata_trn:
            if img_name not in images_top50_new:
                images_top50_new[img_name] = {}

            images_top50_new[img_name]['top50'] = images_top50[img_name]
            images_top50_new[img_name]['class'] = img_class

        return images_top50_new

    def get_top50_images_trn(self):
        images_top50_new = {}
        for img_name, img_class in self.img_metadata_trn:
            if img_name not in images_top50_new:
                images_top50_new[img_name] = {}

            images_top50_new[img_name]['class'] = img_class

        return images_top50_new

    def create_gradiant_grid_images(self, support_img, support_mask, query_img, query_mask, arr):
        # create grid image for suppot images and query image.
        canvas = torch.ones((support_img.shape[0], 2 * support_img.shape[1] + 2 * self.padding,
                             2 * support_img.shape[2] + 2 * self.padding))

        content_list = [support_img, support_mask, query_img, query_mask]

        if arr == 'a1':
            support_img = content_list[0]
            support_mask = content_list[1]
            query_img = content_list[2]
            query_mask = content_list[3]

        elif arr == 'a2':
            support_img = content_list[1]
            support_mask = content_list[0]
            query_img = content_list[3]
            query_mask = content_list[2]

        elif arr == 'a3':
            support_img = content_list[3]
            support_mask = content_list[2]
            query_img = content_list[1]
            query_mask = content_list[0]

        elif arr == 'a4':
            support_img = content_list[2]
            support_mask = content_list[3]
            query_img = content_list[0]
            query_mask = content_list[1]

        elif arr == 'a5':
            support_img = content_list[1]
            support_mask = content_list[3]
            query_img = content_list[0]
            query_mask = content_list[2]

        elif arr == 'a6':
            support_img = content_list[3]
            support_mask = content_list[1]
            query_img = content_list[2]
            query_mask = content_list[0]

        elif arr == 'a7':
            support_img = content_list[2]
            support_mask = content_list[0]
            query_img = content_list[3]
            query_mask = content_list[1]

        elif arr == 'a8':
            support_img = content_list[0]
            support_mask = content_list[2]
            query_img = content_list[1]
            query_mask = content_list[3]

        canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
        canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
        canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
        canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask

        return canvas

    def create_ensemble_grid_from_images(self, support_img, support_mask, query_img, query_mask, positions):
        canvas = torch.ones((support_img.shape[0], 2 * support_img.shape[1] + 2 * self.padding,
                             2 * support_img.shape[2] + 2 * self.padding))

        if positions == 'a1':
            canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask
        elif positions == 'a2':
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_img
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_img
            canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_mask
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_mask
        elif positions == 'a3':
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = support_img
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = query_img
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = support_mask
            canvas[:, :support_img.shape[1], :support_img.shape[2]] = query_mask
        elif positions == 'a4':
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = support_img
            canvas[:, :support_img.shape[1], :support_img.shape[2]] = query_img
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = support_mask
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = query_mask
        elif positions == 'a5':
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = support_img
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_img
            canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_mask
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = query_mask
        elif positions == 'a6':
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = support_img
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
            canvas[:, :support_img.shape[1], :support_img.shape[2]] = query_mask
        elif positions == 'a7':
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_img
            canvas[:, :support_img.shape[1], :support_img.shape[2]] = query_img
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = support_mask
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_mask
        elif positions == 'a8':
            canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = query_img
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = support_mask
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask

        return canvas

    def create_all_grids(self, support_img, support_mask, query_img, query_mask):
        canvas_list = []

        # List of all possible arrangements
        arrangements = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8']

        for arr in arrangements:
            canvas = self.create_ensemble_grid_from_images(support_img, support_mask, query_img, query_mask, arr)
            canvas_list.append(canvas)

        return canvas_list

    def __getitem__(self, idx):
        grid_stack = torch.tensor([]).cuda()

        for sim_idx in range(1):
            query_name, support_name, class_sample_query, class_sample_support = self.sample_episode_for_training(idx, sim_idx)
            query_img, query_cmask, support_img, support_cmask, org_qry_imsize = self.load_frame(query_name,
                                                                                                 support_name)
            if self.image_transform:
                query_img = self.image_transform(query_img)
                query_mask, query_ignore_idx = self.extract_ignore_idx(query_cmask, class_sample_query,
                                                                       purple=self.purple)
            if self.mask_transform:
                query_mask = self.mask_transform(query_mask)

            # The support image no need for transformation.
            if self.image_transform:
                support_img = self.image_transform(support_img)
            support_mask, support_ignore_idx = self.extract_ignore_idx(support_cmask, class_sample_support,
                                                                       purple=self.purple)
            if self.mask_transform:
                support_mask = self.mask_transform(support_mask)

            if self.arr != 'ensemble':
                grid = self.create_gradiant_grid_images(support_img, support_mask, query_img, query_mask, self.arr)

            else:
                grid = self.create_all_grids(support_img, support_mask, query_img, query_mask)

            if len(grid_stack) == 0:
                grid_stack = grid
            else:
                grid_stack = torch.cat((grid_stack, grid))

        # print('grid stack: ', grid_stack.shape)
        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'support_img': support_img,
                 'support_mask': support_mask,
                 'grid_stack': grid_stack
                 }

        return batch


    def extract_ignore_idx(self, mask, class_id, purple):
        mask = np.array(mask)
        boundary = np.floor(mask / 255.)
        if not purple:
            mask[mask != class_id + 1] = 0
            mask[mask == class_id + 1] = 255
            return Image.fromarray(mask), boundary
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if mask[x, y] != class_id + 1:
                    color_mask[x, y] = np.array(PURPLE)
                else:
                    color_mask[x, y] = np.array(YELLOW)
        return Image.fromarray(color_mask), boundary

    def load_frame(self, query_name, support_name):
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        support_img = self.read_img(support_name)
        support_mask = self.read_mask(support_name)
        org_qry_imsize = query_img.size

        return query_img, query_mask, support_img, support_mask, org_qry_imsize

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = Image.open(os.path.join(self.ann_path, img_name) + '.png')
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg')

    def sample_episode_for_training(self, idx, sim_idx):
        """Returns the index of the query, support and class."""
        query_name, class_sample = self.img_few_shot_metadata_trn[idx]

        support_name = self.images_top50_for_training[query_name]['top50'][sim_idx]
        support_class = self.images_top50_for_training[support_name]['class']

        return query_name, support_name, class_sample, support_class

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        return class_ids_val

    def build_few_shot_metadata(self, split, n_shot):

        def read_metadata(split, fold_id):
            cwd = 'evaluate'

            fold_n_metadata_path = os.path.join(cwd, 'splits/pascal/%s/fold%d.txt' % (split, fold_id))

            with open(fold_n_metadata_path, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]

            classes = {}
            for item in fold_n_metadata:
                if item[1] not in classes:
                    classes[item[1]] = []
                classes[item[1]].append(item)

            n = n_shot
            random.seed(1)
            selected_items = []
            for class_items in classes.values():
                selected_items.extend(random.sample(class_items, min(n, len(class_items))))

            return selected_items

        img_metadata = []
        img_metadata = read_metadata(split, self.fold)

        print('Total (%s) images are : %d' % (split, len(img_metadata)))

        return img_metadata

    def build_img_metadata(self, split):

        def read_metadata(split, fold_id):
            # cwd = os.path.dirname(os.path.abspath(__file__))
            cwd = 'evaluate'

            if self.cluster:
                fold_n_metadata_path = os.path.join(cwd, 'splits/pascal/%s/fold_cluster%d.txt' % (split, fold_id))
            else:
                fold_n_metadata_path = os.path.join(cwd, 'splits/pascal/%s/fold%d.txt' % (split, fold_id))

            with open(fold_n_metadata_path, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            if self.cluster:
                fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1, int(data.split('__')[2]) - 1] for
                                   data in fold_n_metadata]
            else:
                fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]

            return fold_n_metadata

        img_metadata = []
        img_metadata = read_metadata(split, self.fold)

        print('Total (%s) images are : %d' % (split, len(img_metadata)))

        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        if len(self.img_metadata[0]) != 3:
            for img_name, img_class in self.img_metadata:
                img_metadata_classwise[img_class] += [img_name]
        else:
            for img_name, img_class, _ in self.img_metadata:
                img_metadata_classwise[img_class] += [img_name]

        return img_metadata_classwise
