import torch.utils.data as data
from evaluate_detection.voc_orig import VOCDetection4Val, VOCDetection4Train, make_transforms
import cv2
from PIL import Image
from evaluate_detection.voc import make_transforms, create_grid_from_images
import torch
import numpy as np
import torchvision.transforms as T


def box_to_img(mask, target, border_width=4):
    if mask is None:
        mask = np.zeros((112, 112, 3))
    h, w, _ = mask.shape
    for box in target['boxes']:
        x_min, y_min, x_max, y_max = list((box * (h - 1)).round().int().numpy())
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), (255, 255, 255), border_width)
    return Image.fromarray(mask.astype('uint8'))


def get_annotated_image(img, boxes, border_width=3, mode='draw', bgcolor='white', fg='image'):
    if mode == 'draw':
        image_copy = np.array(img.copy())
        for box in boxes:
            box = box.numpy().astype('int')
            cv2.rectangle(image_copy, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), border_width)
    elif mode == 'keep':
        image_copy = np.array(Image.new('RGB', (img.shape[1], img.shape[0]), color=bgcolor))

        for box in boxes:
            box = box.numpy().astype('int')
            if fg == 'image':
                image_copy[box[1]:box[3], box[0]:box[2]] = img[box[1]:box[3], box[0]:box[2]]
            elif fg == 'white':
                image_copy[box[1]:box[3], box[0]:box[2]] = 255

    return image_copy


class CanvasDataset4Val(data.Dataset):
    def __init__(self, pascal_path='pascal-5i', years=("2012",), random=False, **kwargs):
        self.train_ds = VOCDetection4Val(pascal_path, years, image_sets=['train'], transforms=None,
                                         keep_single_objs_only=1, filter_by_mask_size=1)
        self.val_ds = VOCDetection4Val(pascal_path, years, image_sets=['val'], transforms=None,
                                       keep_single_objs_only=1, filter_by_mask_size=1)
        self.background_transforms = T.Compose([
            T.Resize((224, 224)),
            T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ])
        self.transforms = make_transforms('val')
        self.random = random

    def __len__(self):
        return len(self.val_ds)

    def __getitem__(self, idx):

        grid_stack = torch.tensor([]).cuda()

        query_image, query_target = self.val_ds[idx]
        label = query_target['labels'].numpy()[0]


        support_image, support_target = self.train_ds[idx]
        support_label = support_target['labels'].numpy()[0]


        boxes = support_target['boxes'][torch.where(support_target['labels'] == support_label)[0]]
        support_image_copy = get_annotated_image(np.array(support_image), boxes, border_width=-1, mode='keep', bgcolor='black', fg='white')
        support_image_copy_pil = Image.fromarray(support_image_copy)

        boxes = query_target['boxes'][torch.where(query_target['labels'] == label)[0]]
        query_image_copy = get_annotated_image(np.array(query_image), boxes, border_width=-1, mode='keep', bgcolor='black', fg='white')
        query_image_copy_pil = Image.fromarray(query_image_copy)

        query_image_ten = self.transforms(query_image, None)[0]
        query_target_ten = self.transforms(query_image_copy_pil, None)[0]
        support_target_ten = self.transforms(support_image_copy_pil, None)[0]
        support_image_ten = self.transforms(support_image, None)[0]

        background_image = Image.new('RGB', (224, 224), color='white')
        background_image = self.background_transforms(background_image)
        grid = create_grid_from_images(background_image, support_image_ten, support_target_ten, query_image_ten,
                                         query_target_ten)

        if len(grid_stack) == 0:
            grid_stack = grid
        else:
            grid_stack = torch.cat((grid_stack, grid))

        return {'query_img': query_image_ten,
                'query_mask': query_target_ten,
                'support_img': support_image_ten,
                'support_mask': support_target_ten,
                'grid_stack': grid_stack}


class CanvasDataset4Train(data.Dataset):
    def __init__(self, pascal_path='pascal-5i', years=("2012",), random=False, **kwargs):
        self.train_ds = VOCDetection4Train(pascal_path, years, image_sets=['train'], transforms=None, keep_single_objs_only=1, filter_by_mask_size=1)
        self.val_ds = VOCDetection4Train(pascal_path, years, image_sets=['val'], transforms=None, keep_single_objs_only=1, filter_by_mask_size=1)
        self.background_transforms = T.Compose([
            T.Resize((224, 224)),
            T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ])
        self.transforms = make_transforms('val')
        self.random = random

    def __len__(self):
        return len(self.val_ds)

    def __getitem__(self, idx):

        grid_stack = torch.tensor([]).cuda()

        query_image, query_target = self.val_ds[idx]

        label = query_target['labels'].numpy()[0]


        support_image, support_target = self.train_ds[idx]
        support_label = support_target['labels'].numpy()[0]


        boxes = support_target['boxes'][torch.where(support_target['labels'] == support_label)[0]]
        support_image_copy = get_annotated_image(np.array(support_image), boxes, border_width=-1, mode='keep', bgcolor='black', fg='white')
        support_image_copy_pil = Image.fromarray(support_image_copy)

        boxes = query_target['boxes'][torch.where(query_target['labels'] == label)[0]]
        query_image_copy = get_annotated_image(np.array(query_image), boxes, border_width=-1, mode='keep', bgcolor='black', fg='white')
        query_image_copy_pil = Image.fromarray(query_image_copy)

        query_image_ten = self.transforms(query_image, None)[0]
        query_target_ten = self.transforms(query_image_copy_pil, None)[0]
        support_target_ten = self.transforms(support_image_copy_pil, None)[0]
        support_image_ten = self.transforms(support_image, None)[0]

        background_image = Image.new('RGB', (224, 224), color='white')
        background_image = self.background_transforms(background_image)
        grid = create_grid_from_images(background_image, support_image_ten, support_target_ten, query_image_ten,
                                         query_target_ten)

        if len(grid_stack) == 0:
            grid_stack = grid
        else:
            grid_stack = torch.cat((grid_stack, grid))

        return {'query_img': query_image_ten,
                'query_mask': query_target_ten,
                'support_img': support_image_ten,
                'support_mask': support_target_ten,
                'grid_stack': grid_stack}