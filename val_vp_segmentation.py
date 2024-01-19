import os.path
from tqdm import tqdm
from trainer import val_pascal_dataloader
from evaluate.reasoning_dataloader import *
import torchvision
from evaluate.mae_utils import *
import argparse
from pathlib import Path
from evaluate.segmentation_utils import *
from PIL import Image
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from trainer.train_models import _generate_result_for_canvas, CustomVP
from evaluate_detection.voc_orig import CLASS_NAMES


def get_args():
    parser = argparse.ArgumentParser('InMeMo inference for segmentation', add_help=False)
    parser.add_argument('--mae_model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument("--mode", type=str, default='spimg_spmask',
                        choices=['no_vp', 'spimg_spmask', 'spimg', 'spimg_qrimg', 'qrimg', 'spimg_spmask_qrimg'],
                        help="mode of adding visual prompt on img.")
    parser.add_argument('--output_dir', default=f'./output_samples/')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--base_dir', default='./pascal-5i', help='pascal base dir')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--t', default=[0, 0, 0], type=float, nargs='+')
    parser.add_argument('--task', default='segmentation', choices=['segmentation', 'detection'])
    parser.add_argument('--ckpt', default='./weights/checkpoint-1000.pth', help='model checkpoint')
    parser.add_argument('--dataset_type', default='pascal',
                        choices=['pascal', 'pascal_det'])
    parser.add_argument('--fold', default=3, type=int)
    parser.add_argument('--split', default='trn', type=str)
    parser.add_argument('--purple', default=0, type=int)
    parser.add_argument('--flip', default=0, type=int)
    parser.add_argument('--feature_name', default='features_vit-laion2b_pixel-level_trn', type=str)
    parser.add_argument('--percentage', default='', type=str)
    parser.add_argument('--cluster', action='store_true')
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--ensemble', action='store_true')

    parser.add_argument("--p-eps", type=int, default=1,
                        help="Number of mae weight hyperparameter,[0, 1].")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--arr", type=str, default='a1',
                        help="the setting of arrangements of canvas")
    parser.add_argument("--vp-model", type=str, default='pad',  #  ['pad']
                        help="pad prompter")
    parser.add_argument('--save_model_path',
                        help='model checkpoint')

    return parser


def test_for_generate_results(args):
    padding = 1
    image_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((224 // 2 - padding, 224 // 2 - padding), 3),
         torchvision.transforms.ToTensor()])
    mask_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((224 // 2 - padding, 224 // 2 - padding), 3),
         torchvision.transforms.ToTensor()])

    val_dataset = {
        'pascal': val_pascal_dataloader.DatasetPASCAL,
    }[args.dataset_type](args.base_dir, fold=args.fold, split=args.split, image_transform=image_transform,
                         mask_transform=mask_transform,
                         flipped_order=args.flip, purple=args.purple, random=args.random, cluster=args.cluster,
                         feature_name=args.feature_name, percentage=args.percentage, seed=args.seed, mode=args.mode,
                         arr=args.arr)

    dataloaders = {}
    dataloaders['val'] = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print('val datalaoder: ', len(dataloaders['val']))

    print("load data over")

    # MAE_VQGAN model
    vqgan = prepare_model(args.ckpt, arch=args.mae_model)

    if args.vp_model == 'pad':
        print('load pad prompter.')
        VP = CustomVP(args=args, vqgan=vqgan.to(args.device), mode=args.mode, arr=args.arr, p_eps=args.p_eps)
    else:
        raise ValueError("No VP model is loaded, Please use 'pad'")

    if args.mode != 'no_vp':
        state_dict = torch.load(args.save_model_path, map_location='cuda:0')
        VP.PadPrompter.load_state_dict(state_dict["visual_prompt_dict"])

        VP.eval()
        VP.to(args.device)

    setting = f'{args.mode}_fold{args.fold}_{args.task}_{args.arr}'
    eg_save_path = f'{args.output_dir}/{args.vp_model}_output_examples/{args.mode}'
    os.makedirs(eg_save_path, exist_ok=True)

    print(f'This is the mode of {args.mode}.')
    print(f'This is the arrangement of {args.arr}.')

    eval_dict = {'iou': 0, 'color_blind_iou': 0, 'accuracy': 0}
    examples_save_path = eg_save_path + f'/{setting}/'
    os.makedirs(examples_save_path, exist_ok=True)

    with open(os.path.join(examples_save_path, 'log.txt'), 'w') as log:
        log.write(str(args) + '\n')

    image_number = 0

    # Inference phase
    for i, data in enumerate(tqdm(dataloaders["val"])):
        len_dataloader = len(dataloaders["val"])
        # print('len dataloader: ', len_dataloader)
        support_img, support_mask, query_img, query_mask, grid_stack = \
            data['support_img'], data['support_mask'], data['query_img'], data['query_mask'], data['grid_stack']
        support_img = support_img.to(args.device, dtype=torch.float32)
        support_mask = support_mask.to(args.device, dtype=torch.float32)
        query_img = query_img.to(args.device, dtype=torch.float32)
        query_mask = query_mask.to(args.device, dtype=torch.float32)
        grid_stack = grid_stack.to(args.device, dtype=torch.float32)

        _, canvas_pred, canvas_label = VP(support_img, support_mask, query_img, query_mask, grid_stack)

        if args.dataset_type != 'pascal_det':
            # convert to imagenet distribution.
            imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).to(args.device)
            imagenet_std = torch.tensor([0.229, 0.224, 0.225]).to(args.device)
            canvas_pred = (canvas_pred - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
            canvas_label = (canvas_label - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]


        original_image_list, generated_result_list = _generate_result_for_canvas(args, vqgan.to(args.device),
                                                                                 canvas_pred, canvas_label, args.arr)
        for index in range(len(original_image_list)):

            # Image.fromarray(original_image_list[index]).save(
            #     examples_save_path + f'original_image_{image_number}.png')
            Image.fromarray(generated_result_list[index]).save(
                examples_save_path + f'generated_image_{image_number}.png')

            original_image = round_image(original_image_list[index], [WHITE, BLACK])
            generated_result = round_image(generated_result_list[index], [WHITE, BLACK], t=args.t)

            current_metric = calculate_metric(args, original_image, generated_result, fg_color=WHITE, bg_color=BLACK)
            # print('current_metric: ', current_metric)
            with open(os.path.join(examples_save_path, 'log.txt'), 'a') as log:
                # log.write(str(idx) + '\t' + str(current_metric) + '\n')
                log.write(str(image_number) + '\t' + str(current_metric) + '\n')
            image_number += 1

            for i, j in current_metric.items():
                eval_dict[i] += (j / len(val_dataset))

        # print('eval_dict: ', eval_dict)
    print('val metric: {}'.format(eval_dict))
    with open(os.path.join(examples_save_path, 'log.txt'), 'a') as log:
        log.write('all\t' + str(eval_dict) + '\n')


if __name__ == '__main__':
    mp.set_start_method('spawn')
    args = get_args()

    args = args.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    test_for_generate_results(args)
