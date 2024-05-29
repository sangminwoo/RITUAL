import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/experiments')
# print(sys.path)

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import Conversation, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

from utils import dist_util
from utils.logger import create_logger
from glob import glob

import re
from PIL import Image
from torchvision.transforms import v2

from pope_loader import POPEDataSet

# import kornia
from ritual_utils.ritual_sample import evolve_ritual_sampling
from ritual_utils.vcd_add_noise import add_diffusion_noise

evolve_ritual_sampling()
torch.multiprocessing.set_sharing_strategy('file_system')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description="POPE evaluation on LVLMs.")
    parser.add_argument("--model_path", type=str, default="/mnt/server8_hard1/donguk/checkpoints/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default=None)
    
    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    
    parser.add_argument("--data_path", type=str, default="/mnt/server18_hard0/jhjang/LVLM/crg/data/coco/val2014")
    parser.add_argument("--pope_path", type=str, default="/mnt/server8_hard1/donguk/rips2024/experiments/data/POPE/coco/coco_pope_random.json")
    parser.add_argument("--log_path", type=str, default="/mnt/server16_hard0/sangmin/code/neurips2024/logs/pope")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--use_ritual", type=str2bool, default=False)

    parser.add_argument("--use_vcd", type=str2bool, default=False)
    parser.add_argument("--noise_step", type=int, default=500)
    
    parser.add_argument("--use_m3id", type=str2bool, default=False)
    
    parser.add_argument("--ritual_alpha_pos", type=float, default=3)
    parser.add_argument("--ritual_alpha_neg", type=float, default=1)
    parser.add_argument("--ritual_beta", type=float, default=0.1)
    
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--experiment_index", type=int, default=0)

    args = parser.parse_args()
    return args


def print_acc(pred_list, label_list, logger):
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)
    # unknown_ratio = pred_list.count(2) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    logger.info('TP\tFP\tTN\tFN\t')
    logger.info('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)

    return acc, precision, recall, f1, yes_ratio


def recorder(out, pred_list):
    NEG_WORDS = ["No", "not", "no", "NO"]
    for line in out.split('\n'):

        line = line.replace('.', '')
        line = line.replace(',', '')
        words = line.split(' ')

        if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
            pred_list.append(0)
        else:
            pred_list.append(1)
    
    return pred_list


def main():
    args = parse_args()
    # Setup DDP:
    dist_util.setup_dist(args)
    device = dist_util.device()

    # Setup an experiment folder:
    if dist.get_rank() == 0:
        os.makedirs(
            args.log_path, exist_ok=True
        )  # Make results folder (holds all experiment subfolders)
        model_string_name = args.model_path.split("/")[-1]
        # experiment_index = len(glob(f"{args.log_path}/{model_string_name}/*")) + args.experiment_index
        # experiment_index = args.experiment_index
        # experiment_dir = f"{args.log_path}/{model_string_name}/{experiment_index:03d}"  # Create an experiment folder
        experiment_dir = f"{args.log_path}/{model_string_name}/{args.ritual_alpha_pos}_{args.ritual_alpha_neg}_{args.ritual_beta}"  # Create an experiment folder
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # ========================================
    #             Model & Dataset
    # ========================================
    logger.info('Initializing Model')

    #### for ritual
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

    pope_dataset = POPEDataSet(
        pope_path=args.pope_path, 
        data_path=args.data_path,
        trans=image_processor,
        model=args.model_base
    )
    pope_loader = torch.utils.data.DataLoader(
        pope_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        drop_last=False
    )

    # ==============================================
    #               Augmentations
    # ==============================================

    aug_dict = {
    'horizontal flip':v2.RandomHorizontalFlip(p=1),
    'vertical flip':v2.RandomVerticalFlip(p=1),
    'rotation':v2.RandomRotation(degrees=180),
    'color jitter':v2.ColorJitter(brightness=1, contrast=1, saturation=1, hue=0.5),
    'gaussian blur':v2.GaussianBlur(kernel_size=13, sigma=(1.5, 2.0)),
    'crop':v2.RandomResizedCrop(size=336),
    }
    
    # For statistics
    pos_aug_counter = {k:0 for k in aug_dict}
    pos_aug_counter.update({None: 0})

    # ========================================
    #            Start Generation
    # ========================================
    logger.info("Start eval...")
    pred_list, label_list = [], []
    for batch_id, data in tqdm(enumerate(pope_loader), total=len(pope_loader)):
        image = data["image"][0]
        qs = data["query"][0]
        label = data["label"]
        image_path = data["image_path"][0]
        label_list = label_list + list(label)

        image_pos = None
        image_neg = None

        if args.use_ritual:
            # ==============================================
            #              Image Transforms
            # ==============================================
            raw_image = Image.open(image_path)
            pos_aug = random.choice(list(aug_dict.keys()))

            if pos_aug is not None:
                raw_image_pos = aug_dict[pos_aug](raw_image)
                image_pos = image_processor.preprocess(raw_image_pos, return_tensor='pt')['pixel_values'][0] 
                image_pos = torch.tensor(image_pos)
        
            pos_aug_counter[pos_aug] += 1
            logger.info(f"RITUAL Transformation: {pos_aug}")

        elif args.use_vcd:
            image_neg = add_diffusion_noise(image, args.noise_step)
            
        
        
        # ==============================================
        #              Text prompt setting
        # ==============================================
        conv_out = Conversation(
            system="A chat between a curious human and an artificial intelligence assistant. "
                   "The assistant gives helpful, detailed, and polite answers to the human's questions.",
            roles=("USER", "ASSISTANT"),
            version="v1",
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.TWO,
            sep=" ",
            sep2="</s>",
        )
        
        qu_out = DEFAULT_IMAGE_TOKEN + '\n' + qs # for opera? setting
        # qu_out = DEFAULT_IMAGE_TOKEN + '\n' + qs + " Please answer this question with one word." # for VCD setting
        conv_out.append_message(conv_out.roles[0], qu_out)
        conv_out.append_message(conv_out.roles[1], None)
        prompt_out = conv_out.get_prompt()
        
        input_ids = tokenizer_image_token(prompt_out, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv_out.sep if conv_out.sep_style != SeparatorStyle.TWO else conv_out.sep2

        # ==============================================
        #                ritual method
        # ==============================================
        with torch.inference_mode():
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    images=image.unsqueeze(0).half().cuda(),
                    images_pos=(image_pos.unsqueeze(0).half().cuda() if image_pos is not None else None),
                    images_neg=(image_neg.unsqueeze(0).half().cuda() if image_neg is not None else None),
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    use_ritual=args.use_ritual,
                    use_vcd=args.use_vcd,
                    use_m3id=args.use_m3id,
                    ritual_alpha_pos=args.ritual_alpha_pos,
                    ritual_alpha_neg=args.ritual_alpha_neg,
                    ritual_beta=args.ritual_beta,
                )
                
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        pred_list = recorder(outputs, pred_list)

        logger.info(f"[VQA for ritual]")
        logger.info(f"V: {image_path}")
        logger.info(f"Q: {qs}")
        logger.info(f"A: {outputs}")
        if label == 1: logger.info(f"GT: Yes")
        elif label == 0: logger.info(f"GT: No")
        logger.info(f"="*50)

    if len(pred_list) != 0:
        logger.info(vars(args))
        # logger.info("Prompt for Aug:", prompt_aug)
        # logger.info("Prompt for ritual:", prompt_out)
        acc, precision, recall, f1, yes_ratio = print_acc(pred_list, label_list, logger)
        
        acc = round(acc*100,2)
        precision = round(precision*100,2)
        recall = round(recall*100,2)
        f1 = round(f1*100,2)
        yes_ratio = round(yes_ratio*100,2)
        
        logger.info(
            f"acc: {acc}, precision: {precision}, recall: {recall}, f1: {f1}, yes_ratio: {yes_ratio}"
        )
        if args.use_ritual:
            logger.info(f"RITUAL Transformation: {pos_aug_counter}")

if __name__ == "__main__":
    main()
