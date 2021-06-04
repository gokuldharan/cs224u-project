import argparse
import cv2
import numpy as np
import torch
from torchvision import models
import os
import pickle as pkl
from pc.clip_classifier import ClipClassifierDebug
from pc.finetune_clip import ClipDataset
from pc.data import (
    Task,
    get,
    TASK_SHORTHAND,
    TASK_MEDIUMHAND,
    TASK_REV_MEDIUMHAND,
    TASK_LABELS,
)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-name', type=str, default='',
                        help='Input image dir')
    parser.add_argument('--model-path', type=str, default = '',
                        help='Path to saved ClipClassifier model')
    parser.add_argument('--gan', action='store_true', default=False)
    parser.add_argument('--coco', action='store_true', default=False)
    parser.add_argument('--text', action='store_true', default=False)
                        

    args = parser.parse_args()

    if args.model_path == '':
        raise Exception(f"No model path specified")
    
    if args.gan + args.coco + args.text in [0, 2, 3]:
        raise Exception(f"Specify one model type")



    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args

if __name__ == '__main__':
    """ python vit_gradcam.py -image-path <path_to_image>
    Example usage of using cam-methods on a VIT network.
        
    """

    args = get_args()

    if args.image_name == '':
        print("No image specified, looking for horses!")
        image_name = 'COCO_val2014_000000079229.jpg'
    else:
        image_name = args.image_name

    model, preprocess = ClipClassifierDebug.loadNonJitFromFolder(args.model_path, text_only=False)
    model.eval()

    if args.use_cuda:
        model = model.cuda()
    
    device = torch.cuda.current_device()

    if args.coco:
        img_data = ClipDataset(Task.Situated_AffordancesProperties, preprocess, False, gan_imgs=False, text_only=False)
        flav = "coco"
    elif args.gan:
        img_data = ClipDataset(Task.Situated_AffordancesProperties, preprocess, False, gan_imgs=True, text_only=False)
        flav = "gan"
    else:
        img_data = ClipDataset(Task.Situated_AffordancesProperties, preprocess, False, gan_imgs=True, text_only=False)
        flav = "text"

    #sent_idx_to_image = pkl.load(open("data/clip/sent_idx_to_image.pkl", "rb"))

    with open("data/sentences/sentences.txt", "r") as f:
            all_sentences = [line.strip() for line in f.readlines()]

    outF = open(flav + "_AP_probe.txt", "w")
    testimgs = open("test_imgs.txt", "w")
    seen_imgs = set()
    seen_sents = set()
    cnt, total = 0, 0
    for i, data in enumerate(img_data):
        image_file = img_data.images[i]
        img_name = image_file.split('/')[-1]
        if img_name not in seen_imgs:
            seen_imgs.add(img_name)
            testimgs.write(img_name + "\n")
        if img_name == image_name: #Horse!
            sent_idx =  img_data.line_mapping[data["label"]]
            if sent_idx not in seen_sents:
                seen_sents.add(sent_idx)
                input_ids = torch.unsqueeze(data["input_ids"],0).to(device)
                input_image = torch.unsqueeze(data["input_image"],0).to(device)
                output = model([input_ids, input_image]).data.tolist()[0][0]
                gold = data["y"][0]
                if round(output) == gold:
                    cnt += 1
                line = all_sentences[sent_idx] + " pred = " + str(output) + " gold = " + str(gold) + "\n"
                total += 1
                outF.write(line)
    print("Accuracy = " + str(cnt / total))
    testimgs.close()
    outF.close()
    

    gan_data = ClipDataset(Task.Situated_AffordancesProperties, preprocess, False, gan_imgs=True, text_only=False)