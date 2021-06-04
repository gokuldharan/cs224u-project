#Sourced from: https://github.com/jacobgil/pytorch-grad-cam/blob/master/usage_examples/vit_example.py

import argparse
import cv2
import numpy as np
import torch
from torchvision import models
import os
import pickle as pkl

from pytorch_grad_cam import GradCAM, \
                             ScoreCAM, \
                             GradCAMPlusPlus, \
                             AblationCAM, \
                             XGradCAM, \
                             EigenCAM, \
                             EigenGradCAM

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image

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
    parser.add_argument('--image-dir', type=str, default='./examples/both.png',
                        help='Input image dir')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true',
                        help='Reduce noise by taking the first principle componenet'
                        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')
    parser.add_argument('--model-path', type=str, default = '',
                        help='Path to saved ClipClassifier model')
                        

    args = parser.parse_args()

    if args.model_path == '':
        raise Exception(f"No model path specified")

    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args

def reshape_transform(tensor, height=7, width=7):
    result = tensor[1:, :  , :].reshape(tensor.size(1), 
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

class ClipInput():
    def __init__(self, text, images):
        self.text = text
        self.images = images

if __name__ == '__main__':
    """ python vit_gradcam.py -image-path <path_to_image>
    Example usage of using cam-methods on a VIT network.
        
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM, 
         "scorecam": ScoreCAM, 
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")
    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    model, preprocess = ClipClassifierDebug.loadNonJitFromFolder(args.model_path, text_only=False)
    model.eval()

    if args.use_cuda:
        model = model.cuda()

    lastBlock = None
    for block in model.clip_model.visual.transformer.resblocks.children():
        lastBlock = block

    target_layer = lastBlock.ln_1

    
    cam = methods[args.method](model=model, 
                               target_layer=target_layer,
                               use_cuda=args.use_cuda,
                               reshape_transform=reshape_transform)



    device = torch.cuda.current_device()
    gan_data = ClipDataset(Task.Situated_AffordancesProperties, preprocess, False, gan_imgs=True, text_only=False)
    img_data = ClipDataset(Task.Situated_AffordancesProperties, preprocess, False, gan_imgs=False, text_only=False)
    sent_idx_to_image = pkl.load(open("data/clip/sent_idx_to_image.pkl", "rb"))

    with open("data/sentences/sentences.txt", "r") as f:
            all_sentences = [line.strip() for line in f.readlines()]

    for i, data in enumerate(img_data):
        image_file = img_data.images[i]
        img_name = image_file.split('/')[-1]
        if img_name == 'COCO_val2014_000000079229.jpg': #Horse!
            print(all_sentences[img_data.line_mapping[data["label"]]])

    img_train_data = ClipDataset(Task.Situated_AffordancesProperties, preprocess, True, gan_imgs=False, text_only=False)

    for i, data in enumerate(img_train_data):
        image_file = img_data.images[i]
        img_name = image_file.split('/')[-1]
        assert(img_name != 'COCO_val2014_000000079229.jpg') #Horse!

    for id in list(range(0, 10, 1)):
        y = torch.from_numpy(gan_data[id]["y"]).to(device, dtype=torch.half)
        input_ids = torch.unsqueeze(gan_data[id]["input_ids"],0).to(device)
        input_images = torch.unsqueeze(gan_data[id]["input_image"],0).to(device)
        image_file = gan_data.images[id]
        img_id = image_file.split('/')[-1].rstrip('.png')


        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested category.
        target_category = None

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 1

        grayscale_cam = cam([input_ids, input_images],
                            target_category=None,
                            eigen_smooth=args.eigen_smooth,
                            aug_smooth=args.aug_smooth)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]
        

        rgb_img = cv2.imread(image_file, 1)[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, (224, 224))
        rgb_img = np.float32(rgb_img) / 255

        cam_image = show_cam_on_image(rgb_img, grayscale_cam)
        save_path = os.path.join("visualizations", f'{args.method}_{img_id}.png')
        cv2.imwrite(save_path, cam_image)