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
    parser.add_argument('--save-dir', type=str, default='./visualizations/',
                        help='Save dir')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true',
                        help='Reduce noise by taking the first principle componenet'
                        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')
    parser.add_argument('--task', type=str, default='AP',
                        help='Can be AP/OP/OA')
    parser.add_argument('--gan-model-path', type=str, default = '',
                        help='Path to saved ClipClassifier model')
    parser.add_argument('--coco-model-path', type=str, default = '',
                        help='Path to saved ClipClassifier model')
    parser.add_argument('--sent-idx', type=int, default = '54292',
                        help='sentence index')                  

    args = parser.parse_args()

    #if args.gan_model_path == '' or args.coco_model_path == '':
    #    raise Exception(f"No model path specified for either coco or gan")

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
         
    tasks = \
        {"AP": Task.Situated_AffordancesProperties, 
         "OP": Task.Situated_ObjectsProperties, 
         "OA": Task.Situated_ObjectsAffordances}


    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")
    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    with open("data/sentences/sentences.txt", "r") as f:
            all_sentences = [line.strip() for line in f.readlines()]
            
    #Temporary convenience hack
    gan_model_path = os.path.join("data","situated-"+args.task,"gan_imgs_clip_classifier")
    coco_model_path = os.path.join("data","situated-"+args.task,"mscoco_imgs_clip_classifier")
    text_model_path = os.path.join("data","situated-"+args.task,"text_only_clip_classifier")
    gan_model, preprocess1 = ClipClassifierDebug.loadNonJitFromFolder(gan_model_path, text_only=False)
    coco_model, preprocess2 = ClipClassifierDebug.loadNonJitFromFolder(coco_model_path, text_only=False)
    text_model, _ = ClipClassifierDebug.loadNonJitFromFolder(text_model_path, text_only=True)
    gan_model.eval()
    coco_model.eval()


    outF = open(args.save_dir + "/results.txt", "a+")

    if args.use_cuda:
        gan_model = gan_model.cuda()
        coco_model = coco_model.cuda()

    lastBlock = None
    for block in coco_model.clip_model.visual.transformer.resblocks.children():
        lastBlock = block

    coco_target_layer = lastBlock.ln_1

    lastBlock = None
    for block in gan_model.clip_model.visual.transformer.resblocks.children():
        lastBlock = block

    gan_target_layer = lastBlock.ln_1

    
    coco_cam = methods[args.method](model=coco_model, 
                               target_layer=coco_target_layer,
                               use_cuda=args.use_cuda,
                               reshape_transform=reshape_transform)

    gan_cam = methods[args.method](model=gan_model, 
                               target_layer=gan_target_layer,
                               use_cuda=args.use_cuda,
                               reshape_transform=reshape_transform)

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    coco_cam.batch_size = 1
    gan_cam.batch_size = 1


    device = torch.cuda.current_device()
    gan_data = ClipDataset(tasks[args.task], preprocess1, False, gan_imgs=True, text_only=False) 
    img_data = ClipDataset(tasks[args.task], preprocess2, False, gan_imgs=False, text_only=False)
    sent_idx = args.sent_idx #54292 for horse!
    print(gan_data.task_idxs)
    id = gan_data.sent_idx_to_dataset_id[sent_idx]
    assert(id == img_data.sent_idx_to_dataset_id[sent_idx])
    y = torch.from_numpy(gan_data[id]["y"]).to(device, dtype=torch.half)
    input_ids = torch.unsqueeze(gan_data[id]["input_ids"],0).to(device)
    input_images = torch.unsqueeze(gan_data[id]["input_image"],0).to(device)
    image_file = gan_data.images[id]
    img_name = image_file.split('/')[-1]
    output = gan_model([input_ids, input_images]).data.tolist()[0][0]
    gold = gan_data[id]["y"][0]
    line = all_sentences[sent_idx] + " gan pred = " + str(output) + " gold = " + str(gold) + "\n"
    outF.write(line)


    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None



    grayscale_cam = gan_cam([input_ids, input_images],
                        target_category=None,
                        eigen_smooth=args.eigen_smooth,
                        aug_smooth=args.aug_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]
    

    rgb_img = cv2.imread(image_file, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    save_path = os.path.join(args.save_dir, f'{args.method}_{img_name}')
    cv2.imwrite(save_path, cam_image)



    y = torch.from_numpy(img_data[id]["y"]).to(device, dtype=torch.half)
    input_ids = torch.unsqueeze(img_data[id]["input_ids"],0).to(device)
    input_images = torch.unsqueeze(img_data[id]["input_image"],0).to(device)
    image_file = img_data.images[id]
    img_name = image_file.split('/')[-1]
    output = coco_model([input_ids, input_images]).data.tolist()[0][0]
    gold = img_data[id]["y"][0]
    output_t = text_model([input_ids, None]).data.tolist()[0][0]
    line = all_sentences[sent_idx] + " coco pred = " + str(output) + " gold = " + str(gold) + "\n"
    outF.write(line)
    line = all_sentences[sent_idx] + " text pred = " + str(output_t) + " gold = " + str(gold) + "\n"
    outF.write(line)

    grayscale_cam = coco_cam([input_ids, input_images],
                        target_category=None,
                        eigen_smooth=args.eigen_smooth,
                        aug_smooth=args.aug_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]
    

    rgb_img = cv2.imread(image_file, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    save_path = os.path.join(args.save_dir, f'{args.method}_{img_name}')
    cv2.imwrite(save_path, cam_image)


    

    outF.close()