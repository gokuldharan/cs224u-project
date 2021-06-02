import torch
import clip
import torch.nn as nn

from pc.clip_classifier import ClipClassifier

clip_model, preprocess = clip.load("ViT-B/32")
model = ClipClassifier(clip_model, 1024, 0.0, 128, nn.ReLU, 0.0, 1, False, text_only=False)

print("loading model weights")
model.load_state_dict(torch.load("runs/clip/situated-OA/mscoco_imgs_clip_classifier.pt"))
model.eval()


