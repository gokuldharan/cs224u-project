import os
import torch
import clip

import torch.nn as nn
import tarfile

from pc.models import mlp, init_weights

class ClipClassifier(nn.Module):
    
    def __init__(self, 
                 clip_model, 
                 d_in=None, 
                 input_dropout=None, 
                 h=None, 
                 activation=None, 
                 innner_dropout=None, 
                 d_out=None, 
                 train=None, 
                 text_only=None, 
                 mlp_model=None):

        super(ClipClassifier, self).__init__()
        self.text_only = text_only
        self.clip_model = clip_model

        if mlp_model: 
            self.mlp = mlp_model
        else:
            self.mlp = mlp(d_in, input_dropout, h, activation, innner_dropout, d_out).half()

    def forward(self, text, image=None):
        text_feats = self.clip_model.encode_text(text)

        if not self.text_only:
            image_feats = self.clip_model.encode_image(image)
            mlp_input = torch.cat([text_feats, image_feats], dim=1)
        else:
            mlp_input = text_feats

        pred = self.mlp(mlp_input)

        return pred

    def save(self, path):
        path_list = path.split("/")
        dir_path, filename = path_list[:-1], path_list[-1]
        dir_path = "/".join(dir_path)
        clip_model_path = os.path.join(dir_path, "clip_model.pt")
        mlp_model_path = os.path.join(dir_path, "mlp_model.pt")
        
        # save clip model
        self.clip_model.save(clip_model_path)

        # save mlp
        torch.save(self.mlp, mlp_model_path)

        # combine model files into one tar
        with tarfile.open(path, "w:gz") as tar:
            tar.add(clip_model_path)
            tar.add(mlp_model_path)

        tar.close()
        
        # remove model files that are outside the tar file
        os.remove(clip_model_path)
        os.remove(mlp_model_path)
    
    @classmethod
    def load(cls, path, text_only=False):
        path_list = path.split("/")
        dir_path, filename = path_list[:-1], path_list[-1]
        dir_path = "/".join(dir_path)
        clip_model_path = os.path.join(dir_path, "clip_model.pt")
        mlp_model_path = os.path.join(dir_path, "mlp_model.pt")

        # extract model weights from tar file
        tar = tarfile.open(path, "r:gz")
        tar.extractall()
        tar.close()

        # load clip model
        clip_model = torch.jit.load(clip_model_path)
        mlp_model = torch.load(mlp_model_path)
        model = cls(clip_model, text_only=text_only, mlp_model=mlp_model)

        return model

    @classmethod
    def loadNonJitFromFolder(cls, path, text_only=False):
        clip_model_path = os.path.join(path, "clip_model.pt")
        mlp_model_path = os.path.join(path, "mlp_model.pt")

        # load clip model
        clip_model_jit = torch.jit.load(clip_model_path)
        clip_model, _ = clip.load("ViT-B/32", jit=False)
        states = clip_model_jit.state_dict()
        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in states:
                del states[key]
        clip_model.load_state_dict(states)
        mlp_model = torch.load(mlp_model_path)
        model = cls(clip_model, text_only=text_only, mlp_model=mlp_model)

        return model



def get_clip_classifier(d_in, input_dropout, h, activation, innner_dropout, d_out, train=True, text_only=False):
    clip_model, preprocess = clip.load("ViT-B/32", jit=False)
    classifier = ClipClassifier(clip_model, d_in, input_dropout, h, activation, innner_dropout, d_out, train, text_only) 
    
    return classifier, preprocess




### This class is a non-JIT version of CLIP Classifier that is easier to use with probing/visualization tools
class ClipClassifierDebug(ClipClassifier):
    def __init__(self, 
                 clip_model, 
                 d_in=None, 
                 input_dropout=None, 
                 h=None, 
                 activation=None, 
                 innner_dropout=None, 
                 d_out=None, 
                 train=None, 
                 text_only=None, 
                 mlp_model=None):

        super(ClipClassifierDebug, self).__init__(clip_model, 
                 d_in=d_in, 
                 input_dropout=input_dropout, 
                 h=h, 
                 activation=activation, 
                 innner_dropout=innner_dropout, 
                 d_out=d_out, 
                 train=train, 
                 text_only=text_only, 
                 mlp_model=mlp_model)
    #Assumes Text and Image concatenated in one list along dimension 0
    def forward(self, x):
        text = x[0]
        image = x[1]
        text_feats = self.clip_model.encode_text(text)

        if not self.text_only:
            image_feats = self.clip_model.encode_image(image)
            mlp_input = torch.cat([text_feats, image_feats], dim=1)
        else:
            mlp_input = text_feats

        pred = self.mlp(mlp_input)

        return pred

    @classmethod
    def loadNonJitFromFolder(cls, path, text_only=False):
        clip_model_path = os.path.join(path, "clip_model.pt")
        mlp_model_path = os.path.join(path, "mlp_model.pt")

        # load clip model
        clip_model_jit = torch.jit.load(clip_model_path)
        clip_model, preprocess = clip.load("ViT-B/32", jit=False)
        states = clip_model_jit.state_dict()
        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in states:
                del states[key]
        clip_model.load_state_dict(states)
        mlp_model = torch.load(mlp_model_path)
        model = cls(clip_model, text_only=text_only, mlp_model=mlp_model)

        return model, preprocess