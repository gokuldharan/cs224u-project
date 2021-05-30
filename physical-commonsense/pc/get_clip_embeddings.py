import clip
import json
import torch
import numpy as np
import pandas as pd
import pickle as pkl

from PIL import Image
from tqdm import tqdm
from pc.data import Task, _uids2sentidx, _expand

NUM_PROPS = 50
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def get_situated_clip_full_emb():
    # Load and save clip sentence features
    sent_feats = get_situated_clip_text_emb()
    imgid2file = id_to_file_map()

    prop_img_feats = get_image_feats("data/pc/situated-properties.csv", imgid2file, "data/clip/prop_img_feats.npz")
    aff_img_feats = get_image_feats("data/pc/situated-affordances-sampled.csv", imgid2file, "data/clip/aff_img_feats.npz")

    #sent_feats = np.load("data/clip/sentences.clip_text.npz")["matrix"]
    #imgid2file = pkl.load(open("data/mscoco/mscoco_id_to_file.pkl", "rb"))

    #prop_img_feats = np.load("data/clip/prop_img_feats.npz")["matrix"]
    #aff_img_feats = np.load("data/clip/aff_img_feats.npz")["matrix"]

    label_to_idx = {}
    merge_prop_img_sent_feats(prop_img_feats, sent_feats, label_to_idx)
    merge_aff_img_sent_feats(aff_img_feats, sent_feats, label_to_idx)
    merge_aff_prop_img_sent_feats(aff_img_feats, sent_feats, label_to_idx)

    pkl.dump(label_to_idx, open("data/clip/labels_to_idx.pkl", "wb"))
    

def get_situated_clip_text_emb(batch_size=64):
    print("Getting sentence embeddings")
    sents = open("data/sentences/sentences.txt", "r").readlines()
    tokenized_sents = clip.tokenize(sents).to(device)

    sent_features = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sents), batch_size)):
            batch = tokenized_sents[i:batch_size+i]
            feats = model.encode_text(batch)
            sent_features.extend(feats.cpu().numpy())

    print("Saving file")
    np.savez("data/clip/sentences.clip_text.npz", matrix=sent_features)

    return sent_features
        

# Map image filenames to sentence idxs
def id_to_file_map():
    print("Making MSCOCO ID to filename map")
    # Load image meta data
    coco_meta_data = json.load(open("data/mscoco/annotations/instances_train2014.json", "rb"))["images"]
    meta_data2 = json.load(open("data/mscoco/annotations/instances_val2014.json", "rb"))["images"]
    coco_meta_data.extend(meta_data2)

    # Map image ID to image filename
    imgid2file = {}
    for img in tqdm(coco_meta_data):
        imgid = img["id"]
        filename = img["file_name"]
        imgid2file[imgid] = filename

    pkl.dump(imgid2file, open("data/mscoco/mscoco_id_to_file.pkl", "wb"))

    return imgid2file


def get_image_feats(csv_path, imgid2file, save_name):
    print(f"Getting image features for images in {csv_path}")
    df = pd.read_csv(csv_path, index_col="objectUID")
    image_features = []

    for _, row in tqdm(df.iterrows()):
        image_id = row["cocoImgID"]
        filename = imgid2file[image_id]
        image = preprocess(Image.open(f"data/mscoco/images/{filename}")).unsqueeze(0).to(device)

        with torch.no_grad():
            features = model.encode_image(image)
            image_features.append(features.cpu().numpy())

    np.savez(save_name, matrix=image_features)

    return image_features

def merge_prop_img_sent_feats(image_feats, sent_feats, label_to_idx):
    print("Mergeing image and sent feats for object property")
    # Need to get subset of sentence indices that map to each image
    label_df = pd.read_csv("data/pc/situated-properties.csv", index_col="objectUID").drop(
        columns=["cocoImgID", "cocoAnnID"]
    )

    image_idxs = []
    rows = [row for row, _ in label_df.iterrows()]
    cols = label_df.columns.to_list()
    labels = []
    for row_idx, row in enumerate(rows):
        for col in cols:
            labels.append("{}/{}".format(row, col))
            image_idxs.append(row_idx)

    sent_idxs = _uids2sentidx(Task.Situated_ObjectsProperties, labels)

    # Match sentences to images of objects
    assert len(sent_idxs) == len(image_idxs)
    full_feats = [np.concatenate((sent_feats[sent_idxs[i]], np.squeeze(image_feats[image_idxs[i]]))) for i in range(len(labels))]
    label_to_idx[Task.Situated_ObjectsProperties] = {}

    for label_idx, label in enumerate(labels):
        label_to_idx[Task.Situated_ObjectsProperties][label] = label_idx

    print("Saving file")
    np.savez("data/clip/sentences.clip_obj_prop_full.npz", matrix=full_feats)


def merge_aff_img_sent_feats(image_feats, sent_feats, label_to_idx):
    print("Mergeing image and sent feats for object affordance")
    df = pd.read_csv(
        "data/pc/situated-affordances-sampled.csv", index_col="objectUID"
    ).drop(columns=["cocoImgID", "cocoAnnID", "objectHuman"])

    labels = []
    image_idxs = []
    for row_idx, (_, row) in enumerate(df.iterrows()):
        obj = row.name

        # record positive examples
        for aff_yes in row["affordancesYes"].split(","):
            labels.append("{}/{}".format(obj, aff_yes))
            image_idxs.append(row_idx)

        # record negative examples
        for aff_no in row["affordancesNo"].split(","):
            labels.append("{}/{}".format(obj, aff_no))
            image_idxs.append(row_idx)

    sent_idxs = _uids2sentidx(Task.Situated_ObjectsAffordances, labels)

    assert len(sent_idxs) == len(image_idxs)
    full_feats = [np.concatenate((sent_feats[sent_idxs[i]], np.squeeze(image_feats[image_idxs[i]]))) for i in range(len(labels))]
    label_to_idx[Task.Situated_ObjectsAffordances] = {}

    for label_idx, label in enumerate(labels):
        label_to_idx[Task.Situated_ObjectsAffordances][label] = label_idx

    print("Saving file")
    np.savez("data/clip/sentences.clip_obj_aff_full.npz", matrix=full_feats)


def merge_aff_prop_img_sent_feats(image_feats, sent_feats, label_to_idx):
    print("Mergeing image and sent feats for property affordance")
    aff_df = pd.read_csv(
        "data/pc/situated-affordances-sampled.csv", index_col="objectUID"
    ).drop(columns=["cocoImgID", "objectHuman"])
    prop_df = pd.read_csv(
        "data/pc/situated-properties.csv", index_col="objectUID"
    ).drop(columns=["cocoImgID"])

    image_idxs = []
    props = list(prop_df.drop(columns=["cocoAnnID"]).columns)
    labels = []

    # for each object, pick its 3 affordances. set the same property vector for each
    # of those 3 affordances.
    for i, (_, aff_row) in enumerate(aff_df.iterrows()):
        prop_data = (
            prop_df[prop_df["cocoAnnID"] == aff_row["cocoAnnID"]]
            .drop(columns=["cocoAnnID"])
            .to_numpy()
            .squeeze()
        )

        for j, aff in enumerate(aff_row["affordancesYes"].split(",")):
            for prop in props:
                labels.append("{}/{}".format(aff, prop))
                image_idxs.append(i)

    sent_idxs = _uids2sentidx(Task.Situated_AffordancesProperties, labels)

    assert len(sent_idxs) == len(image_idxs)
    full_feats = [np.concatenate((sent_feats[sent_idxs[i]], np.squeeze(image_feats[image_idxs[i]]))) for i in range(len(labels))]
    label_to_idx[Task.Situated_AffordancesProperties] = {}

    for label_idx, label in enumerate(labels):
        label_to_idx[Task.Situated_AffordancesProperties][label] = label_idx

    print("Saving file")
    np.savez("data/clip/sentences.clip_aff_prop_full.npz", matrix=full_feats)

get_situated_clip_full_emb()
