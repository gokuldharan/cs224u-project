import random

random.seed(224)
val_split_total = 16
train_split_uids = open("data/pc/situated-train-object-uids.txt", "r")
train_split_uids = [line for line in train_split_uids.readlines()]


dev_val_uids = random.sample(train_split_uids, val_split_total)
dev_train_uids = [uid for uid in train_split_uids if uid not in dev_val_uids]

with open("data/pc/situated-dev-train-object-uids.txt", "w") as f:
    for uid in dev_train_uids:
        f.write(uid)
    f.close()

with open("data/pc/situated-dev-val-object-uids.txt", "w") as f:
    for uid in dev_val_uids:
        f.write(uid)
    f.close()
