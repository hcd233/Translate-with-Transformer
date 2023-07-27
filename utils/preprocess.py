import os
import random

import tqdm

save_dir = "./processed"

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

train_ds = {
    "en": "./dataset/train.en.txt",
    "zh": "./dataset/train.ch.txt"
}

dev_ds = {
    "en": "./dataset/dev.en.txt",
    "zh": "./dataset/dev.ch.txt"
}


def preprocess(ds, save_path):
    with open(ds["en"], "r", encoding="utf-8") as en_file, \
            open(ds["zh"], "r", encoding="utf-8") as zh_file:
        en = en_file.read().replace("\t", "").split("\n")[:-1]
        zh = zh_file.read().replace("\t", "").split("\n")[:-1]
        assert len(en) == len(zh)

    nums = len(en)

    random.seed(114514)
    pair = list(zip(en, zh))
    random.shuffle(pair)

    with open(save_path, "w+", encoding='utf-8') as file:
        for en, zh in tqdm.tqdm(pair[:nums // 2]):
            file.write(en + "\t" + "<en2zh>" + zh + "<end>" + "\n")

        for en, zh in tqdm.tqdm(pair[nums // 2:]):
            file.write(zh + "\t" + "<zh2en>" + en + "<end>" + "\n")


if __name__ == '__main__':
    preprocess(train_ds, save_dir + "/train.txt")
    preprocess(dev_ds, save_dir + "/dev.txt")
