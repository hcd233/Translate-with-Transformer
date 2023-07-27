import torch
import json
import tqdm
import time
import os.path
from datetime import datetime

import torch.nn.functional as F
import torch.utils.data as Data
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from model.transformer import Transformer
from utils import *

# HyperParameters
EPOCH = 8
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
VOCAB_SIZE = 65003
MAX_SEQ_LEN = 50
PAD_IDX = 65000
EMBEDDING_SIZE = 1024
NUM_HEADS = 16
NUM_LAYERS = 18
DROPOUT = 0.1
PRINT_PER_BATCH_NUM = 200
SAVE_PER_EPOCH_RATIO = 0.2
WARMUP_RATIO = 0.2
WEIGHT_DECAY = 2e-4

NUM_GPUS = 8
DEVICE = torch.device("cuda")

MODEL = Transformer(src_vocab_size=VOCAB_SIZE,
                    trg_vocab_size=VOCAB_SIZE,
                    src_pad_idx=PAD_IDX,
                    trg_pad_idx=PAD_IDX,
                    embed_size=EMBEDDING_SIZE,
                    heads=NUM_HEADS,
                    num_layers=NUM_LAYERS,
                    dropout=DROPOUT,
                    device=DEVICE,
                    max_length=MAX_SEQ_LEN).to(DEVICE)

CHECKPOINTS = "./checkpoints"
LOGS = None

src_pe_path = "./vocab/source.spm"
tgt_pe_path = "./vocab/target.spm"

train_src_path = "./dataset/train.en.txt"
train_tgt_path = "./dataset/train.ch.txt"

dev_src_path = "./dataset/dev.en.txt"
dev_tgt_path = "./dataset/dev.ch.txt"

vocab_path = "./vocab/vocab.json"
accelerator = Accelerator(project_dir=CHECKPOINTS)


def print_logs(msg):
    print(msg)
    with open(LOGS, mode="a+", encoding="utf-8") as log:
        log.write(msg + "\n")


for i in range(0xFF):
    try_log = f"./logs/log_{i}.txt"
    if not os.path.exists(try_log):
        LOGS = try_log
        break

if not os.path.exists(CHECKPOINTS):
    os.mkdir(CHECKPOINTS)
else:
    ckpts = os.listdir(CHECKPOINTS)
    if len(ckpts) != 0:
        accelerator.load_state(CHECKPOINTS)
        print_logs(f"Load Checkpoints from {CHECKPOINTS}")


# MODEL = nn.parallel.DistributedDataParallel(MODEL, device_ids=DEVICES, output_device=DEVICES[0],)


class SeqDataset(Data.Dataset):
    def __init__(self, src, tgt):
        super(SeqDataset, self).__init__()
        self.src = src
        self.tgt = tgt

    def __getitem__(self, idx):
        return torch.tensor(self.src[idx]), torch.tensor(self.tgt[idx])

    def __len__(self):
        return len(self.src)


def load_dataset(src_path, tgt_path):
    with open(src_path, mode='r', encoding='utf-8') as file:
        src = file.read().split("\n")[:-1]  # test [:-1]

    with open(tgt_path, mode='r', encoding='utf-8') as file:
        tgt = file.read().split("\n")[:-1]

    return sp.BatchProcess(src, tgt)


def print_configuration():
    print_logs("\nTraining Configuration")
    print_logs("\tdevices:{}".format(DEVICE))
    print_logs("\tlearning rate:{:.6f}".format(LEARNING_RATE))
    print_logs("\tbatch size:{}".format(BATCH_SIZE))
    print_logs("\tvocab size:{}".format(VOCAB_SIZE))
    print_logs("\tmax length:{}".format(MAX_SEQ_LEN))
    print_logs("\twarmup ratio:{:.2f}".format(WARMUP_RATIO))
    print_logs("\tcheckpoints:{}".format(CHECKPOINTS))

    print_logs("model Configuration")
    print_logs("\thead number:{}".format(NUM_HEADS))
    print_logs("\tembedding size:{}".format(EMBEDDING_SIZE))
    print_logs("\tlayer number:{}".format(NUM_LAYERS))

    print_logs("Dataset Configuration")
    print_logs("\ttrain sequence number:{}".format(len(train_dl) * BATCH_SIZE))
    print_logs("\ttrain batch number:{}".format(len(train_dl)))
    print_logs("\tvalidate sequence number:{}".format(len(dev_dl) * BATCH_SIZE))
    print_logs("\tvalidate batch number:{}".format(len(dev_dl)))


with open(vocab_path, mode='r', encoding='utf-8') as file:
    enc_vocab = json.load(file)
    dec_vocab = {v: k for k, v in enc_vocab.items()}

# Initialize utils class

sp = SPModel()
sp.LoadModel(src_path=src_pe_path, tgt_path=tgt_pe_path)

se = SeqEncoder(enc_vocab=enc_vocab, dec_vocab=dec_vocab, max_seq_len=MAX_SEQ_LEN)

train_src, train_tgt = load_dataset(src_path=train_src_path, tgt_path=train_tgt_path)
dev_src, dev_tgt = load_dataset(src_path=dev_src_path, tgt_path=dev_tgt_path)

# Preprocess data
half = len(train_src) // 2
train_ds = SeqDataset(src=se.SrcEncode(train_src[:half] + train_tgt[half:]),
                      tgt=se.TgtEncode(train_tgt[:half], tgt_lang="zh") + se.TgtEncode(train_src[half:], tgt_lang="en"))
half = len(dev_src) // 2
dev_ds = SeqDataset(src=se.SrcEncode(dev_src[:half] + dev_tgt[half:]),
                    tgt=se.TgtEncode(dev_tgt[:half], tgt_lang="zh") + se.TgtEncode(dev_src[half:], tgt_lang="en"))
# train_ds
train_dl = Data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
dev_dl = Data.DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
del train_ds, dev_ds, train_src, train_tgt, dev_src, dev_tgt


def main(model, train_dataloader, dev_dataloader):
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(len(train_dataloader) * WARMUP_RATIO),
        num_training_steps=len(train_dataloader) * EPOCH
    )
    # accelerate
    MODEL = accelerator.prepare_model(model=model)
    optimizer = accelerator.prepare_optimizer(optimizer)
    train_dl = accelerator.prepare_data_loader(train_dataloader)
    dev_dl = accelerator.prepare_data_loader(dev_dataloader)
    lr_scheduler = accelerator.prepare_scheduler(lr_scheduler)

    accelerator.register_for_checkpointing(lr_scheduler)
    DEVICE = accelerator.device
    MODEL.to(DEVICE)

    epoch = tqdm.tqdm(range(EPOCH))
    epoch.set_description("EPOCH")

    print_configuration()

    for e in epoch:

        train_dl = tqdm.tqdm(train_dl)
        dev_dl = tqdm.tqdm(dev_dl)
        train_dl.set_description("Training")
        dev_dl.set_description("Validating")

        # train
        avg_train_loss = 0.
        save_ratio = int(len(train_dl) * SAVE_PER_EPOCH_RATIO)

        MODEL.train()
        for batch_idx, (src, tgt) in enumerate(train_dl):
            optimizer.zero_grad()

            output = MODEL(src, tgt[:, :-1])

            output = output.reshape(-1, output.shape[2])
            tgt = tgt[:, 1:].reshape(-1)

            loss = F.cross_entropy(output, tgt, ignore_index=PAD_IDX)
            avg_train_loss += loss.item()
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()

            if batch_idx % PRINT_PER_BATCH_NUM == 0 and batch_idx != 0:
                avg_train_loss /= PRINT_PER_BATCH_NUM
                # f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch:{epoch} Batch:{batch_idx} Loss:{loss}"
                print_logs(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch:{e} Batch:{batch_idx} Loss:{avg_train_loss}")
                avg_train_loss = 0.

            # save checkpoints
            if batch_idx % save_ratio == 0 and batch_idx != 0 or batch_idx == len(train_dl) - 1:
                ckpt_name = CHECKPOINTS + "/pytorch_model_epoch{}_batch_idx{}_loss{:.4f}.bin".format(e, batch_idx,
                                                                                                     loss.item())
                accelerator.save(MODEL.state_dict(), ckpt_name)
                print_logs(f"save to {ckpt_name}")

            time.sleep(3e-3)
        # validate
        avg_dev_loss = 0.
        MODEL.eval()
        for src, tgt in dev_dl:
            with torch.no_grad():
                output = MODEL(src, tgt[:, :-1])
                output = output.reshape(-1, output.shape[2])
                tgt = tgt[:, 1:].reshape(-1)

                loss = F.cross_entropy(output, tgt, ignore_index=PAD_IDX)
                avg_dev_loss += loss.item()

        avg_dev_loss /= len(dev_dl)
        print_logs(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch:{e}, Validation Loss:{avg_dev_loss}")


if __name__ == '__main__':
    main(MODEL, train_dl, dev_dl)
