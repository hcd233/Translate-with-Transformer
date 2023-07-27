import torch
import json
from model.transformer import Transformer
from utils import *

# HyperParameters
BATCH_SIZE = 64
VOCAB_SIZE = 65002
MAX_SEQ_LEN = 40
PAD_IDX = 65000
EMBEDDING_SIZE = 768
NUM_HEADS = 8
DROPOUT = 0.1
PRINT_PER_BATCH_NUM = 100
SAVE_PER_EPOCH_RATIO = 0.2
WARMUP_RATIO = 0.5
NUM_GPUS = 8
DEVICES = [torch.device(f"cuda:{i}") for i in range(min(NUM_GPUS, torch.cuda.device_count()))] + [torch.device("cpu")]

src_pe_path = "./vocab/source.spm"
tgt_pe_path = "./vocab/target.spm"

train_src_path = "./dataset/train.en.txt"
train_tgt_path = "./dataset/train.ch.txt"

dev_src_path = "./dataset/dev.en.txt"
dev_tgt_path = "./dataset/dev.ch.txt"

vocab_path = "./vocab/vocab.json"

checkpoint = "./checkpoints/pytorch_model_epoch3_batch_idx44999_loss2.8455.bin"

with open(vocab_path, mode='r', encoding='utf-8') as file:
    enc_vocab = json.load(file)
    dec_vocab = {v: k for k, v in enc_vocab.items()}

# Initialize utils class

sp = SPModel()

sp.LoadModel(src_path=src_pe_path, tgt_path=tgt_pe_path)

se = SeqEncoder(enc_vocab=enc_vocab, dec_vocab=dec_vocab, max_seq_len=MAX_SEQ_LEN)

MODEL = Transformer(src_vocab_size=VOCAB_SIZE,
                    trg_vocab_size=VOCAB_SIZE,
                    src_pad_idx=PAD_IDX,
                    trg_pad_idx=PAD_IDX,
                    embed_size=EMBEDDING_SIZE,
                    num_layers=NUM_HEADS,
                    dropout=DROPOUT,
                    device=DEVICES[0],
                    max_length=MAX_SEQ_LEN).to(DEVICES[0])

MODEL.load_state_dict(torch.load(checkpoint, map_location=DEVICES[0]))


def preprocess(src_seq, tgt_lang) -> torch.tensor:
    src_seq = sp.EncodeENSentence(seq=src_seq)
    # print(seq)
    src_seq = se.SrcEncode(src_seq=src_seq)
    tgt_seq = None
    assert tgt_lang in ["en", "zh"], f"invalid input {tgt_lang}"
    if tgt_lang == 'en':
        tgt_seq = [[enc_vocab["<zh2en>"]] + [enc_vocab["<pad>"]] * (MAX_SEQ_LEN - 1) for _ in range(len(src_seq))]
    elif tgt_lang == 'zh':
        tgt_seq = [[enc_vocab["<en2zh>"]] + [enc_vocab["<pad>"]] * (MAX_SEQ_LEN - 1) for _ in range(len(src_seq))]

    # print(seq)
    return torch.tensor(src_seq).reshape(-1, MAX_SEQ_LEN).to(DEVICES[0]), \
        torch.tensor(tgt_seq).reshape(-1, MAX_SEQ_LEN).to(DEVICES[0])


def postprocess(tgt_seq, tgt_lang) -> str:
    return sp.DecodeTgtSentence(se.Decode(tgt_seq), tgt_lang)


def infer(src_seqs, target_lang):
    def __infer(src_seq, tgt_lang):
        # "</s>": 0, "<unk>": 2, "<pad>": 65000
        src_seq, tgt_seq = preprocess(src_seq, tgt_lang)
        MODEL.eval()
        with torch.no_grad():
            for idx in range(1, MAX_SEQ_LEN):
                out = MODEL(src_seq, tgt_seq[:, :-1])
                # print(out, out.shape)
                next_tensor = torch.argmax(input=out[:, idx - 1], dim=1)
                tgt_seq[:, idx] = next_tensor
        tgt_seq = postprocess(tgt_seq, tgt_lang)
        return tgt_seq

    tgt_seqs = []
    src_batch_seqs = []

    num_seqs = len(src_seqs)
    for i in range(num_seqs):
        if BATCH_SIZE * (i + 1) < num_seqs:
            src_batch_seqs.append(src_seqs[BATCH_SIZE * i:BATCH_SIZE * (i + 1)])
        elif BATCH_SIZE * i < num_seqs <= BATCH_SIZE * (i + 1):
            src_batch_seqs.append(src_seqs[BATCH_SIZE * i:])
        else:
            break
    for src_seq in src_batch_seqs:
        tgt_seqs += __infer(src_seq, target_lang)
    return tgt_seqs


if __name__ == "__main__":
    src = [
        "For the people who lacking basic digital capabilities, "
        "they may think online public services and business transactions are burdens,"
        "let alone enjoy the benefits that are offered by them.",

        "If you're in before 6:00, it's the whatchamacallit...",
        "Companies have an incentive to keep minors from betting and to operate transparently.",
        "Come on, you two, let's go.",
        "Apparently, but the weirdest part ",
        "And then lastly your marketing campaign.",
        "He had a cooked breakfast, thank God. ",
        "You need a montage - montage ",
        "Went to get a drink, she disappeared. No, sir. ",
        "I got a pen.",
        "He is one of the most important justices in American history ",
        "Congratulations on making the finest microwave I've ever seen. ",
    ]
    print(infer(src, target_lang="zh"))
