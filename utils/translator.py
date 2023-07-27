import json
import safetensors.torch
import torch
from utils import SPModel, SeqEncoder
from model.transformer import Transformer


class NmtTranslator:
    def __init__(self, cfg: dict):
        self.batch_size = cfg.get("batch_size", 64)

        self.vocab_size = cfg.get("vocab_size", 65003)

        self.enc_vocab = None
        self.dec_vocab = None

        self.max_seq_len = cfg.get("max_seq_len", 50)
        self.pad_idx = cfg.get("pad_idx", 65000)
        self.embedding_size = cfg.get("embedding_size", 1024)
        self.num_heads = cfg.get("num_heads", 10)
        self.num_layers = cfg.get("num_layers", 8)

        self.num_gpus = cfg.get("num_gpus", 8)
        self.devices = [torch.device(i) for i in range(self.num_gpus)]
        if not self.devices:
            self.devices = [torch.device("cpu")]
            print("Using cpu.")

        self.src_pe_path = cfg.get("src_pe_path", None)
        self.tgt_pe_path = cfg.get("tgt_pe_path", None)

        self.vocab_path = cfg.get("vocab_path", None)

        self.checkpoint = cfg.get("checkpoint", None)

        self.sp_model = SPModel()
        self.__set_sp_model()

        self.seq_encoder = None
        self.__set_seq_encoder()

        self.transformer = None
        self.__set_transformer()

    def __set_sp_model(self):
        assert self.src_pe_path and self.tgt_pe_path
        self.sp_model.LoadModel(src_path=self.src_pe_path, tgt_path=self.tgt_pe_path)

    def __set_seq_encoder(self):
        with open(file=self.vocab_path, mode='r', encoding='utf-8') as vocab_file:
            self.enc_vocab = json.load(vocab_file)
            self.dec_vocab = {v: k for k, v in self.enc_vocab.items()}
        self.seq_encoder = SeqEncoder(enc_vocab=self.enc_vocab,
                                      dec_vocab=self.dec_vocab,
                                      max_seq_len=self.max_seq_len)

    def __set_transformer(self):
        self.transformer = Transformer(src_vocab_size=self.vocab_size,
                                       trg_vocab_size=self.vocab_size,
                                       src_pad_idx=self.pad_idx,
                                       trg_pad_idx=self.pad_idx,
                                       embed_size=self.embedding_size,
                                       heads=self.num_heads,
                                       num_layers=self.num_layers,
                                       max_length=self.max_seq_len,
                                       device=self.devices[0]
                                       ).to(self.devices[0])
        safetensors.torch.load_model(self.transformer, self.checkpoint)
        # self.transformer.load_state_dict(torch.load(self.checkpoint, map_location=self.devices[0]), strict=False)

    def translate(self, src_seqs, target_lang):
        def __preprocess(_src_seq, _tgt_lang) -> torch.tensor:
            assert _tgt_lang in ["en", "zh"], f"invalid input {_tgt_lang}"

            tgt_seq = None

            if _tgt_lang == 'en':
                _src_seq = self.sp_model.EncodeZHSentence(seq=_src_seq)
                # print(seq)
                _src_seq = self.seq_encoder.SrcEncode(src_seq=_src_seq)
                tgt_seq = [[self.enc_vocab["<zh2en>"]] + [self.enc_vocab["<pad>"]] * (self.max_seq_len - 1) for _ in
                           range(len(_src_seq))]
            elif _tgt_lang == 'zh':
                _src_seq = self.sp_model.EncodeENSentence(seq=_src_seq)
                # print(seq)
                _src_seq = self.seq_encoder.SrcEncode(src_seq=_src_seq)
                tgt_seq = [[self.enc_vocab["<en2zh>"]] + [self.enc_vocab["<pad>"]] * (self.max_seq_len - 1) for _ in
                           range(len(_src_seq))]

            # print(seq)
            return torch.tensor(_src_seq).reshape(-1, self.max_seq_len).to(self.devices[0]), \
                torch.tensor(tgt_seq).reshape(-1, self.max_seq_len).to(self.devices[0])

        def __postprocess(tgt_seq, tgt_lang) -> str:
            return self.sp_model.DecodeTgtSentence(self.seq_encoder.Decode(tgt_seq), tgt_lang)

        def __infer(_src_seq, _tgt_lang):
            # "</s>": 0, "<unk>": 2, "<pad>": 65000
            _src_seq, tgt_seq = __preprocess(_src_seq, _tgt_lang)
            self.transformer.eval()
            translated = [1 for _ in range(len(_src_seq))]
            with torch.no_grad():
                for _idx in range(1, self.max_seq_len):
                    out = self.transformer(_src_seq, tgt_seq[:, :-1])
                    # print(out, out.shape)
                    next_tensor = torch.argmax(input=out[:, _idx - 1], dim=1)
                    tgt_seq[:, _idx] = next_tensor
                    for i in range(len(next_tensor)):
                        if next_tensor[i] == self.enc_vocab["</e>"]:
                            translated[i] = 0
                    if sum(translated) == 0:
                        break
            tgt_seq = __postprocess(tgt_seq, _tgt_lang)

            return tgt_seq

        assert len(src_seqs) <= self.batch_size * 5, f"Input too many sentence.({len(src_seqs)})"
        src_batch_seqs = []
        num_seqs = len(src_seqs)
        tgt_seqs = []

        for idx in range(num_seqs):
            if self.batch_size * (idx + 1) < num_seqs:
                src_batch_seqs.append(src_seqs[self.batch_size * idx:self.batch_size * (idx + 1)])
            elif self.batch_size * idx < num_seqs <= self.batch_size * (idx + 1):
                src_batch_seqs.append(src_seqs[self.batch_size * idx:])
            else:
                break
        for src_seq in src_batch_seqs:
            tgt_seqs += __infer(src_seq, target_lang)
        return tgt_seqs
