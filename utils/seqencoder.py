import tqdm


class SeqEncoder(object):
    def __init__(self, enc_vocab, dec_vocab, max_seq_len):
        self.__enc_vocab = enc_vocab
        self.__dec_vocab = dec_vocab
        self.__seq_max_len = max_seq_len

    def encode(self, seq):
        return [self.__enc_vocab.get(i, self.__enc_vocab["<unk>"]) for i in seq]

    def SrcEncode(self, src_seq):
        assert len(src_seq) > 0
        if isinstance(src_seq[0], list):
            t = tqdm.tqdm(range(len(src_seq)))
            for i in t:
                src_seq[i] = self.CutOrPad(src_seq[i], mode="src")
                src_seq[i] = self.encode(src_seq[i])
            return src_seq
        return self.encode(self.CutOrPad(src_seq, mode="src"))

    def TgtEncode(self, tgt_seq, tgt_lang):
        assert len(tgt_seq) > 0
        if isinstance(tgt_seq[0], list):
            for i in range(len(tgt_seq)):
                tgt_seq[i] = self.CutOrPad(tgt_seq[i], mode="tgt", tgt_lang=tgt_lang)
                tgt_seq[i] = self.encode(tgt_seq[i])
            return tgt_seq
        return self.encode(self.CutOrPad(tgt_seq, mode="tgt", tgt_lang=tgt_lang))

    def __decode(self, tensors):
        dec_seq = []
        for i in tensors:

            dec_chr = self.__dec_vocab[int(i)]
            if dec_chr in ['<pad>', '<zh2en>', '<en2zh>']:
                continue
            if dec_chr == "</e>":
                break
            dec_seq.append(dec_chr)
        return dec_seq

    def Decode(self, tgt_tsr):
        assert len(tgt_tsr) > 0

        if isinstance(tgt_tsr[0], torch.Tensor):
            tgt_dec = []
            for i in range(len(tgt_tsr)):
                tgt_dec.append(self.__decode(tgt_tsr[i]))
            return tgt_dec
        return self.__decode(tgt_tsr)

    def CutOrPad(self, seq: list[str], mode, tgt_lang=None):
        if mode == "tgt":
            if tgt_lang == "en":
                seq.insert(0, "<zh2en>")
            elif tgt_lang == "zh":
                seq.insert(0, "<en2zh>")
            seq.append("</e>")

        seq_len = len(seq)

        if seq_len <= self.__seq_max_len:
            seq += ["<pad>"] * (self.__seq_max_len - seq_len)
        else:
            seq = seq[:self.__seq_max_len]

        return seq