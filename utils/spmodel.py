import tqdm
import sentencepiece as spm


class SPModel(object):
    def __init__(self):
        self.__en_sp_model = spm.SentencePieceProcessor()
        self.__zh_sp_model = spm.SentencePieceProcessor()

    def LoadModel(self, src_path, tgt_path):
        self.__en_sp_model.Load(src_path)
        self.__zh_sp_model.Load(tgt_path)

    def EncodeENSentence(self, seq):
        return self.__en_sp_model.EncodeAsPieces(seq)

    def EncodeZHSentence(self, seq):
        return self.__zh_sp_model.EncodeAsPieces(seq)

    def ProcessSentence(self, src_sentence, tgt_sentence):
        return self.EncodeENSentence(src_sentence), self.EncodeZHSentence(tgt_sentence)

    def DecodeTgtSentence(self, tgt_sentence, tgt_lang):
        assert tgt_lang in ["en", "zh"], f"invalid input {tgt_lang}"
        if tgt_lang == "en":
            return self.__en_sp_model.DecodePieces(input=tgt_sentence, out_type=str)
        elif tgt_lang == "zh":
            return self.__zh_sp_model.DecodePieces(input=tgt_sentence, out_type=str)

    def BatchProcess(self, src_sentences, tgt_sentences):
        assert len(src_sentences) == len(tgt_sentences)
        src_sentences = tqdm.tqdm(src_sentences)
        return [self.EncodeENSentence(s) for s in src_sentences], \
            [self.EncodeZHSentence(s) for s in tgt_sentences]
