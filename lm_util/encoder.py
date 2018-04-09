import numpy as np
import json
import codecs

class Encoder(object):
    loaders = {
        "tsv": (lambda x: dict([(int(l.split("\t")[0]), u"".join(l.split("\t")[1:])) for l in x.strip().split("\n")])),
        "json": lambda x: json.loads(x),
    }

    def __init__(self, code_2_utf={}, loader_file_contents="", loader="",is_dictionary=False):
        if loader == "":
            self.code_2_utf = code_2_utf
        else:
            print loader
            self.code_2_utf = EncoDeco.loaders[loader](loader_file_contents)
        self.utf_2_code = {v: k for k, v in self.code_2_utf.iteritems()}
        self.default_utf = u"."
        self.default_code = max(self.code_2_utf.keys())
        self.is_dictionary=is_dictionary

    def add_null(self, symbol=u"\u2205"):
        self.code_2_utf[max(self.code_2_utf.keys()) + 1] = symbol
        self.utf_2_code = {v: k for k, v in self.code_2_utf.iteritems()}

    @property
    def alphabet_size(self):
        return len(self.code_2_utf)

    @classmethod
    def load_tsv(cls,fname,is_dictionary=False):
        tsv=codecs.open(fname,"r","utf-8").read().strip()
        code_2_utf=dict([(int(l[:l.find("\t")]), u"".join(l.split("\t")[1:])) for l in tsv.split("\n")])
        encoder=cls(code_2_utf=code_2_utf,is_dictionary=is_dictionary)
        return encoder

    @classmethod
    def load_json(cls,fname,is_dictionary=False):
        code_2_utf=json.loads(codecs.open(fname,"r","utf-8").read())
        encoder=cls(code_2_utf=code_2_utf,is_dictionary=is_dictionary)
        return encoder


    def encode(self, msg_string):
        if self.is_dictionary:
            return np.array([self.utf_2_code.get(word, self.default_code) for word in msg_string.split() if len(word)>0], dtype="int64")
        else:
            return np.array([self.utf_2_code.get(char, self.default_code) for char in msg_string], dtype="int64")

    def encode_onehot(self,msg_string):
        res=np.zeros([len(msg_string),self.alphabet_size])
        res[np.arange(len(msg_string)),self.encode(msg_string)]=1
        return res

    def decode(self, msg_nparray):
        if self.is_dictionary:
            return u" ".join([self.code_2_utf.get(code, self.default_utf) for code in msg_nparray.tolist()])
        else:
            return u"".join([self.code_2_utf.get(code, self.default_utf) for code in msg_nparray.tolist()])

    def decode_ctc(self, msg_nparray, null_val):
        keep_idx = np.zeros(msg_nparray.shape, dtype="bool")
        keep_idx[1:] = msg_nparray[1:] != msg_nparray[:-1]
        keep_idx = np.logical_and(keep_idx, msg_nparray != null_val)
        if self.is_dictionary:
            return u" ".join([self.code_2_utf.get(code, self.default_utf) for code in msg_nparray[keep_idx].tolist()])
        else:
            return u"".join([self.code_2_utf.get(code, self.default_utf) for code in msg_nparray[keep_idx].tolist()])

    def encode_batch(self, msg_list):
        res_lengths = np.array([len(msg) for msg in msg_list], dtype="int64")
        res = np.zeros([len(msg_list), res_lengths.max()], dtype="int64")
        count = 0
        for msg_string in msg_list:
            res[count, :len(msg_string)] = self.encode(msg_string)
            count += 1
        return res, res_lengths

    def decode_batch(self, msg_nparray, lengths=None):
        if lengths is None:
            lengths = np.ones(msg_nparray.shape[0]) * msg_nparray.shape[1]
        res = []
        for k in range(msg_nparray.shape[0]):
            res.append(self.decode(msg_nparray[k, :lengths[k]]))
        return res

    def decode_ctc_batch(self, msg_nparray, null_val, lengths=None):
        if lengths is None:
            lengths = np.ones(msg_nparray.shape[0]) * msg_nparray.shape[1]
        res = []
        for k in range(msg_nparray.shape[0]):
            res.append(self.decode_ctc(msg_nparray[k, :lengths[k]], null_val))
        return res

    def get_encoder(self):
        return lambda x: self.encode(x)

    def get_decoder(self):
        return lambda x: self.decode(x)

    def get_ctc_decoder(self):
        return lambda x: self.decode_ctc(x)


def get_int_2_uc_dict(map_tsv, separator=None):
    try:
        map_tsv = open(map_tsv).read().decode("utf8").strip()
    except IOError:
        map_tsv = map_tsv.strip()
    lines = [line.split("\t")[0] for line in map_tsv.split(separator)]
    try:
        code2uc = {int(line[0]): unicode(line[1]) for line in lines}
    except IndexError:
        code2uc = {counter: unicode(line[0]) for counter, line in enumerate(lines)}
    return code2uc
