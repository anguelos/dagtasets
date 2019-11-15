import numpy as np
import sys
import json
import urllib
#import urllib.request, urllib.parse, urllib.error
import xml.etree.ElementTree as ET
import io
import codecs
import re

class OcrCorpus(object):

    @staticmethod
    def get_iliad_unicode(lang="eng"):
        if lang == "eng":
            book_url = lambda \
                    n: "http://www.perseus.tufts.edu/hopper/xmlchunk?doc=Perseus%3Atext%3A1999.01.0134%3Abook%3D" + str(
                int(n + 1))
        elif lang == "gr":
            book_url = lambda \
                    n: "http://www.perseus.tufts.edu/hopper/xmlchunk?doc=Perseus%3Atext%3A1999.01.0133%3Abook%3D" + str(
                int(n + 1))
        else:
            raise ValueError()
        book_str_list = []
        for book_no in range(1):#range(24):
            xml_book = urllib.request.urlopen(book_url(book_no)).read()
            root = ET.fromstring(xml_book)
            book_text = ET.tostring(root, encoding='utf-8', method='text')
            book_text = book_text.replace("  ", "\n").replace('\r','')
            book_str_list.append(book_text.decode("utf-8"))
        corpus = "\n".join(book_str_list)
        return corpus

    @staticmethod
    def corpus_to_alphabet_tsv(corpus_unicode_str):
        alphabet = list(set(re.sub("\s+"," ",corpus_unicode_str).replace('\r', '')))
        return "\n".join([str(int(n)) + "\t" + s for n, s in enumerate(alphabet)])

    def __init__(self,*args):
        if len(args):
            raise NotImplementedError()

    @classmethod
    def create_iliad_corpus(cls,lang='eng',alphabet=None):
        """Synthecizer constructor."""
        corpus=cls()
        corpus_str=OcrCorpus.get_iliad_unicode(lang)
        if alphabet is None:
            corpus.alphabet=list(set(re.sub("\s+"," ",corpus_str).replace('\r', '')))
        else:
            if set(alphabet)>set(corpus_str):
                corpus.alphabet=list(alphabet)
            else: # the alphabet is missing some of the occuring characters
                raise ValueError()
        corpus.data_stream=io.StringIO(corpus_str)
        corpus.alphabet_to_num={v:n for n,v in enumerate(corpus.alphabet)}
        return corpus

    @classmethod
    def create_file_corpus(cls,fname,alphabet=None,encoding="utf-8"):
        """Synthecizer constructor."""
        corpus=cls()
        corpus_str=open(fname).read().decode('utf8')
        if alphabet is None:
            corpus.alphabet=list(set(re.sub("\s+"," ",corpus_str).replace('\r', '')))
        else:
            if set(alphabet)>set(corpus_str):
                corpus.alphabet=list(alphabet)
            else: # the alphabet is missing some of the occuring characters
                raise ValueError()
        corpus.data_stream=io.StringIO(corpus_str)
        corpus.alphabet_to_num={v:n for n,v in enumerate(corpus.alphabet)}
        return corpus

    def read_str(self,nchars=-1):
        if nchars<=0:
            res_str = self.data_stream.read(nchars)
        else:
            res_str=self.data_stream.read(nchars)
            if len(res_str)<nchars:
                self.data_stream.seek(0)
                res_str+=self.data_stream.read(nchars-len(res_str))
            if len(res_str) < nchars:
                raise IOError()
        return res_str

    def read_symbol_ids(self,nchars=-1):
        res_str=self.read_str()
        return np.array([self.alphabet_to_num[c] for c in res_str],dtype='int32')

    def get_alphabet_tsv(self):
        return "\n".join([str(int(n)) + "\t" + s for n, s in enumerate(self.alphabet)])
