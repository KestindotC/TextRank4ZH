import os
import re
import logging
from typing import List

import jieba.posseg as pseg

from . import util


logger = logging.getLogger(__name__)


def get_default_stop_words_filepath():
    logger.info("Get default package stopwords.")
    dir_name = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dir_name, 'stopwords.txt')


class WordSegmentation(object):

    def __init__(self,
                 stop_words_file: str = None,
                 allow_speech_tags: List = util.allow_speech_tags):

        self.default_speech_tag_filter = allow_speech_tags
        self.stop_words = None
        if not stop_words_file:
            stop_words_file = get_default_stop_words_filepath()

        with open(stop_words_file, mode="r", encoding="utf-8", errors="ignore") as f:
            self.stop_words = set(f.read().splitlines())

    def segment(self, text,
                lower=True,
                use_stop_words=True,
                use_speech_tags_filter=False):

        jieba_result = pseg.cut(text)

        if use_speech_tags_filter:
            jieba_result = [w for w in jieba_result if w.flag in self.default_speech_tag_filter]
        else:
            jieba_result = [w for w in jieba_result]

        word_list = [w.word.strip() for w in jieba_result if w.flag != 'x']
        word_list = [word for word in word_list if len(word) > 0]

        if lower:
            word_list = [word.lower() for word in word_list]

        if use_stop_words:
            word_list = [word.strip() for word in word_list if word.strip() not in self.stop_words]

        return word_list

    def segment_sentences(self, sentences, lower=True, use_stop_words=True, use_speech_tags_filter=False):

        res = []
        for sentence in sentences:
            res.append(self.segment(text=sentence,
                                    lower=lower,
                                    use_stop_words=use_stop_words,
                                    use_speech_tags_filter=use_speech_tags_filter))
        return res


class SentenceSegmentation(object):

    def __init__(self, delimiters: List = None):
        if not delimiters:
            self.delimiters = f"[{''.join(util.sentence_delimiters)}]"
        else:
            self.delimiters = f"[{''.join(list(set(delimiters)))}]"

    def segment(self, text: str):
        logger.debug(f"Sentence to segment: {text}")
        logger.debug(f"Delimiters to applied on text: {str(self.delimiters)}")

        result = re.split(self.delimiters, text)
        return list(filter(None, result))


class Segmentation(object):

    def __init__(self, stop_words_file=None,
                 allow_speech_tags=util.allow_speech_tags,
                 delimiters=util.sentence_delimiters):

        self.ws = WordSegmentation(stop_words_file=stop_words_file, allow_speech_tags=allow_speech_tags)
        self.ss = SentenceSegmentation(delimiters=delimiters)

    def segment(self, text, lower=False):
        sentences = self.ss.segment(text)
        words_no_filter = self.ws.segment_sentences(sentences=sentences,
                                                    lower=lower,
                                                    use_stop_words=False,
                                                    use_speech_tags_filter=False)
        words_no_stop_words = self.ws.segment_sentences(sentences=sentences,
                                                        lower=lower,
                                                        use_stop_words=True,
                                                        use_speech_tags_filter=False)

        words_all_filters = self.ws.segment_sentences(sentences=sentences,
                                                      lower=lower,
                                                      use_stop_words=True,
                                                      use_speech_tags_filter=True)

        return util.AttrDict(
            sentences=sentences,
            words_no_filter=words_no_filter,
            words_no_stop_words=words_no_stop_words,
            words_all_filters=words_all_filters
        )
