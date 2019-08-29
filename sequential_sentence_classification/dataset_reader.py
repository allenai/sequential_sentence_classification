import itertools
import json
from typing import Dict, List
from overrides import overrides

import numpy as np

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data import Tokenizer
from allennlp.data.instance import Instance
from allennlp.data.fields.field import Field
from allennlp.data.fields import TextField, LabelField, ListField, ArrayField, MultiLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.word_splitter import SimpleWordSplitter, WordSplitter


@DatasetReader.register("SeqClassificationReader")
class SeqClassificationReader(DatasetReader):
    """
    Reads a file from Pubmed-RCT dataset. Each instance contains an abstract_id, 
    a list of sentences and a list of labels (one per sentence).
    Input File Format: Example abstract below:
        {
        "abstract_id": 5337700, 
        "sentences": ["this is motivation", "this is method", "this is conclusion"], 
        "labels": ["BACKGROUND", "RESULTS", "CONCLUSIONS"]
        }
    """

    def __init__(self,
                 lazy: bool = False,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 word_splitter: WordSplitter = None,
                 tokenizer: Tokenizer = None,
                 sent_max_len: int = 100,
                 max_sent_per_example: int = 20,
                 use_sep: bool = True,
                 sci_sum: bool = False,
                 use_abstract_scores: bool = True,
                 predict: bool = False,
                 ) -> None:
        super().__init__(lazy)
        self._word_splitter = word_splitter or SimpleWordSplitter()
        self._tokenizer = tokenizer or WordTokenizer(self._word_splitter)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.sent_max_len = sent_max_len
        self.use_sep = use_sep
        self.predict = predict
        self.sci_sum = sci_sum
        self.max_sent_per_example = max_sent_per_example
        self.use_abstract_scores = use_abstract_scores

    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path)

        with open(file_path) as f:
            for line in f:
                json_dict = json.loads(line)
                instances = self.read_one_example(json_dict)
                for instance in instances:
                    yield instance

    def read_one_example(self, json_dict):
        instances = []
        sentences = json_dict["sentences"]

        if not self.predict:
            labels = json_dict["labels"]
        else:
            labels = None

        additional_features = None
        if self.sci_sum:
            labels = [s if s > 0 else 0.000001 for s in json_dict["highlight_scores"]]

            if self.use_abstract_scores:
                features = []
                if self.use_abstract_scores:
                    features.append(json_dict["abstract_scores"])
                additional_features = list(map(list, zip(*features)))  # some magic transpose function

            sentences, labels = self.filter_bad_sci_sum_sentences(sentences, labels)

            if len(sentences) == 0:
                return []

        for sentences_loop, labels_loop, additional_features_loop in  \
                self.enforce_max_sent_per_example(sentences, labels, additional_features):

            instance = self.text_to_instance(
                sentences=sentences_loop,
                labels=labels_loop,
                additional_features=additional_features_loop,
                )
            instances.append(instance)
        return instances

    def enforce_max_sent_per_example(self, sentences, labels=None, additional_features=None):
        """
        Splits examples with len(sentences) > self.max_sent_per_example into multiple smaller examples
        with len(sentences) <= self.max_sent_per_example.
        Recursively split the list of sentences into two halves until each half
        has len(sentences) < <= self.max_sent_per_example. The goal is to produce splits that are of almost
        equal size to avoid the scenario where all splits are of size
        self.max_sent_per_example then the last split is 1 or 2 sentences
        This will result into losing context around the edges of each examples.
        """
        if labels is not None:
            assert len(sentences) == len(labels)
        if additional_features is not None:
            assert len(sentences) == len(additional_features)

        if len(sentences) > self.max_sent_per_example and self.max_sent_per_example > 0:
            i = len(sentences) // 2
            l1 = self.enforce_max_sent_per_example(
                    sentences[:i], None if labels is None else labels[:i],
                    None if additional_features is None else additional_features[:i])
            l2 = self.enforce_max_sent_per_example(
                    sentences[i:], None if labels is None else labels[i:],
                    None if additional_features is None else additional_features[i:])
            return l1 + l2
        else:
            return [(sentences, labels, additional_features)]

    def is_bad_sentence(self, sentence: str):
        if len(sentence) > 10 and len(sentence) < 600:
            return False
        else:
            return True

    def filter_bad_sci_sum_sentences(self, sentences, labels):
        filtered_sentences = []
        filtered_labels = []
        if not self.predict:
            for sentence, label in zip(sentences, labels):
                # most sentences outside of this range are bad sentences
                if not self.is_bad_sentence(sentence):
                    filtered_sentences.append(sentence)
                    filtered_labels.append(label)
                else:
                    filtered_sentences.append("BADSENTENCE")
                    filtered_labels.append(0.000001)
            sentences = filtered_sentences
            labels = filtered_labels
        else:
            for sentence in sentences:
                # most sentences outside of this range are bad sentences
                if not self.is_bad_sentence(sentence):
                    filtered_sentences.append(sentence)
                else:
                    filtered_sentences.append("BADSENTENCE")
            sentences = filtered_sentences

        return sentences, labels

    @overrides
    def text_to_instance(self,
                         sentences: List[str],
                         labels: List[str] = None,
                         additional_features: List[float] = None,
                         ) -> Instance:
        if not self.predict:
            assert len(sentences) == len(labels)
        if additional_features is not None:
            assert len(sentences) == len(additional_features)

        if self.use_sep:
            tokenized_sentences = [self._tokenizer.tokenize(s)[:self.sent_max_len] + [Token("[SEP]")] for s in sentences]
            sentences = [list(itertools.chain.from_iterable(tokenized_sentences))[:-1]]
        else:
            # Tokenize the sentences
            sentences = [
                self._tokenizer.tokenize(sentence_text)[:self.sent_max_len]
                for sentence_text in sentences
            ]

        fields: Dict[str, Field] = {}
        fields["sentences"] = ListField([
                TextField(sentence, self._token_indexers)
                for sentence in sentences
        ])

        if labels is not None:
            if isinstance(labels[0], list):
                fields["labels"] = ListField([
                        MultiLabelField(label) for label in labels
                    ])
            else:
                # make the labels strings for easier identification of the neutral label
                # probably not strictly necessary
                if self.sci_sum:
                    fields["labels"] = ArrayField(np.array(labels))
                else:
                    fields["labels"] = ListField([
                            LabelField(str(label)+"_label") for label in labels
                        ])

        if additional_features is not None:
            fields["additional_features"] = ArrayField(np.array(additional_features))

        return Instance(fields)