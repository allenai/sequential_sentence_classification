import logging
from typing import Dict

import torch
from torch.nn import Linear
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, TimeDistributed, Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import F1Measure, CategoricalAccuracy
from allennlp.modules.conditional_random_field import ConditionalRandomField

logger = logging.getLogger(__name__)

@Model.register("SeqClassificationModel")
class SeqClassificationModel(Model):
    """
    Question answering model where answers are sentences
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 use_sep: bool = True,
                 with_crf: bool = False,
                 self_attn: Seq2SeqEncoder = None,
                 bert_dropout: float = 0.1,
                 sci_sum: bool = False,
                 additional_feature_size: int = 0,
                 ) -> None:
        super(SeqClassificationModel, self).__init__(vocab)

        self.text_field_embedder = text_field_embedder
        self.vocab = vocab
        self.use_sep = use_sep
        self.with_crf = with_crf
        self.sci_sum = sci_sum
        self.self_attn = self_attn
        self.additional_feature_size = additional_feature_size

        self.dropout = torch.nn.Dropout(p=bert_dropout)

       # define loss
        if self.sci_sum:
            self.loss = torch.nn.MSELoss(reduction='none')  # labels are rouge scores
            self.labels_are_scores = True
            self.num_labels = 1
        else:
            self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
            self.labels_are_scores = False
            self.num_labels = self.vocab.get_vocab_size(namespace='labels')
            # define accuracy metrics
            self.label_accuracy = CategoricalAccuracy()
            self.label_f1_metrics = {}

            # define F1 metrics per label
            for label_index in range(self.num_labels):
                label_name = self.vocab.get_token_from_index(namespace='labels', index=label_index)
                self.label_f1_metrics[label_name] = F1Measure(label_index)

        encoded_senetence_dim = text_field_embedder._token_embedders['bert'].get_output_dim()

        ff_in_dim = encoded_senetence_dim if self.use_sep else self_attn.get_output_dim()
        ff_in_dim += self.additional_feature_size

        self.time_distributed_aggregate_feedforward = TimeDistributed(Linear(ff_in_dim, self.num_labels))

        if self.with_crf:
            self.crf = ConditionalRandomField(
                self.num_labels, constraints=None,
                include_start_end_transitions=True
            )

    def forward(self,  # type: ignore
                sentences: torch.LongTensor,
                labels: torch.IntTensor = None,
                confidences: torch.Tensor = None,
                additional_features: torch.Tensor = None,
                ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        TODO: add description

        Returns
        -------
        An output dictionary consisting of:
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        # ===========================================================================================================
        # Layer 1: For each sentence, participant pair: create a Glove embedding for each token
        # Input: sentences
        # Output: embedded_sentences

        # embedded_sentences: batch_size, num_sentences, sentence_length, embedding_size
        embedded_sentences = self.text_field_embedder(sentences, num_wrapping_dims= 1)
        mask = get_text_field_mask(sentences, num_wrapping_dims=1).float()
        batch_size, num_sentences, _, _ = embedded_sentences.size()

        if self.use_sep:
            # The following code collects vectors of the SEP tokens from all the examples in the batch,
            # and arrange them in one list. It does the same for the labels and confidences.
            # TODO: replace 103 with '[SEP]'
            sentences_mask = sentences['bert']["token_ids"] == 103  # mask for all the SEP tokens in the batch
            embedded_sentences = embedded_sentences[sentences_mask]  # given batch_size x num_sentences_per_example x sent_len x vector_len
                                                                        # returns num_sentences_per_batch x vector_len
            assert embedded_sentences.dim() == 2
            num_sentences = embedded_sentences.shape[0]
            # for the rest of the code in this model to work, think of the data we have as one example
            # with so many sentences and a batch of size 1
            batch_size = 1
            embedded_sentences = embedded_sentences.unsqueeze(dim=0)
            embedded_sentences = self.dropout(embedded_sentences)

            if labels is not None:
                if self.labels_are_scores:
                    labels_mask = labels != 0.0  # mask for all the labels in the batch (no padding)
                else:
                    labels_mask = labels != -1  # mask for all the labels in the batch (no padding)

                labels = labels[labels_mask]  # given batch_size x num_sentences_per_example return num_sentences_per_batch
                assert labels.dim() == 1
                if confidences is not None:
                    confidences = confidences[labels_mask]
                    assert confidences.dim() == 1
                if additional_features is not None:
                    additional_features = additional_features[labels_mask]
                    assert additional_features.dim() == 2

                num_labels = labels.shape[0]
                if num_labels != num_sentences:  # bert truncates long sentences, so some of the SEP tokens might be gone
                    assert num_labels > num_sentences  # but `num_labels` should be at least greater than `num_sentences`
                    logger.warning(f'Found {num_labels} labels but {num_sentences} sentences')
                    labels = labels[:num_sentences]  # Ignore some labels. This is ok for training but bad for testing.
                                                        # We are ignoring this problem for now.
                                                        # TODO: fix, at least for testing

                # do the same for `confidences`
                if confidences is not None:
                    num_confidences = confidences.shape[0]
                    if num_confidences != num_sentences:
                        assert num_confidences > num_sentences
                        confidences = confidences[:num_sentences]

                # and for `additional_features`
                if additional_features is not None:
                    num_additional_features = additional_features.shape[0]
                    if num_additional_features != num_sentences:
                        assert num_additional_features > num_sentences
                        additional_features = additional_features[:num_sentences]

                # similar to `embedded_sentences`, add an additional dimension that corresponds to batch_size=1
                labels = labels.unsqueeze(dim=0)
                if confidences is not None:
                    confidences = confidences.unsqueeze(dim=0)
                if additional_features is not None:
                    additional_features = additional_features.unsqueeze(dim=0)
        else:
            # ['CLS'] token
            embedded_sentences = embedded_sentences[:, :, 0, :]
            embedded_sentences = self.dropout(embedded_sentences)
            batch_size, num_sentences, _ = embedded_sentences.size()
            sent_mask = (mask.sum(dim=2) != 0)
            embedded_sentences = self.self_attn(embedded_sentences, sent_mask)

        if additional_features is not None:
            embedded_sentences = torch.cat((embedded_sentences, additional_features), dim=-1)

        label_logits = self.time_distributed_aggregate_feedforward(embedded_sentences)
        # label_logits: batch_size, num_sentences, num_labels

        if self.labels_are_scores:
            label_probs = label_logits
        else:
            label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        # Create output dictionary for the trainer
        # Compute loss and epoch metrics
        output_dict = {"action_probs": label_probs}

        # =====================================================================

        if self.with_crf:
            # Layer 4 = CRF layer across labels of sentences in an abstract
            mask_sentences = (labels != -1)
            best_paths = self.crf.viterbi_tags(label_logits, mask_sentences)
            #
            # # Just get the tags and ignore the score.
            predicted_labels = [x for x, y in best_paths]
            # print(f"len(predicted_labels):{len(predicted_labels)}, (predicted_labels):{predicted_labels}")

            label_loss = 0.0
        if labels is not None:
            # Compute cross entropy loss
            flattened_logits = label_logits.view((batch_size * num_sentences), self.num_labels)
            flattened_gold = labels.contiguous().view(-1)

            if not self.with_crf:
                label_loss = self.loss(flattened_logits.squeeze(), flattened_gold)
                if confidences is not None:
                    label_loss = label_loss * confidences.type_as(label_loss).view(-1)
                label_loss = label_loss.mean()
                flattened_probs = torch.softmax(flattened_logits, dim=-1)
            else:
                clamped_labels = torch.clamp(labels, min=0)
                log_likelihood = self.crf(label_logits, clamped_labels, mask_sentences)
                label_loss = -log_likelihood
                # compute categorical accuracy
                crf_label_probs = label_logits * 0.
                for i, instance_labels in enumerate(predicted_labels):
                    for j, label_id in enumerate(instance_labels):
                        crf_label_probs[i, j, label_id] = 1
                flattened_probs = crf_label_probs.view((batch_size * num_sentences), self.num_labels)

            if not self.labels_are_scores:
                evaluation_mask = (flattened_gold != -1)
                self.label_accuracy(flattened_probs.float().contiguous(), flattened_gold.squeeze(-1), mask=evaluation_mask)

                # compute F1 per label
                for label_index in range(self.num_labels):
                    label_name = self.vocab.get_token_from_index(namespace='labels', index=label_index)
                    metric = self.label_f1_metrics[label_name]
                    metric(flattened_probs, flattened_gold, mask=evaluation_mask)
        
        if labels is not None:
            output_dict["loss"] = label_loss
        output_dict['action_logits'] = label_logits
        return output_dict

    def get_metrics(self, reset: bool = False):
        metric_dict = {}

        if not self.labels_are_scores:
            type_accuracy = self.label_accuracy.get_metric(reset)
            metric_dict['acc'] = type_accuracy

            average_F1 = 0.0
            for name, metric in self.label_f1_metrics.items():
                metric_val = metric.get_metric(reset)
                metric_dict[name + 'F'] = metric_val["f1"]
                average_F1 += metric_val["f1"]

            average_F1 /= len(self.label_f1_metrics.items())
            metric_dict['avgF'] = average_F1

        return metric_dict
