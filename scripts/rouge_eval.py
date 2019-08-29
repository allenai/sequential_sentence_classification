import os
import json
import argparse
import numpy as np
from tqdm import tqdm

from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

# Rouge computation is taken from https://github.com/EdCo95/scientific-paper-summarisation/blob/master/Evaluation/rouge.py
# 
# File Name : https://github.com/EdCo95/scientific-paper-summarisation/blob/master/Evaluation/rouge.py
#
# Description : Computes ROUGE-L metric as described by Lin and Hovey (2004)
#
# Creation Date : 2015-01-07 06:03
# Author : Ramakrishna Vedantam <vrama91@vt.edu>

def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings

    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if(len(string)< len(sub)):
        sub, string = string, sub

    lengths = [[0 for i in range(0,len(sub)+1)] for j in range(0,len(string)+1)]

    for j in range(1,len(sub)+1):
        for i in range(1,len(string)+1):
            if(string[i-1] == sub[j-1]):
                lengths[i][j] = lengths[i-1][j-1] + 1
            else:
                lengths[i][j] = max(lengths[i-1][j] , lengths[i][j-1])

    return lengths[len(string)][len(sub)]

class Rouge():
    '''
    Class for computing ROUGE-L score for a set of candidate sentences for the MS COCO test set

    '''
    def __init__(self):
        # vrama91: updated the value below based on discussion with Hovey
        self.beta = 1.2

    def calc_score(self, candidate, refs):
        """
        Compute ROUGE-L score given one candidate and references for an image
        :param candidate: str : candidate sentence to be evaluated
        :param refs: list of str : COCO reference sentences for the particular image to be evaluated
        :returns score: int (ROUGE-L score for the candidate evaluated against references)
        """
        assert(len(candidate)==1)	
        assert(len(refs)>0)         
        prec = []
        rec = []

        # split into tokens
        token_c = candidate[0].split(" ")
    	
        for reference in refs:
            # split into tokens
            token_r = reference.split(" ")
            # compute the longest common subsequence
            lcs = my_lcs(token_r, token_c)
            prec.append(lcs/float(len(token_c)))
            rec.append(lcs/float(len(token_r)))

        prec_max = max(prec)
        rec_max = max(rec)

        if(prec_max!=0 and rec_max !=0):
            score = ((1 + self.beta**2)*prec_max*rec_max)/float(rec_max + self.beta**2*prec_max)
        else:
            score = 0.0
        return score

    def compute_score(self, gts, res):
        """
        Computes Rouge-L score given a set of reference and candidate sentences for the dataset
        Invoked by evaluate_captions.py 
        :param hypo_for_image: dict : candidate / test sentences with "image name" key and "tokenized sentences" as values 
        :param ref_for_image: dict : reference MS-COCO sentences with "image name" key and "tokenized sentences" as values
        :returns: average_score: float (mean ROUGE-L score computed by averaging scores for all the images)
        """
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        score = []
        for id in imgIds:
            hypo = res[id]
            ref  = gts[id]

            score.append(self.calc_score(hypo, ref))

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)

        average_score = np.mean(np.array(score))
        return average_score, np.array(score)

    def method(self):
        return "Rouge"

def main(model_path: str, test_jsonl_file: str, test_highlights_path: str):
    rouge = Rouge()
    # Load paper highlights
    with open(test_highlights_path) as _highlights_json_file:
        higlights_by_id = json.load(_highlights_json_file)
    
    with open(test_abstracts_path) as _abstracts_json_file:
        abstracts_by_id = json.load(_abstracts_json_file)

    # Load allennlp model
    text_field_embedder = {"token_embedders": {"bert": {"pretrained_model": "/net/nfs.corp/s2-research/scibert/scibert_scivocab_uncased.tar.gz"}}}
    token_indexers = {"bert": {"pretrained_model": "/net/nfs.corp/s2-research/scibert/scivocab_uncased.vocab"}}
    overrides = {"model": {"text_field_embedder": text_field_embedder},
                 "dataset_reader": {"token_indexers": token_indexers},}
    model_archive = load_archive(model_path, overrides=json.dumps(overrides), cuda_device=0)
    predictor = Predictor.from_archive(model_archive, 'SeqClassificationPredictor')
    dataset_reader = Predictor._dataset_reader

    # Load papers to predict on
    with open(test_jsonl_file) as _test_jsonl_file:
        test_lines = [json.loads(line) for line in _test_jsonl_file.read().split('\n')[:-1]]

    print("{} test lines loaded".format(len(test_lines)))

    abstract_total_score = 0
    abstract_total_instances = 0
    # Using abstracts as the predictions
    for line in test_lines:
        paper_id = line["abstract_id"]
        abstract_sentences = abstracts_by_id[paper_id]
        highlights = higlights_by_id[paper_id]

        summary_score = 0
        summary_sentences = 0
        for sentence in abstract_sentences:
            score = rouge.calc_score([sentence], highlights)
            summary_score += score
            summary_sentences += 1

        avg_summary_score = summary_score / summary_sentences
        abstract_total_score += avg_summary_score
        abstract_total_instances += 1

    print("final score:", abstract_total_score / abstract_total_instances)

    test_jsons = []
    with open(test_jsonl_file) as f:
        for line in f:
            test_jsons.append(json.loads(line))

    print("{} test jsons loaded".format(len(test_jsons)))

    # Predict on said papers

    total_score = 0
    total_instances = 0
    for json_dict in tqdm(test_jsons, desc="Predicting..."):
        instances = dataset_reader.read_one_example(json_dict)
        if not isinstance(instances, list):  # if the datareader returns one instnace, put it in a list
            instances = [instances]

        sentences = json_dict['sentences']
        gold_scores_list = json_dict['highlight_scores']
        paper_id = instances[0].fields["abstract_id"].metadata
        highlights = higlights_by_id[paper_id]

        scores_list = []
        for instance in instances:
            prediction = predictor.predict_instance(instance)
            probs = prediction['action_probs']
            scores_list.extend(probs)

        assert len(sentences) == len(scores_list)
        assert len(sentences) == len(gold_scores_list)

        sentences_with_scores = list(zip(sentences, scores_list))

        # Note: the following line should get Oracle performance
        # sentences_with_scores = list(zip(sentences, gold_scores_list))
        sentences_with_scores = sorted(sentences_with_scores, key=lambda x: x[1], reverse=True)

        top_10_sentences = [s[0] for s in sentences_with_scores[:10]]

        summary_score = 0
        summary_sentences = 0
        for sentence in top_10_sentences:
            score = rouge.calc_score([sentence], highlights)
            summary_score += score
            summary_sentences += 1
        
        avg_summary_score = summary_score / summary_sentences
        total_score += avg_summary_score
        total_instances += 1

    print("final score:", total_score / total_instances)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_model",
        help="Path to the model to evaluate"
    )
    args = parser.parse_args()

    test_jsonl_file = os.path.join("data", "sci_sum", "rouge_test.jsonl")
    test_highlights_path = os.path.join("data", "sci_sum", "test_highlights.json")
    test_abstracts_path = os.path.join("data", "sci_sum", "test_abstracts.json")
    
    main(args.path_to_model, test_jsonl_file, test_highlights_path)
