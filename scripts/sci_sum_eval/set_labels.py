import numpy as np
import json

def add_labels(filename, skip=True):
    total = 0
    failed = 0
    new_jsons = []
    with open(f'data/sci_sum/old/{filename}_no_labels.jsonl') as f:
        for line in f: 
            json_dict = json.loads(line)
            scores = json_dict['highlight_scores']
            assert len(scores) == len(json_dict['sentences'])
            total += 1
            if len(scores) < 50 and skip:
                failed += 1
                continue
            sorting_index = np.argsort(scores)
            labels = [2] * len(scores)
            for i in sorting_index[:20]:
                labels[i] = 0
            for i in sorting_index[-20:]:
                labels[i] = 1
            json_dict['labels'] = labels
            new_jsons.append(json_dict)
    print(f'total: {total}, good: {total - failed}, bad: {failed}')
    with open(f'data/sci_sum/{filename}.jsonl', 'w') as f:
        for json_dict in new_jsons:
            line = json.dumps(json_dict)
            f.write(f'{line}\n')

add_labels('train')
add_labels('dev')
add_labels('test')
add_labels('rouge_test', skip=False)