# Split the sci_sum data into train/dev/test (90/10/10) randomly

import os
import json
import random
random.seed(1111)

all_filename = "abstruct_converted_data.jsonl"
train = "train.jsonl"
dev = "dev.jsonl"
test = "test.jsonl"

train_jsonl = []
dev_jsonl = []
test_jsonl = []

with open(all_filename) as _all_jsonl_file:
    result = [json.loads(jline) for jline in _all_jsonl_file.read().split('\n')[:-1]]

for paper in result:
    random_flip = random.uniform(0, 1)
    if random_flip < 0.8:
        train_jsonl.append(paper)
    elif random_flip < 0.9:
        dev_jsonl.append(paper)
    else:
        test_jsonl.append(paper)

print("Train", len(train_jsonl))
print("Dev", len(dev_jsonl))
print("Test", len(test_jsonl))

with open(train, 'w') as _train_jsonl_file:
    for line in train_jsonl:
        _train_jsonl_file.write(json.dumps(line))
        _train_jsonl_file.write('\n')

with open(dev, 'w') as _dev_jsonl_file:
    for line in dev_jsonl:
        _dev_jsonl_file.write(json.dumps(line))
        _dev_jsonl_file.write('\n')

with open(test, 'w') as _test_jsonl_file:
    for line in test_jsonl:
        _test_jsonl_file.write(json.dumps(line))
        _test_jsonl_file.write('\n')