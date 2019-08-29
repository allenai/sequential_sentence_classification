# When downloading the data, both train and test papers ended up with their data in the same folder
# This script filters out the data for the test paper ids into a separate file
import os
import json

def main():
    sci_sum_data_path = os.path.join(os.getcwd(), "data", "sci_sum")
    test_ids_path = os.path.join(sci_sum_data_path, "cspubsum_test_ids.txt")
    with open(test_ids_path) as _test_ids_file:
        test_ids = set([url_id.split('/')[-1] for url_id in _test_ids_file.read().split('\n')[:-1]])

    print("Number of test ids:", len(test_ids))

    train_path = os.path.join(sci_sum_data_path, "train.jsonl")
    dev_path = os.path.join(sci_sum_data_path, "dev.jsonl")
    test_path = os.path.join(sci_sum_data_path, "test.jsonl")

    filtered_train_examples = []
    filtered_dev_examples = []
    filtered_test_examples = []

    actual_test_examples = []

    with open(train_path) as _train_jsonl_file:
        train_lines = [json.loads(line) for line in _train_jsonl_file.read().split('\n')[:-1]]

    with open(dev_path) as _dev_jsonl_file:
        dev_lines = [json.loads(line) for line in _dev_jsonl_file.read().split('\n')[:-1]]

    with open(test_path) as _test_jsonl_file:
        test_lines = [json.loads(line) for line in _test_jsonl_file.read().split('\n')[:-1]]

    print("Train examples read:", len(train_lines))
    print("Dev examples read:", len(dev_lines))
    print("Test examples read:", len(test_lines))

    ids_filtered_out = set()
    for line in train_lines:
        paper_id = line["abstract_id"]
        if paper_id in test_ids:
            ids_filtered_out.add(paper_id)
            actual_test_examples.append(line)
        else:
            filtered_train_examples.append(line)

    for line in dev_lines:
        paper_id = line["abstract_id"]
        if paper_id in test_ids:
            ids_filtered_out.add(paper_id)
            actual_test_examples.append(line)
        else:
            filtered_dev_examples.append(line)

    for line in test_lines:
        paper_id = line["abstract_id"]
        if paper_id in test_ids:
            ids_filtered_out.add(paper_id)
            actual_test_examples.append(line)
        else:
            filtered_test_examples.append(line)

    print("Number of examples filtered out:", len(actual_test_examples))
    print("Ids not found in original set:", test_ids - ids_filtered_out)
    print("Train examples to write:", len(filtered_train_examples))
    print("dev examples to write:", len(filtered_dev_examples))
    print("Test examples to write:", len(filtered_test_examples))

    new_train_path = os.path.join(sci_sum_data_path, "train_new.jsonl")
    new_dev_path = os.path.join(sci_sum_data_path, "dev_new.jsonl")
    new_test_path = os.path.join(sci_sum_data_path, "test_new.jsonl")
    actual_test_path = os.path.join(sci_sum_data_path, "rouge_test.jsonl")

    # Commented out writing to file to prevent me from accidentally overwriting data files again
    # with open(new_train_path, "w") as _new_train_file:
    #     for line in filtered_train_examples:
    #         _new_train_file.write(json.dumps(line))
    #         _new_train_file.write('\n')

    # with open(new_dev_path, "w") as _new_dev_file:
    #     for line in filtered_dev_examples:
    #         _new_dev_file.write(json.dumps(line))
    #         _new_dev_file.write('\n')

    # with open(new_test_path, "w") as _new_test_file:
    #     for line in filtered_test_examples:
    #         _new_test_file.write(json.dumps(line))
    #         _new_test_file.write('\n')

    # with open(actual_test_path, "w") as _actual_test_file:
    #     for line in actual_test_examples:
    #         _actual_test_file.write(json.dumps(line))
    #         _actual_test_file.write('\n')

if __name__ == "__main__":
    main()
