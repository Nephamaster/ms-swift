import json


def count_samples_num(data_path:str):
    with open(data_path, 'r', encoding='utf-8') as f:
        if 'jsonl' in data_path or 'txt' in data_path:
            print(len(f.readlines()))
        elif 'json' in data_path:
            data = json.load(f)
            print (len(data))


if __name__ == '__main__':
    count_samples_num('../data/tiger_pt_zh_dataflow.jsonl')