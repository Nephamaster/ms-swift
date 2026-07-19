import json


def normalize(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    normalized = []
    for term in data:
        normalized.append({'messages': [{'role': 'assistant', 'content': term['content']}]})

    with open(output_path, 'w', encoding='utf-8') as f:
        for term in normalized:
            f.write(json.dumps(term, ensure_ascii=False) + '\n')


normalize('../dataset/tiger_pt_zh_dataflow.json', '../dataset/tiger_pt_zh_dataflow.jsonl')
