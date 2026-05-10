import json

def convert_item(item):
    conversations = item.get("conversations", [])
    result = []

    i = 0
    while i < len(conversations) - 1:
        cur = conversations[i]
        nxt = conversations[i + 1]

        # 只处理 human -> gpt 成对情况
        if cur.get("from") == "human" and nxt.get("from") == "gpt":
            result.append({
                "human": cur.get("value", ""),
                "assistant": nxt.get("value", "")
            })
            i += 2
        else:
            # 跳过异常或不成对数据
            i += 1

    return {"conversation": result}


def convert_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)   # 如果是 jsonl，可以改成逐行读

    new_data = [convert_item(item) for item in data]

    with open(output_path, "w", encoding="utf-8") as f:
        for item in new_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    input_path = "../dataset/SFT_novel_10000.json"
    output_path = "../dataset/SFT_novel_10000.jsonl"
    convert_file(input_path, output_path)