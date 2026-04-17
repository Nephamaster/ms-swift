## 数据格式

### SFT

query-response格式:
```jsonl
{"system": "<system>", "query": "<query2>", "response": "<response2>", "history": [["<query1>", "<response1>"]]}
```

按以上形式处理数据条目然后存储为jsonl文件

## 数据信息

### csc_mix.jsonl

#### 数据来源

- twnlp/csc_data
    hf_dataset_id：twnlp/csc_data
    github：https://github.com/TW-NLP/ChineseErrorCorrector
    paper: 
    ```bibtex
    @misc{tian2025chineseerrorcorrector34bstateoftheartchinesespelling,
        title={ChineseErrorCorrector3-4B: State-of-the-Art Chinese Spelling and Grammar Corrector}, 
        author={Wei Tian and YuhaoZhou},
        year={2025},
        eprint={2511.17562},
        archivePrefix={arXiv},
        primaryClass={cs.CL},
        url={https://arxiv.org/abs/2511.17562}, 
    }
    ```

- SIGHAN13-15 train
    github: https://github.com/onebula/sighan_raw/tree/master/pair_data/simplified

#### 数据分布

W271K(279,816)
Medical(39,303)
Lemon(22,259)
ECSpell(6,688)
CSCD(35,001)
SIGHAN

负样本比例：85%