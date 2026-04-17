pip install -i https://mirrors.aliyun.com/pypi/simple \
    addict aiohttp binpacking charset_normalizer cpm_kernels dacite einops fastapi zstandard bitsandbytes
pip install -i https://mirrors.aliyun.com/pypi/simple \
    "gradio>=3.40.0,<6.0" importlib_metadata modelscope nltk openai oss2 rouge "trl>=0.15,<0.30"
pip install -i https://mirrors.aliyun.com/pypi/simple \
    scipy sentencepiece simplejson sortedcontainers tensorboard tiktoken transformers_stream_generator
pip install -i https://mirrors.aliyun.com/pypi/simple "datasets>=3.0,<4.0" "peft>=0.11,<0.19"
pip install -i https://mirrors.aliyun.com/pypi/simple accelerate
pip install -i https://mirrors.aliyun.com/pypi/simple deepspeed
pip install -i https://mirrors.aliyun.com/pypi/simple torchvision
pip install -i https://mirrors.aliyun.com/pypi/simple flash-attn --no-build-isolation
pip install -i https://mirrors.aliyun.com/pypi/simple -e .
pip install -i https://mirrors.aliyun.com/pypi/simple "qwen_vl_utils>=0.0.14" "decord" -U