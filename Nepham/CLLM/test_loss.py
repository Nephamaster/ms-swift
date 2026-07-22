from swift import get_processor, get_template

data = {"messages": [{"role": "user", "content": "任务: 纠错文本\n输入: 目前区次事件的细节还不清楚，伤亡人数也未确定。\n输出: "}, {"role": "assistant", "content": "目前这次事件的细节还不清楚，伤亡人数也未确定。"}], "src": "目前区次事件的细节还不清楚，伤亡人数也未确定。", "tgt": "目前这次事件的细节还不清楚，伤亡人数也未确定。"}

template = get_template(
    get_processor('/share/project/wuhaiming/spaces/CLLM/models/Qwen3-4B-Base-Char',model_type='qwen3'),
    template_type='qwen3',
    loss_scale='default+ignore_empty_think',
    is_binary_loss_scale=False
)
template.set_mode('train')
inputs = template.encode(data)
print(inputs)
print(template.safe_decode(inputs['labels']))
print(inputs['loss_scale'])