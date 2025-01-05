import time

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def translate(prompt):
    messages = [
        {"role": "system", "content": "你的工作是将日文翻译成中文，要求内容准确，不得遗漏或曲解原意，并且只返回翻译结果"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=128
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


if __name__ == '__main__':
    input_file = 'transcription.srt'
    output_file = 'translation.srt'
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("")

    # 读取并翻译
    start_time = time.time()
    with open(input_file, 'r', encoding='utf-8') as file:
        srt_list = file.readlines()
    for i in tqdm(range(2, len(srt_list), 3), desc='Translating: ', unit='sentence'):
        srt_list[i] = translate(srt_list[i])

    # 将翻译结果写入文件
    with open(output_file, 'a', encoding='utf-8') as file:
        file.writelines(srt_list)

    print(f'translation finished, costs：{(time.time() - start_time) / 60:.2f} minutes')
