import os.path

from tqdm import tqdm
from transformers import pipeline
from openai import OpenAI

from utils import *


def chat_local(content: list[str] = None,
               output_language: str = "zh",
               model_name: str = default["Qwen"]["local"]):
    pipe = pipeline("text-generation",
                    model_name,
                    device_map="auto"
                    )

    result = []
    for message in tqdm(content, desc="Translating: ", unit="sentence"):
        messages = [
            {'role': 'system', 'content': sys_message[output_language]},
            {'role': 'user', 'content': f'translate:{message}'}
        ]
        output = pipe(messages, max_new_tokens=128)
        result.append(output[0]['generated_text'][-1]['content'])

    return result


def chat_api(content: list[str] = None,
             output_language: str = "zh",
             model_name: str = default["Qwen"]["api"]):
    client = OpenAI(
        api_key=os.getenv("ALIYUN_API"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    result = []
    for message in tqdm(content, desc="Translating: ", unit="sentence"):
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {'role': 'system', 'content': sys_message[output_language]},
                {'role': 'user', 'content': f'translate:{message}'}],
        )
        result.append(completion.choices[0].message.content)

    return result


def translate(input_file: str = None,
              mode: str = "api",
              model_name: str = default["Qwen"],
              output_language: str = "zh",
              output_dir: str = None
              ):
    with open(input_file, 'r', encoding='utf-8') as file:
        srt_list = file.readlines()
    content = [srt_list[i].replace('\n', '') for i in range(2, len(srt_list), 3)]
    if mode == "local":
        logger.info('Translating with local model')
        response = chat_local(content, output_language, model_name=model_name[mode])
    elif mode == "api":
        logger.info('Translating with api')
        response = chat_api(content, output_language, model_name=model_name[mode])
    else:
        raise Exception("mode must be local or api")
    for index, r in enumerate(response):
        srt_list[2 + 3 * index] = r + '\n'

    output_dir = "output/translation/" if output_dir is None else os.path.join(output_dir, 'translation')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_name = os.path.splitext(os.path.basename(input_file))[0] + f'_{output_language}.srt'
    output_path = os.path.join(output_dir, file_name)
    with open(output_path, "w", encoding="utf-8") as file:
        file.writelines(srt_list)

    output_path = os.path.abspath(output_path)
    logger.info(f'Translation saved to {output_path}')
    return output_path


if __name__ == '__main__':
    input_file = 'output/transcription/example_transcription.srt'
    translate(input_file, mode="api", output_dir="output/translation", output_language="jp")
