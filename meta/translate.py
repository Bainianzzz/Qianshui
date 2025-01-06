sys_message = {
    "zh": "你的任务是将用户提供的sentence全部翻译成中文，只返回翻译后的结果",
    "en": "Translate the user-provided text into English without omitting any information or misinterpreting the meaning, and return only the translated result."
}

default = {
    "Qwen": {
        "local": "Qwen/Qwen2.5-1.5B-Instruct",
        "api": "qwen-turbo"
    },
    "output_file": "translation.srt"
}
