import logging
from logging.handlers import RotatingFileHandler

sys_message = {
    "zh": "你的任务是将用户提供的文本全部翻译成中文，只返回翻译后的结果，无法翻译时保留原文，无需注释",
    "jp": "あなたの任務は、ユーザーが提供したテキストをすべて日本語に翻訳し、翻訳結果のみを返すことです。翻訳できない場合は原文を保持し、コメントは必要ありません。"
}

default = {
    "Qwen": {
        "local": "Qwen/Qwen2.5-1.5B-Instruct",
        "api": "qwen-turbo"
    },
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False
formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(name)s: %(message)s')

# 保存日志到文件
file_handler = RotatingFileHandler('output/log.log', maxBytes=1024 * 1024, backupCount=5)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 控制台日志输出
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
