import time


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    formatted_seconds = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{formatted_seconds:02d},{milliseconds:03d}"


# 提取文本内容并保存为SRT格式
def transcription2srt(index, segment, output_file):
    # 转换时间格式
    start_time_srt = format_time(segment.start)
    end_time_srt = format_time(segment.end)

    # 添加到 SRT 内容
    srt_content = ""
    srt_content += f"{index}\n"
    srt_content += f"{start_time_srt} --> {end_time_srt}\n"
    srt_content += f"{segment.text}\n"
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

    # 保存到文件
    with open(output_file, "a", encoding="utf-8") as file:
        file.write(srt_content)
