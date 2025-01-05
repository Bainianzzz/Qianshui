import os.path
import time

import torch.cuda
from faster_whisper import WhisperModel, BatchedInferencePipeline
from tqdm import tqdm

from format_srt import transcription2srt


def transcribe(
        audio_file,
        model_size="large-v3",
        compute_type="int8",
        output="transcription.srt"
):
    if torch.cuda.is_available():
        model = WhisperModel(model_size, device="cuda", compute_type=compute_type)
        batched_model = BatchedInferencePipeline(model=model)

        segments, info = batched_model.transcribe(audio_file, task="transcribe", beam_size=5, chunk_length=8,
                                                  vad_filter=True, batch_size=8,
                                                  vad_parameters=dict(min_silence_duration_ms=800))
    else:
        model = WhisperModel(model_size, device="cpu", compute_type=compute_type)

        segments, info = model.transcribe(audio_file, task="transcribe", beam_size=5,
                                          vad_filter=True,
                                          vad_parameters=dict(min_silence_duration_ms=800))

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    with open(output, "w", encoding="utf-8") as file:
        file.write("")

    # 提取文本内容
    for index, segment in enumerate(segments):
        transcription2srt(index, segment, output)

    print("Transcription saved to transcription.srt")
    # print(f'transcription finished, costs：{(time.time() - start_time) / 60:.2f} minutes')
