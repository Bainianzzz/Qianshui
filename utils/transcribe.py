import os.path

import torch.cuda
from faster_whisper import WhisperModel, BatchedInferencePipeline

from format_srt import transcription2srt
from meta.transcribe import *


def transcribe(
        audio_file,
        model_size: str = default["model_size"],
        compute_type: str = "int8",
        output_dir: str = None,
        output: str = default["output"],
        beam_size: int = 5,
        chunk_length: int = 8,
        batch_size: int = 8,
        vad_parameters=None,
        multilingual: bool = False
):
    if vad_parameters is None:
        vad_parameters = dict(min_silence_duration_ms=500)
    if torch.cuda.is_available():
        model = WhisperModel(model_size, device="cuda", compute_type=compute_type)
        batched_model = BatchedInferencePipeline(model=model)

        segments, info = batched_model.transcribe(audio_file, task="transcribe", beam_size=beam_size,
                                                  chunk_length=chunk_length,
                                                  batch_size=batch_size,
                                                  multilingual=multilingual,
                                                  vad_filter=True, vad_parameters=vad_parameters)
    else:
        model = WhisperModel(model_size, device="cpu", compute_type=compute_type)

        segments, info = model.transcribe(audio_file, task="transcribe", beam_size=beam_size,
                                          multilingual=multilingual,
                                          vad_filter=True, vad_parameters=vad_parameters)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    output_dir = default["output_dir"] if output_dir is None else output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output = os.path.join(output_dir, output)
    with open(output, "w", encoding="utf-8") as file:
        file.write("")

    # 提取文本内容
    for index, segment in enumerate(segments):
        transcription2srt(index, segment, output)

    print(f'Transcription saved to {output}')


if __name__ == '__main__':
    transcribe('../tmp/example.mp3', output_dir="../tmp")
