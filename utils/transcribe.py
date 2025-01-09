import os.path

import torch.cuda
from faster_whisper import WhisperModel, BatchedInferencePipeline

from utils.format_srt import transcription2srt
from utils import logger


def transcribe(
        audio_file,
        model_size: str = "distil-large-v3",
        compute_type: str = "float16",
        output_dir: str = None,
        beam_size: int = 5,
        chunk_length: int = 5,
        batch_size: int = 8,
        vad_parameters=None,
        multilingual: bool = False
):
    if vad_parameters is None:
        vad_parameters = dict(min_silence_duration_ms=500)
    if torch.cuda.is_available():
        logger.info(f'The model will be running on {torch.cuda.get_device_name()}')
        model = WhisperModel(model_size, device="cuda", compute_type=compute_type)
        batched_model = BatchedInferencePipeline(model=model)

        segments, info = batched_model.transcribe(audio_file, task="transcribe", beam_size=beam_size,
                                                  chunk_length=chunk_length,
                                                  batch_size=batch_size,
                                                  multilingual=multilingual,
                                                  vad_filter=True, vad_parameters=vad_parameters)
        logger.info(f'running paramters:\n'
                    f'beam_size:{beam_size}  chunk_length:{chunk_length}  batch_size:{batch_size}  '
                    f'multilingual:{multilingual}  vad_filter:{vad_parameters}')
    else:
        logger.info(f'The model will be running on cpu')
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

        segments, info = model.transcribe(audio_file, task="transcribe", beam_size=beam_size,
                                          multilingual=multilingual,
                                          vad_filter=True, vad_parameters=vad_parameters)
        logger.info(f'running paramters:\n'
                    f'beam_size:{beam_size}  multilingual:{multilingual}  vad_filter:{vad_parameters}')

    logger.info("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    output_dir = "output/transcription/" if output_dir is None else output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_name = os.path.splitext(os.path.basename(audio_file))[0] + '_transcription.srt'
    output_path = os.path.join(output_dir, file_name)
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("")

    # 提取文本内容
    for index, segment in enumerate(segments):
        transcription2srt(index, segment, output_path)

    output_path=os.path.abspath(output_path)
    logger.info(f'Transcription saved to {output_path}')
    return output_path


if __name__ == '__main__':
    transcribe('../tmp/example.mp3', output_dir="../tmp")
