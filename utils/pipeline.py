import os.path

from utils import default, logger
from utils.transcribe import transcribe
from utils.translate import translate


class Pipeline:
    def __init__(self,
                 task: str = 'transcribe'):
        if task is None:
            raise RuntimeError('Task must be specified')
        if task == 'transcribe':
            self.task = task
        elif task == 'translate':
            self.task = task
        else:
            raise ValueError('Task must be either "transcribe" or "translate"')

    def predict(self,
                file: str = None,
                srt_file: str = None,
                model_size: str = "distil-large-v3",
                compute_type: str = "float16",
                output_dir: str = None,
                beam_size: int = 5,
                chunk_length: int = 5,
                batch_size: int = 8,
                vad_parameters=None,
                multilingual: bool = False,
                mode: str = "api",
                model_name: str = default["Qwen"],
                output_language: str = "zh"):
        """'
        :param file: an audio file or media file to be transcribed/translated
        :param srt_file: a srt file to be translated
        :param model_size: the model which will be used to transcribe
            See https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py#L604
        :param compute_type: Type to use for computation.
            See https://opennmt.net/CTranslate2/quantization.html.
        :param output_dir: the directory where the output will be saved
        :param beam_size:
        :param chunk_length:
        :param batch_size:
        :param vad_parameters:
        :param multilingual:
            See https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py#L753
        :param mode: the way to finish the translation
            "api": use the model's API (please set the API_KEY ad your system variable),
            "local": download and run model on your own computer
        :param model_name: the model which is used to translate
        :param output_language: the language which will be translated to
        :return: None
        """

        if file is None and srt_file is None:
            raise RuntimeError('File or srt_file must be specified')
        if file is not None and not os.path.exists(file):
            raise FileNotFoundError(f'{file} is not an available file')
        if srt_file is not None and not os.path.exists(srt_file):
            raise FileNotFoundError(f'{srt_file} is not an available file')

        if srt_file is None:
            if model_size is None:
                compute_type = "float16" if compute_type is None else compute_type
                logger.info(f'Using default model size: \"distil-large-v3\" with {compute_type}')
                model_size = "distil-large-v3"

            transcription = transcribe(audio_file=file, model_size=model_size, compute_type=compute_type,
                                       output_dir=output_dir, beam_size=beam_size,
                                       chunk_length=chunk_length, batch_size=batch_size,
                                       vad_parameters=vad_parameters, multilingual=multilingual)

            if self.task == 'translate':
                translate(input_file=transcription, mode=mode, model_name=model_name,
                          output_language=output_language, output_dir=output_dir)
        else:
            if self.task == 'translate':
                logger.info(f'Pipeline will directly translate the {srt_file}')
                translate(input_file=srt_file, mode=mode, model_name=model_name,
                          output_language=output_language, output_dir=output_dir)
            else:
                logger.info(f'{srt_file} already exists, no need to transcribe')
                pass


if __name__ == '__main__':
    pipe = Pipeline('translate')
    pipe.predict(srt_file='output/transcription/example_transcription.srt', mode='api')
