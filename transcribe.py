import io
import requests
import ffmpeg
from pydub import AudioSegment
import whisper
import argparse

class VideoDownloader:
    """
    Класс для скачивания видео по URL.
    """
    def __init__(self, video_url: str):
        self.video_url = video_url

    def download_video(self) -> bytes:
        """
        Скачивает видео с указанного URL и возвращает его в виде байтов.
        """
        response = requests.get(self.video_url, stream=True)
        response.raise_for_status()  # Проверка на ошибки
        video_data = io.BytesIO(response.content)
        return video_data


class AudioExtractor:
    """
    Класс для извлечения аудио из видеофайла.
    """
    def __init__(self, video_data: bytes):
        self.video_data = video_data

    def extract_audio(self) -> io.BytesIO:
        """
        Извлекает аудио из видео и возвращает его в виде объекта BytesIO.
        """
        # Сохраняем видео во временный файл для обработки
        video_file = io.BytesIO(self.video_data)
        video_file.seek(0)
        
        # Извлечение аудио с помощью ffmpeg
        audio_output = io.BytesIO()
        ffmpeg.input('pipe:0').output('pipe:1', format='wav').run(input=self.video_data, capture_stdout=True, capture_stderr=True)
        
        return audio_output


class AudioSplitter:
    """
    Класс для разбиения аудио на части фиксированной длительности.
    """
    def __init__(self, audio_data: io.BytesIO, chunk_duration: int = 30):
        self.audio_data = audio_data
        self.chunk_duration = chunk_duration

    def split_audio(self) -> list:
        """
        Разбивает аудио на части фиксированной длительности (в секундах).
        Возвращает список объектов BytesIO.
        """
        audio = AudioSegment.from_file(self.audio_data, format="wav")
        duration_ms = len(audio)
        chunks = []

        for start_ms in range(0, duration_ms, self.chunk_duration * 1000):
            chunk = audio[start_ms:start_ms + self.chunk_duration * 1000]
            chunk_data = io.BytesIO()
            chunk.export(chunk_data, format="wav")
            chunk_data.seek(0)
            chunks.append(chunk_data)

        return chunks


class Transcriber:
    """
    Класс для транскрипции аудио в текст с использованием Whisper.
    """
    def __init__(self):
        self.model = whisper.load_model("base")  # Загрузим модель whisper

    def transcribe(self, audio_data: io.BytesIO) -> str:
        """
        Транскрибирует аудио в текст.
        """
        audio = whisper.load_audio(audio_data)
        result = self.model.transcribe(audio)
        return result['text']


class TextSaver:
    """
    Класс для сохранения транскрибированного текста в файл.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path

    def save_text(self, text: str):
        """
        Сохраняет текст в файл.
        """
        with open(self.file_path, 'w', encoding='utf-8') as f:
            f.write(text)


class VideoProcessor:
    """
    Класс для обработки видео, извлечения аудио, транскрибирования и сохранения текста.
    """
    def __init__(self, video_url: str, output_text_file: str):
        self.video_url = video_url
        self.output_text_file = output_text_file

    def process_video(self):
        """
        Процесс обработки видео: скачивание, извлечение аудио, транскрипция и сохранение текста.
        """
        # 1. Скачиваем видео
        downloader = VideoDownloader(self.video_url)
        video_data = downloader.download_video()

        # 2. Извлекаем аудио из видео
        extractor = AudioExtractor(video_data)
        audio_data = extractor.extract_audio()

        # 3. Разбиваем аудио на части
        splitter = AudioSplitter(audio_data)
        chunks = splitter.split_audio()

        # 4. Транскрибируем каждую часть
        transcriber = Transcriber()
        full_text = ""
        for chunk in chunks:
            text = transcriber.transcribe(chunk)
            full_text += text + "\n"  # Добавляем текст с новой строки

        # 5. Сохраняем текст в файл
        saver = TextSaver(self.output_text_file)
        saver.save_text(full_text)
        print(f"Текст успешно сохранен в {self.output_text_file}")


# Добавление обработки аргументов командной строки
def main():
    parser = argparse.ArgumentParser(description="Обработка видео: скачивание, извлечение аудио и транскрипция текста.")
    parser.add_argument("video_url", type=str, help="URL видео для скачивания.")
    parser.add_argument("output_text_file", type=str, help="Путь к файлу для сохранения транскрибированного текста.")

    args = parser.parse_args()

    # Создаем процессор и запускаем обработку
    processor = VideoProcessor(args.video_url, args.output_text_file)
    processor.process_video()


# Запуск программы
if __name__ == "__main__":
    main()
