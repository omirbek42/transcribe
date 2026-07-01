# Агент транскрибации аудио/видео (RU/KZ/EN)

MVP-пайплайн для транскрибации совещаний, интервью и выступлений на русском,
казахском и английском языках (включая смешанную казахско-русскую речь), с
автоматической коррекцией текста под корпоративный деловой стиль. Подробности
требований — в [prd.md](prd.md).

## Структура

- `pipeline.py` — извлечение аудио → ASR → диаризация → LLM-коррекция → экспорт
- `glossary.json` — редактируемый словарь терминов (компании, аббревиатуры, ФИО)
- `app.py` — Streamlit-интерфейс поверх `pipeline.py`
- `requirements.txt` — зависимости Python
- `.env.example` — шаблон переменных окружения

## Установка

### 1. ffmpeg

Требуется системный `ffmpeg` (используется для извлечения/нормализации аудио).

- Windows: `winget install ffmpeg` (или скачать с ffmpeg.org и добавить в PATH)
- macOS: `brew install ffmpeg`
- Linux: `apt install ffmpeg`

Проверка: `ffmpeg -version`

### 2. Python-окружение

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. GPU (опционально)

Если доступна CUDA, пайплайн автоматически использует модель `large-v3` на
GPU; иначе — `medium` на CPU. Для GPU нужен PyTorch со сборкой под CUDA (см.
https://pytorch.org/get-started/locally/ для команды под вашу версию CUDA).

### 4. Переменные окружения

Скопируйте `.env.example` в `.env` и заполните:

```
ANTHROPIC_API_KEY=...   # обязателен для шага LLM-коррекции
HF_TOKEN=...             # обязателен только при включённой диаризации спикеров
```

- `ANTHROPIC_API_KEY` — ключ Anthropic API (https://console.anthropic.com/).
- `HF_TOKEN` — HuggingFace access token с принятыми условиями использования
  моделей `pyannote/speaker-diarization-3.1` и `pyannote/segmentation-3.0`
  (https://huggingface.co/settings/tokens).

## Запуск

### Веб-интерфейс

```bash
streamlit run app.py
```

Откроется в браузере: загрузите файл, при необходимости включите диаризацию,
нажмите «Запустить транскрибацию», скачайте `.docx` / `.txt` / `.srt`.

### Командная строка

```bash
python pipeline.py meeting.mp4 --output-dir out --diarize
```

Флаги:
- `--diarize` — включить диаризацию спикеров (требует `HF_TOKEN`)
- `--glossary path/to/glossary.json` — свой путь к глоссарию
- `--skip-llm` — пропустить шаг LLM-коррекции (для отладки одного ASR)

## Глоссарий

`glossary.json` подставляется в system prompt шага LLM-коррекции и не требует
изменения кода:

```json
{
  "companies": [],
  "abbreviations": {},
  "people": []
}
```

Заполняется владельцем продукта по мере необходимости.

## Ограничения MVP

- Только batch-обработка готовых файлов, без real-time транскрибации.
- Без перевода между языками — язык каждой реплики сохраняется как есть.
- Без аутентификации пользователей.
- Данные отправляются в Anthropic API (шаг коррекции) и, при включённой
  диаризации, обрабатываются моделью pyannote локально — HuggingFace token
  используется только для загрузки весов модели, аудио никуда не выгружается.
