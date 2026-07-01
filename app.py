"""Streamlit-интерфейс поверх pipeline.py."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from pipeline import run_pipeline

load_dotenv()

st.set_page_config(page_title="Транскрибация RU/KZ/EN", layout="centered")
st.title("Агент транскрибации аудио/видео (RU/KZ/EN)")
st.caption(
    "Загрузите запись совещания, интервью или выступления — получите протокол "
    "(.docx), чистый текст (.txt) и субтитры (.srt)."
)

uploaded = st.file_uploader(
    "Аудио (mp3, wav, m4a) или видео (mp4)",
    type=["mp3", "wav", "m4a", "mp4"],
)

diarize = st.checkbox(
    "Диаризация спикеров (требуется HF_TOKEN)",
    value=False,
    help="Определяет, кто говорит. Замедляет обработку и требует HuggingFace token.",
)

run_clicked = st.button("Запустить транскрибацию", disabled=uploaded is None)

if run_clicked and uploaded is not None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / uploaded.name
        input_path.write_bytes(uploaded.getbuffer())
        output_dir = tmp_path / "output"

        with st.status("Обработка...", expanded=True) as status:
            def progress(message: str) -> None:
                status.write(message)

            try:
                result = run_pipeline(
                    input_path,
                    output_dir,
                    diarize_flag=diarize,
                    progress_cb=progress,
                )
            except Exception as exc:  # noqa: BLE001 - показываем ошибку пользователю
                status.update(label="Ошибка", state="error")
                st.error(str(exc))
                st.stop()

            status.update(label="Готово", state="complete")

        st.success("Транскрибация завершена")

        downloads = [
            ("Протокол (.docx)", "docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            ("Текст (.txt)", "txt", "text/plain"),
            ("Субтитры (.srt)", "srt", "text/plain"),
        ]
        cols = st.columns(3)
        for col, (label, key, mime) in zip(cols, downloads):
            path = result.get(key)
            if path and path.exists():
                col.download_button(
                    label,
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime=mime,
                )
