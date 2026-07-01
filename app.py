"""Streamlit-интерфейс поверх pipeline.py."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from pipeline import load_glossary, run_pipeline

load_dotenv()

st.set_page_config(
    page_title="Транскрибация RU/KZ/EN",
    page_icon="🎙️",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 760px; }
    [data-testid="stFileUploaderDropzone"] { border-radius: 12px; }
    div[data-testid="stStatusWidget"] { border-radius: 12px; }
    .app-step-icon { font-size: 1.1rem; margin-right: 0.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

ANTHROPIC_KEY_SET = bool(os.environ.get("ANTHROPIC_API_KEY"))
HF_TOKEN_SET = bool(os.environ.get("HF_TOKEN"))

STEP_ICONS = [
    ("готово", "✅"),
    ("аудио", "🎧"),
    ("модел", "📥"),
    ("распознав", "🗣️"),
    ("диариз", "👥"),
    ("коррекц", "✍️"),
    ("экспорт", "📦"),
]


def iconize(message: str) -> str:
    lowered = message.lower()
    for keyword, icon in STEP_ICONS:
        if keyword in lowered:
            return f"{icon} {message}"
    return f"⏳ {message}"


# --- Sidebar -----------------------------------------------------------

with st.sidebar:
    st.subheader("О проекте")
    st.caption(
        "Транскрибация совещаний, интервью и выступлений на русском, "
        "казахском и английском (включая смешанную ru/kk речь) с "
        "приведением текста к деловому стилю."
    )

    st.markdown("**Пайплайн**")
    st.markdown(
        "1. Извлечение аудио (ffmpeg)\n"
        "2. Распознавание речи (faster-whisper)\n"
        "3. Диаризация спикеров (опционально)\n"
        "4. LLM-коррекция стиля и глоссария\n"
        "5. Экспорт .docx / .txt / .srt"
    )

    st.divider()
    st.markdown("**Статус ключей окружения**")
    if ANTHROPIC_KEY_SET:
        st.success("ANTHROPIC_API_KEY найден", icon="✅")
    else:
        st.error("ANTHROPIC_API_KEY не задан", icon="⚠️")
    if HF_TOKEN_SET:
        st.success("HF_TOKEN найден", icon="✅")
    else:
        st.info("HF_TOKEN не задан (нужен только для диаризации)", icon="ℹ️")

    st.divider()
    try:
        glossary = load_glossary(Path("glossary.json"))
        st.markdown("**Глоссарий**")
        gc1, gc2, gc3 = st.columns(3)
        gc1.metric("Компании", len(glossary.get("companies", [])))
        gc2.metric("Аббр.", len(glossary.get("abbreviations", {})))
        gc3.metric("ФИО", len(glossary.get("people", [])))
    except Exception:
        st.caption("glossary.json не найден")

# --- Main ---------------------------------------------------------------

st.title("🎙️ Агент транскрибации аудио/видео")
st.caption("RU / KZ / EN, включая смешанную казахско-русскую речь")
st.write(
    "Загрузите запись совещания, интервью или выступления — получите "
    "протокол (**.docx**), чистый текст (**.txt**) и субтитры (**.srt**)."
)

if not ANTHROPIC_KEY_SET:
    st.warning(
        "ANTHROPIC_API_KEY не задан — шаг LLM-коррекции завершится ошибкой. "
        "Задайте ключ в `.env` или включите тестовый режим ниже.",
        icon="⚠️",
    )

with st.container(border=True):
    uploaded = st.file_uploader(
        "Аудио (mp3, wav, m4a) или видео (mp4)",
        type=["mp3", "wav", "m4a", "mp4"],
    )
    if uploaded is not None:
        size_mb = len(uploaded.getbuffer()) / (1024 * 1024)
        st.caption(f"📄 {uploaded.name} · {size_mb:.1f} MB")

with st.expander("Дополнительные настройки"):
    diarize = st.checkbox(
        "Диаризация спикеров",
        value=False,
        disabled=not HF_TOKEN_SET,
        help="Определяет, кто говорит. Замедляет обработку, требует HF_TOKEN.",
    )
    if not HF_TOKEN_SET:
        st.caption("Недоступно без HF_TOKEN в окружении.")

    skip_llm = st.checkbox(
        "Тестовый режим — без LLM-коррекции",
        value=not ANTHROPIC_KEY_SET,
        help="Пропускает шаг коррекции стиля/глоссария. Полезно для проверки ASR без ключа Anthropic.",
    )

run_disabled = uploaded is None or (not skip_llm and not ANTHROPIC_KEY_SET)
run_clicked = st.button(
    "Запустить транскрибацию",
    type="primary",
    use_container_width=True,
    disabled=run_disabled,
)

if run_clicked and uploaded is not None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / uploaded.name
        input_path.write_bytes(uploaded.getbuffer())
        output_dir = tmp_path / "output"

        with st.status("Обработка...", expanded=True) as status:
            def progress(message: str) -> None:
                status.write(iconize(message))

            try:
                result = run_pipeline(
                    input_path,
                    output_dir,
                    diarize_flag=diarize,
                    skip_llm=skip_llm,
                    progress_cb=progress,
                )
            except Exception as exc:  # noqa: BLE001 - показываем ошибку пользователю
                status.update(label="Ошибка", state="error")
                st.error(str(exc), icon="🚫")
                st.stop()

            status.update(label="Готово", state="complete")

        st.success("Транскрибация завершена", icon="✅")

        downloads = [
            ("Протокол", "docx", "📝", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            ("Текст", "txt", "📄", "text/plain"),
            ("Субтитры", "srt", "🎬", "text/plain"),
        ]
        cols = st.columns(3)
        for col, (label, key, icon, mime) in zip(cols, downloads):
            path = result.get(key)
            with col:
                with st.container(border=True):
                    st.markdown(f"{icon} **{label}**")
                    if path and path.exists():
                        st.download_button(
                            "Скачать",
                            data=path.read_bytes(),
                            file_name=path.name,
                            mime=mime,
                            use_container_width=True,
                        )
                    else:
                        st.caption("Недоступно")

        txt_path = result.get("txt")
        if txt_path and txt_path.exists():
            with st.expander("Предпросмотр текста"):
                st.text(txt_path.read_text(encoding="utf-8")[:5000])
