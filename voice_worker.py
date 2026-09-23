import argparse
import asyncio
import json
import os

VOICE_BY_LANGUAGE = {
    "en": "en-US-AriaNeural",
    "te": "te-IN-ShrutiNeural",
    "hi": "hi-IN-SwaraNeural",
    "ta": "ta-IN-PallaviNeural",
    "kn": "kn-IN-SapnaNeural",
    "ml": "ml-IN-SobhanaNeural",
    "bn": "bn-IN-TanishaaNeural",
    "mr": "mr-IN-AarohiNeural",
    "gu": "gu-IN-DhwaniNeural",
    "pa": "pa-IN-OjasNeural",
    "ur": "ur-IN-GulNeural",
    "ar": "ar-SA-ZariyahNeural",
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
}


def voice_for_language(language):
    language_code = language.split("-")[0].lower()
    return VOICE_BY_LANGUAGE.get(language_code, VOICE_BY_LANGUAGE["en"])


def transcribe(audio_path):
    from faster_whisper import WhisperModel

    model = WhisperModel(
        os.getenv("WHISPER_MODEL", "tiny"),
        device="cpu",
        compute_type="int8"
    )
    segments, info = model.transcribe(audio_path, language=None, vad_filter=True)
    text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
    print(json.dumps({
        "text": text,
        "language": info.language,
        "language_probability": info.language_probability,
    }, ensure_ascii=False))


async def speak(text, language, output):
    import edge_tts

    voice = voice_for_language(language)
    await edge_tts.Communicate(text, voice).save(output)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    transcribe_parser = subparsers.add_parser("transcribe")
    transcribe_parser.add_argument("audio")

    speak_parser = subparsers.add_parser("speak")
    speak_parser.add_argument("--text", required=True)
    speak_parser.add_argument("--language", default="en")
    speak_parser.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "transcribe":
        transcribe(args.audio)
    else:
        asyncio.run(speak(args.text, args.language, args.output))


if __name__ == "__main__":
    main()
