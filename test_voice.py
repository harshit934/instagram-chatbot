import unittest

from voice_worker import voice_for_language


class VoiceWorkerTests(unittest.TestCase):
    def test_uses_language_specific_voice(self):
        self.assertEqual(
            voice_for_language("te"),
            "te-IN-ShrutiNeural"
        )

    def test_uses_base_language_for_locale(self):
        self.assertEqual(
            voice_for_language("hi-IN"),
            "hi-IN-SwaraNeural"
        )

    def test_falls_back_to_english(self):
        self.assertEqual(
            voice_for_language("unknown"),
            "en-US-AriaNeural"
        )


if __name__ == "__main__":
    unittest.main()