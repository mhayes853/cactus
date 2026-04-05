import unittest
import json
import sys
from pathlib import Path

from src.downloads import ensure_model
from src.cactus import (
    cactus_init,
    cactus_destroy,
    cactus_complete,
    cactus_embed,
    cactus_image_embed,
    cactus_audio_embed,
    cactus_transcribe,
    cactus_grammar_init_json_schema,
    cactus_grammar_json_schema_default_options,
    cactus_grammar_destroy,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent.parent
ASSETS_DIR = PROJECT_ROOT / "tests" / "assets"

def _has_asset(name):
    return (ASSETS_DIR / name).exists()


class TestVLMModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.weights_dir = ensure_model("LiquidAI/LFM2-VL-450M")
        cls.model = cactus_init(str(cls.weights_dir), None, False)

    @classmethod
    def tearDownClass(cls):
        cactus_destroy(cls.model)

    def test_text_completion(self):
        messages = json.dumps([{"role": "user", "content": "What is 2+2?"}])
        response = cactus_complete(self.model, messages, None, None, None)
        result = json.loads(response)
        print(f"\n  completion: {json.dumps(result, indent=2)}")
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("success", False))
        self.assertGreater(len(result.get("response", "")), 0)

    def test_text_embedding(self):
        embedding = cactus_embed(self.model, "Hello world", True)
        print(f"\n  text embedding dim: {len(embedding)}, first 5: {embedding[:5]}")
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)

    @unittest.skipUnless(_has_asset("test_monkey.png"), "test_monkey.png not found")
    def test_image_embedding(self):
        embedding = cactus_image_embed(self.model, str(ASSETS_DIR / "test_monkey.png"))
        print(f"\n  image embedding dim: {len(embedding)}, first 5: {embedding[:5]}")
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)

    @unittest.skipUnless(_has_asset("test_monkey.png"), "test_monkey.png not found")
    def test_vlm_image_completion(self):
        messages = json.dumps([{
            "role": "user",
            "content": "Describe this image",
            "images": [str(ASSETS_DIR / "test_monkey.png")],
        }])
        response = cactus_complete(self.model, messages, None, None, None)
        result = json.loads(response)
        print(f"\n  vlm completion: {json.dumps(result, indent=2)}")
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("success", False))
        self.assertGreater(len(result.get("response", "")), 0)

    def test_grammar_constrained_json_schema(self):
        schema = json.dumps({
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "age": {"type": "integer"},
                "bio": {"type": "string", "minLength": 1}
            },
            "required": ["name", "age", "bio"],
            "additionalProperties": False
        })

        grammar = cactus_grammar_init_json_schema(schema, cactus_grammar_json_schema_default_options())
        self.assertIsNotNone(grammar)

        try:
            messages = json.dumps([{"role": "user", "content": "Create a fictional character."}])
            response = cactus_complete(self.model, messages, None, None, None, grammar=grammar)
            result = json.loads(response)
            print(f"\n  grammar constrained: {json.dumps(result, indent=2)}")

            self.assertIsInstance(result, dict)
            self.assertTrue(result.get("success", False))

            response_text = result.get("response", "")
            self.assertGreater(len(response_text), 0)

            parsed = json.loads(response_text)
            self.assertIsInstance(parsed.get("name"), str)
            self.assertGreater(len(parsed["name"]), 0)
            self.assertIsInstance(parsed.get("bio"), str)
            self.assertGreater(len(parsed["bio"]), 0)
            self.assertIsInstance(parsed.get("age"), int)
        finally:
            cactus_grammar_destroy(grammar)


class TestWhisperModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.weights_dir = ensure_model("openai/whisper-small")
        cls.model = cactus_init(str(cls.weights_dir), None, False)

    @classmethod
    def tearDownClass(cls):
        cactus_destroy(cls.model)

    @unittest.skipUnless(_has_asset("test.wav"), "test.wav not found")
    def test_transcription(self):
        prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
        response = cactus_transcribe(
            self.model,
            str(ASSETS_DIR / "test.wav"),
            prompt,
            None,
            None,
            None,
        )
        result = json.loads(response)
        print(f"\n  transcription: {json.dumps(result, indent=2)}")
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("success", False))
        self.assertIn("segments", result)
        self.assertGreater(len(result["segments"]), 0)

    @unittest.skipUnless(_has_asset("test.wav"), "test.wav not found")
    def test_audio_embedding(self):
        embedding = cactus_audio_embed(self.model, str(ASSETS_DIR / "test.wav"))
        print(f"\n  audio embedding dim: {len(embedding)}, first 5: {embedding[:5]}")
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)


if __name__ == "__main__":
    unittest.main()
