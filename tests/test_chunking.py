import unittest

from app.schemas import Document
from app.chunking import FixedChunker

class MyTestCase(unittest.TestCase):
    def test_chunk_document_produces_chunks(self):
        doc = Document(
            doc_id="wiki::Test",
            text=" ".join(["Ada"] * 1000),
            document_type="wiki_article",
            topic_id="Test Topic",
            timestamp=None,
            is_malicious=False,
            attack_type=None,
        )

        chunker = FixedChunker(
            tokenizer_name="Qwen/Qwen2.5-7B-Instruct",
            chunk_size=256,
            overlap=32,
        )

        chunks = chunker.chunk_document(doc)

        assert len(chunks) > 1
        assert chunks[0].doc_id == "wiki::Test"
        assert chunks[0].topic_id == "Test Topic"
        assert chunks[0].document_type == "wiki_chunk"
        assert chunks[0].text.strip() != ""


if __name__ == '__main__':
    unittest.main()
