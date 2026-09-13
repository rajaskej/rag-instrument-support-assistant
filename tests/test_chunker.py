from core.ingestion.models import Block, BlockType, Chunk
from core.ingestion.chunker import chunk_blocks


def test_text_blocks_under_same_heading_are_combined_into_one_chunk():
    blocks = [
        Block(["Overview"], BlockType.TEXT, "First sentence."),
        Block(["Overview"], BlockType.TEXT, "Second sentence."),
    ]
    chunks = chunk_blocks(blocks, doc_id="doc1", model_number="DM-1000", doc_type="manual")
    assert len(chunks) == 1
    assert "First sentence." in chunks[0].text
    assert "Second sentence." in chunks[0].text
    assert chunks[0].section_path == "Overview"
    assert chunks[0].doc_id == "doc1"
    assert chunks[0].model_number == "DM-1000"
    assert chunks[0].doc_type == "manual"


def test_heading_change_starts_a_new_chunk():
    blocks = [
        Block(["Overview"], BlockType.TEXT, "Overview text."),
        Block(["Specifications"], BlockType.TEXT, "Spec text."),
    ]
    chunks = chunk_blocks(blocks, doc_id="doc1", model_number=None, doc_type="manual")
    assert len(chunks) == 2
    assert chunks[0].section_path == "Overview"
    assert chunks[1].section_path == "Specifications"


def test_table_is_never_split_and_gets_its_own_chunk():
    blocks = [
        Block(["Specifications"], BlockType.TEXT, "Intro text."),
        Block(["Specifications"], BlockType.TABLE, "| Range | 0-3 |"),
        Block(["Specifications"], BlockType.TEXT, "Trailing text."),
    ]
    chunks = chunk_blocks(blocks, doc_id="doc1", model_number=None, doc_type="manual")
    table_chunks = [c for c in chunks if "Range" in c.text]
    assert len(table_chunks) == 1
    assert table_chunks[0].text == "| Range | 0-3 |"


def test_oversized_list_block_is_split_only_at_step_boundaries():
    long_step = "word " * 100
    steps = "\n".join(f"{i}. {long_step}" for i in range(1, 8))
    blocks = [Block(["Calibration Procedure"], BlockType.LIST, steps)]
    chunks = chunk_blocks(blocks, doc_id="doc1", model_number=None, doc_type="manual", max_tokens=300)
    assert len(chunks) > 1
    for chunk in chunks:
        first_line = chunk.text.splitlines()[0]
        assert first_line.strip()[0].isdigit()
        assert len(chunk.text.split()) <= 300


def test_chunk_ids_are_unique_and_sequential():
    blocks = [
        Block(["Overview"], BlockType.TEXT, "A"),
        Block(["Specifications"], BlockType.TEXT, "B"),
    ]
    chunks = chunk_blocks(blocks, doc_id="doc1", model_number=None, doc_type="manual")
    ids = [c.chunk_id for c in chunks]
    assert ids == ["doc1::0", "doc1::1"]
