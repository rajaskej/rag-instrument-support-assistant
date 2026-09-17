from core.config import MAX_CHUNK_TOKENS
from core.ingestion.models import Block, BlockType, Chunk


def _token_count(text: str) -> int:
    return len(text.split())


def chunk_blocks(
    blocks: list[Block],
    doc_id: str,
    model_number: str | None,
    doc_type: str,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    buffer_blocks: list[Block] = []
    buffer_tokens = 0
    current_heading: list[str] | None = None

    def make_chunk(heading_path: list[str], text: str) -> Chunk:
        section_path = " > ".join(heading_path)
        return Chunk(
            chunk_id=f"{doc_id}::{len(chunks)}",
            text=f"{section_path}\n\n{text}",
            doc_id=doc_id,
            model_number=model_number,
            section_path=section_path,
            doc_type=doc_type,
        )

    def flush() -> None:
        nonlocal buffer_blocks, buffer_tokens
        if not buffer_blocks:
            return
        text = "\n\n".join(b.content for b in buffer_blocks)
        chunks.append(make_chunk(buffer_blocks[0].heading_path, text))
        buffer_blocks = []
        buffer_tokens = 0

    for block in blocks:
        if block.heading_path != current_heading:
            flush()
            current_heading = block.heading_path

        if block.block_type == BlockType.TABLE:
            flush()
            chunks.append(make_chunk(block.heading_path, block.content))
            continue

        if block.block_type == BlockType.LIST and _token_count(block.content) > max_tokens:
            flush()
            chunks.extend(_split_list_block(block, doc_id, model_number, doc_type, max_tokens, len(chunks)))
            continue

        block_tokens = _token_count(block.content)
        if buffer_blocks and buffer_tokens + block_tokens > max_tokens:
            flush()
            current_heading = block.heading_path

        buffer_blocks.append(block)
        buffer_tokens += block_tokens

    flush()
    return chunks


def _split_list_block(
    block: Block,
    doc_id: str,
    model_number: str | None,
    doc_type: str,
    max_tokens: int,
    start_index: int,
) -> list[Chunk]:
    steps = [line for line in block.content.split("\n") if line.strip()]
    groups: list[list[str]] = []
    buffer: list[str] = []
    buffer_tokens = 0

    for step in steps:
        step_tokens = _token_count(step)
        if buffer and buffer_tokens + step_tokens > max_tokens:
            groups.append(buffer)
            buffer = []
            buffer_tokens = 0
        buffer.append(step)
        buffer_tokens += step_tokens
    if buffer:
        groups.append(buffer)

    section_path = " > ".join(block.heading_path)
    return [
        Chunk(
            chunk_id=f"{doc_id}::{start_index + i}",
            text=f"{section_path}\n\n" + "\n".join(group),
            doc_id=doc_id,
            model_number=model_number,
            section_path=section_path,
            doc_type=doc_type,
        )
        for i, group in enumerate(groups)
    ]
