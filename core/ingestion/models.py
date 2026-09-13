from dataclasses import dataclass
from enum import Enum


class BlockType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    LIST = "list"


@dataclass
class Block:
    heading_path: list[str]
    block_type: BlockType
    content: str


@dataclass
class Chunk:
    chunk_id: str
    text: str
    doc_id: str
    model_number: str | None
    section_path: str
    doc_type: str
