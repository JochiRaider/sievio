import sievio.core.chunk as chunk_module
from sievio.core.chunk import ChunkPolicy, count_tokens, iter_chunk_dicts


def test_iter_chunk_dicts_code_single_chunk() -> None:
    text = "\n".join(
        [
            "def fib(n):",
            "    if n <= 1:",
            "        return n",
            "    return fib(n - 1) + fib(n - 2)",
        ]
    )
    policy = ChunkPolicy(mode="code", target_tokens=200, min_tokens=0, overlap_tokens=0)

    chunks = list(iter_chunk_dicts(text, mode="code", policy=policy))

    assert len(chunks) == 1
    assert chunks[0]["text"].rstrip("\n") == text.rstrip("\n")
    assert chunks[0]["n_tokens"] > 0
    assert chunks[0]["n_tokens"] < len(text)


def test_iter_chunk_dicts_doc_multi_chunk() -> None:
    md_text = """# Title

Intro paragraph that is reasonably long and mentions multiple concepts so that token counts accumulate quickly.

## Section One
Here is the first real paragraph. It has several sentences to make sure chunking has to consider boundaries and spacing across sentences. Another sentence here keeps things lengthy.

```
code block line 1
code block line 2
```

Another paragraph after the code block with some additional words to trigger more tokens. Ending here.

## Section Two
A short paragraph.

Final wrap up paragraph that should make the document long enough to split into more than one chunk when the target token size is small.
"""
    policy = ChunkPolicy(mode="doc", target_tokens=80, min_tokens=20, overlap_tokens=0)

    chunks = list(iter_chunk_dicts(md_text, mode="doc", policy=policy))
    joined = "".join(chunk["text"] for chunk in chunks)

    assert len(chunks) > 1
    assert joined == md_text
    for chunk in chunks[1:]:
        start = chunk["start"]
        assert start == 0 or md_text[start - 1].isspace()


def test_iter_chunk_dicts_mode_doc_vs_code() -> None:
    text = """# Heading

Alpha paragraph with enough content to require more than one block in doc mode when the target size is small.

## Details
More prose follows in this section to ensure that the doc splitter keeps headings and paragraphs together, while the code splitter treats everything as lines.
"""
    policy_doc = ChunkPolicy(mode="doc", target_tokens=80, min_tokens=20, overlap_tokens=0)
    policy_code = ChunkPolicy(mode="code", target_tokens=80, min_tokens=20, overlap_tokens=0)

    doc_chunks = list(iter_chunk_dicts(text, mode="doc", policy=policy_doc))
    code_chunks = list(iter_chunk_dicts(text, mode="code", policy=policy_code))

    assert doc_chunks != code_chunks
    assert len(doc_chunks) >= 1 and len(code_chunks) >= 1


def test_count_tokens_falls_back_without_tokenizer(monkeypatch) -> None:
    sample = "foo bar baz " * 50

    n1 = count_tokens(sample)
    monkeypatch.setattr(chunk_module, "_get_tokenizer", lambda name=None: None)
    n2 = count_tokens(sample)

    assert n2 > 0
    if n1 > 0:
        assert abs(n2 - n1) <= max(1, int(n1 * 0.5))
