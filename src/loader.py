import re
from pathlib import Path
from collections import Counter
from pypdf import PdfReader

def load_documents(data_dir="data"):
    """Load all txt and pdf files in the data directory and return [{source, text},...]"""

    docs = []
    for path in Path(data_dir).iterdir():
        suffix = path.suffix.lower()
        if suffix == ".txt":
            text = path.read_text(encoding="utf-8")
        elif suffix == ".pdf":
            reader = PdfReader(path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            continue
        docs.append({"source": path.name, "text": text.strip()})
    return docs

def split_sentences(text):
    """Splits text into sentences using (.!?;)"""
    sentences = []
    parts = re.split(r"(?<=[。！？；\n])", text)
    for p in parts:
        if p.strip():
            sentences.append(p)
        elif sentences:
            sentences[-1] += p
    return sentences

def split_text(text, chunk_size =500, overlap = 100):
    """Split text into chunks of chunk_size with overlap"""
    sentences = []
    for s in split_sentences(text):
        for i in range(0, len(s), chunk_size):
            sentences.append(s[i:i+chunk_size])

    chunks = []
    current = []
    for sent in sentences:
        current_len = sum(len(s) for s in current)
        if current_len + len(sent) > chunk_size:
            chunks.append("".join(current))
            tail = []
            for s in reversed(current):
                if sum(len(t) for t in tail) + len(s) > overlap:
                    break
                tail.insert(0, s)

            if sum(len(t) for t in tail) + len(sent) > chunk_size:
                tail = []

            current = tail
        current.append(sent)
    if current:
        chunks.append("".join(current))
    return chunks



def load_and_split(data_dir = "data", chunk_size = 500, overlap = 100):
    """Load all documents and split them into chunks"""
    docs = load_documents(data_dir)
    chunks = []
    for doc in docs:
        text_chunks = split_text(doc["text"], chunk_size, overlap)
        for i, chunk in enumerate(text_chunks):
            chunks.append({"source": doc["source"], "chunk_index": i, "text": chunk})
    return chunks

if __name__ == "__main__":
    chunks = load_and_split()
    print(f"共有 {len(chunks)} 个文本块")

    counts = Counter(c["source"] for c in chunks)

    for source, n in counts.items():
        print(f"{source}: {n} 个文本块")

    print("\n第一块的结尾：", chunks[0]["text"][-50:])