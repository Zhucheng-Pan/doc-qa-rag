import faiss
from sentence_transformers import SentenceTransformer

from loader import load_and_split

QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

class VectorRetriever:

    def __init__(self, model_name = "BAAI/bge-small-zh-v1.5"):
        """Initialize the retriever with a sentence transformer model"""
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def build(self, chunks):
        """Build the FAISS index from the provided text chunks"""
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)

    def search(self, query, top_k=5):
        q_vec = self.model.encode([QUERY_INSTRUCTION + query], normalize_embeddings=True)
        scores,ids = self.index.search(q_vec, top_k)
        results = []
        for score, i in zip(scores[0], ids[0]):
            if i == -1:
                continue
            results.append({**self.chunks[i], "score": float(score)})

        return results

if __name__ == "__main__":
    chunks = load_and_split()
    print(f"共有 {len(chunks)} 个文本块, 开始向量化...")

    retriever = VectorRetriever()
    retriever.build(chunks)
    print("索引建立完成，输入问题测试检索，直接按回车退出。\n")

    while True:
        query = input("请输入问题(输入回车退出)：").strip()
        if not query:
            break
        for rank, r in enumerate(retriever.search(query, top_k=3), start=1):
            preview = r["text"][:80].replace("\n", " ")
            print(f" {rank}. [{r['score']:.3f}] {r['source']} 第{r['chunk_index']}块: {preview}...")
        print()