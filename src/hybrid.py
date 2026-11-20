import re
import jieba
from rank_bm25 import BM25Okapi
from loader import load_and_split
from retriever import VectorRetriever

jieba.setLogLevel(20)  # make the output less verbose

for word in ["宁人社规", "苏人保规", "苏府规字", "沪人才规"]:
    jieba.add_word(word)  # add custom words to the dictionary

STOPWORDS = {"需要","的", "了", "吗", "呢", "是", "在", "和", "与", "及", "或", "我", "什么", "怎么", "如何", "多少"}

def tokenize(text):
    """tokenize the text using jieba and remove stopwords"""
    words = jieba.lcut(text)
    return [w for w in words if w not in STOPWORDS and re.search(r'\w', w)]

class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.chunks = []

    def build(self, chunks):
        """ build the BM25 index from the provided chunks"""
        self.chunks = chunks
        corpus = [tokenize(c['text']) for c in chunks]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query, top_k=5):
        scores = self.bm25.get_scores(tokenize(query))
        ranked  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return[
            {**self.chunks[i], 'score': scores[i]} for i in ranked[:top_k] if scores[i] > 0
        ]


def rrf_fuse(result_lists, k=60):
    """RRF: Reciprocal Rank Fusion, a method to combine multiple ranked lists into a single ranked list."""
    fused={}
    for results in result_lists:
        for rank, r in enumerate(results, start=1):
            key = (r['source'], r['chunk_index'])
            if key not in fused:
                fused[key] = {**r, "score": 0.0}

            fused[key]["score"] += 1 / (k + rank)

    return sorted(fused.values(), key=lambda r: r["score"], reverse=True)

class HybridRetriever:
    def __init__(self):
        self.bm25 = BM25Retriever()
        self.vector = VectorRetriever()

    def build(self, chunks):
        self.bm25.build(chunks)
        self.vector.build(chunks)

    def search(self, query, top_k=5, candidates = 20):
        vec_results = self.vector.search(query, top_k=candidates)
        bm25_results = self.bm25.search(query, top_k=candidates)
        return rrf_fuse([vec_results, bm25_results])[:top_k]

def show(title, results):
    print(f" 【{title}】")
    for rank, r in enumerate(results, start =1):
        print(f" {rank}. [{r['score']:.3f}] {r['source']} 第{r['chunk_index']}块")

if __name__ == "__main__":
    chunks = load_and_split("data")
    retriever = HybridRetriever()
    retriever.build(chunks)
    print("索引建立完成，输入问题对比三种检索，直接回车退出。\n")

    while True:
        query = input("请输入问题：").strip()
        if not query:
            break

        print("分词结果：", tokenize(query))
        print("\n检索结果对比：")
        show("向量检索", retriever.vector.search(query))
        show("BM25检索", retriever.bm25.search(query))
        show("混合检索", retriever.search(query))
        print("\n")