from loader import load_and_split
from reranker import RerankRetriever 
from generator import Generator

TOP_K = 5

def main():
    chunks = load_and_split()
    print(f"共有 {len(chunks)} 个文本块, 开始向量化...")

    retriever = RerankRetriever() 
    retriever.build(chunks)
    print("索引建立完成，输入问题测试检索，直接按回车退出。\n")

    generator = Generator()

    while True:
        query = input("请输入问题(输入回车退出)：").strip()
        if not query:
            break
        results = retriever.search(query, top_k=TOP_K)
        for rank, r in enumerate(results, start=1):
            preview = r["text"][:80].replace("\n", " ")
            print(f" {rank}. [{r['score']:.3f}] {r['source']} 第{r['chunk_index']}块: {preview}...")
        print("\n正在生成答案，请稍候...\n")
        answer = generator.generate_answer(query, results)
        print(f"【回答】\n{answer}\n")


if __name__ == "__main__":
    main()