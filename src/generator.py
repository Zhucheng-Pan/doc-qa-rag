import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SYSTEM_PROMPT = """你是一个知识库问答助手，你的任务是根据用户的问题和提供的上下文信息，给出准确、简洁的回答。请确保你的回答基于上下文信息，并且尽量避免编造内容。如果上下文中没有相关信息，请礼貌地告知用户无法回答。
规则：
1. 仅使用提供的上下文信息回答问题，不要引入外部知识。
2. 如果上下文中没有相关信息，直接回答「参考资料中没有找到相关信息」，不要编造。
3. 每句关键信息后面用 [编号] 标注出处，比如「硕士每人每月800元[2]」。
4. 如果不同资料的说法不一致，分别列出，并说明各自来自哪份文件、发布时间是什么。
5. 回答简洁清楚，用中文。
"""

def build_prompt(query, chunks):
    """combine the user query and context chunks and id the source of each chunk"""
    parts = []
    for i, c in enumerate(chunks, start=1):
        header = f"[{i}] 来源: {c['source']} (第{c['chunk_index']}块)"
        parts.append(f"{header}\n{c['text']}")
    context = "\n\n".join(parts)
    return f"【参考资料】\n{context}\n\n【问题】\n{query}"

class Generator:
    def __init__(self, model = "deepseek-chat", base_url="https://api.deepseek.com"):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=base_url,
        )
        self.model = model

    def generate_answer(self, query, chunks):
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(query, chunks)}
        ]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
        )
        return resp.choices[0].message.content