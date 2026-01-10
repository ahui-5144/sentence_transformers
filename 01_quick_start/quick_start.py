
from sentence_transformers import SentenceTransformer

# 1.加载预训练的句子转换器模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 待编码的句子
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium",
]

# 2. 计算向量
embeddings = model.encode(sentences)
print(embeddings.shape)
#(3, 384)

# 3. 计算嵌入向量相似度
similarities = model.similarity(embeddings, embeddings)
print(similarities)
#tensor([[1.0000, 0.6660, 0.1058],
#        [0.6660, 1.0000, 0.1471],
#        [0.1058, 0.1471, 1.0000]])

"""
这个和之前学习rag的时候使用的LlamaIndex 中的HuggingFaceEmbedding的区别
  主要区别总结
  ┌────────────┬───────────────────────────┬──────────────────────────┐
  │    特性    │        LlamaIndex         │  Sentence-Transformers   │
  ├────────────┼───────────────────────────┼──────────────────────────┤
  │ 易用性     │ 通过 LlamaIndex 生态封装  │ 原生库，更直观           │
  ├────────────┼───────────────────────────┼──────────────────────────┤
  │ 返回值     │ list，需转 np.array       │ 直接 np.array            │
  ├────────────┼───────────────────────────┼──────────────────────────┤
  │ 相似度计算 │ 需手动实现                │ 内置 model.similarity()  │
  ├────────────┼───────────────────────────┼──────────────────────────┤
  │ 模型选择   │ bge-base-en-v1.5 (768维)  │ all-MiniLM-L6-v2 (384维) │
  ├────────────┼───────────────────────────┼──────────────────────────┤
  │ 适用场景   │ RAG 系统，配合 LlamaIndex │ 通用语义相似度任务       │
  └────────────┴───────────────────────────┴──────────────────────────┘
  两者的核心原理相同，都是将文本转换为高维向量，但 Sentence-Transformers 作为专门库，提供了更友好的 API 和更多功能（如内置相似度计算）。


"""