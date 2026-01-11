from sentence_transformers import SparseEncoder

# 1. 加载一个预训练的 SparseEncoder 模型
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

# 2. Calculate sparse embeddings by calling model.encode() 稀疏嵌入
embeddings = model.encode(sentences)
print(embeddings.shape) # sparse representation with vocabulary size dimensions 具有词汇量维度的稀疏表示
#torch.Size([3, 30522])

# 3. Calculate the embedding similarities (using dot product by default) 计算嵌入相似度（默认使用点积）
similarities = model.similarity(embeddings, embeddings)
print(similarities)
"""
tensor([[3.5629e+01, 9.1541e+00, 9.8058e-02],
        [9.1541e+00, 2.7478e+01, 1.9062e-02],
        [9.8058e-02, 1.9062e-02, 2.9553e+01]], device='cuda:0')
"""

# 4. Check sparsity statistics 查看稀疏度统计
stats = SparseEncoder.sparsity(embeddings)
print(f"Sparsity: {stats['sparsity_ratio']:.2%}") #Sparsity: 99.84%     99.84% 的维度是 0
print(f"Avg non-zero dimensions per embedding: {stats['active_dims']:.2f}") #Avg non-zero dimensions per embedding: 49.67    平均每个句子只有 50 个非零维度

