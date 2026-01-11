import numpy as np
from sentence_transformers import CrossEncoder

# 1. Load a pretrained CrossEncoder model 加载预训练的CrossEncoder模型
model = CrossEncoder("cross-encoder/stsb-distilroberta-base")

# We want to compute the similarity between the query sentence...
query = "A man is eating pasta."

# ... and all sentences in the corpus
corpus = [
    "A man is eating food.",
    "A man is eating a piece of bread.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
    "A woman is playing violin.",
    "Two men pushed carts through the woods.",
    "A man is riding a white horse on an enclosed ground.",
    "A monkey is playing drums.",
    "A cheetah is running behind its prey.",
]

# 2. We rank all sentences in the corpus for the query 我们对语料库中的所有句子进行排序
ranks = model.rank(query, corpus)
print("Query:", query)
for rank in ranks:
    print("Rank:", rank)
    print(f"{rank['score']:.2f}\t{corpus[rank['corpus_id']]}")

#corpus_id: 0 → 表示这是 corpus 列表中索引为 0 的句子

"""console
Query: A man is eating pasta.
Rank: {'corpus_id': 0, 'score': np.float32(0.6732371)}
0.67	A man is eating food.
Rank: {'corpus_id': 1, 'score': np.float32(0.34102532)}
0.34	A man is eating a piece of bread.
Rank: {'corpus_id': 3, 'score': np.float32(0.07569342)}
0.08	A man is riding a horse.
Rank: {'corpus_id': 6, 'score': np.float32(0.06676244)}
0.07	A man is riding a white horse on an enclosed ground.
Rank: {'corpus_id': 2, 'score': np.float32(0.005424653)}
0.01	The girl is carrying a baby.
Rank: {'corpus_id': 5, 'score': np.float32(0.005368141)}
0.01	Two men pushed carts through the woods.
Rank: {'corpus_id': 7, 'score': np.float32(0.0053482456)}
0.01	A monkey is playing drums.
Rank: {'corpus_id': 4, 'score': np.float32(0.00525378)}
0.01	A woman is playing violin.
Rank: {'corpus_id': 8, 'score': np.float32(0.005167174)}
0.01	A cheetah is running behind its prey.
"""

# 3. Alternatively, you can also manually compute the score between two sentences 或者，你也可以手动计算两句话之间的分数
sentence_combinations = []
for sentence in corpus:
    sentence_combinations.append((query, sentence))

scores = model.predict(sentence_combinations)

# Sort the scores in decreasing order to get the corpus indices 按分数递减排序以获得语料库索引
"""
np.argsort 是 NumPy 中用于返回排序后索引的函数，而不是直接返回排序后的值。

  基本语法

  numpy.argsort(a, axis=-1, kind=None, order=None)

  核心概念

  返回的是索引数组，按照排序后的顺序。

  import numpy as np

  arr = np.array([30, 10, 20])
  indices = np.argsort(arr)
  print(indices)  # [1, 2, 0]

  解读：
  - 最小值 10 的索引是 1
  - 中间值 20 的索引是 2
  - 最大值 30 的索引是 0
# 方法1: 使用 [::-1] 反转

  与 np.sort 的区别
  ┌─────────────────┬──────────────┐
  │      函数        │    返回值    │
  ├─────────────────┼──────────────┤
  │ np.sort(arr)    │ 排序后的值   │
  ├─────────────────┼──────────────┤
  │ np.argsort(arr) │ 排序后的索引 │
  └─────────────────┴──────────────┘
  arr = np.array([30, 10, 20])
  np.sort(arr)     # [10, 20, 30]
  np.argsort(arr)  # [1, 2, 0]
"""

ranked_indices = np.argsort(scores)[::-1]
print("Scores:", scores)
print("Indices:", ranked_indices)
""" console 
Scores: [0.6732371  0.34102532 0.00542465 0.07569342 0.00525378 0.00536814
 0.06676244 0.00534825 0.00516717]
Indices: [0 1 3 6 2 5 7 4 8]
"""


""" model.rank()和model.predict()的区别
  model.rank() 就是根据相似度分数对语料库句子进行排序。

  工作流程

  from sentence_transformers import CrossEncoder

  model = CrossEncoder("cross-encoder/stsb-distilroberta-base")
  query = "A man is eating pasta."
  corpus = ["句子1", "句子2", ...]

  # 内部发生了什么：
  ranks = model.rank(query, corpus)

  内部步骤

  # 1. 将 query 与 corpus 中每个句子配对
  pairs = [(query, sentence) for sentence in corpus]

  # 2. 模型预测每对的相似度分数
  scores = model.predict(pairs)
  # 例如: [0.67, 0.34, 0.01, 0.08, ...]

  # 3. 按 score 降序排序，记录原始索引（类似 argsort +[::-1]）
  ranks = [
      {'corpus_id': 0, 'score': 0.67},  # corpus[0] 最相似
      {'corpus_id': 1, 'score': 0.34},  # corpus[1] 第二
      {'corpus_id': 3, 'score': 0.08},  # corpus[3] 第三
      ...
  ]

  返回结果

  返回一个列表，按分数从高到低排列：
  ┌───────────┬────────────────────────┐
  │   字段    │          说明          │
  ├───────────┼────────────────────────┤
  │ corpus_id │ 原始 corpus 中的索引   │
  ├───────────┼────────────────────────┤
  │ score     │ 相似度分数（0-1 之间） │
  └───────────┴────────────────────────┘
  与 model.predict 的对比

  # predict: 只算分数，不排序
  scores = model.predict([(query, sent) for sent in corpus])
  # 返回: [0.67, 0.34, 0.01, 0.08, ...]

  # rank: 算分数 + 排序
  ranks = model.rank(query, corpus)
  # 返回: [{'corpus_id': 0, 'score': 0.67}, ...]

  所以 rank = predict + argsort + 结果封装，一步到位完成相似度排序。

"""