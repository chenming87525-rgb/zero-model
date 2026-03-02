import json
import numpy as np

# ====== 1. 加载词表 ======
with open("Tokenizer-full.json", "r", encoding="utf-8") as f:
    dicts = json.load(f)

id2word = {v: k for k, v in dicts.items()}
V = len(dicts)

# ====== 2. 加载模型参数 ======
data = np.load("model_weights.npz")
W_out = data["W_out"]
b_out = data["b_out"]

# ====== 3. 加载词向量 ======
with open("word2vec.json", "r", encoding="utf-8") as f:
    word2vec_json = json.load(f)

word2vec = {int(k): np.array(v) for k, v in word2vec_json.items()}

print("模型加载成功！")

# ====== 4. 预测函数 ======
def predict_next_word(sentence, topk=5):
    # 分词
    temp = list(sentence)
    words = []
    for k in temp:
        words.append(dicts[k])

    if len(words) == 0:
        print("输入词不在词表中")
        return

    # 取所有词的向量
    context_vecs = [word2vec[i] for i in words]

    # 平均
    h = np.mean(context_vecs, axis=0)

    # 前向传播
    z = np.dot(W_out, h) + b_out
    exp_z = np.exp(z - np.max(z))
    y_pred = exp_z / np.sum(exp_z)

    # 取概率最高的 topk 个词
    top_indices = np.argsort(y_pred)[-topk:][::-1]

    print("输入：", sentence)
    print("预测结果：")
    for idx in top_indices:
        print(id2word[idx], "概率:", float(y_pred[idx]))

# ====== 5. 交互式预测 ======
while True:
    sentence = input("\n请输入一句话（输入exit退出）：")
    if sentence == "exit":
        break
    predict_next_word(sentence)