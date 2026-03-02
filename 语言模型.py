import random
import json
import numpy as np

with open('output.txt', 'r', encoding='utf-8') as f:
    text = f.readlines()

print("语料加载成功！")

with open("Tokenizer-full.json", "r", encoding="UTF-8") as f:
    dicts = json.load(f)

print("词表加载成！")

V = len(dicts)
D = 128
lr = 0.05

W_out = np.random.randn(V, D) * 0.01
b_out = np.zeros(V)
y_true = np.zeros(V)

E = []
word2vec = {}

for keys in dicts.keys():
    word2vec[dicts[keys]] = [random.uniform(-0.01, 0.01) for _ in range(D)]

print(word2vec[0])
print("向量初始化成功！")
def savemodel():
    np.savez("model_weights.npz", W_out=W_out, b_out=b_out)
    word2vec_tolist = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in word2vec.items()}

    # 保存词向量
    with open("word2vec.json", "w", encoding="utf-8") as f:
        json.dump(word2vec_tolist, f, ensure_ascii=False)
    print("模型保存成功！")
maxtrain = len(text)
epoch_times = 10
loss = 0
for epoch in range(epoch_times):
    for a in range(0, maxtrain):
        temp = list(text[a])
        words = []
        for k in temp:
            words.append(dicts[k])
        contentsvec = []
        for i in range(1, len(words)):
            context = words[:i]  # 前面所有词
            target = words[i]
            # contentsvec.append(word2vec[str(i)])
            context_vecs = [word2vec[i] for i in context]
            h = np.mean(context_vecs, axis=0)
            z = np.dot(W_out, h) + b_out
            exp_z = np.exp(z - np.max(z))
            y_pred = exp_z / np.sum(exp_z)
            loss = -np.log(y_pred[target] + 1e-9)
            y_true = np.zeros(V)
            y_true[target] = 1

            dz = y_pred - y_true
            dW_out = np.outer(dz, h)
            db_out = dz

            W_out -= lr * dW_out
            b_out -= lr * db_out

            dh = np.dot(W_out.T, dz)

            grad_each = dh / len(context)

            for t in context:
                word2vec[t] -= lr * grad_each

        if a % 100 == 0:
            print(f"epoch {epoch + 1}, step {a}, loss={loss:.4f}")
            savemodel()
            print("保存成功")
    savemodel()
    print("保存成功")

        # result = np.sum(contentsvec, axis=0) / len(words)

        # print(result)

# 保存矩阵参数
