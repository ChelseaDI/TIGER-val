# TIGER implementation

## 📚Dataset

The data used in this project comes from the paper [NineRec](https://arxiv.org/pdf/2309.07705).Data can be downloaded
from this [link](https://drive.google.com/file/d/15RlthgPczrFbP4U7l6QflSImK5wSGP5K/view). \
Specifically, __Bili_Cartoon__ from __Downstream Dataset__ is used in this project. After downloading, put it into the
folder `./data/Bili_Cartoon`

## 🪄Large Language Model

T5-base is chosen for this project, it can be downloaded
from [here](https://huggingface.co/google-t5/t5-base/tree/main).

After downloading, create a folder __LLM__ at the same level as the project, and create a subfolder named __t5-base__
under the folder. Put the downloaded large language model into the subfolder.

## 💡What improved in this project

This project implements four kinds of ID:<br />

-**SID from text**: Using SentenceT5 convert item title and description(concatenate) to get semantic embedding. Then training RQ-VAE to get SID<br />
-**SID from pretrained model**: As same as **SID from text**, but replacing semantic embedding with embedding table of pretrained sequential model(SASRec from RecBole)<br />
-**SID from text without conflict**: In **SID from text** doesn't solve hash conflict problem, this variant adds additional token to the end of SID(the same as TIGER)<br /> 
-**CID(chunked ID)**: Using k-base to get CID, which described in [MBGen](https://arxiv.org/pdf/2405.16871)


## 🚀Quick Start

___Step 1: generating text embedding___.

```bash
python ./data/generate_text_embed.py
```

___Step 2: processing interaction data___

```bash
python ./data/preprocess.py
```

___Step 3: item tokenization___

```bash
python ./tokenizer/main.py
```

___Step 4: supervised fine-tuning LLM___

```bash
python sh train.sh
```

___Step 5: evaluation___

```bash
python sh evaluation.sh
```

## 🙇Acknowledgement

Most of the code in this project references from [LC-Rec](https://arxiv.org/pdf/2311.09049). And I make some
improvements and add some comments.
## SID 冲突与类内选择实验

所有命令从 `TIGER-val` 目录运行。`sid` / `pretrained` 使用原始 D 位
`*_index.json`，允许多个 item 共用完整 SID，训练历史和标签均不追加消歧位。
`sid_nc` / `pretrained_nc` 使用 D+1 位 `*_index_nc.json`，最后一位消除冲突；
这里 **nc 表示 no conflict**，不是允许冲突。`cid` 不是本实验的语义 SID 对照。
原始 SID 分支已有支持，本次补充对照指标、映射校验、冲突统计及结果导出。

### 运行方法

准备 `data/Games/inter_data.json`、本地 `../LLM/t5-base` 和同一次 RQ-VAE
产生的 `tokenizer/sid_result/Games_sentence-t5-base/K=256_D=3_index.json`。
交互数据为 `{user_id: [item_id, ...]}`，item 必须在映射中，序列至少三项。
使用同一个原始映射生成消歧版本，**不要分别训练两套 RQ-VAE**：

```bash
(cd tokenizer && python solve_conflict.py --dataset Games --token_type sid --K 256 --D 3)

# 传统对照：D+1 位唯一 SID
python main.py --dataset Games --token_type sid_nc --K 256 --D 3 --precision fp32
python evaluation.py --dataset Games --token_type sid_nc --K 256 --D 3 --device cuda:0 --results_file results/Games_unique.json

# 实验组：D 位共享 SID；需要单独训练推荐模型
python main.py --dataset Games --token_type sid --K 256 --D 3 --precision fp32
python evaluation.py --dataset Games --token_type sid --K 256 --D 3 --device cuda:0 --results_file results/Games_shared.json
```

两组默认种子均为 2024；保持数据划分、初始 T5、训练参数及 beam 数一致。
模型分别保存到 `checkpoints/sid_nc/Games_sentence-t5-base/K=256_D=3/`
和 `checkpoints/sid/Games_sentence-t5-base/K=256_D=3/`，不会互相覆盖。
旧版 `sid_nc` 检查点可能位于不含 `_sentence-t5-base` 的目录，需先迁移到新路径。
若使用 SASRec embedding，将上述 token_type 替换为 `pretrained_nc` / `pretrained`，
映射目录为 `tokenizer/pretrained_result/Games/`，检查点数据集目录为 `Games/`。
训练及评估仍需安装原项目依赖；本次未执行完整模型训练。

### 指标含义

结果 JSON 包含参数、检查点路径、目录级冲突统计，以及 `all`、`collision`、
`singleton` 三个测试样本分层。分层依据 ground truth 的原始 SID 类大小，
每层独立按用户样本数归一化，空层指标为空。`collision_rate = 1 - 类数/item 数`，
同时报告发生冲突的类数、这些类覆盖的 item 数和最大类大小。

- `item/hit@K`、`item/ndcg@K`：消歧模型的严格 item 指标。
- `sid_at_item_rank/*`：仍用原始 item 候选排名，只放宽为 D 位 SID 相等。
  同类多个候选只计首次命中，NDCG 不会重复累加。适合与同模型 item 指标对照。
- `sid_hit_item_miss/hit@K`：同一份排名中 SID 命中而 item 未命中的样本比例，
  是二者 Hit@K 之差，**不是**以 SID 命中样本数为分母的条件错误率。
- `sid_unique/*`：消歧模型 beam 结果按 SID 类去重，保留得分最高候选的类排名。
  只覆盖已经生成的 beam，既非全目录类概率求和，也不保证取得 K 个不同类。
- `sid/*`：共享 SID 模型直接生成 SID 类的指标；不随机选择类内 item，
  因而不输出无法确定的 item 指标。

默认输出 Hit@5/10 与 NDCG@5/10。可通过
`--metrics "['recall@5', 'recall@10', 'ndcg@5', 'ndcg@10']"` 输出 Recall；
单个 ground truth 时 Recall 等于 Hit。无效、非有限分数或重复候选被过滤，
不足 K 个候选时不会补入真实标签，缺失命中记为零。建议 beam 数不小于最大 K。

类级指标的命中条件更宽松，候选空间也更小；不同训练模型间的指标提升不能单独
证明“剩余误差由决策因素导致”。同模型 `sid_at_item_rank` 与 `item` 的差距，
尤其冲突组中的差距，可量化类内区分的改进空间。完整 SID 相同也不保证语义品类
相同，仍需检查类内物品语义一致性。后续结合决策特征消融和严格 item 指标提升，
才能进一步检验提案的机制假设。

验证评估逻辑：`python -m unittest discover -s tests -v`。
