# LViT 现代运行适配（阶段二至四）

本适配从用户上传并归档的 `src/lvit` 源码构建原二分类网络，共 39,933,345 参数；未将该快照宣称为官方某个提交的逐字复现。来源与许可证仍见 `src/lvit/SOURCE.md`、`LICENSE`、`source_snapshot.json`。`src/lvit` 不作修改，模型结构及 state_dict 名称不改，不导入包含默认 `monuseg_2` 路径和环境副作用的 `Config.py`。

## 环境与接口

环境沿用 Python 3.12、Torch 2.5.1+cu124、torchvision 0.20.1+cu124、Transformers 4.51.3；仅新增 `requirements/lvit.txt` 的 timm 1.0.15 与 opencv-python-headless 4.11.0.86。timm 用于 vendor Vit.py 中 DropPath 导入，不需原清单中的 MXNet、bert-embedding、旧 numpy、pandas、scipy、ml_collections、TensorBoard 等。

模型工厂：`src.experiment.model.build_model()`；输入 `image:float32[B,3,224,224]` 和 `text:float32[B,10,768]`，输出 sigmoid 概率 `[B,1,224,224]`。网络从头初始化，不装载旧 LViT 权重。

## 数据、损失与指标

数据由 `configs/data_protocol_v1.json` 和冻结 CSV 唯一确定。启动时核对实际样本数与原图/掩码 SHA-256，缓存核对清单、协议和文本来源。图像 OpenCV BGR、bilinear 224、除 255；掩码先将原灰度 `>0` 二值化再 nearest 224。两组均不做几何增强，避免图像与描述中左右、上下信息不一致。训练程序仅实例化 train/val loader，test 只在选定检查点后的独立评价命令中加载。

损失精确保留上传版本二分类 WeightedDiceBCE 的默认训练权重：0.5 WeightedDiceLoss + 0.5 WeightedBCE。Dice 的 p/t 各乘 0.5，分母使用平方和，smooth=1e-5；BCE 在整个 batch 内按正负像素分别归一化，各占 0.5。全空/全前景样本缺失的一类贡献为0，这是原公式的行为。此 soft Dice 与报告 hard Dice 不是同一指标；适配代码没有调用原 `_show_dice`。`check_lvit.py` 用 AST 提取原损失类，验证混合、全空、全前景三种输入的值和梯度与适配实现一致。

报告直接使用 `scripts/protocol_metrics.py`，概率 `>=0.5`，逐图宏平均、224 网格、实际样本数分母；双空=1，单空=0。最佳检查点按 val macro Dice 选取，严格大于才替换，平分保留最早。JSON 同时保留 fraction 和显式 percent 字段及每样本结果。

## 文本兼容性与公平比较

核对旧 [bert-embedding 实现](https://github.com/imgarylai/bert-embedding/blob/master/bert_embedding/bert.py) 后确认：默认 uncased BERT，最大序列25（含 special），最终隐藏层，移除 CLS/SEP/PAD，将连续 `##` 子词平均后合并为一个词。上传 DataLoader 再取合并后前10项，原 utils.read_text 为不足9空格词的描述添加重复 ` EOF XXX`，随后 DataLoader 只使用第一行。

现代适配冻结本地 `weights/bert-base-uncased`，采用 Hugging Face BertTokenizer/BertModel，在 eval/no-grad 下用相同的25位置、连续 `##` 平均、合并词前10项语义。保留上传 filler 和第一行政策；仍不足10位置时显式补零，避免旧版短句变长引发 batch 拼接失败。legacy 和 Qwen 两组共享这套政策及相同权重/分词文件 SHA-256。这里不是“截取10个 subword”；每个样本缓存选中词、实际 WordPiece、补零数、原文/准备后文本 hash，便于审查长文本被截断的实际情况。

这保证实验组间编码设置一致，不保证与 GluonNLP/MXNet 旧 bert-embedding 数值等价：权重格式/实现、special token ID、tokenizer细节、短句补齐均可能不同。旧 checkpoint 即使形状相容也不能据此宣称性能等价；本轮全部从头训练。provided_legacy_text 来源未独立核实，可能含额外标签信息，不当作自然可获得的临床报告；Qwen 普通描述是另一输入条件。

缓存 `cache_text_features.py` 从 legacy Excel 原文或 Qwen JSONL `plain_text` 读取。Qwen JSONL 必须恰好包含全体清单 ID、image_path、image_sha256、dataset、split 和同一 config_sha256、model_sha256；不读取掩码生成文本。特征metadata显式记录生成模型hash集合。文本编码缓存阶段核对数据 hash、生成配置和编码器文件 hash，记录CPU/GPU时间与缓存空间。训练每个 epoch 不再调用 Qwen/BERT。

## 训练、恢复和测量

固定 Adam lr=1e-3、batch=2、weight_decay=0，完整 float32；参考原 CosineAnnealingWarmRestarts(T_0=10,T_mult=1,eta_min=1e-4)，使用现代 Torch 实现并在 epoch末更新。配置示例 max_epochs=2000、min_epochs=50、patience=50；最终统一预算由速度/过拟合验收后主代理冻结，在两种文本来源及 seeds 666/667/668 中保持一致。

种子控制 Python、NumPy、Torch、CUDA；禁用 TF32/cuDNN benchmark，开启 deterministic algorithms，CLI 在导入 torch 前设置 CUBLAS_WORKSPACE_CONFIG=:4096:8。shuffle 使用独立 Torch Generator，以 seed+epoch×100003 确定当前 epoch 顺序；无随机数据增强。空文本或第一行空白会拒绝，避免原 first-line 政策静默产生全零文本特征。

每个正式 run 只保留 best.pt 与 last.pt，通过临时文件 fsync 后原子替换。检查点包含模型、Adam moments、scheduler、Python/NumPy/Torch/所有CUDA RNG、完整配置/协议/清单/缓存/代码指纹、epoch、步数和最优 val。恢复粒度为“完整 epoch”；中断时最多重跑当前未完成 epoch，不承诺 batch 中途恢复。`--resume` 要求原输出目录保留 best.pt，并拒绝配置/缓存/代码漂移。若当前 epoch 已保存而日志写入前中断，history 可能缺该行，应以检查点 epoch 为准；若检查点已达停止规则，恢复不再多训练一轮。正式评价也拒绝代码/指标/配置/缓存/清单漂移，test 入口要求检查点epoch等于其记录的best_epoch，防止误用非最优last。

每轮日志记录实际样本数、loss、梯度L2、概率范围/均值、前景比例、train/val同步耗时、吞吐与Torch分配/保留显存峰值；正式结果另记总墙钟时间。梯度范数逐批检查，计入训练速度，checkpoint I/O 不计入 train samples/s。过拟合检查使用真实 train 固定batch；保存恢复检查比较包含dropout的下一更新的所有参数/BN缓冲、Adam矩、scheduler、loss是否严格零差异。测试掩码仅参与最终分割评分，不参与Qwen提示。

## 命令示例（项目根）

```bash
python scripts/cache_text_features.py --dataset monuseg --source provided_legacy_text --output cache/text_features/monuseg_provided_legacy_text_v1.pt --device cpu
python scripts/cache_text_features.py --dataset monuseg --source qwen_predicted_text --descriptions cache/qwen/monuseg_plain_v1.jsonl --output cache/text_features/monuseg_qwen_predicted_text_v1.pt --device cpu
python scripts/check_lvit.py --config configs/training_monuseg_legacy_v1.json --output-dir outputs/lvit_checks/legacy --mode full --steps 100
python scripts/train_lvit.py --config configs/training_monuseg_legacy_v1.json --seed 666 --output-dir outputs/lvit/monuseg/legacy_seed666
python scripts/train_lvit.py --config configs/training_monuseg_legacy_v1.json --seed 666 --output-dir outputs/lvit/monuseg/legacy_seed666 --resume outputs/lvit/monuseg/legacy_seed666/last.pt
python scripts/train_lvit.py --evaluate-only --checkpoint outputs/lvit/monuseg/legacy_seed666/best.pt --split test --output-dir outputs/lvit/monuseg/legacy_seed666/test --save-predictions
python scripts/infer_lvit.py --image /path/new_image.png --description-file /path/qwen_plain_text.txt --checkpoint /path/best.pt --output-dir outputs/new_image
```

`infer_lvit.py` 接收已从新图生成的描述，不读取GT；联合 Qwen→此入口由主代理串联。输出224概率图/二值PNG，记录编码和分割前向耗时；生成描述耗时应由上游联合入口单列。
