# Qwen2.5-VL-3B + LViT

面向医学图像分割的研究项目：先建立冻结 Qwen 自动描述与 LViT 分割基线，再验证短描述、预测定位框与多尺度分割特征之间的空间对应机制。

## 当前状态

- Qwen 已在云端下载并通过离线单图推理验证。
- 阶段一固定用户确认的第二篇论文历史划分：legacy_subset_v1。
- 首个完整训练任务为 MoNuSeg，随后是 cov19、MosMedData。
- LViT 来自用户上传的改版仓库，训练环境适配属于阶段二。
- 数据、权重、缓存和运行结果保留在云端，不进入普通 Git。

## 固定数据规模

| 数据集 | 训练 | 验证 | 测试 |
|---|---:|---:|---:|
| monuseg | 24 | 6 | 14 |
| cov19 | 160 | 40 | 200 |
| mosmeddata | 160 | 40 | 273 |

已有划分种子未知，使用固定清单及哈希追溯。cov19 与 MosMedData 有已记录的病例/序列交叉，不能将当前历史协议表述为严格患者独立泛化实验。

## 目录

configs 保存协议配置；docs 保存文档；manifests/legacy_subset_v1 保存清单与哈希；scripts 保存运行和检查脚本；requirements 保存环境记录；src/lvit 保存源码副本。

datasets、weights、cache、outputs 保存本地数据和运行产物，仅其说明文档进入 Git。

完整路线见 [项目任务路线与实验计划](项目任务路线与实验计划.md)，阶段一见 [数据与实验协议](docs/阶段一数据与实验协议.md)。

## 验证阶段一

在项目根目录运行：

```bash
python scripts/verify_stage1.py
```

该命令检查数据副本、哈希、清单、Excel 映射及协议一致性，不训练模型或下载数据。src/lvit/requirements.txt 是旧依赖记录，不应直接覆盖安装到当前 Qwen 环境。

## Qwen 已完成验证

RTX 4080 SUPER，约 32GB 显存；Python 3.12.3、PyTorch 2.5.1+cu124。640×384 合成图，BF16、SDPA、batch=1：加载 2.6376 秒；首次生成 2.2727 秒；后两次平均 1.1922 秒；整卡显存采样峰值 7.9766GiB。

这些结果仅证明功能可用，不代表医学图像质量。

## 来源

LViT 原项目：[HUANGLIZI/LViT](https://github.com/HUANGLIZI/LViT)。用户上传版本的来源见 [SOURCE](src/lvit/SOURCE.md)，原许可证保留于 src/lvit/LICENSE。

数据和预训练模型遵循各自使用条件，本仓库不重新分发原始数据或 Qwen 权重。
