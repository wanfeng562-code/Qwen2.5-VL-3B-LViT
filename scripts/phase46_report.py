"""Write a reviewable report from terminal, audited experiment artifacts."""
import csv
import datetime
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiment.common import atomic_json


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    summary = read('outputs/phase46/confirmation/summary.json')
    audit = read('outputs/phase46/development_audit.json')
    review = read('outputs/phase46/visual_review/review.json')
    assert audit['status'] == 'passed' and audit['experiment_status'] == summary['status']
    assert summary['status'] in ('validation_passed', 'stopped_validation_gate')
    assert review['independently_rescored_pngs'] == 72
    registry = read('outputs/phase46/registry.json')
    old = read('outputs/phase45/registry.json')
    gate = summary['validation_gate']
    final_path = ROOT / 'outputs/phase46/final_test/independent_audit.json'
    final = read(str(final_path.relative_to(ROOT))) if final_path.exists() else None
    if final:
        assert gate['passed'] and final['status'] == 'passed'
    else:
        assert not (ROOT / 'cache/phase46/V_mid_test.pt').exists(), 'Do not report incomplete final testing as untested'
    lines = ['# 阶段4.6：Qwen描述与视觉特征实验结果', '',
        '本报告由真实运行记录生成；日期：' + datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds') + '。', '',
        '## 结论与完成边界', '']
    if final:
        lines += [('**达到预定数值门槛。**' if final['target_achieved'] else '**未达到全部预定门槛，不能宣称目标已完成。**') +
                  '已完成四组条件各三种子的原划分最终复测；这14张测试图此前已查看，不称为新的盲测。']
    else:
        lines += ['**三种子验证门槛未通过，未进行新的测试集复测。**' if not gate['passed'] else
                  '**三种子验证门槛已通过，最终复测尚未执行。目标仍未验收。**']
    lines += ['', '本轮的正负结论仅针对MoNuSeg 24/6/14。Qwen视觉路线使用冻结预训练视觉塔，不证明自然语言描述或医学推理的贡献。cov19、MosMedData尚未完成本轮正式实验。', '',
        f"本轮正式训练{len(registry['runs'])}次，加旧阶段4.5的{len(old['runs'])}次，共{len(registry['runs'])+len(old['runs'])}/18次；S提示筛选未通过，不消耗正式训练名额。未用名额不自动转为额外搜索。LoRA未启动，训练成本为0。", '',
        '## 候选与匹配对照', '',
        'S在每个查询中提供三张训练示例，训练查询排除自身；30条回答仍为相同的high/unknown，验证字段准确率2/12，故跳过分割训练。V_mid使用视觉塔第16个block、1280维特征；V_merged使用最终merger、2048维特征。两者恢复空间顺序后池化到28×28，通过门控轻量卷积分支注入LViT最后解码特征。', '',
        '| seed666候选 | 最佳验证Dice% |', '|---|---:|']
    for name, value in summary['selection']['scores'].items():
        lines.append(f'| {name} | {100*value:.4f} |')
    lines += ['', '按预先规则选择V_mid；没有用测试成绩选择层或超参数。四组条件如下：', '',
        '| 条件 | 视觉输入 | 文本输入 |', '|---|---|---|',
        '| candidate | 对应图像的Qwen特征 | 24张训练原文的均值 |',
        '| constant | 24张训练视觉特征的逐位置均值 | 同一训练文本均值 |',
        '| shuffled | 各划分内部固定错位的视觉特征 | 同一训练文本均值 |',
        '| legacy | 同一训练视觉均值 | 对应图像的原有文本 |', '',
        '固定特征均不使用验证或测试均值。打乱偏移为1+seed%(划分大小-1)，不跨划分交换。同一种子的四组初始权重指纹相同。LViT图像仍为BGR224，Qwen图像输入上限1003520像素，新增分辨率和预训练资源需要披露；尚无与其他专业视觉编码器的公平比较。原文本来源未独立证实。', '',
        '## 三种子验证结果', '', '| 条件 | seed666 Dice% | seed667 Dice% | seed668 Dice% | 均值±样本标准差% |', '|---|---:|---:|---:|---:|']
    aggregates = {}
    for arm in ('candidate', 'legacy', 'constant', 'shuffled'):
        values = [100*next(r['best_validation_dice'] for r in summary['records'] if r['arm']==arm and r['seed']==seed) for seed in (666,667,668)]
        aggregates[arm] = dict(mean=statistics.mean(values), sd=statistics.stdev(values))
        lines.append(f"| {arm} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} | {statistics.mean(values):.4f} ± {statistics.stdev(values):.4f} |")
    lines += ['', '验证仅6张图，种子间标准差不代表跨病例总体不确定性，不作统计显著性结论。', '',
        '| 验证门槛 | 结果 |', '|---|---|']
    for key in ('beats_constant','beats_shuffled','not_below_legacy'):
        lines.append(f"| {key} | {'通过' if gate[key] else '未通过'} |")
    sensitivity = read('outputs/phase46/confirmation/seed666_sensitivity.json')
    lines += ['', f"已训练candidate666的五种验证错位使Dice平均下降{sensitivity['permuted_drop_pp']:.4f}个百分点。这说明推理依赖特征；它与重新训练的shuffled对照是不同证据，不能相互替代。", '',
        '所有开发决策只用训练/验证；FP32 Adam、学习率0.001、batch2、损失、无几何增强、max2000/min50/patience50和最佳验证Dice最早选模均保持冻结。', '',
        '## 全部正式运行与成本', '',
        '| 运行 | 验证Dice% | 验证IoU% | 实际/最佳epoch | 训练秒 | allocated/reserved峰值GiB |',
        '|---|---:|---:|---:|---:|---:|']
    total = 0
    for entry in registry['runs']:
        name = entry['run_id']; result = read('outputs/phase46/runs/'+name+'/result.json')
        metric = read('outputs/phase46/runs/'+name+'/best_validation.json')
        total += result['training_elapsed_seconds']
        lines.append(f"| {name} | {100*metric['dice']:.4f} | {100*metric['iou']:.4f} | {result['epochs_completed']}/{result['best_epoch']} | {result['training_elapsed_seconds']:.2f} | {result['peak_allocated_bytes']/2**30:.3f}/{result['peak_reserved_bytes']/2**30:.3f} |")
    lines += ['', f'本轮分割训练墙钟合计{total:.2f}秒（{total/60:.2f}分钟），不含Qwen/BERT缓存生成、文件校验、分析与模型加载。', '']
    extraction = read('cache/phase46/extraction_receipt.json')
    descriptions = [json.loads(line) for line in (ROOT/'cache/phase46/fewshot_attributes.jsonl').read_text().splitlines()]
    assert extraction['count']==len(descriptions)==30
    visual_seconds = [r['extract_seconds'] for r in extraction['feature_records']]
    generation_seconds = [r['generate_seconds'] for r in descriptions]
    lines += [f"30张训练/验证图的双层视觉缓存与三示例属性生成，生成器记录墙钟{extraction['wall_seconds']:.2f}秒，含模型加载{extraction['model_load_seconds']:.2f}秒，不含进程启动和此前模型文件hash。视觉提取平均{statistics.mean(visual_seconds):.4f}秒/图，属性generate平均{statistics.mean(generation_seconds):.4f}秒/图。属性生成显存allocated/reserved峰值{max(r['peak_allocated_bytes'] for r in descriptions)/2**30:.3f}/{max(r['peak_reserved_bytes'] for r in descriptions)/2**30:.3f}GiB。分步计时的边界与端到端测速不同，不简单相加为服务延迟。", '']
    benchmark_path = ROOT / 'outputs/phase46/benchmark.json'
    if benchmark_path.exists():
        b = read(str(benchmark_path.relative_to(ROOT))); timing = b['timings']
        lines += [f"第一张验证图batch1，完整Qwen权重与LViT同时驻留，两次预热后10次实测：平均端到端{timing['total_seconds']['mean']:.4f}秒，视觉提取{timing['vision_seconds']['mean']:.4f}秒，分割及二值图CPU传输{timing['segmentation_and_binary_copy_seconds']['mean']:.4f}秒；显存allocated/reserved峰值{b['peak_allocated_bytes']/2**30:.3f}/{b['peak_reserved_bytes']/2**30:.3f}GiB。模型加载{b['model_load_seconds']:.2f}秒单列。常量训练文本已预载，不运行语言解码器；端到端含图像读取/预处理，不含输出写盘或初始文件hash。缓存与实时视觉特征、分割概率逐元素一致。", '']
    else:
        lines += ['同时驻留单图测速尚未完成，不以缓存验证导出的耗时冒充完整推理耗时。', '']
    lines += ['## 独立核验与逐图结果', '',
        f"开发审查通过{audit['check_count']}项，复核旧文件与冻结核心、运行预算、最早最佳epoch、早停、配置/缓存/检查点指纹和同种子初始权重。全部四组×三种子×六张验证预测共{review['independently_rescored_pngs']}份PNG独立重算通过。", '',
        '![验证对照](phase46_evidence/comparison_validation.png)', '',
        '![训练与验证曲线](phase46_evidence/training_curves.png)', '']
    table = []
    for sid in [r['sample_id'] for r in review['metrics']['candidate']]:
        row = dict(sample_id=sid)
        for arm in ('candidate','constant','shuffled','legacy'):
            for metric in ('dice','iou','fp','fn'):
                values = [next(x[metric] for x in review['all_seed_metrics'][f'{arm}_{seed}'] if x['sample_id']==sid) for seed in (666,667,668)]
                row[arm+'_'+metric] = statistics.mean(values)
        table.append(row)
    out = ROOT / 'outputs/phase46'
    with (out/'validation_paired_per_image.csv').open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]),lineterminator='\n');w.writeheader();w.writerows(table)
    if final:
        lines += ['## 原划分最终复测', '', '| 条件 | 测试Dice均值±SD% | 测试IoU均值±SD% |', '|---|---:|---:|']
        for arm, values in final['aggregates'].items():
            d, i = values['dice'], values['iou']
            lines.append(f"| {arm} | {d['mean_percent']:.4f} ± {d['sample_sd_percent']:.4f} | {i['mean_percent']:.4f} ± {i['sample_sd_percent']:.4f} |")
        old_scores=[read(f'outputs/lvit/monuseg/legacy_seed{seed}/test/test_metrics.json') for seed in (666,667,668)]
        old_dice=100*statistics.mean(x['dice'] for x in old_scores)
        old_iou=100*statistics.mean(x['iou'] for x in old_scores)
        lines += ['', f"旧原文本基线测试Dice为{old_dice:.4f}%，IoU为{old_iou:.4f}%；本轮候选分别低{old_dice-final['aggregates']['candidate']['dice']['mean_percent']:.4f}和{old_iou-final['aggregates']['candidate']['iou']['mean_percent']:.4f}个百分点。相同新结构的原文本对照也低于旧基线，不能把视觉特征相对新对照的提升解释为整套框架超过原方案。本轮未分别隔离原文本编码变化和融合结构变化的因果影响。", '',
            '| 条件 | seed | 测试Dice% | 测试IoU% |', '|---|---:|---:|---:|']
        for arm in ('candidate','constant','shuffled','legacy'):
            for seed in (666,667,668):
                scores=final['records'][f'{arm}_{seed}']
                lines.append(f"| {arm} | {seed} | {100*statistics.mean(x['dice'] for x in scores):.4f} | {100*statistics.mean(x['iou'] for x in scores):.4f} |")
        lines += ['', '| 成功条件 | 结果 |', '|---|---|']
        for name, passed in final['success_gates'].items():
            lines.append(f"| {name} | {'通过' if passed else '未通过'} |")
        lines += ['', '三个种子相对旧原文本基线的Dice差值（百分点）：'+json.dumps(final['seed_deltas_vs_old_pp'])+'。', '',
            '独立重算168份新预测和42份旧原文本基线预测，共210份PNG；复算三种子均值、SD及全部六项成功条件。未根据复测结果重新选择候选或调整配置。', '',
            '![最终逐图对照](phase46_evidence/comparison_final.png)', '',
            '![误检漏检案例](phase46_evidence/delta_cases.png)', '',
            '![种子与逐图差值](phase46_evidence/final_scores.png)', '']
    lines += ['## 证据位置与剩余边界', '',
        '- 云端：`outputs/phase46/registry.json`、`confirmation/summary.json`、`development_audit.json`、`visual_review/`、`validation_paired_per_image.csv`。',
        '- 若已复测：`final_test/independent_audit.json`、`audited_per_image.csv`、各组各种子的预测PNG与metrics。',
        '- 计划、代码及配置纳入Git；数据、模型、缓存和检查点不纳入Git。',
        '- 图表生成不等于目视验收；最终人工/主代理看图记录另存，不能声称临床有效性。',
        '- 若数值门槛未全通过，保留当前瓶颈与后续假设，不降低门槛，不以单种子或单张改善宣称成功。更细空间特征、任务监督和其他视觉骨干比较仍是待检验假设。', '']
    if (ROOT/'docs/阶段4.6视觉核验.md').exists():
        lines += ['主代理已实际打开图表核验，详见[视觉核验与失败案例](阶段4.6视觉核验.md)。图中展示seed666，案例按三种子均值选取，二者不可混淆。', '']
    (ROOT/'docs/阶段4.6实验结果.md').write_text('\n'.join(lines),encoding='utf-8')
    atomic_json(ROOT/'docs/phase46_result.json',dict(validation_status=summary['status'],formal_runs=len(registry['runs']),
        combined_runs=len(registry['runs'])+len(old['runs']),validation=aggregates,final_test_evaluated=bool(final),
        target_achieved=bool(final and final['target_achieved']),training_seconds=total))
    print('REPORT_WRITTEN',summary['status'],'FINAL_TEST',bool(final))


if __name__ == '__main__':
    main()
