# AccentRoute

8 类英语口音识别的多源数据管线与弱监督消融实验。项目重心是**数据管线**(多源整合、
LLM 弱标注、质量控制与评测设计),模型侧刻意保持标准化(冻结 whisper-small encoder
+ LoRA)作为验证手段。

**状态:** 管线代码与评测设施已完成并有单元测试;数据下载与实验尚未开始(见下方 Gates)。

## 类别体系(锁定 8 类)

母语变体 `en-US` `en-GB` `en-AU` `en-IN`;L2 口音(按说话人母语)`L1-Mandarin`
`L1-Spanish` `L1-Korean` `L1-Arabic`。映射不进 8 类的样本丢弃并计数
(`configs/taxonomy_v1.yaml`,版本化白名单)。

## 头条数字:三臂公平消融

| 臂 | 数据 | 训练预算 |
| --- | --- | --- |
| A `a_gold` | gold + 自报 | epoch 对齐 |
| B `b_gold_oversampled` | gold + 自报 | **optimizer 步数与 C 完全相同** |
| C `c_gold_weak` | gold + 自报 + 已接受弱标 | epoch 对齐(定出 S_C) |

头条 = **C − B**(弱标数据本身的价值);A − B 单独报告以隔离训练预算效应。
三臂共享 `configs/train_common.yaml` 的采样器、增强、LR schedule、checkpoint 规则,
臂配置若试图覆盖任一共享字段,配置加载器直接报错。

## 统计口径与措辞纪律

每臂 3 seeds(17/42/1337)。报告两个**分开呈现、不合并**的量:

1. 按类分层的 **test-speaker bootstrap 95% CI**(层内等量有放回重采样 speaker,
   稀有类不会在采样中消失);
2. **seed 变异**:逐 seed 配对 Δ 的明细与 std。

CI 未覆盖训练随机性,所以只写 `test-speaker bootstrap CI excludes zero`,
**不泛称 statistically significant**。这条纪律由代码强制:`AblationStats` 没有
`significant` 字段,`eval.tables.check_wording` 会拒收含禁用措辞的报告文本。

## 已知有效性限制(不藏)

- **Source-label confounding**:类别与数据源高度绑定时,模型可能学到麦克风/录音
  环境而非口音。四道防线:`accentroute report` 的 source × accent 混杂矩阵与
  `confounded` 标记、测试集每类尽量 ≥2 源、按源分层的每类 F1、LOSO-L2 诊断。
  单源主导的类在 datasheet 明确标注,结论措辞必须限定。
- **L2 说话人稀缺**:L2-ARCTIC 每类仅 4 个金标说话人,speaker-disjoint 后测试集
  L2 类 CI 会很宽,如实报告。
- **弱标注双重角色**:Qwen2-Audio 既是弱标签来源也是零样本基线。缓解:pin
  revision sha 与 prompt sha256、三池人工盲审(含 reject 池以看筛选偏差)、
  每类 precision < 0.80 触发 kill 规则。头条 C−B 双方用同一金标测试集。
- **去重范围**:核心范围做 YouTube 集内说话人合并与 CV 内近重复;全局跨源说话人
  聚类在 backlog,残余风险写入 datasheet。

## 管线

```
ingest → taxonomy → filter → dedup/split → weak-label → augment → emit
```

每段是「读 Parquet manifest → 变换 → 按阶段 schema 校验 → 写新 Parquet」的纯函数。
三条不变量由 schema 机器强制(`src/accentroute/schema.py`):

- rejected 行必有 `reject_reason`
- **`label_source == "weak"` 绝不出现在 val/test/ood_test**
- YouTube 的 accepted 行必须是 E1/E2 证据等级

```bash
uv sync --group dev                       # 核心依赖
uv sync --group dev --extra audio --extra ml   # 加音频栈与训练栈

uv run accentroute ingest common-voice
uv run accentroute filter data/manifests/raw_common_voice.parquet data/manifests/qc.parquet
uv run accentroute split data/manifests/qc.parquet data/manifests/split.parquet
uv run accentroute report data/manifests/split.parquet     # G1 混杂矩阵
uv run accentroute emit data/manifests/split.parquet c_gold_weak data/datasets/c_gold_weak
```

GPU 阶段(Qwen2-Audio 弱标注、训练)走 AICR rtx-batch:

```bash
sbatch scripts/weaklabel_qwen.sbatch <manifest.parquet> <out.parquet>
sbatch scripts/train.sbatch c_gold_weak 17
sbatch scripts/train.sbatch b_gold_oversampled 17 <S_C>   # B 必须对齐 C 的步数
bash scripts/watch_jobs.sh <jobid>...
```

## 数据源与许可

| 源 | 用途 | 许可 |
| --- | --- | --- |
| Common Voice(HF `common_voice_17_0`)| 主训练集,自报口音 | CC0 |
| L2-ARCTIC | L2 四类金标 | CC BY-NC 4.0,TAMU 表单申请,**repo 不分发音频** |
| EdAcc | **只作域外测试** | CC-BY-SA |
| YouTube 访谈 | 弱标扩充,只进 train | **repo 只发 URL+时间戳+标签清单** |

VCTK 与 Speech Accent Archive 在 backlog。`data/` 已 gitignore。

## Gates

- **G1**:金标 manifest 每类 ≥200 clips;母语类 ≥20 speakers、L2 类 ≥8 speakers;
  泄漏检查通过;混杂矩阵评审;EdAcc supported-class 集合确定。
- **G2**:A 臂 3 seeds 可复现且超多数类基线;弱标注端到端跑通并有接受率。
- **G3**:三臂消融 + 域外数字落定 —— 之后才动 datasheet 定稿与发布类工作。

## 开发

```bash
uv run pytest -q       # 162 tests
uv run ruff check .
```

CI 只装核心依赖,**绝不下载模型**:所有模型调用都经参数注入,单测用合成 fixture;
需要 torch 的测试自动跳过。

设计 spec 与实施计划见 `docs/superpowers/`。
