# AccentRoute 实施计划 v1.2

> **执行方式：** 普通任务流程，按任务顺序逐个执行（每任务 TDD 五步 + commit），不依赖任何会话特定 skill。批准后第一步：把本计划另存为 `docs/superpowers/plans/2026-08-12-accentroute-implementation.md`。
> **v1.2 修订摘要（响应第二轮 review）：** ① source-label confounding 控制（混杂矩阵、多源测试集、按源分层报告、LOSO、confounded 标记）；② 三臂公平消融 A/B/C；③ 统计口径改为分层 bootstrap + seed 变异分开报告 + 严格措辞；④ 阶段化 schema；⑤ 范围收缩：核心 = CV + L2-ARCTIC + EdAcc + 小规模 YouTube；VCTK/SAA/全局说话人聚类/HF 发布/demo 后置。

**Goal:** 建成 8 类英语口音识别的多源数据管线 + 弱监督三臂消融（核心数字 C−B），whisper-small + LoRA 作验证模型，3 周 part-time 交付可信版本。

**Architecture:** 7 段管线（ingest → taxonomy → filter → dedup/split → weak-label → augment → emit），每段是「读 Parquet manifest → 变换 → 按阶段 schema 校验 → 写新 Parquet」的纯函数。模型侧冻结 whisper-small encoder + LoRA(r=16, q/v) + masked mean-pooling（有效帧数由 processor attention mask 推导）+ 线性头。

**Tech Stack:** Python 3.11 + uv、pandera + pyarrow、Silero VAD（pip 包）、faster-whisper（tiny, int8）、fastText LID、SpeechBrain ECAPA（`speechbrain/spkrec-ecapa-voxceleb`，经 `speechbrain.inference`）、`Qwen/Qwen2-Audio-7B-Instruct`（pin revision sha）、transformers + peft、typer、pytest、GitHub Actions。

**计算分工：** 数据管线 Mac 本地 CPU；GPU 侧（Qwen2-Audio 推理 ~17GB BF16、训练 3 臂×3 seeds=9 次 + LOSO 1 次、基线）走 **AICR rtx-batch**（p2026_0038_neu）。集群交互 = repo 内显式 `scripts/*.sbatch` + `scripts/watch_jobs.sh`（squeue/sacct 轮询、失败告警），不以任何 skill 为前提。yt-dlp 只在本地（音频不出本机）。

## Context

设计 spec 已批准后经两轮 review：第一轮 9 条（数据血缘/防泄漏/弱标签审计/统计设计）已固化为决策；第二轮指出两个实验有效性缺口——**模型可能学到数据源而非口音**（类别与源高度绑定时,即使 speaker-disjoint 也挡不住麦克风/环境/朗读材料 shortcut）与**消融对照不公平**（gold+weak 同时多了样本量/步数/音频域），以及 CI 口径偏乐观、范围过载。本 v1.2 全部纳入。repo 目前仅有 spec 一份，从零搭建。

## 已核实的外部事实（2026-08-12 web 调研）

1. **EdAcc**：CC-BY-SA；HF `edinburghcstr/edacc` 或 Edinburgh DataShare（10283/8983）。逐说话人含 `accent`（语言学家标准化）、`raw_accent`、`l1`（母语）→ 8 类映射可行；每类 speaker 数（尤其 Korean/Arabic）W1 实测。仅 dev/test，无 train，与「只作域外」一致。
2. **Qwen2-Audio**：`Qwen/Qwen2-Audio-7B-Instruct`（Apache-2.0，BF16 ~17GB）→ 必须集群 GPU。transformers 主线集成（`Qwen2AudioForConditionalGeneration` + `AutoProcessor` + `apply_chat_template`）。pin：`HfApi().model_info(...).sha` → `configs/weaklabel.yaml` → `from_pretrained(revision=sha)`。
3. **L2-ARCTIC**：CC BY-NC 4.0，TAMU 表单（自动回链，小时级）。Arabic/Mandarin/Korean/Spanish 各 4 speakers → **每类仅 4 金标说话人，全项目最大统计约束**。
4. **Common Voice**：2025-10 起独家走 Mozilla Data Collective；**主路径 = HF `common_voice_17_0`**（gated 点击即过，CC0，自由文本 `accents` 字段）；MDC 注册 W1 第 1 天启动作可选升级。accent 填充率 ingest 实测。
5. **API 现状**：Silero VAD 用 pip 包；ECAPA 用 `speechbrain.inference.classifiers.EncoderClassifier`（`speechbrain.pretrained` 已废弃）。

## 设计决策（两轮 review 合并，均已定案）

1. **三臂公平消融（头条数字 = C−B）**：
   - **A** gold-only（epoch 对齐：与 C 相同 epoch 数）；**B** gold-only oversampled（**optimizer 步数与 C 完全相同**）；**C** gold + accepted weak。
   - 三臂共享 `configs/train_common.yaml`：同 augmentation、同 class-balanced sampler、同 LR schedule、**固定步数预算替代 early stopping**、同一 checkpoint 选择规则（val macro-F1）。C−B ≈ 弱标数据本身的价值；A−B 隔离训练预算效应。
2. **统计口径（措辞纪律）**：每臂 3 seeds（17/42/1337）。主报告两个量，**分开呈现、不合并**：
   - **按类分层的 test-speaker bootstrap 95% CI**：以真实类别为层，各层内有放回重采样 speaker cluster（防稀有类在采样中消失），对 seed 平均后的 Δmacro-F1 出 percentile CI；
   - **seed 变异**：逐 seed 配对 Δ 的 mean±std。
   - 措辞规则：只允许写 "test-speaker bootstrap CI excludes zero"，**不得泛称 statistically significant**（除非另做 seed×speaker hierarchical bootstrap，本期不做）。
3. **Source-confounding 控制**：
   - G1 前产出 **source × accent 矩阵**（n_speakers、hours、n_clips）；训练时长 >90% 来自单一源的类标记 `confounded=True`，写入 datasheet，对这些类的结论必须限定措辞（不得声称纯口音能力）；
   - 测试集构成：每类尽量 ≥2 个源；结果表**按源分层报告**每类 F1；
   - **LOSO 诊断**（W3，单 seed）：L2 四类训练剔除 L2-ARCTIC（只用 CV 自报 L2），在 L2-ARCTIC holdout speakers 上测——量化跨源泛化 vs source shortcut。EdAcc 本身是全体类的跨源域外测。
4. **弱标注循环论证**：pin Qwen revision sha + prompt 版本化（8 选 1 + `unsure`，k=3 自洽投票，k 可配降 1）；接受规则见共识规范；**审计覆盖三个池**：accepted 25 条/类 + rejected/review 池按 reject_reason 分层抽 50 条（理解筛选器选择偏差与 false-reject），datasheet 报每类 precision（Wilson 区间，n=25 噪声明示）；kill 规则：accepted 池某类 precision < 0.80 → 该类弱标签整体剔除。弱标签只进 train（schema 机器强制）。Qwen 零样本只是次要对照；头条 C−B 双方用同一金标测试集。
5. **YouTube 证据等级**：`lists/youtube_v1.csv` 人工策展（**小规模严策展：目标 300–600 clips**），标 evidence_level（E1 自述/出生地；E2 频道地区+内容线索；E3 仅模型）。只接受 E1/E2 且 Qwen 多数票与先验一致；只进 train。
6. **去重（v1.2 范围化）**：split 键 = `speaker_key`（缺省 `f"{source}:{speaker_id_raw}"`）。核心范围内做三件事：① YouTube 集内部 ECAPA 去重（跨视频/频道同一访谈对象；ANN 候选边 + union-find，架构从一开始就可扩展）；② CV 内部近重复检测（转写 Jaccard ≥0.8 且 |Δ时长|≤0.5s）；③ 阈值校准用 L2-ARCTIC/CV 已知同人对作正例、**跨源配对（CV×YouTube）作 hard negatives**。全局跨源说话人聚类进 backlog；残余风险（同人跨 CV/YouTube 未检出）写入 datasheet。
7. **阶段化 schema**：`RawManifestSchema` →（filter/taxonomy 后）`QCManifestSchema` →（dedup/split/weaklabel 后）`SplitManifestSchema`，继承 + `validate_manifest(df, stage)` 按阶段校验；不变量挂对应阶段。命名纪律：`snr_proxy_db`（单通道代理量非真实 SNR）、`consensus_score`（多数票×证据权重，非校准置信度）、基线称 **frozen ECAPA embedding probe**、EdAcc 排除后指标称 **supported-class macro-F1**（并给域内模型在同一支持类子集上的对照值，不与完整 8 类横比）。
8. **Masked mean-pooling**：有效帧数**从 WhisperFeatureExtractor 的 attention mask 推导**（`return_attention_mask=True`，mel mask → conv2 下采样），手算公式只作单测参考实现交叉验证。策略：>30s 取中心窗；<5s 已被 filter 丢弃；单窗不滑窗；归一化沿用 extractor 默认。
9. **止损阶梯（8 类锁死）**：① W1 滞后 → EdAcc 域外测缩为 supported-class 报告即可，不加源；② W2 滞后 → k_votes 3→1、YouTube 清单缩到 ~300 条（**弱标注绝不推迟——它是核心论点**；缩规模不砍环节）；③ 仍不够 → 砍 LOSO 与 proposal。任何情况不改任务定义、不砍三臂消融。
10. **排期门禁**：HF 发布、路由 demo 全部后置出 3 周（backlog）；datasheet 是核心交付。G1/G2/G3 见任务表。

## Global Constraints

- 8 类锁定：en-US、en-GB、en-AU、en-IN、L1-Mandarin、L1-Spanish、L1-Korean、L1-Arabic；映射不进丢弃并计数。
- 核心数据源 = Common Voice + L2-ARCTIC + EdAcc（域外）+ YouTube（弱标，只进 train）。VCTK/SAA 为 backlog。`label_source="self_report"` 允许进 eval，`weak` 禁止。
- 底座 `openai/whisper-small` encoder 冻结；LoRA r=16 只打 `q_proj`/`v_proj`。
- repo 不再分发 L2-ARCTIC / YouTube 音频；YouTube 只发布 URL+时间戳+标签清单；`data/` gitignored。
- **CI 绝不下载模型**：模型调用全部注入/monkeypatch，CI 只跑合成 fixture。
- 每任务 TDD：写失败测试 → 确认失败 → 最小实现 → 通过 → commit；CI 绿才算完成。
- 措辞纪律：不称 SOTA；不泛称 statistically significant（见决策 2）；confounded 类结论限定；weak supervision / data-centric 叙事。

## 项目结构

```
accent-route/
├── pyproject.toml                # py3.11+uv
├── .github/workflows/ci.yml     # ruff + pytest（合成 fixture）
├── configs/
│   ├── taxonomy_v1.yaml
│   ├── sources/{common_voice,l2_arctic,edacc,youtube}.yaml
│   ├── filter.yaml               # min_dur=5.0 max_dur=30.0 min_snr_proxy_db=10 min_vad_ratio=0.5 min_lang_prob=0.8
│   ├── dedup.yaml
│   ├── weaklabel.yaml            # model_id、PIN sha、prompt 文件+sha256、k_votes=3、kill_precision=0.80
│   ├── train_common.yaml         # 三臂共享：steps、sampler、augment、ckpt 规则、seeds [17,42,1337]
│   └── arms/{a_gold,b_gold_oversampled,c_gold_weak,loso_l2}.yaml
├── prompts/qwen2audio_accent_v1.txt
├── lists/youtube_v1.csv          # url,start_s,end_s,prior_label,evidence_level,evidence_note
├── data/                         # gitignored
├── scripts/                      # weaklabel_qwen.sbatch train.sbatch watch_jobs.sh run_experiments.py
├── src/accentroute/
│   ├── schema.py  taxonomy.py  audio.py  cli.py
│   ├── ingest/{base,common_voice,l2_arctic,edacc,youtube}.py
│   ├── filter.py  dedup.py  split.py
│   ├── weaklabel/{qwen,consensus,audit}.py
│   ├── augment.py  emit.py
│   ├── model/{pooling,whisper_lora}.py  train.py
│   ├── eval/{metrics,bootstrap,baselines,tables}.py
│   └── reports/{coverage_confounding,dataset_stats}.py
├── tests/
└── docs/（datasheet.md、clipto-proposal.md）
```

## 核心算法规范（normative）

### (a) 阶段化 schema（T1）

```python
# src/accentroute/schema.py
import pandera as pa
from pandera.typing import Series

ACCENTS = ["en-US", "en-GB", "en-AU", "en-IN",
           "L1-Mandarin", "L1-Spanish", "L1-Korean", "L1-Arabic"]
SOURCES = ["common_voice", "l2_arctic", "edacc", "youtube"]   # vctk/saa 进 backlog 时再扩
SPLITS = ["train", "val", "test", "ood_test", "unassigned"]

class RawManifestSchema(pa.DataFrameModel):        # ingest 产出
    clip_id: Series[str] = pa.Field(unique=True)
    source: Series[str] = pa.Field(isin=SOURCES)
    source_uri: Series[str];  orig_file: Series[str]
    offset_start_s: Series[float] = pa.Field(ge=0)
    offset_end_s: Series[float] = pa.Field(gt=0)
    sample_rate_orig: Series[int] = pa.Field(gt=0)
    duration_s: Series[float] = pa.Field(gt=0)
    license: Series[str];  speaker_id_raw: Series[str]
    accent_raw: Series[str] = pa.Field(nullable=True)

class QCManifestSchema(RawManifestSchema):         # taxonomy + filter 后
    accent_label: Series[str] = pa.Field(isin=ACCENTS, nullable=True)
    taxonomy_version: Series[str]
    snr_proxy_db: Series[float] = pa.Field(nullable=True)      # 单通道代理量，非真实 SNR
    vad_speech_ratio: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    lang_prob: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    transcript: Series[str] = pa.Field(nullable=True)
    status: Series[str] = pa.Field(isin=["pending", "accepted", "rejected", "review"])
    reject_reason: Series[str] = pa.Field(nullable=True)

    @pa.dataframe_check(name="rejected_has_reason")
    def _c1(cls, df): return ~((df["status"] == "rejected") & df["reject_reason"].isna())

class SplitManifestSchema(QCManifestSchema):       # dedup + split + weaklabel 后
    speaker_key: Series[str]                       # 缺省 f"{source}:{speaker_id_raw}"，去重合并后更新
    split: Series[str] = pa.Field(isin=SPLITS)
    label_source: Series[str] = pa.Field(isin=["gold", "self_report", "weak"])
    consensus_score: Series[float] = pa.Field(ge=0, le=1, nullable=True)  # 非校准置信度
    evidence_level: Series[str] = pa.Field(isin=["E1", "E2", "E3"], nullable=True)

    @pa.dataframe_check(name="weak_never_in_eval")
    def _c2(cls, df): return ~((df["label_source"] == "weak")
                               & df["split"].isin(["val", "test", "ood_test"]))

    @pa.dataframe_check(name="youtube_requires_evidence")
    def _c3(cls, df): return ~((df["source"] == "youtube") & (df["status"] == "accepted")
                               & ~df["evidence_level"].isin(["E1", "E2"]))

STAGE_SCHEMAS = {"raw": RawManifestSchema, "qc": QCManifestSchema, "split": SplitManifestSchema}
def validate_manifest(df, stage: str):
    return STAGE_SCHEMAS[stage].validate(df, lazy=True)   # lazy → 一次报全部违规
```

### (b) Pooling：attention-mask 推导有效帧数（T9）

```python
# src/accentroute/model/pooling.py
import torch
N_ENC_MAX = 1500  # whisper 30s 窗

def valid_encoder_frames(mel_attention_mask: torch.Tensor) -> torch.Tensor:
    """mel_attention_mask: [B, 3000]，来自 WhisperFeatureExtractor(..., return_attention_mask=True)。
    conv1 保长，conv2 k=3,s=2,p=1 → L_out = (L_in - 1)//2 + 1。"""
    n_mel = mel_attention_mask.sum(-1)
    return ((n_mel - 1) // 2 + 1).clamp(max=N_ENC_MAX)

def num_valid_encoder_frames_ref(n_samples: int, hop: int = 160) -> int:
    """仅作单测参考实现：与 mask 推导路径交叉验证，不用于生产。"""
    return min(((n_samples // hop) - 1) // 2 + 1, N_ENC_MAX)

def masked_mean(hidden: torch.Tensor, n_valid: torch.Tensor) -> torch.Tensor:
    """hidden: [B,1500,D]; n_valid: [B] -> [B,D]"""
    idx = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
    mask = (idx < n_valid[:, None]).to(hidden.dtype)
    return (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
```

单测锚点：真实 extractor 上 30s→1500、5s→250 且两条路径一致；`masked_mean([x; 垃圾padding])` == `x` 的 plain mean。

### (c) 去重：ANN 候选边 + union-find（T7，架构可扩展）

```python
# src/accentroute/dedup.py
def candidate_edges(embs, k: int = 20, sim_threshold: float = 0.45) -> list[tuple[int, int]]:
    """embs: [n,192] L2 归一化。top-k 余弦近邻生成候选边，避免 O(n²) 全矩阵。
    核心范围 n 只有数百（YouTube 集内），用 sklearn NearestNeighbors 精确解；
    签名不变，backlog 全局聚类时换 faiss。"""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(embs)), metric="cosine").fit(embs)
    dist, idx = nn.kneighbors(embs)
    return [(i, int(j)) for i in range(len(embs))
            for j, d in zip(idx[i][1:], dist[i][1:]) if 1.0 - d >= sim_threshold]

def union_find_clusters(n: int, edges: list[tuple[int, int]]) -> list[int]:
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra
    return [find(i) for i in range(n)]
```

阈值校准（T7 内脚本）：正例 = L2-ARCTIC/CV 已知同人 clip 对；**负例必须含跨源 hard negatives（CV×YouTube 随机配对）**；输出类内/类间相似度直方图，按误合并率 ≤1e-3 定阈值，0.45 只是校准前缺省。近重复：embedding 余弦 ≥0.92 且 |Δ时长|≤0.5s 候选 → 转写 Jaccard ≥0.8 确认 → 留一拒余（`reject_reason="near_duplicate"`）。

### (d) 分层 bootstrap + seed 变异（T12）

```python
# src/accentroute/eval/bootstrap.py
@dataclass(frozen=True)
class AblationStats:
    delta_mean: float          # seed 平均后 Δmacro-F1 的 bootstrap 均值
    ci_low: float; ci_high: float; n_boot: int
    ci_excludes_zero: bool     # 报告措辞只允许引用这个字段
    seed_deltas: tuple[float, ...]   # 逐 seed 配对 Δ
    seed_delta_std: float

def stratified_cluster_bootstrap(y_true, preds_a, preds_b, speaker_keys, classes,
                                 n_boot: int = 10_000, seed: int = 0) -> AblationStats:
    """按真实类别分层：每类内部有放回重采样该类的 speaker_key（等量），
    防止稀有类在某次采样中消失。preds_a/b: [n_seeds, n]。"""
    rng = np.random.default_rng(seed)
    idx_of = {k: np.flatnonzero(speaker_keys == k) for k in np.unique(speaker_keys)}
    strata = {c: np.unique(speaker_keys[classes == c]) for c in np.unique(classes)}
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        chosen = np.concatenate([rng.choice(ks, size=len(ks)) for ks in strata.values()])
        idx = np.concatenate([idx_of[k] for k in chosen])
        fa = np.mean([macro_f1(y_true[idx], p[idx]) for p in preds_a])
        fb = np.mean([macro_f1(y_true[idx], p[idx]) for p in preds_b])
        deltas[b] = fa - fb
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    seed_d = tuple(float(macro_f1(y_true, a) - macro_f1(y_true, b))
                   for a, b in zip(preds_a, preds_b))
    return AblationStats(float(deltas.mean()), float(lo), float(hi), n_boot,
                         bool(lo > 0 or hi < 0), seed_d, float(np.std(seed_d)))
```

单测：同预测 → CI 跨 0；注入 +5 点偏移 → 检出；每次重采样中每类至少 1 个 cluster；定 seed 逐字节复现。报告模板固定："Δmacro-F1 = X, test-speaker stratified bootstrap 95% CI [l, h] (excludes zero: yes/no); per-seed Δ = [...], std = Y" —— 不出现 "statistically significant" 字样。

### (e) 弱标签共识（T13）

```python
# src/accentroute/weaklabel/consensus.py
@dataclass(frozen=True)
class WeakLabelDecision:
    accepted: bool; label: str | None; consensus_score: float; reason: str

EVIDENCE_WEIGHT = {"E1": 1.0, "E2": 0.85}

def consensus(evidence_level: str, prior_label: str,
              qwen_votes: list[str]) -> WeakLabelDecision:
    """consensus_score = 多数票比例 × 证据权重；工程排序分，非校准置信度。"""
    if evidence_level not in EVIDENCE_WEIGHT:
        return WeakLabelDecision(False, None, 0.0, "evidence_E3")
    top, n = Counter(qwen_votes).most_common(1)[0]
    if top != prior_label or n < 2:
        return WeakLabelDecision(False, None, 0.0, "qwen_disagrees")
    score = (n / len(qwen_votes)) * EVIDENCE_WEIGHT[evidence_level]
    return WeakLabelDecision(True, prior_label, score, "consensus")
```

审计（`weaklabel/audit.py`）：`draw_audit_sample(df, accepted_per_class=25, reject_pool_n=50, seed=0)` —— accepted 池按类分层 + rejected/review 池按 reject_reason 分层，合并出一份盲听 CSV（不含标签列）；`audit_report(annotated) -> AuditReport`（每类 precision + Wilson 区间 + reject 池 false-reject 率）。kill 规则：accepted 池某类 precision < 0.80 → 该类弱标签整体剔除，datasheet 记录。

### (f) 三臂训练预算协议（T10/T14）

```yaml
# configs/train_common.yaml —— 三臂与 LOSO 共享，任何字段不得被臂配置覆盖
epochs_c: 15                 # 以 C 臂数据量为基准
batch_size: 32
sampler: class_balanced      # 同一实现、同一 seed 流
augment: {speed: [0.9, 1.1], musan: true, rir: true}
lr_schedule: cosine, lr: 1.0e-4, warmup_ratio: 0.05
ckpt_select: best_val_macro_f1   # 固定步数预算，无 early stopping
seeds: [17, 42, 1337]
```

- **C**：gold+weak，训 `epochs_c` → 得 `S_C` 总步数（写入 run 元数据）。
- **B**：gold-only，**训满同样的 `S_C` 步**（数据循环即 oversample）。
- **A**：gold-only，epoch 对齐（`epochs_c` 个 epoch，步数 < S_C）。
- 头条 = C−B（弱标数据本身价值）；A−B 隔离预算效应，一并报告。

## 任务分解（16 任务 · 3 周 · 3 门禁 · backlog 另列）

每任务 TDD 五步：① 写失败测试 ② 确认失败 ③ 最小实现 ④ 通过 ⑤ commit。

### Week 1 — 金标管线（CV + L2-ARCTIC + EdAcc）→ Gate G1

- [ ] **T1 脚手架 + 阶段化 schema**：`pyproject.toml`、`ci.yml`、`schema.py`（规范 (a)）、`tests/test_schema.py`。测试：raw 合法行过 raw 校验但（缺列）不过 split 校验；weak-in-test 拒；rejected 无 reason 拒；CI 绿。**同时**：9+10 条决策以「修订 v1.2」回写 spec；本计划落盘 `docs/superpowers/plans/`。依赖：无。
- [ ] **T2 Taxonomy**：`taxonomy.py` + `configs/taxonomy_v1.yaml`。`load_taxonomy(path)->Taxonomy`；`Taxonomy.map(raw)->str|None`；`.version`。测试："united states english"→en-US；"scottish"→None 且计数；大小写/空白鲁棒。依赖 T1。
- [ ] **T3 音频工具 + ingest 基类**：`audio.py`、`ingest/base.py`。`to_wav16k_mono(src,dst)->AudioMeta`；`SourceIngestor.iter_records()`；`run_ingest(ing,out)->Path`（产出过 raw 校验）。测试：44.1k 立体声 fixture → 16k mono；切片偏移精确。依赖 T1。
- [ ] **T4 三个源适配器**：`ingest/{common_voice,l2_arctic,edacc}.py` + `configs/sources/*.yaml`。测试：微型 fixture → 行/许可串正确、CV accent_raw 填充率统计输出。**W1 第 1 天并行启动**：MDC 注册、L2-ARCTIC 表单、HF CV17 gate。依赖 T2、T3。
- [ ] **T5 覆盖与混杂报告（G1 输入）**：`reports/coverage_confounding.py`。`source_accent_matrix(df)->DataFrame`（source×accent：n_speakers/n_clips/hours）；`flag_confounded(matrix, dominance=0.9)->DataFrame`（类级 confounded 标记）；`edacc_class_coverage(df)->DataFrame`（<5 speakers 的类标记 excluded → supported-class 集合）。测试：fixture 上矩阵/标记/排除正确。依赖 T2、T4。
- [ ] **T6 Filter**：`filter.py` + `configs/filter.yaml`。`compute_vad_ratio`、`estimate_snr_proxy_db`、`transcribe_tiny`、`apply_filters(df,cfg)->df`（产出过 qc 校验）。测试：静音拒 `low_vad`；已知信噪比合成音误差 <1dB；模型全 monkeypatch。依赖 T3。
- [ ] **T7 范围化去重**：`dedup.py`（规范 (c)）+ 校准脚本 + `configs/dedup.yaml`。`assign_speaker_keys(df)->df`（缺省 source:speaker_id_raw）；`dedup_youtube_speakers(df, embs)->df`（union-find 合并 speaker_key）；`find_near_duplicates(df)->df`（CV 内转写+时长）。测试：合成 embedding 同人合并/异人不合并；近重复拒；校准脚本含跨源负例并输出直方图数据。依赖 T6。
- [ ] **T8 Speaker-disjoint 切分 + 多源测试集**：`split.py` + `configs/split.yaml`。`assign_splits(df,ratios=(0.8,0.1,0.1),seed=17)->df`（产出过 split 校验）；`write_speaker_report(df,out)`。测试：**无 speaker_key 跨 split**；edacc 只落 ood_test；类分层容差内；seed 确定性；**每类测试集来源数 ≥2 或触发 confounded 记录**（L2 类 = L2-ARCTIC holdout + CV 自报；结果表按 label_source 与 source 分列）。依赖 T7。

**Gate G1（W1 末）**：金标 manifest 每类 ≥200 clips；母语类 ≥20 speakers、L2 类 ≥8 speakers；泄漏检查通过；**混杂矩阵 + confounded 标记评审**；EdAcc supported-class 集合确定。失败 → 止损阶梯 ①。

### Week 2 — 模型 + 评测设施 + 弱标注 → Gate G2

- [ ] **T9 Pooling + 模型**：`model/pooling.py`（规范 (b)）、`model/whisper_lora.py`。`WhisperEncoderClassifier.forward(input_features, n_valid)->Tensor[B,8]`；`build_model(cfg)`（peft LoRA r=16 on q/v，encoder 冻结）。测试：两条帧数路径一致；masked mean 精确忽略 padding；只有 LoRA+head requires_grad。依赖 T1。
- [ ] **T10 训练循环 + 预算协议**：`train.py` + `configs/train_common.yaml` + `configs/arms/*.yaml`（规范 (f)）。`train(cfg:TrainConfig)->TrainResult(ckpt_path,val_macro_f1,seed,total_steps)`。测试：16-clip 合成 batch 可过拟合；**B 臂步数 == C 臂步数断言**；三臂共享字段不可覆盖（配置加载器拒绝）；metrics json 落盘。依赖 T8、T9。
- [ ] **T11 指标 + 基线**：`eval/{metrics,baselines}.py`。`macro_f1`、`confusion`、`majority_baseline`、`ecapa_embedding_probe`（冻结 embedding + 逻辑回归；命名即措辞）、`qwen_zero_shot(df,cfg)`（与 T13 共用 pin 的 revision+prompt）。测试：对照 sklearn；多数类精确；qwen 输出解析含 `unsure`。依赖 T8。
- [ ] **T12 分层 bootstrap**：`eval/bootstrap.py`（规范 (d)）。测试见规范。依赖 T11。
- [ ] **T13 弱标注管线**：`ingest/youtube.py`、`lists/youtube_v1.csv`（300–600 clips 严策展）、`weaklabel/{qwen,consensus,audit}.py`、`prompts/qwen2audio_accent_v1.txt`、`configs/weaklabel.yaml`、`scripts/weaklabel_qwen.sbatch` + `scripts/watch_jobs.sh`（squeue/sacct 轮询）。`qwen_label_batch(manifest,cfg)->Path`（GPU）；`consensus`（规范 (e)）；`draw_audit_sample`；`audit_report`。测试：共识规则表驱动（E3→拒；不一致→review；E1+3/3→1.0；E2+2/3→0.567）；审计抽样含 reject 池分层；schema 挡 weak 出 eval。依赖 T6、T8。

**Gate G2（W2 末）**：A 臂 3 seeds 可复现、val macro-F1 超多数类基线且 CI 不跨 0（对多数类）；弱标注端到端跑通有接受率。失败 → 止损阶梯 ②（k_votes 降 1、清单缩规模；弱标注不推迟）。

### Week 3 — 三臂实验 + 分层评测 → Gate G3 → datasheet

- [ ] **T14 增强 + 三个数据集变体**：`augment.py`、`emit.py`。`augment_train(df,wav_dir,cfg)->df`（只对 train 行）；`emit_dataset(df, arm: Literal["a_gold","b_gold_oversampled","c_gold_weak"], out_dir)->DatasetStats`（B 的 oversample 由训练端步数控制，emit 只标记 arm 与数据内容）。测试：增强行只在 train；A/B 变体零 weak 行；统计对账。依赖 T8、T13。
- [ ] **T15 实验矩阵 + 分层评测**：`scripts/run_experiments.py`、`scripts/train.sbatch`、`eval/tables.py`。矩阵 = 3 臂×3 seeds（9 次）+ LOSO-L2 单 seed 诊断（10 次训练，AICR rtx，watch_jobs.sh 盯）。`make_results_tables(runs_dir)->Path`：① 三臂消融表（C−B 头条 + A−B 预算效应，bootstrap CI + seed std 分列）；② **按源分层的每类 F1**；③ EdAcc supported-class macro-F1（附域内模型在同一支持类子集的对照值）；④ 混淆矩阵重点 en-IN vs L2、en-GB vs en-AU；⑤ LOSO 表。测试：从 fixture run 目录生成全部表；CI 列与 T12 输出一致；措辞模板校验（禁 "statistically significant"）。依赖 T10–T12、T14。
- [ ] **T16 Datasheet + proposal + 简历 bullets（Gate G3：T15 数字落定后启动）**：`docs/datasheet.md`、`docs/clipto-proposal.md`。datasheet 必含：许可表、丢弃统计、弱标注接受率、审计三池结果（accepted precision + Wilson 区间、reject 池 false-reject）、kill 触发记录、taxonomy 版本、**混杂矩阵与 confounded 类清单**、去重残余风险声明、L2 测试集构成与宽 CI 说明。proposal（2–3 页）：端侧蒸馏/量化路线、ASR 前置路由集成点、MCP 工具形态。简历 3 条 bullet（canonical 三段式，禁 SOTA）。依赖 T15。

### Backlog（3 周外，按价值排序）

1. HF 发布（adapter + 模型卡，push 脚本带 dry-run）；2. VCTK/SAA 适配器（en-GB/en-AU 补强 + 混杂矩阵更新）；3. 全局跨源说话人聚类（faiss 替换 candidate_edges 即可）；4. 口音感知路由 demo（`demo/route_asr.py`，WER 对比）；5. seed×speaker hierarchical bootstrap。

## 风险与预案（按杀伤力排序）

1. **Source-label confounding（最大有效性风险）**：混杂矩阵（T5）+ 多源测试集（T8）+ 按源分层报告与 LOSO（T15）+ confounded 限定措辞（T16）四道防线；若某 L2 类测试集只有 L2-ARCTIC 一个源，如实标 confounded 而不是硬凑。
2. **L2 测试说话人稀缺**：L2-ARCTIC 每类 4 speakers；CV 自报补测试集、按 label_source 分列；CI 宽如实报。
3. **W2/W3 超载**：训练从 6 次涨到 10 次，但范围收缩（源 3 个、无 HF/demo）对冲；预案 = k_votes 3→1、YouTube 缩到 300 条、砍 LOSO 与 proposal（决策 9 阶梯）。
4. **CV 渠道变化**：主路径 HF v17.0 已定，MDC 失败无影响。
5. **EdAcc Korean/Arabic 覆盖不足**：T5 在 W1 出 supported-class 集合，W3 无惊吓。
6. **CI 下载模型**：全部注入/monkeypatch（Global Constraints）。

## Verification（端到端验证）

1. **每任务**：`pytest tests/test_<module>.py -v` 红→绿；完成即 commit。
2. **管线冒烟**：`accentroute` CLI 在合成 fixture 上全链路（ingest→emit），各阶段 manifest 过对应 `validate_manifest(df, stage)`。
3. **泄漏审计（G1 硬门）**：脚本断言 train/val/test 的 `speaker_key` 两两交集为空；EdAcc 全在 ood_test；weak 行 100% 在 train；每类测试来源数落盘。
4. **预算对齐审计（三臂公平性硬门）**：训练日志断言 B/C 总步数相等、三臂 augment/sampler/ckpt 规则哈希一致。
5. **可复现（G2 硬门）**：同 config+seed 两次训练 val macro-F1 差 <0.5 点。
6. **头条数字（G3 硬门）**：`stratified_cluster_bootstrap` 输出 C−B 的 CI 与 seed std，按固定措辞模板写入结果表。
7. **CI**：GitHub Actions ruff+pytest 全绿（无网络模型下载）。
