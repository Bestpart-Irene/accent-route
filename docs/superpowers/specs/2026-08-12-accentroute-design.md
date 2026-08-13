# AccentRoute — 面向转写产品的英语口音识别数据管线(设计 spec)

日期:2026-08-12 · 状态:v1.2(两轮 review 修订已并入,实施计划已批准,见 `../plans/2026-08-12-accentroute-implementation.md`)
用途:① mle 简历 data-pipeline 格位项目(建成后替换 LLM Serving);② Clipto 投递敲门砖
落地位置:`/Users/xxiellan/accent-route`(项目根目录,本 spec 随 repo;GitHub: Bestpart-Irene)

## 1. 目标

输入 5–30 秒英语语音片段,输出 8 类口音标签。项目重心是**数据管线**(多源整合、LLM 弱标注、质量控制),模型侧刻意保持标准化(Whisper + LoRA),对应岗位族定义的 data pipeline + evaluation 轴。

成功标准(全部可量化):
1. speaker-disjoint 测试集 macro-F1 显著高于 ECAPA-TDNN 基线与 Qwen2-Audio 零样本;
2. **头条消融数字**:金标+弱标训练 vs 仅金标训练的 macro-F1 提升(证明数据管线与 LLM 标注的价值);
3. EdAcc 域外测试(自发对话)上性能不崩(报告域内外差距)。

## 2. 口音类别体系(8 类,锁定)

- 母语变体:`en-US`、`en-GB`、`en-AU`、`en-IN`
- L2 口音(按说话人母语):`L1-Mandarin`、`L1-Spanish`、`L1-Korean`、`L1-Arabic`

映射不进 8 类的样本丢弃(如 Scottish、Filipino 等,记录丢弃统计)。en-IN 定义为印度英语(不区分母语/L2)。

## 3. 数据源与许可

| 源 | 用途 | 许可要点 |
| --- | --- | --- |
| Common Voice (en) | 主训练集(自报口音标签,自由文本→映射表归类) | CC0,可再分发 |
| L2-ARCTIC | L2 四类金标 | 研究许可,repo 不再分发音频,给下载脚本 |
| VCTK | en-GB 等母语类补充 | CC BY 4.0 |
| Speech Accent Archive | 少量补充/审计集 | CC BY-NC-SA,仅评测用 |
| EdAcc | **只作域外测试集,不训练** | 实施第 1 周核对许可证,不符合则换 CommonVoice 自发子集 |
| YouTube 访谈/播客 | 弱标注扩充集 | **不再分发音频**,repo 只发布 URL+时间戳+标签清单;yt-dlp 本地抓取 |

## 4. 数据管线(7 段,每段独立模块 + 单元测试)

1. **ingest**:各源 → 16 kHz mono WAV 分片 + 统一元数据 schema(Parquet:source, speaker_id, accent_raw, duration, split)
2. **taxonomy**:accent_raw → 8 类映射表(YAML 版本化;CommonVoice 自由文本白名单映射,映射外丢弃并计数)
3. **filter**:Silero VAD 去静音、时长 ≥5s、SNR 下限、Whisper-tiny 转写 + fastText LID 确认是英语
4. **dedup & split**:说话人级去重;**speaker-disjoint** train/val/test 切分(同一说话人绝不跨 split);切分脚本输出可复核的 speaker 清单
5. **weak-label**(上网大模型自动标注):yt-dlp 抓取 → 频道地区元数据先验 + Qwen2-Audio 零样本标注 → **两信号一致才收录**;不一致样本落盘进人工抽查池;记录接受率
6. **balance & augment**:按类配额采样;变速(0.9/1.1)、加噪(MUSAN)、混响(RIR)增强,只用于训练集
7. **emit**:两个训练集版本(gold-only / gold+weak,供消融)+ 数据统计报告(类分布、时长分布、说话人数)

## 5. 模型与训练

- 底座:`openai/whisper-small` encoder(MIT,冻结)
- 适配:LoRA(r=16,注意力 q/v 投影)+ mean-pooling + 线性分类头
- 训练:交叉熵 + 类权重;单卡(AICR rtx 档 / Explorer);预计单次训练 ≤ 数小时
- 基线:多数类、ECAPA-TDNN(SpeechBrain,Apache-2.0)、Qwen2-Audio 零样本(Apache-2.0,本地推理)
- 全栈开源,LoRA adapter + 分类头以本人名义发 Hugging Face(带模型卡)

## 6. 评测

- 主指标 macro-F1;混淆矩阵重点分析 en-IN vs L2 类、en-GB vs en-AU
- EdAcc 域外测试单独报告
- 消融:gold-only vs gold+weak(头条数字)
- **Stretch(时间允许才做)**:口音感知路由 demo——按预测口音设置 Whisper 解码 prompt,对比默认解码 WER,给 Clipto proposal 提供业务数字

## 7. 交付物

1. 公开 GitHub repo(管线代码 + 复现脚本 + 单元测试 + CI)
2. Hugging Face 模型页(adapter 权重 + 模型卡)
3. 数据 datasheet(源、许可、丢弃统计、弱标注接受率)
4. Clipto integration proposal(2–3 页):端侧蒸馏/量化路线(ONNX/CoreML)、ASR 前置路由集成点、MCP 工具形态
5. mle 简历 3 条 bullet(建成后写,替换 LLM Serving 格位;遵守 canonical 三段式,禁 SOTA 措辞,用 weak supervision / data-centric 叙事)

## 8. 周期(3 周 part-time)

- W1:ingest + taxonomy + filter + dedup/split 跑通,金标数据集成型;EdAcc 许可核对
- W2:LoRA 训练 + 基线 + 域内评测;weak-label 管线跑通
- W3:消融 + EdAcc 域外 + HF 发布 + datasheet;有余力做路由 demo;写 Clipto proposal

## 9. 风险与诚实边界

- CommonVoice 自报标签噪声 → 白名单映射 + 每类人工抽查 50 条
- YouTube ToS → 不再分发音频,只发 URL 清单;抓取限速
- 语音栈对作者是新领域 → 周期按 3 周计,W1 结束若 ingest 未跑通则砍 L2 类别数止损
- 简历/面试措辞:LoRA fine-tune 开源底座(非从零训练);弱标注是 weak supervision(非人工金标);不声称 SOTA

## 10. 修订记录(v1.1 + v1.2,取代上文冲突条款)

两轮 review 后固化的设计修订,完整落地方案见实施计划。与 §1–§9 冲突处以本节为准:

**v1.1(数据血缘/防泄漏/审计/统计,9 条)**
1. 消融统计:每配置 3 seeds(17/42/1337);Δmacro-F1 用测试集 speaker 级分层 bootstrap 95% CI。
2. 弱标注防循环:pin Qwen2-Audio revision sha + prompt 版本化;已接受弱标签人工盲审(25 条/类);某类 precision < 0.80 → 该类弱标签整体剔除;弱标签只进 train(schema 机器强制)。
3. YouTube 证据等级 E1/E2/E3,只接受 E1/E2 且与 Qwen 一致;弱标数据只进 train。
4. 去重超越 speaker_id:ECAPA 说话人识别 + 近重复音频 + 转写重合检查。
5. 统一 schema 扩充:clip_id、原始文件引用、许可、采样率、taxonomy 版本、质量指标、label provenance、consensus_score、reject_reason。
6. EdAcc W1 双重验证(许可 + 标签映射);覆盖不足的类从域外 macro-F1 排除(supported-class macro-F1)。
7. Whisper masked mean-pooling(attention mask 推导有效帧数);>30s 取中心窗。
8. 止损不砍 8 类:先砍补充源,再缩弱标注规模,最后砍发布类工作。
9. demo/HF 发布/proposal 排在核心实验之后;每周 go/no-go 门禁。

**v1.2(实验有效性,响应第二轮 review)**
10. **Source-confounding 控制**:G1 前产出 source × accent 矩阵;单源占比 >90% 的类标记 confounded 并限定结论措辞;测试集每类尽量 ≥2 源;结果按源分层报告;加 LOSO 诊断。
11. **三臂公平消融**:A gold(epoch 对齐)/ B gold oversampled(步数与 C 相同)/ C gold+weak;共享 augmentation、sampler、固定步数预算、ckpt 规则;**头条数字 = C−B**,A−B 隔离预算效应。
12. **统计措辞**:test-speaker bootstrap CI 与 seed 变异分开报告;只写 "bootstrap CI excludes zero",不泛称 statistically significant。
13. 阶段化 schema(raw/qc/split);命名纪律:snr_proxy_db、consensus_score、frozen ECAPA embedding probe、supported-class macro-F1。
14. 范围收缩:3 周核心 = Common Voice + L2-ARCTIC + EdAcc + 小规模严策展 YouTube(300–600 clips);VCTK、SAA、全局说话人聚类、HF 发布、路由 demo 后置 backlog;**弱标注绝不推迟**(核心论点),只缩规模。
15. 实施不依赖会话特定 skill;集群交互为 repo 内显式 sbatch + 轮询脚本。
