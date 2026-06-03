<div align="center">

# 🎧 TTSsolo — AI 多人有声书生成器

<p>
  <a href="https://github.com/Youth-ANKE/TTSsolo/stargazers"><img src="https://img.shields.io/github/stars/Youth-ANKE/TTSsolo?style=flat-square&color=6366f1" alt="Stars"></a>
  <a href="https://github.com/Youth-ANKE/TTSsolo/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Youth-ANKE/TTSsolo?style=flat-square&color=22c55e" alt="License"></a>
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&color=06b6d4" alt="Python">
  <img src="https://img.shields.io/badge/DeepSeek-LLM-6366f1?style=flat-square" alt="DeepSeek">
  <img src="https://img.shields.io/badge/MiMo-TTS-f59e0b?style=flat-square" alt="MiMo TTS">
</p>

**基于 DeepSeek LLM + 小米 MiMo-V2.5-TTS 的多人 AI 有声书自动生成项目**

支持智能角色识别、三种音色模式（预置/设计/克隆）、流水线并行加速、Web 可视化界面

[快速开始](#-快速开始) · [配置说明](#-配置详解) · [Web 界面](#-web-可视化界面) · [API 参考](#-api-文档参考)

</div>

---

## 📑 目录

- [功能特性](#-功能特性)
- [项目架构](#-项目架构)
- [快速开始](#-快速开始)
- [配置详解](#-配置详解)
- [使用指南](#-使用指南)
- [Web 可视化界面](#-web-可视化界面)
- [工作流程](#-工作流程)
- [性能优化](#-性能优化)
- [故障排除](#-故障排除)
- [更新日志](#-更新日志)
- [贡献指南](#-贡献指南)
- [许可证](#-许可证)

---

## ✨ 功能特性

### 📖 多格式输入支持
- **TXT** — 纯文本文件，自动识别章节标题
- **EPUB** — 电子书格式（需安装 `ebooklib`）
- **PDF** — 扫描版/文字版 PDF（需安装 `PyMuPDF`）
- 智能章节拆分：自动识别 `第X章`、`Chapter X`、`CHAPTER X` 等格式

### 🎭 智能角色识别与音色分配
- DeepSeek LLM 自动分析文本，识别所有出场角色
- 为每个角色生成性格特征与音色描述
- 支持旁白与对话的自动区分

### 🎨 三种音色模式

| 模式 | 模型 ID | 特点 | 所需素材 |
|------|---------|------|----------|
| **预置音色** `preset` | `mimo-v2.5-tts` | 内置精品音色，开箱即用 | 无 |
| **音色设计** `voicedesign` | `mimo-v2.5-tts-voicedesign` | 文字描述生成任意音色，最灵活 | 文字描述 |
| **音色克隆** `voiceclone` | `mimo-v2.5-tts-voiceclone` | 基于音频样本复刻真实声音，最逼真 | 10-30秒音频样本 |

**预置音色列表**：冰糖、茉莉、苏打、白桦、青竹、琥珀、星尘、夜雨

### 🎬 情绪与风格控制
- 自动标注情绪标签：`平静`、`温和`、`惊讶`、`紧张`、`恐惧`、`愤怒`、`激动`、`怅然`、`悲伤`、`冷漠`、`开心`
- 支持导演模式：通过情绪标签精细控制每句台词的演绎风格
- 段落间自动插入静音间隔（可配置时长）

### ⚡ 双模式处理引擎

| 模式 | 特点 | 适用场景 |
|------|------|----------|
| **串行模式** | 逐章依次处理，稳定可靠，内存占用低 | 长文本、资源受限环境 |
| **流水线模式** `--pipeline` | LLM→TTS→Output 三阶段并行，片段级流式处理 | 追求速度、短篇/中篇 |

流水线架构：
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ LLM Worker   │────→│ TTS Worker   │────→│ Output Worker│
│ (剧本生成)    │  Q1 │ (语音合成)    │  Q2 │ (增量写入)    │
└─────────────┘     └─────────────┘     └─────────────┘
```

### 🖥️ Web 可视化界面
- 单文件 HTML，无需构建工具，直接浏览器打开
- 实时流水线可视化：进度条、波形动画、情绪热力图
- 成本估算器：实时计算 LLM + TTS 预估费用
- 音色试听：设计/预置音色即时预览
- 配置模板管理：保存/加载/导出/导入 YAML 配置

### 📦 灵活输出
- **格式**：WAV（无损）/ MP3（192kbps，需 ffmpeg/pydub）
- **剧本导出**：结构化 JSON，可审查和手动调整
- **分段保留**：可选保留每句台词的独立音频文件

---

## 🏗️ 项目架构

```
ai-audiobook/
├── config.example.yaml    # 配置模板
├── config.yaml            # 你的配置（从模板复制并修改）
├── requirements.txt       # Python 依赖
├── README.md              # 本文档
│
├── src/                   # 核心源码
│   ├── __init__.py        # 包信息
│   ├── __main__.py        # python -m src 入口
│   ├── main.py            # CLI 主流程编排（串行/流水线模式）
│   ├── config.py          # YAML 配置加载与环境变量覆盖
│   ├── parser.py          # 文本解析器（TXT/EPUB/PDF + 智能分章）
│   ├── llm_processor.py   # DeepSeek LLM 文本处理（同步 + 异步）
│   ├── tts_engine.py      # MiMo TTS 语音合成（同步 + 异步）
│   ├── audio_processor.py # 音频拼接、格式转换、流式 WAV 写入
│   └── pipeline.py        # 异步流水线（三 Worker + 双队列）
│
├── web/                   # Web 可视化界面
│   ├── index.html         # 单文件完整界面
│   └── assets/
│       └── hero-bg.jpg    # 首页背景图
│
├── voices/                # 音色样本文件（voiceclone 模式）
├── output/                # 输出目录
│   ├── scripts/           # 生成的剧本 JSON
│   └── segments/          # 分段音频（可选保留）
│
└── sample/
    └── sample_novel.txt   # 示例小说文本（悬疑短篇）
```

---

## 🚀 快速开始

### 环境要求

- **Python**: 3.9+
- **操作系统**: Linux / macOS / Windows (WSL 推荐)
- **可选**: ffmpeg（用于 MP3 转换）

### 1. 克隆项目

```bash
git clone https://github.com/Youth-ANKE/TTSsolo.git
cd TTSsolo
```

### 2. 安装依赖

```bash
$ 核心依赖
pip install -r requirements.txt

$ 可选：EPUB 支持
pip install ebooklib>=0.18

$ 可选：PDF 支持
pip install PyMuPDF>=1.23.0

$ 可选：MP3 转换（推荐安装 ffmpeg）
$ Ubuntu/Debian
sudo apt install ffmpeg
$ macOS
brew install ffmpeg
$ 或安装 Python 备选方案
pip install pydub>=0.25.1
```

### 3. 配置 API 密钥

**方式一：环境变量（推荐，适合 CI/CD）**

```bash
export DEEPSEEK_API_KEY="your_deepseek_key"
export MIMO_API_KEY="your_mimo_key"
```

**方式二：配置文件（推荐，适合本地开发）**

```bash
cp config.example.yaml config.yaml
$ 编辑 config.yaml，填入你的 API 密钥
```

> 🔑 获取 API Key：
> - [DeepSeek 开放平台](https://platform.deepseek.com/)
> - [小米 MiMo 开放平台](https://platform.xiaomimimo.com/)

### 4. 运行

```bash
$ 基本用法（使用默认配置）
python -m src sample/sample_novel.txt

$ 指定配置文件
python -m src input.txt --config my_config.yaml

$ 只处理第 1-5 章
python -m src input.txt --chapters 1 5

$ 仅生成剧本（不合成语音，用于预览和费用估算）
python -m src input.txt --script-only

$ 使用流水线并行加速
python -m src input.txt --pipeline

$ 使用预置音色模式
python -m src input.txt --voice-mode preset

$ 输出 WAV 无损格式
python -m src input.txt --output-format wav

$ 查看所有参数
python -m src --help
```

---

## ⚙️ 配置详解

`config.yaml` 完整字段说明：

### API 密钥配置

```yaml
api:
  deepseek:
    api_key: "YOUR_DEEPSEEK_API_KEY"    # 必填
    base_url: "https://api.deepseek.com" # 默认
    model: "deepseek-v4-flash"           # v4-flash(快/便宜) 或 v4-pro(强/贵)
  mimo:
    api_key: "YOUR_MIMO_API_KEY"         # 必填
    base_url: "https://api.xiaomimimo.com/v1" # 默认
```

### LLM 文本处理配置

```yaml
llm:
  chunk_max_chars: 500        # 每个文本块最大字符数（避免超出 TTS 限制）
  thinking_mode: false        # 是否启用 DeepSeek 思考模式
  max_retries: 3              # API 请求失败重试次数
  request_interval: 1         # 请求间隔（秒），控制速率
```

### TTS 语音合成配置

```yaml
tts:
  voice_mode: "voicedesign"   # preset / voicedesign / voiceclone
  audio_format: "wav"         # wav / pcm16
  max_retries: 3              # API 请求失败重试次数
  request_interval: 2         # 请求间隔（秒），控制速率
  silence_between_paragraphs: 1.0  # 段落间静音时长（秒）
```

### 音色配置（根据 voice_mode 选择对应配置）

**预置音色模式：**
```yaml
voices:
  preset:
    narrator: "白桦"          # 旁白音色
    voice_map:                # 角色 → 音色映射（覆盖自动识别结果）
      "林晓": "冰糖"
      "神秘男人": "苏打"
```

**音色设计模式（推荐）：**
```yaml
voices:
  voicedesign:
    narrator: |
      一位中年男性，嗓音醇厚温暖，语速适中偏慢，
      像一位经验丰富的电台播音员在朗读文学作品。
    voice_descriptions:       # 角色 → 音色描述
      "林晓": "年轻女性，声音清亮但带着一丝紧张感，语速偏快。"
      "神秘男人": "中年男性，声音低沉沙哑，语速缓慢而神秘，带着压迫感。"
```

**音色克隆模式：**
```yaml
voices:
  voiceclone:
    narrator: "voices/narrator_sample.wav"  # 旁白样本路径
    voice_samples:                          # 角色 → 样本路径
      "林晓": "voices/linxiao_sample.wav"
      "神秘男人": "voices/man_sample.wav"
```

### 音频输出配置

```yaml
output:
  format: "mp3"               # wav / mp3
  mp3_bitrate: "192k"         # MP3 比特率（仅 mp3 有效）
  dir: "output"               # 输出目录
  keep_segments: false        # 是否保留分段音频
  sample_rate: 24000          # 采样率（MiMo TTS 固定 24kHz）
```

### 有声书元数据

```yaml
metadata:
  title: "AI有声书"           # 书名
  author: "未知"              # 作者
  chapter_max_chars: 10000    # 每章最大字符数（用于自动分章）
```

---

## 📖 使用指南

### 完整 CLI 参数

```bash
python -m src <input_file> [选项]

位置参数:
  input_file              输入文本文件路径 (TXT/EPUB/PDF)

可选参数:
  -h, --help              显示帮助信息
  --config CONFIG         指定配置文件路径 (默认: config.yaml)
  --chapters START END    只处理指定章节范围 (1-based)
  --script-only           仅生成剧本 JSON，跳过 TTS 合成
  --pipeline              使用流水线并行模式（默认串行）
  --voice-mode MODE       强制指定音色模式 (preset/voicedesign/voiceclone)
  --output-format FORMAT  输出格式 (wav/mp3)
  --log-file LOG_FILE     指定日志文件路径
```

### 典型使用场景

**场景 1：快速体验（预置音色）**
```bash
python -m src novel.txt --voice-mode preset --chapters 1 3
```

**场景 2：高质量制作（音色设计 + 流水线）**
```bash
$ 先预览剧本（不花钱做 TTS）
python -m src novel.txt --script-only
$ 确认剧本无误后，正式生成
python -m src novel.txt --pipeline --voice-mode voicedesign
```

**场景 3：复刻真人声音（音色克隆）**
```bash
$ 1. 准备 10-30 秒的清晰音频样本放入 voices/ 目录
$ 2. 配置 config.yaml 的 voiceclone 部分
$ 3. 运行
python -m src novel.txt --voice-mode voiceclone
```

**场景 4：长篇分批处理**
```bash
$ 先生成第 1-10 章
python -m src novel.txt --chapters 1 10 --output-format wav
$ 再生成第 11-20 章
python -m src novel.txt --chapters 11 20 --output-format wav
$ 最后用 ffmpeg 合并
ffmpeg -i "concat:chapter_001_*.wav|chapter_002_*.wav" -acodec copy full_book.wav
```

---

## 🖥️ Web 可视化界面

TTSsolo 提供一个**单文件 HTML 前端**，无需构建工具，直接在浏览器中打开即可使用。

### 启动方式

```bash
$ 方式一：Python 内置服务器
cd web
python3 -m http.server 8765
$ 浏览器访问 http://localhost:8765

$ 方式二：直接打开文件
$ 用浏览器打开 web/index.html（部分功能需要服务器环境）
```

### 界面功能

| 模块 | 功能说明 |
|------|---------|
| **文件上传** | 拖拽/选择 TXT/EPUB/PDF，自动解析章节列表 |
| **章节选择** | 勾选需要生成的章节，查看每章字数 |
| **音色配置** | 旁白音色设计、添加多个角色音色、预置音色卡片选择 |
| **音色试听** | 点击试听按钮预览音色效果（模拟） |
| **参数设置** | 输出格式、LLM 模型、处理模式、章节范围 |
| **配置模板** | 保存当前配置、导出 YAML、导入配置、加载历史模板 |
| **成本估算** | 根据文本长度实时估算 LLM + TTS 费用 |
| **流水线可视化** | 实时进度条、波形动画、情绪热力图 |
| **对比播放器** | 同一文本片段用不同音色/情绪对比播放 |
| **输出管理** | 下载生成的音频文件、查看生成日志 |

### 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl + G` | 开始生成 |
| `Ctrl + S` | 保存配置模板 |
| `Ctrl + O` | 导入配置 |
| `Esc` | 取消/关闭弹窗 |

---

## 🔄 工作流程

```
输入文件
   │
   ▼
┌─────────────┐
│  文本解析器   │  ← 提取纯文本、自动识别章节标题、分章
│   parser.py  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ LLM 角色分析 │  ← 识别角色、生成性格/音色描述
│llm_processor │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ LLM 剧本生成 │  ← 拆分旁白/对话、标注情绪标签
│  (流式输出)  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  TTS 语音合成 │  ← 多角色多音色并行合成
│  tts_engine  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 音频后处理   │  ← 合并分段、插入静音、格式转换
│audio_processor│
└──────┬──────┘
       │
       ▼
    输出文件 (WAV/MP3 + JSON 剧本)
```

---

## ⚡ 性能优化

### 流水线模式 vs 串行模式

| 指标 | 串行模式 | 流水线模式 | 提升 |
|------|---------|-----------|------|
| 内存占用 | 低 | 中 | — |
| CPU 占用 | 低 | 中 | — |
| 总耗时（短篇） | 基准 | -30%~50% | ⬆️ |
| 总耗时（长篇） | 基准 | -20%~40% | ⬆️ |
| 稳定性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | — |

> 建议：短篇/中篇使用 `--pipeline`，长篇使用默认串行模式。

### 费用优化建议

1. **先用 `--script-only` 预览**：确认剧本质量后再付费做 TTS
2. **分章节处理**：避免一次性处理全书导致 API 超时
3. **使用 `deepseek-v4-flash`**：性价比最高，质量已足够用于角色识别
4. **合理设置 `chunk_max_chars`**：默认 500 字符，可根据内容调整
5. **控制 `request_interval`**：适当降低间隔可提速，但注意 API 速率限制

---

## 🛠️ 故障排除

### 常见问题

**Q: 安装依赖时 `soundfile` 报错？**
```bash
$ Linux 需要系统级音频库
sudo apt-get install libsndfile1
$ macOS
brew install libsndfile
```

**Q: EPUB/PDF 解析失败？**
```bash
$ 安装对应解析库
pip install ebooklib>=0.18      # EPUB
pip install PyMuPDF>=1.23.0     # PDF
```

**Q: MP3 转换失败？**
```bash
$ 方案一：安装 ffmpeg（推荐）
sudo apt install ffmpeg  # Linux
brew install ffmpeg      # macOS

$ 方案二：使用 pydub
pip install pydub>=0.25.1
```

**Q: API 返回 429（Too Many Requests）？**
- 增大 `request_interval` 配置值
- 检查 API 平台的速率限制文档
- 考虑升级 API 套餐

**Q: 流水线模式内存占用过高？**
- 减少并发数（修改 `tts_engine.py` 中的 `Semaphore` 值）
- 切换为串行模式
- 分更小的章节范围处理

**Q: 角色识别不准确？**
- 确保输入文本前 15000 字包含主要角色
- 在 `config.yaml` 中手动配置 `voice_map`/`voice_descriptions` 覆盖自动识别结果

**Q: 生成的音频有杂音/断句不自然？**
- 检查 `silence_between_paragraphs` 设置
- 尝试调整 `chunk_max_chars` 为更小的值
- 使用 WAV 格式输出，避免 MP3 编码损失

### 调试模式

```bash
$ 启用详细日志
python -m src input.txt --log-file debug.log
$ 查看日志
tail -f debug.log
```

---

## 📝 更新日志

### v1.1.0 (2025-06)
- ✨ 新增 Web 可视化界面（单文件 HTML）
- ✨ 新增流水线并行模式（`--pipeline`）
- ✨ 新增三种音色模式：预置/设计/克隆
- ✨ 新增 `--script-only` 仅生成剧本模式
- ✨ 新增情绪热力图、成本估算器、对比播放器
- 🔧 修复操作符优先级、async 并发、f-string 语法等 Bug
- 🔧 添加文件名安全净化、localStorage 兼容、移动端适配

### v1.0.0 (2025-05)
- 🎉 项目初始发布
- 支持 TXT/EPUB/PDF 输入
- 支持 DeepSeek LLM 角色识别与剧本生成
- 支持 MiMo TTS 语音合成
- 支持串行处理模式

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 开发环境搭建

```bash
git clone https://github.com/Youth-ANKE/TTSsolo.git
cd TTSsolo
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install ebooklib PyMuPDF pydub  # 可选依赖
```

### 代码规范

- 遵循 PEP 8 风格指南
- 所有公共函数添加类型注解
- 新功能需附带使用示例

### 提交 Issue

请包含以下信息：
- 操作系统和 Python 版本
- 复现步骤
- 错误日志（脱敏后）
- 配置文件（脱敏后）

---

## 📚 API 文档参考

- [DeepSeek API 文档](https://api-docs.deepseek.com/zh-cn/)
- [小米 MiMo TTS v2.5 文档](https://platform.xiaomimimo.com/docs/zh-CN/usage-guide/speech-synthesis-v2.5)

---

## ⚠️ 注意事项

1. **API 费用**：DeepSeek 和 MiMo API 均按用量计费，建议先用 `--script-only` 预览剧本
2. **文本长度**：长篇小说建议分章节处理，使用 `--chapters` 参数控制范围
3. **音色样本**：voiceclone 模式需要提供 10-30 秒的清晰音频样本
4. **MP3 转换**：需要安装 ffmpeg 或 pydub，否则只能输出 WAV 格式
5. **并发限制**：API 有调用频率限制，程序内置了请求间隔控制
6. **版权合规**：请确保你有权将输入文本转换为有声书

---

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源。

---

<div align="center">

**Made with ❤️ by Youth-ANKE**

如果这个项目对你有帮助，请给个 ⭐ Star！

</div>
