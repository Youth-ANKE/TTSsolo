# 🎧 AI有声书生成器

基于 **DeepSeek LLM** + **小米MiMo-V2.5-TTS** 的多人AI有声书自动生成项目。

## ✨ 核心特性

- 📖 **多格式输入**：支持 TXT、EPUB、PDF 电子书格式
- 🎭 **智能角色识别**：DeepSeek LLM 自动识别小说角色并生成角色描述
- 🎨 **三种音色模式**：
  - **预置音色**（preset）：使用内置精品音色（冰糖/茉莉/苏打/白桦等）
  - **音色设计**（voicedesign）：通过文字描述定制角色音色
  - **音色克隆**（voiceclone）：基于音频样本复刻任意声音
- 🎬 **情绪风格控制**：自动标注情绪标签和风格指令，支持导演模式
- 📦 **可配置输出**：支持 WAV/MP3 格式，可调比特率
- 📝 **剧本导出**：生成结构化JSON剧本，可审查和手动调整

## 🏗️ 项目架构

```
ai-audiobook/
├── config.example.yaml    # 配置模板
├── config.yaml            # 你的配置（需自行创建）
├── requirements.txt       # Python依赖
├── src/
│   ├── __init__.py
│   ├── __main__.py        # python -m src 入口
│   ├── main.py            # 主流程编排
│   ├── config.py          # 配置管理
│   ├── parser.py          # 文本解析（TXT/EPUB/PDF）
│   ├── llm_processor.py   # DeepSeek LLM文本处理
│   ├── tts_engine.py      # MiMo TTS语音合成
│   └── audio_processor.py # 音频拼接与格式转换
├── voices/                # 音色样本文件（voiceclone模式）
├── output/                # 输出目录
│   ├── scripts/            # 生成的剧本JSON
│   └── segments/           # 分段音频（可选保留）
└── sample/
    └── sample_novel.txt   # 示例小说文本
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt

# 可选：安装EPUB/PDF支持
pip install ebooklib PyMuPDF

# 可选：安装MP3转换支持
sudo apt install ffmpeg
# 或
pip install pydub
```

### 2. 配置API密钥

```bash
# 方式一：环境变量
export DEEPSEEK_API_KEY="your_deepseek_key"
export MIMO_API_KEY="your_mimo_key"

# 方式二：配置文件
cp config.example.yaml config.yaml
# 编辑 config.yaml 填入API密钥
```

### 3. 运行

```bash
# 基本用法
python -m src sample/sample_novel.txt

# 指定配置文件
python -m src input.txt --config my_config.yaml

# 只处理第1-5章
python -m src input.txt --chapters 1 5

# 仅生成剧本（不合成语音，用于预览）
python -m src input.txt --script-only

# 使用预置音色
python -m src input.txt --voice-mode preset

# 输出WAV格式
python -m src input.txt --output-format wav
```

## ⚙️ 配置说明

### 音色模式对比

| 模式 | Model ID | 适用场景 | 需要素材 |
|------|----------|----------|----------|
| `preset` | `mimo-v2.5-tts` | 快速体验，角色区分度有限 | 无 |
| `voicedesign` | `mimo-v2.5-tts-voicedesign` | 自定义角色音色，最灵活 | 文字描述 |
| `voiceclone` | `mimo-v2.5-tts-voiceclone` | 克隆真实声音，最逼真 | 音频样本 |

### 音色设计示例（voicedesign模式）

```yaml
voices:
  voicedesign:
    narrator: |
      一位中年男性，嗓音醇厚温暖，语速适中偏慢，
      像一位经验丰富的电台播音员在朗读文学作品。
    voice_descriptions:
      "林晓": "年轻女性，声音清亮但带着一丝紧张感，语速偏快，像二十出头的聪明女孩。"
      "神秘男人": "中年男性，声音低沉沙哑，语速缓慢而神秘，带着一种压迫感。"
```

### 音色克隆示例（voiceclone模式）

```yaml
voices:
  voiceclone:
    narrator: "voices/narrator_sample.wav"
    voice_samples:
      "林晓": "voices/linxiao_sample.wav"
      "神秘男人": "voices/man_sample.wav"
```

## 📖 工作流程

```
输入文件 → 文本解析 → LLM角色分析 → LLM剧本生成 → TTS语音合成 → 音频拼接 → 输出文件
   │           │            │               │              │              │           │
  TXT/       提取纯文本    识别角色、       拆分旁白/       多角色多音色     合并分段     WAV/MP3
  EPUB/PDF    自动分章     生成音色描述     对话、标注情绪   语音合成        插入静音
```

## 🔑 API文档参考

- [DeepSeek API文档](https://api-docs.deepseek.com/zh-cn/)
- [小米MiMo TTS v2.5文档](https://platform.xiaomimimo.com/docs/zh-CN/usage-guide/speech-synthesis-v2.5)

## 📝 注意事项

1. **API费用**：DeepSeek和MiMo API均按用量计费，建议先用 `--script-only` 预览剧本
2. **文本长度**：长篇小说建议分章节处理，使用 `--chapters` 参数控制范围
3. **音色样本**：voiceclone模式需要提供10-30秒的清晰音频样本
4. **MP3转换**：需要安装 ffmpeg 或 pydub，否则只能输出WAV格式
5. **并发限制**：API有调用频率限制，程序内置了请求间隔控制

## 📄 License

MIT
