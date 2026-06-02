"""
配置管理模块
加载和管理项目配置
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DeepSeekConfig:
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"


@dataclass
class MiMoConfig:
    api_key: str = ""
    base_url: str = "https://api.xiaomimimo.com/v1"


@dataclass
class LLMConfig:
    chunk_max_chars: int = 500
    thinking_mode: bool = False
    max_retries: int = 3
    request_interval: float = 1.0


@dataclass
class TTSConfig:
    voice_mode: str = "voicedesign"  # preset / voicedesign / voiceclone
    audio_format: str = "wav"
    max_retries: int = 3
    request_interval: float = 2.0
    silence_between_paragraphs: float = 1.0


@dataclass
class OutputConfig:
    format: str = "mp3"
    mp3_bitrate: str = "192k"
    dir: str = "output"
    keep_segments: bool = False
    sample_rate: int = 24000


@dataclass
class MetadataConfig:
    title: str = "AI有声书"
    author: str = "未知"
    chapter_max_chars: int = 10000


@dataclass
class AppConfig:
    deepseek: DeepSeekConfig = field(default_factory=DeepSeekConfig)
    mimo: MiMoConfig = field(default_factory=MiMoConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    # 音色配置（原始字典，按模式不同结构不同）
    voices: dict = field(default_factory=dict)

    # 项目根目录
    project_root: str = ""


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径（相对于项目根目录或绝对路径）
    
    Returns:
        AppConfig 配置对象
    """
    # 确定项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 如果是相对路径，相对于项目根目录查找
    if not os.path.isabs(config_path):
        abs_path = os.path.join(project_root, config_path)
    else:
        abs_path = config_path
    
    if not os.path.exists(abs_path):
        raise FileNotFoundError(
            f"配置文件不存在: {abs_path}\n"
            f"请复制 config.example.yaml 为 config.yaml 并填入你的API密钥"
        )
    
    with open(abs_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    
    # 环境变量覆盖（优先使用环境变量中的API Key）
    if os.environ.get("DEEPSEEK_API_KEY"):
        raw.setdefault("api", {}).setdefault("deepseek", {})["api_key"] = os.environ["DEEPSEEK_API_KEY"]
    if os.environ.get("MIMO_API_KEY"):
        raw.setdefault("api", {}).setdefault("mimo", {})["api_key"] = os.environ["MIMO_API_KEY"]
    
    # 解析各模块配置
    api_cfg = raw.get("api", {})
    ds_cfg = api_cfg.get("deepseek", {})
    mimo_cfg = api_cfg.get("mimo", {})
    llm_cfg = raw.get("llm", {})
    tts_cfg = raw.get("tts", {})
    output_cfg = raw.get("output", {})
    meta_cfg = raw.get("metadata", {})
    voices_cfg = raw.get("voices", {})
    
    config = AppConfig(
        deepseek=DeepSeekConfig(
            api_key=ds_cfg.get("api_key", ""),
            base_url=ds_cfg.get("base_url", "https://api.deepseek.com"),
            model=ds_cfg.get("model", "deepseek-v4-flash"),
        ),
        mimo=MiMoConfig(
            api_key=mimo_cfg.get("api_key", ""),
            base_url=mimo_cfg.get("base_url", "https://api.xiaomimimo.com/v1"),
        ),
        llm=LLMConfig(
            chunk_max_chars=llm_cfg.get("chunk_max_chars", 500),
            thinking_mode=llm_cfg.get("thinking_mode", False),
            max_retries=llm_cfg.get("max_retries", 3),
            request_interval=llm_cfg.get("request_interval", 1.0),
        ),
        tts=TTSConfig(
            voice_mode=tts_cfg.get("voice_mode", "voicedesign"),
            audio_format=tts_cfg.get("audio_format", "wav"),
            max_retries=tts_cfg.get("max_retries", 3),
            request_interval=tts_cfg.get("request_interval", 2.0),
            silence_between_paragraphs=tts_cfg.get("silence_between_paragraphs", 1.0),
        ),
        output=OutputConfig(
            format=output_cfg.get("format", "mp3"),
            mp3_bitrate=output_cfg.get("mp3_bitrate", "192k"),
            dir=output_cfg.get("dir", "output"),
            keep_segments=output_cfg.get("keep_segments", False),
            sample_rate=output_cfg.get("sample_rate", 24000),
        ),
        metadata=MetadataConfig(
            title=meta_cfg.get("title", "AI有声书"),
            author=meta_cfg.get("author", "未知"),
            chapter_max_chars=meta_cfg.get("chapter_max_chars", 10000),
        ),
        voices=voices_cfg,
        project_root=project_root,
    )
    
    return config
