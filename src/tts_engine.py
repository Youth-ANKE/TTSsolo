"""
TTS语音合成模块
使用小米MiMo-V2.5-TTS进行多角色语音合成
支持三种音色模式：
1. preset - 预置音色
2. voicedesign - 文本设计音色
3. voiceclone - 音频克隆音色

支持同步和异步两种模式。
"""

import os
import base64
import time
import asyncio
import logging
from typing import List, Optional, Dict
from dataclasses import dataclass

from openai import OpenAI, AsyncOpenAI

from .config import AppConfig
from .llm_processor import SpeechSegment, Character, ChapterScript

logger = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    """音频片段"""
    index: int
    speaker: str
    file_path: str  # 保存的音频文件路径
    duration: float = 0.0  # 时长（秒）
    chapter_index: int = -1  # 所属章节


# ============================================================
# 同步版本（保留原有功能）
# ============================================================

class TTSEngine:
    """MiMo TTS语音合成引擎（同步版本）"""
    
    PRESET_VOICE_MAP = {
        "旁白": "白桦",
        "冰糖": "冰糖",
        "茉莉": "茉莉",
        "苏打": "苏打",
        "白桦": "白桦",
        "Mia": "Mia",
        "Chloe": "Chloe",
        "Milo": "Milo",
        "Dean": "Dean",
    }
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.mimo.api_key,
            base_url=config.mimo.base_url,
        )
        self.voice_mode = config.tts.voice_mode
        self.audio_format = config.tts.audio_format
        self.max_retries = config.tts.max_retries
        self.request_interval = config.tts.request_interval
        self.output_dir = config.output.dir
        self._load_voice_config()
    
    def _load_voice_config(self):
        """加载音色配置"""
        voices = self.config.voices
        if self.voice_mode == "preset":
            self.preset_voices = voices.get("preset", {}).get("voice_map", {})
            self.narrator_voice = voices.get("preset", {}).get("narrator", "白桦")
        elif self.voice_mode == "voicedesign":
            self.voice_descriptions = voices.get("voicedesign", {}).get("voice_descriptions", {})
            self.narrator_voice_desc = voices.get("voicedesign", {}).get("narrator", "")
        elif self.voice_mode == "voiceclone":
            self.voice_samples = voices.get("voiceclone", {}).get("voice_samples", {})
            self.narrator_voice_sample = voices.get("voiceclone", {}).get("narrator", "")
    
    def _get_voice_for_speaker(self, speaker: str, character: Optional[Character] = None):
        """根据说话者获取对应的音色配置"""
        if self.voice_mode == "preset":
            voice_name = self.preset_voices.get(speaker) or self.PRESET_VOICE_MAP.get(speaker, "白桦")
            if speaker == "旁白":
                voice_name = self.narrator_voice
            return "mimo-v2.5-tts", {"voice": voice_name}
        elif self.voice_mode == "voicedesign":
            voice_desc = ""
            if speaker == "旁白":
                voice_desc = self.narrator_voice_desc
            elif character and character.voice_description:
                voice_desc = character.voice_description
            elif speaker in self.voice_descriptions:
                voice_desc = self.voice_descriptions[speaker]
            if not voice_desc:
                voice_desc = "沉稳的讲述者声音，语速适中，吐字清晰。"
            return "mimo-v2.5-tts-voicedesign", {"voice_description": voice_desc}
        elif self.voice_mode == "voiceclone":
            voice_sample = ""
            if speaker == "旁白":
                voice_sample = self.narrator_voice_sample
            elif character and character.voice_sample:
                voice_sample = character.voice_sample
            elif speaker in self.voice_samples:
                voice_sample = self.voice_samples[speaker]
            if not voice_sample:
                logger.warning(f"角色 '{speaker}' 没有音色样本，回退到旁白音色")
                voice_sample = self.narrator_voice_sample
            return "mimo-v2.5-tts-voiceclone", {"voice_sample": voice_sample}
        return "mimo-v2.5-tts", {"voice": "白桦"}
    
    def _build_tts_request(self, text: str, speaker: str,
                           character: Optional[Character] = None,
                           style_instruction: str = "",
                           emotion: str = "") -> dict:
        """构建TTS API请求参数"""
        model, voice_config = self._get_voice_for_speaker(speaker, character)
        messages = []
        
        if self.voice_mode == "preset":
            user_content = style_instruction if style_instruction else ""
            if emotion and not user_content:
                user_content = f"用{emotion}的语气朗读。"
            if user_content:
                messages.append({"role": "user", "content": user_content})
            assistant_text = text
            if emotion:
                assistant_text = f"({emotion}){text}"
            messages.append({"role": "assistant", "content": assistant_text})
            return {"model": model, "messages": messages, "audio": {"format": self.audio_format, "voice": voice_config["voice"]}}
        
        elif self.voice_mode == "voicedesign":
            voice_desc = voice_config["voice_description"]
            user_content = voice_desc
            if style_instruction:
                user_content += f"\n{style_instruction}"
            messages.append({"role": "user", "content": user_content})
            assistant_text = text
            if emotion:
                assistant_text = f"({emotion}){text}"
            messages.append({"role": "assistant", "content": assistant_text})
            return {"model": model, "messages": messages, "audio": {"format": self.audio_format}}
        
        elif self.voice_mode == "voiceclone":
            user_content = style_instruction if style_instruction else ""
            if emotion and not user_content:
                user_content = f"用{emotion}的语气朗读。"
            if user_content:
                messages.append({"role": "user", "content": user_content})
            assistant_text = text
            if emotion:
                assistant_text = f"({emotion}){text}"
            messages.append({"role": "assistant", "content": assistant_text})
            voice_sample_path = voice_config["voice_sample"]
            abs_path = voice_sample_path if os.path.isabs(voice_sample_path) else os.path.join(self.config.project_root, voice_sample_path)
            with open(abs_path, "rb") as f:
                audio_data = base64.b64encode(f.read()).decode("utf-8")
            return {"model": model, "messages": messages, "audio": {"format": self.audio_format, "voice": {"audio": audio_data}}}
        
        return {"model": "mimo-v2.5-tts", "messages": [{"role": "assistant", "content": text}], "audio": {"format": self.audio_format, "voice": "白桦"}}
    
    def synthesize_segment(self, segment: SpeechSegment,
                           character: Optional[Character] = None,
                           output_path: str = "") -> AudioSegment:
        """合成单个语音片段"""
        if not output_path:
            output_path = os.path.join(self.output_dir, "segments", f"seg_{segment.index:04d}_{segment.speaker}.wav")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        request_params = self._build_tts_request(
            text=segment.text, speaker=segment.speaker,
            character=character, style_instruction=segment.style_instruction,
            emotion=segment.emotion,
        )
        
        logger.info(f"合成片段 [{segment.index}] 说话者={segment.speaker} 模式={self.voice_mode} 文本长度={len(segment.text)}")
        
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(**request_params)
                message = completion.choices[0].message
                audio_bytes = base64.b64decode(message.audio.data)
                with open(output_path, "wb") as f:
                    f.write(audio_bytes)
                logger.info(f"  ✓ 片段 [{segment.index}] 保存至 {output_path} ({len(audio_bytes)} bytes)")
                return AudioSegment(index=segment.index, speaker=segment.speaker, file_path=output_path)
            except Exception as e:
                logger.warning(f"  ✗ 片段 [{segment.index}] 合成失败 (尝试 {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.request_interval * (attempt + 1))
                else:
                    raise RuntimeError(f"片段 [{segment.index}] 合成失败，已重试{self.max_retries}次: {e}")
        raise RuntimeError(f"片段 [{segment.index}] 合成失败")
    
    def synthesize_chapter(self, script: ChapterScript, segment_dir: str = "") -> List[AudioSegment]:
        """合成整个章节的所有语音片段"""
        if not segment_dir:
            segment_dir = os.path.join(self.output_dir, "segments", f"chapter_{script.chapter_index:03d}")
        character_map = {c.name: c for c in script.characters}
        audio_segments = []
        total = len(script.segments)
        for i, segment in enumerate(script.segments):
            logger.info(f"合成进度: {i + 1}/{total}")
            character = character_map.get(segment.speaker)
            output_path = os.path.join(segment_dir, f"seg_{segment.index:04d}.wav")
            audio_seg = self.synthesize_segment(segment=segment, character=character, output_path=output_path)
            audio_seg.chapter_index = script.chapter_index
            audio_segments.append(audio_seg)
            if i < total - 1:
                time.sleep(self.request_interval)
        logger.info(f"章节 [{script.chapter_index}] 合成完成，共 {len(audio_segments)} 个片段")
        return audio_segments


# ============================================================
# 异步版本（支持并发合成 + 流水线）
# ============================================================

class AsyncTTSEngine:
    """MiMo TTS语音合成引擎（异步版本，支持并发控制）"""
    
    PRESET_VOICE_MAP = {
        "旁白": "白桦", "冰糖": "冰糖", "茉莉": "茉莉", "苏打": "苏打", "白桦": "白桦",
        "Mia": "Mia", "Chloe": "Chloe", "Milo": "Milo", "Dean": "Dean",
    }
    
    def __init__(self, config: AppConfig, max_concurrent: int = 3):
        """
        Args:
            config: 应用配置
            max_concurrent: 最大并发TTS请求数（避免触发API限速）
        """
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.mimo.api_key,
            base_url=config.mimo.base_url,
        )
        self.voice_mode = config.tts.voice_mode
        self.audio_format = config.tts.audio_format
        self.max_retries = config.tts.max_retries
        self.request_interval = config.tts.request_interval
        self.output_dir = config.output.dir
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        
        # 复用同步版本的配置加载和请求构建逻辑
        self._sync_engine = TTSEngine(config)
        self._load_voice_config()
    
    def _load_voice_config(self):
        """加载音色配置（复用同步版本）"""
        self._sync_engine._load_voice_config()
    
    async def synthesize_segment(self, segment: SpeechSegment,
                                  character: Optional[Character] = None,
                                  output_path: str = "",
                                  chapter_index: int = -1) -> AudioSegment:
        """
        异步合成单个语音片段（受Semaphore控制并发）
        """
        if not output_path:
            output_path = os.path.join(
                self.output_dir, "segments",
                f"seg_{segment.index:04d}_{segment.speaker}.wav"
            )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        request_params = self._sync_engine._build_tts_request(
            text=segment.text, speaker=segment.speaker,
            character=character, style_instruction=segment.style_instruction,
            emotion=segment.emotion,
        )
        
        logger.info(f"[Async] 合成片段 [{segment.index}] 说话者={segment.speaker} 文本长度={len(segment.text)}")
        
        async with self._semaphore:
            for attempt in range(self.max_retries):
                try:
                    completion = await self.client.chat.completions.create(**request_params)
                    message = completion.choices[0].message
                    audio_bytes = base64.b64decode(message.audio.data)
                    
                    with open(output_path, "wb") as f:
                        f.write(audio_bytes)
                    
                    logger.info(f"[Async]  ✓ 片段 [{segment.index}] 保存至 {output_path} ({len(audio_bytes)} bytes)")
                    
                    return AudioSegment(
                        index=segment.index,
                        speaker=segment.speaker,
                        file_path=output_path,
                        chapter_index=chapter_index,
                    )
                
                except Exception as e:
                    logger.warning(f"[Async]  ✗ 片段 [{segment.index}] 合成失败 (尝试 {attempt + 1}): {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.request_interval * (attempt + 1))
                    else:
                        raise RuntimeError(f"片段 [{segment.index}] 合成失败，已重试{self.max_retries}次: {e}")
        
        raise RuntimeError(f"片段 [{segment.index}] 合成失败")
    
    async def synthesize_chapter(self, script: ChapterScript,
                                  segment_dir: str = "") -> List[AudioSegment]:
        """
        异步合成整个章节（并发控制，但保持片段顺序）
        
        注意：虽然并发发送请求，但返回结果按顺序排列，
        以确保音频拼接时片段顺序正确。
        """
        if not segment_dir:
            segment_dir = os.path.join(
                self.output_dir, "segments",
                f"chapter_{script.chapter_index:03d}"
            )
        
        character_map = {c.name: c for c in script.characters}
        total = len(script.segments)
        
        # 创建所有异步任务
        tasks = []
        for i, segment in enumerate(script.segments):
            character = character_map.get(segment.speaker)
            output_path = os.path.join(segment_dir, f"seg_{segment.index:04d}.wav")
            
            task = self.synthesize_segment(
                segment=segment,
                character=character,
                output_path=output_path,
                chapter_index=script.chapter_index,
            )
            tasks.append(task)
        
        logger.info(f"[Async] 并发提交 {total} 个TTS任务（最大并发: {self.max_concurrent}）")
        
        # 并发执行，按顺序收集结果
        audio_segments = []
        for i, task in enumerate(tasks):
            result = await task
            audio_segments.append(result)
            logger.info(f"[Async] 章节合成进度: {i + 1}/{total}")
        
        logger.info(f"[Async] 章节 [{script.chapter_index}] 合成完成，共 {len(audio_segments)} 个片段")
        return audio_segments
