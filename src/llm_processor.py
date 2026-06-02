"""
LLM文本处理模块
使用DeepSeek LLM进行：
1. 角色识别与提取
2. 对话/旁白拆分
3. 情绪与风格标注
4. 生成结构化剧本
"""

import json
import time
import logging
import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from openai import OpenAI

from .config import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class Character:
    """角色信息"""
    name: str                    # 角色名称
    description: str = ""         # 角色描述
    voice_description: str = ""   # 音色描述（用于voicedesign）
    voice_sample: str = ""        # 音色样本路径（用于voiceclone）
    preset_voice: str = ""        # 预置音色名（用于preset）


@dataclass
class SpeechSegment:
    """语音片段（剧本中的一个合成单元）"""
    index: int                    # 片段序号
    speaker: str                  # 说话者（角色名或"旁白"）
    text: str                    # 要合成的文本
    emotion: str = ""             # 情绪标签
    style_instruction: str = ""   # 自然语言风格指令（放在user消息中）
    audio_tags: str = ""          # 音频标签（放在assistant文本开头）


@dataclass
class ChapterScript:
    """章节剧本"""
    chapter_index: int
    chapter_title: str
    characters: List[Character]
    segments: List[SpeechSegment]


# ============================================================
# Prompt模板
# ============================================================

CHARACTER_ANALYSIS_PROMPT = """你是一位专业的有声书制作导演。请分析以下小说文本，提取所有出现的角色信息。

要求：
1. 识别所有有台词的角色
2. 为每个角色生成一段音色描述（用于TTS音色设计），描述应包含性别、年龄、音色质感、说话风格等
3. 用JSON格式输出

输出格式（严格JSON，不要markdown代码块）：
{
  "characters": [
    {
      "name": "角色名",
      "description": "角色简介（一句话描述身份和性格）",
      "voice_description": "音色描述（用于TTS合成，描述性别、年龄、音色特点、说话风格）"
    }
  ]
}

小说文本：
{text}
"""

SCRIPT_GENERATION_PROMPT = """你是一位专业的有声书剧本编剧。请将以下小说文本转换为有声书剧本。

已有角色信息：
{character_info}

要求：
1. 将文本拆分为"旁白"和各角色的"对话"片段
2. 每个片段不超过{max_chars}个字符
3. 为每个片段标注说话者(speaker)、情绪(emotion)和风格指令(style_instruction)
4. 旁白部分使用沉稳的讲述风格
5. 对话部分根据角色性格和场景情绪添加风格指令
6. 风格指令用自然语言描述，告诉TTS模型应该如何演绎这段话
7. 情绪标签从以下选择：平静/开心/悲伤/愤怒/恐惧/惊讶/兴奋/委屈/冷漠/怅然/无奈/愧疚/紧张/温柔/严肃/激动/疲惫

输出格式（严格JSON数组，不要markdown代码块）：
[
  {
    "speaker": "旁白/角色名",
    "text": "要朗读的文本内容",
    "emotion": "情绪标签",
    "style_instruction": "自然语言风格描述，指导TTS如何演绎"
  }
]

小说文本：
{text}
"""


class LLMProcessor:
    """DeepSeek LLM文本处理器"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.deepseek.api_key,
            base_url=config.deepseek.base_url,
        )
        self.model = config.deepseek.model
        self.max_retries = config.llm.max_retries
        self.request_interval = config.llm.request_interval
        self.chunk_max_chars = config.llm.chunk_max_chars
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        调用DeepSeek LLM（带重试）
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
        
        Returns:
            LLM返回的文本内容
        """
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,  # 低温度保证结构化输出稳定
            "stream": False,
        }
        
        # 思考模式
        if self.config.llm.thinking_mode:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            kwargs["reasoning_effort"] = "high"
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"调用DeepSeek LLM (尝试 {attempt + 1}/{self.max_retries})")
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                logger.info(f"LLM返回内容长度: {len(content)} 字符")
                return content
            except Exception as e:
                logger.warning(f"LLM调用失败 (尝试 {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.request_interval * (attempt + 1))
                else:
                    raise RuntimeError(f"LLM调用失败，已重试{self.max_retries}次: {e}")
        
        raise RuntimeError("LLM调用失败")
    
    def _parse_json_response(self, response_text: str) -> dict:
        """解析LLM返回的JSON（容错处理）"""
        # 尝试直接解析
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # 尝试提取JSON代码块
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass
        
        # 尝试找到第一个 { 和最后一个 }
        start = response_text.find('{')
        end = response_text.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(response_text[start:end + 1])
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"无法解析LLM返回的JSON:\n{response_text[:500]}")
    
    def analyze_characters(self, text: str) -> List[Character]:
        """
        分析文本中的角色
        
        Args:
            text: 小说文本（可以是节选）
        
        Returns:
            角色列表
        """
        logger.info("开始角色分析...")
        
        # 如果文本太长，取前面部分进行分析
        sample_text = text[:8000] if len(text) > 8000 else text
        
        prompt = CHARACTER_ANALYSIS_PROMPT.format(text=sample_text)
        
        # 使用简单的system prompt
        response = self._call_llm(
            system_prompt="你是一位专业的有声书制作导演，擅长角色分析和声音设计。",
            user_prompt=prompt,
        )
        
        result = self._parse_json_response(response)
        characters = []
        
        for char_data in result.get("characters", []):
            character = Character(
                name=char_data.get("name", "未知"),
                description=char_data.get("description", ""),
                voice_description=char_data.get("voice_description", ""),
            )
            characters.append(character)
            logger.info(f"  识别角色: {character.name} - {character.description}")
        
        # 添加旁白角色
        narrator = Character(
            name="旁白",
            description="有声书旁白/讲述者",
            voice_description=self.config.voices.get("voicedesign", {}).get("narrator", ""),
        )
        characters.insert(0, narrator)
        
        logger.info(f"共识别 {len(characters)} 个角色（含旁白）")
        return characters
    
    def generate_script(self, text: str, characters: List[Character]) -> List[SpeechSegment]:
        """
        生成有声书剧本
        
        Args:
            text: 章节文本
            characters: 已识别的角色列表
        
        Returns:
            语音片段列表
        """
        logger.info(f"生成剧本（文本长度: {len(text)} 字符）...")
        
        # 构建角色信息字符串
        char_info = "\n".join([
            f"- {c.name}: {c.description} (音色: {c.voice_description})"
            for c in characters
        ])
        
        prompt = SCRIPT_GENERATION_PROMPT.format(
            character_info=char_info,
            max_chars=self.chunk_max_chars,
            text=text,
        )
        
        response = self._call_llm(
            system_prompt="你是一位专业的有声书剧本编剧。你只输出JSON格式的剧本数据，不输出其他任何内容。",
            user_prompt=prompt,
        )
        
        result = self._parse_json_response(response)
        
        # 如果返回的是包含segments的对象，提取segments
        if isinstance(result, dict) and "segments" in result:
            segments_data = result["segments"]
        elif isinstance(result, list):
            segments_data = result
        else:
            segments_data = []
        
        segments = []
        for i, seg_data in enumerate(segments_data):
            segment = SpeechSegment(
                index=i,
                speaker=seg_data.get("speaker", "旁白"),
                text=seg_data.get("text", ""),
                emotion=seg_data.get("emotion", "平静"),
                style_instruction=seg_data.get("style_instruction", ""),
            )
            segments.append(segment)
        
        logger.info(f"生成 {len(segments)} 个语音片段")
        return segments
    
    def process_chapter(self, chapter_index: int, chapter_title: str,
                         text: str, characters: Optional[List[Character]] = None) -> ChapterScript:
        """
        处理单个章节
        
        Args:
            chapter_index: 章节序号
            chapter_title: 章节标题
            text: 章节文本
            characters: 已有的角色列表（如果为None则自动分析）
        
        Returns:
            章节剧本
        """
        logger.info(f"处理章节 [{chapter_index}] {chapter_title}")
        
        # 如果没有提供角色，先分析角色
        if characters is None:
            characters = self.analyze_characters(text)
        
        # 生成剧本
        segments = self.generate_script(text, characters)
        
        return ChapterScript(
            chapter_index=chapter_index,
            chapter_title=chapter_title,
            characters=characters,
            segments=segments,
        )
