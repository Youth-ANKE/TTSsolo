"""
流水线编排模块
实现生产者-消费者模式的异步流水线：
  LLM剧本生成(Producer) → TTS语音合成(Worker) → 音频增量写入(Consumer)

三个阶段通过 asyncio.Queue 连接，实现片段级流水线并行。
"""

import re
import os
import json
import time
import asyncio
import logging
from typing import List, Optional, Dict
from dataclasses import dataclass

from .config import AppConfig
from .parser import parse_book, ParsedBook
from .llm_processor import (
    AsyncLLMProcessor, Character, SpeechSegment, ChapterScript
)
from .tts_engine import AsyncTTSEngine, AudioSegment
from .audio_processor import AudioPostProcessor

logger = logging.getLogger(__name__)


# 队列中的消息类型
@dataclass
class PipelineItem:
    """流水线中传递的数据单元"""
    chapter_index: int
    chapter_title: str
    segment: SpeechSegment          # 剧本片段
    character: Optional[Character]  # 对应角色
    is_last_in_chapter: bool = False  # 是否是本章最后一个片段


@dataclass
class AudioResult:
    """音频合成结果"""
    chapter_index: int
    chapter_title: str
    segment_index: int
    speaker: str
    file_path: str
    is_last_in_chapter: bool = False


class AudiobookPipeline:
    """
    AI有声书异步流水线生成器
    
    架构：
        ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
        │ LLM Worker   │────→│ TTS Worker   │────→│ Output Worker│
        │ (剧本生成)    │ Q1  │ (语音合成)    │ Q2  │ (增量写入)    │
        └─────────────┘     └─────────────┘     └─────────────┘
        
    特性：
    - 片段级流水线：第一章的TTS合成与第二章的剧本生成可并行
    - TTS并发控制：Semaphore限制最大并发数，避免API限速
    - 增量音频写入：边合成边写入WAV，无需等所有片段完成
    - 实时进度反馈：通过回调函数报告进度
    """
    
    def __init__(self, config: AppConfig,
                 tts_concurrent: int = 3,
                 queue_size: int = 20):
        """
        Args:
            config: 应用配置
            tts_concurrent: TTS最大并发数
            queue_size: 队列缓冲大小
        """
        self.config = config
        self.tts_concurrent = tts_concurrent
        self.queue_size = queue_size
        
        # 初始化异步引擎
        self.llm = AsyncLLMProcessor(config)
        self.tts = AsyncTTSEngine(config, max_concurrent=tts_concurrent)
        self.audio_processor = AudioPostProcessor(config)
        
        # 输出目录
        self.output_dir = os.path.join(config.project_root, config.output.dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.script_dir = os.path.join(self.output_dir, "scripts")
        os.makedirs(self.script_dir, exist_ok=True)
        
        # 统计信息
        self.stats = {
            "total_segments": 0,
            "completed_segments": 0,
            "total_chapters": 0,
            "completed_chapters": 0,
            "start_time": 0,
        }
        
        # 进度回调
        self.on_segment_complete = None  # func(segment_index, speaker, chapter_title)
        self.on_chapter_complete = None  # func(chapter_index, chapter_title, output_path)
    
    async def generate(self, input_file: str,
                       chapter_range: Optional[tuple] = None,
                       skip_tts: bool = False) -> List[str]:
        """
        流水线生成有声书
        
        Args:
            input_file: 输入文件路径
            chapter_range: 章节范围 (start, end)
            skip_tts: 是否跳过 TTS 合成（仅生成剧本）
        
        Returns:
            生成的音频文件路径列表
        """
        start_time = time.time()
        self.stats["start_time"] = start_time
        
        logger.info("=" * 60)
        logger.info("🚀 AI有声书流水线模式启动")
        logger.info(f"输入文件: {input_file}")
        logger.info(f"音色模式: {self.config.tts.voice_mode}")
        logger.info(f"TTS并发数: {self.tts_concurrent}")
        logger.info(f"输出格式: {self.config.output.format}")
        logger.info("=" * 60)
        
        # ===== 第一步：解析文本 =====
        logger.info("\n📚 解析文本文件")
        book = parse_book(input_file)
        logger.info(f"书名: {book.title}, 作者: {book.author}, 章节数: {len(book.chapters)}")
        
        chapters = book.chapters
        if chapter_range:
            start, end = chapter_range
            chapters = [ch for ch in chapters if start <= ch.index <= end]
            logger.info(f"处理章节范围: {start}-{end}，共 {len(chapters)} 章")
        
        self.stats["total_chapters"] = len(chapters)
        
        # ===== 第二步：角色分析 =====
        logger.info("\n🎭 角色分析")
        analysis_text = book.raw_text[:15000] if len(book.raw_text) > 15000 else book.raw_text
        characters = await self.llm.analyze_characters(analysis_text)
        self._merge_voice_config(characters)
        
        # ===== 第三步：启动流水线 =====
        logger.info("\n🔧 启动流水线...")
        
        output_files = []
        
        if skip_tts:
            # 仅生成剧本模式
            logger.info("🚩 跳过TTS合成，仅生成剧本")
            script_queue = asyncio.Queue()
            skip_worker = asyncio.create_task(
                self._skip_tts_worker(chapters, characters, script_queue)
            )
            await skip_worker
            elapsed = time.time() - start_time
            logger.info(f"\n剧本生成完成！总耗时: {elapsed:.1f}秒")
            return output_files
        
        # 创建队列
        script_queue = asyncio.Queue(maxsize=self.queue_size)   # 剧本队列
        audio_queue = asyncio.Queue(maxsize=self.queue_size)     # 音频队列
        
        # 启动三个Worker
        llm_task = asyncio.create_task(
            self._llm_worker(chapters, characters, script_queue)
        )
        tts_task = asyncio.create_task(
            self._tts_worker(script_queue, audio_queue)
        )
        output_task = asyncio.create_task(
            self._output_worker(audio_queue, chapters, output_files)
        )
        
        # 等待所有Worker完成
        await asyncio.gather(llm_task, tts_task, output_task)
        
        # ===== 汇总 =====
        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 60)
        logger.info("🎉 流水线生成完成！")
        logger.info(f"总耗时: {elapsed:.1f}秒 ({elapsed / 60:.1f}分钟)")
        logger.info(f"处理片段: {self.stats['completed_segments']}/{self.stats['total_segments']}")
        logger.info(f"生成文件: {len(output_files)}")
        for f in output_files:
            logger.info(f"  📁 {f}")
        logger.info("=" * 60)
        
        return output_files
    
    # ============================================================
    # Worker 1: LLM剧本生成（生产者）
    # ============================================================
    
    async def _llm_worker(self, chapters, characters, queue: asyncio.Queue):
        """
        LLM Worker：逐章生成剧本，将片段推入队列
        
        当第一章的片段开始被TTS Worker消费时，
        LLM Worker已经在为第二章生成剧本。
        """
        try:
            for chapter in chapters:
                logger.info(f"\n📖 [LLM] 处理章节 [{chapter.index}] {chapter.title}")
                
                # 生成剧本
                segments = await self.llm.generate_script(
                    text=chapter.content,
                    characters=characters,
                )
                
                self.stats["total_segments"] += len(segments)
                
                # 保存剧本JSON
                script = ChapterScript(
                    chapter_index=chapter.index,
                    chapter_title=chapter.title,
                    characters=characters,
                    segments=segments,
                )
                safe_title = re.sub(r'[\\/*?:"<>|]', '_', chapter.title)
                script_file = os.path.join(
                    self.script_dir,
                    f"chapter_{chapter.index:03d}_{safe_title}.json"
                )
                self._save_script(script, script_file)
                
                # 构建角色查找表
                character_map = {c.name: c for c in characters}
                
                # 将每个片段推入队列
                for i, segment in enumerate(segments):
                    is_last = (i == len(segments) - 1)
                    item = PipelineItem(
                        chapter_index=chapter.index,
                        chapter_title=chapter.title,
                        segment=segment,
                        character=character_map.get(segment.speaker),
                        is_last_in_chapter=is_last,
                    )
                    await queue.put(item)
                    logger.info(f"[LLM] 片段 [{segment.index}] 已入队 (说话者={segment.speaker})")
                
                logger.info(f"[LLM] 章节 [{chapter.index}] {len(segments)} 个片段全部入队")
        
        except Exception as e:
            logger.error(f"[LLM Worker] 错误: {e}", exc_info=True)
        finally:
            # 发送终止信号
            await queue.put(None)
            logger.info("[LLM Worker] 已完成，发送终止信号")
    
    # ============================================================
    # Worker 2: TTS语音合成（中间处理者）
    # ============================================================
    
    async def _tts_worker(self, script_queue: asyncio.Queue,
                          audio_queue: asyncio.Queue):
        """
        TTS Worker：从剧本队列取出片段，合成语音，推入音频队列
        
        支持并发控制（Semaphore），多个TTS请求可同时进行。
        """
        try:
            while True:
                item = await script_queue.get()
                
                if item is None:
                    # 收到终止信号，传递并退出
                    await audio_queue.put(None)
                    break
                
                # 合成语音
                segment_dir = os.path.join(
                    self.output_dir, "segments",
                    f"chapter_{item.chapter_index:03d}"
                )
                output_path = os.path.join(segment_dir, f"seg_{item.segment.index:04d}.wav")
                
                audio_seg = await self.tts.synthesize_segment(
                    segment=item.segment,
                    character=item.character,
                    output_path=output_path,
                    chapter_index=item.chapter_index,
                )
                
                result = AudioResult(
                    chapter_index=item.chapter_index,
                    chapter_title=item.chapter_title,
                    segment_index=item.segment.index,
                    speaker=item.segment.speaker,
                    file_path=audio_seg.file_path,
                    is_last_in_chapter=item.is_last_in_chapter,
                )
                
                await audio_queue.put(result)
                script_queue.task_done()
        
        except Exception as e:
            logger.error(f"[TTS Worker] 错误: {e}", exc_info=True)
            await audio_queue.put(None)
    
    # ============================================================
    # Worker 3: 音频增量写入（消费者）
    # ============================================================
    
    async def _output_worker(self, audio_queue: asyncio.Queue,
                              chapters, output_files: List[str]):
        """
        Output Worker：从音频队列取出片段，增量追加写入WAV文件
        
        每个章节维护一个StreamWavWriter，边收边写。
        章节结束时关闭Writer并转换格式。
        """
        try:
            current_writer = None
            current_chapter_idx = -1
            current_output_path = ""
            current_ext = self.config.output.format
            segment_count_in_chapter = 0
            
            while True:
                result = await audio_queue.get()
                
                if result is None:
                    # 收到终止信号，关闭当前Writer
                    if current_writer:
                        current_writer.close()
                        final_path = self.audio_processor.finalize_stream(
                            current_output_path, current_output_path
                        )
                        if final_path and final_path not in output_files:
                            output_files.append(final_path)
                        self.stats["completed_chapters"] += 1
                        if self.on_chapter_complete:
                            self.on_chapter_complete(
                                current_chapter_idx, "", final_path
                            )
                    break
                
                # 如果是新章节，关闭上一个Writer，创建新的
                if result.chapter_index != current_chapter_idx:
                    # 关闭上一章
                    if current_writer:
                        current_writer.close()
                        final_path = self.audio_processor.finalize_stream(
                            current_output_path, current_output_path
                        )
                        if final_path and final_path not in output_files:
                            output_files.append(final_path)
                        self.stats["completed_chapters"] += 1
                        if self.on_chapter_complete:
                            self.on_chapter_complete(
                                current_chapter_idx, "", final_path
                            )
                        # 清理分段文件
                        logger.info(f"[Output] 章节 [{current_chapter_idx}] 音频写入完成")
                    
                    # 创建新章节Writer
                    current_chapter_idx = result.chapter_index
                    ext = current_ext
                    safe_chapter_title = re.sub(r'[\\/*?:"<>|]', '_', result.chapter_title)
                    output_filename = f"chapter_{result.chapter_index:03d}_{safe_chapter_title}.{ext}"
                    current_output_path = os.path.join(self.output_dir, output_filename)
                    
                    current_writer = self.audio_processor.create_stream_writer(
                        current_output_path
                    )
                    segment_count_in_chapter = 0
                    
                    logger.info(f"[Output] 开始写入章节 [{result.chapter_index}] {result.chapter_title}")
                
                # 追加音频数据
                current_writer.append_file(result.file_path)
                segment_count_in_chapter += 1
                self.stats["completed_segments"] += 1
                
                # 片段间插入静音（非最后一个片段）
                if not result.is_last_in_chapter and self.config.tts.silence_between_paragraphs > 0:
                    current_writer.append_silence(self.config.tts.silence_between_paragraphs)
                
                # 进度回调
                if self.on_segment_complete:
                    self.on_segment_complete(
                        result.segment_index, result.speaker, result.chapter_title
                    )
                
                elapsed = time.time() - self.stats["start_time"]
                logger.info(
                    f"[Output] ✓ 片段 [{result.segment_index}] 已写入 "
                    f"(累计: {self.stats['completed_segments']}段, "
                    f"耗时: {elapsed:.1f}s, "
                    f"当前时长: {current_writer.duration:.1f}s)"
                )
                
                audio_queue.task_done()
        
        except Exception as e:
            logger.error(f"[Output Worker] 错误: {e}", exc_info=True)
    
    async def _skip_tts_worker(self, chapters, characters, queue: asyncio.Queue):
        """仅生成剧本的Worker：逐章生成JSON剧本文件，跳过TTS"""
        try:
            for chapter in chapters:
                logger.info(f"\n📖 [纯剧本] 处理章节 [{chapter.index}] {chapter.title}")
                script = await self.llm.generate_script(
                    chapter_index=chapter.index,
                    chapter_title=chapter.title,
                    text=chapter.content,
                    characters=characters,
                )
                safe_title = re.sub(r'[\\/*?:"<>|]', '_', chapter.title)
                script_file = os.path.join(
                    self.output_dir, "scripts",
                    f"chapter_{chapter.index:03d}_{safe_title}.json"
                )
                os.makedirs(os.path.dirname(script_file), exist_ok=True)
                self._save_script(script, script_file)
                logger.info(f"✅ 剧本已保存: {script_file}")
                self.stats["total_segments"] = (
                    self.stats.get("total_segments", 0) + len(script.segments)
                )
        except Exception as e:
            logger.error(f"[纯剧本Worker] 出错: {e}", exc_info=True)
        finally:
            queue.put_nowait(None)  # 发送终止信号
    
    # ============================================================
    # 辅助方法
    # ============================================================
    
    def _merge_voice_config(self, characters: List[Character]):
        """将配置文件中的音色信息合并到角色对象"""
        voice_mode = self.config.tts.voice_mode
        voices = self.config.voices
        
        if voice_mode == "preset":
            preset_map = voices.get("preset", {}).get("voice_map", {})
            for char in characters:
                if char.name in preset_map:
                    char.preset_voice = preset_map[char.name]
        elif voice_mode == "voicedesign":
            desc_map = voices.get("voicedesign", {}).get("voice_descriptions", {})
            for char in characters:
                if char.name in desc_map:
                    char.voice_description = desc_map[char.name]
        elif voice_mode == "voiceclone":
            sample_map = voices.get("voiceclone", {}).get("voice_samples", {})
            for char in characters:
                if char.name in sample_map:
                    char.voice_sample = sample_map[char.name]
    
    def _save_script(self, script: ChapterScript, file_path: str):
        """保存剧本文本为JSON"""
        data = {
            "chapter_index": script.chapter_index,
            "chapter_title": script.chapter_title,
            "characters": [
                {"name": c.name, "description": c.description, "voice_description": c.voice_description}
                for c in script.characters
            ],
            "segments": [
                {
                    "index": s.index, "speaker": s.speaker, "text": s.text,
                    "emotion": s.emotion, "style_instruction": s.style_instruction,
                }
                for s in script.segments
            ],
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
