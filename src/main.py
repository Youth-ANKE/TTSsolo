"""
AI有声书 - 主流程编排模块
串联文本解析 → LLM处理 → TTS合成 → 音频拼接 的完整流程
支持两种模式：
  - 串行模式（默认）：逐章依次处理，稳定可靠
  - 流水线模式（--pipeline）：异步并行处理，速度更快
"""

import re
import os
import sys
import json
import time
import asyncio
import logging
import argparse
from datetime import datetime
from typing import List, Optional

from .config import load_config, AppConfig
from .parser import parse_book, ParsedBook
from .llm_processor import LLMProcessor, ChapterScript, Character
from .tts_engine import TTSEngine, AudioSegment
from .audio_processor import AudioPostProcessor
from .pipeline import AudiobookPipeline


def setup_logging(log_file: str = ""):
    """配置日志"""
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
    )


class AudiobookGenerator:
    """AI有声书生成器（串行模式）"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.llm = LLMProcessor(config)
        self.tts = TTSEngine(config)
        self.audio_processor = AudioPostProcessor(config)
        
        self.output_dir = os.path.join(config.project_root, config.output.dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.script_dir = os.path.join(self.output_dir, "scripts")
        os.makedirs(self.script_dir, exist_ok=True)
    
    def generate(self, input_file: str, 
                 chapter_range: Optional[tuple] = None,
                 skip_tts: bool = False) -> List[str]:
        """生成完整有声书（串行模式）"""
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("AI有声书生成器启动（串行模式）")
        logger.info(f"输入文件: {input_file}")
        logger.info(f"音色模式: {self.config.tts.voice_mode}")
        logger.info(f"输出格式: {self.config.output.format}")
        logger.info("=" * 60)
        
        # 解析文本
        logger.info("\n📚 第一步：解析文本文件")
        book = parse_book(input_file)
        logger.info(f"书名: {book.title}")
        logger.info(f"作者: {book.author}")
        logger.info(f"章节数: {len(book.chapters)}")
        
        chapters = book.chapters
        if chapter_range:
            start, end = chapter_range
            chapters = [ch for ch in chapters if start <= ch.index <= end]
            logger.info(f"处理章节范围: {start}-{end}，共 {len(chapters)} 章")
        
        # 角色分析
        logger.info("\n🎭 第二步：角色分析")
        analysis_text = book.raw_text[:15000] if len(book.raw_text) > 15000 else book.raw_text
        characters = self.llm.analyze_characters(analysis_text)
        self._merge_voice_config(characters)
        
        # 逐章处理
        output_files = []
        for chapter in chapters:
            logger.info(f"\n📖 处理章节 [{chapter.index}] {chapter.title}")
            logger.info("-" * 40)
            
            script = self.llm.process_chapter(
                chapter_index=chapter.index,
                chapter_title=chapter.title,
                text=chapter.content,
                characters=characters,
            )
            
            # 生成安全的文件名（替换非法字符）
            safe_title = re.sub(r'[\\/*?:"<>|]', '_', chapter.title)
            script_file = os.path.join(
                self.script_dir,
                f"chapter_{chapter.index:03d}_{safe_title}.json"
            )
            self._save_script(script, script_file)
            logger.info(f"剧本已保存: {script_file}")
            
            if skip_tts:
                logger.info("跳过TTS合成（skip_tts=True）")
                continue
            
            audio_segments = self.tts.synthesize_chapter(script)
            
            ext = self.config.output.format
            output_filename = f"chapter_{chapter.index:03d}_{safe_title}.{ext}"
            output_path = os.path.join(self.output_dir, output_filename)
            
            final_path = self.audio_processor.concat_segments(
                audio_segments, output_path
            )
            
            if final_path:
                output_files.append(final_path)
                logger.info(f"✅ 章节 [{chapter.index}] 生成完成: {final_path}")
            
            self.audio_processor.cleanup_segments(audio_segments)
        
        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 60)
        logger.info("🎉 有声书生成完成！")
        logger.info(f"总耗时: {elapsed:.1f}秒 ({elapsed / 60:.1f}分钟)")
        logger.info(f"生成文件数: {len(output_files)}")
        for f in output_files:
            logger.info(f"  📁 {f}")
        logger.info("=" * 60)
        
        return output_files
    
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
                {"index": s.index, "speaker": s.speaker, "text": s.text,
                 "emotion": s.emotion, "style_instruction": s.style_instruction}
                for s in script.segments
            ],
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


async def run_pipeline(config: AppConfig, input_file: str,
                       chapter_range: Optional[tuple] = None,
                       tts_concurrent: int = 3,
                       script_only: bool = False) -> List[str]:
    """
    流水线模式入口（异步）
    
    Args:
        config: 应用配置
        input_file: 输入文件路径
        chapter_range: 章节范围
        tts_concurrent: TTS最大并发数
        script_only: 仅生成剧本
    """
    pipeline = AudiobookPipeline(
        config=config,
        tts_concurrent=tts_concurrent,
    )
    
    return await pipeline.generate(
        input_file=input_file,
        chapter_range=chapter_range,
        skip_tts=script_only,
    )


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="AI有声书生成器 - 使用DeepSeek LLM + 小米MiMo TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 串行模式生成（默认）
  python -m src input.txt
  
  # 流水线模式（更快，推荐）
  python -m src input.txt --pipeline
  
  # 流水线模式 + 自定义TTS并发数
  python -m src input.txt --pipeline --tts-concurrent 5
  
  # 指定配置文件
  python -m src input.txt --config my_config.yaml
  
  # 只处理第1-5章
  python -m src input.txt --chapters 1 5
  
  # 仅生成剧本（不合成语音）
  python -m src input.txt --script-only
  
  # 使用预置音色模式
  python -m src input.txt --voice-mode preset
        """
    )
    
    parser.add_argument("input", help="输入文件路径（支持TXT/EPUB/PDF）")
    parser.add_argument("--config", "-c", default="config.yaml",
                        help="配置文件路径（默认: config.yaml）")
    parser.add_argument("--chapters", "-ch", nargs=2, type=int, metavar=("START", "END"),
                        help="章节范围（如: --chapters 1 5）")
    parser.add_argument("--voice-mode", "-vm", choices=["preset", "voicedesign", "voiceclone"],
                        help="覆盖音色模式")
    parser.add_argument("--output-format", "-of", choices=["wav", "mp3"],
                        help="覆盖输出格式")
    parser.add_argument("--script-only", "-s", action="store_true",
                        help="仅生成剧本，不合成语音")
    parser.add_argument("--title", "-t", default="",
                        help="有声书标题（覆盖配置）")
    parser.add_argument("--author", "-a", default="",
                        help="有声书作者（覆盖配置）")
    parser.add_argument("--log-file", default="",
                        help="日志文件路径")
    
    # 流水线模式参数
    parser.add_argument("--pipeline", "-p", action="store_true",
                        help="启用流水线模式（异步并行，速度更快）")
    parser.add_argument("--tts-concurrent", "-tc", type=int, default=3,
                        help="TTS最大并发数（流水线模式，默认: 3）")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_file)
    
    # 加载配置
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        logging.error(str(e))
        sys.exit(1)
    
    # 命令行参数覆盖配置
    if args.voice_mode:
        config.tts.voice_mode = args.voice_mode
    if args.output_format:
        config.output.format = args.output_format
    if args.title:
        config.metadata.title = args.title
    if args.author:
        config.metadata.author = args.author
    
    # 验证API Key
    if not config.deepseek.api_key or config.deepseek.api_key == "YOUR_DEEPSEEK_API_KEY":
        logging.error("请设置DeepSeek API Key（配置文件或DEEPSEEK_API_KEY环境变量）")
        sys.exit(1)
    if not config.mimo.api_key or config.mimo.api_key == "YOUR_MIMO_API_KEY":
        logging.error("请设置小米MiMo API Key（配置文件或MIMO_API_KEY环境变量）")
        sys.exit(1)
    
    chapter_range = None
    if args.chapters:
        # 用户输入为1-based，内部使用0-based
        chapter_range = (args.chapters[0] - 1, args.chapters[1] - 1)
    
    try:
        if args.pipeline:
            # ===== 流水线模式 =====
            logger.info("🚀 使用流水线模式（异步并行）")
            output_files = asyncio.run(run_pipeline(
                config=config,
                input_file=args.input,
                chapter_range=chapter_range,
                tts_concurrent=args.tts_concurrent,
                script_only=args.script_only,
            ))
        else:
            # ===== 串行模式 =====
            generator = AudiobookGenerator(config)
            output_files = generator.generate(
                input_file=args.input,
                chapter_range=chapter_range,
                skip_tts=args.script_only,
            )
        
        if output_files:
            print(f"\n✅ 生成完成！共 {len(output_files)} 个音频文件")
        elif args.script_only:
            print(f"\n✅ 剧本生成完成！查看 {config.output.dir}/scripts/ 目录")
        else:
            print("\n⚠️ 没有生成任何音频文件")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断，正在退出...")
        sys.exit(1)
    except Exception as e:
        logging.error(f"生成失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
