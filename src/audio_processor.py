"""
音频后处理模块
负责音频片段拼接、静音插入、格式转换
"""

import os
import logging
from typing import List

import numpy as np
import soundfile as sf

from .config import AppConfig
from .tts_engine import AudioSegment

logger = logging.getLogger(__name__)


class AudioPostProcessor:
    """音频后处理器"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.sample_rate = config.output.sample_rate  # 24000 Hz
        self.silence_duration = config.tts.silence_between_paragraphs
        self.output_format = config.output.format
        self.mp3_bitrate = config.output.mp3_bitrate
        self.keep_segments = config.output.keep_segments
    
    def _generate_silence(self, duration: float) -> np.ndarray:
        """生成静音音频数据"""
        num_samples = int(self.sample_rate * duration)
        return np.zeros(num_samples, dtype=np.float32)
    
    def _load_audio(self, file_path: str) -> np.ndarray:
        """加载音频文件为float32 numpy数组"""
        try:
            data, sr = sf.read(file_path, dtype='float32')
            
            # 如果采样率不匹配，进行重采样
            if sr != self.sample_rate:
                logger.warning(f"音频采样率 {sr}Hz 与目标 {self.sample_rate}Hz 不匹配，进行重采样")
                data = self._resample(data, sr, self.sample_rate)
            
            # 如果是立体声，转为单声道
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            
            return data
        except Exception as e:
            logger.error(f"加载音频失败 {file_path}: {e}")
            return np.array([], dtype=np.float32)
    
    def _resample(self, data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """简单线性插值重采样"""
        try:
            import librosa
            return librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # 简单的线性插值重采样
            duration = len(data) / orig_sr
            target_length = int(duration * target_sr)
            indices = np.linspace(0, len(data) - 1, target_length)
            return np.interp(indices, np.arange(len(data)), data)
    
    def concat_segments(self, audio_segments: List[AudioSegment],
                         output_path: str) -> str:
        """
        拼接多个音频片段为一个完整音频文件
        
        Args:
            audio_segments: 音频片段列表
            output_path: 输出文件路径
        
        Returns:
            输出文件路径
        """
        if not audio_segments:
            logger.warning("没有音频片段需要拼接")
            return ""
        
        logger.info(f"拼接 {len(audio_segments)} 个音频片段...")
        
        all_audio = []
        silence = self._generate_silence(self.silence_duration)
        
        for i, seg in enumerate(audio_segments):
            audio_data = self._load_audio(seg.file_path)
            
            if len(audio_data) == 0:
                logger.warning(f"片段 [{seg.index}] 加载失败，跳过")
                continue
            
            all_audio.append(audio_data)
            
            # 在片段之间插入静音（最后一个片段除外）
            if i < len(audio_segments) - 1 and self.silence_duration > 0:
                all_audio.append(silence)
        
        if not all_audio:
            logger.error("所有音频片段加载失败")
            return ""
        
        # 合并所有音频
        combined = np.concatenate(all_audio)
        duration = len(combined) / self.sample_rate
        logger.info(f"合并完成，总时长: {duration:.1f}秒 ({duration / 60:.1f}分钟)")
        
        # 先保存为WAV
        if output_path.endswith('.mp3'):
            wav_path = output_path.rsplit('.', 1)[0] + '.wav'
        else:
            wav_path = output_path
        
        sf.write(wav_path, combined, self.sample_rate)
        logger.info(f"WAV文件保存至: {wav_path}")
        
        # 如果需要MP3格式，进行转换
        if self.output_format == "mp3" and output_path.endswith('.mp3'):
            self._convert_to_mp3(wav_path, output_path)
            # 删除中间WAV文件
            if os.path.exists(wav_path):
                os.remove(wav_path)
            return output_path
        
        return wav_path
    
    def _convert_to_mp3(self, wav_path: str, mp3_path: str):
        """将WAV转换为MP3"""
        try:
            import subprocess
            
            # 尝试使用ffmpeg
            cmd = [
                "ffmpeg", "-y",
                "-i", wav_path,
                "-codec:a", "libmp3lame",
                "-b:a", self.mp3_bitrate,
                "-ar", str(self.sample_rate),
                "-ac", "1",
                mp3_path,
            ]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            
            if result.returncode == 0:
                logger.info(f"MP3文件保存至: {mp3_path}")
            else:
                logger.error(f"ffmpeg转换失败: {result.stderr}")
                # 回退到pydub
                self._convert_to_mp3_pydub(wav_path, mp3_path)
        
        except FileNotFoundError:
            logger.warning("ffmpeg未安装，尝试使用pydub转换")
            self._convert_to_mp3_pydub(wav_path, mp3_path)
        except Exception as e:
            logger.error(f"MP3转换失败: {e}")
            # 保留WAV文件作为回退
            import shutil
            shutil.copy2(wav_path, mp3_path.replace('.mp3', '.wav'))
            logger.warning(f"转换失败，已保存为WAV格式: {mp3_path.replace('.mp3', '.wav')}")
    
    def _convert_to_mp3_pydub(self, wav_path: str, mp3_path: str):
        """使用pydub进行MP3转换（备选方案）"""
        try:
            from pydub import AudioSegment as PydubSegment
            
            audio = PydubSegment.from_wav(wav_path)
            audio.export(mp3_path, format="mp3", bitrate=self.mp3_bitrate)
            logger.info(f"MP3文件保存至(pydub): {mp3_path}")
        
        except ImportError:
            raise ImportError(
                "MP3转换需要安装 ffmpeg 或 pydub:\n"
                "  sudo apt install ffmpeg\n"
                "  或: pip install pydub"
            )
    
    def cleanup_segments(self, audio_segments: List[AudioSegment]):
        """清理分段音频文件"""
        if self.keep_segments:
            logger.info("保留分段音频文件（keep_segments=True）")
            return
        
        for seg in audio_segments:
            try:
                if os.path.exists(seg.file_path):
                    os.remove(seg.file_path)
                    logger.debug(f"已删除分段文件: {seg.file_path}")
            except Exception as e:
                logger.warning(f"删除分段文件失败 {seg.file_path}: {e}")
        
        logger.info("分段音频文件清理完成")
