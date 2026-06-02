"""
音频后处理模块
负责音频片段拼接、静音插入、格式转换
新增：增量追加写入（支持流水线边合成边写入）
"""

import os
import struct
import wave
import logging
from typing import List, Optional

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
            if sr != self.sample_rate:
                logger.warning(f"音频采样率 {sr}Hz 与目标 {self.sample_rate}Hz 不匹配，进行重采样")
                data = self._resample(data, sr, self.sample_rate)
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
            duration = len(data) / orig_sr
            target_length = int(duration * target_sr)
            indices = np.linspace(0, len(data) - 1, target_length)
            return np.interp(indices, np.arange(len(data)), data)
    
    def concat_segments(self, audio_segments: List[AudioSegment],
                         output_path: str) -> str:
        """拼接多个音频片段为一个完整音频文件"""
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
            if i < len(audio_segments) - 1 and self.silence_duration > 0:
                all_audio.append(silence)
        
        if not all_audio:
            logger.error("所有音频片段加载失败")
            return ""
        
        combined = np.concatenate(all_audio)
        duration = len(combined) / self.sample_rate
        logger.info(f"合并完成，总时长: {duration:.1f}秒 ({duration / 60:.1f}分钟)")
        
        wav_path = output_path
        if output_path.endswith('.mp3'):
            wav_path = output_path.rsplit('.', 1)[0] + '.wav'
        
        sf.write(wav_path, combined, self.sample_rate)
        logger.info(f"WAV文件保存至: {wav_path}")
        
        if self.output_format == "mp3" and output_path.endswith('.mp3'):
            self._convert_to_mp3(wav_path, output_path)
            if os.path.exists(wav_path):
                os.remove(wav_path)
            return output_path
        
        return wav_path
    
    # ============================================================
    # 增量追加写入（流水线模式核心）
    # ============================================================
    
    class StreamWavWriter:
        """
        WAV流式写入器
        支持边合成边追加写入，无需等所有片段就绪。
        
        使用方式：
            writer = StreamWavWriter("output.wav", sample_rate=24000)
            writer.append(audio_data_np)  # 追加一个片段
            writer.append_silence(1.0)    # 追加1秒静音
            writer.close()                # 完成写入
        """
        
        def __init__(self, output_path: str, sample_rate: int = 24000):
            self.output_path = output_path
            self.sample_rate = sample_rate
            self._total_samples = 0
            self._temp_path = output_path + ".tmp"
            
            # 创建临时WAV文件
            self._wf = wave.open(self._temp_path, 'wb')
            self._wf.setnchannels(1)       # 单声道
            self._wf.setsampwidth(2)       # 16-bit
            self._wf.setframerate(sample_rate)
        
        def append(self, audio_data: np.ndarray):
            """
            追加音频数据（float32 numpy数组）
            
            Args:
                audio_data: float32格式的音频数据
            """
            # float32 -> int16
            pcm_data = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
            self._wf.writeframes(pcm_data.tobytes())
            self._total_samples += len(pcm_data)
        
        def append_silence(self, duration: float):
            """追加静音"""
            num_samples = int(self.sample_rate * duration)
            silence = np.zeros(num_samples, dtype=np.float32)
            self.append(silence)
        
        def append_file(self, file_path: str):
            """从WAV文件追加音频数据"""
            data, sr = sf.read(file_path, dtype='float32')
            if sr != self.sample_rate:
                # 简单重采样
                duration = len(data) / sr
                target_length = int(duration * self.sample_rate)
                indices = np.linspace(0, len(data) - 1, target_length)
                data = np.interp(indices, np.arange(len(data)), data)
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            self.append(data)
        
        @property
        def duration(self) -> float:
            """当前已写入的音频总时长（秒）"""
            return self._total_samples / self.sample_rate
        
        def close(self):
            """关闭文件，更新WAV头"""
            self._wf.close()
            # 重命名为最终文件名
            os.rename(self._temp_path, self.output_path)
            logger.info(f"流式WAV写入完成: {self.output_path} (时长: {self.duration:.1f}秒)")
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            self.close()
    
    def create_stream_writer(self, output_path: str) -> 'AudioPostProcessor.StreamWavWriter':
        """
        创建流式WAV写入器
        
        Args:
            output_path: 输出WAV文件路径
        
        Returns:
            StreamWavWriter 实例
        """
        wav_path = output_path
        if output_path.endswith('.mp3'):
            wav_path = output_path.rsplit('.', 1)[0] + '.wav'
        
        os.makedirs(os.path.dirname(wav_path) if os.path.dirname(wav_path) else ".", exist_ok=True)
        return self.StreamWavWriter(wav_path, self.sample_rate)
    
    def finalize_stream(self, wav_path: str, final_path: str):
        """
        流式写入完成后，将WAV转换为最终格式（如MP3）
        
        Args:
            wav_path: 流式写入的WAV文件路径
            final_path: 最终输出文件路径
        """
        if self.output_format == "mp3" and final_path.endswith('.mp3'):
            self._convert_to_mp3(wav_path, final_path)
            if os.path.exists(wav_path):
                os.remove(wav_path)
            return final_path
        return wav_path
    
    # ============================================================
    # MP3转换（原有功能）
    # ============================================================
    
    def _convert_to_mp3(self, wav_path: str, mp3_path: str):
        """将WAV转换为MP3"""
        try:
            import subprocess
            cmd = [
                "ffmpeg", "-y", "-i", wav_path,
                "-codec:a", "libmp3lame", "-b:a", self.mp3_bitrate,
                "-ar", str(self.sample_rate), "-ac", "1", mp3_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                logger.info(f"MP3文件保存至: {mp3_path}")
            else:
                logger.error(f"ffmpeg转换失败: {result.stderr}")
                self._convert_to_mp3_pydub(wav_path, mp3_path)
        except FileNotFoundError:
            logger.warning("ffmpeg未安装，尝试使用pydub转换")
            self._convert_to_mp3_pydub(wav_path, mp3_path)
        except Exception as e:
            logger.error(f"MP3转换失败: {e}")
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
            raise ImportError("MP3转换需要安装 ffmpeg 或 pydub:\n  sudo apt install ffmpeg\n  或: pip install pydub")
    
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
