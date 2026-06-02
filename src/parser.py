"""
文本解析模块
支持解析 TXT、EPUB、PDF 格式的电子书文本
"""

import os
import re
import logging
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Chapter:
    """章节"""
    index: int
    title: str
    content: str  # 原始文本内容


@dataclass
class ParsedBook:
    """解析后的书籍"""
    title: str
    author: str
    chapters: List[Chapter]
    raw_text: str = ""  # 完整原始文本


class TextParser:
    """文本解析器基类"""
    
    def parse(self, file_path: str) -> ParsedBook:
        raise NotImplementedError
    
    @staticmethod
    def detect_format(file_path: str) -> str:
        """根据文件扩展名检测格式"""
        ext = os.path.splitext(file_path)[1].lower()
        format_map = {
            ".txt": "txt",
            ".epub": "epub",
            ".pdf": "pdf",
            ".md": "txt",  # Markdown当作纯文本处理
        }
        return format_map.get(ext, "txt")


class TxtParser(TextParser):
    """纯文本解析器"""
    
    def parse(self, file_path: str) -> ParsedBook:
        logger.info(f"解析TXT文件: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # 清理文本
        text = self._clean_text(text)
        
        # 尝试检测章节
        chapters = self._split_chapters(text)
        
        return ParsedBook(
            title=os.path.splitext(os.path.basename(file_path))[0],
            author="未知",
            chapters=chapters,
            raw_text=text,
        )
    
    def _clean_text(self, text: str) -> str:
        """清理文本中的多余空白"""
        # 合并连续空行为单个空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 去除行首尾空白
        lines = [line.strip() for line in text.split('\n')]
        return '\n'.join(lines)
    
    def _split_chapters(self, text: str) -> List[Chapter]:
        """尝试按章节标题分割文本"""
        # 常见章节标题模式
        chapter_patterns = [
            r'^第[一二三四五六七八九十百千零\d]+[章节回卷集部篇]\s*.+$',
            r'^Chapter\s+\d+.*$',
            r'^CHAPTER\s+\d+.*$',
            r'^\d+[\.、]\s*.+$',  # "1. 标题" 或 "1、标题"
        ]
        
        combined_pattern = '|'.join(f'({p})' for p in chapter_patterns)
        
        lines = text.split('\n')
        chapters = []
        current_lines = []
        current_title = "序"
        chapter_index = 0
        
        for line in lines:
            if re.match(combined_pattern, line.strip(), re.IGNORECASE):
                # 保存当前章节
                if current_lines:
                    content = '\n'.join(current_lines).strip()
                    if content:
                        chapters.append(Chapter(
                            index=chapter_index,
                            title=current_title,
                            content=content,
                        ))
                        chapter_index += 1
                
                current_title = line.strip()
                current_lines = []
            else:
                current_lines.append(line)
        
        # 保存最后一章
        if current_lines:
            content = '\n'.join(current_lines).strip()
            if content:
                chapters.append(Chapter(
                    index=chapter_index,
                    title=current_title,
                    content=content,
                ))
        
        # 如果没有检测到章节，将整个文本作为一章
        if not chapters and text.strip():
            chapters.append(Chapter(
                index=0,
                title="全文",
                content=text.strip(),
            ))
        
        logger.info(f"共检测到 {len(chapters)} 个章节")
        return chapters


class EpubParser(TextParser):
    """EPUB电子书解析器"""
    
    def parse(self, file_path: str) -> ParsedBook:
        try:
            import ebooklib
            from ebooklib import epub
            from html.parser import HTMLParser
        except ImportError:
            raise ImportError(
                "解析EPUB需要安装 ebooklib 库:\n"
                "  pip install ebooklib"
            )
        
        logger.info(f"解析EPUB文件: {file_path}")
        
        book = epub.read_epub(file_path)
        
        # 提取元数据
        title = book.get_metadata('DC', 'title')
        title = title[0][0] if title else os.path.splitext(os.path.basename(file_path))[0]
        
        author = book.get_metadata('DC', 'creator')
        author = author[0][0] if author else "未知"
        
        # 提取文本内容
        all_text_parts = []
        chapters = []
        chapter_index = 0
        
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            content = item.get_content().decode('utf-8', errors='ignore')
            text = self._html_to_text(content)
            text = text.strip()
            
            if text:
                # 尝试从HTML中提取章节标题
                chapter_title = self._extract_title_from_html(content)
                
                chapters.append(Chapter(
                    index=chapter_index,
                    title=chapter_title or f"第{chapter_index + 1}章",
                    content=text,
                ))
                chapter_index += 1
                all_text_parts.append(text)
        
        # 合并过短的章节
        chapters = self._merge_short_chapters(chapters, min_chars=200)
        
        raw_text = '\n\n'.join(all_text_parts)
        
        return ParsedBook(
            title=title,
            author=author,
            chapters=chapters,
            raw_text=raw_text,
        )
    
    def _html_to_text(self, html_content: str) -> str:
        """将HTML转换为纯文本"""
        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text_parts = []
                self.skip = False
            
            def handle_starttag(self, tag, attrs):
                if tag in ('script', 'style'):
                    self.skip = True
                elif tag in ('p', 'div', 'br', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'):
                    self.text_parts.append('\n')
            
            def handle_endtag(self, tag):
                if tag in ('script', 'style'):
                    self.skip = False
                elif tag in ('p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
                    self.text_parts.append('\n')
            
            def handle_data(self, data):
                if not self.skip:
                    self.text_parts.append(data)
            
            def get_text(self):
                return ''.join(self.text_parts)
        
        extractor = TextExtractor()
        extractor.feed(html_content)
        text = extractor.get_text()
        
        # 清理多余空白
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def _extract_title_from_html(self, html_content: str) -> str:
        """从HTML中提取标题"""
        # 简单匹配 <h1>, <h2>, <h3> 标签
        match = re.search(r'<h[1-3][^>]*>(.*?)</h[1-3]>', html_content, re.DOTALL | re.IGNORECASE)
        if match:
            title = re.sub(r'<[^>]+>', '', match.group(1)).strip()
            if title and len(title) < 100:
                return title
        return ""
    
    def _merge_short_chapters(self, chapters: List[Chapter], min_chars: int = 200) -> List[Chapter]:
        """合并过短的章节"""
        if len(chapters) <= 1:
            return chapters
        
        merged = []
        buffer_lines = []
        buffer_title = chapters[0].title
        buffer_index = 0
        
        for ch in chapters:
            if len(ch.content) < min_chars and merged or buffer_lines:
                buffer_lines.append(ch.content)
            else:
                if buffer_lines:
                    merged.append(Chapter(
                        index=buffer_index,
                        title=buffer_title,
                        content='\n\n'.join(buffer_lines),
                    ))
                buffer_lines = [ch.content]
                buffer_title = ch.title
                buffer_index = len(merged)
        
        if buffer_lines:
            merged.append(Chapter(
                index=buffer_index,
                title=buffer_title,
                content='\n\n'.join(buffer_lines),
            ))
        
        return merged


class PdfParser(TextParser):
    """PDF文档解析器"""
    
    def parse(self, file_path: str) -> ParsedBook:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            try:
                import PyPDF2
                return self._parse_with_pypdf2(file_path, PyPDF2)
            except ImportError:
                raise ImportError(
                    "解析PDF需要安装 PyMuPDF 或 PyPDF2 库:\n"
                    "  pip install PyMuPDF\n"
                    "  或: pip install PyPDF2"
                )
        
        logger.info(f"解析PDF文件: {file_path}")
        
        doc = fitz.open(file_path)
        
        # 提取元数据
        title = doc.metadata.get("title", "") or os.path.splitext(os.path.basename(file_path))[0]
        author = doc.metadata.get("author", "未知")
        
        all_text_parts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            text = text.strip()
            if text:
                all_text_parts.append(text)
        
        doc.close()
        
        raw_text = '\n\n'.join(all_text_parts)
        raw_text = re.sub(r'\n{3,}', '\n\n', raw_text)
        
        # 尝试分章
        chapters = self._split_chapters(raw_text)
        
        return ParsedBook(
            title=title,
            author=author,
            chapters=chapters,
            raw_text=raw_text,
        )
    
    def _parse_with_pypdf2(self, file_path: str, PyPDF2) -> ParsedBook:
        """使用PyPDF2作为备选解析器"""
        logger.info(f"解析PDF文件(PyPDF2): {file_path}")
        
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            
            title = (reader.metadata.title if reader.metadata else None) or \
                    os.path.splitext(os.path.basename(file_path))[0]
            author = (reader.metadata.author if reader.metadata else None) or "未知"
            
            all_text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text and text.strip():
                    all_text_parts.append(text.strip())
        
        raw_text = '\n\n'.join(all_text_parts)
        raw_text = re.sub(r'\n{3,}', '\n\n', raw_text)
        
        chapters = self._split_chapters(raw_text)
        
        return ParsedBook(
            title=title,
            author=author,
            chapters=chapters,
            raw_text=raw_text,
        )
    
    def _split_chapters(self, text: str) -> List[Chapter]:
        """PDF章节分割（复用TxtParser的逻辑）"""
        parser = TxtParser()
        return parser._split_chapters(text)


def parse_book(file_path: str) -> ParsedBook:
    """
    自动检测格式并解析电子书
    
    Args:
        file_path: 电子书文件路径
    
    Returns:
        ParsedBook 解析后的书籍对象
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    file_format = TextParser.detect_format(file_path)
    logger.info(f"检测到文件格式: {file_format}")
    
    parsers = {
        "txt": TxtParser,
        "epub": EpubParser,
        "pdf": PdfParser,
    }
    
    parser_class = parsers.get(file_format, TxtParser)
    parser = parser_class()
    
    return parser.parse(file_path)
