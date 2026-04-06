"""Content Exporter - Export content to various formats"""

import os
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class ExportConfig:
    output_dir: str = "./output"
    file_format: str = "md"
    include_metadata: bool = True
    include_toc: bool = False
    date_format: str = "%Y-%m-%d"


class ContentExporter:
    """Export content to various formats"""
    
    SUPPORTED_FORMATS = ["md", "html", "txt", "json", "pdf"]
    
    def __init__(self, config: Optional[ExportConfig] = None):
        self.config = config or ExportConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def export_markdown(
        self,
        content: str,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Export content as Markdown"""
        
        output = []
        
        if self.config.include_metadata and metadata:
            output.append("---")
            for key, value in metadata.items():
                output.append(f"{key}: {value}")
            output.append("---")
            output.append("")
        
        output.append(content)
        
        filepath = self._get_filepath(filename, "md")
        
        with open(filepath, "w") as f:
            f.write("\n".join(output))
        
        return str(filepath)
    
    def export_html(
        self,
        content: str,
        filename: str,
        title: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        styles: Optional[str] = None,
    ) -> str:
        """Export content as HTML"""
        
        default_styles = """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
            color: #333;
        }
        h1, h2, h3 { color: #2c3e50; }
        code { background: #f4f4f4; padding: 0.2em 0.4em; border-radius: 3px; }
        pre { background: #f4f4f4; padding: 1em; overflow-x: auto; }
        blockquote { border-left: 4px solid #ddd; margin: 0; padding-left: 1em; color: #666; }
        """
        
        html_content = self._markdown_to_html(content)
        
        css_styles = styles or default_styles
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
{css_styles}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""
        
        if metadata and self.config.include_metadata:
            meta_tags = "\n".join(
                f'    <meta name="{k}" content="{v}">'
                for k, v in metadata.items()
            )
            html = html.replace("</head>", f"  {meta_tags}\n</head>")
        
        filepath = self._get_filepath(filename, "html")
        
        with open(filepath, "w") as f:
            f.write(html)
        
        return str(filepath)
    
    def export_txt(
        self,
        content: str,
        filename: str,
    ) -> str:
        """Export content as plain text"""
        
        text_only = self._strip_markdown(content)
        
        filepath = self._get_filepath(filename, "txt")
        
        with open(filepath, "w") as f:
            f.write(text_only)
        
        return str(filepath)
    
    def export_json(
        self,
        content: str,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Export content as JSON"""
        import json
        
        export_data = {
            "content": content,
            "exported_at": datetime.now().isoformat(),
        }
        
        if metadata:
            export_data["metadata"] = metadata
        
        filepath = self._get_filepath(filename, "json")
        
        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)
        
        return str(filepath)
    
    def export_multiple_formats(
        self,
        content: str,
        filename: str,
        formats: List[str],
        **kwargs
    ) -> Dict[str, str]:
        """Export to multiple formats at once"""
        
        results = {}
        
        for fmt in formats:
            if fmt not in self.SUPPORTED_FORMATS:
                continue
            
            try:
                if fmt == "md":
                    results[fmt] = self.export_markdown(content, filename, kwargs.get("metadata"))
                elif fmt == "html":
                    results[fmt] = self.export_html(
                        content,
                        filename,
                        title=kwargs.get("title", filename),
                        metadata=kwargs.get("metadata"),
                    )
                elif fmt == "txt":
                    results[fmt] = self.export_txt(content, filename)
                elif fmt == "json":
                    results[fmt] = self.export_json(content, filename, kwargs.get("metadata"))
            
            except Exception as e:
                results[fmt] = f"Error: {str(e)}"
        
        return results
    
    def create_content_package(
        self,
        content: str,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None,
        include_formats: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Create a complete content package with all formats"""
        
        include_formats = include_formats or ["md", "html", "json"]
        
        return self.export_multiple_formats(
            content=content,
            filename=filename,
            formats=include_formats,
            metadata=metadata,
            title=filename,
        )
    
    def generate_readme(
        self,
        project_name: str,
        description: str,
        content_files: List[str],
    ) -> str:
        """Generate a README for exported content"""
        
        readme = f"""# {project_name}

{description}

## Contents

| File | Description |
|------|-------------|
"""
        
        for file in content_files:
            readme += f"| `{file}` | |\n"
        
        readme += f"""
## Export Details

- **Exported**: {datetime.now().strftime(self.config.date_format)}
- **Files**: {len(content_files)}

---
*Generated with AI Writing Agent*
"""
        
        return readme
    
    def _get_filepath(self, filename: str, extension: str) -> Path:
        """Get the full filepath for export"""
        
        if not filename.endswith(f".{extension}"):
            filename = f"{filename}.{extension}"
        
        return Path(self.config.output_dir) / filename
    
    def _markdown_to_html(self, content: str) -> str:
        """Convert basic Markdown to HTML"""
        
        html = content
        
        html = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)
        html = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
        html = re.sub(r"^# (.+)$", r"<h1>\1</h1>", html, flags=re.MULTILINE)
        
        html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
        html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)
        
        html = re.sub(r"`(.+?)`", r"<code>\1</code>", html)
        
        html = re.sub(r"^\*\s(.+)$", r"<li>\1</li>", html, flags=re.MULTILINE)
        html = re.sub(r"^\d+\.\s(.+)$", r"<li>\1</li>", html, flags=re.MULTILINE)
        
        html = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', html)
        
        paragraphs = re.split(r"\n\n+", html)
        html = "\n".join(
            f"<p>{p}</p>" if not p.startswith("<") and not p.endswith(">") else p
            for p in paragraphs
        )
        
        return html
    
    def _strip_markdown(self, content: str) -> str:
        """Remove Markdown formatting from content"""
        
        text = content
        
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"\*(.+?)\*", r"\1", text)
        text = re.sub(r"__(.+?)__", r"\1", text)
        text = re.sub(r"_(.+?)_", r"\1", text)
        text = re.sub(r"`(.+?)`", r"\1", text)
        text = re.sub(r"```[\s\S]*?```", "", text)
        text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
        text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^---+\s*$", "", text, flags=re.MULTILINE)
        
        return text.strip()
