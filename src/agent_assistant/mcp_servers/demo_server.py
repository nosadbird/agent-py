"""本地 FastMCP stdio 示例服务。"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("agent-demo")


@mcp.tool()
def add(a: float, b: float) -> float:
    """两数相加。"""
    return a + b


@mcp.tool()
def text_stats(text: str) -> dict[str, int]:
    """统计文本的字符数、非空白字符数、单词数与行数。"""
    return {
        "char_count": len(text),
        "non_whitespace_count": sum(1 for char in text if not char.isspace()),
        "word_count": len(text.split()) if text.strip() else 0,
        "line_count": len(text.splitlines()),
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
