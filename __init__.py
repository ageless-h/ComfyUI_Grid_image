# -*- coding: utf-8 -*-
"""
ComfyUI Grid Image Node
将多张图像拼接成网格图的节点
使用新版API (node v2) 实现动态输入

功能特性:
- 动态图像输入: 连接一个图像后自动出现下一个，最多32个
- 预设分辨率: 包含4K/2K/1080p/720p/480p等常用分辨率
- 横竖屏支持: 16:9、21:9、9:16等常见比例
- 网格布局: 可设置行数和列数
- 顺序编号: 每个网格右下角显示顺序号
"""

from typing import List
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .Nodes import GridImageNode


class GridImageExtension(ComfyExtension):
    """网格图像节点扩展"""

    @override
    async def get_node_list(self) -> List[type[io.ComfyNode]]:
        return [GridImageNode]


async def comfy_entrypoint() -> GridImageExtension:
    """ComfyUI入口点"""
    return GridImageExtension()
