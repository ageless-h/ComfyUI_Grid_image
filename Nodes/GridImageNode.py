# -*- coding: utf-8 -*-
"""
网格图像节点 - 将多张图像拼接成网格图
使用 ComfyUI 新版API (node v2) 实现动态输入
"""

from typing import List, Tuple
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from comfy_api.latest import io, _io


# 预设分辨率配置（分辨率后边跟比例）
ALL_RESOLUTIONS = [
    # 横屏 16:9
    ("4K (3840x2160) 16:9", 3840, 2160),
    ("2K (2560x1440) 16:9", 2560, 1440),
    ("1080p (1920x1080) 16:9", 1920, 1080),
    ("720p (1280x720) 16:9", 1280, 720),
    ("480p (854x480) 16:9", 854, 480),
    # 横屏 21:9
    ("1080p (1920x810) 21:9", 1920, 810),
    # 竖屏 9:16
    ("4K (2160x3840) 9:16", 2160, 3840),
    ("2K (1440x2560) 9:16", 1440, 2560),
    ("1080p (1080x1920) 9:16", 1080, 1920),
    ("720p (720x1280) 9:16", 720, 1280),
    ("480p (480x854) 9:16", 480, 854),
]

RESOLUTION_LABELS = [r[0] for r in ALL_RESOLUTIONS]


class GridImageNode(io.ComfyNode):
    """网格图像节点 - 新版API (node v2)"""

    @classmethod
    def define_schema(cls):
        # 动态图像输入模板：prefix="image", 最小1个，最多32个
        image_template = _io.Autogrow.TemplatePrefix(
            input=io.Image.Input("image"),
            prefix="image",
            min=1,
            max=32
        )

        return io.Schema(
            node_id="GridImageNode",
            display_name="网格图像",
            category="Grid image",
            inputs=[
                io.Combo.Input(
                    "resolution",
                    options=RESOLUTION_LABELS,
                    default="1080p (1920x1080) 16:9",
                    display_name="分辨率"
                ),
                io.Int.Input(
                    "rows",
                    default=2,
                    min=1,
                    max=8,
                    display_name="行数"
                ),
                io.Int.Input(
                    "cols",
                    default=2,
                    min=1,
                    max=8,
                    display_name="列数"
                ),
                _io.Autogrow.Input(
                    "images",
                    template=image_template,
                    display_name="图像"
                ),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        """处理图像并生成网格图"""

        # 获取参数
        resolution = kwargs.get("resolution", "1080p (1920x1080) 16:9")
        rows = kwargs.get("rows", 2)
        cols = kwargs.get("cols", 2)

        width, height = cls._get_resolution_size(resolution)

        # 收集动态图像输入
        image_list = []
        for key, value in kwargs.items():
            if key.startswith("images.image") and value is not None:
                image_list.append(value)

        if "images" in kwargs and isinstance(kwargs["images"], dict):
            for key, value in kwargs["images"].items():
                if value is not None:
                    image_list.append(value)

        if not image_list:
            output = torch.zeros((1, height, width, 3), dtype=torch.float32)
            return io.NodeOutput(output)

        cell_width = width // cols
        cell_height = height // rows

        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        for idx, image_tensor in enumerate(image_list):
            if idx >= rows * cols:
                break

            row = idx // cols
            col = idx % cols
            x = col * cell_width
            y = row * cell_height

            img = cls._tensor_to_pil(image_tensor)
            img_resized = img.resize((cell_width, cell_height), Image.LANCZOS)
            img_array = np.array(img_resized)
            canvas[y:y+cell_height, x:x+cell_width] = img_array
            cls._draw_number(canvas, idx + 1, x, y, cell_width, cell_height)

        output = torch.from_numpy(canvas).float() / 255.0
        output = output.unsqueeze(0)

        return io.NodeOutput(output)

    @staticmethod
    def _get_resolution_size(resolution_label: str) -> Tuple[int, int]:
        for label, width, height in ALL_RESOLUTIONS:
            if label == resolution_label:
                return width, height
        return 1920, 1080

    @staticmethod
    def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        if tensor.dim() == 4:
            tensor = tensor[0]
        array = tensor.cpu().numpy()
        array = (array * 255).clip(0, 255).astype(np.uint8)
        if array.shape[-1] == 1:
            array = array[:, :, 0]
            return Image.fromarray(array, mode="L").convert("RGB")
        else:
            return Image.fromarray(array, mode="RGB")

    @staticmethod
    def _draw_number(canvas: np.ndarray, number: int, x: int, y: int, width: int, height: int):
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        font_size = min(width, height) // 8
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        text = str(number)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        padding = min(width, height) // 16
        text_x = x + width - text_width - padding
        text_y = y + height - text_height - padding
        outline_color = "black"
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, 0)]:
            draw.text((text_x + dx, text_y + dy), text, font=font, fill=outline_color)
        draw.text((text_x, text_y), text, font=font, fill="white")
        canvas[:] = np.array(img)
