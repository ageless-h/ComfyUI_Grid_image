# -*- coding: utf-8 -*-
"""
测试 GridImageNode
"""

import pytest
import torch
import numpy as np
from PIL import Image, ImageDraw

# 设置环境变量以避免导入问题
import os
os.environ["COMFYUI_SYSTEM"] = "0"

from Nodes.GridImageNode import (
    GridImageNode,
    ALL_RESOLUTIONS,
    RESOLUTION_LABELS,
)


class TestGetResolutionSize:
    """测试 _get_resolution_size 静态方法"""

    def test_4k_landscape(self):
        """测试4K横屏分辨率"""
        width, height = GridImageNode._get_resolution_size("4K (3840x2160) 16:9")
        assert width == 3840
        assert height == 2160

    def test_2k_landscape(self):
        """测试2K横屏分辨率"""
        width, height = GridImageNode._get_resolution_size("2K (2560x1440) 16:9")
        assert width == 2560
        assert height == 1440

    def test_1080p_landscape(self):
        """测试1080p横屏分辨率"""
        width, height = GridImageNode._get_resolution_size("1080p (1920x1080) 16:9")
        assert width == 1920
        assert height == 1080

    def test_720p_landscape(self):
        """测试720p横屏分辨率"""
        width, height = GridImageNode._get_resolution_size("720p (1280x720) 16:9")
        assert width == 1280
        assert height == 720

    def test_480p_landscape(self):
        """测试480p横屏分辨率"""
        width, height = GridImageNode._get_resolution_size("480p (854x480) 16:9")
        assert width == 854
        assert height == 480

    def test_1080p_21_9(self):
        """测试1080p 21:9分辨率"""
        width, height = GridImageNode._get_resolution_size("1080p (1920x810) 21:9")
        assert width == 1920
        assert height == 810

    def test_4k_portrait(self):
        """测试4K竖屏分辨率"""
        width, height = GridImageNode._get_resolution_size("4K (2160x3840) 9:16")
        assert width == 2160
        assert height == 3840

    def test_2k_portrait(self):
        """测试2K竖屏分辨率"""
        width, height = GridImageNode._get_resolution_size("2K (1440x2560) 9:16")
        assert width == 1440
        assert height == 2560

    def test_1080p_portrait(self):
        """测试1080p竖屏分辨率"""
        width, height = GridImageNode._get_resolution_size("1080p (1080x1920) 9:16")
        assert width == 1080
        assert height == 1920

    def test_720p_portrait(self):
        """测试720p竖屏分辨率"""
        width, height = GridImageNode._get_resolution_size("720p (720x1280) 9:16")
        assert width == 720
        assert height == 1280

    def test_480p_portrait(self):
        """测试480p竖屏分辨率"""
        width, height = GridImageNode._get_resolution_size("480p (480x854) 9:16")
        assert width == 480
        assert height == 854

    def test_unknown_resolution(self):
        """测试未知分辨率返回默认值"""
        width, height = GridImageNode._get_resolution_size("unknown")
        assert width == 1920
        assert height == 1080

    def test_empty_resolution(self):
        """测试空字符串返回默认值"""
        width, height = GridImageNode._get_resolution_size("")
        assert width == 1920
        assert height == 1080


class TestTensorToPIL:
    """测试 _tensor_to_pil 静态方法"""

    def create_test_tensor_3d(self, height: int, width: int, channels: int = 3) -> torch.Tensor:
        """创建3D测试用张量 (H, W, C)"""
        data = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
        return torch.from_numpy(data.astype(np.float32) / 255.0)

    def test_3d_tensor_rgb(self):
        """测试3D张量(RGB)转换为PIL图像"""
        tensor = self.create_test_tensor_3d(100, 100, 3)
        result = GridImageNode._tensor_to_pil(tensor)
        assert isinstance(result, Image.Image)
        assert result.size == (100, 100)
        assert result.mode == "RGB"

    def test_4d_tensor_rgb(self):
        """测试4D张量(BATCH, H, W, C)转换为PIL图像"""
        tensor = self.create_test_tensor_3d(100, 100, 3)
        tensor_4d = tensor.unsqueeze(0)  # (1, H, W, C)
        result = GridImageNode._tensor_to_pil(tensor_4d)
        assert isinstance(result, Image.Image)
        assert result.size == (100, 100)
        assert result.mode == "RGB"

    def test_grayscale_3d_tensor(self):
        """测试3D灰度张量(H, W, 1)转换为RGB图像"""
        # 创建正确的灰度张量形状 (H, W, 1)
        data = np.random.randint(0, 256, (100, 100, 1), dtype=np.uint8)
        tensor = torch.from_numpy(data.astype(np.float32) / 255.0)
        result = GridImageNode._tensor_to_pil(tensor)
        assert isinstance(result, Image.Image)
        assert result.size == (100, 100)
        assert result.mode == "RGB"

    def test_grayscale_4d_tensor(self):
        """测试4D灰度张量(BATCH, H, W, 1)转换为RGB图像"""
        data = np.random.randint(0, 256, (100, 100, 1), dtype=np.uint8)
        tensor = torch.from_numpy(data.astype(np.float32) / 255.0)
        tensor_4d = tensor.unsqueeze(0)
        result = GridImageNode._tensor_to_pil(tensor_4d)
        assert isinstance(result, Image.Image)
        assert result.size == (100, 100)
        assert result.mode == "RGB"


class TestDrawNumber:
    """测试 _draw_number 静态方法"""

    def test_draw_single_number(self):
        """测试绘制单个数字"""
        canvas = np.zeros((200, 200, 3), dtype=np.uint8)
        GridImageNode._draw_number(canvas, 1, 0, 0, 200, 200)
        # 检查画布是否被修改
        assert canvas.sum() > 0

    def test_draw_multiple_numbers(self):
        """测试绘制多个数字"""
        canvas = np.zeros((200, 200, 3), dtype=np.uint8)
        GridImageNode._draw_number(canvas, 1, 0, 0, 100, 100)
        sum1 = canvas.sum()
        GridImageNode._draw_number(canvas, 2, 100, 0, 100, 100)
        assert canvas.sum() > sum1

    def test_draw_two_digit_number(self):
        """测试绘制两位数"""
        canvas = np.zeros((300, 300, 3), dtype=np.uint8)
        GridImageNode._draw_number(canvas, 10, 0, 0, 300, 300)
        assert canvas.sum() > 0

    def test_draw_number_out_of_bounds(self):
        """测试绘制数字时超出边界的情况"""
        canvas = np.zeros((50, 50, 3), dtype=np.uint8)
        # 小的画布可能导致数字位置计算为负
        GridImageNode._draw_number(canvas, 1, 0, 0, 50, 50)
        # 不应该报错

    def test_draw_number_changes_pixel_values(self):
        """测试绘制数字确实改变了像素值"""
        canvas = np.zeros((200, 200, 3), dtype=np.uint8)
        original_sum = canvas.sum()
        GridImageNode._draw_number(canvas, 1, 0, 0, 200, 200)
        assert canvas.sum() > original_sum


class TestDefineSchema:
    """测试 define_schema 类方法"""

    def test_schema_is_dict_like(self):
        """测试schema返回正确的结构"""
        schema = GridImageNode.define_schema()
        assert schema is not None
        assert hasattr(schema, 'node_id')
        assert schema.node_id == "GridImageNode"

    def test_schema_has_display_name(self):
        """测试schema有正确的显示名称"""
        schema = GridImageNode.define_schema()
        assert schema.display_name == "网格图像"

    def test_schema_has_category(self):
        """测试schema有正确的分类"""
        schema = GridImageNode.define_schema()
        assert schema.category == "Grid image"

    def test_schema_has_inputs(self):
        """测试schema有输入配置"""
        schema = GridImageNode.define_schema()
        assert len(schema.inputs) >= 4  # resolution, rows, cols, images

    def test_schema_has_outputs(self):
        """测试schema有输出配置"""
        schema = GridImageNode.define_schema()
        assert len(schema.outputs) == 1

    def test_resolution_options_count(self):
        """测试分辨率选项数量"""
        schema = GridImageNode.define_schema()
        resolution_input = schema.inputs[0]
        assert len(resolution_input.options) == len(RESOLUTION_LABELS)

    def test_rows_input_properties(self):
        """测试行数输入属性"""
        schema = GridImageNode.define_schema()
        rows_input = schema.inputs[1]
        assert rows_input.default == 2
        assert rows_input.min == 1
        assert rows_input.max == 8

    def test_cols_input_properties(self):
        """测试列数输入属性"""
        schema = GridImageNode.define_schema()
        cols_input = schema.inputs[2]
        assert cols_input.default == 2
        assert cols_input.min == 1
        assert cols_input.max == 8


class TestExecute:
    """测试 execute 类方法"""

    def create_test_tensor(self, height: int, width: int, value: int = 128) -> torch.Tensor:
        """创建测试用张量"""
        data = np.full((height, width, 3), value, dtype=np.uint8)
        return torch.from_numpy(data.astype(np.float32) / 255.0)

    def test_execute_with_no_images(self):
        """测试没有图像输入时返回空画布"""
        result = GridImageNode.execute()
        output = result[0]
        assert output.shape == (1, 1080, 1920, 3)
        assert output.dtype == torch.float32

    def test_execute_with_default_params(self):
        """测试默认参数执行"""
        result = GridImageNode.execute(resolution="720p (1280x720) 16:9")
        output = result[0]
        assert output.shape == (1, 720, 1280, 3)

    def test_execute_with_single_image(self):
        """测试单张图像"""
        image = self.create_test_tensor(100, 100, 128)
        result = GridImageNode.execute(
            resolution="1080p (1920x1080) 16:9",
            rows=2,
            cols=2,
            images={"image1": image}
        )
        output = result[0]
        assert output.shape == (1, 1080, 1920, 3)

    def test_execute_with_prefixed_images(self):
        """测试带前缀的图像输入"""
        image = self.create_test_tensor(100, 100, 200)
        result = GridImageNode.execute(
            resolution="720p (1280x720) 16:9",
            rows=2,
            cols=2,
            **{"images.image1": image}
        )
        output = result[0]
        assert output.shape == (1, 720, 1280, 3)

    def test_execute_multiple_images(self):
        """测试多张图像"""
        images = []
        for i in range(4):
            img = self.create_test_tensor(100, 100, int(50 + i * 50))
            images.append(img)

        result = GridImageNode.execute(
            resolution="1080p (1920x1080) 16:9",
            rows=2,
            cols=2,
            images={"image1": images[0], "image2": images[1],
                   "image3": images[2], "image4": images[3]}
        )
        output = result[0]
        assert output.shape == (1, 1080, 1920, 3)

    def test_execute_excess_images_ignored(self):
        """测试超过网格数量的图像被忽略"""
        images = []
        for i in range(10):  # 超过 2x2=4 的限制
            img = self.create_test_tensor(100, 100, int(50 + i * 20))
            images.append(img)

        kwargs = {
            "resolution": "1080p (1920x1080) 16:9",
            "rows": 2,
            "cols": 2,
        }
        for i in range(10):
            kwargs[f"images.image{i+1}"] = images[i]

        result = GridImageNode.execute(**kwargs)
        output = result[0]
        # 只处理前4张图像
        assert output.shape == (1, 1080, 1920, 3)

    def test_execute_vertical_resolution(self):
        """测试竖屏分辨率"""
        image = self.create_test_tensor(100, 100, 100)
        result = GridImageNode.execute(
            resolution="1080p (1080x1920) 9:16",
            rows=2,
            cols=2,
            images={"image1": image}
        )
        output = result[0]
        assert output.shape == (1, 1920, 1080, 3)

    def test_execute_single_row(self):
        """测试单行布局"""
        images = [self.create_test_tensor(100, 100, 100) for _ in range(4)]
        result = GridImageNode.execute(
            resolution="1080p (1920x1080) 16:9",
            rows=1,
            cols=4,
            images={"image1": images[0], "image2": images[1],
                   "image3": images[2], "image4": images[3]}
        )
        output = result[0]
        assert output.shape == (1, 1080, 1920, 3)

    def test_execute_single_column(self):
        """测试单列布局"""
        images = [self.create_test_tensor(100, 100, 100) for _ in range(4)]
        result = GridImageNode.execute(
            resolution="1080p (1920x1080) 16:9",
            rows=4,
            cols=1,
            images={"image1": images[0], "image2": images[1],
                   "image3": images[2], "image4": images[3]}
        )
        output = result[0]
        assert output.shape == (1, 1080, 1920, 3)


class TestConstants:
    """测试常量定义"""

    def test_all_resolutions_not_empty(self):
        """测试ALL_RESOLUTIONS不为空"""
        assert len(ALL_RESOLUTIONS) > 0

    def test_resolution_labels_count(self):
        """测试分辨率标签数量与ALL_RESOLUTIONS一致"""
        assert len(RESOLUTION_LABELS) == len(ALL_RESOLUTIONS)

    def test_all_resolutions_format(self):
        """测试所有分辨率格式正确"""
        for label, width, height in ALL_RESOLUTIONS:
            assert isinstance(label, str)
            assert isinstance(width, int)
            assert isinstance(height, int)
            assert width > 0
            assert height > 0

    def test_contains_common_resolutions(self):
        """测试包含常见分辨率"""
        labels = [r[0] for r in ALL_RESOLUTIONS]
        assert "1080p (1920x1080) 16:9" in labels
        assert "4K (3840x2160) 16:9" in labels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
