"""
ComfyUI自定义节点
将此文件放到 ComfyUI/custom_nodes/ 目录下
"""
import torch
import numpy as np
from PIL import Image


class SD15FromScratchNode:
    """从零实现的SD1.5节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "a beautiful landscape"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 150}),
                "cfg_scale": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.5}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "SD15FromScratch"
    
    def __init__(self):
        self.pipeline = None
    
    def generate(self, prompt, negative_prompt, width, height, steps, cfg_scale, seed):
        """生成图像"""
        # 延迟加载模型
        if self.pipeline is None:
            from .pipeline import StableDiffusionPipeline
            self.pipeline = StableDiffusionPipeline(
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        
        # 生成图像
        image = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            seed=seed
        )
        
        # 转换为ComfyUI格式 [B, H, W, C]
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]
        
        return (image_tensor,)


# ComfyUI节点注册
NODE_CLASS_MAPPINGS = {
    "SD15FromScratch": SD15FromScratchNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SD15FromScratch": "SD1.5 From Scratch"
}
