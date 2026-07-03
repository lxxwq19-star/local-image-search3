#!/usr/bin/env python3
"""
将 PyTorch 模型 (SigLIP2 + CLIP-L/14) 导出为 ONNX 格式。
只需要跑一次，之后运行时不再需要 Python/PyTorch。

用法：
  python scripts/convert_models.py

输出：
  models/siglip2_vision.onnx   — SigLIP2 视觉编码器 → 1024维
  models/cliplarge_text.onnx   — CLIP-L/14 文本编码器 → 768维

前置条件：
  pip install torch torchvision transformers accelerate
"""
import os
import sys
import argparse

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # D:\local-image-search3
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

# Model source directories (from local-image-search2)
# We look at the original models in local-image-search2/
ORIGINAL_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "local-image-search2", "models")
SIGLIP2_SRC = os.path.join(ORIGINAL_DIR, "siglip2-large")
CLIPLARGE_SRC = os.path.join(ORIGINAL_DIR, "clip-large")

OUT_SIGLIP2_VISION = os.path.join(MODELS_DIR, "siglip2_vision.onnx")
OUT_CLIPLARGE_TEXT = os.path.join(MODELS_DIR, "cliplarge_text.onnx")
OUT_CLIPLARGE_VISION = os.path.join(MODELS_DIR, "cliplarge_vision.onnx")


def log(msg):
    print(f"[CONVERT] {msg}", flush=True)


def check_deps():
    """检查 PyTorch/transformers 是否可用"""
    try:
        import torch
        log(f"PyTorch {torch.__version__} — CUDA: {torch.cuda.is_available()}")
        import transformers
        log(f"Transformers {transformers.__version__}")
        return True
    except ImportError as e:
        log(f"❌ 缺少依赖: {e}")
        log("请安装: pip install torch torchvision transformers accelerate")
        return False


def export_siglip2_vision():
    """导出 SigLIP2 视觉编码器为 ONNX"""
    log(f"=" * 60)
    log(f"导出 SigLIP2 视觉编码器")
    log(f"来源: {SIGLIP2_SRC}")
    log(f"目标: {OUT_SIGLIP2_VISION}")

    if not os.path.isdir(SIGLIP2_SRC):
        log(f"⚠️  源目录不存在: {SIGLIP2_SRC}")
        log(f"   跳过 SigLIP2 导出")
        return False

    import torch
    from transformers import AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"加载 SigLIP2 模型到 {device}...")

    model = AutoModel.from_pretrained(SIGLIP2_SRC, torch_dtype=torch.float32)
    model = model.to(device)
    model.eval()

    # SigLIP2 使用 AutoModel，我们需要 vision model 部分
    # 不同版本的 transformers 可能有不同的访问方式
    if hasattr(model, "vision_model"):
        vision_model = model.vision_model
        log(f"使用 model.vision_model")
    else:
        # 对于某些 SigLIP2 变体，可能直接是模型本身
        vision_model = model
        log(f"使用 model 本身作为 vision 编码器")

    # 构造 dummy 输入: (1, 3, 256, 256)
    dummy_input = torch.randn(1, 3, 256, 256, device=device)

    log(f"执行一次前向推理验证...")
    with torch.no_grad():
        # 尝试不同的输出获取方式
        outputs = vision_model(dummy_input)
        if hasattr(outputs, "pooler_output"):
            test_out = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            test_out = outputs.last_hidden_state[:, 0]
        elif isinstance(outputs, torch.Tensor):
            test_out = outputs
        elif hasattr(outputs, "image_embeds"):
            test_out = outputs.image_embeds
        else:
            # 尝试索引
            if isinstance(outputs, (list, tuple)):
                test_out = outputs[0]
            else:
                # 遍历属性
                for key in ["pooler_output", "last_hidden_state", "logits", "embeddings"]:
                    if hasattr(outputs, key):
                        test_out = getattr(outputs, key)
                        break
                else:
                    test_out = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

        if isinstance(test_out, torch.Tensor):
            log(f"输出形状: {test_out.shape}")
            dim = test_out.shape[-1]
        else:
            log(f"⚠️  无法确定输出维度")
            dim = 1024  # 默认

    log(f"导出 ONNX (动态 batch)...")
    import torch.nn as nn

    # 包装器：只输出 pooler_output（1024维向量），不是全序列
    class VisionEncoderWrapper(nn.Module):
        def __init__(self, vision_model):
            super().__init__()
            self.vision_model = vision_model

        def forward(self, pixel_values):
            outputs = self.vision_model(pixel_values)
            return outputs.pooler_output

    wrapped = VisionEncoderWrapper(vision_model).to(device)

    input_names = ["pixel_values"]
    output_names = ["image_embeds"]
    dynamic_axes = {
        "pixel_values": {0: "batch_size"},
        "image_embeds": {0: "batch_size"},
    }

    # 使用 torch.onnx.export
    torch.onnx.export(
        wrapped,
        dummy_input,
        OUT_SIGLIP2_VISION,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )

    file_size_mb = os.path.getsize(OUT_SIGLIP2_VISION) / 1024 / 1024
    log(f"✅ SigLIP2 ONNX 导出完成: {file_size_mb:.1f} MB, dim={dim}")

    # 清理 GPU 内存
    del model, vision_model
    if device == "cuda":
        torch.cuda.empty_cache()

    return True


def export_cliplarge_text():
    """导出 CLIP-L/14 文本编码器为 ONNX"""
    log(f"=" * 60)
    log(f"导出 CLIP-L/14 文本编码器")
    log(f"来源: {CLIPLARGE_SRC}")
    log(f"目标: {OUT_CLIPLARGE_TEXT}")

    if not os.path.isdir(CLIPLARGE_SRC):
        log(f"⚠️  源目录不存在: {CLIPLARGE_SRC}")
        log(f"   跳过 CLIP-L/14 导出")
        return False

    import torch
    from transformers import CLIPModel, CLIPTokenizerFast

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"加载 CLIP-L/14 模型到 {device}...")

    model = CLIPModel.from_pretrained(CLIPLARGE_SRC, torch_dtype=torch.float32)
    model = model.to(device)
    model.eval()

    # 获取文本模型
    text_model = model.text_model

    # 构造 dummy 输入: input_ids (1, 77)
    dummy_input_ids = torch.randint(0, 49408, (1, 77), dtype=torch.int64, device=device)
    dummy_attention_mask = torch.ones(1, 77, dtype=torch.int64, device=device)

    log(f"执行一次前向推理验证...")
    with torch.no_grad():
        text_outputs = text_model(dummy_input_ids, attention_mask=dummy_attention_mask)
        if hasattr(text_outputs, "pooler_output"):
            test_out = text_outputs.pooler_output
            log(f"使用 pooler_output: {test_out.shape}")
        elif hasattr(text_outputs, "last_hidden_state"):
            # CLIP text uses the [EOS] token (last token) as pooled output
            test_out = text_outputs.last_hidden_state[:, -1]
            log(f"使用 last_hidden_state[:, -1]: {test_out.shape}")
        else:
            test_out = text_outputs[0][:, -1]
            log(f"使用 output[0][:, -1]: {test_out.shape}")

        dim = test_out.shape[-1] if test_out.dim() > 1 else 1
        log(f"输出维度: {dim}")

    log(f"导出 ONNX (动态 batch)...")
    import torch.nn as nn

    # 包装器：text_model pooler_output → text_projection → 768维文本特征向量
    class TextEncoderWrapper(nn.Module):
        def __init__(self, text_model, text_projection):
            super().__init__()
            self.text_model = text_model
            self.text_projection = text_projection

        def forward(self, input_ids, attention_mask):
            outputs = self.text_model(input_ids, attention_mask=attention_mask)
            # pooler_output 是未投影的 CLS/EOS 向量
            pooled = outputs.pooler_output
            # text_projection 将其映射到与视觉编码器同一空间
            projected = self.text_projection(pooled)
            return projected

    wrapped = TextEncoderWrapper(text_model, model.text_projection).to(device)

    input_names = ["input_ids", "attention_mask"]
    output_names = ["text_embeds"]
    dynamic_axes = {
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "text_embeds": {0: "batch_size"},
    }

    torch.onnx.export(
        wrapped,
        (dummy_input_ids, dummy_attention_mask),
        OUT_CLIPLARGE_TEXT,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )

    file_size_mb = os.path.getsize(OUT_CLIPLARGE_TEXT) / 1024 / 1024
    log(f"✅ CLIP-L/14 ONNX 导出完成: {file_size_mb:.1f} MB, dim={dim}")

    # 清理
    del model, text_model
    if device == "cuda":
        torch.cuda.empty_cache()

    return True


def export_cliplarge_vision():
    """导出 CLIP-L/14 视觉编码器为 ONNX (输出 768 维向量)"""
    log(f"=" * 60)
    log(f"导出 CLIP-L/14 视觉编码器")
    log(f"来源: {CLIPLARGE_SRC}")
    out_path = os.path.join(MODELS_DIR, "cliplarge_vision.onnx")
    log(f"目标: {out_path}")

    if not os.path.isdir(CLIPLARGE_SRC):
        log(f"⚠️  源目录不存在: {CLIPLARGE_SRC}")
        log(f"   跳过 CLIP 视觉导出")
        return False

    import torch
    import torch.nn as nn
    from transformers import CLIPModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"加载 CLIP-L/14 模型到 {device}...")
    model = CLIPModel.from_pretrained(CLIPLARGE_SRC, torch_dtype=torch.float32)
    model = model.to(device)
    model.eval()

    vision_model = model.vision_model
    visual_projection = model.visual_projection

    # 验证推理
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        outputs = vision_model(dummy_input)
        pooler = outputs.pooler_output  # (1, 1024) CLS token
        projected = visual_projection(pooler)  # (1, 768) final embedding
        log(f"pooler_output shape: {pooler.shape}")
        log(f"visual_projection output shape: {projected.shape}")

    class ClipVisionEncoderWrapper(nn.Module):
        def __init__(self, vision_model, visual_projection):
            super().__init__()
            self.vision_model = vision_model
            self.visual_projection = visual_projection
        def forward(self, pixel_values):
            outputs = self.vision_model(pixel_values)
            # pooler_output = CLS token embedding (1024-dim)
            pooled = outputs.pooler_output
            # project to 768-dim
            return self.visual_projection(pooled)

    wrapped = ClipVisionEncoderWrapper(vision_model, visual_projection).to(device)

    input_names = ["pixel_values"]
    output_names = ["image_embeds"]
    dynamic_axes = {
        "pixel_values": {0: "batch_size"},
        "image_embeds": {0: "batch_size"},
    }

    torch.onnx.export(
        wrapped,
        dummy_input,
        out_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )

    file_size_mb = os.path.getsize(out_path) / 1024 / 1024
    log(f"✅ CLIP-L/14 视觉 ONNX 导出完成: {file_size_mb:.1f} MB, dim=768")

    del model, vision_model, visual_projection
    if device == "cuda":
        torch.cuda.empty_cache()
    return True


def verify_onnx():
    """验证导出的 ONNX 模型"""
    log(f"=" * 60)
    log(f"验证 ONNX 模型")
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        log(f"⚠️  onnx/onnxruntime 未安装，跳过验证")
        log(f"   pip install onnx onnxruntime")
        return

    # 验证 SigLIP2
    if os.path.exists(OUT_SIGLIP2_VISION):
        try:
            model = onnx.load(OUT_SIGLIP2_VISION)
            onnx.checker.check_model(model)
            log(f"✅ SigLIP2 ONNX 格式验证通过")

            # ONNX Runtime 推理测试
            session = ort.InferenceSession(OUT_SIGLIP2_VISION, providers=["CPUExecutionProvider"])
            import numpy as np
            dummy_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
            output = session.run(None, {"pixel_values": dummy_input})
            log(f"✅ SigLIP2 推理测试: 输出形状 {output[0].shape}")
        except Exception as e:
            log(f"⚠️  SigLIP2 验证失败: {e}")

    # 验证 CLIP-L/14 文本
    if os.path.exists(OUT_CLIPLARGE_TEXT):
        try:
            model = onnx.load(OUT_CLIPLARGE_TEXT)
            onnx.checker.check_model(model)
            log(f"✅ CLIP-L/14 text ONNX 格式验证通过")

            # ONNX Runtime 推理测试
            session = ort.InferenceSession(OUT_CLIPLARGE_TEXT, providers=["CPUExecutionProvider"])
            import numpy as np
            dummy_ids = np.zeros((1, 77), dtype=np.int64)
            dummy_mask = np.ones((1, 77), dtype=np.int64)
            output = session.run(None, {"input_ids": dummy_ids, "attention_mask": dummy_mask})
            log(f"✅ CLIP-L/14 text 推理测试: 输出形状 {output[0].shape}")
        except Exception as e:
            log(f"⚠️  CLIP-L/14 text 验证失败: {e}")

    # 验证 CLIP-L/14 视觉
    if os.path.exists(OUT_CLIPLARGE_VISION):
        try:
            model = onnx.load(OUT_CLIPLARGE_VISION)
            onnx.checker.check_model(model)
            log(f"✅ CLIP-L/14 vision ONNX 格式验证通过")

            # ONNX Runtime 推理测试
            session = ort.InferenceSession(OUT_CLIPLARGE_VISION, providers=["CPUExecutionProvider"])
            import numpy as np
            dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
            output = session.run(None, {"pixel_values": dummy_input})
            log(f"✅ CLIP-L/14 vision 推理测试: 输出形状 {output[0].shape}")
        except Exception as e:
            log(f"⚠️  CLIP-L/14 vision 验证失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="将 PyTorch 模型导出为 ONNX")
    parser.add_argument("--verify-only", action="store_true", help="仅验证已有 ONNX 文件")
    parser.add_argument("--skip-verify", action="store_true", help="跳过验证步骤")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    if args.verify_only:
        verify_onnx()
        return

    if not check_deps():
        sys.exit(1)

    log(f"源模型目录: {ORIGINAL_DIR}")
    log(f"输出目录: {MODELS_DIR}")
    log(f"")

    siglip2_ok = export_siglip2_vision()
    cliplarge_ok = export_cliplarge_text()
    cliplarge_vision_ok = export_cliplarge_vision()

    log(f"")
    log(f"=" * 60)
    if siglip2_ok and os.path.exists(OUT_SIGLIP2_VISION):
        log(f"✅ SigLIP2: {OUT_SIGLIP2_VISION}")
    else:
        log(f"❌ SigLIP2: 导出失败或跳过")

    if cliplarge_ok and os.path.exists(OUT_CLIPLARGE_TEXT):
        log(f"✅ CLIP-L/14 text: {OUT_CLIPLARGE_TEXT}")
    else:
        log(f"❌ CLIP-L/14 text: 导出失败或跳过")

    if cliplarge_vision_ok and os.path.exists(OUT_CLIPLARGE_VISION):
        log(f"✅ CLIP-L/14 vision: {OUT_CLIPLARGE_VISION}")
    else:
        log(f"❌ CLIP-L/14 vision: 导出失败或跳过")

    log(f"")

    if not args.skip_verify:
        verify_onnx()

    log(f"=" * 60)
    log(f"完成！运行时不再需要 Python/PyTorch。")


if __name__ == "__main__":
    main()
