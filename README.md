# local-image-search3

本地图片语义搜索引擎 - 支持自然语言描述搜索本地图片

## 功能特性

- 🔍 **语义搜索**：用自然语言描述找图片（如"红色的汽车"、"海边日落"）
- 🖼️ **以图搜图**：上传一张图片，找出相似的图片
- 📁 **文件夹索引**：选择性索引指定文件夹，支持子文件夹层级管理
- ⚡ **GPU 加速**：支持 DirectML (Windows) / CoreML (macOS) 硬件加速
- 🎯 **Top-K 结果**：语义搜索返回最相关的 2000 张图片

## 下载

### Windows 版本

1. 访问 [GitHub Releases](https://github.com/lxxwq19-star/local-image-search3/releases)
2. 下载最新版本的 `local-image-search3-deploy.zip`
3. 解压后双击 `local-image-search.exe` 即可运行
4. 确保 `models/` 子目录跟 exe 在同一目录

### macOS 版本

⚠️ **需要单独下载模型文件**（2.8 GB，未打包进 .dmg）

#### 方式 1：下载预构建 dmg

1. 访问 [GitHub Actions](https://github.com/lxxwq19-star/local-image-search3/actions)
2. 选择最新的 `Build macOS` workflow run
3. 下载对应架构的 artifact：
   - `local-image-search-macos-intel` - Intel Mac（2020 年前）
   - `local-image-search-macos-arm` - Apple Silicon（M1/M2/M3/M4）

#### 方式 2：首次运行放行（必读）

macOS Gatekeeper 会拦截未签名应用，按以下任一方法放行：

**方法 1：右键打开（最简单）**
1. 打开 DMG，将 `Local Image Search.app` 拖到「应用程序」文件夹
2. **不要双击** App
3. `Control + 右键` 点击 App → 选择「打开」
4. 弹窗选择「打开」，系统将永久信任此 App

**方法 2：终端一键清除隔离标记（推荐）**
```bash
sudo xattr -rd com.apple.quarantine /Applications/Local\ Image\ Search.app
```

#### 方式 3：下载模型文件

**从 Hugging Face 自动下载（需科学上网）**

创建 `download_models.sh` 并执行：
```bash
#!/bin/bash
MODEL_DIR="$HOME/Library/Application Support/com.localimagesearch.app/models"
mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

echo "下载 SigLIP2 视觉模型..."
curl -L "https://huggingface.co/google/siglip2-base-patch16-256/resolve/main/model.onnx" -o siglip2_vision.onnx

echo "下载 CLIP Large 文本模型..."
curl -L "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/text_encoder.onnx" -o cliplarge_text.onnx

echo "下载 CLIP Large 视觉模型..."
curl -L "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/vision_encoder.onnx" -o cliplarge_vision.onnx

echo "下载完成！"
```

**从 Windows 版本复制**

如果你已有 Windows 版本，直接从 `local-image-search3-deploy\models\` 复制 3 个 `.onnx` 文件到：
```
~/Library/Application Support/com.localimagesearch.app/models/
```

## 构建（开发者）

### Windows

```bash
cd D:\local-image-search3\src-tauri
cargo build --release
```

### macOS

```bash
# Intel
cargo build --release --target x86_64-apple-darwin

# Apple Silicon
cargo build --release --target aarch64-apple-darwin
```

或使用 Tauri CLI：
```bash
cd /path/to/local-image-search3
npm install --prefix src
npm run build --prefix src
npx tauri build
```

## 技术栈

- **前端**：React + Vite + TypeScript
- **后端**：Rust + Tauri v2
- **AI 推理**：ONNX Runtime
  - SigLIP 2（视觉编码器）
  - CLIP Large（文本 + 视觉编码器）
- **数据库**：SQLite（rusqlite）

## 已知限制

### Windows
- ✅ 无需签名，直接运行
- ⚠️ 首次运行可能被杀毒软件拦截

### macOS
- ❌ 无法公证（需要 Apple Developer 证书 $99/年）
- ⚠️ 用户首次运行必须执行放行步骤
- ✅ Ad-hoc 签名已递归签名所有内嵌二进制，app 不会闪退

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

---

**Build Date**: 2026-07-03  
**Version**: 0.1.0
