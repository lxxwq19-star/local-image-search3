# 编译完成后，运行本脚本把 exe + DLL 复制到部署目录
# 用法：在 PowerShell 中 cd 到 src-tauri 目录，然后：
# .\copy-to-deploy.ps1

$DeployDir = "D:\local-image-search3-deploy"
$ReleaseDir = "D:\local-image-search3\src-tauri\target\release"

# 1. 复制 exe
Copy-Item "$ReleaseDir\local-image-search.exe" "$DeployDir\" -Force
Write-Host "✅ 已复制 local-image-search.exe" -ForegroundColor Green

# 2. 复制 ONNX Runtime DLLs
$Dlls = @(
    "onnxruntime.dll",
    "onnxruntime_providers_shared.dll"
)

foreach ($dll in $Dlls) {
    $src = "$ReleaseDir\$dll"
    if (Test-Path $src) {
        Copy-Item $src "$DeployDir\" -Force
        Write-Host "✅ 已复制 $dll" -ForegroundColor Green
    } else {
        Write-Host "⚠️  未找到 $dll（可能在其他位置）" -ForegroundColor Yellow
    }
}

# 3. 递归查找其他可能需要的 DLL
$AllDlls = Get-ChildItem "$ReleaseDir\*.dll" -ErrorAction SilentlyContinue
Write-Host "`n部署目录 DLL 清单：" -ForegroundColor Cyan
Get-ChildItem "$DeployDir\*.dll" | ForEach-Object { Write-Host "  $($_.Name)" }

Write-Host "`n✅ 部署目录已完整：$DeployDir" -ForegroundColor Cyan
