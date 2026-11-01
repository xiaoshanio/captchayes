# ============================================================
# 一键跑完整「继续训练 + 数据更新」验证链路
#   Step 1  YOLO 检测器：在自造数据上训练 -> 导出 ONNX(nms=True) -> 契约校验
#   Step 2  Siamese 匹配器：在自造配对数据上训练 -> 导出 ONNX -> 契约校验
#   Step 3  端到端验收：回灌 model/ -> 跑未改动的 runtime -> 自动恢复
#
# 用法（在 T-Sec-captcha 目录下）：
#   pwsh -File train/run_all.ps1
#   pwsh -File train/run_all.ps1 -DetEpochs 30 -MatchEpochs 12 -SkipEval
#
# 【环境前提·必读】
#   1) pip install "polars[rtcompat]"
#      本机 CPU 无 AVX2/FMA/BMI，polars 官方 wheel 执行到内核会硬崩溃
#      (STATUS_ILLEGAL_INSTRUCTION 0xC000001D，Python 捕不到)。
#      只设 POLARS_SKIP_CPU_CHECK=1 **无效** —— 那只关告警，不解决崩溃。
#      rtcompat 是可移植运行时构建，装完即可正常写 results.csv。
#   2) ultralytics 在 device=cpu 时会强制 workers=0，并在 select_device()
#      里把 torch 线程数设为 min(8, cpu_count-1)。传参不一定生效。
# ============================================================
param(
    [int]$DetEpochs = 20,
    [int]$MatchEpochs = 8,
    [int]$DetBatch = 8,
    [int]$MatchBatch = 64,
    [int]$Imgsz = 640,
    [int]$Workers = 4,
    [string]$Data = "data/selfgen/data.yaml",
    [string]$Pairs = "data/selfpairs",
    [string]$OutDir = "runs/export_nms",
    [double]$EvalConf = 0.2,
    [int]$EvalLimit = 60,
    [switch]$SkipEval
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:POLARS_SKIP_CPU_CHECK = "1"       # 关告警（真正的修法是装 polars[rtcompat]）
$env:OMP_NUM_THREADS = "10"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ---- 前置检查：polars 是否可用（这是最容易踩的坑，提前挡住）
Write-Host "=========== Step 0/3  环境自检 ===========" -ForegroundColor Cyan
python -c "import polars as pl; pl.DataFrame({'a':[1]}).write_csv(); print('  polars OK', pl.__version__)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [FAIL] polars 不可用。请先执行：pip install `"polars[rtcompat]`"" -ForegroundColor Red
    exit 1
}
python -c "import onnx, onnxruntime, cv2, torch; print('  deps OK')"
if ($LASTEXITCODE -ne 0) { Write-Host "  [FAIL] 依赖缺失" -ForegroundColor Red; exit 1 }

$t0 = Get-Date

Write-Host "`n=========== Step 1/3  YOLO 检测器 ===========" -ForegroundColor Cyan
python train/train_detector.py `
    --data $Data `
    --epochs $DetEpochs `
    --imgsz $Imgsz `
    --batch $DetBatch `
    --workers $Workers `
    --device cpu `
    --cache ram `
    --project runs/detect `
    --name selfgen `
    --out-onnx "$OutDir/best_v3_nms.onnx"
if ($LASTEXITCODE -ne 0) { Write-Host "  [FAIL] 检测器训练/导出失败" -ForegroundColor Red; exit 1 }

Write-Host "`n=========== Step 2/3  Siamese 匹配器 ===========" -ForegroundColor Cyan
python train/train_matcher.py `
    --pairs $Pairs `
    --epochs $MatchEpochs `
    --batch $MatchBatch `
    --workers 0 `
    --threads 10 `
    --project runs/matcher `
    --name selfgen `
    --out-onnx "$OutDir/pre_model_v7_nms.onnx"
if ($LASTEXITCODE -ne 0) { Write-Host "  [FAIL] 匹配器训练/导出失败" -ForegroundColor Red; exit 1 }

if (-not $SkipEval) {
    Write-Host "`n=========== Step 3/3  端到端验收（回灌 + 自动恢复） ===========" -ForegroundColor Cyan
    python eval/eval_new.py `
        --synth (Split-Path $Data -Parent) `
        --new-model $OutDir `
        --limit $EvalLimit `
        --conf $EvalConf `
        --out out/eval_final
    if ($LASTEXITCODE -ne 0) { Write-Host "  [WARN] 验收脚本返回非零（model/ 已在 finally 中恢复）" -ForegroundColor Yellow }
}

Write-Host "`n=========== 完成 ===========" -ForegroundColor Green
Write-Host ("总耗时: {0:N1} 分钟" -f ((Get-Date) - $t0).TotalMinutes)
Get-ChildItem $OutDir -ErrorAction SilentlyContinue | Select-Object Name, Length
Write-Host ""
Write-Host "下一步：确认验收结果后正式回灌（回滚靠 runs/_model_backup/）："
Write-Host "  Copy-Item $OutDir/best_v3_nms.onnx      model/best_v3.onnx      -Force"
Write-Host "  Copy-Item $OutDir/pre_model_v7_nms.onnx model/pre_model_v7.onnx -Force"
Write-Host "  python demo.py"
