# ============================================================
#  RED-CNN same-split baseline, MULTIPLE SEEDS.
#
#  WHY: the paper currently reports RED-CNN at n=1 seed on each
#  split (31.8310 S1 / 32.9471 S2) while the three arms have
#  n=5 and n=3. No seed-level paired test is therefore possible
#  against RED-CNN, and no interval can be quoted. This closes
#  that asymmetry (paper limitation #4).
#
#  COST: RED-CNN is 1,848,865 params. Eval batch size MUST be 1
#  (cuDNN picks a ~7GB-workspace ConvTranspose2d algorithm at
#  batch>=2 on 512x512 -- see train.py evaluate()). Each seed is
#  roughly 6x the cost of a PR-LWT seed.
#
#  WHY THIS FILE IS PURE ASCII (do not add Chinese to it):
#    Windows PowerShell 5.1 reads a .ps1 WITHOUT a BOM as
#    ANSI/GBK. Full-width punctuation decodes wrongly and
#    swallows the following quote, silently breaking the parser.
#    Keep this file ASCII-only.
#
#  WHY SEQUENTIAL: single 6GB GPU. Concurrent runs fight over it.
#
#  Usage:
#      powershell -File experiments/run_redcnn_seeds.ps1
#
#  Progress:  Get-Content experiments/runs/<tag>/progress.json
#  Finished:  experiments/runs/<tag>/DONE
#  Interrupted?  re-run train.py with --resume
# ============================================================

$PY = "D:\huan.yue\tools\miniconda3\envs\prlwt\python.exe"
$env:PYTHONIOENCODING = "utf-8"
Set-Location (Split-Path $PSScriptRoot -Parent)

# Seeds 1..4 -- seed 0 already exists (redcnn_s0 / lit_redcnn_s0).
# Do NOT re-run seed 0: it would overwrite results that the paper's
# section 5.1 already cites.
$SEEDS = @(1, 2, 3, 4)

$COMMON = @(
    "--model", "redcnn",
    "--dataset", "aapm",
    "--epochs", "30",
    "--patch", "128",
    "--batch-size", "8",
    "--val-interval", "5"
)

foreach ($s in $SEEDS) {
    $tag = "redcnn_s$s"
    Write-Output "==== $tag (S1) start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===="
    & $PY -u experiments/train.py --tag $tag --seed $s @COMMON
    if ($LASTEXITCODE -ne 0) {
        Write-Output "!! $tag exited $LASTEXITCODE -- stopping."
        exit $LASTEXITCODE
    }
}

foreach ($s in $SEEDS) {
    $tag = "lit_redcnn_s$s"
    Write-Output "==== $tag (S2) start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===="
    & $PY -u experiments/train.py --tag $tag --seed $s `
        --split-file splits/aapm_mayo_3mm_lit.json @COMMON
    if ($LASTEXITCODE -ne 0) {
        Write-Output "!! $tag exited $LASTEXITCODE -- stopping."
        exit $LASTEXITCODE
    }
}

Write-Output "ALL DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
