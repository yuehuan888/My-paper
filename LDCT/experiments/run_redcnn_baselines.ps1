# ============================================================
#  Run RED-CNN baselines on the same splits as the main method.
#  Closes Limitation #3 of the paper: "no same-split comparison
#  with real baselines" (the published 32.93 dB is quoted, and
#  the published split uses 9 training patients vs our 8).
#
#  WHY THIS FILE IS PURE ASCII (do not add Chinese to it):
#    Windows PowerShell 5.1 reads a .ps1 file WITHOUT a BOM as
#    ANSI/GBK. Chinese full-width punctuation like ) encodes as
#    EF BC 89 in UTF-8; decoded as GBK those bytes swallow the
#    following quote character and the parser silently breaks
#    (a line gets echoed instead of executed). That is exactly
#    what happened on 2026-09-14 15:08 -- the first run produced
#    no output at all. Keep this file ASCII-only.
#
#  WHY SEQUENTIAL: this is a 4GB RTX 3050 Laptop. A previous
#  session ran several trainings at once, they fought over the
#  one GPU, and cleanup_procs.ps1 was written to kill them.
#  Serial is slower but stable and gives clean timings.
#
#  Usage:
#      powershell -File experiments/run_redcnn_baselines.ps1
#
#  Progress at any time:
#      Get-Content experiments/runs/<tag>/train.log -Tail 20
#      Get-Content experiments/runs/<tag>/progress.json
#  Finished marker: experiments/runs/<tag>/DONE
#  If killed mid-run, re-run train.py with --resume to continue.
# ============================================================

$PY = "D:\DeveloperTools\miniconda\envs\mgfnet\python.exe"
$env:PYTHONIOENCODING = "utf-8"
Set-Location (Split-Path $PSScriptRoot -Parent)

$ARGS_COMMON = @(
    "--model", "redcnn",
    "--dataset", "aapm",
    "--epochs", "30",
    "--patch", "128",
    "--batch-size", "8",
    "--val-interval", "5",
    "--seed", "0"
)

Write-Output "==== [1/2] S1 split: 7 train / L291 val / L506+L067 test ===="
Write-Output "start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
& $PY -u experiments/train.py --tag redcnn_s0 @ARGS_COMMON
if ($LASTEXITCODE -ne 0) {
    Write-Output "!! redcnn_s0 exited with code $LASTEXITCODE -- stopping."
    exit $LASTEXITCODE
}
Write-Output "done  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

Write-Output "==== [2/2] S2 split: 8 train / L067 val / L506 test (literature-aligned) ===="
Write-Output "start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
& $PY -u experiments/train.py --tag lit_redcnn_s0 `
    --split-file splits/aapm_mayo_3mm_lit.json @ARGS_COMMON
if ($LASTEXITCODE -ne 0) {
    Write-Output "!! lit_redcnn_s0 exited with code $LASTEXITCODE"
    exit $LASTEXITCODE
}
Write-Output "done  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

Write-Output "ALL DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
