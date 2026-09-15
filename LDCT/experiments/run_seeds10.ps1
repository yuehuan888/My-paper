# ============================================================
#  Extend the three-arm comparison from n=5 to n=10 seeds (S1).
#
#  WHY 30 EPOCHS AND NOT 60:
#    Every existing run has best_epoch == 30, i.e. validation was
#    still improving when the budget ran out -- the arms are
#    compared at a FIXED BUDGET, not at convergence. Running the
#    new seeds at 60 epochs would make them incomparable with the
#    existing n=5 and silently mix two protocols in one table.
#    So this script deliberately matches the existing 30-epoch
#    protocol. A separate, complete 60-epoch re-run of ALL seeds
#    is the correct follow-up if convergence is to be addressed
#    (see run_convergence60.ps1) -- do not half-do it here.
#
#  Seeds 5..9 are new; seeds 0..4 already exist and must not be
#  re-run (the paper cites them).
#
#  WHY THIS FILE IS PURE ASCII: PowerShell 5.1 reads a .ps1
#  without a BOM as ANSI/GBK; full-width punctuation breaks the
#  parser silently. Keep it ASCII-only.
#
#  Usage:
#      powershell -File experiments/run_seeds10.ps1
# ============================================================

$PY = "D:\huan.yue\tools\miniconda3\envs\prlwt\python.exe"
$env:PYTHONIOENCODING = "utf-8"
Set-Location (Split-Path $PSScriptRoot -Parent)

$SEEDS = @(5, 6, 7, 8, 9)
$WAVELETS = @("pr", "unconstrained", "fixed")

$COMMON = @(
    "--dataset", "aapm",
    "--epochs", "30",
    "--patch", "128",
    "--batch-size", "8",
    "--val-interval", "5"
)

foreach ($w in $WAVELETS) {
    foreach ($s in $SEEDS) {
        $tag = "w3_${w}_s$s"
        Write-Output "==== $tag start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===="
        & $PY -u experiments/train.py --tag $tag --wavelet $w --seed $s @COMMON
        if ($LASTEXITCODE -ne 0) {
            Write-Output "!! $tag exited $LASTEXITCODE -- stopping."
            exit $LASTEXITCODE
        }
    }
}

Write-Output "ALL DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Output "Now re-run:  python experiments/analyze_arms.py --prefix w3_"
