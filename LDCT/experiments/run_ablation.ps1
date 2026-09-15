# ============================================================
#  Ablation sweep: the lifting knobs the paper never varied.
#
#  PRIORITY 1 -- BOUND SWEEP (this is contribution 3).
#    The paper claims "structural PR does not enforce itself in
#    float32; bounding the taps is what makes it hold" but backs
#    it with a SINGLE observation point: unbound taps drift to
#    |P| ~ 5 and the round-trip error goes 3.6e-07 -> 5.2e+00.
#    One point is not a dose-response. Sweeping `bound` turns
#    that assertion into a curve -- and, crucially, lets us test
#    whether the round-trip degradation IS WHAT COSTS PSNR, in
#    exactly the way the epsilon-intervention did in section 5.3.
#    If PSNR tracks round-trip error here too, contribution 3
#    gains the same causal status as contribution 2.
#
#    bound=0.5 is the current default (|taps| <= 1.5) and already
#    has n=5 results -- it is the control arm, NOT re-run here.
#
#  PRIORITY 2 -- CAPACITY/depth knobs (n_taps, levels).
#    Cheap, and reviewers routinely ask whether the conclusion
#    survives a different transform size.
#
#  WHY THIS FILE IS PURE ASCII: PowerShell 5.1 reads a .ps1
#  without a BOM as ANSI/GBK; full-width punctuation breaks the
#  parser silently. Keep it ASCII-only.
#
#  WHY SEQUENTIAL: single 6GB GPU.
#
#  Usage:
#      powershell -File experiments/run_ablation.ps1
#      powershell -File experiments/run_ablation.ps1 -Only bound
# ============================================================

param(
    [string]$Only = "all"   # all | bound | size
)

$PY = "D:\huan.yue\tools\miniconda3\envs\prlwt\python.exe"
$env:PYTHONIOENCODING = "utf-8"
Set-Location (Split-Path $PSScriptRoot -Parent)

$COMMON = @(
    "--wavelet", "pr",
    "--dataset", "aapm",
    "--epochs", "30",
    "--patch", "128",
    "--batch-size", "8",
    "--val-interval", "5",
    "--seed", "0"
)

function Run-One($tag, $extra) {
    Write-Output "==== $tag start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===="
    & $PY -u experiments/train.py --tag $tag @COMMON @extra
    if ($LASTEXITCODE -ne 0) {
        Write-Output "!! $tag exited $LASTEXITCODE -- stopping."
        exit $LASTEXITCODE
    }
}

# ---------------------------------------------------------------
# PRIORITY 1: bound sweep (tag: bnd<b>)
#   0.5 = control (already exists as w3_pr_s*), do not re-run.
#   Larger bound => taps allowed to drift further => float32
#   round-trip should degrade. We expect PSNR to follow.
# ---------------------------------------------------------------
if ($Only -eq "all" -or $Only -eq "bound") {
    foreach ($b in @("1.0", "2.0", "4.0", "8.0")) {
        $t = $b.Replace(".", "")
        Run-One "bnd${t}_s0" @("--bound", $b)
    }
}

# ---------------------------------------------------------------
# PRIORITY 2: transform-size knobs
#   n_taps must be odd (zero-padding needs a centre tap).
#   levels=1 -> 4 subbands; levels=2 (default) -> 7.
# ---------------------------------------------------------------
if ($Only -eq "all" -or $Only -eq "size") {
    foreach ($nt in @("5", "7")) {
        Run-One "taps${nt}_s0" @("--n-taps", $nt)
    }
    foreach ($lv in @("1", "3")) {
        Run-One "lvl${lv}_s0" @("--levels", $lv)
    }
}

Write-Output "ALL DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Output ""
Write-Output "Round-trip error for each arm comes from:"
Write-Output "  python experiments/verify_roundtrip.py --checkpoints"
