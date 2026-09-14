# ============================================================
#  Intervention experiment: inject a controlled round-trip error into the PR arm.
#
#  PURPOSE
#    The paper currently *attributes* the unconstrained arm's failure to its
#    lost invertibility, but never intervenes. This script supplies the missing
#    intervention: hold everything fixed, inject a controlled round-trip error
#    into the PR arm, and see whether performance degrades.
#
#    If performance falls monotonically with the injected error, the mechanism
#    claim becomes causal rather than correlational.
#
#  MECHANISM
#    models/lifting_dwt.py, _LiftingBank1D.U_synth: the SYNTHESIS step uses
#    (1-eps)*U while the ANALYSIS step still uses U. This reproduces exactly the
#    failure mode of the unconstrained arm ("the inverse is not the true
#    inverse"). eps=0 is a verified no-op (round-trip unchanged at 1.619e-07).
#
#  CALIBRATION (experiments/calibrate_mismatch.py, probe-verified)
#    eps = 0.111  -> round-trip ~6%
#    eps = 0.222  -> round-trip ~12.5%   <- matches the unconstrained arm's mean
#    eps = 0.444  -> round-trip ~25%
#    Error is approximately linear in eps, so this gives a DOSE-RESPONSE curve.
#
#  PROTOCOL - identical to the w3_pr_* control runs:
#    epochs=30 patch=128 batch=8 lr=1e-3 val_interval=5, split S1
#    seeds 0,1,2 so they pair with the existing w3_pr_s0..s2 control.
#
#  WHY THIS FILE IS PURE ASCII: Windows PowerShell 5.1 reads a .ps1 without a
#    BOM as GBK; Chinese full-width punctuation then swallows quotes and the
#    parser silently breaks. Learned the hard way on 2026-09-14. Keep it ASCII.
#
#  USAGE:  powershell -File experiments/run_intervention.ps1
#  PROGRESS: Get-Content experiments/runs/mis222_s0/progress.json
#  DONE MARKER: experiments/runs/mis222_s0/DONE
# ============================================================

$PY = "D:\DeveloperTools\miniconda\envs\mgfnet\python.exe"
$env:PYTHONIOENCODING = "utf-8"
Set-Location (Split-Path $PSScriptRoot -Parent)

$EPS_LIST = @("0.111", "0.222", "0.444")
$SEEDS = @("0", "1", "2", "3", "4")

$total = $EPS_LIST.Count * $SEEDS.Count
$n = 0

foreach ($eps in $EPS_LIST) {
    $tageps = $eps.Replace("0.", "")     # 0.111 -> "111"
    foreach ($seed in $SEEDS) {
        $n++
        $tag = "mis${tageps}_s${seed}"
        # Skip runs already finished, so this script can be re-run to extend a
        # partial sweep (e.g. from 3 seeds to 5) without redoing completed work.
        if (Test-Path "experiments/runs/$tag/DONE") {
            Write-Output "==== [$n/$total] tag=$tag  already done, skipping ===="
            continue
        }
        Write-Output "==== [$n/$total] tag=$tag  eps=$eps  seed=$seed ===="
        Write-Output "start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

        & $PY -u experiments/train.py `
            --tag $tag `
            --model pr_wavelet `
            --dataset aapm `
            --wavelet pr `
            --synth-mismatch $eps `
            --seed $seed `
            --epochs 30 `
            --patch 128 `
            --batch-size 8 `
            --val-interval 5

        if ($LASTEXITCODE -ne 0) {
            Write-Output "!! $tag exited with code $LASTEXITCODE"
            exit $LASTEXITCODE
        }
        Write-Output "done  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    }
}

Write-Output "INTERVENTION COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
