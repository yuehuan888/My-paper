# 清理本项目遗留的后台进程。
#
# 背景：反复用 `Stop-Process python` 重启训练时，只杀掉了 python，
# 父级 bash 脚本仍然存活并继续循环启动下一个训练，导致多个训练进程
# 同时抢同一块 GPU（实测曾累积到 3 个），严重拖慢速度。
#
# 本脚本先列出、再按确认的 PID 杀掉，不使用按镜像名的通配杀法。

$patterns = 'redcnn|chain_ms|\.done|download\.sh|dl_chain|quickcfg|gamma|leak|diagnose|wave|ldct_(fast|3arm|final|seeds)|bench|sweep'

$victims = @()

Get-CimInstance Win32_Process -Filter "Name='bash.exe'" | ForEach-Object {
    if ($_.CommandLine -match $patterns) {
        $victims += [pscustomobject]@{ PID = $_.ProcessId; Kind = 'bash'; Cmd = $_.CommandLine }
    }
}
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ForEach-Object {
    $tag = (($_.CommandLine -split '--tag ')[1] -split ' ')[0]
    $victims += [pscustomobject]@{ PID = $_.ProcessId; Kind = 'python'; Cmd = $tag }
}

if ($victims.Count -eq 0) {
    Write-Output "没有找到遗留进程"
    exit 0
}

Write-Output "将清理以下进程："
$victims | ForEach-Object { Write-Output ("  {0,-6} {1,-8} {2}" -f $_.PID, $_.Kind, $_.Cmd) }

Stop-Process -Id ($victims | ForEach-Object { $_.PID }) -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

$left = Get-Process bash, python, sleep -ErrorAction SilentlyContinue
Write-Output ""
Write-Output ("清理完成。剩余 bash/python/sleep 进程数: " + (($left | Measure-Object).Count))
