# Follow a live schedule from `run.cmd resource-parallel PLAN` as it runs.
# It prints the key lines of every run's log, prefixed with the run name, and a line
# whenever a run or the schedule changes status. It only reads files: it never touches
# the runs, the repository or a Minecraft server. Ctrl+C stops watching, not the runs.
#
#   powershell -ExecutionPolicy Bypass -File scripts\watch_schedule.ps1 runs\iq-experiment
param(
    [Parameter(Mandatory = $true)][string]$Schedule,
    [int]$Backlog = 5
)
$folder = Resolve-Path $Schedule -ErrorAction Stop
$logs = Join-Path $folder "logs"
$pattern = '^(E\d+ step|Episode \d+ finished|Condition:|Starting fresh|MASKED RANDOM|UNIFORM RANDOM|Error|Traceback|Saved checkpoint|Stopping|Refusing)'
$seen = @{}
$unitStatus = @{}
$scheduleStatus = $null
Write-Host "Watching $folder (Ctrl+C stops watching; the schedule keeps running)." -ForegroundColor Cyan

function Read-Schedule {
    # The scheduler replaces schedule.json atomically; a read can still meet the moment of
    # replacement, so a failed read is simply retried on the next round.
    foreach ($name in "schedule.final.json", "schedule.json") {
        $path = Join-Path $folder $name
        if (Test-Path $path) {
            try { return Get-Content $path -Raw -ErrorAction Stop | ConvertFrom-Json } catch { return $null }
        }
    }
    return $null
}

while ($true) {
    $record = Read-Schedule
    if ($record) {
        foreach ($unit in $record.units) {
            $now = "$($unit.status) $($unit.attempted)/$($unit.episodes)"
            if ($unitStatus[$unit.name] -ne $now) {
                $unitStatus[$unit.name] = $now
                $colour = if ($unit.status -eq "done") { "Green" } elseif ($unit.status -in "failed", "gave_up", "blocked", "interrupted") { "Red" } else { "Cyan" }
                Write-Host ("RUN    {0,-32} {1}" -f $unit.name, $now) -ForegroundColor $colour
            }
        }
        if ($record.status -ne $scheduleStatus) {
            $scheduleStatus = $record.status
            Write-Host "SCHEDULE $scheduleStatus" -ForegroundColor Yellow
        }
    }
    if (Test-Path $logs) {
        foreach ($log in Get-ChildItem (Join-Path $logs "*.log") | Sort-Object Name) {
            $lines = @(Get-Content $log.FullName -ErrorAction SilentlyContinue)
            if (-not $seen.ContainsKey($log.Name)) { $seen[$log.Name] = [Math]::Max(0, $lines.Count - $Backlog) }
            if ($lines.Count -le $seen[$log.Name]) { continue }
            $run = $log.BaseName
            foreach ($line in $lines[$seen[$log.Name]..($lines.Count - 1)]) {
                if ($line -notmatch $pattern) { continue }
                $text = "{0,-32} {1}" -f $run, $line
                if ($line -match '^Episode \d+ finished: pickaxe=True') { Write-Host $text -ForegroundColor Green }
                elseif ($line -match '^Episode \d+ finished') { Write-Host $text -ForegroundColor DarkYellow }
                elseif ($line -match '^(Error|Traceback)') { Write-Host $text -ForegroundColor Red }
                else { Write-Host $text }
            }
            $seen[$log.Name] = $lines.Count
        }
    }
    if ($scheduleStatus -and $scheduleStatus -ne "running") {
        Write-Host "The schedule has ended: $scheduleStatus." -ForegroundColor Cyan
        break
    }
    Start-Sleep -Seconds 2
}
