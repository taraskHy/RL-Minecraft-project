param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ProjectArgs
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:PYTHONUNBUFFERED = '1'
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'

try {
    if (-not (Test-Path -LiteralPath $projectPython)) {
        $candidates = @(
            @{ Name = 'py'; Prefix = @('-3.12') },
            @{ Name = 'py'; Prefix = @('-3') },
            @{ Name = 'python'; Prefix = @() }
        )
        $created = $false
        foreach ($candidate in $candidates) {
            $executable = Get-Command $candidate.Name -ErrorAction SilentlyContinue
            if (-not $executable) { continue }
            $prefix = $candidate.Prefix
            try {
                & $executable.Source @prefix -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' 2>$null
            } catch {
                # A missing py -3.12 installation writes stderr on PowerShell 5.
                # Continue to py -3 / python instead of aborting the entire setup.
                continue
            }
            if ($LASTEXITCODE -ne 0) { continue }
            Write-Host 'Creating this project''s Python environment...'
            & $executable.Source @prefix -m venv (Join-Path $projectRoot '.venv')
            if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
            $created = $true
            break
        }
        if (-not $created) {
            throw 'Install Python 3.12 or newer from https://www.python.org/downloads/windows/ (include the Python launcher), then run setup.cmd again.'
        }
    }
    & $projectPython -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)'
    if ($LASTEXITCODE -ne 0) { throw 'The project .venv is unusable or older than Python 3.12. Rename .venv and run setup.cmd to recreate it.' }
    # Native stderr carries real diagnostics, not PowerShell exceptions. Keep
    # the Python command's exit code instead of converting every error into 1.
    $ErrorActionPreference = 'Continue'
    & $projectPython -m infrastructure @ProjectArgs
    exit $LASTEXITCODE
} catch {
    Write-Host "Setup/launcher error: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
