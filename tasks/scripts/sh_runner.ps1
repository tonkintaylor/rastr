if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "Git is not installed"
    exit 1
}

$gitPath = (Get-Command git).Path
# Recursively check parents of gitPath until we find that $parent\bin\bash.exe exists.
# Otherwise, give up and error out.
$gitInstallPath = $gitPath
while ($gitInstallPath -and -not (Test-Path "$gitInstallPath\bin\bash.exe")) {
    $gitInstallPath = Split-Path $gitInstallPath -Parent
}
if (-not $gitInstallPath) {
    Write-Error "Could not find git bash"
    exit 1
}
$gitBashPath = "$gitInstallPath\bin\bash.exe"

function RunShFileWithGitBash {
    param (
        [Parameter(Mandatory = $true)]
        [string]$ShFilePath,
        [Parameter(Mandatory = $false)]
        [string]$Args
    )
    $ThisArgs = $Args

    if (-not $ThisArgs) {
        $ThisArgs = ""
    }

    $process = Start-Process -FilePath $gitBashPath -ArgumentList "--login", "-c", "`"$ShFilePath $ThisArgs`"" -NoNewWindow -Wait -PassThru
    $exitCode = $process.ExitCode
    if ($exitCode -ne 0) {
        exit $exitCode
    }
}
