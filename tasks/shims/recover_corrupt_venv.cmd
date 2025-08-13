@ECHO OFF
WHERE pwsh > nul 2> nul
IF %ERRORLEVEL% NEQ 0 (
    WHERE powershell > nul 2> nul
    IF %ERRORLEVEL% NEQ 0 (
        C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -ExecutionPolicy Bypass -File "./tasks/shims/recover_corrupt_venv.ps1"
    ) ELSE (
        powershell -ExecutionPolicy Bypass -File "./tasks/shims/recover_corrupt_venv.ps1"
    )
) ELSE (
    pwsh -ExecutionPolicy Bypass -File "./tasks/shims/recover_corrupt_venv.ps1"
)
