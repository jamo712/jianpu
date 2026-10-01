param(
    [switch]$InstallPyInstaller,
    [switch]$SkipFrontend,
    [switch]$SkipInno
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Frontend = Join-Path $Root "jpeditor"
$Build = Join-Path $Root "build"
# Keep the original Chinese program name. The Qt prefix is a relative ASCII path.
$ExeName = -join ([char]0x4E09, [char]0x89D2, [char]0x6D32, [char]0x53E3, [char]0x98CE, [char]0x7434)
$Bundle = Join-Path $Build $ExeName

function Require-Command([string]$Name, [string]$Hint) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Missing command: $Name. $Hint"
    }
}

Require-Command "python" "Install Python 3.10+ and add it to PATH."
$PythonHome = Split-Path -Parent (Get-Command python).Source
$CondaLibrary = Join-Path $PythonHome "Library\bin"
if (Test-Path $CondaLibrary) {
    # conda keeps ffi/sqlite/expat here. Without them on PATH, PyInstaller omits the DLLs
    # and the frozen app exits while importing ctypes.
    $env:Path = "$CondaLibrary;$env:Path"
}
if (-not $SkipFrontend) {
    Require-Command "npm" "Install Node.js 18+ and add it to PATH."
    Push-Location $Frontend
    try {
        npm run build
    } finally {
        Pop-Location
    }
}

$PyInstaller = Get-Command "pyinstaller" -ErrorAction SilentlyContinue
if (-not $PyInstaller -and $InstallPyInstaller) {
    python -m pip install pyinstaller
    $PyInstaller = Get-Command "pyinstaller" -ErrorAction SilentlyContinue
}
if (-not $PyInstaller) {
    throw "PyInstaller is missing. Run 'python -m pip install pyinstaller' or use -InstallPyInstaller."
}

$FrontendDist = Join-Path $Frontend "dist"
if (-not (Test-Path (Join-Path $FrontendDist "index.html"))) {
    throw "Missing jpeditor/dist/index.html. Run npm run build first."
}

New-Item -ItemType Directory -Force -Path $Build | Out-Null
Remove-Item -LiteralPath $Bundle -Recurse -Force -ErrorAction SilentlyContinue

$PyArgs = @(
    "--noconfirm", "--clean", "--windowed",
    "--name", $ExeName,
    "--icon", (Join-Path $Root "static\app.ico"),
    "--runtime-hook", (Join-Path $Root "rthook_qt_prefix.py"),
    "--distpath", $Build,
    "--workpath", (Join-Path $Build "pyi-work"),
    "--specpath", $Build,
    "--add-data", "$Root\static;static",
    "--add-data", "$FrontendDist;jpeditor/dist",
    "--exclude-module", "torch",
    "--exclude-module", "torchvision",
    "--exclude-module", "torchaudio",
    "--exclude-module", "transformers",
    "main.py"
)
& pyinstaller @PyArgs

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE. No usable application was created."
}

$ExpectedExe = Join-Path $Bundle ($ExeName + ".exe")
if (-not (Test-Path -LiteralPath $ExpectedExe)) {
    throw "PyInstaller did not create the expected executable: $ExpectedExe"
}

$Internal = Join-Path $Bundle "_internal"
$QtBin = Join-Path $Internal "PyQt5\Qt5\bin"
if (Test-Path $CondaLibrary) {
    foreach ($dll in @("ffi.dll", "liblzma.dll", "libbz2.dll", "libexpat.dll", "sqlite3.dll")) {
        $src = Join-Path $CondaLibrary $dll
        if (Test-Path -LiteralPath $src) {
            Copy-Item -LiteralPath $src -Destination (Join-Path $Internal $dll) -Force
        }
    }
}
# Qt ships an old MSVCP140. Image recognition loads a newer C++ runtime and then crashes
# inside that old DLL. Replace it with the system copy, which stays compatible with Qt.
foreach ($dll in @("msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "vcruntime140.dll", "vcruntime140_1.dll", "vcruntime140_threads.dll", "concrt140.dll")) {
    $src = Join-Path $env:SystemRoot "System32\$dll"
    if (-not (Test-Path -LiteralPath $src)) {
        $src = Join-Path $PythonHome $dll
    }
    if (-not (Test-Path -LiteralPath $src) -and (Test-Path $CondaLibrary)) {
        $src = Join-Path $CondaLibrary $dll
    }
    if (Test-Path -LiteralPath $src) {
        Copy-Item -LiteralPath $src -Destination (Join-Path $Internal $dll) -Force
        if (Test-Path $QtBin) {
            Copy-Item -LiteralPath $src -Destination (Join-Path $QtBin $dll) -Force
        }
    }
}

if (-not $SkipInno) {
    $IsccCommand = Get-Command "ISCC.exe" -ErrorAction SilentlyContinue
    if ($IsccCommand) {
        $IsccPath = $IsccCommand.Source
    } else {
        $IsccPath = $null
    }
    if (-not $IsccPath) {
        $Candidates = @(
            "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
            "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
        )
        $UninstallRoots = @(
            "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*",
            "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*",
            "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*"
        )
        $InnoInstall = Get-ItemProperty $UninstallRoots -ErrorAction SilentlyContinue |
            Where-Object { $_.DisplayName -match "Inno Setup" -and $_.InstallLocation } |
            Select-Object -First 1 -ExpandProperty InstallLocation
        if ($InnoInstall) {
            $Candidates += (Join-Path $InnoInstall "ISCC.exe")
        }
        $IsccPath = $Candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    }
    if ($IsccPath) {
        & $IsccPath (Join-Path $Root "installer.iss")
    } else {
        Write-Warning "Inno Setup was not found. Portable output is ready at: $Bundle"
        Write-Warning "Install Inno Setup and run this script again to create the installer."
    }
}

Write-Host "Build complete: $Bundle"
