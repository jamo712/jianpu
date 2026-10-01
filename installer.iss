[Setup]
AppName=三角洲口风琴
AppVersion=1.0.0
DefaultDirName={autopf}\三角洲口风琴
DefaultGroupName=三角洲口风琴
OutputDir=build\installer
OutputBaseFilename=三角洲口风琴-Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64
SetupIconFile={#SourcePath}\static\app.ico
UninstallDisplayIcon={app}\三角洲口风琴.exe

[Files]
Source: "build\三角洲口风琴\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\三角洲口风琴"; Filename: "{app}\三角洲口风琴.exe"; IconFilename: "{app}\三角洲口风琴.exe"
Name: "{autodesktop}\三角洲口风琴"; Filename: "{app}\三角洲口风琴.exe"; IconFilename: "{app}\三角洲口风琴.exe"

[Run]
Filename: "{app}\三角洲口风琴.exe"; Description: "启动三角洲口风琴"; Flags: nowait postinstall skipifsilent
