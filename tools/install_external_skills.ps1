$CodexHome = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $HOME ".codex" }
$Installer = Join-Path $CodexHome "skills/.system/skill-installer/scripts/install-skill-from-github.py"
<<<<<<< HEAD
Write-Host "== Installing / reminding external skills =="
$manifest = Get-Content skills.manifest.json | ConvertFrom-Json
foreach ($s in $manifest.external_skills) {
    $name = $s.name
    $method = if ($s.install_method) { $s.install_method } else { "skill_installer_git" }
    if ($method -eq "skill_installer_git") {
        if (-Not (Test-Path $Installer)) {
            Write-Host "[manual] $name: Codex skill installer not found: $Installer"
            Write-Host "         Repo: $($s.repo) Path: $($s.path) Ref: $($s.ref)"
            continue
        }
        Write-Host "[install] $name from $($s.repo):$($s.path)"
        python $Installer --repo $s.repo --ref $s.ref --path $s.path --method git
    }
    elseif ($method -eq "codex_prompt") {
        Write-Host "[prompt] $name: install inside Codex with:"
        Write-Host "         $($s.install_prompt)"
    }
    elseif ($method -eq "codex_plugin_marketplace") {
        Write-Host "[plugin] $name: install as Codex plugin:"
        Write-Host "         CLI: $($s.codex_cli_install)"
        Write-Host "         App: $($s.codex_app_install)"
    }
    else {
        Write-Host "[unknown] $name: install_method=$method. Check skills.manifest.json"
    }
}
Write-Host "After installation, restart Codex and run /skills or /plugins to verify availability."
=======
if (-Not (Test-Path $Installer)) {
    Write-Host "未找到 Codex skill installer: $Installer"
    Write-Host "请在 Codex 中使用 `$skill-installer，或手动安装 skills.manifest.json 中的外部 Skill。"
    exit 0
}
$manifest = Get-Content skills.manifest.json | ConvertFrom-Json
foreach ($s in $manifest.external_skills) {
    Write-Host "Installing $($s.name)..."
    python $Installer --repo $s.repo --ref $s.ref --path $s.path --method git
}
Write-Host "完成。请重启 Codex 并运行 /skills。"
>>>>>>> 1bd3e85eba289b200cbc1799c28eb5dd4f06b03f
