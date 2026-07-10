$CodexHome = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $HOME ".codex" }
$Installer = Join-Path $CodexHome "skills/.system/skill-installer/scripts/install-skill-from-github.py"
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
