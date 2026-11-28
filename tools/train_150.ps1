param(
    [string]$Output = "results/training-150",
    [ValidateSet("first-pass", "all", "off")]
    [string]$Preview = "first-pass"
)

Set-Location (Join-Path $PSScriptRoot "..")
$Host.UI.RawUI.WindowTitle = "Hermae | CUDA training | 150 epochs per model"
$python = (Resolve-Path ".venv/Scripts/python.exe").Path
if (Test-Path $Output) {
    $Output = "$Output-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
}
New-Item -ItemType Directory -Path $Output -ErrorAction Stop | Out-Null
$Output = (Resolve-Path $Output).Path
$refit = Join-Path $Output "refit"
$uv = Join-Path $Output "uv"
$epochs = 150
$previewArgs = @()
if ($Preview -ne "off") {
    $previewArgs = @("--live-preview", $Preview)
}

function Read-Report($Path) {
    Get-Content -Raw -LiteralPath $Path -ErrorAction Stop | ConvertFrom-Json
}

Write-Host "Hermae training on CUDA. Each model will complete $epochs epochs."
Write-Host "Results and live logs: $Output"
Write-Host "Stage 1/2  Galaxy detector, four host rankers, and optical transient classifier"
$refitArgs = @(
    "-u", "-m", "hermae.refit", "--v1", "../Hermae_v1",
    "--schedules", "results/quality-training", "--manifest", "results/ztf-development/manifest.json",
    "--supplemental-manifest", "results/ztf-cached-supplement/manifest.json",
    "--out", $refit, "--epochs", $epochs, "--device", "cuda"
)
& $python @refitArgs @previewArgs 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath (Join-Path $Output "refit.log")
if ($LASTEXITCODE -ne 0) { throw "Development refit failed. See refit.log." }

Write-Host "Stage 2/2  GALEX UV localizer"
$uvArgs = @(
    "-u", "-m", "hermae.train_uv", "--manifest", "results/expanded-archive/uv.json",
    "--out", $uv, "--epochs", $epochs, "--device", "cuda"
)
& $python @uvArgs @previewArgs 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath (Join-Path $Output "uv.log")
if ($LASTEXITCODE -ne 0) { throw "UV training failed. See uv.log." }

$galaxy = Read-Report (Join-Path $refit "galaxy/report.json")
$classical = Read-Report (Join-Path $refit "ranker-classical/report.json")
$unet = Read-Report (Join-Path $refit "ranker-unet/report.json")
$transient = Read-Report (Join-Path $refit "transient/report.json")
$localizer = Read-Report (Join-Path $uv "evaluation.json")
$completed = [ordered]@{
    galaxy = $galaxy.history.Count
    classical_image_only = $classical.models.image_only.history.Count
    classical_catalogue_fusion = $classical.models.catalogue_fusion.history.Count
    unet_image_only = $unet.models.image_only.history.Count
    unet_catalogue_fusion = $unet.models.catalogue_fusion.history.Count
    optical_transient = $transient.history.Count
    uv_localizer = $localizer.history.Count
}
if ($completed.Values | Where-Object { $_ -ne $epochs }) {
    throw "At least one training report did not record $epochs epochs."
}
$completed | ConvertTo-Json | Set-Content -Encoding ASCII -LiteralPath (Join-Path $Output "completed.json")
$completed | Format-Table
Write-Host "Verified $epochs completed epochs for all seven models."
