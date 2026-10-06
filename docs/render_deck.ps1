# Renders every slide of a .pptx to PNG using PowerPoint (for visual QA). Usage: render_deck.ps1 <deck.pptx> <outdir>
param([string]$Deck, [string]$OutDir)
$Deck = (Resolve-Path $Deck).Path
New-Item -ItemType Directory -Force $OutDir | Out-Null
$OutDir = (Resolve-Path $OutDir).Path
Get-ChildItem $OutDir -Filter *.PNG | Remove-Item -Force
$app = New-Object -ComObject PowerPoint.Application
$pres = $app.Presentations.Open($Deck, $true, $false, $false)
$pres.Export($OutDir, "PNG", 1600, 900)
$pres.Close()
$app.Quit()
Get-ChildItem $OutDir -Filter *.PNG | Sort-Object Name | ForEach-Object { $_.FullName }
