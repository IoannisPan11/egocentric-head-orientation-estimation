<#
00_extract_frames.ps1

Exports every frame of the scene-camera video of each recording as a PNG image, using FFmpeg.

Thesis step : Frame extraction (Section 3.2.2)
Dataset     : ET - Head Mounted
Run on      : Windows workstation at TU Wien (PowerShell + FFmpeg)

Purpose
- Export all frames from each recording's "PI world" video (the scene camera of the Pupil Labs
  Invisible glasses) to PNG images using FFmpeg, with a safe no-overwrite policy that avoids
  mixing partial outputs.

Inputs
- $projectRoot: dataset root containing recording folders.
- $okList: text file listing recordings to process (one path per line). Expected at:
    $projectRoot\results\export_ok.txt
  Each line should be an absolute path to a recording folder (timestamp folder) that contains
  a "PI world*.mp4" (preferred) or "PI world*.mjpeg" file.

Processing
- For each recording path in export_ok.txt:
  1) Verify the path exists.
  2) Locate the world video:
     - Prefer the first match of "PI world*.mp4"
     - Fallback to the first match of "PI world*.mjpeg"
     - Skip the recording if neither exists.
  3) Create a deterministic output folder name by converting the recording's relative path
     (w.r.t. $projectRoot) into a filesystem-safe string (non [A-Za-z0-9_] -> "_").
  4) Safe/no-overwrite policy:
     - If the output folder already contains >= $minFramesToTreatAsComplete PNGs, treat it as complete
       and SKIP the recording.
     - If it contains a smaller number of PNGs, treat it as partial, delete the existing PNGs, and
       re-export from scratch to avoid mixed/duplicated frame numbering.
  5) Run FFmpeg to export all frames to:
       frame_%06d.png

Outputs
- PNG frames are written under:
    $outRoot\<safe_recording_key>\frame_%06d.png
- A timestamped log file is written under:
    $logRoot\frames_export_YYYYMMDD_HHMMSS.log
  containing per-recording status and any FFmpeg failures.

Dependencies / Assumptions
- FFmpeg is available on PATH (or $env:Path is configured before running).
- This script exports every frame (no FPS downsampling). Output can be very large.
- $minFramesToTreatAsComplete is a heuristic threshold; increase it if you want stricter "complete" detection.

Notes
- Frame timestamps are not exported here. They come from `export/world.csv` and are matched to the
  frames by index in step 09 (frame_000001.png corresponds to the first data row of world.csv).
#>


'''
       '# ================================)
# FRAME EXPORT (PNG) - SAFE MODE
# Output: D:\john.panagiotopoulos\ET-IMU-and-head-mounted-imu\results\frames
# Input list: .\results\export_ok.txt  (relative to project root on E:)
# NO OVERWRITE: skips folders that look complete, re-runs folders that are partial
# ================================

# --- 0) SETTINGS ---
$projectRoot = "E:\john\ET-IMU-and-head-mounted-imu"   # your dataset root (input)
$okList      = Join-Path $projectRoot "results\export_ok.txt"

$outRoot     = "D:\john.panagiotopoulos\ET-IMU-and-head-mounted-imu\results\frames"
$logRoot     = "D:\john.panagiotopoulos\ET-IMU-and-head-mounted-imu\results\logs"

# If a folder has at least this many frames, consider it COMPLETE and SKIP.
# (For your first two, it's 55k+ and 90k+, so 1000 is safe. Raise if you want stricter.)
$minFramesToTreatAsComplete = 1000

# FFmpeg: assume it's already in PATH (you did $env:Path=... earlier).
# If not, uncomment and set this:
# $env:Path = "C:\Users\john.panagiotopoulos\Downloads\ffmpeg-master-latest-win64-gpl-shared\bin;" + $env:Path

# --- 1) PREP ---
if (-not (Test-Path $okList)) {
    Write-Host "ERROR: export_ok.txt not found at: $okList" -ForegroundColor Red
    return
}

New-Item -ItemType Directory -Force -Path $outRoot | Out-Null
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

$stamp   = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $logRoot "frames_export_$stamp.log"

"Started: $(Get-Date)" | Out-File -FilePath $logFile -Encoding UTF8
"projectRoot=$projectRoot" | Out-File -FilePath $logFile -Append -Encoding UTF8
"outRoot=$outRoot" | Out-File -FilePath $logFile -Append -Encoding UTF8
"minFramesToTreatAsComplete=$minFramesToTreatAsComplete" | Out-File -FilePath $logFile -Append -Encoding UTF8
"okList=$okList" | Out-File -FilePath $logFile -Append -Encoding UTF8
"" | Out-File -FilePath $logFile -Append -Encoding UTF8

$paths = Get-Content $okList | Where-Object { $_.Trim() -ne "" }
Write-Host "OK recordings loaded: $($paths.Count)" -ForegroundColor Cyan
"OK recordings loaded: $($paths.Count)" | Out-File -FilePath $logFile -Append -Encoding UTF8

# --- helper: count frames fast ---
function Get-FrameCount($dir) {
    if (-not (Test-Path $dir)) { return 0 }
    try {
        return (Get-ChildItem -Path $dir -Filter "*.png" -File -ErrorAction SilentlyContinue).Count
    } catch {
        return 0
    }
}

# --- 2) LOOP ---
$idx = 0
foreach ($rec in $paths) {
    $idx++
    Write-Host "`n[$idx/$($paths.Count)] $rec" -ForegroundColor Cyan
    "[$idx/$($paths.Count)] $rec" | Out-File -FilePath $logFile -Append -Encoding UTF8

    if (-not (Test-Path $rec)) {
        Write-Host "  -> SKIP (path not found)" -ForegroundColor Yellow
        "  -> SKIP (path not found)" | Out-File -FilePath $logFile -Append -Encoding UTF8
        continue
    }

    # Find world video (prefer mp4, fallback to mjpeg)
    $worldMp4 = Get-ChildItem -Path $rec -File -Filter "PI world*.mp4" -ErrorAction SilentlyContinue | Select-Object -First 1
    $worldMj  = $null
    if (-not $worldMp4) {
        $worldMj = Get-ChildItem -Path $rec -File -Filter "PI world*.mjpeg" -ErrorAction SilentlyContinue | Select-Object -First 1
    }

    if (-not $worldMp4 -and -not $worldMj) {
        Write-Host "  -> SKIP (no PI world video)" -ForegroundColor Yellow
        "  -> SKIP (no PI world video)" | Out-File -FilePath $logFile -Append -Encoding UTF8
        continue
    }

    $video = if ($worldMp4) { $worldMp4.FullName } else { $worldMj.FullName }

    # Build safe output folder name from relative path
    $rel  = $rec.Replace("$projectRoot\", "")
    $safe = ($rel -replace "[^A-Za-z0-9_]", "_")
    $outDir = Join-Path $outRoot $safe
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    # NO OVERWRITE logic:
    # - if >= minFramesToTreatAsComplete -> SKIP
    # - else -> delete partial pngs and re-export cleanly (prevents mixed numbering)
    $existingCount = Get-FrameCount $outDir
    if ($existingCount -ge $minFramesToTreatAsComplete) {
        Write-Host "  -> SKIP (already has $existingCount png frames)" -ForegroundColor Green
        "  -> SKIP (already has $existingCount png frames)" | Out-File -FilePath $logFile -Append -Encoding UTF8
        continue
    }
    elseif ($existingCount -gt 0) {
        Write-Host "  -> PARTIAL ($existingCount frames). Cleaning partial output and re-exporting..." -ForegroundColor Yellow
        "  -> PARTIAL ($existingCount frames). Cleaning partial output and re-exporting..." | Out-File -FilePath $logFile -Append -Encoding UTF8
        Remove-Item -Path (Join-Path $outDir "*.png") -Force -ErrorAction SilentlyContinue
    }

    Write-Host "  -> exporting from: $video" -ForegroundColor Magenta
    "  -> exporting from: $video" | Out-File -FilePath $logFile -Append -Encoding UTF8

    # Export PNG frames
    $outPattern = Join-Path $outDir "frame_%06d.png"

    & ffmpeg -hide_banner -loglevel error -i "$video" "$outPattern"
    $code = $LASTEXITCODE

    if ($code -ne 0) {
        Write-Host "  -> FAILED (ffmpeg exit code $code)" -ForegroundColor Red
        "  -> FAILED (ffmpeg exit code $code)" | Out-File -FilePath $logFile -Append -Encoding UTF8
        continue
    }

    $count = Get-FrameCount $outDir
    Write-Host "  -> OK ($count frames)" -ForegroundColor Green
    "  -> OK ($count frames)" | Out-File -FilePath $logFile -Append -Encoding UTF8
}

"`nFinished: $(Get-Date)" | Out-File -FilePath $logFile -Append -Encoding UTF8
Write-Host "`nDONE. Log: $logFile" -ForegroundColor Cyan
'''