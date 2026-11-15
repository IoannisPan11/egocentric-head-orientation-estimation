<#
00_export_recordings_leg0.ps1

Exports the leg0 eye-tracker recordings of the ET - Upper Body dataset with pl-rec-export, working on
copies so that the raw recordings are never modified.

Thesis step : Frame extraction (Section 3.2.2)
Dataset     : ET - Upper Body (explored, not used for the final model)
Run on      : Windows workstation at TU Wien (PowerShell)
Note        : "leg0" is the naming used in the code and folders; it corresponds to
              "video0 (leg1)" in the thesis.

About the "Neon" file names
- The recordings come from Pupil Labs Invisible glasses. pl-rec-export expects the scene-camera files
  to have the Neon-style names "Neon Scene Camera v1 ps1.mp4" / ".time", so the script renames
  world.mp4 / world.time to these names inside the working copy. "Neon" in this script refers only to
  these file names, not to the device that was used.

Purpose
- Safely export ET recordings using pl-rec-export across the entire
  D:\upper_body_sync_2 dataset.
- Ensures original raw recordings remain untouched.

Strategy
- For every folder containing info.json (i.e., valid Pupil Labs recording):
  1) Create a copy with suffix "_WORK" (if not already existing).
  2) Rename world files inside the _WORK copy to match Neon naming
     expected by pl-rec-export (if required).
  3) Run pl-rec-export on the _WORK folder with force overwrite enabled.

Inputs
- $root: dataset root directory (D:\upper_body_sync_2)
- Requires pl-rec-export installed in the active Python virtual environment.

Processing Logic
- Only recordings with info.json are considered valid.
- Copy happens once (idempotent behavior).
- Export is executed only on _WORK folders.
- Originals are never modified.

Outputs
- Each recording will contain:
    <recording>_WORK\export\
        events.csv
        gaze.csv
        imu.csv
        world.csv
        template.json

Safety
- Original RAW recordings remain untouched.
- Export overwrites only inside _WORK folder (-f flag).
- Script can be re-run safely.

Dependencies
- PowerShell
- Active .venv with pl-rec-export available in PATH
- pl-rec-export (the Invisible recordings are given Neon-style file names first, see above)

Notes
- The recordings are Pupil Labs Invisible recordings. If their world files use the legacy names
  (world.mp4, world.time), they are renamed to the Neon-style names before export.
#>

$root = "D:\upper_body_sync_2"

Get-ChildItem $root -Recurse -Directory | Where-Object {
    Test-Path "$($_.FullName)\info.json"
} | ForEach-Object {

    $rec = $_.FullName
    $work = "${rec}_WORK"

    Write-Host "Processing $rec"

    # 1) Copy if not already copied
    if (!(Test-Path $work)) {
        Copy-Item $rec $work -Recurse
    }

    # 2) Rename world files inside WORK
    #    The Invisible files get the Neon-style names that pl-rec-export expects.
    $worldTime = Join-Path $work "world.time"
    $worldMp4  = Join-Path $work "world.mp4"

    if (Test-Path $worldTime) {
        Rename-Item $worldTime "Neon Scene Camera v1 ps1.time" -ErrorAction SilentlyContinue
    }

    if (Test-Path $worldMp4) {
        Rename-Item $worldMp4 "Neon Scene Camera v1 ps1.mp4" -ErrorAction SilentlyContinue
    }

    # 3) Run export
    pl-rec-export $work -f
}