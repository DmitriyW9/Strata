# Troubleshooting

This page is for anyone whose install or startup fails. Find your symptom below. Each entry gives one fix.
Start from the route you installed: [desktop app or Terminal](INSTALL.md), or [source checkout](SOURCE_INSTALL.md).
The normal release uses the app or `strata launch` and needs no `.bat` file. Detailed engine errors are in the
[engine reference](DETAILS.md#troubleshooting).

**Quick check.** Run these and keep the output for a bug report:

```sh
strata --version
strata doctor --live
strata launch --status
```

## The desktop app and the Terminal

**I cannot find `START-HERE.bat` or `SETUP.bat`.**
Open the installed Strata app, or run `strata launch`. Those wrappers are in the source repository and are not the
startup path for the app. For a fresh desktop install, download the one installer for your system
(`Strata-0.2.6-windows-x64-setup.exe`, `Strata-0.2.6-linux-x86_64.AppImage` or `Strata-0.2.6-macos-arm64.dmg`), not a
`strata-engine-…` archive.

**`strata` is not found.**
On Windows, open a new PowerShell window after installing. On Linux or Mac, use the full path the installer printed,
normally `~/.local/bin/strata`, and add `~/.local/bin` to PATH if needed. Check `strata --version`.

**Strata could not find an engine for this computer.**
Run `strata doctor --live` and `strata update --check`. Check the selected GPU, driver and
[backend guide](INSTALL.md#what-you-need). Supported paths include Apple Silicon and Intel Arc on Linux; do not
assume every machine without an NVIDIA/AMD discrete card is unsupported. A model download can finish even when
no engine starts. Verify `strata launch --status` and `/health` before calling the installation successful.

**This model needs a newer engine.**
Unload running models, then use the app's engine update or `strata update`. Minimum engine versions vary by model and
backend. Check the model guide and the installed engine, not only the app version. If the feed has no compatible engine,
report the missing combination.

**An update says it succeeded but `strata --version` is still old.**
The Terminal switches to the updated Control on the next command, so run `strata --version` again. If it is still old,
run the one-line installer once more; it keeps models and settings. The app, Control and engine have separate version
numbers; see [updating](INSTALL.md#updating) and the [release notes](https://github.com/Niko1221/Strata/releases).

**A new engine misbehaves after an update.**
Stop the model, then use the app's rollback control or `strata components rollback`. Source-checkout engines use a
separate rollback workflow, `python setup.py --rollback-engine`, described in [source setup](SOURCE_INSTALL.md).

**The Mac app cannot be opened.**
Check that the download is for Apple Silicon and that macOS is 15 or newer. Follow
[Mac installation](MAC.md#the-desktop-app-and-strata-launch-on-a-mac). A warning alone does not establish that a file
is safe or intact: verify the download source and release before changing quarantine settings. Release signing and
test-build handling are described in [Mac packaging](../packaging/mac/README.md).

**The Linux AppImage does not open.**
Make the downloaded file executable. It may need FUSE 2 (`libfuse2` on Ubuntu 22.04, `libfuse2t64` on newer Ubuntu;
package names vary by distribution). Without FUSE, run the file with `--appimage-extract-and-run`. For a crash log,
run it with `--save-logs`; see the logs section below.

## While installing or starting

**A download stopped.**
Retry it in the app, or rerun the same `strata launch` command. Completed model-download steps are reused. Check
free space and the error message. For an interrupted Terminal installer, rerun the install command. Only a source
checkout uses `START-HERE.bat` / `./setup.sh` to resume source setup.

**Startup takes a long time or the computer becomes slow.**
Large models need time to check files, prepare packs and fill memory. A first launch can take many minutes; there is
no fixed startup time. Watch progress and the engine log. If memory pressure is high, close memory-heavy programs and try a
smaller model or context. DeepSeek V4.1-Flash needs 46 GB of RAM at least ([model page](MODELS.md#deepseek-v41-flash));
`strata launch --plan --family deepseek41 --model ist-das-2.00bpw` shows what your computer is missing.

**The NVIDIA driver is too old.**
Use the driver required by the engine package. CUDA 13 needs driver 580 or newer; older cards use the
[CUDA 12 route](OLDER_GPUS.md). After updating the driver and restarting if needed, open the app or rerun
`strata launch`.

**Port 8080 is already in use, or I changed a running model's settings.**
Check `strata launch --status`. To replace your existing launch, rerun the intended command with `--restart`;
it stops current answers. To run beside it, choose another port, for example `--port 8081`, if memory permits.
Do not stop an unrelated program just because it holds the port. `--yes` alone does not restart a running model.

**The engine stops on a small GPU or low-memory computer.**
Try a smaller pack or context. The draft layer also uses memory. In the app, set the model's Draft layer to Off;
in the Terminal, an explicit example is:

```sh
strata launch --family qwen --model q2_0 --context 32k --mtp off
```

Use `--restart` if replacing an active run. Turning MTP off can slow decoding. Keep some VRAM free for the display;
setting the reserve to zero is not a general speed fix.

**Pictures do not work.**
The model, backend and installed engine package must include a compatible picture reader. Inspect the startup
error and the model guide. `--vision off` disables reading pictures; it is different from optional
[image generation](IMAGES.md). In the Terminal, use `on|off|gpu|cpu`, not source setup's `yes|no` spelling.

**"Windows blocked the Strata engine (or the image encoder) ... Smart App Control".**
On a clean Windows 11, Smart App Control refuses programs that are not signed, and Strata's `strata.exe` and
`strata-vision.exe` are not (#735). Turn it off (Windows Security > App & browser control > Smart App Control
settings; it cannot be turned back on without a reset of Windows), or, if only the image encoder is blocked, run
`START-HERE.bat --setup --vision no`.

**A new engine misbehaves after an update.**
An update keeps the engine it replaced in `engine\.previous` (one generation, about 210 MiB).
`python setup.py --rollback-engine` puts it back (and keeps the newer one there: run it again to go forward) (#670).

**UPDATE.bat / update.sh say "git pull did not succeed", or "has no commit in common" (#1276).**
The repository's history was cleaned up on 2026-10-06, so a clone made before that cannot be updated with `git pull`.
The 0.1.40.2 scripts handle it: when no tracked file is edited they keep your old commits in the branch
`pre-cleanup-backup` and move the clone to the new history; otherwise they print the two commands for you. Your models,
settings and engine are untracked files and are never touched. An older UPDATE.bat / update.sh (before 0.1.40.2) cannot
do this itself: run these once in the Strata folder, then use UPDATE.bat as usual (add `git stash` first if `git status`
lists edited files):

    git fetch origin
    git branch pre-cleanup-backup
    git checkout -B main origin/main

**Python or the build tools could not be installed.**
Install what it names (links are printed), then run it again. Everything already done is kept.

**Linux: the engine does not compile (`unsupported GNU version`, or `exception specification is incompatible` for
`cospi`/`sinpi`/`rsqrt`).** Two known mismatches between the CUDA toolkit and a new Linux (#601):
- gcc newer than 14 (Ubuntu 26.04's default 15): CUDA 12.x and 13.0 refuse it. Install g++-14 and run
  `CXX=g++-14 CUDAHOSTCXX=g++-14 ./setup.sh` (CC, CXX and CUDAHOSTCXX name the build's compilers; setup passes them
  to CMake itself, so an already-configured build folder picks them up too, #1645).
- glibc 2.41 or newer with CUDA 12.x: CUDA and glibc's C23 math declarations disagree about `noexcept`, already in
  CMake's first compiler test. Setup pre-includes a host-only compatibility header for the CUDA 12 build; CUDA kernels
  and runtime behavior are unchanged.

**The first start takes minutes.**
It is reading 34-55 GB into RAM; the second start is faster while the files are in the OS cache. Started from Task
Scheduler, it can be 24x slower: see [Running it at startup](DETAILS.md#running-it-at-startup-task-scheduler).
On Linux the engine now asks the kernel to read the model files ahead (a cold start went from ~920 s to 70 s on one
PC); `STRATA_READ_AHEAD=0` turns that off. On Linux with transparent huge pages on `always`, the arena no longer asks
for `MADV_HUGEPAGE` on top (a fragmented machine spent minutes compacting memory, #771); `STRATA_NO_ARENA_THP=1` skips
that request on any setting.

**Setup says the engine does not match GitHub's checksum.**
Setup checks the downloaded ready-made engine against the SHA-256 GitHub publishes. On a mismatch the file is corrupt or was changed on the way, so setup deletes it and stops; run setup again to download it afresh. If GitHub gives no checksum (offline, rate limited, your own mirror), setup warns and installs anyway. `STRATA_SKIP_SHA256=1` skips the check, only if you insist.
142
=======
**Windows blocks an executable.**
Verify the official release and the component named in the message. Feed signatures and Windows code-signing are
different checks. If only the picture reader is blocked, `--vision off` may let text requests run. Do not disable
system-wide protection as the routine installation fix; include the blocked file and exact message in a bug report.

## While it answers

**It is slow and the disk stays busy.**
The chosen model may exceed available RAM/VRAM and be using the file tier. Close other programs, check memory in
the monitor, and choose a smaller model or context. Compare performance only with the same hardware and workload.

**The prompt exceeds the context.**
Start a new chat, shorten the input, or use the app's Summarize and continue action. A larger context needs more
memory. In the Terminal, restart your chosen model with an explicit `--context`, such as `64k`; preserve its family,
model and other choices. Do not look for `SETUP.bat` in an app install.

**On Linux the engine exits with code -9.**
Check system logs for an out-of-memory kill. Also check `ulimit -R`: a desktop-session real-time limit can cause a
kill even when memory is available. The engine prints a warning for that condition; see the
[detailed troubleshooting table](DETAILS.md#troubleshooting) before changing system limits.

**A hand-edited engine option disappears.**
The run config tracks explicit options in `user_args` and the generated baseline in `args_written`. Prefer the
app's Expert settings or Terminal `--set` for supported overrides. Context, KV, picture reading and drafting are
launch choices; set them through the app or the corresponding `strata launch` flags. Inspect the effective config
rather than assuming a saved preference was applied.

## Source-checkout problems

Python/venv, CUDA compiler versions, ROCm builds, `UPDATE.bat`, `update.sh` and generated `run-<model>` scripts
belong to [SOURCE_INSTALL.md](SOURCE_INSTALL.md). Use the backend guide for build prerequisites. The packaged
Terminal carries Python; a missing system Python is not a reason to install a compiler for the normal release.

If `git pull` fails in an old clone, back up your local work and follow the recovery steps the source update script prints.
Do not overwrite local branches or edits just to make `git pull` succeed.

## Logs and bug reports

- **App:** Settings → Engine → Diagnostics → View logs → Save logs. The diagnostic file goes to Downloads.
- **Terminal:** include `strata --version`, `strata doctor --live`, `strata launch --status`, and the relevant log excerpt.
- **Windows app crash:** inspect `%LOCALAPPDATA%\Strata\logs` for `desktop-crash-<date>.log`.
- **Linux app crash:** start the AppImage with `--save-logs` to collect diagnostics.

Include model and quantization, OS, GPU/VRAM, RAM, driver, install route, exact command, expected result and actual
error. Remove API keys and private data before sharing. Follow [CONTRIBUTING.md](../CONTRIBUTING.md).
