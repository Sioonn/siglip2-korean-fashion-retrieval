# Repository Safety Rules

- Do not read, list, stat, search, open, modify, delete, move, copy, or otherwise access these paths:
  - `/root/code/dust3r`
  - `/root/dataset`
- The restriction includes every file or directory below those paths.
- Do not follow symlinks or generated paths that resolve into those paths.
- If a task appears to require either path, stop and ask the user for explicit direction instead of inspecting it.
