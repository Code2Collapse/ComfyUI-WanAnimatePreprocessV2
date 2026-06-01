# Vendored helpers from ComfyUI-CustomNodePacks/js/

These 5 files are COPIES of upstream sources in `ComfyUI-CustomNodePacks/js/`:
- `_c2c_theme.js`
- `_c2c_dialog.js`
- `_c2c_report.js`
- `_c2c_i18n.js`
- `_c2c_window.js`

## Why duplicated
Each ComfyUI custom-node pack is served under its own static URL. A relative
`import "./_c2c_theme.js"` from this pack cannot reach the other pack.
Vendoring is the simplest, dependency-free way for both packs to speak the
same advanced design language (theme tokens `C/T/z`, dialog helpers,
error-routing via `reportFailure`, i18n `t()`, draggable windows).

## Sync rule
**If you change any of these files in `ComfyUI-CustomNodePacks/js/`, copy
the new version here too.** No build-step automation — manual copy keeps
the dependency surface zero.

## How to verify
`_c2c_vendor_sanity.js` loads on every page open and logs
`[wanv2] vendored helpers OK` in the browser console.
