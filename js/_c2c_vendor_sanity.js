// Track B sanity-check: confirm the 5 vendored helper modules load without
// error and expose their public surface. Logs once at page load. No-op on
// failure (helpers are optional polish — never crash the pack on them).

import { app } from "../../scripts/app.js";

const TAG = "[wanv2 vendored]";

(async () => {
    try {
        const [theme, dialog, report, i18n, win] = await Promise.all([
            import("./_c2c_theme.js"),
            import("./_c2c_dialog.js"),
            import("./_c2c_report.js"),
            import("./_c2c_i18n.js"),
            import("./_c2c_window.js"),
        ]);

        const missing = [];
        if (!theme || typeof theme.C === "undefined") missing.push("_c2c_theme.C");
        if (!theme || typeof theme.T === "undefined") missing.push("_c2c_theme.T");
        if (!theme || typeof theme.z === "undefined") missing.push("_c2c_theme.z");
        if (!dialog || typeof dialog.c2cConfirm !== "function") missing.push("_c2c_dialog.c2cConfirm");
        if (!report || typeof report.reportFailure !== "function") missing.push("_c2c_report.reportFailure");
        if (!i18n || typeof i18n.t !== "function") missing.push("_c2c_i18n.t");
        if (!win || typeof win.makeDraggable !== "function") missing.push("_c2c_window.makeDraggable");

        if (missing.length) {
            console.warn(`${TAG} loaded with missing exports:`, missing);
        } else {
            console.info(`${TAG} OK — theme C/T/z + dialog + report + i18n + window all present`);
        }
    } catch (err) {
        console.warn(`${TAG} failed to load:`, err && err.message ? err.message : err);
    }
})();

// Touch `app` so the import isn't tree-shaken if a bundler ever runs.
void app;
