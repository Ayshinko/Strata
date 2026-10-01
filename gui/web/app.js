/* Strata Manager - vanilla JS, no framework.  Talks to gui/manager.py's local JSON API only. */
"use strict";

const $ = (id) => document.getElementById(id);
const POLL_MS = 4000;

const state = { models: [], editor: null, status: { server: { state: "unknown" } }, tool: null,
                busy: false, lastStart: 0 };

/* ---------------------------------------------------------------- API */
async function api(path, opts) {
  const r = await fetch(path, Object.assign({ method: "GET", headers: { "Accept": "application/json" } }, opts));
  let j = {};
  try { j = await r.json(); } catch (e) { /* not json */ }
  if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
  if (j.ok === false) throw new Error(j.error || "request failed");
  return j.data !== undefined ? j.data : j;
}

function post(path, body) {
  return api(path, { method: "POST", headers: { "Content-Type": "application/json" },
                     body: JSON.stringify(body) });
}

/* ---------------------------------------------------------------- toast */
let toastTimer = null;
function toast(msg, kind) {
  const t = $("toast");
  t.textContent = msg;
  t.dataset.kind = kind || "";
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, kind === "error" ? 7000 : 3500);
}

/* ---------------------------------------------------------------- system + state pill */
async function refreshSystem() {
  try {
    const s = await api("/api/system");
    const g = s.gpus.map(g =>
      `${g.name} · ${Math.round(g.vram_used_gb || 0)}/${g.vram_gb.toFixed(0)} GB VRAM`).join(" · ");
    const ram = s.ram_gb ? ` · RAM ${s.ram_used_gb != null ? Math.round(s.ram_used_gb) + "/" : ""}${Math.round(s.ram_gb)} GB` : "";
    $("sys-line").textContent = `Strata v${s.strata_version} · ${s.cpu} · ${g || "no NVIDIA GPU"}${ram}`;
  } catch (e) { $("sys-line").textContent = "GPU / RAM unavailable: " + e.message; }
}

const PILL_TEXT = { "running": "running", "starting": "starting…", "stopped": "stopped",
                    "unknown": "…", "error": "error" };
function setPill(status, note) {
  const p = $("state-pill");
  p.dataset.state = status.state || "unknown";
  p.textContent = (PILL_TEXT[status.state] || status.state) + (note ? ` — ${note}` : "");
}

function refreshButtons() {
  const s = state.status.server;
  const on = s.state === "running" || s.state === "starting";
  $("start").disabled = state.busy || on || !state.models.length;
  $("stop").disabled = state.busy || !on || s.state === "stopped";
  $("restart").disabled = state.busy || !on;
  $("save").disabled = state.busy || !state.editor;
}

async function refreshStatus() {
  try {
    const st = await api("/api/status");
    state.status = st;
    state.tool = st.tool;
    setPill(st.server,
      st.server.state === "running" ? `port ${st.server.port}` :
      st.server.state === "starting" ? "model loading…" : "");
    refreshButtons();
    handleLog();
    handleTool();
  } catch (e) {
    setPill({ state: "error" }, e.message);
  }
}

/* ---------------------------------------------------------------- models list */
function renderModels(defaultName) {
  const ul = $("models");
  ul.textContent = "";
  const tpl = $("tpl-model");
  for (const m of state.models) {
    const li = tpl.content.firstElementChild ? tpl.content.firstElementChild.cloneNode(true)
                                             : document.createElement("li");
    li.className = "model";
    li.querySelector(".model__title").textContent = m.title;
    li.querySelector(".badge.quant").textContent = m.quant || m.family || "?";
    if (m.error) {
      li.querySelector(".model__meta").textContent = "unreadable: " + m.error;
      ul.appendChild(li);
      continue;
    }
    const cur = li.querySelector(".badge.current");
    cur.hidden = !(defaultName && m.config === defaultName);
    const meta = [];
    meta.push(`family ${m.family || "?"}`);
    meta.push(`context ${m.context ? m.context.toLocaleString() : "?"}`);
    if (m.kv) meta.push(`KV ${m.kv}`);
    meta.push(`vision ${m.vision || "off"}`);
    meta.push(m.low_ram !== "off" ? `low-RAM ${m.low_ram}` : "RAM-resident");
    meta.push(`port ${m.port}`);
    if (m.layer_split) meta.push(`GPUs ${Array.isArray(m.gpu) ? m.gpu.join("+") : m.gpu}`);
    if (!m.gguf_ok) meta.push("⚠ missing files");
    li.querySelector(".model__meta").textContent = meta.join("  ·  ");
    const gg = li.querySelector(".model__gguf");
    gg.textContent = m.gguf && m.gguf[0] ? m.gguf[0] : (m.pack ? "pack: " + m.pack : "");
    li.querySelector("button.select").addEventListener("click", () => selectModel(m.config));
    li.querySelector("button.edit").addEventListener("click", () => { selectModel(m.config); window.scrollTo({ top: 0, behavior: "smooth" }); });
    ul.appendChild(li);
  }
  ul.hidden = !state.models.length;
  if (!state.models.length) {
    const li = document.createElement("li");
    li.className = "model";
    li.innerHTML = `<div class="model__title">None yet</div>
      <div class="model__meta">Install a model with SETUP.bat / START-HERE.bat first — the Manager manages, it does not install.</div>`;
    ul.appendChild(li);
    ul.hidden = false;
  }
}

function renderModelSelect(defaultName) {
  const sel = $("model-sel");
  sel.textContent = "";
  for (const m of state.models) {
    const o = document.createElement("option");
    o.value = m.config;
    o.textContent = `${m.title}  (${m.quant || m.family || "?"} · ${m.context ? m.context.toLocaleString() : "?"} ctx)`;
    sel.appendChild(o);
  }
  sel.value = defaultName && state.models.length ? defaultName : (state.models[0] || {}).config || "";
}

/* ---------------------------------------------------------------- editor */
function fillEditor() {
  const e = state.editor, s = e.summary;
  $("ctx-num").value = e.context || 32768;
  renderChips(e);
  updateCtxHint();
  for (const r of document.querySelectorAll('input[name="vision"]')) r.disabled = false;
  const vis = e.vision || "off";
  for (const r of document.querySelectorAll('input[name="vision"]')) r.checked = r.value === vis;
  $("lowram").checked = e.low_ram;
  $("lowram-hint").textContent = e.low_ram
    ? (e.low_ram_resident ? "resident: the experts the GPU does not hold stay in RAM"
                         : "mapped: the experts are read from the model folder through the OS file cache")
    : "";
  $("kv-sel").value = (e.kv_options && e.kv_options.length && (e.kv === "fp16" ? "int8" : e.kv)) || "int8";
  $("kv-hint").textContent = e.kv_options && e.kv_options.length
    ? (e.kv === "fp16" ? "" : "above 8K context; the engine streams it from 64K up")
    : "below 8K context the engine uses fp16 (no flag)";
  $("kv-sel").disabled = !(e.kv_options && e.kv_options.length);
  fillGpu(e);
  $("port-num").value = e.port || 8080;
  $("host-sel").value = (e.host === "0.0.0.0") ? "0.0.0.0" : "127.0.0.1";
  $("api-key").value = e.api_key || "";
  $("vision-hint").textContent = e.vision_available ? "" : "Vision not installed for this model — run SETUP.bat --vision to add it (the Manager never downloads).";
  for (const r of document.querySelectorAll('input[name="vision"]')) r.disabled = !e.vision_available;
  const firstGguf = s.gguf && s.gguf[0];
  if (firstGguf && !$("gguf-path").value) $("gguf-path").value = firstGguf.replace(new RegExp("/?[^/\\\\]+$"), "");
}

function fillGpu(e) {
  const sel = $("gpu-sel");
  const layerSplit = Array.isArray(e.gpu);
  sel.textContent = "";
  const auto = document.createElement("option"); auto.value = "auto"; auto.textContent = "Auto (Strata's choice)";
  sel.appendChild(auto);
  for (const g of e.gpus || []) {
    const o = document.createElement("option");
    o.value = String(g.index);
    o.textContent = `GPU ${g.index} — ${g.name} (${g.vram_gb.toFixed(0)} GB)`;
    sel.appendChild(o);
  }
  if (layerSplit) {
    sel.disabled = true;
    $("gpu-hint").textContent = `layer split across ${e.gpu.length} GPUs — change with SETUP.bat --gpus`;
  } else {
    sel.disabled = false;
    sel.value = e.gpu !== undefined && e.gpu !== null ? String(e.gpu) : "auto";
    $("gpu-hint").textContent = "which card runs the model";
  }
}

function renderChips(e) {
  const box = $("ctx-chips");
  box.textContent = "";
  for (const c of e.presets) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    b.dataset.ctx = String(c);
    const k = c / 1024;
    b.textContent = (k >= 1024 ? k / 1024 : k) + (k >= 1024 ? "M" : "K") + (c > e.trained_context ? " ⚠" : "");
    b.title = c.toLocaleString() + (c > e.trained_context ? " tokens — past the trained 262144; the setup adds yarn rope scaling" : " tokens");
    b.addEventListener("click", () => {
      $("ctx-num").value = String(c);
      syncChips();
      updateCtxHint();
    });
    box.appendChild(b);
  }
  syncChips();
}

function syncChips() {
  const v = $("ctx-num").value;
  for (const b of $("ctx-chips").querySelectorAll("button.chip")) b.setAttribute("aria-checked", b.dataset.ctx === v ? "true" : "false");
}

function updateCtxHint() {
  const v = parseInt($("ctx-num").value, 10) || 0;
  $("ctx-hint").textContent = v > state.editor.trained_context
    ? "past the trained 262144: the setup adds yarn rope scaling (factor " + (v / 262144).toFixed(2) + ")"
    : v > 0 ? "tokens the model can see at once" : "";
  if (state.editor) {
    $("kv-hint").textContent = v > 8192 ? "above 8K context; the engine streams it from 64K up" : "below 8K context the engine uses fp16 (no flag)";
    $("kv-sel").disabled = !(v > 8192);
  }
}

async function selectModel(name) {
  if (!name) return;
  state.busy = true;
  try {
    const e = await api("/api/config?config=" + encodeURIComponent(name));
    state.editor = e;
    fillEditor();
    $("model-sel").value = name;
  } catch (err) { toast(err.message, "error"); }
  state.busy = false;
  refreshButtons();
}

function editorChanges() {
  const e = state.editor;
  const board = {
    config: e.summary.config,
    context: parseInt($("ctx-num").value, 10) || e.context,
    kv: $("kv-sel").value,
    vision: [...document.querySelectorAll('input[name="vision"]')].find(r => r.checked)?.value || "off",
    low_ram: $("lowram").checked,
    port: parseInt($("port-num").value, 10) || e.port,
    host: $("host-sel").value,
    api_key: $("api-key").value.trim(),
    gpu: $("gpu-sel").value,
  };
  if (Array.isArray(e.gpu)) delete board.gpu;      // a layer split is not edited here
  return board;
}

async function doSave() {
  if (!state.editor) return;
  state.busy = true; refreshButtons();
  try {
    await post("/api/save", editorChanges());
    toast("saved — Strata reads the same config file format", "ok");
    await reloadModels();
    await selectModel(state.editor.summary.config);
  } catch (err) { toast(err.message, "error"); }
  state.busy = false; refreshButtons();
}

/* ---------------------------------------------------------------- start / stop / restart */
async function doStart() { await doControl("start"); }
async function doStop()   { await doControl("stop"); }
async function doRestart(){ await doControl("restart"); }

async function doControl(kind) {
  if (!state.editor) return;
  state.busy = true; refreshButtons();
  try {
    const body = { config: state.editor.summary.config, port: parseInt($("port-num").value, 10) || undefined, open_chat: false };
    const r = kind === "stop" ? await post("/api/stop") : await post("/api/" + kind, body);
    if (r.error) { toast(r.error, "error"); return; }
    toast(kind === "start" ? "starting Strata — the model loads in the background" :
          kind === "stop" ? "stopped Strata" : "restarting Strata", "ok");
    await refreshStatus();
    if (kind === "stop") setTimeout(refreshStatus, 1200);
  } catch (err) { toast(err.message, "error"); }
  state.busy = false; refreshButtons();
}

/* ---------------------------------------------------------------- log + tool */
let logRefreshAt = 0;
async function handleLog() {
  const s = state.status.server;
  if ((s.state !== "starting" && s.state !== "running") || !s.log) { $("logcard").hidden = true; return; }
  const now = Date.now();
  if (now < logRefreshAt) return;
  logRefreshAt = now + 3000;
  $("logcard").hidden = false;
  $("log-where").textContent = s.config || "";
  try {
    const d = await api("/api/log?path=" + encodeURIComponent(s.log) + "&tail=140");
    const box = $("logbox");
    box.textContent = d.text || "(no output yet — the model is loading, this can take a minute or two)";
    box.scrollTop = box.scrollHeight;
  } catch (e) { /* transient */ }
}

function handleTool() {
  const t = state.tool;
  if (!t) { $("gguf-note").textContent = ""; return; }
  const alive = t.alive;
  const el = Math.round((Date.now() / 1000) - (t.started || 0));
  $("gguf-note").textContent = (alive ? "⏳ " : "✓ ") + t.what + (alive ? ` (${el}s — log: ${t.log})` : " finished — refresh the list");
  if (!alive) setTimeout(reloadModels, 1500);
}

/* ---------------------------------------------------------------- browse + custom GGUF */
let browsePath = "";
async function openBrowse(startPath) {
  const d = await api("/api/browse", { method: "POST", headers: { "Content-Type": "application/json" },
                                       body: JSON.stringify({ path: startPath || browsePath }) });
  browsePath = d.path;
  $("browse-path").value = d.path;
  $("browse-modal").hidden = false;
  renderDirs(d);
}

function renderDirs(d) {
  const ul = $("browse-dirs");
  ul.textContent = "";
  if (d.parent && d.parent !== d.path) {
    const up = document.createElement("li");
    up.appendChild(Object.assign(document.createElement("span"), { textContent: "↑ .." }));
    ul.appendChild(up);
    up.addEventListener("click", () => setBrowseDir(d.parent));
  }
  for (const dir of d.dirs) {
    const li = document.createElement("li");
    li.dataset.path = dir;
    const t = document.createElement("span");
    t.className = "dir";
    t.textContent = dir.replace(/^.*[\\/]/, "");
    li.appendChild(t);
    li.addEventListener("click", () => browsePreview(dir));
    ul.appendChild(li);
  }
  browsePreview(d.path);
}

async function setBrowseDir(path) {
  const d = await api("/api/browse", { method: "POST", headers: { "Content-Type": "application/json" },
                                       body: JSON.stringify({ path }) });
  browsePath = d.path;
  $("browse-path").value = d.path;
  renderDirs(d);
}

async function browsePreview(path) {
  browsePath = path;
  const dd = $("browse-detect");
  try {
    const det = await api("/api/gguf-detect", { method: "POST", headers: { "Content-Type": "application/json" },
                                                body: JSON.stringify({ path }) });
    if (det.ok) {
      dd.textContent = `✓ ${det.family_title} ${det.quant} — shards: ${det.shards.join(", ")}`;
      dd.dataset.ok = "1";
      $("gguf-prepare").hidden = !det.tag;
      state.ggufDetect = det;
    } else {
      dd.textContent = det.error;
      dd.dataset.ok = "0";
      $("gguf-prepare").hidden = true;
      state.ggufDetect = null;
    }
  } catch (e) { dd.textContent = e.message; }
}

function useBrowseFolder() {
  $("browse-modal").hidden = true;
  $("gguf-path").value = browsePath;
  browsePreview(browsePath);
}

async function doPrepareGguf() {
  const det = state.ggufDetect;
  if (!det) return;
  try {
    const r = await post("/api/prepare-gguf", { path: det.dir, context: parseInt($("ctx-num").value, 10) || 32768 });
    if (r.already) { toast(`already installed as ${r.config}`, "ok"); await selectModel(r.config); return; }
    toast("setup.py is preparing the model in the background — its log appears below");
    setTimeout(refreshStatus, 1500);
  } catch (e) { toast(e.message, "error"); }
}

async function reloadModels(keep) {
  const d = await api("/api/models");
  state.models = d.models;
  const defaultName = keep && state.editor ? state.editor.summary.config : d.default;
  renderModels(defaultName);
  renderModelSelect(defaultName);
  if (keep && state.editor) $("model-sel").value = state.editor.summary.config;
  return defaultName;
}

async function boot() {
  try { await refreshSystem(); } catch (e) { /* non-fatal */ }
  try {
    const defaultName = await reloadModels();
    if (defaultName) await selectModel(defaultName);
    else toast("No models installed yet — SETUP.bat first.", "error");
  } catch (e) { toast(e.message, "error"); }
  await refreshStatus();
  setInterval(refreshStatus, POLL_MS);
  setInterval(async () => { if (!state.busy) await reloadModels(true); }, POLL_MS * 8);
}

/* ---------------------------------------------------------------- wire up */
$("model-sel").addEventListener("change", (ev) => selectModel(ev.target.value));
$("ctx-num").addEventListener("input", () => { syncChips(); updateCtxHint(); });
$("save").addEventListener("click", doSave);
$("start").addEventListener("click", doStart);
$("stop").addEventListener("click", doStop);
$("restart").addEventListener("click", doRestart);
$("gguf-browse").addEventListener("click", () => openBrowse($("gguf-path").value || ""));
$("browse-close").addEventListener("click", () => { $("browse-modal").hidden = true; });
$("browse-use").addEventListener("click", useBrowseFolder);
$("browse-go").addEventListener("click", () => setBrowseDir($("browse-path").value.trim()));
$("browse-path").addEventListener("keydown", (ev) => { if (ev.key === "Enter") setBrowseDir($("browse-path").value.trim()); });
$("gguf-prepare").addEventListener("click", doPrepareGguf);

boot();