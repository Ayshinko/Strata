#!/usr/bin/env python3
"""Strata Manager - an optional local GUI for managing a Strata install (Windows and Linux).

It is a thin management layer ONLY.  It does not contain a second inference engine or a duplicate setup
system: everything Strata already knows - the installed models, the config format, the model families, the
context/rope/Vision/low-RAM semantics, the system checks - is read from setup.py (`import setup`), and the
manager's "Save" writes the SAME strata-*.json files that setup.py writes, edited surgically.

  - Model discovery....... each strata-*.json in the Strata folder, parsed with setup.choices_from_config()
                            (the same function setup.py uses to read back its own configs)
  - Context / Vision / Low-RAM / KV / GPU ..... edited through the existing config keys/args, never a second
                            settings system
  - Custom GGUF ........... only detected here (shards, family, size): preparing it runs the EXISTING setup.py
                            pipeline (`setup.py --family ... --model ... --gguf-dir ... --no-start --yes`)
  - Start / Stop / Restart  a tiny supervisor: launches serve/server.py (the exact command the run-*.bat
                            scripts use) detached, remembers its PID, stops it by closing the process tree -
                            the same outcome as closing Strata's own console window.  Nothing is added to
                            serve/server.py; Strata's own port-lock check also prevents duplicate instances.

The server binds http://127.0.0.1:<port>/ only (default port 8275).  No secrets leave this PC.

    python gui/manager.py            (or double-click START-MANAGER.bat / run ./start-manager.sh)

Options: --port N, --root DIR (another Strata folder; mainly for tests), --no-browser.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# The Strata folder this manager belongs to; setup.py's model/config/rope/Vision/Low-RAM logic is imported
# (never duplicated), so the Manager and Strata always agree on what is possible.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import setup  # noqa: E402  (setup.py is written to be importable: see tools/test_setup_rope.py)

DEFAULT_PORT = 8275                     # the Manager's own port (Strata's server keeps its 8080)
BACKUP_SUFFIX = ".bak-"                 # strata-<m>.json.bak-<timestamp>, like setup's own backups
KEEP_BACKUPS = 5
TRAINED_CONTEXT = 262144                # the model's trained length (setup.py's rope "trained" default)
# the contexts setup.py offers (its CONTEXTS), plus 16K - a valid --max-context setup accepts without rope
PRESET_CONTEXTS = sorted(set([8192, 16384, 32768, 65536, 131072, 262144]) | set(setup.CONTEXTS))

# the supervisor (Start/Stop/Restart, the runtime state, the server's log) lives in gui/launcher.py and is
# platform-neutral there - the OS-specific half is gui/platforms/
from gui import launcher                                     # noqa: E402
from gui.launcher import server_status, start_server, stop_server, restart_server, run_tool, tail_lines  # noqa: E402


# ====================================================================================================== small helpers
def arg_index(args: list, key: str) -> int:
    """The index of `key` in an args list, or -1."""
    try:
        return args.index(key)
    except ValueError:
        return -1


def arg_val(args: list, key: str, default=None):
    """The value after `key` in an engine args list (the way setup.py's choices_from_config reads it)."""
    i = arg_index(args, key)
    return args[i + 1] if i >= 0 and i + 1 < len(args) else default


def set_arg(args: list, key: str) -> list:
    """Add a flag (no value) to the args list, keeping its current position if already there."""
    return list(args) if arg_index(args, key) >= 0 else list(args) + [key]


def set_arg_val(args: list, key: str, value) -> list:
    """Add or replace a `--key value` pair, preserving the list's other entries in place."""
    out = list(args)
    i = arg_index(out, key)
    if i >= 0:
        if i + 1 < len(out):
            out[i + 1] = str(value)
            return out
        del out[i]                       # a dangling flag at the very end: drop it and append properly
    return out + [key, str(value)]


def drop_arg(args: list, key: str) -> list:
    """Remove a `--key [value]` pair (a value is everything up to the next flag)."""
    out, i = list(args), arg_index(args, key)
    if i < 0:
        return out
    if i + 1 < len(out) and not str(out[i + 1]).startswith("--"):
        del out[i:i + 2]
    else:
        del out[i]
    return out


def read_config(path: Path) -> dict:
    """A strata-*.json config, as JSON (BOM-tolerant, like setup.py)."""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def vision_mode(cfg: dict) -> str:
    """The config's Vision setting in the semantics setup.py uses: 'off' | 'gpu' | 'cpu'."""
    vis = cfg.get("vision")
    return "gpu" if isinstance(vis, dict) and vis.get("gpu") else ("cpu" if isinstance(vis, dict) else "off")


def low_ram_mode(cfg: dict) -> str:
    """The config's low-RAM mode: 'resident' | 'mmap' | 'off' (from the flag setup.py writes)."""
    args = cfg.get("args", [])
    if "--resident-experts" in args:
        return "resident"
    if "--mmap-experts" in args:
        return "mmap"
    return "off"


def kv_mode(cfg: dict):
    """The config's KV cache precision ('fp16' when no --kv flag: what setup writes below 8K context)."""
    args = cfg.get("args", [])
    ctx = arg_val(args, "--max-context")
    kv = arg_val(args, "--kv")
    return kv if kv else ("fp16" if (ctx is None or int(ctx) <= 8192) else "int8")


def gguf_paths(cfg: dict) -> list:
    """The GGUF shards the config points at (--native and --ple-gguf, in setup.py's order)."""
    args = cfg.get("args", [])
    return [p for p in (arg_val(args, "--native"), arg_val(args, "--ple-gguf")) if p]


# ====================================================================================================== model discovery
def model_summary(cfg_path: Path) -> dict:
    """Everything reliably known about one installed model, from its own config file - no invented metadata."""
    try:
        cfg = read_config(cfg_path)
    except (OSError, ValueError) as e:
        return {"config": cfg_path.name, "title": cfg_path.stem[len("strata-"):], "error": str(e)}
    ch = setup.choices_from_config(cfg_path)
    args = cfg.get("args", [])
    ctx = ch["context"]
    return {
        "config": cfg_path.name,
        "title": cfg.get("model_name") or cfg_path.stem[len("strata-"):],
        "family": ch["family"],
        "quant": ch["model"],                                # setup's MODELS key (Q2_0, IQ3_XXS, ...)
        "quant_about": setup.MODELS.get(ch["model"], {}).get("about"),
        "gguf": gguf_paths(cfg),
        "pack": arg_val(args, "--pack"),
        "tokenizer": cfg.get("tokenizer"),
        "context": ctx,
        "kv": kv_mode(cfg),
        "vision": vision_mode(cfg),
        "low_ram": low_ram_mode(cfg),
        "gpu": cfg.get("gpu"),
        "layer_split": cfg.get("layer_split"),
        "port": cfg.get("port", 8080),
        "host": cfg.get("host", "127.0.0.1"),
        "api_key_set": bool(cfg.get("api_key")),
        "engine": Path(cfg.get("exe", "")).name if cfg.get("exe") else None,
        "exe_ok": bool(cfg.get("exe")) and Path(cfg["exe"]).exists(),
        "gguf_ok": all(Path(p).exists() for p in gguf_paths(cfg)),
        "mtime": float(cfg_path.stat().st_mtime),
    }


def discover(root: Path) -> list:
    """Installed models, most recently used first - setup.py's installed_configs() ordering."""
    configs = sorted(root.glob("strata-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [model_summary(p) for p in configs]


# ====================================================================================================== config editing
def apply_context(args: list, ctx: int) -> list:
    """--max-context, and the surrounding rules setup.py applies itself:
      - KV: below 8K the engine uses fp16 and setup writes no --kv; from 8K up int8 unless chosen otherwise.
      - KV streaming (--kv-resident) is only added from 64K up: a shorter context drops it again.
      - past the trained 262144 the setup adds rope yarn with the derived factor; inside it an explicit rope
        choice stays untouched.
    """
    args = set_arg_val(args, "--max-context", ctx)
    if ctx <= 8192:
        args = drop_arg(args, "--kv")
    elif arg_val(args, "--kv") is None:
        args = set_arg_val(args, "--kv", "int8")
    if ctx < 65536:
        args = drop_arg(args, "--kv-resident")
    if ctx > TRAINED_CONTEXT and arg_val(args, "--rope-scaling") is None:
        args = set_arg_val(args, "--rope-scaling", "yarn")
        args = set_arg_val(args, "--rope-scale", f"{setup.derived_factor(ctx):g}")
    return args


def apply_kv(args: list, ctx: int, kv: str) -> list:
    """An explicit KV cache precision: only meaningful from 8K up (below, setup leaves it to fp16)."""
    if ctx <= 8192:
        return list(args)
    if kv in ("int8", "q4_0", "k8v4"):
        return set_arg_val(args, "--kv", kv)
    return list(args)


def _find_mmproj(root: Path, cfg: dict, family: str) -> Path | None:
    """The family's vision encoder file where setup.py would have put it (used to re-enable Vision whose section
    an earlier "Off" removed).  Same discovery rules as setup.py: next to the model's GGUF, or in the data
    folder's models."""
    fam = setup.FAMILIES.get(family, {})
    name = fam.get("mmproj")
    if not name:
        return None
    cands = []
    for p in gguf_paths(cfg):
        cands.append(Path(p).parent / name)
    for d in (root / "models", root.parent / "Strata-data" / "models"):
        cands += list(d.rglob(name)) if d.is_dir() else []
    try:
        s = setup.load_settings()
        if s.get("data_dir"):
            cands += list((Path(s["data_dir"]) / "models").rglob(name))
    except Exception:
        pass
    for p in cands:
        if Path(p).is_file():
            return Path(p)
    return None


def detect_family(cfg: dict) -> str:
    """Which model family a config is: setup.py's own markers - the Coder/Swift names in the model_name or the
    GGUF file paths, else the original."""
    name = str(cfg.get("model_name", "")).lower()
    joined = " ".join(p.lower() for p in gguf_paths(cfg))
    if "coder" in name or "coder" in joined:
        return "coder"
    if "swift" in name or "swift" in joined:
        return "swift"
    return "qwen"


def apply_vision(root: Path, cfg: dict, mode: str) -> dict:
    """Vision Off/GPU/CPU with the config structure setup.py writes.  Off removes the section exactly as setup
    does (serve/server.py loads the encoder only when the section exists) and drops the engine's --vision
    flags.  Turning it on reconstructs the section with the same values setup.py writes, found in the same
    places (the engine's image encoder, the family's mmproj next to the model's GGUF) - it never downloads
    anything."""
    cfg = dict(cfg)
    args = list(cfg.get("args", []))
    if mode in ("off", "none"):
        cfg.pop("vision", None)
        cfg["args"] = drop_arg(drop_arg(args, "--vram-reserve-mib"), "--vision")
        return cfg
    vis = cfg.get("vision")
    if not isinstance(vis, dict):
        family = detect_family(cfg)
        exe = Path(root) / "engine" / setup.VEXE
        shard = arg_val(args, "--native")
        mmproj = _find_mmproj(root, cfg, family)
        if not shard or not Path(shard).exists() or not mmproj or not exe.exists():
            raise ValueError(
                "Vision is not installed for this model (no encoder in its config and its files were not "
                "found: engine/" + setup.VEXE + " and the model family's mmproj-*.gguf). Adding it downloads "
                "~1 GB, so the Manager does not fetch it: run SETUP.bat --vision for this model, then choose "
                "a Vision mode here.")
        vis = {"exe": str(exe), "mmproj": str(mmproj), "model": shard, "gpu": True,
               "max_tokens": setup.VISION[mode]["max_tokens"]}
    reserve = setup.VISION[mode]["reserve_mib"]
    args = set_arg(args, "--vision")
    args = set_arg_val(args, "--vram-reserve-mib", reserve)
    vis = dict(vis)
    vis["gpu"] = mode == "gpu"
    vis["max_tokens"] = setup.VISION[mode]["max_tokens"]
    if mode == "cpu" and "threads" not in vis:
        vis["threads"] = max(1, (os.cpu_count() or 8) // 2)   # setup.py's own default
    cfg["args"] = args
    cfg["vision"] = vis
    return cfg


def apply_low_ram(cfg: dict, on: bool, resident: bool | None = None) -> dict:
    """The low-RAM mode: the flag setup.py writes (--resident-experts or --mmap-experts).  `resident` is the
    setup.py decision (setup.low_ram_resident) made by the caller with this PC's RAM/VRAM; None falls back to
    the always-available mapped mode."""
    cfg = dict(cfg)
    args = drop_arg(drop_arg(list(cfg.get("args", [])), "--resident-experts"), "--mmap-experts")
    if on:
        args = set_arg(args, "--resident-experts" if resident else "--mmap-experts")
    cfg["args"] = args
    return cfg


def apply_network(cfg: dict, port: int | None, host: str | None, api_key: str | None) -> dict:
    """The config fields setup.py writes when it serves the model on the network (its --port/--host/--api-key)."""
    cfg = dict(cfg)
    if port is not None and str(port).isdigit():
        cfg["port"] = int(port)
    if host is not None:
        if host.strip() in ("", "127.0.0.1", "localhost") or host is None:
            cfg.pop("host", None)
        else:
            cfg["host"] = host.strip()
    if api_key is not None:
        if api_key.strip():
            cfg["api_key"] = api_key.strip()
        else:
            cfg.pop("api_key", None)
    return cfg


def apply_gpu(cfg: dict, gpu) -> dict:
    """The GPU the model runs on: cfg["gpu"] as an index (setup.py's issue-#51 semantics).  A multi-GPU layer
    split ("gpu" a list) is left alone - the editor disables the picker for it (the Manager does not re-split
    layers: that is SETUP.bat's --gpus job)."""
    cfg = dict(cfg)
    if isinstance(cfg.get("gpu"), list):
        return cfg
    if gpu in (None, "", "auto"):
        cfg.pop("gpu", None)
    else:
        cfg["gpu"] = int(gpu)
    return cfg


def edit_config(root: Path, cfg: dict, changes: dict, low_ram_resident: bool | None = None) -> dict:
    """Apply a settings edit (from the Manager's Save) to a loaded config.  Pure: returns the new config dict.
    `changes` keys: context, kv, vision (off|gpu|cpu), low_ram (bool), port, host, api_key, gpu."""
    cfg = dict(cfg)
    if changes.get("context") is not None:
        cfg["args"] = apply_context(list(cfg.get("args", [])), int(changes["context"]))
    ctx = int(arg_val(cfg.get("args", []), "--max-context") or 0)
    if changes.get("kv"):
        cfg["args"] = apply_kv(list(cfg.get("args", [])), ctx, changes["kv"])
    if changes.get("vision") and changes["vision"] != vision_mode(cfg):
        cfg = apply_vision(root, cfg, changes["vision"])
    if changes.get("low_ram") is not None and changes["low_ram"] != (low_ram_mode(cfg) != "off"):
        cfg = apply_low_ram(cfg, bool(changes["low_ram"]), low_ram_resident)
    if any(k in changes for k in ("port", "host", "api_key")):
        cfg = apply_network(cfg, changes.get("port"), changes.get("host"), changes.get("api_key"))
    if "gpu" in changes:
        cfg = apply_gpu(cfg, changes["gpu"])
    return cfg


def save_config(cfg_path: Path, cfg: dict, backup: bool = True) -> Path:
    """Write a config in the exact format setup.py writes (indent=1, UTF-8), atomically (temp + os.replace, so
    a crash never leaves a half-written config), keeping a timestamped .bak copy of the previous version."""
    text = json.dumps(cfg, indent=1, ensure_ascii=False)
    if backup and cfg_path.exists():
        try:
            old = cfg_path.read_text(encoding="utf-8-sig")
        except OSError:
            old = None
        if old != text:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            shutil.copyfile(cfg_path, cfg_path.with_name(cfg_path.name + BACKUP_SUFFIX + stamp))
            keep = sorted(cfg_path.parent.glob(cfg_path.name + BACKUP_SUFFIX + "*"))
            for stale in keep[:-KEEP_BACKUPS]:
                stale.unlink(missing_ok=True)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=cfg_path.name + ".", suffix=".tmp", dir=str(cfg_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, cfg_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return cfg_path


# ====================================================================================================== custom GGUF
GGUF_SHARD_RE = re.compile(r"(?i)(\d+)-of-(\d+)\.gguf$")


def detect_gguf_dir(path: str) -> dict:
    """What Strata would make of a chosen folder with GGUF files: the family and size from the file names
    (setup.py's own naming conventions: Qwen3.8-Flash-Next-GSQ-RCO-<Q> or Swift-.../<Q> or Coder-.../<Q>,
    shards "...-0000i-of-00002.gguf").  Preparation itself is left to setup.py (prepare_gguf)."""
    d = Path(path)
    if d.is_file():
        d = d.parent
    if not d.is_dir():
        return {"ok": False, "error": f"not a folder: {path}"}
    shards = [p for p in sorted(d.iterdir()) if p.is_file() and GGUF_SHARD_RE.search(p.name)
              and int(GGUF_SHARD_RE.search(p.name).group(2)) >= 2]
    if not shards:
        return {"ok": False,
                "error": f"no multi-shard GGUF files (…-NNNN1-of-NNNN2.gguf) in {d}"}
    name = shards[0].name
    q = None
    for size in sorted(setup.MODELS, key=len, reverse=True):     # IQ2_XS before IQ2 etc.; longest match wins
        if size in name:
            q = size
            break
    if q is None:
        example = setup.FAMILIES["qwen"]["file"].format(q="Q2_0", i=1)
        return {"ok": False, "error": f"cannot tell which Strata size {name} is. Strata knows these sizes: "
                f"{', '.join(setup.MODELS)}; its files are named like {example}."}
    # the family follows the size: only the Coder has IQ1_M, Swift's files say "Swift" in their name,
    # and setup.py itself never takes a family from the GGUF name (a Coder IQ1_M file is named exactly like
    # a Qwen one).  A name marker only overrides when that family actually has the size.
    fams = setup.MODELS[q].get("families", ("qwen", "swift"))
    if re.search(r"(?i)swift", name) and "swift" in fams:
        family = "swift"
    elif re.search(r"(?i)coder", name) and "coder" in fams:
        family = "coder"
    else:
        family = fams[0]
    fam = setup.FAMILIES[family]
    return {"ok": True, "dir": str(d), "shards": [s.name for s in shards], "family": family,
            "family_title": fam["title"], "quant": q, "tag": (fam["tag"] + q).lower()}


# ====================================================================================================== system info
def system_info() -> dict:
    """GPU / VRAM / RAM / CPU, from the same sources setup.py uses (nvidia-smi, the OS) - measured values only,
    no invented estimates."""
    gpus = []
    for g in setup.gpus():
        used = None
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout
            for line in out.splitlines():
                i, _, v = line.partition(",")
                if i.strip() == str(g["index"]):
                    used = float(v.strip()) / 1024.0
                    break
        except (OSError, subprocess.SubprocessError, ValueError):
            used = None
        gpus.append({**g, "vram_used_gb": used})
    ram = setup.ram_gb()
    ram_used = None
    try:
        import psutil                              # already a Strata dependency (requirements.txt)
        ram_used = max(0.0, ram - psutil.virtual_memory().available / 2 ** 30)
    except Exception:
        pass
    return {"gpus": gpus, "ram_gb": round(ram, 1) if ram else None,
            "ram_used_gb": round(ram_used, 1) if ram_used is not None else None,
            "cpu": setup.cpu_info()[0], "strata_version": setup.source_version()}


# ====================================================================================================== HTTP layer
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "StrataManager/0.1"

    def log_message(self, fmt, *args):                   # keep the console clean
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode("utf-8"))

    def _read_json(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        except (ValueError, json.JSONDecodeError):
            return {}

    def _handle(self, fn):
        try:
            fn()
        except Exception as e:                           # never leave the browser hanging
            import traceback
            traceback.print_exc()
            self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)

    # ---- routes ----------------------------------------------------------------------
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        m = self.server.manager

        def route():
            if u.path in ("/", "/index.html"):
                self._send(200, (m.web / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif u.path == "/app.css":
                self._send(200, (m.web / "app.css").read_bytes(), "text/css; charset=utf-8")
            elif u.path == "/app.js":
                self._send(200, (m.web / "app.js").read_bytes(), "text/javascript; charset=utf-8")
            elif u.path == "/api/system":
                self._json({"ok": True, "data": system_info()})
            elif u.path == "/api/models":
                models = discover(m.root)
                self._json({"ok": True, "data": {"models": models,
                                                 "default": models[0]["config"] if models else None,
                                                 "status": server_status(m.root)}})
            elif u.path == "/api/config":
                cfg_path = m.resolve((q.get("config") or [""])[0])
                if not cfg_path:
                    self._json({"ok": False, "error": "no such config"}, 404)
                    return
                self._json({"ok": True, "data": m.config_state(cfg_path)})
            elif u.path == "/api/status":
                self._json({"ok": True, "data": m.status_payload()})
            elif u.path == "/api/log":
                p = (q.get("path") or [""])[0]
                if not m.inside(str(m.root), p):
                    self._json({"ok": False, "error": "refusing to read files outside the Strata folder"}, 403)
                    return
                n = min(int((q.get("tail") or ["200"])[0]), 1000)
                self._json({"ok": True, "data": {"path": p, "text": tail_lines(p, n)}})
            elif u.path == "/api/open-chat":
                webbrowser.open(f"http://127.0.0.1:{int((q.get('port') or ['8080'])[0])}/")
                self._json({"ok": True})
            else:
                self._json({"ok": False, "error": f"unknown endpoint {u.path}"}, 404)
        self._handle(route)

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        body = self._read_json()
        m = self.server.manager

        def route():
            if u.path == "/api/save":
                cfg_path = m.resolve(body.get("config", ""))
                if not cfg_path:
                    self._json({"ok": False, "error": "no such config"}, 404)
                    return
                self._json(m.save(body, cfg_path))
            elif u.path == "/api/start":
                self._json(m.start(body))
            elif u.path == "/api/stop":
                self._json(stop_server(m.root))
            elif u.path == "/api/restart":
                self._json(m.restart(body))
            elif u.path == "/api/browse":
                self._json({"ok": True, "data": m.browse(body.get("path", ""))})
            elif u.path == "/api/gguf-detect":
                self._json(detect_gguf_dir(body.get("path", "")))
            elif u.path == "/api/prepare-gguf":
                self._json(m.prepare_gguf(body))
            else:
                self._json({"ok": False, "error": f"unknown endpoint {u.path}"}, 404)
        self._handle(route)


class Manager:
    """The Manager application: owns the HTTP callbacks, the config editor state, and the supervisor glue."""

    def __init__(self, root: Path = ROOT):
        self.root = Path(root).resolve()
        self.web = Path(__file__).resolve().parent / "web"

    @staticmethod
    def inside(base: str, path: str) -> bool:
        try:
            p = Path(path).resolve()
            return p == Path(base).resolve() or p.is_relative_to(Path(base).resolve())
        except (OSError, ValueError):
            return False

    def resolve(self, name: str) -> Path | None:
        """A config file *in this Strata folder* (never outside it), by file name."""
        p = (self.root / name).resolve()
        if p.parent != self.root or p.suffix != ".json" or not p.is_file():
            return None
        return p

    # ---- state for the editor -----------------------------------------------------------
    def config_state(self, cfg_path: Path) -> dict:
        """Everything the editor needs for one model: its current values, and the options Strata actually has
        (context presets, Vision, KV, GPUs, the low-RAM decision) - nothing the engine does not know."""
        cfg = read_config(cfg_path)
        args = cfg.get("args", [])
        ctx = int(arg_val(args, "--max-context") or 0)
        ch = setup.choices_from_config(cfg_path)
        gpu_sel = cfg.get("gpu")
        vram_gb = None
        if isinstance(gpu_sel, int):
            for g in setup.gpus() or []:
                if g["index"] == gpu_sel:
                    vram_gb = g["vram_gb"]
                    break
        vision = vision_mode(cfg)
        low_ram = low_ram_mode(cfg) != "off"
        resident = None
        if low_ram and ch["model"]:
            ram = setup.ram_gb()
            vram = vram_gb if vram_gb is not None else max((g["vram_gb"] for g in setup.gpus() or []),
                                                           default=0.0)
            if vision != "off":
                vram -= setup.VISION[vision]["reserve_mib"] / 1024.0
            resident = bool(setup.low_ram_resident(ch["model"], ram, max(0.0, vram), ctx, kv_mode(cfg)))
        return {"summary": model_summary(cfg_path), "context": ctx, "presets": PRESET_CONTEXTS,
                "trained_context": TRAINED_CONTEXT,
                "kv": kv_mode(cfg), "kv_options": ["int8", "q4_0", "k8v4"] if ctx > 8192 else [],
                "vision": vision,
                "vision_available": isinstance(cfg.get("vision"), dict)
                or bool(_find_mmproj(self.root, cfg, ch["family"] or "qwen"))
                and (self.root / "engine" / setup.VEXE).exists(),
                "low_ram": low_ram, "low_ram_resident": resident,
                "port": cfg.get("port", 8080), "host": cfg.get("host", "127.0.0.1"),
                "api_key": cfg.get("api_key", ""),
                "gpu": gpu_sel, "gpus": setup.gpus() or [],
                "rope": arg_val(args, "--rope-scaling"),
                "serve_log": str(self.root / (cfg_path.name[:-5] + launcher.SERVE_LOG_SUFFIX)),
                }

    def save(self, body: dict, cfg_path: Path) -> dict:
        """Load, apply the edit, write back atomically with a backup, and return the new summary."""
        cfg = read_config(cfg_path)
        low_ram_resident = None
        if body.get("low_ram") is True and low_ram_mode(cfg) == "off":
            low_ram_resident = self._resident_decision(cfg_path, cfg)
        try:
            new_cfg = edit_config(self.root, cfg, body, low_ram_resident)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        save_config(cfg_path, new_cfg)
        return {"ok": True, "data": {"summary": model_summary(cfg_path), "config": cfg_path.name}}

    def _resident_decision(self, cfg_path: Path, cfg: dict) -> bool:
        """setup.low_ram_resident with this PC's RAM/VRAM - the same decision setup.py makes at install time."""
        ch = setup.choices_from_config(cfg_path)
        ctx = int(arg_val(cfg.get("args", []), "--max-context") or 0)
        vis = vision_mode(cfg)
        gpus = setup.gpus() or []
        sel = cfg.get("gpu")
        vram = next((g["vram_gb"] for g in gpus if g["index"] == sel),
                    max((g["vram_gb"] for g in gpus), default=0.0))
        if vis != "off":
            vram -= setup.VISION[vis]["reserve_mib"] / 1024.0
        if ch["model"] and gpus:
            return bool(setup.low_ram_resident(ch["model"], setup.ram_gb(), max(0.0, vram), ctx, kv_mode(cfg)))
        return False

    # ---- supervisor glue --------------------------------------------------------------------
    def status_payload(self) -> dict:
        st = launcher.ServerState(self.root).read()
        tool = st.get("tool") or {}
        return {"server": server_status(self.root),
                "tool": {**tool, "alive": launcher.pid_alive(tool.get("pid"))} if tool else None}

    def start(self, body: dict) -> dict:
        return start_server(body.get("config", ""), body.get("port"), self.root, body.get("open_chat", False))

    def restart(self, body: dict) -> dict:
        return restart_server(body.get("config", ""), body.get("port"), self.root, body.get("open_chat", False))

    # ---- folder picker and custom GGUF -----------------------------------------------------------
    def browse(self, path: str) -> dict:
        """A minimal server-side folder picker (the browser has no native file dialog without new
        dependencies): lists the folders under `path` so the user can walk to their model files."""
        try:
            d = (Path(os.path.expanduser(path)).resolve() if path else Path.cwd().resolve())
            if not d.is_dir():
                d = Path.home().resolve()
            kids = sorted((p for p in d.iterdir() if p.is_dir() and not p.name.startswith(".")),
                          key=lambda p: p.name.lower())
            return {"path": str(d), "anchor": d.anchor, "parent": str(d.parent) if d.parent != d else None,
                    "dirs": [str(p) for p in kids], "detected": detect_gguf_dir(str(d))}
        except OSError as e:
            return {"ok": False, "error": str(e)}

    def prepare_gguf(self, body: dict) -> dict:
        """A chosen custom GGUF folder becomes an installed model through the EXISTING setup.py pipeline
        (--gguf-dir, --no-start, --yes): no second packing/downloading system lives here.  setup.py's own
        output streams to a log the UI tails, and the new config lands in the Strata folder for the editor."""
        det = detect_gguf_dir(body.get("path", ""))
        if not det["ok"]:
            return det
        fam = setup.FAMILIES[det["family"]]
        cfg_name = f"strata-{det['tag']}.json"
        if (self.root / cfg_name).exists():
            return {"ok": True, "already": cfg_name, "config": cfg_name}
        ctx = int(body.get("context") or 32768)
        cmd = [sys.executable, str(self.root / "setup.py"), "--family", det["family"], "--model", det["quant"],
               "--gguf-dir", det["dir"], "--context", str(ctx), "--no-start", "--yes"]
        log_path = self.root / "logs" / f"manager-prepare-{det['tag']}.log"
        tool = run_tool(cmd, log_path, f"preparing {fam['title']} {det['quant']} from {det['dir']}",
                        root=self.root)
        return {"ok": True, "config": cfg_name, "tool": tool, "log": str(log_path)}


def make_server(root: Path, port: int, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    class S(ThreadingHTTPServer):
        manager = Manager(root)
    srv = S((host, port), Handler)
    srv.daemon_threads = True
    return srv


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help=f"the Manager's local port (default {DEFAULT_PORT})")
    ap.add_argument("--root", default=str(ROOT), help="another Strata folder to manage (mainly for tests)")
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    if not (root / "setup.py").is_file():
        print(f"  [x] {root} is no Strata folder (no setup.py) - launch the Manager from inside Strata.")
        return 1
    try:
        httpd = make_server(root, a.port)
    except OSError as e:
        print(f"  [x] the Manager's port {a.port} is taken ({e}). Use --port N for another one.")
        return 1
    print("Strata Manager")
    print(f"  managing        {root}")
    print(f"  open            http://127.0.0.1:{a.port}/  (close this window or Ctrl+C to stop the Manager)")
    if not a.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(f"http://127.0.0.1:{a.port}/")).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())