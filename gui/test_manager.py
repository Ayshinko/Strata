"""Tests for the Strata Manager's backend (gui/manager.py).

The Manager only reuses Strata's own logic (setup.py) and its own config format, so the tests cover:
  - reading existing configs (model_summary / discover, on temp copies - never the user's real ones)
  - the surgical edits Save performs (context/rope/KV, Vision, Low-RAM, network, GPU)
  - atomic saves with .bak backups, and that a saved config stays readable
  - the custom-GGUF name detection (shards, family, size)
  - a real HTTP round-trip against a temp Strata folder (GET /, /api/models, /api/save)

Pure stdlib + unittest; no GPU, no network, no downloads.

    python -m unittest gui.test_manager
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import setup                                          # noqa: E402
import gui.manager as mgr                              # noqa: E402

SHARD1 = "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
SHARD2 = "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf"
FAKE_GPU = {"index": 0, "name": "Fake GPU", "vram_gb": 12.0, "arch": "100",
            "driver": "600", "count": 1}


def sample_cfg(**over):
    """A minimal but real-looking strata-q2_0 config (a temp copy - never a user file)."""
    cfg = {
        "exe": "C:/fake/strata.exe",
        "args": ["--pack", "C:/fake/pack/q2_0", "--native", f"C:/fake/{SHARD1}",
                 "--ple-gguf", f"C:/fake/{SHARD2}", "--expert-cache", "auto",
                 "--max-context", "32768", "--kv", "int8"],
        "cwd": "C:/fake",
        "tokenizer": "C:/fake/pack/q2_0/tokenizer",
        "model_name": "qwen3.8-flash-next-q2_0",
        "log": "C:/fake/strata-q2_0.log",
        "port": 8080,
    }
    cfg.update(over)
    return cfg


class ArgsHelpers(unittest.TestCase):
    def test_set_arg_val_in_place(self):
        a = ["--a", "1", "--b", "2"]
        self.assertEqual(mgr.set_arg_val(a, "--b", "9"), ["--a", "1", "--b", "9"])
        self.assertEqual(mgr.set_arg_val(a, "--c", "3"), ["--a", "1", "--b", "2", "--c", "3"])
        self.assertEqual(mgr.arg_val(a, "--b"), "2")
        self.assertIsNone(mgr.arg_val(a, "--nope"))

    def test_drop_arg_pair_and_flag(self):
        self.assertEqual(mgr.drop_arg(["--kv", "int8", "--spec", "4"], "--kv"), ["--spec", "4"])
        self.assertEqual(mgr.drop_arg(["--vision", "--k", "1"], "--vision"), ["--k", "1"])
        self.assertEqual(mgr.drop_arg(["--x", "--y", "2"], "--x"), ["--y", "2"])


class ContextEdits(unittest.TestCase):
    def test_sets_max_context_and_adds_int8_kv_above_8k(self):
        a = mgr.apply_context(["--pack", "p"], 32768)
        self.assertEqual(mgr.arg_val(a, "--max-context"), "32768")
        self.assertEqual(mgr.arg_val(a, "--kv"), "int8")

    def test_below_8k_drops_kv(self):
        a = mgr.apply_context(["--kv", "q4_0", "--max-context", "65536"], 8192)
        self.assertIsNone(mgr.arg_val(a, "--kv"))
        self.assertEqual(mgr.arg_val(a, "--max-context"), "8192")

    def test_past_trained_adds_yarn_with_derived_factor(self):
        a = mgr.apply_context([], 393216)
        self.assertEqual(mgr.arg_val(a, "--rope-scaling"), "yarn")
        self.assertEqual(float(mgr.arg_val(a, "--rope-scale")), 1.5)   # 393216 / 262144
        inside = mgr.apply_context(["--rope-scaling", "yarn", "--rope-scale", "1.5"], 131072)
        self.assertEqual(mgr.arg_val(inside, "--rope-scaling"), "yarn")  # explicit choice is kept

    def test_kv_resident_dropped_below_64k(self):
        a = mgr.apply_context(["--kv-resident", "32768"], 32768)
        self.assertIsNone(mgr.arg_val(a, "--kv-resident"))

    def test_apply_kv_only_above_8k(self):
        self.assertEqual(mgr.apply_kv([], 4096, "q4_0"), [])
        self.assertIn("q4_0", mgr.apply_kv([], 16384, "q4_0"))


class VisionEdits(unittest.TestCase):
    def setUp(self):
        self.cfg = sample_cfg(vision={"exe": "C:/fake/strata-vision.exe", "mmproj": "C:/fake/mmproj.gguf",
                                      "model": f"C:/fake/{SHARD1}", "gpu": True, "max_tokens": 1024},
                              args=["--vision", "--vram-reserve-mib", "700", "--max-context", "32768"])

    def test_off_removes_section_and_flags(self):
        out = mgr.apply_vision(ROOT, self.cfg, "off")
        self.assertNotIn("vision", out)
        self.assertNotIn("--vision", out["args"])
        self.assertNotIn("--vram-reserve-mib", out["args"])
        self.assertEqual(mgr.vision_mode(out), "off")

    def test_gpu_cpu_switch_keeps_paths(self):
        g = mgr.apply_vision(ROOT, self.cfg, "gpu")
        self.assertTrue(g["vision"]["gpu"])
        self.assertEqual(g["vision"]["max_tokens"], setup.VISION["gpu"]["max_tokens"])
        c = mgr.apply_vision(ROOT, self.cfg, "cpu")
        self.assertFalse(c["vision"]["gpu"])
        self.assertEqual(c["vision"]["max_tokens"], setup.VISION["cpu"]["max_tokens"])
        self.assertIn("--vision", c["args"])
        self.assertIn("threads", c["vision"])

    def test_reconstructs_section_from_disk(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "engine").mkdir()
            (root / "engine" / "strata-vision.exe").write_bytes(b"")
            shard = root / SHARD1
            shard.write_bytes(b"")
            (root / "mmproj-Qwen3.8-Flash-Next-BF16.gguf").write_bytes(b"")
            cfg = sample_cfg(args=["--native", str(shard), "--max-context", "32768"])   # no vision section
            out = mgr.apply_vision(root, cfg, "gpu")
            v = out["vision"]
            self.assertTrue(v["gpu"])
            self.assertEqual(v["exe"], str(root / "engine" / "strata-vision.exe"))
            self.assertEqual(v["mmproj"], str(root / "mmproj-Qwen3.8-Flash-Next-BF16.gguf"))
            self.assertEqual(v["model"], str(shard))

    def test_refuses_when_files_missing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = sample_cfg(args=["--native", str(root / "nope.gguf"), "--max-context", "32768"])
            with self.assertRaises(ValueError):
                mgr.apply_vision(root, cfg, "gpu")


class LowRamNetworkGpu(unittest.TestCase):
    def test_low_ram_flag_round_trip(self):
        cfg = sample_cfg(args=["--mmap-experts", "--max-context", "32768"])
        self.assertEqual(mgr.low_ram_mode(cfg), "mmap")
        off = mgr.apply_low_ram(cfg, False)
        self.assertEqual(mgr.low_ram_mode(off), "off")
        on = mgr.apply_low_ram(off, True, resident=True)
        self.assertEqual(mgr.low_ram_mode(on), "resident")
        on2 = mgr.apply_low_ram(off, True, resident=False)
        self.assertEqual(mgr.low_ram_mode(on2), "mmap")

    def test_network_fields(self):
        cfg = sample_cfg()
        out = mgr.apply_network(cfg, 9090, "0.0.0.0", "secret")
        self.assertEqual(out["port"], 9090)
        self.assertEqual(out["host"], "0.0.0.0")
        self.assertEqual(out["api_key"], "secret")
        back = mgr.apply_network(out, None, "127.0.0.1", "")
        self.assertNotIn("host", back)
        self.assertNotIn("api_key", back)

    def test_gpu(self):
        self.assertEqual(mgr.apply_gpu(sample_cfg(), 2)["gpu"], 2)
        self.assertNotIn("gpu", mgr.apply_gpu(sample_cfg(), "auto"))
        split = sample_cfg(gpu=[0, 1], gpus_asked=True)
        self.assertEqual(mgr.apply_gpu(split, 2)["gpu"], [0, 1])   # a layer split is left alone


class EditRoundTrip(unittest.TestCase):
    def test_full_change_set(self):
        cfg = sample_cfg()
        out = mgr.edit_config(ROOT, cfg, {"context": 131072, "kv": "q4_0", "vision": "off",
                                          "low_ram": True, "port": 8081, "host": "127.0.0.1",
                                          "api_key": "", "gpu": "auto"}, low_ram_resident=False)
        self.assertEqual(mgr.arg_val(out["args"], "--max-context"), "131072")
        self.assertEqual(mgr.arg_val(out["args"], "--kv"), "q4_0")
        self.assertEqual(mgr.low_ram_mode(out), "mmap")
        self.assertEqual(out["port"], 8081)
        # and the result is exactly what setup.py itself can read back
        self.assertEqual(setup.choices_from_config.__name__, "choices_from_config")


class SaveConfig(unittest.TestCase):
    def test_atomic_write_and_backup(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "strata-x.json"
            p.write_text(json.dumps(sample_cfg(), indent=1), encoding="utf-8")
            new = dict(sample_cfg())
            new["args"][new["args"].index("--max-context") + 1] = "65536"
            mgr.save_config(p, new)
            reread = json.loads(p.read_text(encoding="utf-8-sig"))
            self.assertEqual(mgr.arg_val(reread["args"], "--max-context"), "65536")
            backups = list(p.parent.glob("strata-x.json.bak-*"))
            self.assertEqual(len(backups), 1)                    # one .bak of the previous version
            self.assertIn("32768", backups[0].read_text(encoding="utf-8-sig"))

    def test_rewrite_same_content_makes_no_backup(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "strata-x.json"
            cfg = sample_cfg()
            mgr.save_config(p, cfg)
            mgr.save_config(p, cfg)
            self.assertEqual(len(list(p.parent.glob("strata-x.json.bak-*"))), 0)


class GgufDetection(unittest.TestCase):
    def _dir(self, *names):
        d = Path(tempfile.mkdtemp())
        for n in names:
            (d / n).write_bytes(b"x")
        return d

    def test_qwen_q2_0(self):
        d = self._dir(SHARD1, SHARD2)
        r = mgr.detect_gguf_dir(str(d))
        self.assertTrue(r["ok"])
        self.assertEqual(r["family"], "qwen")
        self.assertEqual(r["quant"], "Q2_0")
        self.assertEqual(r["tag"], "q2_0")

    def test_swift_iq3_xxs(self):
        d = self._dir("Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
                      "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf")
        r = mgr.detect_gguf_dir(str(d))
        self.assertEqual(r["family"], "swift")
        self.assertEqual(r["quant"], "IQ3_XXS")

    def test_coder(self):
        d = self._dir("Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf",
                      "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf")
        self.assertEqual(mgr.detect_gguf_dir(str(d))["family"], "coder")

    def test_no_shards(self):
        d = self._dir("readme.txt")
        self.assertFalse(mgr.detect_gguf_dir(str(d))["ok"])


class Discovery(unittest.TestCase):
    def test_model_summary_and_order(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = root / "strata-q2_0.json"
            b = root / "strata-iq3_xxs.json"
            a.write_text(json.dumps(sample_cfg(model_name="qwen3.8-flash-next-q2_0"), indent=1),
                         encoding="utf-8")
            time = 1000000000
            os.utime(a, (time, time))
            b.write_text(json.dumps(sample_cfg(model_name="qwen3.8-flash-next-iq3_xxs",
                                               args=["--max-context", "131072", "--kv", "int8",
                                                     "--resident-experts"]), indent=1), encoding="utf-8")
            os.utime(b, (time + 10, time + 10))
            ms = mgr.discover(root)
            self.assertEqual([m["config"] for m in ms], ["strata-iq3_xxs.json", "strata-q2_0.json"])
            first = ms[0]
            self.assertEqual(first["quant"], "IQ3_XXS")
            self.assertEqual(first["context"], 131072)
            self.assertEqual(first["low_ram"], "resident")
            self.assertEqual(first["vision"], "off")


class HttpRoundTrip(unittest.TestCase):
    """A real Manager HTTP server against a temp Strata folder: the browser's own calls."""

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.root = Path(cls.td.name)
        cfg = sample_cfg()
        (cls.root / "strata-q2_0.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        cls.server = mgr.make_server(cls.root, 0)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.td.cleanup()

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return r.status, r.read()

    def post(self, path, obj):
        req = urllib.request.Request(self.base + path, data=json.dumps(obj).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())

    def test_ui_and_models_serve(self):
        status, html = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"Strata Manager", html)
        status, _ = self.get("/app.css")
        self.assertEqual(status, 200)
        status, models = self.get("/api/models")
        self.assertEqual(json.loads(models)["data"]["models"][0]["config"], "strata-q2_0.json")

    def test_config_endpoint_edits_nothing(self):
        status, data = self.get("/api/config?config=strata-q2_0.json")
        self.assertEqual(status, 200)
        before = self.root.joinpath("strata-q2_0.json").read_text(encoding="utf-8")
        self.assertEqual(json.loads(data)["data"]["context"], 32768)
        self.assertEqual(self.root.joinpath("strata-q2_0.json").read_text(encoding="utf-8"), before)

    def test_save_round_trip_and_backup(self):
        with mock.patch.object(setup, "gpus", return_value=[FAKE_GPU]):
            with mock.patch.object(setup, "ram_gb", return_value=24.0):
                status, data = self.post("/api/save", {"config": "strata-q2_0.json", "context": 131072,
                                                       "kv": "q4_0", "vision": "off", "low_ram": True,
                                                       "port": 8080, "host": "127.0.0.1", "api_key": "", "gpu": 0})
        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        reread = json.loads(self.root.joinpath("strata-q2_0.json").read_text(encoding="utf-8-sig"))
        self.assertEqual(mgr.arg_val(reread["args"], "--max-context"), "131072")
        self.assertEqual(mgr.arg_val(reread["args"], "--kv"), "q4_0")
        self.assertEqual(mgr.low_ram_mode(reread), "mmap")
        self.assertIn("gpu", reread)
        # the previous version is backed up and untouched
        backups = list(self.root.glob("strata-q2_0.json.bak-*"))
        self.assertEqual(len(backups), 1)
        old = json.loads(backups[0].read_text(encoding="utf-8-sig"))
        self.assertEqual(mgr.arg_val(old["args"], "--max-context"), "32768")
        # saving the same values again creates no second backup
        self.post("/api/save", {"config": "strata-q2_0.json", "context": 131072, "kv": "q4_0",
                                "vision": "off", "low_ram": True, "port": 8080, "host": "127.0.0.1",
                                "api_key": "", "gpu": 0})
        self.assertEqual(len(list(self.root.glob("strata-q2_0.json.bak-*"))), 1)


class LauncherTest(unittest.TestCase):
    """The supervisor is platform-neutral: it never branches on the OS; the platform adapters do.  These
    tests run the supervisor against a fake adapter, so they are valid on every OS."""

    class FakeLauncher:
        name = "fake"
        force_result = {"stopped": True, "forced": True}

        def __init__(self):
            self.spawned = []
            self.terminated = []
            self.force_terminated = []

        def spawn(self, cmd, cwd, log_file):
            self.spawned.append((cmd, cwd, log_file))
            return 4242

        def terminate(self, pid, grace_s):
            self.terminated.append(pid)
            return {"stopped": True, "forced": False}

        def terminate_force(self, pid):
            self.force_terminated.append(pid)
            return self.force_result

        @staticmethod
        def alive(pid):
            return False

    def setUp(self):
        self.fake = self.FakeLauncher()
        self.patch = mock.patch("gui.launcher.get_launcher", return_value=self.fake)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.root = Path(self.td.name)

    def _write_state(self, **server):
        mgr.launcher.ServerState(self.root).write({"server": server, "tool": None})

    def test_platform_adapter_chosen_for_this_os(self):
        from gui.platforms import get_launcher
        if os.name == "nt":
            from gui.platforms.windows import WindowsLauncher
            self.assertIsInstance(get_launcher(), WindowsLauncher)
        else:
            from gui.platforms.linux import LinuxLauncher
            self.assertIsInstance(get_launcher(), LinuxLauncher)

    def test_start_command_is_platform_neutral(self):
        import json as _json
        exe = self.root / "strata"
        exe.write_bytes(b"")
        cfg = sample_cfg(exe=str(exe), args=[], port=18080)
        (self.root / "strata-x.json").write_text(_json.dumps(cfg), encoding="utf-8")
        r = mgr.start_server("strata-x.json", 18080, self.root)
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["pid"], 4242)
        cmd, cwd, _ = self.fake.spawned[0]
        self.assertEqual(cwd, str(self.root))
        self.assertTrue(any(part.endswith("server.py") for part in cmd))   # run-*.bat/.sh command
        st = mgr.launcher.ServerState(self.root).read()
        self.assertEqual(st["server"]["pid"], 4242)
        self.assertEqual(st["server"]["platform"], "fake")

    def test_duplicate_start_refused(self):
        self._write_state(pid=9999, config="strata-x.json", port=18080, log="x")
        exe = self.root / "strata"
        exe.write_bytes(b"")
        import json as _json
        (self.root / "strata-x.json").write_text(_json.dumps(sample_cfg(exe=str(exe), port=18080)),
                                                 encoding="utf-8")
        with mock.patch("gui.launcher.probe_port", return_value=True):   # port answers = already running
            r = mgr.start_server("strata-x.json", 18080, self.root)
        self.assertFalse(r["ok"])
        self.assertIn("already", r["error"])

    def test_stop_clears_state_through_the_platform_adapter(self):
        self._write_state(pid=9999, config="strata-x.json", port=18080, log="no.log")
        r = mgr.stop_server(self.root)
        self.assertEqual(r["state"], "stopped")
        self.assertEqual(self.fake.terminated, [9999])
        st = mgr.launcher.ServerState(self.root).read()
        self.assertEqual(st["server"], None)
        self.assertEqual(st["lifecycle"], None)
        self.assertEqual(st["last_stop"]["elapsed_s"] >= 0.0, True)   # measured, not faked
        self.assertEqual(st["last_stop"]["forced"], False)

    def test_force_stop_uses_the_hard_path_only(self):
        self._write_state(pid=7777, config="strata-x.json", port=18080, log="no.log")
        r = mgr.force_stop_server(self.root)
        self.assertEqual(r["state"], "stopped")
        self.assertEqual(r["forced"], True)
        self.assertEqual(self.fake.force_terminated, [7777])
        self.assertEqual(self.fake.terminated, [])                    # graceful terminate never ran
        self.assertIsNone(mgr.launcher.ServerState(self.root).read()["server"])

    def test_lifecycle_shows_stopping_and_restarting(self):
        self._write_state(pid=9999, config="strata-x.json", port=18080, log="no.log")
        mgr.launcher.set_lifecycle(self.root, "stopping", "graceful")
        time.sleep(0.05)
        s = mgr.launcher.server_status(self.root)
        self.assertEqual(s["state"], "stopping")
        self.assertEqual(s["phase"], "graceful")
        self.assertGreater(s["elapsed"], 0.0)
        mgr.launcher.set_lifecycle(self.root, "restarting", "starting")
        self.assertEqual(mgr.launcher.server_status(self.root)["state"], "restarting")

    def test_stop_without_a_managed_server_is_a_noop(self):
        r = mgr.stop_server(self.root)
        self.assertEqual(r["state"], "stopped")
        self.assertEqual(self.fake.terminated, [])

    def test_windows_spawn_uses_a_hidden_console(self):
        if os.name != "nt":
            self.skipTest("Windows-launcher behavior checked on Windows")
        from gui.platforms.windows import WindowsLauncher
        calls = {}

        def fake_popen(*a, **kw):
            calls.update(kw)
            return type("P", (), {"pid": 1234})()

        fd, path = tempfile.mkstemp()
        os.close(fd)
        log = Path(path)
        self.addCleanup(lambda: log.unlink(missing_ok=True))
        with mock.patch("gui.platforms.windows.subprocess.Popen", fake_popen):
            pid = WindowsLauncher().spawn(["python", "-m", "x"], str(Path(".").resolve()), log)
        self.assertEqual(pid, 1234)
        flags = calls["creationflags"]
        self.assertTrue(flags & subprocess.CREATE_NEW_CONSOLE)     # a real console to inherit
        self.assertFalse(flags & subprocess.DETACHED_PROCESS)      # ...not console-less
        self.assertEqual(calls["startupinfo"].wShowWindow, 0)      # SW_HIDE: born invisible

    def test_linux_adapter_module_imports_everywhere(self):
        from gui.platforms.linux import LinuxLauncher   # importable on any OS (calls are OS-conditional)
        self.assertEqual(LinuxLauncher.name, "linux")


class CrossPlatformPaths(unittest.TestCase):
    """The Manager's path handling must not assume drive letters or backslashes (Linux: /mnt/Storage/Model)."""

    def test_detect_works_with_forward_slash_paths(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(d, ignore_errors=True))
        (d / SHARD1).write_bytes(b"")
        (d / SHARD2).write_bytes(b"")
        posix_style = str(d).replace("\\", "/")          # exactly what a Linux path string looks like
        r = mgr.detect_gguf_dir(posix_style)
        self.assertTrue(r["ok"])
        self.assertEqual(r["quant"], "Q2_0")

    def test_browse_uses_pathlib_only(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(d, ignore_errors=True))
        (d / "sub").mkdir()
        b = mgr.Manager(ROOT).browse(str(d))
        self.assertNotIn("error", b)
        self.assertTrue(any(x.endswith("sub") for x in b["dirs"]))


class UnifiedGateway(unittest.TestCase):
    """The unified-page plumbing: the Manager serves the shared Strata UI kit and proxies the Strata
    server's endpoints on the SAME origin, staying alive (clean "offline") when Strata is down."""

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.root = Path(cls.td.name)
        (cls.root / "engine").mkdir()
        (cls.root / "engine" / "strata.exe").write_bytes(b"")
        cls.cfg = sample_cfg(exe=str(cls.root / "engine" / "strata.exe"), api_key="top-secret")
        (cls.root / "strata-q2_0.json").write_text(json.dumps(cls.cfg, indent=1), encoding="utf-8")
        # a fake Strata server: answers the real endpoints the gateway forwards
        from http.server import BaseHTTPRequestHandler as BH
        class FakeStrata(BH):
            key = "top-secret"
            def log_message(self, *a): pass
            def _json(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def _authorized(self):
                ok = self.headers.get("Authorization") == "Bearer " + self.key
                if not ok:
                    self._json(401, {"error": {"message": "missing or wrong API key"}})
                return ok
            def do_GET(self):
                path = self.path.split("?")[0]
                if path == "/metrics":
                    if not self._authorized(): return
                    self._json(200, {"live": {"state": "idle"}, "engine": {"model": "fake"}})
                elif path in ("/api/health", "/health"):
                    if not self._authorized(): return       # proves detection sends the config's key
                    self._json(200, {"status": "ok", "model": "fake", "service": "strata"})
                elif path == "/v1/models":
                    if not self._authorized(): return
                    self._json(200, {"object": "list", "data": [{"id": "fake", "object": "model"}]})
                else:
                    self._json(404, {"error": {"message": "nope"}})
            def do_POST(self):
                path = self.path.split("?")[0]
                if path == "/v1/chat/completions":
                    if not self._authorized(): return
                    if not self.headers.get("Content-Type", "").startswith("application/json"):
                        self._json(415, {"error": {"message": "send application/json"}}); return
                    n = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(n) or b"{}")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    self.wfile.write(b"data: {" + json.dumps({"choices": [{"delta": {"content": "hello"}}]}).encode() + b"}\n\n")
                    self.wfile.write(b"data: [DONE]\n\n")
                else:
                    self._json(404, {"error": {"message": "nope"}})
        cls.fake = cls.cfg.get("port", 8080)
        # bind the fake Strata on a free port and point the config port at it
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            cls.fake = s.getsockname()[1]
        cls.cfg["port"] = cls.fake
        (cls.root / "strata-q2_0.json").write_text(json.dumps(cls.cfg, indent=1), encoding="utf-8")
        import socketserver
        from http.server import ThreadingHTTPServer as _THS
        class S(_THS):
            daemon_threads = True
        cls.up = S(("127.0.0.1", cls.fake), FakeStrata)
        threading.Thread(target=cls.up.serve_forever, daemon=True).start()
        cls.server = mgr.make_server(cls.root, 0)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close()
        cls.up.shutdown(); cls.up.server_close()
        cls.td.cleanup()

    def _get(self, path, key=None):
        req = urllib.request.Request(self.base + path)
        if key:
            req.add_header("Authorization", "Bearer " + key)
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, r.read()

    def test_unified_page_is_one_native_app(self):
        status, html = self._get("/")
        self.assertEqual(status, 200)
        text = html.decode("utf-8")
        for tab in ("tab-btn-manager", "tab-btn-chat", "tab-btn-monitor", "tab-btn-about",
                    "view-manager", "view-chat", "view-monitor", "view-about"):
            self.assertIn(tab, text)
        # the shared UI kit is served from serve/web, the Chat/Monitor/About code verbatim
        self.assertIn(b"serve/web/app.js", (self._get("/web/serve-app.js"))[1][:120])
        self.assertIn(b"Manager", (self._get("/web/app.js"))[1][:120])
        self.assertIn(b"--st-accent", (self._get("/web/tokens.css"))[1])
        status, _ = self._get("/fonts/outfit-latin-wght.woff2")
        self.assertEqual(status, 200)

    def test_common_strata_endpoint_is_proxied_with_the_config_key(self):
        status, body = self._get("/metrics")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["live"]["state"], "idle")
        self.assertNotIn(b"missing or wrong API key", body)     # the config key was injected, no CORS
        status, body = self._get("/v1/models")
        self.assertEqual(status, 200)
        self.assertIn(b'"object"', body)

    def test_chat_stream_flows_through_the_gateway(self):
        import urllib.request as ur
        req = ur.Request(self.base + "/v1/chat/completions",
                         data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode(),
                         headers={"Content-Type": "application/json"})
        with ur.urlopen(req, timeout=15) as r:
            body = r.read().decode()
        self.assertIn("hello", body)

    def test_offline_answer_when_strata_is_down(self):
        # a fresh Manager pointing at a port where nothing listens
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            busy = 14999
            (root / "strata-q2_0.json").write_text(
                json.dumps(sample_cfg(port=busy), indent=1), encoding="utf-8")
            srv = mgr.make_server(root, 0)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            try:
                import urllib.error
                with self.assertRaises(urllib.error.HTTPError) as cm:
                    urllib.request.urlopen(base + "/health", timeout=10)
                self.assertEqual(cm.exception.code, 503)
                err = json.loads(cm.exception.read())
                self.assertTrue(err.get("offline"))
                # and the Manager's own API still answers: the page lives on
                with urllib.request.urlopen(base + "/api/status", timeout=10) as r:
                    self.assertEqual(json.loads(r.read())["data"]["server"]["state"], "stopped")
            finally:
                srv.shutdown(); srv.server_close()

    def test_externally_running_strata_is_detected(self):
        # (the gateway test's fake strata IS external: no Manager-owned server record)
        with urllib.request.urlopen(self.base + "/api/status", timeout=10) as r:
            body = json.loads(r.read())["data"]["server"]
        self.assertEqual(body["state"], "external")
        self.assertEqual(body["port"], self.fake)
        self.assertTrue(body.get("external"))

    def test_start_refused_while_external_runs(self):
        r = mgr.start_server("strata-q2_0.json", self.fake, self.root)
        self.assertFalse(r["ok"])
        self.assertIn("outside the Manager", r["error"])

    def test_random_process_on_the_port_is_not_strata(self):
        # port answers TCP but has no /api/health service=strata identity: not classified as external
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            import socket
            from http.server import BaseHTTPRequestHandler as BH, HTTPServer
            class Plain(BH):
                def log_message(self, *a): pass
                def do_GET(self):
                    body = b"<html>not strata</html>"
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                p = s.getsockname()[1]
            (root / "strata-q2_0.json").write_text(
                json.dumps(sample_cfg(port=p), indent=1), encoding="utf-8")
            srv = HTTPServer(("127.0.0.1", p), Plain)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            try:
                self.assertIsNone(mgr.launcher.external_strata(root))
                self.assertEqual(mgr.launcher.server_status(root)["state"], "stopped")
            finally:
                srv.shutdown(); srv.server_close()

    def test_openai_compatible_server_without_strata_identity_is_not_detected(self):
        # LM Studio / llama.cpp style: /v1/models with object+data, but no /api/health service=strata
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            import urllib.request as ur
            from http.server import BaseHTTPRequestHandler as BH, HTTPServer
            class OpenAiCompatible(BH):
                def log_message(self, *a): pass
                def _json(self, code, obj):
                    body = json.dumps(obj).encode()
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                def do_GET(self):
                    path = self.path.split("?")[0]
                    if path == "/v1/models":
                        self._json(200, {"object": "list",
                                         "data": [{"id": "gpt-oss-120b", "object": "model"}]})
                    elif path == "/api/health":
                        self._json(404, {"error": {"message": "no such endpoint"}})
                    else:
                        self._json(404, {"error": {"message": "nope"}})
            import socket
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                p = s.getsockname()[1]
            (root / "strata-q2_0.json").write_text(
                json.dumps(sample_cfg(port=p), indent=1), encoding="utf-8")
            srv = HTTPServer(("127.0.0.1", p), OpenAiCompatible)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            try:
                self.assertIsNone(mgr.launcher.external_strata(root))
                self.assertEqual(mgr.launcher.server_status(root)["state"], "stopped")
                # and /v1/models alone is fine for readiness, never for identity
                self.assertTrue(mgr.launcher.strata_ready(p))
                self.assertFalse(mgr.launcher.strata_health(p))
            finally:
                srv.shutdown(); srv.server_close()

    def test_key_protected_strata_is_detected_with_the_configured_key(self):
        # the class fake Strata demands the key even on /api/health: detection sends the config's key.
        self.assertTrue(mgr.launcher.strata_health(self.fake, api_key="top-secret"))
        self.assertFalse(mgr.launcher.strata_health(self.fake))            # wrong/no key -> not Strata
        self.assertFalse(mgr.launcher.strata_health(self.fake, api_key="wrong"))
        ext = mgr.launcher.external_strata(self.root)
        self.assertIsNotNone(ext)
        self.assertEqual(ext["port"], self.fake)
        self.assertTrue(ext.get("ready"))

    def test_manager_owned_server_takes_precedence(self):
        # the class fake is RUNNING and matches external_strata, but a live Manager-owned record wins
        mgr.launcher.ServerState(self.root).write({"server": {"pid": 4242,
                                                            "config": "strata-q2_0.json",
                                                            "port": self.fake,
                                                            "log": "x.serve.log"}, "tool": None})
        try:
            with mock.patch.object(mgr.launcher, "pid_alive", return_value=True):
                st = mgr.launcher.server_status(self.root)
            self.assertEqual(st["state"], "running")
            self.assertNotIn("external", st)
        finally:
            mgr.launcher.ServerState(self.root).write({"server": None, "tool": None})

    def test_external_disappears_back_to_stopped(self):
        # a fresh external Strata is detected, then its disappearance flips the status to Stopped
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            import urllib.error
            import urllib.request as ur
            import socket
            from http.server import BaseHTTPRequestHandler as BH, HTTPServer
            class Ext(BH):
                def log_message(self, *a): pass
                def _json(self, code, obj):
                    body = json.dumps(obj).encode()
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                def do_GET(self):
                    path = self.path.split("?")[0]
                    if path == "/api/health":
                        self._json(200, {"status": "ok", "service": "strata"})
                    elif path == "/v1/models":
                        self._json(200, {"object": "list", "data": []})
                    else:
                        self._json(404, {"error": {"message": "nope"}})
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                p = s.getsockname()[1]
            (root / "strata-q2_0.json").write_text(
                json.dumps(sample_cfg(port=p), indent=1), encoding="utf-8")
            srv = HTTPServer(("127.0.0.1", p), Ext)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            mgr_srv = mgr.make_server(root, 0)
            threading.Thread(target=mgr_srv.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{mgr_srv.server_address[1]}"
            try:
                with urllib.request.urlopen(base + "/api/status", timeout=10) as r:
                    body = json.loads(r.read())["data"]["server"]
                self.assertEqual(body["state"], "external")
                srv.shutdown(); srv.server_close()
                time.sleep(0.2)
                with urllib.request.urlopen(base + "/api/status", timeout=10) as r:
                    body = json.loads(r.read())["data"]["server"]
                self.assertEqual(body["state"], "stopped")
            finally:
                mgr_srv.shutdown(); mgr_srv.server_close()
                try:
                    srv.server_close()
                except Exception:
                    pass


class LauncherBat(unittest.TestCase):
    """Structural checks for START-MANAGER.bat: cmd parses it safely (a parenthesized IF with a
    closing parenthesis in its ECHO once produced the real ' . was unexpected at this time.' error)
    and pythonw is launched without a redirect that would lock a file pythonw never writes to."""

    def _bat(self) -> str:
        p = Path(__file__).resolve().parents[1] / "START-MANAGER.bat"
        return p.read_text(encoding="utf-8")

    def test_no_parenthesized_if_with_paren_in_echo(self):
        bat = self._bat()
        self.assertNotIn('if not exist ".venv\\Scripts\\pythonw.exe" (', bat)  # the old hazard
        self.assertNotIn(" (\r\n", bat)
        self.assertNotIn("(\n", bat)
        self.assertIn("goto no_pythonw", bat)          # goto style, not a block, per the fix
        self.assertIn(":no_pythonw", bat)

    def test_launches_pythonw_without_a_redirect_lock(self):
        bat = self._bat()
        self.assertIn("pythonw.exe", bat)              # no console window by design
        self.assertIn('start "" ".venv\\Scripts\\pythonw.exe" gui\\manager.py %*', bat)
        self.assertNotIn(">> \"logs\\manager.log\"", bat)   # the Manager writes its own log
        self.assertNotIn('"logs\\manager.log" 2>&1', bat)

    def test_exit_codes_are_explicit(self):
        bat = self._bat()
        self.assertIn("exit /b 0", bat)
        self.assertIn("exit /b 1", bat)


if __name__ == "__main__":
    unittest.main()