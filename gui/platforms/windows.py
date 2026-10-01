"""Windows process handling for the Manager's launcher.

Spawn: a detached process (no console window) in its own process group.  Stop: close the whole process tree
with taskkill /T - the documented Strata outcome ("closing the console window stops the model": server,
engine, vision encoder and MCP children end together).  taskkill without /F posts a WM_CLOSE a windowless
detached server cannot answer, so after a short grace the tree is force-closed.
"""

from __future__ import annotations

import os
import subprocess
import time

from .base import Launcher


class WindowsLauncher(Launcher):
    name = "windows"

    def spawn(self, cmd: list, cwd: str, log_file) -> int:
        with open(log_file, "a", encoding="utf-8") as f:
            p = subprocess.Popen(
                cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
                close_fds=False)
        return p.pid

    def terminate(self, pid: int, grace_s: float) -> bool:
        subprocess.run(["taskkill", "/PID", str(pid), "/T"], capture_output=True, timeout=grace_s + 1)
        for _ in range(max(1, int(grace_s * 2))):
            if not self.alive(pid):
                return True
            time.sleep(0.5)
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=grace_s + 1)
        for _ in range(int(grace_s)):
            if not self.alive(pid):
                return True
            time.sleep(0.5)
        return not self.alive(pid)

    @staticmethod
    def alive(pid: int) -> bool:
        if not pid:
            return False
        try:
            os.kill(int(pid), 0)
            return True
        except OSError:
            return False