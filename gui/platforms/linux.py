"""Linux/Unix process handling for the Manager's launcher.

Spawn: a new session (start_new_session=True) - the server becomes a process-group leader, and everything it
starts (the engine, the vision encoder, the MCP servers: plain children) stays in that group.  Stop: SIGTERM
to the group first - serve/server.py's own signal handler turns it into a graceful shutdown (it sends QUIT to
the engine) - then SIGKILL to the group after the grace.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time

from .base import Launcher


class LinuxLauncher(Launcher):
    name = "linux"

    def spawn(self, cmd: list, cwd: str, log_file) -> int:
        f = open(log_file, "a", encoding="utf-8")
        try:
            proc = subprocess.Popen(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True)
        finally:
            f.close()
        return proc.pid

    def terminate(self, pid: int, grace_s: float) -> bool:
        try:
            os.killpg(int(pid), signal.SIGTERM)              # graceful: server.py QUITs the engine
        except OSError:
            pass
        for _ in range(max(1, int(grace_s * 2))):
            if not self.alive(pid):
                return True
            time.sleep(0.5)
        try:
            os.killpg(int(pid), signal.SIGKILL)
        except OSError:
            pass
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