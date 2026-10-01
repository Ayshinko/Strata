"""The launcher interface: how a platform spawns and terminates the Strata server.  Small on purpose."""

from __future__ import annotations


class Launcher:
    """What every platform adapter provides.

    spawn(cmd, cwd, log_file) -> pid
        Start `cmd` detached, its stdout/stderr appended to `log_file`, working from `cwd`.  The returned
        process (or its group) must survive the Manager exiting, and everything Strata is expected to manage
        (the server itself and the engine / vision encoder / MCP servers it starts) must be reachable through
        terminate(pid, grace_s).

    terminate(pid, grace_s) -> bool
        Best-effort graceful stop of the whole Strata process tree; after `grace_s` it must be forced down.
        Returns True when nothing is alive any more.

    alive(pid) -> bool
        Is the process still running?  (os.kill(pid, 0): the platform-neutral existence probe.)
    """

    name = "base"

    def spawn(self, cmd: list, cwd: str, log_file):  # pragma: no cover - the interface
        raise NotImplementedError

    def terminate(self, pid: int, grace_s: float) -> bool:  # pragma: no cover
        raise NotImplementedError

    def alive(self, pid: int) -> bool:  # pragma: no cover
        raise NotImplementedError