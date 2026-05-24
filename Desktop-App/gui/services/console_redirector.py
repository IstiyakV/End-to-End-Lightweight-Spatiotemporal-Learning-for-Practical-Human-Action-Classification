import sys
import queue


class ConsoleRedirector:
    """Thread-safe stdout/stderr interceptor using a queue.

    The background training thread calls write() which puts text into a
    queue.Queue — no direct UI calls from the thread.  The main thread
    drains the queue via a polling loop (after(120,...)) in the monitor
    frame, keeping Tk's event loop clean.
    """

    def __init__(self, log_queue: queue.Queue):
        self._queue = log_queue
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

    # ── stream interface ──────────────────────────────────────────
    def write(self, text: str):
        # Mirror to real terminal so VS Code / cmd output still works
        try:
            self._original_stdout.write(text)
        except Exception:
            pass
        if text:
            self._queue.put(text)

    def flush(self):
        try:
            self._original_stdout.flush()
        except Exception:
            pass

    # ── lifecycle ─────────────────────────────────────────────────
    def start(self):
        sys.stdout = self
        sys.stderr = self

    def stop(self):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
