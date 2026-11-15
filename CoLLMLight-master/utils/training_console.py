"""Stream training subprocesses to the terminal and one append-only run log."""
import codecs
import os
import re
import selectors
import signal
import subprocess
import sys
import time

from .experiment import training_log
from .training_config import ROOT


_LOG_OWNER = 'COLLMLIGHT_TRAIN_LOG'
_SIMULATION_DETAIL = re.compile(r'^(?:time|Step time):\s+[\d.eE+\-]+\s*$')


def run_logged_process(cfg, command, label, env):
    """One outer process owns logging; nested workers inherit its output pipes."""
    log_path = str((cfg.out/'logs/train.log').resolve())
    child_env = dict(env, PYTHONUNBUFFERED='1', **{_LOG_OWNER:log_path})
    if os.environ.get(_LOG_OWNER) == log_path:
        print(f'[{label}] starting', flush=True)
        return subprocess.run(command, cwd=ROOT, env=child_env)

    with training_log(cfg, label) as log:
        process = subprocess.Popen(command, cwd=ROOT, env=child_env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   bufsize=0, start_new_session=True)
        decoders = {stream: codecs.getincrementaldecoder('utf-8')('replace')
                    for stream in (process.stdout, process.stderr)}
        stdout_pending = ''
        started = last_output = last_notice = time.monotonic()

        def emit(stream, data, final=False):
            nonlocal stdout_pending
            text = decoders[stream].decode(data, final=final)
            log.write(text)
            log.flush()
            if stream is process.stderr:
                # Preserve tqdm carriage returns and ANSI cursor movement immediately.
                sys.stderr.write(text)
                sys.stderr.flush()
            else:
                stdout_pending += text
                lines = stdout_pending.splitlines(keepends=True)
                stdout_pending = ''
                for line in lines:
                    if not line.endswith(('\n', '\r')) and not final:
                        stdout_pending = line
                    elif not _SIMULATION_DETAIL.fullmatch(line.strip()):
                        sys.stdout.write(line)
                sys.stdout.flush()

        try:
            with selectors.DefaultSelector() as selector:
                for stream in decoders:
                    selector.register(stream, selectors.EVENT_READ)
                while selector.get_map():
                    for key, _ in selector.select(timeout=1):
                        chunk = os.read(key.fd, 65536)
                        if chunk:
                            last_output = time.monotonic()
                            emit(key.fileobj, chunk)
                        else:
                            emit(key.fileobj, b'', final=True)
                            selector.unregister(key.fileobj)
                    now = time.monotonic()
                    if now-last_output >= 30 and now-last_notice >= 30:
                        notice = f'\n[{label}] still running; elapsed {now-started:.0f}s, waiting for worker output\n'
                        sys.stderr.write(notice)
                        sys.stderr.flush()
                        log.write(notice)
                        log.flush()
                        last_notice = now
            return subprocess.CompletedProcess(command, process.wait())
        except BaseException as exc:
            if process.poll() is None:
                try:os.killpg(process.pid, signal.SIGINT if isinstance(exc, KeyboardInterrupt) else signal.SIGTERM)
                except ProcessLookupError:pass
            try:
                remaining = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                try:os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:pass
                remaining = process.communicate()
            for stream, data in zip(decoders, remaining):
                emit(stream, data, final=True)
            raise
        finally:
            process.stdout.close()
            process.stderr.close()
