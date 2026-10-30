import ctypes
import hashlib
import json
import os
import shutil
import signal
import socket
import struct
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HEADER = struct.Struct("<ii")
LIBC = ctypes.CDLL(None, use_errno=True)


def die_with_parent():
    LIBC.prctl(1, signal.SIGKILL)


def sweep(parent):
    for directory in parent.glob("kernel-*"):
        pid = directory / "pid"
        try:
            alive = pid.exists() and Path(f"/proc/{int(pid.read_text())}").exists()
        except (ValueError, FileNotFoundError):
            alive = False
        try:
            stale = not alive and (pid.exists() or time.time() - directory.stat().st_mtime > 3600)
        except FileNotFoundError:
            continue
        if stale:
            shutil.rmtree(directory, ignore_errors=True)


REQUEST = struct.Struct("<i")
BOOT = 120


class KernelError(RuntimeError):
    pass


def game_directory():
    default = Path.home() / ".local/share/Steam/steamapps/common/Slay the Spire 2"
    return Path(os.environ.get("STS2_DIR", default)).resolve()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def engine():
    source = game_directory()
    binary = source / "SlayTheSpire2"
    package = ROOT / "artifacts/dist/ai4sts2"
    if not binary.is_file() or not (package / "ai4sts2.dll").is_file():
        raise FileNotFoundError("Install STS2 and run scripts/build.sh first.")
    pack = binary.with_suffix(".pck").stat()
    game = digest(source / "data_sts2_linuxbsd_x86_64/sts2.dll")
    stamp = hashlib.sha256(f"{pack.st_size}:{pack.st_mtime_ns}".encode()).hexdigest()[:6]
    target = ROOT / "artifacts/engine" / f"{game}{stamp}-{digest(package / 'ai4sts2.dll')}"
    if not (target / binary.name).exists():
        staging = target.with_name(f"{target.name}.{os.getpid()}")
        ignore = shutil.ignore_patterns("mods", "steam_appid.txt", "libsentry*")
        shutil.copytree(source, staging, copy_function=os.link, ignore=ignore)
        shutil.copytree(package, staging / "mods/ai4sts2")
        try:
            staging.rename(target)
        except OSError:
            shutil.rmtree(staging, ignore_errors=True)
    return target / binary.name


class Kernel:
    def __init__(self, executable, connect=True):
        parent = ROOT / "artifacts/kernels"
        parent.mkdir(parents=True, exist_ok=True)
        sweep(parent)
        self.directory = Path(tempfile.mkdtemp(prefix="kernel-", dir=parent))
        self.socket = None
        self.process = None
        self.listener = None
        self.failed = False
        try:
            self._spawn(executable)
            if connect:
                self.connect()
        except BaseException:
            self.failed = True
            self.close()
            raise

    def _spawn(self, executable):
        profile = self.directory / "data/SlayTheSpire2"
        settings = profile / "default/1/settings.save"
        settings.parent.mkdir(parents=True)
        settings.write_text(
            json.dumps(
                {
                    "schema_version": 8,
                    "mod_settings": {"mods_enabled": True, "mod_list": []},
                    "volume_master": 0,
                    "skip_intro_logo": True,
                    "seen_ea_disclaimer": True,
                    "fullscreen": False,
                    "language": "eng",
                }
            )
        )
        (profile / ".ai4sts2-kernel").touch()
        address = self.directory / "kernel.sock"
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(str(address))
        self.listener.listen(1)
        self.listener.settimeout(1)
        command = [
            str(executable),
            "--headless",
            "--audio-driver",
            "Dummy",
            "--force-steam=off",
            "--ai4sts2-kernel",
        ]
        self.log = (self.directory / "game.log").open("wb")
        self.process = subprocess.Popen(
            command,
            cwd=Path(executable).parent,
            env=os.environ
            | {
                "XDG_DATA_HOME": str(self.directory / "data"),
                "XDG_CONFIG_HOME": str(self.directory / "config"),
                "AI4STS2_SOCKET": str(address),
                "DOTNET_GCgen0size": "0x4000000",
                "DOTNET_gcConcurrent": "0",
                "DOTNET_TieredPGO": "0",
            },
            stdin=subprocess.DEVNULL,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=die_with_parent,
        )
        temporary = self.directory / "pid.tmp"
        temporary.write_text(str(self.process.pid))
        temporary.replace(self.directory / "pid")
        self.started = time.monotonic()

    def connect(self):
        try:
            while self.socket is None:
                if self.process.poll() is not None or time.monotonic() > self.started + BOOT:
                    raise KernelError(f"Kernel did not connect. {self.tail()}")
                try:
                    self.socket, _ = self.listener.accept()
                except TimeoutError:
                    pass
            self.socket.settimeout(None)
            self.listener.close()
            self.listener = None
            self.vocab = [None]
            self.hello = self.call("hello")
            self.schema = self.hello["schema"]
        except BaseException:
            self.failed = True
            self.close()
            raise

    def tail(self, lines=40):
        path = self.directory / "game.log"
        text = path.read_text(errors="replace") if path.exists() else ""
        return "\n".join(text.splitlines()[-lines:])

    def call(self, op, **arguments):
        self.send(op, **arguments)
        return self.receive()

    def send(self, op, **arguments):
        body = json.dumps({"op": op, **arguments}).encode()
        try:
            self.socket.sendall(REQUEST.pack(len(body)) + body)
        except OSError as error:
            self.failed = True
            raise KernelError(f"Kernel connection failed. {self.tail()}") from error

    def receive(self):
        try:
            header = self._exactly(HEADER.size)
            size, binary = HEADER.unpack(header)
            response = json.loads(self._exactly(size))
            payload = self._exactly(binary) if binary else None
        except (OSError, EOFError) as error:
            self.failed = True
            raise KernelError(f"Kernel connection failed. {self.tail()}") from error
        if not response["ok"]:
            self.failed = True
            raise KernelError(response["error"])
        result = response.get("result")
        if payload is not None:
            self.vocab.extend(result.pop("vocab"))
            values = np.frombuffer(payload, dtype=np.float32)
            observation, offset = {}, 0
            for table, columns in self.schema.items():
                rows = result["layout"][table]
                count = rows * len(columns)
                observation[table] = values[offset : offset + count].reshape(rows, len(columns))
                offset += count
            if offset != values.size:
                raise KernelError("Observation layout does not match its payload.")
            result["observation"] = observation
        return result

    def fileno(self):
        return self.socket.fileno()

    def _exactly(self, size):
        buffer = bytearray(size)
        view = memoryview(buffer)
        while view:
            count = self.socket.recv_into(view)
            if not count:
                raise EOFError
            view = view[count:]
        return buffer

    def close(self):
        if self.listener is not None:
            self.listener.close()
            self.listener = None
        graceful = False
        if self.socket is not None:
            try:
                if not self.failed:
                    self.call("close")
                    graceful = True
            except KernelError:
                pass
            self.socket.close()
            self.socket = None
        if self.process is not None:
            if not graceful and self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGKILL)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.failed = True
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
            if graceful and self.process.returncode:
                self.failed = True
            self.process = None
        if getattr(self, "log", None) is not None:
            self.log.close()
            self.log = None
        if not self.failed:
            shutil.rmtree(self.directory, ignore_errors=True)
        elif self.directory.name.startswith("kernel-"):
            name = self.directory.name.removeprefix("kernel-")
            failed = self.directory.with_name(f"failed-{name}")
            self.directory.rename(failed)
            self.directory = failed

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
