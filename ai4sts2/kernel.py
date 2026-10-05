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

ROOT = Path(__file__).resolve().parents[1]
HEADER = struct.Struct("<i")


class KernelError(RuntimeError):
    pass


def game_directory():
    default = Path.home() / ".local/share/Steam/steamapps/common/Slay the Spire 2"
    return Path(os.environ.get("STS2_DIR", default)).resolve()


def engine():
    source = game_directory()
    binary = source / "SlayTheSpire2"
    package = ROOT / "artifacts/dist/ai4sts2"
    if not binary.is_file() or not (package / "ai4sts2.dll").is_file():
        raise FileNotFoundError("Install STS2 and run scripts/build.sh first.")
    target = ROOT / "artifacts/engine"
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name in {"mods", "steam_appid.txt"} or item.name.startswith("libsentry"):
            continue
        destination = target / item.name
        if item.name == binary.name:
            if (
                not destination.exists()
                or destination.stat().st_mtime_ns != item.stat().st_mtime_ns
            ):
                shutil.copy2(item, destination)
        elif not destination.is_symlink():
            destination.symlink_to(item.resolve())
    mods = target / "mods"
    if mods.exists():
        shutil.rmtree(mods)
    shutil.copytree(package, mods / "ai4sts2")
    return target / binary.name


class Kernel:
    def __init__(self, executable, boot_timeout=120, connect=True):
        parent = ROOT / "artifacts/kernels"
        parent.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="kernel-", dir=parent))
        self.socket = None
        self.process = None
        self.listener = None
        self.failed = False
        self.boot_timeout = boot_timeout
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
            },
            stdin=subprocess.DEVNULL,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.started = time.monotonic()

    def connect(self):
        try:
            while self.socket is None:
                if (
                    self.process.poll() is not None
                    or time.monotonic() > self.started + self.boot_timeout
                ):
                    raise KernelError(f"Kernel did not connect. {self.tail()}")
                try:
                    self.socket, _ = self.listener.accept()
                except TimeoutError:
                    pass
            self.socket.settimeout(None)
            self.listener.close()
            self.listener = None
            self.hello = self.call("hello")
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
            self.socket.sendall(HEADER.pack(len(body)) + body)
        except OSError as error:
            self.failed = True
            raise KernelError(f"Kernel connection failed. {self.tail()}") from error

    def receive(self):
        try:
            response = json.loads(self._receive())
        except (OSError, EOFError) as error:
            self.failed = True
            raise KernelError(f"Kernel connection failed. {self.tail()}") from error
        if not response["ok"]:
            self.failed = True
            raise KernelError(response["error"])
        return response.get("result")

    def fileno(self):
        return self.socket.fileno()

    def _receive(self):
        header = self._exactly(HEADER.size)
        return self._exactly(HEADER.unpack(header)[0])

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

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
