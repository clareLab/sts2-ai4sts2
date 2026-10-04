import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from ai4sts2.execution import REFERENCE

ROOT = Path(__file__).resolve().parents[2]


class WorkerFailure(RuntimeError):
    pass


def game_path():
    return Path(
        os.environ.get(
            "STS2_DIR", Path.home() / ".local/share/Steam/steamapps/common/Slay the Spire 2"
        )
    ).resolve()


def prepare_game():
    source = game_path()
    binary = source / "SlayTheSpire2"
    package = ROOT / "artifacts/dist/ai4sts2"
    if not binary.is_file() or not (package / "ai4sts2.dll").is_file():
        raise FileNotFoundError("Install STS2 and run scripts/build.sh first.")
    target = ROOT / "artifacts/engine"
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name in {"mods", "steam_appid.txt"}:
            continue
        destination = target / item.name
        if item.name == binary.name:
            if (
                not destination.exists()
                or destination.stat().st_mtime_ns != item.stat().st_mtime_ns
            ):
                shutil.copy2(item, destination)
        elif not destination.exists():
            destination.symlink_to(item.resolve())
    shutil.copytree(package, target / "mods/ai4sts2", dirs_exist_ok=True)
    return target / binary.name


class OfficialGame:
    def __init__(
        self,
        executable=None,
        timeout=75,
        execution=REFERENCE,
        audit=False,
        raw_selection=False,
        diagnostic=False,
        diagnostic_event=None,
    ):
        if diagnostic_event and not diagnostic:
            raise ValueError("A diagnostic event requires diagnostic mode.")
        executable = Path(executable or prepare_game())
        self.timeout = timeout
        self.measurements = {}
        self._has_episode = False
        self._launch = {
            "executable": executable,
            "execution": execution,
            "audit": audit,
            "raw_selection": raw_selection,
            "diagnostic": diagnostic,
            "diagnostic_event": diagnostic_event,
        }
        self._open(**self._launch)

    def _open(self, executable, execution, audit, raw_selection, diagnostic, diagnostic_event):
        directory = ROOT / "artifacts/workers"
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="worker-", dir=directory))
        userdata = self.directory / "userdata"
        profile = userdata / "SlayTheSpire2"
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
                    "fps_limit": execution.fps,
                    "language": "eng",
                }
            )
        )
        (profile / ".ai4sts2-worker").touch()
        environment = os.environ | {
            "XDG_DATA_HOME": str(userdata),
            "XDG_CONFIG_HOME": str(self.directory / "config"),
            "LP_NUM_THREADS": "1",
            "DOTNET_PROCESSOR_COUNT": "2",
            "AI4STS2_EXECUTION": json.dumps(execution.to_dict()),
            "AI4STS2_AUDIT": "1" if audit else "0",
            "AI4STS2_RAW_SELECTION": "1" if raw_selection else "0",
            "AI4STS2_DIAGNOSTIC": "1" if diagnostic else "0",
            "AI4STS2_DIAGNOSTIC_EVENT": diagnostic_event or "",
        }
        command = [
            str(executable),
            "--headless",
            "--audio-driver",
            "Dummy",
            "--force-steam=off",
            "--ai4sts2-worker",
        ]
        if execution.fixed_fps:
            command += ["--fixed-fps", str(execution.fixed_fps)]
        if shutil.which("steam-run"):
            command.insert(0, "steam-run")
        self.sequence = 0
        self.responses = queue.Queue()
        self.log_path = self.directory / "game.log"
        self.log = self.log_path.open("w")
        self.journal = (self.directory / "requests.jsonl").open("w", buffering=1)
        self.process = subprocess.Popen(
            command,
            cwd=executable.parent,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            self.hello = self.request("hello")
            if self.hello["engine"] != "official" or self.hello["test_mode"]:
                raise RuntimeError("The worker is not running normal official game rules.")
            if self.hello.get("diagnostic", False) != diagnostic:
                raise RuntimeError("The worker diagnostic mode does not match the requested mode.")
        except BaseException:
            self.close()
            raise

    def _read(self):
        try:
            for line in self.process.stdout:
                if line.startswith("AI4STS2 "):
                    try:
                        self.responses.put(json.loads(line[8:]))
                    except json.JSONDecodeError:
                        self.log.write(f"Invalid worker response: {line!r}\n")
                        self.log.flush()
                        self.responses.put(None)
                else:
                    self.log.write(line)
                    self.log.flush()
        finally:
            self.responses.put(None)

    def request(self, method, parameters=None):
        started = time.perf_counter()
        restart_ms = 0.0
        if method == "reset":
            if getattr(self, "_has_episode", False) and not self._launch["execution"].reuse_process:
                self.close()
                self._open(**self._launch)
                restart_ms = (time.perf_counter() - started) * 1000
            self._has_episode = True
        if self.process.poll() is not None:
            raise WorkerFailure(
                f"Official worker exited ({self.process.returncode}). See {self.log_path}"
            )
        self.sequence += 1
        identifier = str(self.sequence)
        request = json.dumps({"id": identifier, "method": method, "params": parameters or {}})
        self.journal.write(request + "\n")
        try:
            self.process.stdin.write(request + "\n")
            self.process.stdin.flush()
        except OSError as error:
            self.close()
            raise WorkerFailure(
                f"Official worker connection failed. See {self.log_path}"
            ) from error
        try:
            response = self.responses.get(timeout=self.timeout)
        except queue.Empty as error:
            self.close()
            raise WorkerFailure(
                f"Official worker timed out during {method}: {self.log_path}"
            ) from error
        if not isinstance(response, dict) or response.get("id") != identifier:
            detail = {"expected_id": identifier, "response": response, "exit": self.process.poll()}
            self.close()
            raise WorkerFailure(f"Official worker protocol failed: {detail}. See {self.log_path}")
        if not response["ok"]:
            self.close()
            raise RuntimeError(response["error"])
        measurement = self.measurements.setdefault(method, {})
        timing = response.get("timing", {}) | {"restart_ms": restart_ms}
        wall_ms = (time.perf_counter() - started) * 1000
        for key, value in (timing | {"wall_ms": wall_ms, "calls": 1}).items():
            measurement[key] = measurement.get(key, 0) + value
        measurement["transport_ms"] = measurement.get("transport_ms", 0) + max(
            0, wall_ms - timing.get("engine_ms", wall_ms) - restart_ms
        )
        return response["result"]

    def drain_measurements(self):
        result, self.measurements = self.measurements, {}
        return result

    def close(self):
        import signal

        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=5)
        self.reader.join(timeout=2)
        for stream in (self.process.stdin, self.process.stdout, self.log, self.journal):
            stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def probe(character="IRONCLAD", steps=40):
    started = time.monotonic()
    with OfficialGame() as game:
        state = game.request(
            "reset", {"character": character, "seed": "AI4STS2-PROBE-1", "scope": "first_combat"}
        )
        print(json.dumps({"hello": game.hello, "startup_seconds": time.monotonic() - started}))
        for index in range(steps):
            print(json.dumps({"step": index, "state": state}), flush=True)
            if state["terminated"]:
                break
            state = game.request("step", {"revision": state["revision"], "action": 0})
