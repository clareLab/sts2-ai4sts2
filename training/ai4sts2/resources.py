import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

GIB = 1024**3
TRIAL_MEMORY = 11 * GIB // 4
COORDINATOR_MEMORY = GIB


def cgroups():
    path = Path("/proc/self/cgroup")
    if not path.is_file():
        return []
    relative = next(
        (line[3:] for line in path.read_text().splitlines() if line.startswith("0::")), ""
    )
    root = Path("/sys/fs/cgroup")
    directory = root / relative.lstrip("/")
    return [
        parent
        for parent in (directory, *directory.parents)
        if parent == root or root in parent.parents
    ]


def capacity(available=True):
    cpus = float(
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    )
    memory = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        fields = {
            line.split(":")[0]: int(line.split()[1]) * 1024
            for line in meminfo.read_text().splitlines()
        }
        memory = fields["MemAvailable"]
    for directory in cgroups():
        cpu_limit = directory / "cpu.max"
        if cpu_limit.is_file():
            quota, period = cpu_limit.read_text().split()
            if quota != "max":
                cpus = min(cpus, int(quota) / int(period))
        memory_limit = directory / "memory.max"
        if memory_limit.is_file() and (value := memory_limit.read_text().strip()) != "max":
            remaining = int(value)
            current = directory / "memory.current"
            if available and current.is_file():
                remaining -= int(current.read_text())
            memory = remaining if memory is None else min(memory, remaining)
    if memory is None:
        raise RuntimeError("Available memory could not be measured.")
    return cpus, max(0, memory)


@dataclass(frozen=True)
class Budget:
    cpus: float
    memory: int

    def concurrency(self, requested=0):
        if not math.isfinite(self.cpus) or self.cpus < 1 or self.memory < 4 * GIB:
            raise RuntimeError("Training requires one CPU and a 4 GiB memory budget.")
        maximum = min(int(self.cpus), (self.memory - COORDINATOR_MEMORY) // TRIAL_MEMORY)
        if requested < 0 or requested > maximum:
            raise ValueError(f"The resource budget supports at most {maximum} concurrent trials.")
        return requested or maximum

    def report(self, requested=0):
        return asdict(self) | {
            "concurrent_trials": self.concurrency(requested),
            "trial_memory": TRIAL_MEMORY,
        }


def budget():
    supplied = "AI4STS2_MEMORY_BUDGET" in os.environ
    cpus, memory = capacity(available=not supplied)
    if supplied:
        cpus = min(cpus, float(os.environ["AI4STS2_CPU_BUDGET"]))
        memory = min(memory, int(os.environ["AI4STS2_MEMORY_BUDGET"]))
    else:
        cpus = min(2.0, cpus)
        memory = min(8 * GIB, memory // 2)
    result = Budget(cpus, memory)
    result.concurrency()
    return result


if __name__ == "__main__":
    limits = budget()
    print(limits.cpus, limits.memory)
