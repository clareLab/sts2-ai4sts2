from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Execution:
    fps: int = 60
    settle_frames: int = 3
    step_frames: int = 2
    non_interactive: bool = False
    fixed_fps: int = 0

    def __post_init__(self):
        if self.fps < 0 or self.settle_frames < 1 or self.step_frames < 1 or self.fixed_fps < 0:
            raise ValueError("Invalid execution settings.")

    def to_dict(self):
        return asdict(self)


REFERENCE = Execution()
CANDIDATES = (
    Execution(fps=240),
    Execution(fps=0, fixed_fps=60),
    Execution(fps=0, fixed_fps=60, settle_frames=1, step_frames=1),
    Execution(fps=0, fixed_fps=60, non_interactive=True),
)
