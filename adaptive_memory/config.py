from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    line_bytes: int = 64
    l1_lines: int = 32
    l2_lines: int = 128
    l3_lines: int = 512
    l1_ways: int = 4
    l2_ways: int = 8
    l3_ways: int = 8
    # Incremental serial lookup costs, in modeled cycles, not hardware measurements.
    l1_cycles: int = 4
    l2_cycles: int = 12
    l3_cycles: int = 40
    ram_cycles: int = 200
    horizon: int = 256
    learning_rate: float = 0.05
    min_labels: int = 64
    window: int = 1000

    def __post_init__(self):
        integer_fields = [k for k in asdict(self) if k != "learning_rate"]
        for key in integer_fields:
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.line_bytes & (self.line_bytes - 1):
            raise ValueError("line_bytes must be a power of two")
        for level in (1, 2, 3):
            lines = getattr(self, f"l{level}_lines")
            ways = getattr(self, f"l{level}_ways")
            if lines % ways:
                raise ValueError(f"L{level} lines must be divisible by associativity")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")

    def to_dict(self):
        return asdict(self)

