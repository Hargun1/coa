"""Seeded byte-address generators and strict, streaming CSV trace import."""

import csv
import random
from pathlib import Path

WORKLOADS = ("sequential", "looping", "random", "zipf", "phase_shift", "mixed_scan")


def generate(name, length=25000, seed=0, line_bytes=64):
    if name not in WORKLOADS:
        raise ValueError(f"Unknown workload {name!r}")
    if length <= 0 or line_bytes <= 0:
        raise ValueError("length and line_bytes must be positive")
    rng = random.Random(seed)
    if name == "sequential":
        blocks = list(range(length))
    elif name == "looping":
        # A stable loop larger than L2. Offset depends on seed, size does not.
        offset = rng.randrange(4096)
        blocks = [offset + i % 192 for i in range(length)]
    elif name == "random":
        blocks = [rng.randrange(4096) for _ in range(length)]
    elif name == "zipf":
        blocks = rng.choices(range(4096), weights=[1 / (i + 1) ** 1.15 for i in range(4096)], k=length)
    elif name == "phase_shift":
        blocks = []
        for i in range(length):
            phase = min(3, 4 * i // length)
            # Four disjoint 384-line hotspots; 15% independent background noise.
            block = phase * 768 + rng.randrange(384) if rng.random() < 0.85 else rng.randrange(4096)
            blocks.append(block)
    else:
        blocks = []
        scan = 8192
        for i in range(length):
            # Alternating hot-region reuse and short streaming pollution bursts.
            if i % 512 < 384:
                blocks.append(rng.randrange(192))
            else:
                blocks.append(scan)
                scan += 1
    return [block * line_bytes for block in blocks]


def read_trace(path, limit=2_000_000):
    """Read address or address,operation CSV; only R/READ rows are accepted.

    Headerless one-address-per-line files are also supported. Addresses may use
    decimal or a 0x prefix. Blank rows and leading # comments are skipped.
    """
    addresses = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.reader(stream)
        address_column = 0
        operation_column = None
        first = True
        for number, row in enumerate(rows, 1):
            if not row or not any(cell.strip() for cell in row) or row[0].lstrip().startswith("#"):
                continue
            lowered = [cell.strip().lower() for cell in row]
            if first and "address" in lowered:
                address_column = lowered.index("address")
                operation_column = lowered.index("operation") if "operation" in lowered else None
                first = False
                continue
            first = False
            try:
                if operation_column is not None and row[operation_column].strip().upper() not in ("R", "READ"):
                    raise ValueError("only read operations are supported")
                raw = row[address_column].strip()
                address = int(raw, 16) if raw.lower().startswith("0x") else int(raw, 10)
                if not 0 <= address < 2 ** 64:
                    raise ValueError("address must fit an unsigned 64-bit integer")
            except (ValueError, IndexError) as error:
                raise ValueError(f"Invalid trace row {number}: {error}") from error
            addresses.append(address)
            if len(addresses) > limit:
                raise ValueError(f"Trace exceeds the {limit:,}-request limit")
    if not addresses:
        raise ValueError("Trace contains no addresses")
    return addresses

