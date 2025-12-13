import sys
import time
import shutil
import hashlib
import numpy as np
from collections import defaultdict
from threading import Lock

_service_counts = defaultdict(int)
_display_lock = Lock()
_last_display_lines = ["", ""]
_render_initialized = False
_MIN_CELL_WIDTH = 3


def _render_counts_locked():
    """Render a small two-row panel with service name and active process counts."""
    global _last_display_lines, _render_initialized

    if _service_counts:
        services = sorted(_service_counts.items())
        columns_data = [("Service", "Active processes")]
        columns_data.extend((name, str(cnt)) for name, cnt in services)
    else:
        columns_data = [("Service", "Active processes"), ("-", "0")]

    widths = [max(len(top), len(bottom), _MIN_CELL_WIDTH) for top, bottom in columns_data]

    separators = len(columns_data) - 1
    term_columns = max(shutil.get_terminal_size(fallback=(120, 20)).columns - 1, 10)

    def total_width(ws):
        return sum(ws) + 3 * separators

    excess = total_width(widths) - term_columns
    while excess > 0 and any(w > _MIN_CELL_WIDTH for w in widths):
        for idx in range(len(widths)):
            if excess == 0:
                break
            if widths[idx] > _MIN_CELL_WIDTH:
                widths[idx] -= 1
                excess -= 1
        if not any(w > _MIN_CELL_WIDTH for w in widths):
            break

    def fmt_cell(text, width):
        if len(text) > width:
            if width <= 1:
                trimmed = text[:width]
            elif width == 2:
                trimmed = text[:1] + "."
            elif width == 3:
                trimmed = text[:1] + ".."
            else:
                trimmed = text[: width - 3] + "..."
        else:
            trimmed = text
        return trimmed.ljust(width)

    line1 = " | ".join(fmt_cell(top, width) for (top, _), width in zip(columns_data, widths))
    line2 = " | ".join(fmt_cell(bottom, width) for (_, bottom), width in zip(columns_data, widths))
    lines = [line1, line2]

    if not _render_initialized:
        sys.stdout.write("\n" * len(lines))
        sys.stdout.flush()
        _render_initialized = True

    sys.stdout.write("\033[F" * len(lines))

    for idx, line in enumerate(lines):
        sys.stdout.write("\r\033[K" + line)
        sys.stdout.write("\n")

    _last_display_lines = lines
    sys.stdout.flush()


def _modify_service_count(service, delta):
    """Update active process count for a service and refresh the display."""
    if not service:
        return

    with _display_lock:
        new_count = _service_counts[service] + delta
        if new_count <= 0:
            _service_counts.pop(service, None)
        else:
            _service_counts[service] = new_count
        _render_counts_locked()

def run_burst(host, client_id, config, processes):
    """
    Execute ON/OFF traffic following an N-Burst model defined by a Q matrix.

    Args:
    - host: Containernet container (Host object)
    - client_id: unique client identifier (can include suffix, e.g., clientVod2_rtsp1)
    - config: client config dict (cmd, states, Q_matrix, limits)
    - processes: shared dict that tracks running processes by client_id
    """
    states = config["estados"]               # List of state names (e.g., ["OFF1", "OFF2", "ON1", "ON2"])
    Q = config["Q_matrix"]                   # Transition matrix (NxN) with rates and probabilities
    cmd_base = config["cmd"]                 # Base command with {tiempo}
    limits = config["limits"]                # List of (min, max) tuples per state for duration

    # Seed based on client_id for reproducibility (local RNG, leaves global untouched)
    seed = int(hashlib.sha256(client_id.encode()).hexdigest(), 16) % 2**32
    rng = np.random.default_rng(seed)

    state_idx = rng.choice(len(states))  # random initial state

    service = client_id.split("_")[0] if "_" in client_id else client_id

    while True:
        state = states[state_idx]
        exit_rate = -Q[state_idx][state_idx]  # exit rate from current state
        duration = rng.exponential(1 / exit_rate)
        
        # Clamp duration to the configured range for the state
        interval = int(min(max(limits[state_idx][0], duration), limits[state_idx][1]))

        if state.startswith("ON"):
            cmd = cmd_base.format(tiempo=interval)
            process = host.popen(cmd, shell=True)
            processes[client_id] = process
            _modify_service_count(service, 1)

        time.sleep(interval)

        if state.startswith("ON") and client_id in processes:
            processes[client_id].terminate()
            processes[client_id].wait()
            del processes[client_id]
            _modify_service_count(service, -1)

        # Choose the next state based on the current Q row
        row = Q[state_idx].copy()
        row[state_idx] = 0  # no self-transition
        probs = row / row.sum()  # normalize
        state_idx = rng.choice(len(states), p=probs)
