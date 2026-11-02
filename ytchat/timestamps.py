"""Convert between seconds and YouTube-style timestamps like 4:05 or 1:02:03."""


def fmt_ts(seconds: float) -> str:
    total = int(max(seconds, 0))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def parse_ts(label: str) -> int:
    total = 0
    for part in label.strip().split(":"):
        total = total * 60 + int(part)
    return total
