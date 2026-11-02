from ytchat.ratelimit import DAY, RateLimiter


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_per_minute_limit_resets_after_a_minute():
    clock = Clock()
    rl = RateLimiter(per_minute=3, per_day=0, global_per_day=0, clock=clock)
    assert [rl.check("a") for _ in range(3)] == [None, None, None]
    refused = rl.check("a")
    assert refused and "wait 60 seconds" in refused
    assert rl.check("b") is None  # other visitors are unaffected
    clock.now += 61
    assert rl.check("a") is None


def test_refused_requests_are_not_counted():
    clock = Clock()
    rl = RateLimiter(per_minute=1, per_day=0, global_per_day=0, clock=clock)
    assert rl.check("a") is None
    for _ in range(5):
        assert rl.check("a") is not None
    clock.now += 61
    assert rl.check("a") is None  # the refusals didn't extend the wait


def test_per_day_limit():
    clock = Clock()
    rl = RateLimiter(per_minute=0, per_day=2, global_per_day=0, clock=clock)
    assert rl.check("a") is None
    clock.now += 3600
    assert rl.check("a") is None
    clock.now += 3600
    assert "today's limit" in rl.check("a")
    clock.now += DAY
    assert rl.check("a") is None


def test_global_daily_cap_covers_all_visitors():
    clock = Clock()
    rl = RateLimiter(per_minute=0, per_day=0, global_per_day=3, clock=clock)
    assert [rl.check(ip) for ip in ("a", "b", "c")] == [None, None, None]
    assert "daily limit" in rl.check("d")


def test_zero_disables_every_limit():
    rl = RateLimiter(per_minute=0, per_day=0, global_per_day=0, clock=Clock())
    assert all(rl.check("a") is None for _ in range(500))
