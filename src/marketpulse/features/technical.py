"""Pure technical indicator functions.

Every function here takes a plain, already-time-windowed ``list[float]`` (or
two aligned ones) — no timestamps, no I/O, no randomness — so each is
testable with nothing more than a list of numbers. Deciding *which* window
of observations feeds *which* feature, and whether that window has enough
history to trust, is ``features.windows`` and ``features.pipeline``'s job,
not this module's.

Every ratio/division guards its denominator: a zero or missing baseline
returns ``None``, never ``inf`` or ``nan``.
"""


def moving_average(values: list[float]) -> float | None:
    """Arithmetic mean of ``values``, or ``None`` if empty."""
    if not values:
        return None
    return sum(values) / len(values)


def ratio_to_baseline(current: float, baseline: float | None) -> float | None:
    """``current / baseline - 1``, or ``None`` if ``baseline`` is missing or zero."""
    if baseline is None or baseline == 0:
        return None
    return current / baseline - 1.0


def exponential_moving_average(values: list[float]) -> float | None:
    """EMA over ``values``, seeded at the first value.

    Uses ``alpha = 2 / (n + 1)``, the conventional smoothing factor for an
    n-period EMA, computed fresh from exactly the values given — so the same
    input list always yields the same result, with no hidden carry-over
    state between calls.
    """
    if not values:
        return None
    alpha = 2.0 / (len(values) + 1)
    ema = values[0]
    for value in values[1:]:
        ema = alpha * value + (1 - alpha) * ema
    return ema


def _returns(prices: list[float]) -> list[float] | None:
    """Consecutive pct-returns, or ``None`` if too short or a zero denominator appears."""
    if len(prices) < 2:
        return None
    returns = []
    for previous, current in zip(prices, prices[1:], strict=False):
        if previous == 0:
            return None
        returns.append((current - previous) / previous)
    return returns


def return_volatility(prices: list[float]) -> float | None:
    """Sample standard deviation of consecutive returns."""
    returns = _returns(prices)
    if returns is None or len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return float(variance**0.5)


def realised_volatility(prices: list[float]) -> float | None:
    """Root sum of squared consecutive returns (realised variance ** 0.5)."""
    returns = _returns(prices)
    if returns is None:
        return None
    return float(sum(r**2 for r in returns) ** 0.5)


def high_low_range(prices: list[float]) -> float | None:
    """``(max - min) / min`` over ``prices``."""
    if not prices:
        return None
    low, high = min(prices), max(prices)
    if low == 0:
        return None
    return (high - low) / low


def rate_of_change(prices: list[float]) -> float | None:
    """``(last - first) / first`` over the given window."""
    if len(prices) < 2:
        return None
    first = prices[0]
    if first == 0:
        return None
    return (prices[-1] - first) / first


def rsi(prices: list[float]) -> float | None:
    """Classic RSI (0-100) from average gain/loss over consecutive moves.

    A perfectly flat window (no gains, no losses) returns the neutral value
    50.0 rather than dividing zero by zero.
    """
    if len(prices) < 2:
        return None
    gains = []
    losses = []
    for previous, current in zip(prices, prices[1:], strict=False):
        delta = current - previous
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
    avg_gain = sum(gains) / len(gains)
    avg_loss = sum(losses) / len(losses)
    if avg_gain == 0 and avg_loss == 0:
        return 50.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def direction_streak(prices: list[float]) -> int:
    """Signed count of consecutive same-direction moves ending at the last price.

    Positive for an up-streak, negative for a down-streak, 0 if the most
    recent move was flat or there are fewer than two points.
    """
    if len(prices) < 2:
        return 0
    streak = 0
    direction = 0
    for current, previous in zip(reversed(prices), reversed(prices[:-1]), strict=False):
        delta = current - previous
        step = 1 if delta > 0 else (-1 if delta < 0 else 0)
        if step == 0:
            break
        if direction == 0:
            direction = step
        elif step != direction:
            break
        streak += 1
    return streak * direction
