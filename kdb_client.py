"""Hyper-optimized async KDB+ client built on qroissant.

Zero-copy pipeline:
  KDB (IPC) -> Rust parallel decode (SIMD) -> Arrow PyCapsules -> Polars

The standard decoded path (not raw streaming) is fastest because qroissant's
Rust-side parallel column decoder with SIMD is more efficient than any
Python-side streaming + decode approach.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Union, overload

import polars as pl
import qroissant as q

if TYPE_CHECKING:
    from types import TracebackType

__all__ = ["CadeKdb", "CadeKdbPoolConfig", "CadeKdbTransformConfig"]

_log = logging.getLogger(__name__)

# Wall-clock bound on a single prewarm attempt so that a stalled Rust-side
# handshake cannot freeze the connect path forever.
_PREWARM_TIMEOUT_S = 10.0

# ---------------------------------------------------------------------------
# Pre-computed constants
# ---------------------------------------------------------------------------

_SEP_RE = re.compile(r"[_.\-\s]+")
_IS_BOOL_COL_RE = re.compile(r"^[iI]s[A-Z]")  # "isActive" yes, "isin" no
_BOOL_TRUE = ["Y", "TRUE", "true"]
_BOOL_FALSE = ["N", "FALSE", "false"]
_BOOL_ALL = frozenset(_BOOL_TRUE + _BOOL_FALSE)
# Validation set includes "NA"/"" since those are nullified anyway.
_BOOL_VALID = frozenset(_BOOL_TRUE + _BOOL_FALSE + ["NA", ""])
_BOOL_VALID_SERIES = pl.Series("_v", list(_BOOL_VALID))  # Polars-native for is_in
_STR_DTYPES = frozenset({pl.String})
_LIST_STR_DT = pl.List(pl.String)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _parse_conn(
    conn_str: str | None,
    host: str | None,
    port: int | None,
    username: str | None,
    password: str | None,
) -> tuple[str, int, str | None, str | None]:
    """Parse KDB connection parameters from various formats.

    Supported formats:
      ``:host:port:user:pass``   (KDB classic)
      ``host:port:user:pass``
      ``host:port``  + separate *username* / *password* kwargs
      All four as keyword arguments
    """
    if conn_str is not None:
        s = conn_str.lstrip(":")
        # Cap splits at 3 so a password containing ``:`` is not silently
        # truncated at the first embedded colon.  Previously ``host:port:u:p:w``
        # split into 5 parts and everything after the first password colon
        # was lost, producing silent authentication failures on high-entropy
        # passwords.
        parts = s.split(":", 3)
        if len(parts) == 4:
            return parts[0], int(parts[1]), parts[2] or username, parts[3] or password
        if len(parts) == 3:
            return parts[0], int(parts[1]), parts[2] or username, password
        if len(parts) == 2:
            return parts[0], int(parts[1]), username, password
        raise ValueError(f"Cannot parse connection string: {conn_str!r}")
    if host is not None and port is not None:
        return host, int(port), username, password
    raise ValueError("Provide a connection string or host + port.")


@functools.lru_cache(maxsize=512)
def _to_camel(name: str) -> str:
    """Convert a column name to camelCase.  Results are cached."""
    if not name:
        return name
    if _SEP_RE.search(name):
        parts = [p for p in _SEP_RE.split(name) if p]
        if not parts:
            return name
        return parts[0].lower() + "".join(p.capitalize() for p in parts[1:])
    # PascalCase -> camelCase: lowercase first char only
    if name[0].isupper():
        return name[0].lower() + name[1:]
    return name


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CadeKdbTransformConfig:
    """Post-query DataFrame transformation options.

    All flags default to ``True`` for the opinionated fast path.

    Attributes
    ----------
    camel_case_headers : bool
        Rename columns to camelCase.
    nullify_na_strings : bool
        Replace ``"NA"`` and ``""`` with ``null`` in string columns.
    convert_boolean_strings : bool
        For columns matching ``is[A-Z]...`` whose distinct values are
        a subset of {Y, N, TRUE, FALSE, true, false}, cast to Int8 (0/1).
    bools_to_int8 : bool
        Cast native ``Boolean`` columns to ``Int8``.
    lazy : bool
        Return a ``LazyFrame`` instead of ``DataFrame``.
    """

    camel_case_headers: bool = True
    nullify_na_strings: bool = True
    convert_boolean_strings: bool = True
    bools_to_int8: bool = True
    lazy: bool = True


@dataclass(frozen=True, slots=True)
class CadeKdbPoolConfig:
    """Connection pool tuning parameters.

    ``idle_timeout_ms`` controls how long unused connections stay alive;
    expired connections are evicted automatically, satisfying the requirement
    to remove stale entries from the pool.

    Attributes
    ----------
    max_size : int
        Maximum connections in the pool.
    min_idle : int
        Idle connections to keep warm.
    checkout_timeout_ms : int
        Max wait for a connection checkout (ms).
    idle_timeout_ms : int
        Evict idle connections after this duration (ms).
    max_lifetime_ms : int
        Replace connections after this total lifetime (ms).
    test_on_checkout : bool
        Validate connections before handing them out.
    healthcheck_query : str
        q expression used for validation (``"::"`` = generic null).
    retry_attempts : int
        Retries after initial query failure.
    retry_backoff_ms : int
        Delay between retries (ms).
    prewarm : bool
        Eagerly create ``min_idle`` connections on ``connect()``.
    """

    max_size: int = 10
    min_idle: int = 2
    checkout_timeout_ms: int = 5_000
    idle_timeout_ms: int = 60_000
    max_lifetime_ms: int = 300_000
    test_on_checkout: bool = True
    healthcheck_query: str = "::"
    retry_attempts: int = 1
    retry_backoff_ms: int = 100
    prewarm: bool = True


# ---------------------------------------------------------------------------
# Arrow -> Polars conversion
# ---------------------------------------------------------------------------


def _expand_dict(d: q.Dictionary) -> pl.DataFrame:
    """Expand a qroissant Dictionary into a flat DataFrame.

    Keyed tables in KDB are represented as ``Dictionary(Table, Table)``.
    A naive ``pl.from_arrow(dict)`` collapses them into two struct columns
    named *keys* / *values* and loses column headers.  We detect this case
    and horizontally concatenate the key and value tables instead.

    If the key and value tables share column names (e.g. both carry
    ``date``), the value-side duplicates are dropped so that
    ``pl.concat(..., how="horizontal")`` does not raise ``DuplicateError``.
    """
    k, v = d.keys, d.values
    if isinstance(k, q.Table) and isinstance(v, q.Table):
        kdf = pl.from_arrow(k)
        vdf = pl.from_arrow(v)
        overlap = [c for c in vdf.columns if c in kdf.columns]
        if overlap:
            vdf = vdf.drop(overlap)
        return pl.concat([kdf, vdf], how="horizontal")
    # Non-table dictionary: let Polars handle via PyCapsule
    result = pl.from_arrow(d)
    return result.to_frame("value") if isinstance(result, pl.Series) else result


def _apply_transforms(df: pl.DataFrame, cfg: CadeKdbTransformConfig) -> pl.DataFrame:
    """Apply all configured transforms in minimal passes over the data."""
    schema = df.schema
    cols = df.columns

    # -- Pass 1: type coercions (char-list -> String, Boolean -> Int8) -----
    #    Only build expressions for columns that actually need transformation.
    dirty_exprs: list[pl.Expr] = []
    for c in cols:
        dt = schema[c]
        if dt == _LIST_STR_DT:
            dirty_exprs.append(pl.col(c).list.join("").alias(c))
        elif dt == pl.Boolean and cfg.bools_to_int8:
            dirty_exprs.append(pl.col(c).cast(pl.Int8))
    if dirty_exprs:
        df = df.with_columns(dirty_exprs)
        schema = df.schema  # refresh after dtype changes

    # -- Detect is[A-Z]* boolean-string candidates -------------------------
    bool_candidates: set[str] = set()
    if cfg.convert_boolean_strings:
        bool_candidates = {
            c
            for c in cols
            if schema[c] in _STR_DTYPES
            and _IS_BOOL_COL_RE.match(
                _to_camel(c) if cfg.camel_case_headers else c
            )
        }
        # Validate candidates: all unique non-null values must be boolean-like
        # OR a null-equivalent ("NA", "").  Uses Polars-native is_in().all().
        validated: set[str] = set()
        for c in bool_candidates:
            uniq_s = df[c].drop_nulls().unique()
            if len(uniq_s) > 0 and uniq_s.is_in(_BOOL_VALID_SERIES).all():
                validated.add(c)
        bool_candidates = validated

    # -- Pass 2: nullify "NA" / "" in string columns -----------------------
    #    Merge with boolean conversion for qualifying is* columns (Pass 2+3).
    if cfg.nullify_na_strings or bool_candidates:
        str_exprs: list[pl.Expr] = []
        for c in cols:
            if schema[c] not in _STR_DTYPES:
                continue
            if c in bool_candidates:
                # Merged null + boolean conversion in a single expression.
                str_exprs.append(
                    pl.when(pl.col(c).is_in(["NA", ""]))
                    .then(pl.lit(None).cast(pl.Int8))
                    .when(pl.col(c).is_in(_BOOL_TRUE))
                    .then(pl.lit(1).cast(pl.Int8))
                    .when(pl.col(c).is_in(_BOOL_FALSE))
                    .then(pl.lit(0).cast(pl.Int8))
                    .otherwise(pl.lit(None).cast(pl.Int8))
                    .alias(c)
                )
            elif cfg.nullify_na_strings:
                str_exprs.append(
                    pl.when(pl.col(c).is_in(["NA", ""]))
                    .then(None)
                    .otherwise(pl.col(c))
                    .alias(c)
                )
        if str_exprs:
            df = df.with_columns(str_exprs)

    # -- Pass 3: camelCase rename (metadata-only, no data copy) ------------
    if cfg.camel_case_headers:
        renames = {c: _to_camel(c) for c in df.columns}
        renames = {k: v for k, v in renames.items() if k != v}
        if renames:
            df = df.rename(renames)

    return df


def _to_polars(
    value: q.Value, cfg: CadeKdbTransformConfig
) -> pl.LazyFrame | pl.DataFrame:
    """Convert a qroissant Value to a (possibly lazy) Polars DataFrame."""
    if isinstance(value, q.Dictionary):
        df = _expand_dict(value)
    elif isinstance(value, q.Table):
        df = pl.from_arrow(value)  # zero-copy via __arrow_c_stream__
    elif isinstance(value, q.Atom):
        df = pl.DataFrame({"value": [value.as_py()]})
        return df.lazy() if cfg.lazy else df
    else:
        # Vector / List -> PyCapsule fallback
        result = pl.from_arrow(value)  # zero-copy via __arrow_c_array__
        df = result.to_frame("value") if isinstance(result, pl.Series) else result

    # Release Rust-side Arrow buffer reference as early as possible.
    del value

    df = _apply_transforms(df, cfg)
    return df.lazy() if cfg.lazy else df


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class CadeKdb:
    """High-performance async KDB+ client with connection pooling.

    **Pipeline**: KDB IPC -> Rust parallel decode -> Arrow zero-copy -> Polars

    Usable as an async context manager or standalone (with ``auto_connect``).

    Parameters
    ----------
    conn_str : str, optional
        Connection string in any of:
        ``:host:port:user:pass`` | ``host:port:user:pass`` | ``host:port``
    host, port, username, password
        Individual connection parameters. Fill gaps left by *conn_str*.
    connection_timeout_ms : int
        TCP connect timeout (milliseconds).
    query_timeout : float
        Default per-query timeout (seconds).
    auto_connect : bool
        Lazily create the pool on first query if ``connect()`` was not called.
    max_timeouts_before_reset : int
        Consecutive timeout/transport errors before the pool is destroyed and
        recreated (circuit-breaker).  Set to ``0`` to disable.
    pool : CadeKdbPoolConfig, optional
        Pool sizing, idle eviction, healthcheck, and retry settings.
    transform : CadeKdbTransformConfig, optional
        Default post-query transforms applied when ``output="polars"``.

    Examples
    --------
    Context manager (recommended)::

        async with CadeKdb(":kdb-host:7015") as kdb:
            lf = await kdb.query("select from trade where date = .z.d")
            df = lf.collect()

    Standalone with explicit lifecycle::

        kdb = CadeKdb(host="kdb-prod", port=5010, username="svc", password="pw")
        lf = await kdb.query("meta .mt.get[`.credit.refData]")
        await kdb.close()

    Concurrent queries (pool handles checkout/checkin)::

        async with CadeKdb(":host:7015") as kdb:
            lf_trade, lf_quote = await asyncio.gather(
                kdb.query("select from trade"),
                kdb.query("select from quote"),
            )
    """

    __slots__ = (
        "_endpoint",
        "_pool",
        "_pool_opts",
        "_decode_opts",
        "_transform",
        "_auto_connect",
        "_query_timeout",
        "_closed",
        "_lock",
        "_connect_lock",
        "_prewarm",
        "_consecutive_timeouts",
        "_max_timeouts_before_reset",
        "_last_success",
        "_created_at",
    )

    def __init__(
        self,
        conn_str: str | None = None,
        *,
        host: str | None = None,
        port: int | None = None,
        username: str | None = None,
        password: str | None = None,
        connection_timeout_ms: int = 5_000,
        query_timeout: float = 30.0,
        auto_connect: bool = True,
        max_timeouts_before_reset: int = 3,
        pool: CadeKdbPoolConfig | None = None,
        transform: CadeKdbTransformConfig | None = None,
    ) -> None:
        h, p, u, pw = _parse_conn(conn_str, host, port, username, password)
        pcfg = pool or CadeKdbPoolConfig()

        self._endpoint = q.Endpoint.tcp(
            h, p, username=u, password=pw, timeout_ms=connection_timeout_ms,
        )
        self._pool_opts = q.PoolOptions(
            max_size=pcfg.max_size,
            min_idle=pcfg.min_idle,
            checkout_timeout_ms=pcfg.checkout_timeout_ms,
            idle_timeout_ms=pcfg.idle_timeout_ms,
            max_lifetime_ms=pcfg.max_lifetime_ms,
            test_on_checkout=pcfg.test_on_checkout,
            healthcheck_query=pcfg.healthcheck_query,
            retry_attempts=pcfg.retry_attempts,
            retry_backoff_ms=pcfg.retry_backoff_ms,
        )
        self._decode_opts = (
            q.DecodeOptions.builder()
            .with_parallel(True)          # multi-threaded column decode
            .with_temporal_nulls(True)    # q null sentinels -> None
            .with_treat_infinity_as_null(True)
            .build()
        )
        self._transform = transform or CadeKdbTransformConfig()
        self._auto_connect = auto_connect
        self._query_timeout = query_timeout
        self._prewarm = pcfg.prewarm
        self._pool: q.AsyncPool | None = None
        self._closed = False
        self._lock = asyncio.Lock()
        # Separate lock serializes actual connect I/O so a cold-start stampede
        # of N concurrent queries produces exactly ONE prewarm, not N.
        self._connect_lock = asyncio.Lock()
        self._consecutive_timeouts = 0
        self._max_timeouts_before_reset = max_timeouts_before_reset
        # Initialize last_success to creation time.  A literal 0.0 would make
        # every brand-new pool eligible for immediate eviction by the
        # maintenance sweep because monotonic() >> 0.
        now = time.monotonic()
        self._last_success = now
        self._created_at = now

    # -- Lifecycle ---------------------------------------------------------

    async def connect(self, *, connect_timeout: float | None = None) -> None:
        """Initialize the connection pool (idempotent).

        If ``prewarm`` is enabled, eagerly opens ``min_idle`` connections,
        bounded by ``_PREWARM_TIMEOUT_S`` so a stalled handshake cannot
        freeze the connect path.

        ``connect_timeout`` (optional) overrides the hardcoded prewarm
        cap with a tighter value (never a looser one).  The manager
        passes the caller's *remaining* deadline budget here so a slow
        handshake cannot silently blow the per-host budget: previously,
        ``connect()`` would happily wait the full ``_PREWARM_TIMEOUT_S``
        regardless of how much time the caller actually had left.

        A closed client stays closed: ``connect()`` on a closed instance
        raises ``RuntimeError`` instead of silently resurrecting — once
        ``close()`` has been called the client is terminal.

        ``_connect_lock`` serializes the slow I/O so N concurrent callers
        on a cold client produce exactly ONE prewarm, not N (preventing
        a reconnect storm against upstream KDB).
        """
        # Compute the prewarm deadline.  The caller-supplied value only
        # tightens the hardcoded cap — it never relaxes it.
        if connect_timeout is None:
            prewarm_timeout = _PREWARM_TIMEOUT_S
        elif connect_timeout <= 0:
            raise asyncio.TimeoutError(
                "connect_timeout expired before prewarm could start"
            )
        else:
            prewarm_timeout = min(connect_timeout, _PREWARM_TIMEOUT_S)

        # Fast check under state lock — are we already connected / closed?
        async with self._lock:
            if self._closed:
                raise RuntimeError("Client is closed; create a new instance.")
            if self._pool is not None:
                return

        # Serialize the actual connect I/O.  Other callers that lose the
        # race will wake up, find self._pool populated, and return.
        async with self._connect_lock:
            async with self._lock:
                if self._closed:
                    raise RuntimeError("Client is closed; create a new instance.")
                if self._pool is not None:
                    return

            # Slow I/O outside the state lock.
            pool = q.AsyncPool(
                self._endpoint,
                options=self._decode_opts,
                pool=self._pool_opts,
            )
            try:
                if self._prewarm:
                    await asyncio.wait_for(
                        pool.prewarm(), timeout=prewarm_timeout,
                    )
            except BaseException:
                with contextlib.suppress(Exception):
                    await pool.close()
                raise

            # Commit under state lock.  Defer pool.close() to outside the
            # lock to avoid blocking other tasks behind Rust-side I/O.
            discard = False
            async with self._lock:
                if self._closed:
                    discard = True
                else:
                    self._pool = pool

            if discard:
                with contextlib.suppress(Exception):
                    await pool.close()
                raise RuntimeError("Client closed during connect().")

    async def close(self) -> None:
        """Shut down the pool, release all connections, reject new queries.

        Snapshot-and-nil under lock, then close outside to avoid blocking.
        After ``close()`` the client is terminal — subsequent ``connect()``
        calls raise ``RuntimeError`` instead of silently resurrecting.

        Exceptions from the underlying ``pool.close()`` are logged rather
        than silently swallowed — operators need this signal to diagnose
        resource leaks and shutdown-path bugs in production.
        """
        async with self._lock:
            old, self._pool = self._pool, None
            self._closed = True
        if old is not None:
            try:
                await old.close()
            except asyncio.CancelledError:
                raise
            except Exception:
                _log.exception("kdb pool close failed during CadeKdb.close()")

    async def _reset_pool(self) -> None:
        """Destroy and recreate the pool.

        Used as a circuit-breaker when consecutive timeouts suggest the pool
        holds poisoned connections that qroissant failed to reclaim.
        Reset does NOT flip ``_closed`` — the client remains usable.

        Exceptions from the underlying close are logged, not silently
        swallowed — a failing close under pressure is a diagnosis signal,
        not noise to hide.
        """
        async with self._lock:
            old = self._pool
            self._pool = None
            self._consecutive_timeouts = 0
        if old is not None:
            try:
                await asyncio.wait_for(old.close(), timeout=2.0)
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                _log.warning(
                    "kdb pool close timed out during reset; abandoning handle"
                )
            except Exception:
                _log.exception("kdb pool close failed during reset")
        # Next query triggers _ensure_pool -> connect() automatically.

    async def reset(self) -> None:
        """Force-reset the connection pool.

        Useful when an external system detects that this host is unhealthy.
        """
        await self._reset_pool()

    @property
    def last_success_time(self) -> float:
        """Monotonic timestamp of the last successful query.

        Initialized to the creation time so a brand-new pool is not
        immediately eligible for idle eviction by the maintenance sweep.
        """
        return self._last_success

    @property
    def last_activity_time(self) -> float:
        """Monotonic timestamp of most recent relevant activity.

        Returns the later of the creation time or the last successful
        query.  Used by the manager's idle-eviction sweep so that a fresh
        pool that has not served a query yet is not incorrectly trimmed.
        """
        return max(self._last_success, self._created_at)

    async def __aenter__(self) -> CadeKdb:
        if self._auto_connect:
            await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    @property
    def connected(self) -> bool:
        """``True`` if the pool is alive and accepting queries."""
        return self._pool is not None and not self._closed

    async def _ensure_pool(self) -> q.AsyncPool:
        """Return the live pool, auto-connecting if allowed.

        State transitions are checked under ``_lock`` to eliminate the
        TOCTOU between ``_pool``/``_closed`` reads that the previous
        lockless implementation exhibited.

        A bounded retry loop handles the narrow race where a concurrent
        ``_reset_pool()`` nils ``_pool`` between ``connect()`` returning
        and the post-connect re-lock — the previous implementation
        surfaced a spurious ``RuntimeError`` in that case, causing
        callers to observe transient failures during circuit-breaker
        resets that had nothing to do with their query.
        """
        for _ in range(3):
            async with self._lock:
                if self._pool is not None:
                    return self._pool
                if self._closed:
                    raise RuntimeError("Client is closed.")
                if not self._auto_connect:
                    raise RuntimeError(
                        "Not connected. Call connect() or set auto_connect=True."
                    )
            await self.connect()
            async with self._lock:
                if self._pool is not None:
                    return self._pool
                if self._closed:
                    raise RuntimeError("Client is closed.")
                # Reset raced with us — loop and try again.
        raise RuntimeError("Pool unavailable after connect() retries.")

    # -- Querying ----------------------------------------------------------

    @overload
    async def query(
        self,
        expr: str,
        *,
        timeout: float | None = ...,
        output: Literal["polars"] = ...,
        transform: CadeKdbTransformConfig | None = ...,
        decode: q.DecodeOptions | None = ...,
    ) -> pl.LazyFrame | pl.DataFrame: ...

    @overload
    async def query(
        self,
        expr: str,
        *,
        timeout: float | None = ...,
        output: Literal["arrow"],
        transform: CadeKdbTransformConfig | None = ...,
        decode: q.DecodeOptions | None = ...,
    ) -> q.Value: ...

    async def query(
        self,
        expr: str,
        *,
        timeout: float | None = None,
        output: Literal["polars", "arrow"] = "polars",
        transform: CadeKdbTransformConfig | None = None,
        decode: q.DecodeOptions | None = None,
    ) -> Union[pl.LazyFrame, pl.DataFrame, q.Value]:
        """Execute a q expression and return the result.

        Parameters
        ----------
        expr : str
            q expression to evaluate remotely.
        timeout : float, optional
            Per-query timeout in seconds.  Overrides the client default.
            Raises ``asyncio.TimeoutError`` on expiry.  Must be > 0.
        output : ``"polars"`` | ``"arrow"``
            ``"polars"`` (default): apply transforms, return LazyFrame/DataFrame.
            ``"arrow"``: return the raw qroissant ``Value`` (zero-copy PyCapsule).
        transform : CadeKdbTransformConfig, optional
            Override the client's default transforms for this single query.
        decode : DecodeOptions, optional
            Override the client's default qroissant decode options for this query
            (e.g. different symbol interpretation).
        """
        pool = await self._ensure_pool()
        t = timeout if timeout is not None else self._query_timeout

        if t <= 0:
            raise ValueError(f"timeout must be positive, got {t!r}")

        # -- execute with cancellation + timeout hardening -----------------
        try:
            value: q.Value = await asyncio.wait_for(
                pool.query(expr, decode=decode), timeout=t,
            )
        except asyncio.CancelledError:
            # External cancellation — let it propagate cleanly.
            # qroissant's Rust backend drops the future (and the connection)
            # via its Drop impl, so no explicit cleanup needed here.
            raise
        except (asyncio.TimeoutError, q.TransportError):
            # Dead socket or hung query — record for circuit-breaker.
            should_reset = False
            async with self._lock:
                self._consecutive_timeouts += 1
                if (
                    self._max_timeouts_before_reset > 0
                    and self._consecutive_timeouts >= self._max_timeouts_before_reset
                ):
                    should_reset = True
            if should_reset:
                await self._reset_pool()
            raise
        except q.PoolClosedError:
            # Pool was closed by a concurrent close() / reset().
            raise
        else:
            # Update success counters and timestamp under a single lock
            # so a racing reader (e.g. ``last_activity_time``) never sees
            # a torn pairing of reset-counter + stale timestamp.  Under
            # free-threaded / no-GIL CPython, a bare float attribute
            # write can tear relative to the counter reset.
            now = time.monotonic()
            async with self._lock:
                self._consecutive_timeouts = 0
                self._last_success = now

        if output == "arrow":
            return value

        cfg = transform or self._transform

        # Skip thread dispatch for scalar results — the overhead of
        # asyncio.to_thread (~50-200µs) exceeds the inline transform time.
        if isinstance(value, q.Atom):
            return _to_polars(value, cfg)

        # Run the Polars conversion off the event loop.  pl.from_arrow is
        # zero-copy (wraps Arrow memory), but the transform expressions
        # hold the GIL briefly.  to_thread keeps the loop responsive.
        return await asyncio.to_thread(_to_polars, value, cfg)

    # -- Diagnostics -------------------------------------------------------

    async def metrics(self) -> q.PoolMetrics:
        """Snapshot of pool occupancy, idle count, and configuration."""
        pool = await self._ensure_pool()
        return await pool.metrics()

    async def try_metrics(self) -> q.PoolMetrics | None:
        """Metrics snapshot that NEVER triggers auto-connect.

        Returns ``None`` if the pool is absent or the client is closed.
        Intended for maintenance sweeps and diagnostic callers that must
        not accidentally resurrect an evicted / reset pool — calling
        ``metrics()`` goes through ``_ensure_pool()`` which auto-connects
        under default settings, turning a read-only probe into a
        fire-and-forget reconnection storm under unstable conditions.
        """
        async with self._lock:
            if self._pool is None or self._closed:
                return None
            pool = self._pool
        return await pool.metrics()
