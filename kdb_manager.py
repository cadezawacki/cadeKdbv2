"""Multi-host KDB+ query orchestration built on CadeKdb.

Provides:
  - **CadeKdbManager** — pool registry with maintenance, multi-host failover,
    aggressive fan-out, credential rotation, and backdoor routing.
  - **query_kdb()** — convenience function matching the legacy API.

Architecture::

    query_kdb(expr, host, port, ...)
        |
        v
    CadeKdbManager.query(expr, hosts=[...])
        |
        +-- _sequential_query()          # try hosts one-by-one
        |       +-- _try_host()          # credential rotation per host
        |               v
        +-- _aggressive_query()          # fan-out, first wins
                +-- _try_host()
                        v
                CadeKdb.query(expr)    # single-pool, zero-copy pipeline
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import threading
import time
from collections import OrderedDict, UserList
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import zip_longest
from typing import TYPE_CHECKING, Any, Literal, Union, overload

import polars as pl
import qroissant as q

from kdb_client import CadeKdb, CadeKdbPoolConfig, CadeKdbTransformConfig

if TYPE_CHECKING:
    from types import TracebackType

_log = logging.getLogger(__name__)

# Hard bound on a single metrics probe during the maintenance sweep.  A
# stalled Rust-side call must never freeze the eviction loop.
_METRICS_TIMEOUT_S = 1.0

__all__ = [
    "CadeKdbManager",
    "HostConfig",
    "AuthCache",
    "ConnectionList",
    "fconn",
    "as_connection",
    "query_kdb",
    "_norm_arg_names",
]

# ---------------------------------------------------------------------------
# Error classifiers (ported from old kdbManager.py)
# ---------------------------------------------------------------------------

_AUTH_KEYWORDS = frozenset(
    {
        "auth",
        "credential",
        "permission",
        "unauthoriz",
        "access",
        "login",
        "password",
        "invalid user",
        "denied",
        "authentication",
    }
)
_TRANSPORT_KEYWORDS = frozenset(
    {
        "broken pipe",
        "connection reset",
        "connection refused",
        "connection abort",
        "network is unreachable",
        "remote closed",
        "transport is closing",
        "0 bytes read",
        "readexactly",
        "eof",
    }
)


def _is_auth_error(exc: BaseException) -> bool:
    """True if *exc* looks like a KDB authentication / credential failure."""
    if isinstance(exc, ConnectionRefusedError):
        return False
    msg = str(exc).lower()
    return any(kw in msg for kw in _AUTH_KEYWORDS)


def _is_transport_error(exc: BaseException) -> bool:
    """True if *exc* looks like a socket / IO failure."""
    if isinstance(exc, (EOFError, BrokenPipeError, OSError)):
        return True
    msg = str(exc).lower()
    return any(kw in msg for kw in _TRANSPORT_KEYWORDS)


# ---------------------------------------------------------------------------
# ConnectionList / fconn  (ported from old connections module)
# ---------------------------------------------------------------------------


def _interweave_lists(l1: list, l2: list | None = None, *args: list) -> list:
    l2 = [] if l2 is None else l2
    return [
        x
        for x in itertools.chain.from_iterable(zip_longest(l1, l2, *args))
        if x is not None
    ]


def _str_get_or_none(
    c: dict, k: str, v: Any, upper: bool = True, strict: bool = False
) -> Any:
    r = c.get(k, None if strict else (v.upper() if isinstance(v, str) and upper else v))
    return r.upper() if (isinstance(r, str) and upper) else r


def _check_condition(
    c: dict, k: str, v: Any, strict: bool, upper: bool = True
) -> bool:
    if isinstance(v, (list, tuple)):
        if not len(v):
            raise ValueError(f"Cannot filter on empty list: {k}, {v}")
        r = _str_get_or_none(c, k, v[0], upper, strict)
        rv = [vr.upper() if isinstance(vr, str) else vr for vr in v]
        return r in rv
    r = _str_get_or_none(c, k, v, upper, strict)
    return r == (v.upper() if (isinstance(v, str) and upper) else v)


class ConnectionList(UserList):
    """Thin list wrapper so downstream code can distinguish filtered configs
    from raw lists (``isinstance(cfg, ConnectionList)``)."""

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)


def fconn(
    config: list[dict] | dict,
    *args: list,
    weave: bool = True,
    strict: bool = True,
    **kwargs: Any,
) -> ConnectionList:
    """Filter and combine KDB connection configs.

    Parameters
    ----------
    config : list[dict] | dict
        Base connection config(s).
    *args : list
        Additional config lists to interweave or append.
    weave : bool
        If ``True`` (default), interweave *config* and *args* so entries
        alternate.  If ``False``, simple concatenation.
    strict : bool
        When ``True`` (default), filter keys must exist in the config dict.
    **kwargs
        Key/value filters applied to each config dict.  A config is kept
        only if it passes **all** filters.  Values may be scalars or
        lists (membership test).

    Returns
    -------
    ConnectionList
        Filtered subset of configs.

    Examples
    --------
    ::

        fconn(PANOPROXY, region="US", dbtype=["prod", "dr"])
    """
    config = config if isinstance(config, list) else [config]
    full_args = _interweave_lists(config, *args) if weave else config + list(args)
    # List-comprehension filter avoids the O(n*m) .remove() path and, more
    # importantly, the value-equality bug where two equal-but-distinct
    # config dicts cause the wrong entry to be removed.
    passes_filters = [
        c for c in full_args
        if all(_check_condition(c, k, v, strict) for k, v in kwargs.items())
    ]
    return ConnectionList(passes_filters)


def as_connection(config: list[dict] | dict) -> dict[str, Any]:
    """Normalize a config dict into ``{host, port, user, password}``."""
    if isinstance(config, list):
        config = config[0]
    return {
        "host": config.get("host"),
        "port": config.get("port"),
        "user": config.get("user", config.get("username", config.get("usr"))),
        "password": config.get("password", config.get("pass", config.get("pwd"))),
    }


# ---------------------------------------------------------------------------
# HostConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HostConfig:
    """Describes a single KDB host target for routing."""

    host: str
    port: int
    username: str | None = None
    password: str | None = None
    name: str | None = None
    backdoor: bool = False

    @property
    def key(self) -> str:
        base = f"{self.host}:{self.port}"
        return f"{base}:{self.name}" if self.name else base


def _resolve_hosts(
    *,
    host: str | None = None,
    port: int | None = None,
    hosts: Sequence[tuple[str, int]] | None = None,
    config: ConnectionList | Sequence[dict[str, Any]] | dict[str, Any] | None = None,
    name: str | None = None,
    username: str | None = None,
    password: str | None = None,
) -> list[HostConfig]:
    """Normalize the many input formats into a flat ``list[HostConfig]``.

    *config* is auto-detected:
      - Already a ``ConnectionList`` (from ``fconn``) → use directly.
      - A single ``dict`` → wrapped in a list.
      - A plain ``list[dict]`` → used as-is.
    """
    result: list[HostConfig] = []

    if config is not None:
        # Auto-wrap: single dict → list, ConnectionList → list, raw list → list
        if isinstance(config, dict):
            cfg_list: Sequence[dict[str, Any]] = [config]
        elif isinstance(config, ConnectionList):
            cfg_list = list(config)
        else:
            cfg_list = config

        for c in cfg_list:
            result.append(
                HostConfig(
                    host=str(c.get("host", "")),
                    port=int(c.get("port", 0)),
                    username=c.get("usr") or c.get("user") or c.get("username"),
                    password=(
                        c.get("pwd") or c.get("pass") or c.get("password")
                    ),
                    name=name or c.get("name"),
                    backdoor=bool(c.get("backdoor", False)),
                )
            )
    elif hosts is not None:
        for h, p in hosts:
            result.append(
                HostConfig(
                    host=h, port=int(p), username=username,
                    password=password, name=name,
                )
            )
    elif host is not None and port is not None:
        result.append(
            HostConfig(
                host=host, port=int(port), username=username,
                password=password, name=name,
            )
        )
    else:
        raise ValueError("Provide host+port, hosts=[], or config=[].")

    if not result:
        raise ValueError("No hosts resolved.")
    return result


# ---------------------------------------------------------------------------
# AuthCache
# ---------------------------------------------------------------------------


class AuthCache:
    """LRU-bounded cache of successful (host, port) -> (user, password) mappings.

    Uses an explicit ``threading.Lock`` so that concurrent reads/writes are
    safe under free-threaded / no-GIL CPython builds, not just relying on
    the implicit atomicity of individual ``OrderedDict`` ops under the GIL.
    """

    __slots__ = ("_cache", "_maxsize", "_lock")

    def __init__(self, maxsize: int = 1024) -> None:
        self._cache: OrderedDict[tuple[str, int], tuple[str, str]] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, host: str, port: int) -> tuple[str, str] | None:
        with self._lock:
            pair = self._cache.get((host, port))
            if pair is not None:
                self._cache.move_to_end((host, port))
            return pair

    def remember(self, host: str, port: int, user: str, password: str) -> None:
        key = (host, port)
        with self._lock:
            self._cache[key] = (user, password)
            self._cache.move_to_end(key)
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def forget(self, host: str, port: int) -> None:
        with self._lock:
            self._cache.pop((host, port), None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


# ---------------------------------------------------------------------------
# Credential iteration
# ---------------------------------------------------------------------------


def _iter_credentials(
    hc: HostConfig,
    auth_cache: AuthCache,
    default_creds: Sequence[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Build an ordered list of (user, password) to try for *hc*.

    Priority: explicit on HostConfig > auth cache > default combos.
    Deduplicates while preserving order.
    """
    seen: set[tuple[str, str]] = set()
    result: list[tuple[str, str]] = []

    def _add(u: str | None, p: str | None) -> None:
        if u is None or p is None:
            return
        pair = (u, p)
        if pair not in seen:
            seen.add(pair)
            result.append(pair)

    _add(hc.username, hc.password)

    cached = auth_cache.get(hc.host, hc.port)
    if cached:
        _add(cached[0], cached[1])

    for u, p in default_creds:
        _add(u, p)

    return result


# ---------------------------------------------------------------------------
# CadeKdbManager
# ---------------------------------------------------------------------------

_DEFAULT_CREDS: list[tuple[str, str]] = [
    ("credituser", "creditpass"),
    ("produser", "prodpass"),
]

# Pre-built defaults to avoid per-call dataclass allocation.
_DEFAULT_TRANSFORM_LAZY = CadeKdbTransformConfig(lazy=True)
_DEFAULT_TRANSFORM_EAGER = CadeKdbTransformConfig(lazy=False)


class CadeKdbManager:
    """Multi-host KDB pool registry with lifecycle management.

    Parameters
    ----------
    credentials : list of (user, password) tuples
        Default credential combos tried when connecting to a new host.
    pool_config : CadeKdbPoolConfig
        Default pool settings for newly created ``CadeKdb`` instances.
    transform : CadeKdbTransformConfig
        Default post-query transforms.
    connection_timeout_ms : int
        TCP connect timeout for new pools.
    query_timeout : float
        Default per-query timeout (seconds).
    maintenance_interval : float
        Seconds between maintenance sweeps (idle-pool reaping).
        Set to ``0`` to disable.

    Examples
    --------
    ::

        async with CadeKdbManager() as mgr:
            lf = await mgr.query("select from trade",
                                 host="kdb-host", port=7015)

            # Multi-host failover
            lf = await mgr.query("select from trade",
                                 hosts=[("host-a", 7015), ("host-b", 7015)])

            # Aggressive fan-out (first result wins)
            lf = await mgr.query("select from trade",
                                 hosts=[("host-a", 7015), ("host-b", 7015)],
                                 aggressive=True)
    """

    __slots__ = (
        "_pools",
        "_lock",
        "_key_locks",
        "_auth_cache",
        "_credentials",
        "_default_pool_config",
        "_default_transform",
        "_connection_timeout_ms",
        "_query_timeout",
        "_maintenance_interval",
        "_shutdown",
        "_maintenance_task",
    )

    def __init__(
        self,
        *,
        credentials: Sequence[tuple[str, str]] | None = None,
        pool_config: CadeKdbPoolConfig | None = None,
        transform: CadeKdbTransformConfig | None = None,
        connection_timeout_ms: int = 5_000,
        query_timeout: float = 30.0,
        maintenance_interval: float = 20.0,
    ) -> None:
        self._pools: dict[str, CadeKdb] = {}
        self._lock = asyncio.Lock()
        # Per-key creation locks serialize concurrent ``get_or_create`` calls
        # targeting the SAME ``host:port[:name]``.  Without this, N cold
        # queries against a single host each open a full TCP-auth-prewarm
        # cycle in parallel before the commit-lock deduplicates — a
        # reconnect storm that CadeKdb goes to lengths to prevent per-instance
        # but the manager previously re-introduced at the registry level.
        self._key_locks: dict[str, asyncio.Lock] = {}
        self._auth_cache = AuthCache()
        self._credentials: list[tuple[str, str]] = list(credentials or _DEFAULT_CREDS)
        self._default_pool_config = pool_config or CadeKdbPoolConfig()
        self._default_transform = transform or CadeKdbTransformConfig()
        self._connection_timeout_ms = connection_timeout_ms
        self._query_timeout = query_timeout
        self._maintenance_interval = maintenance_interval
        self._shutdown = asyncio.Event()
        self._maintenance_task: asyncio.Task[None] | None = None

    # -- Lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Start the maintenance loop."""
        if self._maintenance_interval > 0 and (
            self._maintenance_task is None or self._maintenance_task.done()
        ):
            self._shutdown.clear()
            self._maintenance_task = asyncio.create_task(
                self._maintenance_loop(), name="kdb-manager-maintenance"
            )
            self._maintenance_task.add_done_callback(self._on_maintenance_done)

    def _on_maintenance_done(self, task: asyncio.Task[None]) -> None:
        """Clear the task ref so ``start()`` can restart the loop."""
        if self._maintenance_task is task:
            self._maintenance_task = None

    async def shutdown(self, timeout: float = 5.0) -> None:
        """Stop maintenance, close all pools."""
        self._shutdown.set()
        if self._maintenance_task is not None:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
            self._maintenance_task = None

        close_tasks: list[asyncio.Task[None]] = []
        async with self._lock:
            for client in self._pools.values():
                close_tasks.append(asyncio.create_task(client.close()))
            self._pools.clear()
        if close_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*close_tasks, return_exceptions=True),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                # Cancel any close tasks that didn't finish in time.
                for t in close_tasks:
                    if not t.done():
                        t.cancel()
                await asyncio.gather(*close_tasks, return_exceptions=True)

    async def __aenter__(self) -> CadeKdbManager:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.shutdown()

    # -- Pool registry -----------------------------------------------------

    def _pool_key(
        self, host: str, port: int, name: str | None = None
    ) -> str:
        base = f"{host}:{port}"
        return f"{base}:{name}" if name else base

    async def get_or_create(
        self,
        host: str,
        port: int,
        *,
        username: str | None = None,
        password: str | None = None,
        name: str | None = None,
        pool_config: CadeKdbPoolConfig | None = None,
        connection_timeout_ms: int | None = None,
        connect_timeout: float | None = None,
    ) -> CadeKdb:
        """Return an existing pool or create a new one for ``host:port``.

        Creation is serialized per-key via ``self._key_locks[key]`` so
        that N concurrent cold queries for the SAME host produce exactly
        ONE connect+prewarm cycle, not N (preventing a reconnect storm
        against upstream KDB).  Concurrent callers targeting DIFFERENT
        hosts never touch each other's per-key lock and remain parallel.

        ``connect_timeout`` (optional) bounds the prewarm phase to the
        caller's remaining deadline budget, so a slow handshake cannot
        silently exceed a per-host timeout that ``_try_host`` already
        accounted for.
        """
        key = self._pool_key(host, port, name)

        # Fast path — no lock, no I/O.
        existing = self._pools.get(key)
        if existing is not None and existing.connected:
            return existing

        # Obtain (or lazily create) the per-key creation lock under the
        # global lock.  The per-key lock itself is only held for the
        # duration of one creation attempt.
        async with self._lock:
            key_lock = self._key_locks.get(key)
            if key_lock is None:
                key_lock = asyncio.Lock()
                self._key_locks[key] = key_lock

        async with key_lock:
            # Re-check under the per-key lock — another task may have
            # just completed creation.
            existing = self._pools.get(key)
            if existing is not None and existing.connected:
                return existing

            # Resolve credentials and construct client under the global
            # lock (Python-only, no I/O).
            async with self._lock:
                stale = self._pools.pop(key, None)

                u, p = username, password
                if u is None or p is None:
                    cached = self._auth_cache.get(host, port)
                    if cached:
                        u, p = cached
                    elif self._credentials:
                        u, p = self._credentials[0]

                ct = connection_timeout_ms or self._connection_timeout_ms
                client = CadeKdb(
                    host=host,
                    port=port,
                    username=u,
                    password=p,
                    connection_timeout_ms=ct,
                    query_timeout=self._query_timeout,
                    pool=pool_config or self._default_pool_config,
                    transform=self._default_transform,
                )

            # Best-effort close of stale entry (outside global lock, still
            # inside per-key lock so no other task can race us here).
            if stale is not None:
                try:
                    await asyncio.wait_for(stale.close(), timeout=2.0)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    _log.exception(
                        "stale pool close failed for %s:%s", host, port
                    )

            # Slow connect+prewarm I/O, bounded by the caller's remaining
            # deadline when supplied.
            try:
                await client.connect(connect_timeout=connect_timeout)
            except BaseException:
                with contextlib.suppress(Exception):
                    await client.close()
                raise

            # Commit under global lock.  With the per-key lock held we are
            # guaranteed there is no concurrent creator for this key, so
            # the old "race winner" branch can be simplified to a direct
            # install.
            async with self._lock:
                self._pools[key] = client

            return client

    async def remove_pool(
        self, host: str, port: int, name: str | None = None
    ) -> None:
        """Close and remove a specific pool."""
        key = self._pool_key(host, port, name)
        async with self._lock:
            client = self._pools.pop(key, None)
        if client is not None:
            try:
                await asyncio.wait_for(client.close(), timeout=5.0)
            except asyncio.CancelledError:
                raise
            except Exception:
                _log.exception("pool close failed for %s", key)

    # -- Maintenance -------------------------------------------------------

    async def _maintenance_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(), timeout=self._maintenance_interval
                )
                break
            except asyncio.TimeoutError:
                pass

            try:
                await self._trim_idle_pools()
            except asyncio.CancelledError:
                raise
            except Exception:
                # Surface maintenance-sweep failures instead of swallowing
                # them silently — operators need this signal to diagnose
                # broken pool state in production.
                _log.exception("kdb manager maintenance sweep failed")

    async def _trim_idle_pools(self) -> None:
        """Remove pools that have zero connections and no recent activity.

        Captures the *scanned* client reference in the candidate list and
        re-checks identity under the lock before eviction.  The previous
        implementation compared ``self._pools.get(key)`` to itself, which
        is trivially ``True`` and allowed the sweep to close brand-new
        pools that had replaced stale ones between scan and trim.

        Uses ``try_metrics()`` (not ``metrics()``) so the probe cannot
        silently auto-reconnect a pool that was nilled by a reset mid-scan.
        """
        now = time.monotonic()
        idle_cutoff = now - self._default_pool_config.idle_timeout_ms / 1000.0
        candidates: list[tuple[str, CadeKdb]] = []

        # Take the snapshot under the global lock so concurrent
        # ``get_or_create`` / ``remove_pool`` cannot mutate the dict
        # mid-iteration — under free-threaded CPython a plain
        # ``list(self._pools.items())`` would race and can raise
        # ``RuntimeError: dictionary changed size during iteration``
        # or produce a torn snapshot.
        async with self._lock:
            snapshot = list(self._pools.items())

        for key, client in snapshot:
            if not client.connected:
                candidates.append((key, client))
                continue
            try:
                m = await asyncio.wait_for(
                    client.try_metrics(), timeout=_METRICS_TIMEOUT_S,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                # Hung / errored pool — treat as eviction candidate so we
                # reclaim it instead of stalling the maintenance loop.
                candidates.append((key, client))
                continue
            if m is None:
                # Pool was nilled between the connected check and the probe.
                # Do NOT resurrect it via auto-connect — mark for eviction.
                candidates.append((key, client))
                continue
            if m.connections == 0 and client.last_activity_time < idle_cutoff:
                candidates.append((key, client))

        for key, scanned in candidates:
            victim: CadeKdb | None = None
            async with self._lock:
                # True identity comparison: only evict if the dict still
                # points to the same object we scanned.
                current = self._pools.get(key)
                if current is scanned:
                    victim = self._pools.pop(key, None)
            if victim is not None:
                try:
                    await asyncio.wait_for(victim.close(), timeout=2.0)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    _log.exception("kdb pool close failed for %s", key)

    # -- Single-host attempt -----------------------------------------------

    async def _try_host(
        self,
        expr: str,
        hc: HostConfig,
        creds: list[tuple[str, str]],
        *,
        timeout: float,
        output: Literal["polars", "arrow"],
        transform: CadeKdbTransformConfig | None,
        decode: q.DecodeOptions | None,
    ) -> Union[pl.LazyFrame, pl.DataFrame, q.Value]:
        """Try *expr* on a single host, rotating through *creds* on auth failure.

        All credential attempts share a single deadline so total wall-clock
        cannot exceed *timeout*.  Without this, N credentials × M hosts ×
        timeout can silently blow the caller's budget.

        On auth failure the pool is evicted so the next credential creates a
        fresh pool with the new user/password.
        """
        deadline = time.monotonic() + timeout
        last_exc: BaseException | None = None

        for user, password in creds:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise last_exc or TimeoutError(
                    f"Credential rotation deadline exceeded for "
                    f"{hc.host}:{hc.port}"
                )
            try:
                # Bound pool creation by the remaining query timeout so that
                # a slow connect() can't silently blow the deadline.
                conn_ms = min(
                    self._connection_timeout_ms,
                    max(int(remaining * 1000), 500),
                )
                client = await self.get_or_create(
                    hc.host,
                    hc.port,
                    username=user,
                    password=password,
                    name=hc.name,
                    connection_timeout_ms=conn_ms,
                    # Pass the remaining deadline budget so prewarm cannot
                    # silently exceed the caller's timeout.
                    connect_timeout=remaining,
                )
                # Recompute the deadline budget AFTER pool creation — a
                # slow connect / prewarm / stale-close can consume most
                # of the original ``remaining``.  Reusing the stale value
                # would hand ``client.query`` a budget longer than the
                # caller's actual wall-clock headroom, silently leaking
                # the failover deadline.
                query_budget = deadline - time.monotonic()
                if query_budget <= 0:
                    raise TimeoutError(
                        f"Deadline exceeded after pool creation for "
                        f"{hc.host}:{hc.port}"
                    )
                result = await client.query(
                    expr,
                    timeout=query_budget,
                    output=output,
                    transform=transform,
                    decode=decode,
                )
                self._auth_cache.remember(hc.host, hc.port, user, password)
                return result
            except asyncio.CancelledError:
                raise
            except q.QRuntimeError:
                # Query-level error (bad q expression) — not a host problem.
                raise
            except (q.DecodeError, q.ProtocolError, q.OperationError, q.PoolError):
                # Non-retriable qroissant errors — never rotate credentials.
                raise
            except Exception as e:
                last_exc = e
                if _is_auth_error(e):
                    self._auth_cache.forget(hc.host, hc.port)
                    # Evict pool so next iteration creates one with new creds.
                    await self.remove_pool(hc.host, hc.port, hc.name)
                    continue
                raise

        raise last_exc or ConnectionError(
            f"All credentials exhausted for {hc.host}:{hc.port}"
        )

    # -- Sequential failover (Feature 8) -----------------------------------

    async def _sequential_query(
        self,
        expr: str,
        host_list: list[HostConfig],
        effective_creds: Sequence[tuple[str, str]],
        *,
        timeout: float,
        output: Literal["polars", "arrow"],
        transform: CadeKdbTransformConfig | None,
        decode: q.DecodeOptions | None,
    ) -> Union[pl.LazyFrame, pl.DataFrame, q.Value]:
        """Try hosts one-by-one until one succeeds.  Deadline-based budget."""
        deadline = time.monotonic() + timeout
        last_exc: BaseException | None = None

        for hc in host_list:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Deadline exceeded before trying all hosts.")

            creds = _iter_credentials(hc, self._auth_cache, effective_creds)
            if not creds:
                continue

            try:
                return await self._try_host(
                    expr, hc, creds,
                    timeout=remaining,
                    output=output,
                    transform=transform,
                    decode=decode,
                )
            except asyncio.CancelledError:
                raise
            except q.QRuntimeError:
                raise
            except Exception as e:
                last_exc = e
                continue

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Exhausted all hosts with no result.")

    # -- Aggressive fan-out (Feature 9) ------------------------------------

    async def _aggressive_query(
        self,
        expr: str,
        host_list: list[HostConfig],
        effective_creds: Sequence[tuple[str, str]],
        *,
        timeout: float,
        output: Literal["polars", "arrow"],
        transform: CadeKdbTransformConfig | None,
        decode: q.DecodeOptions | None,
    ) -> Union[pl.LazyFrame, pl.DataFrame, q.Value]:
        """Race queries across all hosts.  First success wins, rest cancelled.

        The entire fan-out is bounded by *timeout* — including connection
        establishment, so a slow ``connect()`` cannot blow the deadline.
        """
        tasks: list[asyncio.Task[Any]] = []
        for hc in host_list:
            creds = _iter_credentials(hc, self._auth_cache, effective_creds)
            if not creds:
                continue
            task = asyncio.create_task(
                self._try_host(
                    expr, hc, creds,
                    timeout=timeout,
                    output=output,
                    transform=transform,
                    decode=decode,
                ),
                name=hc.key,
            )
            tasks.append(task)

        if not tasks:
            raise RuntimeError("No hosts to query.")

        # Closure-captured slot for a committed race win.  Populated
        # BEFORE ``_race()`` enters the shielded post-win drain.  If the
        # outer ``wait_for`` fires during that drain, the ``shield`` call
        # cannot actually prevent the awaiter itself from receiving
        # ``CancelledError`` — it only keeps the inner gather running.
        # Without this slot, the successful result is silently dropped
        # and the caller sees a spurious ``TimeoutError`` against an
        # already-answered query.  This is the whole point of exposing
        # the result to the outer ``except TimeoutError`` handler.
        result_holder: list[Any] = []

        async def _race() -> Union[pl.LazyFrame, pl.DataFrame, q.Value]:
            pending: set[asyncio.Task[Any]] = set(tasks)
            errors: dict[str, BaseException] = {}

            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )

                for t in done:
                    label = t.get_name()
                    if t.cancelled():
                        errors[label] = asyncio.CancelledError()
                        continue
                    try:
                        result = t.result()
                    except q.QRuntimeError:
                        for p in pending:
                            p.cancel()
                        if pending:
                            # Shield cleanup so an outer wait_for cannot
                            # leave tasks orphaned.
                            await asyncio.shield(
                                asyncio.gather(
                                    *pending, return_exceptions=True
                                )
                            )
                        raise
                    except Exception as e:
                        errors[label] = e
                    else:
                        # Commit the result to the outer holder BEFORE
                        # the shielded drain.  The shield keeps the
                        # inner gather alive under cancellation but
                        # does NOT prevent the await here from raising
                        # CancelledError, so ``return result`` below
                        # may never execute under a racing outer
                        # timeout.  The holder is the single source of
                        # truth recoverable from the except handler.
                        result_holder.append(result)
                        for p in pending:
                            p.cancel()
                        if pending:
                            await asyncio.shield(
                                asyncio.gather(
                                    *pending, return_exceptions=True
                                )
                            )
                        return result

            if errors:
                raise next(iter(errors.values()))
            raise RuntimeError("Aggressive fan-out: all hosts failed.")

        try:
            # Wrap the entire race with a hard timeout so that slow
            # connect()/prewarm() phases cannot exceed the deadline.
            return await asyncio.wait_for(_race(), timeout=timeout)
        except asyncio.TimeoutError:
            # If the race already committed a result, hand it back
            # rather than raising — the outer wait_for cancelled
            # ``_race`` mid-drain, but the winning host DID answer.
            if result_holder:
                return result_holder[0]
            raise
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
            # Shield the cleanup gather so an external CancelledError
            # cannot interrupt it and orphan running tasks.  Bound the
            # wait so a task stuck in a non-cooperative Rust call cannot
            # hang the caller forever — if it misses the deadline we let
            # the tasks run to completion in the background.
            pending = [t for t in tasks if not t.done()]
            if pending:
                drain = asyncio.gather(*pending, return_exceptions=True)
                try:
                    await asyncio.shield(
                        asyncio.wait_for(drain, timeout=2.0)
                    )
                except asyncio.TimeoutError:
                    _log.warning(
                        "aggressive fan-out: %d task(s) did not cancel "
                        "within 2s; leaving in background",
                        len(pending),
                    )
                except Exception:
                    pass

    # -- Backdoor routing (Feature 11) -------------------------------------

    @staticmethod
    def _build_backdoor_query(
        expr: str,
        target_host: str,
        target_port: int,
        user: str,
        password: str,
    ) -> str:
        """Wrap *expr* to execute via IPC proxy through a backdoor host.

        Rejects credentials/hosts containing characters that would corrupt
        the q IPC address grammar (``:``) or the outer string literal
        (``"``, ``\\``, CR/LF/TAB).  Without these checks a password
        containing ``"`` becomes a remote code execution primitive on the
        intermediary.  The expression itself has BOTH backslash and quote
        escaped — the previous implementation only escaped quotes, allowing
        an attacker-controlled ``\\"`` to pivot-escape.
        """
        _forbidden = ':"\\\n\r\t'
        for field, val in (
            ("user", user),
            ("password", password),
            ("host", target_host),
        ):
            if val is None or any(c in val for c in _forbidden):
                raise ValueError(
                    f"Unsafe character in backdoor {field}; refusing to build"
                )
        escaped = expr.replace("\\", "\\\\").replace('"', '\\"')
        return (
            f'(`$":{target_host}:{target_port}:{user}:{password}")'
            f'"{escaped}"'
        )

    async def _backdoor_query(
        self,
        expr: str,
        target_host: str,
        target_port: int,
        *,
        backdoor_hosts: list[HostConfig],
        effective_creds: Sequence[tuple[str, str]],
        timeout: float,
        output: Literal["polars", "arrow"],
        transform: CadeKdbTransformConfig | None,
        decode: q.DecodeOptions | None,
    ) -> Union[pl.LazyFrame, pl.DataFrame, q.Value]:
        """Route *expr* through backdoor intermediary hosts."""
        target_creds = _iter_credentials(
            HostConfig(host=target_host, port=target_port),
            self._auth_cache,
            effective_creds,
        )

        last_exc: BaseException | None = None
        for user, password in target_creds:
            wrapped = self._build_backdoor_query(
                expr, target_host, target_port, user, password
            )
            try:
                result = await self._sequential_query(
                    wrapped,
                    backdoor_hosts,
                    effective_creds,
                    timeout=timeout,
                    output=output,
                    transform=transform,
                    decode=decode,
                )
                self._auth_cache.remember(
                    target_host, target_port, user, password
                )
                return result
            except asyncio.CancelledError:
                raise
            except q.QRuntimeError:
                raise
            except Exception as e:
                last_exc = e
                if _is_auth_error(e):
                    continue
                raise

        raise last_exc or ConnectionError(
            f"Backdoor failed for {target_host}:{target_port}"
        )

    # -- Public query API --------------------------------------------------

    @overload
    async def query(
        self,
        expr: str,
        *,
        host: str | None = ...,
        port: int | None = ...,
        hosts: Sequence[tuple[str, int]] | None = ...,
        config: ConnectionList | Sequence[dict[str, Any]] | dict[str, Any] | None = ...,
        name: str | None = ...,
        username: str | None = ...,
        password: str | None = ...,
        timeout: float | None = ...,
        aggressive: bool = ...,
        backdoor: bool = ...,
        backdoor_hosts: Sequence[tuple[str, int]] | None = ...,
        credentials: Sequence[tuple[str, str]] | None = ...,
        none_is_failure: bool = ...,
        empty_is_failure: bool = ...,
        output: Literal["polars"] = ...,
        transform: CadeKdbTransformConfig | None = ...,
        decode: q.DecodeOptions | None = ...,
    ) -> pl.LazyFrame | pl.DataFrame: ...

    @overload
    async def query(
        self,
        expr: str,
        *,
        host: str | None = ...,
        port: int | None = ...,
        hosts: Sequence[tuple[str, int]] | None = ...,
        config: ConnectionList | Sequence[dict[str, Any]] | dict[str, Any] | None = ...,
        name: str | None = ...,
        username: str | None = ...,
        password: str | None = ...,
        timeout: float | None = ...,
        aggressive: bool = ...,
        backdoor: bool = ...,
        backdoor_hosts: Sequence[tuple[str, int]] | None = ...,
        credentials: Sequence[tuple[str, str]] | None = ...,
        none_is_failure: bool = ...,
        empty_is_failure: bool = ...,
        output: Literal["arrow"] = ...,
        transform: CadeKdbTransformConfig | None = ...,
        decode: q.DecodeOptions | None = ...,
    ) -> q.Value: ...

    async def query(
        self,
        expr: str,
        *,
        host: str | None = None,
        port: int | None = None,
        hosts: Sequence[tuple[str, int]] | None = None,
        config: ConnectionList | Sequence[dict[str, Any]] | dict[str, Any] | None = None,
        name: str | None = None,
        username: str | None = None,
        password: str | None = None,
        timeout: float | None = None,
        aggressive: bool = False,
        backdoor: bool = False,
        backdoor_hosts: Sequence[tuple[str, int]] | None = None,
        credentials: Sequence[tuple[str, str]] | None = None,
        none_is_failure: bool = True,
        empty_is_failure: bool = False,
        output: Literal["polars", "arrow"] = "polars",
        transform: CadeKdbTransformConfig | None = None,
        decode: q.DecodeOptions | None = None,
    ) -> Union[pl.LazyFrame, pl.DataFrame, q.Value]:
        """Execute a q expression with multi-host routing.

        Parameters
        ----------
        expr : str
            q expression to evaluate.
        host, port : str, int
            Single target host.  Mutually exclusive with *hosts* / *config*.
        hosts : list of (host, port) tuples
            Multiple targets for failover or fan-out.
        config : ConnectionList | list[dict] | dict
            Host configs with keys ``host``, ``port``, ``usr``/``user``,
            ``pwd``/``password``, optionally ``backdoor``, ``name``.
            Accepts raw dicts, a ``ConnectionList`` from ``fconn()``, or
            a single dict (auto-wrapped).
        name : str, optional
            Logical pool name (allows multiple pools to the same host:port).
        username, password : str, optional
            Override credentials for this query.
        timeout : float, optional
            Total query budget in seconds.
        aggressive : bool
            If ``True``, query all hosts in parallel (first wins).
        backdoor : bool
            If ``True``, proxy the query through *backdoor_hosts*.
        backdoor_hosts : list of (host, port) tuples
            Intermediary hosts for backdoor routing.
        credentials : list of (user, password) tuples
            Override default credential combos for this query.
        none_is_failure : bool
            If ``True`` (default) and the result is ``None``, return an
            empty LazyFrame/DataFrame instead.
        empty_is_failure : bool
            If ``True`` and the result is an empty table, raise so that
            multi-host failover tries the next host.
        output : ``"polars"`` | ``"arrow"``
            Return format.
        transform : CadeKdbTransformConfig, optional
            Override default transforms.
        decode : DecodeOptions, optional
            Override default qroissant decode options.
        """
        t = timeout if timeout is not None else self._query_timeout

        # Per-query credential override is passed through the call chain
        # instead of mutating shared state (concurrency-safe).
        effective_creds: Sequence[tuple[str, str]] = (
            list(credentials) if credentials is not None else self._credentials
        )

        result = await self._route(
            expr,
            host=host,
            port=port,
            hosts=hosts,
            config=config,
            name=name,
            username=username,
            password=password,
            timeout=t,
            aggressive=aggressive,
            backdoor=backdoor,
            backdoor_hosts=backdoor_hosts,
            effective_creds=effective_creds,
            output=output,
            transform=transform,
            decode=decode,
        )

        # -- post-result checks (backwards compat with old query_kdb) ------
        if none_is_failure and result is None:
            if output == "polars":
                cfg = transform or self._default_transform
                return pl.LazyFrame() if cfg.lazy else pl.DataFrame()
            raise RuntimeError("Query returned None (none_is_failure=True).")

        if empty_is_failure and isinstance(result, (pl.DataFrame, pl.LazyFrame)):
            if isinstance(result, pl.LazyFrame):
                # Probe with .limit(1).collect() instead of materializing
                # the whole lazy pipeline — a full .collect() just to check
                # emptiness defeats the zero-copy lazy optimization and on
                # multi-GB result sets causes OOM under concurrent queries.
                probe = result.limit(1).collect()
                is_empty = probe.is_empty()
            else:
                is_empty = result.is_empty()
            if is_empty:
                raise RuntimeError(
                    "Query returned empty result (empty_is_failure=True)."
                )

        return result

    async def _route(
        self,
        expr: str,
        *,
        host: str | None,
        port: int | None,
        hosts: Sequence[tuple[str, int]] | None,
        config: ConnectionList | Sequence[dict[str, Any]] | dict[str, Any] | None,
        name: str | None,
        username: str | None,
        password: str | None,
        timeout: float,
        aggressive: bool,
        backdoor: bool,
        backdoor_hosts: Sequence[tuple[str, int]] | None,
        effective_creds: Sequence[tuple[str, str]],
        output: Literal["polars", "arrow"],
        transform: CadeKdbTransformConfig | None,
        decode: q.DecodeOptions | None,
    ) -> Union[pl.LazyFrame, pl.DataFrame, q.Value, None]:
        """Internal routing dispatch — separated from result checks."""
        host_list = _resolve_hosts(
            host=host,
            port=port,
            hosts=hosts,
            config=config,
            name=name,
            username=username,
            password=password,
        )

        # Explicit backdoor routing.
        if backdoor:
            if not backdoor_hosts:
                raise ValueError("backdoor=True requires backdoor_hosts.")
            bd_list = _resolve_hosts(hosts=backdoor_hosts)
            target = host_list[0]
            return await self._backdoor_query(
                expr,
                target.host,
                target.port,
                backdoor_hosts=bd_list,
                effective_creds=effective_creds,
                timeout=timeout,
                output=output,
                transform=transform,
                decode=decode,
            )

        # Separate backdoor-flagged hosts (from config) from direct hosts.
        direct: list[HostConfig] = []
        bd_flagged: list[HostConfig] = []
        for hc in host_list:
            (bd_flagged if hc.backdoor else direct).append(hc)

        # Try direct hosts first.
        if direct:
            strategy = (
                self._aggressive_query if aggressive else self._sequential_query
            )
            try:
                return await strategy(
                    expr, direct, effective_creds,
                    timeout=timeout,
                    output=output,
                    transform=transform,
                    decode=decode,
                )
            except q.QRuntimeError:
                raise
            except Exception:
                if not bd_flagged:
                    raise
                # Fall through to backdoor hosts.

        # Backdoor-flagged hosts from config.
        if bd_flagged:
            if not direct:
                raise ValueError(
                    "Config has only backdoor hosts but no direct target. "
                    "Use backdoor=True with explicit backdoor_hosts instead."
                )
            target = direct[0]
            return await self._backdoor_query(
                expr,
                target.host,
                target.port,
                backdoor_hosts=bd_flagged,
                effective_creds=effective_creds,
                timeout=timeout,
                output=output,
                transform=transform,
                decode=decode,
            )

        raise RuntimeError("No hosts available.")

    # -- Diagnostics -------------------------------------------------------

    async def pool_stats(self) -> dict[str, dict[str, Any]]:
        """Return metrics for all managed pools.

        Each per-pool metrics call is bounded so that a single hung pool
        cannot freeze the entire diagnostics call.  Uses ``try_metrics()``
        so diagnostic probes cannot accidentally auto-reconnect an evicted
        or reset pool.
        """
        stats: dict[str, dict[str, Any]] = {}
        # Snapshot under the lock — see ``_trim_idle_pools`` for the
        # rationale (free-threaded CPython dict-mutation race).
        async with self._lock:
            snapshot = list(self._pools.items())
        for key, client in snapshot:
            try:
                m = await asyncio.wait_for(
                    client.try_metrics(), timeout=_METRICS_TIMEOUT_S,
                )
                if m is None:
                    stats[key] = {"status": "disconnected"}
                    continue
                stats[key] = {
                    "connections": m.connections,
                    "idle": m.idle_connections,
                    "max_size": m.max_size,
                    "last_success": client.last_success_time,
                }
            except asyncio.CancelledError:
                raise
            except Exception:
                stats[key] = {"status": "error"}
        return stats

    @property
    def auth_cache(self) -> AuthCache:
        """Access the credential cache (e.g. for pre-seeding)."""
        return self._auth_cache


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def _norm_arg_names(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy KDB argument name aliases.

    Accepts ``usr``/``user``/``username``/``login`` → ``username``,
    ``pwd``/``pass``/``passwd``/``password`` → ``password``,
    ``hostname``/``server`` → ``host``, ``svc``/``service`` → ``port``.
    """
    m = dict(kwargs)

    def _first(*keys: str) -> Any:
        for k in keys:
            v = m.pop(k, None)
            if v is not None:
                return v
        return None

    username = _first("usr", "user", "username", "login")
    password = _first("pwd", "pass", "passwd", "password")
    host = _first("host", "hostname", "server")
    port = _first("port", "svc", "service")

    m["username"] = username
    m["password"] = password
    m["host"] = host
    m["port"] = port
    return m


async def query_kdb(
    expr: str | None = None,
    *,
    manager: CadeKdbManager,
    config: ConnectionList | Sequence[dict[str, Any]] | dict[str, Any] | None = None,
    host: str | None = None,
    port: int | None = None,
    name: str | None = None,
    usr: str | None = None,
    pwd: str | None = None,
    username: str | None = None,
    password: str | None = None,
    user: str | None = None,
    hosts: Sequence[tuple[str, int]] | None = None,
    timeout: float | None = None,
    aggressive: bool = False,
    backdoor: bool = False,
    backdoor_hosts: Sequence[tuple[str, int]] | None = None,
    credentials: Sequence[tuple[str, str]] | None = None,
    credential_combos: Sequence[tuple[str, str]] | None = None,
    none_is_failure: bool = True,
    empty_is_failure: bool = False,
    output: Literal["polars", "arrow"] = "polars",
    transform: CadeKdbTransformConfig | None = None,
    decode: q.DecodeOptions | None = None,
    lazy: bool = True,
    _kdb: CadeKdbManager | None = None,
    **kwargs: Any,
) -> Union[pl.LazyFrame, pl.DataFrame, q.Value]:
    """Execute a q expression via a ``CadeKdbManager``.

    Backwards-compatible wrapper around :meth:`CadeKdbManager.query` that
    accepts the same argument names and conventions as the legacy
    ``query_kdb()`` function.

    The primary parameter name is ``expr``; the legacy ``q=`` kwarg alias
    is still accepted for backwards compatibility.  (Renamed so the function
    body no longer shadows the ``qroissant`` module import.)

    Parameters
    ----------
    expr : str
        q expression to evaluate (positional, like the old API).  Legacy
        callers may pass ``q=`` as a keyword alias.
    manager : CadeKdbManager
        **Required.**  The pool manager to route through.
        Legacy alias ``_kdb`` is also accepted.
    config : ConnectionList | list[dict] | dict, optional
        Host configs.  Accepts raw dicts, ``fconn()`` result, or a single dict.
    host, port : str, int
        Single host target.
    usr / pwd / user / username / password
        Credential aliases — all normalised internally.
    credential_combos
        Legacy alias for *credentials*.
    none_is_failure : bool
        If ``True`` (default) and result is ``None``, return empty frame.
    empty_is_failure : bool
        If ``True`` and result is empty, raise so failover continues.
    lazy : bool
        If ``True`` (default), return a LazyFrame.
    """
    # Normalise legacy aliases.  Accept ``q=`` as a positional-equivalent
    # kwarg so existing callers keep working.
    mgr = _kdb or manager
    if expr is None and "q" in kwargs:
        expr = kwargs.pop("q")
    if expr is None:
        raise ValueError("Query expression is required (pass as positional or expr=).")

    normalized = _norm_arg_names({
        "usr": usr, "user": user, "username": username,
        "pwd": pwd, "password": password,
        "host": host, "port": port,
        **kwargs,
    })
    effective_user = normalized["username"]
    effective_pass = normalized["password"]
    effective_host = normalized.get("host", host)
    effective_port = normalized.get("port", port)

    creds = credentials or credential_combos

    if transform is None:
        transform = _DEFAULT_TRANSFORM_LAZY if lazy else _DEFAULT_TRANSFORM_EAGER

    return await mgr.query(
        expr,
        host=effective_host,
        port=effective_port,
        hosts=hosts,
        config=config,
        name=name,
        username=effective_user,
        password=effective_pass,
        timeout=timeout,
        aggressive=aggressive,
        backdoor=backdoor,
        backdoor_hosts=backdoor_hosts,
        credentials=creds,
        none_is_failure=none_is_failure,
        empty_is_failure=empty_is_failure,
        output=output,
        transform=transform,
        decode=decode,
    )
