# CadeKdb

High-performance async Python wrapper for [qroissant](https://pypi.org/project/qroissant/) — a Rust-based KDB+/q IPC client with first-class Apache Arrow support.

**Pipeline**: KDB IPC &rarr; Rust parallel decode (SIMD) &rarr; Arrow PyCapsules &rarr; Polars (zero-copy)

---

## Architecture

```
query_kdb("select from trade", manager=mgr, config=PANOPROXY)
    |
    v
CadeKdbManager.query()              # multi-host orchestration
    |
    +-- _sequential_query()          # try hosts one-by-one (deadline budget)
    |       +-- _try_host()          # credential rotation per host
    |               |
    +-- _aggressive_query()          # fan-out: race all hosts, first wins
            +-- _try_host()
                    |
                    v
            CadeKdb.query()          # single-pool, circuit-breaker protected
                    |
                    v
            q.AsyncPool.query()      # qroissant Rust backend
                    |
                    v
            _to_polars(value, cfg)   # zero-copy Arrow -> Polars + transforms
```

### Files

| File | Purpose |
|------|---------|
| `kdb_client.py` | `CadeKdb` — single-host async client with connection pooling, circuit-breaker, zero-copy transforms |
| `kdb_manager.py` | `CadeKdbManager` — multi-host pool registry, failover, fan-out, credential rotation, backdoor routing, maintenance loop |
| `kdb_helpers.py` | Query builders (`kdb_where`, `kdb_by`, `kdb_fby`), region routing, insert helpers, convenience wrappers |

---

## Installation

```bash
pip install qroissant polars
```

Requires Python 3.10+.

---

## Quick Start

### Single Host

```python
import asyncio
from kdb_client import CadeKdb

async def main():
    async with CadeKdb(":kdb-host:7015:user:pass") as kdb:
        lf = await kdb.query("select from trade where date = .z.d")
        df = lf.collect()
        print(df)

asyncio.run(main())
```

### Multi-Host with Manager

```python
import asyncio
from kdb_manager import CadeKdbManager, fconn, query_kdb

PANOPROXY = [
    {"host": "kdb-panoproxy-credit-nyk", "port": 7015, "region": "US", "dbtype": "prod"},
    {"host": "kdb-panoproxy-credit-ldn", "port": 7015, "region": "EU", "dbtype": "prod"},
    {"host": "kdb-panoproxy-credit-dr",  "port": 7015, "region": "US", "dbtype": "dr"},
]

async def main():
    async with CadeKdbManager() as mgr:
        # Sequential failover across all hosts
        lf = await query_kdb(
            "select from .mt.get[`.credit.refData]",
            manager=mgr,
            config=fconn(PANOPROXY, region="US"),
        )
        print(lf.collect())

asyncio.run(main())
```

---

## Connection Strings

`CadeKdb` accepts connections in multiple formats:

```python
# KDB classic
kdb = CadeKdb(":host:7015:user:pass")

# Without leading colon
kdb = CadeKdb("host:7015:user:pass")

# Host:port with separate credentials
kdb = CadeKdb("host:7015", username="user", password="pass")

# All keyword arguments
kdb = CadeKdb(host="host", port=7015, username="user", password="pass")
```

---

## CadeKdb — Single-Host Client

The low-level client. Wraps a single qroissant `AsyncPool` with circuit-breaker protection and automatic Polars conversion.

```python
async with CadeKdb(
    host="kdb-prod", port=5010,
    username="svc", password="pw",
    connection_timeout_ms=5_000,     # TCP connect timeout
    query_timeout=30.0,              # default per-query timeout (seconds)
    max_timeouts_before_reset=3,     # circuit-breaker threshold
    pool=CadeKdbPoolConfig(
        max_size=10,
        min_idle=2,
        idle_timeout_ms=60_000,      # evict idle connections after 60s
        prewarm=True,                # eagerly open min_idle connections
    ),
    transform=CadeKdbTransformConfig(
        camel_case_headers=True,     # snake_case/PascalCase -> camelCase
        nullify_na_strings=True,     # "NA" and "" -> null in string columns
        convert_boolean_strings=True,# isActive: "Y"/"N" -> 1/0 Int8
        bools_to_int8=True,          # Boolean columns -> Int8
        lazy=True,                   # return LazyFrame (default)
    ),
) as kdb:
    # Returns a LazyFrame by default
    lf = await kdb.query("select from trade")

    # Per-query timeout override
    lf = await kdb.query("select from bigTable", timeout=120.0)

    # Raw Arrow output (skip Polars transforms)
    value = await kdb.query("select from trade", output="arrow")

    # Concurrent queries (pool handles checkout/checkin)
    lf_trade, lf_quote = await asyncio.gather(
        kdb.query("select from trade"),
        kdb.query("select from quote"),
    )
```

### Circuit Breaker

After `max_timeouts_before_reset` consecutive timeout or transport errors, the pool is automatically destroyed and recreated. This prevents poisoned connections from silently hanging forever.

```python
kdb = CadeKdb(":host:7015", max_timeouts_before_reset=3)
# After 3 consecutive timeouts, the pool is nuked and rebuilt on the next query.
# Set to 0 to disable.
```

### Pool Metrics

```python
metrics = await kdb.metrics()
print(metrics.connections, metrics.idle_connections, metrics.max_size)
```

---

## CadeKdbManager — Multi-Host Orchestration

The production-grade entry point. Manages a registry of `CadeKdb` pools, routes queries across hosts, handles credential rotation, and runs a background maintenance loop.

```python
async with CadeKdbManager(
    credentials=[("credituser", "creditpass"), ("produser", "prodpass")],
    pool_config=CadeKdbPoolConfig(max_size=10, min_idle=2),
    connection_timeout_ms=5_000,
    query_timeout=30.0,
    maintenance_interval=20.0,  # seconds between idle-pool reaping
) as mgr:
    ...
```

### Pool Registry

Pools are created lazily on first query and cached by `host:port[:name]`. Overlapping hosts share the same pool automatically:

```python
async with CadeKdbManager() as mgr:
    # These two queries share the same underlying CadeKdb pool:
    lf1 = await mgr.query("select from trade", host="kdb-nyk", port=7015)
    lf2 = await mgr.query("select from quote", host="kdb-nyk", port=7015)

    # This uses a different pool (different host):
    lf3 = await mgr.query("select from trade", host="kdb-ldn", port=7015)

    # Named pools allow multiple pools to the same host:port:
    lf4 = await mgr.query("q1", host="kdb-gw", port=5010, name="trades")
    lf5 = await mgr.query("q2", host="kdb-gw", port=5010, name="quotes")
```

### Sequential Failover

Tries hosts one-by-one with a shared deadline budget. If host A times out, the remaining time is given to host B:

```python
lf = await mgr.query(
    "select from trade",
    hosts=[("kdb-nyk", 7015), ("kdb-ldn", 7015), ("kdb-sgp", 7015)],
    timeout=10.0,   # total budget across all hosts
)
```

### Aggressive Fan-Out

Race the query across all hosts simultaneously. First successful result wins; the rest are cancelled:

```python
lf = await mgr.query(
    "select from trade",
    hosts=[("kdb-nyk", 7015), ("kdb-ldn", 7015)],
    aggressive=True,
    timeout=5.0,
)
```

### Config-Driven Routing with `fconn`

Define host configurations as dicts, filter with `fconn`:

```python
from kdb_manager import fconn

PANOPROXY = [
    {"host": "kdb-pano-nyk",  "port": 7015, "region": "US", "dbtype": "prod"},
    {"host": "kdb-pano-ldn",  "port": 7015, "region": "EU", "dbtype": "prod"},
    {"host": "kdb-pano-dr",   "port": 7015, "region": "US", "dbtype": "dr"},
    {"host": "kdb-pano-nyk2", "port": 7015, "region": "US", "dbtype": "prod"},
]

# Filter to US prod hosts only
lf = await mgr.query(
    "select from trade",
    config=fconn(PANOPROXY, region="US", dbtype="prod"),
)

# Filter to prod OR dr
lf = await mgr.query(
    "select from trade",
    config=fconn(PANOPROXY, dbtype=["prod", "dr"]),
    aggressive=True,
)

# Single dict (auto-wrapped)
lf = await mgr.query("select from trade", config=PANOPROXY[0])

# Raw list (no fconn needed)
lf = await mgr.query("select from trade", config=PANOPROXY)
```

### Credential Rotation

On authentication failure, the manager tries the next credential combo automatically. Successful credentials are cached per `(host, port)`:

```python
mgr = CadeKdbManager(
    credentials=[
        ("credituser", "creditpass"),
        ("produser", "prodpass"),
        ("readonly", "readonly"),
    ]
)

# Per-query credential override:
lf = await mgr.query(
    "select from trade",
    host="kdb-secure", port=5010,
    credentials=[("admin", "s3cret")],
)

# Pre-seed the auth cache:
mgr.auth_cache.remember("kdb-prod", 7015, "svc", "pw")
```

### Backdoor Routing

Proxy a query through intermediary KDB hosts when the target is not directly reachable:

```python
BACKDOOR_HOSTS = [
    {"host": "kdb-jump-1", "port": 5000},
    {"host": "kdb-jump-2", "port": 5000},
]

lf = await mgr.query(
    "select from trade",
    host="kdb-internal", port=7015,
    backdoor=True,
    backdoor_hosts=[("kdb-jump-1", 5000), ("kdb-jump-2", 5000)],
)
```

Backdoor hosts can also be flagged in config dicts:

```python
MIXED_CONFIG = [
    {"host": "kdb-direct", "port": 7015},                      # tried first
    {"host": "kdb-jump",   "port": 5000, "backdoor": True},    # fallback proxy
]
lf = await mgr.query("select from trade", config=MIXED_CONFIG)
```

### Result Validation

```python
# none_is_failure=True (default): if query returns None, return empty DataFrame
lf = await mgr.query("select from trade", host="h", port=7015)

# empty_is_failure=True: if result is an empty table, raise RuntimeError
lf = await mgr.query(
    "select from trade",
    config=HOSTS,
    empty_is_failure=True,
)
```

### Pool Statistics

```python
stats = await mgr.pool_stats()
# {'kdb-nyk:7015': {'connections': 3, 'idle': 2, 'max_size': 10, 'last_success': 1234567.89},
#  'kdb-ldn:7015': {'connections': 1, 'idle': 1, 'max_size': 10, 'last_success': 1234560.00}}
```

---

## query_kdb — Legacy-Compatible Function

Drop-in replacement for the old `query_kdb()` with full argument alias support:

```python
from kdb_manager import CadeKdbManager, query_kdb, fconn

mgr = CadeKdbManager()
await mgr.start()

# All of these are equivalent:
lf = await query_kdb("select from trade", manager=mgr, host="h", port=7015, usr="u", pwd="p")
lf = await query_kdb("select from trade", manager=mgr, host="h", port=7015, user="u", password="p")
lf = await query_kdb("select from trade", manager=mgr, host="h", port=7015, username="u", password="p")

# Legacy _kdb alias:
lf = await query_kdb("select from trade", _kdb=mgr, host="h", port=7015)

# With config:
lf = await query_kdb("select from trade", manager=mgr, config=fconn(PANOPROXY))

# credential_combos alias:
lf = await query_kdb("q", manager=mgr, host="h", port=7015,
                     credential_combos=[("u1", "p1"), ("u2", "p2")])

await mgr.shutdown()
```

---

## Real-World Example: Concurrent Tasks with Overlapping Hosts

A common production pattern: many async tasks querying different tables across a mix of KDB hosts. The manager deduplicates pools automatically.

```python
import asyncio
from kdb_manager import CadeKdbManager, CadeKdbPoolConfig, fconn, query_kdb
from kdb_helpers import construct_panoproxy_triplet, kdb_where

# -- Host configurations --

PANOPROXY_US = [
    {"host": "kdb-panoproxy-credit-nyk", "port": 7015, "region": "US", "dbtype": "prod"},
    {"host": "kdb-panoproxy-credit-dr",  "port": 7015, "region": "US", "dbtype": "dr"},
]

PANOPROXY_EU = [
    {"host": "kdb-panoproxy-credit-ldn", "port": 7015, "region": "EU", "dbtype": "prod"},
]

GATEWAY = [
    {"host": "kdb-gateway-credit-nyk", "port": 5010, "region": "US", "dbtype": "prod"},
    {"host": "kdb-gateway-credit-ldn", "port": 5010, "region": "EU", "dbtype": "prod"},
]


# -- Task definitions --

async def fetch_ref_data(mgr: CadeKdbManager):
    """Fetch reference data from US panoproxy."""
    table = construct_panoproxy_triplet("US", "refData")
    where = kdb_where({"Ticker": ["AAPL", "MSFT", "GOOG"]})
    return await query_kdb(
        f"select from {table} where {where}",
        manager=mgr,
        config=fconn(PANOPROXY_US, dbtype="prod"),
        timeout=10.0,
    )


async def fetch_trades(mgr: CadeKdbManager, region: str):
    """Fetch today's trades for a region — aggressive across prod/dr."""
    table = construct_panoproxy_triplet(region, "trades")
    config = PANOPROXY_US if region == "US" else PANOPROXY_EU
    return await query_kdb(
        f"select from {table}",
        manager=mgr,
        config=fconn(config, dbtype=["prod", "dr"]),
        aggressive=True,   # race prod vs dr, first wins
        timeout=15.0,
    )


async def fetch_positions(mgr: CadeKdbManager):
    """Fetch positions via gateway — sequential failover."""
    return await query_kdb(
        "select from .credit.nyk.positions",
        manager=mgr,
        config=fconn(GATEWAY),
        timeout=20.0,
    )


async def fetch_risk_metrics(mgr: CadeKdbManager):
    """Fetch risk from the same US panoproxy (shares pool with ref data)."""
    table = construct_panoproxy_triplet("US", "riskMetrics")
    return await query_kdb(
        f"select from {table}",
        manager=mgr,
        config=fconn(PANOPROXY_US, dbtype="prod"),
        timeout=10.0,
    )


# -- Main --

async def main():
    async with CadeKdbManager(
        credentials=[("credituser", "creditpass"), ("produser", "prodpass")],
        pool_config=CadeKdbPoolConfig(max_size=10, min_idle=2, prewarm=True),
        query_timeout=30.0,
    ) as mgr:

        # Launch all tasks concurrently.
        # fetch_ref_data and fetch_risk_metrics both hit kdb-panoproxy-credit-nyk:7015
        # — they automatically share the same pool.
        ref_data, us_trades, eu_trades, positions, risk = await asyncio.gather(
            fetch_ref_data(mgr),
            fetch_trades(mgr, "US"),
            fetch_trades(mgr, "EU"),
            fetch_positions(mgr),
            fetch_risk_metrics(mgr),
        )

        # All results are LazyFrames by default
        print("Ref data:", ref_data.collect().shape)
        print("US trades:", us_trades.collect().shape)
        print("EU trades:", eu_trades.collect().shape)
        print("Positions:", positions.collect().shape)
        print("Risk:", risk.collect().shape)

        # Check pool sharing
        stats = await mgr.pool_stats()
        for key, info in stats.items():
            print(f"  {key}: {info['connections']} conns, {info['idle']} idle")

asyncio.run(main())
```

In this example:
- **5 concurrent tasks** share **4 unique pools** (ref data and risk metrics share the US panoproxy pool)
- US trades uses **aggressive fan-out** across prod and DR
- Positions uses **sequential failover** across gateway hosts
- Credential rotation is automatic across all hosts
- Pools are prewarmed on startup, reaped by the maintenance loop when idle

---

## Query Building Helpers

All helpers are pure functions (no I/O) — import from `kdb_helpers`:

```python
from kdb_helpers import kdb_where, kdb_by, kdb_fby, kdb_col_select_helper

# WHERE clause from dicts
where = kdb_where({"sym": "AAPL"}, {"date": "2024.01.15"})
# => 'sym=`AAPL,date=`2024.01.15'

# List membership
where = kdb_where({"sym": ["AAPL", "MSFT", "GOOG"]})
# => 'sym in `AAPL`MSFT`GOOG'

# Typed values
where = kdb_where({"price": {"value": 150.0, "dtype": "float"}})
# => 'price=150.0'

# String contains (LIKE)
where = kdb_where({"name": {"value": "Apple", "dtype": "string"}})
# => 'name like "*Apple*"'

# BY clause
by = kdb_by(["sym", "date"])
# => ' by sym,date'

# FBY expression
fby = kdb_fby(["sym", "date"], d="last")
# => 'j = (last; j) fby ([];sym;date)'

# Column select with aggregation
cols = kdb_col_select_helper(["sym", "alias:price", "volume"], method="last")
# => 'last sym,alias:last price,last volume'
```

### Insert Helpers

```python
from kdb_helpers import q_symbols, q_strings, q_floats, q_ints, q_now
from kdb_helpers import build_insert_query_generic, build_insert_query_panoproxy

cols = [q_now(3), q_symbols(["AAPL", "MSFT", "GOOG"]), q_floats([150.0, 280.5, 140.2])]

# Generic insert
q = build_insert_query_generic("trade", cols)
# => '`trade insert (.z.t;`$("AAPL";"MSFT";"GOOG");150.000 280.500 140.200)'

# Panoproxy insert
q = build_insert_query_panoproxy("trade", cols)
# => '.utils.publishToRDB[`trade;(.z.t;`$("AAPL";"MSFT";"GOOG");150.000 280.500 140.200)]'
```

### Region Routing

```python
from kdb_helpers import (
    region_to_gateway, construct_gateway_triplet,
    region_to_panoproxy, construct_panoproxy_triplet,
)

region_to_gateway("US")        # => "nyk"
region_to_gateway("EU")        # => "ldn"

construct_gateway_triplet("credit", "US", "refData")
# => '.credit.nyk.refData'

construct_gateway_triplet("credit", "US", "refData", stripe="s1")
# => '.credit.nyk.s1.refData'

construct_panoproxy_triplet("US", "trades")
# => '.mt.get[`.credit.us.trades.realtime]'

construct_panoproxy_triplet("US", "trades", dates=["2024.01.10"])
# => '.mt.get[`.credit.us.trades.historical]'
```

---

## Data Transforms

All transforms are applied automatically by default. They run in a background thread to avoid blocking the event loop, with merged Polars passes for efficiency.

| Transform | Default | What it does |
|-----------|---------|-------------|
| `camel_case_headers` | `True` | `trade_date` &rarr; `tradeDate`, `BidSize` &rarr; `bidSize` |
| `nullify_na_strings` | `True` | `"NA"`, `""` &rarr; `null` in string columns |
| `convert_boolean_strings` | `True` | `isActive`: `"Y"`/`"N"` &rarr; `1`/`0` Int8 (only for `is[A-Z]*` columns) |
| `bools_to_int8` | `True` | Native `Boolean` &rarr; `Int8` |
| `lazy` | `True` | Return `LazyFrame` instead of `DataFrame` |

### Keyed Table Expansion

KDB keyed tables (which are `Dictionary(Table, Table)`) are automatically expanded into a flat DataFrame with all key and value columns, instead of collapsing into `keys`/`values` struct columns.

### Disable Transforms

```python
from kdb_client import CadeKdbTransformConfig

raw_transform = CadeKdbTransformConfig(
    camel_case_headers=False,
    nullify_na_strings=False,
    convert_boolean_strings=False,
    bools_to_int8=False,
    lazy=False,
)

# Per-query override
df = await kdb.query("select from trade", transform=raw_transform)

# Or set as default on the client
kdb = CadeKdb(":host:7015", transform=raw_transform)
```

---

## Performance Design

| Technique | Where |
|-----------|-------|
| **Zero-copy Arrow PyCapsules** | `pl.from_arrow(value)` wraps qroissant's Rust memory without copying |
| **Parallel column decode** | `DecodeOptions.with_parallel(True)` — Rust multi-threaded with SIMD |
| **Connection pooling** | qroissant `AsyncPool` with LIFO reuse, idle eviction, health checks |
| **Merged transform passes** | Boolean detection + null replacement in a single `with_columns()` |
| **Cached camelCase** | `@lru_cache(512)` on `_to_camel()` — regex runs once per column name |
| **Polars-native boolean detection** | `is_in(_BOOL_ALL_SERIES).all()` stays in Rust, no Python list materialization |
| **Inline scalar path** | `q.Atom` results skip `asyncio.to_thread` entirely |
| **Lock-free hot path** | `get_or_create()` returns existing pools without acquiring the lock |
| **Non-blocking pool creation** | `connect()` and `prewarm()` run outside locks — concurrent hosts don't block each other |

---

## Error Handling

```python
import qroissant as q

try:
    lf = await kdb.query("invalid q expression")
except q.QRuntimeError as e:
    print(f"q error: {e}")           # bad query syntax
except q.TransportError as e:
    print(f"connection lost: {e}")   # socket/IO failure
except q.PoolClosedError as e:
    print(f"pool closed: {e}")       # pool shut down mid-query
except asyncio.TimeoutError:
    print("query timed out")         # per-query timeout
except q.DecodeError as e:
    print(f"corrupt IPC: {e}")       # malformed response
```

Non-retriable errors (`QRuntimeError`, `DecodeError`, `ProtocolError`, `OperationError`, `PoolError`) are never retried or credential-rotated — they propagate immediately.

---

## Lifecycle & Shutdown

```python
# Context manager (recommended)
async with CadeKdbManager() as mgr:
    ...
# All pools closed automatically

# Manual lifecycle
mgr = CadeKdbManager()
await mgr.start()       # starts maintenance loop
...
await mgr.shutdown()     # stops maintenance, closes all pools

# Single client
kdb = CadeKdb(":host:7015")
await kdb.connect()      # explicit connect (or auto_connect=True)
...
await kdb.close()        # release connections

# Force-reset a misbehaving pool
await kdb.reset()
```

---

## Configuration Reference

### CadeKdbPoolConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_size` | `10` | Maximum connections in the pool |
| `min_idle` | `2` | Idle connections to keep warm |
| `checkout_timeout_ms` | `5000` | Max wait for a connection (ms) |
| `idle_timeout_ms` | `60000` | Evict idle connections after (ms) |
| `max_lifetime_ms` | `300000` | Replace connections after (ms) |
| `test_on_checkout` | `True` | Validate before handing out |
| `healthcheck_query` | `"::"` | q expression for health checks |
| `retry_attempts` | `1` | Retries after initial failure |
| `retry_backoff_ms` | `100` | Delay between retries (ms) |
| `prewarm` | `True` | Open `min_idle` connections eagerly |

### CadeKdbTransformConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `camel_case_headers` | `True` | Rename columns to camelCase |
| `nullify_na_strings` | `True` | `"NA"` / `""` &rarr; null |
| `convert_boolean_strings` | `True` | `is[A-Z]*` columns: Y/N &rarr; 0/1 |
| `bools_to_int8` | `True` | Boolean &rarr; Int8 |
| `lazy` | `True` | Return LazyFrame |

### CadeKdbManager

| Parameter | Default | Description |
|-----------|---------|-------------|
| `credentials` | `[("credituser","creditpass"), ("produser","prodpass")]` | Default credential combos |
| `pool_config` | `CadeKdbPoolConfig()` | Default pool settings |
| `transform` | `CadeKdbTransformConfig()` | Default transforms |
| `connection_timeout_ms` | `5000` | TCP connect timeout |
| `query_timeout` | `30.0` | Default per-query timeout (seconds) |
| `maintenance_interval` | `20.0` | Seconds between idle-pool reaping |
