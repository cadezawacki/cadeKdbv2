"""KDB query building helpers, routing utilities, and convenience wrappers.

Ported from the legacy ``kdb.py`` module.  All functions are pure (no I/O)
except the async convenience wrappers at the bottom which require a
``CadeKdbManager`` instance.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import polars as pl
    from kdb_manager import ConnectionList, CadeKdbManager

__all__ = [
    # Table name extraction
    "extract_table_name",
    # Query building
    "kdb_where",
    "build_kdb_where",
    "kdb_fby",
    "kdb_by",
    "kdb_col_select_helper",
    # Value list builders
    "q_symbols",
    "q_strings",
    "q_floats",
    "q_ints",
    "q_now",
    # Insert builders
    "build_insert_query_panoproxy",
    "build_insert_query_generic",
    # Region / routing helpers
    "region_to_gateway",
    "construct_gateway_triplet",
    "region_to_panoproxy",
    "construct_panoproxy_triplet",
    # Convenience async wrappers
    "pano",
    "gateway",
    "ldn",
    "nyk",
    "panoa",
]


# ---------------------------------------------------------------------------
# Table name extraction
# ---------------------------------------------------------------------------

_FROM_TABLE_RE = re.compile(
    r"(?i)\bfrom\s+(?:\.mt\.get\[\s*`?(\.[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*)`?"
    r"\s*\]|(\.?[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*))"
)


def extract_table_name(s: str) -> str | None:
    """Extract the table name from a q ``select from ...`` expression."""
    m = _FROM_TABLE_RE.search(s)
    if m:
        return m.group(1) or m.group(2)
    return None


# ---------------------------------------------------------------------------
# KDB WHERE / BY / FBY clause builders
# ---------------------------------------------------------------------------

_SAFE_COL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_. ]*$")


def _parse_condition(col: str, condition: Any) -> str:
    """Convert a single (column, condition) pair into a q WHERE fragment."""

    def _compact_seq(seq: Sequence[Any]) -> list[Any]:
        return [x for x in seq if x is not None]

    # Guard against q injection via crafted column names
    col_name = col.split(" ")[-1] if " " in col else col
    if not _SAFE_COL_RE.match(col_name):
        raise ValueError(f"Unsafe column name rejected: {col_name!r}")

    if condition is None:
        return f"not null {col.split(' ')[-1]}" if "not" in col else f"null {col}"

    if isinstance(condition, (int, float)):
        return f"{col}={condition}"

    if isinstance(condition, str):
        return f"{col}=`{condition}"

    if isinstance(condition, (list, tuple, set)):
        items = _compact_seq(list(condition))
        if not items:
            return f"{col} in ()"

        if all(isinstance(x, str) for x in items):
            symbols = "`".join([""] + list(items))
            return f"{col} in {symbols}"

        return f"{col} in ({' '.join(map(str, items))})"

    if isinstance(condition, dict):
        value = condition.get("value")
        dtype = condition.get("dtype", "auto")

        if value is None:
            return f"{col} is null"

        if isinstance(value, (list, tuple, set)):
            items = _compact_seq(list(value))
            if not items:
                return f"{col} in ()"

            if dtype == "string":
                if len(items) == 1:
                    escaped = str(items[0]).replace('"', '\\"')
                    return f'{col} like "*{escaped}*"'
                quoted = "; ".join(
                    f'"{str(v).replace(chr(34), chr(92) + chr(34))}"'
                    for v in items
                )
                return f"{col} in ({quoted})"

            if dtype == "string_exact":
                if len(items) == 1:
                    escaped = str(items[0]).replace('"', '\\"')
                    return f'{col} like "{escaped}"'
                quoted = "; ".join(
                    f'"{str(v).replace(chr(34), chr(92) + chr(34))}"'
                    for v in items
                )
                return f"{col} in ({quoted})"

            if dtype in ("sym", "symbol"):
                symbols = "`".join([""] + [str(v) for v in items])
                return f"{col} in {symbols}"

            return f"{col} in ({'; '.join(map(str, items))})"

        # scalar dict value
        if dtype == "string":
            escaped = str(value).replace('"', '\\"')
            return f'{col} like "*{escaped}*"'
        if dtype == "string_exact":
            escaped = str(value).replace('"', '\\"')
            return f'{col} like "{escaped}"'
        if dtype in ("sym", "symbol"):
            return f"{col}=`{value}"
        if isinstance(value, str) and dtype == "auto":
            return f"{col}=`{value}"
        return f"{col}={value}"

    return f"{col}={condition}"


def build_kdb_where(filters: list[dict[str, Any]]) -> str:
    """Build a q WHERE clause from a list of ``{col: condition}`` dicts.

    Examples
    --------
    >>> build_kdb_where([{"sym": "AAPL"}, {"date": {"value": "2024.01.01"}}])
    'sym=`AAPL,date=`2024.01.01'
    """
    if not filters:
        return ""
    conditions = [
        _parse_condition(col, cond)
        for filter_dict in filters
        for col, cond in filter_dict.items()
    ]
    return ",".join(conditions)


def kdb_where(*filters: dict[str, Any]) -> str:
    """Convenience wrapper: ``kdb_where({"sym":"AAPL"}, {"date": ...})``."""
    if len(filters) == 1 and isinstance(filters[0], list):
        return build_kdb_where(filters[0])
    return build_kdb_where(list(filters))


def kdb_fby(
    vars: str | list[str], d: str = "last", letter: str | None = None
) -> str:
    """Build a q ``fby`` expression.

    Parameters
    ----------
    vars : str or list[str]
        Group-by variable(s).
    d : str
        Aggregation direction (``"first"`` or ``"last"``).
    letter : str, optional
        Row-index variable (defaults to ``"i"`` for first, ``"j"`` for last).
    """
    l = letter or {"first": "i", "last": "j"}.get(d)
    f = f"([]{';'.join(vars)})" if isinstance(vars, list) else vars
    return f"{l} = ({d}; {l}) fby {f}"


def kdb_by(vars: str | list[str]) -> str:
    """Build a q ``by`` clause.

    >>> kdb_by(["sym", "date"])
    ' by sym,date'
    """
    if isinstance(vars, str):
        vars = [vars]
    return " by " + ",".join(vars)


def kdb_col_select_helper(
    col_lst: Sequence[str],
    method: str | None = "first",
    fills: bool = False,
) -> str:
    """Build a q column-select expression.

    Parameters
    ----------
    col_lst : list[str]
        Column names.  ``"alias:col"`` syntax supported.
    method : str or None
        Aggregation (``"first"``, ``"last"``, ``"avg"``, …).
        ``None`` or ``"none"`` for no aggregation.
    fills : bool
        Wrap each column in ``fills[...]``.
    """
    if not col_lst:
        return ""
    if method is None or method == "none":
        method = ""
    out: list[str] = []
    seen: set[str] = set()
    if method != "" and not method.endswith(" "):
        method += " "
    for col in col_lst:
        if col in seen:
            continue
        seen.add(col)
        if ":" in col:
            a, b = col.split(":", 1)
            if fills:
                b = f"fills[{b}]"
            out.append(f"{a}:{method}{b}")
        else:
            c = f"fills[{col}]" if fills else col
            out.append(f"{method}{c}")
    return ",".join(out) if out else ""


# ---------------------------------------------------------------------------
# Value list builders (for INSERT queries)
# ---------------------------------------------------------------------------


class _ColExpr:
    """Opaque wrapper holding a q expression string for a column vector."""

    __slots__ = ("expr",)

    def __init__(self, expr: str) -> None:
        self.expr = expr

    def __repr__(self) -> str:
        return f"_ColExpr({self.expr!r})"


def _q_escape_string(s: str) -> str:
    return '"' + s.replace('"', r"\"") + '"'


def _q_symbol_list(strings: Sequence[str]) -> str:
    if not strings:
        return "`$()"
    return "`$(" + ";".join(_q_escape_string(str(x)) for x in strings) + ")"


def _q_string_list(strings: Sequence[str]) -> str:
    if not strings:
        return "()"
    return "(" + ";".join(_q_escape_string(str(x)) for x in strings) + ")"


def _q_float_list(vals: Sequence[float | None]) -> str:
    if not vals:
        return "0#0f"
    out: list[str] = []
    for v in vals:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out.append("0n")
            continue
        fv = float(v)
        txt = (
            f"{fv:.3f}"
            if fv == round(fv, 3)
            else f"{fv:.6f}".rstrip("0").rstrip(".")
        )
        if "." not in txt and "e" not in txt and "E" not in txt:
            txt += ".0"
        out.append(txt)
    return " ".join(out)


def _q_int_list(vals: Sequence[int]) -> str:
    if not vals:
        return "0#0j"
    return " ".join(str(int(v)) for v in vals)


def _q_now_repeated(n: int) -> str:
    return f"{int(n)}#enlist .z.t" if n > 1 else ".z.t"


def q_symbols(strings: Sequence[str]) -> _ColExpr:
    """Build a q symbol list expression."""
    return _ColExpr(_q_symbol_list(strings))


def q_strings(strings: Sequence[str]) -> _ColExpr:
    """Build a q string list expression."""
    return _ColExpr(_q_string_list(strings))


def q_floats(values: Sequence[float | None]) -> _ColExpr:
    """Build a q float list expression."""
    return _ColExpr(_q_float_list(values))


def q_ints(values: Sequence[int]) -> _ColExpr:
    """Build a q integer list expression."""
    return _ColExpr(_q_int_list(values))


def q_now(n_rows: int) -> _ColExpr:
    """Build a q expression for *n_rows* copies of ``.z.t``."""
    return _ColExpr(_q_now_repeated(n_rows))


# ---------------------------------------------------------------------------
# INSERT query builders
# ---------------------------------------------------------------------------


def build_insert_query_panoproxy(
    table_name: str, columns: Sequence[_ColExpr]
) -> str:
    """Build a panoproxy-style insert expression."""
    parts = "; ".join(c.expr for c in columns)
    return f".utils.publishToRDB[`{table_name};({parts})]"


def build_insert_query_generic(
    table_name: str, columns: Sequence[_ColExpr]
) -> str:
    """Build a generic q insert expression."""
    parts = "; ".join(c.expr for c in columns)
    return f"`{table_name} insert ({parts})"


# ---------------------------------------------------------------------------
# Region / routing helpers
# ---------------------------------------------------------------------------

_GATEWAY_MAP = {"US": "nyk", "EU": "ldn", "SGP": "sgp"}
_PANOPROXY_MAP = {"US": "us", "EU": "eu", "SGP": "sgp"}


@lru_cache(maxsize=128)
def region_to_gateway(region: str | None = None) -> str:
    """Map a region code to a gateway suffix (``"US"`` -> ``"nyk"``)."""
    region = region or "US"
    return _GATEWAY_MAP.get(region.upper(), "nyk")


@lru_cache(maxsize=128)
def construct_gateway_triplet(
    schema: str,
    region: str,
    table: str,
    stripe: str | None = None,
) -> str:
    """Build a gateway-style table path.

    >>> construct_gateway_triplet("credit", "US", "refData")
    '.credit.nyk.refData'
    >>> construct_gateway_triplet("credit", "US", "refData", stripe="s1")
    '.credit.nyk.s1.refData'
    """
    region = region_to_gateway(region)
    if stripe is not None:
        return f".{schema}.{region}.{stripe}.{table}"
    return f".{schema}.{region}.{table}"


@lru_cache(maxsize=128)
def region_to_panoproxy(region: str | None = None) -> str:
    """Map a region code to a panoproxy suffix (``"US"`` -> ``"us"``)."""
    region = region or "US"
    return _PANOPROXY_MAP.get(region.upper(), "us")


def construct_panoproxy_triplet(
    region: str,
    table: str,
    dates: Any = None,
    base: str = "credit",
) -> str:
    """Build a panoproxy-style ``.mt.get[...]`` path.

    Automatically chooses ``realtime`` vs ``historical`` based on *dates*.
    If date logic dependencies (``is_today``, ``parse_date``, ``get_today``)
    are unavailable, defaults to ``"realtime"``.

    >>> construct_panoproxy_triplet("US", "refData")
    '.mt.get[`.credit.us.refData.realtime]'
    """
    region = region_to_panoproxy(region)

    # Determine realtime vs historical.
    d = "realtime"
    try:
        # Try to import app-specific date helpers (optional dependency).
        from app.helpers.date_helpers import get_today, is_today, parse_date

        if isinstance(dates, list) and len(dates) > 1:
            dts = [parse_date(x, biz=True) for x in dates]
            d = "historical" if min(dts) < get_today(utc=True) else "realtime"
        else:
            d = "realtime" if is_today(dates, utc=True) else "historical"
    except ImportError:
        # Date helpers not available — default to realtime.
        if dates is not None:
            d = "historical"

    return f".mt.get[`.{base}.{region}.{table}.{d}]"


# ---------------------------------------------------------------------------
# Convenience async wrappers
#
# These require a CadeKdbManager and connection configs.  They are factory
# functions: call ``make_convenience_wrappers(mgr, configs)`` once at
# startup to get bound ``pano()``, ``gateway()`` etc.
# ---------------------------------------------------------------------------


async def pano(
    q: str,
    *,
    manager: CadeKdbManager,
    config: Any,
    name: str = "panoproxy",
    **kwargs: Any,
) -> pl.LazyFrame | pl.DataFrame:
    """Query the panoproxy cluster."""
    from kdb_manager import fconn, query_kdb

    cfg = config if isinstance(config, list) else [config]
    return await query_kdb(q, manager=manager, config=fconn(cfg), name=name, **kwargs)


async def gateway(
    q: str,
    *,
    manager: CadeKdbManager,
    config: Any,
    name: str = "gateway",
    **kwargs: Any,
) -> pl.LazyFrame | pl.DataFrame:
    """Query the gateway cluster."""
    from kdb_manager import fconn, query_kdb

    cfg = config if isinstance(config, list) else [config]
    return await query_kdb(q, manager=manager, config=fconn(cfg), name=name, **kwargs)


async def ldn(
    q: str,
    *,
    manager: CadeKdbManager,
    config: Any,
    name: str = "panoproxy_eu",
    **kwargs: Any,
) -> pl.LazyFrame | pl.DataFrame:
    """Query the London panoproxy cluster."""
    from kdb_manager import fconn, query_kdb

    cfg = config if isinstance(config, list) else [config]
    return await query_kdb(q, manager=manager, config=fconn(cfg, region="EU"), name=name, **kwargs)


async def nyk(
    q: str,
    *,
    manager: CadeKdbManager,
    config: Any,
    name: str = "panoproxy_us",
    **kwargs: Any,
) -> pl.LazyFrame | pl.DataFrame:
    """Query the New York panoproxy cluster."""
    from kdb_manager import fconn, query_kdb

    cfg = config if isinstance(config, list) else [config]
    return await query_kdb(q, manager=manager, config=fconn(cfg, region="US"), name=name, **kwargs)


async def panoa(
    q: str,
    *,
    manager: CadeKdbManager,
    config: Any,
    region: str | None = None,
    name: str = "panoproxy",
    **kwargs: Any,
) -> pl.LazyFrame | pl.DataFrame:
    """Query panoproxy with aggressive fan-out across prod/dr hosts."""
    from kdb_manager import fconn, query_kdb

    cfg = config if isinstance(config, list) else [config]
    if region:
        return await query_kdb(
            q,
            manager=manager,
            config=fconn(cfg, region=region, dbtype=["dr", "prod"]),
            aggressive=True,
            name=name,
            **kwargs,
        )
    return await query_kdb(
        q,
        manager=manager,
        config=fconn(cfg, dbtype=["dr", "prod"]),
        aggressive=True,
        name=name,
        **kwargs,
    )
