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

# Strict column identifier: letters/digits/underscore/dot only.  Whitespace,
# semicolons, quotes, backticks — anything that could terminate or splice the
# surrounding q expression — is rejected outright.
_SAFE_COL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
# Support the "not col" form used when condition is None (i.e. `not null col`).
_SAFE_NOT_COL_RE = re.compile(r"^not\s+([A-Za-z_][A-Za-z0-9_.]*)$")
# Valid aggregation method names for kdb_col_select_helper.
_SAFE_METHOD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Characters that are unsafe inside a q symbol literal — backticks, semicolons,
# whitespace, quotes, and backslashes can all terminate or splice the symbol
# and allow injection of arbitrary q code.
_UNSAFE_SYM_RE = re.compile(r"[`;\s\"\\]")


def _validate_col(col: str) -> tuple[str, str | None]:
    """Validate a column expression.

    Returns ``(bare_col, not_form_col | None)``.  If the caller passed
    ``"not colname"``, *not_form_col* is the full matched expression and
    *bare_col* is the underlying column name.  Otherwise *not_form_col*
    is ``None`` and *bare_col* is the validated column name.

    This is the sole injection guard for WHERE fragments — callers must
    rely on it to reject any input that could splice arbitrary q code.
    """
    if not isinstance(col, str):
        raise ValueError(f"Unsafe column rejected: {col!r}")
    m = _SAFE_NOT_COL_RE.match(col)
    if m is not None:
        return m.group(1), col
    if _SAFE_COL_RE.match(col):
        return col, None
    raise ValueError(f"Unsafe column rejected: {col!r}")


def _q_escape_str(s: str) -> str:
    """Escape a string for safe embedding in a q string literal.

    Backslash MUST be escaped before quote to avoid double-processing.
    """
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _q_sym(s: Any) -> str:
    """Build a q symbol literal, rejecting unsafe characters.

    This is the primary defense against q-expression injection via filter
    values.  Any character that could terminate or splice the symbol is
    rejected outright — callers must use ``dtype="string"`` for values with
    whitespace / quotes.
    """
    s = str(s)
    if _UNSAFE_SYM_RE.search(s):
        raise ValueError(f"Unsafe symbol value rejected: {s!r}")
    return f"`{s}"


def _q_number(v: Any) -> str:
    """Serialize a number for safe q embedding.

    Rejects ``bool`` (a subclass of int, but carries different semantics),
    NaN, and infinity — all of which either misrepresent intent or produce
    invalid q tokens.
    """
    if isinstance(v, bool):
        raise ValueError(f"Boolean rejected in numeric context: {v!r}")
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        raise ValueError(f"Non-finite numeric rejected: {v!r}")
    if not isinstance(v, (int, float)):
        raise ValueError(f"Non-numeric rejected: {v!r}")
    return str(v)


def _parse_condition(col: str, condition: Any) -> str:
    """Convert a single (column, condition) pair into a q WHERE fragment."""

    def _compact_seq(seq: Sequence[Any]) -> list[Any]:
        return [x for x in seq if x is not None]

    # Guard against q injection via crafted column names.  The FULL col
    # expression is validated — not just a trailing token — so that
    # strings like ``"foo;system\"rm -rf\""`` cannot splice arbitrary q.
    bare_col, not_form = _validate_col(col)

    if condition is None:
        if not_form is not None:
            return f"not null {bare_col}"
        return f"null {bare_col}"

    # Conditions with operators (``=``, ``in``) only make sense on a plain
    # column, not the ``not col`` form.  Reject to prevent producing
    # invalid / ambiguous q expressions like ``"not col=val"``.
    if not_form is not None:
        raise ValueError(
            f"'not col' form only valid with None condition: {col!r}"
        )
    col = bare_col

    # ``bool`` MUST be checked before ``int`` — bool is a subclass of int.
    if isinstance(condition, bool):
        raise ValueError(f"Boolean scalar rejected: {condition!r}")

    if isinstance(condition, (int, float)):
        return f"{col}={_q_number(condition)}"

    if isinstance(condition, str):
        return f"{col}={_q_sym(condition)}"

    if isinstance(condition, (list, tuple, set)):
        items = _compact_seq(list(condition))
        if not items:
            return f"{col} in ()"

        if all(isinstance(x, str) for x in items):
            # Each symbol validated independently — no raw interpolation.
            symbols = "".join(_q_sym(x) for x in items)
            return f"{col} in {symbols}"

        return f"{col} in ({' '.join(_q_number(x) for x in items)})"

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
                    return f'{col} like "*{_q_escape_str(str(items[0]))}*"'
                quoted = "; ".join(
                    f'"{_q_escape_str(str(v))}"' for v in items
                )
                return f"{col} in ({quoted})"

            if dtype == "string_exact":
                if len(items) == 1:
                    return f'{col} like "{_q_escape_str(str(items[0]))}"'
                quoted = "; ".join(
                    f'"{_q_escape_str(str(v))}"' for v in items
                )
                return f"{col} in ({quoted})"

            if dtype in ("sym", "symbol"):
                symbols = "".join(_q_sym(v) for v in items)
                return f"{col} in {symbols}"

            return f"{col} in ({'; '.join(_q_number(v) for v in items)})"

        # scalar dict value
        if dtype == "string":
            return f'{col} like "*{_q_escape_str(str(value))}*"'
        if dtype == "string_exact":
            return f'{col} like "{_q_escape_str(str(value))}"'
        if dtype in ("sym", "symbol"):
            return f"{col}={_q_sym(value)}"
        if isinstance(value, str) and dtype == "auto":
            return f"{col}={_q_sym(value)}"
        if isinstance(value, bool):
            raise ValueError(f"Boolean dict value rejected: {value!r}")
        if isinstance(value, (int, float)):
            return f"{col}={_q_number(value)}"
        raise ValueError(f"Unsupported dict value: {value!r}")

    raise ValueError(f"Unsupported condition type: {type(condition).__name__}")


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


_FBY_DIRECTIONS = frozenset({"first", "last"})


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

    All inputs are validated against strict identifier regexes to prevent
    q-expression injection — prior versions spliced *d*, *letter*, and
    *vars* directly into the output string, allowing a caller-controlled
    value such as ``d='first;system "rm -rf /"'`` to execute arbitrary q.
    """
    if d not in _FBY_DIRECTIONS:
        raise ValueError(f"Invalid fby direction: {d!r}")
    if letter is not None and not _SAFE_METHOD_RE.match(letter):
        raise ValueError(f"Invalid fby letter: {letter!r}")
    l = letter or {"first": "i", "last": "j"}[d]

    if isinstance(vars, list):
        for v in vars:
            if not isinstance(v, str) or not _SAFE_COL_RE.match(v):
                raise ValueError(f"Invalid fby var: {v!r}")
        f = f"([]{';'.join(vars)})"
    else:
        if not isinstance(vars, str) or not _SAFE_COL_RE.match(vars):
            raise ValueError(f"Invalid fby var: {vars!r}")
        f = vars
    return f"{l} = ({d}; {l}) fby {f}"


def kdb_by(vars: str | list[str]) -> str:
    """Build a q ``by`` clause.

    >>> kdb_by(["sym", "date"])
    ' by sym,date'
    """
    if isinstance(vars, str):
        vars = [vars]
    for v in vars:
        if not isinstance(v, str) or not _SAFE_COL_RE.match(v):
            raise ValueError(f"Invalid by var: {v!r}")
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
        Column names.  ``"alias:col"`` syntax supported — both halves must
        be valid q identifiers.
    method : str or None
        Aggregation (``"first"``, ``"last"``, ``"avg"``, …).
        ``None`` or ``"none"`` for no aggregation.
    fills : bool
        Wrap each column in ``fills[...]``.

    All caller-provided strings are validated against strict identifier
    regexes to block q-expression injection.  Prior versions interpolated
    raw strings into the output, allowing ``";system \"rm -rf /\""`` and
    similar to execute arbitrary q on the remote host.
    """
    if not col_lst:
        return ""
    if method is None or method == "none":
        method = ""
    elif not _SAFE_METHOD_RE.match(method):
        raise ValueError(f"Unsafe aggregation method: {method!r}")
    out: list[str] = []
    seen: set[str] = set()
    if method != "" and not method.endswith(" "):
        method += " "
    for col in col_lst:
        if not isinstance(col, str):
            raise ValueError(f"Unsafe column entry: {col!r}")
        if col in seen:
            continue
        seen.add(col)
        if ":" in col:
            a, b = col.split(":", 1)
            if not _SAFE_METHOD_RE.match(a) or not _SAFE_COL_RE.match(b):
                raise ValueError(f"Unsafe column entry: {col!r}")
            if fills:
                b = f"fills[{b}]"
            out.append(f"{a}:{method}{b}")
        else:
            if not _SAFE_COL_RE.match(col):
                raise ValueError(f"Unsafe column entry: {col!r}")
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
    # Backslash MUST be escaped before quote to avoid double-processing.
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


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
