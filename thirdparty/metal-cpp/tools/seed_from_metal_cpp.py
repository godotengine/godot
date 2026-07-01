#!/usr/bin/env python3
"""Seed per-class member filters from upstream metal-cpp.

Walks `thirdparty/metal-cpp` to discover, for each ObjC class upstream
wraps, the set of selectors its inline impls touch via
`_<PREFIX>_PRIVATE_SEL(<accessor>)`. Also parses each framework's macOS
SDK headers to discover the *full* SDK selector surface per class.

For every class, we then choose the more compact representation:
  * `include.members`  — the upstream selector allow-list
  * `exclude.members`  — the SDK selectors upstream *omits*
                          (only when significantly smaller; see _EXCLUDE_RATIO)

Also writes each framework's `include.classes` to exactly the ObjC types
upstream defines, so the generator doesn't wrap SDK siblings upstream chose
to leave out (e.g. `URLQueryItem`, `URLComponents`, `FileSecurity` from
`NSURL.h`).

ruamel.yaml's round-trip loader preserves comments / key order in the
in-place config edit.

Usage:
    venv/bin/python3 tools/seed_from_metal_cpp.py \
        --metal-cpp ../metal-cpp \
        --config tools/config.yaml
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

try:
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap, CommentedSeq
except ImportError:
    print("ruamel.yaml required: pip install ruamel.yaml", file=sys.stderr)
    sys.exit(1)

# `metalcpp_common` lives next to this script; ensure we can import it
# whether the script is run from the repo root or from tools/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metalcpp_common import ObjCParser  # noqa: E402


# C++ namespaces upstream uses (in priority order — longer prefixes first so
# `MTL4FX` is tried before `MTL4` before `MTL`).
NAMESPACES = ["NS", "MTL4FX", "MTL4", "MTLFX", "MTL", "CA"]

# Pattern alternation built once: longest-match first so `MTL4` doesn't
# swallow the `MTL` of `MTL4Foo`.
_NS_ALT = "|".join(re.escape(n) for n in NAMESPACES)

# Matches an inline-impl signature header: `<ret> NS::<Class>::<method>(...)`
# anywhere on a line. Class is captured for the next selector association.
_SIG_RE = re.compile(rf"\b({_NS_ALT})::(\w+)::\w+\s*\(")
# Matches the selector reference inside an inline-impl body.
_SEL_RE = re.compile(rf"_(?:{_NS_ALT})_PRIVATE_SEL\((\w+)\)")
# Matches a full class DEFINITION line: `class [_EXPORT] <Name> : public ...`.
# Forward decls `class <Name>;` are intentionally not matched — they don't
# imply upstream wraps the class.
_CLASS_DEF_RE = re.compile(r"^\s*class\s+(?:_\w+\s+)?(\w+)\s*:\s*public\b")

# Prefer `exclude` only when it would shrink the per-class list by at least
# this factor. Tuned so very-large classes (MTLDevice, MTLRenderCommandEncoder)
# collapse to a small drop list while small classes keep the explicit
# include — `include` documents intent better when the lists are comparable.
_EXCLUDE_RATIO = 2  # len(exclude) * _EXCLUDE_RATIO < len(include)


def accessor_to_selector(accessor: str) -> str:
    """`foo_bar_` → `foo:bar:`; `commit` → `commit`."""
    return accessor.replace("_", ":") if accessor.endswith("_") else accessor


def parse_upstream_hpp(path: Path) -> dict[str, set[str]]:
    """Scan one upstream `.hpp`. Returns `{ObjCClassName: {selectors}}`.

    State machine: each line that contains an `<Namespace>::<Class>::method(`
    signature sets the "current class"; each subsequent `_<NS>_PRIVATE_SEL(...)`
    on the same or following lines is attributed to that class. Resets when
    a new signature appears.
    """
    out: dict[str, set[str]] = {}
    current: str | None = None  # ObjC class name (`<ns><class>`)
    for line in path.read_text().splitlines():
        sig = _SIG_RE.search(line)
        if sig:
            current = f"{sig.group(1)}{sig.group(2)}"
        if current is None:
            continue
        for m in _SEL_RE.finditer(line):
            out.setdefault(current, set()).add(accessor_to_selector(m.group(1)))
    return out


def parse_upstream_classes(path: Path) -> set[str]:
    """Return the set of ObjC class names upstream defines in this `.hpp`.

    Prefix is derived from the filename (`MTL4CommandQueue.hpp` → `MTL4`), so
    a definition `class CommitOptions : public ...` in that file produces
    `MTL4CommitOptions`. Files not matching any known prefix are ignored.
    """
    prefix = next((n for n in NAMESPACES if path.stem.startswith(n)), None)
    if prefix is None:
        return set()
    out: set[str] = set()
    for line in path.read_text().splitlines():
        if m := _CLASS_DEF_RE.match(line):
            out.add(f"{prefix}{m.group(1)}")
    return out


def collect_upstream(metal_cpp_root: Path) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Walk upstream metal-cpp once.

    Returns `(selectors_by_class, classes_by_prefix)`:
      * `selectors_by_class[ObjCClass]` — selectors referenced by inline impls
      * `classes_by_prefix[prefix]` — ObjC class names whose full definitions
        appear in upstream (drives the framework-level `include.classes`).
    """
    selectors: dict[str, set[str]] = {}
    classes: dict[str, set[str]] = {}
    for hpp in sorted(metal_cpp_root.glob("**/*.hpp")):
        for cls, sels in parse_upstream_hpp(hpp).items():
            selectors.setdefault(cls, set()).update(sels)
        for cls in parse_upstream_classes(hpp):
            prefix = next((n for n in NAMESPACES if cls.startswith(n)), None)
            if prefix:
                classes.setdefault(prefix, set()).add(cls)
    return selectors, classes


def _macos_sdk_path() -> Path:
    """Resolve the macOS SDK via xcrun. macOS exposes the broadest method
    surface — picking it ensures we see every selector that any platform
    might support (per-platform `API_AVAILABLE` shrinks libclang's AST on
    other SDKs)."""
    out = subprocess.check_output(["xcrun", "--sdk", "macosx",
                                    "--show-sdk-path"]).decode().strip()
    return Path(out)


def collect_sdk_classes(cfg: dict, sdk_path: Path) -> dict[str, "ParsedClass"]:
    """Parse every ObjC SDK header listed in the config and return
    `{ObjCClassName: ParsedClass}`. C-API frameworks (apinotes-driven) are
    skipped — they don't have ObjC selectors."""
    parser = ObjCParser(sdk_path)
    out: dict[str, ParsedClass] = {}
    for fw in cfg["frameworks"]:
        if fw.get("api_notes"):
            continue
        sdk_fw_name = fw.get("sdk_framework") or fw["name"]
        fw_dir = (sdk_path / "System" / "Library" / "Frameworks"
                   / f"{sdk_fw_name}.framework" / "Headers")
        for header in fw.get("headers", []) or []:
            hp = fw_dir / header
            if not hp.exists():
                continue
            data = parser.parse_header(hp)
            for cls in data.classes:
                # Each class shows up in at most one header (and re-parsing
                # the same one would duplicate), so first-write wins.
                out.setdefault(cls.name, ParsedClass.from_cls(cls))
    return out


class ParsedClass:
    """Subset of `ObjCClass` we keep for selector comparison."""
    __slots__ = ("methods", "properties")

    def __init__(self, methods: list[str], properties: list[tuple[str, bool, str]]):
        # (selector,)
        self.methods = methods
        # (prop_name, is_readonly, objc_type)
        self.properties = properties

    @classmethod
    def from_cls(cls, parsed_cls) -> "ParsedClass":
        return cls(
            methods=[m.selector for m in parsed_cls.methods],
            properties=[(p.name, p.is_readonly, p.objc_type)
                         for p in parsed_cls.properties],
        )


def _prop_in_upstream(prop_name: str, is_readonly: bool, objc_type: str,
                      upstream: set[str]) -> bool:
    """True when upstream's selector set references this SDK property under
    any of its filter-recognized forms.

    The generator's `filter_class` matches a property when either `prop.name`
    or its `set<Prop>:` form is in the allow-list. Upstream metal-cpp,
    however, references properties via their actual ObjC selector
    (`_PRIVATE_SEL(isRasterizationEnabled)`, `_PRIVATE_SEL(setLabel_)`, …),
    not by property name. We therefore probe every plausible upstream-style
    selector to decide whether upstream "wraps" the property."""
    cap = prop_name[0].upper() + prop_name[1:]
    if prop_name in upstream:
        return True
    if not is_readonly and f"set{cap}:" in upstream:
        return True
    # Apple's BOOL property convention: getter selector is `is<Prop>` even
    # though the property name is `<prop>`.
    if objc_type.strip() in {"BOOL", "bool", "_Bool"} and f"is{cap}" in upstream:
        return True
    return False


def compute_exclude(parsed: ParsedClass, upstream: set[str]) -> set[str]:
    """Return the set of SDK identifiers (method selectors or property
    names) that upstream does *not* wrap. Suitable for `exclude.members`
    — the generator's filter will drop these and keep the rest."""
    excl: set[str] = set()
    for selector in parsed.methods:
        if selector not in upstream:
            excl.add(selector)
    for prop_name, is_readonly, objc_type in parsed.properties:
        if not _prop_in_upstream(prop_name, is_readonly, objc_type, upstream):
            excl.add(prop_name)
    return excl


def update_config(config_path: Path, upstream_sels: dict[str, set[str]],
                  classes_by_prefix: dict[str, set[str]],
                  sdk_classes: dict[str, ParsedClass],
                  dry_run: bool) -> dict[str, int]:
    """In-place merge. Writes:
      * `frameworks[i].include.classes` — exactly the upstream-defined types
      * `class_overrides[Class].include.members` or `.exclude.members`,
        whichever is more compact.

    Returns counters: `{"include": n, "exclude": n, "wrapped": n, "sels": n}`.
    """
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = 200
    with config_path.open() as f:
        cfg = yaml.load(f)

    counts = {"include": 0, "exclude": 0, "no_filter": 0}

    def _clear_members(entry: CommentedMap, key: str) -> None:
        """Drop `entry[key].members` and the now-empty parent if needed."""
        sub = entry.get(key)
        if isinstance(sub, CommentedMap):
            sub.pop("members", None)
            if not sub:
                entry.pop(key)

    for fw in cfg["frameworks"]:
        prefix: str = fw["prefix"]
        # Constrain the framework's class universe to exactly what upstream
        # defines for this prefix. Without this the generator wraps every
        # @interface the SDK packs into a header (e.g. NSURL.h declares URL,
        # URLQueryItem, URLComponents, FileSecurity — but upstream only
        # wraps URL).
        upstream_classes = sorted(classes_by_prefix.get(prefix, set()))
        if upstream_classes:
            inc = fw.setdefault("include", CommentedMap())
            inc["classes"] = CommentedSeq(upstream_classes)
        co = fw.setdefault("class_overrides", CommentedMap())
        for cls in sorted(upstream_sels):
            owning_prefix = next(
                (p for p in NAMESPACES if cls.startswith(p)), ""
            )
            if owning_prefix != prefix:
                continue
            upstream = upstream_sels[cls]
            if not upstream:
                continue
            entry = co.get(cls)
            if not isinstance(entry, CommentedMap):
                entry = CommentedMap()
                co[cls] = entry
            # Skip structs / skipped entries — they don't take include/exclude.
            if entry.get("skip"):
                continue

            # Three outcomes per class:
            #   * exclude:   upstream wraps most of the SDK; the SDK diff is
            #                significantly smaller than the include list.
            #   * no_filter: upstream wraps the entire SDK surface (empty
            #                diff) — a filter would be a no-op, so drop it.
            #   * include:   anything else (default).
            mode = "include"
            include_members = sorted(upstream)
            exclude_members: list[str] = []
            if (parsed := sdk_classes.get(cls)) is not None:
                excl = compute_exclude(parsed, upstream)
                if not excl:
                    mode = "no_filter"
                elif len(excl) * _EXCLUDE_RATIO < len(upstream):
                    mode = "exclude"
                    exclude_members = sorted(excl)

            if mode == "exclude":
                _clear_members(entry, "include")
                exc = entry.get("exclude")
                if not isinstance(exc, CommentedMap):
                    exc = CommentedMap()
                    entry["exclude"] = exc
                exc["members"] = CommentedSeq(exclude_members)
            elif mode == "no_filter":
                _clear_members(entry, "include")
                _clear_members(entry, "exclude")
                # Drop the override entry entirely if nothing else is set
                # (rename / append / skip / packed / …). Keeps the seeded
                # config free of `ClassName: {}` noise.
                if not entry:
                    co.pop(cls)
            else:  # include
                _clear_members(entry, "exclude")
                inc = entry.get("include")
                if not isinstance(inc, CommentedMap):
                    inc = CommentedMap()
                    entry["include"] = inc
                inc["members"] = CommentedSeq(include_members)
            counts[mode] += 1

    if dry_run:
        yaml.dump(cfg, sys.stdout)
    else:
        with config_path.open("w") as f:
            yaml.dump(cfg, f)

    return counts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metal-cpp", type=Path, required=True,
                   help="Path to upstream metal-cpp (thirdparty/metal-cpp)")
    p.add_argument("--config", type=Path, required=True,
                   help="Path to config.yaml to update in place")
    p.add_argument("--dry-run", action="store_true",
                   help="Print updated YAML to stdout instead of writing back")
    args = p.parse_args()

    if not args.metal_cpp.is_dir():
        print(f"--metal-cpp not a directory: {args.metal_cpp}", file=sys.stderr)
        return 1
    if not args.config.is_file():
        print(f"--config not a file: {args.config}", file=sys.stderr)
        return 1

    upstream_sels, classes_by_prefix = collect_upstream(args.metal_cpp)
    if not upstream_sels:
        print("no selectors found — check --metal-cpp path", file=sys.stderr)
        return 1

    # SDK parsing is the slow step (~10s); load config once for both passes.
    yaml = YAML(typ="rt")
    with args.config.open() as f:
        cfg = yaml.load(f)
    sdk_path = _macos_sdk_path()
    sdk_classes = collect_sdk_classes(cfg, sdk_path)

    counts = update_config(args.config, upstream_sels, classes_by_prefix,
                           sdk_classes, args.dry_run)
    n_sels = sum(len(s) for s in upstream_sels.values())
    n_wrapped = sum(len(v) for v in classes_by_prefix.values())
    print(f"{counts['include']} include / {counts['exclude']} exclude / "
          f"{counts['no_filter']} no-filter classes, "
          f"{n_sels} upstream selectors, "
          f"{n_wrapped} wrapped types → {args.config}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
