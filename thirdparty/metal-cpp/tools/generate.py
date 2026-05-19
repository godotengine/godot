#!/usr/bin/env python3
"""Generate metal-cpp style C++ wrappers for Objective-C frameworks.

Parses ObjC framework headers using libclang and generates C++ wrappers
following the same patterns as Apple's metal-cpp (sendMessage, selector
registration, inline implementations).

Usage:
    python generate.py --metal-cpp PATH                 # all SDKs
    python generate.py --metal-cpp PATH --sdk macos     # single SDK
    python generate.py --metal-cpp PATH --strict        # fail on unresolvable types

`--metal-cpp` points at Apple's metal-cpp checkout; the hand-written hpps
listed under each framework's `keep_upstream:` are copied verbatim from
there into the generated tree.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from metalcpp_common import (  # noqa: F401 – re-exported for backwards compat
    BUILTIN_TYPE_MAP,
    PASSTHROUGH_TYPES,
    SYSTEM_HEADER_FOR_TYPE,
    Availability,
    CodeGenerator,
    FrameworkData,
    ObjCClass,
    ObjCEnum,
    ObjCEnumValue,
    ObjCMethod,
    ObjCParam,
    ObjCParser,
    ObjCProperty,
    TypeResolver,
    parse_apinotes_renames,
    strip_objc_prefix,
    strip_objc_prefix_aggressive,
)

log = logging.getLogger(__name__)


# ── Parallel parse workers ────────────────────────────────────────────────
# libclang holds the GIL while parsing, so threads serialize. ProcessPool
# spawns one worker per CPU; each builds its own `ObjCParser` once and reuses
# it across header parses (avoiding the ~100ms libclang Index setup per call).
_WORKER_PARSER: Optional[ObjCParser] = None


def _parse_worker_init(sdk_path: str) -> None:
    """ProcessPoolExecutor initializer — one ObjCParser per worker."""
    global _WORKER_PARSER
    _WORKER_PARSER = ObjCParser(Path(sdk_path))


def _parse_worker(header_path: str) -> "FrameworkData":
    """Parse one header in the worker process; returns the picklable
    FrameworkData. ObjCParser owns libclang resources that aren't picklable,
    so we set it once via the initializer and never cross the process
    boundary with it."""
    assert _WORKER_PARSER is not None, "worker not initialized"
    return _WORKER_PARSER.parse_header(Path(header_path))

TOOL_DIR = Path(__file__).resolve().parent
ROOT_DIR = TOOL_DIR.parent  # thirdparty/metal-cpp/


# ── Filtering ─────────────────────────────────────────────────────────────

def matches_any(name: str, patterns: list[str]) -> bool:
    """Check if name matches any of the given regex patterns."""
    return any(re.fullmatch(p, name) for p in patterns)


def _override_for(fw: "FrameworkConfig", name: str) -> dict:
    """Look up a class_overrides entry for `name`, falling back to its
    canonical (underscore-stripped) form. Returns `{}` when no entry exists."""
    return (fw.class_overrides.get(name)
            or fw.class_overrides.get(name.lstrip("_"))
            or {})


def filter_class(
    cls: ObjCClass,
    fw_exclude: dict,
    class_override: Optional[dict],
    apinotes_renames: Optional[dict[str, str]] = None,
) -> Optional[ObjCClass]:
    """Apply include/exclude filters to a class. Returns filtered copy or None.

    `apinotes_renames` is the `{selector: cpp_name}` map for this class derived
    from the SDK's `<Framework>.apinotes` SwiftName entries. Applied before any
    `class_override["rename"]`, so config overrides win.
    """
    props = list(cls.properties)
    methods = list(cls.methods)

    if class_override:
        inc = class_override.get("include", {})
        exc = class_override.get("exclude", {})

        # Per-class `members:` is a single allow-list keyed by ObjC selector
        # (for methods) or property name (for properties) — the SDK-stable
        # identifier in both cases. When present, anything not in the list
        # is dropped. A property also matches if its setter-form selector
        # (`set<Prop>:`) is in the list — covers Apple's `@property
        # (getter=isFoo) BOOL foo;` form, where the getter selector differs
        # from the property name and only the setter form is hand-known.
        def _prop_matches(p, allow):
            if matches_any(p.name, allow):
                return True
            return matches_any(f"set{p.name[0].upper()}{p.name[1:]}:", allow)

        if inc and "members" in inc:
            allow = inc["members"]
            props = [p for p in props if _prop_matches(p, allow)]
            methods = [m for m in methods if matches_any(m.selector, allow)]

        if exc_members := exc.get("members"):
            props = [p for p in props if not _prop_matches(p, exc_members)]
            methods = [m for m in methods if not matches_any(m.selector, exc_members)]

    # Framework-level excludes
    if exc_m := fw_exclude.get("methods"):
        methods = [m for m in methods if not matches_any(m.selector, exc_m)]
    if exc_p := fw_exclude.get("properties"):
        props = [p for p in props if not matches_any(p.name, exc_p)]

    # Even empty classes get emitted — subclasses inherit them, and the
    # umbrella header expects every <ClassName>.hpp the resolver knows about
    # to exist on disk.

    # Per-method renames. Apinotes SwiftName entries provide canonical short
    # names (`setBlendColorRed:green:blue:alpha:` → `setBlendColor`); the
    # config `rename:` map wins on conflict so we can override Apple's
    # SwiftName when upstream metal-cpp diverged (e.g. `encodeWait`).
    renames: dict[str, str] = {}
    if apinotes_renames:
        renames.update(apinotes_renames)
    renames.update((class_override or {}).get("rename", {}) or {})
    if renames:
        for m in methods:
            if m.selector in renames:
                m.cpp_name_override = renames[m.selector]

    return ObjCClass(
        name=cls.name,
        superclass=cls.superclass,
        properties=props,
        methods=methods,
        protocols=cls.protocols,
        is_protocol=cls.is_protocol,
        availability=cls.availability,
    )


# ── Configuration ─────────────────────────────────────────────────────────

@dataclass
class SDKConfig:
    name: str
    xcrun_sdk: str


@dataclass
class FrameworkConfig:
    name: str
    namespace: str
    prefix: str
    strip_prefix: str
    sdks: list[str]
    headers: list[str]
    include: dict
    exclude: dict
    class_overrides: dict
    # Hand-written hpps copied verbatim from upstream metal-cpp. These are
    # infrastructure / hard-to-generate files (NSDefines, NSObject base
    # class, MTLAccelerationStructureTypes' SIMD constructors). Listed as
    # bare names without `.hpp`.
    keep_upstream: list[str] = field(default_factory=list)
    # SDK framework directory if it differs from `name` — e.g. a virtual
    # framework "Metal4" that pulls MTL4*.h files out of Metal.framework but
    # emits them in a separate `MTL4::` namespace.
    sdk_framework: str = ""
    # Output directory if it differs from `name`. Used by virtual frameworks
    # that share an on-disk directory with another (Metal4 sits inside
    # Metal/ alongside the MTL classes — matches upstream metal-cpp).
    output_subdir: str = ""
    # Extra umbrella includes prepended to this framework's <Fw>.hpp. Lets
    # a base framework's umbrella pull in a sibling's (Metal.hpp →
    # Metal4.hpp) so client code only includes one umbrella.
    extra_umbrella_includes: list[str] = field(default_factory=list)
    shared: bool = False


def load_config(config_path: Path) -> tuple[list[SDKConfig], list[FrameworkConfig]]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    sdks = [SDKConfig(**s) for s in cfg["sdks"]]

    frameworks = []
    for fw in cfg["frameworks"]:
        frameworks.append(FrameworkConfig(
            name=fw["name"],
            namespace=fw["namespace"],
            prefix=fw["prefix"],
            strip_prefix=fw.get("strip_prefix", ""),
            sdks=fw["sdks"],
            headers=fw["headers"],
            include=fw.get("include", {}),
            exclude=fw.get("exclude", {}),
            class_overrides=fw.get("class_overrides", {}),
            keep_upstream=fw.get("keep_upstream", []),
            sdk_framework=fw.get("sdk_framework", ""),
            output_subdir=fw.get("output_subdir", ""),
            extra_umbrella_includes=fw.get("extra_umbrella_includes", []),
            shared=fw.get("shared", False),
        ))

    return sdks, frameworks


def get_sdk_path(xcrun_sdk: str) -> Path:
    result = subprocess.run(
        ["xcrun", "--sdk", xcrun_sdk, "--show-sdk-path"],
        capture_output=True, text=True, check=True,
    )
    return Path(result.stdout.strip())


# ── Parsed ObjC framework index ───────────────────────────────────────────

ParsedFrameworkData = list[tuple[str, FrameworkData]]


def framework_headers_dir(fw: FrameworkConfig, sdk_path: Path) -> Path:
    sdk_fw_name = fw.sdk_framework or fw.name
    return (
        sdk_path / "System" / "Library" / "Frameworks"
        / f"{sdk_fw_name}.framework" / "Headers"
    )


def parse_objc_framework_headers(
    fw: FrameworkConfig,
    sdk_path: Path,
) -> ParsedFrameworkData:
    """Parse every configured ObjC header for a framework once.

    The returned data feeds both the SDK-wide symbol index and the later
    emitter. This keeps cross-framework type discovery based on libclang's
    view of the headers rather than a regex pre-scan.
    """
    fw_headers_dir = framework_headers_dir(fw, sdk_path)
    header_jobs: list[tuple[str, Path]] = []
    for header_name in fw.headers:
        hp = fw_headers_dir / header_name
        if not hp.exists():
            log.warning("    Header not found: %s", hp)
            continue
        header_jobs.append((header_name, hp))

    import concurrent.futures as _cf
    n_workers = max(1, min(len(header_jobs), os.cpu_count() or 1))
    parsed_data: list[FrameworkData]
    if n_workers > 1 and len(header_jobs) > 1:
        with _cf.ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_parse_worker_init,
            initargs=(str(sdk_path),),
        ) as ex:
            parsed_data = list(ex.map(
                _parse_worker,
                [str(hp) for _, hp in header_jobs],
            ))
    else:
        objc_parser = ObjCParser(sdk_path)
        parsed_data = [objc_parser.parse_header(hp) for _, hp in header_jobs]

    parsed: ParsedFrameworkData = []
    for (header_name, _), data in zip(header_jobs, parsed_data):
        log.info("    Parsed: %s", header_name)
        parsed.append((header_name, data))
    return parsed


def register_objc_declarations(
    fw: FrameworkConfig,
    parsed: ParsedFrameworkData,
    resolver: TypeResolver,
) -> None:
    """Register classes, enums, blocks, structs, and simple aliases from
    already-parsed framework data.

    This is the SDK-wide symbol-index pass. It runs for every ObjC framework
    before any header is emitted, so generated signatures can resolve
    cross-framework and later-framework references without regex scanning.
    """
    for header_name, data in parsed:
        source_base = header_name.removesuffix(".h")
        cpp_name = strip_objc_prefix(source_base, fw.strip_prefix)
        resolver.register(source_base, f"{fw.namespace}::{cpp_name}",
                          kind="class", framework_prefix=fw.prefix)

        for enum in data.enums:
            if not enum.name:
                continue
            cpp_name = strip_objc_prefix(enum.name, fw.strip_prefix)
            cpp_qual = f"{fw.namespace}::{cpp_name}"
            resolver.register(enum.name, cpp_qual,
                              kind="enum", framework_prefix=fw.prefix)
            resolver.cpp_to_source[cpp_qual] = source_base
            underlying = (resolver.resolve(enum.underlying_type)
                          if enum.underlying_type else "NS::UInteger")
            resolver.enum_underlying[cpp_qual] = underlying
            resolver.enum_is_options[cpp_qual] = enum.is_options

        for block in data.blocks:
            cpp_name = strip_objc_prefix(block.name, fw.strip_prefix)
            resolver.register(block.name, f"{fw.namespace}::{cpp_name}",
                              kind="block", framework_prefix=fw.prefix)

        for s in data.structs:
            override = _override_for(fw, s.name)
            if override.get("cpp_name"):
                cpp_qual = override["cpp_name"]
            else:
                cpp_name = strip_objc_prefix_aggressive(s.name.lstrip("_"), fw.strip_prefix)
                cpp_qual = f"{fw.namespace}::{cpp_name}"
            resolver.register(s.name, cpp_qual,
                              kind="struct", framework_prefix=fw.prefix)

        for alias_name, underlying in data.primitive_aliases:
            alias_cpp = strip_objc_prefix(alias_name, fw.strip_prefix)
            alias_cpp_qual = f"{fw.namespace}::{alias_cpp}"
            if resolver.kinds.get(alias_cpp_qual) == "enum":
                continue
            resolved_underlying = resolver.resolve(underlying)
            if resolved_underlying.startswith("void"):
                continue
            resolver.register(alias_name, alias_cpp_qual,
                              kind="typedef", framework_prefix=fw.prefix)
            resolver.alias_underlying[alias_cpp_qual] = resolved_underlying

        for cls in data.classes:
            cpp_name = strip_objc_prefix(cls.name, fw.strip_prefix)
            resolver.register(cls.name, f"{fw.namespace}::{cpp_name}",
                              kind="class", framework_prefix=fw.prefix)
            resolver.class_to_source[cls.name] = source_base


def register_objc_aliases(
    fw: FrameworkConfig,
    parsed: ParsedFrameworkData,
    resolver: TypeResolver,
) -> None:
    """Register aliases whose target may live in another header/framework.

    This runs after `register_objc_declarations` has populated base classes
    and structs for the whole SDK.
    """
    for _, data in parsed:
        for alias_name, target_name in data.struct_aliases:
            target_cpp_qual = resolver.type_map.get(target_name)
            if not target_cpp_qual:
                continue
            alias_cpp = strip_objc_prefix(alias_name, fw.strip_prefix)
            alias_cpp_qual = f"{fw.namespace}::{alias_cpp}"
            resolver.register(alias_name, alias_cpp_qual,
                              kind="typedef", framework_prefix=fw.prefix)
            resolver.alias_underlying[alias_cpp_qual] = target_cpp_qual

        for alias_name, target_class in data.class_pointer_aliases:
            target_cpp_qual = resolver.type_map.get(target_class)
            if not target_cpp_qual:
                continue
            target_base = target_cpp_qual.rstrip("*").strip()
            target_with_ptr = f"{target_base}*"
            alias_cpp = strip_objc_prefix(alias_name, fw.strip_prefix)
            alias_cpp_qual = f"{fw.namespace}::{alias_cpp}"
            resolver.register(alias_name, alias_cpp_qual,
                              kind="typedef", framework_prefix=fw.prefix)
            resolver.alias_underlying[alias_cpp_qual] = target_with_ptr


def _ver_tuple(v: str) -> Optional[tuple[int, ...]]:
    try:
        return tuple(int(x) for x in v.split("."))
    except ValueError:
        return None


def _introduced_earlier(child: Availability, parent: Availability) -> bool:
    """True iff `child` is introduced strictly earlier than `parent` on at
    least one shared platform."""
    if not child.platforms or not parent.platforms:
        return False
    for plat, (c_intro, _, _, c_unavail) in child.platforms.items():
        if c_unavail or not c_intro:
            continue
        p_entry = parent.platforms.get(plat)
        if not p_entry:
            continue
        p_intro, _, _, p_unavail = p_entry
        if p_unavail or not p_intro:
            continue
        c_tup, p_tup = _ver_tuple(c_intro), _ver_tuple(p_intro)
        if c_tup is None or p_tup is None:
            continue
        if c_tup < p_tup:
            return True
    return False


def suppress_late_base_availability(
    parsed_objc: dict[str, ParsedFrameworkData],
) -> None:
    """When a class C has a subclass introduced on an earlier OS than C
    itself (Apple's pattern of retro-fitting a new base onto a long-lived
    protocol — e.g. `MTLAllocation` added in macOS 15 as a base for
    `MTLRenderPipelineState` from macOS 10.11), clear C's class-level
    availability.

    The C++ language limitation: there's no syntax for "this subclass
    inherits from this base only when the OS is new enough." Inheritance
    is fully resolved at the subclass's declaration site, so the older
    subclass's declaration unconditionally references the newer base —
    which then trips -Wunguarded-availability-new because the deployment
    target is older than the base's `introduced=`. ObjC's protocol-
    conformance model resolves this at runtime; C++ can't. The only
    workable answer is to drop the type-declaration availability on the
    base. C's own methods keep their per-decl availability, so calls to
    base methods on older OSes still get the correct warning."""
    by_name: dict[str, ObjCClass] = {}
    for parsed in parsed_objc.values():
        for _, data in parsed:
            for cls in data.classes:
                by_name[cls.name] = cls
    for parent in by_name.values():
        kids = [c for c in by_name.values() if c.superclass == parent.name]
        if any(_introduced_earlier(k.availability, parent.availability)
               for k in kids):
            parent.availability = Availability()


def collect_generator_declarations(
    fw: FrameworkConfig,
    parsed: ParsedFrameworkData,
    resolver: TypeResolver,
    gen: CodeGenerator,
) -> None:
    """Populate a framework generator from parsed data already registered in
    the SDK-wide resolver."""
    packed_overrides = {
        name for name, ovr in fw.class_overrides.items()
        if isinstance(ovr, dict) and ovr.get("packed")
    }

    for header_name, data in parsed:
        source_base = header_name.removesuffix(".h")

        for enum in data.enums:
            gen.collect_enum(enum)

        for block in data.blocks:
            gen.collect_block(block)

        for s in data.structs:
            override = _override_for(fw, s.name)
            if override.get("skip"):
                continue
            if s.name in packed_overrides or s.name.lstrip("_") in packed_overrides:
                s.packed = True
            if extra := override.get("members"):
                s.extra_members = extra
            gen.collect_struct(s)

        for alias_name, target_name in data.struct_aliases:
            target_cpp_qual = resolver.type_map.get(target_name)
            if not target_cpp_qual:
                continue
            alias_cpp = strip_objc_prefix(alias_name, fw.strip_prefix)
            gen.collect_struct_alias(alias_cpp, target_cpp_qual)

        for alias_name, target_class in data.class_pointer_aliases:
            target_cpp_qual = resolver.type_map.get(target_class)
            if not target_cpp_qual:
                continue
            target_base = target_cpp_qual.rstrip("*").strip()
            alias_cpp = strip_objc_prefix(alias_name, fw.strip_prefix)
            gen.collect_struct_alias(alias_cpp, f"{target_base}*")

        for alias_name, underlying in data.primitive_aliases:
            alias_cpp = strip_objc_prefix(alias_name, fw.strip_prefix)
            alias_cpp_qual = f"{fw.namespace}::{alias_cpp}"
            if resolver.kinds.get(alias_cpp_qual) != "typedef":
                continue
            resolved_underlying = resolver.alias_underlying.get(alias_cpp_qual)
            if not resolved_underlying:
                resolved_underlying = resolver.resolve(underlying)
            if resolved_underlying.startswith("void"):
                continue
            gen.collect_struct_alias(alias_cpp, resolved_underlying)

        for cls in data.classes:
            gen.class_to_source[cls.name] = source_base


# ── Framework processing ──────────────────────────────────────────────────

def process_objc_framework(
    fw: FrameworkConfig,
    sdk_path: Path,
    parsed: ParsedFrameworkData,
    fw_output: Path,
    resolver: TypeResolver,
    output_dir: Path,
    metal_cpp_dir: Path,
    dir_expected: dict[Path, set[str]],
    emit_availability_types: bool = False,
    emit_availability_members: bool = False,
) -> None:
    """Process an ObjC framework: parse headers and overwrite per-class hpps in
    a metal-cpp-shaped tree. Hand-written hpps listed under `keep_upstream:`
    are copied verbatim from `metal_cpp_dir/<fw>/` so generated wrappers can
    reuse them.
    """
    expected = dir_expected.setdefault(fw_output, set())
    expected.add(f"{fw.name}.hpp")  # umbrella
    expected.add(f"{fw.prefix}Defines.hpp")
    # Hand-written supplements pulled into the umbrella (e.g.
    # `MTLDeviceExtras.hpp`). The user maintains these; the sweep must
    # not touch them.
    for extra in fw.extra_umbrella_includes:
        expected.add(extra)

    # Copy upstream-preserved infrastructure / hand-written hpps.
    if fw.keep_upstream:
        src_dir = metal_cpp_dir / fw.name
        for name in fw.keep_upstream:
            src = src_dir / f"{name}.hpp"
            if not src.exists():
                log.warning("    keep_upstream miss: %s", src)
                continue
            shutil.copyfile(src, fw_output / src.name)
            expected.add(src.name)
            log.info("    keep: %s", src.name)

    gen = CodeGenerator(
        namespace=fw.namespace,
        prefix=fw.prefix,
        strip_prefix=fw.strip_prefix,
        resolver=resolver,
    )
    gen.emit_availability_types = emit_availability_types
    gen.emit_availability_members = emit_availability_members

    # Seed the umbrella with the upstream-kept files so a single
    # `<Framework>.hpp` include surfaces NS::SharedPtr / NS::TransferPtr /
    # NS::Range etc. without the user having to know which header owns them.
    for name in fw.keep_upstream:
        gen.generated_headers.append(f"{name}.hpp")
    # Also tell <P>Structs.hpp which upstream files to chain so a generated
    # class that uses a skipped struct (MTL::BufferRange, MTL::PackedFloat3)
    # only needs to include <P>Structs.hpp.
    gen.keep_upstream = list(fw.keep_upstream)
    gen.extra_umbrella_includes = list(fw.extra_umbrella_includes)
    gen.output_subdir = fw.output_subdir or fw.name

    # Optional <Framework>.apinotes file ships next to the SDK headers and
    # carries Apple's canonical Swift-style short names (the same source Apple
    # uses to expose `setBlendColor` for `setBlendColorRed:green:blue:alpha:`).
    # Honor renames from the framework whose headers we're parsing rather than
    # the framework name itself — a virtual framework (Metal4) shares
    # Metal.apinotes.
    sdk_fw_name = fw.sdk_framework or fw.name
    apinotes_path = (
        sdk_path / "System" / "Library" / "Frameworks"
        / f"{sdk_fw_name}.framework" / "Headers" / f"{sdk_fw_name}.apinotes"
    )
    apinotes_renames: dict[str, dict[str, str]] = (
        parse_apinotes_renames(apinotes_path) if apinotes_path.exists() else {}
    )
    if apinotes_renames:
        log.info("    apinotes renames: %d classes",
                 len(apinotes_renames))

    collect_generator_declarations(fw, parsed, resolver, gen)

    # Write per-framework Enums.hpp + Blocks.hpp before classes (class
    # headers include both).
    # Emit a generated <Prefix>Defines.hpp when no upstream-kept one exists
    # — virtual frameworks like Metal4 share Metal/ but need their own
    # macro aliases (_MTL4_INLINE, _MTL4_ENUM, ...).
    if fw.prefix not in {n for n in fw.keep_upstream if n.endswith("Defines")} and \
            f"{fw.prefix}Defines" not in fw.keep_upstream:
        defines_path = fw_output / f"{fw.prefix}Defines.hpp"
        if not defines_path.exists():
            defines_path.write_text(gen.generate_defines())
            log.info("    -> %s", defines_path.relative_to(output_dir))
    # Skip — and delete any stale copy of — empty aggregates. Some frameworks
    # (MTL4FX, CA) have no blocks/structs at all; emitting a header with
    # only `#pragma once` and an empty namespace is dead weight.
    # Enums no longer have a per-framework aggregate — each enum is
    # emitted into the per-source-header hpp that declared it (see the
    # `generate_class_header(... enums=data.enums)` call below).
    for kind, has, emit in (
        ("Blocks", gen.has_blocks, gen.generate_blocks),
        ("Structs", gen.has_structs, gen.generate_structs),
    ):
        path = fw_output / f"{fw.prefix}{kind}.hpp"
        if has:
            path.write_text(emit())
            expected.add(path.name)
            log.info("    -> %s", path.relative_to(output_dir))
        elif path.exists():
            path.unlink()

    # Foundation: emit the CRTP root (NS::Object + Referencing/Copying/
    # SecureCoding templates) and the templated NS::Enumerator container
    # ourselves rather than copying Apple's `_NS_PRIVATE_SEL`-driven
    # versions. The generated forms route through the same
    # `_objc_msgSend$<sel>` stub trampolines every other wrapper uses;
    # `sendMessage` / `sendMessageSafe` are kept as escape hatches.
    if fw.prefix == "NS":
        for name, emit in (
            ("NSObject.hpp", gen.generate_nsobject_header),
            ("NSEnumerator.hpp", gen.generate_nsenumerator_header),
        ):
            path = fw_output / name
            path.write_text(emit())
            gen.generated_headers.append(name)
            expected.add(name)
            log.info("    -> %s", path.relative_to(output_dir))

    def _write_umbrella():
        umbrella_path = fw_output / f"{fw.name}.hpp"
        umbrella_path.write_text(gen.generate_umbrella(fw.name))
        log.info("    -> %s", umbrella_path.relative_to(output_dir))

    # Pass 2: emit class headers.
    # Skip protocols/classes that collide with upstream NSObject.hpp's CRTP
    # bases (NSCopying, NSSecureCoding, etc.) — those are the dispatch
    # templates we inherit from.
    UPSTREAM_RESERVED = {
        "NSObject", "NSCopying", "NSSecureCoding", "NSMutableCopying",
        "NSCoding", "NSFastEnumeration", "NSLocking", "NSDiscardableContent",
    }
    # Group filtered classes by their source SDK header so each output .hpp
    # mirrors upstream's layout (MTLEvent.h → MTLEvent.hpp containing every
    # @interface / @protocol the SDK packed in there).
    for header_name, data in parsed:
        group: list[tuple[ObjCClass, Optional[dict]]] = []
        for cls in data.classes:
            if cls.name in UPSTREAM_RESERVED:
                continue
            inc_classes = fw.include.get("classes", [".*"])
            if not matches_any(cls.name, inc_classes):
                continue
            exc_classes = fw.exclude.get("classes", [])
            if exc_classes and matches_any(cls.name, exc_classes):
                continue
            override = fw.class_overrides.get(cls.name)
            filtered = filter_class(
                cls, fw.exclude, override,
                apinotes_renames=apinotes_renames.get(cls.name),
            )
            if not filtered:
                continue
            log.info("      Class: %s (%d props, %d methods)",
                     filtered.name,
                     len(filtered.properties),
                     len(filtered.methods))
            gen.collect_selectors(filtered)
            group.append((filtered, override))

        # Emit a per-source-header `.hpp` whenever the SDK header
        # contributed anything we surface: a class, an extern global, a
        # string-typedef alias, or an enum (the new per-header enum
        # emission means enum-only SDK headers like `MTLDataType.h` now
        # produce their own `.hpp`).
        if (not group and not data.constants and not data.string_typedefs
                and not data.enums):
            continue

        out_name = header_name.removesuffix(".h") + ".hpp"
        header_content = gen.generate_class_header(
            group, header_basename=out_name.removesuffix(".hpp"),
            constants=data.constants, string_typedefs=data.string_typedefs,
            enums=data.enums)
        out_path = fw_output / out_name
        out_path.write_text(header_content)
        gen.generated_headers.append(out_name)
        expected.add(out_name)
        log.info("      -> %s", out_path.relative_to(output_dir))

    # Bridge: every per-class header `#include`s `<P>Bridge.hpp`, which
    # collapses identical (return, args, selector) trampolines into one
    # extern "C" decl. Emitted after every class header so the registry
    # has seen every signature it needs to declare.
    bridge_path = fw_output / f"{fw.prefix}Bridge.hpp"
    bridge_path.write_text(gen.generate_bridge_header())
    expected.add(bridge_path.name)
    calls = gen._bridge_call_count
    uniq = len(gen._bridge_entries)
    ratio = (calls / uniq) if uniq else 0.0
    log.info("    -> %s (%d call sites → %d trampolines, %.1fx dedup)",
             bridge_path.relative_to(output_dir), calls, uniq, ratio)

    _write_umbrella()


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate metal-cpp style C++ wrappers for ObjC frameworks")
    parser.add_argument("--sdk", help="Generate for a single SDK (e.g. macos)")
    parser.add_argument("--strict", action="store_true",
                        help="Fail on unresolvable types")
    parser.add_argument("--config", type=Path, default=TOOL_DIR / "config.yaml")
    parser.add_argument("--output", type=Path, default=ROOT_DIR,
                        help="Output directory for generated files (metal-cpp root)")
    parser.add_argument("--metal-cpp", type=Path, required=True,
                        help="Path to Apple's metal-cpp checkout. Files "
                             "listed under each framework's `keep_upstream:` "
                             "are copied verbatim from this tree.")
    parser.add_argument("--api-availability-types",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Emit `API_AVAILABLE(...)` on type declarations "
                             "(classes, enums, structs). Default: off.")
    parser.add_argument("--api-availability-members",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Emit `API_AVAILABLE(...)` on type members "
                             "(methods, properties, enum values). Default: off.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    output_dir = args.output.resolve()
    metal_cpp_dir = args.metal_cpp.resolve()

    sdks, frameworks = load_config(args.config)

    if args.sdk:
        sdks = [s for s in sdks if s.name == args.sdk]
        if not sdks:
            log.error("Unknown SDK: %s", args.sdk)
            sys.exit(1)

    processed_shared: set[str] = set()
    # Per-output-dir expected file set, collected across every framework
    # emission. After the SDK loop, anything in the dir that's not in this
    # set is a stale leftover (typically files that used to be in
    # `keep_upstream:` and got removed) and gets unlinked.
    dir_expected: dict[Path, set[str]] = {}

    for sdk in sdks:
        # Skip SDKs whose every applicable framework was already emitted by
        # an earlier SDK (everything is `shared: true`). Pass 1 type
        # registration would still run otherwise — re-parsing every header
        # for no emission gain.
        applicable = [fw for fw in frameworks if sdk.name in fw.sdks]
        if applicable and all(fw.shared and fw.name in processed_shared
                              for fw in applicable):
            log.info("Skipping SDK %s: all applicable frameworks already emitted", sdk.name)
            continue

        log.info("Processing SDK: %s", sdk.name)
        sdk_path = get_sdk_path(sdk.xcrun_sdk)
        log.info("  SDK path: %s", sdk_path)

        resolver = TypeResolver()

        # Pass 1: parse ObjC headers for every applicable framework once,
        # then build a SDK-wide type index from that libclang data before any
        # framework emits. Cross-framework references (e.g. Metal methods
        # naming Metal4 classes/enums) resolve without relying on regex header
        # scans or framework ordering.
        parsed_objc: dict[str, ParsedFrameworkData] = {}
        for fw in applicable:
            log.info("  Parsing framework: %s", fw.name)
            parsed_objc[fw.name] = parse_objc_framework_headers(fw, sdk_path)

        for fw in applicable:
            register_objc_declarations(fw, parsed_objc[fw.name], resolver)
        for fw in applicable:
            register_objc_aliases(fw, parsed_objc[fw.name], resolver)
        suppress_late_base_availability(parsed_objc)

        for fw in applicable:
            log.info("  Framework: %s", fw.name)

            # Flat layout: <metal-cpp>/<framework>/
            fw_output = output_dir / (fw.output_subdir or fw.name)
            fw_output.mkdir(parents=True, exist_ok=True)

            if not (fw.shared and fw.name in processed_shared):
                process_objc_framework(
                    fw, sdk_path, parsed_objc[fw.name],
                    fw_output, resolver,
                    output_dir, metal_cpp_dir,
                    dir_expected,
                    emit_availability_types=args.api_availability_types,
                    emit_availability_members=args.api_availability_members,
                )
                if fw.shared:
                    processed_shared.add(fw.name)

        # Report unresolvable types
        if resolver.unresolved:
            log.warning("  Unresolvable types in %s:", sdk.name)
            for typ, ctx in resolver.unresolved:
                log.warning("    %s (in %s)", typ, ctx)
            if args.strict:
                log.error("Strict mode: failing due to unresolvable types")
                sys.exit(1)

    # Sweep stale files. Any `.hpp` in an output directory that isn't part
    # of the current expected set is a leftover from a previous run (e.g.
    # a class removed from `headers:` or a file moved out of
    # `keep_upstream:`). Removing it now prevents includes from drifting
    # apart from what the generator owns.
    for out_dir, names in dir_expected.items():
        if not out_dir.is_dir():
            continue
        for path in out_dir.glob("*.hpp"):
            if path.name not in names:
                log.info("    sweep stale: %s", path.relative_to(output_dir))
                path.unlink()

    log.info("Done.")


if __name__ == "__main__":
    main()
