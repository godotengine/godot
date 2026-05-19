"""Shared types, parsing, and code generation for metal-cpp tooling.

Contains ObjC header parsing (libclang), type resolution, data classes,
and C++ code generation following the metal-cpp patterns used by Apple.
Used by generate.py to produce the full metal-cpp tree from SDK headers.
"""

from __future__ import annotations

import ctypes
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import clang.cindex as _cc
from clang.cindex import CursorKind, Index, TranslationUnit, TypeKind

log = logging.getLogger(__name__)

# ── Availability extraction via ctypes ────────────────────────────────────
# The Python libclang binding doesn't expose clang_getCursorPlatformAvailability;
# we call it directly. Layout matches the CXPlatformAvailability struct from
# Index.h.

# Raw CXString — same memory layout as cc._CXString but without the __del__
# that calls clang_disposeString. Using the binding's _CXString inside a
# Structure causes Python to dispose the strings out of order vs the
# clang_disposeCXPlatformAvailability call, which can double-free.

class _CXStringRaw(ctypes.Structure):
    _fields_ = [("spelling", ctypes.c_char_p),
                ("free", ctypes.c_int)]


class _CXVersion(ctypes.Structure):
    _fields_ = [("Major", ctypes.c_int),
                ("Minor", ctypes.c_int),
                ("Subminor", ctypes.c_int)]


class _CXPlatformAvailability(ctypes.Structure):
    _fields_ = [("Platform", _CXStringRaw),
                ("Introduced", _CXVersion),
                ("Deprecated", _CXVersion),
                ("Obsoleted", _CXVersion),
                ("Unavailable", ctypes.c_int),
                ("Message", _CXStringRaw)]


_cc.conf.lib.clang_getCursorPlatformAvailability.argtypes = [
    _cc.Cursor,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(_CXStringRaw),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(_CXStringRaw),
    ctypes.POINTER(_CXPlatformAvailability),
    ctypes.c_int,
]
_cc.conf.lib.clang_getCursorPlatformAvailability.restype = ctypes.c_int
_cc.conf.lib.clang_disposeCXPlatformAvailability.argtypes = [ctypes.POINTER(_CXPlatformAvailability)]
_cc.conf.lib.clang_disposeCXPlatformAvailability.restype = None

# SDK platform name → C++ attribute platform name. The clang attribute uses
# the same identifiers as the SDK's API_AVAILABLE macro.
_PLAT_ALIAS = {
    "ios": "ios",
    "ios_app_extension": "ios",
    "macos": "macos",
    "macos_app_extension": "macos",
    "macosx": "macos",
    "tvos": "tvos",
    "watchos": "watchos",
    "visionos": "visionos",
}


@dataclass
class Availability:
    """Per-platform availability. Each entry is
    (introduced, deprecated, obsoleted, unavailable).
    `unavailable=True` (or all-None versions) means the API isn't available
    on that platform — Apple's `API_AVAILABLE(macos(...))` expands to
    unavailable attributes for every other platform."""
    platforms: dict[str, tuple[Optional[str], Optional[str], Optional[str], bool]] = field(default_factory=dict)
    unavailable: bool = False


def _ver(v: _CXVersion) -> Optional[str]:
    if v.Major < 0:
        return None
    parts = [str(v.Major)]
    if v.Minor >= 0:
        parts.append(str(v.Minor))
    if v.Subminor > 0:
        parts.append(str(v.Subminor))
    return ".".join(parts)


def cursor_availability(cursor) -> Availability:
    """Pull platform availability for a cursor. Returns empty Availability
    when the cursor has no annotations."""
    arr = (_CXPlatformAvailability * 8)()
    always_dep = ctypes.c_int()
    dep_msg = _CXStringRaw()
    always_unavail = ctypes.c_int()
    unavail_msg = _CXStringRaw()
    n = _cc.conf.lib.clang_getCursorPlatformAvailability(
        cursor,
        ctypes.byref(always_dep), ctypes.byref(dep_msg),
        ctypes.byref(always_unavail), ctypes.byref(unavail_msg),
        arr, 8,
    )
    av = Availability(unavailable=bool(always_unavail.value))
    try:
        for j in range(n):
            plat_raw = arr[j].Platform.spelling.decode("utf-8") if arr[j].Platform.spelling else ""
            plat = _PLAT_ALIAS.get(plat_raw, plat_raw)
            if not plat:
                continue
            intro = _ver(arr[j].Introduced)
            dep = _ver(arr[j].Deprecated)
            obs = _ver(arr[j].Obsoleted)
            unavail = bool(arr[j].Unavailable)
            # Apple's API_AVAILABLE(macos(...)) makes libclang report every
            # other platform with all-None versions — that's the "excluded"
            # signal. Treat it as unavailable.
            if not unavail and intro is None and dep is None and obs is None:
                unavail = True
            av.platforms[plat] = (intro, dep, obs, unavail)
    finally:
        # libclang owns the CXString backing buffers for entries 0..n-1; free
        # via the proper dispose function. We don't dispose the standalone
        # dep_msg / unavail_msg CXStrings here — those use the binding's
        # type marshaling and a separate dispose path, and we don't read them.
        for j in range(n):
            _cc.conf.lib.clang_disposeCXPlatformAvailability(ctypes.byref(arr[j]))
    return av


# Standard Apple platforms relevant to Metal-consuming code. Any platform in
# this set that's NOT explicitly introduced for a given API is treated as
# unavailable — matching Apple's `API_AVAILABLE(...)` semantics, where
# unmentioned platforms are excluded rather than inherited.
# `maccatalyst` is intentionally omitted: it has no `__API_AVAILABLE_PLATFORM_*`
# entry in `<os/availability.h>`, so the macro expansion fails to parse. The iOS
# version bound already covers Mac Catalyst targets in practice.
_STANDARD_PLATFORMS = ("ios", "macos", "tvos", "visionos")


def format_availability(av: Availability) -> str:
    """Render an Availability as Apple `<Availability.h>` macros —
    `API_AVAILABLE(macos(10.13), ios(11.0))`, `API_UNAVAILABLE(tvos)`,
    `API_DEPRECATED("", macos(10.13, 12.0))`. Empty when there's nothing
    to emit.

    Only platforms libclang explicitly enumerated for the cursor are emitted.
    Apple's convention: when a declaration has its own API_AVAILABLE, libclang
    enumerates every platform (with all-None for excluded ones, surfaced here
    as `unavailable`). When a declaration inherits from its enclosing context
    (no own attribute), libclang reports only the platforms the parent
    mentioned — missing platforms inherit too (e.g. visionOS inheriting iOS
    availability for CAMetalLayer.colorspace), so we must NOT synthesize
    `unavailable` for them."""
    if av.unavailable:
        return "__attribute__((unavailable))"
    if not av.platforms:
        return ""
    introduced: list[str] = []
    deprecated: list[str] = []
    unavail: list[str] = []
    for plat in sorted(av.platforms):
        if plat not in _STANDARD_PLATFORMS:
            continue
        intro, dep, obs, plat_unavail = av.platforms[plat]
        if plat_unavail:
            unavail.append(plat)
            continue
        # `obsoleted` is rarer than `deprecated` and stricter (removed,
        # not just discouraged). Fold it into API_DEPRECATED's end-version
        # slot — `<Availability.h>` has no obsoleted-specific macro.
        end_ver = dep or obs
        if intro and end_ver:
            deprecated.append(f"{plat}({intro}, {end_ver})")
        elif intro:
            introduced.append(f"{plat}({intro})")
    parts: list[str] = []
    if introduced:
        parts.append(f"API_AVAILABLE({', '.join(introduced)})")
    if deprecated:
        parts.append(f'API_DEPRECATED("", {", ".join(deprecated)})')
    if unavail:
        parts.append(f"API_UNAVAILABLE({', '.join(unavail)})")
    return " ".join(parts)

# ── Built-in type map (types already in metal-cpp) ───────────────────────

BUILTIN_TYPE_MAP: dict[str, str] = {
    # Foundation
    "NSObject": "NS::Object",
    "NSString": "NS::String",
    "NSError": "NS::Error",
    "NSArray": "NS::Array",
    "NSDictionary": "NS::Dictionary",
    "NSNumber": "NS::Number",
    "NSURL": "NS::URL",
    "NSBundle": "NS::Bundle",
    "NSData": "NS::Data",
    "NSValue": "NS::Value",
    "NSSet": "NS::Set",
    "NSUInteger": "NS::UInteger",
    "NSInteger": "NS::Integer",
    "NSTimeInterval": "NS::TimeInterval",
    "BOOL": "bool",
    "NSRect": "CGRect",
    "NSSize": "CGSize",
    "NSPoint": "CGPoint",
    # Common Foundation typedefs
    "NSStringEncoding": "NS::UInteger",
    "unichar": "unsigned short",
    "NSComparator": "void*",
    "NSComparisonResult": "long",
    "NSRange": "NS::Range",
    "NSErrorDomain": "NS::String*",
    "NSErrorUserInfoKey": "NS::String*",
    "NSNotificationName": "NS::String*",
    "NSExceptionName": "NS::String*",
    "NSRunLoopMode": "NS::String*",
    # Metal
    "MTLDevice": "MTL::Device",
    "MTLTexture": "MTL::Texture",
    "MTLBuffer": "MTL::Buffer",
    "MTLLibrary": "MTL::Library",
    "MTLCommandQueue": "MTL::CommandQueue",
    "MTLCommandBuffer": "MTL::CommandBuffer",
    "MTLRenderPipelineState": "MTL::RenderPipelineState",
    "MTLComputePipelineState": "MTL::ComputePipelineState",
    "MTLPixelFormat": "MTL::PixelFormat",
    "MTLResourceOptions": "MTL::ResourceOptions",
    "MTLResidencySet": "MTL::ResidencySet",
    "MTLGPUAddress": "MTL::GPUAddress",
    # QuartzCore
    "CAMetalLayer": "CA::MetalLayer",
    "CAMetalDrawable": "CA::MetalDrawable",
}

# Types that pass through unchanged (C, CoreFoundation, etc.)
PASSTHROUGH_TYPES: set[str] = {
    "void",
    "bool",
    "float",
    "double",
    "char",
    "short",
    "int",
    "long",
    "unsigned char",
    "unsigned short",
    "unsigned int",
    "unsigned long",
    "long long",
    "unsigned long long",
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
    "size_t",
    "ssize_t",
    "ptrdiff_t",
    "CGRect",
    "CGSize",
    "CGPoint",
    "CGFloat",
    "CGColorSpaceRef",
    "CGDirectDisplayID",
    "CGColorRef",
    "CGAffineTransform",
    "CFStringRef",
    "CFTypeRef",
    "CFTimeInterval",
    "dispatch_queue_t",
    "dispatch_data_t",
    "IOSurfaceRef",
    "SEL",
    "Class",
}

# Resolved type → system header needed
SYSTEM_HEADER_FOR_TYPE: dict[str, str] = {
    "CGRect": "CoreGraphics/CoreGraphics.h",
    "CGSize": "CoreGraphics/CoreGraphics.h",
    "CGPoint": "CoreGraphics/CoreGraphics.h",
    "CGFloat": "CoreGraphics/CoreGraphics.h",
    "CGColorSpaceRef": "CoreGraphics/CoreGraphics.h",
    "CGDirectDisplayID": "CoreGraphics/CoreGraphics.h",
    "CGColorRef": "CoreGraphics/CoreGraphics.h",
    "CGAffineTransform": "CoreGraphics/CoreGraphics.h",
    "IOSurfaceRef": "IOSurface/IOSurfaceRef.h",
}


# ── Shared helpers ────────────────────────────────────────────────────────


def setter_name(prop_name: str) -> str:
    """ObjC property name → setter name (e.g. 'foo' → 'setFoo')."""
    return f"set{prop_name[0].upper()}{prop_name[1:]}"


def cpp_method_name_from_first_segment(first: str) -> str:
    """Apply upstream metal-cpp's acronym-prefix lower-casing rule to a
    method or property name: `UTF8String` → `utf8String`. Leaves
    camelcase-into-camelcase names like `URLWithString` alone (upstream
    keeps them as-is)."""
    if not first or not first[0].isupper():
        return first
    m = re.match(r"^([A-Z]{2,})(?=\d)", first)
    if m:
        run = m.group(1)
        return run.lower() + first[len(run):]
    return first


def selector_accessor(selector: str) -> str:
    """ObjC selector → Private.hpp accessor (colons become underscores)."""
    return selector.replace(":", "_")


def strip_objc_prefix(name: str, strip_prefix: str) -> str:
    """Strip ObjC prefix to get C++ name (e.g. 'CAMetalLayer' → 'MetalLayer').

    Skips stripping when the result would start with a digit — that happens
    for Apple's MTL4* class line (MTL4CommandQueue, MTL4ComputePipeline, ...).
    Those keep the full name and live as MTL::MTL4CommandQueue in C++.
    """
    if strip_prefix and name.startswith(strip_prefix):
        stripped = name[len(strip_prefix) :]
        if stripped and not stripped[0].isdigit():
            return stripped
    return name


def strip_objc_prefix_aggressive(name: str, strip_prefix: str) -> str:
    """Stripping for structs, which match upstream metal-cpp's convention of
    dropping both the prefix AND any version digits ('MTL4BufferRange' →
    'BufferRange'). Classes keep the digits to avoid collisions with their
    versionless counterparts (MTL4CommandQueue vs MTLCommandQueue protocol)."""
    if not strip_prefix or not name.startswith(strip_prefix):
        return name
    rest = name[len(strip_prefix):]
    # Eat trailing digits (handles MTL4Foo, NS5Foo, etc.)
    while rest and rest[0].isdigit():
        rest = rest[1:]
    if rest and rest[0].isupper():
        return rest
    return name


def resolve_type(
    resolver: TypeResolver,
    objc_type: str,
    namespace: str,
    cpp_class_name: str,
) -> str:
    """Resolve an ObjC type, handling instancetype → concrete class pointer."""
    cpp = resolver.resolve(objc_type, cpp_class_name)
    if cpp == "__instancetype__":
        return f"{namespace}::{cpp_class_name}*"
    return cpp


# ── Data classes ──────────────────────────────────────────────────────────


@dataclass
class ObjCProperty:
    name: str
    objc_type: str
    is_readonly: bool
    is_class_property: bool = False
    availability: Availability = field(default_factory=Availability)


@dataclass
class ObjCParam:
    name: str
    objc_type: str


@dataclass
class ObjCMethod:
    selector: str
    return_type: str
    params: list[ObjCParam] = field(default_factory=list)
    is_class_method: bool = False
    availability: Availability = field(default_factory=Availability)
    # When non-empty, the C++ method name to emit instead of the value
    # computed from the selector. Used by class_overrides.<Cls>.rename
    # to handle Apple-specific selector groupings (e.g.
    # `setBlendColorRed:green:blue:alpha:` → `setBlendColor`).
    cpp_name_override: str = ""

    @property
    def sel_accessor(self) -> str:
        """Selector accessor for _PRIVATE_DEF_SEL: colons replaced with underscores."""
        return selector_accessor(self.selector)

    @property
    def cpp_name(self) -> str:
        if self.cpp_name_override:
            return self.cpp_name_override
        return self._compute_cpp_name()

    def _compute_cpp_name(self) -> str:
        """C++ method name. Two upstream-metal-cpp conventions applied:
          1. Strip trailing `With<Word>` from the first selector segment,
             but only when the selector takes args (`commandBufferWithDescriptor:`
             → `commandBuffer`, `commandBufferWithUnretainedReferences` →
             unchanged — the latter has no args and would collide with the
             no-arg `commandBuffer()` overload).
          2. Lowercase a leading run of capital letters (acronym prefix)
             when it's followed by a lowercase letter — `UTF8String` →
             `utf8String`, `URLString` → `urlString`. Names like
             `URLWithString` keep the prefix because the run is followed
             by an uppercase letter."""
        first = self.selector.split(":")[0]
        if not first:
            return first
        has_args = ":" in self.selector
        # Step 1: preposition-strip (lower-case selectors only, with args).
        # Apple's selector convention attaches descriptive prepositions to
        # the first segment that the actual args make redundant in C++:
        #   commandBufferWithDescriptor:        → commandBuffer
        #   objectAtIndexedSubscript:           → object
        #   encodeWaitForEvent:value:           → encodeWait
        #   maximumLengthOfBytesUsingEncoding:  → maximumLengthOfBytes
        # The strip is gated on the selector having at least one arg, so
        # `commandBufferWithUnretainedReferences` (no args) stays put.
        if has_args and first[0].islower():
            m = re.search(r"(?:With|At|For|Using)[A-Z]", first)
            if m and m.start() > 0:
                first = first[:m.start()]
        # Step 2: acronym-prefix lowercase (`UTF8String` → `utf8String`).
        return cpp_method_name_from_first_segment(first)


@dataclass
class ObjCEnumValue:
    name: str
    value: Optional[int]
    # Original source-form initializer (`(1ULL << 40)`, `Foo | Bar`, ...).
    # Preserved verbatim from the SDK header so the emitted enum keeps the
    # hand-readable expressions Apple used — falling back to the evaluated
    # integer only when there's no explicit `= <expr>`.
    value_expr: Optional[str] = None
    availability: Availability = field(default_factory=Availability)


@dataclass
class ObjCEnum:
    name: str
    underlying_type: str
    values: list[ObjCEnumValue] = field(default_factory=list)
    # True when declared via `NS_OPTIONS` / `CF_OPTIONS` — should emit as
    # `_<P>_OPTIONS(...)` so bitwise operators work without casts.
    is_options: bool = False
    availability: Availability = field(default_factory=Availability)


@dataclass
class ObjCBlockTypedef:
    """`typedef void (^MTLFooHandler)(...)` — emitted as both the raw block
    alias and an std::function-taking ergonomic overload."""
    name: str               # "MTLIOCommandBufferHandler"
    return_type: str        # "void"
    arg_types: list[str]    # ["id<MTLIOCommandBuffer>"]


@dataclass
class ObjCStructField:
    name: str
    objc_type: str
    # For C fixed-size arrays (`uint16_t edge[4]`), the element type goes
    # into `objc_type` ("uint16_t") and the dimension comes out here. In
    # C++ the dimension must trail the field name (`uint16_t edge[4];`),
    # not the type (`uint16_t[4] edge;` doesn't parse).
    array_size: Optional[int] = None


@dataclass
class ObjCStruct:
    """Plain C struct exposed in an Obj-C framework header (e.g.
    MTL4BufferRange). Emitted as a regular C++ struct, with the framework
    prefix stripped from the C++ name (NSOperatingSystemVersion → NS::
    OperatingSystemVersion to match upstream metal-cpp)."""
    name: str
    fields: list[ObjCStructField] = field(default_factory=list)
    packed: bool = False  # True ⇒ emit with _<PREFIX>_PACKED
    # Extra C++ members (constructors, static factories) injected verbatim
    # at the top of the struct body. Matches upstream metal-cpp's hand-written
    # convenience helpers (`MTL::Size::Make`, `MTL::Region::Make3D`, ...).
    extra_members: str = ""
    availability: Availability = field(default_factory=Availability)


@dataclass
class ObjCClass:
    name: str
    superclass: str = "NSObject"
    properties: list[ObjCProperty] = field(default_factory=list)
    methods: list[ObjCMethod] = field(default_factory=list)
    protocols: list[str] = field(default_factory=list)
    # True when this came from `@protocol Foo` rather than `@interface Foo`.
    # Protocols are uninstantiable: there's no `+alloc` and the Metal/
    # Foundation runtime hands them to you via factory methods on a concrete
    # object (`MTL::Device::newBuffer(...)`). Suppress the auto-emitted
    # `alloc()`/`init()` cluster for them.
    is_protocol: bool = False
    availability: Availability = field(default_factory=Availability)


@dataclass
class ObjCConstant:
    """An `extern <type> const <name>` global declared in an SDK header
    (e.g. `extern CADynamicRange const CADynamicRangeHigh`). Surfaced in the
    framework namespace via a `__asm__`-label binding to the system C symbol
    so call sites can use `CA::DynamicRangeHigh` without redeclaring Apple's
    `CADynamicRangeHigh` extern (which would clash under ObjC ARC in `.mm`
    files that pull in Apple's headers transitively)."""
    c_name: str       # `CADynamicRangeHigh`
    cpp_name: str     # `DynamicRangeHigh`
    cpp_type: str     # `CA::DynamicRange` or `NS::String*`


@dataclass
class FrameworkData:
    """Parsed data for one framework from one SDK."""

    classes: list[ObjCClass] = field(default_factory=list)
    enums: list[ObjCEnum] = field(default_factory=list)
    blocks: list[ObjCBlockTypedef] = field(default_factory=list)
    structs: list[ObjCStruct] = field(default_factory=list)
    constants: list[ObjCConstant] = field(default_factory=list)
    # ObjC typedefs of the form `typedef <ClassName> * <Alias>` — surfaced as
    # `using <Alias> = <C++Class>*;` so constants and members can name the
    # Swift-friendly enum-style alias (`CA::DynamicRange`).
    string_typedefs: list[tuple[str, str]] = field(default_factory=list)  # (name, resolved_cpp_type)
    # `typedef MTLSamplePosition MTLCoordinate2D;` — alias for an existing
    # struct/record. Without this, the resolver only knows MTLSamplePosition
    # and treats the alias as an unresolvable type. Captured as
    # (alias_name, target_objc_name) so framework registration can route
    # the alias to the same C++ type as the target.
    struct_aliases: list[tuple[str, str]] = field(default_factory=list)
    # `typedef MTLRenderPipelineReflection * MTLAutoreleasedRenderPipelineReflection;`
    # — typedef whose underlying is a class pointer. Stored as
    # (alias_name, target_class_objc_name) so the alias surfaces as
    # `using AutoreleasedRenderPipelineReflection = MTL::RenderPipelineReflection*;`.
    class_pointer_aliases: list[tuple[str, str]] = field(default_factory=list)
    # `typedef uint64_t MTLTimestamp;` — typedef whose underlying is a
    # primitive/builtin type. Stored as (alias_name, underlying_spelling)
    # so the alias surfaces as `using Timestamp = uint64_t;`.
    primitive_aliases: list[tuple[str, str]] = field(default_factory=list)


# ── Type resolution ───────────────────────────────────────────────────────


class TypeResolver:
    """Maps ObjC type spellings to C++ types."""

    def __init__(self) -> None:
        self.type_map: dict[str, str] = dict(BUILTIN_TYPE_MAP)
        self.unresolved: list[tuple[str, str]] = []  # (type, context)
        # Per-cpp-qualified-name kind: "class" | "enum" | "block" | "struct"
        # | "typedef". Used by the forward-decl scanner so it doesn't emit
        # `class MTL::PixelFormat;` when PixelFormat is in fact an enum.
        self.kinds: dict[str, str] = {}
        # Pre-seed kinds for the BUILTIN_TYPE_MAP entries that are typedefs
        # / structs / values rather than classes — so the forward-decl
        # scanner doesn't emit `class UInteger;` etc.
        for _cpp in BUILTIN_TYPE_MAP.values():
            bare = _cpp.rstrip("*").strip()
            self.kinds[bare] = "typedef"
        # Per-cpp-qualified-name framework prefix, so the scanner can request
        # cross-framework <Prefix>Enums.hpp / Blocks.hpp / Structs.hpp.
        self.frameworks: dict[str, str] = {}  # cpp name → fw prefix
        # ObjC class name → SDK header basename (no .h). Shared across all
        # frameworks so a class that extends a type defined in a sibling
        # framework (CA::MetalDrawable extending MTL::Drawable) can locate
        # the parent's header for its #include line.
        self.class_to_source: dict[str, str] = {}
        # C++ qualified name → SDK header basename (no .h). Drives per-
        # source-header enum includes — when a class references
        # `MTL::IndexType`, the emitter consults this map to pick the
        # specific `<Fw>/MTLArgument.hpp` include rather than a per-
        # framework aggregate.
        self.cpp_to_source: dict[str, str] = {}
        # Enum cpp qualified name → resolved underlying type (e.g.
        # `NS::UInteger`). Lets the scanner emit a C++11 opaque enum
        # forward decl (`enum Foo : NS::UInteger;`) for cross-file enum
        # references — avoids pulling the whole declaring hpp through and
        # the include cycles that creates.
        self.enum_underlying: dict[str, str] = {}
        # Enum cpp qualified name → True for OPTIONS-style enums
        # (`_<P>_OPTIONS` expands to a `using <Name> = <Underlying>;` plus
        # an unnamed `enum : <Name>`). Forward-declaring those as
        # `enum Foo : Underlying;` would clash with the type alias —
        # they're instead forward-declared as `using Foo = Underlying;`.
        self.enum_is_options: dict[str, bool] = {}
        # String-typedef aliases keyed by their qualified C++ name
        # (`NS::ErrorDomain` → `NS::String*`). Lets emitters that can't
        # see the per-header `using` definition (notably `<P>Bridge.hpp`)
        # rewrite signatures to the underlying type so they don't depend
        # on the alias being declared earlier in the include chain.
        self.alias_underlying: dict[str, str] = {}

    def register(self, objc_name: str, cpp_type: str, kind: str = "class",
                 framework_prefix: str = "") -> None:
        """Register a generated type mapping. `kind` distinguishes class
        from non-class names so emission can pick the right declaration
        form. `framework_prefix` keys into per-framework support headers
        like MTLEnums.hpp."""
        self.type_map[objc_name] = cpp_type
        bare = cpp_type.rstrip("*").strip()
        self.kinds[bare] = kind
        if framework_prefix:
            self.frameworks[bare] = framework_prefix

    def resolve(self, objc_type: str, context: str = "") -> str:
        """Resolve an ObjC type string to its C++ equivalent.

        Returns 'void*' and logs a warning for unresolvable types.
        Returns '__instancetype__' for instancetype (caller handles).
        """
        s = self._normalize(objc_type, strip_generics=False)

        # `NSEnumerator<ObjectType> *` (or any inner type) collapses to the
        # kept-upstream `NS::Enumerator<NS::Object>*` — matches upstream
        # metal-cpp's choice to type-erase the element type at the wrapper
        # boundary. Done before the generic-strip pass so template arity
        # stays on the C++ side.
        if re.match(r"^NSEnumerator\s*<[^<>]*>\s*\*?$", s):
            return "NS::Enumerator<NS::Object>*"

        # `id<Protocol>` and `id<Protocol> const *` — recognized BEFORE
        # `_strip_generics` runs, since that pass now also collapses the
        # nested `id<X>` form (so `NSArray<id<MTLAllocation>> *` reduces to
        # `NSArray *`) and would otherwise erase the protocol info needed
        # here.
        m = re.match(r"^id<(\w+)>$", s)
        if m:
            proto = m.group(1)
            cpp = self.type_map.get(proto)
            if cpp:
                return f"{cpp}*"
            return self._unresolved(objc_type, context)

        m = re.match(r"^id<(\w+)>\s*const\s*\*$", s)
        if m:
            proto = m.group(1)
            cpp = self.type_map.get(proto)
            if cpp:
                return f"const {cpp}* const *"
            return self._unresolved(objc_type, context)

        s = self._strip_generics(s)

        # instancetype → handled by caller with the concrete class
        if s == "instancetype":
            return "__instancetype__"

        # `id *` / `const id *` — typeless object-pointer arrays. Apple
        # collection factories take these as `(const ObjectType _Nonnull [])`,
        # which clang decays to `ObjectType const *` (then ObjectType → id).
        # Upstream metal-cpp exposes them as `const NS::Object* const *`.
        if re.fullmatch(r"id\s*\*|const\s+id\s*\*|id\s+const\s*\*", s):
            return "const NS::Object* const *"

        # Plain `id` → `NS::Object*` (matches upstream metal-cpp). After
        # `_strip_generics`, `ObjectType`/`KeyType`/`ValueType` collapse to
        # `id`, so collection factories like `NSArray::array(NS::Object*)`
        # and bare-`id` properties (`MTLCaptureDescriptor.captureObject`)
        # both land here.
        if s == "id":
            return "NS::Object*"

        # ObjC object pointer-to-pointer: `<Class> *const *` /
        # `<Class> * const *` — common in factories that take a C array of
        # typed objects. Emit `const <Cpp>* const *` to match upstream.
        m = re.match(r"^(\w+)\s*\*\s*const\s*\*$", s)
        if m:
            cls = m.group(1)
            cpp = self.type_map.get(cls)
            if cpp:
                base = cpp.rstrip("*").strip()
                return f"const {base}* const *"

        # `ClassName * *` — writable pointer-to-pointer (the Cocoa out-error
        # idiom: `(NSError **)error`). Distinct from the `* const *` array
        # case above; emitted plainly as `<C++>**` so callers pass
        # `&myError` to populate it.
        m = re.match(r"^(\w+)\s*\*\s*\*$", s)
        if m:
            cls = m.group(1)
            cpp = self.type_map.get(cls)
            if cpp:
                base = cpp.rstrip("*").strip()
                return f"{base}**"

        # `ClassName * const` — Apple's `extern Type const Name` globals
        # canonicalize to this (const-qualified pointer to an ObjC class).
        # Emit `<C++>* const` so the variable's symbol matches a system
        # `<Type> const` extern at link time.
        m = re.match(r"^(\w+)\s*\*\s*const$", s)
        if m:
            cls = m.group(1)
            cpp = self.type_map.get(cls)
            if cpp:
                base = cpp.rstrip("*").strip()
                return f"{base}* const"

        # `const <TypedefName>` — the non-canonical spelling of the same
        # globals (typedef-named, const-qualified). The typedef is itself
        # a pointer-typed alias, so append `const` to whatever the resolver
        # produces for the bare name.
        m = re.match(r"^const\s+(\w+)$", s)
        if m:
            cls = m.group(1)
            cpp = self.type_map.get(cls)
            if cpp:
                return f"{cpp} const"

        # ObjC object pointer: ClassName *
        m = re.match(r"^(\w+)\s*\*$", s)
        if m:
            cls = m.group(1)
            cpp = self.type_map.get(cls)
            if cpp:
                # Some typedefs already include the pointer in their mapped
                # form (e.g. NSErrorDomain → NS::String*); don't double it.
                return cpp if cpp.endswith("*") else f"{cpp}*"
            if cls in PASSTHROUGH_TYPES or cls in ("void", "char", "unsigned"):
                return s
            return self._unresolved(objc_type, context)

        # Direct lookup (typedef or basic type)
        if s in self.type_map:
            return self.type_map[s]
        if s in PASSTHROUGH_TYPES:
            return s

        # Pointer to passthrough: void *, const char *, etc.
        if s.endswith("*"):
            base = s[:-1].strip()
            if base in PASSTHROUGH_TYPES:
                return s
            # Const-qualified pointer: re-resolve the unqualified base, then
            # reattach const + *. Falls back to void* if base can't be resolved.
            if base.startswith("const "):
                inner = base[6:].strip()
                if inner in self.type_map:
                    return f"const {self.type_map[inner]} *"
                if inner in PASSTHROUGH_TYPES:
                    return s
                return self._unresolved(objc_type, context)

        return self._unresolved(objc_type, context)

    def _normalize(self, s: str, strip_generics: bool = True) -> str:
        """Strip nullability/ownership qualifiers, ObjC generic type params,
        and extra whitespace. `NSDictionary<NSString *, NSObject *> *` →
        `NSDictionary *` — upstream metal-cpp drops generic params and exposes
        the type-erased class, recovering element types via template methods.
        Pass `strip_generics=False` to keep the `<...>` for callers that want
        to pattern-match on parameterized types (e.g. `NSEnumerator<T>`)."""
        s = re.sub(
            r"\b(__nullable|__nonnull|_Nullable|_Nonnull|"
            r"__kindof|__autoreleasing|__unsafe_unretained|__weak|__strong)\b",
            "",
            s,
        )
        if strip_generics:
            s = self._strip_generics(s)
        return re.sub(r"\s+", " ", s).strip()

    def _strip_generics(self, s: str) -> str:
        """`Class<A, B>` → `Class`; `ObjectType`/`KeyType`/`ValueType` → `id`.
        Split out of `_normalize` so resolve() can pattern-match against
        parameterized types before this drops them."""
        prev = None
        while prev != s:
            prev = s
            # Strip inner `id<Protocol>` → `id` first, so nested forms like
            # `NSArray<id<MTLAllocation>> *` collapse to `NSArray<id> *` and
            # then to `NSArray *` on the next pass. The outer class-stripping
            # regex requires `[A-Z]\w*`, so `id<...>` would otherwise stick.
            s = re.sub(r"\bid\s*<[^<>]*>", "id", s)
            s = re.sub(r"\b([A-Z]\w*)\s*<[^<>]*>", r"\1", s)
        s = re.sub(r"\b(ObjectType|KeyType|ValueType)\b", "id", s)
        return re.sub(r"\s+", " ", s).strip()

    def _unresolved(self, objc_type: str, context: str) -> str:
        log.warning("Unresolvable type: '%s' (in %s)", objc_type, context or "?")
        self.unresolved.append((objc_type, context))
        return "void*"


# ── ObjC header parsing ──────────────────────────────────────────────────


def _parse_block_typedef(name: str, spelling: str) -> Optional[ObjCBlockTypedef]:
    """Parse `void (^)(arg1, arg2)` (libclang block-pointer spelling).
    Returns None if the spelling doesn't look like a block."""
    m = re.match(r"^(.+?)\s*\(\^[^)]*\)\((.*)\)\s*$", spelling)
    if not m:
        return None
    ret = m.group(1).strip()
    raw_args = m.group(2).strip()
    args = [a.strip() for a in raw_args.split(",")] if raw_args else []
    # Empty-arg blocks come back as ["void"] or [""]; normalize.
    if args == ["void"] or args == [""]:
        args = []
    return ObjCBlockTypedef(name=name, return_type=ret, arg_types=args)


# ObjC generic-parameter identifier → upstream-style C++ template name.
_GENERIC_PARAM_RETURNS: dict[str, str] = {
    "ObjectType": "_Object",
    "KeyType": "_KeyType",
    "ValueType": "_ValueType",
}

# libclang TypeKinds for `typedef <primitive> <Alias>;` detection. Used
# to filter typedefs that wrap a builtin scalar (`typedef uint64_t
# MTLTimestamp`) from those that wrap a class pointer / struct / enum.
_PRIMITIVE_TYPE_KINDS = frozenset({
    TypeKind.BOOL, TypeKind.CHAR_S, TypeKind.CHAR_U, TypeKind.SCHAR,
    TypeKind.UCHAR, TypeKind.WCHAR, TypeKind.CHAR16, TypeKind.CHAR32,
    TypeKind.SHORT, TypeKind.USHORT, TypeKind.INT, TypeKind.UINT,
    TypeKind.LONG, TypeKind.ULONG, TypeKind.LONGLONG, TypeKind.ULONGLONG,
    TypeKind.FLOAT, TypeKind.DOUBLE, TypeKind.LONGDOUBLE,
})


def _generic_return_info(objc_type: str) -> Optional[tuple[str, str]]:
    """Return `(template_param, cpp_return_type)` when `objc_type` is a
    generic-parameter return that upstream metal-cpp emits as a templated
    accessor, else None. Two shapes are recognized:

      * Bare generic param (`ObjectType`/`KeyType`/`ValueType`) — emitted as
        `template <class _X = Object> _X* foo(...)`.
      * `NSEnumerator<GenericParam> *` — emitted as
        `template <class _X = Object> Enumerator<_X>* foo(...)`. Matches
        upstream's `Dictionary::keyEnumerator<_KeyType>()` form.
    """
    s = re.sub(
        r"\b(_Nullable|_Nonnull|_Null_unspecified|"
        r"__autoreleasing|__strong|__weak|__unsafe_unretained)\b",
        "", objc_type)
    s = re.sub(r"\s+", " ", s).strip()
    if s in _GENERIC_PARAM_RETURNS:
        t = _GENERIC_PARAM_RETURNS[s]
        return (t, f"{t}*")
    m = re.match(r"^NSEnumerator\s*<\s*(\w+)\s*>\s*\*$", s)
    if m and m.group(1) in _GENERIC_PARAM_RETURNS:
        t = _GENERIC_PARAM_RETURNS[m.group(1)]
        return (t, f"Enumerator<{t}>*")
    return None


def _is_generic_return(objc_type: str) -> bool:
    """True when a method's ObjC return type is a generic parameter — see
    `_generic_return_info` for the recognized shapes."""
    return _generic_return_info(objc_type) is not None


def _enum_value_expr(cursor) -> Optional[str]:
    """Source text of an enum constant's initializer expression, or None
    when the constant has no explicit `= <expr>`. Lets emitted enums keep
    Apple's hand-written forms (`(1ULL << 40)`, `Foo | Bar`) instead of
    the libclang-evaluated integer."""
    try:
        ext = cursor.extent
        f = ext.start.file
        if f is None:
            return None
        length = ext.end.offset - ext.start.offset
        if length <= 0:
            return None
        with open(f.name, "rb") as fp:
            fp.seek(ext.start.offset)
            src = fp.read(length).decode("utf-8", errors="replace")
    except (OSError, AttributeError, ValueError):
        return None
    eq = src.find("=")
    if eq < 0:
        return None
    return src[eq + 1:].strip()


def _synth_block_name(method_cpp_name: str) -> str:
    """Synthesize a block typedef name from a method's cpp name. Mirrors
    upstream metal-cpp's naming (`addLogHandler` → `LogHandlerBlock`).
    Strips trailing 'Block' before appending so we don't end up with
    'EnumerateByteRangesUsingBlockBlock'."""
    name = method_cpp_name
    for prefix in ("add", "set", "with", "register", "remove", "insert"):
        rest = name[len(prefix):]
        if name.startswith(prefix) and rest and rest[0].isupper():
            name = rest
            break
    else:
        name = name[0].upper() + name[1:]
    if name.endswith("Block"):
        name = name[:-len("Block")]
    return name + "Block"


def parse_apinotes_renames(path: Path) -> dict[str, dict[str, str]]:
    """Parse Apple's `<Framework>.apinotes` file, returning a nested map
    `{class_name: {selector: cpp_name}}` derived from SwiftName entries.

    Apple's apinotes are YAML with directives like
        Classes:
        - Name: MTLRenderCommandEncoder
          Methods:
          - Selector: 'setBlendColorRed:green:blue:alpha:'
            SwiftName: setBlendColor(red:green:blue:alpha:)

    The SwiftName's method part (before the `(`) is the canonical short
    name Apple intends. metal-cpp follows ObjC selector conventions rather
    than Swift's, so we apply only the renames Swift derived by stripping
    the first argument label from the selector's first segment — the
    standard ObjC convention (`setBlendColorRed:green:blue:alpha:` →
    `setBlendColor` with first label `red:`). SwiftNames that rewrite the
    method itself (`copyFromTexture:toTexture:` → `copy`, `commandBuffer`
    → `makeCommandBuffer`) diverge from upstream metal-cpp and are
    skipped — heuristic handles those."""
    import yaml  # local import — apinotes are optional
    try:
        with open(path) as f:
            doc = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}

    result: dict[str, dict[str, str]] = {}
    for section in ("Classes", "Protocols"):
        for entry in doc.get(section, []) or []:
            cls = entry.get("Name")
            methods = entry.get("Methods") or []
            if not cls or not methods:
                continue
            renames: dict[str, str] = {}
            for m in methods:
                sel = m.get("Selector")
                swift = m.get("SwiftName")
                if not sel or not swift:
                    continue
                # Split `setBlendColor(red:green:blue:alpha:)` into
                # ("setBlendColor", ["red", "green", "blue", "alpha"]).
                paren = swift.find("(")
                if paren < 0:
                    continue
                cpp_name = swift[:paren]
                arg_labels = [a for a in swift[paren + 1:swift.rfind(")")].split(":") if a]
                if not cpp_name or cpp_name.startswith("_") or not cpp_name[0].isalpha():
                    continue
                first_seg = sel.split(":", 1)[0]
                # Apply only when SwiftName preserves the ObjC method root —
                # i.e. first segment = cpp_name + capitalize(first_arg).
                # `_` is Swift's "unnamed first parameter" marker, which means
                # the selector already embeds the label in its method name
                # (so first segment = cpp_name with nothing appended).
                first_arg = arg_labels[0] if arg_labels else ""
                if first_arg == "_":
                    first_arg = ""
                expected = cpp_name + (first_arg[:1].upper() + first_arg[1:] if first_arg else "")
                if first_seg != expected:
                    continue
                renames[sel] = cpp_name
            if renames:
                # Same class name may appear in both Classes and Protocols
                # (interface + protocol with matching name). Merge.
                result.setdefault(cls, {}).update(renames)
    return result

def _safe_kind(cursor) -> Optional[CursorKind]:
    """Get cursor kind, returning None for unknown kinds from newer SDKs."""
    try:
        return cursor.kind
    except ValueError:
        return None


class ObjCParser:
    """Parse Objective-C headers using libclang."""

    def __init__(self, sdk_path: Path) -> None:
        self.sdk_path = sdk_path
        self.index = Index.create()

    def parse_header(self, header_path: Path) -> FrameworkData:
        """Parse a single ObjC header and extract classes/enums."""
        args = [
            "-x",
            "objective-c",
            "-isysroot",
            str(self.sdk_path),
            "-fno-objc-arc",
            "-Wno-everything",
        ]
        tu = self.index.parse(
            str(header_path),
            args=args,
            options=(TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD | TranslationUnit.PARSE_SKIP_FUNCTION_BODIES),
        )

        data = FrameworkData()
        target = str(header_path)

        # First pass: collect @interface declarations
        classes_by_name: dict[str, ObjCClass] = {}
        for cursor in tu.cursor.get_children():
            loc = cursor.location
            if not loc.file or loc.file.name != target:
                continue
            kind = _safe_kind(cursor)
            if kind == CursorKind.OBJC_INTERFACE_DECL:
                cls = self._parse_class(cursor)
                if cls:
                    classes_by_name[cls.name] = cls
            elif kind == CursorKind.OBJC_PROTOCOL_DECL:
                # metal-cpp wraps protocols as concrete classes — same shape
                # as @interface, but the "superclass" comes from a parent
                # protocol (the first OBJC_PROTOCOL_REF child) rather than
                # OBJC_SUPER_CLASS_REF.
                cls = self._parse_protocol(cursor)
                if cls:
                    classes_by_name[cls.name] = cls
            elif kind == CursorKind.TYPEDEF_DECL:
                # `typedef void (^MTLFooHandler)(...)` — Obj-C block typedef.
                # Emit a `using` alias + std::function overload per upstream.
                try:
                    ut = cursor.underlying_typedef_type
                    if ut.kind == TypeKind.BLOCKPOINTER:
                        block = _parse_block_typedef(cursor.spelling, ut.spelling)
                        if block:
                            data.blocks.append(block)
                            continue
                except (AttributeError, ValueError):
                    pass
                # Catch `typedef NS_ENUM(Underlying, Name) { ... }` — clang
                # emits an anonymous ENUM_DECL plus a typedef pointing at it.
                # _parse_enum returns None for the anonymous form, so re-enter
                # through the typedef and pull both the underlying type and
                # the enum values from the referenced declaration.
                try:
                    decl = cursor.underlying_typedef_type.get_declaration()
                    if _safe_kind(decl) == CursorKind.ENUM_DECL:
                        underlying = decl.enum_type.spelling
                        if underlying:
                            # Inspect the typedef's tokens to distinguish
                            # NS_OPTIONS (flag enum, bitwise OR allowed) from
                            # NS_ENUM. libclang exposes the macro names via
                            # the cursor's tokens.
                            is_options = False
                            try:
                                for tok in cursor.get_tokens():
                                    if tok.spelling in ("NS_OPTIONS", "CF_OPTIONS", "MTL_OPTIONS"):
                                        is_options = True
                                        break
                                    if tok.spelling in ("NS_ENUM", "CF_ENUM"):
                                        break
                            except Exception:
                                pass
                            values: list[ObjCEnumValue] = []
                            for child in decl.get_children():
                                if _safe_kind(child) == CursorKind.ENUM_CONSTANT_DECL:
                                    try:
                                        v: Optional[int] = child.enum_value
                                    except (ValueError, AttributeError):
                                        v = None
                                    values.append(ObjCEnumValue(
                                        name=child.spelling, value=v,
                                        value_expr=_enum_value_expr(child),
                                        availability=cursor_availability(child)))
                            # Apple attaches the type-level `API_AVAILABLE` to
                            # the typedef, not the inner ENUM_DECL — pull from
                            # the cursor we're parsing here.
                            data.enums.append(ObjCEnum(
                                name=cursor.spelling, underlying_type=underlying,
                                values=values, is_options=is_options,
                                availability=cursor_availability(cursor)))
                except (AttributeError, ValueError, AssertionError):
                    pass
                # `typedef NSString * MyAlias` (NS_TYPED_ENUM-style string
                # typedefs that back Apple's extern const groupings — e.g.
                # `CADynamicRange`). Captured so the constants emitter can
                # surface them as `using` aliases inside the namespace.
                # Limited to NSString-backed aliases — typedefs over other
                # ObjC classes (e.g. `MTLAutoreleasedRenderPipelineReflection
                # = MTLRenderPipelineReflection*`) are intentionally not
                # surfaced here; they'd require cross-header includes that
                # the scanner doesn't yet emit.
                try:
                    ut = cursor.underlying_typedef_type
                    spelling = ut.spelling
                    if spelling.strip() == "NSString *":
                        data.string_typedefs.append((cursor.spelling, spelling))
                except (AttributeError, ValueError):
                    pass
                # `typedef MTLSamplePosition MTLCoordinate2D;` — record-type
                # alias. The resolver wouldn't otherwise know MTLCoordinate2D
                # is the same shape as a registered struct, and every method
                # using it would fall through to `void*`. Chase typedef
                # chains (the target may itself be a typedef wrapping a
                # STRUCT_DECL), and record the immediate parent name —
                # that's what's already registered with the resolver.
                try:
                    ut = cursor.underlying_typedef_type
                    decl = ut.get_declaration()
                    target_name = ""
                    found_struct = False
                    for _ in range(8):  # bounded walk to avoid pathological chains
                        k = _safe_kind(decl)
                        if k == CursorKind.STRUCT_DECL:
                            found_struct = True
                            # Apple's `typedef struct _Foo {...} Foo;` lands
                            # here on the first hop with no intermediate
                            # typedef. Fall back to the struct tag so the
                            # alias points at what `register_objc_declarations`
                            # actually registered.
                            if not target_name:
                                target_name = decl.spelling
                            break
                        if k == CursorKind.TYPEDEF_DECL:
                            # Remember the immediate alias name; we use it
                            # as the link target if the chain bottoms out
                            # at a STRUCT_DECL.
                            target_name = decl.spelling
                            decl = decl.underlying_typedef_type.get_declaration()
                            continue
                        break
                    if (found_struct and target_name
                            and target_name != cursor.spelling):
                        data.struct_aliases.append(
                            (cursor.spelling, target_name))
                except (AttributeError, ValueError, AssertionError):
                    pass
                # `typedef MTLRenderPipelineReflection *
                #         MTLAutoreleasedRenderPipelineReflection;` — typedef
                # to an ObjC class pointer. Surface as
                # `using AutoreleasedRenderPipelineReflection = MTL::RenderPipelineReflection*;`.
                # Captures cleanly via the libclang type kind.
                try:
                    ut = cursor.underlying_typedef_type
                    if ut.kind == TypeKind.OBJCOBJECTPOINTER:
                        pointee = ut.get_pointee()
                        target_cls = pointee.get_declaration().spelling
                        if target_cls and target_cls != cursor.spelling:
                            data.class_pointer_aliases.append(
                                (cursor.spelling, target_cls))
                except (AttributeError, ValueError):
                    pass
                # `typedef uint64_t MTLTimestamp;` — typedef over a builtin
                # primitive type. libclang's type-system loses the original
                # spelling (`uint64_t` reduces to `int`), so recover it from
                # the typedef's source tokens — the underlying spelling
                # always sits between `typedef` and the alias name.
                try:
                    ut = cursor.underlying_typedef_type
                    # libclang reports `typedef uint64_t Foo;` with
                    # `ut.kind == ELABORATED` (wrapped typedef-alias name);
                    # the canonical type is the actual primitive kind.
                    canon_kind = ut.get_canonical().kind
                    if canon_kind in _PRIMITIVE_TYPE_KINDS:
                        toks = [t.spelling for t in cursor.get_tokens()]
                        if "typedef" in toks and cursor.spelling in toks:
                            i0 = toks.index("typedef") + 1
                            i1 = toks.index(cursor.spelling, i0)
                            underlying_spell = " ".join(toks[i0:i1]).strip()
                            if (underlying_spell
                                    and underlying_spell != cursor.spelling):
                                data.primitive_aliases.append(
                                    (cursor.spelling, underlying_spell))
                except (AttributeError, ValueError):
                    pass
                continue
            elif kind == CursorKind.VAR_DECL:
                # `extern <Type> const <Name>` — surface as namespaced const
                # bound to the system C symbol via __asm__ label. The non-
                # canonical spelling preserves typedef aliases so emitted
                # constants name them (`CA::DynamicRange const` rather than
                # `NS::String* const`).
                try:
                    if cursor.storage_class.name == "EXTERN":
                        data.constants.append(ObjCConstant(
                            c_name=cursor.spelling,
                            cpp_name=cursor.spelling,
                            cpp_type=cursor.type.spelling,
                        ))
                except (AttributeError, ValueError):
                    pass
            elif kind == CursorKind.ENUM_DECL:
                enum = self._parse_enum(cursor)
                if enum:
                    data.enums.append(enum)
            elif kind == CursorKind.STRUCT_DECL:
                # Named, file-scope C struct (e.g. MTL4BufferRange). Anonymous
                # / nested structs are ignored — they reach us only through
                # the TYPEDEF_DECL path and aren't generally part of the
                # framework's public API.
                if cursor.spelling and not cursor.spelling.startswith("("):
                    fields = []
                    packed = False
                    for ch in cursor.get_children():
                        ck = _safe_kind(ch)
                        if ck == CursorKind.FIELD_DECL:
                            ft = ch.type
                            if ft.kind == TypeKind.CONSTANTARRAY:
                                fields.append(ObjCStructField(
                                    name=ch.spelling,
                                    objc_type=ft.element_type.spelling,
                                    array_size=ft.element_count))
                            else:
                                fields.append(ObjCStructField(
                                    name=ch.spelling, objc_type=ft.spelling))
                        elif ck == CursorKind.PACKED_ATTR:
                            packed = True
                    if fields:
                        data.structs.append(ObjCStruct(
                            name=cursor.spelling, fields=fields, packed=packed,
                            availability=cursor_availability(cursor)))

        # Second pass: merge members from categories/extensions into classes
        for cursor in tu.cursor.get_children():
            loc = cursor.location
            if not loc.file or loc.file.name != target:
                continue
            if _safe_kind(cursor) != CursorKind.OBJC_CATEGORY_DECL:
                continue
            # Find which class this category extends
            class_name = None
            for child in cursor.get_children():
                if _safe_kind(child) == CursorKind.OBJC_CLASS_REF:
                    class_name = child.spelling
                    break
            if not class_name or class_name not in classes_by_name:
                continue
            # Parse the category like a class and merge members
            cat = self._parse_class_members(cursor)
            target_cls = classes_by_name[class_name]
            existing_props = {p.name for p in target_cls.properties}
            existing_sels = {m.selector for m in target_cls.methods}
            for p in cat.properties:
                if p.name not in existing_props:
                    target_cls.properties.append(p)
            for m in cat.methods:
                if m.selector not in existing_sels:
                    target_cls.methods.append(m)

        data.classes = list(classes_by_name.values())

        # `typedef NS_ENUM(Underlying, Name) { ... }` produces two cursors —
        # a TYPEDEF_DECL and an anonymous ENUM_DECL whose name we
        # synthesize from the typedef. Both paths above append to
        # `data.enums`; collapse the dupes (first occurrence wins, since
        # the TYPEDEF_DECL handler runs first). Anonymous loose-constant
        # blocks have empty `name` and stay distinct.
        if data.enums:
            seen: set[str] = set()
            deduped: list[ObjCEnum] = []
            for e in data.enums:
                if e.name:
                    if e.name in seen:
                        continue
                    seen.add(e.name)
                deduped.append(e)
            data.enums = deduped

        # Synthesize block typedefs for inline (anonymous) block parameters.
        # Apple's SDK frequently declares them without a `typedef`, so the
        # spelling we see on the param is `void (^)(arg1, arg2)` and the
        # resolver has nothing to map it to. Each unique block spelling gets
        # a synthetic typedef named after the method that first used it
        # (matching upstream metal-cpp's `LogHandlerBlock` etc.).
        block_by_spelling: dict[str, str] = {b.name: b.name for b in data.blocks}
        existing_names = {b.name for b in data.blocks}
        for cls in data.classes:
            for m in cls.methods:
                for p in m.params:
                    if "(^" not in p.objc_type:
                        continue
                    spelling = p.objc_type
                    if spelling in block_by_spelling:
                        p.objc_type = block_by_spelling[spelling]
                        continue
                    base = _synth_block_name(m.cpp_name)
                    synth = base
                    k = 2
                    while synth in existing_names:
                        synth = f"{base}{k}"
                        k += 1
                    blk = _parse_block_typedef(synth, spelling)
                    if not blk:
                        continue
                    data.blocks.append(blk)
                    existing_names.add(synth)
                    block_by_spelling[spelling] = synth
                    p.objc_type = synth
        return data

    def _parse_class_members(self, cursor) -> ObjCClass:
        """Parse properties/methods from a cursor (class or category)."""
        cls = ObjCClass(name="")
        for child in cursor.get_children():
            kind = _safe_kind(child)
            if kind == CursorKind.OBJC_PROPERTY_DECL:
                prop = self._parse_property(child)
                if prop:
                    cls.properties.append(prop)
            elif kind == CursorKind.OBJC_INSTANCE_METHOD_DECL:
                method = self._parse_method(child, is_class=False)
                if method:
                    cls.methods.append(method)
            elif kind == CursorKind.OBJC_CLASS_METHOD_DECL:
                method = self._parse_method(child, is_class=True)
                if method:
                    cls.methods.append(method)
            elif kind == CursorKind.OBJC_PROTOCOL_REF:
                cls.protocols.append(child.spelling)
        # Remove synthesized accessor methods that duplicate properties.
        prop_sels = set()
        for prop in cls.properties:
            prop_sels.add(prop.name)
            if not prop.is_readonly:
                prop_sels.add(f"set{prop.name[0].upper()}{prop.name[1:]}:")
        cls.methods = [m for m in cls.methods if m.selector not in prop_sels]
        return cls

    def _parse_class(self, cursor) -> Optional[ObjCClass]:
        name = cursor.spelling
        if not name:
            return None

        superclass = ""
        for child in cursor.get_children():
            if _safe_kind(child) == CursorKind.OBJC_SUPER_CLASS_REF:
                superclass = child.spelling
                break

        cls = self._parse_class_members(cursor)
        cls.name = name
        cls.superclass = superclass or "NSObject"
        cls.availability = cursor_availability(cursor)
        return cls

    def _parse_protocol(self, cursor) -> Optional[ObjCClass]:
        """Parse an @protocol as a class wrapper, matching upstream metal-cpp's
        shape (e.g. MTLBuffer protocol → class MTL::Buffer inheriting from
        MTL::Resource). The superclass is the first conformed-to protocol;
        falls back to NSObject when the protocol only conforms to NSObject."""
        name = cursor.spelling
        if not name:
            return None

        cls = self._parse_class_members(cursor)
        cls.name = name
        cls.is_protocol = True
        # _parse_class_members already populated cls.protocols from
        # OBJC_PROTOCOL_REF children. Pick the first non-NSObject one as base.
        super_proto = next(
            (p for p in cls.protocols if p not in ("NSObject", "NSCopying", "NSSecureCoding")),
            "",
        )
        cls.superclass = super_proto or "NSObject"
        cls.availability = cursor_availability(cursor)
        return cls

    def _parse_property(self, cursor) -> Optional[ObjCProperty]:
        name = cursor.spelling
        if not name:
            return None

        objc_type = cursor.type.spelling
        tokens = [t.spelling for t in cursor.get_tokens()]
        is_readonly = "readonly" in tokens
        is_class = "class" in tokens

        return ObjCProperty(
            name=name,
            objc_type=objc_type,
            is_readonly=is_readonly,
            is_class_property=is_class,
            availability=cursor_availability(cursor),
        )

    def _parse_method(self, cursor, is_class: bool) -> Optional[ObjCMethod]:
        selector = cursor.spelling
        if not selector:
            return None

        return_type = cursor.result_type.spelling

        params = []
        for child in cursor.get_children():
            if _safe_kind(child) == CursorKind.PARM_DECL:
                params.append(
                    ObjCParam(
                        name=child.spelling or f"param{len(params)}",
                        objc_type=child.type.spelling,
                    )
                )

        return ObjCMethod(
            selector=selector,
            return_type=return_type,
            params=params,
            is_class_method=is_class,
            availability=cursor_availability(cursor),
        )

    def _parse_enum(self, cursor) -> Optional[ObjCEnum]:
        name = cursor.spelling
        is_anonymous = (not name) or name.startswith("enum ") or "unnamed" in name
        if is_anonymous:
            # Anonymous `NS_ENUM(NSStringEncoding) { ... }` — Apple's one-arg
            # macro form: the underlying typedef *is* the logical enum name.
            # Synthesize the name from it so the emitter produces a real
            # `_NS_ENUM(NS::UInteger, StringEncoding) { ... }` instead of
            # loose `inline constexpr` constants. Underlyings that are
            # primitive integer aliases (NSUInteger, NSInteger) stay
            # anonymous — those rarely carry a logical group identity.
            children = [ch for ch in cursor.get_children()
                         if _safe_kind(ch) == CursorKind.ENUM_CONSTANT_DECL]
            if not children:
                return None
            values = []
            for child in children:
                try:
                    v = child.enum_value
                except (ValueError, AttributeError):
                    v = None
                values.append(ObjCEnumValue(
                    name=child.spelling, value=v,
                    value_expr=_enum_value_expr(child),
                    availability=cursor_availability(child)))
            underlying_spelling = cursor.enum_type.spelling if cursor.enum_type else "NSUInteger"
            synth_name = ""
            underlying = underlying_spelling
            if (re.fullmatch(r"[A-Z]\w*", underlying_spelling)
                    and underlying_spelling not in {"NSUInteger", "NSInteger", "NSTimeInterval"}):
                synth_name = underlying_spelling
                # Walk one typedef hop to get the real integer type for the
                # macro's first arg (NSStringEncoding → NSUInteger).
                try:
                    typedef_decl = cursor.enum_type.get_declaration()
                    if _safe_kind(typedef_decl) == CursorKind.TYPEDEF_DECL:
                        underlying = typedef_decl.underlying_typedef_type.spelling
                except (AttributeError, ValueError):
                    pass
            return ObjCEnum(name=synth_name, underlying_type=underlying,
                            values=values,
                            availability=cursor_availability(cursor))
        # Skip forward declarations: clang emits two ENUM_DECL cursors for a
        # typedef NS_ENUM (one forward, one with the body). Take only the one
        # with constants so collect_enum sees the populated values.
        if not any(_safe_kind(ch) == CursorKind.ENUM_CONSTANT_DECL
                   for ch in cursor.get_children()):
            return None

        underlying = cursor.enum_type.spelling if cursor.enum_type else "unsigned long"

        values = []
        for child in cursor.get_children():
            if _safe_kind(child) == CursorKind.ENUM_CONSTANT_DECL:
                values.append(
                    ObjCEnumValue(
                        name=child.spelling,
                        value=child.enum_value,
                        value_expr=_enum_value_expr(child),
                        availability=cursor_availability(child),
                    )
                )

        return ObjCEnum(name=name, underlying_type=underlying, values=values,
                        availability=cursor_availability(cursor))


# ── Code generation ──────────────────────────────────────────────────────


# Special-case emission for NS::Object — the CRTP root. Was originally
# kept verbatim from Apple's metal-cpp; now generated so the directly-
# named selectors (retain/release/autorelease/copy/hash/description/…) go
# through the same `_objc_msgSend$<sel>` stub trampolines every other
# wrapper uses. `sendMessage` / `sendMessageSafe` remain as an escape
# hatch for arbitrary-selector dispatch (used by custom code that needs
# to call a selector not in the wrapped surface).
_NSOBJECT_HPP = """\
#pragma once

// NS::Object — the CRTP root. Emitted by tools/generate.py; was formerly
// copied verbatim from Apple's metal-cpp/Foundation/NSObject.hpp.
//
// Directly-named selectors (retain/release/autorelease/retainCount/copy/
// hash/isEqual:/description/debugDescription/init/alloc/
// respondsToSelector:/methodSignatureForSelector:) dispatch through the
// linker-synthesized `_objc_msgSend$<sel>` trampolines — same shape every
// other generated class uses. `sendMessage` / `sendMessageSafe` are kept
// for callers that need to dispatch an arbitrary runtime SEL.

#include "NSDefines.hpp"
#include "NSTypes.hpp"
#include "NSBridge.hpp"

#include <objc/message.h>
#include <objc/runtime.h>

#include <type_traits>

namespace NS
{
class Object;
class String;
class MethodSignature;
} // namespace NS

namespace NS
{
template <class _Class, class _Base = class Object>
class _NS_EXPORT Referencing : public _Base
{
public:
    _Class*  retain();
    void     release();
    _Class*  autorelease();
    UInteger retainCount() const;
};

template <class _Class, class _Base = class Object>
class Copying : public Referencing<_Class, _Base>
{
public:
    _Class* copy() const;
};

template <class _Class, class _Base = class Object>
class SecureCoding : public Referencing<_Class, _Base>
{
};

class Object : public Referencing<Object, objc_object>
{
public:
    UInteger      hash() const;
    bool          isEqual(const Object* pObject) const;

    class String* description() const;
    class String* debugDescription() const;

    template <typename _Ret, typename... _Args>
    static _Ret sendMessage(const void* pObj, SEL selector, _Args... args);

    template <typename _Ret, typename... _Args>
    static _Ret sendMessageSafe(const void* pObj, SEL selector, _Args... args);

protected:
    friend class Referencing<Object, objc_object>;

    template <class _Class>
    static _Class* alloc(const char* pClassName);
    template <class _Class>
    static _Class* alloc(const void* pClass);
    template <class _Class>
    _Class* init();

    template <class _Dst>
    static _Dst                   bridgingCast(const void* pObj);
    static class MethodSignature* methodSignatureForSelector(const void* pObj, SEL selector);
    static bool                   respondsToSelector(const void* pObj, SEL selector);
    template <typename _Type>
    static constexpr bool doesRequireMsgSendStret();

private:
    Object() = delete;
    Object(const Object&) = delete;
    ~Object() = delete;

    Object& operator=(const Object&) = delete;
};
} // namespace NS

// --- Inline implementations ---

template <class _Class, class _Base>
_NS_INLINE _Class* NS::Referencing<_Class, _Base>::retain()
{
    return reinterpret_cast<_Class*>(_NS_msg_NSObject_retain((const void*)this, nullptr));
}

template <class _Class, class _Base>
_NS_INLINE void NS::Referencing<_Class, _Base>::release()
{
    _NS_msg_NSObject_release((const void*)this, nullptr);
}

template <class _Class, class _Base>
_NS_INLINE _Class* NS::Referencing<_Class, _Base>::autorelease()
{
    return reinterpret_cast<_Class*>(_NS_msg_NSObject_autorelease((const void*)this, nullptr));
}

template <class _Class, class _Base>
_NS_INLINE NS::UInteger NS::Referencing<_Class, _Base>::retainCount() const
{
    return _NS_msg_NSObject_retainCount((const void*)this, nullptr);
}

template <class _Class, class _Base>
_NS_INLINE _Class* NS::Copying<_Class, _Base>::copy() const
{
    return reinterpret_cast<_Class*>(_NS_msg_NSObject_copy((const void*)this, nullptr));
}

template <class _Dst>
_NS_INLINE _Dst NS::Object::bridgingCast(const void* pObj)
{
#ifdef __OBJC__
    return (__bridge _Dst)pObj;
#else
    return (_Dst)pObj;
#endif // __OBJC__
}

template <typename _Type>
_NS_INLINE constexpr bool NS::Object::doesRequireMsgSendStret()
{
#if (defined(__i386__) || defined(__x86_64__))
    constexpr size_t kStructLimit = (sizeof(std::uintptr_t) << 1);
    return sizeof(_Type) > kStructLimit;
#elif defined(__arm64__)
    return false;
#elif defined(__arm__)
    constexpr size_t kStructLimit = sizeof(std::uintptr_t);
    return std::is_class_v<_Type> && (sizeof(_Type) > kStructLimit);
#else
#error "Unsupported architecture!"
#endif
}

template <>
_NS_INLINE constexpr bool NS::Object::doesRequireMsgSendStret<void>()
{
    return false;
}

// `sendMessage` keeps the upstream architecture-aware dispatch: x86 uses
// objc_msgSend_fpret for floating-point returns and objc_msgSend_stret
// for large structs, arm64 always uses regular objc_msgSend.
template <typename _Ret, typename... _Args>
_NS_INLINE _Ret NS::Object::sendMessage(const void* pObj, SEL selector, _Args... args)
{
#if (defined(__i386__) || defined(__x86_64__))
    if constexpr (std::is_floating_point<_Ret>())
    {
        using SendMessageProcFpret = _Ret (*)(const void*, SEL, _Args...);
        const SendMessageProcFpret pProc = reinterpret_cast<SendMessageProcFpret>(&objc_msgSend_fpret);
        return (*pProc)(pObj, selector, args...);
    }
    else
#endif
#if !defined(__arm64__)
        if constexpr (doesRequireMsgSendStret<_Ret>())
    {
        using SendMessageProcStret = void (*)(_Ret*, const void*, SEL, _Args...);
        const SendMessageProcStret pProc = reinterpret_cast<SendMessageProcStret>(&objc_msgSend_stret);
        _Ret                       ret;
        (*pProc)(&ret, pObj, selector, args...);
        return ret;
    }
    else
#endif
    {
        using SendMessageProc = _Ret (*)(const void*, SEL, _Args...);
        const SendMessageProc pProc = reinterpret_cast<SendMessageProc>(&objc_msgSend);
        return (*pProc)(pObj, selector, args...);
    }
}

_NS_INLINE NS::MethodSignature* NS::Object::methodSignatureForSelector(const void* pObj, SEL selector)
{
    return _NS_msg_NSObject_methodSignatureForSelector_(pObj, nullptr, selector);
}

_NS_INLINE bool NS::Object::respondsToSelector(const void* pObj, SEL selector)
{
    return _NS_msg_NSObject_respondsToSelector_(pObj, nullptr, selector);
}

template <typename _Ret, typename... _Args>
_NS_INLINE _Ret NS::Object::sendMessageSafe(const void* pObj, SEL selector, _Args... args)
{
    if ((respondsToSelector(pObj, selector)) || (nullptr != methodSignatureForSelector(pObj, selector)))
    {
        return sendMessage<_Ret>(pObj, selector, args...);
    }

    if constexpr (!std::is_void<_Ret>::value)
    {
        return _Ret(0);
    }
}

template <class _Class>
_NS_INLINE _Class* NS::Object::alloc(const char* pClassName)
{
    // objc_lookUpClass returns `Class` (objc_class*). Under ObjC ARC,
    // bridging a Class to a non-retainable pointer needs `__bridge`;
    // in pure C++ translation units the macro expands to nothing.
#if __has_feature(objc_arc)
    const void* cls = (__bridge const void*)objc_lookUpClass(pClassName);
#else
    const void* cls = (const void*)objc_lookUpClass(pClassName);
#endif
    return reinterpret_cast<_Class*>(_NS_msg_NSObject_alloc(cls, nullptr));
}

template <class _Class>
_NS_INLINE _Class* NS::Object::alloc(const void* pClass)
{
    return reinterpret_cast<_Class*>(_NS_msg_NSObject_alloc(pClass, nullptr));
}

template <class _Class>
_NS_INLINE _Class* NS::Object::init()
{
    return reinterpret_cast<_Class*>(_NS_msg_NSObject_init((const void*)this, nullptr));
}

_NS_INLINE NS::UInteger NS::Object::hash() const
{
    return _NS_msg_NSObject_hash((const void*)this, nullptr);
}

_NS_INLINE bool NS::Object::isEqual(const Object* pObject) const
{
    return _NS_msg_NSObject_isEqual_((const void*)this, nullptr, pObject);
}

_NS_INLINE NS::String* NS::Object::description() const
{
    return _NS_msg_NSObject_description((const void*)this, nullptr);
}

_NS_INLINE NS::String* NS::Object::debugDescription() const
{
    return _NS_msg_NSObject_debugDescription((const void*)this, nullptr);
}
"""


# Special-case emission for NSEnumerator.hpp — used to be kept verbatim
# because libclang can't surface `template <class _ObjectType>` methods
# on the wrapped protocol. Emitted by the generator so it uses the same
# stub trampolines as everything else; drops the `_NS_PRIVATE_SEL`
# dependency entirely.
_NSENUMERATOR_HPP = """\
#pragma once

// NS::Enumerator / NS::FastEnumeration — emitted by tools/generate.py
// (used to live verbatim under metal-cpp-apple). The kept-upstream
// version routed through `_NS_PRIVATE_SEL`; this version dispatches
// directly through the linker-synthesized `_objc_msgSend$<sel>` stubs
// so it stops pulling Apple's selector-registration machinery into the
// tree.

#include "NSDefines.hpp"
#include "NSObject.hpp"
#include "NSTypes.hpp"
#include "NSBridge.hpp"

namespace NS
{
class Array;
class FastEnumeration;
template <class _ObjectType> class Enumerator;
} // namespace NS

namespace NS
{
struct FastEnumerationState
{
    unsigned long  state;
    Object**       itemsPtr;
    unsigned long* mutationsPtr;
    unsigned long  extra[5];
} _NS_PACKED;

class FastEnumeration : public Referencing<FastEnumeration>
{
public:
    NS::UInteger countByEnumerating(FastEnumerationState* pState, Object** pBuffer, NS::UInteger len);
};

template <class _ObjectType>
class Enumerator : public Referencing<Enumerator<_ObjectType>, FastEnumeration>
{
public:
    _ObjectType* nextObject();
    class Array* allObjects();
};
} // namespace NS

// --- Inline implementations ---

_NS_INLINE NS::UInteger NS::FastEnumeration::countByEnumerating(
    FastEnumerationState* pState, Object** pBuffer, NS::UInteger len)
{
    return _NS_msg_NSFastEnumeration_countByEnumeratingWithState_objects_count_(
        (const void*)this, nullptr, pState, pBuffer, len);
}

template <class _ObjectType>
_NS_INLINE _ObjectType* NS::Enumerator<_ObjectType>::nextObject()
{
    return reinterpret_cast<_ObjectType*>(
        _NS_msg_NSEnumerator_nextObject((const void*)this, nullptr));
}

template <class _ObjectType>
_NS_INLINE NS::Array* NS::Enumerator<_ObjectType>::allObjects()
{
    return _NS_msg_NSEnumerator_allObjects((const void*)this, nullptr);
}
"""


class CodeGenerator:
    """Generate metal-cpp style C++ wrappers for a single framework."""

    # Namespace → output directory. MTL4 shares Metal/ with MTL so a
    # cross-namespace include from MTL to MTL4 (or vice versa) is a
    # same-dir reference, not `../OtherDir/`.
    FW_NS_TO_DIR = {"NS": "Foundation", "MTL": "Metal", "MTL4": "Metal",
                    "MTLFX": "MetalFX", "MTL4FX": "MetalFX",
                    "CA": "QuartzCore"}

    # NS:: typedefs and value types that look like class refs to the
    # `<NS|MTL|...>::<Name>` regex sweep but must NOT be forward-declared
    # as classes (they're primitives / aliases owned by NSTypes.hpp).
    NOT_A_CLASS = {
        "Integer", "UInteger", "TimeInterval", "Range", "ComparisonResult",
        "Comparator", "ErrorDomain", "ErrorUserInfoKey", "DecimalNumber",
    }

    def __init__(
        self,
        namespace: str,
        prefix: str,
        strip_prefix: str,
        resolver: TypeResolver,
    ) -> None:
        self.ns = namespace
        self.prefix = prefix
        self.strip_prefix = strip_prefix
        self.resolver = resolver

        # When False, the corresponding `API_AVAILABLE(...)` annotations are
        # dropped from the generated source. `types` covers class / enum /
        # struct type declarations; `members` covers methods, property
        # getters/setters, and enum values. Both default off (matches the CLI
        # flag default in generate.py).
        self.emit_availability_types: bool = False
        self.emit_availability_members: bool = False

        # Accumulated across all classes for Private.hpp
        self.all_selectors: dict[str, str] = {}  # accessor → ObjC selector string
        self.all_classes: set[str] = set()  # ObjC class names
        self.all_enums: list[ObjCEnum] = []  # in declaration order
        self._enum_names_seen: set[str] = set()
        self.all_blocks: list[ObjCBlockTypedef] = []
        self._block_names_seen: set[str] = set()
        self.all_structs: list[ObjCStruct] = []
        self._struct_names_seen: set[str] = set()
        # `using Coordinate2D = MTL::SamplePosition;` style aliases —
        # `typedef MTLSamplePosition MTLCoordinate2D` in the SDK, emitted
        # alongside the underlying struct in `<P>Structs.hpp` so consumers
        # see the alias name (`MTL::Coordinate2D`) the public API uses.
        # List of (alias_cpp_name, target_cpp_qualified).
        self.all_struct_aliases: list[tuple[str, str]] = []
        self._struct_alias_names_seen: set[str] = set()
        self.generated_headers: list[str] = []
        # Upstream-kept header basenames for this framework (no `.hpp`).
        # `<P>Structs.hpp` includes these so generated class headers that
        # reference a `skip: true` struct (e.g. MTL::BufferRange owned by
        # the hand-written MTLAccelerationStructureTypes.hpp) can pull the
        # definition in via the framework's Structs aggregate.
        self.keep_upstream: list[str] = []
        # Umbrella include lines prepended to <Fw>.hpp (lets one framework's
        # umbrella pull in a co-resident sibling's).
        self.extra_umbrella_includes: list[str] = []
        # Output subdirectory this framework writes to (filename of the dir
        # under `metal-cpp/`). Defaults to the framework name; differs for
        # virtual frameworks like Metal4 that share Metal/.
        self.output_subdir: str = ""
        # ObjC class name → SDK header basename (no `.h`). Used by the
        # per-source-header emitter to pick the right `#include` for a
        # superclass that lives in another header.
        self.class_to_source: dict[str, str] = {}
        # Bridge registry: every (return type, arg types, selector) tuple
        # used by any class trampoline in this framework collapses to a
        # single deduped extern "C" decl emitted in `<P>Bridge.hpp`.
        # Identical selectors with the same C++ signature (e.g. `label`
        # returning `NS::String*` across N classes) share one entry.
        self._bridge_by_sig: dict[tuple, str] = {}  # (ret, args, sel) → name
        self._bridge_by_name: dict[str, tuple] = {}  # name → sig (collision check)
        self._bridge_entries: list[tuple[str, str, list[str], str]] = []  # (name, ret, args, sel)
        self._bridge_call_count: int = 0  # call sites — pre-dedup extern count

    @property
    def has_blocks(self) -> bool:
        """True when `<Prefix>Blocks.hpp` would carry any block typedef."""
        return bool(self.all_blocks)

    @property
    def has_structs(self) -> bool:
        """True when `<Prefix>Structs.hpp` would carry any struct or
        struct-alias `using` declaration."""
        return bool(self.all_structs) or bool(self.all_struct_aliases)

    @property
    def needs_availability_header(self) -> bool:
        """True when any emitted file references the `API_AVAILABLE` macro
        family — i.e. at least one of the two availability flags is on."""
        return self.emit_availability_types or self.emit_availability_members

    def _av_type(self, av: Availability) -> str:
        """Return ` API_AVAILABLE(...)` for a type declaration (class /
        enum / struct), or empty when type-level availability is disabled."""
        if not self.emit_availability_types:
            return ""
        s = format_availability(av)
        return f" {s}" if s else ""

    def _av_member(self, av: Availability) -> str:
        """Return ` API_AVAILABLE(...)` for a member (method / property /
        enum value), or empty when member-level availability is disabled."""
        if not self.emit_availability_members:
            return ""
        s = format_availability(av)
        return f" {s}" if s else ""

    def collect_enum(self, enum: ObjCEnum) -> None:
        # Anonymous enums have empty `name` — they all need to flow through
        # to emit their constants. Only dedupe named enums.
        if enum.name and enum.name in self._enum_names_seen:
            return
        if enum.name:
            self._enum_names_seen.add(enum.name)
        self.all_enums.append(enum)

    def collect_block(self, block: ObjCBlockTypedef) -> None:
        if block.name in self._block_names_seen:
            return
        self._block_names_seen.add(block.name)
        self.all_blocks.append(block)

    def is_block(self, objc_type: str) -> bool:
        """True iff `objc_type` is one of our generated block typedefs."""
        return objc_type.strip() in self._block_names_seen

    def collect_struct(self, s: ObjCStruct) -> None:
        if s.name in self._struct_names_seen:
            return
        self._struct_names_seen.add(s.name)
        self.all_structs.append(s)

    def collect_struct_alias(self, alias_cpp: str, target_cpp_qual: str) -> None:
        if alias_cpp in self._struct_alias_names_seen:
            return
        self._struct_alias_names_seen.add(alias_cpp)
        self.all_struct_aliases.append((alias_cpp, target_cpp_qual))

    # ── Helpers ────────────────────────────────────────────────────────

    def cpp_class_name(self, objc_name: str) -> str:
        """Strip ObjC prefix to get C++ class name (e.g. NSScreen → Screen)."""
        return strip_objc_prefix(objc_name, self.strip_prefix)

    def _resolve(self, objc_type: str, cls_name: str = "", context: str = "") -> str:
        """Resolve type, handling instancetype → concrete class."""
        cpp_name = self.cpp_class_name(cls_name) if cls_name else ""
        return resolve_type(self.resolver, objc_type, self.ns, cpp_name)

    def _system_includes(self, resolved_types: set[str]) -> list[str]:
        """Determine system #include directives needed for the given resolved types."""
        headers = set()
        for t in resolved_types:
            # Strip pointer/const to get base type
            base = t.rstrip("*").strip()
            if base.startswith("const "):
                base = base[6:].strip()
            if base in SYSTEM_HEADER_FOR_TYPE:
                headers.add(SYSTEM_HEADER_FOR_TYPE[base])
        return sorted(headers)

    # ── Selector collection ───────────────────────────────────────────

    def collect_selectors(self, cls: ObjCClass) -> None:
        """Collect selector/class registrations for Private.hpp."""
        self.all_classes.add(cls.name)

        for prop in cls.properties:
            self.all_selectors[prop.name] = prop.name
            if not prop.is_readonly:
                sname = setter_name(prop.name)
                self.all_selectors[f"{sname}_"] = f"{sname}:"

        for method in cls.methods:
            self.all_selectors[method.sel_accessor] = method.selector

    # ── File generators ───────────────────────────────────────────────

    def generate_bridge_header(self) -> str:
        """Emit `<P>Bridge.hpp`: one extern "C" trampoline decl per unique
        (return type, arg types, selector) tuple recorded for this
        framework. Per-class headers include this file in place of carrying
        their own externs — a `label` returning `NS::String*` across N
        classes collapses to one decl here.

        Cross-namespace types referenced in any signature are forward-
        declared (classes) or opaque-enum-declared (enums); structs and
        blocks need full definitions so the framework's own `<P>Structs.hpp`
        / `<P>Blocks.hpp` are included when populated, and sibling
        frameworks' aggregates are pulled in for cross-framework references.
        """
        p = self.prefix
        ns = self.ns
        ns_root = "" if ns == "NS" else "../Foundation/"

        lines = [
            "#pragma once",
            "",
            "// Consolidated extern \"C\" trampoline decls for this framework.",
            "// One entry per (return, args, selector) — identical C++ signatures",
            "// across multiple classes collapse to a single linker alias of",
            "// `_objc_msgSend$<selector>`. Per-class headers include this file",
            "// instead of declaring their own externs.",
            "",
            f'#include "{p}Defines.hpp"',
            # `SEL` is the second parameter of every trampoline.
            "#include <objc/objc.h>",
            # `NS::UInteger`, `NS::Object`, etc. flow through here. NS framework
            # already gets them from its own NSTypes.hpp; sibling frameworks
            # need the relative path.
            f'#include "{ns_root}NSTypes.hpp"',
        ]
        if self.has_blocks:
            lines.append(f'#include "{p}Blocks.hpp"')
        if self.has_structs:
            lines.append(f'#include "{p}Structs.hpp"')

        # Walk every recorded signature once to learn which other-namespace
        # classes / enums / aggregates we touch and need to surface here.
        sig_types: set[str] = set()
        for _name, ret, args, _sel in self._bridge_entries:
            sig_types.add(ret)
            sig_types.update(args)

        fwd_decls: dict[str, set[str]] = {}
        fwd_template_decls: dict[str, set[str]] = {}  # template forward decls
        fwd_enums: dict[str, dict[str, tuple[str, bool]]] = {}
        cross_fw_includes: set[str] = set()
        # Class templates — declared as template forward decls inline so
        # the bridge stays a leaf header (including their real definitions
        # would cycle: NSEnumerator.hpp itself includes NSBridge.hpp).
        TEMPLATE_CLASSES = {("NS", "Enumerator"): "_ObjectType"}
        for resolved in sig_types:
            for ns_name, cpp_cls in re.findall(
                    r"\b(NS|MTL4FX|MTL4|MTLFX|MTL|CA)::(\w+)", resolved):
                if cpp_cls in self.NOT_A_CLASS:
                    continue
                tparam = TEMPLATE_CLASSES.get((ns_name, cpp_cls))
                if tparam:
                    fwd_template_decls.setdefault(ns_name, set()).add(
                        f"template <class {tparam}> class {cpp_cls};")
                    continue
                qual = f"{ns_name}::{cpp_cls}"
                kind = self.resolver.kinds.get(qual, "class")
                if kind == "enum":
                    underlying = self.resolver.enum_underlying.get(qual)
                    if underlying:
                        is_options = self.resolver.enum_is_options.get(qual, False)
                        fwd_enums.setdefault(ns_name, {})[cpp_cls] = (underlying, is_options)
                    continue
                if kind == "typedef":
                    # Typedef/using aliases flow through Structs (e.g.
                    # `MTL::GPUAddress` = `uint64_t`) or appear later in
                    # a per-class header. The bridge can't forward-declare
                    # them — `class GPUAddress;` would clash with the
                    # actual `using` definition. String-typedef aliases
                    # have already been rewritten to their underlying type
                    # in `_stub`, so anything that reaches here surfaces
                    # via Structs and is already visible.
                    continue
                if kind in ("block", "struct"):
                    # Cross-framework aggregates need a real include — their
                    # definitions can't be forward-declared.
                    if ns_name != self.ns:
                        suffix = "Blocks" if kind == "block" else "Structs"
                        fw_dir = self.FW_NS_TO_DIR.get(ns_name, ns_name)
                        fw_prefix = self.resolver.frameworks.get(qual, ns_name)
                        cross_fw_includes.add(
                            f'#include "../{fw_dir}/{fw_prefix}{suffix}.hpp"')
                    continue
                # Plain class — forward decl is enough since the bridge only
                # uses pointers to it.
                fwd_decls.setdefault(ns_name, set()).add(cpp_cls)

        for inc in sorted(cross_fw_includes):
            lines.append(inc)

        # System headers referenced by signature types (e.g. <cstdint> for
        # uint32_t). The class-header emission path uses _system_includes
        # for the same purpose; reuse it here.
        for header in self._system_includes(sig_types):
            lines.append(f"#include <{header}>")

        if fwd_decls or fwd_enums or fwd_template_decls:
            lines.append("")
            all_ns = sorted(set(fwd_decls) | set(fwd_enums) | set(fwd_template_decls))
            for ns_name in all_ns:
                lines.append(f"namespace {ns_name} {{")
                for cls_fwd in sorted(fwd_decls.get(ns_name, set())):
                    lines.append(f"    class {cls_fwd};")
                for tmpl in sorted(fwd_template_decls.get(ns_name, set())):
                    lines.append(f"    {tmpl}")
                for enum_name, (underlying, is_options) in sorted(
                        fwd_enums.get(ns_name, {}).items()):
                    lines.append(self._format_enum_fwd_decl(
                        enum_name, underlying, is_options, indent="    "))
                lines.append("}")

        lines += [
            "",
            "#pragma clang diagnostic push",
            '#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"',
            # Bridge trampolines reference availability-attributed types as
            # opaque pointers (e.g. `MTL::ResidencySet*`, macOS 15+) without
            # restating the attribute per decl. The per-class wrapper that
            # dispatches through a trampoline carries the public
            # `API_AVAILABLE`, so a call site on an older OS still warns at
            # the right level. Scoped to this block by the surrounding
            # push/pop — clients of the bridge see their original setting.
            '#pragma clang diagnostic ignored "-Wunguarded-availability-new"',
            "",
            'extern "C" {',
        ]
        # Sorting by selector then deduped name keeps emission deterministic
        # and groups all overloads of the same selector together — easier
        # to skim when something looks off.
        for name, ret_cpp, arg_types, selector in sorted(
                self._bridge_entries, key=lambda e: (e[3], e[0])):
            params = ["const void*", "SEL"] + arg_types
            lines.append(
                f'{ret_cpp} {name}'
                f'({", ".join(params)}) __asm__("_objc_msgSend$" "{selector}");'
            )
        lines += [
            "} // extern \"C\"",
            "",
            "#pragma clang diagnostic pop",
            "",
        ]
        return "\n".join(lines)

    def generate_nsobject_header(self) -> str:
        """Emit Foundation/NSObject.hpp — the CRTP root that every other
        wrapper inherits from. Registers every directly-named NSObject
        selector with the bridge registry so the inline impls below pick
        up the deduped trampoline names emitted in `NSBridge.hpp`."""
        # Pre-register signatures. Names returned here are substituted into
        # the static template body so all dispatch goes through the shared
        # bridge externs (no hardcoded `_NS_msg_NSObject_*` decls anymore).
        n_alloc = self._stub("void*", [], "alloc")
        n_init = self._stub("void*", [], "init")
        n_retain = self._stub("void*", [], "retain")
        n_release = self._stub("void", [], "release")
        n_autorelease = self._stub("void*", [], "autorelease")
        n_retainCount = self._stub("NS::UInteger", [], "retainCount")
        n_copy = self._stub("void*", [], "copy")
        n_hash = self._stub("NS::UInteger", [], "hash")
        n_isEqual = self._stub("bool", ["const NS::Object*"], "isEqual:")
        n_description = self._stub("NS::String*", [], "description")
        n_debugDescription = self._stub("NS::String*", [], "debugDescription")
        n_respondsTo = self._stub("bool", ["SEL"], "respondsToSelector:")
        n_methodSig = self._stub("NS::MethodSignature*", ["SEL"],
                                  "methodSignatureForSelector:")
        body = _NSOBJECT_HPP
        # Replace longest placeholders first so prefix-of relationships
        # (`_NS_msg_NSObject_retain` is a prefix of `_NS_msg_NSObject_retainCount`,
        # `_NS_msg_NSObject_description` of `_NS_msg_NSObject_debugDescription`)
        # don't get clobbered by the shorter match.
        replacements = [
            ("_NS_msg_NSObject_alloc", n_alloc),
            ("_NS_msg_NSObject_init", n_init),
            ("_NS_msg_NSObject_retain", n_retain),
            ("_NS_msg_NSObject_release", n_release),
            ("_NS_msg_NSObject_autorelease", n_autorelease),
            ("_NS_msg_NSObject_retainCount", n_retainCount),
            ("_NS_msg_NSObject_copy", n_copy),
            ("_NS_msg_NSObject_hash", n_hash),
            ("_NS_msg_NSObject_isEqual_", n_isEqual),
            ("_NS_msg_NSObject_description", n_description),
            ("_NS_msg_NSObject_debugDescription", n_debugDescription),
            ("_NS_msg_NSObject_respondsToSelector_", n_respondsTo),
            ("_NS_msg_NSObject_methodSignatureForSelector_", n_methodSig),
        ]
        replacements.sort(key=lambda p: -len(p[0]))
        for placeholder, name in replacements:
            body = body.replace(placeholder, name)
        return body

    def generate_nsenumerator_header(self) -> str:
        """Emit Foundation/NSEnumerator.hpp. Like NSObject.hpp, the
        hand-written externs are gone; selectors register with the bridge
        so the inline impls reach them through `NSBridge.hpp`."""
        n_countByEnum = self._stub(
            "NS::UInteger", ["void*", "NS::Object**", "NS::UInteger"],
            "countByEnumeratingWithState:objects:count:")
        n_nextObject = self._stub("void*", [], "nextObject")
        n_allObjects = self._stub("NS::Array*", [], "allObjects")
        body = _NSENUMERATOR_HPP
        replacements = [
            ("_NS_msg_NSFastEnumeration_countByEnumeratingWithState_objects_count_",
             n_countByEnum),
            ("_NS_msg_NSEnumerator_nextObject", n_nextObject),
            ("_NS_msg_NSEnumerator_allObjects", n_allObjects),
        ]
        replacements.sort(key=lambda p: -len(p[0]))
        for placeholder, name in replacements:
            body = body.replace(placeholder, name)
        return body

    def generate_defines(self) -> str:
        p = self.prefix
        return "\n".join([
            "#pragma once",
            "",
            f'#include "../Foundation/NSDefines.hpp"',
            "",
            f"#define _{p}_EXPORT _NS_EXPORT",
            f"#define _{p}_EXTERN _NS_EXTERN",
            f"#define _{p}_INLINE _NS_INLINE",
            f"#define _{p}_PACKED _NS_PACKED",
            "",
            f"#define _{p}_CONST(type, name) _NS_CONST(type, name)",
            f"#define _{p}_ENUM(type, name) _NS_ENUM(type, name)",
            f"#define _{p}_OPTIONS(type, name) _NS_OPTIONS(type, name)",
            "",
            f"#define _{p}_VALIDATE_SIZE(ns, name) _NS_VALIDATE_SIZE(ns, name)",
            f"#define _{p}_VALIDATE_ENUM(ns, name) _NS_VALIDATE_ENUM(ns, name)",
            "",
        ])

    def generate_private(self) -> str:
        """Class-lookup machinery only. Selectors are now dispatched through
        per-method extern "C" decls with __asm__ labels (see emit_stub_decl),
        so there is no global SEL table to emit.
        """
        p = self.prefix
        ns = self.ns
        lines = [
            "#pragma once",
            "",
            f'#include "{p}Defines.hpp"',
            "",
            "#include <objc/runtime.h>",
            "",
            f"#define _{p}_PRIVATE_CLS(symbol) (Private::Class::s_k##symbol)",
            "",
            f"#if defined({p}_PRIVATE_IMPLEMENTATION)",
            "",
            "#ifdef METALCPP_SYMBOL_VISIBILITY_HIDDEN",
            f'#define _{p}_PRIVATE_VISIBILITY __attribute__((visibility("hidden")))',
            "#else",
            f'#define _{p}_PRIVATE_VISIBILITY __attribute__((visibility("default")))',
            "#endif // METALCPP_SYMBOL_VISIBILITY_HIDDEN",
            "",
            "#ifdef __OBJC__",
            f"#define _{p}_PRIVATE_OBJC_LOOKUP_CLASS(symbol) ((__bridge void*)objc_lookUpClass(#symbol))",
            "#else",
            f"#define _{p}_PRIVATE_OBJC_LOOKUP_CLASS(symbol) objc_lookUpClass(#symbol)",
            "#endif // __OBJC__",
            "",
            f"#define _{p}_PRIVATE_DEF_CLS(symbol) "
            f"void* s_k##symbol _{p}_PRIVATE_VISIBILITY = "
            f"_{p}_PRIVATE_OBJC_LOOKUP_CLASS(symbol)",
            "",
            "#else",
            "",
            f"#define _{p}_PRIVATE_DEF_CLS(symbol) extern void* s_k##symbol",
            "",
            f"#endif // {p}_PRIVATE_IMPLEMENTATION",
            "",
        ]

        # Class registrations
        if self.all_classes:
            lines += [
                f"namespace {ns}",
                "{",
                "namespace Private",
                "{",
                "    namespace Class",
                "    {",
            ]
            for cls_name in sorted(self.all_classes):
                lines.append(f"        _{p}_PRIVATE_DEF_CLS({cls_name});")
            lines += [
                "    } // Class",
                "} // Private",
                f"}} // {ns}",
                "",
            ]

        return "\n".join(lines)

    def _collect_enum_fwds(self, resolved_types) -> dict[str, dict[str, tuple[str, bool]]]:
        """Walk an iterable of resolved C++ type strings and return
        `{namespace: {enum_cpp_name: (underlying_type, is_options)}}`
        covering every registered enum found. Used by emitters that need
        to reference enums declared in other source headers without
        pulling those headers through transitively (avoids include
        cycles like `MTLArgument.hpp` ↔ `MTLTexture.hpp`)."""
        fwds: dict[str, dict[str, tuple[str, bool]]] = {}
        for resolved in resolved_types:
            for ns_name, cpp_cls in re.findall(
                    r"\b(NS|MTL4FX|MTL4|MTLFX|MTL|CA)::(\w+)", resolved):
                qual = f"{ns_name}::{cpp_cls}"
                if self.resolver.kinds.get(qual) != "enum":
                    continue
                underlying = self.resolver.enum_underlying.get(qual)
                if not underlying:
                    continue
                is_options = self.resolver.enum_is_options.get(qual, False)
                fwds.setdefault(ns_name, {})[cpp_cls] = (underlying, is_options)
        return fwds

    @staticmethod
    def _format_enum_fwd_decl(cpp_cls: str, underlying: str,
                              is_options: bool, indent: str = "") -> str:
        """Single forward decl. OPTIONS-style enums expand (via
        `_<P>_OPTIONS`) to `using Name = Underlying; enum : Name { … }`
        — declaring them as `enum Name : Underlying;` would clash with
        the type alias. Forward-declare them as the same `using` alias
        the macro produces instead."""
        if is_options:
            return f"{indent}using {cpp_cls} = {underlying};"
        return f"{indent}enum {cpp_cls} : {underlying};"

    def _emit_enum_lines(self, enums: list[ObjCEnum]) -> list[str]:
        """Render a sequence of enums as C++ source lines (no namespace
        wrapper, no `#pragma once`). Used by per-source-header emission;
        the caller embeds these inside its own `namespace <ns> { ... }`
        block. Output style matches upstream metal-cpp's
        `_MTL_ENUM(NS::UInteger, IndexType) { IndexTypeUInt16, ... };`
        form, with the framework prefix stripped from both the enum name
        and its value names.

        Identifiers inside source-form `= <expr>` initializers are
        rewritten with the framework prefix removed (`NSActivityIdle…` →
        `ActivityIdle…`). Expressions that reference anything outside
        `self.all_enums` (SDK `#define` macros, cross-framework symbols)
        fall back to the libclang-evaluated integer literal so the
        emitted header doesn't reference an undeclared identifier."""
        p = self.prefix
        strip = self.strip_prefix
        prefix_re = re.compile(rf"\b{re.escape(strip)}(?=[A-Z])") if strip else None
        known_idents = {
            strip_objc_prefix(v.name, strip)
            for e in self.all_enums for v in e.values
        }

        def _maybe_expr(text: Optional[str]) -> Optional[str]:
            if text is None:
                return None
            stripped = prefix_re.sub("", text) if prefix_re else text
            for m in re.finditer(r"\b[A-Za-z_]\w*\b", stripped):
                if m.group(0) not in known_idents:
                    return None
            return stripped

        lines: list[str] = []
        for enum in enums:
            underlying = self.resolver.resolve(enum.underlying_type) if enum.underlying_type else "NS::UInteger"
            if not enum.name:
                # Anonymous enum (e.g. `NS_ENUM(NSStringEncoding) { ... }`) —
                # the values are global constants of the underlying typedef
                # in the framework namespace.
                for v in enum.values:
                    v_cpp = strip_objc_prefix(v.name, strip)
                    expr = _maybe_expr(v.value_expr)
                    av = self._av_member(v.availability)
                    if expr is not None:
                        lines.append(f"inline constexpr {underlying} {v_cpp}{av} = {expr};")
                    elif v.value is not None:
                        val = f"static_cast<{underlying}>({v.value})" if v.value < 0 else str(v.value)
                        lines.append(f"inline constexpr {underlying} {v_cpp}{av} = {val};")
                lines.append("")
                continue
            cpp_name = strip_objc_prefix(enum.name, strip)
            macro = "OPTIONS" if enum.is_options else "ENUM"
            # Attribute attaches to the closing `};` — the macro form
            # doesn't expose a position between `enum` and the name, and
            # clang rejects a preceding-line attribute on type decls.
            av_close = self._av_type(enum.availability).lstrip()
            lines.append(f"_{p}_{macro}({underlying}, {cpp_name}) {{")
            for v in enum.values:
                v_cpp = strip_objc_prefix(v.name, strip)
                expr = _maybe_expr(v.value_expr)
                # Place the attribute between the enumerator name and the
                # `=`. Trailing position (after the initializer) would let
                # operators in the initializer try to consume the macro —
                # e.g. `1UL << 3 API_AVAILABLE(...)` is parsed as a shift
                # expression and fails.
                av_v = self._av_member(v.availability)
                if expr is not None:
                    lines.append(f"    {v_cpp}{av_v} = {expr},")
                elif v.value is None:
                    lines.append(f"    {v_cpp}{av_v},")
                elif v.value < 0:
                    # Negative literal in an enum backed by an unsigned typedef
                    # would trigger -Wc++11-narrowing. Always cast through the
                    # underlying type — harmless when it's signed, required
                    # when it isn't.
                    lines.append(f"    {v_cpp}{av_v} = static_cast<{underlying}>({v.value}),")
                else:
                    lines.append(f"    {v_cpp}{av_v} = {v.value},")
            close = f"}} {av_close};" if av_close else "};"
            lines += [close, ""]
        return lines

    def generate_blocks(self) -> str:
        """Per-framework <Prefix>Blocks.hpp matching upstream's pattern:
        a `using` alias for the raw Obj-C block and a `Function` alias
        wrapping it in std::function for C++ ergonomics."""
        p = self.prefix
        ns = self.ns
        strip = self.strip_prefix
        ns_root = "" if ns == "NS" else "../Foundation/"
        lines = [
            "#pragma once",
            "",
            f'#include "{p}Defines.hpp"',
            f'#include "{ns_root}NSObjCRuntime.hpp"',
            f'#include "{ns_root}NSTypes.hpp"',
            f'#include "{ns_root}NSRange.hpp"',
            "",
            "#include <functional>",
            "",
        ]
        # Gather every type touched by a block signature once, then split
        # into class refs (forward-declared as `class Foo;`) and enum refs
        # (forward-declared as `enum Foo : Underlying;`). Enums move into
        # the cross-namespace fwd block alongside classes — including the
        # specific declaring `<header>.hpp` would re-introduce the
        # `MTLArgument.hpp` ↔ `MTLTexture.hpp` cycle.
        sig_types: set[str] = set()
        for b in self.all_blocks:
            sig_types.add(self.resolver.resolve(b.return_type))
            for t in b.arg_types:
                sig_types.add(self.resolver.resolve(t))
        cls_refs: set[tuple[str, str]] = set()
        for resolved in sig_types:
            for m_ns, m_cls in re.findall(r"\b(NS|MTL4FX|MTL4|MTLFX|MTL|CA)::(\w+)", resolved):
                if self.resolver.kinds.get(f"{m_ns}::{m_cls}", "class") != "class":
                    continue
                cls_refs.add((m_ns, m_cls))
        cls_by_ns: dict[str, set[str]] = {}
        for ns_name, c in cls_refs:
            cls_by_ns.setdefault(ns_name, set()).add(c)
        enum_fwds = self._collect_enum_fwds(sig_types)
        # Emit cross-namespace enum forward decls outside `namespace {ns}`.
        for ns_name in sorted(enum_fwds):
            if ns_name == ns:
                continue
            lines.append(f"namespace {ns_name} {{")
            for name, (underlying, is_options) in sorted(enum_fwds[ns_name].items()):
                lines.append(self._format_enum_fwd_decl(
                    name, underlying, is_options, indent="    "))
            lines.append("}")
        if any(ns_name != ns for ns_name in enum_fwds):
            lines.append("")
        lines += [f"namespace {ns} {{", ""]
        # Same-ns class forward decls + same-ns enum forward decls go
        # inside the open `namespace {ns}` block.
        for c in sorted(cls_by_ns.get(ns, set())):
            lines.append(f"class {c};")
        for name, (underlying, is_options) in sorted(enum_fwds.get(ns, {}).items()):
            lines.append(self._format_enum_fwd_decl(name, underlying, is_options))
        if cls_by_ns.get(ns) or enum_fwds.get(ns):
            lines.append("")
        # Cross-ns class forward decls.
        for ns_name in sorted(cls_by_ns):
            if ns_name == ns:
                continue
            lines.append(f"}} namespace {ns_name} {{")
            for c in sorted(cls_by_ns[ns_name]):
                lines.append(f"class {c};")
            lines.append(f"}} namespace {ns} {{")
        if any(ns_name != ns for ns_name in cls_by_ns):
            lines.append("")
        for b in self.all_blocks:
            cpp_name = strip_objc_prefix(b.name, strip)
            # Match upstream's convention: `<Name>Block` → `<Name>Function`
            # (replace), otherwise `<Name>Function` (append).
            fn_name = cpp_name[:-len("Block")] + "Function" if cpp_name.endswith("Block") else f"{cpp_name}Function"
            # Resolve types and rewrite any typedef aliases to their
            # underlying spelling — the blocks header is a leaf and can't
            # see the per-header `using` declarations that define those
            # aliases (e.g. `MTL::DeviceNotificationName`).
            ret_cpp = self._unalias_in_type(self.resolver.resolve(b.return_type))
            args_cpp = [self._unalias_in_type(self.resolver.resolve(a))
                        for a in b.arg_types]
            args_joined = ", ".join(args_cpp)
            lines.append(f"using {cpp_name} = {ret_cpp} (^)({args_joined});")
            lines.append(f"using {fn_name} = std::function<{ret_cpp}({args_joined})>;")
            lines.append("")
        lines += [f"}} // {ns}", ""]
        return "\n".join(lines)

    def generate_structs(self) -> str:
        """Per-framework <Prefix>Structs.hpp — plain C-style structs the SDK
        exposes (e.g. MTL4BufferRange). Framework prefix is stripped from the
        struct name (matches upstream metal-cpp: MTLOrigin → MTL::Origin).
        Structs marked packed (either by SDK __attribute__((packed)) or via
        config) get `_<PREFIX>_PACKED` to match upstream's emission."""
        p = self.prefix
        ns = self.ns
        ns_root = "" if ns == "NS" else "../Foundation/"
        lines = ["#pragma once", ""]
        if self.needs_availability_header:
            # `API_AVAILABLE` etc. for per-struct annotations below.
            lines.append("#include <Availability.h>")
        lines += [
            f'#include "{p}Defines.hpp"',
            f'#include "{ns_root}NSTypes.hpp"',
            "",
        ]
        # Emit opaque enum forward decls for every enum referenced by
        # struct field types. Including the declaring `<header>.hpp`
        # would create cycles (e.g. `MTLStructs.hpp` ↔ `MTLTexture.hpp`
        # via `MTL::TextureSwizzle`).
        sig_types: set[str] = set()
        for s in self.all_structs:
            for fld in s.fields:
                sig_types.add(self.resolver.resolve(fld.objc_type))
        enum_fwds = self._collect_enum_fwds(sig_types)
        # Cross-ns enum fwds go before the `namespace {ns}` block.
        for ns_name in sorted(enum_fwds):
            if ns_name == ns:
                continue
            lines.append(f"namespace {ns_name} {{")
            for name, (underlying, is_options) in sorted(enum_fwds[ns_name].items()):
                lines.append(self._format_enum_fwd_decl(
                    name, underlying, is_options, indent="    "))
            lines.append("}")
        if any(ns_name != ns for ns_name in enum_fwds):
            lines.append("")
        lines += [
            f"namespace {ns} {{",
            "",
        ]
        # Same-ns enum fwds inside the namespace block.
        for name, (underlying, is_options) in sorted(enum_fwds.get(ns, {}).items()):
            lines.append(self._format_enum_fwd_decl(name, underlying, is_options))
        if enum_fwds.get(ns):
            lines.append("")
        # Track every struct cpp-name we emit in THIS namespace block so
        # we can split typedef aliases into "target visible here" vs
        # "target lives upstream" (the latter must come after the
        # keep_upstream chain).
        own_struct_cpp_names: set[str] = set()
        # Resolve every field once. A struct that has a by-value member
        # whose type lives in our own namespace but isn't one of the structs
        # we're about to emit must come AFTER the keep_upstream `#include`
        # chain — by elimination, the only way the resolver knows the name
        # is because it was registered for an upstream-kept hpp (e.g.
        # MTL::PackedFloat4x3 from MTLAccelerationStructureTypes.hpp).
        # Cross-namespace refs (`NS::UInteger`) are always fine — NSTypes
        # and sibling-framework headers are pulled in at the top of this
        # file.
        all_struct_cpp_names: set[str] = set()
        for s in self.all_structs:
            base = s.name.lstrip("_")
            all_struct_cpp_names.add(
                strip_objc_prefix_aggressive(base, self.strip_prefix))
        # Forward-declared enums in our own namespace (emitted above the
        # struct block) are usable by-value too — count them as available.
        available_own_names = all_struct_cpp_names | set(enum_fwds.get(ns, {}))
        resolved_fields: dict[int, list[tuple[str, str, Optional[int]]]] = {}
        deferred: set[int] = set()
        own_ns_re = re.compile(rf"\b{re.escape(ns)}::(\w+)")
        for s in self.all_structs:
            rfields: list[tuple[str, str, Optional[int]]] = []
            defer = False
            for fld in s.fields:
                resolved = self.resolver.resolve(fld.objc_type)
                for m in own_ns_re.finditer(resolved):
                    if m.group(1) not in available_own_names:
                        defer = True
                rfields.append((resolved, fld.name, fld.array_size))
            resolved_fields[id(s)] = rfields
            if defer:
                deferred.add(id(s))

        def _emit_struct(s: "ObjCStruct") -> None:
            base = s.name.lstrip("_")
            cpp_name = strip_objc_prefix_aggressive(base, self.strip_prefix)
            own_struct_cpp_names.add(cpp_name)
            av = self._av_type(s.availability).lstrip()
            if av:
                lines.append(av)
            lines.append(f"struct {cpp_name} {{")
            if s.extra_members:
                lines.append(s.extra_members.rstrip())
            for field_type, fname, asize in resolved_fields[id(s)]:
                dim = f"[{asize}]" if asize is not None else ""
                lines.append(f"    {field_type} {fname}{dim};")
            suffix = f" _{p}_PACKED" if s.packed else ""
            lines.extend([f"}}{suffix};", ""])

        for s in self.all_structs:
            if id(s) not in deferred:
                _emit_struct(s)

        def _emit_alias(alias_cpp: str, target_cpp_qual: str) -> str:
            # Same-namespace target: strip the `<ns>::` prefix so the
            # using reads `using Coordinate2D = SamplePosition;` rather
            # than the redundantly qualified form.
            t = target_cpp_qual
            if t.startswith(f"{ns}::"):
                t = t[len(ns) + 2:]
            return f"using {alias_cpp} = {t};"

        # Aliases whose target is defined right above (own struct) — emit
        # before the keep_upstream chain so `<P>Bridge.hpp` (transitively
        # included via keep_upstream) sees them by value.
        own_aliases = [a for a in self.all_struct_aliases
                       if a[1].startswith(f"{ns}::")
                       and a[1][len(ns) + 2:] in own_struct_cpp_names]
        upstream_aliases = [a for a in self.all_struct_aliases
                            if a not in own_aliases]
        for alias_cpp, target in own_aliases:
            lines.append(_emit_alias(alias_cpp, target))
        if own_aliases:
            lines.append("")
        lines += [f"}} // {ns}", ""]
        # Pull in upstream-kept headers AFTER the struct definitions. Some
        # upstream files (notably MTLAccelerationStructureTypes.hpp) include
        # per-class headers that in turn include `<P>Bridge.hpp` — and the
        # bridge needs `MTL::Size` / `Origin` / `Region` already visible
        # when its extern decls reference them by value. Defining the
        # structs first breaks that cycle.
        for keep in self.keep_upstream:
            lines.append(f'#include "{keep}.hpp"')
        if self.keep_upstream:
            lines.append("")
        # Structs (and aliases) whose member types come from the
        # keep_upstream chain only become visible once it's been included.
        # Emit them in a follow-on namespace block.
        deferred_structs = [s for s in self.all_structs if id(s) in deferred]
        if deferred_structs or upstream_aliases:
            lines += [f"namespace {ns} {{", ""]
            for s in deferred_structs:
                _emit_struct(s)
            for alias_cpp, target in upstream_aliases:
                lines.append(_emit_alias(alias_cpp, target))
            lines += ["", f"}} // {ns}", ""]
        return "\n".join(lines)

    # ── Bridge-trampoline registry ───────────────────────────────────
    # Every generated method body calls into `_<P>_msg_<sig>_<sel>` — an
    # `extern "C"` decl with an `__asm__` label that resolves to the
    # linker-synthesized `_objc_msgSend$<sel>` trampoline. Names are keyed
    # by (return type, arg types, selector) so identical C++ signatures
    # collapse to one decl across every class that calls the same selector.
    # The deduped externs all live in a single per-framework `<P>Bridge.hpp`
    # included by each per-class header.

    @staticmethod
    def _mangle_type_id(t: str) -> str:
        """C++ type → identifier-safe fragment. Each non-identifier char
        maps to a distinct escape so two distinct C++ types never collide."""
        if not t or t == "void":
            return "v"
        s = re.sub(r"\s+", "", t)
        table = {"*": "p", "&": "r", ":": "_", "<": "L", ">": "G", ",": "C"}
        out = []
        for ch in s:
            if ch.isalnum() or ch == "_":
                out.append(ch)
            else:
                out.append(table.get(ch, "_"))
        return "".join(out)

    @staticmethod
    def _mangle_selector_id(selector: str) -> str:
        """ObjC selector → identifier-safe fragment. Trailing `:` (which
        every parameterized selector has) becomes a trailing `_`, and
        embedded `:` between segments collapses the same way."""
        return selector.replace(":", "_")

    def _unalias_in_type(self, t: str) -> str:
        """Substitute every recorded string-typedef alias in `t` with its
        underlying C++ type. The bridge header can't see the per-header
        `using` definitions, so we rewrite signatures eagerly to the
        underlying form (`NS::ErrorDomain` → `NS::String*`)."""
        aliases = self.resolver.alias_underlying
        if not aliases:
            return t
        # Replace longest alias names first so a longer key isn't shadowed
        # by a partial match (e.g. `NS::NotificationName` vs `NS::Name`).
        for alias in sorted(aliases, key=len, reverse=True):
            if alias in t:
                t = t.replace(alias, aliases[alias])
        return t

    def _stub(self, ret_cpp: str, arg_types_cpp: list[str], selector: str) -> str:
        """Register a (return, args, selector) trampoline and return its
        deduped extern "C" name. Called wherever a per-class inline impl
        needs to dispatch through a selector stub.

        Receiver is always `const void*` and `SEL` is always the second arg
        (matches upstream metal-cpp's Object::sendMessage shape, and stays
        legal under ObjC ARC since raw pointer ↔ id bridging isn't needed).
        """
        ret_cpp = self._unalias_in_type(ret_cpp)
        arg_types_cpp = [self._unalias_in_type(a) for a in arg_types_cpp]
        self._bridge_call_count += 1
        key = (ret_cpp, tuple(arg_types_cpp), selector)
        existing = self._bridge_by_sig.get(key)
        if existing:
            return existing
        ret_m = self._mangle_type_id(ret_cpp)
        sel_m = self._mangle_selector_id(selector)
        args_m = "_".join(self._mangle_type_id(a) for a in arg_types_cpp)
        base = f"_{self.prefix}_msg_{ret_m}_{sel_m}"
        name = f"{base}_{args_m}" if args_m else base
        # Collision guard (mangling should never produce them, but a
        # signature equal under whitespace canonicalization could):
        if name in self._bridge_by_name and self._bridge_by_name[name] != key:
            idx = 2
            while f"{base}_x{idx}" in self._bridge_by_name:
                idx += 1
            name = f"{base}_x{idx}"
        self._bridge_by_sig[key] = name
        self._bridge_by_name[name] = key
        self._bridge_entries.append((name, ret_cpp, list(arg_types_cpp), selector))
        return name

    # ── Block-typed method overload helpers ──────────────────────────
    # When a method takes a single block parameter we also emit a second
    # overload that takes `const <Block>Function&` (std::function) and wraps
    # it in a block literal — same pattern upstream metal-cpp uses by hand.

    def _block_param_info(self, m: ObjCMethod) -> tuple[Optional[int], Optional[ObjCBlockTypedef]]:
        idx = None
        block = None
        for i, p in enumerate(m.params):
            if self.is_block(p.objc_type):
                if idx is not None:
                    return None, None  # >1 block params: skip
                idx = i
                for b in self.all_blocks:
                    if b.name == p.objc_type:
                        block = b
                        break
        return (idx, block) if block else (None, None)

    def _function_alias_name(self, block: ObjCBlockTypedef) -> str:
        """C++ name of the std::function-typed alias matching `block`. Matches
        upstream metal-cpp's convention: <Foo>Block → <Foo>Function, otherwise
        append Function."""
        cpp = strip_objc_prefix(block.name, self.strip_prefix)
        base = cpp[:-len("Block")] if cpp.endswith("Block") else cpp
        return f"{self.ns}::{base}Function"

    def _fmt_function_overload_params(self, m: ObjCMethod, cls_name: str,
                                       block_idx: int, block: ObjCBlockTypedef) -> str:
        fn_alias = self._function_alias_name(block)
        parts = []
        for i, p in enumerate(m.params):
            if i == block_idx:
                parts.append(f"const {fn_alias}& {p.name}")
            else:
                parts.append(f"{self._resolve(p.objc_type, cls_name)} {p.name}")
        return ", ".join(parts)

    def _fmt_function_overload_body(self, m: ObjCMethod, block_idx: int,
                                     block: ObjCBlockTypedef) -> str:
        fn_alias = self._function_alias_name(block)
        fn_param = m.params[block_idx].name
        # Block literal forwards every arg to the captured std::function.
        block_args = []
        forward_args = []
        for i, arg in enumerate(block.arg_types):
            arg_cpp = self.resolver.resolve(arg)
            name = f"x{i}"
            block_args.append(f"{arg_cpp} {name}")
            forward_args.append(name)
        fwd = ", ".join(forward_args)
        block_ret = self.resolver.resolve(block.return_type) if block.return_type else "void"
        if block_ret == "void":
            block_literal = f"^({', '.join(block_args)}) {{ blockFunction({fwd}); }}"
        else:
            # Non-void block — Apple block syntax puts the return type BEFORE
            # the arg list: `^Ret(args) { ... }`. C++ lambda-style trailing
            # return is not valid here.
            block_literal = (f"^{block_ret}({', '.join(block_args)}) "
                              f"{{ return blockFunction({fwd}); }}")
        call_args = []
        for i, p in enumerate(m.params):
            call_args.append(block_literal if i == block_idx else p.name)
        # Outer method might return a value too (e.g. NS::UInteger from
        # indexOfObjectPassingTest); forward that result.
        m_ret = self._resolve(m.return_type) if m.return_type else "void"
        return_kw = "return " if m_ret != "void" else ""
        return (
            f"    __block {fn_alias} blockFunction = {fn_param};\n"
            f"    {return_kw}{m.cpp_name}({', '.join(call_args)});"
        )

    def generate_class_header(self, classes: list[tuple["ObjCClass", Optional[dict]]],
                                header_basename: str,
                                constants: Optional[list["ObjCConstant"]] = None,
                                string_typedefs: Optional[list[tuple[str, str]]] = None,
                                enums: Optional[list[ObjCEnum]] = None) -> str:
        """Emit one .hpp covering every class that lived in the same SDK
        header — matches upstream metal-cpp's file layout (MTLEvent.h →
        MTLEvent.hpp containing Event + SharedEvent + SharedEventHandle +
        SharedEventListener).

        `classes` is a list of (ObjCClass, override-dict) tuples in source
        order. `header_basename` is the destination .hpp filename
        (e.g. 'MTLEvent') — also used to recognize "same-file" enum
        references in the scanner. `constants`, `string_typedefs`, and
        `enums` are the extern globals / NSString-aliased typedefs /
        enum declarations the SDK header carried at file scope — emitted
        in the namespace before the class declarations so they sit where
        Apple's SDK declares them.
        """
        p = self.prefix
        ns = self.ns
        # Helper: include path from this generator's dir to a target dir.
        def _rel_dir(target_dir: str) -> str:
            return "" if target_dir == (self.output_subdir or "") else f"../{target_dir}/"
        ns_root = "" if ns == "NS" else "../Foundation/"

        # Per-class superclass-include needs (collected first so they go in
        # the preamble before anything that may depend on them). The include
        # path uses the SDK-header basename the superclass lives in — same
        # convention as the rest of the file layout.
        same_file_classes = {c.name for c, _ in classes}
        super_hpps: set[str] = set()
        super_cpps: dict[str, str] = {}  # cls.name → super_cpp
        for cls, _ in classes:
            sc = ""
            shpp = ""
            # Skip superclasses we don't emit (NSObject, NSFastEnumeration,
            # NSLocking, and other CRTP-base / non-generated protocols).
            # They reach us through upstream NSObject.hpp.
            # Look up the parent's source header in this gen's map first,
            # then fall back to the resolver's cross-framework map so a
            # subclass declared in framework A can name a parent from B.
            super_src = (self.class_to_source.get(cls.superclass)
                         or self.resolver.class_to_source.get(cls.superclass, ""))
            if (cls.superclass and cls.superclass != "NSObject"
                    and super_src):
                resolved = resolve_type(self.resolver, cls.superclass, ns,
                                        self.cpp_class_name(cls.superclass))
                if resolved and not resolved.startswith("void"):
                    if resolved.endswith("*"):
                        resolved = resolved[:-1].strip()
                    sc = resolved
                    super_ns, _, _ = sc.rpartition("::")
                    super_dir = self.FW_NS_TO_DIR.get(super_ns, super_ns)
                    same_dir = super_dir == (self.output_subdir or "")
                    if same_dir and cls.superclass not in same_file_classes:
                        shpp = f'"{super_src}.hpp"'
                    elif super_ns and not same_dir:
                        shpp = f'"{_rel_dir(super_dir)}{super_src}.hpp"'
            super_cpps[cls.name] = sc
            if shpp:
                super_hpps.add(shpp)

        lines = ["#pragma once", ""]
        if self.needs_availability_header:
            # `API_AVAILABLE` / `API_UNAVAILABLE` / `API_DEPRECATED` macros
            # used by the per-decl availability annotations below.
            lines.append("#include <Availability.h>")
        lines.append(f'#include "{p}Defines.hpp"')
        # Auxiliary per-framework aggregates are emitted only when non-empty
        # (see `has_blocks` / `has_structs`). Skip the include too —
        # referencing a missing file would otherwise break the build.
        # Enums no longer live in a per-framework aggregate: each enum is
        # emitted into the per-source-header hpp it was declared in, and
        # the resolver's `cpp_to_source` map routes cross-header includes
        # below via `_scan`.
        if self.has_blocks:
            lines.append(f'#include "{p}Blocks.hpp"')
        if self.has_structs:
            lines.append(f'#include "{p}Structs.hpp"')
        # Consolidated extern "C" trampoline decls for this framework.
        # Per-class headers no longer carry their own externs — every
        # selector dispatched from this header is declared once in
        # `<P>Bridge.hpp` and reused across every class that calls it.
        # Listed after Structs/Blocks so by-value struct params in the
        # bridge see complete definitions (Structs may chain back into
        # this header via the upstream keep-list).
        lines.append(f'#include "{p}Bridge.hpp"')
        lines += [
            f'#include "{ns_root}NSObject.hpp"',
            f'#include "{ns_root}NSTypes.hpp"',
            f'#include "{ns_root}NSRange.hpp"',
        ]
        for shpp in sorted(super_hpps):
            lines.append(f"#include {shpp}")

        # Register string-typedef aliases (`typedef NSString *CADynamicRange`)
        # with the resolver BEFORE per-class processing so member type
        # resolution finds them. Emission of the `using` lines happens later
        # (inside the namespace block, where the alias is actually visible
        # to consumers). The mapped value is the alias name — the alias
        # already expands to a pointer type, so callers must NOT add `*`.
        for alias, underlying in (string_typedefs or []):
            stripped = strip_objc_prefix(alias, self.strip_prefix)
            self.resolver.register(alias, f"{ns}::{stripped}",
                                    kind="typedef",
                                    framework_prefix=self.prefix)
            # Record the underlying type so `<P>Bridge.hpp` (which can't
            # see the per-header `using` def) can rewrite signatures.
            resolved_underlying = self._resolve(underlying, "")
            if not resolved_underlying.startswith("void"):
                self.resolver.alias_underlying[f"{ns}::{stripped}"] = resolved_underlying

        # ── Per-class processing (filtering, dedup) ────────────────────
        # Build a working struct of {cls, override, sliced members} for
        # each class so the section emitters below can iterate without
        # repeating the prep work.
        # Class templates from kept-upstream headers — can't be
        # forward-declared as plain classes (`class Foo;` clashes with
        # `template<class T> class Foo;`). Map name → header to #include
        # instead of forward-declaring.
        TEMPLATE_CLASSES = {
            ("NS", "Enumerator"): "NSEnumerator.hpp",
        }
        block_cpp_names = {strip_objc_prefix(b.name, self.strip_prefix) for b in self.all_blocks}
        struct_cpp_names = {s.name for s in self.all_structs}
        # Enums declared in *this* source header — references inside the
        # file resolve in-place and need no extra include.
        same_file_enum_cpp_names = {
            strip_objc_prefix(e.name, self.strip_prefix)
            for e in (enums or []) if e.name
        }

        resolved_types: set[str] = set()
        fwd_decls: dict[str, set[str]] = {}
        # Per-namespace opaque enum forward decls: `enum Foo : NS::UInteger;`.
        # Lets a class header reference an enum declared in another header
        # without pulling that header through — sidesteps the include
        # cycles `MTLArgument.hpp` <-> `MTLTexture.hpp` produced when we
        # tried to emit a full include for every cross-file enum ref.
        # ns -> {cpp_name: (underlying, is_options)}
        fwd_enums: dict[str, dict[str, tuple[str, bool]]] = {}
        cross_fw_includes: set[str] = set()
        same_file_cpp_names = {self.cpp_class_name(c.name) for c, _ in classes}

        def _scan(resolved: str) -> None:
            for ns_name, cpp_cls in re.findall(r"\b(NS|MTL4FX|MTL4|MTLFX|MTL|CA)::(\w+)", resolved):
                if ns_name == self.ns and cpp_cls in same_file_cpp_names:
                    continue
                if cpp_cls in self.NOT_A_CLASS:
                    continue
                # Class templates need a real #include — a `class Foo;`
                # forward decl would clash with the template definition
                # in the kept-upstream header.
                template_hdr = TEMPLATE_CLASSES.get((ns_name, cpp_cls))
                if template_hdr:
                    fw_dir = self.FW_NS_TO_DIR.get(ns_name, ns_name)
                    cross_fw_includes.add(
                        f'#include "{_rel_dir(fw_dir)}{template_hdr}"')
                    continue
                qual = f"{ns_name}::{cpp_cls}"
                kind = self.resolver.kinds.get(qual, "class")
                if kind == "enum":
                    # In-file refs need nothing; cross-file refs get an
                    # opaque enum forward decl. The full definition is
                    # reached transitively when the consumer needs enum
                    # values — methods only ever USE the type, so the
                    # forward decl is enough for them to compile.
                    if ns_name == self.ns and cpp_cls in same_file_enum_cpp_names:
                        continue
                    underlying = self.resolver.enum_underlying.get(qual)
                    if underlying:
                        is_options = self.resolver.enum_is_options.get(qual, False)
                        fwd_enums.setdefault(ns_name, {})[cpp_cls] = (underlying, is_options)
                    continue
                if kind != "class":
                    suffix = {"block": "Blocks", "struct": "Structs"}.get(kind)
                    if suffix and ns_name != self.ns:
                        fw_dir = self.FW_NS_TO_DIR.get(ns_name, ns_name)
                        fw_prefix = self.resolver.frameworks.get(qual, ns_name)
                        cross_fw_includes.add(
                            f'#include "{_rel_dir(fw_dir)}{fw_prefix}{suffix}.hpp"')
                    continue
                if ns_name == self.ns and cpp_cls in block_cpp_names:
                    continue
                if ns_name == self.ns and cpp_cls in struct_cpp_names:
                    continue
                fwd_decls.setdefault(ns_name, set()).add(cpp_cls)

        # Type aliases that collapse to the same C++ scalar — must be treated
        # as equal during overload dedup so that `numberWithLong:` and
        # `numberWithInteger:` (NSInteger ≡ long on Apple's 64-bit ABI) don't
        # both emit `number(long)` and trip a redeclaration error.
        SCALAR_ALIASES = {
            "NS::Integer": "long",
            "NS::UInteger": "unsigned long",
            "NS::TimeInterval": "double",
        }

        def _canon(t: str) -> str:
            for alias, base in SCALAR_ALIASES.items():
                t = t.replace(alias, base)
            # Collapse whitespace around `*` so resolver outputs that differ
            # only in spacing (`void *` vs `void*`) compare equal — the
            # signature tuple is the dedup key, and identical C++ overloads
            # would otherwise leak through.
            return re.sub(r"\s*\*\s*", "*", t)

        def _signature(cls_name: str, cpp_name: str, params: list[ObjCParam]) -> tuple:
            return (cpp_name,) + tuple(_canon(self._resolve(p.objc_type, cls_name)) for p in params)

        prepared: list[dict] = []  # one entry per emitted class
        for cls, override in classes:
            class_props = [p for p in cls.properties if p.is_class_property]
            instance_props = [p for p in cls.properties if not p.is_class_property]
            # Drop methods whose C++ name collides with a reserved keyword
            # (`+new`, `+delete`, etc. — Apple's SDK exposes these on some
            # classes specifically marked unavailable in ObjC).
            CXX_KEYWORDS = {"new", "delete", "class", "template", "operator",
                             "this", "throw", "try", "catch", "typeid",
                             "typename", "using", "namespace"}
            class_methods = [m for m in cls.methods
                              if m.is_class_method and m.cpp_name not in CXX_KEYWORDS]
            instance_methods = [m for m in cls.methods
                                 if not m.is_class_method and m.cpp_name not in CXX_KEYWORDS]

            seen_sigs: set[tuple] = set()
            class_methods = [m for m in class_methods
                              if not (_signature(cls.name, m.cpp_name, m.params) in seen_sigs
                                       or seen_sigs.add(_signature(cls.name, m.cpp_name, m.params)))]
            seen_sigs = set()
            instance_methods = [m for m in instance_methods
                                 if not (_signature(cls.name, m.cpp_name, m.params) in seen_sigs
                                          or seen_sigs.add(_signature(cls.name, m.cpp_name, m.params)))]
            seen_props: set[str] = set()
            instance_props = [p for p in instance_props
                               if not (p.name in seen_props or seen_props.add(p.name))]
            seen_props = set()
            class_props = [p for p in class_props
                            if not (p.name in seen_props or seen_props.add(p.name))]
            class_sigs = {_signature(cls.name, m.cpp_name, m.params) for m in class_methods}
            instance_methods = [m for m in instance_methods
                                 if _signature(cls.name, m.cpp_name, m.params) not in class_sigs]

            # Walk every type touched by this class so we can build the
            # union resolved-types / forward-decl / cross-fw-include set.
            for prop in class_props + instance_props:
                r = self._resolve(prop.objc_type, cls.name)
                resolved_types.add(r); _scan(r)
            for m in class_methods + instance_methods:
                r = self._resolve(m.return_type, cls.name)
                resolved_types.add(r); _scan(r)
                for param in m.params:
                    r = self._resolve(param.objc_type, cls.name)
                    resolved_types.add(r); _scan(r)

            prepared.append({
                "cls": cls,
                "override": override or {},
                "class_props": class_props,
                "instance_props": instance_props,
                "class_methods": class_methods,
                "instance_methods": instance_methods,
            })

        for header in self._system_includes(resolved_types):
            lines.append(f"#include <{header}>")
        for inc in sorted(cross_fw_includes):
            lines.append(inc)

        if fwd_decls or fwd_enums:
            lines.append("")
            # Merge per-namespace class + enum forward decls so each
            # namespace block declares everything once.
            all_ns = sorted(set(fwd_decls) | set(fwd_enums))
            for ns_name in all_ns:
                lines.append(f"namespace {ns_name} {{")
                for cls_fwd in sorted(fwd_decls.get(ns_name, set())):
                    lines.append(f"    class {cls_fwd};")
                for enum_name, (underlying, is_options) in sorted(fwd_enums.get(ns_name, {}).items()):
                    lines.append(self._format_enum_fwd_decl(
                        enum_name, underlying, is_options, indent="    "))
                lines.append("}")

        # ── Class declarations (one namespace block, all classes inside) ──
        # Local alias — every call site in this function annotates a member
        # (method / property / setter), so it routes through the member flag.
        # The single class-level annotation uses `self._av_type` directly.
        _av = self._av_member

        lines += ["", f"namespace {ns}", "{", ""]
        # String-typedef aliases (`typedef NSString *CADynamicRange`) — emit
        # them at the top of the namespace, where Apple's SDK declares them
        # alongside the associated `extern const` globals. Build a local
        # lookup so the constants below can resolve `const <Alias>` against
        # the just-emitted alias rather than the global resolver (which
        # intentionally doesn't register these to avoid cross-header
        # include explosions).
        emitted_any_preamble = False
        local_typedef: dict[str, str] = {}  # ObjC alias name → stripped C++ name
        for alias, underlying in (string_typedefs or []):
            stripped = strip_objc_prefix(alias, self.strip_prefix)
            resolved = self._resolve(underlying, "")
            if resolved.startswith("void"):
                continue
            lines.append(f"using {stripped} = {resolved};")
            local_typedef[alias] = stripped
            emitted_any_preamble = True
        # `extern <Type> const <Name> __asm__("_<C_Name>")` — namespaced
        # constants bound at link time to Apple's underscored C symbol so
        # `.mm` translation units that also include Apple's headers don't
        # see two declarations of the same C name.
        for const in (constants or []):
            if not const.c_name.startswith(self.strip_prefix):
                continue
            stripped = strip_objc_prefix(const.c_name, self.strip_prefix)
            cpp_type = const.cpp_type.strip()
            # Resolve via local typedef first (`const CADynamicRange` →
            # `CA::DynamicRange const`), then fall back to the global
            # resolver (handles BUILTIN-mapped aliases like NSErrorDomain).
            m = re.match(r"^const\s+(\w+)$", cpp_type)
            if m and m.group(1) in local_typedef:
                resolved = f"{local_typedef[m.group(1)]} const"
            else:
                resolved = self._resolve(cpp_type, "")
            if resolved.startswith("void"):
                continue
            lines.append(
                f'extern {resolved} {stripped} __asm__("_{const.c_name}");'
            )
            emitted_any_preamble = True
        # Enums declared in this SDK source header. Emitted inline so each
        # `<header>.hpp` carries the enums Apple's `<header>.h` declared,
        # rather than a per-framework `<Prefix>Enums.hpp` aggregate. The
        # scanner (above) routes cross-file references to the right hpp.
        if enums:
            enum_lines = self._emit_enum_lines(enums)
            if enum_lines:
                lines += enum_lines
                emitted_any_preamble = True
        if emitted_any_preamble:
            lines.append("")
        # Forward-declare every class in this file so a class declared earlier
        # can reference one declared later (common when a Descriptor's factory
        # method returns the protocol it produces, both in the same SDK .h).
        if len(prepared) > 1:
            for info in prepared:
                lines.append(f"class {self.cpp_class_name(info['cls'].name)};")
            lines.append("")
        for idx, info in enumerate(prepared):
            cls = info["cls"]
            cpp_name = self.cpp_class_name(cls.name)
            prepend_block = info["override"].get("prepend", "")
            append_block = info["override"].get("append", "")

            crtp = "Referencing"
            if "NSSecureCoding" in cls.protocols:
                crtp = "SecureCoding"
            elif "NSCopying" in cls.protocols:
                crtp = "Copying"
            sc = super_cpps.get(cls.name, "")
            base = f"NS::{crtp}<{cpp_name}{', ' + sc if sc else ''}>"

            if prepend_block:
                lines += [prepend_block.rstrip(), ""]
            # Auto-emit `static T* alloc()` and `T* init()` — every Obj-C
            # class supports these via NSObject. We don't carry inheritance
            # from the @interface chain so `T::alloc()->init()` would
            # otherwise fail. Skip when the class redeclares its own init().
            has_own_init = any(
                m.cpp_name == "init" and not m.params
                for m in info["instance_methods"]
            )
            info["emit_auto_init"] = not has_own_init
            # Each emitted member becomes a (lhs, rhs, sort_key[, prefix])
            # tuple so we can column-align names within a group (`static
            # MTL::Device*  ` padded to the same width as the longest LHS)
            # and sort the rows by C++ name to match upstream metal-cpp's
            # layout. Overloads of the same name share a sort key and stay
            # adjacent because Python's sort is stable. An optional 4th
            # element holds a header line (`template <class _Object = Object>`)
            # emitted on its own line just before the aligned row.
            def _emit_aligned(rows: list) -> None:
                if not rows:
                    return
                rows.sort(key=lambda r: r[2])
                width = max(len(r[0]) for r in rows) + 1
                for row in rows:
                    if len(row) >= 4 and row[3]:
                        lines.append(f"    {row[3]}")
                    lines.append(f"    {row[0].ljust(width)}{row[1]};")

            # Auto-emitted alloc() / init() — kept as a top group, separated
            # by a blank line from the rest, mirroring upstream's "constructor
            # cluster" convention. Skipped for protocols: those wrap an
            # `@protocol` and aren't directly instantiable; the runtime
            # produces them via factories on a concrete class
            # (e.g. `MTL::Device::newBuffer(...)`).
            # Attribute placement: clang requires `class <attrs> Name`, not
            # `class Name <attrs>` (the latter is ignored for type decls) and
            # not on a preceding line (also ignored for class definitions).
            lines += [
                f"class{self._av_type(cls.availability)} {cpp_name} : public {base}",
                "{",
                "public:",
            ]
            if not cls.is_protocol:
                ctor_rows: list[tuple[str, str, str]] = [
                    (f"static {cpp_name}*", "alloc()", "alloc"),
                ]
                if info["emit_auto_init"]:
                    ctor_rows.append((f"{cpp_name}*", "init() const", "init"))
                _emit_aligned(ctor_rows)
                lines.append("")

            # Class-scope: static methods and class properties, merged.
            class_rows: list[tuple[str, str, str]] = []
            for prop in info["class_props"]:
                cpp_type = self._resolve(prop.objc_type, cls.name)
                getter_name = cpp_method_name_from_first_segment(prop.name)
                class_rows.append((
                    f"static {cpp_type}",
                    f"{getter_name}(){_av(prop.availability)}",
                    getter_name,
                ))
            for m in info["class_methods"]:
                ret = self._resolve(m.return_type, cls.name)
                params_str = self._fmt_params(m.params, cls.name)
                class_rows.append((
                    f"static {ret}",
                    f"{m.cpp_name}({params_str}){_av(m.availability)}",
                    m.cpp_name,
                ))
                bi, blk = self._block_param_info(m)
                if bi is not None and blk is not None:
                    fp = self._fmt_function_overload_params(m, cls.name, bi, blk)
                    class_rows.append((
                        f"static {ret}",
                        f"{m.cpp_name}({fp}){_av(m.availability)}",
                        m.cpp_name,
                    ))
            if class_rows:
                _emit_aligned(class_rows)
                lines.append("")

            # Instance-scope: instance methods + property getters/setters,
            # merged. Setter `setFoo` sorts after getter `foo` naturally
            # ('f' < 's'), matching upstream's getter-then-setter cadence.
            inst_rows: list[tuple[str, str, str]] = []
            for prop in info["instance_props"]:
                getter_name = cpp_method_name_from_first_segment(prop.name)
                # Same generic-return treatment as methods: `@property
                # (readonly) ObjectType firstObject;` becomes templated.
                if gi := _generic_return_info(prop.objc_type):
                    tparam, ret = gi
                    inst_rows.append((
                        ret,
                        f"{getter_name}() const{_av(prop.availability)}",
                        getter_name,
                        f"template <class {tparam} = Object>",
                    ))
                    continue
                cpp_type = self._resolve(prop.objc_type, cls.name)
                inst_rows.append((
                    cpp_type,
                    f"{getter_name}() const{_av(prop.availability)}",
                    getter_name,
                ))
                if not prop.is_readonly:
                    # Setter name matches upstream's pattern: `setUtf8String`
                    # (case-folded) rather than `setUTF8String`. Selector
                    # still tracks the SDK form via stub_decl below.
                    sname_cpp = f"set{getter_name[0].upper()}{getter_name[1:]}"
                    inst_rows.append((
                        "void",
                        f"{sname_cpp}({cpp_type} {prop.name}){_av(prop.availability)}",
                        sname_cpp,
                    ))
            for m in info["instance_methods"]:
                params_str = self._fmt_params(m.params, cls.name)
                # Methods on collection-style classes that return the bare
                # generic parameter (`ObjectType` / `KeyType` / `ValueType`)
                # or an `NSEnumerator<GenericParam>` become template-on-
                # element methods so the caller can name the element type
                # and skip a downstream `reinterpret_cast`. Matches
                # upstream's `template <class _KeyType = Object>
                # Enumerator<_KeyType>* keyEnumerator() const;` form.
                # Defaults to the base `Object*` so untyped callers still
                # compile.
                # Instance methods aren't emitted `const` — most mutate the
                # underlying Obj-C object's state (e.g. `commit()`,
                # `enqueue()`, encoder calls). Property getters keep `const`
                # because they're guaranteed pure reads.
                if gi := _generic_return_info(m.return_type):
                    tparam, ret = gi
                    inst_rows.append((
                        ret,
                        f"{m.cpp_name}({params_str}){_av(m.availability)}",
                        m.cpp_name,
                        f"template <class {tparam} = Object>",
                    ))
                else:
                    ret = self._resolve(m.return_type, cls.name)
                    inst_rows.append((
                        ret,
                        f"{m.cpp_name}({params_str}){_av(m.availability)}",
                        m.cpp_name,
                    ))
                    bi, blk = self._block_param_info(m)
                    if bi is not None and blk is not None:
                        fp = self._fmt_function_overload_params(m, cls.name, bi, blk)
                        inst_rows.append((
                            ret,
                            f"{m.cpp_name}({fp}){_av(m.availability)}",
                            m.cpp_name,
                        ))
            if inst_rows:
                _emit_aligned(inst_rows)
                lines.append("")

            lines += ["};", ""]
            if append_block:
                lines += [append_block.rstrip(), ""]

        lines += [
            f"}} // namespace {ns}",
            "",
        ]

        # Dispatch stubs and inline impls only apply when this header
        # actually declares classes — an enum-only `.hpp` (e.g. emitted
        # for `MTLDataType.h`) closes the namespace and stops here.
        if not prepared:
            return "\n".join(lines)

        # ── ObjC class symbols + inline impls ─────────────────────────
        # Trampoline `extern "C"` decls all live in `<P>Bridge.hpp` (already
        # included up top). Each class only emits its own runtime class
        # symbol here — `&OBJC_CLASS_$_<X>` is what `+alloc` is invoked on.
        lines.append("// --- Class symbols + inline implementations ---")
        lines.append("")
        for info in prepared:
            cls = info["cls"]
            lines.append(f'extern "C" void *OBJC_CLASS_$_{cls.name};')
        lines.append("")

        def _arg_types(cls_name: str, params: list[ObjCParam]) -> list[str]:
            return [self._resolve(p.objc_type, cls_name) for p in params]

        def _call(receiver_expr: str, stub: str, args: str) -> str:
            arg_suffix = f", {args}" if args else ""
            return f"{stub}((const void*){receiver_expr}, nullptr{arg_suffix})"

        for info in prepared:
            cls = info["cls"]
            cpp_name = self.cpp_class_name(cls.name)
            cls_recv = f"&OBJC_CLASS_$_{cls.name}"
            cls_t = f"{self.ns}::{cpp_name}*"
            # Inline body for the auto-emitted alloc() and (optionally) init().
            # Skipped for protocols (no constructor cluster declared).
            if not cls.is_protocol:
                alloc_stub = self._stub(cls_t, [], "alloc")
                lines += [
                    f"_{p}_INLINE {ns}::{cpp_name}* {ns}::{cpp_name}::alloc()",
                    "{",
                    f"    return {_call(cls_recv, alloc_stub, '')};",
                    "}", "",
                ]
                if info.get("emit_auto_init"):
                    init_stub = self._stub(cls_t, [], "init")
                    lines += [
                        f"_{p}_INLINE {ns}::{cpp_name}* {ns}::{cpp_name}::init() const",
                        "{",
                        f"    return {_call('this', init_stub, '')};",
                        "}", "",
                    ]
            for prop in info["class_props"]:
                cpp_type = self._resolve(prop.objc_type, cls.name)
                stub = self._stub(cpp_type, [], prop.name)
                getter_name = cpp_method_name_from_first_segment(prop.name)
                lines += [
                    f"_{p}_INLINE {cpp_type} {ns}::{cpp_name}::{getter_name}()",
                    "{",
                    f"    return {_call(cls_recv, stub, '')};",
                    "}", "",
                ]
            for m in info["class_methods"]:
                ret = self._resolve(m.return_type, cls.name)
                params_str = self._fmt_params(m.params, cls.name)
                args = self._fmt_args(m.params)
                stub = self._stub(ret, _arg_types(cls.name, m.params), m.selector)
                ret_kw = "return " if ret != "void" else ""
                lines += [
                    f"_{p}_INLINE {ret} {ns}::{cpp_name}::{m.cpp_name}({params_str})",
                    "{",
                    f"    {ret_kw}{_call(cls_recv, stub, args)};",
                    "}", "",
                ]
                bi, blk = self._block_param_info(m)
                if bi is not None and blk is not None:
                    fp = self._fmt_function_overload_params(m, cls.name, bi, blk)
                    body = self._fmt_function_overload_body(m, bi, blk)
                    lines += [
                        f"_{p}_INLINE {ret} {ns}::{cpp_name}::{m.cpp_name}({fp})",
                        "{", body, "}", "",
                    ]
            for prop in info["instance_props"]:
                getter_name = cpp_method_name_from_first_segment(prop.name)
                cpp_type = self._resolve(prop.objc_type, cls.name)
                stub = self._stub(cpp_type, [], prop.name)
                if gi := _generic_return_info(prop.objc_type):
                    tparam, ret = gi
                    # The impl is at file scope, so qualify the template
                    # container name with the namespace (the decl form
                    # above is inside `namespace NS {}` and stays bare).
                    ret_q = ret.replace("Enumerator<", f"{ns}::Enumerator<")
                    lines += [
                        f"template <class {tparam}>",
                        f"_{p}_INLINE {ret_q} {ns}::{cpp_name}::{getter_name}() const",
                        "{",
                        f"    return reinterpret_cast<{ret_q}>({_call('this', stub, '')});",
                        "}", "",
                    ]
                    continue
                lines += [
                    f"_{p}_INLINE {cpp_type} {ns}::{cpp_name}::{getter_name}() const",
                    "{",
                    f"    return {_call('this', stub, '')};",
                    "}", "",
                ]
                if not prop.is_readonly:
                    # SDK selector keeps its original case; C++ name matches
                    # the case-folded getter.
                    sname_sel = setter_name(prop.name)
                    sname_cpp = f"set{getter_name[0].upper()}{getter_name[1:]}"
                    stub_s = self._stub("void", [cpp_type], f"{sname_sel}:")
                    lines += [
                        f"_{p}_INLINE void {ns}::{cpp_name}::{sname_cpp}({cpp_type} {prop.name})",
                        "{",
                        f"    {_call('this', stub_s, prop.name)};",
                        "}", "",
                    ]
            for m in info["instance_methods"]:
                params_str = self._fmt_params(m.params, cls.name)
                args = self._fmt_args(m.params)
                ret = self._resolve(m.return_type, cls.name)
                stub = self._stub(ret, _arg_types(cls.name, m.params), m.selector)
                if gi := _generic_return_info(m.return_type):
                    # Templated element-typed method (see declaration side):
                    # the trampoline returns the resolver's default (e.g.
                    # `void*` or `NS::Enumerator<NS::Object>*`), so
                    # reinterpret_cast to the caller's requested type. The
                    # impl is at file scope, so qualify the template
                    # container with the namespace.
                    tparam, ret_g = gi
                    ret_q = ret_g.replace("Enumerator<", f"{ns}::Enumerator<")
                    lines += [
                        f"template <class {tparam}>",
                        f"_{p}_INLINE {ret_q} {ns}::{cpp_name}::{m.cpp_name}({params_str})",
                        "{",
                        f"    return reinterpret_cast<{ret_q}>({_call('this', stub, args)});",
                        "}", "",
                    ]
                    continue
                ret_kw = "return " if ret != "void" else ""
                lines += [
                    f"_{p}_INLINE {ret} {ns}::{cpp_name}::{m.cpp_name}({params_str})",
                    "{",
                    f"    {ret_kw}{_call('this', stub, args)};",
                    "}", "",
                ]
                bi, blk = self._block_param_info(m)
                if bi is not None and blk is not None:
                    fp = self._fmt_function_overload_params(m, cls.name, bi, blk)
                    body = self._fmt_function_overload_body(m, bi, blk)
                    lines += [
                        f"_{p}_INLINE {ret} {ns}::{cpp_name}::{m.cpp_name}({fp})",
                        "{", body, "}", "",
                    ]

        return "\n".join(lines)

    def generate_umbrella(self, fw_name: str) -> str:
        p = self.prefix
        lines = [
            "#pragma once",
            "",
            f'#include "{p}Defines.hpp"',
        ]
        # Enums live in their declaring per-source-header hpp, included
        # below via `generated_headers`. Blocks/Structs are still per-fw
        # aggregates.
        if self.has_blocks:
            lines.append(f'#include "{p}Blocks.hpp"')
        if self.has_structs:
            lines.append(f'#include "{p}Structs.hpp"')
        for header in sorted(self.generated_headers):
            lines.append(f'#include "{header}"')
        # Sibling umbrellas this framework pulls in (e.g. Metal.hpp →
        # Metal4.hpp so a single client include surfaces both namespaces).
        for extra in self.extra_umbrella_includes:
            lines.append(f'#include "{extra}"')
        lines.append("")
        return "\n".join(lines)

    # ── Formatting helpers ────────────────────────────────────────────

    def _fmt_params(self, params: list[ObjCParam], cls_name: str) -> str:
        if not params:
            return ""
        parts = []
        for p in params:
            cpp_type = self._resolve(p.objc_type, cls_name)
            parts.append(f"{cpp_type} {p.name}")
        return ", ".join(parts)

    def _fmt_args(self, params: list[ObjCParam]) -> str:
        return ", ".join(p.name for p in params)
