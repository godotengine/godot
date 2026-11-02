#!/usr/bin/env bash
# Build SPIR-V preprocessing fuzz targets.
#
# Usage:
#   ./drivers/webgpu/tint_cli/fuzz/build_fuzzers.sh           # Build all 3 fuzzers
#   ./drivers/webgpu/tint_cli/fuzz/build_fuzzers.sh --clean   # Clean + rebuild
#   ./drivers/webgpu/tint_cli/fuzz/build_fuzzers.sh --test    # Build + smoke test
#
# Output: bin/fuzz_preprocess_passes, bin/fuzz_split_samplers, bin/fuzz_spirv_to_wgsl
#
# If Homebrew LLVM is installed (brew install llvm), builds with libFuzzer +
# AddressSanitizer for production fuzzing. Otherwise builds with a standalone
# mutation driver that works with any compiler.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TINT_CLI_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$TINT_CLI_DIR/../../../" && pwd)"
BUILD_DIR="$TINT_CLI_DIR/.build"
SHIM_DIR="$TINT_CLI_DIR"

uv run --no-project python "$REPO_ROOT/drivers/webgpu/dependencies.py"

# Parse args.
CLEAN=false
SMOKE_TEST=false
for arg in "$@"; do
    [[ "$arg" == "--clean" ]] && CLEAN=true
    [[ "$arg" == "--test" ]] && SMOKE_TEST=true
done

if [[ "$CLEAN" == true ]]; then
    rm -f "$REPO_ROOT/bin/fuzz_preprocess_passes"
    rm -f "$REPO_ROOT/bin/fuzz_split_samplers"
    rm -f "$REPO_ROOT/bin/fuzz_spirv_to_wgsl"
    rm -f "$BUILD_DIR/fuzz/"*.o
fi

# ─────────────────────────────────────────────────────────────────────────────
# 1. Ensure the tint_cli objects and their response files are built.
# ─────────────────────────────────────────────────────────────────────────────
if [[ ! -f "$BUILD_DIR/spirv_tools.rsp" || ! -f "$BUILD_DIR/cli/tint_wrapper.o" ]]; then
    echo "[fuzz] Building tint_cli dependencies first..."
    bash "$TINT_CLI_DIR/build.sh"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 2. Detect compiler with libFuzzer support.
#    Priority: CXX env var → Homebrew LLVM (macOS) → system clang++ → fallback.
# ─────────────────────────────────────────────────────────────────────────────
USE_LIBFUZZER=false
FUZZ_TEST_SRC='extern "C" int LLVMFuzzerTestOneInput(const unsigned char *d, unsigned long s){return 0;}'

try_libfuzzer() {
    local compiler="$1"
    if echo "$FUZZ_TEST_SRC" | "$compiler" -x c++ -fsanitize=fuzzer,address -o /dev/null - 2>/dev/null; then
        CXX="$compiler"
        USE_LIBFUZZER=true
        return 0
    fi
    return 1
}

# If CXX is explicitly set, try it first.
if [[ -n "${CXX:-}" ]]; then
    try_libfuzzer "$CXX" || true
fi

# Try Homebrew LLVM (macOS).
if [[ "$USE_LIBFUZZER" == false ]]; then
    LLVM_PREFIX="$(brew --prefix llvm 2>/dev/null || echo "")"
    if [[ -n "$LLVM_PREFIX" && -x "$LLVM_PREFIX/bin/clang++" ]]; then
        try_libfuzzer "$LLVM_PREFIX/bin/clang++" || true
    fi
fi

# Try system clang++ (Linux typically ships libFuzzer with clang).
if [[ "$USE_LIBFUZZER" == false ]]; then
    for candidate in clang++ clang++-18 clang++-17 clang++-16 clang++-15; do
        if command -v "$candidate" &>/dev/null; then
            try_libfuzzer "$candidate" && break || true
        fi
    done
fi

if [[ "$USE_LIBFUZZER" == false ]]; then
    CXX="${CXX:-c++}"
    echo "[fuzz] libFuzzer not available — building with standalone mutation driver."
    echo "       macOS: brew install llvm"
    echo "       Linux: apt install clang (or clang-18)"
else
    echo "[fuzz] Using libFuzzer + ASan ($CXX)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3. Gather include/object paths.
# ─────────────────────────────────────────────────────────────────────────────
SPIRV_TOOLS_DIR="$REPO_ROOT/thirdparty/spirv-tools"
SPIRV_HEADERS_DIR="$REPO_ROOT/thirdparty/spirv-headers-webgpu"
TINT_DIR="$REPO_ROOT/thirdparty/tint"

PREPROCESS_INCLUDES=(
    -I"$SHIM_DIR"
    -I"$REPO_ROOT"
    -I"$SPIRV_TOOLS_DIR"
    -I"$SPIRV_TOOLS_DIR/source/"
    -I"$SPIRV_TOOLS_DIR/include/"
    -I"$SPIRV_TOOLS_DIR/generated/"
    -I"$SPIRV_HEADERS_DIR/include/"
    -I"$SPIRV_HEADERS_DIR/include/spirv/unified1/"
)

TINT_DEFINES=(
    -DTINT_BUILD_SPV_READER=1
    -DTINT_BUILD_WGSL_WRITER=1
    -DTINT_BUILD_WGSL_READER=0
    -DTINT_BUILD_SPV_WRITER=0
    -DTINT_BUILD_GLSL_WRITER=0
    -DTINT_BUILD_HLSL_WRITER=0
    -DTINT_BUILD_MSL_WRITER=0
    -DTINT_BUILD_NULL_WRITER=0
    -DTINT_BUILD_SYNTAX_TREE_WRITER=0
    -DTINT_BUILD_IR_BINARY=0
)

TINT_INCLUDES=(
    -I"$TINT_DIR"
    -I"$TINT_DIR/src/"
    -I"$SPIRV_TOOLS_DIR"
    -I"$SPIRV_TOOLS_DIR/source/"
    -I"$SPIRV_TOOLS_DIR/include/"
    -I"$SPIRV_TOOLS_DIR/generated/"
    -I"$SPIRV_HEADERS_DIR/include/"
    -I"$SPIRV_HEADERS_DIR/include/spirv/unified1/"
)

# Library objects selected by build.sh; the response files hold paths relative to the repository root.
cd "$REPO_ROOT"
SPIRV_TOOLS_OBJS="@$BUILD_DIR/spirv_tools.rsp"
TINT_OBJS="@$BUILD_DIR/tint.rsp"
PREPROCESS_OBJ="$BUILD_DIR/fuzz/spirv_preprocess.o"
TINT_WRAPPER_OBJ="$BUILD_DIR/cli/tint_wrapper.o"

# ─────────────────────────────────────────────────────────────────────────────
# 4. Compile and link fuzz targets.
# ─────────────────────────────────────────────────────────────────────────────
mkdir -p "$BUILD_DIR/fuzz" "$REPO_ROOT/bin"

COMMON_FLAGS="-O1 -g -std=c++17 -w"

if [[ "$USE_LIBFUZZER" == true ]]; then
    FUZZ_FLAGS="-fsanitize=fuzzer,address"
    DRIVER_OBJ=""
else
    FUZZ_FLAGS=""
    # Compile standalone driver.
    echo "[fuzz] Compiling standalone driver..."
    $CXX -c "$SCRIPT_DIR/standalone_driver.cpp" \
        -o "$BUILD_DIR/fuzz/standalone_driver.o" \
        $COMMON_FLAGS
    DRIVER_OBJ="$BUILD_DIR/fuzz/standalone_driver.o"
fi

# The fuzzed passes get the fuzzer instrumentation and sanitizers too.
echo "[fuzz] Compiling spirv_preprocess.cpp..."
$CXX -c "$REPO_ROOT/drivers/webgpu/spirv_preprocess.cpp" -o "$PREPROCESS_OBJ" $COMMON_FLAGS "${PREPROCESS_INCLUDES[@]}" $FUZZ_FLAGS

# --- fuzz_preprocess_passes (preprocessing only, no Tint) ---
echo "[fuzz] Building fuzz_preprocess_passes..."
$CXX -c "$SCRIPT_DIR/fuzz_preprocess_passes.cpp" \
    -o "$BUILD_DIR/fuzz/fuzz_preprocess_passes.o" \
    $COMMON_FLAGS "${PREPROCESS_INCLUDES[@]}" $FUZZ_FLAGS

$CXX -o "$REPO_ROOT/bin/fuzz_preprocess_passes" \
    "$BUILD_DIR/fuzz/fuzz_preprocess_passes.o" \
    "$PREPROCESS_OBJ" \
    "$SPIRV_TOOLS_OBJS" \
    $DRIVER_OBJ \
    $FUZZ_FLAGS -O1

# --- fuzz_split_samplers (single pass, no Tint) ---
echo "[fuzz] Building fuzz_split_samplers..."
$CXX -c "$SCRIPT_DIR/fuzz_split_samplers.cpp" \
    -o "$BUILD_DIR/fuzz/fuzz_split_samplers.o" \
    $COMMON_FLAGS "${PREPROCESS_INCLUDES[@]}" $FUZZ_FLAGS

$CXX -o "$REPO_ROOT/bin/fuzz_split_samplers" \
    "$BUILD_DIR/fuzz/fuzz_split_samplers.o" \
    "$PREPROCESS_OBJ" \
    "$SPIRV_TOOLS_OBJS" \
    $DRIVER_OBJ \
    $FUZZ_FLAGS -O1

# --- fuzz_spirv_to_wgsl (full pipeline: preprocessing + Tint) ---
echo "[fuzz] Building fuzz_spirv_to_wgsl..."
$CXX -c "$SCRIPT_DIR/fuzz_spirv_to_wgsl.cpp" \
    -o "$BUILD_DIR/fuzz/fuzz_spirv_to_wgsl.o" \
    -std=c++20 -O1 -g -w \
    "${PREPROCESS_INCLUDES[@]}" "${TINT_INCLUDES[@]}" "${TINT_DEFINES[@]}" \
    -I"$REPO_ROOT/drivers/webgpu/" \
    $FUZZ_FLAGS

$CXX -o "$REPO_ROOT/bin/fuzz_spirv_to_wgsl" \
    "$BUILD_DIR/fuzz/fuzz_spirv_to_wgsl.o" \
    "$PREPROCESS_OBJ" \
    "$TINT_WRAPPER_OBJ" \
    "$TINT_OBJS" \
    "$SPIRV_TOOLS_OBJS" \
    $DRIVER_OBJ \
    $FUZZ_FLAGS -O1

echo ""
echo "[fuzz] Built:"
for f in fuzz_preprocess_passes fuzz_split_samplers fuzz_spirv_to_wgsl; do
    printf "  bin/%-30s %s\n" "$f" "$(du -h "$REPO_ROOT/bin/$f" | cut -f1)"
done

# ─────────────────────────────────────────────────────────────────────────────
# 5. Set up seed corpus from fixtures.
# ─────────────────────────────────────────────────────────────────────────────
CORPUS_DIR="$SCRIPT_DIR/corpus"
FIXTURES_DIR="$REPO_ROOT/drivers/webgpu/tests/shader_corpus/fixtures"

if [[ ! -d "$CORPUS_DIR" ]]; then
    echo ""
    echo "[fuzz] Setting up seed corpus from fixtures..."
    mkdir -p "$CORPUS_DIR"
    for spv in "$FIXTURES_DIR"/*.spv; do
        [[ -f "$spv" ]] && cp "$spv" "$CORPUS_DIR/"
    done
    echo "  Copied $(ls "$CORPUS_DIR"/*.spv 2>/dev/null | wc -l | tr -d ' ') files"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 6. Smoke test (if --test).
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$SMOKE_TEST" == true ]]; then
    echo ""
    echo "[fuzz] Running smoke test (feed each fixture to each fuzzer)..."
    CORPUS_FILES=("$CORPUS_DIR"/*.spv)
    # libFuzzer runs file arguments once; the standalone driver needs an explicit iteration count.
    RUN_ARGS=()
    [[ "$USE_LIBFUZZER" == false ]] && RUN_ARGS=(--iterations 0)

    SMOKE_FAILURES=0
    for fuzzer in fuzz_preprocess_passes fuzz_split_samplers fuzz_spirv_to_wgsl; do
        printf "  %-30s" "$fuzzer"
        status=0
        "$REPO_ROOT/bin/$fuzzer" ${RUN_ARGS[@]+"${RUN_ARGS[@]}"} "${CORPUS_FILES[@]}" 2>/dev/null || status=$?
        if [[ $status -eq 0 ]]; then
            echo "PASS"
        else
            echo "FAIL (exit $status)"
            SMOKE_FAILURES=$((SMOKE_FAILURES + 1))
        fi
    done
    (( SMOKE_FAILURES == 0 )) || exit 1
fi

echo ""
if [[ "$USE_LIBFUZZER" == true ]]; then
    echo "Run a fuzzer:  bin/fuzz_preprocess_passes drivers/webgpu/tint_cli/fuzz/corpus/"
else
    echo "Run a fuzzer:  bin/fuzz_preprocess_passes --iterations 100000 drivers/webgpu/tint_cli/fuzz/corpus/*.spv"
fi
