#!/usr/bin/env bash
# Build tint_convert_cli — native host tool for build-time SPIR-V → WGSL conversion.
#
# Usage:
#   ./drivers/webgpu/tint_cli/build.sh          # Build with auto-detected parallelism
#   ./drivers/webgpu/tint_cli/build.sh --clean   # Remove build artifacts and rebuild
#
# Output: bin/tint_convert_cli
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

uv run --no-project python drivers/webgpu/dependencies.py

# Directories.
TINT_DIR="thirdparty/tint"
SPIRV_TOOLS_DIR="thirdparty/spirv-tools"
SPIRV_HEADERS_DIR="thirdparty/spirv-headers-webgpu"
BUILD_DIR="drivers/webgpu/tint_cli/.build"
SHIM_DIR="drivers/webgpu/tint_cli"

JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
CXX="${CXX:-c++}"

# Parse args.
CLEAN=false
for arg in "$@"; do
    [[ "$arg" == "--clean" ]] && CLEAN=true
done

if [[ "$CLEAN" == true ]]; then
    echo "Cleaning build artifacts..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR/spirv_tools" "$BUILD_DIR/tint" "$BUILD_DIR/cli"
uv run --no-project python drivers/webgpu/sources.py spirv-tools | tr -d '\r' > "$BUILD_DIR/spirv-tools.sources"
# Tint's Windows utilities need MSVC headers, so MinGW builds use the portable variants and link statically
# to avoid picking up another toolchain's runtime DLLs.
case "$(uname -s)" in
    MINGW* | MSYS* | CYGWIN*) TINT_PLATFORM="--platform other" LINK_FLAGS="-static" ;;
    *) TINT_PLATFORM="" LINK_FLAGS="" ;;
esac
uv run --no-project python drivers/webgpu/sources.py tint $TINT_PLATFORM | tr -d '\r' > "$BUILD_DIR/tint.sources"

# Common flags.
WARNINGS="-w"  # Suppress warnings from thirdparty code.
COMMON_FLAGS="-O2 $WARNINGS"

# Include paths for SPIRV-Tools.
SPIRV_TOOLS_INCLUDES=(
    -I"$SPIRV_TOOLS_DIR"
    -I"$SPIRV_TOOLS_DIR/source/"
    -I"$SPIRV_TOOLS_DIR/include/"
    -I"$SPIRV_TOOLS_DIR/generated/"
    -I"$SPIRV_HEADERS_DIR/include/"
    -I"$SPIRV_HEADERS_DIR/include/spirv/unified1/"
)

# Include paths for Tint.
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

# Tint preprocessor defines (must match SCsub).
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

# ─────────────────────────────────────────────────────────────────────────────
# Compile function: thirdparty objects are reused while newer than their source and the dependency
# preparation. The CLI sources include driver headers and shims, so they always rebuild.
# ─────────────────────────────────────────────────────────────────────────────
compile_one() {
    local src="$1"
    local obj="$2"
    local std="$3"
    shift 3
    local flags=("$@")

    if [[ "$src" == thirdparty/* && -f "$obj" && "$obj" -nt "$src" &&
          "$obj" -nt "$TINT_DIR/.webgpu-source.json" &&
          "$obj" -nt "$SPIRV_TOOLS_DIR/.webgpu-source.json" &&
          "$obj" -nt "$SPIRV_HEADERS_DIR/.webgpu-source.json" ]]; then
        return 0
    fi

    mkdir -p "$(dirname "$obj")"
    $CXX -c "$src" -o "$obj" -std="$std" $COMMON_FLAGS "${flags[@]}"
}

COMPILERS=()
compile_async() {
    compile_one "$@" &
    COMPILERS+=("$!")
    if (( ${#COMPILERS[@]} >= JOBS )); then
        wait "${COMPILERS[0]}"
        COMPILERS=("${COMPILERS[@]:1}")
    fi
}

wait_compilers() {
    for compiler in "${COMPILERS[@]}"; do
        wait "$compiler"
    done
    COMPILERS=()
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. Compile SPIRV-Tools
# ─────────────────────────────────────────────────────────────────────────────
echo "[1/4] Compiling SPIRV-Tools..."

SPIRV_TOOLS_OBJS=()
# Collect all .cpp files from the SCsub list (everything under source/).
while IFS= read -r src; do
    objname="${src#$SPIRV_TOOLS_DIR/}"
    objname="${objname%.cpp}.o"
    obj="$BUILD_DIR/spirv_tools/$objname"
    SPIRV_TOOLS_OBJS+=("$obj")
    compile_async "$src" "$obj" "c++17" "${SPIRV_TOOLS_INCLUDES[@]}"
done < "$BUILD_DIR/spirv-tools.sources"
wait_compilers

echo "  ${#SPIRV_TOOLS_OBJS[@]} objects"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Compile Tint
# ─────────────────────────────────────────────────────────────────────────────
echo "[2/4] Compiling Tint..."

# Use the same source selection as the engine build.
TINT_OBJS=()
while IFS= read -r src; do
    objname="${src#$TINT_DIR/}"
    objname="${objname%.cc}.o"
    obj="$BUILD_DIR/tint/$objname"
    TINT_OBJS+=("$obj")
    compile_async "$src" "$obj" "c++20" "${TINT_INCLUDES[@]}" "${TINT_DEFINES[@]}"
done < "$BUILD_DIR/tint.sources"
wait_compilers
echo "  ${#TINT_OBJS[@]} objects"

# ─────────────────────────────────────────────────────────────────────────────
# 3. Compile CLI sources (spirv_preprocess.cpp + tint_wrapper.cpp + main.cpp)
# ─────────────────────────────────────────────────────────────────────────────
echo "[3/4] Compiling CLI sources..."

# spirv_preprocess.cpp — compiled with shim include path (before repo root so
# the shim core/templates/ is found instead of Godot's).
compile_async "drivers/webgpu/spirv_preprocess.cpp" \
    "$BUILD_DIR/cli/spirv_preprocess.o" \
    "c++17" \
    -I"$SHIM_DIR" \
    -I. \
    "${SPIRV_TOOLS_INCLUDES[@]}"

# tint_wrapper.cpp — compiled with Tint C++20 environment.
compile_async "drivers/webgpu/tint_wrapper.cpp" \
    "$BUILD_DIR/cli/tint_wrapper.o" \
    "c++20" \
    "${TINT_INCLUDES[@]}" "${TINT_DEFINES[@]}" \
    -I"drivers/webgpu/" \
    -I.

# main.cpp — compiled with shim + Tint includes.
compile_async "drivers/webgpu/tint_cli/main.cpp" \
    "$BUILD_DIR/cli/main.o" \
    "c++20" \
    -I"$SHIM_DIR" \
    -I"drivers/webgpu/" \
    -I. \
    "${TINT_INCLUDES[@]}" "${TINT_DEFINES[@]}"

wait_compilers

# ─────────────────────────────────────────────────────────────────────────────
# 4. Link
# ─────────────────────────────────────────────────────────────────────────────
echo "[4/4] Linking tint_convert_cli..."

# Link every selected object; missing outputs are build failures. Response files keep the command line
# within Windows limits and are reused by fuzz/build_fuzzers.sh.
printf '%s\n' "${SPIRV_TOOLS_OBJS[@]}" > "$BUILD_DIR/spirv_tools.rsp"
printf '%s\n' "${TINT_OBJS[@]}" > "$BUILD_DIR/tint.rsp"

echo "  Linking $((3 + ${#SPIRV_TOOLS_OBJS[@]} + ${#TINT_OBJS[@]})) objects..."

mkdir -p bin
$CXX -o bin/tint_convert_cli "$BUILD_DIR/cli/main.o" "$BUILD_DIR/cli/spirv_preprocess.o" "$BUILD_DIR/cli/tint_wrapper.o" \
    @"$BUILD_DIR/tint.rsp" @"$BUILD_DIR/spirv_tools.rsp" -O2 $LINK_FLAGS

echo ""
echo "Built: bin/tint_convert_cli ($(du -h bin/tint_convert_cli | cut -f1))"
