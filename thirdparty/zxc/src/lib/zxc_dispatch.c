/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file zxc_dispatch.c
 * @brief Runtime CPU feature detection and SIMD dispatch layer.
 *
 * Detects AVX2/AVX512/NEON at runtime and routes compress/decompress calls
 * to the best available implementation via lazy-initialised function pointers.
 * Also contains the public one-shot buffer API (@ref zxc_compress,
 * @ref zxc_decompress, @ref zxc_get_decompressed_size).
 */

#include "../../include/zxc_dict.h"
#include "../../include/zxc_error.h"
#include "../../include/zxc_seekable.h"
#include "zxc_internal.h"

/*
 * ZXC_DISABLE_SIMD => force ZXC_ONLY_DEFAULT so the dispatcher never selects
 * an AVX2/AVX512/NEON variant.
 */
#if defined(ZXC_DISABLE_SIMD) && !defined(ZXC_ONLY_DEFAULT)
#define ZXC_ONLY_DEFAULT
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#if defined(_M_X64)
#include <immintrin.h>  // _xgetbv (x86-specific header; x64 AVX state check)
#endif
#endif

#if defined(__linux__) && (defined(__arm__) || defined(_M_ARM))
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

/*
 * ============================================================================
 * PROTOTYPES FOR MULTI-VERSIONED VARIANTS
 * ============================================================================
 * These are compiled in separate translation units with different flags.
 */

// Decompression Prototypes
int zxc_decompress_chunk_wrapper_default(const zxc_cctx_t* RESTRICT ctx,
                                         const uint8_t* RESTRICT src, const size_t src_sz,
                                         uint8_t* RESTRICT dst, const size_t dst_cap);
int zxc_decompress_chunk_wrapper_dict_default(const zxc_cctx_t* RESTRICT ctx,
                                              const uint8_t* RESTRICT src, const size_t src_sz,
                                              uint8_t* RESTRICT dst, const size_t dst_cap);
int zxc_decompress_chunk_wrapper_safe_default(const zxc_cctx_t* RESTRICT ctx,
                                              const uint8_t* RESTRICT src, const size_t src_sz,
                                              uint8_t* RESTRICT dst, const size_t dst_cap);

#ifndef ZXC_ONLY_DEFAULT
#if defined(__x86_64__) || defined(_M_X64)
int zxc_decompress_chunk_wrapper_avx2(const zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                      const size_t src_sz, uint8_t* RESTRICT dst,
                                      const size_t dst_cap);
int zxc_decompress_chunk_wrapper_dict_avx2(const zxc_cctx_t* RESTRICT ctx,
                                           const uint8_t* RESTRICT src, const size_t src_sz,
                                           uint8_t* RESTRICT dst, const size_t dst_cap);
int zxc_decompress_chunk_wrapper_avx512(const zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                        const size_t src_sz, uint8_t* RESTRICT dst,
                                        const size_t dst_cap);
int zxc_decompress_chunk_wrapper_dict_avx512(const zxc_cctx_t* RESTRICT ctx,
                                             const uint8_t* RESTRICT src, const size_t src_sz,
                                             uint8_t* RESTRICT dst, const size_t dst_cap);
int zxc_decompress_chunk_wrapper_safe_avx2(const zxc_cctx_t* RESTRICT ctx,
                                           const uint8_t* RESTRICT src, const size_t src_sz,
                                           uint8_t* RESTRICT dst, const size_t dst_cap);
int zxc_decompress_chunk_wrapper_safe_avx512(const zxc_cctx_t* RESTRICT ctx,
                                             const uint8_t* RESTRICT src, const size_t src_sz,
                                             uint8_t* RESTRICT dst, const size_t dst_cap);
int zxc_decompress_chunk_wrapper_sse2(const zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                      const size_t src_sz, uint8_t* RESTRICT dst,
                                      const size_t dst_cap);
int zxc_decompress_chunk_wrapper_dict_sse2(const zxc_cctx_t* RESTRICT ctx,
                                           const uint8_t* RESTRICT src, const size_t src_sz,
                                           uint8_t* RESTRICT dst, const size_t dst_cap);
int zxc_decompress_chunk_wrapper_safe_sse2(const zxc_cctx_t* RESTRICT ctx,
                                           const uint8_t* RESTRICT src, const size_t src_sz,
                                           uint8_t* RESTRICT dst, const size_t dst_cap);
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
int zxc_decompress_chunk_wrapper_neon(const zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                      const size_t src_sz, uint8_t* RESTRICT dst,
                                      const size_t dst_cap);
int zxc_decompress_chunk_wrapper_dict_neon(const zxc_cctx_t* RESTRICT ctx,
                                           const uint8_t* RESTRICT src, const size_t src_sz,
                                           uint8_t* RESTRICT dst, const size_t dst_cap);
int zxc_decompress_chunk_wrapper_safe_neon(const zxc_cctx_t* RESTRICT ctx,
                                           const uint8_t* RESTRICT src, const size_t src_sz,
                                           uint8_t* RESTRICT dst, const size_t dst_cap);
#endif
#endif

// Compression Prototypes
int zxc_compress_chunk_wrapper_default(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                       const size_t src_sz, uint8_t* RESTRICT dst,
                                       const size_t dst_cap);

// Huffman Prototypes (variant TUs of zxc_huffman.c). The compressor and
// decompressor variants resolve their Huffman calls to the matching suffixed
// symbol at compile time (zero dispatch overhead in the hot path); the thin
// wrappers below expose the un-suffixed names for tests and external callers.
int zxc_huf_build_code_lengths_default(const uint32_t* RESTRICT freq, uint8_t* RESTRICT code_len,
                                       void* RESTRICT scratch);
int zxc_huf_encode_section_default(const uint8_t* RESTRICT literals, const size_t n_literals,
                                   const uint8_t* RESTRICT code_len, uint8_t* RESTRICT dst,
                                   const size_t dst_cap);
int zxc_huf_decode_section_default(const uint8_t* RESTRICT payload, const size_t payload_size,
                                   uint8_t* RESTRICT dst, const size_t n_literals);
int zxc_huf_encode_section_dict_default(const uint8_t* RESTRICT literals, const size_t n_literals,
                                        const uint8_t* RESTRICT code_len, uint8_t* RESTRICT dst,
                                        const size_t dst_cap);
int zxc_huf_decode_section_dict_default(const uint8_t* RESTRICT payload, const size_t payload_size,
                                        uint8_t* RESTRICT dst, const size_t n_literals,
                                        const zxc_huf_dec_entry_t* RESTRICT table);
int zxc_huf_build_dec_table_default(const uint8_t* RESTRICT code_len,
                                    zxc_huf_dec_entry_t* RESTRICT table);
void zxc_huf_pack_lengths_default(const uint8_t* RESTRICT code_len, uint8_t* RESTRICT out);
int zxc_huf_unpack_lengths_default(const uint8_t* RESTRICT in, uint8_t* RESTRICT code_len);

#if defined(__x86_64__) || defined(_M_X64)
int zxc_compress_chunk_wrapper_avx2(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                    const size_t src_sz, uint8_t* RESTRICT dst,
                                    const size_t dst_cap);
int zxc_compress_chunk_wrapper_avx512(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                      const size_t src_sz, uint8_t* RESTRICT dst,
                                      const size_t dst_cap);
int zxc_compress_chunk_wrapper_sse2(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                    const size_t src_sz, uint8_t* RESTRICT dst,
                                    const size_t dst_cap);
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
int zxc_compress_chunk_wrapper_neon(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                    const size_t src_sz, uint8_t* RESTRICT dst,
                                    const size_t dst_cap);
#endif

/*
 * ============================================================================
 * CPU DETECTION LOGIC
 * ============================================================================
 */

/**
 * @enum zxc_cpu_feature_t
 * @brief Detected CPU SIMD capability level.
 */
typedef enum {
    ZXC_CPU_GENERIC = 0, /**< @brief Scalar-only fallback.   */
    ZXC_CPU_AVX2 = 1,    /**< @brief x86-64 AVX2 available.  */
    ZXC_CPU_AVX512 = 2,  /**< @brief x86-64 AVX-512F+BW available. */
    ZXC_CPU_NEON = 3,    /**< @brief ARM NEON available.      */
    ZXC_CPU_SSE2 = 4     /**< @brief x86 SSE2 available (no AVX2); x86-64 baseline. */
} zxc_cpu_feature_t;

/**
 * @brief Probes the running CPU for SIMD support.
 *
 * Uses CPUID on x86-64 (MSVC and GCC/Clang paths), `getauxval` on
 * 32-bit ARM Linux, and compile-time constants on AArch64.
 *
 * @return The highest @ref zxc_cpu_feature_t level supported.
 */
// LCOV_EXCL_START
static zxc_cpu_feature_t zxc_detect_cpu_features(void) {
#ifdef ZXC_ONLY_DEFAULT
    return ZXC_CPU_GENERIC;
#else
    zxc_cpu_feature_t features = ZXC_CPU_GENERIC;

#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
    // AVX2/AVX512 need OS-enabled YMM/ZMM state: gate on OSXSAVE + XGETBV/XCR0,
    // not CPUID alone (else a VEX/EVEX op faults #UD when the OS hasn't enabled it).
    int regs[4];
    int sse2 = 0;
    int avx2 = 0;
    int avx512 = 0;

    __cpuid(regs, 1);
    if (regs[3] & (1 << 26)) sse2 = 1;  // SSE2
    if (regs[2] & (1 << 27)) {          // OSXSAVE
        const unsigned long long xcr0 = _xgetbv(0);
        if ((xcr0 & 0x6) == 0x6) {  // SSE+YMM enabled
            __cpuidex(regs, 7, 0);
            if (regs[1] & (1 << 5)) avx2 = 1;
            // AVX512 also needs XCR0[5..7] (opmask/ZMM)
            if ((regs[1] & (1 << 16)) && (regs[1] & (1 << 30)) && (xcr0 & 0xE0) == 0xE0) avx512 = 1;
        }
    }

    if (avx512) {
        features = ZXC_CPU_AVX512;
    } else if (avx2) {
        features = ZXC_CPU_AVX2;
    } else if (sse2) {
        features = ZXC_CPU_SSE2;
    }
#else
    // GCC/Clang built-in detection
    __builtin_cpu_init();

    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw")) {
        features = ZXC_CPU_AVX512;
    } else if (__builtin_cpu_supports("avx2")) {
        features = ZXC_CPU_AVX2;
    } else if (__builtin_cpu_supports("sse2")) {
        features = ZXC_CPU_SSE2;
    }
#endif

#elif defined(__aarch64__) || defined(_M_ARM64)
    // ARM64 usually guarantees NEON
    features = ZXC_CPU_NEON;

#elif defined(__arm__) || defined(_M_ARM)
    // ARM32 Runtime detection for Linux
#if defined(__linux__)
    const unsigned long hwcaps = getauxval(AT_HWCAP);
    if (hwcaps & HWCAP_NEON) {
        features = ZXC_CPU_NEON;
    }
#else
// Fallback for non-Linux: rely on compiler flags.
// If compiled with -mfpu=neon, we assume target supports it.
// Otherwise, safe default is GENERIC.
#if defined(__ARM_NEON)
    features = ZXC_CPU_NEON;
#endif
#endif
#endif

    return features;
#endif
}
// LCOV_EXCL_STOP

/*
 * ============================================================================
 * DISPATCHERS
 * ============================================================================
 * We use a function pointer initialized on first use (lazy initialization).
 */

/** @brief Function pointer type for the chunk decompressor. */
typedef int (*zxc_decompress_func_t)(const zxc_cctx_t* RESTRICT, const uint8_t* RESTRICT,
                                     const size_t, uint8_t* RESTRICT, const size_t);
/** @brief Function pointer type for the chunk compressor. */
typedef int (*zxc_compress_func_t)(zxc_cctx_t* RESTRICT, const uint8_t* RESTRICT, const size_t,
                                   uint8_t* RESTRICT, const size_t);

/** @brief Lazily-resolved pointer to the best decompression variant. */
static ZXC_ATOMIC zxc_decompress_func_t zxc_decompress_ptr = (zxc_decompress_func_t)0;
/** @brief Lazily-resolved pointer to the best dict-decompression variant. */
static ZXC_ATOMIC zxc_decompress_func_t zxc_decompress_dict_ptr = (zxc_decompress_func_t)0;
/** @brief Lazily-resolved pointer to the best safe-decompression variant. */
static ZXC_ATOMIC zxc_decompress_func_t zxc_decompress_safe_ptr = (zxc_decompress_func_t)0;
/** @brief Lazily-resolved pointer to the best compression variant. */
static ZXC_ATOMIC zxc_compress_func_t zxc_compress_ptr = (zxc_compress_func_t)0;

/**
 * @brief First-call initialiser for the decompression dispatcher.
 *
 * Detects CPU features, selects the best implementation, stores the
 * pointer atomically, then tail-calls into it.
 *
 * @param[in]  ctx      Decompression context (its @c dict_size picks the dict variant).
 * @param[in]  src      Compressed input chunk.
 * @param[in]  src_sz   Size of @p src in bytes.
 * @param[out] dst      Destination buffer for decompressed data.
 * @param[in]  dst_cap  Capacity of @p dst in bytes.
 * @return Result of the selected variant: decompressed size, or negative
 *         @ref zxc_error_t.
 */
// LCOV_EXCL_START
static int zxc_decompress_dispatch_init(const zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                        const size_t src_sz, uint8_t* RESTRICT dst,
                                        const size_t dst_cap) {
    const zxc_cpu_feature_t cpu = zxc_detect_cpu_features();
    zxc_decompress_func_t zxc_decompress_ptr_local = NULL;
    zxc_decompress_func_t zxc_decompress_dict_ptr_local = NULL;

#ifndef ZXC_ONLY_DEFAULT
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu == ZXC_CPU_AVX512) {
        zxc_decompress_ptr_local = zxc_decompress_chunk_wrapper_avx512;
        zxc_decompress_dict_ptr_local = zxc_decompress_chunk_wrapper_dict_avx512;
    } else if (cpu == ZXC_CPU_AVX2) {
        zxc_decompress_ptr_local = zxc_decompress_chunk_wrapper_avx2;
        zxc_decompress_dict_ptr_local = zxc_decompress_chunk_wrapper_dict_avx2;
    } else if (cpu == ZXC_CPU_SSE2) {
        zxc_decompress_ptr_local = zxc_decompress_chunk_wrapper_sse2;
        zxc_decompress_dict_ptr_local = zxc_decompress_chunk_wrapper_dict_sse2;
    } else {
        zxc_decompress_ptr_local = zxc_decompress_chunk_wrapper_default;
        zxc_decompress_dict_ptr_local = zxc_decompress_chunk_wrapper_dict_default;
    }
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
    // cppcheck-suppress knownConditionTrueFalse
    if (cpu == ZXC_CPU_NEON) {
        zxc_decompress_ptr_local = zxc_decompress_chunk_wrapper_neon;
        zxc_decompress_dict_ptr_local = zxc_decompress_chunk_wrapper_dict_neon;
    } else {
        zxc_decompress_ptr_local = zxc_decompress_chunk_wrapper_default;
        zxc_decompress_dict_ptr_local = zxc_decompress_chunk_wrapper_dict_default;
    }
#else
    (void)cpu;
    zxc_decompress_ptr_local = zxc_decompress_chunk_wrapper_default;
    zxc_decompress_dict_ptr_local = zxc_decompress_chunk_wrapper_dict_default;
#endif
#else
    (void)cpu;
    zxc_decompress_ptr_local = zxc_decompress_chunk_wrapper_default;
    zxc_decompress_dict_ptr_local = zxc_decompress_chunk_wrapper_dict_default;
#endif

#if ZXC_USE_C11_ATOMICS
    atomic_store_explicit(&zxc_decompress_ptr, zxc_decompress_ptr_local, memory_order_release);
    atomic_store_explicit(&zxc_decompress_dict_ptr, zxc_decompress_dict_ptr_local,
                          memory_order_release);
#else
    zxc_decompress_ptr = zxc_decompress_ptr_local;
    zxc_decompress_dict_ptr = zxc_decompress_dict_ptr_local;
#endif
    return (ctx->dict_size ? zxc_decompress_dict_ptr_local : zxc_decompress_ptr_local)(
        ctx, src, src_sz, dst, dst_cap);
}
// LCOV_EXCL_STOP

/**
 * @brief First-call initialiser for the safe-decompression dispatcher.
 *
 * Mirrors @ref zxc_decompress_dispatch_init but selects the `_safe_*`
 * decoder variants used by @ref zxc_decompress_block_safe.
 *
 * @param[in]  ctx      Decompression context.
 * @param[in]  src      Compressed input chunk.
 * @param[in]  src_sz   Size of @p src in bytes.
 * @param[out] dst      Destination buffer (strict: exact uncompressed size).
 * @param[in]  dst_cap  Capacity of @p dst in bytes.
 * @return Result of the selected variant: decompressed size, or negative
 *         @ref zxc_error_t.
 */
// LCOV_EXCL_START
static int zxc_decompress_safe_dispatch_init(const zxc_cctx_t* RESTRICT ctx,
                                             const uint8_t* RESTRICT src, const size_t src_sz,
                                             uint8_t* RESTRICT dst, const size_t dst_cap) {
    const zxc_cpu_feature_t cpu = zxc_detect_cpu_features();
    zxc_decompress_func_t zxc_decompress_safe_ptr_local = NULL;

#ifndef ZXC_ONLY_DEFAULT
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu == ZXC_CPU_AVX512)
        zxc_decompress_safe_ptr_local = zxc_decompress_chunk_wrapper_safe_avx512;
    else if (cpu == ZXC_CPU_AVX2)
        zxc_decompress_safe_ptr_local = zxc_decompress_chunk_wrapper_safe_avx2;
    else if (cpu == ZXC_CPU_SSE2)
        zxc_decompress_safe_ptr_local = zxc_decompress_chunk_wrapper_safe_sse2;
    else
        zxc_decompress_safe_ptr_local = zxc_decompress_chunk_wrapper_safe_default;
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
    // cppcheck-suppress knownConditionTrueFalse
    if (cpu == ZXC_CPU_NEON)
        zxc_decompress_safe_ptr_local = zxc_decompress_chunk_wrapper_safe_neon;
    else
        zxc_decompress_safe_ptr_local = zxc_decompress_chunk_wrapper_safe_default;
#else
    (void)cpu;
    zxc_decompress_safe_ptr_local = zxc_decompress_chunk_wrapper_safe_default;
#endif
#else
    (void)cpu;
    zxc_decompress_safe_ptr_local = zxc_decompress_chunk_wrapper_safe_default;
#endif

#if ZXC_USE_C11_ATOMICS
    atomic_store_explicit(&zxc_decompress_safe_ptr, zxc_decompress_safe_ptr_local,
                          memory_order_release);
#else
    zxc_decompress_safe_ptr = zxc_decompress_safe_ptr_local;
#endif
    return zxc_decompress_safe_ptr_local(ctx, src, src_sz, dst, dst_cap);
}
// LCOV_EXCL_STOP

/**
 * @brief First-call initialiser for the compression dispatcher.
 *
 * Detects CPU features, selects the best implementation, stores the
 * pointer atomically, then tail-calls into it.
 *
 * @param[in,out] ctx      Compression context.
 * @param[in]     src      Uncompressed input chunk.
 * @param[in]     src_sz   Size of @p src in bytes.
 * @param[out]    dst      Destination buffer for the compressed chunk.
 * @param[in]     dst_cap  Capacity of @p dst in bytes.
 * @return Result of the selected variant: compressed size, or negative
 *         @ref zxc_error_t.
 */
// LCOV_EXCL_START
static int zxc_compress_dispatch_init(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                      const size_t src_sz, uint8_t* RESTRICT dst,
                                      const size_t dst_cap) {
    const zxc_cpu_feature_t cpu = zxc_detect_cpu_features();
    zxc_compress_func_t zxc_compress_ptr_local = NULL;

#ifndef ZXC_ONLY_DEFAULT
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu == ZXC_CPU_AVX512)
        zxc_compress_ptr_local = zxc_compress_chunk_wrapper_avx512;
    else if (cpu == ZXC_CPU_AVX2)
        zxc_compress_ptr_local = zxc_compress_chunk_wrapper_avx2;
    else if (cpu == ZXC_CPU_SSE2)
        zxc_compress_ptr_local = zxc_compress_chunk_wrapper_sse2;
    else
        zxc_compress_ptr_local = zxc_compress_chunk_wrapper_default;
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
    // cppcheck-suppress knownConditionTrueFalse
    if (cpu == ZXC_CPU_NEON)
        zxc_compress_ptr_local = zxc_compress_chunk_wrapper_neon;
    else
        zxc_compress_ptr_local = zxc_compress_chunk_wrapper_default;
#else
    (void)cpu;
    zxc_compress_ptr_local = zxc_compress_chunk_wrapper_default;
#endif
#else
    (void)cpu;
    zxc_compress_ptr_local = zxc_compress_chunk_wrapper_default;
#endif

#if ZXC_USE_C11_ATOMICS
    atomic_store_explicit(&zxc_compress_ptr, zxc_compress_ptr_local, memory_order_release);
#else
    zxc_compress_ptr = zxc_compress_ptr_local;
#endif
    return zxc_compress_ptr_local(ctx, src, src_sz, dst, dst_cap);
}
// LCOV_EXCL_STOP

/**
 * @brief Public decompression dispatcher (calls lazily-resolved implementation).
 *
 * @param[in,out] ctx    Decompression context.
 * @param[in]     src    Compressed input chunk (header + payload + optional checksum).
 * @param[in]     src_sz Size of @p src in bytes.
 * @param[out]    dst    Destination buffer for decompressed data.
 * @param[in]     dst_cap Capacity of @p dst.
 * @return Decompressed size in bytes, or a negative @ref zxc_error_t code.
 */
int zxc_decompress_chunk_wrapper(const zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                 const size_t src_sz, uint8_t* RESTRICT dst, const size_t dst_cap) {
    /* dict_size is constant for a stream; this per-block branch (outside the decode
     * loop) routes to the dict variant only when a dictionary is active, so the
     * no-dict path runs the dict-free chunk wrapper (identical codegen to main). */
#if ZXC_USE_C11_ATOMICS
    const zxc_decompress_func_t func = atomic_load_explicit(
        ctx->dict_size ? &zxc_decompress_dict_ptr : &zxc_decompress_ptr, memory_order_acquire);
#else
    const zxc_decompress_func_t func =
        ctx->dict_size ? zxc_decompress_dict_ptr : zxc_decompress_ptr;
#endif
    if (UNLIKELY(!func)) return zxc_decompress_dispatch_init(ctx, src, src_sz, dst, dst_cap);
    return func(ctx, src, src_sz, dst, dst_cap);
}

/**
 * @brief Internal safe-decompression dispatcher (strict dst_capacity == uncompressed_size).
 *
 * Calls the lazily-resolved `_safe_*` variant, running first-call init if needed.
 *
 * @param[in]  ctx      Decompression context.
 * @param[in]  src      Compressed input chunk.
 * @param[in]  src_sz   Size of @p src in bytes.
 * @param[out] dst      Destination buffer (capacity == exact uncompressed size).
 * @param[in]  dst_cap  Capacity of @p dst in bytes.
 * @return Decompressed size in bytes, or a negative @ref zxc_error_t.
 */
static int zxc_decompress_chunk_wrapper_safe_public(const zxc_cctx_t* RESTRICT ctx,
                                                    const uint8_t* RESTRICT src,
                                                    const size_t src_sz, uint8_t* RESTRICT dst,
                                                    const size_t dst_cap) {
#if ZXC_USE_C11_ATOMICS
    const zxc_decompress_func_t func =
        atomic_load_explicit(&zxc_decompress_safe_ptr, memory_order_acquire);
#else
    const zxc_decompress_func_t func = zxc_decompress_safe_ptr;
#endif
    if (UNLIKELY(!func)) return zxc_decompress_safe_dispatch_init(ctx, src, src_sz, dst, dst_cap);
    return func(ctx, src, src_sz, dst, dst_cap);
}

/**
 * @brief Public compression dispatcher (calls lazily-resolved implementation).
 *
 * @param[in,out] ctx    Compression context.
 * @param[in]     src    Uncompressed input chunk.
 * @param[in]     src_sz Size of @p src in bytes.
 * @param[out]    dst    Destination buffer for compressed data.
 * @param[in]     dst_cap Capacity of @p dst.
 * @return Compressed size in bytes, or a negative @ref zxc_error_t code.
 */
int zxc_compress_chunk_wrapper(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                               const size_t src_sz, uint8_t* RESTRICT dst, const size_t dst_cap) {
#if ZXC_USE_C11_ATOMICS
    const zxc_compress_func_t func = atomic_load_explicit(&zxc_compress_ptr, memory_order_acquire);
#else
    const zxc_compress_func_t func = zxc_compress_ptr;
#endif
    if (UNLIKELY(!func)) return zxc_compress_dispatch_init(ctx, src, src_sz, dst, dst_cap);
    return func(ctx, src, src_sz, dst, dst_cap);
}

/*
 * ============================================================================
 * HUFFMAN TRAMPOLINES
 * ============================================================================
 * The Huffman codec is built per-variant (default / avx2 / avx512 / neon)
 * alongside zxc_compress.c and zxc_decompress.c, so the LZ77 stages and the
 * Huffman stage in a given variant share the same ISA flags (e.g. -mbmi2 on
 * the AVX2/AVX512 variants). The compress/decompress variant TUs resolve
 * their Huffman calls to the matching suffixed symbol at compile time, so
 * the production hot path has zero dispatch overhead.
 *
 * These thin wrappers exist only for tests and external callers that link
 * against the un-suffixed names. They forward to the default (scalar) variant.
 */
/**
 * @brief Build length-limited per-symbol Huffman code lengths from frequencies.
 *
 * Un-suffixed entry forwarding to @ref zxc_huf_build_code_lengths_default; full
 * contract in @c zxc_internal.h.
 *
 * @param[in]  freq      Per-symbol frequency counts.
 * @param[out] code_len  Per-symbol code lengths.
 * @param[in]  scratch   Caller-provided build scratch buffer.
 * @return `ZXC_OK` on success, negative `zxc_error_t` on failure.
 */
int zxc_huf_build_code_lengths(const uint32_t* RESTRICT freq, uint8_t* RESTRICT code_len,
                               void* RESTRICT scratch) {
    return zxc_huf_build_code_lengths_default(freq, code_len, scratch);
}

/**
 * @brief Encode a full Huffman literal section (lengths header + streams).
 *
 * Un-suffixed entry forwarding to @ref zxc_huf_encode_section_default; full
 * contract in @c zxc_internal.h.
 *
 * @param[in]  literals    Source literal bytes.
 * @param[in]  n_literals  Number of source bytes.
 * @param[in]  code_len    Per-symbol code lengths.
 * @param[out] dst         Destination section buffer.
 * @param[in]  dst_cap     Capacity of @p dst in bytes.
 * @return Bytes written on success, negative `zxc_error_t` on failure.
 */
int zxc_huf_encode_section(const uint8_t* RESTRICT literals, const size_t n_literals,
                           const uint8_t* RESTRICT code_len, uint8_t* RESTRICT dst,
                           const size_t dst_cap) {
    return zxc_huf_encode_section_default(literals, n_literals, code_len, dst, dst_cap);
}

/**
 * @brief Decode a full Huffman literal section.
 *
 * Un-suffixed entry forwarding to @ref zxc_huf_decode_section_default; full
 * contract in @c zxc_internal.h.
 *
 * @param[in]  payload       Section payload.
 * @param[in]  payload_size  Payload length in bytes.
 * @param[out] dst           Destination buffer.
 * @param[in]  n_literals    Expected number of decoded bytes.
 * @return `ZXC_OK` on success, negative `zxc_error_t` on failure.
 */
int zxc_huf_decode_section(const uint8_t* RESTRICT payload, const size_t payload_size,
                           uint8_t* RESTRICT dst, const size_t n_literals) {
    return zxc_huf_decode_section_default(payload, payload_size, dst, n_literals);
}

/**
 * @brief Encode a Huffman literal section without lengths header (shared dict table).
 *
 * Un-suffixed entry forwarding to @ref zxc_huf_encode_section_dict_default; full
 * contract in @c zxc_internal.h.
 *
 * @param[in]  literals    Source literal bytes.
 * @param[in]  n_literals  Number of source bytes.
 * @param[in]  code_len    Per-symbol code lengths (from the shared dict table).
 * @param[out] dst         Destination section buffer.
 * @param[in]  dst_cap     Capacity of @p dst in bytes.
 * @return Bytes written on success, negative `zxc_error_t` on failure.
 */
int zxc_huf_encode_section_dict(const uint8_t* RESTRICT literals, const size_t n_literals,
                                const uint8_t* RESTRICT code_len, uint8_t* RESTRICT dst,
                                const size_t dst_cap) {
    return zxc_huf_encode_section_dict_default(literals, n_literals, code_len, dst, dst_cap);
}

/**
 * @brief Decode a Huffman literal section using a prebuilt shared-dict table.
 *
 * Un-suffixed entry forwarding to @ref zxc_huf_decode_section_dict_default; full
 * contract in @c zxc_internal.h.
 *
 * @param[in]  payload       Section payload.
 * @param[in]  payload_size  Payload length in bytes.
 * @param[out] dst           Destination buffer.
 * @param[in]  n_literals    Expected number of decoded bytes.
 * @param[in]  table         Prebuilt shared-dict decode table.
 * @return `ZXC_OK` on success, negative `zxc_error_t` on failure.
 */
int zxc_huf_decode_section_dict(const uint8_t* RESTRICT payload, const size_t payload_size,
                                uint8_t* RESTRICT dst, const size_t n_literals,
                                const zxc_huf_dec_entry_t* RESTRICT table) {
    return zxc_huf_decode_section_dict_default(payload, payload_size, dst, n_literals, table);
}

/**
 * @brief Build the multi-symbol Huffman decode table from code lengths.
 *
 * Un-suffixed entry forwarding to @ref zxc_huf_build_dec_table_default; full
 * contract in @c zxc_internal.h.
 *
 * @param[in]  code_len  Per-symbol code lengths.
 * @param[out] table     Destination decode table.
 * @return `ZXC_OK` on success, `ZXC_ERROR_CORRUPT_DATA` on invalid lengths.
 */
int zxc_huf_build_dec_table(const uint8_t* RESTRICT code_len, zxc_huf_dec_entry_t* RESTRICT table) {
    return zxc_huf_build_dec_table_default(code_len, table);
}

/**
 * @brief Pack per-symbol code lengths into the 128-byte nibble header.
 *
 * Un-suffixed entry forwarding to @ref zxc_huf_pack_lengths_default; full
 * contract in @c zxc_internal.h.
 *
 * @param[in]  code_len  Per-symbol code lengths (one byte each).
 * @param[out] out       Destination 128-byte packed header.
 */
void zxc_huf_pack_lengths(const uint8_t* RESTRICT code_len, uint8_t* RESTRICT out) {
    zxc_huf_pack_lengths_default(code_len, out);
}

/**
 * @brief Unpack and validate a 128-byte packed lengths header.
 *
 * Un-suffixed entry forwarding to @ref zxc_huf_unpack_lengths_default; full
 * contract in @c zxc_internal.h.
 *
 * @param[in]  in        128-byte packed lengths header.
 * @param[out] code_len  Destination per-symbol code lengths.
 * @return `ZXC_OK` on success, `ZXC_ERROR_CORRUPT_DATA` on invalid lengths.
 */
int zxc_huf_unpack_lengths(const uint8_t* RESTRICT in, uint8_t* RESTRICT code_len) {
    return zxc_huf_unpack_lengths_default(in, code_len);
}

/*
 * ============================================================================
 * PUBLIC UTILITY API
 * ============================================================================
 * These wrapper functions provide a simplified interface by managing context
 * allocation and looping over blocks. They call the dispatched wrappers above.
 */

/**
 * @brief Compresses an entire buffer in one call.
 *
 * Manages context allocation internally, loops over blocks, writes the
 * file header / EOF block / footer, and accumulates the global checksum.
 *
 * @param[in]  src              Uncompressed input data.
 * @param[in]  src_size         Size of @p src in bytes.
 * @param[out] dst              Destination buffer (use zxc_compress_bound() to size).
 * @param[in]  dst_capacity     Capacity of @p dst.
 * @param[in]  opts             Compression options (level, block size, checksum,
 *                              dictionary, seekable, threads), or NULL for defaults.
 * @return Total compressed size in bytes, or a negative @ref zxc_error_t code.
 */
// cppcheck-suppress unusedFunction
int64_t zxc_compress(const void* RESTRICT src, const size_t src_size, void* RESTRICT dst,
                     const size_t dst_capacity, const zxc_compress_opts_t* opts) {
    if (UNLIKELY(!dst || dst_capacity == 0 || (src_size > 0 && !src))) return ZXC_ERROR_NULL_INPUT;

    const int checksum_enabled = opts ? opts->checksum_enabled : 0;
    const int seekable = opts ? opts->seekable : 0;
    const int level = (opts && opts->level > 0) ? opts->level : ZXC_LEVEL_DEFAULT;
    const size_t block_size =
        (opts && opts->block_size > 0) ? opts->block_size : ZXC_BLOCK_SIZE_DEFAULT;
    const uint8_t* dict = opts ? (const uint8_t*)opts->dict : NULL;
    const size_t dict_size = (opts && opts->dict) ? opts->dict_size : 0;
    const uint8_t* dict_huf = (opts && opts->dict) ? (const uint8_t*)opts->dict_huf : NULL;

    if (UNLIKELY(dict_size > ZXC_DICT_SIZE_MAX)) return ZXC_ERROR_DICT_TOO_LARGE;
    if (UNLIKELY(!zxc_validate_block_size(block_size))) return ZXC_ERROR_BAD_BLOCK_SIZE;

    const uint32_t did = (dict && dict_size > 0) ? zxc_dict_id(dict, dict_size, dict_huf) : 0;

    const uint8_t* ip = (const uint8_t*)src;
    uint8_t* op = (uint8_t*)dst;
    const uint8_t* op_start = op;
    const uint8_t* op_end = op + dst_capacity;
    uint32_t global_hash = 0;
    zxc_cctx_t ctx;

    const size_t eff_chunk =
        dict_size > 0 ? zxc_block_size_ceil(dict_size + block_size) : block_size;
    // LCOV_EXCL_START
    if (UNLIKELY(zxc_cctx_init(&ctx, eff_chunk, 1, level, checksum_enabled, dict_size) != ZXC_OK))
        return ZXC_ERROR_MEMORY;
    // LCOV_EXCL_STOP
    if (UNLIKELY(zxc_cctx_attach_dict_huf(&ctx, dict_huf) != ZXC_OK)) {
        // LCOV_EXCL_START
        zxc_cctx_free(&ctx);
        return ZXC_ERROR_CORRUPT_DATA;
        // LCOV_EXCL_STOP
    }

    /* Dict input buffer: [dict_content | block_data] for the encoder, carved
     * into the cctx workspace (NULL when no dictionary is active). */
    uint8_t* const dict_input = ctx.dict_buffer;
    if (dict_input) ZXC_MEMCPY(dict_input, dict, dict_size);

    const int h_val =
        zxc_write_file_header(op, (size_t)(op_end - op), block_size, checksum_enabled, did);
    // LCOV_EXCL_START
    if (UNLIKELY(h_val < 0)) {
        zxc_cctx_free(&ctx);
        return h_val;
    }
    // LCOV_EXCL_STOP
    op += h_val;

    /* Seekable: dynamic array for per-block compressed sizes */
    uint32_t* seek_comp = NULL;
    uint32_t seek_count = 0;
    uint32_t seek_cap = 0;
    if (seekable) {
        const size_t block_count = src_size / block_size;
        if (UNLIKELY(block_count > (size_t)UINT32_MAX - 2)) {
            // LCOV_EXCL_START
            zxc_cctx_free(&ctx);
            return ZXC_ERROR_BAD_BLOCK_SIZE;
            // LCOV_EXCL_STOP
        }
        seek_cap = (uint32_t)(block_count + 2);
        seek_comp = (uint32_t*)ZXC_MALLOC(seek_cap * sizeof(uint32_t));
        // LCOV_EXCL_START
        if (UNLIKELY(!seek_comp)) {
            zxc_cctx_free(&ctx);
            return ZXC_ERROR_MEMORY;
        }
        // LCOV_EXCL_STOP
    }

    size_t pos = 0;
    while (pos < src_size) {
        const size_t chunk_len = (src_size - pos > block_size) ? block_size : (src_size - pos);
        const size_t rem_cap = (size_t)(op_end - op);

        int res;
        if (dict_input) {
            ZXC_MEMCPY(dict_input + dict_size, ip + pos, chunk_len);
            res = zxc_compress_chunk_wrapper(&ctx, dict_input, dict_size + chunk_len, op, rem_cap);
        } else {
            res = zxc_compress_chunk_wrapper(&ctx, ip + pos, chunk_len, op, rem_cap);
        }
        if (UNLIKELY(res < 0)) {
            ZXC_FREE(seek_comp);
            zxc_cctx_free(&ctx);
            return res;
        }

        if (checksum_enabled) {
            // Update Global Hash (Rotation + XOR)
            // Block checksum is at the end of the written block data
            if (LIKELY(res >= ZXC_GLOBAL_CHECKSUM_SIZE)) {
                const uint32_t block_hash = zxc_le32(op + res - ZXC_GLOBAL_CHECKSUM_SIZE);
                global_hash = zxc_hash_combine_rotate(global_hash, block_hash);
            }
        }

        /* Seekable: record compressed block size */
        if (seekable) {
            // LCOV_EXCL_START
            if (UNLIKELY(seek_count >= seek_cap)) {
                seek_cap = seek_cap * 2;
                uint32_t* nc = (uint32_t*)ZXC_REALLOC(seek_comp, seek_cap * sizeof(uint32_t));
                if (UNLIKELY(!nc)) {
                    ZXC_FREE(seek_comp);
                    zxc_cctx_free(&ctx);
                    return ZXC_ERROR_MEMORY;
                }
                seek_comp = nc;
            }
            // LCOV_EXCL_STOP
            seek_comp[seek_count] = (uint32_t)res;
            seek_count++;
        }

        op += res;
        pos += chunk_len;
    }

    zxc_cctx_free(&ctx);

    // Write EOF Block
    const size_t rem_cap = (size_t)(op_end - op);
    const zxc_block_header_t eof_bh = {
        .block_type = ZXC_BLOCK_EOF, .block_flags = 0, .reserved = 0, .comp_size = 0};
    const int eof_val = zxc_write_block_header(op, rem_cap, &eof_bh);
    // LCOV_EXCL_START
    if (UNLIKELY(eof_val < 0)) {
        ZXC_FREE(seek_comp);
        return eof_val;
    }
    // LCOV_EXCL_STOP
    op += eof_val;

    /* Seekable: write seek table between EOF block and footer */
    if (seekable && seek_count > 0) {
        const size_t st_cap = (size_t)(op_end - op);
        const int64_t st_val = zxc_write_seek_table(op, st_cap, seek_comp, seek_count);
        ZXC_FREE(seek_comp);
        if (UNLIKELY(st_val < 0)) return st_val;  // LCOV_EXCL_LINE
        op += st_val;
    } else {
        ZXC_FREE(seek_comp);
    }

    if (UNLIKELY((size_t)(op_end - op) < ZXC_FILE_FOOTER_SIZE))
        return ZXC_ERROR_DST_TOO_SMALL;  // LCOV_EXCL_LINE

    // Write 12-byte Footer: [Source Size (8)] + [Global Hash (4)]
    const int footer_val =
        zxc_write_file_footer(op, (size_t)(op_end - op), src_size, global_hash, checksum_enabled);
    if (UNLIKELY(footer_val < 0)) return footer_val;  // LCOV_EXCL_LINE
    op += footer_val;

    return (int64_t)(op - op_start);
}

/**
 * @brief Decompresses an entire buffer in one call.
 *
 * Validates the file header and footer, loops over compressed blocks,
 * and verifies the global checksum when enabled.
 *
 * @param[in]  src              Compressed input data.
 * @param[in]  src_size         Size of @p src in bytes.
 * @param[out] dst              Destination buffer for decompressed data.
 * @param[in]  dst_capacity     Capacity of @p dst.
 * @param[in]  opts             Decompression options (checksum verification,
 *                              dictionary, threads), or NULL for defaults.
 * @return Total decompressed size in bytes, or a negative @ref zxc_error_t code.
 */
// cppcheck-suppress unusedFunction
int64_t zxc_decompress(const void* RESTRICT src, const size_t src_size, void* RESTRICT dst,
                       const size_t dst_capacity, const zxc_decompress_opts_t* opts) {
    if (UNLIKELY(!src || src_size < ZXC_FILE_HEADER_SIZE + ZXC_FILE_FOOTER_SIZE ||
                 (!dst && dst_capacity != 0)))
        return ZXC_ERROR_NULL_INPUT;

    if (UNLIKELY(!dst || dst_capacity == 0)) {
        /* Empty-frame case (stored size == 0). */
        if (UNLIKELY(zxc_le32(src) != ZXC_MAGIC_WORD)) return ZXC_ERROR_NULL_INPUT;
        const uint8_t* footer = (const uint8_t*)src + src_size - ZXC_FILE_FOOTER_SIZE;
        return (zxc_le64(footer) == 0) ? 0 : (int64_t)ZXC_ERROR_DST_TOO_SMALL;
    }

    const int checksum_enabled = opts ? opts->checksum_enabled : 0;
    const uint8_t* dict = opts ? (const uint8_t*)opts->dict : NULL;
    const size_t dict_size = (opts && opts->dict) ? opts->dict_size : 0;
    const uint8_t* dict_huf = (opts && opts->dict) ? (const uint8_t*)opts->dict_huf : NULL;

    const uint8_t* ip = (const uint8_t*)src;
    const uint8_t* ip_end = ip + src_size;
    uint8_t* op = (uint8_t*)dst;
    const uint8_t* op_start = op;
    const uint8_t* op_end = op + dst_capacity;
    size_t runtime_chunk_size = 0;
    zxc_cctx_t ctx;

    int file_has_checksums = 0;
    uint32_t header_dict_id = 0;
    if (UNLIKELY(zxc_read_file_header(ip, src_size, &runtime_chunk_size, &file_has_checksums,
                                      &header_dict_id) != ZXC_OK ||
                 zxc_cctx_init(&ctx, runtime_chunk_size, 0, 0,
                               file_has_checksums && checksum_enabled, dict_size) != ZXC_OK)) {
        return ZXC_ERROR_BAD_HEADER;
    }

    /* Dictionary validation */
    if (header_dict_id != 0) {
        if (UNLIKELY(!dict || dict_size == 0)) {
            zxc_cctx_free(&ctx);
            return ZXC_ERROR_DICT_REQUIRED;
        }
        if (UNLIKELY(zxc_dict_id(dict, dict_size, dict_huf) != header_dict_id)) {
            zxc_cctx_free(&ctx);
            return ZXC_ERROR_DICT_MISMATCH;
        }
    }
    if (UNLIKELY(zxc_cctx_attach_dict_huf(&ctx, dict_huf) != ZXC_OK)) {
        // LCOV_EXCL_START
        zxc_cctx_free(&ctx);
        return ZXC_ERROR_CORRUPT_DATA;
        // LCOV_EXCL_STOP
    }

    ip += ZXC_FILE_HEADER_SIZE;

    const size_t work_sz = runtime_chunk_size + ZXC_DECOMPRESS_TAIL_PAD;

    /* Dict decode buffer: [dict_content | decode_space + PAD], carved into the
     * cctx workspace (NULL when no dictionary is active). */
    uint8_t* const dict_dec = ctx.dict_buffer;
    if (dict_dec) ZXC_MEMCPY(dict_dec, dict, dict_size);

    // Block decompression loop
    uint32_t global_hash = 0;

    while (ip < ip_end) {
        const size_t rem_src = (size_t)(ip_end - ip);
        zxc_block_header_t bh;
        // Read the block header to determine the compressed size
        if (UNLIKELY(zxc_read_block_header(ip, rem_src, &bh) != ZXC_OK)) {
            zxc_cctx_free(&ctx);
            return ZXC_ERROR_BAD_HEADER;
        }

        // Handle EOF block separately (not a real chunk to decompress)
        if (UNLIKELY(bh.block_type == ZXC_BLOCK_EOF)) {
            // EOF carries no payload; a non-zero comp_size is a malformed header.
            if (UNLIKELY(bh.comp_size != 0)) {
                zxc_cctx_free(&ctx);
                return ZXC_ERROR_BAD_HEADER;
            }
            // Footer is always the last ZXC_FILE_FOOTER_SIZE bytes of the source,
            // even when a seek table is inserted between EOF block and footer.
            // LCOV_EXCL_START
            if (UNLIKELY(src_size < ZXC_FILE_FOOTER_SIZE)) {
                zxc_cctx_free(&ctx);
                return ZXC_ERROR_SRC_TOO_SMALL;
            }
            // LCOV_EXCL_STOP
            const uint8_t* const footer = (const uint8_t*)src + src_size - ZXC_FILE_FOOTER_SIZE;

            // Validate source size matches what we decompressed
            const uint64_t stored_size = zxc_le64(footer);
            if (UNLIKELY(stored_size != (uint64_t)(op - op_start))) {
                zxc_cctx_free(&ctx);
                return ZXC_ERROR_CORRUPT_DATA;
            }

            // Validate global checksum if enabled and file has checksums
            if (checksum_enabled && file_has_checksums) {
                const uint32_t stored_hash = zxc_le32(footer + sizeof(uint64_t));
                if (UNLIKELY(stored_hash != global_hash)) {
                    zxc_cctx_free(&ctx);
                    return ZXC_ERROR_BAD_CHECKSUM;
                }
            }
            break;  // EOF reached, exit loop
        }

        int res;
        const size_t rem_cap = (size_t)(op_end - op);
        if (dict_dec) {
            /* Dict path: decode into bounce buffer with dict prefix so match
             * copies that reference dict content resolve naturally. */
            res = zxc_decompress_chunk_wrapper(&ctx, ip, rem_src, dict_dec + dict_size, work_sz);
            if (LIKELY(res > 0)) {
                if (UNLIKELY((size_t)res > rem_cap)) {
                    // LCOV_EXCL_START
                    zxc_cctx_free(&ctx);
                    return ZXC_ERROR_DST_TOO_SMALL;
                    // LCOV_EXCL_STOP
                }
                ZXC_MEMCPY(op, dict_dec + dict_size, (size_t)res);
            }
        } else if (LIKELY(rem_cap >= work_sz)) {
            // Fast path: decode directly into dst. Cap dst_cap to chunk_size + PAD
            res = zxc_decompress_chunk_wrapper(&ctx, ip, rem_src, op, work_sz);
        } else {
            // Safe path: decode into bounce buffer, then copy exact result.
            res = zxc_decompress_chunk_wrapper(&ctx, ip, rem_src, ctx.work_buf, ctx.work_buf_cap);
            if (LIKELY(res > 0)) {
                // LCOV_EXCL_START
                if (UNLIKELY((size_t)res > rem_cap)) {
                    zxc_cctx_free(&ctx);
                    return ZXC_ERROR_DST_TOO_SMALL;
                }
                // LCOV_EXCL_STOP
                ZXC_MEMCPY(op, ctx.work_buf, (size_t)res);
            }
        }
        if (UNLIKELY(res < 0)) {
            zxc_cctx_free(&ctx);
            return res;
        }

        // Update global hash from block checksum
        if (checksum_enabled && file_has_checksums) {
            const uint32_t block_hash = zxc_le32(ip + ZXC_BLOCK_HEADER_SIZE + bh.comp_size);
            global_hash = zxc_hash_combine_rotate(global_hash, block_hash);
        }

        ip += ZXC_BLOCK_HEADER_SIZE + bh.comp_size +
              (file_has_checksums ? ZXC_BLOCK_CHECKSUM_SIZE : 0);
        op += res;
    }

    zxc_cctx_free(&ctx);
    return (int64_t)(op - op_start);
}

/**
 * @brief Reads the decompressed size from a ZXC-compressed buffer.
 *
 * The size is stored in the file footer (last @ref ZXC_FILE_FOOTER_SIZE bytes).
 *
 * @param[in] src      Compressed data.
 * @param[in] src_size Size of @p src in bytes.
 * @return Original uncompressed size, or 0 on error.
 */
uint64_t zxc_get_decompressed_size(const void* src, const size_t src_size) {
    if (UNLIKELY(src_size < ZXC_FILE_HEADER_SIZE + ZXC_FILE_FOOTER_SIZE)) return 0;

    const uint8_t* const p = (const uint8_t*)src;
    if (UNLIKELY(zxc_le32(p) != ZXC_MAGIC_WORD)) return 0;

    const uint8_t* const footer = p + src_size - ZXC_FILE_FOOTER_SIZE;
    return zxc_le64(footer);
}

/**
 * @brief Reads the dictionary id from a compressed archive's file header.
 *
 * Public API; see @c zxc_buffer.h. Validates the magic, then returns the
 * header's @c dict_id field when the dictionary flag is set. Does not decompress.
 *
 * @param[in] src       Start of the compressed archive (>= @c ZXC_FILE_HEADER_SIZE).
 * @param[in] src_size  Size of @p src in bytes.
 * @return The dictionary id, or 0 if @p src is invalid or the archive uses no
 *         dictionary.
 */
// cppcheck-suppress unusedFunction
uint32_t zxc_get_dict_id(const void* src, const size_t src_size) {
    if (UNLIKELY(!src || src_size < ZXC_FILE_HEADER_SIZE)) return 0;

    const uint8_t* const p = (const uint8_t*)src;
    if (UNLIKELY(zxc_le32(p) != ZXC_MAGIC_WORD)) return 0;

    return (p[6] & ZXC_FILE_FLAG_HAS_DICTIONARY) ? zxc_le32(p + 7) : 0;
}

/*
 * ============================================================================
 * REUSABLE CONTEXT API (Opaque)
 * ============================================================================
 *
 * Provides heap-allocated, opaque contexts that integrators can reuse across
 * multiple compress / decompress calls, eliminating per-call malloc/free
 * overhead.
 */

/* --- Compression --------------------------------------------------------- */

/**
 * @brief Opaque reusable compression context (public handle @ref zxc_cctx).
 *
 * Wraps one internal @ref zxc_cctx_t plus the sticky options and bookkeeping
 * needed to reuse buffers across calls and re-init only when the block size
 * changes.
 */
struct zxc_cctx_s {
    zxc_cctx_t inner;       /* existing internal context */
    int initialized;        /* 1 if inner has live allocations */
    int owns_workspace;     /* 0 = library-allocated (free in zxc_free_cctx),
                               1 = caller-supplied static workspace (no-op free,
                               block_size pinned at init) */
    size_t last_block_size; /* block size used for last init */
    /* Sticky options (remembered from create or last compress call). */
    int stored_level;
    int stored_checksum;
    size_t stored_block_size;
};

/**
 * @brief Creates a reusable compression context.
 *
 * Public API; full contract in @c zxc_buffer.h. With non-NULL @p opts the
 * internal buffers are pre-allocated for the given level / block size /
 * checksum; with NULL @p opts allocation is deferred to the first
 * @ref zxc_compress_cctx call. The resolved settings become sticky defaults.
 *
 * @param[in] opts  Initial compression options, or NULL to defer allocation.
 * @return A context to release with @ref zxc_free_cctx, or NULL on allocation
 *         failure or invalid @p opts.
 */
zxc_cctx* zxc_create_cctx(const zxc_compress_opts_t* opts) {
    zxc_cctx* const cctx = (zxc_cctx*)ZXC_CALLOC(1, sizeof(zxc_cctx));
    if (UNLIKELY(!cctx)) return NULL;  // LCOV_EXCL_LINE

    /* Resolve and store sticky defaults. */
    cctx->stored_level = (opts && opts->level > 0) ? opts->level : ZXC_LEVEL_DEFAULT;
    cctx->stored_block_size =
        (opts && opts->block_size > 0) ? opts->block_size : ZXC_BLOCK_SIZE_DEFAULT;
    cctx->stored_checksum = opts ? opts->checksum_enabled : 0;

    if (opts) {
        // LCOV_EXCL_START
        if (UNLIKELY(!zxc_validate_block_size(cctx->stored_block_size) ||
                     zxc_cctx_init(&cctx->inner, cctx->stored_block_size, 1, cctx->stored_level,
                                   cctx->stored_checksum, 0) != ZXC_OK)) {
            ZXC_FREE(cctx);
            return NULL;
        }
        // LCOV_EXCL_STOP
        cctx->last_block_size = cctx->stored_block_size;
        cctx->initialized = 1;
    }

    return cctx;
}

/**
 * @brief Releases a reusable compression context.
 *
 * Public API; see @c zxc_buffer.h. Frees the inner buffers and the handle.
 * NULL-safe. For a static (caller-workspace) context this is a no-op, since the
 * caller owns the workspace.
 *
 * @param[in] cctx  Context from @ref zxc_create_cctx (may be NULL).
 */
void zxc_free_cctx(zxc_cctx* cctx) {
    if (UNLIKELY(!cctx)) return;
    /* Static cctx: handle + inner buffers live inside the caller's workspace,
     * which we do not own. Free is a no-op; the caller owns the workspace. */
    if (cctx->owns_workspace) return;
    if (cctx->initialized) zxc_cctx_free(&cctx->inner);
    ZXC_FREE(cctx);
}

/**
 * @brief Compresses a whole buffer into a framed archive, reusing @p cctx.
 *
 * Public API; full contract in @c zxc_buffer.h. Resolves per-call options over
 * the context's sticky defaults, re-initialises the inner buffers only when the
 * block size changes (level / checksum update in place), then writes the file
 * header, the compressed blocks, the EOF block and the footer.
 *
 * @param[in,out] cctx          Reusable compression context.
 * @param[in]     src           Source bytes.
 * @param[in]     src_size      Number of source bytes (must be > 0).
 * @param[out]    dst           Destination buffer for the archive.
 * @param[in]     dst_capacity  Capacity of @p dst in bytes.
 * @param[in]     opts          Per-call option overrides, or NULL for the
 *                              context defaults.
 * @return Archive size in bytes on success, or a negative @ref zxc_error_t.
 */
int64_t zxc_compress_cctx(zxc_cctx* cctx, const void* RESTRICT src, const size_t src_size,
                          void* RESTRICT dst, const size_t dst_capacity,
                          const zxc_compress_opts_t* opts) {
    if (UNLIKELY(!cctx)) return ZXC_ERROR_NULL_INPUT;
    if (UNLIKELY(!src || !dst || src_size == 0 || dst_capacity == 0)) return ZXC_ERROR_NULL_INPUT;

    const int checksum_enabled = opts ? opts->checksum_enabled : cctx->stored_checksum;
    const int level = (opts && opts->level > 0) ? opts->level : cctx->stored_level;
    const size_t block_size =
        (opts && opts->block_size > 0) ? opts->block_size : cctx->stored_block_size;

    if (UNLIKELY(!zxc_validate_block_size(block_size))) return ZXC_ERROR_BAD_BLOCK_SIZE;

    /* Static cctx: block_size is locked at workspace init.  Reject any opts
     * that would force a re-partition, since the workspace cannot grow.
     * level / checksum_enabled may still vary per call. */
    if (UNLIKELY(cctx->owns_workspace && block_size != cctx->last_block_size))
        return ZXC_ERROR_BAD_BLOCK_SIZE;

    cctx->stored_level = level;
    cctx->stored_block_size = block_size;
    cctx->stored_checksum = checksum_enabled;

    /* Re-init only when block_size changed (it drives buffer sizes). */
    if (UNLIKELY(!cctx->initialized || cctx->last_block_size != block_size)) {
        if (cctx->initialized) {
            // LCOV_EXCL_START
            zxc_cctx_free(&cctx->inner);
            cctx->initialized = 0;
            // LCOV_EXCL_STOP
        }
        // LCOV_EXCL_START
        if (UNLIKELY(zxc_cctx_init(&cctx->inner, block_size, 1, level, checksum_enabled, 0) !=
                     ZXC_OK))
            return ZXC_ERROR_MEMORY;
        // LCOV_EXCL_STOP
        cctx->last_block_size = block_size;
        cctx->initialized = 1;
    } else {
        /* Same block_size: update level + checksum without realloc. */
        cctx->inner.compression_level = level;
        cctx->inner.checksum_enabled = checksum_enabled;
    }

    zxc_cctx_t* const ctx = &cctx->inner;

    uint8_t* op = (uint8_t*)dst;
    const uint8_t* const op_start = op;
    const uint8_t* const op_end = op + dst_capacity;
    const uint8_t* const ip = (const uint8_t*)src;
    uint32_t global_hash = 0;

    const int h_val =
        zxc_write_file_header(op, (size_t)(op_end - op), block_size, checksum_enabled, 0);
    if (UNLIKELY(h_val < 0)) return h_val;  // LCOV_EXCL_LINE
    op += h_val;

    size_t pos = 0;
    while (pos < src_size) {
        const size_t chunk_len = (src_size - pos > block_size) ? block_size : (src_size - pos);
        const size_t rem_cap = (size_t)(op_end - op);

        const int res = zxc_compress_chunk_wrapper(ctx, ip + pos, chunk_len, op, rem_cap);
        if (UNLIKELY(res < 0)) return res;

        if (checksum_enabled) {
            if (LIKELY(res >= ZXC_GLOBAL_CHECKSUM_SIZE)) {
                const uint32_t block_hash = zxc_le32(op + res - ZXC_GLOBAL_CHECKSUM_SIZE);
                global_hash = zxc_hash_combine_rotate(global_hash, block_hash);
            }
        }

        op += res;
        pos += chunk_len;
    }

    /* EOF block */
    const size_t rem_cap = (size_t)(op_end - op);
    const zxc_block_header_t eof_bh = {
        .block_type = ZXC_BLOCK_EOF, .block_flags = 0, .reserved = 0, .comp_size = 0};
    const int eof_val = zxc_write_block_header(op, rem_cap, &eof_bh);
    if (UNLIKELY(eof_val < 0)) return eof_val;  // LCOV_EXCL_LINE
    op += eof_val;

    if (UNLIKELY(rem_cap < (size_t)eof_val + ZXC_FILE_FOOTER_SIZE))
        return ZXC_ERROR_DST_TOO_SMALL;  // LCOV_EXCL_LINE

    const int footer_val =
        zxc_write_file_footer(op, (size_t)(op_end - op), src_size, global_hash, checksum_enabled);
    if (UNLIKELY(footer_val < 0)) return footer_val;  // LCOV_EXCL_LINE
    op += footer_val;

    return (int64_t)(op - op_start);
}

/* --- Decompression ------------------------------------------------------- */

/**
 * @brief Opaque reusable decompression context (public handle @ref zxc_dctx).
 *
 * Reuses the internal @ref zxc_cctx_t type for decode, tracking the last block
 * and dict sizes so the inner buffers are re-carved only when they change.
 */
struct zxc_dctx_s {
    zxc_cctx_t inner;       /* reuses the same internal context type */
    size_t last_block_size; /* block size from last header parse */
    size_t last_dict_size;  /* dict_size the inner buffer was carved for (drives re-init) */
    int initialized;        /* 1 if inner has live allocations */
    int owns_workspace;     /* 0 = library-allocated (free in zxc_free_dctx),
                               1 = caller-supplied static workspace (no-op free,
                               block_size pinned at init) */
};

/**
 * @brief Creates a reusable decompression context.
 *
 * Public API; see @c zxc_buffer.h. The inner buffers are allocated lazily on
 * the first decode (sized from the archive header), so this only allocates the
 * handle itself.
 *
 * @return A context to release with @ref zxc_free_dctx, or NULL on allocation
 *         failure.
 */
zxc_dctx* zxc_create_dctx(void) {
    zxc_dctx* const dctx = (zxc_dctx*)ZXC_CALLOC(1, sizeof(zxc_dctx));
    return dctx;
}

/**
 * @brief Releases a reusable decompression context.
 *
 * Public API; see @c zxc_buffer.h. Frees the inner buffers and the handle.
 * NULL-safe; a no-op for a static (caller-workspace) context.
 *
 * @param[in] dctx  Context from @ref zxc_create_dctx (may be NULL).
 */
void zxc_free_dctx(zxc_dctx* dctx) {
    if (UNLIKELY(!dctx)) return;
    /* Static dctx: handle + inner buffers live inside the caller's workspace,
     * which we do not own. Free is a no-op; the caller owns the workspace. */
    if (dctx->owns_workspace) return;
    if (dctx->initialized) zxc_cctx_free(&dctx->inner);
    ZXC_FREE(dctx);
}

/**
 * @brief Decompresses a framed archive into @p dst, reusing @p dctx.
 *
 * Public API; full contract in @c zxc_buffer.h. Parses the file header,
 * re-initialises the inner buffers only when the block size changes (or a prior
 * dict call left a prefix), then decodes each block - straight into @p dst when
 * the tail padding fits, otherwise through a bounce buffer - and verifies the
 * footer size and optional checksum.
 *
 * @param[in,out] dctx          Reusable decompression context.
 * @param[in]     src           Compressed archive bytes.
 * @param[in]     src_size      Archive size (>= @c ZXC_FILE_HEADER_SIZE).
 * @param[out]    dst           Destination for the decompressed output.
 * @param[in]     dst_capacity  Capacity of @p dst in bytes.
 * @param[in]     opts          Per-call options (e.g. checksum), or NULL.
 * @return Decompressed size in bytes on success, or a negative @ref zxc_error_t.
 */
int64_t zxc_decompress_dctx(zxc_dctx* dctx, const void* RESTRICT src, const size_t src_size,
                            void* RESTRICT dst, const size_t dst_capacity,
                            const zxc_decompress_opts_t* opts) {
    if (UNLIKELY(!dctx || !src || !dst || src_size < ZXC_FILE_HEADER_SIZE))
        return ZXC_ERROR_NULL_INPUT;

    const int checksum_enabled = opts ? opts->checksum_enabled : 0;

    const uint8_t* ip = (const uint8_t*)src;
    const uint8_t* const ip_end = ip + src_size;
    uint8_t* op = (uint8_t*)dst;
    const uint8_t* const op_start = op;
    const uint8_t* const op_end = op + dst_capacity;
    size_t runtime_chunk_size = 0;
    int file_has_checksums = 0;
    uint32_t global_hash = 0;

    if (UNLIKELY(zxc_read_file_header(ip, src_size, &runtime_chunk_size, &file_has_checksums,
                                      NULL) != ZXC_OK))
        return ZXC_ERROR_BAD_HEADER;

    /* Static dctx: block_size is locked at workspace init; reject any
     * archive whose declared block_size would require a re-partition. */
    if (UNLIKELY(dctx->owns_workspace && runtime_chunk_size != dctx->last_block_size))
        return ZXC_ERROR_BAD_BLOCK_SIZE;

    /* Re-init when block size changed, or when a prior dict-using call (block
     * API) left the inner context carrying a dict prefix. */
    if (UNLIKELY(!dctx->initialized || dctx->last_block_size != runtime_chunk_size ||
                 dctx->last_dict_size != 0)) {
        if (dctx->initialized) {
            // LCOV_EXCL_START
            zxc_cctx_free(&dctx->inner);
            dctx->initialized = 0;
            // LCOV_EXCL_STOP
        }
        // LCOV_EXCL_START
        if (UNLIKELY(zxc_cctx_init(&dctx->inner, runtime_chunk_size, 0, 0,
                                   file_has_checksums && checksum_enabled, 0) != ZXC_OK))
            return ZXC_ERROR_MEMORY;
        // LCOV_EXCL_STOP
        dctx->last_block_size = runtime_chunk_size;
        dctx->last_dict_size = 0;
        dctx->initialized = 1;
    } else {
        dctx->inner.checksum_enabled = file_has_checksums && checksum_enabled;
    }

    zxc_cctx_t* const ctx = &dctx->inner;
    ip += ZXC_FILE_HEADER_SIZE;

    /* work_buf was pre-sized to runtime_chunk_size + ZXC_DECOMPRESS_TAIL_PAD
     * inside the matching zxc_cctx_init call above; the re-init guard ensures
     * it stays in sync when chunk_size changes between calls. */
    const size_t work_sz = runtime_chunk_size + ZXC_DECOMPRESS_TAIL_PAD;

    while (ip < ip_end) {
        const size_t rem_src = (size_t)(ip_end - ip);
        zxc_block_header_t bh;
        if (UNLIKELY(zxc_read_block_header(ip, rem_src, &bh) != ZXC_OK))
            return ZXC_ERROR_BAD_HEADER;

        if (UNLIKELY(bh.block_type == ZXC_BLOCK_EOF)) {
            if (UNLIKELY(bh.comp_size != 0)) return ZXC_ERROR_BAD_HEADER;
            if (UNLIKELY(rem_src < ZXC_BLOCK_HEADER_SIZE + ZXC_FILE_FOOTER_SIZE))
                return ZXC_ERROR_SRC_TOO_SMALL;

            const uint8_t* const footer = ip + ZXC_BLOCK_HEADER_SIZE;
            const uint64_t stored_size = zxc_le64(footer);
            if (UNLIKELY(stored_size != (uint64_t)(op - op_start))) return ZXC_ERROR_CORRUPT_DATA;

            if (checksum_enabled && file_has_checksums) {
                const uint32_t stored_hash = zxc_le32(footer + sizeof(uint64_t));
                if (UNLIKELY(stored_hash != global_hash)) return ZXC_ERROR_BAD_CHECKSUM;
            }
            break;
        }

        const size_t rem_cap = (size_t)(op_end - op);
        int res;
        if (LIKELY(rem_cap >= work_sz)) {
            // Fast path: decode directly into dst (enough padding for wild copies).
            res = zxc_decompress_chunk_wrapper(ctx, ip, rem_src, op, rem_cap);
        } else {
            // Safe path: decode into bounce buffer, then copy exact result.
            res = zxc_decompress_chunk_wrapper(ctx, ip, rem_src, ctx->work_buf, ctx->work_buf_cap);
            if (LIKELY(res > 0)) {
                if (UNLIKELY((size_t)res > rem_cap))
                    return ZXC_ERROR_DST_TOO_SMALL;  // LCOV_EXCL_LINE
                ZXC_MEMCPY(op, ctx->work_buf, (size_t)res);
            }
        }
        if (UNLIKELY(res < 0)) return res;

        if (checksum_enabled && file_has_checksums) {
            const uint32_t block_hash = zxc_le32(ip + ZXC_BLOCK_HEADER_SIZE + bh.comp_size);
            global_hash = zxc_hash_combine_rotate(global_hash, block_hash);
        }

        ip += ZXC_BLOCK_HEADER_SIZE + bh.comp_size +
              (file_has_checksums ? ZXC_BLOCK_CHECKSUM_SIZE : 0);
        op += res;
    }

    return (int64_t)(op - op_start);
}

/* ========================================================================= */
/*  Block-Level API (no file framing)                                        */
/* ========================================================================= */

/**
 * @brief Compresses a single block (no file framing), reusing @p cctx.
 *
 * Public API; full contract in @c zxc_buffer.h. Produces one format-conformant
 * block with no header / EOF / footer, so @p src_size must not exceed
 * @c ZXC_BLOCK_SIZE_MAX (use the frame or streaming APIs for larger inputs).
 * With a dictionary in @p opts, [dict | block] is assembled in the cctx-owned
 * bounce buffer before encoding. Inner buffers are re-initialised only when the
 * effective block size changes.
 *
 * @param[in,out] cctx          Reusable compression context.
 * @param[in]     src           Source block bytes.
 * @param[in]     src_size      Source length (0 < @p src_size <= @c ZXC_BLOCK_SIZE_MAX).
 * @param[out]    dst           Destination buffer for the block payload.
 * @param[in]     dst_capacity  Capacity of @p dst in bytes.
 * @param[in]     opts          Per-call options (level, dict, ...), or NULL.
 * @return Block payload size in bytes on success, or a negative @ref zxc_error_t.
 */
int64_t zxc_compress_block(zxc_cctx* cctx, const void* RESTRICT src, const size_t src_size,
                           void* RESTRICT dst, const size_t dst_capacity,
                           const zxc_compress_opts_t* opts) {
    if (UNLIKELY(!cctx || !src || !dst || src_size == 0 || dst_capacity == 0))
        return ZXC_ERROR_NULL_INPUT;

    /* Block API processes a single format-conformant block: src_size must not
     * exceed ZXC_BLOCK_SIZE_MAX. Callers with larger inputs should use the
     * frame or streaming APIs which chunk transparently. */
    if (UNLIKELY(src_size > ZXC_BLOCK_SIZE_MAX)) return ZXC_ERROR_BAD_BLOCK_SIZE;

    const int checksum_enabled = opts ? opts->checksum_enabled : cctx->stored_checksum;
    const int level = (opts && opts->level > 0) ? opts->level : cctx->stored_level;
    /* For block API, block_size == src_size (the caller compresses one block at a time). */
    const size_t block_size =
        (opts && opts->block_size > 0) ? opts->block_size : cctx->stored_block_size;
    const size_t min_bs = zxc_block_size_ceil(src_size);

    /* Always ensure internal buffers can hold src_size.
     * When a dictionary is active, offset_bits must accommodate dict + block. */
    const uint8_t* b_dict = opts ? (const uint8_t*)opts->dict : NULL;
    const size_t b_dict_size = (opts && opts->dict) ? opts->dict_size : 0;
    const size_t base_block_size = (block_size > min_bs) ? block_size : min_bs;
    const size_t effective_block_size =
        b_dict_size > 0 ? zxc_block_size_ceil(b_dict_size + base_block_size) : base_block_size;

    cctx->stored_level = level;
    cctx->stored_block_size = effective_block_size;
    cctx->stored_checksum = checksum_enabled;

    /* Re-init only when block_size changed. */
    if (UNLIKELY(!cctx->initialized || cctx->last_block_size != effective_block_size)) {
        if (cctx->initialized) {
            // LCOV_EXCL_START
            zxc_cctx_free(&cctx->inner);
            cctx->initialized = 0;
            // LCOV_EXCL_STOP
        }
        // LCOV_EXCL_START
        if (UNLIKELY(zxc_cctx_init(&cctx->inner, effective_block_size, 1, level, checksum_enabled,
                                   b_dict_size) != ZXC_OK))
            return ZXC_ERROR_MEMORY;
        // LCOV_EXCL_STOP
        cctx->last_block_size = effective_block_size;
        cctx->initialized = 1;
    } else {
        cctx->inner.compression_level = level;
        cctx->inner.checksum_enabled = checksum_enabled;
    }

    cctx->inner.dict_size = b_dict_size;

    int res;
    if (b_dict && b_dict_size > 0) {
        /* [dict | block] assembled in the cctx-owned dict_buffer */
        uint8_t* const combined = cctx->inner.dict_buffer;
        ZXC_MEMCPY(combined, b_dict, b_dict_size);
        ZXC_MEMCPY(combined + b_dict_size, src, src_size);
        res = zxc_compress_chunk_wrapper(&cctx->inner, combined, b_dict_size + src_size,
                                         (uint8_t*)dst, dst_capacity);
    } else {
        res = zxc_compress_chunk_wrapper(&cctx->inner, (const uint8_t*)src, src_size, (uint8_t*)dst,
                                         dst_capacity);
    }
    if (UNLIKELY(res < 0)) return res;
    return (int64_t)res;
}

/**
 * @brief Decompresses a single block (no file framing), reusing @p dctx.
 *
 * Public API; full contract in @c zxc_buffer.h. Decodes one format-conformant
 * block; the decoded payload cannot exceed @c ZXC_BLOCK_SIZE_MAX, so
 * @p dst_capacity is bounded by @c ZXC_BLOCK_SIZE_MAX + @c ZXC_DECOMPRESS_TAIL_PAD.
 * With a dictionary in @p opts the decode runs through the [dict | decode]
 * bounce buffer; otherwise it goes straight into @p dst when the tail padding
 * fits, or via @c work_buf when it doesn't.
 *
 * @param[in,out] dctx          Reusable decompression context.
 * @param[in]     src           Compressed block bytes.
 * @param[in]     src_size      Source length (>= @c ZXC_BLOCK_HEADER_SIZE).
 * @param[out]    dst           Destination for the decoded payload.
 * @param[in]     dst_capacity  Capacity of @p dst in bytes.
 * @param[in]     opts          Per-call options (dict, checksum), or NULL.
 * @return Decoded payload size in bytes on success, or a negative @ref zxc_error_t.
 */
int64_t zxc_decompress_block(zxc_dctx* dctx, const void* RESTRICT src, const size_t src_size,
                             void* RESTRICT dst, const size_t dst_capacity,
                             const zxc_decompress_opts_t* opts) {
    if (UNLIKELY(!dctx || !src || !dst || src_size < ZXC_BLOCK_HEADER_SIZE || dst_capacity == 0))
        return ZXC_ERROR_NULL_INPUT;

    /* Block API decompresses a single format-conformant block. Decoded payload
     * cannot exceed ZXC_BLOCK_SIZE_MAX; dst_capacity is bounded accordingly to
     * include the tail-pad needed for safe wild copies. Callers expecting
     * larger outputs should use the frame or streaming APIs. */
    if (UNLIKELY(dst_capacity > ZXC_BLOCK_SIZE_MAX + ZXC_DECOMPRESS_TAIL_PAD))
        return ZXC_ERROR_BAD_BLOCK_SIZE;

    const int checksum_enabled = opts ? opts->checksum_enabled : 0;

    const uint8_t* dict = opts ? (const uint8_t*)opts->dict : NULL;
    const size_t dict_size = (opts && opts->dict) ? opts->dict_size : 0;

    /* Derive the block_size from dst_capacity (callers know the original size) */
    const size_t block_size = zxc_block_size_ceil(dst_capacity);
    if (UNLIKELY(!dctx->initialized || dctx->last_block_size != block_size ||
                 dctx->last_dict_size != dict_size)) {
        if (dctx->initialized) {
            zxc_cctx_free(&dctx->inner);
            dctx->initialized = 0;
        }
        // LCOV_EXCL_START
        if (UNLIKELY(zxc_cctx_init(&dctx->inner, block_size, 0, 0, checksum_enabled, dict_size) !=
                     ZXC_OK))
            return ZXC_ERROR_MEMORY;
        // LCOV_EXCL_STOP
        dctx->last_block_size = block_size;
        dctx->last_dict_size = dict_size;
        dctx->initialized = 1;
    } else {
        dctx->inner.checksum_enabled = checksum_enabled;
    }

    zxc_cctx_t* const ctx = &dctx->inner;
    ctx->dict_size = dict_size;

    /* work_buf was pre-sized to block_size + ZXC_DECOMPRESS_TAIL_PAD inside
     * the matching zxc_cctx_init call above. */
    const size_t work_sz = block_size + ZXC_DECOMPRESS_TAIL_PAD;

    int res;
    if (dict && dict_size > 0) {
        /* [dict | decode] assembled in the cctx-owned dict_buffer */
        uint8_t* const dec_buf = ctx->dict_buffer;
        ZXC_MEMCPY(dec_buf, dict, dict_size);
        res = zxc_decompress_chunk_wrapper(ctx, (const uint8_t*)src, src_size, dec_buf + dict_size,
                                           work_sz);
        if (LIKELY(res > 0)) {
            if (UNLIKELY((size_t)res > dst_capacity)) return ZXC_ERROR_DST_TOO_SMALL;
            ZXC_MEMCPY(dst, dec_buf + dict_size, (size_t)res);
        }
    } else if (LIKELY(dst_capacity >= work_sz)) {
        res = zxc_decompress_chunk_wrapper(ctx, (const uint8_t*)src, src_size, (uint8_t*)dst,
                                           dst_capacity);
    } else {
        /* Bounce through work_buf when output can't absorb wild copies. */
        res = zxc_decompress_chunk_wrapper(ctx, (const uint8_t*)src, src_size, ctx->work_buf,
                                           ctx->work_buf_cap);
        if (LIKELY(res > 0)) {
            if (UNLIKELY((size_t)res > dst_capacity)) return ZXC_ERROR_DST_TOO_SMALL;
            ZXC_MEMCPY(dst, ctx->work_buf, (size_t)res);
        }
    }
    if (UNLIKELY(res < 0)) return res;
    return (int64_t)res;
}

/**
 * @brief Safe-variant block decompressor: accepts dst_capacity == uncompressed_size.
 *
 * Dict inputs and RAW blocks route to @ref zxc_decompress_block; plain GLO/GHI
 * use the strict safe decoder (no bounce buffer, no +ZXC_DECOMPRESS_TAIL_PAD).
 *
 * Public API; full contract in @c zxc_buffer.h.
 *
 * @param[in,out] dctx          Reusable decompression context.
 * @param[in]     src           Compressed block bytes.
 * @param[in]     src_size      Source length (>= @c ZXC_BLOCK_HEADER_SIZE).
 * @param[out]    dst           Destination for the decoded payload.
 * @param[in]     dst_capacity  Exact uncompressed size (<= @c ZXC_BLOCK_SIZE_MAX).
 * @param[in]     opts          Per-call options (dict, checksum), or NULL.
 * @return Decoded payload size in bytes on success, or a negative @ref zxc_error_t.
 */
int64_t zxc_decompress_block_safe(zxc_dctx* dctx, const void* RESTRICT src, const size_t src_size,
                                  void* RESTRICT dst, const size_t dst_capacity,
                                  const zxc_decompress_opts_t* opts) {
    if (UNLIKELY(!dctx || !src || !dst || src_size < ZXC_BLOCK_HEADER_SIZE || dst_capacity == 0))
        return ZXC_ERROR_NULL_INPUT;

    /* Strict-tail variant: dst_capacity matches the exact uncompressed size */
    if (UNLIKELY(dst_capacity > ZXC_BLOCK_SIZE_MAX)) return ZXC_ERROR_BAD_BLOCK_SIZE;

    /* A dict needs the [dict|payload] bounce; route to the bounce-capable path. */
    if (opts && opts->dict && opts->dict_size > 0) {
        return zxc_decompress_block(dctx, src, src_size, dst, dst_capacity, opts);
    }

    const uint8_t type = ((const uint8_t*)src)[0];
    /* RAW never wild-writes past dst_capacity: route to the existing fast API. */
    if (type == ZXC_BLOCK_RAW) {
        return zxc_decompress_block(dctx, src, src_size, dst, dst_capacity, opts);
    }

    /* GLO/GHI: use the strict-tail decoder (no bounce buffer required). */
    const int checksum_enabled = opts ? opts->checksum_enabled : 0;
    const size_t block_size = zxc_block_size_ceil(dst_capacity);
    if (UNLIKELY(!dctx->initialized || dctx->last_block_size != block_size ||
                 dctx->last_dict_size != 0)) {
        if (dctx->initialized) {
            zxc_cctx_free(&dctx->inner);
            dctx->initialized = 0;
        }
        // LCOV_EXCL_START
        if (UNLIKELY(zxc_cctx_init(&dctx->inner, block_size, 0, 0, checksum_enabled, 0) != ZXC_OK))
            return ZXC_ERROR_MEMORY;
        // LCOV_EXCL_STOP
        dctx->last_block_size = block_size;
        dctx->last_dict_size = 0;
        dctx->initialized = 1;
    } else {
        dctx->inner.checksum_enabled = checksum_enabled;
    }
    dctx->inner.dict_size = 0;

    const int res = zxc_decompress_chunk_wrapper_safe_public(&dctx->inner, (const uint8_t*)src,
                                                             src_size, (uint8_t*)dst, dst_capacity);
    if (UNLIKELY(res < 0)) return res;
    return (int64_t)res;
}

/*
 * ============================================================================
 * STATIC CONTEXT API (caller-allocated workspace)
 * ============================================================================
 * Places the public handle struct at the start of the workspace, then carves
 * the persistent buffer (via zxc_cctx_init_in_workspace) in the remaining
 * cache-line-aligned tail.  The caller owns the whole workspace; free
 * functions become no-ops via the owns_workspace flag.
 */

/* Size occupied by the opaque handle at the start of the workspace, rounded
 * up to a cache-line boundary so the persistent buffer (which expects 64 B
 * alignment for the hot zones) starts aligned. */
#define ZXC_STATIC_CCTX_HDR_SIZE ZXC_ALIGN_CL(sizeof(struct zxc_cctx_s))
#define ZXC_STATIC_DCTX_HDR_SIZE ZXC_ALIGN_CL(sizeof(struct zxc_dctx_s))

/**
 * @brief Workspace size needed for a static compression context.
 *
 * Public API; see @c zxc_buffer.h. Sum of the cache-line-aligned handle header
 * and the persistent buffer that @ref zxc_init_static_cctx carves for the given
 * @p block_size / @p level. Performs no allocation.
 *
 * @param[in] block_size  Block size the context will be pinned to.
 * @param[in] level       Compression level.
 * @return Required workspace size in bytes, or 0 if the parameters are invalid.
 */
size_t zxc_static_cctx_workspace_size(const size_t block_size, const int level) {
    if (UNLIKELY(!zxc_validate_block_size(block_size))) return 0;
    if (UNLIKELY(level < ZXC_LEVEL_FASTEST || level > ZXC_LEVEL_DENSITY)) return 0;
    const size_t inner_sz = zxc_cctx_compute_workspace_size(block_size, 1, level, 0);
    if (UNLIKELY(inner_sz == 0)) return 0;
    return ZXC_STATIC_CCTX_HDR_SIZE + inner_sz;
}

/**
 * @brief Initialises a compression context inside a caller-supplied workspace.
 *
 * Public API; full contract in @c zxc_buffer.h. Places the opaque handle at the
 * start of @p workspace and carves the persistent buffer in the aligned tail -
 * no heap allocation. The block size is pinned for the context's lifetime, and
 * @ref zxc_free_cctx becomes a no-op (the caller owns @p workspace).
 *
 * @param[in] workspace       Caller buffer (>= @ref zxc_static_cctx_workspace_size).
 * @param[in] workspace_size  Capacity of @p workspace in bytes.
 * @param[in] opts            Compression options (non-NULL: level, block_size,
 *                            checksum).
 * @return A ready context owned by @p workspace, or NULL on invalid input or an
 *         undersized workspace.
 */
zxc_cctx* zxc_init_static_cctx(void* RESTRICT workspace, const size_t workspace_size,
                               const zxc_compress_opts_t* RESTRICT opts) {
    if (UNLIKELY(!workspace || !opts)) return NULL;

    const int level = (opts->level > 0) ? opts->level : ZXC_LEVEL_DEFAULT;
    const size_t block_size = (opts->block_size > 0) ? opts->block_size : ZXC_BLOCK_SIZE_DEFAULT;
    const int checksum_enabled = opts->checksum_enabled;

    if (UNLIKELY(!zxc_validate_block_size(block_size))) return NULL;
    if (UNLIKELY(level < ZXC_LEVEL_FASTEST || level > ZXC_LEVEL_DENSITY)) return NULL;

    const size_t inner_sz = zxc_cctx_compute_workspace_size(block_size, 1, level, 0);
    if (UNLIKELY(inner_sz == 0)) return NULL;
    if (UNLIKELY(workspace_size < ZXC_STATIC_CCTX_HDR_SIZE + inner_sz)) return NULL;

    zxc_cctx* const cctx = (zxc_cctx*)workspace;
    ZXC_MEMSET(cctx, 0, sizeof(*cctx));

    uint8_t* const inner_ws = (uint8_t*)workspace + ZXC_STATIC_CCTX_HDR_SIZE;
    if (UNLIKELY(zxc_cctx_init_in_workspace(&cctx->inner, inner_ws, inner_sz, block_size, 1, level,
                                            checksum_enabled, 0) != ZXC_OK))
        return NULL;

    cctx->owns_workspace = 1;
    cctx->initialized = 1;
    cctx->last_block_size = block_size;
    cctx->stored_level = level;
    cctx->stored_block_size = block_size;
    cctx->stored_checksum = checksum_enabled;
    return cctx;
}

/**
 * @brief Workspace size needed for a static decompression context.
 *
 * Public API; see @c zxc_buffer.h. Sum of the cache-line-aligned handle header
 * and the persistent buffer that @ref zxc_init_static_dctx carves for the given
 * @p block_size. Performs no allocation.
 *
 * @param[in] block_size  Block size the context will be pinned to.
 * @return Required workspace size in bytes, or 0 if @p block_size is invalid.
 */
size_t zxc_static_dctx_workspace_size(const size_t block_size) {
    if (UNLIKELY(!zxc_validate_block_size(block_size))) return 0;
    const size_t inner_sz = zxc_cctx_compute_workspace_size(block_size, 0, 0, 0);
    if (UNLIKELY(inner_sz == 0)) return 0;
    return ZXC_STATIC_DCTX_HDR_SIZE + inner_sz;
}

/**
 * @brief Initialises a decompression context inside a caller-supplied workspace.
 *
 * Public API; full contract in @c zxc_buffer.h. Places the opaque handle at the
 * start of @p workspace and carves the persistent buffer in the aligned tail -
 * no heap allocation. The block size is pinned, so decoded archives must match
 * it; @ref zxc_free_dctx becomes a no-op (the caller owns @p workspace).
 *
 * @param[in] workspace       Caller buffer (>= @ref zxc_static_dctx_workspace_size).
 * @param[in] workspace_size  Capacity of @p workspace in bytes.
 * @param[in] block_size      Block size to pin the context to.
 * @return A ready context owned by @p workspace, or NULL on invalid input or an
 *         undersized workspace.
 */
zxc_dctx* zxc_init_static_dctx(void* RESTRICT workspace, const size_t workspace_size,
                               const size_t block_size) {
    if (UNLIKELY(!workspace)) return NULL;
    if (UNLIKELY(!zxc_validate_block_size(block_size))) return NULL;

    const size_t inner_sz = zxc_cctx_compute_workspace_size(block_size, 0, 0, 0);
    if (UNLIKELY(inner_sz == 0)) return NULL;
    if (UNLIKELY(workspace_size < ZXC_STATIC_DCTX_HDR_SIZE + inner_sz)) return NULL;

    zxc_dctx* const dctx = (zxc_dctx*)workspace;
    ZXC_MEMSET(dctx, 0, sizeof(*dctx));

    uint8_t* const inner_ws = (uint8_t*)workspace + ZXC_STATIC_DCTX_HDR_SIZE;
    /* mode == 0 init: checksum_enabled is updated per-call from the file
     * header flags, so it does not need to be locked at workspace init. */
    if (UNLIKELY(zxc_cctx_init_in_workspace(&dctx->inner, inner_ws, inner_sz, block_size, 0, 0, 0,
                                            0) != ZXC_OK))
        return NULL;

    dctx->owns_workspace = 1;
    dctx->initialized = 1;
    dctx->last_block_size = block_size;
    return dctx;
}
