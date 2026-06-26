/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file zxc_internal.h
 * @brief Internal definitions, constants, SIMD helpers, and utility functions.
 *
 * This header is **not** part of the public API.  It is shared across the
 * library's translation units and contains:
 * - Platform detection and SIMD intrinsic includes.
 * - Compiler-abstraction macros (LIKELY, PREFETCH, MEMCPY, ALIGN, ...).
 * - Endianness detection and byte-swap helpers.
 * - File-format constants (magic word, header sizes, block sizes, ...).
 * - Inline helpers for hashing, endian-safe loads/stores, bit manipulation,
 *   aligned allocation, and bitstream reading.
 * - Internal function prototypes for chunk-level compression/decompression.
 *
 * @warning Do not include this header from user code; use the public headers
 *          zxc_buffer.h or zxc_stream.h instead.
 */

#ifndef ZXC_INTERNAL_H
#define ZXC_INTERNAL_H

#include "zxc_deps.h" /* libc deps: <limits.h>, <stdint.h>, <stdlib.h>, <string.h>,
                        and the ZXC_MALLOC / ZXC_ALIGNED_MALLOC macros.
                        Vendor this file to retarget non-libc environments. */

#include "../../include/zxc_buffer.h"
#include "../../include/zxc_constants.h"
#include "../../include/zxc_error.h"
#include "../../include/zxc_seekable.h"
#include "rapidhash.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup internal Internal Helpers
 * @brief Platform abstractions, constants, and utility functions (private).
 * @{
 */

/**
 * @name Atomic Qualifier
 * @brief Provides a portable atomic / volatile qualifier.
 *
 * If C11 atomics are available, @c ZXC_ATOMIC expands to @c _Atomic;
 * otherwise it falls back to @c volatile.
 * @{
 */
#if !defined(__cplusplus) && defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && \
    !defined(__STDC_NO_ATOMICS__)
#include <stdatomic.h>
#define ZXC_ATOMIC _Atomic
#define ZXC_USE_C11_ATOMICS 1
#else
#define ZXC_ATOMIC volatile
#define ZXC_USE_C11_ATOMICS 0
#endif
/** @} */ /* end of Atomic Qualifier */

/**
 * @name SIMD Intrinsics & Compiler Macros
 * @brief Auto-detected SIMD feature macros for x86 (SSE/AVX) and ARM (NEON).
 *
 * Depending on the target architecture and compiler flags the following macros
 * may be defined:
 * - @c ZXC_USE_AVX512 - AVX-512F + AVX-512BW available.
 * - @c ZXC_USE_AVX2   - AVX2 available.
 * - @c ZXC_USE_SSE2   - SSE2 (x86-64 baseline) available.
 * - @c ZXC_USE_NEON64 - AArch64 NEON available.
 * - @c ZXC_USE_NEON32 - ARMv7 NEON available.
 *
 * Note: @c -mavx2 / @c -mavx512f imply @c __SSE2__, so @c ZXC_USE_SSE2 is
 * also defined in the AVX variants. The hand-written SIMD code paths therefore
 * order their preprocessor branches AVX512 -> AVX2 -> SSE2 so the widest
 * available path wins; the SSE2 branch is the active one only in the dedicated
 * @c _sse2 variant (no AVX2/AVX512 flags). SSE2 is the x86-64 baseline, so this
 * tier covers every 64-bit x86 CPU (and i686 with @c -msse2). The handful of
 * operations that would otherwise require SSE4.1 (@c _mm_max_epu32,
 * @c _mm_blendv_epi8, @c _mm_packus_epi32) or SSSE3 (@c _mm_shuffle_epi8) are
 * emulated with pure SSE2 instruction sequences or fall back to scalar code.
 *
 * Define @c ZXC_DISABLE_SIMD to gate all hand-written SIMD paths (intrinsics,
 * inline assembly).  Compiler auto-vectorisation is unaffected.
 * @{
 */
#ifndef ZXC_DISABLE_SIMD
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#include <nmmintrin.h>
#if defined(__AVX512F__) && defined(__AVX512BW__)
#ifndef ZXC_USE_AVX512
#define ZXC_USE_AVX512
#endif
#endif
#if defined(__AVX2__)
#ifndef ZXC_USE_AVX2
#define ZXC_USE_AVX2
#endif
#endif
#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#ifndef ZXC_USE_SSE2
#define ZXC_USE_SSE2
#endif
#endif
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64) || \
       defined(ZXC_USE_NEON32) || defined(ZXC_USE_NEON64))
#if !defined(_MSC_VER)
#include <arm_acle.h>
#endif
#include <arm_neon.h>
#if defined(__aarch64__) || defined(_M_ARM64)
#ifndef ZXC_USE_NEON64
#define ZXC_USE_NEON64
#endif
#else
#ifndef ZXC_USE_NEON32
#define ZXC_USE_NEON32
#endif
#endif
#endif
#endif    /* ZXC_DISABLE_SIMD */
/** @} */ /* end of SIMD Intrinsics */

/**
 * @name Compiler Abstractions
 * @brief Portable wrappers for branch hints, prefetch, memory ops, alignment,
 *        and forced inlining.
 * @{
 */

#if defined(__GNUC__) || defined(__clang__)
/** @def LIKELY
 * @brief Branch prediction hint: expression is likely true.
 * @param x Expression to evaluate.
 */
#define LIKELY(x) (__builtin_expect(!!(x), 1))

/** @def UNLIKELY
 * @brief Branch prediction hint: expression is unlikely to be true.
 * @param x Expression to evaluate.
 */
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))

/** @def RESTRICT
 * @brief Pointer aliasing hint (maps to __restrict__).
 */
#define RESTRICT __restrict__

/** @def ZXC_PREFETCH_READ
 * @brief Prefetch data for reading.
 * @param ptr Pointer to data to prefetch.
 */
#define ZXC_PREFETCH_READ(ptr) __builtin_prefetch((const void*)(ptr), 0, 3)

/** @def ZXC_PREFETCH_WRITE
 * @brief Prefetch data for writing.
 * @param ptr Pointer to data to prefetch.
 */
#define ZXC_PREFETCH_WRITE(ptr) __builtin_prefetch((const void*)(ptr), 1, 3)

/** @def ZXC_MEMCPY
 * @brief Optimized memory copy using compiler built-in.
 */
#define ZXC_MEMCPY(dst, src, n) __builtin_memcpy(dst, src, n)

/** @def ZXC_MEMSET
 * @brief Optimized memory set using compiler built-in.
 */
#define ZXC_MEMSET(dst, val, n) __builtin_memset(dst, val, n)

/** @def ZXC_ALIGN
 * @brief Specifies memory alignment for a variable or structure.
 * @param x Alignment boundary in bytes (must be a power of 2).
 */
#define ZXC_ALIGN(x) __attribute__((aligned(x)))

/** @def ZXC_ALWAYS_INLINE
 * @brief Forces a function to be inlined at all optimization levels.
 */
#define ZXC_ALWAYS_INLINE inline __attribute__((always_inline))

/** @def ZXC_NOINLINE
 * @brief Prevents a function from being inlined into its callers.
 */
#define ZXC_NOINLINE __attribute__((noinline))

#elif defined(_MSC_VER)
#include <intrin.h>
#if defined(_M_IX86) || defined(_M_X64) || defined(_M_AMD64)
#include <xmmintrin.h>
#define ZXC_PREFETCH_READ(ptr) _mm_prefetch((const char*)(ptr), _MM_HINT_T0)
#define ZXC_PREFETCH_WRITE(ptr) _mm_prefetch((const char*)(ptr), _MM_HINT_T0)
#else
#define ZXC_PREFETCH_READ(ptr) __prefetch((const void*)(ptr))
#define ZXC_PREFETCH_WRITE(ptr) __prefetch((const void*)(ptr))
#endif
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#define RESTRICT __restrict
#pragma intrinsic(memcpy, memset)
#define ZXC_MEMCPY(dst, src, n) memcpy(dst, src, n)
#define ZXC_MEMSET(dst, val, n) memset(dst, val, n)

/** @def ZXC_ALIGN
 * @brief Specifies memory alignment for a variable or structure (MSVC).
 * @param x Alignment boundary in bytes (must be a power of 2).
 */
#define ZXC_ALIGN(x) __declspec(align(x))

/** @def ZXC_ALWAYS_INLINE
 * @brief Forces a function to be inlined at all optimization levels (MSVC).
 */
#define ZXC_ALWAYS_INLINE __forceinline

/** @def ZXC_NOINLINE
 * @brief Prevents a function from being inlined into its callers (MSVC).
 */
#define ZXC_NOINLINE __declspec(noinline)
#pragma intrinsic(_BitScanReverse)
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#define RESTRICT
#define ZXC_PREFETCH_READ(ptr)
#define ZXC_PREFETCH_WRITE(ptr)
#define ZXC_MEMCPY(dst, src, n) memcpy(dst, src, n)
#define ZXC_MEMSET(dst, val, n) memset(dst, val, n)

/** @def ZXC_ALWAYS_INLINE
 * @brief Forces a function to be inlined (fallback for non-GCC/Clang/MSVC compilers).
 */
#define ZXC_ALWAYS_INLINE inline

/** @def ZXC_NOINLINE
 * @brief Prevents inlining (best-effort no-op fallback for unknown compilers).
 */
#define ZXC_NOINLINE

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#include <stdalign.h>
/** @def ZXC_ALIGN
 * @brief Specifies memory alignment using C11 _Alignas.
 * @param x Alignment boundary in bytes (must be a power of 2).
 */
#define ZXC_ALIGN(x) _Alignas(x)
#else
/** @def ZXC_ALIGN
 * @brief No-op alignment macro for compilers without alignment support.
 * @param x Ignored (alignment not supported).
 */
#define ZXC_ALIGN(x)
#endif
#endif
/** @} */ /* end of Compiler Abstractions */

/* Heap allocator and cache-line-aligned allocator macros are now defined
 * in @c zxc_deps.h (included at the top of this header), so non-libc
 * targets can override them by vendoring that single file. */

/**
 * @name Endianness Detection
 * @brief Compile-time detection of host byte order.
 *
 * Defines exactly one of @c ZXC_LITTLE_ENDIAN or @c ZXC_BIG_ENDIAN.
 * @{
 */
#ifndef ZXC_LITTLE_ENDIAN
#if defined(_WIN32) || defined(__LITTLE_ENDIAN__) || \
    (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#define ZXC_LITTLE_ENDIAN
#elif defined(__BIG_ENDIAN__) || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#define ZXC_BIG_ENDIAN
#else
#warning "Endianness not detected, defaulting to little-endian"
#define ZXC_LITTLE_ENDIAN
#endif
#endif
/** @} */ /* end of Endianness Detection */

/**
 * @name Byte-Swap Helpers
 * @brief 16/32/64-bit byte-swap macros (only defined under @c ZXC_BIG_ENDIAN).
 * @{
 */
#ifdef ZXC_BIG_ENDIAN
#if defined(__GNUC__) || defined(__clang__)
#define ZXC_BSWAP16(x) __builtin_bswap16(x)
#define ZXC_BSWAP32(x) __builtin_bswap32(x)
#define ZXC_BSWAP64(x) __builtin_bswap64(x)
#elif defined(_MSC_VER)
#define ZXC_BSWAP16(x) _byteswap_ushort(x)
#define ZXC_BSWAP32(x) _byteswap_ulong(x)
#define ZXC_BSWAP64(x) _byteswap_uint64(x)
#else
#define ZXC_BSWAP16(x) ((uint16_t)(((x) >> 8) | ((x) << 8)))
#define ZXC_BSWAP32(x) \
    ((uint32_t)(((x) >> 24) | (((x) >> 8) & 0xFF00) | (((x) << 8) & 0xFF0000) | ((x) << 24)))
#define ZXC_BSWAP64(x) \
    ((uint64_t)(((uint64_t)ZXC_BSWAP32((uint32_t)(x)) << 32) | ZXC_BSWAP32((uint32_t)((x) >> 32))))
#endif
#endif
/** @} */ /* end of Byte-Swap Helpers */

/**
 * @name File Format Constants
 * @brief Magic words, header sizes, block sizes, and related constants.
 * @{
 */

/** @brief Magic word identifying ZXC files (little-endian 0x9CB02EF5). */
#define ZXC_MAGIC_WORD 0x9CB02EF5U
/** @brief Current on-disk file format version. The decoder accepts only this
 *  version; Older versions are rejected with ZXC_ERROR_BAD_VERSION. */
#define ZXC_FILE_FORMAT_VERSION 6

/** @brief Safety padding appended to buffers to tolerate overruns. */
#define ZXC_PAD_SIZE 32
/**
 * @brief Tail padding required on the decompression destination buffer.
 *
 * The decoder's fast path uses speculative wild-copy writes and gates
 * fast-loop entry on @c d_end - ZXC_DECOMPRESS_TAIL_PAD. Sizing
 * @c dst_capacity to @c uncompressed_size + ZXC_DECOMPRESS_TAIL_PAD
 * guarantees the fast path is reachable and that tail bounds checks
 * never spuriously reject the last literals of a valid block.
 *
 * @see zxc_decompress_block_bound()
 */
#define ZXC_DECOMPRESS_TAIL_PAD (ZXC_PAD_SIZE * 66)
/** @brief Assumed CPU cache line size for alignment. */
#define ZXC_CACHE_LINE_SIZE 64
/** @brief Bitmask for cache-line alignment checks. */
#define ZXC_ALIGNMENT_MASK (ZXC_CACHE_LINE_SIZE - 1)
/** @brief Round @p x up to the next cache-line boundary. */
#define ZXC_ALIGN_CL(x) (((x) + ZXC_ALIGNMENT_MASK) & ~(size_t)ZXC_ALIGNMENT_MASK)

/**
 * @brief Number of @c uint64_t words needed to hold a bitmap of @p n_bits.
 *
 * Equivalent to @c ceil(n_bits / 64).
 */
#define ZXC_BITMAP_WORDS(n_bits) (((n_bits) + 63) / 64)

/** @brief Bit flag in the Flags byte indicating checksum presence (bit 7). */
#define ZXC_FILE_FLAG_HAS_CHECKSUM 0x80U
/** @brief Bit flag in the Flags byte indicating a dictionary is required (bit 6). */
#define ZXC_FILE_FLAG_HAS_DICTIONARY 0x40U
/** @brief Mask for the checksum algorithm id (bits 0-3). */
#define ZXC_FILE_CHECKSUM_ALGO_MASK 0x0FU

/** @brief Magic word identifying ZXC dictionary files (.zxd). */
#define ZXC_DICT_MAGIC 0x9CB0D1C7U
/** @brief Current dictionary file format version. A 128-byte packed Huffman
 *         code-lengths table (shared literal table) always follows the
 *         dictionary content. */
#define ZXC_DICT_VERSION 1
/** @brief K-gram length scanned by the dictionary trainer. Aligned on the LZ
 *         minimum match length so trained patterns are matchable at encode time. */
#define ZXC_DICT_KGRAM_LEN ZXC_LZ_MIN_MATCH_LEN
/** @brief Address bits for the dictionary trainer's k-gram frequency table. */
#define ZXC_DICT_HASH_BITS 16
/** @brief Maximum number of candidate segments the dictionary trainer keeps. */
#define ZXC_DICT_MAX_SEGMENTS (1U << 16)
/** @brief Target number of sampled k-gram positions for the trainer's frequency
 *  estimate. Bounds the count so 16-bit counters stay unsaturated on large
 *  corpora; the trainer strides the corpus to hit roughly this many positions. */
#define ZXC_DICT_SAMPLE_TARGET (1U << 19)
/** @brief Number of buckets in the dictionary trainer's frequency table. */
#define ZXC_DICT_HASH_SIZE (1U << ZXC_DICT_HASH_BITS)
/** @brief Training block size for the shared-table literal statistics. */
#define ZXC_DICT_HUF_TRAIN_BLOCK 4096U
/** @brief Cap on the corpus bytes compressed by the literal-table trainer: the
 *         histogram converges early, so past it slices are strided evenly instead. */
#define ZXC_DICT_HUF_SAMPLE_BUDGET (8U << 20)

/** @brief Block header size: Type(1)+Flags(1)+Reserved(1)+CRC(1)+CompSize(4). */
#define ZXC_BLOCK_HEADER_SIZE 8
/** @brief Size of the per-block checksum field in bytes. */
#define ZXC_BLOCK_CHECKSUM_SIZE 4
/** @brief Binary size of a GLO block sub-header. */
#define ZXC_GLO_HEADER_BINARY_SIZE 16
/** @brief Binary size of a GHI block sub-header. */
#define ZXC_GHI_HEADER_BINARY_SIZE 16

/** @brief Worst-case format overhead inside a single block beyond the outer
 *  8-byte block header and the optional 4-byte checksum.
 *
 *  Covers the inner GLO/GHI sub-header (16 B) plus four section descriptors
 *  (4 x 8 = 32 B) = 48 B, with a 16 B safety margin for future format
 *  evolution. Used by zxc_compress_block_bound() and zxc_compress_bound()
 *  to size the destination buffer in the worst (incompressible) case. */
#define ZXC_BLOCK_FORMAT_OVERHEAD 64

/** @brief Binary size of a section descriptor (comp_size + raw_size). */
#define ZXC_SECTION_DESC_BINARY_SIZE 8
/** @brief 32-bit mask for extracting sizes from a section descriptor. */
#define ZXC_SECTION_SIZE_MASK 0xFFFFFFFFU
/** @brief Number of sections in a GLO block. */
#define ZXC_GLO_SECTIONS 4
/** @brief Number of sections in a GHI block. */
#define ZXC_GHI_SECTIONS 3

/** @brief Checksum algorithm id for RapidHash (default, sole implementation). */
#define ZXC_CHECKSUM_RAPIDHASH 0

/** @brief Size of the global checksum appended after EOF block (4 bytes). */
#define ZXC_GLOBAL_CHECKSUM_SIZE 4

/** @name Seekable Format Constants
 *  @brief Seek table block appended between EOF block and footer.
 *
 *  The seek table is optional (opt-in at compression time) and allows
 *  random-access decompression by recording per-block compressed and
 *  decompressed sizes.  It uses a standard ZXC block header with
 *  @c block_type = @c ZXC_BLOCK_SEK.
 *
 *  Detection from the end of the file: the reader derives @c num_blocks
 *  from the file footer (total decompressed size) and file header (block size).
 *  It then seeks backward to validate the SEK block header.
 *  @{ */
/** @brief Per-block entry size: comp_size(4) only.  decomp_size is derived
 *  from the file header's block_size (all blocks except the last are full). */
#define ZXC_SEEK_ENTRY_SIZE 4
/** @} */ /* end of Seekable Format Constants */

/** @name GLO Token Constants
 *  @brief 4-bit literal length / 4-bit match length / 16-bit offset.
 *  @{ */
/** @brief Bits for Literal Length in a GLO token. */
#define ZXC_TOKEN_LIT_BITS 4
/** @brief Bits for Match Length in a GLO token. */
#define ZXC_TOKEN_ML_BITS 4
/** @brief Mask to extract Literal Length from a GLO token. */
#define ZXC_TOKEN_LL_MASK ((1U << ZXC_TOKEN_LIT_BITS) - 1)
/** @brief Mask to extract Match Length from a GLO token. */
#define ZXC_TOKEN_ML_MASK ((1U << ZXC_TOKEN_ML_BITS) - 1)
/** @} */

/** @name GHI Sequence Constants
 *  @brief 8-bit literal length / 8-bit match length / 16-bit offset.
 *  @{ */
/** @brief Bits for Literal Length in a GHI sequence. */
#define ZXC_SEQ_LL_BITS 8
/** @brief Bits for Match Length in a GHI sequence. */
#define ZXC_SEQ_ML_BITS 8
/** @brief Bits for Offset in a GHI sequence. */
#define ZXC_SEQ_OFF_BITS 16
/** @brief Mask to extract Literal Length from a GHI sequence. */
#define ZXC_SEQ_LL_MASK ((1U << ZXC_SEQ_LL_BITS) - 1)
/** @brief Mask to extract Match Length from a GHI sequence. */
#define ZXC_SEQ_ML_MASK ((1U << ZXC_SEQ_ML_BITS) - 1)
/** @brief Mask to extract Offset from a GHI sequence. */
#define ZXC_SEQ_OFF_MASK ((1U << ZXC_SEQ_OFF_BITS) - 1)
/** @} */

/** @name Literal Stream Encoding
 *  @{ */
/** @brief Flag bit indicating an RLE run in the literal stream (0x80). */
#define ZXC_LIT_RLE_FLAG 0x80U
/** @brief Mask to extract the run/literal length (lower 7 bits). */
#define ZXC_LIT_LEN_MASK (ZXC_LIT_RLE_FLAG - 1)
/** @} */

/** @name LZ77 Constants
 *  @brief Hash table geometry, sliding window, and match parameters.
 *
 *  The hash table uses a split layout with 15-bit addressing (32 768 buckets):
 *  - `hash_table[]`: uint32_t, stores `(epoch << offset_bits) | position` (128 KB).
 *  - `hash_tags[]`:      uint8_t, stores an 8-bit tag for fast rejection (32 KB).
 *  Total: 160 KB.  The tag table fits in L1 cache, enabling a
 *  "filter-first" access pattern that avoids cold loads into hash_table
 *  on the ~60-75% of lookups where the tag mismatches.
 *  The 64 KB sliding window allows `chain_table` to use `uint16_t`.
 *  @{ */
/** @brief Address bits for the LZ77 hash table (2^15 = 32 768 buckets). */
#define ZXC_LZ_HASH_BITS 15
/** @brief Marsaglia multiplicative hash constant for 4-byte hashing. */
#define ZXC_LZ_HASH_PRIME1 0x2D35182DU
/** @brief Marsaglia/Vigna xorshift* multiplier for 5-byte hashing. */
#define ZXC_LZ_HASH_PRIME2 0x2545F4914F6CDD1DULL
/** @brief Maximum number of entries in the hash table. */
#define ZXC_LZ_HASH_SIZE (1U << ZXC_LZ_HASH_BITS)
/** @brief Sliding window size (64 KB). */
#define ZXC_LZ_WINDOW_SIZE (1U << 16)
/** @brief Mask for ring-buffer indexing into chain_table (power-of-two window). */
#define ZXC_LZ_WINDOW_MASK (ZXC_LZ_WINDOW_SIZE - 1U)
/** @brief Minimum match length for an LZ77 match. */
#define ZXC_LZ_MIN_MATCH_LEN 5
/** @brief Maximum legitimate value a varint can decode to.
 *
 * A varint value represents (ll - MASK) or (ml - MASK) and is therefore always
 * strictly less than ZXC_BLOCK_SIZE_MAX (enforced by the Block API entry
 * points). The cap is set to (ZXC_BLOCK_SIZE_MAX - 1), which fits cleanly in a
 * 3-byte varint (21 bits): the decoder rejects any 4- or 5-byte encoding, and
 * the encoder refuses to emit values above this bound. Together they bound the
 * varint surface to exactly the format-defined block size limit. */
#define ZXC_MAX_VARINT_VALUE ((uint32_t)(ZXC_BLOCK_SIZE_MAX - 1U))
/** @brief Maximum decoded output of a single sequence with INLINE ll/ml
 *         (non-varint). Used by 4x decoder bounds checks to reserve space for
 *         subsequent inline sequences in the same batch when the current
 *         sequence has a varint-extended ml. */
#define ZXC_GLO_MAX_INLINE_OUT_PER_SEQ \
    ((ZXC_TOKEN_LL_MASK - 1U) + (ZXC_TOKEN_ML_MASK - 1U) + ZXC_LZ_MIN_MATCH_LEN) /* 33 */
#define ZXC_GHI_MAX_INLINE_OUT_PER_SEQ \
    ((ZXC_SEQ_LL_MASK - 1U) + (ZXC_SEQ_ML_MASK - 1U) + ZXC_LZ_MIN_MATCH_LEN) /* 513 */
/** @brief Base bias added to encoded offsets (stored = actual - bias). */
#define ZXC_LZ_OFFSET_BIAS 1
/** @brief Maximum allowed offset distance. */
#define ZXC_LZ_MAX_DIST (ZXC_LZ_WINDOW_SIZE - 1)
/** @brief Bytes at the block end where match search stops (left as literals).
 *  Equals the 8-byte word the finder reads at each probe, so @c ip+8<=iend. */
#define ZXC_LZ_SEARCH_MARGIN (sizeof(uint64_t))
/** @} */

/** @name Optimal Parser Tuning (level >= 6)
 *  @brief Static prices and complexity guards used by the level-6 optimal
 *         LZ77 parser DP.
 *  @{ */
/** @brief Static price (bits) of a match token before varint extras: 1 byte
 *         token + 2 byte offset. */
#define ZXC_OPT_MATCH_COST_BASE ((uint32_t)(3U * CHAR_BIT))
/** @brief Threshold above which `find_best_match` is skipped at intra-match
 *         positions, keeping the parser O(N) on highly repetitive data. */
#define ZXC_OPT_LONG_MATCH_SKIP ((size_t)256)
/** @brief Minimum literal count for the sample-based Huffman cost estimator
 *         used by the optimal parser. Below this, the strided sample is too
 *         small for the resulting code-lengths to be statistically reliable,
 *         so the estimator falls back to RAW cost (8 bits/byte). */
#define ZXC_OPT_LIT_SAMPLE_MIN 1024

/** @} */

/** @name Hash Prime Constants
 *  @brief Mixing primes used by internal hash functions.
 *  @{ */
/** @brief Hash prime 1. */
#define ZXC_HASH_PRIME1 0x9E3779B97F4A7C15ULL
/** @brief Hash prime 2. */
#define ZXC_HASH_PRIME2 0xD2D84A61D2D84A61ULL
/** @} */

/** @name Huffman Codec Constants
 *  @brief Length-limited canonical Huffman codec for the GLO literal stream
 *         (active at compression level >= 6).
 *
 *  On-disk section payload layout:
 *  - @c ZXC_HUF_TABLE_SIZE bytes: @c ZXC_HUF_NUM_SYMBOLS code lengths
 *    packed two per byte (4 bits each). The same packed table is used as the
 *    per-block lengths header (enc_lit=2) and as the shared table carried by
 *    a .zxd dictionary (enc_lit=3) -- hence the public constant.
 *  - @c ZXC_HUF_STREAM_SIZES_HEADER_SIZE bytes: the first
 *    `ZXC_HUF_NUM_STREAMS - 1` sub-stream sizes as little-endian @c uint16_t;
 *    the last sub-stream size is derived from the enclosing section length.
 *  - Payload: @c ZXC_HUF_NUM_STREAMS concatenated LSB-first bit-streams,
 *    each covering an equal share of the literal indices (the last absorbs
 *    the remainder).
 *
 *  The decoder uses a single lookup table of @c ZXC_HUF_DEC_TABLE_SIZE entries
 *  (width @c ZXC_HUF_LOOKUP_BITS) that yields 1 or 2 symbols per lookup,
 *  feeding a `ZXC_HUF_NUM_STREAMS`-way interleaved hot loop.
 *  @{ */
/** @brief Maximum code length, in bits. Capped well below the package-merge
 *         algorithmic ceiling (14) to keep the decoder LUT small. */
#define ZXC_HUF_MAX_CODE_LEN 8
/** @brief Decoder LUT width: each lookup consumes this many bits and yields
 *         1 or 2 symbols. */
#define ZXC_HUF_LOOKUP_BITS 11
/** @brief Number of entries in the multi-symbol decoder lookup table. */
#define ZXC_HUF_DEC_TABLE_SIZE (1U << ZXC_HUF_LOOKUP_BITS)
/** @brief Alphabet size: one entry per possible byte value. */
#define ZXC_HUF_NUM_SYMBOLS 256
/** @brief Interleaved bit-stream count for parallel decoding. */
#define ZXC_HUF_NUM_STREAMS 4
/** @brief Sub-stream sizes header: `(ZXC_HUF_NUM_STREAMS - 1)` little-endian
 *         @c uint16_t values; the last sub-stream size is derived from the
 *         enclosing section length. */
#define ZXC_HUF_STREAM_SIZES_HEADER_SIZE ((int)((ZXC_HUF_NUM_STREAMS - 1) * sizeof(uint16_t)))
/** @brief Total Huffman header size: packed code lengths + sub-stream sizes. */
#define ZXC_HUF_HEADER_SIZE (ZXC_HUF_TABLE_SIZE + ZXC_HUF_STREAM_SIZES_HEADER_SIZE)
/** @brief Absolute floor below which Huffman cannot beat RAW even with
 *         zero-entropy literals after the 3 % savings margin. Above this
 *         floor, the precise size accounting at the call site decides per
 *         block, so the threshold is corpus-agnostic.
 *
 *         Derivation: the call site requires `huf_total < baseline * 31/32`
 *         (3 % margin = `baseline >> 5`). At zero-entropy literals the
 *         payload vanishes and `huf_total = HEADER`, giving
 *         `N > HEADER x 32/31`. The `+30` is the standard ceiling-division
 *         offset (`b - 1` with `b = 31`). Constants:
 *           - 32 = inverse of the 3 % margin (`1/32`)
 *           - 31 = `32 - 1`, the fraction kept after the margin
 *           - 30 = `31 - 1`, ceiling-division rounding offset */
#define ZXC_HUF_MIN_LITERALS ((ZXC_HUF_HEADER_SIZE * 32 + 30) / 31)
/** @brief Width of the decoder bit accumulator, in bits
 *         (`sizeof(uint64_t) * CHAR_BIT`). */
#define ZXC_HUF_ACCUM_BITS 64
/** @brief Decoder batch size: lookups per stream between two refills. */
#define ZXC_HUF_BATCH 5
/** @brief Worst-case bits consumed per stream per batch. Must stay <= 57 so
 *         that an 8-byte refill always brings the bit accumulator back to
 *         >= 56 bits before the next batch. */
#define ZXC_HUF_BATCH_BITS (ZXC_HUF_BATCH * ZXC_HUF_LOOKUP_BITS)
/** @brief Mask for indexing into the multi-symbol decoder lookup table. */
#define ZXC_HUF_TBL_MASK ((uint64_t)(ZXC_HUF_DEC_TABLE_SIZE - 1))
/** @brief Per-stream output headroom required to enter the batched fast loop:
 *         each iteration speculatively writes 2 bytes per stream and runs
 *         @c ZXC_HUF_BATCH iterations before re-checking the bound. */
#define ZXC_HUF_SAFE_MARGIN ((size_t)(2 * ZXC_HUF_BATCH))

/**
 * @brief Multi-symbol decoder lookup table entry. Bit layout:
 *   bits  0..7   sym1       - first decoded symbol
 *   bits  8..15  sym2       - second decoded symbol (junk if n_extra == 0)
 *   bits 16..19  len1       - bit length of sym1's code (1..8)
 *   bits 20..23  len_total  - total bits consumed (1..11)
 *   bit  24      n_extra    - 0 if 1 symbol, 1 if 2 symbols decoded
 *
 * Lives here (not in zxc_huffman.c) so a prebuilt table can be carried by the
 * compression context for the shared dictionary literal table.
 */
typedef struct {
    uint32_t entry;
} zxc_huf_dec_entry_t;

/**
 * @brief Boundary package-merge work item.
 *
 * Each level holds at most `2 * ZXC_HUF_NUM_SYMBOLS` of these; exposed so
 * callers can size pre-allocated scratch via ::ZXC_HUF_BUILD_SCRATCH_SIZE.
 */
typedef struct {
    uint32_t weight; /**< Accumulated weight (summed frequency) of the package. */
    int16_t left;    /**< Left child index, or -1 for a leaf. */
    int16_t right;   /**< Right child index, or -1 for a leaf. */
    int16_t sym;     /**< Symbol index for a leaf, or -1 for an internal node. */
} zxc_huf_pm_item_t;

/** @brief Trace-back stack frame for the package-merge code-length recovery. */
typedef struct {
    int8_t lvl;  /**< Package-merge level being traced back. */
    int16_t idx; /**< Item index within that level. */
} zxc_huf_pm_frame_t;

/** @brief Per-level item bound: at most leaves + paired packages from the
 *         previous level. */
#define ZXC_HUF_PM_LEVEL_BOUND (2 * ZXC_HUF_NUM_SYMBOLS)

/** @brief Worst-case scratch size (bytes) for ::zxc_huf_build_code_lengths.
 *         Carved by the function into items / counts / stack regions; sized
 *         for the worst-case alphabet (n = `ZXC_HUF_NUM_SYMBOLS`). Includes
 *         a small alignment slack between regions. */
#define ZXC_HUF_BUILD_SCRATCH_SIZE                                                               \
    ((size_t)ZXC_HUF_MAX_CODE_LEN * (size_t)ZXC_HUF_PM_LEVEL_BOUND * sizeof(zxc_huf_pm_item_t) + \
     8U + (size_t)ZXC_HUF_MAX_CODE_LEN * sizeof(int) + 8U +                                      \
     (size_t)ZXC_HUF_MAX_CODE_LEN * (size_t)ZXC_HUF_PM_LEVEL_BOUND * sizeof(zxc_huf_pm_frame_t))
/** @} */

/** @name Block Size Helpers
 *  @brief Runtime helpers for variable block sizes.
 *  @{ */

/**
 * @brief Integer log-base-2 for a 32-bit value.
 * @param v Must be a power of two (returns 0 for zero).
 * @return Floor of log2(v).
 */
static ZXC_ALWAYS_INLINE uint32_t zxc_log2_u32(const uint32_t v) {
#ifdef _MSC_VER
    unsigned long index;
    return (v == 0) ? 0 : (_BitScanReverse(&index, v) ? index : 0);
#else
    return (v == 0) ? 0 : (uint32_t)(31 - __builtin_clz(v));
#endif
}

/**
 * @brief Branchless bit_ceil: smallest power of two >= v, clamped to ZXC_BLOCK_SIZE_MIN.
 * @param[in] v Input size (must be > 0).
 * @return Smallest power of two >= @p v, clamped up to @ref ZXC_BLOCK_SIZE_MIN.
 */
static ZXC_ALWAYS_INLINE size_t zxc_block_size_ceil(const size_t v) {
    uint64_t x = (uint64_t)v - 1;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    x++;
    const size_t bs = (size_t)x;
    return (bs < ZXC_BLOCK_SIZE_MIN) ? ZXC_BLOCK_SIZE_MIN : bs;
}

/**
 * @brief Validates a block size.
 * Must be a power of two in [ZXC_BLOCK_SIZE_MIN, ZXC_BLOCK_SIZE_MAX].
 * @param[in] bs Block size to validate.
 * @return 1 if valid, 0 otherwise.
 */
static ZXC_ALWAYS_INLINE int zxc_validate_block_size(const size_t bs) {
    return bs >= ZXC_BLOCK_SIZE_MIN && bs <= ZXC_BLOCK_SIZE_MAX && (bs & (bs - 1)) == 0;
}
/** @} */

/** @} */ /* end of File Format Constants */

/**
 * @struct zxc_lz77_params_t
 * @brief Search parameters for LZ77 compression levels.
 *
 * Each compression level maps to a specific set of parameters that control the
 * trade-off between compression speed and ratio.  Higher search depths and lazy
 * matching improve ratio at the expense of throughput; larger step values
 * accelerate literal scanning but may miss short matches.
 */
typedef struct {
    /** Maximum number of candidates explored in the hash chain per position.
     *  Higher values find better matches but increase CPU cost linearly. */
    int search_depth;

    /** "Good enough" match length: once a match reaches this threshold the
     *  chain walk stops immediately, avoiding wasted effort on an already
     *  excellent match. */
    int sufficient_len;

    /** Enable lazy matching.  When set, after finding a match at position
     *  @c ip the compressor probes @c ip+1 (and @c ip+2 for level >= 4) to
     *  see if a longer match exists.  If so, a literal is emitted and the
     *  better match is taken instead.  Improves ratio but costs extra work. */
    int use_lazy;

    /** Maximum number of candidates explored during lazy evaluation (same
     *  semantics as @ref search_depth but applied to the ip+1 / ip+2 probes).
     *  Only meaningful when @ref use_lazy is non-zero. */
    int lazy_attempts;

    /** Skip lazy evaluation when the current match length already reaches
     *  this threshold: a match this long is unlikely to be beaten at the
     *  next byte.  Set to 0 when @ref use_lazy is disabled. */
    int lazy_len_threshold;

    /** Base step size when advancing through unmatched literals.
     *  1 = test every byte (best ratio), 4 = skip aggressively (fastest). */
    uint32_t step_base;

    /** Acceleration factor for step size: @c step = step_base + (distance >> step_shift).
     *  A larger value keeps the step conservative (grows slowly with distance);
     *  a smaller value ramps up quickly, skipping more in long literal runs. */
    uint32_t step_shift;
} zxc_lz77_params_t;

/**
 * @brief Retrieves LZ77 compression parameters based on the specified compression level.
 *
 * This inline function returns the appropriate LZ77 parameters configuration
 * for the given compression level.
 *
 * @param[in] level The compression level to use for determining LZ77 parameters.
 * @return zxc_lz77_params_t The LZ77 parameters structure corresponding to the specified level.
 */
static ZXC_ALWAYS_INLINE zxc_lz77_params_t zxc_get_lz77_params(const int level) {
    if (level >= ZXC_LEVEL_DENSITY) return (zxc_lz77_params_t){64, 256, 0, 0, 0, 1, 8};
    // search_depth, sufficient_len, use_lazy, lazy_attempts, lazy_len_threshold, step_base,
    // step_shift
    static const zxc_lz77_params_t table[6] = {
        {3, 16, 0, 0, 0, 4, 4},      // fallback
        {3, 16, 0, 0, 0, 4, 4},      // level 1
        {3, 18, 0, 0, 0, 3, 6},      // level 2
        {3, 16, 1, 4, 128, 1, 4},    // level 3
        {3, 18, 1, 4, 128, 1, 5},    // level 4
        {64, 256, 1, 16, 128, 1, 8}  // level 5
    };
    return table[level < ZXC_LEVEL_FASTEST ? ZXC_LEVEL_FASTEST : level];
}

/**
 * @enum zxc_block_type_t
 * @brief Defines the different types of data blocks supported by the ZXC
 * format.
 *
 * This enumeration categorizes blocks based on the compression strategy
 * applied:
 * - `ZXC_BLOCK_RAW` (0): No compression. Used when data is incompressible (high
 * entropy) or when compression would expand the data size.
 * - `ZXC_BLOCK_GLO` (1): General-purpose compression (LZ77 + Bitpacking). This
 * is the default for most data (text, binaries, JSON, etc.). Includes 4 sections descriptors.
 * - `ZXC_BLOCK_GHI` (2): General-purpose high-velocity mode using LZ77 with advanced
 * techniques (lazy matching, step skipping) for maximum ratio. Includes 3 sections descriptors.
 * - `ZXC_BLOCK_SEK` (254): Seek table block. Contains per-block compressed/decompressed sizes
 *   for random-access decompression. Placed between EOF block and file footer.
 * - `ZXC_BLOCK_EOF` (255): End of file marker.
 */
typedef enum {
    ZXC_BLOCK_RAW = 0,
    ZXC_BLOCK_GLO = 1,
    ZXC_BLOCK_GHI = 2,
    ZXC_BLOCK_SEK = 254,
    ZXC_BLOCK_EOF = 255
} zxc_block_type_t;

/**
 * @enum zxc_section_encoding_t
 * @brief Specifies the encoding methods used for internal data sections.
 *
 * These modes determine how specific components (like literals, match lengths,
 * or offsets) are stored within a block.
 * - `ZXC_SECTION_ENCODING_RAW`: Data is stored uncompressed.
 * - `ZXC_SECTION_ENCODING_RLE`: Run-Length Encoding.
 * - `ZXC_SECTION_ENCODING_HUFFMAN`: Canonical Huffman, 4-way interleaved
 *   sub-streams, max 8-bit codes, LSB-first. Only valid for the literal
 *   stream (`enc_lit`) of GLO blocks. Produced exclusively at level >= 6.
 * - `ZXC_SECTION_ENCODING_HUFFMAN_DICT`: same bitstream layout as HUFFMAN but
 *   the 128-byte code-lengths header is omitted: codes come from the shared
 *   table carried by the dictionary (.zxd). Only valid for `enc_lit` of GLO
 *   blocks in dictionary-compressed archives; requires the same dictionary
 *   (content + table, bound by dict_id) at decode time.
 */
typedef enum {
    ZXC_SECTION_ENCODING_RAW = 0,
    ZXC_SECTION_ENCODING_RLE = 1,
    ZXC_SECTION_ENCODING_HUFFMAN = 2,
    ZXC_SECTION_ENCODING_HUFFMAN_DICT = 3
} zxc_section_encoding_t;

/**
 * @struct zxc_gnr_header_t
 * @brief Header specific to General (LZ-based) compression blocks.
 *
 * This header follows the main block header when the block type is GLO/GHI. It
 * describes the layout of sequences and literals.
 *
 * @var zxc_gnr_header_t::n_sequences
 * The total count of LZ sequences in the block.
 * @var zxc_gnr_header_t::n_literals
 * The total count of literal bytes.
 * @var zxc_gnr_header_t::enc_lit
 * Encoding method used for the literal stream.
 * @var zxc_gnr_header_t::enc_litlen
 * Encoding method used for the literal lengths stream.
 * @var zxc_gnr_header_t::enc_mlen
 * Encoding method used for the match lengths stream.
 * @var zxc_gnr_header_t::enc_off
 * Encoding method used for the offset stream.
 */
typedef struct {
    uint32_t n_sequences;  // Number of sequences
    uint32_t n_literals;   // Number of literals
    uint8_t enc_lit;       // Literal encoding
    uint8_t enc_litlen;    // Literal lengths encoding
    uint8_t enc_mlen;      // Match lengths encoding
    uint8_t enc_off;       // Offset encoding (Unused in Token format, kept for alignment)
} zxc_gnr_header_t;

/**
 * @struct zxc_section_desc_t
 * @brief Describes the size attributes of a specific data section.
 *
 * Used to track the compressed and uncompressed sizes of sub-components
 * (e.g., a literal stream or offset stream) within a block.
 */
typedef struct {
    uint64_t sizes; /**< Packed sizes: compressed size (low 32 bits) | raw size (high 32 bits). */
} zxc_section_desc_t;

/**
 * @struct zxc_bit_reader_t
 * @brief Internal bit reader structure for ZXC compression/decompression.
 *
 * This structure maintains the state of the bit stream reading operation.
 * It buffers bits from the input byte stream into an accumulator to allow
 * reading variable-length bit sequences.
 */
typedef struct {
    const uint8_t* ptr; /**< Pointer to the current position in the input byte stream. */
    const uint8_t* end; /**< Pointer to the end of the input byte stream. */
    uint64_t accum;     /**< Bit accumulator holding buffered bits (64-bit buffer). */
    int bits;           /**< Number of valid bits currently in the accumulator. */
} zxc_bit_reader_t;

/**
 * ============================================================================
 * MEMORY & ENDIANNESS HELPERS
 * ============================================================================
 * Functions to handle unaligned memory access and Little Endian conversion.
 */

/**
 * @brief Reads a 16-bit unsigned integer from memory in little-endian format.
 *
 * This function interprets the bytes at the given memory address as a
 * little-endian 16-bit integer, regardless of the host system's endianness.
 * It is marked as always inline for performance critical paths.
 *
 * @param[in] p Pointer to the memory location to read from.
 * @return The 16-bit unsigned integer value read from memory.
 */
static ZXC_ALWAYS_INLINE uint16_t zxc_le16(const void* p) {
    uint16_t v;
    ZXC_MEMCPY(&v, p, sizeof(v));
#ifdef ZXC_BIG_ENDIAN
    return ZXC_BSWAP16(v);
#else
    return v;
#endif
}

/**
 * @brief Reads a 32-bit unsigned integer from memory in little-endian format.
 *
 * This function interprets the bytes at the given pointer address as a
 * little-endian 32-bit integer, regardless of the host system's endianness.
 * It is marked as always inline for performance critical paths.
 *
 * @param[in] p Pointer to the memory location to read from.
 * @return The 32-bit unsigned integer value read from memory.
 */
static ZXC_ALWAYS_INLINE uint32_t zxc_le32(const void* p) {
    uint32_t v;
    ZXC_MEMCPY(&v, p, sizeof(v));
#ifdef ZXC_BIG_ENDIAN
    return ZXC_BSWAP32(v);
#else
    return v;
#endif
}

/**
 * @brief Reads a 64-bit unsigned integer from memory in little-endian format.
 *
 * This function interprets the bytes at the given memory address as a
 * little-endian 64-bit integer, regardless of the host system's endianness.
 * It is marked as always inline for performance critical paths.
 *
 * @param[in] p Pointer to the memory location to read from.
 * @return The 64-bit unsigned integer value read from memory.
 */
static ZXC_ALWAYS_INLINE uint64_t zxc_le64(const void* p) {
    uint64_t v;
    ZXC_MEMCPY(&v, p, sizeof(v));
#ifdef ZXC_BIG_ENDIAN
    return ZXC_BSWAP64(v);
#else
    return v;
#endif
}

/**
 * @brief Stores a 16-bit integer in memory using little-endian byte order.
 *
 * This function copies the value of a 16-bit unsigned integer to the specified
 * memory location. It uses memcpy to avoid strict aliasing violations and
 * potential unaligned access issues.
 *
 * @note This function assumes the system is little-endian or that the compiler
 * optimizes the memcpy to a store instruction that handles endianness if necessary
 * (though the implementation shown is a direct copy).
 *
 * @param[out] p Pointer to the destination memory where the value will be stored.
 *          Must point to a valid memory region of at least 2 bytes.
 * @param[in] v The 16-bit unsigned integer value to store.
 */
static ZXC_ALWAYS_INLINE void zxc_store_le16(void* p, const uint16_t v) {
#ifdef ZXC_BIG_ENDIAN
    const uint16_t s = ZXC_BSWAP16(v);
    ZXC_MEMCPY(p, &s, sizeof(s));
#else
    ZXC_MEMCPY(p, &v, sizeof(v));
#endif
}

/**
 * @brief Stores a 32-bit unsigned integer in little-endian format at the specified memory location.
 *
 * This function writes the 32-bit value `v` to the memory pointed to by `p`.
 * It uses `ZXC_MEMCPY` to ensure safe memory access, avoiding potential alignment issues
 * that could occur with direct pointer casting on some architectures.
 *
 * @note This function is marked as `ZXC_ALWAYS_INLINE` to minimize function call overhead.
 *
 * @param[out] p Pointer to the destination memory where the value will be stored.
 * @param[in] v The 32-bit unsigned integer value to store.
 */
static ZXC_ALWAYS_INLINE void zxc_store_le32(void* p, const uint32_t v) {
#ifdef ZXC_BIG_ENDIAN
    const uint32_t s = ZXC_BSWAP32(v);
    ZXC_MEMCPY(p, &s, sizeof(s));
#else
    ZXC_MEMCPY(p, &v, sizeof(v));
#endif
}

/**
 * @brief Stores a 64-bit unsigned integer in little-endian format at the specified memory location.
 *
 * This function copies the 64-bit value `v` to the memory pointed to by `p`.
 * It uses `ZXC_MEMCPY` to ensure safe memory access, avoiding potential alignment issues
 * that might occur with direct pointer dereferencing on some architectures.
 *
 * @note This function assumes the system is little-endian or that the compiler optimizes
 * the memcpy to a store instruction that handles endianness correctly if `ZXC_MEMCPY`
 * is defined appropriately.
 *
 * @param[out] p Pointer to the destination memory where the value will be stored.
 * @param[in] v The 64-bit unsigned integer value to store.
 */
static ZXC_ALWAYS_INLINE void zxc_store_le64(void* p, const uint64_t v) {
#ifdef ZXC_BIG_ENDIAN
    const uint64_t s = ZXC_BSWAP64(v);
    ZXC_MEMCPY(p, &s, sizeof(s));
#else
    ZXC_MEMCPY(p, &v, sizeof(v));
#endif
}

/**
 * @brief Computes the 1-byte checksum for block headers.
 *
 * Implementation based on Marsaglia's Xorshift (PRNG) principles.
 *
 * @param[in] p Pointer to the input data to be hashed (8 bytes)
 * @return uint8_t The computed hash value.
 */
static ZXC_ALWAYS_INLINE uint8_t zxc_hash8(const uint8_t* p) {
    const uint64_t v = zxc_le64(p);
    uint64_t h = v ^ ZXC_HASH_PRIME1;
    h ^= h << 13;
    h ^= h >> 7;
    h ^= h << 17;
    return (uint8_t)((h >> 32) ^ h);
}

/**
 * @brief Computes the 2-byte checksum for file headers.
 *
 * This function generates a hash value by reading data from the given pointer.
 * The result is a 16-bit hash.
 * Implementation based on Marsaglia's Xorshift (PRNG) principles.
 *
 * @param[in] p Pointer to the input data to be hashed (16 bytes)
 * @return uint16_t The computed hash value.
 */
static ZXC_ALWAYS_INLINE uint16_t zxc_hash16(const uint8_t* p) {
    const uint64_t v1 = zxc_le64(p);
    const uint64_t v2 = zxc_le64(p + 8);
    uint64_t h = v1 ^ v2 ^ ZXC_HASH_PRIME2;
    h ^= h << 13;
    h ^= h >> 7;
    h ^= h << 17;
    const uint32_t res = (uint32_t)((h >> 32) ^ h);
    return (uint16_t)((res >> 16) ^ res);
}

/**
 * @brief Copies 16 bytes from the source memory location to the destination memory location.
 *
 * This function is forced to be inlined and uses SIMD intrinsics when available.
 * SSE2 on x86/x64, NEON on ARM, or memcpy as fallback.
 *
 * @param[out] dst Pointer to the destination memory block.
 * @param[in] src Pointer to the source memory block.
 */
static ZXC_ALWAYS_INLINE void zxc_copy16(void* dst, const void* src) {
#if defined(ZXC_USE_AVX2) || defined(ZXC_USE_AVX512) || defined(ZXC_USE_SSE2)
    // x86 SSE2/AVX2/AVX512: Single 128-bit unaligned load/store
    _mm_storeu_si128((__m128i*)dst, _mm_loadu_si128((const __m128i*)src));
#elif defined(ZXC_USE_NEON64) || defined(ZXC_USE_NEON32)
    vst1q_u8((uint8_t*)dst, vld1q_u8((const uint8_t*)src));
#else
    ZXC_MEMCPY(dst, src, 16);
#endif
}

/**
 * @brief Copies 32 bytes from source to destination using SIMD when available.
 *
 * Uses AVX2 on x86, NEON on ARM64/ARM32, or two 16-byte copies as fallback.
 *
 * @param[out] dst Pointer to the destination memory block.
 * @param[in] src Pointer to the source memory block.
 */
static ZXC_ALWAYS_INLINE void zxc_copy32(void* dst, const void* src) {
#if defined(ZXC_USE_AVX2) || defined(ZXC_USE_AVX512)
    // AVX2/AVX512: Single 256-bit (32 byte) unaligned load/store
    _mm256_storeu_si256((__m256i*)dst, _mm256_loadu_si256((const __m256i*)src));
#elif defined(ZXC_USE_SSE2)
    // SSE2: Two 128-bit (16 byte) unaligned load/stores (no 256-bit regs)
    _mm_storeu_si128((__m128i*)dst, _mm_loadu_si128((const __m128i*)src));
    _mm_storeu_si128((__m128i*)((uint8_t*)dst + 16),
                     _mm_loadu_si128((const __m128i*)((const uint8_t*)src + 16)));
#elif defined(ZXC_USE_NEON64) || defined(ZXC_USE_NEON32)
    // NEON: Two 128-bit (16 byte) unaligned load/stores
    vst1q_u8((uint8_t*)dst, vld1q_u8((const uint8_t*)src));
    vst1q_u8((uint8_t*)dst + 16, vld1q_u8((const uint8_t*)src + 16));
#else
    ZXC_MEMCPY(dst, src, 32);
#endif
}

/**
 * @brief Counts trailing zeros in a 32-bit unsigned integer.
 *
 * This function returns the number of contiguous zero bits starting from the
 * least significant bit (LSB). If the input is 0, it returns 32.
 *
 * It utilizes compiler-specific built-ins for GCC/Clang (`__builtin_ctz`) and
 * MSVC (`_BitScanForward`) for optimal performance. If no supported compiler
 * is detected, it falls back to a portable De Bruijn sequence implementation.
 *
 * @param[in] x The 32-bit unsigned integer to scan.
 * @return The number of trailing zeros (0-32).
 */
static ZXC_ALWAYS_INLINE int zxc_ctz32(const uint32_t x) {
    if (x == 0) return 32;
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctz(x);
#elif defined(_MSC_VER)
    unsigned long r;
    _BitScanForward(&r, x);
    return (int)r;
#else
    // Fallback De Bruijn (32 bits)
    static const int DeBruijn32[32] = {0,  1,  28, 2,  29, 14, 24, 3,  30, 22, 20,
                                       15, 25, 17, 4,  8,  31, 27, 13, 23, 21, 19,
                                       16, 7,  26, 12, 18, 6,  11, 5,  10, 9};
    return DeBruijn32[((uint32_t)((x & (0U - x)) * 0x077CB531U)) >> 27];
#endif
}

/**
 * @brief Counts the number of trailing zeros in a 64-bit unsigned integer.
 *
 * This function determines the number of zero bits following the least significant
 * one bit in the binary representation of `x`.
 *
 * @param[in] x The 64-bit unsigned integer to scan.
 * @return The number of trailing zeros. Returns 64 if `x` is 0.
 *
 * @note This implementation uses compiler built-ins for GCC/Clang (`__builtin_ctzll`)
 *       and MSVC (`_BitScanForward64`) when available for optimal performance.
 *       It falls back to a De Bruijn sequence multiplication method for other compilers.
 */
static ZXC_ALWAYS_INLINE int zxc_ctz64(const uint64_t x) {
    if (x == 0) return 64;
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(x);
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
    unsigned long r;
    _BitScanForward64(&r, x);
    return (int)r;
#elif defined(_MSC_VER)
    // Use two 32-bit scans to avoid fragile 64-bit De Bruijn multiplication.
    unsigned long r;
    const uint32_t lo = (uint32_t)x;
    if (_BitScanForward(&r, lo)) return (int)r;
    _BitScanForward(&r, (uint32_t)(x >> 32));
    return 32 + (int)r;
#else
    // Fallback De Bruijn for non-GCC/non-MSVC compilers
    static const int Debruijn64[64] = {
        0,  1,  48, 2,  57, 49, 28, 3,  61, 58, 50, 42, 38, 29, 17, 4,  62, 55, 59, 36, 53, 51,
        43, 22, 45, 39, 33, 30, 24, 18, 12, 5,  63, 47, 56, 27, 60, 41, 37, 16, 54, 35, 52, 21,
        44, 32, 23, 11, 46, 26, 40, 15, 34, 20, 31, 10, 25, 14, 19, 9,  13, 8,  7,  6};
    return Debruijn64[((x & (0ULL - x)) * 0x03F79D71B4CA8B09ULL) >> 58];
#endif
}

/**
 * @brief Allocates aligned memory in a cross-platform manner.
 *
 * This function provides a unified interface for allocating memory with a specific
 * alignment requirement. It wraps `_aligned_malloc` for Windows
 * environments and `posix_memalign` for POSIX-compliant systems.
 *
 * @param[in] size The size of the memory block to allocate, in bytes.
 * @param[in] alignment The alignment value, which must be a power of two and a multiple
 *                  of `sizeof(void *)`.
 * @return A pointer to the allocated memory block, or NULL if the allocation fails.
 *         The returned pointer must be freed using the corresponding aligned free function.
 */
void* zxc_aligned_malloc(const size_t size, const size_t alignment);

/**
 * @brief Frees memory previously allocated with an aligned allocation function.
 *
 * This function provides a cross-platform wrapper for freeing aligned memory.
 * On Windows, it calls `_aligned_free`.
 * On other platforms, it falls back to the standard `free` function.
 *
 * @param[in] ptr A pointer to the memory block to be freed. If ptr is NULL, no operation is
 * performed.
 */
void zxc_aligned_free(void* ptr);

/*
 * ============================================================================
 * COMPRESSION CONTEXT & STRUCTS
 * ============================================================================
 */

/*
 * INTERNAL API
 * ------------
 */

/**
 * @brief Calculates a 32-bit hash for a given input buffer.
 * @param[in] input Pointer to the data buffer.
 * @param[in] len Length of the data in bytes.
 * @param[in] hash_method Checksum algorithm identifier (e.g., ZXC_CHECKSUM_RAPIDHASH).
 * @return The calculated 32-bit hash value.
 */
static ZXC_ALWAYS_INLINE uint32_t zxc_checksum(const void* RESTRICT input, const size_t len,
                                               const uint8_t hash_method) {
    (void)hash_method; /* single algorithm for now; extend when adding more */
    const uint64_t hash = rapidhash(input, len);

    return (uint32_t)(hash ^ (hash >> (sizeof(uint32_t) * CHAR_BIT)));
}

/**
 * @brief Seeded variant of @ref zxc_checksum, for chaining a hash over
 *        non-contiguous buffers: `zxc_checksum_seed(b, bn, zxc_checksum(a, an, m), m)`
 *        hashes each byte once without a concat copy.
 * @param[in] input Pointer to the data buffer.
 * @param[in] len Length of the data in bytes.
 * @param[in] seed Previous 32-bit checksum to chain from.
 * @param[in] hash_method Checksum algorithm identifier (e.g., ZXC_CHECKSUM_RAPIDHASH).
 * @return The calculated 32-bit hash value.
 */
static ZXC_ALWAYS_INLINE uint32_t zxc_checksum_seed(const void* RESTRICT input, const size_t len,
                                                    const uint32_t seed,
                                                    const uint8_t hash_method) {
    (void)hash_method; /* single algorithm for now; extend when adding more */
    const uint64_t hash = rapidhash_withSeed(input, len, seed);

    return (uint32_t)(hash ^ (hash >> (sizeof(uint32_t) * CHAR_BIT)));
}

/**
 * @brief Combines a running hash with a new block hash using rotate-left and XOR.
 *
 * This function updates a global checksum by rotating the current hash left by 1 bit
 * (with wraparound) and XORing with the new block hash. This provides a simple but
 * effective rolling hash that depends on the order of blocks.
 *
 * Formula: result = ((hash << 1) | (hash >> 31)) ^ block_hash
 *
 * @param[in] hash The current running hash value.
 * @param[in] block_hash The hash of the new block to combine.
 * @return The updated combined hash value.
 */
static ZXC_ALWAYS_INLINE uint32_t zxc_hash_combine_rotate(const uint32_t hash,
                                                          const uint32_t block_hash) {
    return ((hash << 1) | (hash >> 31)) ^ block_hash;
}

/**
 * @brief Loads up to 7 bytes from memory in little-endian order into a uint64_t.
 *
 * This is used for partial reads at stream boundaries where fewer than 8 bytes
 * remain. Unlike ZXC_MEMCPY into a uint64_t (which is endian-dependent), this
 * function always produces a value with byte 0 in the least-significant bits.
 *
 * @param[in] p Pointer to the source bytes.
 * @param[in] n Number of bytes to read (must be < 8).
 * @return The loaded value in native host order, with bytes arranged as if
 *         read from a little-endian stream.
 */
static ZXC_ALWAYS_INLINE uint64_t zxc_le_partial(const uint8_t* p, size_t n) {
#ifdef ZXC_BIG_ENDIAN
    uint64_t v = 0;
    for (size_t i = 0; i < n; i++) v |= (uint64_t)p[i] << (i * CHAR_BIT);
    return v;
#else
    uint64_t v = 0;
    n = n > sizeof(v) ? sizeof(v) : n;
    ZXC_MEMCPY(&v, p, n);
    return v;
#endif
}

/**
 * @brief Initializes a bit reader structure.
 *
 * Sets up the internal state of the bit reader to read from the specified
 * source buffer.
 *
 * @param[out] br Pointer to the bit reader structure to initialize.
 * @param[in] src Pointer to the source buffer containing the data to read.
 * @param[in] size The size of the source buffer in bytes.
 */
static ZXC_ALWAYS_INLINE void zxc_br_init(zxc_bit_reader_t* RESTRICT br,
                                          const uint8_t* RESTRICT src, const size_t size) {
    br->ptr = src;
    br->end = src + size;
    // Safety check: ensure we have at least 8 bytes to fill the accumulator
    if (UNLIKELY(size < sizeof(uint64_t))) {
        br->accum = zxc_le_partial(src, size);
        br->ptr += size;
        br->bits = (int)(size * CHAR_BIT);
    } else {
        br->accum = zxc_le64(br->ptr);
        br->ptr += sizeof(uint64_t);
        br->bits = sizeof(uint64_t) * CHAR_BIT;
    }
}

/**
 * @brief Writes a generic header and section descriptors to a destination
 * buffer.
 *
 * Serializes the `zxc_gnr_header_t` and an array of 4 section descriptors.
 *
 * @param[out] dst Pointer to the destination buffer.
 * @param[in] rem The remaining space in the destination buffer.
 * @param[in] gh Pointer to the generic header structure to write.
 * @param[in] desc Array of 4 section descriptors to write.
 * @return int The number of bytes written, or a negative error code if the buffer
 * is too small.
 */
int zxc_write_glo_header_and_desc(uint8_t* RESTRICT dst, const size_t rem,
                                  const zxc_gnr_header_t* RESTRICT gh,
                                  const zxc_section_desc_t desc[ZXC_GLO_SECTIONS]);

/**
 * @brief Reads a generic header and section descriptors from a source buffer.
 *
 * Deserializes data into a `zxc_gnr_header_t` and an array of 4 section
 * descriptors.
 *
 * @param[in] src Pointer to the source buffer.
 * @param[in] len The length of the source buffer available for reading.
 * @param[out] gh Pointer to the generic header structure to populate.
 * @param[out] desc Array of 4 section descriptors to populate.
 *
 * @return int Returns ZXC_OK on success, or a negative zxc_error_t code on failure.
 */
int zxc_read_glo_header_and_desc(const uint8_t* RESTRICT src, const size_t len,
                                 zxc_gnr_header_t* RESTRICT gh,
                                 zxc_section_desc_t desc[ZXC_GLO_SECTIONS]);

/**
 * @brief Writes a record header and description to the destination buffer.
 *
 * @param dst Pointer to the destination buffer where the header and description will be written.
 * @param rem Remaining size available in the destination buffer.
 * @param gh Pointer to the GNR header structure containing header information.
 * @param desc Array of 3 section descriptors to be written along with the header.
 *
 * @return int Returns the number of bytes written on success, or a negative error code on failure.
 */
int zxc_write_ghi_header_and_desc(uint8_t* RESTRICT dst, const size_t rem,
                                  const zxc_gnr_header_t* RESTRICT gh,
                                  const zxc_section_desc_t desc[ZXC_GHI_SECTIONS]);

/**
 * @brief Reads a record header and section descriptors from a buffer.
 *
 * This function parses the source buffer to extract a general header and
 * up to three section descriptors from a ZXC record.
 *
 * @param[in] src Pointer to the source buffer containing the record data.
 * @param[in] len Length of the source buffer in bytes.
 * @param[out] gh Pointer to a zxc_gnr_header_t structure to store the parsed header.
 * @param[out] desc Array of 3 zxc_section_desc_t structures to store the parsed section
 * descriptors.
 *
 * @return int Returns ZXC_OK on success, or a negative zxc_error_t code on failure.
 */
int zxc_read_ghi_header_and_desc(const uint8_t* RESTRICT src, const size_t len,
                                 zxc_gnr_header_t* RESTRICT gh,
                                 zxc_section_desc_t desc[ZXC_GHI_SECTIONS]);

/* ============================================================================
 * Huffman codec for the GLO literal stream (level >= 6).
 *
 * On-disk layout, decoder geometry and tunables: see
 * @ref ZXC_HUF_MAX_CODE_LEN and the surrounding "Huffman Codec Constants"
 * group above.
 * ============================================================================
 */

/**
 * @brief Build length-limited canonical Huffman code lengths from a frequency table.
 *
 * Uses the boundary package-merge algorithm capped at `ZXC_HUF_MAX_CODE_LEN`.
 * Symbols with `freq[i] == 0` get `code_len[i] == 0`; others receive a value
 * in `[1, ZXC_HUF_MAX_CODE_LEN]`.
 *
 * @param[in]  freq     Frequency table of length `ZXC_HUF_NUM_SYMBOLS`.
 * @param[out] code_len Output code-length array of length `ZXC_HUF_NUM_SYMBOLS`.
 * @param[in]  scratch  Optional caller-owned scratch buffer of at least
 *                      ::ZXC_HUF_BUILD_SCRATCH_SIZE bytes. If `NULL`, the
 *                      function allocates its own working memory and frees
 *                      it before returning.
 * @return `ZXC_OK` on success, negative `zxc_error_t` code on failure.
 */
int zxc_huf_build_code_lengths(const uint32_t* RESTRICT freq, uint8_t* RESTRICT code_len,
                               void* RESTRICT scratch);

/**
 * @brief Encode the literal stream into a Huffman section payload.
 *
 * Writes the 128-byte length header, the 6-byte sub-stream size table and
 * the 4 concatenated LSB-first bit-streams.
 *
 * @param[in]  literals   Source literal bytes (must not alias `dst`).
 * @param[in]  n_literals Number of source bytes.
 * @param[in]  code_len   Per-symbol code lengths produced by
 *                        ::zxc_huf_build_code_lengths.
 * @param[out] dst        Destination buffer for the section payload.
 * @param[in]  dst_cap    Capacity of @p dst in bytes.
 * @return Total bytes written on success, negative `zxc_error_t` code on failure.
 */
int zxc_huf_encode_section(const uint8_t* RESTRICT literals, const size_t n_literals,
                           const uint8_t* RESTRICT code_len, uint8_t* RESTRICT dst,
                           const size_t dst_cap);

/**
 * @brief Decode a Huffman literal section payload of `payload_size` bytes.
 *
 * Writes exactly `n_literals` decoded bytes into @p dst.
 *
 * @param[in]  payload      Section payload (header + 4 sub-streams).
 * @param[in]  payload_size Total payload length in bytes.
 * @param[out] dst          Destination buffer (must not alias @p payload).
 * @param[in]  n_literals   Expected number of decoded bytes.
 * @return `ZXC_OK` on success, negative `zxc_error_t` code on failure.
 */
int zxc_huf_decode_section(const uint8_t* RESTRICT payload, const size_t payload_size,
                           uint8_t* RESTRICT dst, const size_t n_literals);

/**
 * @brief Encode a Huffman literal section using externally supplied code
 *        lengths, WITHOUT the 128-byte lengths header (shared dictionary
 *        table). Output: 6-byte sub-stream sizes header + 4 sub-streams.
 *
 * @return Bytes written on success, negative `zxc_error_t` code on failure
 *         (including `ZXC_ERROR_CORRUPT_DATA` if a literal has no code).
 */
int zxc_huf_encode_section_dict(const uint8_t* RESTRICT literals, const size_t n_literals,
                                const uint8_t* RESTRICT code_len, uint8_t* RESTRICT dst,
                                const size_t dst_cap);

/**
 * @brief Decode a Huffman literal section that carries no lengths header,
 *        using a prebuilt decode table (shared dictionary table).
 *
 * @param[in]  payload      Section payload (6-byte sizes header + 4 sub-streams).
 * @param[in]  payload_size Total payload length in bytes.
 * @param[out] dst          Destination buffer (must not alias @p payload).
 * @param[in]  n_literals   Expected number of decoded bytes.
 * @param[in]  table        Prebuilt @ref ZXC_HUF_DEC_TABLE_SIZE-entry decode table.
 * @return `ZXC_OK` on success, negative `zxc_error_t` code on failure.
 */
int zxc_huf_decode_section_dict(const uint8_t* RESTRICT payload, const size_t payload_size,
                                uint8_t* RESTRICT dst, const size_t n_literals,
                                const zxc_huf_dec_entry_t* RESTRICT table);

/**
 * @brief Build the @ref ZXC_HUF_DEC_TABLE_SIZE-entry decode table from per-symbol
 *        code lengths. Validates Kraft equality.
 * @return `ZXC_OK` on success, `ZXC_ERROR_CORRUPT_DATA` on invalid lengths.
 */
int zxc_huf_build_dec_table(const uint8_t* RESTRICT code_len, zxc_huf_dec_entry_t* RESTRICT table);

/**
 * @brief Pack per-symbol code lengths into the 128-byte (4-bit nibble) header.
 */
void zxc_huf_pack_lengths(const uint8_t* RESTRICT code_len, uint8_t* RESTRICT out);

/**
 * @brief Unpack and structurally validate a 128-byte packed lengths header.
 * @return `ZXC_OK` on success, `ZXC_ERROR_CORRUPT_DATA` on invalid lengths.
 */
int zxc_huf_unpack_lengths(const uint8_t* RESTRICT in, uint8_t* RESTRICT code_len);

/* ---------------------------------------------------------------------------
 * Compression / decompression context.
 *
 * The context owns the working buffers (hash table, sequence buffers, scratch
 * memory) that the encoder and decoder reuse across blocks. It used to be
 * exposed via zxc_sans_io.h, but no consumer outside of the library itself
 * needs to drive it directly - the public buffer / streaming / seekable APIs
 * already provide opaque wrappers (`zxc_create_cctx` / `zxc_create_dctx`).
 * Keeping the layout private lets us evolve the buffer layout (cache-line
 * placement, additional scratch arenas) without breaking the ABI.
 * --------------------------------------------------------------------------- */

/**
 * @struct zxc_cctx_t
 * @brief Compression / decompression context.
 *
 * Holds the buffers reused across blocks to avoid repeated allocations.
 *
 * **Key fields:**
 * - @c hash_table: epoch-tagged positions (`ZXC_LZ_HASH_SIZE` * 4 bytes).
 * - @c hash_tags:  8-bit tags for fast match rejection
 *   (`ZXC_LZ_HASH_SIZE` * 1 byte).
 * - @c chain_table: collision chain storing the *previous* occurrence of a
 *   hash, forming a linked list per bucket and enabling history traversal.
 * - @c epoch: drives "lazy hash table invalidation". Instead of memset-ing
 *   the hash table for every block, we store `(epoch << offset_bits) | offset`; an
 *   entry whose stored epoch differs from `ctx->epoch` is treated as empty.
 */
typedef struct {
    /* Hot zone: random access / high frequency.
     * Kept at the start to ensure they reside in the first cache line (64 bytes). */
    uint32_t* hash_table;  /**< Hash table for LZ77 match positions (epoch|pos). */
    uint8_t* hash_tags;    /**< Split tag table for fast match rejection (8-bit tags). */
    uint16_t* chain_table; /**< Chain table for collision resolution. */
    void* memory_block;    /**< Single allocation block owner. */
    uint32_t epoch;        /**< Current epoch for lazy hash table invalidation. */

    /* Warm zone: sequential access per sequence. */
    uint32_t* buf_sequences; /**< Buffer for sequence records (packed: LL(8)|ML(8)|Offset(16)). */
    uint8_t* buf_tokens;     /**< Buffer for token sequences. */
    uint16_t* buf_offsets;   /**< Buffer for offsets. */
    uint8_t* buf_extras;     /**< Buffer for extra lengths (vbytes for LL/ML). */
    uint8_t* literals;       /**< Buffer for literal bytes. */

    /* Cold zone: configuration / scratch / resizeable. */
    uint8_t* lit_buffer;                 /**< Scratch buffer for literals (RLE / Huffman). */
    size_t lit_buffer_cap;               /**< Current capacity of the scratch buffer. */
    uint8_t* work_buf;                   /**< Padded scratch buffer for buffer-API decompression. */
    size_t work_buf_cap;                 /**< Capacity of the work buffer. */
    uint8_t* opt_scratch;                /**< Optimal-parser DP scratch (level >= 6 only,
                                              lazy-allocated, packs dp/parent_len/parent_off/actions).
                                              Also reused as transient scratch for the
                                              length-limited Huffman code-length builder. */
    size_t opt_scratch_cap;              /**< Current capacity of opt_scratch in bytes. */
    int checksum_enabled;                /**< 1 if checksum calculation/verification is enabled. */
    int compression_level;               /**< Compression level. */
    size_t dict_size;                    /**< Dictionary prefill size (0 = no dictionary). */
    uint8_t* dict_buffer;                /**< [dict | data] concat scratch carved from memory_block
                                              when dict_size > 0 (NULL otherwise). */
    size_t dict_buffer_cap;              /**< Capacity of dict_buffer in bytes (0 = none). */
    const uint8_t* dict_huf_lengths;     /**< Shared dictionary literal table: 128-byte
                                     packed code-lengths header (NULL = none). Set via
                                     zxc_cctx_attach_dict_huf; caller-owned memory. */
    zxc_huf_dec_entry_t* dict_huf_table; /**< Decode table built once from
                                 dict_huf_lengths; carved from memory_block when
                                 mode == 0 and dict_size > 0 (NULL otherwise). */
    uint32_t* lit_freq_acc;              /**< Trainer hook: when non-NULL, the GLO encoder
                                              accumulates post-LZ literal byte frequencies here
                                              (256 entries). NULL outside dictionary training. */

    /* Block-size derived parameters (computed once at init). */
    size_t chunk_size;    /**< Effective block size in bytes. */
    uint32_t offset_bits; /**< log2(chunk_size) - governs epoch_mark shift. */
    uint32_t offset_mask; /**< (1U << offset_bits) - 1 */
    uint32_t max_epoch;   /**< 1U << (32 - offset_bits) */
} zxc_cctx_t;

/**
 * @brief Initialises a ZXC compression / decompression context in place.
 *
 * Allocates the internal buffers (hash table, sequence buffers, scratch) sized
 * for @p chunk_size and the requested @p mode.
 *
 * @param[out] ctx               Context to initialise.
 * @param[in]  chunk_size        Block size driving buffer sizing.
 * @param[in]  mode              1 for compression, 0 for decompression.
 * @param[in]  level             Compression level (ignored when @p mode == 0).
 * @param[in]  checksum_enabled  Non-zero to enable checksum computation.
 * @param[in]  dict_size         Dictionary prefill size; when > 0 an extra
 *                               [dict | data] concat buffer is carved into the
 *                               workspace and @c ctx->dict_buffer is set.
 *
 * @return @c ZXC_OK on success, or a negative @ref zxc_error_t code (notably
 *         @c ZXC_ERROR_MEMORY on allocation failure).
 */
int zxc_cctx_init(zxc_cctx_t* ctx, const size_t chunk_size, const int mode, const int level,
                  const int checksum_enabled, const size_t dict_size);

/**
 * @brief Attach the shared dictionary literal table to an initialised context.
 *
 * Stores @p lengths (128-byte packed code-lengths header, caller-owned, must
 * outlive the context's use) and, on decompression contexts created with
 * @c dict_size > 0, builds the decode table once into the workspace-carved
 * @c dict_huf_table. A NULL @p lengths is a no-op.
 *
 * @return @ref ZXC_OK on success, @ref ZXC_ERROR_CORRUPT_DATA if the lengths
 *         header is structurally invalid (bad nibble, Kraft inequality).
 */
int zxc_cctx_attach_dict_huf(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT lengths);

/**
 * @brief Returns the byte count that @ref zxc_cctx_init would allocate for
 *        the given parameters.
 *
 * Used by the static-cctx public API to size a caller-supplied workspace
 * before calling @ref zxc_cctx_init_in_workspace.
 *
 * @param[in] chunk_size  Block size in bytes (must satisfy
 *                        @ref zxc_validate_block_size).
 * @param[in] mode        1 = compression, 0 = decompression.
 * @param[in] level       Compression level (only consulted when @p mode == 1).
 * @param[in] dict_size   Dictionary prefill size; when > 0 the figure includes
 *                        the [dict | data] concat buffer.
 * @return Size in bytes, or 0 if the parameters are invalid.
 */
size_t zxc_cctx_compute_workspace_size(const size_t chunk_size, const int mode, const int level,
                                       const size_t dict_size);

/**
 * @brief Initialises a compression / decompression context inside a
 *        caller-supplied workspace.
 *
 * Identical to @ref zxc_cctx_init except that the persistent buffer is
 * carved out of @p workspace instead of being @c ZXC_ALIGNED_MALLOC'd
 * internally.  @p workspace must be cache-line aligned and at least as
 * large as @ref zxc_cctx_compute_workspace_size for the same parameters.
 *
 * The caller owns @p workspace and must keep it alive for the lifetime of
 * @p ctx.  @ref zxc_cctx_free becomes a no-op for contexts initialised
 * this way (the workspace is not freed by the library).
 *
 * @param[out] ctx               Context to initialise (zeroed on entry).
 * @param[in]  workspace         Caller-allocated, cache-line-aligned buffer.
 * @param[in]  workspace_size    Capacity of @p workspace in bytes.
 * @param[in]  chunk_size        Block size in bytes.
 * @param[in]  mode              1 = compression, 0 = decompression.
 * @param[in]  level             Compression level (ignored when @p mode == 0).
 * @param[in]  checksum_enabled  Non-zero to enable checksum computation.
 * @param[in]  dict_size         Dictionary prefill size; when > 0 the workspace
 *                               must include the [dict | data] concat buffer and
 *                               @c ctx->dict_buffer is set into it.
 * @return @c ZXC_OK on success, @c ZXC_ERROR_DST_TOO_SMALL if the workspace
 *         is too small, or another negative @ref zxc_error_t.
 */
int zxc_cctx_init_in_workspace(zxc_cctx_t* RESTRICT ctx, void* RESTRICT workspace,
                               const size_t workspace_size, const size_t chunk_size, const int mode,
                               const int level, const int checksum_enabled, const size_t dict_size);

/**
 * @brief Releases the internal buffers owned by a context.
 *
 * Does NOT free @p ctx itself - the caller owns the struct storage. The
 * context may safely be re-initialised with zxc_cctx_init() afterwards.
 *
 * @param[in,out] ctx Context whose buffers should be released.
 */
void zxc_cctx_free(zxc_cctx_t* ctx);

/**
 * @brief Internal wrapper function to decompress a single chunk of data.
 *
 * This function handles the decompression of a specific chunk from the source
 * buffer into the destination buffer using the provided compression context. It
 * serves as an abstraction layer over the core decompression logic.
 *
 * @param[in,out] ctx     Pointer to the ZXC compression context structure containing
 *                internal state and configuration.
 * @param[in] src     Pointer to the source buffer containing compressed data.
 * @param[in] src_sz  Size of the compressed data in the source buffer (in bytes).
 * @param[out] dst     Pointer to the destination buffer where decompressed data will
 * be written.
 * @param[in] dst_cap Capacity of the destination buffer (maximum bytes that can be
 * written).
 *
 * @return int    Returns ZXC_OK on success, or a negative zxc_error_t code on failure.
 *                Specific error codes depend on the underlying ZXC
 * implementation.
 */
int zxc_decompress_chunk_wrapper(const zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                 const size_t src_sz, uint8_t* RESTRICT dst, const size_t dst_cap);
int zxc_decompress_chunk_wrapper_dict(const zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT src,
                                      const size_t src_sz, uint8_t* RESTRICT dst,
                                      const size_t dst_cap);

/**
 * @brief Wraps the internal chunk compression logic.
 *
 * This function acts as a wrapper to compress a single chunk of data using the
 * provided compression context. It handles the interaction with the underlying
 * compression algorithm for a specific block of memory.
 *
 * @param[in,out] ctx   Pointer to the ZXC compression context containing configuration
 *              and state.
 * @param[in] chunk Pointer to the source buffer containing the raw data to
 * compress.
 * @param[in] src_sz    The size of the source chunk in bytes.
 * @param[out] dst   Pointer to the destination buffer where compressed data will be
 * written.
 * @param[in] dst_cap   The capacity of the destination buffer (maximum bytes to write).
 *
 * @return int      The number of bytes written to the destination buffer on success,
 *                  or a negative error code on failure.
 */
int zxc_compress_chunk_wrapper(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT chunk,
                               const size_t src_sz, uint8_t* RESTRICT dst, const size_t dst_cap);

/* ---------------------------------------------------------------------------
 * Internal frame primitives.
 *
 * Read/write the ZXC file header, block header, and file footer. These were
 * previously exposed via zxc_sans_io.h but no in-tree consumer outside of the
 * library implementation needs them, and exposing them freezes on-disk layout
 * details (block_flags layout, footer composition) that we want to keep free
 * to evolve until the format is declared stable.
 * --------------------------------------------------------------------------- */

/**
 * @brief On-disk header structure for a ZXC block (8 bytes, little-endian).
 *
 * @c raw_size is not stored in the header; decoders derive it from Section
 * Descriptors within the compressed payload.
 */
typedef struct {
    uint8_t block_type;  /**< Block type (see @ref zxc_block_type_t). */
    uint8_t block_flags; /**< Flags (e.g., checksum presence). */
    uint8_t reserved;    /**< Reserved for future protocol extensions. */
    uint8_t header_crc;  /**< Header integrity checksum (1 byte). */
    uint32_t comp_size;  /**< Compressed size excluding this header. */
} zxc_block_header_t;

/**
 * @brief Writes the standard ZXC file header into @p dst.
 *
 * Stores the magic word (little-endian) and the version number into the
 * provided buffer, after checking that it has sufficient capacity.
 *
 * @param[out] dst           Destination buffer.
 * @param[in]  dst_capacity  Total capacity of @p dst in bytes.
 * @param[in]  chunk_size    Block size to encode in the header.
 * @param[in]  has_checksum  Non-zero if the checksum bit must be set.
 * @param[in]  dict_id       Dictionary ID (0 = no dictionary).
 *
 * @return Number of bytes written (@c ZXC_FILE_HEADER_SIZE) on success,
 *         or @c ZXC_ERROR_DST_TOO_SMALL if @p dst_capacity is insufficient.
 */
int zxc_write_file_header(uint8_t* RESTRICT dst, const size_t dst_capacity, const size_t chunk_size,
                          const int has_checksum, const uint32_t dict_id);

/**
 * @brief Validates and reads the ZXC file header from @p src.
 *
 * Checks that the source buffer is large enough to contain a ZXC file header
 * and that the magic word and version number match the expected format.
 *
 * @param[in]  src               Pointer to the source buffer.
 * @param[in]  src_size          Size of the source buffer in bytes.
 * @param[out] out_block_size    Optional pointer that receives the recommended
 *                               block size. May be @c NULL.
 * @param[out] out_has_checksum  Optional pointer that receives the checksum
 *                               flag. May be @c NULL.
 * @param[out] out_dict_id       Optional pointer that receives the dictionary
 *                               ID (0 if none). May be @c NULL.
 *
 * @return @c ZXC_OK on success, or a negative error code (e.g.
 *         @c ZXC_ERROR_SRC_TOO_SMALL, @c ZXC_ERROR_BAD_MAGIC,
 *         @c ZXC_ERROR_BAD_VERSION).
 */
int zxc_read_file_header(const uint8_t* RESTRICT src, const size_t src_size, size_t* out_block_size,
                         int* out_has_checksum, uint32_t* out_dict_id);

/**
 * @brief Encodes a block header into @p dst.
 *
 * Serialises the contents of a @ref zxc_block_header_t structure into a byte
 * array in little-endian format, after checking that @p dst has sufficient
 * capacity.
 *
 * @param[out] dst           Destination buffer.
 * @param[in]  dst_capacity  Total capacity of @p dst in bytes.
 * @param[in]  bh            Source block header structure to serialise.
 *
 * @return Number of bytes written (@c ZXC_BLOCK_HEADER_SIZE) on success,
 *         or @c ZXC_ERROR_DST_TOO_SMALL if @p dst_capacity is insufficient.
 */
int zxc_write_block_header(uint8_t* RESTRICT dst, const size_t dst_capacity,
                           const zxc_block_header_t* bh);

/**
 * @brief Reads and parses a ZXC block header from @p src.
 *
 * Extracts the block type, flags, reserved fields, and compressed size from
 * the first @c ZXC_BLOCK_HEADER_SIZE bytes of @p src. Multi-byte fields are
 * decoded as little-endian.
 *
 * @param[in]  src       Source buffer holding the encoded block header.
 * @param[in]  src_size  Size of @p src in bytes.
 * @param[out] bh        Block header structure populated with the parsed data.
 *
 * @return @c ZXC_OK on success, or @c ZXC_ERROR_SRC_TOO_SMALL if @p src is
 *         smaller than @c ZXC_BLOCK_HEADER_SIZE.
 */
int zxc_read_block_header(const uint8_t* RESTRICT src, const size_t src_size,
                          zxc_block_header_t* bh);

/**
 * @brief Writes the ZXC file footer into @p dst.
 *
 * The footer stores the original uncompressed size and an optional global
 * checksum. It is always @c ZXC_FILE_FOOTER_SIZE (12) bytes long.
 *
 * @param[out] dst               Destination buffer.
 * @param[in]  dst_capacity      Total capacity of @p dst in bytes.
 * @param[in]  src_size          Original uncompressed size of the data.
 * @param[in]  global_hash       Global checksum hash (used only when
 *                               @p checksum_enabled is non-zero).
 * @param[in]  checksum_enabled  Non-zero if the checksum should be emitted.
 *
 * @return Number of bytes written (@c ZXC_FILE_FOOTER_SIZE) on success,
 *         or @c ZXC_ERROR_DST_TOO_SMALL on failure.
 */
int zxc_write_file_footer(uint8_t* RESTRICT dst, const size_t dst_capacity, const uint64_t src_size,
                          const uint32_t global_hash, const int checksum_enabled);

/* ---------------------------------------------------------------------------
 * Seekable cross-TU hooks (defined in zxc_seekable.c, consumed by the
 * FILE*-flavored open helper in zxc_driver.c).
 * ------------------------------------------------------------------------- */

/**
 * @brief Hands ownership of a heap-allocated reader context to a seekable
 *        handle.  The context will be released via @c ZXC_FREE when
 *        @ref zxc_seekable_free is called on @p s.
 *
 * Safe to call exactly once per handle.  Intended for thin wrappers that
 * build a @ref zxc_reader_t over their own allocated state
 * (@ref zxc_seekable_open_file) and need that state to outlive the call
 * site.
 *
 * @param[in,out] s    Seekable handle returned by @ref zxc_seekable_open_reader.
 * @param[in]     ctx  Pointer previously returned by @c ZXC_MALLOC / @c ZXC_CALLOC.
 */
void zxc_seekable_attach_owned_ctx(zxc_seekable* s, void* ctx);

/** @} */ /* end of internal */

#ifdef __cplusplus
}
#endif

#endif  // ZXC_INTERNAL_H