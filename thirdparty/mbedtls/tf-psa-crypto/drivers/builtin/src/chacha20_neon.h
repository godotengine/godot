/**
 * \file chacha20_neon.h
 *
 * \brief Neon implementation of ChaCha20
 *
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */
#ifndef TF_PSA_CRYPTO_CHACHA20_NEON_H
#define TF_PSA_CRYPTO_CHACHA20_NEON_H

#include "tf_psa_crypto_common.h"

/*
 * The Neon implementation can be configured to process multiple blocks in parallel; increasing the
 * number of blocks gains a lot of performance, but adds on average around 250 bytes of code size
 * for each additional block.
 *
 * This is controlled by setting MBEDTLS_CHACHA20_NEON_MULTIBLOCK in the range [0..6] (0 selects
 * the scalar implementation; 1 selects single-block Neon; 2..6 select multi-block Neon).
 *
 * The default (i.e., if MBEDTLS_CHACHA20_NEON_MULTIBLOCK is not set) selects the fastest variant
 * which has better code size than the scalar implementation (based on testing for Aarch64 on clang
 * and gcc).
 *
 * Size & performance notes for Neon implementation from informal tests on Aarch64
 * (applies to both gcc and clang except as noted):
 *   - When single-block is selected, this saves around 400-550 bytes of code-size c.f. the scalar
 *     implementation
 *   - Multi-block Neon is smaller and faster than scalar (up to 2 blocks for gcc, 3 for clang)
 *   - Code size increases consistently with number of blocks
 *   - Performance increases with number of blocks (except at 5 which is slightly slower than 4)
 *   - Performance is within a few % for gcc vs clang at all settings
 *   - Performance at 4 blocks roughly matches our hardware accelerated AES-GCM impl with
 *     better code size
 *   - Performance is worse at 7 or more blocks, due to running out of Neon registers
 */

#if !defined(MBEDTLS_HAVE_NEON_INTRINSICS)
// Select scalar implementation if Neon not available
    #define MBEDTLS_CHACHA20_NEON_MULTIBLOCK 0
#elif !defined(MBEDTLS_CHACHA20_NEON_MULTIBLOCK)
// By default, select the best performing option that is not a code-size regression (based on
// measurements from recent gcc and clang).
#if defined(MBEDTLS_ARCH_IS_THUMB)
    #if defined(MBEDTLS_COMPILER_IS_GCC)
        #define MBEDTLS_CHACHA20_NEON_MULTIBLOCK 1
    #else
        #define MBEDTLS_CHACHA20_NEON_MULTIBLOCK 2
    #endif
#elif defined(MBEDTLS_ARCH_IS_ARM64)
    #define MBEDTLS_CHACHA20_NEON_MULTIBLOCK 3
#else
    #if defined(MBEDTLS_COMPILER_IS_GCC)
        #define MBEDTLS_CHACHA20_NEON_MULTIBLOCK 2
    #else
        #define MBEDTLS_CHACHA20_NEON_MULTIBLOCK 3
    #endif
#endif
#endif

#endif /* TF_PSA_CRYPTO_CHACHA20_NEON_H */
