/**
 * \file arc4.h
 *
 * \brief The ARCFOUR stream cipher
 *
 * \warning   ARC4 is considered a weak cipher and its use constitutes a
 *            security risk. We recommend considering stronger ciphers instead.
 */
/*
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 *
 */
#ifndef MBEDTLS_ARC4_H
#define MBEDTLS_ARC4_H

#if !defined(MBEDTLS_CONFIG_FILE)
#include "mbedtls/config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#include <stddef.h>

/* MBEDTLS_ERR_ARC4_HW_ACCEL_FAILED is deprecated and should not be used. */
/** ARC4 hardware accelerator failed. */
#define MBEDTLS_ERR_ARC4_HW_ACCEL_FAILED                  -0x0019

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(MBEDTLS_ARC4_ALT)
// Regular implementation
//

/**
 * \brief     ARC4 context structure
 *
 * \warning   ARC4 is considered a weak cipher and its use constitutes a
 *            security risk. We recommend considering stronger ciphers instead.
 *
 */
typedef struct mbedtls_arc4_context {
    int x;                      /*!< permutation index */
    int y;                      /*!< permutation index */
    unsigned char m[256];       /*!< permutation table */
}
mbedtls_arc4_context;

#else  /* MBEDTLS_ARC4_ALT */
#include "arc4_alt.h"
#endif /* MBEDTLS_ARC4_ALT */

/**
 * \brief          Initialize ARC4 context
 *
 * \param ctx      ARC4 context to be initialized
 *
 * \warning        ARC4 is considered a weak cipher and its use constitutes a
 *                 security risk. We recommend considering stronger ciphers
 *                 instead.
 *
 */
void mbedtls_arc4_init(mbedtls_arc4_context *ctx);

/**
 * \brief          Clear ARC4 context
 *
 * \param ctx      ARC4 context to be cleared
 *
 * \warning        ARC4 is considered a weak cipher and its use constitutes a
 *                 security risk. We recommend considering stronger ciphers
 *                 instead.
 *
 */
void mbedtls_arc4_free(mbedtls_arc4_context *ctx);

/**
 * \brief          ARC4 key schedule
 *
 * \param ctx      ARC4 context to be setup
 * \param key      the secret key
 * \param keylen   length of the key, in bytes
 *
 * \warning        ARC4 is considered a weak cipher and its use constitutes a
 *                 security risk. We recommend considering stronger ciphers
 *                 instead.
 *
 */
void mbedtls_arc4_setup(mbedtls_arc4_context *ctx, const unsigned char *key,
                        unsigned int keylen);

/**
 * \brief          ARC4 cipher function
 *
 * \param ctx      ARC4 context
 * \param length   length of the input data
 * \param input    buffer holding the input data
 * \param output   buffer for the output data
 *
 * \return         0 if successful
 *
 * \warning        ARC4 is considered a weak cipher and its use constitutes a
 *                 security risk. We recommend considering stronger ciphers
 *                 instead.
 *
 */
int mbedtls_arc4_crypt(mbedtls_arc4_context *ctx, size_t length, const unsigned char *input,
                       unsigned char *output);

#if defined(MBEDTLS_SELF_TEST)

/**
 * \brief          Checkup routine
 *
 * \return         0 if successful, or 1 if the test failed
 *
 * \warning        ARC4 is considered a weak cipher and its use constitutes a
 *                 security risk. We recommend considering stronger ciphers
 *                 instead.
 *
 */
int mbedtls_arc4_self_test(int verbose);

#endif /* MBEDTLS_SELF_TEST */

#ifdef __cplusplus
}
#endif

#endif /* arc4.h */
