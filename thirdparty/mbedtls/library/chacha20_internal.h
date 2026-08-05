/**
 * \file chacha20_internal.h
 *
 * \brief Internal declarations for ChaCha20.
 */
/*
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */

#ifndef MBEDTLS_CHACHA20_INTERNAL_H
#define MBEDTLS_CHACHA20_INTERNAL_H

#include "common.h"

#include "mbedtls/chacha20.h"

#if !defined(MBEDTLS_CHACHA20_ALT)
int mbedtls_chacha20_check_counter_wrap(const mbedtls_chacha20_context *ctx,
                                        size_t size);
#endif /* !MBEDTLS_CHACHA20_ALT */

#endif /* MBEDTLS_CHACHA20_INTERNAL_H */
