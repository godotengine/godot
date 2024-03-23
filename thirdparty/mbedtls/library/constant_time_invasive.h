/**
 * \file constant_time_invasive.h
 *
 * \brief Constant-time module: interfaces for invasive testing only.
 *
 * The interfaces in this file are intended for testing purposes only.
 * They SHOULD NOT be made available in library integrations except when
 * building the library for testing.
 */
/*
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */

#ifndef MBEDTLS_CONSTANT_TIME_INVASIVE_H
#define MBEDTLS_CONSTANT_TIME_INVASIVE_H

#include "common.h"

#if defined(MBEDTLS_TEST_HOOKS)

/** Turn a value into a mask:
 * - if \p low <= \p c <= \p high,
 *   return the all-bits 1 mask, aka (unsigned) -1
 * - otherwise, return the all-bits 0 mask, aka 0
 *
 * \param low   The value to analyze.
 * \param high  The value to analyze.
 * \param c     The value to analyze.
 *
 * \return      All-bits-one if \p low <= \p c <= \p high, otherwise zero.
 */
unsigned char mbedtls_ct_uchar_mask_of_range(unsigned char low,
                                             unsigned char high,
                                             unsigned char c);

#endif /* MBEDTLS_TEST_HOOKS */

#endif /* MBEDTLS_CONSTANT_TIME_INVASIVE_H */
