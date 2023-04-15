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
 *  SPDX-License-Identifier: Apache-2.0
 *
 *  Licensed under the Apache License, Version 2.0 (the "License"); you may
 *  not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
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
unsigned char mbedtls_ct_uchar_mask_of_range( unsigned char low,
                                              unsigned char high,
                                              unsigned char c );

#endif /* MBEDTLS_TEST_HOOKS */

#endif /* MBEDTLS_CONSTANT_TIME_INVASIVE_H */
