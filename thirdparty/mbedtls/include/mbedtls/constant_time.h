/**
 *  Constant-time functions
 *
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */

#ifndef MBEDTLS_CONSTANT_TIME_H
#define MBEDTLS_CONSTANT_TIME_H

#include <stddef.h>


/** Constant-time buffer comparison without branches.
 *
 * This is equivalent to the standard memcmp function, but is likely to be
 * compiled to code using bitwise operation rather than a branch.
 *
 * This function can be used to write constant-time code by replacing branches
 * with bit operations using masks.
 *
 * \param a     Pointer to the first buffer.
 * \param b     Pointer to the second buffer.
 * \param n     The number of bytes to compare in the buffer.
 *
 * \return      Zero if the content of the two buffer is the same,
 *              otherwise non-zero.
 */
int mbedtls_ct_memcmp(const void *a,
                      const void *b,
                      size_t n);

#endif /* MBEDTLS_CONSTANT_TIME_H */
