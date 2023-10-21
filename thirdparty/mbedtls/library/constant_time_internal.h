/**
 *  Constant-time functions
 *
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

#ifndef MBEDTLS_CONSTANT_TIME_INTERNAL_H
#define MBEDTLS_CONSTANT_TIME_INTERNAL_H

#include "common.h"

#if defined(MBEDTLS_BIGNUM_C)
#include "mbedtls/bignum.h"
#endif

#if defined(MBEDTLS_SSL_TLS_C)
#include "ssl_misc.h"
#endif

#include <stddef.h>


/** Turn a value into a mask:
 * - if \p value == 0, return the all-bits 0 mask, aka 0
 * - otherwise, return the all-bits 1 mask, aka (unsigned) -1
 *
 * This function can be used to write constant-time code by replacing branches
 * with bit operations using masks.
 *
 * \param value     The value to analyze.
 *
 * \return          Zero if \p value is zero, otherwise all-bits-one.
 */
unsigned mbedtls_ct_uint_mask(unsigned value);

#if defined(MBEDTLS_SSL_SOME_SUITES_USE_MAC)

/** Turn a value into a mask:
 * - if \p value == 0, return the all-bits 0 mask, aka 0
 * - otherwise, return the all-bits 1 mask, aka (size_t) -1
 *
 * This function can be used to write constant-time code by replacing branches
 * with bit operations using masks.
 *
 * \param value     The value to analyze.
 *
 * \return          Zero if \p value is zero, otherwise all-bits-one.
 */
size_t mbedtls_ct_size_mask(size_t value);

#endif /* MBEDTLS_SSL_SOME_SUITES_USE_MAC */

#if defined(MBEDTLS_BIGNUM_C)

/** Turn a value into a mask:
 * - if \p value == 0, return the all-bits 0 mask, aka 0
 * - otherwise, return the all-bits 1 mask, aka (mbedtls_mpi_uint) -1
 *
 * This function can be used to write constant-time code by replacing branches
 * with bit operations using masks.
 *
 * \param value     The value to analyze.
 *
 * \return          Zero if \p value is zero, otherwise all-bits-one.
 */
mbedtls_mpi_uint mbedtls_ct_mpi_uint_mask(mbedtls_mpi_uint value);

#endif /* MBEDTLS_BIGNUM_C */

#if defined(MBEDTLS_SSL_SOME_SUITES_USE_TLS_CBC)

/** Constant-flow mask generation for "greater or equal" comparison:
 * - if \p x >= \p y, return all-bits 1, that is (size_t) -1
 * - otherwise, return all bits 0, that is 0
 *
 * This function can be used to write constant-time code by replacing branches
 * with bit operations using masks.
 *
 * \param x     The first value to analyze.
 * \param y     The second value to analyze.
 *
 * \return      All-bits-one if \p x is greater or equal than \p y,
 *              otherwise zero.
 */
size_t mbedtls_ct_size_mask_ge(size_t x,
                               size_t y);

#endif /* MBEDTLS_SSL_SOME_SUITES_USE_TLS_CBC */

/** Constant-flow boolean "equal" comparison:
 * return x == y
 *
 * This is equivalent to \p x == \p y, but is likely to be compiled
 * to code using bitwise operation rather than a branch.
 *
 * \param x     The first value to analyze.
 * \param y     The second value to analyze.
 *
 * \return      1 if \p x equals to \p y, otherwise 0.
 */
unsigned mbedtls_ct_size_bool_eq(size_t x,
                                 size_t y);

#if defined(MBEDTLS_BIGNUM_C)

/** Decide if an integer is less than the other, without branches.
 *
 * This is equivalent to \p x < \p y, but is likely to be compiled
 * to code using bitwise operation rather than a branch.
 *
 * \param x     The first value to analyze.
 * \param y     The second value to analyze.
 *
 * \return      1 if \p x is less than \p y, otherwise 0.
 */
unsigned mbedtls_ct_mpi_uint_lt(const mbedtls_mpi_uint x,
                                const mbedtls_mpi_uint y);

/**
 * \brief          Check if one unsigned MPI is less than another in constant
 *                 time.
 *
 * \param A        The left-hand MPI. This must point to an array of limbs
 *                 with the same allocated length as \p B.
 * \param B        The right-hand MPI. This must point to an array of limbs
 *                 with the same allocated length as \p A.
 * \param limbs    The number of limbs in \p A and \p B.
 *                 This must not be 0.
 *
 * \return         The result of the comparison:
 *                 \c 1 if \p A is less than \p B.
 *                 \c 0 if \p A is greater than or equal to \p B.
 */
unsigned mbedtls_mpi_core_lt_ct(const mbedtls_mpi_uint *A,
                                const mbedtls_mpi_uint *B,
                                size_t limbs);
#endif /* MBEDTLS_BIGNUM_C */

/** Choose between two integer values without branches.
 *
 * This is equivalent to `condition ? if1 : if0`, but is likely to be compiled
 * to code using bitwise operation rather than a branch.
 *
 * \param condition     Condition to test.
 * \param if1           Value to use if \p condition is nonzero.
 * \param if0           Value to use if \p condition is zero.
 *
 * \return  \c if1 if \p condition is nonzero, otherwise \c if0.
 */
unsigned mbedtls_ct_uint_if(unsigned condition,
                            unsigned if1,
                            unsigned if0);

#if defined(MBEDTLS_BIGNUM_C)

/** Conditionally assign a value without branches.
 *
 * This is equivalent to `if ( condition ) dest = src`, but is likely
 * to be compiled to code using bitwise operation rather than a branch.
 *
 * \param n             \p dest and \p src must be arrays of limbs of size n.
 * \param dest          The MPI to conditionally assign to. This must point
 *                      to an initialized MPI.
 * \param src           The MPI to be assigned from. This must point to an
 *                      initialized MPI.
 * \param condition     Condition to test, must be 0 or 1.
 */
void mbedtls_ct_mpi_uint_cond_assign(size_t n,
                                     mbedtls_mpi_uint *dest,
                                     const mbedtls_mpi_uint *src,
                                     unsigned char condition);

#endif /* MBEDTLS_BIGNUM_C */

#if defined(MBEDTLS_BASE64_C)

/** Given a value in the range 0..63, return the corresponding Base64 digit.
 *
 * The implementation assumes that letters are consecutive (e.g. ASCII
 * but not EBCDIC).
 *
 * \param value     A value in the range 0..63.
 *
 * \return          A base64 digit converted from \p value.
 */
unsigned char mbedtls_ct_base64_enc_char(unsigned char value);

/** Given a Base64 digit, return its value.
 *
 * If c is not a Base64 digit ('A'..'Z', 'a'..'z', '0'..'9', '+' or '/'),
 * return -1.
 *
 * The implementation assumes that letters are consecutive (e.g. ASCII
 * but not EBCDIC).
 *
 * \param c     A base64 digit.
 *
 * \return      The value of the base64 digit \p c.
 */
signed char mbedtls_ct_base64_dec_value(unsigned char c);

#endif /* MBEDTLS_BASE64_C */

#if defined(MBEDTLS_SSL_SOME_SUITES_USE_MAC)

/** Conditional memcpy without branches.
 *
 * This is equivalent to `if ( c1 == c2 ) memcpy(dest, src, len)`, but is likely
 * to be compiled to code using bitwise operation rather than a branch.
 *
 * \param dest      The pointer to conditionally copy to.
 * \param src       The pointer to copy from. Shouldn't overlap with \p dest.
 * \param len       The number of bytes to copy.
 * \param c1        The first value to analyze in the condition.
 * \param c2        The second value to analyze in the condition.
 */
void mbedtls_ct_memcpy_if_eq(unsigned char *dest,
                             const unsigned char *src,
                             size_t len,
                             size_t c1, size_t c2);

/** Copy data from a secret position with constant flow.
 *
 * This function copies \p len bytes from \p src_base + \p offset_secret to \p
 * dst, with a code flow and memory access pattern that does not depend on \p
 * offset_secret, but only on \p offset_min, \p offset_max and \p len.
 * Functionally equivalent to `memcpy(dst, src + offset_secret, len)`.
 *
 * \note                This function reads from \p dest, but the value that
 *                      is read does not influence the result and this
 *                      function's behavior is well-defined regardless of the
 *                      contents of the buffers. This may result in false
 *                      positives from static or dynamic analyzers, especially
 *                      if \p dest is not initialized.
 *
 * \param dest          The destination buffer. This must point to a writable
 *                      buffer of at least \p len bytes.
 * \param src           The base of the source buffer. This must point to a
 *                      readable buffer of at least \p offset_max + \p len
 *                      bytes. Shouldn't overlap with \p dest.
 * \param offset        The offset in the source buffer from which to copy.
 *                      This must be no less than \p offset_min and no greater
 *                      than \p offset_max.
 * \param offset_min    The minimal value of \p offset.
 * \param offset_max    The maximal value of \p offset.
 * \param len           The number of bytes to copy.
 */
void mbedtls_ct_memcpy_offset(unsigned char *dest,
                              const unsigned char *src,
                              size_t offset,
                              size_t offset_min,
                              size_t offset_max,
                              size_t len);

/** Compute the HMAC of variable-length data with constant flow.
 *
 * This function computes the HMAC of the concatenation of \p add_data and \p
 * data, and does with a code flow and memory access pattern that does not
 * depend on \p data_len_secret, but only on \p min_data_len and \p
 * max_data_len. In particular, this function always reads exactly \p
 * max_data_len bytes from \p data.
 *
 * \param ctx               The HMAC context. It must have keys configured
 *                          with mbedtls_md_hmac_starts() and use one of the
 *                          following hashes: SHA-384, SHA-256, SHA-1 or MD-5.
 *                          It is reset using mbedtls_md_hmac_reset() after
 *                          the computation is complete to prepare for the
 *                          next computation.
 * \param add_data          The first part of the message whose HMAC is being
 *                          calculated. This must point to a readable buffer
 *                          of \p add_data_len bytes.
 * \param add_data_len      The length of \p add_data in bytes.
 * \param data              The buffer containing the second part of the
 *                          message. This must point to a readable buffer
 *                          of \p max_data_len bytes.
 * \param data_len_secret   The length of the data to process in \p data.
 *                          This must be no less than \p min_data_len and no
 *                          greater than \p max_data_len.
 * \param min_data_len      The minimal length of the second part of the
 *                          message, read from \p data.
 * \param max_data_len      The maximal length of the second part of the
 *                          message, read from \p data.
 * \param output            The HMAC will be written here. This must point to
 *                          a writable buffer of sufficient size to hold the
 *                          HMAC value.
 *
 * \retval 0 on success.
 * \retval #MBEDTLS_ERR_PLATFORM_HW_ACCEL_FAILED
 *         The hardware accelerator failed.
 */
#if defined(MBEDTLS_USE_PSA_CRYPTO)
int mbedtls_ct_hmac(mbedtls_svc_key_id_t key,
                    psa_algorithm_t alg,
                    const unsigned char *add_data,
                    size_t add_data_len,
                    const unsigned char *data,
                    size_t data_len_secret,
                    size_t min_data_len,
                    size_t max_data_len,
                    unsigned char *output);
#else
int mbedtls_ct_hmac(mbedtls_md_context_t *ctx,
                    const unsigned char *add_data,
                    size_t add_data_len,
                    const unsigned char *data,
                    size_t data_len_secret,
                    size_t min_data_len,
                    size_t max_data_len,
                    unsigned char *output);
#endif /* MBEDTLS_USE_PSA_CRYPTO */

#endif /* MBEDTLS_SSL_SOME_SUITES_USE_MAC */

#if defined(MBEDTLS_PKCS1_V15) && defined(MBEDTLS_RSA_C) && !defined(MBEDTLS_RSA_ALT)

/** This function performs the unpadding part of a PKCS#1 v1.5 decryption
 *  operation (EME-PKCS1-v1_5 decoding).
 *
 * \note The return value from this function is a sensitive value
 *       (this is unusual). #MBEDTLS_ERR_RSA_OUTPUT_TOO_LARGE shouldn't happen
 *       in a well-written application, but 0 vs #MBEDTLS_ERR_RSA_INVALID_PADDING
 *       is often a situation that an attacker can provoke and leaking which
 *       one is the result is precisely the information the attacker wants.
 *
 * \param input          The input buffer which is the payload inside PKCS#1v1.5
 *                       encryption padding, called the "encoded message EM"
 *                       by the terminology.
 * \param ilen           The length of the payload in the \p input buffer.
 * \param output         The buffer for the payload, called "message M" by the
 *                       PKCS#1 terminology. This must be a writable buffer of
 *                       length \p output_max_len bytes.
 * \param olen           The address at which to store the length of
 *                       the payload. This must not be \c NULL.
 * \param output_max_len The length in bytes of the output buffer \p output.
 *
 * \return      \c 0 on success.
 * \return      #MBEDTLS_ERR_RSA_OUTPUT_TOO_LARGE
 *              The output buffer is too small for the unpadded payload.
 * \return      #MBEDTLS_ERR_RSA_INVALID_PADDING
 *              The input doesn't contain properly formatted padding.
 */
int mbedtls_ct_rsaes_pkcs1_v15_unpadding(unsigned char *input,
                                         size_t ilen,
                                         unsigned char *output,
                                         size_t output_max_len,
                                         size_t *olen);

#endif /* MBEDTLS_PKCS1_V15 && MBEDTLS_RSA_C && ! MBEDTLS_RSA_ALT */

#endif /* MBEDTLS_CONSTANT_TIME_INTERNAL_H */
