/**
 * \file pkcs5.h
 *
 * \brief PKCS#5 functions
 *
 * \author Mathias Olsson <mathias@kompetensum.com>
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
#ifndef MBEDTLS_PKCS5_H
#define MBEDTLS_PKCS5_H

#if !defined(MBEDTLS_CONFIG_FILE)
#include "mbedtls/config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#include "mbedtls/asn1.h"
#include "mbedtls/md.h"

#include <stddef.h>
#include <stdint.h>

/** Bad input parameters to function. */
#define MBEDTLS_ERR_PKCS5_BAD_INPUT_DATA                  -0x2f80
/** Unexpected ASN.1 data. */
#define MBEDTLS_ERR_PKCS5_INVALID_FORMAT                  -0x2f00
/** Requested encryption or digest alg not available. */
#define MBEDTLS_ERR_PKCS5_FEATURE_UNAVAILABLE             -0x2e80
/** Given private key password does not allow for correct decryption. */
#define MBEDTLS_ERR_PKCS5_PASSWORD_MISMATCH               -0x2e00

#define MBEDTLS_PKCS5_DECRYPT      0
#define MBEDTLS_PKCS5_ENCRYPT      1

#ifdef __cplusplus
extern "C" {
#endif

#if defined(MBEDTLS_ASN1_PARSE_C)

/**
 * \brief          PKCS#5 PBES2 function
 *
 * \note           When encrypting, #MBEDTLS_CIPHER_PADDING_PKCS7 must
 *                 be enabled at compile time.
 *
 * \warning        When decrypting:
 *                 - if #MBEDTLS_CIPHER_PADDING_PKCS7 is enabled at compile
 *                   time, this function validates the CBC padding and returns
 *                   #MBEDTLS_ERR_PKCS5_PASSWORD_MISMATCH if the padding is
 *                   invalid. Note that this can help active adversaries
 *                   attempting to brute-forcing the password. Note also that
 *                   there is no guarantee that an invalid password will be
 *                   detected (the chances of a valid padding with a random
 *                   password are about 1/255).
 *                 - if #MBEDTLS_CIPHER_PADDING_PKCS7 is disabled at compile
 *                   time, this function does not validate the CBC padding.
 *
 * \param pbe_params the ASN.1 algorithm parameters
 * \param mode       either #MBEDTLS_PKCS5_DECRYPT or #MBEDTLS_PKCS5_ENCRYPT
 * \param pwd        password to use when generating key
 * \param pwdlen     length of password
 * \param data       data to process
 * \param datalen    length of data
 * \param output     Output buffer.
 *                   On success, it contains the encrypted or decrypted data,
 *                   possibly followed by the CBC padding.
 *                   On failure, the content is indeterminate.
 *                   For decryption, there must be enough room for \p datalen
 *                   bytes.
 *                   For encryption, there must be enough room for
 *                   \p datalen + 1 bytes, rounded up to the block size of
 *                   the block cipher identified by \p pbe_params.
 *
 * \returns        0 on success, or a MBEDTLS_ERR_XXX code if verification fails.
 */
int mbedtls_pkcs5_pbes2(const mbedtls_asn1_buf *pbe_params, int mode,
                        const unsigned char *pwd,  size_t pwdlen,
                        const unsigned char *data, size_t datalen,
                        unsigned char *output);

#if defined(MBEDTLS_CIPHER_PADDING_PKCS7)

/**
 * \brief          PKCS#5 PBES2 function
 *
 * \warning        When decrypting:
 *                 - This function validates the CBC padding and returns
 *                   #MBEDTLS_ERR_PKCS5_PASSWORD_MISMATCH if the padding is
 *                   invalid. Note that this can help active adversaries
 *                   attempting to brute-forcing the password. Note also that
 *                   there is no guarantee that an invalid password will be
 *                   detected (the chances of a valid padding with a random
 *                   password are about 1/255).
 *
 * \param pbe_params the ASN.1 algorithm parameters
 * \param mode       either #MBEDTLS_PKCS5_DECRYPT or #MBEDTLS_PKCS5_ENCRYPT
 * \param pwd        password to use when generating key
 * \param pwdlen     length of password
 * \param data       data to process
 * \param datalen    length of data
 * \param output     Output buffer.
 *                   On success, it contains the decrypted data.
 *                   On failure, the content is indetermidate.
 *                   For decryption, there must be enough room for \p datalen
 *                   bytes.
 *                   For encryption, there must be enough room for
 *                   \p datalen + 1 bytes, rounded up to the block size of
 *                   the block cipher identified by \p pbe_params.
 * \param output_size size of output buffer.
 *                    This must be big enough to accommodate for output plus
 *                    padding data.
 * \param output_len On success, length of actual data written to the output buffer.
 *
 * \returns        0 on success, or a MBEDTLS_ERR_XXX code if parsing or decryption fails.
 */
int mbedtls_pkcs5_pbes2_ext(const mbedtls_asn1_buf *pbe_params, int mode,
                            const unsigned char *pwd,  size_t pwdlen,
                            const unsigned char *data, size_t datalen,
                            unsigned char *output, size_t output_size,
                            size_t *output_len);

#endif /* MBEDTLS_CIPHER_PADDING_PKCS7 */

#endif /* MBEDTLS_ASN1_PARSE_C */

/**
 * \brief          PKCS#5 PBKDF2 using HMAC
 *
 * \param ctx      Generic HMAC context
 * \param password Password to use when generating key
 * \param plen     Length of password
 * \param salt     Salt to use when generating key
 * \param slen     Length of salt
 * \param iteration_count       Iteration count
 * \param key_length            Length of generated key in bytes
 * \param output   Generated key. Must be at least as big as key_length
 *
 * \returns        0 on success, or a MBEDTLS_ERR_XXX code if verification fails.
 */
int mbedtls_pkcs5_pbkdf2_hmac(mbedtls_md_context_t *ctx, const unsigned char *password,
                              size_t plen, const unsigned char *salt, size_t slen,
                              unsigned int iteration_count,
                              uint32_t key_length, unsigned char *output);

#if defined(MBEDTLS_SELF_TEST)

/**
 * \brief          Checkup routine
 *
 * \return         0 if successful, or 1 if the test failed
 */
int mbedtls_pkcs5_self_test(int verbose);

#endif /* MBEDTLS_SELF_TEST */

#ifdef __cplusplus
}
#endif

#endif /* pkcs5.h */
