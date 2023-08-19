/**
 * \file pem.h
 *
 * \brief Privacy Enhanced Mail (PEM) decoding
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
#ifndef MBEDTLS_PEM_H
#define MBEDTLS_PEM_H

#if !defined(MBEDTLS_CONFIG_FILE)
#include "mbedtls/config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#include <stddef.h>

/**
 * \name PEM Error codes
 * These error codes are returned in case of errors reading the
 * PEM data.
 * \{
 */
/** No PEM header or footer found. */
#define MBEDTLS_ERR_PEM_NO_HEADER_FOOTER_PRESENT          -0x1080
/** PEM string is not as expected. */
#define MBEDTLS_ERR_PEM_INVALID_DATA                      -0x1100
/** Failed to allocate memory. */
#define MBEDTLS_ERR_PEM_ALLOC_FAILED                      -0x1180
/** RSA IV is not in hex-format. */
#define MBEDTLS_ERR_PEM_INVALID_ENC_IV                    -0x1200
/** Unsupported key encryption algorithm. */
#define MBEDTLS_ERR_PEM_UNKNOWN_ENC_ALG                   -0x1280
/** Private key password can't be empty. */
#define MBEDTLS_ERR_PEM_PASSWORD_REQUIRED                 -0x1300
/** Given private key password does not allow for correct decryption. */
#define MBEDTLS_ERR_PEM_PASSWORD_MISMATCH                 -0x1380
/** Unavailable feature, e.g. hashing/encryption combination. */
#define MBEDTLS_ERR_PEM_FEATURE_UNAVAILABLE               -0x1400
/** Bad input parameters to function. */
#define MBEDTLS_ERR_PEM_BAD_INPUT_DATA                    -0x1480
/** \} name PEM Error codes */

#ifdef __cplusplus
extern "C" {
#endif

#if defined(MBEDTLS_PEM_PARSE_C)
/**
 * \brief       PEM context structure
 */
typedef struct mbedtls_pem_context {
    unsigned char *buf;     /*!< buffer for decoded data             */
    size_t buflen;          /*!< length of the buffer                */
    unsigned char *info;    /*!< buffer for extra header information */
}
mbedtls_pem_context;

/**
 * \brief       PEM context setup
 *
 * \param ctx   context to be initialized
 */
void mbedtls_pem_init(mbedtls_pem_context *ctx);

/**
 * \brief       Read a buffer for PEM information and store the resulting
 *              data into the specified context buffers.
 *
 * \param ctx       context to use
 * \param header    header string to seek and expect
 * \param footer    footer string to seek and expect
 * \param data      source data to look in (must be nul-terminated)
 * \param pwd       password for decryption (can be NULL)
 * \param pwdlen    length of password
 * \param use_len   destination for total length used (set after header is
 *                  correctly read, so unless you get
 *                  MBEDTLS_ERR_PEM_BAD_INPUT_DATA or
 *                  MBEDTLS_ERR_PEM_NO_HEADER_FOOTER_PRESENT, use_len is
 *                  the length to skip)
 *
 * \note            Attempts to check password correctness by verifying if
 *                  the decrypted text starts with an ASN.1 sequence of
 *                  appropriate length
 *
 * \return          0 on success, or a specific PEM error code
 */
int mbedtls_pem_read_buffer(mbedtls_pem_context *ctx, const char *header, const char *footer,
                            const unsigned char *data,
                            const unsigned char *pwd,
                            size_t pwdlen, size_t *use_len);

/**
 * \brief       PEM context memory freeing
 *
 * \param ctx   context to be freed
 */
void mbedtls_pem_free(mbedtls_pem_context *ctx);
#endif /* MBEDTLS_PEM_PARSE_C */

#if defined(MBEDTLS_PEM_WRITE_C)
/**
 * \brief           Write a buffer of PEM information from a DER encoded
 *                  buffer.
 *
 * \param header    The header string to write.
 * \param footer    The footer string to write.
 * \param der_data  The DER data to encode.
 * \param der_len   The length of the DER data \p der_data in Bytes.
 * \param buf       The buffer to write to.
 * \param buf_len   The length of the output buffer \p buf in Bytes.
 * \param olen      The address at which to store the total length written
 *                  or required (if \p buf_len is not enough).
 *
 * \note            You may pass \c NULL for \p buf and \c 0 for \p buf_len
 *                  to request the length of the resulting PEM buffer in
 *                  `*olen`.
 *
 * \note            This function may be called with overlapping \p der_data
 *                  and \p buf buffers.
 *
 * \return          \c 0 on success.
 * \return          #MBEDTLS_ERR_BASE64_BUFFER_TOO_SMALL if \p buf isn't large
 *                  enough to hold the PEM buffer. In  this case, `*olen` holds
 *                  the required minimum size of \p buf.
 * \return          Another PEM or BASE64 error code on other kinds of failure.
 */
int mbedtls_pem_write_buffer(const char *header, const char *footer,
                             const unsigned char *der_data, size_t der_len,
                             unsigned char *buf, size_t buf_len, size_t *olen);
#endif /* MBEDTLS_PEM_WRITE_C */

#ifdef __cplusplus
}
#endif

#endif /* pem.h */
