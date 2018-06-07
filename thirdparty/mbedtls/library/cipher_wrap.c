/**
 * \file cipher_wrap.c
 *
 * \brief Generic cipher wrapper for mbed TLS
 *
 * \author Adriaan de Jong <dejong@fox-it.com>
 *
 *  Copyright (C) 2006-2015, ARM Limited, All Rights Reserved
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
 *
 *  This file is part of mbed TLS (https://tls.mbed.org)
 */

#if !defined(MBEDTLS_CONFIG_FILE)
#include "mbedtls/config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#if defined(MBEDTLS_CIPHER_C)

#include "mbedtls/cipher_internal.h"

#if defined(MBEDTLS_AES_C)
#include "mbedtls/aes.h"
#endif

#if defined(MBEDTLS_ARC4_C)
#include "mbedtls/arc4.h"
#endif

#if defined(MBEDTLS_CAMELLIA_C)
#include "mbedtls/camellia.h"
#endif

#if defined(MBEDTLS_ARIA_C)
#include "mbedtls/aria.h"
#endif

#if defined(MBEDTLS_DES_C)
#include "mbedtls/des.h"
#endif

#if defined(MBEDTLS_BLOWFISH_C)
#include "mbedtls/blowfish.h"
#endif

#if defined(MBEDTLS_GCM_C)
#include "mbedtls/gcm.h"
#endif

#if defined(MBEDTLS_CCM_C)
#include "mbedtls/ccm.h"
#endif

#if defined(MBEDTLS_CIPHER_NULL_CIPHER)
#include <string.h>
#endif

#if defined(MBEDTLS_PLATFORM_C)
#include "mbedtls/platform.h"
#else
#include <stdlib.h>
#define mbedtls_calloc    calloc
#define mbedtls_free       free
#endif

#if defined(MBEDTLS_GCM_C)
/* shared by all GCM ciphers */
static void *gcm_ctx_alloc( void )
{
    void *ctx = mbedtls_calloc( 1, sizeof( mbedtls_gcm_context ) );

    if( ctx != NULL )
        mbedtls_gcm_init( (mbedtls_gcm_context *) ctx );

    return( ctx );
}

static void gcm_ctx_free( void *ctx )
{
    mbedtls_gcm_free( ctx );
    mbedtls_free( ctx );
}
#endif /* MBEDTLS_GCM_C */

#if defined(MBEDTLS_CCM_C)
/* shared by all CCM ciphers */
static void *ccm_ctx_alloc( void )
{
    void *ctx = mbedtls_calloc( 1, sizeof( mbedtls_ccm_context ) );

    if( ctx != NULL )
        mbedtls_ccm_init( (mbedtls_ccm_context *) ctx );

    return( ctx );
}

static void ccm_ctx_free( void *ctx )
{
    mbedtls_ccm_free( ctx );
    mbedtls_free( ctx );
}
#endif /* MBEDTLS_CCM_C */

#if defined(MBEDTLS_AES_C)

static int aes_crypt_ecb_wrap( void *ctx, mbedtls_operation_t operation,
        const unsigned char *input, unsigned char *output )
{
    return mbedtls_aes_crypt_ecb( (mbedtls_aes_context *) ctx, operation, input, output );
}

#if defined(MBEDTLS_CIPHER_MODE_CBC)
static int aes_crypt_cbc_wrap( void *ctx, mbedtls_operation_t operation, size_t length,
        unsigned char *iv, const unsigned char *input, unsigned char *output )
{
    return mbedtls_aes_crypt_cbc( (mbedtls_aes_context *) ctx, operation, length, iv, input,
                          output );
}
#endif /* MBEDTLS_CIPHER_MODE_CBC */

#if defined(MBEDTLS_CIPHER_MODE_CFB)
static int aes_crypt_cfb128_wrap( void *ctx, mbedtls_operation_t operation,
        size_t length, size_t *iv_off, unsigned char *iv,
        const unsigned char *input, unsigned char *output )
{
    return mbedtls_aes_crypt_cfb128( (mbedtls_aes_context *) ctx, operation, length, iv_off, iv,
                             input, output );
}
#endif /* MBEDTLS_CIPHER_MODE_CFB */

#if defined(MBEDTLS_CIPHER_MODE_CTR)
static int aes_crypt_ctr_wrap( void *ctx, size_t length, size_t *nc_off,
        unsigned char *nonce_counter, unsigned char *stream_block,
        const unsigned char *input, unsigned char *output )
{
    return mbedtls_aes_crypt_ctr( (mbedtls_aes_context *) ctx, length, nc_off, nonce_counter,
                          stream_block, input, output );
}
#endif /* MBEDTLS_CIPHER_MODE_CTR */

static int aes_setkey_dec_wrap( void *ctx, const unsigned char *key,
                                unsigned int key_bitlen )
{
    return mbedtls_aes_setkey_dec( (mbedtls_aes_context *) ctx, key, key_bitlen );
}

static int aes_setkey_enc_wrap( void *ctx, const unsigned char *key,
                                unsigned int key_bitlen )
{
    return mbedtls_aes_setkey_enc( (mbedtls_aes_context *) ctx, key, key_bitlen );
}

static void * aes_ctx_alloc( void )
{
    mbedtls_aes_context *aes = mbedtls_calloc( 1, sizeof( mbedtls_aes_context ) );

    if( aes == NULL )
        return( NULL );

    mbedtls_aes_init( aes );

    return( aes );
}

static void aes_ctx_free( void *ctx )
{
    mbedtls_aes_free( (mbedtls_aes_context *) ctx );
    mbedtls_free( ctx );
}

static const mbedtls_cipher_base_t aes_info = {
    MBEDTLS_CIPHER_ID_AES,
    aes_crypt_ecb_wrap,
#if defined(MBEDTLS_CIPHER_MODE_CBC)
    aes_crypt_cbc_wrap,
#endif
#if defined(MBEDTLS_CIPHER_MODE_CFB)
    aes_crypt_cfb128_wrap,
#endif
#if defined(MBEDTLS_CIPHER_MODE_CTR)
    aes_crypt_ctr_wrap,
#endif
#if defined(MBEDTLS_CIPHER_MODE_STREAM)
    NULL,
#endif
    aes_setkey_enc_wrap,
    aes_setkey_dec_wrap,
    aes_ctx_alloc,
    aes_ctx_free
};

static const mbedtls_cipher_info_t aes_128_ecb_info = {
    MBEDTLS_CIPHER_AES_128_ECB,
    MBEDTLS_MODE_ECB,
    128,
    "AES-128-ECB",
    16,
    0,
    16,
    &aes_info
};

static const mbedtls_cipher_info_t aes_192_ecb_info = {
    MBEDTLS_CIPHER_AES_192_ECB,
    MBEDTLS_MODE_ECB,
    192,
    "AES-192-ECB",
    16,
    0,
    16,
    &aes_info
};

static const mbedtls_cipher_info_t aes_256_ecb_info = {
    MBEDTLS_CIPHER_AES_256_ECB,
    MBEDTLS_MODE_ECB,
    256,
    "AES-256-ECB",
    16,
    0,
    16,
    &aes_info
};

#if defined(MBEDTLS_CIPHER_MODE_CBC)
static const mbedtls_cipher_info_t aes_128_cbc_info = {
    MBEDTLS_CIPHER_AES_128_CBC,
    MBEDTLS_MODE_CBC,
    128,
    "AES-128-CBC",
    16,
    0,
    16,
    &aes_info
};

static const mbedtls_cipher_info_t aes_192_cbc_info = {
    MBEDTLS_CIPHER_AES_192_CBC,
    MBEDTLS_MODE_CBC,
    192,
    "AES-192-CBC",
    16,
    0,
    16,
    &aes_info
};

static const mbedtls_cipher_info_t aes_256_cbc_info = {
    MBEDTLS_CIPHER_AES_256_CBC,
    MBEDTLS_MODE_CBC,
    256,
    "AES-256-CBC",
    16,
    0,
    16,
    &aes_info
};
#endif /* MBEDTLS_CIPHER_MODE_CBC */

#if defined(MBEDTLS_CIPHER_MODE_CFB)
static const mbedtls_cipher_info_t aes_128_cfb128_info = {
    MBEDTLS_CIPHER_AES_128_CFB128,
    MBEDTLS_MODE_CFB,
    128,
    "AES-128-CFB128",
    16,
    0,
    16,
    &aes_info
};

static const mbedtls_cipher_info_t aes_192_cfb128_info = {
    MBEDTLS_CIPHER_AES_192_CFB128,
    MBEDTLS_MODE_CFB,
    192,
    "AES-192-CFB128",
    16,
    0,
    16,
    &aes_info
};

static const mbedtls_cipher_info_t aes_256_cfb128_info = {
    MBEDTLS_CIPHER_AES_256_CFB128,
    MBEDTLS_MODE_CFB,
    256,
    "AES-256-CFB128",
    16,
    0,
    16,
    &aes_info
};
#endif /* MBEDTLS_CIPHER_MODE_CFB */

#if defined(MBEDTLS_CIPHER_MODE_CTR)
static const mbedtls_cipher_info_t aes_128_ctr_info = {
    MBEDTLS_CIPHER_AES_128_CTR,
    MBEDTLS_MODE_CTR,
    128,
    "AES-128-CTR",
    16,
    0,
    16,
    &aes_info
};

static const mbedtls_cipher_info_t aes_192_ctr_info = {
    MBEDTLS_CIPHER_AES_192_CTR,
    MBEDTLS_MODE_CTR,
    192,
    "AES-192-CTR",
    16,
    0,
    16,
    &aes_info
};

static const mbedtls_cipher_info_t aes_256_ctr_info = {
    MBEDTLS_CIPHER_AES_256_CTR,
    MBEDTLS_MODE_CTR,
    256,
    "AES-256-CTR",
    16,
    0,
    16,
    &aes_info
};
#endif /* MBEDTLS_CIPHER_MODE_CTR */

#if defined(MBEDTLS_GCM_C)
static int gcm_aes_setkey_wrap( void *ctx, const unsigned char *key,
                                unsigned int key_bitlen )
{
    return mbedtls_gcm_setkey( (mbedtls_gcm_context *) ctx, MBEDTLS_CIPHER_ID_AES,
                     key, key_bitlen );
}

static const mbedtls_cipher_base_t gcm_aes_info = {
    MBEDTLS_CIPHER_ID_AES,
    NULL,
#if defined(MBEDTLS_CIPHER_MODE_CBC)
    NULL,
#endif
#if defined(MBEDTLS_CIPHER_MODE_CFB)
    NULL,
#endif
#if defined(MBEDTLS_CIPHER_MODE_CTR)
    NULL,
#endif
#if defined(MBEDTLS_CIPHER_MODE_STREAM)
    NULL,
#endif
    gcm_aes_setkey_wrap,
    gcm_aes_setkey_wrap,
    gcm_ctx_alloc,
    gcm_ctx_free,
};

static const mbedtls_cipher_info_t aes_128_gcm_info = {
    MBEDTLS_CIPHER_AES_128_GCM,
    MBEDTLS_MODE_GCM,
    128,
    "AES-128-GCM",
    12,
    MBEDTLS_CIPHER_VARIABLE_IV_LEN,
    16,
    &gcm_aes_info
};

static const mbedtls_cipher_info_t aes_192_gcm_info = {
    MBEDTLS_CIPHER_AES_192_GCM,
    MBEDTLS_MODE_GCM,
    192,
    "AES-192-GCM",
    12,
    MBEDTLS_CIPHER_VARIABLE_IV_LEN,
    16,
    &gcm_aes_info
};

static const mbedtls_cipher_info_t aes_256_gcm_info = {
    MBEDTLS_CIPHER_AES_256_GCM,
    MBEDTLS_MODE_GCM,
    256,
    "AES-256-GCM",
    12,
    MBEDTLS_CIPHER_VARIABLE_IV_LEN,
    16,
    &gcm_aes_info
};
#endif /* MBEDTLS_GCM_C */

#if defined(MBEDTLS_CCM_C)
static int ccm_aes_setkey_wrap( void *ctx, const unsigned char *key,
                                unsigned int key_bitlen )
{
    return mbedtls_ccm_setkey( (mbedtls_ccm_context *) ctx, MBEDTLS_CIPHER_ID_AES,
                     key, key_bitlen );
}

static const mbedtls_cipher_base_t ccm_aes_info = {
    MBEDTLS_CIPHER_ID_AES,
    NULL,
#if defined(MBEDTLS_CIPHER_MODE_CBC)
    NULL,
#endif
#if defined(MBEDTLS_CIPHER_MODE_CFB)
    NULL,
#endif
#if defined(MBEDTLS_CIPHER_MODE_CTR)
    NULL,
#endif
#if defined(MBEDTLS_CIPHER_MODE_STREAM)
    NULL,
#endif
    ccm_aes_setkey_wrap,
    ccm_aes_setkey_wrap,
    ccm_ctx_alloc,
    ccm_ctx_free,
};

static const mbedtls_cipher_info_t aes_128_ccm_info = {
    MBEDTLS_CIPHER_AES_128_CCM,
    MBEDTLS_MODE_CCM,
    128,
    "AES-128-CCM",
    12,
    MBEDTLS_CIPHER_VARIABLE_IV_LEN,
    16,
    &ccm_aes_info
};

static const mbedtls_cipher_info_t aes_192_ccm_info = {
    MBEDTLS_CIPHER_AES_192_CCM,
    MBEDTLS_MODE_CCM,
    192,
    "AES-192-CCM",
    12,
    MBEDTLS_CIPHER_VARIABLE_IV_LEN,
    16,
    &ccm_aes_info
};

static const mbedtls_cipher_info_t aes_256_ccm_info = {
    MBEDTLS_CIPHER_AES_256_CCM,
    MBEDTLS_MODE_CCM,
    256,
    "AES-256-CCM",
    12,
    MBEDTLS_CIPHER_VARIABLE_IV_LEN,
    16,
    &ccm_aes_info
};
#endif /* MBEDTLS_CCM_C */

#endif /* MBEDTLS_AES_C */

#if defined(MBEDTLS_CAMELLIA_C)

static int camellia_crypt_ecb_wrap( void *ctx, mbedtls_operation_t operation,
        const unsigned char *input, unsigned char *output )
{
    return mbedtls_camellia_crypt_ecb( (mbedtls_camellia_context *) ctx, operation, input,
                               output );
}

#if defined(MBEDTLS_CIPHER_MODE_CBC)
static int camellia_crypt_cbc_wrap( void *ctx, mbedtls_operation_t operation,
        size_t length, unsigned char *iv,
        const unsigned char *input, unsigned char *output )
{
    return mbedtls_camellia_crypt_cbc( (mbedtls_camellia_context *) ctx, operation, length, iv,
                               input, output );
}
#endif /* MBEDTLS_CIPHER_MODE_CBC */

#if defined(MBEDTLS_CIPHER_MODE_CFB)
static int camellia_crypt_cfb128_wrap( void *ctx, mbedtls_operation_t operation,
        size_t length, size_t *iv_off, unsigned char *iv,
        const unsigned char *input, unsigned char *output )
{
    return mbedtls_camellia_crypt_cfb128( (mbedtls_camellia_context *) ctx, operation, length,
                                  iv_off, iv, input, output );
}
#endif /* MBEDTLS_CIPHER_MODE_CFB */

#if defined(MBEDTLS_CIPHER_MODE_CTR)
static int camellia_crypt_ctr_wrap( void *ctx, size_t length, size_t *nc_off,
        unsigned char *nonce_counter, unsigned char *stream_block,
        const unsigned char *input, unsigned char *output )
{
    return mbedtls_camellia_crypt_ctr( (mbedtls_camellia_context *) ctx, length, nc_off,
                               nonce_counter, stream_block, input, output );
}
#endif /* MBEDTLS_CIPHER_MODE_CTR */

static int camellia_setkey_dec_wrap( void *ctx, const unsigned char *key,
                                     unsigned int key_bitlen )
{
    return mbedtls_camellia_setkey_dec( (mbedtls_camellia_context *) ctx, key, key_bitlen );
}

static int camellia_setkey_enc_wrap( void *ctx, const unsigned char *key,
                                     unsigned int key_bitlen )
{
    return mbedtls_camellia_setkey_enc( (mbedtls_camellia_context *) ctx, key, key_bitlen );
}

static void * camellia_ctx_alloc( void )
{
    mbedtls_camellia_context *ctx;
    ctx = mbedtls_calloc( 1, sizeof( mbedtls_camellia_context ) );

    if( ctx == NULL )
        return( NULL );

    mbedtls_camellia_init( ctx );

    return( ctx );
}

static void camellia_ctx_free( void *ctx )
{
    mbedtls_camellia_free( (mbedtls_camellia_context *) ctx );
    mbedtls_free( ctx );
}

static const mbedtls_cipher_base_t camellia_info = {
    MBEDTLS_CIPHER_ID_CAMELLIA,
    camellia_crypt_ecb_wrap,
#if defined(MBEDTLS_CIPHER_MODE_CBC)
    camellia_crypt_cbc_wrap,
#endif
#if defined(MBEDTLS_CIPHER_MODE_CFB)
    camellia_crypt_cfb128_wrap,
#endif
#if defined(MBEDTLS_CIPHER_MODE_CTR)
    camellia_crypt_ctr_wrap,
#endif
#if defined(MBEDTLS_CIPHER_MODE_STREAM)
    NULL,
#endif
    camellia_setkey_enc_wrap,
    camellia_setkey_dec_wrap,
    camellia_ctx_alloc,
    camellia_ctx_free
};

static const mbedtls_cipher_info_t camellia_128_ecb_info = {
    MBEDTLS_CIPHER_CAMELLIA_128_ECB,
    MBEDTLS_MODE_ECB,
    128,
    "CAMELLIA-128-ECB",
    16,
    0,
    16,
    &camellia_info
};

static const mbedtls_cipher_info_t camellia_192_ecb_info = {
    MBEDTLS_CIPHER_CAMELLIA_192_ECB,
    MBEDTLS_MODE_ECB,
    192,
    "CAMELLIA-192-ECB",
    16,
    0,
    16,
    &camellia_info
};

static const mbedtls_cipher_info_t camellia_256_ecb_info = {
    MBEDTLS_CIPHER_CAMELLIA_256_ECB,
    MBEDTLS_MODE_ECB,
    256,
    "CAMELLIA-256-ECB",
    16,
    0,
    16,
    &camellia_info
};

#if defined(MBEDTLS_CIPHER_MODE_CBC)
static const mbedtls_cipher_info_t camellia_128_cbc_info = {
    MBEDTLS_CIPHER_CAMELLIA_128_CBC,
    MBEDTLS_MODE_CBC,
    128,
    "CAMELLIA-128-CBC",
    16,
    0,
    16,
    &camellia_info
};

static const mbedtls_cipher_info_t camellia_192_cbc_info = {
    MBEDTLS_CIPHER_CAMELLIA_192_CBC,
    MBEDTLS_MODE_CBC,
    192,
    "CAMELLIA-192-CBC",
    16,
    0,
    16,
    &camellia_info
};

static const mbedtls_cipher_info_t camellia_256_cbc_info = {
    MBEDTLS_CIPHER_CAMELLIA_256_CBC,
    MBEDTLS_MODE_CBC,
    256,
    "CAMELLIA-256-CBC",
    16,
    0,
    16,
    &camellia_info
};
#endif /* MBEDTLS_CIPHER_MODE_CBC */

#if defined(MBEDTLS_CIPHER_MODE_CFB)
static const mbedtls_cipher_info_t camellia_128_cfb128_info = {
    MBEDTLS_CIPHER_CAMELLIA_128_CFB128,
    MBEDTLS_MODE_CFB,
    128,
    "CAMELLIA-128-CFB128",
    16,
    0,
    16,
    &camellia_info
};

static const mbedtls_cipher_info_t camellia_192_cfb128_info = {
    MBEDTLS_CIPHER_CAMELLIA_192_CFB128,
    MBEDTLS_MODE_CFB,
    192,
    "CAMELLIA-192-CFB128",
    16,
    0,
    16,
    &camellia_info
};

static const mbedtls_cipher_info_t camellia_256_cfb128_info = {
    MBEDTLS_CIPHER_CAMELLIA_256_CFB128,
    MBEDTLS_MODE_CFB,
    256,
    "CAMELLIA-256-CFB128",
    16,
    0,
    16,
    &camellia_info
};
#endif /* MBEDTLS_CIPHER_MODE_CFB */

#if defined(MBEDTLS_CIPHER_MODE_CTR)
static const mbedtls_cipher_info_t camellia_128_ctr_info = {
    MBEDTLS_CIPHER_CAMELLIA_128_CTR,
    MBEDTLS_MODE_CTR,
    128,
    "CAMELLIA-128-CTR",
    16,
    0,
    16,
    &camellia_info
};

static const mbedtls_cipher_info_t camellia_192_ctr_info = {
    MBEDTLS_CIPHER_CAMELLIA_192_CTR,
    MBEDTLS_MODE_CTR,
    192,
    "CAMELLIA-192-CTR",
    16,
    0,
    16,
    &camellia_info
};

static const mbedtls_cipher_info_t camellia_256_ctr_info = {
    MBEDTLS_CIPHER_CAMELLIA_256_CTR,
    MBEDTLS_MODE_CTR,
    256,
    "CAMELLIA-256-CTR",
    16,
    0,
    16,
    &camellia_info
};
#endif /* MBEDTLS_CIPHER_MODE_CTR */

#if defined(MBEDTLS_GCM_C)
static int gcm_camellia_setkey_wrap( void *ctx, const unsigned char *key,
                                     unsigned int key_bitlen )
{
    return mbedtls_gcm_setkey( (mbedtls_gcm_context *) ctx, MBEDTLS_CIPHER_ID_CAMELLIA,
                     key, key_bitlen );
}

static const mbedtls_cipher_base_t gcm_camellia_info = {
    MBEDTLS_CIPHER_ID_CAMELLIA,
    NULL,
#if defined(MBEDTLS_CIPHER_MODE_CBC)
    NULL,
#endif
#if defined(MBEDTLS_CIPHER_MODE_CFB)
    NULL,
#endif
#if defined(MBEDTLS_CIPHER_MODE_CTR)
    NULL,
#endif
#if defined(MBEDTLS_CIPHER_MODE_STREAM)
    NULL,
#endif
    gcm_camellia_setkey_wrap,
    gcm_camellia_setkey_wrap,
    gcm_ctx_alloc,
    gcm_ctx_free,
};

static const mbedtls_cipher_info_t camellia_128_gcm_info = {
    MBEDTLS_CIPHER_CAMELLIA_128_GCM,
    MBEDTLS_MODE_GCM,
    128,
    "CAMELLIA-128-GCM",
    12,
    MBEDTLS_CIPHER_VARIABLE_IV_LEN,
    16,
    &gcm_camellia_info
};

static const mbedtls_cipher_info_t camellia_192_gcm_info = {
    MBEDTLS_CIPHER_CAMELLIA_192_GCM,
    MBEDTLS_MODE_GCM,
    192,
    "CAMELLIA-192-GCM",
    12,
    MBEDTLS_CIPHER_VARIABLE_IV_LEN,
    16,
    &gcm_camellia_info
};

static const mbedtls_cipher_info_t camellia_256_gcm_info = {
    MBEDTLS_CIPHER_CAMELLIA_256_GCM,
    MBEDTLS_MODE_GCM,
    256,
    "CAMELLIA-256-GCM",
    12,
    MBEDTLS_CIPHER_VARIABLE_IV_LEN,
    16,
    &gcm_camellia_info
};
#endif /* MBEDTLS_GCM_C */

#if defined(MBEDTLS_CCM_C)
static int ccm_camellia_setkey_wrap( void *ctx, const unsigned char *key,
                                     unsigned int key_bitlen )
{
    return mbedtls_ccm_setkey( (mbedtls_ccm_context *) ctx, MBEDTLS_CIPHER_ID_CAMELLIA,
                     key, key_bitlen );
}

static const mbedtls_cipher_base_t ccm_camellia_info = {
    MBEDTLS_CIPHER_ID_CAMELLIA,
    NULL,
#if defined(MBEDTLS_CIPHER_MODE_CBC)
    NULL,
#endif
#if defined(MBEDTLS_CIPHER_MODE_CFB)
    NULL,
#endif
#if defined(MBEDTLS_CIPHER_MODE_CTR)
    NULL,
#endif
#if defined(MBEDTLS_CIPHER_MODE_STREAM)
    NULL,
#endif
    ccm_camellia_setkey_wrap,
    ccm_camellia_setkey_wrap,
    ccm_ctx_alloc,
    ccm_ctx_free,
};

static const mbedtls_cipher_info_t camellia_128_ccm_info = {
    MBEDTLS_CIPHER_CAMELLIA_128_CCM,
    MBEDTLS_MODE_CCM,
    128,
    "CAMELLIA-128-CCM",
    12,
    MBEDTLS_CIPHER_VARIABLE_IV_LEN,
    16,
    &ccm_camellia_info
};

static const mbedtls_cipher_info_t camellia_192_ccm_info = {
    MBEDTLS_CIPHER_CAMELLIA_192_CCM,
    MBEDTLS_MODE_CCM,
    192,
    "CAMELLIA-192-CCM",
    12,
    MBEDTLS_CIPHER_VARIABLE_IV_LEN,
    16,
    &ccm_camellia_info
};

static const mbedtls_cipher_info_t camellia_256_ccm_info = {
    MBEDTLS_CIPHER_CAMELLIA_256_CCM,
    MBEDTLS_MODE_CCM,
    256,
    "CAMELLIA-256-CCM",
    12,
    MBEDTLS_CIPHER_VARIABLE_IV_LEN,
    16,
    &ccm_camellia_info
};
#endif /* MBEDTLS_CCM_C */

#endif /* MBEDTLS_CAMELLIA_C */

#if defined(MBEDTLS_ARIA_C)

static int aria_crypt_ecb_wrap( void *ctx, mbedtls_operation_t operation,
        const unsigned char *input, unsigned char *output )
{
    (void) operation;
    return mbedtls_aria_crypt_ecb( (mbedtls_aria_context *) ctx, input,
                               output );
}

#if defined(MBEDTLS_CIPHER_MODE_CBC)
static int aria_crypt_cbc_wrap( void *ctx, mbedtls_operation_t operation,
        size_t length, unsigned char *iv,
        const unsigned char *input, unsigned char *output )
{
    return mbedtls_aria_crypt_cbc( (mbedtls_aria_context *) ctx, operation, length, iv,
                               input, output );
}
#endif /* MBEDTLS_CIPHER_MODE_CBC */

#if defined(MBEDTLS_CIPHER_MODE_CFB)
static int aria_crypt_cfb128_wrap( void *ctx, mbedtls_operation_t operation,
        size_t length, size_t *iv_off, unsigned char *iv,
        const unsigned char *input, unsigned char *output )
{
    return mbedtls_aria_crypt_cfb128( (mbedtls_aria_context *) ctx, operation, length,
                                  iv_off, iv, input, output );
}
#endif /* MBEDTLS_CIPHER_MODE_CFB */

#if defined(MBEDTLS_CIPHER_MODE_CTR)
static int aria_crypt_ctr_wrap( void *ctx, size_t length, size_t *nc_off,
        unsigned char *nonce_counter, unsigned char *stream_block,
        const unsigned char *input, unsigned char *output )
{
    return mbedtls_aria_crypt_ctr( (mbedtls_aria_context *) ctx, length, nc_off,
                               nonce_counter, stream_block, input, output );
}
#endif /* MBEDTLS_CIPHER_MODE_CTR */

static int aria_setkey_dec_wrap( void *ctx, const unsigned char *key,
                                     unsigned int key_bitlen )
{
    return mbedtls_aria_setkey_dec( (mbedtls_aria_context *) ctx, key, key_bitlen );
}

static int aria_setkey_enc_wrap( void *ctx, const unsigned char *key,
                                     unsigned int key_bitlen )
{
    return mbedtls_aria_setkey_enc( (mbedtls_aria_context *) ctx, key, key_bitlen );
}

static void * aria_ctx_alloc( void )
{
    mbedtls_aria_context *ctx;
    ctx = mbedtls_calloc( 1, sizeof( mbedtls_aria_context ) );

    if( ctx == NULL )
        return( NULL );

    mbedtls_aria_init( ctx );

    return( ctx );
}

static void aria_ctx_free( void *ctx )
{
    mbedtls_aria_free( (mbedtls_aria_context *) ctx );
    mbedtls_free( ctx );
}

static const mbedtls_cipher_base_t aria_info = {
    MBEDTLS_CIPHER_ID_ARIA,
    aria_crypt_ecb_wrap,
#if defined(MBEDTLS_CIPHER_MODE_CBC)
    aria_crypt_cbc_wrap,
#endif
#if defined(MBEDTLS_CIPHER_MODE_CFB)
    aria_crypt_cfb128_wrap,
#endif
#if defined(MBEDTLS_CIPHER_MODE_CTR)
    aria_crypt_ctr_wrap,
#endif
#if defined(MBEDTLS_CIPHER_MODE_STREAM)
    NULL,
#endif
    aria_setkey_enc_wrap,
    aria_setkey_dec_wrap,
    aria_ctx_alloc,
    aria_ctx_free
};

static const mbedtls_cipher_info_t aria_128_ecb_info = {
    MBEDTLS_CIPHER_ARIA_128_ECB,
    MBEDTLS_MODE_ECB,
    128,
    "ARIA-128-ECB",
    16,
    0,
    16,
    &aria_info
};

static const mbedtls_cipher_info_t aria_192_ecb_info = {
    MBEDTLS_CIPHER_ARIA_192_ECB,
    MBEDTLS_MODE_ECB,
    192,
    "ARIA-192-ECB",
    16,
    0,
    16,
    &aria_info
};

static const mbedtls_cipher_info_t aria_256_ecb_info = {
    MBEDTLS_CIPHER_ARIA_256_ECB,
    MBEDTLS_MODE_ECB,
    256,
    "ARIA-256-ECB",
    16,
    0,
    16,
    &aria_info
};

#if defined(MBEDTLS_CIPHER_MODE_CBC)
static const mbedtls_cipher_info_t aria_128_cbc_info = {
    MBEDTLS_CIPHER_ARIA_128_CBC,
    MBEDTLS_MODE_CBC,
    128,
    "ARIA-128-CBC",
    16,
    0,
    16,
    &aria_info
};

static const mbedtls_cipher_info_t aria_192_cbc_info = {
    MBEDTLS_CIPHER_ARIA_192_CBC,
    MBEDTLS_MODE_CBC,
    192,
    "ARIA-192-CBC",
    16,
    0,
    16,
    &aria_info
};

static const mbedtls_cipher_info_t aria_256_cbc_info = {
    MBEDTLS_CIPHER_ARIA_256_CBC,
    MBEDTLS_MODE_CBC,
    256,
    "ARIA-256-CBC",
    16,
    0,
    16,
    &aria_info
};
#endif /* MBEDTLS_CIPHER_MODE_CBC */

#if defined(MBEDTLS_CIPHER_MODE_CFB)
static const mbedtls_cipher_info_t aria_128_cfb128_info = {
    MBEDTLS_CIPHER_ARIA_128_CFB128,
    MBEDTLS_MODE_CFB,
    128,
    "ARIA-128-CFB128",
    16,
    0,
    16,
    &aria_info
};

static const mbedtls_cipher_info_t aria_192_cfb128_info = {
    MBEDTLS_CIPHER_ARIA_192_CFB128,
    MBEDTLS_MODE_CFB,
    192,
    "ARIA-192-CFB128",
    16,
    0,
    16,
    &aria_info
};

static const mbedtls_cipher_info_t aria_256_cfb128_info = {
    MBEDTLS_CIPHER_ARIA_256_CFB128,
    MBEDTLS_MODE_CFB,
    256,
    "ARIA-256-CFB128",
    16,
    0,
    16,
    &aria_info
};
#endif /* MBEDTLS_CIPHER_MODE_CFB */

#if defined(MBEDTLS_CIPHER_MODE_CTR)
static const mbedtls_cipher_info_t aria_128_ctr_info = {
    MBEDTLS_CIPHER_ARIA_128_CTR,
    MBEDTLS_MODE_CTR,
    128,
    "ARIA-128-CTR",
    16,
    0,
    16,
    &aria_info
};

static const mbedtls_cipher_info_t aria_192_ctr_info = {
    MBEDTLS_CIPHER_ARIA_192_CTR,
    MBEDTLS_MODE_CTR,
    192,
    "ARIA-192-CTR",
    16,
    0,
    16,
    &aria_info
};

static const mbedtls_cipher_info_t aria_256_ctr_info = {
    MBEDTLS_CIPHER_ARIA_256_CTR,
    MBEDTLS_MODE_CTR,
    256,
    "ARIA-256-CTR",
    16,
    0,
    16,
    &aria_info
};
#endif /* MBEDTLS_CIPHER_MODE_CTR */

#if defined(MBEDTLS_GCM_C)
static int gcm_aria_setkey_wrap( void *ctx, const unsigned char *key,
                                     unsigned int key_bitlen )
{
    return mbedtls_gcm_setkey( (mbedtls_gcm_context *) ctx, MBEDTLS_CIPHER_ID_ARIA,
                     key, key_bitlen );
}

static const mbedtls_cipher_base_t gcm_aria_info = {
    MBEDTLS_CIPHER_ID_ARIA,
    NULL,
#if defined(MBEDTLS_CIPHER_MODE_CBC)
    NULL,
#endif
#if defined(MBEDTLS_CIPHER_MODE_CFB)
    NULL,
#endif
#if defined(MBEDTLS_CIPHER_MODE_CTR)
    NULL,
#endif
#if defined(MBEDTLS_CIPHER_MODE_STREAM)
    NULL,
#endif
    gcm_aria_setkey_wrap,
    gcm_aria_setkey_wrap,
    gcm_ctx_alloc,
    gcm_ctx_free,
};

static const mbedtls_cipher_info_t aria_128_gcm_info = {
    MBEDTLS_CIPHER_ARIA_128_GCM,
    MBEDTLS_MODE_GCM,
    128,
    "ARIA-128-GCM",
    12,
    MBEDTLS_CIPHER_VARIABLE_IV_LEN,
    16,
    &gcm_aria_info
};

static const mbedtls_cipher_info_t aria_192_gcm_info = {
    MBEDTLS_CIPHER_ARIA_192_GCM,
    MBEDTLS_MODE_GCM,
    192,
    "ARIA-192-GCM",
    12,
    MBEDTLS_CIPHER_VARIABLE_IV_LEN,
    16,
    &gcm_aria_info
};

static const mbedtls_cipher_info_t aria_256_gcm_info = {
    MBEDTLS_CIPHER_ARIA_256_GCM,
    MBEDTLS_MODE_GCM,
    256,
    "ARIA-256-GCM",
    12,
    MBEDTLS_CIPHER_VARIABLE_IV_LEN,
    16,
    &gcm_aria_info
};
#endif /* MBEDTLS_GCM_C */

#if defined(MBEDTLS_CCM_C)
static int ccm_aria_setkey_wrap( void *ctx, const unsigned char *key,
                                     unsigned int key_bitlen )
{
    return mbedtls_ccm_setkey( (mbedtls_ccm_context *) ctx, MBEDTLS_CIPHER_ID_ARIA,
                     key, key_bitlen );
}

static const mbedtls_cipher_base_t ccm_aria_info = {
    MBEDTLS_CIPHER_ID_ARIA,
    NULL,
#if defined(MBEDTLS_CIPHER_MODE_CBC)
    NULL,
#endif
#if defined(MBEDTLS_CIPHER_MODE_CFB)
    NULL,
#endif
#if defined(MBEDTLS_CIPHER_MODE_CTR)
    NULL,
#endif
#if defined(MBEDTLS_CIPHER_MODE_STREAM)
    NULL,
#endif
    ccm_aria_setkey_wrap,
    ccm_aria_setkey_wrap,
    ccm_ctx_alloc,
    ccm_ctx_free,
};

static const mbedtls_cipher_info_t aria_128_ccm_info = {
    MBEDTLS_CIPHER_ARIA_128_CCM,
    MBEDTLS_MODE_CCM,
    128,
    "ARIA-128-CCM",
    12,
    MBEDTLS_CIPHER_VARIABLE_IV_LEN,
    16,
    &ccm_aria_info
};

static const mbedtls_cipher_info_t aria_192_ccm_info = {
    MBEDTLS_CIPHER_ARIA_192_CCM,
    MBEDTLS_MODE_CCM,
    192,
    "ARIA-192-CCM",
    12,
    MBEDTLS_CIPHER_VARIABLE_IV_LEN,
    16,
    &ccm_aria_info
};

static const mbedtls_cipher_info_t aria_256_ccm_info = {
    MBEDTLS_CIPHER_ARIA_256_CCM,
    MBEDTLS_MODE_CCM,
    256,
    "ARIA-256-CCM",
    12,
    MBEDTLS_CIPHER_VARIABLE_IV_LEN,
    16,
    &ccm_aria_info
};
#endif /* MBEDTLS_CCM_C */

#endif /* MBEDTLS_ARIA_C */

#if defined(MBEDTLS_DES_C)

static int des_crypt_ecb_wrap( void *ctx, mbedtls_operation_t operation,
        const unsigned char *input, unsigned char *output )
{
    ((void) operation);
    return mbedtls_des_crypt_ecb( (mbedtls_des_context *) ctx, input, output );
}

static int des3_crypt_ecb_wrap( void *ctx, mbedtls_operation_t operation,
        const unsigned char *input, unsigned char *output )
{
    ((void) operation);
    return mbedtls_des3_crypt_ecb( (mbedtls_des3_context *) ctx, input, output );
}

#if defined(MBEDTLS_CIPHER_MODE_CBC)
static int des_crypt_cbc_wrap( void *ctx, mbedtls_operation_t operation, size_t length,
        unsigned char *iv, const unsigned char *input, unsigned char *output )
{
    return mbedtls_des_crypt_cbc( (mbedtls_des_context *) ctx, operation, length, iv, input,
                          output );
}
#endif /* MBEDTLS_CIPHER_MODE_CBC */

#if defined(MBEDTLS_CIPHER_MODE_CBC)
static int des3_crypt_cbc_wrap( void *ctx, mbedtls_operation_t operation, size_t length,
        unsigned char *iv, const unsigned char *input, unsigned char *output )
{
    return mbedtls_des3_crypt_cbc( (mbedtls_des3_context *) ctx, operation, length, iv, input,
                           output );
}
#endif /* MBEDTLS_CIPHER_MODE_CBC */

static int des_setkey_dec_wrap( void *ctx, const unsigned char *key,
                                unsigned int key_bitlen )
{
    ((void) key_bitlen);

    return mbedtls_des_setkey_dec( (mbedtls_des_context *) ctx, key );
}

static int des_setkey_enc_wrap( void *ctx, const unsigned char *key,
                                unsigned int key_bitlen )
{
    ((void) key_bitlen);

    return mbedtls_des_setkey_enc( (mbedtls_des_context *) ctx, key );
}

static int des3_set2key_dec_wrap( void *ctx, const unsigned char *key,
                                  unsigned int key_bitlen )
{
    ((void) key_bitlen);

    return mbedtls_des3_set2key_dec( (mbedtls_des3_context *) ctx, key );
}

static int des3_set2key_enc_wrap( void *ctx, const unsigned char *key,
                                  unsigned int key_bitlen )
{
    ((void) key_bitlen);

    return mbedtls_des3_set2key_enc( (mbedtls_des3_context *) ctx, key );
}

static int des3_set3key_dec_wrap( void *ctx, const unsigned char *key,
                                  unsigned int key_bitlen )
{
    ((void) key_bitlen);

    return mbedtls_des3_set3key_dec( (mbedtls_des3_context *) ctx, key );
}

static int des3_set3key_enc_wrap( void *ctx, const unsigned char *key,
                                  unsigned int key_bitlen )
{
    ((void) key_bitlen);

    return mbedtls_des3_set3key_enc( (mbedtls_des3_context *) ctx, key );
}

static void * des_ctx_alloc( void )
{
    mbedtls_des_context *des = mbedtls_calloc( 1, sizeof( mbedtls_des_context ) );

    if( des == NULL )
        return( NULL );

    mbedtls_des_init( des );

    return( des );
}

static void des_ctx_free( void *ctx )
{
    mbedtls_des_free( (mbedtls_des_context *) ctx );
    mbedtls_free( ctx );
}

static void * des3_ctx_alloc( void )
{
    mbedtls_des3_context *des3;
    des3 = mbedtls_calloc( 1, sizeof( mbedtls_des3_context ) );

    if( des3 == NULL )
        return( NULL );

    mbedtls_des3_init( des3 );

    return( des3 );
}

static void des3_ctx_free( void *ctx )
{
    mbedtls_des3_free( (mbedtls_des3_context *) ctx );
    mbedtls_free( ctx );
}

static const mbedtls_cipher_base_t des_info = {
    MBEDTLS_CIPHER_ID_DES,
    des_crypt_ecb_wrap,
#if defined(MBEDTLS_CIPHER_MODE_CBC)
    des_crypt_cbc_wrap,
#endif
#if defined(MBEDTLS_CIPHER_MODE_CFB)
    NULL,
#endif
#if defined(MBEDTLS_CIPHER_MODE_CTR)
    NULL,
#endif
#if defined(MBEDTLS_CIPHER_MODE_STREAM)
    NULL,
#endif
    des_setkey_enc_wrap,
    des_setkey_dec_wrap,
    des_ctx_alloc,
    des_ctx_free
};

static const mbedtls_cipher_info_t des_ecb_info = {
    MBEDTLS_CIPHER_DES_ECB,
    MBEDTLS_MODE_ECB,
    MBEDTLS_KEY_LENGTH_DES,
    "DES-ECB",
    8,
    0,
    8,
    &des_info
};

#if defined(MBEDTLS_CIPHER_MODE_CBC)
static const mbedtls_cipher_info_t des_cbc_info = {
    MBEDTLS_CIPHER_DES_CBC,
    MBEDTLS_MODE_CBC,
    MBEDTLS_KEY_LENGTH_DES,
    "DES-CBC",
    8,
    0,
    8,
    &des_info
};
#endif /* MBEDTLS_CIPHER_MODE_CBC */

static const mbedtls_cipher_base_t des_ede_info = {
    MBEDTLS_CIPHER_ID_DES,
    des3_crypt_ecb_wrap,
#if defined(MBEDTLS_CIPHER_MODE_CBC)
    des3_crypt_cbc_wrap,
#endif
#if defined(MBEDTLS_CIPHER_MODE_CFB)
    NULL,
#endif
#if defined(MBEDTLS_CIPHER_MODE_CTR)
    NULL,
#endif
#if defined(MBEDTLS_CIPHER_MODE_STREAM)
    NULL,
#endif
    des3_set2key_enc_wrap,
    des3_set2key_dec_wrap,
    des3_ctx_alloc,
    des3_ctx_free
};

static const mbedtls_cipher_info_t des_ede_ecb_info = {
    MBEDTLS_CIPHER_DES_EDE_ECB,
    MBEDTLS_MODE_ECB,
    MBEDTLS_KEY_LENGTH_DES_EDE,
    "DES-EDE-ECB",
    8,
    0,
    8,
    &des_ede_info
};

#if defined(MBEDTLS_CIPHER_MODE_CBC)
static const mbedtls_cipher_info_t des_ede_cbc_info = {
    MBEDTLS_CIPHER_DES_EDE_CBC,
    MBEDTLS_MODE_CBC,
    MBEDTLS_KEY_LENGTH_DES_EDE,
    "DES-EDE-CBC",
    8,
    0,
    8,
    &des_ede_info
};
#endif /* MBEDTLS_CIPHER_MODE_CBC */

static const mbedtls_cipher_base_t des_ede3_info = {
    MBEDTLS_CIPHER_ID_3DES,
    des3_crypt_ecb_wrap,
#if defined(MBEDTLS_CIPHER_MODE_CBC)
    des3_crypt_cbc_wrap,
#endif
#if defined(MBEDTLS_CIPHER_MODE_CFB)
    NULL,
#endif
#if defined(MBEDTLS_CIPHER_MODE_CTR)
    NULL,
#endif
#if defined(MBEDTLS_CIPHER_MODE_STREAM)
    NULL,
#endif
    des3_set3key_enc_wrap,
    des3_set3key_dec_wrap,
    des3_ctx_alloc,
    des3_ctx_free
};

static const mbedtls_cipher_info_t des_ede3_ecb_info = {
    MBEDTLS_CIPHER_DES_EDE3_ECB,
    MBEDTLS_MODE_ECB,
    MBEDTLS_KEY_LENGTH_DES_EDE3,
    "DES-EDE3-ECB",
    8,
    0,
    8,
    &des_ede3_info
};
#if defined(MBEDTLS_CIPHER_MODE_CBC)
static const mbedtls_cipher_info_t des_ede3_cbc_info = {
    MBEDTLS_CIPHER_DES_EDE3_CBC,
    MBEDTLS_MODE_CBC,
    MBEDTLS_KEY_LENGTH_DES_EDE3,
    "DES-EDE3-CBC",
    8,
    0,
    8,
    &des_ede3_info
};
#endif /* MBEDTLS_CIPHER_MODE_CBC */
#endif /* MBEDTLS_DES_C */

#if defined(MBEDTLS_BLOWFISH_C)

static int blowfish_crypt_ecb_wrap( void *ctx, mbedtls_operation_t operation,
        const unsigned char *input, unsigned char *output )
{
    return mbedtls_blowfish_crypt_ecb( (mbedtls_blowfish_context *) ctx, operation, input,
                               output );
}

#if defined(MBEDTLS_CIPHER_MODE_CBC)
static int blowfish_crypt_cbc_wrap( void *ctx, mbedtls_operation_t operation,
        size_t length, unsigned char *iv, const unsigned char *input,
        unsigned char *output )
{
    return mbedtls_blowfish_crypt_cbc( (mbedtls_blowfish_context *) ctx, operation, length, iv,
                               input, output );
}
#endif /* MBEDTLS_CIPHER_MODE_CBC */

#if defined(MBEDTLS_CIPHER_MODE_CFB)
static int blowfish_crypt_cfb64_wrap( void *ctx, mbedtls_operation_t operation,
        size_t length, size_t *iv_off, unsigned char *iv,
        const unsigned char *input, unsigned char *output )
{
    return mbedtls_blowfish_crypt_cfb64( (mbedtls_blowfish_context *) ctx, operation, length,
                                 iv_off, iv, input, output );
}
#endif /* MBEDTLS_CIPHER_MODE_CFB */

#if defined(MBEDTLS_CIPHER_MODE_CTR)
static int blowfish_crypt_ctr_wrap( void *ctx, size_t length, size_t *nc_off,
        unsigned char *nonce_counter, unsigned char *stream_block,
        const unsigned char *input, unsigned char *output )
{
    return mbedtls_blowfish_crypt_ctr( (mbedtls_blowfish_context *) ctx, length, nc_off,
                               nonce_counter, stream_block, input, output );
}
#endif /* MBEDTLS_CIPHER_MODE_CTR */

static int blowfish_setkey_wrap( void *ctx, const unsigned char *key,
                                 unsigned int key_bitlen )
{
    return mbedtls_blowfish_setkey( (mbedtls_blowfish_context *) ctx, key, key_bitlen );
}

static void * blowfish_ctx_alloc( void )
{
    mbedtls_blowfish_context *ctx;
    ctx = mbedtls_calloc( 1, sizeof( mbedtls_blowfish_context ) );

    if( ctx == NULL )
        return( NULL );

    mbedtls_blowfish_init( ctx );

    return( ctx );
}

static void blowfish_ctx_free( void *ctx )
{
    mbedtls_blowfish_free( (mbedtls_blowfish_context *) ctx );
    mbedtls_free( ctx );
}

static const mbedtls_cipher_base_t blowfish_info = {
    MBEDTLS_CIPHER_ID_BLOWFISH,
    blowfish_crypt_ecb_wrap,
#if defined(MBEDTLS_CIPHER_MODE_CBC)
    blowfish_crypt_cbc_wrap,
#endif
#if defined(MBEDTLS_CIPHER_MODE_CFB)
    blowfish_crypt_cfb64_wrap,
#endif
#if defined(MBEDTLS_CIPHER_MODE_CTR)
    blowfish_crypt_ctr_wrap,
#endif
#if defined(MBEDTLS_CIPHER_MODE_STREAM)
    NULL,
#endif
    blowfish_setkey_wrap,
    blowfish_setkey_wrap,
    blowfish_ctx_alloc,
    blowfish_ctx_free
};

static const mbedtls_cipher_info_t blowfish_ecb_info = {
    MBEDTLS_CIPHER_BLOWFISH_ECB,
    MBEDTLS_MODE_ECB,
    128,
    "BLOWFISH-ECB",
    8,
    MBEDTLS_CIPHER_VARIABLE_KEY_LEN,
    8,
    &blowfish_info
};

#if defined(MBEDTLS_CIPHER_MODE_CBC)
static const mbedtls_cipher_info_t blowfish_cbc_info = {
    MBEDTLS_CIPHER_BLOWFISH_CBC,
    MBEDTLS_MODE_CBC,
    128,
    "BLOWFISH-CBC",
    8,
    MBEDTLS_CIPHER_VARIABLE_KEY_LEN,
    8,
    &blowfish_info
};
#endif /* MBEDTLS_CIPHER_MODE_CBC */

#if defined(MBEDTLS_CIPHER_MODE_CFB)
static const mbedtls_cipher_info_t blowfish_cfb64_info = {
    MBEDTLS_CIPHER_BLOWFISH_CFB64,
    MBEDTLS_MODE_CFB,
    128,
    "BLOWFISH-CFB64",
    8,
    MBEDTLS_CIPHER_VARIABLE_KEY_LEN,
    8,
    &blowfish_info
};
#endif /* MBEDTLS_CIPHER_MODE_CFB */

#if defined(MBEDTLS_CIPHER_MODE_CTR)
static const mbedtls_cipher_info_t blowfish_ctr_info = {
    MBEDTLS_CIPHER_BLOWFISH_CTR,
    MBEDTLS_MODE_CTR,
    128,
    "BLOWFISH-CTR",
    8,
    MBEDTLS_CIPHER_VARIABLE_KEY_LEN,
    8,
    &blowfish_info
};
#endif /* MBEDTLS_CIPHER_MODE_CTR */
#endif /* MBEDTLS_BLOWFISH_C */

#if defined(MBEDTLS_ARC4_C)
static int arc4_crypt_stream_wrap( void *ctx, size_t length,
                                   const unsigned char *input,
                                   unsigned char *output )
{
    return( mbedtls_arc4_crypt( (mbedtls_arc4_context *) ctx, length, input, output ) );
}

static int arc4_setkey_wrap( void *ctx, const unsigned char *key,
                             unsigned int key_bitlen )
{
    /* we get key_bitlen in bits, arc4 expects it in bytes */
    if( key_bitlen % 8 != 0 )
        return( MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA );

    mbedtls_arc4_setup( (mbedtls_arc4_context *) ctx, key, key_bitlen / 8 );
    return( 0 );
}

static void * arc4_ctx_alloc( void )
{
    mbedtls_arc4_context *ctx;
    ctx = mbedtls_calloc( 1, sizeof( mbedtls_arc4_context ) );

    if( ctx == NULL )
        return( NULL );

    mbedtls_arc4_init( ctx );

    return( ctx );
}

static void arc4_ctx_free( void *ctx )
{
    mbedtls_arc4_free( (mbedtls_arc4_context *) ctx );
    mbedtls_free( ctx );
}

static const mbedtls_cipher_base_t arc4_base_info = {
    MBEDTLS_CIPHER_ID_ARC4,
    NULL,
#if defined(MBEDTLS_CIPHER_MODE_CBC)
    NULL,
#endif
#if defined(MBEDTLS_CIPHER_MODE_CFB)
    NULL,
#endif
#if defined(MBEDTLS_CIPHER_MODE_CTR)
    NULL,
#endif
#if defined(MBEDTLS_CIPHER_MODE_STREAM)
    arc4_crypt_stream_wrap,
#endif
    arc4_setkey_wrap,
    arc4_setkey_wrap,
    arc4_ctx_alloc,
    arc4_ctx_free
};

static const mbedtls_cipher_info_t arc4_128_info = {
    MBEDTLS_CIPHER_ARC4_128,
    MBEDTLS_MODE_STREAM,
    128,
    "ARC4-128",
    0,
    0,
    1,
    &arc4_base_info
};
#endif /* MBEDTLS_ARC4_C */

#if defined(MBEDTLS_CIPHER_NULL_CIPHER)
static int null_crypt_stream( void *ctx, size_t length,
                              const unsigned char *input,
                              unsigned char *output )
{
    ((void) ctx);
    memmove( output, input, length );
    return( 0 );
}

static int null_setkey( void *ctx, const unsigned char *key,
                        unsigned int key_bitlen )
{
    ((void) ctx);
    ((void) key);
    ((void) key_bitlen);

    return( 0 );
}

static void * null_ctx_alloc( void )
{
    return( (void *) 1 );
}

static void null_ctx_free( void *ctx )
{
    ((void) ctx);
}

static const mbedtls_cipher_base_t null_base_info = {
    MBEDTLS_CIPHER_ID_NULL,
    NULL,
#if defined(MBEDTLS_CIPHER_MODE_CBC)
    NULL,
#endif
#if defined(MBEDTLS_CIPHER_MODE_CFB)
    NULL,
#endif
#if defined(MBEDTLS_CIPHER_MODE_CTR)
    NULL,
#endif
#if defined(MBEDTLS_CIPHER_MODE_STREAM)
    null_crypt_stream,
#endif
    null_setkey,
    null_setkey,
    null_ctx_alloc,
    null_ctx_free
};

static const mbedtls_cipher_info_t null_cipher_info = {
    MBEDTLS_CIPHER_NULL,
    MBEDTLS_MODE_STREAM,
    0,
    "NULL",
    0,
    0,
    1,
    &null_base_info
};
#endif /* defined(MBEDTLS_CIPHER_NULL_CIPHER) */

const mbedtls_cipher_definition_t mbedtls_cipher_definitions[] =
{
#if defined(MBEDTLS_AES_C)
    { MBEDTLS_CIPHER_AES_128_ECB,          &aes_128_ecb_info },
    { MBEDTLS_CIPHER_AES_192_ECB,          &aes_192_ecb_info },
    { MBEDTLS_CIPHER_AES_256_ECB,          &aes_256_ecb_info },
#if defined(MBEDTLS_CIPHER_MODE_CBC)
    { MBEDTLS_CIPHER_AES_128_CBC,          &aes_128_cbc_info },
    { MBEDTLS_CIPHER_AES_192_CBC,          &aes_192_cbc_info },
    { MBEDTLS_CIPHER_AES_256_CBC,          &aes_256_cbc_info },
#endif
#if defined(MBEDTLS_CIPHER_MODE_CFB)
    { MBEDTLS_CIPHER_AES_128_CFB128,       &aes_128_cfb128_info },
    { MBEDTLS_CIPHER_AES_192_CFB128,       &aes_192_cfb128_info },
    { MBEDTLS_CIPHER_AES_256_CFB128,       &aes_256_cfb128_info },
#endif
#if defined(MBEDTLS_CIPHER_MODE_CTR)
    { MBEDTLS_CIPHER_AES_128_CTR,          &aes_128_ctr_info },
    { MBEDTLS_CIPHER_AES_192_CTR,          &aes_192_ctr_info },
    { MBEDTLS_CIPHER_AES_256_CTR,          &aes_256_ctr_info },
#endif
#if defined(MBEDTLS_GCM_C)
    { MBEDTLS_CIPHER_AES_128_GCM,          &aes_128_gcm_info },
    { MBEDTLS_CIPHER_AES_192_GCM,          &aes_192_gcm_info },
    { MBEDTLS_CIPHER_AES_256_GCM,          &aes_256_gcm_info },
#endif
#if defined(MBEDTLS_CCM_C)
    { MBEDTLS_CIPHER_AES_128_CCM,          &aes_128_ccm_info },
    { MBEDTLS_CIPHER_AES_192_CCM,          &aes_192_ccm_info },
    { MBEDTLS_CIPHER_AES_256_CCM,          &aes_256_ccm_info },
#endif
#endif /* MBEDTLS_AES_C */

#if defined(MBEDTLS_ARC4_C)
    { MBEDTLS_CIPHER_ARC4_128,             &arc4_128_info },
#endif

#if defined(MBEDTLS_BLOWFISH_C)
    { MBEDTLS_CIPHER_BLOWFISH_ECB,         &blowfish_ecb_info },
#if defined(MBEDTLS_CIPHER_MODE_CBC)
    { MBEDTLS_CIPHER_BLOWFISH_CBC,         &blowfish_cbc_info },
#endif
#if defined(MBEDTLS_CIPHER_MODE_CFB)
    { MBEDTLS_CIPHER_BLOWFISH_CFB64,       &blowfish_cfb64_info },
#endif
#if defined(MBEDTLS_CIPHER_MODE_CTR)
    { MBEDTLS_CIPHER_BLOWFISH_CTR,         &blowfish_ctr_info },
#endif
#endif /* MBEDTLS_BLOWFISH_C */

#if defined(MBEDTLS_CAMELLIA_C)
    { MBEDTLS_CIPHER_CAMELLIA_128_ECB,     &camellia_128_ecb_info },
    { MBEDTLS_CIPHER_CAMELLIA_192_ECB,     &camellia_192_ecb_info },
    { MBEDTLS_CIPHER_CAMELLIA_256_ECB,     &camellia_256_ecb_info },
#if defined(MBEDTLS_CIPHER_MODE_CBC)
    { MBEDTLS_CIPHER_CAMELLIA_128_CBC,     &camellia_128_cbc_info },
    { MBEDTLS_CIPHER_CAMELLIA_192_CBC,     &camellia_192_cbc_info },
    { MBEDTLS_CIPHER_CAMELLIA_256_CBC,     &camellia_256_cbc_info },
#endif
#if defined(MBEDTLS_CIPHER_MODE_CFB)
    { MBEDTLS_CIPHER_CAMELLIA_128_CFB128,  &camellia_128_cfb128_info },
    { MBEDTLS_CIPHER_CAMELLIA_192_CFB128,  &camellia_192_cfb128_info },
    { MBEDTLS_CIPHER_CAMELLIA_256_CFB128,  &camellia_256_cfb128_info },
#endif
#if defined(MBEDTLS_CIPHER_MODE_CTR)
    { MBEDTLS_CIPHER_CAMELLIA_128_CTR,     &camellia_128_ctr_info },
    { MBEDTLS_CIPHER_CAMELLIA_192_CTR,     &camellia_192_ctr_info },
    { MBEDTLS_CIPHER_CAMELLIA_256_CTR,     &camellia_256_ctr_info },
#endif
#if defined(MBEDTLS_GCM_C)
    { MBEDTLS_CIPHER_CAMELLIA_128_GCM,     &camellia_128_gcm_info },
    { MBEDTLS_CIPHER_CAMELLIA_192_GCM,     &camellia_192_gcm_info },
    { MBEDTLS_CIPHER_CAMELLIA_256_GCM,     &camellia_256_gcm_info },
#endif
#if defined(MBEDTLS_CCM_C)
    { MBEDTLS_CIPHER_CAMELLIA_128_CCM,     &camellia_128_ccm_info },
    { MBEDTLS_CIPHER_CAMELLIA_192_CCM,     &camellia_192_ccm_info },
    { MBEDTLS_CIPHER_CAMELLIA_256_CCM,     &camellia_256_ccm_info },
#endif
#endif /* MBEDTLS_CAMELLIA_C */

#if defined(MBEDTLS_ARIA_C)
    { MBEDTLS_CIPHER_ARIA_128_ECB,     &aria_128_ecb_info },
    { MBEDTLS_CIPHER_ARIA_192_ECB,     &aria_192_ecb_info },
    { MBEDTLS_CIPHER_ARIA_256_ECB,     &aria_256_ecb_info },
#if defined(MBEDTLS_CIPHER_MODE_CBC)
    { MBEDTLS_CIPHER_ARIA_128_CBC,     &aria_128_cbc_info },
    { MBEDTLS_CIPHER_ARIA_192_CBC,     &aria_192_cbc_info },
    { MBEDTLS_CIPHER_ARIA_256_CBC,     &aria_256_cbc_info },
#endif
#if defined(MBEDTLS_CIPHER_MODE_CFB)
    { MBEDTLS_CIPHER_ARIA_128_CFB128,  &aria_128_cfb128_info },
    { MBEDTLS_CIPHER_ARIA_192_CFB128,  &aria_192_cfb128_info },
    { MBEDTLS_CIPHER_ARIA_256_CFB128,  &aria_256_cfb128_info },
#endif
#if defined(MBEDTLS_CIPHER_MODE_CTR)
    { MBEDTLS_CIPHER_ARIA_128_CTR,     &aria_128_ctr_info },
    { MBEDTLS_CIPHER_ARIA_192_CTR,     &aria_192_ctr_info },
    { MBEDTLS_CIPHER_ARIA_256_CTR,     &aria_256_ctr_info },
#endif
#if defined(MBEDTLS_GCM_C)
    { MBEDTLS_CIPHER_ARIA_128_GCM,     &aria_128_gcm_info },
    { MBEDTLS_CIPHER_ARIA_192_GCM,     &aria_192_gcm_info },
    { MBEDTLS_CIPHER_ARIA_256_GCM,     &aria_256_gcm_info },
#endif
#if defined(MBEDTLS_CCM_C)
    { MBEDTLS_CIPHER_ARIA_128_CCM,     &aria_128_ccm_info },
    { MBEDTLS_CIPHER_ARIA_192_CCM,     &aria_192_ccm_info },
    { MBEDTLS_CIPHER_ARIA_256_CCM,     &aria_256_ccm_info },
#endif
#endif /* MBEDTLS_ARIA_C */

#if defined(MBEDTLS_DES_C)
    { MBEDTLS_CIPHER_DES_ECB,              &des_ecb_info },
    { MBEDTLS_CIPHER_DES_EDE_ECB,          &des_ede_ecb_info },
    { MBEDTLS_CIPHER_DES_EDE3_ECB,         &des_ede3_ecb_info },
#if defined(MBEDTLS_CIPHER_MODE_CBC)
    { MBEDTLS_CIPHER_DES_CBC,              &des_cbc_info },
    { MBEDTLS_CIPHER_DES_EDE_CBC,          &des_ede_cbc_info },
    { MBEDTLS_CIPHER_DES_EDE3_CBC,         &des_ede3_cbc_info },
#endif
#endif /* MBEDTLS_DES_C */

#if defined(MBEDTLS_CIPHER_NULL_CIPHER)
    { MBEDTLS_CIPHER_NULL,                 &null_cipher_info },
#endif /* MBEDTLS_CIPHER_NULL_CIPHER */

    { MBEDTLS_CIPHER_NONE, NULL }
};

#define NUM_CIPHERS sizeof mbedtls_cipher_definitions / sizeof mbedtls_cipher_definitions[0]
int mbedtls_cipher_supported[NUM_CIPHERS];

#endif /* MBEDTLS_CIPHER_C */
