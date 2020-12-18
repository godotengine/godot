/**
 * \file pkcs5.c
 *
 * \brief PKCS#5 functions
 *
 * \author Mathias Olsson <mathias@kompetensum.com>
 *
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 *
 *  This file is provided under the Apache License 2.0, or the
 *  GNU General Public License v2.0 or later.
 *
 *  **********
 *  Apache License 2.0:
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
 *  **********
 *
 *  **********
 *  GNU General Public License v2.0 or later:
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 *  **********
 */
/*
 * PKCS#5 includes PBKDF2 and more
 *
 * http://tools.ietf.org/html/rfc2898 (Specification)
 * http://tools.ietf.org/html/rfc6070 (Test vectors)
 */

#if !defined(MBEDTLS_CONFIG_FILE)
#include "mbedtls/config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#if defined(MBEDTLS_PKCS5_C)

#include "mbedtls/pkcs5.h"

#if defined(MBEDTLS_ASN1_PARSE_C)
#include "mbedtls/asn1.h"
#include "mbedtls/cipher.h"
#include "mbedtls/oid.h"
#endif /* MBEDTLS_ASN1_PARSE_C */

#include <string.h>

#if defined(MBEDTLS_PLATFORM_C)
#include "mbedtls/platform.h"
#else
#include <stdio.h>
#define mbedtls_printf printf
#endif

#if defined(MBEDTLS_ASN1_PARSE_C)
static int pkcs5_parse_pbkdf2_params( const mbedtls_asn1_buf *params,
                                      mbedtls_asn1_buf *salt, int *iterations,
                                      int *keylen, mbedtls_md_type_t *md_type )
{
    int ret;
    mbedtls_asn1_buf prf_alg_oid;
    unsigned char *p = params->p;
    const unsigned char *end = params->p + params->len;

    if( params->tag != ( MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE ) )
        return( MBEDTLS_ERR_PKCS5_INVALID_FORMAT +
                MBEDTLS_ERR_ASN1_UNEXPECTED_TAG );
    /*
     *  PBKDF2-params ::= SEQUENCE {
     *    salt              OCTET STRING,
     *    iterationCount    INTEGER,
     *    keyLength         INTEGER OPTIONAL
     *    prf               AlgorithmIdentifier DEFAULT algid-hmacWithSHA1
     *  }
     *
     */
    if( ( ret = mbedtls_asn1_get_tag( &p, end, &salt->len, MBEDTLS_ASN1_OCTET_STRING ) ) != 0 )
        return( MBEDTLS_ERR_PKCS5_INVALID_FORMAT + ret );

    salt->p = p;
    p += salt->len;

    if( ( ret = mbedtls_asn1_get_int( &p, end, iterations ) ) != 0 )
        return( MBEDTLS_ERR_PKCS5_INVALID_FORMAT + ret );

    if( p == end )
        return( 0 );

    if( ( ret = mbedtls_asn1_get_int( &p, end, keylen ) ) != 0 )
    {
        if( ret != MBEDTLS_ERR_ASN1_UNEXPECTED_TAG )
            return( MBEDTLS_ERR_PKCS5_INVALID_FORMAT + ret );
    }

    if( p == end )
        return( 0 );

    if( ( ret = mbedtls_asn1_get_alg_null( &p, end, &prf_alg_oid ) ) != 0 )
        return( MBEDTLS_ERR_PKCS5_INVALID_FORMAT + ret );

    if( mbedtls_oid_get_md_hmac( &prf_alg_oid, md_type ) != 0 )
        return( MBEDTLS_ERR_PKCS5_FEATURE_UNAVAILABLE );

    if( p != end )
        return( MBEDTLS_ERR_PKCS5_INVALID_FORMAT +
                MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );

    return( 0 );
}

int mbedtls_pkcs5_pbes2( const mbedtls_asn1_buf *pbe_params, int mode,
                 const unsigned char *pwd,  size_t pwdlen,
                 const unsigned char *data, size_t datalen,
                 unsigned char *output )
{
    int ret, iterations = 0, keylen = 0;
    unsigned char *p, *end;
    mbedtls_asn1_buf kdf_alg_oid, enc_scheme_oid, kdf_alg_params, enc_scheme_params;
    mbedtls_asn1_buf salt;
    mbedtls_md_type_t md_type = MBEDTLS_MD_SHA1;
    unsigned char key[32], iv[32];
    size_t olen = 0;
    const mbedtls_md_info_t *md_info;
    const mbedtls_cipher_info_t *cipher_info;
    mbedtls_md_context_t md_ctx;
    mbedtls_cipher_type_t cipher_alg;
    mbedtls_cipher_context_t cipher_ctx;

    p = pbe_params->p;
    end = p + pbe_params->len;

    /*
     *  PBES2-params ::= SEQUENCE {
     *    keyDerivationFunc AlgorithmIdentifier {{PBES2-KDFs}},
     *    encryptionScheme AlgorithmIdentifier {{PBES2-Encs}}
     *  }
     */
    if( pbe_params->tag != ( MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE ) )
        return( MBEDTLS_ERR_PKCS5_INVALID_FORMAT +
                MBEDTLS_ERR_ASN1_UNEXPECTED_TAG );

    if( ( ret = mbedtls_asn1_get_alg( &p, end, &kdf_alg_oid, &kdf_alg_params ) ) != 0 )
        return( MBEDTLS_ERR_PKCS5_INVALID_FORMAT + ret );

    // Only PBKDF2 supported at the moment
    //
    if( MBEDTLS_OID_CMP( MBEDTLS_OID_PKCS5_PBKDF2, &kdf_alg_oid ) != 0 )
        return( MBEDTLS_ERR_PKCS5_FEATURE_UNAVAILABLE );

    if( ( ret = pkcs5_parse_pbkdf2_params( &kdf_alg_params,
                                           &salt, &iterations, &keylen,
                                           &md_type ) ) != 0 )
    {
        return( ret );
    }

    md_info = mbedtls_md_info_from_type( md_type );
    if( md_info == NULL )
        return( MBEDTLS_ERR_PKCS5_FEATURE_UNAVAILABLE );

    if( ( ret = mbedtls_asn1_get_alg( &p, end, &enc_scheme_oid,
                              &enc_scheme_params ) ) != 0 )
    {
        return( MBEDTLS_ERR_PKCS5_INVALID_FORMAT + ret );
    }

    if( mbedtls_oid_get_cipher_alg( &enc_scheme_oid, &cipher_alg ) != 0 )
        return( MBEDTLS_ERR_PKCS5_FEATURE_UNAVAILABLE );

    cipher_info = mbedtls_cipher_info_from_type( cipher_alg );
    if( cipher_info == NULL )
        return( MBEDTLS_ERR_PKCS5_FEATURE_UNAVAILABLE );

    /*
     * The value of keylen from pkcs5_parse_pbkdf2_params() is ignored
     * since it is optional and we don't know if it was set or not
     */
    keylen = cipher_info->key_bitlen / 8;

    if( enc_scheme_params.tag != MBEDTLS_ASN1_OCTET_STRING ||
        enc_scheme_params.len != cipher_info->iv_size )
    {
        return( MBEDTLS_ERR_PKCS5_INVALID_FORMAT );
    }

    mbedtls_md_init( &md_ctx );
    mbedtls_cipher_init( &cipher_ctx );

    memcpy( iv, enc_scheme_params.p, enc_scheme_params.len );

    if( ( ret = mbedtls_md_setup( &md_ctx, md_info, 1 ) ) != 0 )
        goto exit;

    if( ( ret = mbedtls_pkcs5_pbkdf2_hmac( &md_ctx, pwd, pwdlen, salt.p, salt.len,
                                   iterations, keylen, key ) ) != 0 )
    {
        goto exit;
    }

    if( ( ret = mbedtls_cipher_setup( &cipher_ctx, cipher_info ) ) != 0 )
        goto exit;

    if( ( ret = mbedtls_cipher_setkey( &cipher_ctx, key, 8 * keylen, (mbedtls_operation_t) mode ) ) != 0 )
        goto exit;

    if( ( ret = mbedtls_cipher_crypt( &cipher_ctx, iv, enc_scheme_params.len,
                              data, datalen, output, &olen ) ) != 0 )
        ret = MBEDTLS_ERR_PKCS5_PASSWORD_MISMATCH;

exit:
    mbedtls_md_free( &md_ctx );
    mbedtls_cipher_free( &cipher_ctx );

    return( ret );
}
#endif /* MBEDTLS_ASN1_PARSE_C */

int mbedtls_pkcs5_pbkdf2_hmac( mbedtls_md_context_t *ctx, const unsigned char *password,
                       size_t plen, const unsigned char *salt, size_t slen,
                       unsigned int iteration_count,
                       uint32_t key_length, unsigned char *output )
{
    int ret = 0, j;
    unsigned int i;
    unsigned char md1[MBEDTLS_MD_MAX_SIZE];
    unsigned char work[MBEDTLS_MD_MAX_SIZE];
    unsigned char md_size = mbedtls_md_get_size( ctx->md_info );
    size_t use_len;
    unsigned char *out_p = output;
    unsigned char counter[4];

    memset( counter, 0, 4 );
    counter[3] = 1;

#if UINT_MAX > 0xFFFFFFFF
    if( iteration_count > 0xFFFFFFFF )
        return( MBEDTLS_ERR_PKCS5_BAD_INPUT_DATA );
#endif

    while( key_length )
    {
        // U1 ends up in work
        //
        if( ( ret = mbedtls_md_hmac_starts( ctx, password, plen ) ) != 0 )
            goto cleanup;

        if( ( ret = mbedtls_md_hmac_update( ctx, salt, slen ) ) != 0 )
            goto cleanup;

        if( ( ret = mbedtls_md_hmac_update( ctx, counter, 4 ) ) != 0 )
            goto cleanup;

        if( ( ret = mbedtls_md_hmac_finish( ctx, work ) ) != 0 )
            goto cleanup;

        memcpy( md1, work, md_size );

        for( i = 1; i < iteration_count; i++ )
        {
            // U2 ends up in md1
            //
            if( ( ret = mbedtls_md_hmac_starts( ctx, password, plen ) ) != 0 )
                goto cleanup;

            if( ( ret = mbedtls_md_hmac_update( ctx, md1, md_size ) ) != 0 )
                goto cleanup;

            if( ( ret = mbedtls_md_hmac_finish( ctx, md1 ) ) != 0 )
                goto cleanup;

            // U1 xor U2
            //
            for( j = 0; j < md_size; j++ )
                work[j] ^= md1[j];
        }

        use_len = ( key_length < md_size ) ? key_length : md_size;
        memcpy( out_p, work, use_len );

        key_length -= (uint32_t) use_len;
        out_p += use_len;

        for( i = 4; i > 0; i-- )
            if( ++counter[i - 1] != 0 )
                break;
    }

cleanup:
    /* Zeroise buffers to clear sensitive data from memory. */
    mbedtls_platform_zeroize( work, MBEDTLS_MD_MAX_SIZE );
    mbedtls_platform_zeroize( md1, MBEDTLS_MD_MAX_SIZE );

    return( ret );
}

#if defined(MBEDTLS_SELF_TEST)

#if !defined(MBEDTLS_SHA1_C)
int mbedtls_pkcs5_self_test( int verbose )
{
    if( verbose != 0 )
        mbedtls_printf( "  PBKDF2 (SHA1): skipped\n\n" );

    return( 0 );
}
#else

#define MAX_TESTS   6

static const size_t plen[MAX_TESTS] =
    { 8, 8, 8, 24, 9 };

static const unsigned char password[MAX_TESTS][32] =
{
    "password",
    "password",
    "password",
    "passwordPASSWORDpassword",
    "pass\0word",
};

static const size_t slen[MAX_TESTS] =
    { 4, 4, 4, 36, 5 };

static const unsigned char salt[MAX_TESTS][40] =
{
    "salt",
    "salt",
    "salt",
    "saltSALTsaltSALTsaltSALTsaltSALTsalt",
    "sa\0lt",
};

static const uint32_t it_cnt[MAX_TESTS] =
    { 1, 2, 4096, 4096, 4096 };

static const uint32_t key_len[MAX_TESTS] =
    { 20, 20, 20, 25, 16 };

static const unsigned char result_key[MAX_TESTS][32] =
{
    { 0x0c, 0x60, 0xc8, 0x0f, 0x96, 0x1f, 0x0e, 0x71,
      0xf3, 0xa9, 0xb5, 0x24, 0xaf, 0x60, 0x12, 0x06,
      0x2f, 0xe0, 0x37, 0xa6 },
    { 0xea, 0x6c, 0x01, 0x4d, 0xc7, 0x2d, 0x6f, 0x8c,
      0xcd, 0x1e, 0xd9, 0x2a, 0xce, 0x1d, 0x41, 0xf0,
      0xd8, 0xde, 0x89, 0x57 },
    { 0x4b, 0x00, 0x79, 0x01, 0xb7, 0x65, 0x48, 0x9a,
      0xbe, 0xad, 0x49, 0xd9, 0x26, 0xf7, 0x21, 0xd0,
      0x65, 0xa4, 0x29, 0xc1 },
    { 0x3d, 0x2e, 0xec, 0x4f, 0xe4, 0x1c, 0x84, 0x9b,
      0x80, 0xc8, 0xd8, 0x36, 0x62, 0xc0, 0xe4, 0x4a,
      0x8b, 0x29, 0x1a, 0x96, 0x4c, 0xf2, 0xf0, 0x70,
      0x38 },
    { 0x56, 0xfa, 0x6a, 0xa7, 0x55, 0x48, 0x09, 0x9d,
      0xcc, 0x37, 0xd7, 0xf0, 0x34, 0x25, 0xe0, 0xc3 },
};

int mbedtls_pkcs5_self_test( int verbose )
{
    mbedtls_md_context_t sha1_ctx;
    const mbedtls_md_info_t *info_sha1;
    int ret, i;
    unsigned char key[64];

    mbedtls_md_init( &sha1_ctx );

    info_sha1 = mbedtls_md_info_from_type( MBEDTLS_MD_SHA1 );
    if( info_sha1 == NULL )
    {
        ret = 1;
        goto exit;
    }

    if( ( ret = mbedtls_md_setup( &sha1_ctx, info_sha1, 1 ) ) != 0 )
    {
        ret = 1;
        goto exit;
    }

    for( i = 0; i < MAX_TESTS; i++ )
    {
        if( verbose != 0 )
            mbedtls_printf( "  PBKDF2 (SHA1) #%d: ", i );

        ret = mbedtls_pkcs5_pbkdf2_hmac( &sha1_ctx, password[i], plen[i], salt[i],
                                  slen[i], it_cnt[i], key_len[i], key );
        if( ret != 0 ||
            memcmp( result_key[i], key, key_len[i] ) != 0 )
        {
            if( verbose != 0 )
                mbedtls_printf( "failed\n" );

            ret = 1;
            goto exit;
        }

        if( verbose != 0 )
            mbedtls_printf( "passed\n" );
    }

    if( verbose != 0 )
        mbedtls_printf( "\n" );

exit:
    mbedtls_md_free( &sha1_ctx );

    return( ret );
}
#endif /* MBEDTLS_SHA1_C */

#endif /* MBEDTLS_SELF_TEST */

#endif /* MBEDTLS_PKCS5_C */
