/*
 *  X.509 Certificate Signing Request writing
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
/*
 * References:
 * - CSRs: PKCS#10 v1.7 aka RFC 2986
 * - attributes: PKCS#9 v2.0 aka RFC 2985
 */

#if !defined(MBEDTLS_CONFIG_FILE)
#include "mbedtls/config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#if defined(MBEDTLS_X509_CSR_WRITE_C)

#include "mbedtls/x509_csr.h"
#include "mbedtls/oid.h"
#include "mbedtls/asn1write.h"

#include <string.h>
#include <stdlib.h>

#if defined(MBEDTLS_PEM_WRITE_C)
#include "mbedtls/pem.h"
#endif

/* Implementation that should never be optimized out by the compiler */
static void mbedtls_zeroize( void *v, size_t n ) {
    volatile unsigned char *p = v; while( n-- ) *p++ = 0;
}

void mbedtls_x509write_csr_init( mbedtls_x509write_csr *ctx )
{
    memset( ctx, 0, sizeof( mbedtls_x509write_csr ) );
}

void mbedtls_x509write_csr_free( mbedtls_x509write_csr *ctx )
{
    mbedtls_asn1_free_named_data_list( &ctx->subject );
    mbedtls_asn1_free_named_data_list( &ctx->extensions );

    mbedtls_zeroize( ctx, sizeof( mbedtls_x509write_csr ) );
}

void mbedtls_x509write_csr_set_md_alg( mbedtls_x509write_csr *ctx, mbedtls_md_type_t md_alg )
{
    ctx->md_alg = md_alg;
}

void mbedtls_x509write_csr_set_key( mbedtls_x509write_csr *ctx, mbedtls_pk_context *key )
{
    ctx->key = key;
}

int mbedtls_x509write_csr_set_subject_name( mbedtls_x509write_csr *ctx,
                                    const char *subject_name )
{
    return mbedtls_x509_string_to_names( &ctx->subject, subject_name );
}

int mbedtls_x509write_csr_set_extension( mbedtls_x509write_csr *ctx,
                                 const char *oid, size_t oid_len,
                                 const unsigned char *val, size_t val_len )
{
    return mbedtls_x509_set_extension( &ctx->extensions, oid, oid_len,
                               0, val, val_len );
}

int mbedtls_x509write_csr_set_key_usage( mbedtls_x509write_csr *ctx, unsigned char key_usage )
{
    unsigned char buf[4];
    unsigned char *c;
    int ret;

    c = buf + 4;

    if( ( ret = mbedtls_asn1_write_bitstring( &c, buf, &key_usage, 7 ) ) != 4 )
        return( ret );

    ret = mbedtls_x509write_csr_set_extension( ctx, MBEDTLS_OID_KEY_USAGE,
                                       MBEDTLS_OID_SIZE( MBEDTLS_OID_KEY_USAGE ),
                                       buf, 4 );
    if( ret != 0 )
        return( ret );

    return( 0 );
}

int mbedtls_x509write_csr_set_ns_cert_type( mbedtls_x509write_csr *ctx,
                                    unsigned char ns_cert_type )
{
    unsigned char buf[4];
    unsigned char *c;
    int ret;

    c = buf + 4;

    if( ( ret = mbedtls_asn1_write_bitstring( &c, buf, &ns_cert_type, 8 ) ) != 4 )
        return( ret );

    ret = mbedtls_x509write_csr_set_extension( ctx, MBEDTLS_OID_NS_CERT_TYPE,
                                       MBEDTLS_OID_SIZE( MBEDTLS_OID_NS_CERT_TYPE ),
                                       buf, 4 );
    if( ret != 0 )
        return( ret );

    return( 0 );
}

int mbedtls_x509write_csr_der( mbedtls_x509write_csr *ctx, unsigned char *buf, size_t size,
                       int (*f_rng)(void *, unsigned char *, size_t),
                       void *p_rng )
{
    int ret;
    const char *sig_oid;
    size_t sig_oid_len = 0;
    unsigned char *c, *c2;
    unsigned char hash[64];
    unsigned char sig[MBEDTLS_MPI_MAX_SIZE];
    unsigned char tmp_buf[2048];
    size_t pub_len = 0, sig_and_oid_len = 0, sig_len;
    size_t len = 0;
    mbedtls_pk_type_t pk_alg;

    /*
     * Prepare data to be signed in tmp_buf
     */
    c = tmp_buf + sizeof( tmp_buf );

    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_x509_write_extensions( &c, tmp_buf, ctx->extensions ) );

    if( len )
    {
        MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( &c, tmp_buf, len ) );
        MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( &c, tmp_buf, MBEDTLS_ASN1_CONSTRUCTED |
                                                        MBEDTLS_ASN1_SEQUENCE ) );

        MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( &c, tmp_buf, len ) );
        MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( &c, tmp_buf, MBEDTLS_ASN1_CONSTRUCTED |
                                                        MBEDTLS_ASN1_SET ) );

        MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_oid( &c, tmp_buf, MBEDTLS_OID_PKCS9_CSR_EXT_REQ,
                                          MBEDTLS_OID_SIZE( MBEDTLS_OID_PKCS9_CSR_EXT_REQ ) ) );

        MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( &c, tmp_buf, len ) );
        MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( &c, tmp_buf, MBEDTLS_ASN1_CONSTRUCTED |
                                                        MBEDTLS_ASN1_SEQUENCE ) );
    }

    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( &c, tmp_buf, len ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( &c, tmp_buf, MBEDTLS_ASN1_CONSTRUCTED |
                                                    MBEDTLS_ASN1_CONTEXT_SPECIFIC ) );

    MBEDTLS_ASN1_CHK_ADD( pub_len, mbedtls_pk_write_pubkey_der( ctx->key,
                                                tmp_buf, c - tmp_buf ) );
    c -= pub_len;
    len += pub_len;

    /*
     *  Subject  ::=  Name
     */
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_x509_write_names( &c, tmp_buf, ctx->subject ) );

    /*
     *  Version  ::=  INTEGER  {  v1(0), v2(1), v3(2)  }
     */
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_int( &c, tmp_buf, 0 ) );

    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( &c, tmp_buf, len ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( &c, tmp_buf, MBEDTLS_ASN1_CONSTRUCTED |
                                                    MBEDTLS_ASN1_SEQUENCE ) );

    /*
     * Prepare signature
     */
    mbedtls_md( mbedtls_md_info_from_type( ctx->md_alg ), c, len, hash );

    if( ( ret = mbedtls_pk_sign( ctx->key, ctx->md_alg, hash, 0, sig, &sig_len,
                                 f_rng, p_rng ) ) != 0 )
    {
        return( ret );
    }

    if( mbedtls_pk_can_do( ctx->key, MBEDTLS_PK_RSA ) )
        pk_alg = MBEDTLS_PK_RSA;
    else if( mbedtls_pk_can_do( ctx->key, MBEDTLS_PK_ECDSA ) )
        pk_alg = MBEDTLS_PK_ECDSA;
    else
        return( MBEDTLS_ERR_X509_INVALID_ALG );

    if( ( ret = mbedtls_oid_get_oid_by_sig_alg( pk_alg, ctx->md_alg,
                                                &sig_oid, &sig_oid_len ) ) != 0 )
    {
        return( ret );
    }

    /*
     * Write data to output buffer
     */
    c2 = buf + size;
    MBEDTLS_ASN1_CHK_ADD( sig_and_oid_len, mbedtls_x509_write_sig( &c2, buf,
                                        sig_oid, sig_oid_len, sig, sig_len ) );

    if( len > (size_t)( c2 - buf ) )
        return( MBEDTLS_ERR_ASN1_BUF_TOO_SMALL );

    c2 -= len;
    memcpy( c2, c, len );

    len += sig_and_oid_len;
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( &c2, buf, len ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( &c2, buf, MBEDTLS_ASN1_CONSTRUCTED |
                                                 MBEDTLS_ASN1_SEQUENCE ) );

    return( (int) len );
}

#define PEM_BEGIN_CSR           "-----BEGIN CERTIFICATE REQUEST-----\n"
#define PEM_END_CSR             "-----END CERTIFICATE REQUEST-----\n"

#if defined(MBEDTLS_PEM_WRITE_C)
int mbedtls_x509write_csr_pem( mbedtls_x509write_csr *ctx, unsigned char *buf, size_t size,
                       int (*f_rng)(void *, unsigned char *, size_t),
                       void *p_rng )
{
    int ret;
    unsigned char output_buf[4096];
    size_t olen = 0;

    if( ( ret = mbedtls_x509write_csr_der( ctx, output_buf, sizeof(output_buf),
                                   f_rng, p_rng ) ) < 0 )
    {
        return( ret );
    }

    if( ( ret = mbedtls_pem_write_buffer( PEM_BEGIN_CSR, PEM_END_CSR,
                                  output_buf + sizeof(output_buf) - ret,
                                  ret, buf, size, &olen ) ) != 0 )
    {
        return( ret );
    }

    return( 0 );
}
#endif /* MBEDTLS_PEM_WRITE_C */

#endif /* MBEDTLS_X509_CSR_WRITE_C */
