/**
 * \file pkcs11.c
 *
 * \brief Wrapper for PKCS#11 library libpkcs11-helper
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

#include "mbedtls/pkcs11.h"

#if defined(MBEDTLS_PKCS11_C)

#include "mbedtls/md.h"
#include "mbedtls/oid.h"
#include "mbedtls/x509_crt.h"

#if defined(MBEDTLS_PLATFORM_C)
#include "mbedtls/platform.h"
#else
#include <stdlib.h>
#define mbedtls_calloc    calloc
#define mbedtls_free       free
#endif

#include <string.h>

void mbedtls_pkcs11_init( mbedtls_pkcs11_context *ctx )
{
    memset( ctx, 0, sizeof( mbedtls_pkcs11_context ) );
}

int mbedtls_pkcs11_x509_cert_bind( mbedtls_x509_crt *cert, pkcs11h_certificate_t pkcs11_cert )
{
    int ret = 1;
    unsigned char *cert_blob = NULL;
    size_t cert_blob_size = 0;

    if( cert == NULL )
    {
        ret = 2;
        goto cleanup;
    }

    if( pkcs11h_certificate_getCertificateBlob( pkcs11_cert, NULL,
                                                &cert_blob_size ) != CKR_OK )
    {
        ret = 3;
        goto cleanup;
    }

    cert_blob = mbedtls_calloc( 1, cert_blob_size );
    if( NULL == cert_blob )
    {
        ret = 4;
        goto cleanup;
    }

    if( pkcs11h_certificate_getCertificateBlob( pkcs11_cert, cert_blob,
                                                &cert_blob_size ) != CKR_OK )
    {
        ret = 5;
        goto cleanup;
    }

    if( 0 != mbedtls_x509_crt_parse( cert, cert_blob, cert_blob_size ) )
    {
        ret = 6;
        goto cleanup;
    }

    ret = 0;

cleanup:
    if( NULL != cert_blob )
        mbedtls_free( cert_blob );

    return( ret );
}


int mbedtls_pkcs11_priv_key_bind( mbedtls_pkcs11_context *priv_key,
        pkcs11h_certificate_t pkcs11_cert )
{
    int ret = 1;
    mbedtls_x509_crt cert;

    mbedtls_x509_crt_init( &cert );

    if( priv_key == NULL )
        goto cleanup;

    if( 0 != mbedtls_pkcs11_x509_cert_bind( &cert, pkcs11_cert ) )
        goto cleanup;

    priv_key->len = mbedtls_pk_get_len( &cert.pk );
    priv_key->pkcs11h_cert = pkcs11_cert;

    ret = 0;

cleanup:
    mbedtls_x509_crt_free( &cert );

    return( ret );
}

void mbedtls_pkcs11_priv_key_free( mbedtls_pkcs11_context *priv_key )
{
    if( NULL != priv_key )
        pkcs11h_certificate_freeCertificate( priv_key->pkcs11h_cert );
}

int mbedtls_pkcs11_decrypt( mbedtls_pkcs11_context *ctx,
                       int mode, size_t *olen,
                       const unsigned char *input,
                       unsigned char *output,
                       size_t output_max_len )
{
    size_t input_len, output_len;

    if( NULL == ctx )
        return( MBEDTLS_ERR_RSA_BAD_INPUT_DATA );

    if( MBEDTLS_RSA_PRIVATE != mode )
        return( MBEDTLS_ERR_RSA_BAD_INPUT_DATA );

    output_len = input_len = ctx->len;

    if( input_len < 16 || input_len > output_max_len )
        return( MBEDTLS_ERR_RSA_BAD_INPUT_DATA );

    /* Determine size of output buffer */
    if( pkcs11h_certificate_decryptAny( ctx->pkcs11h_cert, CKM_RSA_PKCS, input,
            input_len, NULL, &output_len ) != CKR_OK )
    {
        return( MBEDTLS_ERR_RSA_BAD_INPUT_DATA );
    }

    if( output_len > output_max_len )
        return( MBEDTLS_ERR_RSA_OUTPUT_TOO_LARGE );

    if( pkcs11h_certificate_decryptAny( ctx->pkcs11h_cert, CKM_RSA_PKCS, input,
            input_len, output, &output_len ) != CKR_OK )
    {
        return( MBEDTLS_ERR_RSA_BAD_INPUT_DATA );
    }
    *olen = output_len;
    return( 0 );
}

int mbedtls_pkcs11_sign( mbedtls_pkcs11_context *ctx,
                    int mode,
                    mbedtls_md_type_t md_alg,
                    unsigned int hashlen,
                    const unsigned char *hash,
                    unsigned char *sig )
{
    size_t sig_len = 0, asn_len = 0, oid_size = 0;
    unsigned char *p = sig;
    const char *oid;

    if( NULL == ctx )
        return( MBEDTLS_ERR_RSA_BAD_INPUT_DATA );

    if( MBEDTLS_RSA_PRIVATE != mode )
        return( MBEDTLS_ERR_RSA_BAD_INPUT_DATA );

    if( md_alg != MBEDTLS_MD_NONE )
    {
        const mbedtls_md_info_t *md_info = mbedtls_md_info_from_type( md_alg );
        if( md_info == NULL )
            return( MBEDTLS_ERR_RSA_BAD_INPUT_DATA );

        if( mbedtls_oid_get_oid_by_md( md_alg, &oid, &oid_size ) != 0 )
            return( MBEDTLS_ERR_RSA_BAD_INPUT_DATA );

        hashlen = mbedtls_md_get_size( md_info );
        asn_len = 10 + oid_size;
    }

    sig_len = ctx->len;
    if( hashlen > sig_len || asn_len > sig_len ||
        hashlen + asn_len > sig_len )
    {
        return( MBEDTLS_ERR_RSA_BAD_INPUT_DATA );
    }

    if( md_alg != MBEDTLS_MD_NONE )
    {
        /*
         * DigestInfo ::= SEQUENCE {
         *   digestAlgorithm DigestAlgorithmIdentifier,
         *   digest Digest }
         *
         * DigestAlgorithmIdentifier ::= AlgorithmIdentifier
         *
         * Digest ::= OCTET STRING
         */
        *p++ = MBEDTLS_ASN1_SEQUENCE | MBEDTLS_ASN1_CONSTRUCTED;
        *p++ = (unsigned char) ( 0x08 + oid_size + hashlen );
        *p++ = MBEDTLS_ASN1_SEQUENCE | MBEDTLS_ASN1_CONSTRUCTED;
        *p++ = (unsigned char) ( 0x04 + oid_size );
        *p++ = MBEDTLS_ASN1_OID;
        *p++ = oid_size & 0xFF;
        memcpy( p, oid, oid_size );
        p += oid_size;
        *p++ = MBEDTLS_ASN1_NULL;
        *p++ = 0x00;
        *p++ = MBEDTLS_ASN1_OCTET_STRING;
        *p++ = hashlen;
    }

    memcpy( p, hash, hashlen );

    if( pkcs11h_certificate_signAny( ctx->pkcs11h_cert, CKM_RSA_PKCS, sig,
            asn_len + hashlen, sig, &sig_len ) != CKR_OK )
    {
        return( MBEDTLS_ERR_RSA_BAD_INPUT_DATA );
    }

    return( 0 );
}

#endif /* defined(MBEDTLS_PKCS11_C) */
