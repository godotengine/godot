/*
 *  X.509 certificate parsing and verification
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
 *  The ITU-T X.509 standard defines a certificate format for PKI.
 *
 *  http://www.ietf.org/rfc/rfc5280.txt (Certificates and CRLs)
 *  http://www.ietf.org/rfc/rfc3279.txt (Alg IDs for CRLs)
 *  http://www.ietf.org/rfc/rfc2986.txt (CSRs, aka PKCS#10)
 *
 *  http://www.itu.int/ITU-T/studygroups/com17/languages/X.680-0207.pdf
 *  http://www.itu.int/ITU-T/studygroups/com17/languages/X.690-0207.pdf
 */

#if !defined(MBEDTLS_CONFIG_FILE)
#include "mbedtls/config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#if defined(MBEDTLS_X509_CRT_PARSE_C)

#include "mbedtls/x509_crt.h"
#include "mbedtls/oid.h"

#include <stdio.h>
#include <string.h>

#if defined(MBEDTLS_PEM_PARSE_C)
#include "mbedtls/pem.h"
#endif

#if defined(MBEDTLS_PLATFORM_C)
#include "mbedtls/platform.h"
#else
#include <stdlib.h>
#define mbedtls_free       free
#define mbedtls_calloc    calloc
#define mbedtls_snprintf   snprintf
#endif

#if defined(MBEDTLS_THREADING_C)
#include "mbedtls/threading.h"
#endif

#if defined(_WIN32) && !defined(EFIX64) && !defined(EFI32)
#include <windows.h>
#else
#include <time.h>
#endif

#if defined(MBEDTLS_FS_IO)
#include <stdio.h>
#if !defined(_WIN32) || defined(EFIX64) || defined(EFI32)
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#endif /* !_WIN32 || EFIX64 || EFI32 */
#endif

/* Implementation that should never be optimized out by the compiler */
static void mbedtls_zeroize( void *v, size_t n ) {
    volatile unsigned char *p = v; while( n-- ) *p++ = 0;
}

/*
 * Default profile
 */
const mbedtls_x509_crt_profile mbedtls_x509_crt_profile_default =
{
#if defined(MBEDTLS_TLS_DEFAULT_ALLOW_SHA1_IN_CERTIFICATES)
    /* Allow SHA-1 (weak, but still safe in controlled environments) */
    MBEDTLS_X509_ID_FLAG( MBEDTLS_MD_SHA1 ) |
#endif
    /* Only SHA-2 hashes */
    MBEDTLS_X509_ID_FLAG( MBEDTLS_MD_SHA224 ) |
    MBEDTLS_X509_ID_FLAG( MBEDTLS_MD_SHA256 ) |
    MBEDTLS_X509_ID_FLAG( MBEDTLS_MD_SHA384 ) |
    MBEDTLS_X509_ID_FLAG( MBEDTLS_MD_SHA512 ),
    0xFFFFFFF, /* Any PK alg    */
    0xFFFFFFF, /* Any curve     */
    2048,
};

/*
 * Next-default profile
 */
const mbedtls_x509_crt_profile mbedtls_x509_crt_profile_next =
{
    /* Hashes from SHA-256 and above */
    MBEDTLS_X509_ID_FLAG( MBEDTLS_MD_SHA256 ) |
    MBEDTLS_X509_ID_FLAG( MBEDTLS_MD_SHA384 ) |
    MBEDTLS_X509_ID_FLAG( MBEDTLS_MD_SHA512 ),
    0xFFFFFFF, /* Any PK alg    */
#if defined(MBEDTLS_ECP_C)
    /* Curves at or above 128-bit security level */
    MBEDTLS_X509_ID_FLAG( MBEDTLS_ECP_DP_SECP256R1 ) |
    MBEDTLS_X509_ID_FLAG( MBEDTLS_ECP_DP_SECP384R1 ) |
    MBEDTLS_X509_ID_FLAG( MBEDTLS_ECP_DP_SECP521R1 ) |
    MBEDTLS_X509_ID_FLAG( MBEDTLS_ECP_DP_BP256R1 ) |
    MBEDTLS_X509_ID_FLAG( MBEDTLS_ECP_DP_BP384R1 ) |
    MBEDTLS_X509_ID_FLAG( MBEDTLS_ECP_DP_BP512R1 ) |
    MBEDTLS_X509_ID_FLAG( MBEDTLS_ECP_DP_SECP256K1 ),
#else
    0,
#endif
    2048,
};

/*
 * NSA Suite B Profile
 */
const mbedtls_x509_crt_profile mbedtls_x509_crt_profile_suiteb =
{
    /* Only SHA-256 and 384 */
    MBEDTLS_X509_ID_FLAG( MBEDTLS_MD_SHA256 ) |
    MBEDTLS_X509_ID_FLAG( MBEDTLS_MD_SHA384 ),
    /* Only ECDSA */
    MBEDTLS_X509_ID_FLAG( MBEDTLS_PK_ECDSA ),
#if defined(MBEDTLS_ECP_C)
    /* Only NIST P-256 and P-384 */
    MBEDTLS_X509_ID_FLAG( MBEDTLS_ECP_DP_SECP256R1 ) |
    MBEDTLS_X509_ID_FLAG( MBEDTLS_ECP_DP_SECP384R1 ),
#else
    0,
#endif
    0,
};

/*
 * Check md_alg against profile
 * Return 0 if md_alg acceptable for this profile, -1 otherwise
 */
static int x509_profile_check_md_alg( const mbedtls_x509_crt_profile *profile,
                                      mbedtls_md_type_t md_alg )
{
    if( ( profile->allowed_mds & MBEDTLS_X509_ID_FLAG( md_alg ) ) != 0 )
        return( 0 );

    return( -1 );
}

/*
 * Check pk_alg against profile
 * Return 0 if pk_alg acceptable for this profile, -1 otherwise
 */
static int x509_profile_check_pk_alg( const mbedtls_x509_crt_profile *profile,
                                      mbedtls_pk_type_t pk_alg )
{
    if( ( profile->allowed_pks & MBEDTLS_X509_ID_FLAG( pk_alg ) ) != 0 )
        return( 0 );

    return( -1 );
}

/*
 * Check key against profile
 * Return 0 if pk_alg acceptable for this profile, -1 otherwise
 */
static int x509_profile_check_key( const mbedtls_x509_crt_profile *profile,
                                   mbedtls_pk_type_t pk_alg,
                                   const mbedtls_pk_context *pk )
{
#if defined(MBEDTLS_RSA_C)
    if( pk_alg == MBEDTLS_PK_RSA || pk_alg == MBEDTLS_PK_RSASSA_PSS )
    {
        if( mbedtls_pk_get_bitlen( pk ) >= profile->rsa_min_bitlen )
            return( 0 );

        return( -1 );
    }
#endif

#if defined(MBEDTLS_ECP_C)
    if( pk_alg == MBEDTLS_PK_ECDSA ||
        pk_alg == MBEDTLS_PK_ECKEY ||
        pk_alg == MBEDTLS_PK_ECKEY_DH )
    {
        mbedtls_ecp_group_id gid = mbedtls_pk_ec( *pk )->grp.id;

        if( ( profile->allowed_curves & MBEDTLS_X509_ID_FLAG( gid ) ) != 0 )
            return( 0 );

        return( -1 );
    }
#endif

    return( -1 );
}

/*
 *  Version  ::=  INTEGER  {  v1(0), v2(1), v3(2)  }
 */
static int x509_get_version( unsigned char **p,
                             const unsigned char *end,
                             int *ver )
{
    int ret;
    size_t len;

    if( ( ret = mbedtls_asn1_get_tag( p, end, &len,
            MBEDTLS_ASN1_CONTEXT_SPECIFIC | MBEDTLS_ASN1_CONSTRUCTED | 0 ) ) != 0 )
    {
        if( ret == MBEDTLS_ERR_ASN1_UNEXPECTED_TAG )
        {
            *ver = 0;
            return( 0 );
        }

        return( ret );
    }

    end = *p + len;

    if( ( ret = mbedtls_asn1_get_int( p, end, ver ) ) != 0 )
        return( MBEDTLS_ERR_X509_INVALID_VERSION + ret );

    if( *p != end )
        return( MBEDTLS_ERR_X509_INVALID_VERSION +
                MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );

    return( 0 );
}

/*
 *  Validity ::= SEQUENCE {
 *       notBefore      Time,
 *       notAfter       Time }
 */
static int x509_get_dates( unsigned char **p,
                           const unsigned char *end,
                           mbedtls_x509_time *from,
                           mbedtls_x509_time *to )
{
    int ret;
    size_t len;

    if( ( ret = mbedtls_asn1_get_tag( p, end, &len,
            MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE ) ) != 0 )
        return( MBEDTLS_ERR_X509_INVALID_DATE + ret );

    end = *p + len;

    if( ( ret = mbedtls_x509_get_time( p, end, from ) ) != 0 )
        return( ret );

    if( ( ret = mbedtls_x509_get_time( p, end, to ) ) != 0 )
        return( ret );

    if( *p != end )
        return( MBEDTLS_ERR_X509_INVALID_DATE +
                MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );

    return( 0 );
}

/*
 * X.509 v2/v3 unique identifier (not parsed)
 */
static int x509_get_uid( unsigned char **p,
                         const unsigned char *end,
                         mbedtls_x509_buf *uid, int n )
{
    int ret;

    if( *p == end )
        return( 0 );

    uid->tag = **p;

    if( ( ret = mbedtls_asn1_get_tag( p, end, &uid->len,
            MBEDTLS_ASN1_CONTEXT_SPECIFIC | MBEDTLS_ASN1_CONSTRUCTED | n ) ) != 0 )
    {
        if( ret == MBEDTLS_ERR_ASN1_UNEXPECTED_TAG )
            return( 0 );

        return( ret );
    }

    uid->p = *p;
    *p += uid->len;

    return( 0 );
}

static int x509_get_basic_constraints( unsigned char **p,
                                       const unsigned char *end,
                                       int *ca_istrue,
                                       int *max_pathlen )
{
    int ret;
    size_t len;

    /*
     * BasicConstraints ::= SEQUENCE {
     *      cA                      BOOLEAN DEFAULT FALSE,
     *      pathLenConstraint       INTEGER (0..MAX) OPTIONAL }
     */
    *ca_istrue = 0; /* DEFAULT FALSE */
    *max_pathlen = 0; /* endless */

    if( ( ret = mbedtls_asn1_get_tag( p, end, &len,
            MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE ) ) != 0 )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );

    if( *p == end )
        return( 0 );

    if( ( ret = mbedtls_asn1_get_bool( p, end, ca_istrue ) ) != 0 )
    {
        if( ret == MBEDTLS_ERR_ASN1_UNEXPECTED_TAG )
            ret = mbedtls_asn1_get_int( p, end, ca_istrue );

        if( ret != 0 )
            return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );

        if( *ca_istrue != 0 )
            *ca_istrue = 1;
    }

    if( *p == end )
        return( 0 );

    if( ( ret = mbedtls_asn1_get_int( p, end, max_pathlen ) ) != 0 )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );

    if( *p != end )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );

    (*max_pathlen)++;

    return( 0 );
}

static int x509_get_ns_cert_type( unsigned char **p,
                                       const unsigned char *end,
                                       unsigned char *ns_cert_type)
{
    int ret;
    mbedtls_x509_bitstring bs = { 0, 0, NULL };

    if( ( ret = mbedtls_asn1_get_bitstring( p, end, &bs ) ) != 0 )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );

    if( bs.len != 1 )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                MBEDTLS_ERR_ASN1_INVALID_LENGTH );

    /* Get actual bitstring */
    *ns_cert_type = *bs.p;
    return( 0 );
}

static int x509_get_key_usage( unsigned char **p,
                               const unsigned char *end,
                               unsigned int *key_usage)
{
    int ret;
    size_t i;
    mbedtls_x509_bitstring bs = { 0, 0, NULL };

    if( ( ret = mbedtls_asn1_get_bitstring( p, end, &bs ) ) != 0 )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );

    if( bs.len < 1 )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                MBEDTLS_ERR_ASN1_INVALID_LENGTH );

    /* Get actual bitstring */
    *key_usage = 0;
    for( i = 0; i < bs.len && i < sizeof( unsigned int ); i++ )
    {
        *key_usage |= (unsigned int) bs.p[i] << (8*i);
    }

    return( 0 );
}

/*
 * ExtKeyUsageSyntax ::= SEQUENCE SIZE (1..MAX) OF KeyPurposeId
 *
 * KeyPurposeId ::= OBJECT IDENTIFIER
 */
static int x509_get_ext_key_usage( unsigned char **p,
                               const unsigned char *end,
                               mbedtls_x509_sequence *ext_key_usage)
{
    int ret;

    if( ( ret = mbedtls_asn1_get_sequence_of( p, end, ext_key_usage, MBEDTLS_ASN1_OID ) ) != 0 )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );

    /* Sequence length must be >= 1 */
    if( ext_key_usage->buf.p == NULL )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                MBEDTLS_ERR_ASN1_INVALID_LENGTH );

    return( 0 );
}

/*
 * SubjectAltName ::= GeneralNames
 *
 * GeneralNames ::= SEQUENCE SIZE (1..MAX) OF GeneralName
 *
 * GeneralName ::= CHOICE {
 *      otherName                       [0]     OtherName,
 *      rfc822Name                      [1]     IA5String,
 *      dNSName                         [2]     IA5String,
 *      x400Address                     [3]     ORAddress,
 *      directoryName                   [4]     Name,
 *      ediPartyName                    [5]     EDIPartyName,
 *      uniformResourceIdentifier       [6]     IA5String,
 *      iPAddress                       [7]     OCTET STRING,
 *      registeredID                    [8]     OBJECT IDENTIFIER }
 *
 * OtherName ::= SEQUENCE {
 *      type-id    OBJECT IDENTIFIER,
 *      value      [0] EXPLICIT ANY DEFINED BY type-id }
 *
 * EDIPartyName ::= SEQUENCE {
 *      nameAssigner            [0]     DirectoryString OPTIONAL,
 *      partyName               [1]     DirectoryString }
 *
 * NOTE: we only parse and use dNSName at this point.
 */
static int x509_get_subject_alt_name( unsigned char **p,
                                      const unsigned char *end,
                                      mbedtls_x509_sequence *subject_alt_name )
{
    int ret;
    size_t len, tag_len;
    mbedtls_asn1_buf *buf;
    unsigned char tag;
    mbedtls_asn1_sequence *cur = subject_alt_name;

    /* Get main sequence tag */
    if( ( ret = mbedtls_asn1_get_tag( p, end, &len,
            MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE ) ) != 0 )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );

    if( *p + len != end )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );

    while( *p < end )
    {
        if( ( end - *p ) < 1 )
            return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                    MBEDTLS_ERR_ASN1_OUT_OF_DATA );

        tag = **p;
        (*p)++;
        if( ( ret = mbedtls_asn1_get_len( p, end, &tag_len ) ) != 0 )
            return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );

        if( ( tag & MBEDTLS_ASN1_CONTEXT_SPECIFIC ) != MBEDTLS_ASN1_CONTEXT_SPECIFIC )
            return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                    MBEDTLS_ERR_ASN1_UNEXPECTED_TAG );

        /* Skip everything but DNS name */
        if( tag != ( MBEDTLS_ASN1_CONTEXT_SPECIFIC | 2 ) )
        {
            *p += tag_len;
            continue;
        }

        /* Allocate and assign next pointer */
        if( cur->buf.p != NULL )
        {
            if( cur->next != NULL )
                return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS );

            cur->next = mbedtls_calloc( 1, sizeof( mbedtls_asn1_sequence ) );

            if( cur->next == NULL )
                return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                        MBEDTLS_ERR_ASN1_ALLOC_FAILED );

            cur = cur->next;
        }

        buf = &(cur->buf);
        buf->tag = tag;
        buf->p = *p;
        buf->len = tag_len;
        *p += buf->len;
    }

    /* Set final sequence entry's next pointer to NULL */
    cur->next = NULL;

    if( *p != end )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );

    return( 0 );
}

/*
 * X.509 v3 extensions
 *
 */
static int x509_get_crt_ext( unsigned char **p,
                             const unsigned char *end,
                             mbedtls_x509_crt *crt )
{
    int ret;
    size_t len;
    unsigned char *end_ext_data, *end_ext_octet;

    if( ( ret = mbedtls_x509_get_ext( p, end, &crt->v3_ext, 3 ) ) != 0 )
    {
        if( ret == MBEDTLS_ERR_ASN1_UNEXPECTED_TAG )
            return( 0 );

        return( ret );
    }

    while( *p < end )
    {
        /*
         * Extension  ::=  SEQUENCE  {
         *      extnID      OBJECT IDENTIFIER,
         *      critical    BOOLEAN DEFAULT FALSE,
         *      extnValue   OCTET STRING  }
         */
        mbedtls_x509_buf extn_oid = {0, 0, NULL};
        int is_critical = 0; /* DEFAULT FALSE */
        int ext_type = 0;

        if( ( ret = mbedtls_asn1_get_tag( p, end, &len,
                MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE ) ) != 0 )
            return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );

        end_ext_data = *p + len;

        /* Get extension ID */
        extn_oid.tag = **p;

        if( ( ret = mbedtls_asn1_get_tag( p, end, &extn_oid.len, MBEDTLS_ASN1_OID ) ) != 0 )
            return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );

        extn_oid.p = *p;
        *p += extn_oid.len;

        if( ( end - *p ) < 1 )
            return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                    MBEDTLS_ERR_ASN1_OUT_OF_DATA );

        /* Get optional critical */
        if( ( ret = mbedtls_asn1_get_bool( p, end_ext_data, &is_critical ) ) != 0 &&
            ( ret != MBEDTLS_ERR_ASN1_UNEXPECTED_TAG ) )
            return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );

        /* Data should be octet string type */
        if( ( ret = mbedtls_asn1_get_tag( p, end_ext_data, &len,
                MBEDTLS_ASN1_OCTET_STRING ) ) != 0 )
            return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );

        end_ext_octet = *p + len;

        if( end_ext_octet != end_ext_data )
            return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                    MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );

        /*
         * Detect supported extensions
         */
        ret = mbedtls_oid_get_x509_ext_type( &extn_oid, &ext_type );

        if( ret != 0 )
        {
            /* No parser found, skip extension */
            *p = end_ext_octet;

#if !defined(MBEDTLS_X509_ALLOW_UNSUPPORTED_CRITICAL_EXTENSION)
            if( is_critical )
            {
                /* Data is marked as critical: fail */
                return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                        MBEDTLS_ERR_ASN1_UNEXPECTED_TAG );
            }
#endif
            continue;
        }

        /* Forbid repeated extensions */
        if( ( crt->ext_types & ext_type ) != 0 )
            return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS );

        crt->ext_types |= ext_type;

        switch( ext_type )
        {
        case MBEDTLS_X509_EXT_BASIC_CONSTRAINTS:
            /* Parse basic constraints */
            if( ( ret = x509_get_basic_constraints( p, end_ext_octet,
                    &crt->ca_istrue, &crt->max_pathlen ) ) != 0 )
                return( ret );
            break;

        case MBEDTLS_X509_EXT_KEY_USAGE:
            /* Parse key usage */
            if( ( ret = x509_get_key_usage( p, end_ext_octet,
                    &crt->key_usage ) ) != 0 )
                return( ret );
            break;

        case MBEDTLS_X509_EXT_EXTENDED_KEY_USAGE:
            /* Parse extended key usage */
            if( ( ret = x509_get_ext_key_usage( p, end_ext_octet,
                    &crt->ext_key_usage ) ) != 0 )
                return( ret );
            break;

        case MBEDTLS_X509_EXT_SUBJECT_ALT_NAME:
            /* Parse subject alt name */
            if( ( ret = x509_get_subject_alt_name( p, end_ext_octet,
                    &crt->subject_alt_names ) ) != 0 )
                return( ret );
            break;

        case MBEDTLS_X509_EXT_NS_CERT_TYPE:
            /* Parse netscape certificate type */
            if( ( ret = x509_get_ns_cert_type( p, end_ext_octet,
                    &crt->ns_cert_type ) ) != 0 )
                return( ret );
            break;

        default:
            return( MBEDTLS_ERR_X509_FEATURE_UNAVAILABLE );
        }
    }

    if( *p != end )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );

    return( 0 );
}

/*
 * Parse and fill a single X.509 certificate in DER format
 */
static int x509_crt_parse_der_core( mbedtls_x509_crt *crt, const unsigned char *buf,
                                    size_t buflen )
{
    int ret;
    size_t len;
    unsigned char *p, *end, *crt_end;
    mbedtls_x509_buf sig_params1, sig_params2, sig_oid2;

    memset( &sig_params1, 0, sizeof( mbedtls_x509_buf ) );
    memset( &sig_params2, 0, sizeof( mbedtls_x509_buf ) );
    memset( &sig_oid2, 0, sizeof( mbedtls_x509_buf ) );

    /*
     * Check for valid input
     */
    if( crt == NULL || buf == NULL )
        return( MBEDTLS_ERR_X509_BAD_INPUT_DATA );

    // Use the original buffer until we figure out actual length
    p = (unsigned char*) buf;
    len = buflen;
    end = p + len;

    /*
     * Certificate  ::=  SEQUENCE  {
     *      tbsCertificate       TBSCertificate,
     *      signatureAlgorithm   AlgorithmIdentifier,
     *      signatureValue       BIT STRING  }
     */
    if( ( ret = mbedtls_asn1_get_tag( &p, end, &len,
            MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE ) ) != 0 )
    {
        mbedtls_x509_crt_free( crt );
        return( MBEDTLS_ERR_X509_INVALID_FORMAT );
    }

    if( len > (size_t) ( end - p ) )
    {
        mbedtls_x509_crt_free( crt );
        return( MBEDTLS_ERR_X509_INVALID_FORMAT +
                MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );
    }
    crt_end = p + len;

    // Create and populate a new buffer for the raw field
    crt->raw.len = crt_end - buf;
    crt->raw.p = p = mbedtls_calloc( 1, crt->raw.len );
    if( p == NULL )
        return( MBEDTLS_ERR_X509_ALLOC_FAILED );

    memcpy( p, buf, crt->raw.len );

    // Direct pointers to the new buffer 
    p += crt->raw.len - len;
    end = crt_end = p + len;

    /*
     * TBSCertificate  ::=  SEQUENCE  {
     */
    crt->tbs.p = p;

    if( ( ret = mbedtls_asn1_get_tag( &p, end, &len,
            MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE ) ) != 0 )
    {
        mbedtls_x509_crt_free( crt );
        return( MBEDTLS_ERR_X509_INVALID_FORMAT + ret );
    }

    end = p + len;
    crt->tbs.len = end - crt->tbs.p;

    /*
     * Version  ::=  INTEGER  {  v1(0), v2(1), v3(2)  }
     *
     * CertificateSerialNumber  ::=  INTEGER
     *
     * signature            AlgorithmIdentifier
     */
    if( ( ret = x509_get_version(  &p, end, &crt->version  ) ) != 0 ||
        ( ret = mbedtls_x509_get_serial(   &p, end, &crt->serial   ) ) != 0 ||
        ( ret = mbedtls_x509_get_alg(      &p, end, &crt->sig_oid,
                                            &sig_params1 ) ) != 0 )
    {
        mbedtls_x509_crt_free( crt );
        return( ret );
    }

    if( crt->version < 0 || crt->version > 2 )
    {
        mbedtls_x509_crt_free( crt );
        return( MBEDTLS_ERR_X509_UNKNOWN_VERSION );
    }

    crt->version++;

    if( ( ret = mbedtls_x509_get_sig_alg( &crt->sig_oid, &sig_params1,
                                  &crt->sig_md, &crt->sig_pk,
                                  &crt->sig_opts ) ) != 0 )
    {
        mbedtls_x509_crt_free( crt );
        return( ret );
    }

    /*
     * issuer               Name
     */
    crt->issuer_raw.p = p;

    if( ( ret = mbedtls_asn1_get_tag( &p, end, &len,
            MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE ) ) != 0 )
    {
        mbedtls_x509_crt_free( crt );
        return( MBEDTLS_ERR_X509_INVALID_FORMAT + ret );
    }

    if( ( ret = mbedtls_x509_get_name( &p, p + len, &crt->issuer ) ) != 0 )
    {
        mbedtls_x509_crt_free( crt );
        return( ret );
    }

    crt->issuer_raw.len = p - crt->issuer_raw.p;

    /*
     * Validity ::= SEQUENCE {
     *      notBefore      Time,
     *      notAfter       Time }
     *
     */
    if( ( ret = x509_get_dates( &p, end, &crt->valid_from,
                                         &crt->valid_to ) ) != 0 )
    {
        mbedtls_x509_crt_free( crt );
        return( ret );
    }

    /*
     * subject              Name
     */
    crt->subject_raw.p = p;

    if( ( ret = mbedtls_asn1_get_tag( &p, end, &len,
            MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE ) ) != 0 )
    {
        mbedtls_x509_crt_free( crt );
        return( MBEDTLS_ERR_X509_INVALID_FORMAT + ret );
    }

    if( len && ( ret = mbedtls_x509_get_name( &p, p + len, &crt->subject ) ) != 0 )
    {
        mbedtls_x509_crt_free( crt );
        return( ret );
    }

    crt->subject_raw.len = p - crt->subject_raw.p;

    /*
     * SubjectPublicKeyInfo
     */
    if( ( ret = mbedtls_pk_parse_subpubkey( &p, end, &crt->pk ) ) != 0 )
    {
        mbedtls_x509_crt_free( crt );
        return( ret );
    }

    /*
     *  issuerUniqueID  [1]  IMPLICIT UniqueIdentifier OPTIONAL,
     *                       -- If present, version shall be v2 or v3
     *  subjectUniqueID [2]  IMPLICIT UniqueIdentifier OPTIONAL,
     *                       -- If present, version shall be v2 or v3
     *  extensions      [3]  EXPLICIT Extensions OPTIONAL
     *                       -- If present, version shall be v3
     */
    if( crt->version == 2 || crt->version == 3 )
    {
        ret = x509_get_uid( &p, end, &crt->issuer_id,  1 );
        if( ret != 0 )
        {
            mbedtls_x509_crt_free( crt );
            return( ret );
        }
    }

    if( crt->version == 2 || crt->version == 3 )
    {
        ret = x509_get_uid( &p, end, &crt->subject_id,  2 );
        if( ret != 0 )
        {
            mbedtls_x509_crt_free( crt );
            return( ret );
        }
    }

#if !defined(MBEDTLS_X509_ALLOW_EXTENSIONS_NON_V3)
    if( crt->version == 3 )
#endif
    {
        ret = x509_get_crt_ext( &p, end, crt );
        if( ret != 0 )
        {
            mbedtls_x509_crt_free( crt );
            return( ret );
        }
    }

    if( p != end )
    {
        mbedtls_x509_crt_free( crt );
        return( MBEDTLS_ERR_X509_INVALID_FORMAT +
                MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );
    }

    end = crt_end;

    /*
     *  }
     *  -- end of TBSCertificate
     *
     *  signatureAlgorithm   AlgorithmIdentifier,
     *  signatureValue       BIT STRING
     */
    if( ( ret = mbedtls_x509_get_alg( &p, end, &sig_oid2, &sig_params2 ) ) != 0 )
    {
        mbedtls_x509_crt_free( crt );
        return( ret );
    }

    if( crt->sig_oid.len != sig_oid2.len ||
        memcmp( crt->sig_oid.p, sig_oid2.p, crt->sig_oid.len ) != 0 ||
        sig_params1.len != sig_params2.len ||
        ( sig_params1.len != 0 &&
          memcmp( sig_params1.p, sig_params2.p, sig_params1.len ) != 0 ) )
    {
        mbedtls_x509_crt_free( crt );
        return( MBEDTLS_ERR_X509_SIG_MISMATCH );
    }

    if( ( ret = mbedtls_x509_get_sig( &p, end, &crt->sig ) ) != 0 )
    {
        mbedtls_x509_crt_free( crt );
        return( ret );
    }

    if( p != end )
    {
        mbedtls_x509_crt_free( crt );
        return( MBEDTLS_ERR_X509_INVALID_FORMAT +
                MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );
    }

    return( 0 );
}

/*
 * Parse one X.509 certificate in DER format from a buffer and add them to a
 * chained list
 */
int mbedtls_x509_crt_parse_der( mbedtls_x509_crt *chain, const unsigned char *buf,
                        size_t buflen )
{
    int ret;
    mbedtls_x509_crt *crt = chain, *prev = NULL;

    /*
     * Check for valid input
     */
    if( crt == NULL || buf == NULL )
        return( MBEDTLS_ERR_X509_BAD_INPUT_DATA );

    while( crt->version != 0 && crt->next != NULL )
    {
        prev = crt;
        crt = crt->next;
    }

    /*
     * Add new certificate on the end of the chain if needed.
     */
    if( crt->version != 0 && crt->next == NULL )
    {
        crt->next = mbedtls_calloc( 1, sizeof( mbedtls_x509_crt ) );

        if( crt->next == NULL )
            return( MBEDTLS_ERR_X509_ALLOC_FAILED );

        prev = crt;
        mbedtls_x509_crt_init( crt->next );
        crt = crt->next;
    }

    if( ( ret = x509_crt_parse_der_core( crt, buf, buflen ) ) != 0 )
    {
        if( prev )
            prev->next = NULL;

        if( crt != chain )
            mbedtls_free( crt );

        return( ret );
    }

    return( 0 );
}

/*
 * Parse one or more PEM certificates from a buffer and add them to the chained
 * list
 */
int mbedtls_x509_crt_parse( mbedtls_x509_crt *chain, const unsigned char *buf, size_t buflen )
{
#if defined(MBEDTLS_PEM_PARSE_C)
    int success = 0, first_error = 0, total_failed = 0;
    int buf_format = MBEDTLS_X509_FORMAT_DER;
#endif

    /*
     * Check for valid input
     */
    if( chain == NULL || buf == NULL )
        return( MBEDTLS_ERR_X509_BAD_INPUT_DATA );

    /*
     * Determine buffer content. Buffer contains either one DER certificate or
     * one or more PEM certificates.
     */
#if defined(MBEDTLS_PEM_PARSE_C)
    if( buflen != 0 && buf[buflen - 1] == '\0' &&
        strstr( (const char *) buf, "-----BEGIN CERTIFICATE-----" ) != NULL )
    {
        buf_format = MBEDTLS_X509_FORMAT_PEM;
    }

    if( buf_format == MBEDTLS_X509_FORMAT_DER )
        return mbedtls_x509_crt_parse_der( chain, buf, buflen );
#else
    return mbedtls_x509_crt_parse_der( chain, buf, buflen );
#endif

#if defined(MBEDTLS_PEM_PARSE_C)
    if( buf_format == MBEDTLS_X509_FORMAT_PEM )
    {
        int ret;
        mbedtls_pem_context pem;

        /* 1 rather than 0 since the terminating NULL byte is counted in */
        while( buflen > 1 )
        {
            size_t use_len;
            mbedtls_pem_init( &pem );

            /* If we get there, we know the string is null-terminated */
            ret = mbedtls_pem_read_buffer( &pem,
                           "-----BEGIN CERTIFICATE-----",
                           "-----END CERTIFICATE-----",
                           buf, NULL, 0, &use_len );

            if( ret == 0 )
            {
                /*
                 * Was PEM encoded
                 */
                buflen -= use_len;
                buf += use_len;
            }
            else if( ret == MBEDTLS_ERR_PEM_BAD_INPUT_DATA )
            {
                return( ret );
            }
            else if( ret != MBEDTLS_ERR_PEM_NO_HEADER_FOOTER_PRESENT )
            {
                mbedtls_pem_free( &pem );

                /*
                 * PEM header and footer were found
                 */
                buflen -= use_len;
                buf += use_len;

                if( first_error == 0 )
                    first_error = ret;

                total_failed++;
                continue;
            }
            else
                break;

            ret = mbedtls_x509_crt_parse_der( chain, pem.buf, pem.buflen );

            mbedtls_pem_free( &pem );

            if( ret != 0 )
            {
                /*
                 * Quit parsing on a memory error
                 */
                if( ret == MBEDTLS_ERR_X509_ALLOC_FAILED )
                    return( ret );

                if( first_error == 0 )
                    first_error = ret;

                total_failed++;
                continue;
            }

            success = 1;
        }
    }

    if( success )
        return( total_failed );
    else if( first_error )
        return( first_error );
    else
        return( MBEDTLS_ERR_X509_CERT_UNKNOWN_FORMAT );
#endif /* MBEDTLS_PEM_PARSE_C */
}

#if defined(MBEDTLS_FS_IO)
/*
 * Load one or more certificates and add them to the chained list
 */
int mbedtls_x509_crt_parse_file( mbedtls_x509_crt *chain, const char *path )
{
    int ret;
    size_t n;
    unsigned char *buf;

    if( ( ret = mbedtls_pk_load_file( path, &buf, &n ) ) != 0 )
        return( ret );

    ret = mbedtls_x509_crt_parse( chain, buf, n );

    mbedtls_zeroize( buf, n );
    mbedtls_free( buf );

    return( ret );
}

int mbedtls_x509_crt_parse_path( mbedtls_x509_crt *chain, const char *path )
{
    int ret = 0;
#if defined(_WIN32) && !defined(EFIX64) && !defined(EFI32)
    int w_ret;
    WCHAR szDir[MAX_PATH];
    char filename[MAX_PATH];
    char *p;
    size_t len = strlen( path );

    WIN32_FIND_DATAW file_data;
    HANDLE hFind;

    if( len > MAX_PATH - 3 )
        return( MBEDTLS_ERR_X509_BAD_INPUT_DATA );

    memset( szDir, 0, sizeof(szDir) );
    memset( filename, 0, MAX_PATH );
    memcpy( filename, path, len );
    filename[len++] = '\\';
    p = filename + len;
    filename[len++] = '*';

    w_ret = MultiByteToWideChar( CP_ACP, 0, filename, (int)len, szDir,
                                 MAX_PATH - 3 );
    if( w_ret == 0 )
        return( MBEDTLS_ERR_X509_BAD_INPUT_DATA );

    hFind = FindFirstFileW( szDir, &file_data );
    if( hFind == INVALID_HANDLE_VALUE )
        return( MBEDTLS_ERR_X509_FILE_IO_ERROR );

    len = MAX_PATH - len;
    do
    {
        memset( p, 0, len );

        if( file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY )
            continue;

        w_ret = WideCharToMultiByte( CP_ACP, 0, file_data.cFileName,
                                     lstrlenW( file_data.cFileName ),
                                     p, (int) len - 1,
                                     NULL, NULL );
        if( w_ret == 0 )
        {
            ret = MBEDTLS_ERR_X509_FILE_IO_ERROR;
            goto cleanup;
        }

        w_ret = mbedtls_x509_crt_parse_file( chain, filename );
        if( w_ret < 0 )
            ret++;
        else
            ret += w_ret;
    }
    while( FindNextFileW( hFind, &file_data ) != 0 );

    if( GetLastError() != ERROR_NO_MORE_FILES )
        ret = MBEDTLS_ERR_X509_FILE_IO_ERROR;

cleanup:
    FindClose( hFind );
#else /* _WIN32 */
    int t_ret;
    int snp_ret;
    struct stat sb;
    struct dirent *entry;
    char entry_name[MBEDTLS_X509_MAX_FILE_PATH_LEN];
    DIR *dir = opendir( path );

    if( dir == NULL )
        return( MBEDTLS_ERR_X509_FILE_IO_ERROR );

#if defined(MBEDTLS_THREADING_C)
    if( ( ret = mbedtls_mutex_lock( &mbedtls_threading_readdir_mutex ) ) != 0 )
    {
        closedir( dir );
        return( ret );
    }
#endif /* MBEDTLS_THREADING_C */

    while( ( entry = readdir( dir ) ) != NULL )
    {
        snp_ret = mbedtls_snprintf( entry_name, sizeof entry_name,
                                    "%s/%s", path, entry->d_name );

        if( snp_ret < 0 || (size_t)snp_ret >= sizeof entry_name )
        {
            ret = MBEDTLS_ERR_X509_BUFFER_TOO_SMALL;
            goto cleanup;
        }
        else if( stat( entry_name, &sb ) == -1 )
        {
            ret = MBEDTLS_ERR_X509_FILE_IO_ERROR;
            goto cleanup;
        }

        if( !S_ISREG( sb.st_mode ) )
            continue;

        // Ignore parse errors
        //
        t_ret = mbedtls_x509_crt_parse_file( chain, entry_name );
        if( t_ret < 0 )
            ret++;
        else
            ret += t_ret;
    }

cleanup:
    closedir( dir );

#if defined(MBEDTLS_THREADING_C)
    if( mbedtls_mutex_unlock( &mbedtls_threading_readdir_mutex ) != 0 )
        ret = MBEDTLS_ERR_THREADING_MUTEX_ERROR;
#endif /* MBEDTLS_THREADING_C */

#endif /* _WIN32 */

    return( ret );
}
#endif /* MBEDTLS_FS_IO */

static int x509_info_subject_alt_name( char **buf, size_t *size,
                                       const mbedtls_x509_sequence *subject_alt_name )
{
    size_t i;
    size_t n = *size;
    char *p = *buf;
    const mbedtls_x509_sequence *cur = subject_alt_name;
    const char *sep = "";
    size_t sep_len = 0;

    while( cur != NULL )
    {
        if( cur->buf.len + sep_len >= n )
        {
            *p = '\0';
            return( MBEDTLS_ERR_X509_BUFFER_TOO_SMALL );
        }

        n -= cur->buf.len + sep_len;
        for( i = 0; i < sep_len; i++ )
            *p++ = sep[i];
        for( i = 0; i < cur->buf.len; i++ )
            *p++ = cur->buf.p[i];

        sep = ", ";
        sep_len = 2;

        cur = cur->next;
    }

    *p = '\0';

    *size = n;
    *buf = p;

    return( 0 );
}

#define PRINT_ITEM(i)                           \
    {                                           \
        ret = mbedtls_snprintf( p, n, "%s" i, sep );    \
        MBEDTLS_X509_SAFE_SNPRINTF;                        \
        sep = ", ";                             \
    }

#define CERT_TYPE(type,name)                    \
    if( ns_cert_type & type )                   \
        PRINT_ITEM( name );

static int x509_info_cert_type( char **buf, size_t *size,
                                unsigned char ns_cert_type )
{
    int ret;
    size_t n = *size;
    char *p = *buf;
    const char *sep = "";

    CERT_TYPE( MBEDTLS_X509_NS_CERT_TYPE_SSL_CLIENT,         "SSL Client" );
    CERT_TYPE( MBEDTLS_X509_NS_CERT_TYPE_SSL_SERVER,         "SSL Server" );
    CERT_TYPE( MBEDTLS_X509_NS_CERT_TYPE_EMAIL,              "Email" );
    CERT_TYPE( MBEDTLS_X509_NS_CERT_TYPE_OBJECT_SIGNING,     "Object Signing" );
    CERT_TYPE( MBEDTLS_X509_NS_CERT_TYPE_RESERVED,           "Reserved" );
    CERT_TYPE( MBEDTLS_X509_NS_CERT_TYPE_SSL_CA,             "SSL CA" );
    CERT_TYPE( MBEDTLS_X509_NS_CERT_TYPE_EMAIL_CA,           "Email CA" );
    CERT_TYPE( MBEDTLS_X509_NS_CERT_TYPE_OBJECT_SIGNING_CA,  "Object Signing CA" );

    *size = n;
    *buf = p;

    return( 0 );
}

#define KEY_USAGE(code,name)    \
    if( key_usage & code )      \
        PRINT_ITEM( name );

static int x509_info_key_usage( char **buf, size_t *size,
                                unsigned int key_usage )
{
    int ret;
    size_t n = *size;
    char *p = *buf;
    const char *sep = "";

    KEY_USAGE( MBEDTLS_X509_KU_DIGITAL_SIGNATURE,    "Digital Signature" );
    KEY_USAGE( MBEDTLS_X509_KU_NON_REPUDIATION,      "Non Repudiation" );
    KEY_USAGE( MBEDTLS_X509_KU_KEY_ENCIPHERMENT,     "Key Encipherment" );
    KEY_USAGE( MBEDTLS_X509_KU_DATA_ENCIPHERMENT,    "Data Encipherment" );
    KEY_USAGE( MBEDTLS_X509_KU_KEY_AGREEMENT,        "Key Agreement" );
    KEY_USAGE( MBEDTLS_X509_KU_KEY_CERT_SIGN,        "Key Cert Sign" );
    KEY_USAGE( MBEDTLS_X509_KU_CRL_SIGN,             "CRL Sign" );
    KEY_USAGE( MBEDTLS_X509_KU_ENCIPHER_ONLY,        "Encipher Only" );
    KEY_USAGE( MBEDTLS_X509_KU_DECIPHER_ONLY,        "Decipher Only" );

    *size = n;
    *buf = p;

    return( 0 );
}

static int x509_info_ext_key_usage( char **buf, size_t *size,
                                    const mbedtls_x509_sequence *extended_key_usage )
{
    int ret;
    const char *desc;
    size_t n = *size;
    char *p = *buf;
    const mbedtls_x509_sequence *cur = extended_key_usage;
    const char *sep = "";

    while( cur != NULL )
    {
        if( mbedtls_oid_get_extended_key_usage( &cur->buf, &desc ) != 0 )
            desc = "???";

        ret = mbedtls_snprintf( p, n, "%s%s", sep, desc );
        MBEDTLS_X509_SAFE_SNPRINTF;

        sep = ", ";

        cur = cur->next;
    }

    *size = n;
    *buf = p;

    return( 0 );
}

/*
 * Return an informational string about the certificate.
 */
#define BEFORE_COLON    18
#define BC              "18"
int mbedtls_x509_crt_info( char *buf, size_t size, const char *prefix,
                   const mbedtls_x509_crt *crt )
{
    int ret;
    size_t n;
    char *p;
    char key_size_str[BEFORE_COLON];

    p = buf;
    n = size;

    if( NULL == crt )
    {
        ret = mbedtls_snprintf( p, n, "\nCertificate is uninitialised!\n" );
        MBEDTLS_X509_SAFE_SNPRINTF;

        return( (int) ( size - n ) );
    }

    ret = mbedtls_snprintf( p, n, "%scert. version     : %d\n",
                               prefix, crt->version );
    MBEDTLS_X509_SAFE_SNPRINTF;
    ret = mbedtls_snprintf( p, n, "%sserial number     : ",
                               prefix );
    MBEDTLS_X509_SAFE_SNPRINTF;

    ret = mbedtls_x509_serial_gets( p, n, &crt->serial );
    MBEDTLS_X509_SAFE_SNPRINTF;

    ret = mbedtls_snprintf( p, n, "\n%sissuer name       : ", prefix );
    MBEDTLS_X509_SAFE_SNPRINTF;
    ret = mbedtls_x509_dn_gets( p, n, &crt->issuer  );
    MBEDTLS_X509_SAFE_SNPRINTF;

    ret = mbedtls_snprintf( p, n, "\n%ssubject name      : ", prefix );
    MBEDTLS_X509_SAFE_SNPRINTF;
    ret = mbedtls_x509_dn_gets( p, n, &crt->subject );
    MBEDTLS_X509_SAFE_SNPRINTF;

    ret = mbedtls_snprintf( p, n, "\n%sissued  on        : " \
                   "%04d-%02d-%02d %02d:%02d:%02d", prefix,
                   crt->valid_from.year, crt->valid_from.mon,
                   crt->valid_from.day,  crt->valid_from.hour,
                   crt->valid_from.min,  crt->valid_from.sec );
    MBEDTLS_X509_SAFE_SNPRINTF;

    ret = mbedtls_snprintf( p, n, "\n%sexpires on        : " \
                   "%04d-%02d-%02d %02d:%02d:%02d", prefix,
                   crt->valid_to.year, crt->valid_to.mon,
                   crt->valid_to.day,  crt->valid_to.hour,
                   crt->valid_to.min,  crt->valid_to.sec );
    MBEDTLS_X509_SAFE_SNPRINTF;

    ret = mbedtls_snprintf( p, n, "\n%ssigned using      : ", prefix );
    MBEDTLS_X509_SAFE_SNPRINTF;

    ret = mbedtls_x509_sig_alg_gets( p, n, &crt->sig_oid, crt->sig_pk,
                             crt->sig_md, crt->sig_opts );
    MBEDTLS_X509_SAFE_SNPRINTF;

    /* Key size */
    if( ( ret = mbedtls_x509_key_size_helper( key_size_str, BEFORE_COLON,
                                      mbedtls_pk_get_name( &crt->pk ) ) ) != 0 )
    {
        return( ret );
    }

    ret = mbedtls_snprintf( p, n, "\n%s%-" BC "s: %d bits", prefix, key_size_str,
                          (int) mbedtls_pk_get_bitlen( &crt->pk ) );
    MBEDTLS_X509_SAFE_SNPRINTF;

    /*
     * Optional extensions
     */

    if( crt->ext_types & MBEDTLS_X509_EXT_BASIC_CONSTRAINTS )
    {
        ret = mbedtls_snprintf( p, n, "\n%sbasic constraints : CA=%s", prefix,
                        crt->ca_istrue ? "true" : "false" );
        MBEDTLS_X509_SAFE_SNPRINTF;

        if( crt->max_pathlen > 0 )
        {
            ret = mbedtls_snprintf( p, n, ", max_pathlen=%d", crt->max_pathlen - 1 );
            MBEDTLS_X509_SAFE_SNPRINTF;
        }
    }

    if( crt->ext_types & MBEDTLS_X509_EXT_SUBJECT_ALT_NAME )
    {
        ret = mbedtls_snprintf( p, n, "\n%ssubject alt name  : ", prefix );
        MBEDTLS_X509_SAFE_SNPRINTF;

        if( ( ret = x509_info_subject_alt_name( &p, &n,
                                            &crt->subject_alt_names ) ) != 0 )
            return( ret );
    }

    if( crt->ext_types & MBEDTLS_X509_EXT_NS_CERT_TYPE )
    {
        ret = mbedtls_snprintf( p, n, "\n%scert. type        : ", prefix );
        MBEDTLS_X509_SAFE_SNPRINTF;

        if( ( ret = x509_info_cert_type( &p, &n, crt->ns_cert_type ) ) != 0 )
            return( ret );
    }

    if( crt->ext_types & MBEDTLS_X509_EXT_KEY_USAGE )
    {
        ret = mbedtls_snprintf( p, n, "\n%skey usage         : ", prefix );
        MBEDTLS_X509_SAFE_SNPRINTF;

        if( ( ret = x509_info_key_usage( &p, &n, crt->key_usage ) ) != 0 )
            return( ret );
    }

    if( crt->ext_types & MBEDTLS_X509_EXT_EXTENDED_KEY_USAGE )
    {
        ret = mbedtls_snprintf( p, n, "\n%sext key usage     : ", prefix );
        MBEDTLS_X509_SAFE_SNPRINTF;

        if( ( ret = x509_info_ext_key_usage( &p, &n,
                                             &crt->ext_key_usage ) ) != 0 )
            return( ret );
    }

    ret = mbedtls_snprintf( p, n, "\n" );
    MBEDTLS_X509_SAFE_SNPRINTF;

    return( (int) ( size - n ) );
}

struct x509_crt_verify_string {
    int code;
    const char *string;
};

static const struct x509_crt_verify_string x509_crt_verify_strings[] = {
    { MBEDTLS_X509_BADCERT_EXPIRED,       "The certificate validity has expired" },
    { MBEDTLS_X509_BADCERT_REVOKED,       "The certificate has been revoked (is on a CRL)" },
    { MBEDTLS_X509_BADCERT_CN_MISMATCH,   "The certificate Common Name (CN) does not match with the expected CN" },
    { MBEDTLS_X509_BADCERT_NOT_TRUSTED,   "The certificate is not correctly signed by the trusted CA" },
    { MBEDTLS_X509_BADCRL_NOT_TRUSTED,    "The CRL is not correctly signed by the trusted CA" },
    { MBEDTLS_X509_BADCRL_EXPIRED,        "The CRL is expired" },
    { MBEDTLS_X509_BADCERT_MISSING,       "Certificate was missing" },
    { MBEDTLS_X509_BADCERT_SKIP_VERIFY,   "Certificate verification was skipped" },
    { MBEDTLS_X509_BADCERT_OTHER,         "Other reason (can be used by verify callback)" },
    { MBEDTLS_X509_BADCERT_FUTURE,        "The certificate validity starts in the future" },
    { MBEDTLS_X509_BADCRL_FUTURE,         "The CRL is from the future" },
    { MBEDTLS_X509_BADCERT_KEY_USAGE,     "Usage does not match the keyUsage extension" },
    { MBEDTLS_X509_BADCERT_EXT_KEY_USAGE, "Usage does not match the extendedKeyUsage extension" },
    { MBEDTLS_X509_BADCERT_NS_CERT_TYPE,  "Usage does not match the nsCertType extension" },
    { MBEDTLS_X509_BADCERT_BAD_MD,        "The certificate is signed with an unacceptable hash." },
    { MBEDTLS_X509_BADCERT_BAD_PK,        "The certificate is signed with an unacceptable PK alg (eg RSA vs ECDSA)." },
    { MBEDTLS_X509_BADCERT_BAD_KEY,       "The certificate is signed with an unacceptable key (eg bad curve, RSA too short)." },
    { MBEDTLS_X509_BADCRL_BAD_MD,         "The CRL is signed with an unacceptable hash." },
    { MBEDTLS_X509_BADCRL_BAD_PK,         "The CRL is signed with an unacceptable PK alg (eg RSA vs ECDSA)." },
    { MBEDTLS_X509_BADCRL_BAD_KEY,        "The CRL is signed with an unacceptable key (eg bad curve, RSA too short)." },
    { 0, NULL }
};

int mbedtls_x509_crt_verify_info( char *buf, size_t size, const char *prefix,
                          uint32_t flags )
{
    int ret;
    const struct x509_crt_verify_string *cur;
    char *p = buf;
    size_t n = size;

    for( cur = x509_crt_verify_strings; cur->string != NULL ; cur++ )
    {
        if( ( flags & cur->code ) == 0 )
            continue;

        ret = mbedtls_snprintf( p, n, "%s%s\n", prefix, cur->string );
        MBEDTLS_X509_SAFE_SNPRINTF;
        flags ^= cur->code;
    }

    if( flags != 0 )
    {
        ret = mbedtls_snprintf( p, n, "%sUnknown reason "
                                       "(this should not happen)\n", prefix );
        MBEDTLS_X509_SAFE_SNPRINTF;
    }

    return( (int) ( size - n ) );
}

#if defined(MBEDTLS_X509_CHECK_KEY_USAGE)
int mbedtls_x509_crt_check_key_usage( const mbedtls_x509_crt *crt,
                                      unsigned int usage )
{
    unsigned int usage_must, usage_may;
    unsigned int may_mask = MBEDTLS_X509_KU_ENCIPHER_ONLY
                          | MBEDTLS_X509_KU_DECIPHER_ONLY;

    if( ( crt->ext_types & MBEDTLS_X509_EXT_KEY_USAGE ) == 0 )
        return( 0 );

    usage_must = usage & ~may_mask;

    if( ( ( crt->key_usage & ~may_mask ) & usage_must ) != usage_must )
        return( MBEDTLS_ERR_X509_BAD_INPUT_DATA );

    usage_may = usage & may_mask;

    if( ( ( crt->key_usage & may_mask ) | usage_may ) != usage_may )
        return( MBEDTLS_ERR_X509_BAD_INPUT_DATA );

    return( 0 );
}
#endif

#if defined(MBEDTLS_X509_CHECK_EXTENDED_KEY_USAGE)
int mbedtls_x509_crt_check_extended_key_usage( const mbedtls_x509_crt *crt,
                                       const char *usage_oid,
                                       size_t usage_len )
{
    const mbedtls_x509_sequence *cur;

    /* Extension is not mandatory, absent means no restriction */
    if( ( crt->ext_types & MBEDTLS_X509_EXT_EXTENDED_KEY_USAGE ) == 0 )
        return( 0 );

    /*
     * Look for the requested usage (or wildcard ANY) in our list
     */
    for( cur = &crt->ext_key_usage; cur != NULL; cur = cur->next )
    {
        const mbedtls_x509_buf *cur_oid = &cur->buf;

        if( cur_oid->len == usage_len &&
            memcmp( cur_oid->p, usage_oid, usage_len ) == 0 )
        {
            return( 0 );
        }

        if( MBEDTLS_OID_CMP( MBEDTLS_OID_ANY_EXTENDED_KEY_USAGE, cur_oid ) == 0 )
            return( 0 );
    }

    return( MBEDTLS_ERR_X509_BAD_INPUT_DATA );
}
#endif /* MBEDTLS_X509_CHECK_EXTENDED_KEY_USAGE */

#if defined(MBEDTLS_X509_CRL_PARSE_C)
/*
 * Return 1 if the certificate is revoked, or 0 otherwise.
 */
int mbedtls_x509_crt_is_revoked( const mbedtls_x509_crt *crt, const mbedtls_x509_crl *crl )
{
    const mbedtls_x509_crl_entry *cur = &crl->entry;

    while( cur != NULL && cur->serial.len != 0 )
    {
        if( crt->serial.len == cur->serial.len &&
            memcmp( crt->serial.p, cur->serial.p, crt->serial.len ) == 0 )
        {
            if( mbedtls_x509_time_is_past( &cur->revocation_date ) )
                return( 1 );
        }

        cur = cur->next;
    }

    return( 0 );
}

/*
 * Check that the given certificate is not revoked according to the CRL.
 * Skip validation is no CRL for the given CA is present.
 */
static int x509_crt_verifycrl( mbedtls_x509_crt *crt, mbedtls_x509_crt *ca,
                               mbedtls_x509_crl *crl_list,
                               const mbedtls_x509_crt_profile *profile )
{
    int flags = 0;
    unsigned char hash[MBEDTLS_MD_MAX_SIZE];
    const mbedtls_md_info_t *md_info;

    if( ca == NULL )
        return( flags );

    while( crl_list != NULL )
    {
        if( crl_list->version == 0 ||
            crl_list->issuer_raw.len != ca->subject_raw.len ||
            memcmp( crl_list->issuer_raw.p, ca->subject_raw.p,
                    crl_list->issuer_raw.len ) != 0 )
        {
            crl_list = crl_list->next;
            continue;
        }

        /*
         * Check if the CA is configured to sign CRLs
         */
#if defined(MBEDTLS_X509_CHECK_KEY_USAGE)
        if( mbedtls_x509_crt_check_key_usage( ca, MBEDTLS_X509_KU_CRL_SIGN ) != 0 )
        {
            flags |= MBEDTLS_X509_BADCRL_NOT_TRUSTED;
            break;
        }
#endif

        /*
         * Check if CRL is correctly signed by the trusted CA
         */
        if( x509_profile_check_md_alg( profile, crl_list->sig_md ) != 0 )
            flags |= MBEDTLS_X509_BADCRL_BAD_MD;

        if( x509_profile_check_pk_alg( profile, crl_list->sig_pk ) != 0 )
            flags |= MBEDTLS_X509_BADCRL_BAD_PK;

        md_info = mbedtls_md_info_from_type( crl_list->sig_md );
        if( md_info == NULL )
        {
            /*
             * Cannot check 'unknown' hash
             */
            flags |= MBEDTLS_X509_BADCRL_NOT_TRUSTED;
            break;
        }

        mbedtls_md( md_info, crl_list->tbs.p, crl_list->tbs.len, hash );

        if( x509_profile_check_key( profile, crl_list->sig_pk, &ca->pk ) != 0 )
            flags |= MBEDTLS_X509_BADCERT_BAD_KEY;

        if( mbedtls_pk_verify_ext( crl_list->sig_pk, crl_list->sig_opts, &ca->pk,
                           crl_list->sig_md, hash, mbedtls_md_get_size( md_info ),
                           crl_list->sig.p, crl_list->sig.len ) != 0 )
        {
            flags |= MBEDTLS_X509_BADCRL_NOT_TRUSTED;
            break;
        }

        /*
         * Check for validity of CRL (Do not drop out)
         */
        if( mbedtls_x509_time_is_past( &crl_list->next_update ) )
            flags |= MBEDTLS_X509_BADCRL_EXPIRED;

        if( mbedtls_x509_time_is_future( &crl_list->this_update ) )
            flags |= MBEDTLS_X509_BADCRL_FUTURE;

        /*
         * Check if certificate is revoked
         */
        if( mbedtls_x509_crt_is_revoked( crt, crl_list ) )
        {
            flags |= MBEDTLS_X509_BADCERT_REVOKED;
            break;
        }

        crl_list = crl_list->next;
    }

    return( flags );
}
#endif /* MBEDTLS_X509_CRL_PARSE_C */

/*
 * Like memcmp, but case-insensitive and always returns -1 if different
 */
static int x509_memcasecmp( const void *s1, const void *s2, size_t len )
{
    size_t i;
    unsigned char diff;
    const unsigned char *n1 = s1, *n2 = s2;

    for( i = 0; i < len; i++ )
    {
        diff = n1[i] ^ n2[i];

        if( diff == 0 )
            continue;

        if( diff == 32 &&
            ( ( n1[i] >= 'a' && n1[i] <= 'z' ) ||
              ( n1[i] >= 'A' && n1[i] <= 'Z' ) ) )
        {
            continue;
        }

        return( -1 );
    }

    return( 0 );
}

/*
 * Return 0 if name matches wildcard, -1 otherwise
 */
static int x509_check_wildcard( const char *cn, mbedtls_x509_buf *name )
{
    size_t i;
    size_t cn_idx = 0, cn_len = strlen( cn );

    if( name->len < 3 || name->p[0] != '*' || name->p[1] != '.' )
        return( 0 );

    for( i = 0; i < cn_len; ++i )
    {
        if( cn[i] == '.' )
        {
            cn_idx = i;
            break;
        }
    }

    if( cn_idx == 0 )
        return( -1 );

    if( cn_len - cn_idx == name->len - 1 &&
        x509_memcasecmp( name->p + 1, cn + cn_idx, name->len - 1 ) == 0 )
    {
        return( 0 );
    }

    return( -1 );
}

/*
 * Compare two X.509 strings, case-insensitive, and allowing for some encoding
 * variations (but not all).
 *
 * Return 0 if equal, -1 otherwise.
 */
static int x509_string_cmp( const mbedtls_x509_buf *a, const mbedtls_x509_buf *b )
{
    if( a->tag == b->tag &&
        a->len == b->len &&
        memcmp( a->p, b->p, b->len ) == 0 )
    {
        return( 0 );
    }

    if( ( a->tag == MBEDTLS_ASN1_UTF8_STRING || a->tag == MBEDTLS_ASN1_PRINTABLE_STRING ) &&
        ( b->tag == MBEDTLS_ASN1_UTF8_STRING || b->tag == MBEDTLS_ASN1_PRINTABLE_STRING ) &&
        a->len == b->len &&
        x509_memcasecmp( a->p, b->p, b->len ) == 0 )
    {
        return( 0 );
    }

    return( -1 );
}

/*
 * Compare two X.509 Names (aka rdnSequence).
 *
 * See RFC 5280 section 7.1, though we don't implement the whole algorithm:
 * we sometimes return unequal when the full algorithm would return equal,
 * but never the other way. (In particular, we don't do Unicode normalisation
 * or space folding.)
 *
 * Return 0 if equal, -1 otherwise.
 */
static int x509_name_cmp( const mbedtls_x509_name *a, const mbedtls_x509_name *b )
{
    /* Avoid recursion, it might not be optimised by the compiler */
    while( a != NULL || b != NULL )
    {
        if( a == NULL || b == NULL )
            return( -1 );

        /* type */
        if( a->oid.tag != b->oid.tag ||
            a->oid.len != b->oid.len ||
            memcmp( a->oid.p, b->oid.p, b->oid.len ) != 0 )
        {
            return( -1 );
        }

        /* value */
        if( x509_string_cmp( &a->val, &b->val ) != 0 )
            return( -1 );

        /* structure of the list of sets */
        if( a->next_merged != b->next_merged )
            return( -1 );

        a = a->next;
        b = b->next;
    }

    /* a == NULL == b */
    return( 0 );
}

/*
 * Check if 'parent' is a suitable parent (signing CA) for 'child'.
 * Return 0 if yes, -1 if not.
 *
 * top means parent is a locally-trusted certificate
 * bottom means child is the end entity cert
 */
static int x509_crt_check_parent( const mbedtls_x509_crt *child,
                                  const mbedtls_x509_crt *parent,
                                  int top, int bottom )
{
    int need_ca_bit;

    /* Parent must be the issuer */
    if( x509_name_cmp( &child->issuer, &parent->subject ) != 0 )
        return( -1 );

    /* Parent must have the basicConstraints CA bit set as a general rule */
    need_ca_bit = 1;

    /* Exception: v1/v2 certificates that are locally trusted. */
    if( top && parent->version < 3 )
        need_ca_bit = 0;

    /* Exception: self-signed end-entity certs that are locally trusted. */
    if( top && bottom &&
        child->raw.len == parent->raw.len &&
        memcmp( child->raw.p, parent->raw.p, child->raw.len ) == 0 )
    {
        need_ca_bit = 0;
    }

    if( need_ca_bit && ! parent->ca_istrue )
        return( -1 );

#if defined(MBEDTLS_X509_CHECK_KEY_USAGE)
    if( need_ca_bit &&
        mbedtls_x509_crt_check_key_usage( parent, MBEDTLS_X509_KU_KEY_CERT_SIGN ) != 0 )
    {
        return( -1 );
    }
#endif

    return( 0 );
}

static int x509_crt_verify_top(
                mbedtls_x509_crt *child, mbedtls_x509_crt *trust_ca,
                mbedtls_x509_crl *ca_crl,
                const mbedtls_x509_crt_profile *profile,
                int path_cnt, int self_cnt, uint32_t *flags,
                int (*f_vrfy)(void *, mbedtls_x509_crt *, int, uint32_t *),
                void *p_vrfy )
{
    int ret;
    uint32_t ca_flags = 0;
    int check_path_cnt;
    unsigned char hash[MBEDTLS_MD_MAX_SIZE];
    const mbedtls_md_info_t *md_info;
    mbedtls_x509_crt *future_past_ca = NULL;

    if( mbedtls_x509_time_is_past( &child->valid_to ) )
        *flags |= MBEDTLS_X509_BADCERT_EXPIRED;

    if( mbedtls_x509_time_is_future( &child->valid_from ) )
        *flags |= MBEDTLS_X509_BADCERT_FUTURE;

    if( x509_profile_check_md_alg( profile, child->sig_md ) != 0 )
        *flags |= MBEDTLS_X509_BADCERT_BAD_MD;

    if( x509_profile_check_pk_alg( profile, child->sig_pk ) != 0 )
        *flags |= MBEDTLS_X509_BADCERT_BAD_PK;

    /*
     * Child is the top of the chain. Check against the trust_ca list.
     */
    *flags |= MBEDTLS_X509_BADCERT_NOT_TRUSTED;

    md_info = mbedtls_md_info_from_type( child->sig_md );
    if( md_info == NULL )
    {
        /*
         * Cannot check 'unknown', no need to try any CA
         */
        trust_ca = NULL;
    }
    else
        mbedtls_md( md_info, child->tbs.p, child->tbs.len, hash );

    for( /* trust_ca */ ; trust_ca != NULL; trust_ca = trust_ca->next )
    {
        if( x509_crt_check_parent( child, trust_ca, 1, path_cnt == 0 ) != 0 )
            continue;

        check_path_cnt = path_cnt + 1;

        /*
         * Reduce check_path_cnt to check against if top of the chain is
         * the same as the trusted CA
         */
        if( child->subject_raw.len == trust_ca->subject_raw.len &&
            memcmp( child->subject_raw.p, trust_ca->subject_raw.p,
                            child->issuer_raw.len ) == 0 )
        {
            check_path_cnt--;
        }

        /* Self signed certificates do not count towards the limit */
        if( trust_ca->max_pathlen > 0 &&
            trust_ca->max_pathlen < check_path_cnt - self_cnt )
        {
            continue;
        }

        if( mbedtls_pk_verify_ext( child->sig_pk, child->sig_opts, &trust_ca->pk,
                           child->sig_md, hash, mbedtls_md_get_size( md_info ),
                           child->sig.p, child->sig.len ) != 0 )
        {
            continue;
        }

        if( mbedtls_x509_time_is_past( &trust_ca->valid_to ) ||
            mbedtls_x509_time_is_future( &trust_ca->valid_from ) )
        {
            if ( future_past_ca == NULL )
                future_past_ca = trust_ca;

            continue;
        }

        break;
    }

    if( trust_ca != NULL || ( trust_ca = future_past_ca ) != NULL )
    {
        /*
         * Top of chain is signed by a trusted CA
         */
        *flags &= ~MBEDTLS_X509_BADCERT_NOT_TRUSTED;

        if( x509_profile_check_key( profile, child->sig_pk, &trust_ca->pk ) != 0 )
            *flags |= MBEDTLS_X509_BADCERT_BAD_KEY;
    }

    /*
     * If top of chain is not the same as the trusted CA send a verify request
     * to the callback for any issues with validity and CRL presence for the
     * trusted CA certificate.
     */
    if( trust_ca != NULL &&
        ( child->subject_raw.len != trust_ca->subject_raw.len ||
          memcmp( child->subject_raw.p, trust_ca->subject_raw.p,
                            child->issuer_raw.len ) != 0 ) )
    {
#if defined(MBEDTLS_X509_CRL_PARSE_C)
        /* Check trusted CA's CRL for the chain's top crt */
        *flags |= x509_crt_verifycrl( child, trust_ca, ca_crl, profile );
#else
        ((void) ca_crl);
#endif

        if( mbedtls_x509_time_is_past( &trust_ca->valid_to ) )
            ca_flags |= MBEDTLS_X509_BADCERT_EXPIRED;

        if( mbedtls_x509_time_is_future( &trust_ca->valid_from ) )
            ca_flags |= MBEDTLS_X509_BADCERT_FUTURE;

        if( NULL != f_vrfy )
        {
            if( ( ret = f_vrfy( p_vrfy, trust_ca, path_cnt + 1,
                                &ca_flags ) ) != 0 )
            {
                return( ret );
            }
        }
    }

    /* Call callback on top cert */
    if( NULL != f_vrfy )
    {
        if( ( ret = f_vrfy( p_vrfy, child, path_cnt, flags ) ) != 0 )
            return( ret );
    }

    *flags |= ca_flags;

    return( 0 );
}

static int x509_crt_verify_child(
                mbedtls_x509_crt *child, mbedtls_x509_crt *parent,
                mbedtls_x509_crt *trust_ca, mbedtls_x509_crl *ca_crl,
                const mbedtls_x509_crt_profile *profile,
                int path_cnt, int self_cnt, uint32_t *flags,
                int (*f_vrfy)(void *, mbedtls_x509_crt *, int, uint32_t *),
                void *p_vrfy )
{
    int ret;
    uint32_t parent_flags = 0;
    unsigned char hash[MBEDTLS_MD_MAX_SIZE];
    mbedtls_x509_crt *grandparent;
    const mbedtls_md_info_t *md_info;

    /* Counting intermediate self signed certificates */
    if( ( path_cnt != 0 ) && x509_name_cmp( &child->issuer, &child->subject ) == 0 )
        self_cnt++;

    /* path_cnt is 0 for the first intermediate CA */
    if( 1 + path_cnt > MBEDTLS_X509_MAX_INTERMEDIATE_CA )
    {
        /* return immediately as the goal is to avoid unbounded recursion */
        return( MBEDTLS_ERR_X509_FATAL_ERROR );
    }

    if( mbedtls_x509_time_is_past( &child->valid_to ) )
        *flags |= MBEDTLS_X509_BADCERT_EXPIRED;

    if( mbedtls_x509_time_is_future( &child->valid_from ) )
        *flags |= MBEDTLS_X509_BADCERT_FUTURE;

    if( x509_profile_check_md_alg( profile, child->sig_md ) != 0 )
        *flags |= MBEDTLS_X509_BADCERT_BAD_MD;

    if( x509_profile_check_pk_alg( profile, child->sig_pk ) != 0 )
        *flags |= MBEDTLS_X509_BADCERT_BAD_PK;

    md_info = mbedtls_md_info_from_type( child->sig_md );
    if( md_info == NULL )
    {
        /*
         * Cannot check 'unknown' hash
         */
        *flags |= MBEDTLS_X509_BADCERT_NOT_TRUSTED;
    }
    else
    {
        mbedtls_md( md_info, child->tbs.p, child->tbs.len, hash );

        if( x509_profile_check_key( profile, child->sig_pk, &parent->pk ) != 0 )
            *flags |= MBEDTLS_X509_BADCERT_BAD_KEY;

        if( mbedtls_pk_verify_ext( child->sig_pk, child->sig_opts, &parent->pk,
                           child->sig_md, hash, mbedtls_md_get_size( md_info ),
                           child->sig.p, child->sig.len ) != 0 )
        {
            *flags |= MBEDTLS_X509_BADCERT_NOT_TRUSTED;
        }
    }

#if defined(MBEDTLS_X509_CRL_PARSE_C)
    /* Check trusted CA's CRL for the given crt */
    *flags |= x509_crt_verifycrl(child, parent, ca_crl, profile );
#endif

    /* Look for a grandparent in trusted CAs */
    for( grandparent = trust_ca;
         grandparent != NULL;
         grandparent = grandparent->next )
    {
        if( x509_crt_check_parent( parent, grandparent,
                                   0, path_cnt == 0 ) == 0 )
            break;
    }

    if( grandparent != NULL )
    {
        ret = x509_crt_verify_top( parent, grandparent, ca_crl, profile,
                                path_cnt + 1, self_cnt, &parent_flags, f_vrfy, p_vrfy );
        if( ret != 0 )
            return( ret );
    }
    else
    {
        /* Look for a grandparent upwards the chain */
        for( grandparent = parent->next;
             grandparent != NULL;
             grandparent = grandparent->next )
        {
            /* +2 because the current step is not yet accounted for
             * and because max_pathlen is one higher than it should be.
             * Also self signed certificates do not count to the limit. */
            if( grandparent->max_pathlen > 0 &&
                grandparent->max_pathlen < 2 + path_cnt - self_cnt )
            {
                continue;
            }

            if( x509_crt_check_parent( parent, grandparent,
                                       0, path_cnt == 0 ) == 0 )
                break;
        }

        /* Is our parent part of the chain or at the top? */
        if( grandparent != NULL )
        {
            ret = x509_crt_verify_child( parent, grandparent, trust_ca, ca_crl,
                                         profile, path_cnt + 1, self_cnt, &parent_flags,
                                         f_vrfy, p_vrfy );
            if( ret != 0 )
                return( ret );
        }
        else
        {
            ret = x509_crt_verify_top( parent, trust_ca, ca_crl, profile,
                                       path_cnt + 1, self_cnt, &parent_flags,
                                       f_vrfy, p_vrfy );
            if( ret != 0 )
                return( ret );
        }
    }

    /* child is verified to be a child of the parent, call verify callback */
    if( NULL != f_vrfy )
        if( ( ret = f_vrfy( p_vrfy, child, path_cnt, flags ) ) != 0 )
            return( ret );

    *flags |= parent_flags;

    return( 0 );
}

/*
 * Verify the certificate validity
 */
int mbedtls_x509_crt_verify( mbedtls_x509_crt *crt,
                     mbedtls_x509_crt *trust_ca,
                     mbedtls_x509_crl *ca_crl,
                     const char *cn, uint32_t *flags,
                     int (*f_vrfy)(void *, mbedtls_x509_crt *, int, uint32_t *),
                     void *p_vrfy )
{
    return( mbedtls_x509_crt_verify_with_profile( crt, trust_ca, ca_crl,
                &mbedtls_x509_crt_profile_default, cn, flags, f_vrfy, p_vrfy ) );
}


/*
 * Verify the certificate validity, with profile
 */
int mbedtls_x509_crt_verify_with_profile( mbedtls_x509_crt *crt,
                     mbedtls_x509_crt *trust_ca,
                     mbedtls_x509_crl *ca_crl,
                     const mbedtls_x509_crt_profile *profile,
                     const char *cn, uint32_t *flags,
                     int (*f_vrfy)(void *, mbedtls_x509_crt *, int, uint32_t *),
                     void *p_vrfy )
{
    size_t cn_len;
    int ret;
    int pathlen = 0, selfsigned = 0;
    mbedtls_x509_crt *parent;
    mbedtls_x509_name *name;
    mbedtls_x509_sequence *cur = NULL;
    mbedtls_pk_type_t pk_type;

    *flags = 0;

    if( profile == NULL )
    {
        ret = MBEDTLS_ERR_X509_BAD_INPUT_DATA;
        goto exit;
    }

    if( cn != NULL )
    {
        name = &crt->subject;
        cn_len = strlen( cn );

        if( crt->ext_types & MBEDTLS_X509_EXT_SUBJECT_ALT_NAME )
        {
            cur = &crt->subject_alt_names;

            while( cur != NULL )
            {
                if( cur->buf.len == cn_len &&
                    x509_memcasecmp( cn, cur->buf.p, cn_len ) == 0 )
                    break;

                if( cur->buf.len > 2 &&
                    memcmp( cur->buf.p, "*.", 2 ) == 0 &&
                    x509_check_wildcard( cn, &cur->buf ) == 0 )
                {
                    break;
                }

                cur = cur->next;
            }

            if( cur == NULL )
                *flags |= MBEDTLS_X509_BADCERT_CN_MISMATCH;
        }
        else
        {
            while( name != NULL )
            {
                if( MBEDTLS_OID_CMP( MBEDTLS_OID_AT_CN, &name->oid ) == 0 )
                {
                    if( name->val.len == cn_len &&
                        x509_memcasecmp( name->val.p, cn, cn_len ) == 0 )
                        break;

                    if( name->val.len > 2 &&
                        memcmp( name->val.p, "*.", 2 ) == 0 &&
                        x509_check_wildcard( cn, &name->val ) == 0 )
                        break;
                }

                name = name->next;
            }

            if( name == NULL )
                *flags |= MBEDTLS_X509_BADCERT_CN_MISMATCH;
        }
    }

    /* Check the type and size of the key */
    pk_type = mbedtls_pk_get_type( &crt->pk );

    if( x509_profile_check_pk_alg( profile, pk_type ) != 0 )
        *flags |= MBEDTLS_X509_BADCERT_BAD_PK;

    if( x509_profile_check_key( profile, pk_type, &crt->pk ) != 0 )
        *flags |= MBEDTLS_X509_BADCERT_BAD_KEY;

    /* Look for a parent in trusted CAs */
    for( parent = trust_ca; parent != NULL; parent = parent->next )
    {
        if( x509_crt_check_parent( crt, parent, 0, pathlen == 0 ) == 0 )
            break;
    }

    if( parent != NULL )
    {
        ret = x509_crt_verify_top( crt, parent, ca_crl, profile,
                                   pathlen, selfsigned, flags, f_vrfy, p_vrfy );
        if( ret != 0 )
            goto exit;
    }
    else
    {
        /* Look for a parent upwards the chain */
        for( parent = crt->next; parent != NULL; parent = parent->next )
            if( x509_crt_check_parent( crt, parent, 0, pathlen == 0 ) == 0 )
                break;

        /* Are we part of the chain or at the top? */
        if( parent != NULL )
        {
            ret = x509_crt_verify_child( crt, parent, trust_ca, ca_crl, profile,
                                         pathlen, selfsigned, flags, f_vrfy, p_vrfy );
            if( ret != 0 )
                goto exit;
        }
        else
        {
            ret = x509_crt_verify_top( crt, trust_ca, ca_crl, profile,
                                       pathlen, selfsigned, flags, f_vrfy, p_vrfy );
            if( ret != 0 )
                goto exit;
        }
    }

exit:
    /* prevent misuse of the vrfy callback - VERIFY_FAILED would be ignored by
     * the SSL module for authmode optional, but non-zero return from the
     * callback means a fatal error so it shouldn't be ignored */
    if( ret == MBEDTLS_ERR_X509_CERT_VERIFY_FAILED )
        ret = MBEDTLS_ERR_X509_FATAL_ERROR;

    if( ret != 0 )
    {
        *flags = (uint32_t) -1;
        return( ret );
    }

    if( *flags != 0 )
        return( MBEDTLS_ERR_X509_CERT_VERIFY_FAILED );

    return( 0 );
}

/*
 * Initialize a certificate chain
 */
void mbedtls_x509_crt_init( mbedtls_x509_crt *crt )
{
    memset( crt, 0, sizeof(mbedtls_x509_crt) );
}

/*
 * Unallocate all certificate data
 */
void mbedtls_x509_crt_free( mbedtls_x509_crt *crt )
{
    mbedtls_x509_crt *cert_cur = crt;
    mbedtls_x509_crt *cert_prv;
    mbedtls_x509_name *name_cur;
    mbedtls_x509_name *name_prv;
    mbedtls_x509_sequence *seq_cur;
    mbedtls_x509_sequence *seq_prv;

    if( crt == NULL )
        return;

    do
    {
        mbedtls_pk_free( &cert_cur->pk );

#if defined(MBEDTLS_X509_RSASSA_PSS_SUPPORT)
        mbedtls_free( cert_cur->sig_opts );
#endif

        name_cur = cert_cur->issuer.next;
        while( name_cur != NULL )
        {
            name_prv = name_cur;
            name_cur = name_cur->next;
            mbedtls_zeroize( name_prv, sizeof( mbedtls_x509_name ) );
            mbedtls_free( name_prv );
        }

        name_cur = cert_cur->subject.next;
        while( name_cur != NULL )
        {
            name_prv = name_cur;
            name_cur = name_cur->next;
            mbedtls_zeroize( name_prv, sizeof( mbedtls_x509_name ) );
            mbedtls_free( name_prv );
        }

        seq_cur = cert_cur->ext_key_usage.next;
        while( seq_cur != NULL )
        {
            seq_prv = seq_cur;
            seq_cur = seq_cur->next;
            mbedtls_zeroize( seq_prv, sizeof( mbedtls_x509_sequence ) );
            mbedtls_free( seq_prv );
        }

        seq_cur = cert_cur->subject_alt_names.next;
        while( seq_cur != NULL )
        {
            seq_prv = seq_cur;
            seq_cur = seq_cur->next;
            mbedtls_zeroize( seq_prv, sizeof( mbedtls_x509_sequence ) );
            mbedtls_free( seq_prv );
        }

        if( cert_cur->raw.p != NULL )
        {
            mbedtls_zeroize( cert_cur->raw.p, cert_cur->raw.len );
            mbedtls_free( cert_cur->raw.p );
        }

        cert_cur = cert_cur->next;
    }
    while( cert_cur != NULL );

    cert_cur = crt;
    do
    {
        cert_prv = cert_cur;
        cert_cur = cert_cur->next;

        mbedtls_zeroize( cert_prv, sizeof( mbedtls_x509_crt ) );
        if( cert_prv != crt )
            mbedtls_free( cert_prv );
    }
    while( cert_cur != NULL );
}

#endif /* MBEDTLS_X509_CRT_PARSE_C */
