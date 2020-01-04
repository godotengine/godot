/*
 *  X.509 common functions for parsing and verification
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

#if defined(MBEDTLS_X509_USE_C)

#include "mbedtls/x509.h"
#include "mbedtls/asn1.h"
#include "mbedtls/oid.h"

#include <stdio.h>
#include <string.h>

#if defined(MBEDTLS_PEM_PARSE_C)
#include "mbedtls/pem.h"
#endif

#if defined(MBEDTLS_PLATFORM_C)
#include "mbedtls/platform.h"
#else
#include <stdio.h>
#include <stdlib.h>
#define mbedtls_free      free
#define mbedtls_calloc    calloc
#define mbedtls_printf    printf
#define mbedtls_snprintf  snprintf
#endif

#if defined(MBEDTLS_HAVE_TIME)
#include "mbedtls/platform_time.h"
#endif
#if defined(MBEDTLS_HAVE_TIME_DATE)
#include "mbedtls/platform_util.h"
#include <time.h>
#endif

#define CHECK(code) if( ( ret = ( code ) ) != 0 ){ return( ret ); }
#define CHECK_RANGE(min, max, val)                      \
    do                                                  \
    {                                                   \
        if( ( val ) < ( min ) || ( val ) > ( max ) )    \
        {                                               \
            return( ret );                              \
        }                                               \
    } while( 0 )

/*
 *  CertificateSerialNumber  ::=  INTEGER
 */
int mbedtls_x509_get_serial( unsigned char **p, const unsigned char *end,
                     mbedtls_x509_buf *serial )
{
    int ret;

    if( ( end - *p ) < 1 )
        return( MBEDTLS_ERR_X509_INVALID_SERIAL +
                MBEDTLS_ERR_ASN1_OUT_OF_DATA );

    if( **p != ( MBEDTLS_ASN1_CONTEXT_SPECIFIC | MBEDTLS_ASN1_PRIMITIVE | 2 ) &&
        **p !=   MBEDTLS_ASN1_INTEGER )
        return( MBEDTLS_ERR_X509_INVALID_SERIAL +
                MBEDTLS_ERR_ASN1_UNEXPECTED_TAG );

    serial->tag = *(*p)++;

    if( ( ret = mbedtls_asn1_get_len( p, end, &serial->len ) ) != 0 )
        return( MBEDTLS_ERR_X509_INVALID_SERIAL + ret );

    serial->p = *p;
    *p += serial->len;

    return( 0 );
}

/* Get an algorithm identifier without parameters (eg for signatures)
 *
 *  AlgorithmIdentifier  ::=  SEQUENCE  {
 *       algorithm               OBJECT IDENTIFIER,
 *       parameters              ANY DEFINED BY algorithm OPTIONAL  }
 */
int mbedtls_x509_get_alg_null( unsigned char **p, const unsigned char *end,
                       mbedtls_x509_buf *alg )
{
    int ret;

    if( ( ret = mbedtls_asn1_get_alg_null( p, end, alg ) ) != 0 )
        return( MBEDTLS_ERR_X509_INVALID_ALG + ret );

    return( 0 );
}

/*
 * Parse an algorithm identifier with (optional) parameters
 */
int mbedtls_x509_get_alg( unsigned char **p, const unsigned char *end,
                  mbedtls_x509_buf *alg, mbedtls_x509_buf *params )
{
    int ret;

    if( ( ret = mbedtls_asn1_get_alg( p, end, alg, params ) ) != 0 )
        return( MBEDTLS_ERR_X509_INVALID_ALG + ret );

    return( 0 );
}

#if defined(MBEDTLS_X509_RSASSA_PSS_SUPPORT)
/*
 * HashAlgorithm ::= AlgorithmIdentifier
 *
 * AlgorithmIdentifier  ::=  SEQUENCE  {
 *      algorithm               OBJECT IDENTIFIER,
 *      parameters              ANY DEFINED BY algorithm OPTIONAL  }
 *
 * For HashAlgorithm, parameters MUST be NULL or absent.
 */
static int x509_get_hash_alg( const mbedtls_x509_buf *alg, mbedtls_md_type_t *md_alg )
{
    int ret;
    unsigned char *p;
    const unsigned char *end;
    mbedtls_x509_buf md_oid;
    size_t len;

    /* Make sure we got a SEQUENCE and setup bounds */
    if( alg->tag != ( MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE ) )
        return( MBEDTLS_ERR_X509_INVALID_ALG +
                MBEDTLS_ERR_ASN1_UNEXPECTED_TAG );

    p = (unsigned char *) alg->p;
    end = p + alg->len;

    if( p >= end )
        return( MBEDTLS_ERR_X509_INVALID_ALG +
                MBEDTLS_ERR_ASN1_OUT_OF_DATA );

    /* Parse md_oid */
    md_oid.tag = *p;

    if( ( ret = mbedtls_asn1_get_tag( &p, end, &md_oid.len, MBEDTLS_ASN1_OID ) ) != 0 )
        return( MBEDTLS_ERR_X509_INVALID_ALG + ret );

    md_oid.p = p;
    p += md_oid.len;

    /* Get md_alg from md_oid */
    if( ( ret = mbedtls_oid_get_md_alg( &md_oid, md_alg ) ) != 0 )
        return( MBEDTLS_ERR_X509_INVALID_ALG + ret );

    /* Make sure params is absent of NULL */
    if( p == end )
        return( 0 );

    if( ( ret = mbedtls_asn1_get_tag( &p, end, &len, MBEDTLS_ASN1_NULL ) ) != 0 || len != 0 )
        return( MBEDTLS_ERR_X509_INVALID_ALG + ret );

    if( p != end )
        return( MBEDTLS_ERR_X509_INVALID_ALG +
                MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );

    return( 0 );
}

/*
 *    RSASSA-PSS-params  ::=  SEQUENCE  {
 *       hashAlgorithm     [0] HashAlgorithm DEFAULT sha1Identifier,
 *       maskGenAlgorithm  [1] MaskGenAlgorithm DEFAULT mgf1SHA1Identifier,
 *       saltLength        [2] INTEGER DEFAULT 20,
 *       trailerField      [3] INTEGER DEFAULT 1  }
 *    -- Note that the tags in this Sequence are explicit.
 *
 * RFC 4055 (which defines use of RSASSA-PSS in PKIX) states that the value
 * of trailerField MUST be 1, and PKCS#1 v2.2 doesn't even define any other
 * option. Enfore this at parsing time.
 */
int mbedtls_x509_get_rsassa_pss_params( const mbedtls_x509_buf *params,
                                mbedtls_md_type_t *md_alg, mbedtls_md_type_t *mgf_md,
                                int *salt_len )
{
    int ret;
    unsigned char *p;
    const unsigned char *end, *end2;
    size_t len;
    mbedtls_x509_buf alg_id, alg_params;

    /* First set everything to defaults */
    *md_alg = MBEDTLS_MD_SHA1;
    *mgf_md = MBEDTLS_MD_SHA1;
    *salt_len = 20;

    /* Make sure params is a SEQUENCE and setup bounds */
    if( params->tag != ( MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE ) )
        return( MBEDTLS_ERR_X509_INVALID_ALG +
                MBEDTLS_ERR_ASN1_UNEXPECTED_TAG );

    p = (unsigned char *) params->p;
    end = p + params->len;

    if( p == end )
        return( 0 );

    /*
     * HashAlgorithm
     */
    if( ( ret = mbedtls_asn1_get_tag( &p, end, &len,
                    MBEDTLS_ASN1_CONTEXT_SPECIFIC | MBEDTLS_ASN1_CONSTRUCTED | 0 ) ) == 0 )
    {
        end2 = p + len;

        /* HashAlgorithm ::= AlgorithmIdentifier (without parameters) */
        if( ( ret = mbedtls_x509_get_alg_null( &p, end2, &alg_id ) ) != 0 )
            return( ret );

        if( ( ret = mbedtls_oid_get_md_alg( &alg_id, md_alg ) ) != 0 )
            return( MBEDTLS_ERR_X509_INVALID_ALG + ret );

        if( p != end2 )
            return( MBEDTLS_ERR_X509_INVALID_ALG +
                    MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );
    }
    else if( ret != MBEDTLS_ERR_ASN1_UNEXPECTED_TAG )
        return( MBEDTLS_ERR_X509_INVALID_ALG + ret );

    if( p == end )
        return( 0 );

    /*
     * MaskGenAlgorithm
     */
    if( ( ret = mbedtls_asn1_get_tag( &p, end, &len,
                    MBEDTLS_ASN1_CONTEXT_SPECIFIC | MBEDTLS_ASN1_CONSTRUCTED | 1 ) ) == 0 )
    {
        end2 = p + len;

        /* MaskGenAlgorithm ::= AlgorithmIdentifier (params = HashAlgorithm) */
        if( ( ret = mbedtls_x509_get_alg( &p, end2, &alg_id, &alg_params ) ) != 0 )
            return( ret );

        /* Only MFG1 is recognised for now */
        if( MBEDTLS_OID_CMP( MBEDTLS_OID_MGF1, &alg_id ) != 0 )
            return( MBEDTLS_ERR_X509_FEATURE_UNAVAILABLE +
                    MBEDTLS_ERR_OID_NOT_FOUND );

        /* Parse HashAlgorithm */
        if( ( ret = x509_get_hash_alg( &alg_params, mgf_md ) ) != 0 )
            return( ret );

        if( p != end2 )
            return( MBEDTLS_ERR_X509_INVALID_ALG +
                    MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );
    }
    else if( ret != MBEDTLS_ERR_ASN1_UNEXPECTED_TAG )
        return( MBEDTLS_ERR_X509_INVALID_ALG + ret );

    if( p == end )
        return( 0 );

    /*
     * salt_len
     */
    if( ( ret = mbedtls_asn1_get_tag( &p, end, &len,
                    MBEDTLS_ASN1_CONTEXT_SPECIFIC | MBEDTLS_ASN1_CONSTRUCTED | 2 ) ) == 0 )
    {
        end2 = p + len;

        if( ( ret = mbedtls_asn1_get_int( &p, end2, salt_len ) ) != 0 )
            return( MBEDTLS_ERR_X509_INVALID_ALG + ret );

        if( p != end2 )
            return( MBEDTLS_ERR_X509_INVALID_ALG +
                    MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );
    }
    else if( ret != MBEDTLS_ERR_ASN1_UNEXPECTED_TAG )
        return( MBEDTLS_ERR_X509_INVALID_ALG + ret );

    if( p == end )
        return( 0 );

    /*
     * trailer_field (if present, must be 1)
     */
    if( ( ret = mbedtls_asn1_get_tag( &p, end, &len,
                    MBEDTLS_ASN1_CONTEXT_SPECIFIC | MBEDTLS_ASN1_CONSTRUCTED | 3 ) ) == 0 )
    {
        int trailer_field;

        end2 = p + len;

        if( ( ret = mbedtls_asn1_get_int( &p, end2, &trailer_field ) ) != 0 )
            return( MBEDTLS_ERR_X509_INVALID_ALG + ret );

        if( p != end2 )
            return( MBEDTLS_ERR_X509_INVALID_ALG +
                    MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );

        if( trailer_field != 1 )
            return( MBEDTLS_ERR_X509_INVALID_ALG );
    }
    else if( ret != MBEDTLS_ERR_ASN1_UNEXPECTED_TAG )
        return( MBEDTLS_ERR_X509_INVALID_ALG + ret );

    if( p != end )
        return( MBEDTLS_ERR_X509_INVALID_ALG +
                MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );

    return( 0 );
}
#endif /* MBEDTLS_X509_RSASSA_PSS_SUPPORT */

/*
 *  AttributeTypeAndValue ::= SEQUENCE {
 *    type     AttributeType,
 *    value    AttributeValue }
 *
 *  AttributeType ::= OBJECT IDENTIFIER
 *
 *  AttributeValue ::= ANY DEFINED BY AttributeType
 */
static int x509_get_attr_type_value( unsigned char **p,
                                     const unsigned char *end,
                                     mbedtls_x509_name *cur )
{
    int ret;
    size_t len;
    mbedtls_x509_buf *oid;
    mbedtls_x509_buf *val;

    if( ( ret = mbedtls_asn1_get_tag( p, end, &len,
            MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE ) ) != 0 )
        return( MBEDTLS_ERR_X509_INVALID_NAME + ret );

    end = *p + len;

    if( ( end - *p ) < 1 )
        return( MBEDTLS_ERR_X509_INVALID_NAME +
                MBEDTLS_ERR_ASN1_OUT_OF_DATA );

    oid = &cur->oid;
    oid->tag = **p;

    if( ( ret = mbedtls_asn1_get_tag( p, end, &oid->len, MBEDTLS_ASN1_OID ) ) != 0 )
        return( MBEDTLS_ERR_X509_INVALID_NAME + ret );

    oid->p = *p;
    *p += oid->len;

    if( ( end - *p ) < 1 )
        return( MBEDTLS_ERR_X509_INVALID_NAME +
                MBEDTLS_ERR_ASN1_OUT_OF_DATA );

    if( **p != MBEDTLS_ASN1_BMP_STRING && **p != MBEDTLS_ASN1_UTF8_STRING      &&
        **p != MBEDTLS_ASN1_T61_STRING && **p != MBEDTLS_ASN1_PRINTABLE_STRING &&
        **p != MBEDTLS_ASN1_IA5_STRING && **p != MBEDTLS_ASN1_UNIVERSAL_STRING &&
        **p != MBEDTLS_ASN1_BIT_STRING )
        return( MBEDTLS_ERR_X509_INVALID_NAME +
                MBEDTLS_ERR_ASN1_UNEXPECTED_TAG );

    val = &cur->val;
    val->tag = *(*p)++;

    if( ( ret = mbedtls_asn1_get_len( p, end, &val->len ) ) != 0 )
        return( MBEDTLS_ERR_X509_INVALID_NAME + ret );

    val->p = *p;
    *p += val->len;

    if( *p != end )
    {
        return( MBEDTLS_ERR_X509_INVALID_NAME +
                MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );
    }

    cur->next = NULL;

    return( 0 );
}

/*
 *  Name ::= CHOICE { -- only one possibility for now --
 *       rdnSequence  RDNSequence }
 *
 *  RDNSequence ::= SEQUENCE OF RelativeDistinguishedName
 *
 *  RelativeDistinguishedName ::=
 *    SET OF AttributeTypeAndValue
 *
 *  AttributeTypeAndValue ::= SEQUENCE {
 *    type     AttributeType,
 *    value    AttributeValue }
 *
 *  AttributeType ::= OBJECT IDENTIFIER
 *
 *  AttributeValue ::= ANY DEFINED BY AttributeType
 *
 * The data structure is optimized for the common case where each RDN has only
 * one element, which is represented as a list of AttributeTypeAndValue.
 * For the general case we still use a flat list, but we mark elements of the
 * same set so that they are "merged" together in the functions that consume
 * this list, eg mbedtls_x509_dn_gets().
 */
int mbedtls_x509_get_name( unsigned char **p, const unsigned char *end,
                   mbedtls_x509_name *cur )
{
    int ret;
    size_t set_len;
    const unsigned char *end_set;

    /* don't use recursion, we'd risk stack overflow if not optimized */
    while( 1 )
    {
        /*
         * parse SET
         */
        if( ( ret = mbedtls_asn1_get_tag( p, end, &set_len,
                MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SET ) ) != 0 )
            return( MBEDTLS_ERR_X509_INVALID_NAME + ret );

        end_set  = *p + set_len;

        while( 1 )
        {
            if( ( ret = x509_get_attr_type_value( p, end_set, cur ) ) != 0 )
                return( ret );

            if( *p == end_set )
                break;

            /* Mark this item as being no the only one in a set */
            cur->next_merged = 1;

            cur->next = mbedtls_calloc( 1, sizeof( mbedtls_x509_name ) );

            if( cur->next == NULL )
                return( MBEDTLS_ERR_X509_ALLOC_FAILED );

            cur = cur->next;
        }

        /*
         * continue until end of SEQUENCE is reached
         */
        if( *p == end )
            return( 0 );

        cur->next = mbedtls_calloc( 1, sizeof( mbedtls_x509_name ) );

        if( cur->next == NULL )
            return( MBEDTLS_ERR_X509_ALLOC_FAILED );

        cur = cur->next;
    }
}

static int x509_parse_int( unsigned char **p, size_t n, int *res )
{
    *res = 0;

    for( ; n > 0; --n )
    {
        if( ( **p < '0') || ( **p > '9' ) )
            return ( MBEDTLS_ERR_X509_INVALID_DATE );

        *res *= 10;
        *res += ( *(*p)++ - '0' );
    }

    return( 0 );
}

static int x509_date_is_valid(const mbedtls_x509_time *t )
{
    int ret = MBEDTLS_ERR_X509_INVALID_DATE;
    int month_len;

    CHECK_RANGE( 0, 9999, t->year );
    CHECK_RANGE( 0, 23,   t->hour );
    CHECK_RANGE( 0, 59,   t->min  );
    CHECK_RANGE( 0, 59,   t->sec  );

    switch( t->mon )
    {
        case 1: case 3: case 5: case 7: case 8: case 10: case 12:
            month_len = 31;
            break;
        case 4: case 6: case 9: case 11:
            month_len = 30;
            break;
        case 2:
            if( ( !( t->year % 4 ) && t->year % 100 ) ||
                !( t->year % 400 ) )
                month_len = 29;
            else
                month_len = 28;
            break;
        default:
            return( ret );
    }
    CHECK_RANGE( 1, month_len, t->day );

    return( 0 );
}

/*
 * Parse an ASN1_UTC_TIME (yearlen=2) or ASN1_GENERALIZED_TIME (yearlen=4)
 * field.
 */
static int x509_parse_time( unsigned char **p, size_t len, size_t yearlen,
                            mbedtls_x509_time *tm )
{
    int ret;

    /*
     * Minimum length is 10 or 12 depending on yearlen
     */
    if ( len < yearlen + 8 )
        return ( MBEDTLS_ERR_X509_INVALID_DATE );
    len -= yearlen + 8;

    /*
     * Parse year, month, day, hour, minute
     */
    CHECK( x509_parse_int( p, yearlen, &tm->year ) );
    if ( 2 == yearlen )
    {
        if ( tm->year < 50 )
            tm->year += 100;

        tm->year += 1900;
    }

    CHECK( x509_parse_int( p, 2, &tm->mon ) );
    CHECK( x509_parse_int( p, 2, &tm->day ) );
    CHECK( x509_parse_int( p, 2, &tm->hour ) );
    CHECK( x509_parse_int( p, 2, &tm->min ) );

    /*
     * Parse seconds if present
     */
    if ( len >= 2 )
    {
        CHECK( x509_parse_int( p, 2, &tm->sec ) );
        len -= 2;
    }
    else
        return ( MBEDTLS_ERR_X509_INVALID_DATE );

    /*
     * Parse trailing 'Z' if present
     */
    if ( 1 == len && 'Z' == **p )
    {
        (*p)++;
        len--;
    }

    /*
     * We should have parsed all characters at this point
     */
    if ( 0 != len )
        return ( MBEDTLS_ERR_X509_INVALID_DATE );

    CHECK( x509_date_is_valid( tm ) );

    return ( 0 );
}

/*
 *  Time ::= CHOICE {
 *       utcTime        UTCTime,
 *       generalTime    GeneralizedTime }
 */
int mbedtls_x509_get_time( unsigned char **p, const unsigned char *end,
                           mbedtls_x509_time *tm )
{
    int ret;
    size_t len, year_len;
    unsigned char tag;

    if( ( end - *p ) < 1 )
        return( MBEDTLS_ERR_X509_INVALID_DATE +
                MBEDTLS_ERR_ASN1_OUT_OF_DATA );

    tag = **p;

    if( tag == MBEDTLS_ASN1_UTC_TIME )
        year_len = 2;
    else if( tag == MBEDTLS_ASN1_GENERALIZED_TIME )
        year_len = 4;
    else
        return( MBEDTLS_ERR_X509_INVALID_DATE +
                MBEDTLS_ERR_ASN1_UNEXPECTED_TAG );

    (*p)++;
    ret = mbedtls_asn1_get_len( p, end, &len );

    if( ret != 0 )
        return( MBEDTLS_ERR_X509_INVALID_DATE + ret );

    return x509_parse_time( p, len, year_len, tm );
}

int mbedtls_x509_get_sig( unsigned char **p, const unsigned char *end, mbedtls_x509_buf *sig )
{
    int ret;
    size_t len;
    int tag_type;

    if( ( end - *p ) < 1 )
        return( MBEDTLS_ERR_X509_INVALID_SIGNATURE +
                MBEDTLS_ERR_ASN1_OUT_OF_DATA );

    tag_type = **p;

    if( ( ret = mbedtls_asn1_get_bitstring_null( p, end, &len ) ) != 0 )
        return( MBEDTLS_ERR_X509_INVALID_SIGNATURE + ret );

    sig->tag = tag_type;
    sig->len = len;
    sig->p = *p;

    *p += len;

    return( 0 );
}

/*
 * Get signature algorithm from alg OID and optional parameters
 */
int mbedtls_x509_get_sig_alg( const mbedtls_x509_buf *sig_oid, const mbedtls_x509_buf *sig_params,
                      mbedtls_md_type_t *md_alg, mbedtls_pk_type_t *pk_alg,
                      void **sig_opts )
{
    int ret;

    if( *sig_opts != NULL )
        return( MBEDTLS_ERR_X509_BAD_INPUT_DATA );

    if( ( ret = mbedtls_oid_get_sig_alg( sig_oid, md_alg, pk_alg ) ) != 0 )
        return( MBEDTLS_ERR_X509_UNKNOWN_SIG_ALG + ret );

#if defined(MBEDTLS_X509_RSASSA_PSS_SUPPORT)
    if( *pk_alg == MBEDTLS_PK_RSASSA_PSS )
    {
        mbedtls_pk_rsassa_pss_options *pss_opts;

        pss_opts = mbedtls_calloc( 1, sizeof( mbedtls_pk_rsassa_pss_options ) );
        if( pss_opts == NULL )
            return( MBEDTLS_ERR_X509_ALLOC_FAILED );

        ret = mbedtls_x509_get_rsassa_pss_params( sig_params,
                                          md_alg,
                                          &pss_opts->mgf1_hash_id,
                                          &pss_opts->expected_salt_len );
        if( ret != 0 )
        {
            mbedtls_free( pss_opts );
            return( ret );
        }

        *sig_opts = (void *) pss_opts;
    }
    else
#endif /* MBEDTLS_X509_RSASSA_PSS_SUPPORT */
    {
        /* Make sure parameters are absent or NULL */
        if( ( sig_params->tag != MBEDTLS_ASN1_NULL && sig_params->tag != 0 ) ||
              sig_params->len != 0 )
        return( MBEDTLS_ERR_X509_INVALID_ALG );
    }

    return( 0 );
}

/*
 * X.509 Extensions (No parsing of extensions, pointer should
 * be either manually updated or extensions should be parsed!)
 */
int mbedtls_x509_get_ext( unsigned char **p, const unsigned char *end,
                          mbedtls_x509_buf *ext, int tag )
{
    int ret;
    size_t len;

    /* Extension structure use EXPLICIT tagging. That is, the actual
     * `Extensions` structure is wrapped by a tag-length pair using
     * the respective context-specific tag. */
    ret = mbedtls_asn1_get_tag( p, end, &ext->len,
              MBEDTLS_ASN1_CONTEXT_SPECIFIC | MBEDTLS_ASN1_CONSTRUCTED | tag );
    if( ret != 0 )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );

    ext->tag = MBEDTLS_ASN1_CONTEXT_SPECIFIC | MBEDTLS_ASN1_CONSTRUCTED | tag;
    ext->p   = *p;
    end      = *p + ext->len;

    /*
     * Extensions  ::=  SEQUENCE SIZE (1..MAX) OF Extension
     */
    if( ( ret = mbedtls_asn1_get_tag( p, end, &len,
            MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE ) ) != 0 )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );

    if( end != *p + len )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );

    return( 0 );
}

/*
 * Store the name in printable form into buf; no more
 * than size characters will be written
 */
int mbedtls_x509_dn_gets( char *buf, size_t size, const mbedtls_x509_name *dn )
{
    int ret;
    size_t i, n;
    unsigned char c, merge = 0;
    const mbedtls_x509_name *name;
    const char *short_name = NULL;
    char s[MBEDTLS_X509_MAX_DN_NAME_SIZE], *p;

    memset( s, 0, sizeof( s ) );

    name = dn;
    p = buf;
    n = size;

    while( name != NULL )
    {
        if( !name->oid.p )
        {
            name = name->next;
            continue;
        }

        if( name != dn )
        {
            ret = mbedtls_snprintf( p, n, merge ? " + " : ", " );
            MBEDTLS_X509_SAFE_SNPRINTF;
        }

        ret = mbedtls_oid_get_attr_short_name( &name->oid, &short_name );

        if( ret == 0 )
            ret = mbedtls_snprintf( p, n, "%s=", short_name );
        else
            ret = mbedtls_snprintf( p, n, "\?\?=" );
        MBEDTLS_X509_SAFE_SNPRINTF;

        for( i = 0; i < name->val.len; i++ )
        {
            if( i >= sizeof( s ) - 1 )
                break;

            c = name->val.p[i];
            if( c < 32 || c == 127 || ( c > 128 && c < 160 ) )
                 s[i] = '?';
            else s[i] = c;
        }
        s[i] = '\0';
        ret = mbedtls_snprintf( p, n, "%s", s );
        MBEDTLS_X509_SAFE_SNPRINTF;

        merge = name->next_merged;
        name = name->next;
    }

    return( (int) ( size - n ) );
}

/*
 * Store the serial in printable form into buf; no more
 * than size characters will be written
 */
int mbedtls_x509_serial_gets( char *buf, size_t size, const mbedtls_x509_buf *serial )
{
    int ret;
    size_t i, n, nr;
    char *p;

    p = buf;
    n = size;

    nr = ( serial->len <= 32 )
        ? serial->len  : 28;

    for( i = 0; i < nr; i++ )
    {
        if( i == 0 && nr > 1 && serial->p[i] == 0x0 )
            continue;

        ret = mbedtls_snprintf( p, n, "%02X%s",
                serial->p[i], ( i < nr - 1 ) ? ":" : "" );
        MBEDTLS_X509_SAFE_SNPRINTF;
    }

    if( nr != serial->len )
    {
        ret = mbedtls_snprintf( p, n, "...." );
        MBEDTLS_X509_SAFE_SNPRINTF;
    }

    return( (int) ( size - n ) );
}

/*
 * Helper for writing signature algorithms
 */
int mbedtls_x509_sig_alg_gets( char *buf, size_t size, const mbedtls_x509_buf *sig_oid,
                       mbedtls_pk_type_t pk_alg, mbedtls_md_type_t md_alg,
                       const void *sig_opts )
{
    int ret;
    char *p = buf;
    size_t n = size;
    const char *desc = NULL;

    ret = mbedtls_oid_get_sig_alg_desc( sig_oid, &desc );
    if( ret != 0 )
        ret = mbedtls_snprintf( p, n, "???"  );
    else
        ret = mbedtls_snprintf( p, n, "%s", desc );
    MBEDTLS_X509_SAFE_SNPRINTF;

#if defined(MBEDTLS_X509_RSASSA_PSS_SUPPORT)
    if( pk_alg == MBEDTLS_PK_RSASSA_PSS )
    {
        const mbedtls_pk_rsassa_pss_options *pss_opts;
        const mbedtls_md_info_t *md_info, *mgf_md_info;

        pss_opts = (const mbedtls_pk_rsassa_pss_options *) sig_opts;

        md_info = mbedtls_md_info_from_type( md_alg );
        mgf_md_info = mbedtls_md_info_from_type( pss_opts->mgf1_hash_id );

        ret = mbedtls_snprintf( p, n, " (%s, MGF1-%s, 0x%02X)",
                              md_info ? mbedtls_md_get_name( md_info ) : "???",
                              mgf_md_info ? mbedtls_md_get_name( mgf_md_info ) : "???",
                              pss_opts->expected_salt_len );
        MBEDTLS_X509_SAFE_SNPRINTF;
    }
#else
    ((void) pk_alg);
    ((void) md_alg);
    ((void) sig_opts);
#endif /* MBEDTLS_X509_RSASSA_PSS_SUPPORT */

    return( (int)( size - n ) );
}

/*
 * Helper for writing "RSA key size", "EC key size", etc
 */
int mbedtls_x509_key_size_helper( char *buf, size_t buf_size, const char *name )
{
    char *p = buf;
    size_t n = buf_size;
    int ret;

    ret = mbedtls_snprintf( p, n, "%s key size", name );
    MBEDTLS_X509_SAFE_SNPRINTF;

    return( 0 );
}

#if defined(MBEDTLS_HAVE_TIME_DATE)
/*
 * Set the time structure to the current time.
 * Return 0 on success, non-zero on failure.
 */
static int x509_get_current_time( mbedtls_x509_time *now )
{
    struct tm *lt, tm_buf;
    mbedtls_time_t tt;
    int ret = 0;

    tt = mbedtls_time( NULL );
    lt = mbedtls_platform_gmtime_r( &tt, &tm_buf );

    if( lt == NULL )
        ret = -1;
    else
    {
        now->year = lt->tm_year + 1900;
        now->mon  = lt->tm_mon  + 1;
        now->day  = lt->tm_mday;
        now->hour = lt->tm_hour;
        now->min  = lt->tm_min;
        now->sec  = lt->tm_sec;
    }

    return( ret );
}

/*
 * Return 0 if before <= after, 1 otherwise
 */
static int x509_check_time( const mbedtls_x509_time *before, const mbedtls_x509_time *after )
{
    if( before->year  > after->year )
        return( 1 );

    if( before->year == after->year &&
        before->mon   > after->mon )
        return( 1 );

    if( before->year == after->year &&
        before->mon  == after->mon  &&
        before->day   > after->day )
        return( 1 );

    if( before->year == after->year &&
        before->mon  == after->mon  &&
        before->day  == after->day  &&
        before->hour  > after->hour )
        return( 1 );

    if( before->year == after->year &&
        before->mon  == after->mon  &&
        before->day  == after->day  &&
        before->hour == after->hour &&
        before->min   > after->min  )
        return( 1 );

    if( before->year == after->year &&
        before->mon  == after->mon  &&
        before->day  == after->day  &&
        before->hour == after->hour &&
        before->min  == after->min  &&
        before->sec   > after->sec  )
        return( 1 );

    return( 0 );
}

int mbedtls_x509_time_is_past( const mbedtls_x509_time *to )
{
    mbedtls_x509_time now;

    if( x509_get_current_time( &now ) != 0 )
        return( 1 );

    return( x509_check_time( &now, to ) );
}

int mbedtls_x509_time_is_future( const mbedtls_x509_time *from )
{
    mbedtls_x509_time now;

    if( x509_get_current_time( &now ) != 0 )
        return( 1 );

    return( x509_check_time( from, &now ) );
}

#else  /* MBEDTLS_HAVE_TIME_DATE */

int mbedtls_x509_time_is_past( const mbedtls_x509_time *to )
{
    ((void) to);
    return( 0 );
}

int mbedtls_x509_time_is_future( const mbedtls_x509_time *from )
{
    ((void) from);
    return( 0 );
}
#endif /* MBEDTLS_HAVE_TIME_DATE */

#if defined(MBEDTLS_SELF_TEST)

#include "mbedtls/x509_crt.h"
#include "mbedtls/certs.h"

/*
 * Checkup routine
 */
int mbedtls_x509_self_test( int verbose )
{
    int ret = 0;
#if defined(MBEDTLS_CERTS_C) && defined(MBEDTLS_SHA256_C)
    uint32_t flags;
    mbedtls_x509_crt cacert;
    mbedtls_x509_crt clicert;

    if( verbose != 0 )
        mbedtls_printf( "  X.509 certificate load: " );

    mbedtls_x509_crt_init( &cacert );
    mbedtls_x509_crt_init( &clicert );

    ret = mbedtls_x509_crt_parse( &clicert, (const unsigned char *) mbedtls_test_cli_crt,
                           mbedtls_test_cli_crt_len );
    if( ret != 0 )
    {
        if( verbose != 0 )
            mbedtls_printf( "failed\n" );

        goto cleanup;
    }

    ret = mbedtls_x509_crt_parse( &cacert, (const unsigned char *) mbedtls_test_ca_crt,
                          mbedtls_test_ca_crt_len );
    if( ret != 0 )
    {
        if( verbose != 0 )
            mbedtls_printf( "failed\n" );

        goto cleanup;
    }

    if( verbose != 0 )
        mbedtls_printf( "passed\n  X.509 signature verify: ");

    ret = mbedtls_x509_crt_verify( &clicert, &cacert, NULL, NULL, &flags, NULL, NULL );
    if( ret != 0 )
    {
        if( verbose != 0 )
            mbedtls_printf( "failed\n" );

        goto cleanup;
    }

    if( verbose != 0 )
        mbedtls_printf( "passed\n\n");

cleanup:
    mbedtls_x509_crt_free( &cacert  );
    mbedtls_x509_crt_free( &clicert );
#else
    ((void) verbose);
#endif /* MBEDTLS_CERTS_C && MBEDTLS_SHA1_C */
    return( ret );
}

#endif /* MBEDTLS_SELF_TEST */

#endif /* MBEDTLS_X509_USE_C */
