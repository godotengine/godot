/*
 *  X.509 base functions for creating certificates / CSRs
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

#if defined(MBEDTLS_X509_CREATE_C)

#include "mbedtls/x509.h"
#include "mbedtls/asn1write.h"
#include "mbedtls/oid.h"

#include <string.h>

typedef struct {
    const char *name;
    size_t name_len;
    const char*oid;
} x509_attr_descriptor_t;

#define ADD_STRLEN( s )     s, sizeof( s ) - 1

static const x509_attr_descriptor_t x509_attrs[] =
{
    { ADD_STRLEN( "CN" ),                       MBEDTLS_OID_AT_CN },
    { ADD_STRLEN( "commonName" ),               MBEDTLS_OID_AT_CN },
    { ADD_STRLEN( "C" ),                        MBEDTLS_OID_AT_COUNTRY },
    { ADD_STRLEN( "countryName" ),              MBEDTLS_OID_AT_COUNTRY },
    { ADD_STRLEN( "O" ),                        MBEDTLS_OID_AT_ORGANIZATION },
    { ADD_STRLEN( "organizationName" ),         MBEDTLS_OID_AT_ORGANIZATION },
    { ADD_STRLEN( "L" ),                        MBEDTLS_OID_AT_LOCALITY },
    { ADD_STRLEN( "locality" ),                 MBEDTLS_OID_AT_LOCALITY },
    { ADD_STRLEN( "R" ),                        MBEDTLS_OID_PKCS9_EMAIL },
    { ADD_STRLEN( "OU" ),                       MBEDTLS_OID_AT_ORG_UNIT },
    { ADD_STRLEN( "organizationalUnitName" ),   MBEDTLS_OID_AT_ORG_UNIT },
    { ADD_STRLEN( "ST" ),                       MBEDTLS_OID_AT_STATE },
    { ADD_STRLEN( "stateOrProvinceName" ),      MBEDTLS_OID_AT_STATE },
    { ADD_STRLEN( "emailAddress" ),             MBEDTLS_OID_PKCS9_EMAIL },
    { ADD_STRLEN( "serialNumber" ),             MBEDTLS_OID_AT_SERIAL_NUMBER },
    { ADD_STRLEN( "postalAddress" ),            MBEDTLS_OID_AT_POSTAL_ADDRESS },
    { ADD_STRLEN( "postalCode" ),               MBEDTLS_OID_AT_POSTAL_CODE },
    { ADD_STRLEN( "dnQualifier" ),              MBEDTLS_OID_AT_DN_QUALIFIER },
    { ADD_STRLEN( "title" ),                    MBEDTLS_OID_AT_TITLE },
    { ADD_STRLEN( "surName" ),                  MBEDTLS_OID_AT_SUR_NAME },
    { ADD_STRLEN( "SN" ),                       MBEDTLS_OID_AT_SUR_NAME },
    { ADD_STRLEN( "givenName" ),                MBEDTLS_OID_AT_GIVEN_NAME },
    { ADD_STRLEN( "GN" ),                       MBEDTLS_OID_AT_GIVEN_NAME },
    { ADD_STRLEN( "initials" ),                 MBEDTLS_OID_AT_INITIALS },
    { ADD_STRLEN( "pseudonym" ),                MBEDTLS_OID_AT_PSEUDONYM },
    { ADD_STRLEN( "generationQualifier" ),      MBEDTLS_OID_AT_GENERATION_QUALIFIER },
    { ADD_STRLEN( "domainComponent" ),          MBEDTLS_OID_DOMAIN_COMPONENT },
    { ADD_STRLEN( "DC" ),                       MBEDTLS_OID_DOMAIN_COMPONENT },
    { NULL, 0, NULL }
};

static const char *x509_at_oid_from_name( const char *name, size_t name_len )
{
    const x509_attr_descriptor_t *cur;

    for( cur = x509_attrs; cur->name != NULL; cur++ )
        if( cur->name_len == name_len &&
            strncmp( cur->name, name, name_len ) == 0 )
            break;

    return( cur->oid );
}

int mbedtls_x509_string_to_names( mbedtls_asn1_named_data **head, const char *name )
{
    int ret = 0;
    const char *s = name, *c = s;
    const char *end = s + strlen( s );
    const char *oid = NULL;
    int in_tag = 1;
    char data[MBEDTLS_X509_MAX_DN_NAME_SIZE];
    char *d = data;

    /* Clear existing chain if present */
    mbedtls_asn1_free_named_data_list( head );

    while( c <= end )
    {
        if( in_tag && *c == '=' )
        {
            if( ( oid = x509_at_oid_from_name( s, c - s ) ) == NULL )
            {
                ret = MBEDTLS_ERR_X509_UNKNOWN_OID;
                goto exit;
            }

            s = c + 1;
            in_tag = 0;
            d = data;
        }

        if( !in_tag && *c == '\\' && c != end )
        {
            c++;

            /* Check for valid escaped characters */
            if( c == end || *c != ',' )
            {
                ret = MBEDTLS_ERR_X509_INVALID_NAME;
                goto exit;
            }
        }
        else if( !in_tag && ( *c == ',' || c == end ) )
        {
            if( mbedtls_asn1_store_named_data( head, oid, strlen( oid ),
                                       (unsigned char *) data,
                                       d - data ) == NULL )
            {
                return( MBEDTLS_ERR_X509_ALLOC_FAILED );
            }

            while( c < end && *(c + 1) == ' ' )
                c++;

            s = c + 1;
            in_tag = 1;
        }

        if( !in_tag && s != c + 1 )
        {
            *(d++) = *c;

            if( d - data == MBEDTLS_X509_MAX_DN_NAME_SIZE )
            {
                ret = MBEDTLS_ERR_X509_INVALID_NAME;
                goto exit;
            }
        }

        c++;
    }

exit:

    return( ret );
}

/* The first byte of the value in the mbedtls_asn1_named_data structure is reserved
 * to store the critical boolean for us
 */
int mbedtls_x509_set_extension( mbedtls_asn1_named_data **head, const char *oid, size_t oid_len,
                        int critical, const unsigned char *val, size_t val_len )
{
    mbedtls_asn1_named_data *cur;

    if( ( cur = mbedtls_asn1_store_named_data( head, oid, oid_len,
                                       NULL, val_len + 1 ) ) == NULL )
    {
        return( MBEDTLS_ERR_X509_ALLOC_FAILED );
    }

    cur->val.p[0] = critical;
    memcpy( cur->val.p + 1, val, val_len );

    return( 0 );
}

/*
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
 */
static int x509_write_name( unsigned char **p, unsigned char *start,
                            const char *oid, size_t oid_len,
                            const unsigned char *name, size_t name_len )
{
    int ret;
    size_t len = 0;

    // Write PrintableString for all except MBEDTLS_OID_PKCS9_EMAIL
    //
    if( MBEDTLS_OID_SIZE( MBEDTLS_OID_PKCS9_EMAIL ) == oid_len &&
        memcmp( oid, MBEDTLS_OID_PKCS9_EMAIL, oid_len ) == 0 )
    {
        MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_ia5_string( p, start,
                                                  (const char *) name,
                                                  name_len ) );
    }
    else
    {
        MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_printable_string( p, start,
                                                        (const char *) name,
                                                        name_len ) );
    }

    // Write OID
    //
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_oid( p, start, oid, oid_len ) );

    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( p, start, len ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( p, start, MBEDTLS_ASN1_CONSTRUCTED |
                                                 MBEDTLS_ASN1_SEQUENCE ) );

    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( p, start, len ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( p, start, MBEDTLS_ASN1_CONSTRUCTED |
                                                 MBEDTLS_ASN1_SET ) );

    return( (int) len );
}

int mbedtls_x509_write_names( unsigned char **p, unsigned char *start,
                      mbedtls_asn1_named_data *first )
{
    int ret;
    size_t len = 0;
    mbedtls_asn1_named_data *cur = first;

    while( cur != NULL )
    {
        MBEDTLS_ASN1_CHK_ADD( len, x509_write_name( p, start, (char *) cur->oid.p,
                                            cur->oid.len,
                                            cur->val.p, cur->val.len ) );
        cur = cur->next;
    }

    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( p, start, len ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( p, start, MBEDTLS_ASN1_CONSTRUCTED |
                                                 MBEDTLS_ASN1_SEQUENCE ) );

    return( (int) len );
}

int mbedtls_x509_write_sig( unsigned char **p, unsigned char *start,
                    const char *oid, size_t oid_len,
                    unsigned char *sig, size_t size )
{
    int ret;
    size_t len = 0;

    if( *p < start || (size_t)( *p - start ) < size )
        return( MBEDTLS_ERR_ASN1_BUF_TOO_SMALL );

    len = size;
    (*p) -= len;
    memcpy( *p, sig, len );

    if( *p - start < 1 )
        return( MBEDTLS_ERR_ASN1_BUF_TOO_SMALL );

    *--(*p) = 0;
    len += 1;

    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( p, start, len ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( p, start, MBEDTLS_ASN1_BIT_STRING ) );

    // Write OID
    //
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_algorithm_identifier( p, start, oid,
                                                        oid_len, 0 ) );

    return( (int) len );
}

static int x509_write_extension( unsigned char **p, unsigned char *start,
                                 mbedtls_asn1_named_data *ext )
{
    int ret;
    size_t len = 0;

    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_raw_buffer( p, start, ext->val.p + 1,
                                              ext->val.len - 1 ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( p, start, ext->val.len - 1 ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( p, start, MBEDTLS_ASN1_OCTET_STRING ) );

    if( ext->val.p[0] != 0 )
    {
        MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_bool( p, start, 1 ) );
    }

    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_raw_buffer( p, start, ext->oid.p,
                                              ext->oid.len ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( p, start, ext->oid.len ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( p, start, MBEDTLS_ASN1_OID ) );

    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( p, start, len ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( p, start, MBEDTLS_ASN1_CONSTRUCTED |
                                                 MBEDTLS_ASN1_SEQUENCE ) );

    return( (int) len );
}

/*
 * Extension  ::=  SEQUENCE  {
 *     extnID      OBJECT IDENTIFIER,
 *     critical    BOOLEAN DEFAULT FALSE,
 *     extnValue   OCTET STRING
 *                 -- contains the DER encoding of an ASN.1 value
 *                 -- corresponding to the extension type identified
 *                 -- by extnID
 *     }
 */
int mbedtls_x509_write_extensions( unsigned char **p, unsigned char *start,
                           mbedtls_asn1_named_data *first )
{
    int ret;
    size_t len = 0;
    mbedtls_asn1_named_data *cur_ext = first;

    while( cur_ext != NULL )
    {
        MBEDTLS_ASN1_CHK_ADD( len, x509_write_extension( p, start, cur_ext ) );
        cur_ext = cur_ext->next;
    }

    return( (int) len );
}

#endif /* MBEDTLS_X509_CREATE_C */
