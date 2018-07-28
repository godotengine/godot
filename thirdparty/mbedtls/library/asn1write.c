/*
 * ASN.1 buffer writing functionality
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

#if defined(MBEDTLS_ASN1_WRITE_C)

#include "mbedtls/asn1write.h"

#include <string.h>

#if defined(MBEDTLS_PLATFORM_C)
#include "mbedtls/platform.h"
#else
#include <stdlib.h>
#define mbedtls_calloc    calloc
#define mbedtls_free       free
#endif

int mbedtls_asn1_write_len( unsigned char **p, unsigned char *start, size_t len )
{
    if( len < 0x80 )
    {
        if( *p - start < 1 )
            return( MBEDTLS_ERR_ASN1_BUF_TOO_SMALL );

        *--(*p) = (unsigned char) len;
        return( 1 );
    }

    if( len <= 0xFF )
    {
        if( *p - start < 2 )
            return( MBEDTLS_ERR_ASN1_BUF_TOO_SMALL );

        *--(*p) = (unsigned char) len;
        *--(*p) = 0x81;
        return( 2 );
    }

    if( len <= 0xFFFF )
    {
        if( *p - start < 3 )
            return( MBEDTLS_ERR_ASN1_BUF_TOO_SMALL );

        *--(*p) = ( len       ) & 0xFF;
        *--(*p) = ( len >>  8 ) & 0xFF;
        *--(*p) = 0x82;
        return( 3 );
    }

    if( len <= 0xFFFFFF )
    {
        if( *p - start < 4 )
            return( MBEDTLS_ERR_ASN1_BUF_TOO_SMALL );

        *--(*p) = ( len       ) & 0xFF;
        *--(*p) = ( len >>  8 ) & 0xFF;
        *--(*p) = ( len >> 16 ) & 0xFF;
        *--(*p) = 0x83;
        return( 4 );
    }

#if SIZE_MAX > 0xFFFFFFFF
    if( len <= 0xFFFFFFFF )
#endif
    {
        if( *p - start < 5 )
            return( MBEDTLS_ERR_ASN1_BUF_TOO_SMALL );

        *--(*p) = ( len       ) & 0xFF;
        *--(*p) = ( len >>  8 ) & 0xFF;
        *--(*p) = ( len >> 16 ) & 0xFF;
        *--(*p) = ( len >> 24 ) & 0xFF;
        *--(*p) = 0x84;
        return( 5 );
    }

#if SIZE_MAX > 0xFFFFFFFF
    return( MBEDTLS_ERR_ASN1_INVALID_LENGTH );
#endif
}

int mbedtls_asn1_write_tag( unsigned char **p, unsigned char *start, unsigned char tag )
{
    if( *p - start < 1 )
        return( MBEDTLS_ERR_ASN1_BUF_TOO_SMALL );

    *--(*p) = tag;

    return( 1 );
}

int mbedtls_asn1_write_raw_buffer( unsigned char **p, unsigned char *start,
                           const unsigned char *buf, size_t size )
{
    size_t len = 0;

    if( *p < start || (size_t)( *p - start ) < size )
        return( MBEDTLS_ERR_ASN1_BUF_TOO_SMALL );

    len = size;
    (*p) -= len;
    memcpy( *p, buf, len );

    return( (int) len );
}

#if defined(MBEDTLS_BIGNUM_C)
int mbedtls_asn1_write_mpi( unsigned char **p, unsigned char *start, const mbedtls_mpi *X )
{
    int ret;
    size_t len = 0;

    // Write the MPI
    //
    len = mbedtls_mpi_size( X );

    if( *p < start || (size_t)( *p - start ) < len )
        return( MBEDTLS_ERR_ASN1_BUF_TOO_SMALL );

    (*p) -= len;
    MBEDTLS_MPI_CHK( mbedtls_mpi_write_binary( X, *p, len ) );

    // DER format assumes 2s complement for numbers, so the leftmost bit
    // should be 0 for positive numbers and 1 for negative numbers.
    //
    if( X->s ==1 && **p & 0x80 )
    {
        if( *p - start < 1 )
            return( MBEDTLS_ERR_ASN1_BUF_TOO_SMALL );

        *--(*p) = 0x00;
        len += 1;
    }

    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( p, start, len ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( p, start, MBEDTLS_ASN1_INTEGER ) );

    ret = (int) len;

cleanup:
    return( ret );
}
#endif /* MBEDTLS_BIGNUM_C */

int mbedtls_asn1_write_null( unsigned char **p, unsigned char *start )
{
    int ret;
    size_t len = 0;

    // Write NULL
    //
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( p, start, 0) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( p, start, MBEDTLS_ASN1_NULL ) );

    return( (int) len );
}

int mbedtls_asn1_write_oid( unsigned char **p, unsigned char *start,
                    const char *oid, size_t oid_len )
{
    int ret;
    size_t len = 0;

    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_raw_buffer( p, start,
                                  (const unsigned char *) oid, oid_len ) );
    MBEDTLS_ASN1_CHK_ADD( len , mbedtls_asn1_write_len( p, start, len ) );
    MBEDTLS_ASN1_CHK_ADD( len , mbedtls_asn1_write_tag( p, start, MBEDTLS_ASN1_OID ) );

    return( (int) len );
}

int mbedtls_asn1_write_algorithm_identifier( unsigned char **p, unsigned char *start,
                                     const char *oid, size_t oid_len,
                                     size_t par_len )
{
    int ret;
    size_t len = 0;

    if( par_len == 0 )
        MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_null( p, start ) );
    else
        len += par_len;

    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_oid( p, start, oid, oid_len ) );

    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( p, start, len ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( p, start,
                                       MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE ) );

    return( (int) len );
}

int mbedtls_asn1_write_bool( unsigned char **p, unsigned char *start, int boolean )
{
    int ret;
    size_t len = 0;

    if( *p - start < 1 )
        return( MBEDTLS_ERR_ASN1_BUF_TOO_SMALL );

    *--(*p) = (boolean) ? 255 : 0;
    len++;

    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( p, start, len ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( p, start, MBEDTLS_ASN1_BOOLEAN ) );

    return( (int) len );
}

int mbedtls_asn1_write_int( unsigned char **p, unsigned char *start, int val )
{
    int ret;
    size_t len = 0;

    if( *p - start < 1 )
        return( MBEDTLS_ERR_ASN1_BUF_TOO_SMALL );

    len += 1;
    *--(*p) = val;

    if( val > 0 && **p & 0x80 )
    {
        if( *p - start < 1 )
            return( MBEDTLS_ERR_ASN1_BUF_TOO_SMALL );

        *--(*p) = 0x00;
        len += 1;
    }

    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( p, start, len ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( p, start, MBEDTLS_ASN1_INTEGER ) );

    return( (int) len );
}

int mbedtls_asn1_write_printable_string( unsigned char **p, unsigned char *start,
                                 const char *text, size_t text_len )
{
    int ret;
    size_t len = 0;

    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_raw_buffer( p, start,
                  (const unsigned char *) text, text_len ) );

    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( p, start, len ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( p, start, MBEDTLS_ASN1_PRINTABLE_STRING ) );

    return( (int) len );
}

int mbedtls_asn1_write_ia5_string( unsigned char **p, unsigned char *start,
                           const char *text, size_t text_len )
{
    int ret;
    size_t len = 0;

    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_raw_buffer( p, start,
                  (const unsigned char *) text, text_len ) );

    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( p, start, len ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( p, start, MBEDTLS_ASN1_IA5_STRING ) );

    return( (int) len );
}

int mbedtls_asn1_write_bitstring( unsigned char **p, unsigned char *start,
                          const unsigned char *buf, size_t bits )
{
    int ret;
    size_t len = 0, size;

    size = ( bits / 8 ) + ( ( bits % 8 ) ? 1 : 0 );

    // Calculate byte length
    //
    if( *p < start || (size_t)( *p - start ) < size + 1 )
        return( MBEDTLS_ERR_ASN1_BUF_TOO_SMALL );

    len = size + 1;
    (*p) -= size;
    memcpy( *p, buf, size );

    // Write unused bits
    //
    *--(*p) = (unsigned char) (size * 8 - bits);

    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( p, start, len ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( p, start, MBEDTLS_ASN1_BIT_STRING ) );

    return( (int) len );
}

int mbedtls_asn1_write_octet_string( unsigned char **p, unsigned char *start,
                             const unsigned char *buf, size_t size )
{
    int ret;
    size_t len = 0;

    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_raw_buffer( p, start, buf, size ) );

    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( p, start, len ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( p, start, MBEDTLS_ASN1_OCTET_STRING ) );

    return( (int) len );
}

mbedtls_asn1_named_data *mbedtls_asn1_store_named_data( mbedtls_asn1_named_data **head,
                                        const char *oid, size_t oid_len,
                                        const unsigned char *val,
                                        size_t val_len )
{
    mbedtls_asn1_named_data *cur;

    if( ( cur = mbedtls_asn1_find_named_data( *head, oid, oid_len ) ) == NULL )
    {
        // Add new entry if not present yet based on OID
        //
        cur = (mbedtls_asn1_named_data*)mbedtls_calloc( 1,
                                            sizeof(mbedtls_asn1_named_data) );
        if( cur == NULL )
            return( NULL );

        cur->oid.len = oid_len;
        cur->oid.p = mbedtls_calloc( 1, oid_len );
        if( cur->oid.p == NULL )
        {
            mbedtls_free( cur );
            return( NULL );
        }

        memcpy( cur->oid.p, oid, oid_len );

        cur->val.len = val_len;
        cur->val.p = mbedtls_calloc( 1, val_len );
        if( cur->val.p == NULL )
        {
            mbedtls_free( cur->oid.p );
            mbedtls_free( cur );
            return( NULL );
        }

        cur->next = *head;
        *head = cur;
    }
    else if( cur->val.len < val_len )
    {
        /*
         * Enlarge existing value buffer if needed
         * Preserve old data until the allocation succeeded, to leave list in
         * a consistent state in case allocation fails.
         */
        void *p = mbedtls_calloc( 1, val_len );
        if( p == NULL )
            return( NULL );

        mbedtls_free( cur->val.p );
        cur->val.p = p;
        cur->val.len = val_len;
    }

    if( val != NULL )
        memcpy( cur->val.p, val, val_len );

    return( cur );
}
#endif /* MBEDTLS_ASN1_WRITE_C */
