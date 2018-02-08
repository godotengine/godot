/**
 * \file asn1write.h
 *
 * \brief ASN.1 buffer writing functionality
 */
/*
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
#ifndef MBEDTLS_ASN1_WRITE_H
#define MBEDTLS_ASN1_WRITE_H

#include "asn1.h"

#define MBEDTLS_ASN1_CHK_ADD(g, f) do { if( ( ret = f ) < 0 ) return( ret ); else   \
                                g += ret; } while( 0 )

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief           Write a length field in ASN.1 format
 *                  Note: function works backwards in data buffer
 *
 * \param p         reference to current position pointer
 * \param start     start of the buffer (for bounds-checking)
 * \param len       the length to write
 *
 * \return          the length written or a negative error code
 */
int mbedtls_asn1_write_len( unsigned char **p, unsigned char *start, size_t len );

/**
 * \brief           Write a ASN.1 tag in ASN.1 format
 *                  Note: function works backwards in data buffer
 *
 * \param p         reference to current position pointer
 * \param start     start of the buffer (for bounds-checking)
 * \param tag       the tag to write
 *
 * \return          the length written or a negative error code
 */
int mbedtls_asn1_write_tag( unsigned char **p, unsigned char *start,
                    unsigned char tag );

/**
 * \brief           Write raw buffer data
 *                  Note: function works backwards in data buffer
 *
 * \param p         reference to current position pointer
 * \param start     start of the buffer (for bounds-checking)
 * \param buf       data buffer to write
 * \param size      length of the data buffer
 *
 * \return          the length written or a negative error code
 */
int mbedtls_asn1_write_raw_buffer( unsigned char **p, unsigned char *start,
                           const unsigned char *buf, size_t size );

#if defined(MBEDTLS_BIGNUM_C)
/**
 * \brief           Write a big number (MBEDTLS_ASN1_INTEGER) in ASN.1 format
 *                  Note: function works backwards in data buffer
 *
 * \param p         reference to current position pointer
 * \param start     start of the buffer (for bounds-checking)
 * \param X         the MPI to write
 *
 * \return          the length written or a negative error code
 */
int mbedtls_asn1_write_mpi( unsigned char **p, unsigned char *start, const mbedtls_mpi *X );
#endif /* MBEDTLS_BIGNUM_C */

/**
 * \brief           Write a NULL tag (MBEDTLS_ASN1_NULL) with zero data in ASN.1 format
 *                  Note: function works backwards in data buffer
 *
 * \param p         reference to current position pointer
 * \param start     start of the buffer (for bounds-checking)
 *
 * \return          the length written or a negative error code
 */
int mbedtls_asn1_write_null( unsigned char **p, unsigned char *start );

/**
 * \brief           Write an OID tag (MBEDTLS_ASN1_OID) and data in ASN.1 format
 *                  Note: function works backwards in data buffer
 *
 * \param p         reference to current position pointer
 * \param start     start of the buffer (for bounds-checking)
 * \param oid       the OID to write
 * \param oid_len   length of the OID
 *
 * \return          the length written or a negative error code
 */
int mbedtls_asn1_write_oid( unsigned char **p, unsigned char *start,
                    const char *oid, size_t oid_len );

/**
 * \brief           Write an AlgorithmIdentifier sequence in ASN.1 format
 *                  Note: function works backwards in data buffer
 *
 * \param p         reference to current position pointer
 * \param start     start of the buffer (for bounds-checking)
 * \param oid       the OID of the algorithm
 * \param oid_len   length of the OID
 * \param par_len   length of parameters, which must be already written.
 *                  If 0, NULL parameters are added
 *
 * \return          the length written or a negative error code
 */
int mbedtls_asn1_write_algorithm_identifier( unsigned char **p, unsigned char *start,
                                     const char *oid, size_t oid_len,
                                     size_t par_len );

/**
 * \brief           Write a boolean tag (MBEDTLS_ASN1_BOOLEAN) and value in ASN.1 format
 *                  Note: function works backwards in data buffer
 *
 * \param p         reference to current position pointer
 * \param start     start of the buffer (for bounds-checking)
 * \param boolean   0 or 1
 *
 * \return          the length written or a negative error code
 */
int mbedtls_asn1_write_bool( unsigned char **p, unsigned char *start, int boolean );

/**
 * \brief           Write an int tag (MBEDTLS_ASN1_INTEGER) and value in ASN.1 format
 *                  Note: function works backwards in data buffer
 *
 * \param p         reference to current position pointer
 * \param start     start of the buffer (for bounds-checking)
 * \param val       the integer value
 *
 * \return          the length written or a negative error code
 */
int mbedtls_asn1_write_int( unsigned char **p, unsigned char *start, int val );

/**
 * \brief           Write a printable string tag (MBEDTLS_ASN1_PRINTABLE_STRING) and
 *                  value in ASN.1 format
 *                  Note: function works backwards in data buffer
 *
 * \param p         reference to current position pointer
 * \param start     start of the buffer (for bounds-checking)
 * \param text      the text to write
 * \param text_len  length of the text
 *
 * \return          the length written or a negative error code
 */
int mbedtls_asn1_write_printable_string( unsigned char **p, unsigned char *start,
                                 const char *text, size_t text_len );

/**
 * \brief           Write an IA5 string tag (MBEDTLS_ASN1_IA5_STRING) and
 *                  value in ASN.1 format
 *                  Note: function works backwards in data buffer
 *
 * \param p         reference to current position pointer
 * \param start     start of the buffer (for bounds-checking)
 * \param text      the text to write
 * \param text_len  length of the text
 *
 * \return          the length written or a negative error code
 */
int mbedtls_asn1_write_ia5_string( unsigned char **p, unsigned char *start,
                           const char *text, size_t text_len );

/**
 * \brief           Write a bitstring tag (MBEDTLS_ASN1_BIT_STRING) and
 *                  value in ASN.1 format
 *                  Note: function works backwards in data buffer
 *
 * \param p         reference to current position pointer
 * \param start     start of the buffer (for bounds-checking)
 * \param buf       the bitstring
 * \param bits      the total number of bits in the bitstring
 *
 * \return          the length written or a negative error code
 */
int mbedtls_asn1_write_bitstring( unsigned char **p, unsigned char *start,
                          const unsigned char *buf, size_t bits );

/**
 * \brief           Write an octet string tag (MBEDTLS_ASN1_OCTET_STRING) and
 *                  value in ASN.1 format
 *                  Note: function works backwards in data buffer
 *
 * \param p         reference to current position pointer
 * \param start     start of the buffer (for bounds-checking)
 * \param buf       data buffer to write
 * \param size      length of the data buffer
 *
 * \return          the length written or a negative error code
 */
int mbedtls_asn1_write_octet_string( unsigned char **p, unsigned char *start,
                             const unsigned char *buf, size_t size );

/**
 * \brief           Create or find a specific named_data entry for writing in a
 *                  sequence or list based on the OID. If not already in there,
 *                  a new entry is added to the head of the list.
 *                  Warning: Destructive behaviour for the val data!
 *
 * \param list      Pointer to the location of the head of the list to seek
 *                  through (will be updated in case of a new entry)
 * \param oid       The OID to look for
 * \param oid_len   Size of the OID
 * \param val       Data to store (can be NULL if you want to fill it by hand)
 * \param val_len   Minimum length of the data buffer needed
 *
 * \return      NULL if if there was a memory allocation error, or a pointer
 *              to the new / existing entry.
 */
mbedtls_asn1_named_data *mbedtls_asn1_store_named_data( mbedtls_asn1_named_data **list,
                                        const char *oid, size_t oid_len,
                                        const unsigned char *val,
                                        size_t val_len );

#ifdef __cplusplus
}
#endif

#endif /* MBEDTLS_ASN1_WRITE_H */
