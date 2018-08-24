/*
 *  Diffie-Hellman-Merkle key exchange
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
 *  The following sources were referenced in the design of this implementation
 *  of the Diffie-Hellman-Merkle algorithm:
 *
 *  [1] Handbook of Applied Cryptography - 1997, Chapter 12
 *      Menezes, van Oorschot and Vanstone
 *
 */

#if !defined(MBEDTLS_CONFIG_FILE)
#include "mbedtls/config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#if defined(MBEDTLS_DHM_C)

#include "mbedtls/dhm.h"
#include "mbedtls/platform_util.h"

#include <string.h>

#if defined(MBEDTLS_PEM_PARSE_C)
#include "mbedtls/pem.h"
#endif

#if defined(MBEDTLS_ASN1_PARSE_C)
#include "mbedtls/asn1.h"
#endif

#if defined(MBEDTLS_PLATFORM_C)
#include "mbedtls/platform.h"
#else
#include <stdlib.h>
#include <stdio.h>
#define mbedtls_printf     printf
#define mbedtls_calloc    calloc
#define mbedtls_free       free
#endif

#if !defined(MBEDTLS_DHM_ALT)

/*
 * helper to validate the mbedtls_mpi size and import it
 */
static int dhm_read_bignum( mbedtls_mpi *X,
                            unsigned char **p,
                            const unsigned char *end )
{
    int ret, n;

    if( end - *p < 2 )
        return( MBEDTLS_ERR_DHM_BAD_INPUT_DATA );

    n = ( (*p)[0] << 8 ) | (*p)[1];
    (*p) += 2;

    if( (int)( end - *p ) < n )
        return( MBEDTLS_ERR_DHM_BAD_INPUT_DATA );

    if( ( ret = mbedtls_mpi_read_binary( X, *p, n ) ) != 0 )
        return( MBEDTLS_ERR_DHM_READ_PARAMS_FAILED + ret );

    (*p) += n;

    return( 0 );
}

/*
 * Verify sanity of parameter with regards to P
 *
 * Parameter should be: 2 <= public_param <= P - 2
 *
 * This means that we need to return an error if
 *              public_param < 2 or public_param > P-2
 *
 * For more information on the attack, see:
 *  http://www.cl.cam.ac.uk/~rja14/Papers/psandqs.pdf
 *  http://web.nvd.nist.gov/view/vuln/detail?vulnId=CVE-2005-2643
 */
static int dhm_check_range( const mbedtls_mpi *param, const mbedtls_mpi *P )
{
    mbedtls_mpi L, U;
    int ret = 0;

    mbedtls_mpi_init( &L ); mbedtls_mpi_init( &U );

    MBEDTLS_MPI_CHK( mbedtls_mpi_lset( &L, 2 ) );
    MBEDTLS_MPI_CHK( mbedtls_mpi_sub_int( &U, P, 2 ) );

    if( mbedtls_mpi_cmp_mpi( param, &L ) < 0 ||
        mbedtls_mpi_cmp_mpi( param, &U ) > 0 )
    {
        ret = MBEDTLS_ERR_DHM_BAD_INPUT_DATA;
    }

cleanup:
    mbedtls_mpi_free( &L ); mbedtls_mpi_free( &U );
    return( ret );
}

void mbedtls_dhm_init( mbedtls_dhm_context *ctx )
{
    memset( ctx, 0, sizeof( mbedtls_dhm_context ) );
}

/*
 * Parse the ServerKeyExchange parameters
 */
int mbedtls_dhm_read_params( mbedtls_dhm_context *ctx,
                     unsigned char **p,
                     const unsigned char *end )
{
    int ret;

    if( ( ret = dhm_read_bignum( &ctx->P,  p, end ) ) != 0 ||
        ( ret = dhm_read_bignum( &ctx->G,  p, end ) ) != 0 ||
        ( ret = dhm_read_bignum( &ctx->GY, p, end ) ) != 0 )
        return( ret );

    if( ( ret = dhm_check_range( &ctx->GY, &ctx->P ) ) != 0 )
        return( ret );

    ctx->len = mbedtls_mpi_size( &ctx->P );

    return( 0 );
}

/*
 * Setup and write the ServerKeyExchange parameters
 */
int mbedtls_dhm_make_params( mbedtls_dhm_context *ctx, int x_size,
                     unsigned char *output, size_t *olen,
                     int (*f_rng)(void *, unsigned char *, size_t),
                     void *p_rng )
{
    int ret, count = 0;
    size_t n1, n2, n3;
    unsigned char *p;

    if( mbedtls_mpi_cmp_int( &ctx->P, 0 ) == 0 )
        return( MBEDTLS_ERR_DHM_BAD_INPUT_DATA );

    /*
     * Generate X as large as possible ( < P )
     */
    do
    {
        MBEDTLS_MPI_CHK( mbedtls_mpi_fill_random( &ctx->X, x_size, f_rng, p_rng ) );

        while( mbedtls_mpi_cmp_mpi( &ctx->X, &ctx->P ) >= 0 )
            MBEDTLS_MPI_CHK( mbedtls_mpi_shift_r( &ctx->X, 1 ) );

        if( count++ > 10 )
            return( MBEDTLS_ERR_DHM_MAKE_PARAMS_FAILED );
    }
    while( dhm_check_range( &ctx->X, &ctx->P ) != 0 );

    /*
     * Calculate GX = G^X mod P
     */
    MBEDTLS_MPI_CHK( mbedtls_mpi_exp_mod( &ctx->GX, &ctx->G, &ctx->X,
                          &ctx->P , &ctx->RP ) );

    if( ( ret = dhm_check_range( &ctx->GX, &ctx->P ) ) != 0 )
        return( ret );

    /*
     * export P, G, GX
     */
#define DHM_MPI_EXPORT( X, n )                                          \
    do {                                                                \
        MBEDTLS_MPI_CHK( mbedtls_mpi_write_binary( ( X ),               \
                                                   p + 2,               \
                                                   ( n ) ) );           \
        *p++ = (unsigned char)( ( n ) >> 8 );                           \
        *p++ = (unsigned char)( ( n )      );                           \
        p += ( n );                                                     \
    } while( 0 )

    n1 = mbedtls_mpi_size( &ctx->P  );
    n2 = mbedtls_mpi_size( &ctx->G  );
    n3 = mbedtls_mpi_size( &ctx->GX );

    p = output;
    DHM_MPI_EXPORT( &ctx->P , n1 );
    DHM_MPI_EXPORT( &ctx->G , n2 );
    DHM_MPI_EXPORT( &ctx->GX, n3 );

    *olen = p - output;

    ctx->len = n1;

cleanup:

    if( ret != 0 )
        return( MBEDTLS_ERR_DHM_MAKE_PARAMS_FAILED + ret );

    return( 0 );
}

/*
 * Set prime modulus and generator
 */
int mbedtls_dhm_set_group( mbedtls_dhm_context *ctx,
                           const mbedtls_mpi *P,
                           const mbedtls_mpi *G )
{
    int ret;

    if( ctx == NULL || P == NULL || G == NULL )
        return( MBEDTLS_ERR_DHM_BAD_INPUT_DATA );

    if( ( ret = mbedtls_mpi_copy( &ctx->P, P ) ) != 0 ||
        ( ret = mbedtls_mpi_copy( &ctx->G, G ) ) != 0 )
    {
        return( MBEDTLS_ERR_DHM_SET_GROUP_FAILED + ret );
    }

    ctx->len = mbedtls_mpi_size( &ctx->P );
    return( 0 );
}

/*
 * Import the peer's public value G^Y
 */
int mbedtls_dhm_read_public( mbedtls_dhm_context *ctx,
                     const unsigned char *input, size_t ilen )
{
    int ret;

    if( ctx == NULL || ilen < 1 || ilen > ctx->len )
        return( MBEDTLS_ERR_DHM_BAD_INPUT_DATA );

    if( ( ret = mbedtls_mpi_read_binary( &ctx->GY, input, ilen ) ) != 0 )
        return( MBEDTLS_ERR_DHM_READ_PUBLIC_FAILED + ret );

    return( 0 );
}

/*
 * Create own private value X and export G^X
 */
int mbedtls_dhm_make_public( mbedtls_dhm_context *ctx, int x_size,
                     unsigned char *output, size_t olen,
                     int (*f_rng)(void *, unsigned char *, size_t),
                     void *p_rng )
{
    int ret, count = 0;

    if( ctx == NULL || olen < 1 || olen > ctx->len )
        return( MBEDTLS_ERR_DHM_BAD_INPUT_DATA );

    if( mbedtls_mpi_cmp_int( &ctx->P, 0 ) == 0 )
        return( MBEDTLS_ERR_DHM_BAD_INPUT_DATA );

    /*
     * generate X and calculate GX = G^X mod P
     */
    do
    {
        MBEDTLS_MPI_CHK( mbedtls_mpi_fill_random( &ctx->X, x_size, f_rng, p_rng ) );

        while( mbedtls_mpi_cmp_mpi( &ctx->X, &ctx->P ) >= 0 )
            MBEDTLS_MPI_CHK( mbedtls_mpi_shift_r( &ctx->X, 1 ) );

        if( count++ > 10 )
            return( MBEDTLS_ERR_DHM_MAKE_PUBLIC_FAILED );
    }
    while( dhm_check_range( &ctx->X, &ctx->P ) != 0 );

    MBEDTLS_MPI_CHK( mbedtls_mpi_exp_mod( &ctx->GX, &ctx->G, &ctx->X,
                          &ctx->P , &ctx->RP ) );

    if( ( ret = dhm_check_range( &ctx->GX, &ctx->P ) ) != 0 )
        return( ret );

    MBEDTLS_MPI_CHK( mbedtls_mpi_write_binary( &ctx->GX, output, olen ) );

cleanup:

    if( ret != 0 )
        return( MBEDTLS_ERR_DHM_MAKE_PUBLIC_FAILED + ret );

    return( 0 );
}

/*
 * Use the blinding method and optimisation suggested in section 10 of:
 *  KOCHER, Paul C. Timing attacks on implementations of Diffie-Hellman, RSA,
 *  DSS, and other systems. In : Advances in Cryptology-CRYPTO'96. Springer
 *  Berlin Heidelberg, 1996. p. 104-113.
 */
static int dhm_update_blinding( mbedtls_dhm_context *ctx,
                    int (*f_rng)(void *, unsigned char *, size_t), void *p_rng )
{
    int ret, count;

    /*
     * Don't use any blinding the first time a particular X is used,
     * but remember it to use blinding next time.
     */
    if( mbedtls_mpi_cmp_mpi( &ctx->X, &ctx->pX ) != 0 )
    {
        MBEDTLS_MPI_CHK( mbedtls_mpi_copy( &ctx->pX, &ctx->X ) );
        MBEDTLS_MPI_CHK( mbedtls_mpi_lset( &ctx->Vi, 1 ) );
        MBEDTLS_MPI_CHK( mbedtls_mpi_lset( &ctx->Vf, 1 ) );

        return( 0 );
    }

    /*
     * Ok, we need blinding. Can we re-use existing values?
     * If yes, just update them by squaring them.
     */
    if( mbedtls_mpi_cmp_int( &ctx->Vi, 1 ) != 0 )
    {
        MBEDTLS_MPI_CHK( mbedtls_mpi_mul_mpi( &ctx->Vi, &ctx->Vi, &ctx->Vi ) );
        MBEDTLS_MPI_CHK( mbedtls_mpi_mod_mpi( &ctx->Vi, &ctx->Vi, &ctx->P ) );

        MBEDTLS_MPI_CHK( mbedtls_mpi_mul_mpi( &ctx->Vf, &ctx->Vf, &ctx->Vf ) );
        MBEDTLS_MPI_CHK( mbedtls_mpi_mod_mpi( &ctx->Vf, &ctx->Vf, &ctx->P ) );

        return( 0 );
    }

    /*
     * We need to generate blinding values from scratch
     */

    /* Vi = random( 2, P-1 ) */
    count = 0;
    do
    {
        MBEDTLS_MPI_CHK( mbedtls_mpi_fill_random( &ctx->Vi, mbedtls_mpi_size( &ctx->P ), f_rng, p_rng ) );

        while( mbedtls_mpi_cmp_mpi( &ctx->Vi, &ctx->P ) >= 0 )
            MBEDTLS_MPI_CHK( mbedtls_mpi_shift_r( &ctx->Vi, 1 ) );

        if( count++ > 10 )
            return( MBEDTLS_ERR_MPI_NOT_ACCEPTABLE );
    }
    while( mbedtls_mpi_cmp_int( &ctx->Vi, 1 ) <= 0 );

    /* Vf = Vi^-X mod P */
    MBEDTLS_MPI_CHK( mbedtls_mpi_inv_mod( &ctx->Vf, &ctx->Vi, &ctx->P ) );
    MBEDTLS_MPI_CHK( mbedtls_mpi_exp_mod( &ctx->Vf, &ctx->Vf, &ctx->X, &ctx->P, &ctx->RP ) );

cleanup:
    return( ret );
}

/*
 * Derive and export the shared secret (G^Y)^X mod P
 */
int mbedtls_dhm_calc_secret( mbedtls_dhm_context *ctx,
                     unsigned char *output, size_t output_size, size_t *olen,
                     int (*f_rng)(void *, unsigned char *, size_t),
                     void *p_rng )
{
    int ret;
    mbedtls_mpi GYb;

    if( ctx == NULL || output_size < ctx->len )
        return( MBEDTLS_ERR_DHM_BAD_INPUT_DATA );

    if( ( ret = dhm_check_range( &ctx->GY, &ctx->P ) ) != 0 )
        return( ret );

    mbedtls_mpi_init( &GYb );

    /* Blind peer's value */
    if( f_rng != NULL )
    {
        MBEDTLS_MPI_CHK( dhm_update_blinding( ctx, f_rng, p_rng ) );
        MBEDTLS_MPI_CHK( mbedtls_mpi_mul_mpi( &GYb, &ctx->GY, &ctx->Vi ) );
        MBEDTLS_MPI_CHK( mbedtls_mpi_mod_mpi( &GYb, &GYb, &ctx->P ) );
    }
    else
        MBEDTLS_MPI_CHK( mbedtls_mpi_copy( &GYb, &ctx->GY ) );

    /* Do modular exponentiation */
    MBEDTLS_MPI_CHK( mbedtls_mpi_exp_mod( &ctx->K, &GYb, &ctx->X,
                          &ctx->P, &ctx->RP ) );

    /* Unblind secret value */
    if( f_rng != NULL )
    {
        MBEDTLS_MPI_CHK( mbedtls_mpi_mul_mpi( &ctx->K, &ctx->K, &ctx->Vf ) );
        MBEDTLS_MPI_CHK( mbedtls_mpi_mod_mpi( &ctx->K, &ctx->K, &ctx->P ) );
    }

    *olen = mbedtls_mpi_size( &ctx->K );

    MBEDTLS_MPI_CHK( mbedtls_mpi_write_binary( &ctx->K, output, *olen ) );

cleanup:
    mbedtls_mpi_free( &GYb );

    if( ret != 0 )
        return( MBEDTLS_ERR_DHM_CALC_SECRET_FAILED + ret );

    return( 0 );
}

/*
 * Free the components of a DHM key
 */
void mbedtls_dhm_free( mbedtls_dhm_context *ctx )
{
    mbedtls_mpi_free( &ctx->pX ); mbedtls_mpi_free( &ctx->Vf );
    mbedtls_mpi_free( &ctx->Vi ); mbedtls_mpi_free( &ctx->RP );
    mbedtls_mpi_free( &ctx->K  ); mbedtls_mpi_free( &ctx->GY );
    mbedtls_mpi_free( &ctx->GX ); mbedtls_mpi_free( &ctx->X  );
    mbedtls_mpi_free( &ctx->G  ); mbedtls_mpi_free( &ctx->P  );

    mbedtls_platform_zeroize( ctx, sizeof( mbedtls_dhm_context ) );
}

#if defined(MBEDTLS_ASN1_PARSE_C)
/*
 * Parse DHM parameters
 */
int mbedtls_dhm_parse_dhm( mbedtls_dhm_context *dhm, const unsigned char *dhmin,
                   size_t dhminlen )
{
    int ret;
    size_t len;
    unsigned char *p, *end;
#if defined(MBEDTLS_PEM_PARSE_C)
    mbedtls_pem_context pem;

    mbedtls_pem_init( &pem );

    /* Avoid calling mbedtls_pem_read_buffer() on non-null-terminated string */
    if( dhminlen == 0 || dhmin[dhminlen - 1] != '\0' )
        ret = MBEDTLS_ERR_PEM_NO_HEADER_FOOTER_PRESENT;
    else
        ret = mbedtls_pem_read_buffer( &pem,
                               "-----BEGIN DH PARAMETERS-----",
                               "-----END DH PARAMETERS-----",
                               dhmin, NULL, 0, &dhminlen );

    if( ret == 0 )
    {
        /*
         * Was PEM encoded
         */
        dhminlen = pem.buflen;
    }
    else if( ret != MBEDTLS_ERR_PEM_NO_HEADER_FOOTER_PRESENT )
        goto exit;

    p = ( ret == 0 ) ? pem.buf : (unsigned char *) dhmin;
#else
    p = (unsigned char *) dhmin;
#endif /* MBEDTLS_PEM_PARSE_C */
    end = p + dhminlen;

    /*
     *  DHParams ::= SEQUENCE {
     *      prime              INTEGER,  -- P
     *      generator          INTEGER,  -- g
     *      privateValueLength INTEGER OPTIONAL
     *  }
     */
    if( ( ret = mbedtls_asn1_get_tag( &p, end, &len,
            MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE ) ) != 0 )
    {
        ret = MBEDTLS_ERR_DHM_INVALID_FORMAT + ret;
        goto exit;
    }

    end = p + len;

    if( ( ret = mbedtls_asn1_get_mpi( &p, end, &dhm->P  ) ) != 0 ||
        ( ret = mbedtls_asn1_get_mpi( &p, end, &dhm->G ) ) != 0 )
    {
        ret = MBEDTLS_ERR_DHM_INVALID_FORMAT + ret;
        goto exit;
    }

    if( p != end )
    {
        /* This might be the optional privateValueLength.
         * If so, we can cleanly discard it */
        mbedtls_mpi rec;
        mbedtls_mpi_init( &rec );
        ret = mbedtls_asn1_get_mpi( &p, end, &rec );
        mbedtls_mpi_free( &rec );
        if ( ret != 0 )
        {
            ret = MBEDTLS_ERR_DHM_INVALID_FORMAT + ret;
            goto exit;
        }
        if ( p != end )
        {
            ret = MBEDTLS_ERR_DHM_INVALID_FORMAT +
                MBEDTLS_ERR_ASN1_LENGTH_MISMATCH;
            goto exit;
        }
    }

    ret = 0;

    dhm->len = mbedtls_mpi_size( &dhm->P );

exit:
#if defined(MBEDTLS_PEM_PARSE_C)
    mbedtls_pem_free( &pem );
#endif
    if( ret != 0 )
        mbedtls_dhm_free( dhm );

    return( ret );
}

#if defined(MBEDTLS_FS_IO)
/*
 * Load all data from a file into a given buffer.
 *
 * The file is expected to contain either PEM or DER encoded data.
 * A terminating null byte is always appended. It is included in the announced
 * length only if the data looks like it is PEM encoded.
 */
static int load_file( const char *path, unsigned char **buf, size_t *n )
{
    FILE *f;
    long size;

    if( ( f = fopen( path, "rb" ) ) == NULL )
        return( MBEDTLS_ERR_DHM_FILE_IO_ERROR );

    fseek( f, 0, SEEK_END );
    if( ( size = ftell( f ) ) == -1 )
    {
        fclose( f );
        return( MBEDTLS_ERR_DHM_FILE_IO_ERROR );
    }
    fseek( f, 0, SEEK_SET );

    *n = (size_t) size;

    if( *n + 1 == 0 ||
        ( *buf = mbedtls_calloc( 1, *n + 1 ) ) == NULL )
    {
        fclose( f );
        return( MBEDTLS_ERR_DHM_ALLOC_FAILED );
    }

    if( fread( *buf, 1, *n, f ) != *n )
    {
        fclose( f );

        mbedtls_platform_zeroize( *buf, *n + 1 );
        mbedtls_free( *buf );

        return( MBEDTLS_ERR_DHM_FILE_IO_ERROR );
    }

    fclose( f );

    (*buf)[*n] = '\0';

    if( strstr( (const char *) *buf, "-----BEGIN " ) != NULL )
        ++*n;

    return( 0 );
}

/*
 * Load and parse DHM parameters
 */
int mbedtls_dhm_parse_dhmfile( mbedtls_dhm_context *dhm, const char *path )
{
    int ret;
    size_t n;
    unsigned char *buf;

    if( ( ret = load_file( path, &buf, &n ) ) != 0 )
        return( ret );

    ret = mbedtls_dhm_parse_dhm( dhm, buf, n );

    mbedtls_platform_zeroize( buf, n );
    mbedtls_free( buf );

    return( ret );
}
#endif /* MBEDTLS_FS_IO */
#endif /* MBEDTLS_ASN1_PARSE_C */
#endif /* MBEDTLS_DHM_ALT */

#if defined(MBEDTLS_SELF_TEST)

static const char mbedtls_test_dhm_params[] =
"-----BEGIN DH PARAMETERS-----\r\n"
"MIGHAoGBAJ419DBEOgmQTzo5qXl5fQcN9TN455wkOL7052HzxxRVMyhYmwQcgJvh\r\n"
"1sa18fyfR9OiVEMYglOpkqVoGLN7qd5aQNNi5W7/C+VBdHTBJcGZJyyP5B3qcz32\r\n"
"9mLJKudlVudV0Qxk5qUJaPZ/xupz0NyoVpviuiBOI1gNi8ovSXWzAgEC\r\n"
"-----END DH PARAMETERS-----\r\n";

static const size_t mbedtls_test_dhm_params_len = sizeof( mbedtls_test_dhm_params );

/*
 * Checkup routine
 */
int mbedtls_dhm_self_test( int verbose )
{
    int ret;
    mbedtls_dhm_context dhm;

    mbedtls_dhm_init( &dhm );

    if( verbose != 0 )
        mbedtls_printf( "  DHM parameter load: " );

    if( ( ret = mbedtls_dhm_parse_dhm( &dhm,
                    (const unsigned char *) mbedtls_test_dhm_params,
                    mbedtls_test_dhm_params_len ) ) != 0 )
    {
        if( verbose != 0 )
            mbedtls_printf( "failed\n" );

        ret = 1;
        goto exit;
    }

    if( verbose != 0 )
        mbedtls_printf( "passed\n\n" );

exit:
    mbedtls_dhm_free( &dhm );

    return( ret );
}

#endif /* MBEDTLS_SELF_TEST */

#endif /* MBEDTLS_DHM_C */
