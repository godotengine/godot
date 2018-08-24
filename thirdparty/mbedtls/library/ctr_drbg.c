/*
 *  CTR_DRBG implementation based on AES-256 (NIST SP 800-90)
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
 *  The NIST SP 800-90 DRBGs are described in the following publication.
 *
 *  http://csrc.nist.gov/publications/nistpubs/800-90/SP800-90revised_March2007.pdf
 */

#if !defined(MBEDTLS_CONFIG_FILE)
#include "mbedtls/config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#if defined(MBEDTLS_CTR_DRBG_C)

#include "mbedtls/ctr_drbg.h"
#include "mbedtls/platform_util.h"

#include <string.h>

#if defined(MBEDTLS_FS_IO)
#include <stdio.h>
#endif

#if defined(MBEDTLS_SELF_TEST)
#if defined(MBEDTLS_PLATFORM_C)
#include "mbedtls/platform.h"
#else
#include <stdio.h>
#define mbedtls_printf printf
#endif /* MBEDTLS_PLATFORM_C */
#endif /* MBEDTLS_SELF_TEST */

/*
 * CTR_DRBG context initialization
 */
void mbedtls_ctr_drbg_init( mbedtls_ctr_drbg_context *ctx )
{
    memset( ctx, 0, sizeof( mbedtls_ctr_drbg_context ) );

#if defined(MBEDTLS_THREADING_C)
    mbedtls_mutex_init( &ctx->mutex );
#endif
}

/*
 * Non-public function wrapped by mbedtls_ctr_drbg_seed(). Necessary to allow
 * NIST tests to succeed (which require known length fixed entropy)
 */
int mbedtls_ctr_drbg_seed_entropy_len(
                   mbedtls_ctr_drbg_context *ctx,
                   int (*f_entropy)(void *, unsigned char *, size_t),
                   void *p_entropy,
                   const unsigned char *custom,
                   size_t len,
                   size_t entropy_len )
{
    int ret;
    unsigned char key[MBEDTLS_CTR_DRBG_KEYSIZE];

    memset( key, 0, MBEDTLS_CTR_DRBG_KEYSIZE );

    mbedtls_aes_init( &ctx->aes_ctx );

    ctx->f_entropy = f_entropy;
    ctx->p_entropy = p_entropy;

    ctx->entropy_len = entropy_len;
    ctx->reseed_interval = MBEDTLS_CTR_DRBG_RESEED_INTERVAL;

    /*
     * Initialize with an empty key
     */
    if( ( ret = mbedtls_aes_setkey_enc( &ctx->aes_ctx, key, MBEDTLS_CTR_DRBG_KEYBITS ) ) != 0 )
    {
        return( ret );
    }

    if( ( ret = mbedtls_ctr_drbg_reseed( ctx, custom, len ) ) != 0 )
    {
        return( ret );
    }
    return( 0 );
}

int mbedtls_ctr_drbg_seed( mbedtls_ctr_drbg_context *ctx,
                   int (*f_entropy)(void *, unsigned char *, size_t),
                   void *p_entropy,
                   const unsigned char *custom,
                   size_t len )
{
    return( mbedtls_ctr_drbg_seed_entropy_len( ctx, f_entropy, p_entropy, custom, len,
                                       MBEDTLS_CTR_DRBG_ENTROPY_LEN ) );
}

void mbedtls_ctr_drbg_free( mbedtls_ctr_drbg_context *ctx )
{
    if( ctx == NULL )
        return;

#if defined(MBEDTLS_THREADING_C)
    mbedtls_mutex_free( &ctx->mutex );
#endif
    mbedtls_aes_free( &ctx->aes_ctx );
    mbedtls_platform_zeroize( ctx, sizeof( mbedtls_ctr_drbg_context ) );
}

void mbedtls_ctr_drbg_set_prediction_resistance( mbedtls_ctr_drbg_context *ctx, int resistance )
{
    ctx->prediction_resistance = resistance;
}

void mbedtls_ctr_drbg_set_entropy_len( mbedtls_ctr_drbg_context *ctx, size_t len )
{
    ctx->entropy_len = len;
}

void mbedtls_ctr_drbg_set_reseed_interval( mbedtls_ctr_drbg_context *ctx, int interval )
{
    ctx->reseed_interval = interval;
}

static int block_cipher_df( unsigned char *output,
                            const unsigned char *data, size_t data_len )
{
    unsigned char buf[MBEDTLS_CTR_DRBG_MAX_SEED_INPUT + MBEDTLS_CTR_DRBG_BLOCKSIZE + 16];
    unsigned char tmp[MBEDTLS_CTR_DRBG_SEEDLEN];
    unsigned char key[MBEDTLS_CTR_DRBG_KEYSIZE];
    unsigned char chain[MBEDTLS_CTR_DRBG_BLOCKSIZE];
    unsigned char *p, *iv;
    mbedtls_aes_context aes_ctx;
    int ret = 0;

    int i, j;
    size_t buf_len, use_len;

    if( data_len > MBEDTLS_CTR_DRBG_MAX_SEED_INPUT )
        return( MBEDTLS_ERR_CTR_DRBG_INPUT_TOO_BIG );

    memset( buf, 0, MBEDTLS_CTR_DRBG_MAX_SEED_INPUT + MBEDTLS_CTR_DRBG_BLOCKSIZE + 16 );
    mbedtls_aes_init( &aes_ctx );

    /*
     * Construct IV (16 bytes) and S in buffer
     * IV = Counter (in 32-bits) padded to 16 with zeroes
     * S = Length input string (in 32-bits) || Length of output (in 32-bits) ||
     *     data || 0x80
     *     (Total is padded to a multiple of 16-bytes with zeroes)
     */
    p = buf + MBEDTLS_CTR_DRBG_BLOCKSIZE;
    *p++ = ( data_len >> 24 ) & 0xff;
    *p++ = ( data_len >> 16 ) & 0xff;
    *p++ = ( data_len >> 8  ) & 0xff;
    *p++ = ( data_len       ) & 0xff;
    p += 3;
    *p++ = MBEDTLS_CTR_DRBG_SEEDLEN;
    memcpy( p, data, data_len );
    p[data_len] = 0x80;

    buf_len = MBEDTLS_CTR_DRBG_BLOCKSIZE + 8 + data_len + 1;

    for( i = 0; i < MBEDTLS_CTR_DRBG_KEYSIZE; i++ )
        key[i] = i;

    if( ( ret = mbedtls_aes_setkey_enc( &aes_ctx, key, MBEDTLS_CTR_DRBG_KEYBITS ) ) != 0 )
    {
        goto exit;
    }

    /*
     * Reduce data to MBEDTLS_CTR_DRBG_SEEDLEN bytes of data
     */
    for( j = 0; j < MBEDTLS_CTR_DRBG_SEEDLEN; j += MBEDTLS_CTR_DRBG_BLOCKSIZE )
    {
        p = buf;
        memset( chain, 0, MBEDTLS_CTR_DRBG_BLOCKSIZE );
        use_len = buf_len;

        while( use_len > 0 )
        {
            for( i = 0; i < MBEDTLS_CTR_DRBG_BLOCKSIZE; i++ )
                chain[i] ^= p[i];
            p += MBEDTLS_CTR_DRBG_BLOCKSIZE;
            use_len -= ( use_len >= MBEDTLS_CTR_DRBG_BLOCKSIZE ) ?
                       MBEDTLS_CTR_DRBG_BLOCKSIZE : use_len;

            if( ( ret = mbedtls_aes_crypt_ecb( &aes_ctx, MBEDTLS_AES_ENCRYPT, chain, chain ) ) != 0 )
            {
                goto exit;
            }
        }

        memcpy( tmp + j, chain, MBEDTLS_CTR_DRBG_BLOCKSIZE );

        /*
         * Update IV
         */
        buf[3]++;
    }

    /*
     * Do final encryption with reduced data
     */
    if( ( ret = mbedtls_aes_setkey_enc( &aes_ctx, tmp, MBEDTLS_CTR_DRBG_KEYBITS ) ) != 0 )
    {
        goto exit;
    }
    iv = tmp + MBEDTLS_CTR_DRBG_KEYSIZE;
    p = output;

    for( j = 0; j < MBEDTLS_CTR_DRBG_SEEDLEN; j += MBEDTLS_CTR_DRBG_BLOCKSIZE )
    {
        if( ( ret = mbedtls_aes_crypt_ecb( &aes_ctx, MBEDTLS_AES_ENCRYPT, iv, iv ) ) != 0 )
        {
            goto exit;
        }
        memcpy( p, iv, MBEDTLS_CTR_DRBG_BLOCKSIZE );
        p += MBEDTLS_CTR_DRBG_BLOCKSIZE;
    }
exit:
    mbedtls_aes_free( &aes_ctx );
    /*
    * tidy up the stack
    */
    mbedtls_platform_zeroize( buf, sizeof( buf ) );
    mbedtls_platform_zeroize( tmp, sizeof( tmp ) );
    mbedtls_platform_zeroize( key, sizeof( key ) );
    mbedtls_platform_zeroize( chain, sizeof( chain ) );
    if( 0 != ret )
    {
        /*
        * wipe partial seed from memory
        */
        mbedtls_platform_zeroize( output, MBEDTLS_CTR_DRBG_SEEDLEN );
    }

    return( ret );
}

static int ctr_drbg_update_internal( mbedtls_ctr_drbg_context *ctx,
                              const unsigned char data[MBEDTLS_CTR_DRBG_SEEDLEN] )
{
    unsigned char tmp[MBEDTLS_CTR_DRBG_SEEDLEN];
    unsigned char *p = tmp;
    int i, j;
    int ret = 0;

    memset( tmp, 0, MBEDTLS_CTR_DRBG_SEEDLEN );

    for( j = 0; j < MBEDTLS_CTR_DRBG_SEEDLEN; j += MBEDTLS_CTR_DRBG_BLOCKSIZE )
    {
        /*
         * Increase counter
         */
        for( i = MBEDTLS_CTR_DRBG_BLOCKSIZE; i > 0; i-- )
            if( ++ctx->counter[i - 1] != 0 )
                break;

        /*
         * Crypt counter block
         */
        if( ( ret = mbedtls_aes_crypt_ecb( &ctx->aes_ctx, MBEDTLS_AES_ENCRYPT, ctx->counter, p ) ) != 0 )
        {
            return( ret );
        }

        p += MBEDTLS_CTR_DRBG_BLOCKSIZE;
    }

    for( i = 0; i < MBEDTLS_CTR_DRBG_SEEDLEN; i++ )
        tmp[i] ^= data[i];

    /*
     * Update key and counter
     */
    if( ( ret = mbedtls_aes_setkey_enc( &ctx->aes_ctx, tmp, MBEDTLS_CTR_DRBG_KEYBITS ) ) != 0 )
    {
        return( ret );
    }
    memcpy( ctx->counter, tmp + MBEDTLS_CTR_DRBG_KEYSIZE, MBEDTLS_CTR_DRBG_BLOCKSIZE );

    return( 0 );
}

void mbedtls_ctr_drbg_update( mbedtls_ctr_drbg_context *ctx,
                      const unsigned char *additional, size_t add_len )
{
    unsigned char add_input[MBEDTLS_CTR_DRBG_SEEDLEN];

    if( add_len > 0 )
    {
        /* MAX_INPUT would be more logical here, but we have to match
         * block_cipher_df()'s limits since we can't propagate errors */
        if( add_len > MBEDTLS_CTR_DRBG_MAX_SEED_INPUT )
            add_len = MBEDTLS_CTR_DRBG_MAX_SEED_INPUT;

        block_cipher_df( add_input, additional, add_len );
        ctr_drbg_update_internal( ctx, add_input );
    }
}

int mbedtls_ctr_drbg_reseed( mbedtls_ctr_drbg_context *ctx,
                     const unsigned char *additional, size_t len )
{
    unsigned char seed[MBEDTLS_CTR_DRBG_MAX_SEED_INPUT];
    size_t seedlen = 0;
    int ret;

    if( ctx->entropy_len > MBEDTLS_CTR_DRBG_MAX_SEED_INPUT ||
        len > MBEDTLS_CTR_DRBG_MAX_SEED_INPUT - ctx->entropy_len )
        return( MBEDTLS_ERR_CTR_DRBG_INPUT_TOO_BIG );

    memset( seed, 0, MBEDTLS_CTR_DRBG_MAX_SEED_INPUT );

    /*
     * Gather entropy_len bytes of entropy to seed state
     */
    if( 0 != ctx->f_entropy( ctx->p_entropy, seed,
                             ctx->entropy_len ) )
    {
        return( MBEDTLS_ERR_CTR_DRBG_ENTROPY_SOURCE_FAILED );
    }

    seedlen += ctx->entropy_len;

    /*
     * Add additional data
     */
    if( additional && len )
    {
        memcpy( seed + seedlen, additional, len );
        seedlen += len;
    }

    /*
     * Reduce to 384 bits
     */
    if( ( ret = block_cipher_df( seed, seed, seedlen ) ) != 0 )
    {
        return( ret );
    }

    /*
     * Update state
     */
    if( ( ret = ctr_drbg_update_internal( ctx, seed ) ) != 0 )
    {
        return( ret );
    }
    ctx->reseed_counter = 1;

    return( 0 );
}

int mbedtls_ctr_drbg_random_with_add( void *p_rng,
                              unsigned char *output, size_t output_len,
                              const unsigned char *additional, size_t add_len )
{
    int ret = 0;
    mbedtls_ctr_drbg_context *ctx = (mbedtls_ctr_drbg_context *) p_rng;
    unsigned char add_input[MBEDTLS_CTR_DRBG_SEEDLEN];
    unsigned char *p = output;
    unsigned char tmp[MBEDTLS_CTR_DRBG_BLOCKSIZE];
    int i;
    size_t use_len;

    if( output_len > MBEDTLS_CTR_DRBG_MAX_REQUEST )
        return( MBEDTLS_ERR_CTR_DRBG_REQUEST_TOO_BIG );

    if( add_len > MBEDTLS_CTR_DRBG_MAX_INPUT )
        return( MBEDTLS_ERR_CTR_DRBG_INPUT_TOO_BIG );

    memset( add_input, 0, MBEDTLS_CTR_DRBG_SEEDLEN );

    if( ctx->reseed_counter > ctx->reseed_interval ||
        ctx->prediction_resistance )
    {
        if( ( ret = mbedtls_ctr_drbg_reseed( ctx, additional, add_len ) ) != 0 )
        {
            return( ret );
        }
        add_len = 0;
    }

    if( add_len > 0 )
    {
        if( ( ret = block_cipher_df( add_input, additional, add_len ) ) != 0 )
        {
            return( ret );
        }
        if( ( ret = ctr_drbg_update_internal( ctx, add_input ) ) != 0 )
        {
            return( ret );
        }
    }

    while( output_len > 0 )
    {
        /*
         * Increase counter
         */
        for( i = MBEDTLS_CTR_DRBG_BLOCKSIZE; i > 0; i-- )
            if( ++ctx->counter[i - 1] != 0 )
                break;

        /*
         * Crypt counter block
         */
        if( ( ret = mbedtls_aes_crypt_ecb( &ctx->aes_ctx, MBEDTLS_AES_ENCRYPT, ctx->counter, tmp ) ) != 0 )
        {
            return( ret );
        }

        use_len = ( output_len > MBEDTLS_CTR_DRBG_BLOCKSIZE ) ? MBEDTLS_CTR_DRBG_BLOCKSIZE :
                                                       output_len;
        /*
         * Copy random block to destination
         */
        memcpy( p, tmp, use_len );
        p += use_len;
        output_len -= use_len;
    }

    if( ( ret = ctr_drbg_update_internal( ctx, add_input ) ) != 0 )
    {
        return( ret );
    }

    ctx->reseed_counter++;

    return( 0 );
}

int mbedtls_ctr_drbg_random( void *p_rng, unsigned char *output, size_t output_len )
{
    int ret;
    mbedtls_ctr_drbg_context *ctx = (mbedtls_ctr_drbg_context *) p_rng;

#if defined(MBEDTLS_THREADING_C)
    if( ( ret = mbedtls_mutex_lock( &ctx->mutex ) ) != 0 )
        return( ret );
#endif

    ret = mbedtls_ctr_drbg_random_with_add( ctx, output, output_len, NULL, 0 );

#if defined(MBEDTLS_THREADING_C)
    if( mbedtls_mutex_unlock( &ctx->mutex ) != 0 )
        return( MBEDTLS_ERR_THREADING_MUTEX_ERROR );
#endif

    return( ret );
}

#if defined(MBEDTLS_FS_IO)
int mbedtls_ctr_drbg_write_seed_file( mbedtls_ctr_drbg_context *ctx, const char *path )
{
    int ret = MBEDTLS_ERR_CTR_DRBG_FILE_IO_ERROR;
    FILE *f;
    unsigned char buf[ MBEDTLS_CTR_DRBG_MAX_INPUT ];

    if( ( f = fopen( path, "wb" ) ) == NULL )
        return( MBEDTLS_ERR_CTR_DRBG_FILE_IO_ERROR );

    if( ( ret = mbedtls_ctr_drbg_random( ctx, buf, MBEDTLS_CTR_DRBG_MAX_INPUT ) ) != 0 )
        goto exit;

    if( fwrite( buf, 1, MBEDTLS_CTR_DRBG_MAX_INPUT, f ) != MBEDTLS_CTR_DRBG_MAX_INPUT )
        ret = MBEDTLS_ERR_CTR_DRBG_FILE_IO_ERROR;
    else
        ret = 0;

exit:
    mbedtls_platform_zeroize( buf, sizeof( buf ) );

    fclose( f );
    return( ret );
}

int mbedtls_ctr_drbg_update_seed_file( mbedtls_ctr_drbg_context *ctx, const char *path )
{
    int ret = 0;
    FILE *f;
    size_t n;
    unsigned char buf[ MBEDTLS_CTR_DRBG_MAX_INPUT ];

    if( ( f = fopen( path, "rb" ) ) == NULL )
        return( MBEDTLS_ERR_CTR_DRBG_FILE_IO_ERROR );

    fseek( f, 0, SEEK_END );
    n = (size_t) ftell( f );
    fseek( f, 0, SEEK_SET );

    if( n > MBEDTLS_CTR_DRBG_MAX_INPUT )
    {
        fclose( f );
        return( MBEDTLS_ERR_CTR_DRBG_INPUT_TOO_BIG );
    }

    if( fread( buf, 1, n, f ) != n )
        ret = MBEDTLS_ERR_CTR_DRBG_FILE_IO_ERROR;
    else
        mbedtls_ctr_drbg_update( ctx, buf, n );

    fclose( f );

    mbedtls_platform_zeroize( buf, sizeof( buf ) );

    if( ret != 0 )
        return( ret );

    return( mbedtls_ctr_drbg_write_seed_file( ctx, path ) );
}
#endif /* MBEDTLS_FS_IO */

#if defined(MBEDTLS_SELF_TEST)

static const unsigned char entropy_source_pr[96] =
    { 0xc1, 0x80, 0x81, 0xa6, 0x5d, 0x44, 0x02, 0x16,
      0x19, 0xb3, 0xf1, 0x80, 0xb1, 0xc9, 0x20, 0x02,
      0x6a, 0x54, 0x6f, 0x0c, 0x70, 0x81, 0x49, 0x8b,
      0x6e, 0xa6, 0x62, 0x52, 0x6d, 0x51, 0xb1, 0xcb,
      0x58, 0x3b, 0xfa, 0xd5, 0x37, 0x5f, 0xfb, 0xc9,
      0xff, 0x46, 0xd2, 0x19, 0xc7, 0x22, 0x3e, 0x95,
      0x45, 0x9d, 0x82, 0xe1, 0xe7, 0x22, 0x9f, 0x63,
      0x31, 0x69, 0xd2, 0x6b, 0x57, 0x47, 0x4f, 0xa3,
      0x37, 0xc9, 0x98, 0x1c, 0x0b, 0xfb, 0x91, 0x31,
      0x4d, 0x55, 0xb9, 0xe9, 0x1c, 0x5a, 0x5e, 0xe4,
      0x93, 0x92, 0xcf, 0xc5, 0x23, 0x12, 0xd5, 0x56,
      0x2c, 0x4a, 0x6e, 0xff, 0xdc, 0x10, 0xd0, 0x68 };

static const unsigned char entropy_source_nopr[64] =
    { 0x5a, 0x19, 0x4d, 0x5e, 0x2b, 0x31, 0x58, 0x14,
      0x54, 0xde, 0xf6, 0x75, 0xfb, 0x79, 0x58, 0xfe,
      0xc7, 0xdb, 0x87, 0x3e, 0x56, 0x89, 0xfc, 0x9d,
      0x03, 0x21, 0x7c, 0x68, 0xd8, 0x03, 0x38, 0x20,
      0xf9, 0xe6, 0x5e, 0x04, 0xd8, 0x56, 0xf3, 0xa9,
      0xc4, 0x4a, 0x4c, 0xbd, 0xc1, 0xd0, 0x08, 0x46,
      0xf5, 0x98, 0x3d, 0x77, 0x1c, 0x1b, 0x13, 0x7e,
      0x4e, 0x0f, 0x9d, 0x8e, 0xf4, 0x09, 0xf9, 0x2e };

static const unsigned char nonce_pers_pr[16] =
    { 0xd2, 0x54, 0xfc, 0xff, 0x02, 0x1e, 0x69, 0xd2,
      0x29, 0xc9, 0xcf, 0xad, 0x85, 0xfa, 0x48, 0x6c };

static const unsigned char nonce_pers_nopr[16] =
    { 0x1b, 0x54, 0xb8, 0xff, 0x06, 0x42, 0xbf, 0xf5,
      0x21, 0xf1, 0x5c, 0x1c, 0x0b, 0x66, 0x5f, 0x3f };

static const unsigned char result_pr[16] =
    { 0x34, 0x01, 0x16, 0x56, 0xb4, 0x29, 0x00, 0x8f,
      0x35, 0x63, 0xec, 0xb5, 0xf2, 0x59, 0x07, 0x23 };

static const unsigned char result_nopr[16] =
    { 0xa0, 0x54, 0x30, 0x3d, 0x8a, 0x7e, 0xa9, 0x88,
      0x9d, 0x90, 0x3e, 0x07, 0x7c, 0x6f, 0x21, 0x8f };

static size_t test_offset;
static int ctr_drbg_self_test_entropy( void *data, unsigned char *buf,
                                       size_t len )
{
    const unsigned char *p = data;
    memcpy( buf, p + test_offset, len );
    test_offset += len;
    return( 0 );
}

#define CHK( c )    if( (c) != 0 )                          \
                    {                                       \
                        if( verbose != 0 )                  \
                            mbedtls_printf( "failed\n" );  \
                        return( 1 );                        \
                    }

/*
 * Checkup routine
 */
int mbedtls_ctr_drbg_self_test( int verbose )
{
    mbedtls_ctr_drbg_context ctx;
    unsigned char buf[16];

    mbedtls_ctr_drbg_init( &ctx );

    /*
     * Based on a NIST CTR_DRBG test vector (PR = True)
     */
    if( verbose != 0 )
        mbedtls_printf( "  CTR_DRBG (PR = TRUE) : " );

    test_offset = 0;
    CHK( mbedtls_ctr_drbg_seed_entropy_len( &ctx, ctr_drbg_self_test_entropy,
                                (void *) entropy_source_pr, nonce_pers_pr, 16, 32 ) );
    mbedtls_ctr_drbg_set_prediction_resistance( &ctx, MBEDTLS_CTR_DRBG_PR_ON );
    CHK( mbedtls_ctr_drbg_random( &ctx, buf, MBEDTLS_CTR_DRBG_BLOCKSIZE ) );
    CHK( mbedtls_ctr_drbg_random( &ctx, buf, MBEDTLS_CTR_DRBG_BLOCKSIZE ) );
    CHK( memcmp( buf, result_pr, MBEDTLS_CTR_DRBG_BLOCKSIZE ) );

    mbedtls_ctr_drbg_free( &ctx );

    if( verbose != 0 )
        mbedtls_printf( "passed\n" );

    /*
     * Based on a NIST CTR_DRBG test vector (PR = FALSE)
     */
    if( verbose != 0 )
        mbedtls_printf( "  CTR_DRBG (PR = FALSE): " );

    mbedtls_ctr_drbg_init( &ctx );

    test_offset = 0;
    CHK( mbedtls_ctr_drbg_seed_entropy_len( &ctx, ctr_drbg_self_test_entropy,
                            (void *) entropy_source_nopr, nonce_pers_nopr, 16, 32 ) );
    CHK( mbedtls_ctr_drbg_random( &ctx, buf, 16 ) );
    CHK( mbedtls_ctr_drbg_reseed( &ctx, NULL, 0 ) );
    CHK( mbedtls_ctr_drbg_random( &ctx, buf, 16 ) );
    CHK( memcmp( buf, result_nopr, 16 ) );

    mbedtls_ctr_drbg_free( &ctx );

    if( verbose != 0 )
        mbedtls_printf( "passed\n" );

    if( verbose != 0 )
            mbedtls_printf( "\n" );

    return( 0 );
}
#endif /* MBEDTLS_SELF_TEST */

#endif /* MBEDTLS_CTR_DRBG_C */
