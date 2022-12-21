/*
 *  RFC 1186/1320 compliant MD4 implementation
 *
 *  Copyright The Mbed TLS Contributors
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
 */
/*
 *  The MD4 algorithm was designed by Ron Rivest in 1990.
 *
 *  http://www.ietf.org/rfc/rfc1186.txt
 *  http://www.ietf.org/rfc/rfc1320.txt
 */

#include "common.h"

#if defined(MBEDTLS_MD4_C)

#include "mbedtls/md4.h"
#include "mbedtls/platform_util.h"
#include "mbedtls/error.h"

#include <string.h>

#include "mbedtls/platform.h"

#if !defined(MBEDTLS_MD4_ALT)

void mbedtls_md4_init( mbedtls_md4_context *ctx )
{
    memset( ctx, 0, sizeof( mbedtls_md4_context ) );
}

void mbedtls_md4_free( mbedtls_md4_context *ctx )
{
    if( ctx == NULL )
        return;

    mbedtls_platform_zeroize( ctx, sizeof( mbedtls_md4_context ) );
}

void mbedtls_md4_clone( mbedtls_md4_context *dst,
                        const mbedtls_md4_context *src )
{
    *dst = *src;
}

/*
 * MD4 context setup
 */
int mbedtls_md4_starts_ret( mbedtls_md4_context *ctx )
{
    ctx->total[0] = 0;
    ctx->total[1] = 0;

    ctx->state[0] = 0x67452301;
    ctx->state[1] = 0xEFCDAB89;
    ctx->state[2] = 0x98BADCFE;
    ctx->state[3] = 0x10325476;

    return( 0 );
}

#if !defined(MBEDTLS_DEPRECATED_REMOVED)
void mbedtls_md4_starts( mbedtls_md4_context *ctx )
{
    mbedtls_md4_starts_ret( ctx );
}
#endif

#if !defined(MBEDTLS_MD4_PROCESS_ALT)
int mbedtls_internal_md4_process( mbedtls_md4_context *ctx,
                                  const unsigned char data[64] )
{
    struct
    {
        uint32_t X[16], A, B, C, D;
    } local;

    local.X[ 0] = MBEDTLS_GET_UINT32_LE( data,  0 );
    local.X[ 1] = MBEDTLS_GET_UINT32_LE( data,  4 );
    local.X[ 2] = MBEDTLS_GET_UINT32_LE( data,  8 );
    local.X[ 3] = MBEDTLS_GET_UINT32_LE( data, 12 );
    local.X[ 4] = MBEDTLS_GET_UINT32_LE( data, 16 );
    local.X[ 5] = MBEDTLS_GET_UINT32_LE( data, 20 );
    local.X[ 6] = MBEDTLS_GET_UINT32_LE( data, 24 );
    local.X[ 7] = MBEDTLS_GET_UINT32_LE( data, 28 );
    local.X[ 8] = MBEDTLS_GET_UINT32_LE( data, 32 );
    local.X[ 9] = MBEDTLS_GET_UINT32_LE( data, 36 );
    local.X[10] = MBEDTLS_GET_UINT32_LE( data, 40 );
    local.X[11] = MBEDTLS_GET_UINT32_LE( data, 44 );
    local.X[12] = MBEDTLS_GET_UINT32_LE( data, 48 );
    local.X[13] = MBEDTLS_GET_UINT32_LE( data, 52 );
    local.X[14] = MBEDTLS_GET_UINT32_LE( data, 56 );
    local.X[15] = MBEDTLS_GET_UINT32_LE( data, 60 );

#define S(x,n) (((x) << (n)) | (((x) & 0xFFFFFFFF) >> (32 - (n))))

    local.A = ctx->state[0];
    local.B = ctx->state[1];
    local.C = ctx->state[2];
    local.D = ctx->state[3];

#define F(x, y, z) (((x) & (y)) | ((~(x)) & (z)))
#define P(a,b,c,d,x,s)                           \
    do                                           \
    {                                            \
        (a) += F((b),(c),(d)) + (x);             \
        (a) = S((a),(s));                        \
    } while( 0 )


    P( local.A, local.B, local.C, local.D, local.X[ 0],  3 );
    P( local.D, local.A, local.B, local.C, local.X[ 1],  7 );
    P( local.C, local.D, local.A, local.B, local.X[ 2], 11 );
    P( local.B, local.C, local.D, local.A, local.X[ 3], 19 );
    P( local.A, local.B, local.C, local.D, local.X[ 4],  3 );
    P( local.D, local.A, local.B, local.C, local.X[ 5],  7 );
    P( local.C, local.D, local.A, local.B, local.X[ 6], 11 );
    P( local.B, local.C, local.D, local.A, local.X[ 7], 19 );
    P( local.A, local.B, local.C, local.D, local.X[ 8],  3 );
    P( local.D, local.A, local.B, local.C, local.X[ 9],  7 );
    P( local.C, local.D, local.A, local.B, local.X[10], 11 );
    P( local.B, local.C, local.D, local.A, local.X[11], 19 );
    P( local.A, local.B, local.C, local.D, local.X[12],  3 );
    P( local.D, local.A, local.B, local.C, local.X[13],  7 );
    P( local.C, local.D, local.A, local.B, local.X[14], 11 );
    P( local.B, local.C, local.D, local.A, local.X[15], 19 );

#undef P
#undef F

#define F(x,y,z) (((x) & (y)) | ((x) & (z)) | ((y) & (z)))
#define P(a,b,c,d,x,s)                          \
    do                                          \
    {                                           \
        (a) += F((b),(c),(d)) + (x) + 0x5A827999;       \
        (a) = S((a),(s));                               \
    } while( 0 )

    P( local.A, local.B, local.C, local.D, local.X[ 0],  3 );
    P( local.D, local.A, local.B, local.C, local.X[ 4],  5 );
    P( local.C, local.D, local.A, local.B, local.X[ 8],  9 );
    P( local.B, local.C, local.D, local.A, local.X[12], 13 );
    P( local.A, local.B, local.C, local.D, local.X[ 1],  3 );
    P( local.D, local.A, local.B, local.C, local.X[ 5],  5 );
    P( local.C, local.D, local.A, local.B, local.X[ 9],  9 );
    P( local.B, local.C, local.D, local.A, local.X[13], 13 );
    P( local.A, local.B, local.C, local.D, local.X[ 2],  3 );
    P( local.D, local.A, local.B, local.C, local.X[ 6],  5 );
    P( local.C, local.D, local.A, local.B, local.X[10],  9 );
    P( local.B, local.C, local.D, local.A, local.X[14], 13 );
    P( local.A, local.B, local.C, local.D, local.X[ 3],  3 );
    P( local.D, local.A, local.B, local.C, local.X[ 7],  5 );
    P( local.C, local.D, local.A, local.B, local.X[11],  9 );
    P( local.B, local.C, local.D, local.A, local.X[15], 13 );

#undef P
#undef F

#define F(x,y,z) ((x) ^ (y) ^ (z))
#define P(a,b,c,d,x,s)                                  \
    do                                                  \
    {                                                   \
        (a) += F((b),(c),(d)) + (x) + 0x6ED9EBA1;       \
        (a) = S((a),(s));                               \
    } while( 0 )

    P( local.A, local.B, local.C, local.D, local.X[ 0],  3 );
    P( local.D, local.A, local.B, local.C, local.X[ 8],  9 );
    P( local.C, local.D, local.A, local.B, local.X[ 4], 11 );
    P( local.B, local.C, local.D, local.A, local.X[12], 15 );
    P( local.A, local.B, local.C, local.D, local.X[ 2],  3 );
    P( local.D, local.A, local.B, local.C, local.X[10],  9 );
    P( local.C, local.D, local.A, local.B, local.X[ 6], 11 );
    P( local.B, local.C, local.D, local.A, local.X[14], 15 );
    P( local.A, local.B, local.C, local.D, local.X[ 1],  3 );
    P( local.D, local.A, local.B, local.C, local.X[ 9],  9 );
    P( local.C, local.D, local.A, local.B, local.X[ 5], 11 );
    P( local.B, local.C, local.D, local.A, local.X[13], 15 );
    P( local.A, local.B, local.C, local.D, local.X[ 3],  3 );
    P( local.D, local.A, local.B, local.C, local.X[11],  9 );
    P( local.C, local.D, local.A, local.B, local.X[ 7], 11 );
    P( local.B, local.C, local.D, local.A, local.X[15], 15 );

#undef F
#undef P

    ctx->state[0] += local.A;
    ctx->state[1] += local.B;
    ctx->state[2] += local.C;
    ctx->state[3] += local.D;

    /* Zeroise variables to clear sensitive data from memory. */
    mbedtls_platform_zeroize( &local, sizeof( local ) );

    return( 0 );
}

#if !defined(MBEDTLS_DEPRECATED_REMOVED)
void mbedtls_md4_process( mbedtls_md4_context *ctx,
                          const unsigned char data[64] )
{
    mbedtls_internal_md4_process( ctx, data );
}
#endif
#endif /* !MBEDTLS_MD4_PROCESS_ALT */

/*
 * MD4 process buffer
 */
int mbedtls_md4_update_ret( mbedtls_md4_context *ctx,
                            const unsigned char *input,
                            size_t ilen )
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    size_t fill;
    uint32_t left;

    if( ilen == 0 )
        return( 0 );

    left = ctx->total[0] & 0x3F;
    fill = 64 - left;

    ctx->total[0] += (uint32_t) ilen;
    ctx->total[0] &= 0xFFFFFFFF;

    if( ctx->total[0] < (uint32_t) ilen )
        ctx->total[1]++;

    if( left && ilen >= fill )
    {
        memcpy( (void *) (ctx->buffer + left),
                (void *) input, fill );

        if( ( ret = mbedtls_internal_md4_process( ctx, ctx->buffer ) ) != 0 )
            return( ret );

        input += fill;
        ilen  -= fill;
        left = 0;
    }

    while( ilen >= 64 )
    {
        if( ( ret = mbedtls_internal_md4_process( ctx, input ) ) != 0 )
            return( ret );

        input += 64;
        ilen  -= 64;
    }

    if( ilen > 0 )
    {
        memcpy( (void *) (ctx->buffer + left),
                (void *) input, ilen );
    }

    return( 0 );
}

#if !defined(MBEDTLS_DEPRECATED_REMOVED)
void mbedtls_md4_update( mbedtls_md4_context *ctx,
                         const unsigned char *input,
                         size_t ilen )
{
    mbedtls_md4_update_ret( ctx, input, ilen );
}
#endif

static const unsigned char md4_padding[64] =
{
 0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

/*
 * MD4 final digest
 */
int mbedtls_md4_finish_ret( mbedtls_md4_context *ctx,
                            unsigned char output[16] )
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    uint32_t last, padn;
    uint32_t high, low;
    unsigned char msglen[8];

    high = ( ctx->total[0] >> 29 )
         | ( ctx->total[1] <<  3 );
    low  = ( ctx->total[0] <<  3 );

    MBEDTLS_PUT_UINT32_LE( low,  msglen, 0 );
    MBEDTLS_PUT_UINT32_LE( high, msglen, 4 );

    last = ctx->total[0] & 0x3F;
    padn = ( last < 56 ) ? ( 56 - last ) : ( 120 - last );

    ret = mbedtls_md4_update_ret( ctx, (unsigned char *)md4_padding, padn );
    if( ret != 0 )
        return( ret );

    if( ( ret = mbedtls_md4_update_ret( ctx, msglen, 8 ) ) != 0 )
        return( ret );


    MBEDTLS_PUT_UINT32_LE( ctx->state[0], output,  0 );
    MBEDTLS_PUT_UINT32_LE( ctx->state[1], output,  4 );
    MBEDTLS_PUT_UINT32_LE( ctx->state[2], output,  8 );
    MBEDTLS_PUT_UINT32_LE( ctx->state[3], output, 12 );

    return( 0 );
}

#if !defined(MBEDTLS_DEPRECATED_REMOVED)
void mbedtls_md4_finish( mbedtls_md4_context *ctx,
                         unsigned char output[16] )
{
    mbedtls_md4_finish_ret( ctx, output );
}
#endif

#endif /* !MBEDTLS_MD4_ALT */

/*
 * output = MD4( input buffer )
 */
int mbedtls_md4_ret( const unsigned char *input,
                     size_t ilen,
                     unsigned char output[16] )
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    mbedtls_md4_context ctx;

    mbedtls_md4_init( &ctx );

    if( ( ret = mbedtls_md4_starts_ret( &ctx ) ) != 0 )
        goto exit;

    if( ( ret = mbedtls_md4_update_ret( &ctx, input, ilen ) ) != 0 )
        goto exit;

    if( ( ret = mbedtls_md4_finish_ret( &ctx, output ) ) != 0 )
        goto exit;

exit:
    mbedtls_md4_free( &ctx );

    return( ret );
}

#if !defined(MBEDTLS_DEPRECATED_REMOVED)
void mbedtls_md4( const unsigned char *input,
                  size_t ilen,
                  unsigned char output[16] )
{
    mbedtls_md4_ret( input, ilen, output );
}
#endif

#if defined(MBEDTLS_SELF_TEST)

/*
 * RFC 1320 test vectors
 */
static const unsigned char md4_test_str[7][81] =
{
    { "" },
    { "a" },
    { "abc" },
    { "message digest" },
    { "abcdefghijklmnopqrstuvwxyz" },
    { "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789" },
    { "12345678901234567890123456789012345678901234567890123456789012345678901234567890" }
};

static const size_t md4_test_strlen[7] =
{
    0, 1, 3, 14, 26, 62, 80
};

static const unsigned char md4_test_sum[7][16] =
{
    { 0x31, 0xD6, 0xCF, 0xE0, 0xD1, 0x6A, 0xE9, 0x31,
      0xB7, 0x3C, 0x59, 0xD7, 0xE0, 0xC0, 0x89, 0xC0 },
    { 0xBD, 0xE5, 0x2C, 0xB3, 0x1D, 0xE3, 0x3E, 0x46,
      0x24, 0x5E, 0x05, 0xFB, 0xDB, 0xD6, 0xFB, 0x24 },
    { 0xA4, 0x48, 0x01, 0x7A, 0xAF, 0x21, 0xD8, 0x52,
      0x5F, 0xC1, 0x0A, 0xE8, 0x7A, 0xA6, 0x72, 0x9D },
    { 0xD9, 0x13, 0x0A, 0x81, 0x64, 0x54, 0x9F, 0xE8,
      0x18, 0x87, 0x48, 0x06, 0xE1, 0xC7, 0x01, 0x4B },
    { 0xD7, 0x9E, 0x1C, 0x30, 0x8A, 0xA5, 0xBB, 0xCD,
      0xEE, 0xA8, 0xED, 0x63, 0xDF, 0x41, 0x2D, 0xA9 },
    { 0x04, 0x3F, 0x85, 0x82, 0xF2, 0x41, 0xDB, 0x35,
      0x1C, 0xE6, 0x27, 0xE1, 0x53, 0xE7, 0xF0, 0xE4 },
    { 0xE3, 0x3B, 0x4D, 0xDC, 0x9C, 0x38, 0xF2, 0x19,
      0x9C, 0x3E, 0x7B, 0x16, 0x4F, 0xCC, 0x05, 0x36 }
};

/*
 * Checkup routine
 */
int mbedtls_md4_self_test( int verbose )
{
    int i, ret = 0;
    unsigned char md4sum[16];

    for( i = 0; i < 7; i++ )
    {
        if( verbose != 0 )
            mbedtls_printf( "  MD4 test #%d: ", i + 1 );

        ret = mbedtls_md4_ret( md4_test_str[i], md4_test_strlen[i], md4sum );
        if( ret != 0 )
            goto fail;

        if( memcmp( md4sum, md4_test_sum[i], 16 ) != 0 )
        {
            ret = 1;
            goto fail;
        }

        if( verbose != 0 )
            mbedtls_printf( "passed\n" );
    }

    if( verbose != 0 )
        mbedtls_printf( "\n" );

    return( 0 );

fail:
    if( verbose != 0 )
        mbedtls_printf( "failed\n" );

    return( ret );
}

#endif /* MBEDTLS_SELF_TEST */

#endif /* MBEDTLS_MD4_C */
