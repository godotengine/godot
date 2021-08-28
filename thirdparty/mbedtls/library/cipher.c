/**
 * \file cipher.c
 *
 * \brief Generic cipher wrapper for mbed TLS
 *
 * \author Adriaan de Jong <dejong@fox-it.com>
 *
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 *
 *  This file is provided under the Apache License 2.0, or the
 *  GNU General Public License v2.0 or later.
 *
 *  **********
 *  Apache License 2.0:
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
 *  **********
 *
 *  **********
 *  GNU General Public License v2.0 or later:
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 *  **********
 */

#if !defined(MBEDTLS_CONFIG_FILE)
#include "mbedtls/config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#if defined(MBEDTLS_CIPHER_C)

#include "mbedtls/cipher.h"
#include "mbedtls/cipher_internal.h"
#include "mbedtls/platform_util.h"

#include <stdlib.h>
#include <string.h>

#if defined(MBEDTLS_CHACHAPOLY_C)
#include "mbedtls/chachapoly.h"
#endif

#if defined(MBEDTLS_GCM_C)
#include "mbedtls/gcm.h"
#endif

#if defined(MBEDTLS_CCM_C)
#include "mbedtls/ccm.h"
#endif

#if defined(MBEDTLS_CHACHA20_C)
#include "mbedtls/chacha20.h"
#endif

#if defined(MBEDTLS_CMAC_C)
#include "mbedtls/cmac.h"
#endif

#if defined(MBEDTLS_PLATFORM_C)
#include "mbedtls/platform.h"
#else
#define mbedtls_calloc calloc
#define mbedtls_free   free
#endif

#define CIPHER_VALIDATE_RET( cond )    \
    MBEDTLS_INTERNAL_VALIDATE_RET( cond, MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA )
#define CIPHER_VALIDATE( cond )        \
    MBEDTLS_INTERNAL_VALIDATE( cond )

#if defined(MBEDTLS_GCM_C) || defined(MBEDTLS_CHACHAPOLY_C)
/* Compare the contents of two buffers in constant time.
 * Returns 0 if the contents are bitwise identical, otherwise returns
 * a non-zero value.
 * This is currently only used by GCM and ChaCha20+Poly1305.
 */
static int mbedtls_constant_time_memcmp( const void *v1, const void *v2, size_t len )
{
    const unsigned char *p1 = (const unsigned char*) v1;
    const unsigned char *p2 = (const unsigned char*) v2;
    size_t i;
    unsigned char diff;

    for( diff = 0, i = 0; i < len; i++ )
        diff |= p1[i] ^ p2[i];

    return( (int)diff );
}
#endif /* MBEDTLS_GCM_C || MBEDTLS_CHACHAPOLY_C */

static int supported_init = 0;

const int *mbedtls_cipher_list( void )
{
    const mbedtls_cipher_definition_t *def;
    int *type;

    if( ! supported_init )
    {
        def = mbedtls_cipher_definitions;
        type = mbedtls_cipher_supported;

        while( def->type != 0 )
            *type++ = (*def++).type;

        *type = 0;

        supported_init = 1;
    }

    return( mbedtls_cipher_supported );
}

const mbedtls_cipher_info_t *mbedtls_cipher_info_from_type( const mbedtls_cipher_type_t cipher_type )
{
    const mbedtls_cipher_definition_t *def;

    for( def = mbedtls_cipher_definitions; def->info != NULL; def++ )
        if( def->type == cipher_type )
            return( def->info );

    return( NULL );
}

const mbedtls_cipher_info_t *mbedtls_cipher_info_from_string( const char *cipher_name )
{
    const mbedtls_cipher_definition_t *def;

    if( NULL == cipher_name )
        return( NULL );

    for( def = mbedtls_cipher_definitions; def->info != NULL; def++ )
        if( !  strcmp( def->info->name, cipher_name ) )
            return( def->info );

    return( NULL );
}

const mbedtls_cipher_info_t *mbedtls_cipher_info_from_values( const mbedtls_cipher_id_t cipher_id,
                                              int key_bitlen,
                                              const mbedtls_cipher_mode_t mode )
{
    const mbedtls_cipher_definition_t *def;

    for( def = mbedtls_cipher_definitions; def->info != NULL; def++ )
        if( def->info->base->cipher == cipher_id &&
            def->info->key_bitlen == (unsigned) key_bitlen &&
            def->info->mode == mode )
            return( def->info );

    return( NULL );
}

void mbedtls_cipher_init( mbedtls_cipher_context_t *ctx )
{
    CIPHER_VALIDATE( ctx != NULL );
    memset( ctx, 0, sizeof( mbedtls_cipher_context_t ) );
}

void mbedtls_cipher_free( mbedtls_cipher_context_t *ctx )
{
    if( ctx == NULL )
        return;

#if defined(MBEDTLS_CMAC_C)
    if( ctx->cmac_ctx )
    {
       mbedtls_platform_zeroize( ctx->cmac_ctx,
                                 sizeof( mbedtls_cmac_context_t ) );
       mbedtls_free( ctx->cmac_ctx );
    }
#endif

    if( ctx->cipher_ctx )
        ctx->cipher_info->base->ctx_free_func( ctx->cipher_ctx );

    mbedtls_platform_zeroize( ctx, sizeof(mbedtls_cipher_context_t) );
}

int mbedtls_cipher_setup( mbedtls_cipher_context_t *ctx, const mbedtls_cipher_info_t *cipher_info )
{
    CIPHER_VALIDATE_RET( ctx != NULL );
    if( cipher_info == NULL )
        return( MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA );

    memset( ctx, 0, sizeof( mbedtls_cipher_context_t ) );

    if( NULL == ( ctx->cipher_ctx = cipher_info->base->ctx_alloc_func() ) )
        return( MBEDTLS_ERR_CIPHER_ALLOC_FAILED );

    ctx->cipher_info = cipher_info;

#if defined(MBEDTLS_CIPHER_MODE_WITH_PADDING)
    /*
     * Ignore possible errors caused by a cipher mode that doesn't use padding
     */
#if defined(MBEDTLS_CIPHER_PADDING_PKCS7)
    (void) mbedtls_cipher_set_padding_mode( ctx, MBEDTLS_PADDING_PKCS7 );
#else
    (void) mbedtls_cipher_set_padding_mode( ctx, MBEDTLS_PADDING_NONE );
#endif
#endif /* MBEDTLS_CIPHER_MODE_WITH_PADDING */

    return( 0 );
}

int mbedtls_cipher_setkey( mbedtls_cipher_context_t *ctx,
                           const unsigned char *key,
                           int key_bitlen,
                           const mbedtls_operation_t operation )
{
    CIPHER_VALIDATE_RET( ctx != NULL );
    CIPHER_VALIDATE_RET( key != NULL );
    CIPHER_VALIDATE_RET( operation == MBEDTLS_ENCRYPT ||
                         operation == MBEDTLS_DECRYPT );
    if( ctx->cipher_info == NULL )
        return( MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA );

    if( ( ctx->cipher_info->flags & MBEDTLS_CIPHER_VARIABLE_KEY_LEN ) == 0 &&
        (int) ctx->cipher_info->key_bitlen != key_bitlen )
    {
        return( MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA );
    }

    ctx->key_bitlen = key_bitlen;
    ctx->operation = operation;

    /*
     * For OFB, CFB and CTR mode always use the encryption key schedule
     */
    if( MBEDTLS_ENCRYPT == operation ||
        MBEDTLS_MODE_CFB == ctx->cipher_info->mode ||
        MBEDTLS_MODE_OFB == ctx->cipher_info->mode ||
        MBEDTLS_MODE_CTR == ctx->cipher_info->mode )
    {
        return( ctx->cipher_info->base->setkey_enc_func( ctx->cipher_ctx, key,
                                                         ctx->key_bitlen ) );
    }

    if( MBEDTLS_DECRYPT == operation )
        return( ctx->cipher_info->base->setkey_dec_func( ctx->cipher_ctx, key,
                                                         ctx->key_bitlen ) );

    return( MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA );
}

int mbedtls_cipher_set_iv( mbedtls_cipher_context_t *ctx,
                           const unsigned char *iv,
                           size_t iv_len )
{
    size_t actual_iv_size;

    CIPHER_VALIDATE_RET( ctx != NULL );
    CIPHER_VALIDATE_RET( iv_len == 0 || iv != NULL );
    if( ctx->cipher_info == NULL )
        return( MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA );

    /* avoid buffer overflow in ctx->iv */
    if( iv_len > MBEDTLS_MAX_IV_LENGTH )
        return( MBEDTLS_ERR_CIPHER_FEATURE_UNAVAILABLE );

    if( ( ctx->cipher_info->flags & MBEDTLS_CIPHER_VARIABLE_IV_LEN ) != 0 )
        actual_iv_size = iv_len;
    else
    {
        actual_iv_size = ctx->cipher_info->iv_size;

        /* avoid reading past the end of input buffer */
        if( actual_iv_size > iv_len )
            return( MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA );
    }

#if defined(MBEDTLS_CHACHA20_C)
    if ( ctx->cipher_info->type == MBEDTLS_CIPHER_CHACHA20 )
    {
        if ( 0 != mbedtls_chacha20_starts( (mbedtls_chacha20_context*)ctx->cipher_ctx,
                                           iv,
                                           0U ) ) /* Initial counter value */
        {
            return( MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA );
        }
    }
#endif

    if ( actual_iv_size != 0 )
    {
        memcpy( ctx->iv, iv, actual_iv_size );
        ctx->iv_size = actual_iv_size;
    }

    return( 0 );
}

int mbedtls_cipher_reset( mbedtls_cipher_context_t *ctx )
{
    CIPHER_VALIDATE_RET( ctx != NULL );
    if( ctx->cipher_info == NULL )
        return( MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA );

    ctx->unprocessed_len = 0;

    return( 0 );
}

#if defined(MBEDTLS_GCM_C) || defined(MBEDTLS_CHACHAPOLY_C)
int mbedtls_cipher_update_ad( mbedtls_cipher_context_t *ctx,
                      const unsigned char *ad, size_t ad_len )
{
    CIPHER_VALIDATE_RET( ctx != NULL );
    CIPHER_VALIDATE_RET( ad_len == 0 || ad != NULL );
    if( ctx->cipher_info == NULL )
        return( MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA );

#if defined(MBEDTLS_GCM_C)
    if( MBEDTLS_MODE_GCM == ctx->cipher_info->mode )
    {
        return( mbedtls_gcm_starts( (mbedtls_gcm_context *) ctx->cipher_ctx, ctx->operation,
                                    ctx->iv, ctx->iv_size, ad, ad_len ) );
    }
#endif

#if defined(MBEDTLS_CHACHAPOLY_C)
    if (MBEDTLS_CIPHER_CHACHA20_POLY1305 == ctx->cipher_info->type )
    {
        int result;
        mbedtls_chachapoly_mode_t mode;

        mode = ( ctx->operation == MBEDTLS_ENCRYPT )
                ? MBEDTLS_CHACHAPOLY_ENCRYPT
                : MBEDTLS_CHACHAPOLY_DECRYPT;

        result = mbedtls_chachapoly_starts( (mbedtls_chachapoly_context*) ctx->cipher_ctx,
                                                        ctx->iv,
                                                        mode );
        if ( result != 0 )
            return( result );

        return( mbedtls_chachapoly_update_aad( (mbedtls_chachapoly_context*) ctx->cipher_ctx,
                                               ad, ad_len ) );
    }
#endif

    return( 0 );
}
#endif /* MBEDTLS_GCM_C || MBEDTLS_CHACHAPOLY_C */

int mbedtls_cipher_update( mbedtls_cipher_context_t *ctx, const unsigned char *input,
                   size_t ilen, unsigned char *output, size_t *olen )
{
    int ret;
    size_t block_size;

    CIPHER_VALIDATE_RET( ctx != NULL );
    CIPHER_VALIDATE_RET( ilen == 0 || input != NULL );
    CIPHER_VALIDATE_RET( output != NULL );
    CIPHER_VALIDATE_RET( olen != NULL );
    if( ctx->cipher_info == NULL )
        return( MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA );

    *olen = 0;
    block_size = mbedtls_cipher_get_block_size( ctx );
    if ( 0 == block_size )
    {
        return( MBEDTLS_ERR_CIPHER_INVALID_CONTEXT );
    }

    if( ctx->cipher_info->mode == MBEDTLS_MODE_ECB )
    {
        if( ilen != block_size )
            return( MBEDTLS_ERR_CIPHER_FULL_BLOCK_EXPECTED );

        *olen = ilen;

        if( 0 != ( ret = ctx->cipher_info->base->ecb_func( ctx->cipher_ctx,
                    ctx->operation, input, output ) ) )
        {
            return( ret );
        }

        return( 0 );
    }

#if defined(MBEDTLS_GCM_C)
    if( ctx->cipher_info->mode == MBEDTLS_MODE_GCM )
    {
        *olen = ilen;
        return( mbedtls_gcm_update( (mbedtls_gcm_context *) ctx->cipher_ctx, ilen, input,
                                    output ) );
    }
#endif

#if defined(MBEDTLS_CHACHAPOLY_C)
    if ( ctx->cipher_info->type == MBEDTLS_CIPHER_CHACHA20_POLY1305 )
    {
        *olen = ilen;
        return( mbedtls_chachapoly_update( (mbedtls_chachapoly_context*) ctx->cipher_ctx,
                                           ilen, input, output ) );
    }
#endif

    if( input == output &&
       ( ctx->unprocessed_len != 0 || ilen % block_size ) )
    {
        return( MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA );
    }

#if defined(MBEDTLS_CIPHER_MODE_CBC)
    if( ctx->cipher_info->mode == MBEDTLS_MODE_CBC )
    {
        size_t copy_len = 0;

        /*
         * If there is not enough data for a full block, cache it.
         */
        if( ( ctx->operation == MBEDTLS_DECRYPT && NULL != ctx->add_padding &&
                ilen <= block_size - ctx->unprocessed_len ) ||
            ( ctx->operation == MBEDTLS_DECRYPT && NULL == ctx->add_padding &&
                ilen < block_size - ctx->unprocessed_len ) ||
             ( ctx->operation == MBEDTLS_ENCRYPT &&
                ilen < block_size - ctx->unprocessed_len ) )
        {
            memcpy( &( ctx->unprocessed_data[ctx->unprocessed_len] ), input,
                    ilen );

            ctx->unprocessed_len += ilen;
            return( 0 );
        }

        /*
         * Process cached data first
         */
        if( 0 != ctx->unprocessed_len )
        {
            copy_len = block_size - ctx->unprocessed_len;

            memcpy( &( ctx->unprocessed_data[ctx->unprocessed_len] ), input,
                    copy_len );

            if( 0 != ( ret = ctx->cipher_info->base->cbc_func( ctx->cipher_ctx,
                    ctx->operation, block_size, ctx->iv,
                    ctx->unprocessed_data, output ) ) )
            {
                return( ret );
            }

            *olen += block_size;
            output += block_size;
            ctx->unprocessed_len = 0;

            input += copy_len;
            ilen -= copy_len;
        }

        /*
         * Cache final, incomplete block
         */
        if( 0 != ilen )
        {
            /* Encryption: only cache partial blocks
             * Decryption w/ padding: always keep at least one whole block
             * Decryption w/o padding: only cache partial blocks
             */
            copy_len = ilen % block_size;
            if( copy_len == 0 &&
                ctx->operation == MBEDTLS_DECRYPT &&
                NULL != ctx->add_padding)
            {
                copy_len = block_size;
            }

            memcpy( ctx->unprocessed_data, &( input[ilen - copy_len] ),
                    copy_len );

            ctx->unprocessed_len += copy_len;
            ilen -= copy_len;
        }

        /*
         * Process remaining full blocks
         */
        if( ilen )
        {
            if( 0 != ( ret = ctx->cipher_info->base->cbc_func( ctx->cipher_ctx,
                    ctx->operation, ilen, ctx->iv, input, output ) ) )
            {
                return( ret );
            }

            *olen += ilen;
        }

        return( 0 );
    }
#endif /* MBEDTLS_CIPHER_MODE_CBC */

#if defined(MBEDTLS_CIPHER_MODE_CFB)
    if( ctx->cipher_info->mode == MBEDTLS_MODE_CFB )
    {
        if( 0 != ( ret = ctx->cipher_info->base->cfb_func( ctx->cipher_ctx,
                ctx->operation, ilen, &ctx->unprocessed_len, ctx->iv,
                input, output ) ) )
        {
            return( ret );
        }

        *olen = ilen;

        return( 0 );
    }
#endif /* MBEDTLS_CIPHER_MODE_CFB */

#if defined(MBEDTLS_CIPHER_MODE_OFB)
    if( ctx->cipher_info->mode == MBEDTLS_MODE_OFB )
    {
        if( 0 != ( ret = ctx->cipher_info->base->ofb_func( ctx->cipher_ctx,
                ilen, &ctx->unprocessed_len, ctx->iv, input, output ) ) )
        {
            return( ret );
        }

        *olen = ilen;

        return( 0 );
    }
#endif /* MBEDTLS_CIPHER_MODE_OFB */

#if defined(MBEDTLS_CIPHER_MODE_CTR)
    if( ctx->cipher_info->mode == MBEDTLS_MODE_CTR )
    {
        if( 0 != ( ret = ctx->cipher_info->base->ctr_func( ctx->cipher_ctx,
                ilen, &ctx->unprocessed_len, ctx->iv,
                ctx->unprocessed_data, input, output ) ) )
        {
            return( ret );
        }

        *olen = ilen;

        return( 0 );
    }
#endif /* MBEDTLS_CIPHER_MODE_CTR */

#if defined(MBEDTLS_CIPHER_MODE_XTS)
    if( ctx->cipher_info->mode == MBEDTLS_MODE_XTS )
    {
        if( ctx->unprocessed_len > 0 ) {
            /* We can only process an entire data unit at a time. */
            return( MBEDTLS_ERR_CIPHER_FEATURE_UNAVAILABLE );
        }

        ret = ctx->cipher_info->base->xts_func( ctx->cipher_ctx,
                ctx->operation, ilen, ctx->iv, input, output );
        if( ret != 0 )
        {
            return( ret );
        }

        *olen = ilen;

        return( 0 );
    }
#endif /* MBEDTLS_CIPHER_MODE_XTS */

#if defined(MBEDTLS_CIPHER_MODE_STREAM)
    if( ctx->cipher_info->mode == MBEDTLS_MODE_STREAM )
    {
        if( 0 != ( ret = ctx->cipher_info->base->stream_func( ctx->cipher_ctx,
                                                    ilen, input, output ) ) )
        {
            return( ret );
        }

        *olen = ilen;

        return( 0 );
    }
#endif /* MBEDTLS_CIPHER_MODE_STREAM */

    return( MBEDTLS_ERR_CIPHER_FEATURE_UNAVAILABLE );
}

#if defined(MBEDTLS_CIPHER_MODE_WITH_PADDING)
#if defined(MBEDTLS_CIPHER_PADDING_PKCS7)
/*
 * PKCS7 (and PKCS5) padding: fill with ll bytes, with ll = padding_len
 */
static void add_pkcs_padding( unsigned char *output, size_t output_len,
        size_t data_len )
{
    size_t padding_len = output_len - data_len;
    unsigned char i;

    for( i = 0; i < padding_len; i++ )
        output[data_len + i] = (unsigned char) padding_len;
}

static int get_pkcs_padding( unsigned char *input, size_t input_len,
        size_t *data_len )
{
    size_t i, pad_idx;
    unsigned char padding_len, bad = 0;

    if( NULL == input || NULL == data_len )
        return( MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA );

    padding_len = input[input_len - 1];
    *data_len = input_len - padding_len;

    /* Avoid logical || since it results in a branch */
    bad |= padding_len > input_len;
    bad |= padding_len == 0;

    /* The number of bytes checked must be independent of padding_len,
     * so pick input_len, which is usually 8 or 16 (one block) */
    pad_idx = input_len - padding_len;
    for( i = 0; i < input_len; i++ )
        bad |= ( input[i] ^ padding_len ) * ( i >= pad_idx );

    return( MBEDTLS_ERR_CIPHER_INVALID_PADDING * ( bad != 0 ) );
}
#endif /* MBEDTLS_CIPHER_PADDING_PKCS7 */

#if defined(MBEDTLS_CIPHER_PADDING_ONE_AND_ZEROS)
/*
 * One and zeros padding: fill with 80 00 ... 00
 */
static void add_one_and_zeros_padding( unsigned char *output,
                                       size_t output_len, size_t data_len )
{
    size_t padding_len = output_len - data_len;
    unsigned char i = 0;

    output[data_len] = 0x80;
    for( i = 1; i < padding_len; i++ )
        output[data_len + i] = 0x00;
}

static int get_one_and_zeros_padding( unsigned char *input, size_t input_len,
                                      size_t *data_len )
{
    size_t i;
    unsigned char done = 0, prev_done, bad;

    if( NULL == input || NULL == data_len )
        return( MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA );

    bad = 0x80;
    *data_len = 0;
    for( i = input_len; i > 0; i-- )
    {
        prev_done = done;
        done |= ( input[i - 1] != 0 );
        *data_len |= ( i - 1 ) * ( done != prev_done );
        bad ^= input[i - 1] * ( done != prev_done );
    }

    return( MBEDTLS_ERR_CIPHER_INVALID_PADDING * ( bad != 0 ) );

}
#endif /* MBEDTLS_CIPHER_PADDING_ONE_AND_ZEROS */

#if defined(MBEDTLS_CIPHER_PADDING_ZEROS_AND_LEN)
/*
 * Zeros and len padding: fill with 00 ... 00 ll, where ll is padding length
 */
static void add_zeros_and_len_padding( unsigned char *output,
                                       size_t output_len, size_t data_len )
{
    size_t padding_len = output_len - data_len;
    unsigned char i = 0;

    for( i = 1; i < padding_len; i++ )
        output[data_len + i - 1] = 0x00;
    output[output_len - 1] = (unsigned char) padding_len;
}

static int get_zeros_and_len_padding( unsigned char *input, size_t input_len,
                                      size_t *data_len )
{
    size_t i, pad_idx;
    unsigned char padding_len, bad = 0;

    if( NULL == input || NULL == data_len )
        return( MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA );

    padding_len = input[input_len - 1];
    *data_len = input_len - padding_len;

    /* Avoid logical || since it results in a branch */
    bad |= padding_len > input_len;
    bad |= padding_len == 0;

    /* The number of bytes checked must be independent of padding_len */
    pad_idx = input_len - padding_len;
    for( i = 0; i < input_len - 1; i++ )
        bad |= input[i] * ( i >= pad_idx );

    return( MBEDTLS_ERR_CIPHER_INVALID_PADDING * ( bad != 0 ) );
}
#endif /* MBEDTLS_CIPHER_PADDING_ZEROS_AND_LEN */

#if defined(MBEDTLS_CIPHER_PADDING_ZEROS)
/*
 * Zero padding: fill with 00 ... 00
 */
static void add_zeros_padding( unsigned char *output,
                               size_t output_len, size_t data_len )
{
    size_t i;

    for( i = data_len; i < output_len; i++ )
        output[i] = 0x00;
}

static int get_zeros_padding( unsigned char *input, size_t input_len,
                              size_t *data_len )
{
    size_t i;
    unsigned char done = 0, prev_done;

    if( NULL == input || NULL == data_len )
        return( MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA );

    *data_len = 0;
    for( i = input_len; i > 0; i-- )
    {
        prev_done = done;
        done |= ( input[i-1] != 0 );
        *data_len |= i * ( done != prev_done );
    }

    return( 0 );
}
#endif /* MBEDTLS_CIPHER_PADDING_ZEROS */

/*
 * No padding: don't pad :)
 *
 * There is no add_padding function (check for NULL in mbedtls_cipher_finish)
 * but a trivial get_padding function
 */
static int get_no_padding( unsigned char *input, size_t input_len,
                              size_t *data_len )
{
    if( NULL == input || NULL == data_len )
        return( MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA );

    *data_len = input_len;

    return( 0 );
}
#endif /* MBEDTLS_CIPHER_MODE_WITH_PADDING */

int mbedtls_cipher_finish( mbedtls_cipher_context_t *ctx,
                   unsigned char *output, size_t *olen )
{
    CIPHER_VALIDATE_RET( ctx != NULL );
    CIPHER_VALIDATE_RET( output != NULL );
    CIPHER_VALIDATE_RET( olen != NULL );
    if( ctx->cipher_info == NULL )
        return( MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA );

    *olen = 0;

    if( MBEDTLS_MODE_CFB == ctx->cipher_info->mode ||
        MBEDTLS_MODE_OFB == ctx->cipher_info->mode ||
        MBEDTLS_MODE_CTR == ctx->cipher_info->mode ||
        MBEDTLS_MODE_GCM == ctx->cipher_info->mode ||
        MBEDTLS_MODE_XTS == ctx->cipher_info->mode ||
        MBEDTLS_MODE_STREAM == ctx->cipher_info->mode )
    {
        return( 0 );
    }

    if ( ( MBEDTLS_CIPHER_CHACHA20          == ctx->cipher_info->type ) ||
         ( MBEDTLS_CIPHER_CHACHA20_POLY1305 == ctx->cipher_info->type ) )
    {
        return( 0 );
    }

    if( MBEDTLS_MODE_ECB == ctx->cipher_info->mode )
    {
        if( ctx->unprocessed_len != 0 )
            return( MBEDTLS_ERR_CIPHER_FULL_BLOCK_EXPECTED );

        return( 0 );
    }

#if defined(MBEDTLS_CIPHER_MODE_CBC)
    if( MBEDTLS_MODE_CBC == ctx->cipher_info->mode )
    {
        int ret = 0;

        if( MBEDTLS_ENCRYPT == ctx->operation )
        {
            /* check for 'no padding' mode */
            if( NULL == ctx->add_padding )
            {
                if( 0 != ctx->unprocessed_len )
                    return( MBEDTLS_ERR_CIPHER_FULL_BLOCK_EXPECTED );

                return( 0 );
            }

            ctx->add_padding( ctx->unprocessed_data, mbedtls_cipher_get_iv_size( ctx ),
                    ctx->unprocessed_len );
        }
        else if( mbedtls_cipher_get_block_size( ctx ) != ctx->unprocessed_len )
        {
            /*
             * For decrypt operations, expect a full block,
             * or an empty block if no padding
             */
            if( NULL == ctx->add_padding && 0 == ctx->unprocessed_len )
                return( 0 );

            return( MBEDTLS_ERR_CIPHER_FULL_BLOCK_EXPECTED );
        }

        /* cipher block */
        if( 0 != ( ret = ctx->cipher_info->base->cbc_func( ctx->cipher_ctx,
                ctx->operation, mbedtls_cipher_get_block_size( ctx ), ctx->iv,
                ctx->unprocessed_data, output ) ) )
        {
            return( ret );
        }

        /* Set output size for decryption */
        if( MBEDTLS_DECRYPT == ctx->operation )
            return( ctx->get_padding( output, mbedtls_cipher_get_block_size( ctx ),
                                      olen ) );

        /* Set output size for encryption */
        *olen = mbedtls_cipher_get_block_size( ctx );
        return( 0 );
    }
#else
    ((void) output);
#endif /* MBEDTLS_CIPHER_MODE_CBC */

    return( MBEDTLS_ERR_CIPHER_FEATURE_UNAVAILABLE );
}

#if defined(MBEDTLS_CIPHER_MODE_WITH_PADDING)
int mbedtls_cipher_set_padding_mode( mbedtls_cipher_context_t *ctx,
                                     mbedtls_cipher_padding_t mode )
{
    CIPHER_VALIDATE_RET( ctx != NULL );

    if( NULL == ctx->cipher_info || MBEDTLS_MODE_CBC != ctx->cipher_info->mode )
    {
        return( MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA );
    }

    switch( mode )
    {
#if defined(MBEDTLS_CIPHER_PADDING_PKCS7)
    case MBEDTLS_PADDING_PKCS7:
        ctx->add_padding = add_pkcs_padding;
        ctx->get_padding = get_pkcs_padding;
        break;
#endif
#if defined(MBEDTLS_CIPHER_PADDING_ONE_AND_ZEROS)
    case MBEDTLS_PADDING_ONE_AND_ZEROS:
        ctx->add_padding = add_one_and_zeros_padding;
        ctx->get_padding = get_one_and_zeros_padding;
        break;
#endif
#if defined(MBEDTLS_CIPHER_PADDING_ZEROS_AND_LEN)
    case MBEDTLS_PADDING_ZEROS_AND_LEN:
        ctx->add_padding = add_zeros_and_len_padding;
        ctx->get_padding = get_zeros_and_len_padding;
        break;
#endif
#if defined(MBEDTLS_CIPHER_PADDING_ZEROS)
    case MBEDTLS_PADDING_ZEROS:
        ctx->add_padding = add_zeros_padding;
        ctx->get_padding = get_zeros_padding;
        break;
#endif
    case MBEDTLS_PADDING_NONE:
        ctx->add_padding = NULL;
        ctx->get_padding = get_no_padding;
        break;

    default:
        return( MBEDTLS_ERR_CIPHER_FEATURE_UNAVAILABLE );
    }

    return( 0 );
}
#endif /* MBEDTLS_CIPHER_MODE_WITH_PADDING */

#if defined(MBEDTLS_GCM_C) || defined(MBEDTLS_CHACHAPOLY_C)
int mbedtls_cipher_write_tag( mbedtls_cipher_context_t *ctx,
                      unsigned char *tag, size_t tag_len )
{
    CIPHER_VALIDATE_RET( ctx != NULL );
    CIPHER_VALIDATE_RET( tag_len == 0 || tag != NULL );
    if( ctx->cipher_info == NULL )
        return( MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA );

    if( MBEDTLS_ENCRYPT != ctx->operation )
        return( MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA );

#if defined(MBEDTLS_GCM_C)
    if( MBEDTLS_MODE_GCM == ctx->cipher_info->mode )
        return( mbedtls_gcm_finish( (mbedtls_gcm_context *) ctx->cipher_ctx,
                                    tag, tag_len ) );
#endif

#if defined(MBEDTLS_CHACHAPOLY_C)
    if ( MBEDTLS_CIPHER_CHACHA20_POLY1305 == ctx->cipher_info->type )
    {
        /* Don't allow truncated MAC for Poly1305 */
        if ( tag_len != 16U )
            return( MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA );

        return( mbedtls_chachapoly_finish( (mbedtls_chachapoly_context*) ctx->cipher_ctx,
                                           tag ) );
    }
#endif

    return( 0 );
}

int mbedtls_cipher_check_tag( mbedtls_cipher_context_t *ctx,
                      const unsigned char *tag, size_t tag_len )
{
    unsigned char check_tag[16];
    int ret;

    CIPHER_VALIDATE_RET( ctx != NULL );
    CIPHER_VALIDATE_RET( tag_len == 0 || tag != NULL );
    if( ctx->cipher_info == NULL )
        return( MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA );

    if( MBEDTLS_DECRYPT != ctx->operation )
    {
        return( MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA );
    }

#if defined(MBEDTLS_GCM_C)
    if( MBEDTLS_MODE_GCM == ctx->cipher_info->mode )
    {
        if( tag_len > sizeof( check_tag ) )
            return( MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA );

        if( 0 != ( ret = mbedtls_gcm_finish( (mbedtls_gcm_context *) ctx->cipher_ctx,
                                     check_tag, tag_len ) ) )
        {
            return( ret );
        }

        /* Check the tag in "constant-time" */
        if( mbedtls_constant_time_memcmp( tag, check_tag, tag_len ) != 0 )
            return( MBEDTLS_ERR_CIPHER_AUTH_FAILED );

        return( 0 );
    }
#endif /* MBEDTLS_GCM_C */

#if defined(MBEDTLS_CHACHAPOLY_C)
    if ( MBEDTLS_CIPHER_CHACHA20_POLY1305 == ctx->cipher_info->type )
    {
        /* Don't allow truncated MAC for Poly1305 */
        if ( tag_len != sizeof( check_tag ) )
            return( MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA );

        ret = mbedtls_chachapoly_finish( (mbedtls_chachapoly_context*) ctx->cipher_ctx,
                                                     check_tag );
        if ( ret != 0 )
        {
            return( ret );
        }

        /* Check the tag in "constant-time" */
        if( mbedtls_constant_time_memcmp( tag, check_tag, tag_len ) != 0 )
            return( MBEDTLS_ERR_CIPHER_AUTH_FAILED );

        return( 0 );
    }
#endif /* MBEDTLS_CHACHAPOLY_C */

    return( 0 );
}
#endif /* MBEDTLS_GCM_C || MBEDTLS_CHACHAPOLY_C */

/*
 * Packet-oriented wrapper for non-AEAD modes
 */
int mbedtls_cipher_crypt( mbedtls_cipher_context_t *ctx,
                  const unsigned char *iv, size_t iv_len,
                  const unsigned char *input, size_t ilen,
                  unsigned char *output, size_t *olen )
{
    int ret;
    size_t finish_olen;

    CIPHER_VALIDATE_RET( ctx != NULL );
    CIPHER_VALIDATE_RET( iv_len == 0 || iv != NULL );
    CIPHER_VALIDATE_RET( ilen == 0 || input != NULL );
    CIPHER_VALIDATE_RET( output != NULL );
    CIPHER_VALIDATE_RET( olen != NULL );

    if( ( ret = mbedtls_cipher_set_iv( ctx, iv, iv_len ) ) != 0 )
        return( ret );

    if( ( ret = mbedtls_cipher_reset( ctx ) ) != 0 )
        return( ret );

    if( ( ret = mbedtls_cipher_update( ctx, input, ilen, output, olen ) ) != 0 )
        return( ret );

    if( ( ret = mbedtls_cipher_finish( ctx, output + *olen, &finish_olen ) ) != 0 )
        return( ret );

    *olen += finish_olen;

    return( 0 );
}

#if defined(MBEDTLS_CIPHER_MODE_AEAD)
/*
 * Packet-oriented encryption for AEAD modes
 */
int mbedtls_cipher_auth_encrypt( mbedtls_cipher_context_t *ctx,
                         const unsigned char *iv, size_t iv_len,
                         const unsigned char *ad, size_t ad_len,
                         const unsigned char *input, size_t ilen,
                         unsigned char *output, size_t *olen,
                         unsigned char *tag, size_t tag_len )
{
    CIPHER_VALIDATE_RET( ctx != NULL );
    CIPHER_VALIDATE_RET( iv != NULL );
    CIPHER_VALIDATE_RET( ad_len == 0 || ad != NULL );
    CIPHER_VALIDATE_RET( ilen == 0 || input != NULL );
    CIPHER_VALIDATE_RET( output != NULL );
    CIPHER_VALIDATE_RET( olen != NULL );
    CIPHER_VALIDATE_RET( tag_len == 0 || tag != NULL );

#if defined(MBEDTLS_GCM_C)
    if( MBEDTLS_MODE_GCM == ctx->cipher_info->mode )
    {
        *olen = ilen;
        return( mbedtls_gcm_crypt_and_tag( ctx->cipher_ctx, MBEDTLS_GCM_ENCRYPT, ilen,
                                   iv, iv_len, ad, ad_len, input, output,
                                   tag_len, tag ) );
    }
#endif /* MBEDTLS_GCM_C */
#if defined(MBEDTLS_CCM_C)
    if( MBEDTLS_MODE_CCM == ctx->cipher_info->mode )
    {
        *olen = ilen;
        return( mbedtls_ccm_encrypt_and_tag( ctx->cipher_ctx, ilen,
                                     iv, iv_len, ad, ad_len, input, output,
                                     tag, tag_len ) );
    }
#endif /* MBEDTLS_CCM_C */
#if defined(MBEDTLS_CHACHAPOLY_C)
    if ( MBEDTLS_CIPHER_CHACHA20_POLY1305 == ctx->cipher_info->type )
    {
        /* ChachaPoly has fixed length nonce and MAC (tag) */
        if ( ( iv_len != ctx->cipher_info->iv_size ) ||
             ( tag_len != 16U ) )
        {
            return( MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA );
        }

        *olen = ilen;
        return( mbedtls_chachapoly_encrypt_and_tag( ctx->cipher_ctx,
                                ilen, iv, ad, ad_len, input, output, tag ) );
    }
#endif /* MBEDTLS_CHACHAPOLY_C */

    return( MBEDTLS_ERR_CIPHER_FEATURE_UNAVAILABLE );
}

/*
 * Packet-oriented decryption for AEAD modes
 */
int mbedtls_cipher_auth_decrypt( mbedtls_cipher_context_t *ctx,
                         const unsigned char *iv, size_t iv_len,
                         const unsigned char *ad, size_t ad_len,
                         const unsigned char *input, size_t ilen,
                         unsigned char *output, size_t *olen,
                         const unsigned char *tag, size_t tag_len )
{
    CIPHER_VALIDATE_RET( ctx != NULL );
    CIPHER_VALIDATE_RET( iv != NULL );
    CIPHER_VALIDATE_RET( ad_len == 0 || ad != NULL );
    CIPHER_VALIDATE_RET( ilen == 0 || input != NULL );
    CIPHER_VALIDATE_RET( output != NULL );
    CIPHER_VALIDATE_RET( olen != NULL );
    CIPHER_VALIDATE_RET( tag_len == 0 || tag != NULL );

#if defined(MBEDTLS_GCM_C)
    if( MBEDTLS_MODE_GCM == ctx->cipher_info->mode )
    {
        int ret;

        *olen = ilen;
        ret = mbedtls_gcm_auth_decrypt( ctx->cipher_ctx, ilen,
                                iv, iv_len, ad, ad_len,
                                tag, tag_len, input, output );

        if( ret == MBEDTLS_ERR_GCM_AUTH_FAILED )
            ret = MBEDTLS_ERR_CIPHER_AUTH_FAILED;

        return( ret );
    }
#endif /* MBEDTLS_GCM_C */
#if defined(MBEDTLS_CCM_C)
    if( MBEDTLS_MODE_CCM == ctx->cipher_info->mode )
    {
        int ret;

        *olen = ilen;
        ret = mbedtls_ccm_auth_decrypt( ctx->cipher_ctx, ilen,
                                iv, iv_len, ad, ad_len,
                                input, output, tag, tag_len );

        if( ret == MBEDTLS_ERR_CCM_AUTH_FAILED )
            ret = MBEDTLS_ERR_CIPHER_AUTH_FAILED;

        return( ret );
    }
#endif /* MBEDTLS_CCM_C */
#if defined(MBEDTLS_CHACHAPOLY_C)
    if ( MBEDTLS_CIPHER_CHACHA20_POLY1305 == ctx->cipher_info->type )
    {
        int ret;

        /* ChachaPoly has fixed length nonce and MAC (tag) */
        if ( ( iv_len != ctx->cipher_info->iv_size ) ||
             ( tag_len != 16U ) )
        {
            return( MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA );
        }

        *olen = ilen;
        ret = mbedtls_chachapoly_auth_decrypt( ctx->cipher_ctx, ilen,
                                iv, ad, ad_len, tag, input, output );

        if( ret == MBEDTLS_ERR_CHACHAPOLY_AUTH_FAILED )
            ret = MBEDTLS_ERR_CIPHER_AUTH_FAILED;

        return( ret );
    }
#endif /* MBEDTLS_CHACHAPOLY_C */

    return( MBEDTLS_ERR_CIPHER_FEATURE_UNAVAILABLE );
}
#endif /* MBEDTLS_CIPHER_MODE_AEAD */

#endif /* MBEDTLS_CIPHER_C */
