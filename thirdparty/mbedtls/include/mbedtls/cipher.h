/**
 * \file cipher.h
 *
 * \brief The generic cipher wrapper.
 *
 * \author Adriaan de Jong <dejong@fox-it.com>
 */
/*
 *  Copyright (C) 2006-2018, Arm Limited (or its affiliates), All Rights Reserved
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
 *  This file is part of Mbed TLS (https://tls.mbed.org)
 */

#ifndef MBEDTLS_CIPHER_H
#define MBEDTLS_CIPHER_H

#if !defined(MBEDTLS_CONFIG_FILE)
#include "config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#include <stddef.h>

#if defined(MBEDTLS_GCM_C) || defined(MBEDTLS_CCM_C)
#define MBEDTLS_CIPHER_MODE_AEAD
#endif

#if defined(MBEDTLS_CIPHER_MODE_CBC)
#define MBEDTLS_CIPHER_MODE_WITH_PADDING
#endif

#if defined(MBEDTLS_ARC4_C)
#define MBEDTLS_CIPHER_MODE_STREAM
#endif

#if ( defined(__ARMCC_VERSION) || defined(_MSC_VER) ) && \
    !defined(inline) && !defined(__cplusplus)
#define inline __inline
#endif

#define MBEDTLS_ERR_CIPHER_FEATURE_UNAVAILABLE  -0x6080  /**< The selected feature is not available. */
#define MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA       -0x6100  /**< Bad input parameters. */
#define MBEDTLS_ERR_CIPHER_ALLOC_FAILED         -0x6180  /**< Failed to allocate memory. */
#define MBEDTLS_ERR_CIPHER_INVALID_PADDING      -0x6200  /**< Input data contains invalid padding and is rejected. */
#define MBEDTLS_ERR_CIPHER_FULL_BLOCK_EXPECTED  -0x6280  /**< Decryption of block requires a full block. */
#define MBEDTLS_ERR_CIPHER_AUTH_FAILED          -0x6300  /**< Authentication failed (for AEAD modes). */
#define MBEDTLS_ERR_CIPHER_INVALID_CONTEXT      -0x6380  /**< The context is invalid. For example, because it was freed. */
#define MBEDTLS_ERR_CIPHER_HW_ACCEL_FAILED      -0x6400  /**< Cipher hardware accelerator failed. */

#define MBEDTLS_CIPHER_VARIABLE_IV_LEN     0x01    /**< Cipher accepts IVs of variable length. */
#define MBEDTLS_CIPHER_VARIABLE_KEY_LEN    0x02    /**< Cipher accepts keys of variable length. */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief     An enumeration of supported ciphers.
 *
 * \warning   ARC4 and DES are considered weak ciphers and their use
 *            constitutes a security risk. We recommend considering stronger
 *            ciphers instead.
 */
typedef enum {
    MBEDTLS_CIPHER_ID_NONE = 0,
    MBEDTLS_CIPHER_ID_NULL,
    MBEDTLS_CIPHER_ID_AES,
    MBEDTLS_CIPHER_ID_DES,
    MBEDTLS_CIPHER_ID_3DES,
    MBEDTLS_CIPHER_ID_CAMELLIA,
    MBEDTLS_CIPHER_ID_BLOWFISH,
    MBEDTLS_CIPHER_ID_ARC4,
} mbedtls_cipher_id_t;

/**
 * \brief     An enumeration of supported (cipher, mode) pairs.
 *
 * \warning   ARC4 and DES are considered weak ciphers and their use
 *            constitutes a security risk. We recommend considering stronger
 *            ciphers instead.
 */
typedef enum {
    MBEDTLS_CIPHER_NONE = 0,
    MBEDTLS_CIPHER_NULL,
    MBEDTLS_CIPHER_AES_128_ECB,
    MBEDTLS_CIPHER_AES_192_ECB,
    MBEDTLS_CIPHER_AES_256_ECB,
    MBEDTLS_CIPHER_AES_128_CBC,
    MBEDTLS_CIPHER_AES_192_CBC,
    MBEDTLS_CIPHER_AES_256_CBC,
    MBEDTLS_CIPHER_AES_128_CFB128,
    MBEDTLS_CIPHER_AES_192_CFB128,
    MBEDTLS_CIPHER_AES_256_CFB128,
    MBEDTLS_CIPHER_AES_128_CTR,
    MBEDTLS_CIPHER_AES_192_CTR,
    MBEDTLS_CIPHER_AES_256_CTR,
    MBEDTLS_CIPHER_AES_128_GCM,
    MBEDTLS_CIPHER_AES_192_GCM,
    MBEDTLS_CIPHER_AES_256_GCM,
    MBEDTLS_CIPHER_CAMELLIA_128_ECB,
    MBEDTLS_CIPHER_CAMELLIA_192_ECB,
    MBEDTLS_CIPHER_CAMELLIA_256_ECB,
    MBEDTLS_CIPHER_CAMELLIA_128_CBC,
    MBEDTLS_CIPHER_CAMELLIA_192_CBC,
    MBEDTLS_CIPHER_CAMELLIA_256_CBC,
    MBEDTLS_CIPHER_CAMELLIA_128_CFB128,
    MBEDTLS_CIPHER_CAMELLIA_192_CFB128,
    MBEDTLS_CIPHER_CAMELLIA_256_CFB128,
    MBEDTLS_CIPHER_CAMELLIA_128_CTR,
    MBEDTLS_CIPHER_CAMELLIA_192_CTR,
    MBEDTLS_CIPHER_CAMELLIA_256_CTR,
    MBEDTLS_CIPHER_CAMELLIA_128_GCM,
    MBEDTLS_CIPHER_CAMELLIA_192_GCM,
    MBEDTLS_CIPHER_CAMELLIA_256_GCM,
    MBEDTLS_CIPHER_DES_ECB,
    MBEDTLS_CIPHER_DES_CBC,
    MBEDTLS_CIPHER_DES_EDE_ECB,
    MBEDTLS_CIPHER_DES_EDE_CBC,
    MBEDTLS_CIPHER_DES_EDE3_ECB,
    MBEDTLS_CIPHER_DES_EDE3_CBC,
    MBEDTLS_CIPHER_BLOWFISH_ECB,
    MBEDTLS_CIPHER_BLOWFISH_CBC,
    MBEDTLS_CIPHER_BLOWFISH_CFB64,
    MBEDTLS_CIPHER_BLOWFISH_CTR,
    MBEDTLS_CIPHER_ARC4_128,
    MBEDTLS_CIPHER_AES_128_CCM,
    MBEDTLS_CIPHER_AES_192_CCM,
    MBEDTLS_CIPHER_AES_256_CCM,
    MBEDTLS_CIPHER_CAMELLIA_128_CCM,
    MBEDTLS_CIPHER_CAMELLIA_192_CCM,
    MBEDTLS_CIPHER_CAMELLIA_256_CCM,
} mbedtls_cipher_type_t;

/** Supported cipher modes. */
typedef enum {
    MBEDTLS_MODE_NONE = 0,
    MBEDTLS_MODE_ECB,
    MBEDTLS_MODE_CBC,
    MBEDTLS_MODE_CFB,
    MBEDTLS_MODE_OFB, /* Unused! */
    MBEDTLS_MODE_CTR,
    MBEDTLS_MODE_GCM,
    MBEDTLS_MODE_STREAM,
    MBEDTLS_MODE_CCM,
} mbedtls_cipher_mode_t;

/** Supported cipher padding types. */
typedef enum {
    MBEDTLS_PADDING_PKCS7 = 0,     /**< PKCS7 padding (default).        */
    MBEDTLS_PADDING_ONE_AND_ZEROS, /**< ISO/IEC 7816-4 padding.         */
    MBEDTLS_PADDING_ZEROS_AND_LEN, /**< ANSI X.923 padding.             */
    MBEDTLS_PADDING_ZEROS,         /**< zero padding (not reversible). */
    MBEDTLS_PADDING_NONE,          /**< never pad (full blocks only).   */
} mbedtls_cipher_padding_t;

/** Type of operation. */
typedef enum {
    MBEDTLS_OPERATION_NONE = -1,
    MBEDTLS_DECRYPT = 0,
    MBEDTLS_ENCRYPT,
} mbedtls_operation_t;

enum {
    /** Undefined key length. */
    MBEDTLS_KEY_LENGTH_NONE = 0,
    /** Key length, in bits (including parity), for DES keys. */
    MBEDTLS_KEY_LENGTH_DES  = 64,
    /** Key length in bits, including parity, for DES in two-key EDE. */
    MBEDTLS_KEY_LENGTH_DES_EDE = 128,
    /** Key length in bits, including parity, for DES in three-key EDE. */
    MBEDTLS_KEY_LENGTH_DES_EDE3 = 192,
};

/** Maximum length of any IV, in Bytes. */
#define MBEDTLS_MAX_IV_LENGTH      16
/** Maximum block size of any cipher, in Bytes. */
#define MBEDTLS_MAX_BLOCK_LENGTH   16

/**
 * Base cipher information (opaque struct).
 */
typedef struct mbedtls_cipher_base_t mbedtls_cipher_base_t;

/**
 * CMAC context (opaque struct).
 */
typedef struct mbedtls_cmac_context_t mbedtls_cmac_context_t;

/**
 * Cipher information. Allows calling cipher functions
 * in a generic way.
 */
typedef struct {
    /** Full cipher identifier. For example,
     * MBEDTLS_CIPHER_AES_256_CBC.
     */
    mbedtls_cipher_type_t type;

    /** The cipher mode. For example, MBEDTLS_MODE_CBC. */
    mbedtls_cipher_mode_t mode;

    /** The cipher key length, in bits. This is the
     * default length for variable sized ciphers.
     * Includes parity bits for ciphers like DES.
     */
    unsigned int key_bitlen;

    /** Name of the cipher. */
    const char * name;

    /** IV or nonce size, in Bytes.
     * For ciphers that accept variable IV sizes,
     * this is the recommended size.
     */
    unsigned int iv_size;

    /** Flags to set. For example, if the cipher supports variable IV sizes or variable key sizes. */
    int flags;

    /** The block size, in Bytes. */
    unsigned int block_size;

    /** Struct for base cipher information and functions. */
    const mbedtls_cipher_base_t *base;

} mbedtls_cipher_info_t;

/**
 * Generic cipher context.
 */
typedef struct {
    /** Information about the associated cipher. */
    const mbedtls_cipher_info_t *cipher_info;

    /** Key length to use. */
    int key_bitlen;

    /** Operation that the key of the context has been
     * initialized for.
     */
    mbedtls_operation_t operation;

#if defined(MBEDTLS_CIPHER_MODE_WITH_PADDING)
    /** Padding functions to use, if relevant for
     * the specific cipher mode.
     */
    void (*add_padding)( unsigned char *output, size_t olen, size_t data_len );
    int (*get_padding)( unsigned char *input, size_t ilen, size_t *data_len );
#endif

    /** Buffer for input that has not been processed yet. */
    unsigned char unprocessed_data[MBEDTLS_MAX_BLOCK_LENGTH];

    /** Number of Bytes that have not been processed yet. */
    size_t unprocessed_len;

    /** Current IV or NONCE_COUNTER for CTR-mode. */
    unsigned char iv[MBEDTLS_MAX_IV_LENGTH];

    /** IV size in Bytes, for ciphers with variable-length IVs. */
    size_t iv_size;

    /** The cipher-specific context. */
    void *cipher_ctx;

#if defined(MBEDTLS_CMAC_C)
    /** CMAC-specific context. */
    mbedtls_cmac_context_t *cmac_ctx;
#endif
} mbedtls_cipher_context_t;

/**
 * \brief This function retrieves the list of ciphers supported by the generic
 * cipher module.
 *
 * \return      A statically-allocated array of ciphers. The last entry
 *              is zero.
 */
const int *mbedtls_cipher_list( void );

/**
 * \brief               This function retrieves the cipher-information
 *                      structure associated with the given cipher name.
 *
 * \param cipher_name   Name of the cipher to search for.
 *
 * \return              The cipher information structure associated with the
 *                      given \p cipher_name, or NULL if not found.
 */
const mbedtls_cipher_info_t *mbedtls_cipher_info_from_string( const char *cipher_name );

/**
 * \brief               This function retrieves the cipher-information
 *                      structure associated with the given cipher type.
 *
 * \param cipher_type   Type of the cipher to search for.
 *
 * \return              The cipher information structure associated with the
 *                      given \p cipher_type, or NULL if not found.
 */
const mbedtls_cipher_info_t *mbedtls_cipher_info_from_type( const mbedtls_cipher_type_t cipher_type );

/**
 * \brief               This function retrieves the cipher-information
 *                      structure associated with the given cipher ID,
 *                      key size and mode.
 *
 * \param cipher_id     The ID of the cipher to search for. For example,
 *                      #MBEDTLS_CIPHER_ID_AES.
 * \param key_bitlen    The length of the key in bits.
 * \param mode          The cipher mode. For example, #MBEDTLS_MODE_CBC.
 *
 * \return              The cipher information structure associated with the
 *                      given \p cipher_id, or NULL if not found.
 */
const mbedtls_cipher_info_t *mbedtls_cipher_info_from_values( const mbedtls_cipher_id_t cipher_id,
                                              int key_bitlen,
                                              const mbedtls_cipher_mode_t mode );

/**
 * \brief               This function initializes a \p cipher_context as NONE.
 */
void mbedtls_cipher_init( mbedtls_cipher_context_t *ctx );

/**
 * \brief               This function frees and clears the cipher-specific
 *                      context of \p ctx. Freeing \p ctx itself remains the
 *                      responsibility of the caller.
 */
void mbedtls_cipher_free( mbedtls_cipher_context_t *ctx );


/**
 * \brief               This function initializes and fills the cipher-context
 *                      structure with the appropriate values. It also clears
 *                      the structure.
 *
 * \param ctx           The context to initialize. May not be NULL.
 * \param cipher_info   The cipher to use.
 *
 * \return              \c 0 on success,
 *                      #MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA on parameter failure,
 *                      #MBEDTLS_ERR_CIPHER_ALLOC_FAILED if allocation of the
 *                      cipher-specific context failed.
 *
 * \internal Currently, the function also clears the structure.
 * In future versions, the caller will be required to call
 * mbedtls_cipher_init() on the structure first.
 */
int mbedtls_cipher_setup( mbedtls_cipher_context_t *ctx, const mbedtls_cipher_info_t *cipher_info );

/**
 * \brief        This function returns the block size of the given cipher.
 *
 * \param ctx    The context of the cipher. Must be initialized.
 *
 * \return       The size of the blocks of the cipher, or zero if \p ctx
 *               has not been initialized.
 */
static inline unsigned int mbedtls_cipher_get_block_size( const mbedtls_cipher_context_t *ctx )
{
    if( NULL == ctx || NULL == ctx->cipher_info )
        return 0;

    return ctx->cipher_info->block_size;
}

/**
 * \brief        This function returns the mode of operation for
 *               the cipher. For example, MBEDTLS_MODE_CBC.
 *
 * \param ctx    The context of the cipher. Must be initialized.
 *
 * \return       The mode of operation, or #MBEDTLS_MODE_NONE if
 *               \p ctx has not been initialized.
 */
static inline mbedtls_cipher_mode_t mbedtls_cipher_get_cipher_mode( const mbedtls_cipher_context_t *ctx )
{
    if( NULL == ctx || NULL == ctx->cipher_info )
        return MBEDTLS_MODE_NONE;

    return ctx->cipher_info->mode;
}

/**
 * \brief       This function returns the size of the IV or nonce
 *              of the cipher, in Bytes.
 *
 * \param ctx   The context of the cipher. Must be initialized.
 *
 * \return      <ul><li>If no IV has been set: the recommended IV size.
 *              0 for ciphers not using IV or nonce.</li>
 *              <li>If IV has already been set: the actual size.</li></ul>
 */
static inline int mbedtls_cipher_get_iv_size( const mbedtls_cipher_context_t *ctx )
{
    if( NULL == ctx || NULL == ctx->cipher_info )
        return 0;

    if( ctx->iv_size != 0 )
        return (int) ctx->iv_size;

    return (int) ctx->cipher_info->iv_size;
}

/**
 * \brief               This function returns the type of the given cipher.
 *
 * \param ctx           The context of the cipher. Must be initialized.
 *
 * \return              The type of the cipher, or #MBEDTLS_CIPHER_NONE if
 *                      \p ctx has not been initialized.
 */
static inline mbedtls_cipher_type_t mbedtls_cipher_get_type( const mbedtls_cipher_context_t *ctx )
{
    if( NULL == ctx || NULL == ctx->cipher_info )
        return MBEDTLS_CIPHER_NONE;

    return ctx->cipher_info->type;
}

/**
 * \brief               This function returns the name of the given cipher
 *                      as a string.
 *
 * \param ctx           The context of the cipher. Must be initialized.
 *
 * \return              The name of the cipher, or NULL if \p ctx has not
 *                      been not initialized.
 */
static inline const char *mbedtls_cipher_get_name( const mbedtls_cipher_context_t *ctx )
{
    if( NULL == ctx || NULL == ctx->cipher_info )
        return 0;

    return ctx->cipher_info->name;
}

/**
 * \brief               This function returns the key length of the cipher.
 *
 * \param ctx           The context of the cipher. Must be initialized.
 *
 * \return              The key length of the cipher in bits, or
 *                      #MBEDTLS_KEY_LENGTH_NONE if ctx \p has not been
 *                      initialized.
 */
static inline int mbedtls_cipher_get_key_bitlen( const mbedtls_cipher_context_t *ctx )
{
    if( NULL == ctx || NULL == ctx->cipher_info )
        return MBEDTLS_KEY_LENGTH_NONE;

    return (int) ctx->cipher_info->key_bitlen;
}

/**
 * \brief          This function returns the operation of the given cipher.
 *
 * \param ctx      The context of the cipher. Must be initialized.
 *
 * \return         The type of operation: #MBEDTLS_ENCRYPT or
 *                 #MBEDTLS_DECRYPT, or #MBEDTLS_OPERATION_NONE if \p ctx
 *                 has not been initialized.
 */
static inline mbedtls_operation_t mbedtls_cipher_get_operation( const mbedtls_cipher_context_t *ctx )
{
    if( NULL == ctx || NULL == ctx->cipher_info )
        return MBEDTLS_OPERATION_NONE;

    return ctx->operation;
}

/**
 * \brief               This function sets the key to use with the given context.
 *
 * \param ctx           The generic cipher context. May not be NULL. Must have
 *                      been initialized using mbedtls_cipher_info_from_type()
 *                      or mbedtls_cipher_info_from_string().
 * \param key           The key to use.
 * \param key_bitlen    The key length to use, in bits.
 * \param operation     The operation that the key will be used for:
 *                      #MBEDTLS_ENCRYPT or #MBEDTLS_DECRYPT.
 *
 * \returns             \c 0 on success, #MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA if
 *                      parameter verification fails, or a cipher-specific
 *                      error code.
 */
int mbedtls_cipher_setkey( mbedtls_cipher_context_t *ctx, const unsigned char *key,
                   int key_bitlen, const mbedtls_operation_t operation );

#if defined(MBEDTLS_CIPHER_MODE_WITH_PADDING)
/**
 * \brief               This function sets the padding mode, for cipher modes
 *                      that use padding.
 *
 *                      The default passing mode is PKCS7 padding.
 *
 * \param ctx           The generic cipher context.
 * \param mode          The padding mode.
 *
 * \returns             \c 0 on success, #MBEDTLS_ERR_CIPHER_FEATURE_UNAVAILABLE
 *                      if the selected padding mode is not supported, or
 *                      #MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA if the cipher mode
 *                      does not support padding.
 */
int mbedtls_cipher_set_padding_mode( mbedtls_cipher_context_t *ctx, mbedtls_cipher_padding_t mode );
#endif /* MBEDTLS_CIPHER_MODE_WITH_PADDING */

/**
 * \brief           This function sets the initialization vector (IV)
 *                  or nonce.
 *
 * \param ctx       The generic cipher context.
 * \param iv        The IV to use, or NONCE_COUNTER for CTR-mode ciphers.
 * \param iv_len    The IV length for ciphers with variable-size IV.
 *                  This parameter is discarded by ciphers with fixed-size IV.
 *
 * \returns         \c 0 on success, or #MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA
 *
 * \note            Some ciphers do not use IVs nor nonce. For these
 *                  ciphers, this function has no effect.
 */
int mbedtls_cipher_set_iv( mbedtls_cipher_context_t *ctx,
                   const unsigned char *iv, size_t iv_len );

/**
 * \brief         This function resets the cipher state.
 *
 * \param ctx     The generic cipher context.
 *
 * \returns       \c 0 on success, #MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA
 *                if parameter verification fails.
 */
int mbedtls_cipher_reset( mbedtls_cipher_context_t *ctx );

#if defined(MBEDTLS_GCM_C)
/**
 * \brief               This function adds additional data for AEAD ciphers.
 *                      Only supported with GCM. Must be called
 *                      exactly once, after mbedtls_cipher_reset().
 *
 * \param ctx           The generic cipher context.
 * \param ad            The additional data to use.
 * \param ad_len        the Length of \p ad.
 *
 * \return              \c 0 on success, or a specific error code on failure.
 */
int mbedtls_cipher_update_ad( mbedtls_cipher_context_t *ctx,
                      const unsigned char *ad, size_t ad_len );
#endif /* MBEDTLS_GCM_C */

/**
 * \brief               The generic cipher update function. It encrypts or
 *                      decrypts using the given cipher context. Writes as
 *                      many block-sized blocks of data as possible to output.
 *                      Any data that cannot be written immediately is either
 *                      added to the next block, or flushed when
 *                      mbedtls_cipher_finish() is called.
 *                      Exception: For MBEDTLS_MODE_ECB, expects a single block
 *                      in size. For example, 16 Bytes for AES.
 *
 * \param ctx           The generic cipher context.
 * \param input         The buffer holding the input data.
 * \param ilen          The length of the input data.
 * \param output        The buffer for the output data. Must be able to hold at
 *                      least \p ilen + block_size. Must not be the same buffer
 *                      as input.
 * \param olen          The length of the output data, to be updated with the
 *                      actual number of Bytes written.
 *
 * \returns             \c 0 on success, #MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA if
 *                      parameter verification fails,
 *                      #MBEDTLS_ERR_CIPHER_FEATURE_UNAVAILABLE on an
 *                      unsupported mode for a cipher, or a cipher-specific
 *                      error code.
 *
 * \note                If the underlying cipher is GCM, all calls to this
 *                      function, except the last one before
 *                      mbedtls_cipher_finish(). Must have \p ilen as a
 *                      multiple of the block_size.
 */
int mbedtls_cipher_update( mbedtls_cipher_context_t *ctx, const unsigned char *input,
                   size_t ilen, unsigned char *output, size_t *olen );

/**
 * \brief               The generic cipher finalization function. If data still
 *                      needs to be flushed from an incomplete block, the data
 *                      contained in it is padded to the size of
 *                      the last block, and written to the \p output buffer.
 *
 * \param ctx           The generic cipher context.
 * \param output        The buffer to write data to. Needs block_size available.
 * \param olen          The length of the data written to the \p output buffer.
 *
 * \returns             \c 0 on success, #MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA if
 *                      parameter verification fails,
 *                      #MBEDTLS_ERR_CIPHER_FULL_BLOCK_EXPECTED if decryption
 *                      expected a full block but was not provided one,
 *                      #MBEDTLS_ERR_CIPHER_INVALID_PADDING on invalid padding
 *                      while decrypting, or a cipher-specific error code
 *                      on failure for any other reason.
 */
int mbedtls_cipher_finish( mbedtls_cipher_context_t *ctx,
                   unsigned char *output, size_t *olen );

#if defined(MBEDTLS_GCM_C)
/**
 * \brief               This function writes a tag for AEAD ciphers.
 *                      Only supported with GCM.
 *                      Must be called after mbedtls_cipher_finish().
 *
 * \param ctx           The generic cipher context.
 * \param tag           The buffer to write the tag to.
 * \param tag_len       The length of the tag to write.
 *
 * \return              \c 0 on success, or a specific error code on failure.
 */
int mbedtls_cipher_write_tag( mbedtls_cipher_context_t *ctx,
                      unsigned char *tag, size_t tag_len );

/**
 * \brief               This function checks the tag for AEAD ciphers.
 *                      Only supported with GCM.
 *                      Must be called after mbedtls_cipher_finish().
 *
 * \param ctx           The generic cipher context.
 * \param tag           The buffer holding the tag.
 * \param tag_len       The length of the tag to check.
 *
 * \return              \c 0 on success, or a specific error code on failure.
 */
int mbedtls_cipher_check_tag( mbedtls_cipher_context_t *ctx,
                      const unsigned char *tag, size_t tag_len );
#endif /* MBEDTLS_GCM_C */

/**
 * \brief               The generic all-in-one encryption/decryption function,
 *                      for all ciphers except AEAD constructs.
 *
 * \param ctx           The generic cipher context.
 * \param iv            The IV to use, or NONCE_COUNTER for CTR-mode ciphers.
 * \param iv_len        The IV length for ciphers with variable-size IV.
 *                      This parameter is discarded by ciphers with fixed-size
 *                      IV.
 * \param input         The buffer holding the input data.
 * \param ilen          The length of the input data.
 * \param output        The buffer for the output data. Must be able to hold at
 *                      least \p ilen + block_size. Must not be the same buffer
 *                      as input.
 * \param olen          The length of the output data, to be updated with the
 *                      actual number of Bytes written.
 *
 * \note                Some ciphers do not use IVs nor nonce. For these
 *                      ciphers, use \p iv = NULL and \p iv_len = 0.
 *
 * \returns             \c 0 on success, or
 *                      #MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA, or
 *                      #MBEDTLS_ERR_CIPHER_FULL_BLOCK_EXPECTED if decryption
 *                      expected a full block but was not provided one, or
 *                      #MBEDTLS_ERR_CIPHER_INVALID_PADDING on invalid padding
 *                      while decrypting, or a cipher-specific error code on
 *                      failure for any other reason.
 */
int mbedtls_cipher_crypt( mbedtls_cipher_context_t *ctx,
                  const unsigned char *iv, size_t iv_len,
                  const unsigned char *input, size_t ilen,
                  unsigned char *output, size_t *olen );

#if defined(MBEDTLS_CIPHER_MODE_AEAD)
/**
 * \brief               The generic autenticated encryption (AEAD) function.
 *
 * \param ctx           The generic cipher context.
 * \param iv            The IV to use, or NONCE_COUNTER for CTR-mode ciphers.
 * \param iv_len        The IV length for ciphers with variable-size IV.
 *                      This parameter is discarded by ciphers with fixed-size IV.
 * \param ad            The additional data to authenticate.
 * \param ad_len        The length of \p ad.
 * \param input         The buffer holding the input data.
 * \param ilen          The length of the input data.
 * \param output        The buffer for the output data.
 *                      Must be able to hold at least \p ilen.
 * \param olen          The length of the output data, to be updated with the
 *                      actual number of Bytes written.
 * \param tag           The buffer for the authentication tag.
 * \param tag_len       The desired length of the authentication tag.
 *
 * \returns             \c 0 on success, or
 *                      #MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA, or
 *                      a cipher-specific error code.
 */
int mbedtls_cipher_auth_encrypt( mbedtls_cipher_context_t *ctx,
                         const unsigned char *iv, size_t iv_len,
                         const unsigned char *ad, size_t ad_len,
                         const unsigned char *input, size_t ilen,
                         unsigned char *output, size_t *olen,
                         unsigned char *tag, size_t tag_len );

/**
 * \brief               The generic autenticated decryption (AEAD) function.
 *
 * \param ctx           The generic cipher context.
 * \param iv            The IV to use, or NONCE_COUNTER for CTR-mode ciphers.
 * \param iv_len        The IV length for ciphers with variable-size IV.
 *                      This parameter is discarded by ciphers with fixed-size IV.
 * \param ad            The additional data to be authenticated.
 * \param ad_len        The length of \p ad.
 * \param input         The buffer holding the input data.
 * \param ilen          The length of the input data.
 * \param output        The buffer for the output data.
 *                      Must be able to hold at least \p ilen.
 * \param olen          The length of the output data, to be updated with the
 *                      actual number of Bytes written.
 * \param tag           The buffer holding the authentication tag.
 * \param tag_len       The length of the authentication tag.
 *
 * \returns             \c 0 on success, or
 *                      #MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA, or
 *                      #MBEDTLS_ERR_CIPHER_AUTH_FAILED if data is not authentic,
 *                      or a cipher-specific error code on failure for any other reason.
 *
 * \note                If the data is not authentic, then the output buffer
 *                      is zeroed out to prevent the unauthentic plaintext being
 *                      used, making this interface safer.
 */
int mbedtls_cipher_auth_decrypt( mbedtls_cipher_context_t *ctx,
                         const unsigned char *iv, size_t iv_len,
                         const unsigned char *ad, size_t ad_len,
                         const unsigned char *input, size_t ilen,
                         unsigned char *output, size_t *olen,
                         const unsigned char *tag, size_t tag_len );
#endif /* MBEDTLS_CIPHER_MODE_AEAD */

#ifdef __cplusplus
}
#endif

#endif /* MBEDTLS_CIPHER_H */
