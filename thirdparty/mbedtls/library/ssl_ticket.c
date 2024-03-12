/*
 *  TLS server tickets callbacks implementation
 *
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */

#include "common.h"

#if defined(MBEDTLS_SSL_TICKET_C)

#include "mbedtls/platform.h"

#include "mbedtls/ssl_internal.h"
#include "mbedtls/ssl_ticket.h"
#include "mbedtls/error.h"
#include "mbedtls/platform_util.h"

#include <string.h>

/*
 * Initialize context
 */
void mbedtls_ssl_ticket_init(mbedtls_ssl_ticket_context *ctx)
{
    memset(ctx, 0, sizeof(mbedtls_ssl_ticket_context));

#if defined(MBEDTLS_THREADING_C)
    mbedtls_mutex_init(&ctx->mutex);
#endif
}

#define MAX_KEY_BYTES 32    /* 256 bits */

#define TICKET_KEY_NAME_BYTES    4
#define TICKET_IV_BYTES         12
#define TICKET_CRYPT_LEN_BYTES   2
#define TICKET_AUTH_TAG_BYTES   16

#define TICKET_MIN_LEN (TICKET_KEY_NAME_BYTES  +        \
                        TICKET_IV_BYTES        +        \
                        TICKET_CRYPT_LEN_BYTES +        \
                        TICKET_AUTH_TAG_BYTES)
#define TICKET_ADD_DATA_LEN (TICKET_KEY_NAME_BYTES  +        \
                             TICKET_IV_BYTES        +        \
                             TICKET_CRYPT_LEN_BYTES)

/*
 * Generate/update a key
 */
MBEDTLS_CHECK_RETURN_CRITICAL
static int ssl_ticket_gen_key(mbedtls_ssl_ticket_context *ctx,
                              unsigned char index)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    unsigned char buf[MAX_KEY_BYTES];
    mbedtls_ssl_ticket_key *key = ctx->keys + index;

#if defined(MBEDTLS_HAVE_TIME)
    key->generation_time = (uint32_t) mbedtls_time(NULL);
#endif

    if ((ret = ctx->f_rng(ctx->p_rng, key->name, sizeof(key->name))) != 0) {
        return ret;
    }

    if ((ret = ctx->f_rng(ctx->p_rng, buf, sizeof(buf))) != 0) {
        return ret;
    }

    /* With GCM and CCM, same context can encrypt & decrypt */
    ret = mbedtls_cipher_setkey(&key->ctx, buf,
                                mbedtls_cipher_get_key_bitlen(&key->ctx),
                                MBEDTLS_ENCRYPT);

    mbedtls_platform_zeroize(buf, sizeof(buf));

    return ret;
}

/*
 * Rotate/generate keys if necessary
 */
MBEDTLS_CHECK_RETURN_CRITICAL
static int ssl_ticket_update_keys(mbedtls_ssl_ticket_context *ctx)
{
#if !defined(MBEDTLS_HAVE_TIME)
    ((void) ctx);
#else
    if (ctx->ticket_lifetime != 0) {
        uint32_t current_time = (uint32_t) mbedtls_time(NULL);
        uint32_t key_time = ctx->keys[ctx->active].generation_time;

        if (current_time >= key_time &&
            current_time - key_time < ctx->ticket_lifetime) {
            return 0;
        }

        ctx->active = 1 - ctx->active;

        return ssl_ticket_gen_key(ctx, ctx->active);
    } else
#endif /* MBEDTLS_HAVE_TIME */
    return 0;
}

/*
 * Setup context for actual use
 */
int mbedtls_ssl_ticket_setup(mbedtls_ssl_ticket_context *ctx,
                             int (*f_rng)(void *, unsigned char *, size_t), void *p_rng,
                             mbedtls_cipher_type_t cipher,
                             uint32_t lifetime)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    const mbedtls_cipher_info_t *cipher_info;

    ctx->f_rng = f_rng;
    ctx->p_rng = p_rng;

    ctx->ticket_lifetime = lifetime;

    cipher_info = mbedtls_cipher_info_from_type(cipher);
    if (cipher_info == NULL) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    if (cipher_info->mode != MBEDTLS_MODE_GCM &&
        cipher_info->mode != MBEDTLS_MODE_CCM) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    if (cipher_info->key_bitlen > 8 * MAX_KEY_BYTES) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    int do_mbedtls_cipher_setup = 1;
#if defined(MBEDTLS_USE_PSA_CRYPTO)
    ret = mbedtls_cipher_setup_psa(&ctx->keys[0].ctx,
                                   cipher_info, TICKET_AUTH_TAG_BYTES);

    switch (ret) {
        case 0:
            do_mbedtls_cipher_setup = 0;
            break;
        case MBEDTLS_ERR_CIPHER_FEATURE_UNAVAILABLE:
            /* We don't yet expect to support all ciphers through PSA,
             * so allow fallback to ordinary mbedtls_cipher_setup(). */
            do_mbedtls_cipher_setup = 1;
            break;
        default:
            return ret;
    }
#endif /* MBEDTLS_USE_PSA_CRYPTO */
    if (do_mbedtls_cipher_setup) {
        if ((ret = mbedtls_cipher_setup(&ctx->keys[0].ctx, cipher_info))
            != 0) {
            return ret;
        }
    }

    do_mbedtls_cipher_setup = 1;
#if defined(MBEDTLS_USE_PSA_CRYPTO)
    do_mbedtls_cipher_setup = 0;

    ret = mbedtls_cipher_setup_psa(&ctx->keys[1].ctx,
                                   cipher_info, TICKET_AUTH_TAG_BYTES);
    if (ret != 0 && ret != MBEDTLS_ERR_CIPHER_FEATURE_UNAVAILABLE) {
        return ret;
    }
    if (ret == MBEDTLS_ERR_CIPHER_FEATURE_UNAVAILABLE) {
        do_mbedtls_cipher_setup = 1;
    }
#endif /* MBEDTLS_USE_PSA_CRYPTO */
    if (do_mbedtls_cipher_setup) {
        if ((ret = mbedtls_cipher_setup(&ctx->keys[1].ctx, cipher_info))
            != 0) {
            return ret;
        }
    }

    if ((ret = ssl_ticket_gen_key(ctx, 0)) != 0 ||
        (ret = ssl_ticket_gen_key(ctx, 1)) != 0) {
        return ret;
    }

    return 0;
}

/*
 * Create session ticket, with the following structure:
 *
 *    struct {
 *        opaque key_name[4];
 *        opaque iv[12];
 *        opaque encrypted_state<0..2^16-1>;
 *        opaque tag[16];
 *    } ticket;
 *
 * The key_name, iv, and length of encrypted_state are the additional
 * authenticated data.
 */

int mbedtls_ssl_ticket_write(void *p_ticket,
                             const mbedtls_ssl_session *session,
                             unsigned char *start,
                             const unsigned char *end,
                             size_t *tlen,
                             uint32_t *ticket_lifetime)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    mbedtls_ssl_ticket_context *ctx = p_ticket;
    mbedtls_ssl_ticket_key *key;
    unsigned char *key_name = start;
    unsigned char *iv = start + TICKET_KEY_NAME_BYTES;
    unsigned char *state_len_bytes = iv + TICKET_IV_BYTES;
    unsigned char *state = state_len_bytes + TICKET_CRYPT_LEN_BYTES;
    size_t clear_len, ciph_len;

    *tlen = 0;

    if (ctx == NULL || ctx->f_rng == NULL) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    /* We need at least 4 bytes for key_name, 12 for IV, 2 for len 16 for tag,
     * in addition to session itself, that will be checked when writing it. */
    MBEDTLS_SSL_CHK_BUF_PTR(start, end, TICKET_MIN_LEN);

#if defined(MBEDTLS_THREADING_C)
    if ((ret = mbedtls_mutex_lock(&ctx->mutex)) != 0) {
        return ret;
    }
#endif

    if ((ret = ssl_ticket_update_keys(ctx)) != 0) {
        goto cleanup;
    }

    key = &ctx->keys[ctx->active];

    *ticket_lifetime = ctx->ticket_lifetime;

    memcpy(key_name, key->name, TICKET_KEY_NAME_BYTES);

    if ((ret = ctx->f_rng(ctx->p_rng, iv, TICKET_IV_BYTES)) != 0) {
        goto cleanup;
    }

    /* Dump session state */
    if ((ret = mbedtls_ssl_session_save(session,
                                        state, end - state,
                                        &clear_len)) != 0 ||
        (unsigned long) clear_len > 65535) {
        goto cleanup;
    }
    MBEDTLS_PUT_UINT16_BE(clear_len, state_len_bytes, 0);

    /* Encrypt and authenticate */
    if ((ret = mbedtls_cipher_auth_encrypt_ext(&key->ctx,
                                               iv, TICKET_IV_BYTES,
                                               /* Additional data: key name, IV and length */
                                               key_name, TICKET_ADD_DATA_LEN,
                                               state, clear_len,
                                               state, end - state, &ciph_len,
                                               TICKET_AUTH_TAG_BYTES)) != 0) {
        goto cleanup;
    }
    if (ciph_len != clear_len + TICKET_AUTH_TAG_BYTES) {
        ret = MBEDTLS_ERR_SSL_INTERNAL_ERROR;
        goto cleanup;
    }

    *tlen = TICKET_MIN_LEN + ciph_len - TICKET_AUTH_TAG_BYTES;

cleanup:
#if defined(MBEDTLS_THREADING_C)
    if (mbedtls_mutex_unlock(&ctx->mutex) != 0) {
        return MBEDTLS_ERR_THREADING_MUTEX_ERROR;
    }
#endif

    return ret;
}

/*
 * Select key based on name
 */
static mbedtls_ssl_ticket_key *ssl_ticket_select_key(
    mbedtls_ssl_ticket_context *ctx,
    const unsigned char name[4])
{
    unsigned char i;

    for (i = 0; i < sizeof(ctx->keys) / sizeof(*ctx->keys); i++) {
        if (memcmp(name, ctx->keys[i].name, 4) == 0) {
            return &ctx->keys[i];
        }
    }

    return NULL;
}

/*
 * Load session ticket (see mbedtls_ssl_ticket_write for structure)
 */
int mbedtls_ssl_ticket_parse(void *p_ticket,
                             mbedtls_ssl_session *session,
                             unsigned char *buf,
                             size_t len)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    mbedtls_ssl_ticket_context *ctx = p_ticket;
    mbedtls_ssl_ticket_key *key;
    unsigned char *key_name = buf;
    unsigned char *iv = buf + TICKET_KEY_NAME_BYTES;
    unsigned char *enc_len_p = iv + TICKET_IV_BYTES;
    unsigned char *ticket = enc_len_p + TICKET_CRYPT_LEN_BYTES;
    size_t enc_len, clear_len;

    if (ctx == NULL || ctx->f_rng == NULL) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    if (len < TICKET_MIN_LEN) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

#if defined(MBEDTLS_THREADING_C)
    if ((ret = mbedtls_mutex_lock(&ctx->mutex)) != 0) {
        return ret;
    }
#endif

    if ((ret = ssl_ticket_update_keys(ctx)) != 0) {
        goto cleanup;
    }

    enc_len = (enc_len_p[0] << 8) | enc_len_p[1];

    if (len != TICKET_MIN_LEN + enc_len) {
        ret = MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
        goto cleanup;
    }

    /* Select key */
    if ((key = ssl_ticket_select_key(ctx, key_name)) == NULL) {
        /* We can't know for sure but this is a likely option unless we're
         * under attack - this is only informative anyway */
        ret = MBEDTLS_ERR_SSL_SESSION_TICKET_EXPIRED;
        goto cleanup;
    }

    /* Decrypt and authenticate */
    if ((ret = mbedtls_cipher_auth_decrypt_ext(&key->ctx,
                                               iv, TICKET_IV_BYTES,
                                               /* Additional data: key name, IV and length */
                                               key_name, TICKET_ADD_DATA_LEN,
                                               ticket, enc_len + TICKET_AUTH_TAG_BYTES,
                                               ticket, enc_len, &clear_len,
                                               TICKET_AUTH_TAG_BYTES)) != 0) {
        if (ret == MBEDTLS_ERR_CIPHER_AUTH_FAILED) {
            ret = MBEDTLS_ERR_SSL_INVALID_MAC;
        }

        goto cleanup;
    }
    if (clear_len != enc_len) {
        ret = MBEDTLS_ERR_SSL_INTERNAL_ERROR;
        goto cleanup;
    }

    /* Actually load session */
    if ((ret = mbedtls_ssl_session_load(session, ticket, clear_len)) != 0) {
        goto cleanup;
    }

#if defined(MBEDTLS_HAVE_TIME)
    {
        /* Check for expiration */
        mbedtls_time_t current_time = mbedtls_time(NULL);

        if (current_time < session->start ||
            (uint32_t) (current_time - session->start) > ctx->ticket_lifetime) {
            ret = MBEDTLS_ERR_SSL_SESSION_TICKET_EXPIRED;
            goto cleanup;
        }
    }
#endif

cleanup:
#if defined(MBEDTLS_THREADING_C)
    if (mbedtls_mutex_unlock(&ctx->mutex) != 0) {
        return MBEDTLS_ERR_THREADING_MUTEX_ERROR;
    }
#endif

    return ret;
}

/*
 * Free context
 */
void mbedtls_ssl_ticket_free(mbedtls_ssl_ticket_context *ctx)
{
    mbedtls_cipher_free(&ctx->keys[0].ctx);
    mbedtls_cipher_free(&ctx->keys[1].ctx);

#if defined(MBEDTLS_THREADING_C)
    mbedtls_mutex_free(&ctx->mutex);
#endif

    mbedtls_platform_zeroize(ctx, sizeof(mbedtls_ssl_ticket_context));
}

#endif /* MBEDTLS_SSL_TICKET_C */
