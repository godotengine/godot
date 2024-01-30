/*
 *  SSLv3/TLSv1 shared functions
 *
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */
/*
 *  The SSL 3.0 specification was drafted by Netscape in 1996,
 *  and became an IETF standard in 1999.
 *
 *  http://wp.netscape.com/eng/ssl3/
 *  http://www.ietf.org/rfc/rfc2246.txt
 *  http://www.ietf.org/rfc/rfc4346.txt
 */

#include "common.h"

#if defined(MBEDTLS_SSL_TLS_C)

#include "mbedtls/platform.h"

#include "mbedtls/ssl.h"
#include "mbedtls/ssl_internal.h"
#include "mbedtls/debug.h"
#include "mbedtls/error.h"
#include "mbedtls/platform_util.h"
#include "mbedtls/version.h"
#include "mbedtls/constant_time.h"

#include <string.h>

#if defined(MBEDTLS_USE_PSA_CRYPTO)
#include "mbedtls/psa_util.h"
#include "psa/crypto.h"
#endif

#if defined(MBEDTLS_X509_CRT_PARSE_C)
#include "mbedtls/oid.h"
#endif

#if defined(MBEDTLS_SSL_PROTO_DTLS)

#if defined(MBEDTLS_SSL_DTLS_CONNECTION_ID)
/* Top-level Connection ID API */

int mbedtls_ssl_conf_cid(mbedtls_ssl_config *conf,
                         size_t len,
                         int ignore_other_cid)
{
    if (len > MBEDTLS_SSL_CID_IN_LEN_MAX) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    if (ignore_other_cid != MBEDTLS_SSL_UNEXPECTED_CID_FAIL &&
        ignore_other_cid != MBEDTLS_SSL_UNEXPECTED_CID_IGNORE) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    conf->ignore_unexpected_cid = ignore_other_cid;
    conf->cid_len = len;
    return 0;
}

int mbedtls_ssl_set_cid(mbedtls_ssl_context *ssl,
                        int enable,
                        unsigned char const *own_cid,
                        size_t own_cid_len)
{
    if (ssl->conf->transport != MBEDTLS_SSL_TRANSPORT_DATAGRAM) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    ssl->negotiate_cid = enable;
    if (enable == MBEDTLS_SSL_CID_DISABLED) {
        MBEDTLS_SSL_DEBUG_MSG(3, ("Disable use of CID extension."));
        return 0;
    }
    MBEDTLS_SSL_DEBUG_MSG(3, ("Enable use of CID extension."));
    MBEDTLS_SSL_DEBUG_BUF(3, "Own CID", own_cid, own_cid_len);

    if (own_cid_len != ssl->conf->cid_len) {
        MBEDTLS_SSL_DEBUG_MSG(3, ("CID length %u does not match CID length %u in config",
                                  (unsigned) own_cid_len,
                                  (unsigned) ssl->conf->cid_len));
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    memcpy(ssl->own_cid, own_cid, own_cid_len);
    /* Truncation is not an issue here because
     * MBEDTLS_SSL_CID_IN_LEN_MAX at most 255. */
    ssl->own_cid_len = (uint8_t) own_cid_len;

    return 0;
}

int mbedtls_ssl_get_peer_cid(mbedtls_ssl_context *ssl,
                             int *enabled,
                             unsigned char peer_cid[MBEDTLS_SSL_CID_OUT_LEN_MAX],
                             size_t *peer_cid_len)
{
    *enabled = MBEDTLS_SSL_CID_DISABLED;

    if (ssl->conf->transport != MBEDTLS_SSL_TRANSPORT_DATAGRAM ||
        ssl->state != MBEDTLS_SSL_HANDSHAKE_OVER) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    /* We report MBEDTLS_SSL_CID_DISABLED in case the CID extensions
     * were used, but client and server requested the empty CID.
     * This is indistinguishable from not using the CID extension
     * in the first place. */
    if (ssl->transform_in->in_cid_len  == 0 &&
        ssl->transform_in->out_cid_len == 0) {
        return 0;
    }

    if (peer_cid_len != NULL) {
        *peer_cid_len = ssl->transform_in->out_cid_len;
        if (peer_cid != NULL) {
            memcpy(peer_cid, ssl->transform_in->out_cid,
                   ssl->transform_in->out_cid_len);
        }
    }

    *enabled = MBEDTLS_SSL_CID_ENABLED;

    return 0;
}
#endif /* MBEDTLS_SSL_DTLS_CONNECTION_ID */

#endif /* MBEDTLS_SSL_PROTO_DTLS */

#if defined(MBEDTLS_SSL_MAX_FRAGMENT_LENGTH)
/*
 * Convert max_fragment_length codes to length.
 * RFC 6066 says:
 *    enum{
 *        2^9(1), 2^10(2), 2^11(3), 2^12(4), (255)
 *    } MaxFragmentLength;
 * and we add 0 -> extension unused
 */
static unsigned int ssl_mfl_code_to_length(int mfl)
{
    switch (mfl) {
        case MBEDTLS_SSL_MAX_FRAG_LEN_NONE:
            return MBEDTLS_TLS_EXT_ADV_CONTENT_LEN;
        case MBEDTLS_SSL_MAX_FRAG_LEN_512:
            return 512;
        case MBEDTLS_SSL_MAX_FRAG_LEN_1024:
            return 1024;
        case MBEDTLS_SSL_MAX_FRAG_LEN_2048:
            return 2048;
        case MBEDTLS_SSL_MAX_FRAG_LEN_4096:
            return 4096;
        default:
            return MBEDTLS_TLS_EXT_ADV_CONTENT_LEN;
    }
}
#endif /* MBEDTLS_SSL_MAX_FRAGMENT_LENGTH */

int mbedtls_ssl_session_copy(mbedtls_ssl_session *dst,
                             const mbedtls_ssl_session *src)
{
    mbedtls_ssl_session_free(dst);
    memcpy(dst, src, sizeof(mbedtls_ssl_session));

#if defined(MBEDTLS_SSL_SESSION_TICKETS) && defined(MBEDTLS_SSL_CLI_C)
    dst->ticket = NULL;
#endif

#if defined(MBEDTLS_X509_CRT_PARSE_C)

#if defined(MBEDTLS_SSL_KEEP_PEER_CERTIFICATE)
    if (src->peer_cert != NULL) {
        int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;

        dst->peer_cert = mbedtls_calloc(1, sizeof(mbedtls_x509_crt));
        if (dst->peer_cert == NULL) {
            return MBEDTLS_ERR_SSL_ALLOC_FAILED;
        }

        mbedtls_x509_crt_init(dst->peer_cert);

        if ((ret = mbedtls_x509_crt_parse_der(dst->peer_cert, src->peer_cert->raw.p,
                                              src->peer_cert->raw.len)) != 0) {
            mbedtls_free(dst->peer_cert);
            dst->peer_cert = NULL;
            return ret;
        }
    }
#else /* MBEDTLS_SSL_KEEP_PEER_CERTIFICATE */
    if (src->peer_cert_digest != NULL) {
        dst->peer_cert_digest =
            mbedtls_calloc(1, src->peer_cert_digest_len);
        if (dst->peer_cert_digest == NULL) {
            return MBEDTLS_ERR_SSL_ALLOC_FAILED;
        }

        memcpy(dst->peer_cert_digest, src->peer_cert_digest,
               src->peer_cert_digest_len);
        dst->peer_cert_digest_type = src->peer_cert_digest_type;
        dst->peer_cert_digest_len = src->peer_cert_digest_len;
    }
#endif /* MBEDTLS_SSL_KEEP_PEER_CERTIFICATE */

#endif /* MBEDTLS_X509_CRT_PARSE_C */

#if defined(MBEDTLS_SSL_SESSION_TICKETS) && defined(MBEDTLS_SSL_CLI_C)
    if (src->ticket != NULL) {
        dst->ticket = mbedtls_calloc(1, src->ticket_len);
        if (dst->ticket == NULL) {
            return MBEDTLS_ERR_SSL_ALLOC_FAILED;
        }

        memcpy(dst->ticket, src->ticket, src->ticket_len);
    }
#endif /* MBEDTLS_SSL_SESSION_TICKETS && MBEDTLS_SSL_CLI_C */

    return 0;
}

#if defined(MBEDTLS_SSL_VARIABLE_BUFFER_LENGTH)
MBEDTLS_CHECK_RETURN_CRITICAL
static int resize_buffer(unsigned char **buffer, size_t len_new, size_t *len_old)
{
    unsigned char *resized_buffer = mbedtls_calloc(1, len_new);
    if (resized_buffer == NULL) {
        return -1;
    }

    /* We want to copy len_new bytes when downsizing the buffer, and
     * len_old bytes when upsizing, so we choose the smaller of two sizes,
     * to fit one buffer into another. Size checks, ensuring that no data is
     * lost, are done outside of this function. */
    memcpy(resized_buffer, *buffer,
           (len_new < *len_old) ? len_new : *len_old);
    mbedtls_platform_zeroize(*buffer, *len_old);
    mbedtls_free(*buffer);

    *buffer = resized_buffer;
    *len_old = len_new;

    return 0;
}

static void handle_buffer_resizing(mbedtls_ssl_context *ssl, int downsizing,
                                   size_t in_buf_new_len,
                                   size_t out_buf_new_len)
{
    int modified = 0;
    size_t written_in = 0, iv_offset_in = 0, len_offset_in = 0;
    size_t written_out = 0, iv_offset_out = 0, len_offset_out = 0;
    if (ssl->in_buf != NULL) {
        written_in = ssl->in_msg - ssl->in_buf;
        iv_offset_in = ssl->in_iv - ssl->in_buf;
        len_offset_in = ssl->in_len - ssl->in_buf;
        if (downsizing ?
            ssl->in_buf_len > in_buf_new_len && ssl->in_left < in_buf_new_len :
            ssl->in_buf_len < in_buf_new_len) {
            if (resize_buffer(&ssl->in_buf, in_buf_new_len, &ssl->in_buf_len) != 0) {
                MBEDTLS_SSL_DEBUG_MSG(1, ("input buffer resizing failed - out of memory"));
            } else {
                MBEDTLS_SSL_DEBUG_MSG(2, ("Reallocating in_buf to %" MBEDTLS_PRINTF_SIZET,
                                          in_buf_new_len));
                modified = 1;
            }
        }
    }

    if (ssl->out_buf != NULL) {
        written_out = ssl->out_msg - ssl->out_buf;
        iv_offset_out = ssl->out_iv - ssl->out_buf;
        len_offset_out = ssl->out_len - ssl->out_buf;
        if (downsizing ?
            ssl->out_buf_len > out_buf_new_len && ssl->out_left < out_buf_new_len :
            ssl->out_buf_len < out_buf_new_len) {
            if (resize_buffer(&ssl->out_buf, out_buf_new_len, &ssl->out_buf_len) != 0) {
                MBEDTLS_SSL_DEBUG_MSG(1, ("output buffer resizing failed - out of memory"));
            } else {
                MBEDTLS_SSL_DEBUG_MSG(2, ("Reallocating out_buf to %" MBEDTLS_PRINTF_SIZET,
                                          out_buf_new_len));
                modified = 1;
            }
        }
    }
    if (modified) {
        /* Update pointers here to avoid doing it twice. */
        mbedtls_ssl_reset_in_out_pointers(ssl);
        /* Fields below might not be properly updated with record
         * splitting or with CID, so they are manually updated here. */
        ssl->out_msg = ssl->out_buf + written_out;
        ssl->out_len = ssl->out_buf + len_offset_out;
        ssl->out_iv = ssl->out_buf + iv_offset_out;

        ssl->in_msg = ssl->in_buf + written_in;
        ssl->in_len = ssl->in_buf + len_offset_in;
        ssl->in_iv = ssl->in_buf + iv_offset_in;
    }
}
#endif /* MBEDTLS_SSL_VARIABLE_BUFFER_LENGTH */

/*
 * Key material generation
 */
#if defined(MBEDTLS_SSL_PROTO_SSL3)
MBEDTLS_CHECK_RETURN_CRITICAL
static int ssl3_prf(const unsigned char *secret, size_t slen,
                    const char *label,
                    const unsigned char *random, size_t rlen,
                    unsigned char *dstbuf, size_t dlen)
{
    int ret = 0;
    size_t i;
    mbedtls_md5_context md5;
    mbedtls_sha1_context sha1;
    unsigned char padding[16];
    unsigned char sha1sum[20];
    ((void) label);

    mbedtls_md5_init(&md5);
    mbedtls_sha1_init(&sha1);

    /*
     *  SSLv3:
     *    block =
     *      MD5( secret + SHA1( 'A'    + secret + random ) ) +
     *      MD5( secret + SHA1( 'BB'   + secret + random ) ) +
     *      MD5( secret + SHA1( 'CCC'  + secret + random ) ) +
     *      ...
     */
    for (i = 0; i < dlen / 16; i++) {
        memset(padding, (unsigned char) ('A' + i), 1 + i);

        if ((ret = mbedtls_sha1_starts_ret(&sha1)) != 0) {
            goto exit;
        }
        if ((ret = mbedtls_sha1_update_ret(&sha1, padding, 1 + i)) != 0) {
            goto exit;
        }
        if ((ret = mbedtls_sha1_update_ret(&sha1, secret, slen)) != 0) {
            goto exit;
        }
        if ((ret = mbedtls_sha1_update_ret(&sha1, random, rlen)) != 0) {
            goto exit;
        }
        if ((ret = mbedtls_sha1_finish_ret(&sha1, sha1sum)) != 0) {
            goto exit;
        }

        if ((ret = mbedtls_md5_starts_ret(&md5)) != 0) {
            goto exit;
        }
        if ((ret = mbedtls_md5_update_ret(&md5, secret, slen)) != 0) {
            goto exit;
        }
        if ((ret = mbedtls_md5_update_ret(&md5, sha1sum, 20)) != 0) {
            goto exit;
        }
        if ((ret = mbedtls_md5_finish_ret(&md5, dstbuf + i * 16)) != 0) {
            goto exit;
        }
    }

exit:
    mbedtls_md5_free(&md5);
    mbedtls_sha1_free(&sha1);

    mbedtls_platform_zeroize(padding, sizeof(padding));
    mbedtls_platform_zeroize(sha1sum, sizeof(sha1sum));

    return ret;
}
#endif /* MBEDTLS_SSL_PROTO_SSL3 */

#if defined(MBEDTLS_SSL_PROTO_TLS1) || defined(MBEDTLS_SSL_PROTO_TLS1_1)
MBEDTLS_CHECK_RETURN_CRITICAL
static int tls1_prf(const unsigned char *secret, size_t slen,
                    const char *label,
                    const unsigned char *random, size_t rlen,
                    unsigned char *dstbuf, size_t dlen)
{
    size_t nb, hs;
    size_t i, j, k;
    const unsigned char *S1, *S2;
    unsigned char *tmp;
    size_t tmp_len = 0;
    unsigned char h_i[20];
    const mbedtls_md_info_t *md_info;
    mbedtls_md_context_t md_ctx;
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;

    mbedtls_md_init(&md_ctx);

    tmp_len = 20 + strlen(label) + rlen;
    tmp = mbedtls_calloc(1, tmp_len);
    if (tmp == NULL) {
        ret = MBEDTLS_ERR_SSL_ALLOC_FAILED;
        goto exit;
    }

    hs = (slen + 1) / 2;
    S1 = secret;
    S2 = secret + slen - hs;

    nb = strlen(label);
    memcpy(tmp + 20, label, nb);
    memcpy(tmp + 20 + nb, random, rlen);
    nb += rlen;

    /*
     * First compute P_md5(secret,label+random)[0..dlen]
     */
    if ((md_info = mbedtls_md_info_from_type(MBEDTLS_MD_MD5)) == NULL) {
        ret = MBEDTLS_ERR_SSL_INTERNAL_ERROR;
        goto exit;
    }

    if ((ret = mbedtls_md_setup(&md_ctx, md_info, 1)) != 0) {
        goto exit;
    }

    ret = mbedtls_md_hmac_starts(&md_ctx, S1, hs);
    if (ret != 0) {
        goto exit;
    }
    ret = mbedtls_md_hmac_update(&md_ctx, tmp + 20, nb);
    if (ret != 0) {
        goto exit;
    }
    ret = mbedtls_md_hmac_finish(&md_ctx, 4 + tmp);
    if (ret != 0) {
        goto exit;
    }

    for (i = 0; i < dlen; i += 16) {
        ret = mbedtls_md_hmac_reset(&md_ctx);
        if (ret != 0) {
            goto exit;
        }
        ret = mbedtls_md_hmac_update(&md_ctx, 4 + tmp, 16 + nb);
        if (ret != 0) {
            goto exit;
        }
        ret = mbedtls_md_hmac_finish(&md_ctx, h_i);
        if (ret != 0) {
            goto exit;
        }

        ret = mbedtls_md_hmac_reset(&md_ctx);
        if (ret != 0) {
            goto exit;
        }
        ret = mbedtls_md_hmac_update(&md_ctx, 4 + tmp, 16);
        if (ret != 0) {
            goto exit;
        }
        ret = mbedtls_md_hmac_finish(&md_ctx, 4 + tmp);
        if (ret != 0) {
            goto exit;
        }

        k = (i + 16 > dlen) ? dlen % 16 : 16;

        for (j = 0; j < k; j++) {
            dstbuf[i + j]  = h_i[j];
        }
    }

    mbedtls_md_free(&md_ctx);

    /*
     * XOR out with P_sha1(secret,label+random)[0..dlen]
     */
    if ((md_info = mbedtls_md_info_from_type(MBEDTLS_MD_SHA1)) == NULL) {
        ret = MBEDTLS_ERR_SSL_INTERNAL_ERROR;
        goto exit;
    }

    if ((ret = mbedtls_md_setup(&md_ctx, md_info, 1)) != 0) {
        goto exit;
    }

    ret = mbedtls_md_hmac_starts(&md_ctx, S2, hs);
    if (ret != 0) {
        goto exit;
    }
    ret = mbedtls_md_hmac_update(&md_ctx, tmp + 20, nb);
    if (ret != 0) {
        goto exit;
    }
    ret = mbedtls_md_hmac_finish(&md_ctx, tmp);
    if (ret != 0) {
        goto exit;
    }

    for (i = 0; i < dlen; i += 20) {
        ret = mbedtls_md_hmac_reset(&md_ctx);
        if (ret != 0) {
            goto exit;
        }
        ret = mbedtls_md_hmac_update(&md_ctx, tmp, 20 + nb);
        if (ret != 0) {
            goto exit;
        }
        ret = mbedtls_md_hmac_finish(&md_ctx, h_i);
        if (ret != 0) {
            goto exit;
        }

        ret = mbedtls_md_hmac_reset(&md_ctx);
        if (ret != 0) {
            goto exit;
        }
        ret = mbedtls_md_hmac_update(&md_ctx, tmp, 20);
        if (ret != 0) {
            goto exit;
        }
        ret = mbedtls_md_hmac_finish(&md_ctx, tmp);
        if (ret != 0) {
            goto exit;
        }

        k = (i + 20 > dlen) ? dlen % 20 : 20;

        for (j = 0; j < k; j++) {
            dstbuf[i + j] = (unsigned char) (dstbuf[i + j] ^ h_i[j]);
        }
    }

exit:
    mbedtls_md_free(&md_ctx);

    mbedtls_platform_zeroize(tmp, tmp_len);
    mbedtls_platform_zeroize(h_i, sizeof(h_i));

    mbedtls_free(tmp);
    return ret;
}
#endif /* MBEDTLS_SSL_PROTO_TLS1) || MBEDTLS_SSL_PROTO_TLS1_1 */

#if defined(MBEDTLS_SSL_PROTO_TLS1_2)
#if defined(MBEDTLS_USE_PSA_CRYPTO)

static psa_status_t setup_psa_key_derivation(psa_key_derivation_operation_t *derivation,
                                             psa_key_id_t key,
                                             psa_algorithm_t alg,
                                             const unsigned char *seed, size_t seed_length,
                                             const unsigned char *label, size_t label_length,
                                             size_t capacity)
{
    psa_status_t status;

    status = psa_key_derivation_setup(derivation, alg);
    if (status != PSA_SUCCESS) {
        return status;
    }

    if (PSA_ALG_IS_TLS12_PRF(alg) || PSA_ALG_IS_TLS12_PSK_TO_MS(alg)) {
        status = psa_key_derivation_input_bytes(derivation,
                                                PSA_KEY_DERIVATION_INPUT_SEED,
                                                seed, seed_length);
        if (status != PSA_SUCCESS) {
            return status;
        }

        if (mbedtls_svc_key_id_is_null(key)) {
            status = psa_key_derivation_input_bytes(
                derivation, PSA_KEY_DERIVATION_INPUT_SECRET,
                NULL, 0);
        } else {
            status = psa_key_derivation_input_key(
                derivation, PSA_KEY_DERIVATION_INPUT_SECRET, key);
        }
        if (status != PSA_SUCCESS) {
            return status;
        }

        status = psa_key_derivation_input_bytes(derivation,
                                                PSA_KEY_DERIVATION_INPUT_LABEL,
                                                label, label_length);
        if (status != PSA_SUCCESS) {
            return status;
        }
    } else {
        return PSA_ERROR_NOT_SUPPORTED;
    }

    status = psa_key_derivation_set_capacity(derivation, capacity);
    if (status != PSA_SUCCESS) {
        return status;
    }

    return PSA_SUCCESS;
}

MBEDTLS_CHECK_RETURN_CRITICAL
static int tls_prf_generic(mbedtls_md_type_t md_type,
                           const unsigned char *secret, size_t slen,
                           const char *label,
                           const unsigned char *random, size_t rlen,
                           unsigned char *dstbuf, size_t dlen)
{
    psa_status_t status;
    psa_algorithm_t alg;
    psa_key_id_t master_key = MBEDTLS_SVC_KEY_ID_INIT;
    psa_key_derivation_operation_t derivation =
        PSA_KEY_DERIVATION_OPERATION_INIT;

    if (md_type == MBEDTLS_MD_SHA384) {
        alg = PSA_ALG_TLS12_PRF(PSA_ALG_SHA_384);
    } else {
        alg = PSA_ALG_TLS12_PRF(PSA_ALG_SHA_256);
    }

    /* Normally a "secret" should be long enough to be impossible to
     * find by brute force, and in particular should not be empty. But
     * this PRF is also used to derive an IV, in particular in EAP-TLS,
     * and for this use case it makes sense to have a 0-length "secret".
     * Since the key API doesn't allow importing a key of length 0,
     * keep master_key=0, which setup_psa_key_derivation() understands
     * to mean a 0-length "secret" input. */
    if (slen != 0) {
        psa_key_attributes_t key_attributes = psa_key_attributes_init();
        psa_set_key_usage_flags(&key_attributes, PSA_KEY_USAGE_DERIVE);
        psa_set_key_algorithm(&key_attributes, alg);
        psa_set_key_type(&key_attributes, PSA_KEY_TYPE_DERIVE);

        status = psa_import_key(&key_attributes, secret, slen, &master_key);
        if (status != PSA_SUCCESS) {
            return MBEDTLS_ERR_SSL_HW_ACCEL_FAILED;
        }
    }

    status = setup_psa_key_derivation(&derivation,
                                      master_key, alg,
                                      random, rlen,
                                      (unsigned char const *) label,
                                      (size_t) strlen(label),
                                      dlen);
    if (status != PSA_SUCCESS) {
        psa_key_derivation_abort(&derivation);
        psa_destroy_key(master_key);
        return MBEDTLS_ERR_SSL_HW_ACCEL_FAILED;
    }

    status = psa_key_derivation_output_bytes(&derivation, dstbuf, dlen);
    if (status != PSA_SUCCESS) {
        psa_key_derivation_abort(&derivation);
        psa_destroy_key(master_key);
        return MBEDTLS_ERR_SSL_HW_ACCEL_FAILED;
    }

    status = psa_key_derivation_abort(&derivation);
    if (status != PSA_SUCCESS) {
        psa_destroy_key(master_key);
        return MBEDTLS_ERR_SSL_HW_ACCEL_FAILED;
    }

    if (!mbedtls_svc_key_id_is_null(master_key)) {
        status = psa_destroy_key(master_key);
    }
    if (status != PSA_SUCCESS) {
        return MBEDTLS_ERR_SSL_HW_ACCEL_FAILED;
    }

    return 0;
}

#else /* MBEDTLS_USE_PSA_CRYPTO */

MBEDTLS_CHECK_RETURN_CRITICAL
static int tls_prf_generic(mbedtls_md_type_t md_type,
                           const unsigned char *secret, size_t slen,
                           const char *label,
                           const unsigned char *random, size_t rlen,
                           unsigned char *dstbuf, size_t dlen)
{
    size_t nb;
    size_t i, j, k, md_len;
    unsigned char *tmp;
    size_t tmp_len = 0;
    unsigned char h_i[MBEDTLS_MD_MAX_SIZE];
    const mbedtls_md_info_t *md_info;
    mbedtls_md_context_t md_ctx;
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;

    mbedtls_md_init(&md_ctx);

    if ((md_info = mbedtls_md_info_from_type(md_type)) == NULL) {
        return MBEDTLS_ERR_SSL_INTERNAL_ERROR;
    }

    md_len = mbedtls_md_get_size(md_info);

    tmp_len = md_len + strlen(label) + rlen;
    tmp = mbedtls_calloc(1, tmp_len);
    if (tmp == NULL) {
        ret = MBEDTLS_ERR_SSL_ALLOC_FAILED;
        goto exit;
    }

    nb = strlen(label);
    memcpy(tmp + md_len, label, nb);
    memcpy(tmp + md_len + nb, random, rlen);
    nb += rlen;

    /*
     * Compute P_<hash>(secret, label + random)[0..dlen]
     */
    if ((ret = mbedtls_md_setup(&md_ctx, md_info, 1)) != 0) {
        goto exit;
    }

    ret = mbedtls_md_hmac_starts(&md_ctx, secret, slen);
    if (ret != 0) {
        goto exit;
    }
    ret = mbedtls_md_hmac_update(&md_ctx, tmp + md_len, nb);
    if (ret != 0) {
        goto exit;
    }
    ret = mbedtls_md_hmac_finish(&md_ctx, tmp);
    if (ret != 0) {
        goto exit;
    }

    for (i = 0; i < dlen; i += md_len) {
        ret = mbedtls_md_hmac_reset(&md_ctx);
        if (ret != 0) {
            goto exit;
        }
        ret = mbedtls_md_hmac_update(&md_ctx, tmp, md_len + nb);
        if (ret != 0) {
            goto exit;
        }
        ret = mbedtls_md_hmac_finish(&md_ctx, h_i);
        if (ret != 0) {
            goto exit;
        }

        ret = mbedtls_md_hmac_reset(&md_ctx);
        if (ret != 0) {
            goto exit;
        }
        ret = mbedtls_md_hmac_update(&md_ctx, tmp, md_len);
        if (ret != 0) {
            goto exit;
        }
        ret = mbedtls_md_hmac_finish(&md_ctx, tmp);
        if (ret != 0) {
            goto exit;
        }

        k = (i + md_len > dlen) ? dlen % md_len : md_len;

        for (j = 0; j < k; j++) {
            dstbuf[i + j]  = h_i[j];
        }
    }

exit:
    mbedtls_md_free(&md_ctx);

    if (tmp != NULL) {
        mbedtls_platform_zeroize(tmp, tmp_len);
    }

    mbedtls_platform_zeroize(h_i, sizeof(h_i));

    mbedtls_free(tmp);

    return ret;
}
#endif /* MBEDTLS_USE_PSA_CRYPTO */
#if defined(MBEDTLS_SHA256_C)
MBEDTLS_CHECK_RETURN_CRITICAL
static int tls_prf_sha256(const unsigned char *secret, size_t slen,
                          const char *label,
                          const unsigned char *random, size_t rlen,
                          unsigned char *dstbuf, size_t dlen)
{
    return tls_prf_generic(MBEDTLS_MD_SHA256, secret, slen,
                           label, random, rlen, dstbuf, dlen);
}
#endif /* MBEDTLS_SHA256_C */

#if defined(MBEDTLS_SHA512_C) && !defined(MBEDTLS_SHA512_NO_SHA384)
MBEDTLS_CHECK_RETURN_CRITICAL
static int tls_prf_sha384(const unsigned char *secret, size_t slen,
                          const char *label,
                          const unsigned char *random, size_t rlen,
                          unsigned char *dstbuf, size_t dlen)
{
    return tls_prf_generic(MBEDTLS_MD_SHA384, secret, slen,
                           label, random, rlen, dstbuf, dlen);
}
#endif /* MBEDTLS_SHA512_C && !MBEDTLS_SHA512_NO_SHA384 */
#endif /* MBEDTLS_SSL_PROTO_TLS1_2 */

static void ssl_update_checksum_start(mbedtls_ssl_context *, const unsigned char *, size_t);

#if defined(MBEDTLS_SSL_PROTO_SSL3) || defined(MBEDTLS_SSL_PROTO_TLS1) || \
    defined(MBEDTLS_SSL_PROTO_TLS1_1)
static void ssl_update_checksum_md5sha1(mbedtls_ssl_context *, const unsigned char *, size_t);
#endif

#if defined(MBEDTLS_SSL_PROTO_SSL3)
static void ssl_calc_verify_ssl(const mbedtls_ssl_context *, unsigned char *, size_t *);
static void ssl_calc_finished_ssl(mbedtls_ssl_context *, unsigned char *, int);
#endif

#if defined(MBEDTLS_SSL_PROTO_TLS1) || defined(MBEDTLS_SSL_PROTO_TLS1_1)
static void ssl_calc_verify_tls(const mbedtls_ssl_context *, unsigned char *, size_t *);
static void ssl_calc_finished_tls(mbedtls_ssl_context *, unsigned char *, int);
#endif

#if defined(MBEDTLS_SSL_PROTO_TLS1_2)
#if defined(MBEDTLS_SHA256_C)
static void ssl_update_checksum_sha256(mbedtls_ssl_context *, const unsigned char *, size_t);
static void ssl_calc_verify_tls_sha256(const mbedtls_ssl_context *, unsigned char *, size_t *);
static void ssl_calc_finished_tls_sha256(mbedtls_ssl_context *, unsigned char *, int);
#endif

#if defined(MBEDTLS_SHA512_C) && !defined(MBEDTLS_SHA512_NO_SHA384)
static void ssl_update_checksum_sha384(mbedtls_ssl_context *, const unsigned char *, size_t);
static void ssl_calc_verify_tls_sha384(const mbedtls_ssl_context *, unsigned char *, size_t *);
static void ssl_calc_finished_tls_sha384(mbedtls_ssl_context *, unsigned char *, int);
#endif
#endif /* MBEDTLS_SSL_PROTO_TLS1_2 */

#if defined(MBEDTLS_KEY_EXCHANGE_PSK_ENABLED) && \
    defined(MBEDTLS_USE_PSA_CRYPTO)
MBEDTLS_CHECK_RETURN_CRITICAL
static int ssl_use_opaque_psk(mbedtls_ssl_context const *ssl)
{
    if (ssl->conf->f_psk != NULL) {
        /* If we've used a callback to select the PSK,
         * the static configuration is irrelevant. */
        if (!mbedtls_svc_key_id_is_null(ssl->handshake->psk_opaque)) {
            return 1;
        }

        return 0;
    }

    if (!mbedtls_svc_key_id_is_null(ssl->conf->psk_opaque)) {
        return 1;
    }

    return 0;
}
#endif /* MBEDTLS_USE_PSA_CRYPTO &&
          MBEDTLS_KEY_EXCHANGE_PSK_ENABLED */

#if defined(MBEDTLS_SSL_EXPORT_KEYS)
static mbedtls_tls_prf_types tls_prf_get_type(mbedtls_ssl_tls_prf_cb *tls_prf)
{
#if defined(MBEDTLS_SSL_PROTO_SSL3)
    if (tls_prf == ssl3_prf) {
        return MBEDTLS_SSL_TLS_PRF_SSL3;
    } else
#endif
#if defined(MBEDTLS_SSL_PROTO_TLS1) || defined(MBEDTLS_SSL_PROTO_TLS1_1)
    if (tls_prf == tls1_prf) {
        return MBEDTLS_SSL_TLS_PRF_TLS1;
    } else
#endif
#if defined(MBEDTLS_SSL_PROTO_TLS1_2)
#if defined(MBEDTLS_SHA512_C) && !defined(MBEDTLS_SHA512_NO_SHA384)
    if (tls_prf == tls_prf_sha384) {
        return MBEDTLS_SSL_TLS_PRF_SHA384;
    } else
#endif
#if defined(MBEDTLS_SHA256_C)
    if (tls_prf == tls_prf_sha256) {
        return MBEDTLS_SSL_TLS_PRF_SHA256;
    } else
#endif
#endif /* MBEDTLS_SSL_PROTO_TLS1_2 */
    return MBEDTLS_SSL_TLS_PRF_NONE;
}
#endif /* MBEDTLS_SSL_EXPORT_KEYS */

int  mbedtls_ssl_tls_prf(const mbedtls_tls_prf_types prf,
                         const unsigned char *secret, size_t slen,
                         const char *label,
                         const unsigned char *random, size_t rlen,
                         unsigned char *dstbuf, size_t dlen)
{
    mbedtls_ssl_tls_prf_cb *tls_prf = NULL;

    switch (prf) {
#if defined(MBEDTLS_SSL_PROTO_SSL3)
        case MBEDTLS_SSL_TLS_PRF_SSL3:
            tls_prf = ssl3_prf;
            break;
#endif /* MBEDTLS_SSL_PROTO_SSL3 */
#if defined(MBEDTLS_SSL_PROTO_TLS1) || defined(MBEDTLS_SSL_PROTO_TLS1_1)
        case MBEDTLS_SSL_TLS_PRF_TLS1:
            tls_prf = tls1_prf;
            break;
#endif /* MBEDTLS_SSL_PROTO_TLS1 || MBEDTLS_SSL_PROTO_TLS1_1 */

#if defined(MBEDTLS_SSL_PROTO_TLS1_2)
#if defined(MBEDTLS_SHA512_C) && !defined(MBEDTLS_SHA512_NO_SHA384)
        case MBEDTLS_SSL_TLS_PRF_SHA384:
            tls_prf = tls_prf_sha384;
            break;
#endif /* MBEDTLS_SHA512_C && !MBEDTLS_SHA512_NO_SHA384 */
#if defined(MBEDTLS_SHA256_C)
        case MBEDTLS_SSL_TLS_PRF_SHA256:
            tls_prf = tls_prf_sha256;
            break;
#endif /* MBEDTLS_SHA256_C */
#endif /* MBEDTLS_SSL_PROTO_TLS1_2 */
        default:
            return MBEDTLS_ERR_SSL_FEATURE_UNAVAILABLE;
    }

    return tls_prf(secret, slen, label, random, rlen, dstbuf, dlen);
}

/* Type for the TLS PRF */
typedef int ssl_tls_prf_t(const unsigned char *, size_t, const char *,
                          const unsigned char *, size_t,
                          unsigned char *, size_t);

/*
 * Populate a transform structure with session keys and all the other
 * necessary information.
 *
 * Parameters:
 * - [in/out]: transform: structure to populate
 *      [in] must be just initialised with mbedtls_ssl_transform_init()
 *      [out] fully populated, ready for use by mbedtls_ssl_{en,de}crypt_buf()
 * - [in] ciphersuite
 * - [in] master
 * - [in] encrypt_then_mac
 * - [in] trunc_hmac
 * - [in] compression
 * - [in] tls_prf: pointer to PRF to use for key derivation
 * - [in] randbytes: buffer holding ServerHello.random + ClientHello.random
 * - [in] minor_ver: SSL/TLS minor version
 * - [in] endpoint: client or server
 * - [in] ssl: optionally used for:
 *        - MBEDTLS_SSL_HW_RECORD_ACCEL: whole context (non-const)
 *        - MBEDTLS_SSL_EXPORT_KEYS: ssl->conf->{f,p}_export_keys
 *        - MBEDTLS_DEBUG_C: ssl->conf->{f,p}_dbg
 */
MBEDTLS_CHECK_RETURN_CRITICAL
static int ssl_populate_transform(mbedtls_ssl_transform *transform,
                                  int ciphersuite,
                                  const unsigned char master[48],
#if defined(MBEDTLS_SSL_SOME_MODES_USE_MAC)
#if defined(MBEDTLS_SSL_ENCRYPT_THEN_MAC)
                                  int encrypt_then_mac,
#endif /* MBEDTLS_SSL_ENCRYPT_THEN_MAC */
#if defined(MBEDTLS_SSL_TRUNCATED_HMAC)
                                  int trunc_hmac,
#endif /* MBEDTLS_SSL_TRUNCATED_HMAC */
#endif /* MBEDTLS_SSL_SOME_MODES_USE_MAC */
#if defined(MBEDTLS_ZLIB_SUPPORT)
                                  int compression,
#endif
                                  ssl_tls_prf_t tls_prf,
                                  const unsigned char randbytes[64],
                                  int minor_ver,
                                  unsigned endpoint,
#if !defined(MBEDTLS_SSL_HW_RECORD_ACCEL)
                                  const
#endif
                                  mbedtls_ssl_context *ssl)
{
    int ret = 0;
#if defined(MBEDTLS_USE_PSA_CRYPTO)
    int psa_fallthrough;
#endif /* MBEDTLS_USE_PSA_CRYPTO */
    int do_mbedtls_cipher_setup;
    unsigned char keyblk[256];
    unsigned char *key1;
    unsigned char *key2;
    unsigned char *mac_enc;
    unsigned char *mac_dec;
    size_t mac_key_len = 0;
    size_t iv_copy_len;
    unsigned keylen;
    const mbedtls_ssl_ciphersuite_t *ciphersuite_info;
    const mbedtls_cipher_info_t *cipher_info;
    const mbedtls_md_info_t *md_info;

#if !defined(MBEDTLS_SSL_HW_RECORD_ACCEL) && \
    !defined(MBEDTLS_SSL_EXPORT_KEYS) && \
    !defined(MBEDTLS_SSL_DTLS_CONNECTION_ID) && \
    !defined(MBEDTLS_DEBUG_C)
    ssl = NULL; /* make sure we don't use it except for those cases */
    (void) ssl;
#endif

    /*
     * Some data just needs copying into the structure
     */
#if defined(MBEDTLS_SSL_ENCRYPT_THEN_MAC) && \
    defined(MBEDTLS_SSL_SOME_MODES_USE_MAC)
    transform->encrypt_then_mac = encrypt_then_mac;
#endif
    transform->minor_ver = minor_ver;

#if defined(MBEDTLS_SSL_CONTEXT_SERIALIZATION)
    memcpy(transform->randbytes, randbytes, sizeof(transform->randbytes));
#endif

    /*
     * Get various info structures
     */
    ciphersuite_info = mbedtls_ssl_ciphersuite_from_id(ciphersuite);
    if (ciphersuite_info == NULL) {
        MBEDTLS_SSL_DEBUG_MSG(1, ("ciphersuite info for %d not found",
                                  ciphersuite));
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    cipher_info = mbedtls_cipher_info_from_type(ciphersuite_info->cipher);
    if (cipher_info == NULL) {
        MBEDTLS_SSL_DEBUG_MSG(1, ("cipher info for %u not found",
                                  ciphersuite_info->cipher));
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    md_info = mbedtls_md_info_from_type(ciphersuite_info->mac);
    if (md_info == NULL) {
        MBEDTLS_SSL_DEBUG_MSG(1, ("mbedtls_md info for %u not found",
                                  (unsigned) ciphersuite_info->mac));
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

#if defined(MBEDTLS_SSL_DTLS_CONNECTION_ID)
    /* Copy own and peer's CID if the use of the CID
     * extension has been negotiated. */
    if (ssl->handshake->cid_in_use == MBEDTLS_SSL_CID_ENABLED) {
        MBEDTLS_SSL_DEBUG_MSG(3, ("Copy CIDs into SSL transform"));

        transform->in_cid_len = ssl->own_cid_len;
        memcpy(transform->in_cid, ssl->own_cid, ssl->own_cid_len);
        MBEDTLS_SSL_DEBUG_BUF(3, "Incoming CID", transform->in_cid,
                              transform->in_cid_len);

        transform->out_cid_len = ssl->handshake->peer_cid_len;
        memcpy(transform->out_cid, ssl->handshake->peer_cid,
               ssl->handshake->peer_cid_len);
        MBEDTLS_SSL_DEBUG_BUF(3, "Outgoing CID", transform->out_cid,
                              transform->out_cid_len);
    }
#endif /* MBEDTLS_SSL_DTLS_CONNECTION_ID */

    /*
     * Compute key block using the PRF
     */
    ret = tls_prf(master, 48, "key expansion", randbytes, 64, keyblk, 256);
    if (ret != 0) {
        MBEDTLS_SSL_DEBUG_RET(1, "prf", ret);
        return ret;
    }

    MBEDTLS_SSL_DEBUG_MSG(3, ("ciphersuite = %s",
                              mbedtls_ssl_get_ciphersuite_name(ciphersuite)));
    MBEDTLS_SSL_DEBUG_BUF(3, "master secret", master, 48);
    MBEDTLS_SSL_DEBUG_BUF(4, "random bytes", randbytes, 64);
    MBEDTLS_SSL_DEBUG_BUF(4, "key block", keyblk, 256);

    /*
     * Determine the appropriate key, IV and MAC length.
     */

    keylen = cipher_info->key_bitlen / 8;

#if defined(MBEDTLS_GCM_C) ||                           \
    defined(MBEDTLS_CCM_C) ||                           \
    defined(MBEDTLS_CHACHAPOLY_C)
    if (cipher_info->mode == MBEDTLS_MODE_GCM ||
        cipher_info->mode == MBEDTLS_MODE_CCM ||
        cipher_info->mode == MBEDTLS_MODE_CHACHAPOLY) {
        size_t explicit_ivlen;

        transform->maclen = 0;
        mac_key_len = 0;
        transform->taglen =
            ciphersuite_info->flags & MBEDTLS_CIPHERSUITE_SHORT_TAG ? 8 : 16;

        /* All modes haves 96-bit IVs, but the length of the static parts vary
         * with mode and version:
         * - For GCM and CCM in TLS 1.2, there's a static IV of 4 Bytes
         *   (to be concatenated with a dynamically chosen IV of 8 Bytes)
         * - For ChaChaPoly in TLS 1.2, and all modes in TLS 1.3, there's
         *   a static IV of 12 Bytes (to be XOR'ed with the 8 Byte record
         *   sequence number).
         */
        transform->ivlen = 12;
#if defined(MBEDTLS_SSL_PROTO_TLS1_3_EXPERIMENTAL)
        if (minor_ver == MBEDTLS_SSL_MINOR_VERSION_4) {
            transform->fixed_ivlen = 12;
        } else
#endif /* MBEDTLS_SSL_PROTO_TLS1_3_EXPERIMENTAL */
        {
            if (cipher_info->mode == MBEDTLS_MODE_CHACHAPOLY) {
                transform->fixed_ivlen = 12;
            } else {
                transform->fixed_ivlen = 4;
            }
        }

        /* Minimum length of encrypted record */
        explicit_ivlen = transform->ivlen - transform->fixed_ivlen;
        transform->minlen = explicit_ivlen + transform->taglen;
    } else
#endif /* MBEDTLS_GCM_C || MBEDTLS_CCM_C || MBEDTLS_CHACHAPOLY_C */
#if defined(MBEDTLS_SSL_SOME_MODES_USE_MAC)
    if (cipher_info->mode == MBEDTLS_MODE_STREAM ||
        cipher_info->mode == MBEDTLS_MODE_CBC) {
        /* Initialize HMAC contexts */
        if ((ret = mbedtls_md_setup(&transform->md_ctx_enc, md_info, 1)) != 0 ||
            (ret = mbedtls_md_setup(&transform->md_ctx_dec, md_info, 1)) != 0) {
            MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_md_setup", ret);
            goto end;
        }

        /* Get MAC length */
        mac_key_len = mbedtls_md_get_size(md_info);
        transform->maclen = mac_key_len;

#if defined(MBEDTLS_SSL_TRUNCATED_HMAC)
        /*
         * If HMAC is to be truncated, we shall keep the leftmost bytes,
         * (rfc 6066 page 13 or rfc 2104 section 4),
         * so we only need to adjust the length here.
         */
        if (trunc_hmac == MBEDTLS_SSL_TRUNC_HMAC_ENABLED) {
            transform->maclen = MBEDTLS_SSL_TRUNCATED_HMAC_LEN;

#if defined(MBEDTLS_SSL_TRUNCATED_HMAC_COMPAT)
            /* Fall back to old, non-compliant version of the truncated
             * HMAC implementation which also truncates the key
             * (Mbed TLS versions from 1.3 to 2.6.0) */
            mac_key_len = transform->maclen;
#endif
        }
#endif /* MBEDTLS_SSL_TRUNCATED_HMAC */

        /* IV length */
        transform->ivlen = cipher_info->iv_size;

        /* Minimum length */
        if (cipher_info->mode == MBEDTLS_MODE_STREAM) {
            transform->minlen = transform->maclen;
        } else {
            /*
             * GenericBlockCipher:
             * 1. if EtM is in use: one block plus MAC
             *    otherwise: * first multiple of blocklen greater than maclen
             * 2. IV except for SSL3 and TLS 1.0
             */
#if defined(MBEDTLS_SSL_ENCRYPT_THEN_MAC)
            if (encrypt_then_mac == MBEDTLS_SSL_ETM_ENABLED) {
                transform->minlen = transform->maclen
                                    + cipher_info->block_size;
            } else
#endif
            {
                transform->minlen = transform->maclen
                                    + cipher_info->block_size
                                    - transform->maclen % cipher_info->block_size;
            }

#if defined(MBEDTLS_SSL_PROTO_SSL3) || defined(MBEDTLS_SSL_PROTO_TLS1)
            if (minor_ver == MBEDTLS_SSL_MINOR_VERSION_0 ||
                minor_ver == MBEDTLS_SSL_MINOR_VERSION_1) {
                ; /* No need to adjust minlen */
            } else
#endif
#if defined(MBEDTLS_SSL_PROTO_TLS1_1) || defined(MBEDTLS_SSL_PROTO_TLS1_2)
            if (minor_ver == MBEDTLS_SSL_MINOR_VERSION_2 ||
                minor_ver == MBEDTLS_SSL_MINOR_VERSION_3) {
                transform->minlen += transform->ivlen;
            } else
#endif
            {
                MBEDTLS_SSL_DEBUG_MSG(1, ("should never happen"));
                ret = MBEDTLS_ERR_SSL_INTERNAL_ERROR;
                goto end;
            }
        }
    } else
#endif /* MBEDTLS_SSL_SOME_MODES_USE_MAC */
    {
        MBEDTLS_SSL_DEBUG_MSG(1, ("should never happen"));
        return MBEDTLS_ERR_SSL_INTERNAL_ERROR;
    }

    MBEDTLS_SSL_DEBUG_MSG(3, ("keylen: %u, minlen: %u, ivlen: %u, maclen: %u",
                              (unsigned) keylen,
                              (unsigned) transform->minlen,
                              (unsigned) transform->ivlen,
                              (unsigned) transform->maclen));

    /*
     * Finally setup the cipher contexts, IVs and MAC secrets.
     */
#if defined(MBEDTLS_SSL_CLI_C)
    if (endpoint == MBEDTLS_SSL_IS_CLIENT) {
        key1 = keyblk + mac_key_len * 2;
        key2 = keyblk + mac_key_len * 2 + keylen;

        mac_enc = keyblk;
        mac_dec = keyblk + mac_key_len;

        /*
         * This is not used in TLS v1.1.
         */
        iv_copy_len = (transform->fixed_ivlen) ?
                      transform->fixed_ivlen : transform->ivlen;
        memcpy(transform->iv_enc, key2 + keylen,  iv_copy_len);
        memcpy(transform->iv_dec, key2 + keylen + iv_copy_len,
               iv_copy_len);
    } else
#endif /* MBEDTLS_SSL_CLI_C */
#if defined(MBEDTLS_SSL_SRV_C)
    if (endpoint == MBEDTLS_SSL_IS_SERVER) {
        key1 = keyblk + mac_key_len * 2 + keylen;
        key2 = keyblk + mac_key_len * 2;

        mac_enc = keyblk + mac_key_len;
        mac_dec = keyblk;

        /*
         * This is not used in TLS v1.1.
         */
        iv_copy_len = (transform->fixed_ivlen) ?
                      transform->fixed_ivlen : transform->ivlen;
        memcpy(transform->iv_dec, key1 + keylen,  iv_copy_len);
        memcpy(transform->iv_enc, key1 + keylen + iv_copy_len,
               iv_copy_len);
    } else
#endif /* MBEDTLS_SSL_SRV_C */
    {
        MBEDTLS_SSL_DEBUG_MSG(1, ("should never happen"));
        ret = MBEDTLS_ERR_SSL_INTERNAL_ERROR;
        goto end;
    }

#if defined(MBEDTLS_SSL_SOME_MODES_USE_MAC)
#if defined(MBEDTLS_SSL_PROTO_SSL3)
    if (minor_ver == MBEDTLS_SSL_MINOR_VERSION_0) {
        if (mac_key_len > sizeof(transform->mac_enc)) {
            MBEDTLS_SSL_DEBUG_MSG(1, ("should never happen"));
            ret = MBEDTLS_ERR_SSL_INTERNAL_ERROR;
            goto end;
        }

        memcpy(transform->mac_enc, mac_enc, mac_key_len);
        memcpy(transform->mac_dec, mac_dec, mac_key_len);
    } else
#endif /* MBEDTLS_SSL_PROTO_SSL3 */
#if defined(MBEDTLS_SSL_PROTO_TLS1) || defined(MBEDTLS_SSL_PROTO_TLS1_1) || \
    defined(MBEDTLS_SSL_PROTO_TLS1_2)
    if (minor_ver >= MBEDTLS_SSL_MINOR_VERSION_1) {
        /* For HMAC-based ciphersuites, initialize the HMAC transforms.
           For AEAD-based ciphersuites, there is nothing to do here. */
        if (mac_key_len != 0) {
            ret = mbedtls_md_hmac_starts(&transform->md_ctx_enc,
                                         mac_enc, mac_key_len);
            if (ret != 0) {
                goto end;
            }
            ret = mbedtls_md_hmac_starts(&transform->md_ctx_dec,
                                         mac_dec, mac_key_len);
            if (ret != 0) {
                goto end;
            }
        }
    } else
#endif
    {
        MBEDTLS_SSL_DEBUG_MSG(1, ("should never happen"));
        ret = MBEDTLS_ERR_SSL_INTERNAL_ERROR;
        goto end;
    }
#endif /* MBEDTLS_SSL_SOME_MODES_USE_MAC */

#if defined(MBEDTLS_SSL_HW_RECORD_ACCEL)
    if (mbedtls_ssl_hw_record_init != NULL) {
        ret = 0;

        MBEDTLS_SSL_DEBUG_MSG(2, ("going for mbedtls_ssl_hw_record_init()"));

        if ((ret = mbedtls_ssl_hw_record_init(ssl, key1, key2, keylen,
                                              transform->iv_enc, transform->iv_dec,
                                              iv_copy_len,
                                              mac_enc, mac_dec,
                                              mac_key_len)) != 0) {
            MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_ssl_hw_record_init", ret);
            ret = MBEDTLS_ERR_SSL_HW_ACCEL_FAILED;
            goto end;
        }
    }
#else
    ((void) mac_dec);
    ((void) mac_enc);
#endif /* MBEDTLS_SSL_HW_RECORD_ACCEL */

#if defined(MBEDTLS_SSL_EXPORT_KEYS)
    if (ssl->conf->f_export_keys != NULL) {
        ssl->conf->f_export_keys(ssl->conf->p_export_keys,
                                 master, keyblk,
                                 mac_key_len, keylen,
                                 iv_copy_len);
    }

    if (ssl->conf->f_export_keys_ext != NULL) {
        ssl->conf->f_export_keys_ext(ssl->conf->p_export_keys,
                                     master, keyblk,
                                     mac_key_len, keylen,
                                     iv_copy_len,
                                     randbytes + 32,
                                     randbytes,
                                     tls_prf_get_type(tls_prf));
    }
#endif

    do_mbedtls_cipher_setup = 1;
#if defined(MBEDTLS_USE_PSA_CRYPTO)

    /* Only use PSA-based ciphers for TLS-1.2.
     * That's relevant at least for TLS-1.0, where
     * we assume that mbedtls_cipher_crypt() updates
     * the structure field for the IV, which the PSA-based
     * implementation currently doesn't. */
#if defined(MBEDTLS_SSL_PROTO_TLS1_2)
    if (minor_ver == MBEDTLS_SSL_MINOR_VERSION_3) {
        ret = mbedtls_cipher_setup_psa(&transform->cipher_ctx_enc,
                                       cipher_info, transform->taglen);
        if (ret != 0 && ret != MBEDTLS_ERR_CIPHER_FEATURE_UNAVAILABLE) {
            MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_cipher_setup_psa", ret);
            goto end;
        }

        if (ret == 0) {
            MBEDTLS_SSL_DEBUG_MSG(3, ("Successfully setup PSA-based encryption cipher context"));
            psa_fallthrough = 0;
        } else {
            MBEDTLS_SSL_DEBUG_MSG(1,
                                  (
                                      "Failed to setup PSA-based cipher context for record encryption - fall through to default setup."));
            psa_fallthrough = 1;
        }
    } else {
        psa_fallthrough = 1;
    }
#else
    psa_fallthrough = 1;
#endif /* MBEDTLS_SSL_PROTO_TLS1_2 */

    if (psa_fallthrough == 0) {
        do_mbedtls_cipher_setup = 0;
    }
#endif /* MBEDTLS_USE_PSA_CRYPTO */
    if (do_mbedtls_cipher_setup &&
        (ret = mbedtls_cipher_setup(&transform->cipher_ctx_enc,
                                    cipher_info)) != 0) {
        MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_cipher_setup", ret);
        goto end;
    }

    do_mbedtls_cipher_setup = 1;
#if defined(MBEDTLS_USE_PSA_CRYPTO)
    /* Only use PSA-based ciphers for TLS-1.2.
     * That's relevant at least for TLS-1.0, where
     * we assume that mbedtls_cipher_crypt() updates
     * the structure field for the IV, which the PSA-based
     * implementation currently doesn't. */
#if defined(MBEDTLS_SSL_PROTO_TLS1_2)
    if (minor_ver == MBEDTLS_SSL_MINOR_VERSION_3) {
        ret = mbedtls_cipher_setup_psa(&transform->cipher_ctx_dec,
                                       cipher_info, transform->taglen);
        if (ret != 0 && ret != MBEDTLS_ERR_CIPHER_FEATURE_UNAVAILABLE) {
            MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_cipher_setup_psa", ret);
            goto end;
        }

        if (ret == 0) {
            MBEDTLS_SSL_DEBUG_MSG(3, ("Successfully setup PSA-based decryption cipher context"));
            psa_fallthrough = 0;
        } else {
            MBEDTLS_SSL_DEBUG_MSG(1,
                                  (
                                      "Failed to setup PSA-based cipher context for record decryption - fall through to default setup."));
            psa_fallthrough = 1;
        }
    } else {
        psa_fallthrough = 1;
    }
#else
    psa_fallthrough = 1;
#endif /* MBEDTLS_SSL_PROTO_TLS1_2 */

    if (psa_fallthrough == 0) {
        do_mbedtls_cipher_setup = 0;
    }
#endif /* MBEDTLS_USE_PSA_CRYPTO */
    if (do_mbedtls_cipher_setup &&
        (ret = mbedtls_cipher_setup(&transform->cipher_ctx_dec,
                                    cipher_info)) != 0) {
        MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_cipher_setup", ret);
        goto end;
    }

    if ((ret = mbedtls_cipher_setkey(&transform->cipher_ctx_enc, key1,
                                     cipher_info->key_bitlen,
                                     MBEDTLS_ENCRYPT)) != 0) {
        MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_cipher_setkey", ret);
        goto end;
    }

    if ((ret = mbedtls_cipher_setkey(&transform->cipher_ctx_dec, key2,
                                     cipher_info->key_bitlen,
                                     MBEDTLS_DECRYPT)) != 0) {
        MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_cipher_setkey", ret);
        goto end;
    }

#if defined(MBEDTLS_CIPHER_MODE_CBC)
    if (cipher_info->mode == MBEDTLS_MODE_CBC) {
        if ((ret = mbedtls_cipher_set_padding_mode(&transform->cipher_ctx_enc,
                                                   MBEDTLS_PADDING_NONE)) != 0) {
            MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_cipher_set_padding_mode", ret);
            goto end;
        }

        if ((ret = mbedtls_cipher_set_padding_mode(&transform->cipher_ctx_dec,
                                                   MBEDTLS_PADDING_NONE)) != 0) {
            MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_cipher_set_padding_mode", ret);
            goto end;
        }
    }
#endif /* MBEDTLS_CIPHER_MODE_CBC */


    /* Initialize Zlib contexts */
#if defined(MBEDTLS_ZLIB_SUPPORT)
    if (compression == MBEDTLS_SSL_COMPRESS_DEFLATE) {
        MBEDTLS_SSL_DEBUG_MSG(3, ("Initializing zlib states"));

        memset(&transform->ctx_deflate, 0, sizeof(transform->ctx_deflate));
        memset(&transform->ctx_inflate, 0, sizeof(transform->ctx_inflate));

        if (deflateInit(&transform->ctx_deflate,
                        Z_DEFAULT_COMPRESSION)   != Z_OK ||
            inflateInit(&transform->ctx_inflate) != Z_OK) {
            MBEDTLS_SSL_DEBUG_MSG(1, ("Failed to initialize compression"));
            ret = MBEDTLS_ERR_SSL_COMPRESSION_FAILED;
            goto end;
        }
    }
#endif /* MBEDTLS_ZLIB_SUPPORT */

end:
    mbedtls_platform_zeroize(keyblk, sizeof(keyblk));
    return ret;
}

/*
 * Set appropriate PRF function and other SSL / TLS 1.0/1.1 / TLS1.2 functions
 *
 * Inputs:
 * - SSL/TLS minor version
 * - hash associated with the ciphersuite (only used by TLS 1.2)
 *
 * Outputs:
 * - the tls_prf, calc_verify and calc_finished members of handshake structure
 */
MBEDTLS_CHECK_RETURN_CRITICAL
static int ssl_set_handshake_prfs(mbedtls_ssl_handshake_params *handshake,
                                  int minor_ver,
                                  mbedtls_md_type_t hash)
{
#if !defined(MBEDTLS_SSL_PROTO_TLS1_2) ||       \
    !(defined(MBEDTLS_SHA512_C) && !defined(MBEDTLS_SHA512_NO_SHA384))
    (void) hash;
#endif

#if defined(MBEDTLS_SSL_PROTO_SSL3)
    if (minor_ver == MBEDTLS_SSL_MINOR_VERSION_0) {
        handshake->tls_prf = ssl3_prf;
        handshake->calc_verify = ssl_calc_verify_ssl;
        handshake->calc_finished = ssl_calc_finished_ssl;
    } else
#endif
#if defined(MBEDTLS_SSL_PROTO_TLS1) || defined(MBEDTLS_SSL_PROTO_TLS1_1)
    if (minor_ver < MBEDTLS_SSL_MINOR_VERSION_3) {
        handshake->tls_prf = tls1_prf;
        handshake->calc_verify = ssl_calc_verify_tls;
        handshake->calc_finished = ssl_calc_finished_tls;
    } else
#endif
#if defined(MBEDTLS_SSL_PROTO_TLS1_2)
#if defined(MBEDTLS_SHA512_C) && !defined(MBEDTLS_SHA512_NO_SHA384)
    if (minor_ver == MBEDTLS_SSL_MINOR_VERSION_3 &&
        hash == MBEDTLS_MD_SHA384) {
        handshake->tls_prf = tls_prf_sha384;
        handshake->calc_verify = ssl_calc_verify_tls_sha384;
        handshake->calc_finished = ssl_calc_finished_tls_sha384;
    } else
#endif
#if defined(MBEDTLS_SHA256_C)
    if (minor_ver == MBEDTLS_SSL_MINOR_VERSION_3) {
        handshake->tls_prf = tls_prf_sha256;
        handshake->calc_verify = ssl_calc_verify_tls_sha256;
        handshake->calc_finished = ssl_calc_finished_tls_sha256;
    } else
#endif
#endif /* MBEDTLS_SSL_PROTO_TLS1_2 */
    {
        return MBEDTLS_ERR_SSL_INTERNAL_ERROR;
    }

    return 0;
}

/*
 * Compute master secret if needed
 *
 * Parameters:
 * [in/out] handshake
 *          [in] resume, premaster, extended_ms, calc_verify, tls_prf
 *               (PSA-PSK) ciphersuite_info, psk_opaque
 *          [out] premaster (cleared)
 * [out] master
 * [in] ssl: optionally used for debugging, EMS and PSA-PSK
 *      debug: conf->f_dbg, conf->p_dbg
 *      EMS: passed to calc_verify (debug + (SSL3) session_negotiate)
 *      PSA-PSA: minor_ver, conf
 */
MBEDTLS_CHECK_RETURN_CRITICAL
static int ssl_compute_master(mbedtls_ssl_handshake_params *handshake,
                              unsigned char *master,
                              const mbedtls_ssl_context *ssl)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;

    /* cf. RFC 5246, Section 8.1:
     * "The master secret is always exactly 48 bytes in length." */
    size_t const master_secret_len = 48;

#if defined(MBEDTLS_SSL_EXTENDED_MASTER_SECRET)
    unsigned char session_hash[48];
#endif /* MBEDTLS_SSL_EXTENDED_MASTER_SECRET */

    /* The label for the KDF used for key expansion.
     * This is either "master secret" or "extended master secret"
     * depending on whether the Extended Master Secret extension
     * is used. */
    char const *lbl = "master secret";

    /* The salt for the KDF used for key expansion.
     * - If the Extended Master Secret extension is not used,
     *   this is ClientHello.Random + ServerHello.Random
     *   (see Sect. 8.1 in RFC 5246).
     * - If the Extended Master Secret extension is used,
     *   this is the transcript of the handshake so far.
     *   (see Sect. 4 in RFC 7627). */
    unsigned char const *salt = handshake->randbytes;
    size_t salt_len = 64;

#if !defined(MBEDTLS_DEBUG_C) &&                    \
    !defined(MBEDTLS_SSL_EXTENDED_MASTER_SECRET) && \
    !(defined(MBEDTLS_USE_PSA_CRYPTO) &&            \
    defined(MBEDTLS_KEY_EXCHANGE_PSK_ENABLED))
    ssl = NULL; /* make sure we don't use it except for those cases */
    (void) ssl;
#endif

    if (handshake->resume != 0) {
        MBEDTLS_SSL_DEBUG_MSG(3, ("no premaster (session resumed)"));
        return 0;
    }

#if defined(MBEDTLS_SSL_EXTENDED_MASTER_SECRET)
    if (handshake->extended_ms == MBEDTLS_SSL_EXTENDED_MS_ENABLED) {
        lbl  = "extended master secret";
        salt = session_hash;
        handshake->calc_verify(ssl, session_hash, &salt_len);

        MBEDTLS_SSL_DEBUG_BUF(3, "session hash for extended master secret",
                              session_hash, salt_len);
    }
#endif /* MBEDTLS_SSL_EXTENDED_MS_ENABLED */

#if defined(MBEDTLS_USE_PSA_CRYPTO) &&          \
    defined(MBEDTLS_KEY_EXCHANGE_PSK_ENABLED)
    if (handshake->ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_PSK &&
        ssl->minor_ver == MBEDTLS_SSL_MINOR_VERSION_3 &&
        ssl_use_opaque_psk(ssl) == 1) {
        /* Perform PSK-to-MS expansion in a single step. */
        psa_status_t status;
        psa_algorithm_t alg;
        psa_key_id_t psk;
        psa_key_derivation_operation_t derivation =
            PSA_KEY_DERIVATION_OPERATION_INIT;
        mbedtls_md_type_t hash_alg = handshake->ciphersuite_info->mac;

        MBEDTLS_SSL_DEBUG_MSG(2, ("perform PSA-based PSK-to-MS expansion"));

        psk = mbedtls_ssl_get_opaque_psk(ssl);

        if (hash_alg == MBEDTLS_MD_SHA384) {
            alg = PSA_ALG_TLS12_PSK_TO_MS(PSA_ALG_SHA_384);
        } else {
            alg = PSA_ALG_TLS12_PSK_TO_MS(PSA_ALG_SHA_256);
        }

        status = setup_psa_key_derivation(&derivation, psk, alg,
                                          salt, salt_len,
                                          (unsigned char const *) lbl,
                                          (size_t) strlen(lbl),
                                          master_secret_len);
        if (status != PSA_SUCCESS) {
            psa_key_derivation_abort(&derivation);
            return MBEDTLS_ERR_SSL_HW_ACCEL_FAILED;
        }

        status = psa_key_derivation_output_bytes(&derivation,
                                                 master,
                                                 master_secret_len);
        if (status != PSA_SUCCESS) {
            psa_key_derivation_abort(&derivation);
            return MBEDTLS_ERR_SSL_HW_ACCEL_FAILED;
        }

        status = psa_key_derivation_abort(&derivation);
        if (status != PSA_SUCCESS) {
            return MBEDTLS_ERR_SSL_HW_ACCEL_FAILED;
        }
    } else
#endif
    {
        ret = handshake->tls_prf(handshake->premaster, handshake->pmslen,
                                 lbl, salt, salt_len,
                                 master,
                                 master_secret_len);
        if (ret != 0) {
            MBEDTLS_SSL_DEBUG_RET(1, "prf", ret);
            return ret;
        }

        MBEDTLS_SSL_DEBUG_BUF(3, "premaster secret",
                              handshake->premaster,
                              handshake->pmslen);

        mbedtls_platform_zeroize(handshake->premaster,
                                 sizeof(handshake->premaster));
    }

    return 0;
}

int mbedtls_ssl_derive_keys(mbedtls_ssl_context *ssl)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    const mbedtls_ssl_ciphersuite_t * const ciphersuite_info =
        ssl->handshake->ciphersuite_info;

    MBEDTLS_SSL_DEBUG_MSG(2, ("=> derive keys"));

    /* Set PRF, calc_verify and calc_finished function pointers */
    ret = ssl_set_handshake_prfs(ssl->handshake,
                                 ssl->minor_ver,
                                 ciphersuite_info->mac);
    if (ret != 0) {
        MBEDTLS_SSL_DEBUG_RET(1, "ssl_set_handshake_prfs", ret);
        return ret;
    }

    /* Compute master secret if needed */
    ret = ssl_compute_master(ssl->handshake,
                             ssl->session_negotiate->master,
                             ssl);
    if (ret != 0) {
        MBEDTLS_SSL_DEBUG_RET(1, "ssl_compute_master", ret);
        return ret;
    }

    /* Swap the client and server random values:
     * - MS derivation wanted client+server (RFC 5246 8.1)
     * - key derivation wants server+client (RFC 5246 6.3) */
    {
        unsigned char tmp[64];
        memcpy(tmp, ssl->handshake->randbytes, 64);
        memcpy(ssl->handshake->randbytes, tmp + 32, 32);
        memcpy(ssl->handshake->randbytes + 32, tmp, 32);
        mbedtls_platform_zeroize(tmp, sizeof(tmp));
    }

    /* Populate transform structure */
    ret = ssl_populate_transform(ssl->transform_negotiate,
                                 ssl->session_negotiate->ciphersuite,
                                 ssl->session_negotiate->master,
#if defined(MBEDTLS_SSL_SOME_MODES_USE_MAC)
#if defined(MBEDTLS_SSL_ENCRYPT_THEN_MAC)
                                 ssl->session_negotiate->encrypt_then_mac,
#endif /* MBEDTLS_SSL_ENCRYPT_THEN_MAC */
#if defined(MBEDTLS_SSL_TRUNCATED_HMAC)
                                 ssl->session_negotiate->trunc_hmac,
#endif /* MBEDTLS_SSL_TRUNCATED_HMAC */
#endif /* MBEDTLS_SSL_SOME_MODES_USE_MAC */
#if defined(MBEDTLS_ZLIB_SUPPORT)
                                 ssl->session_negotiate->compression,
#endif
                                 ssl->handshake->tls_prf,
                                 ssl->handshake->randbytes,
                                 ssl->minor_ver,
                                 ssl->conf->endpoint,
                                 ssl);
    if (ret != 0) {
        MBEDTLS_SSL_DEBUG_RET(1, "ssl_populate_transform", ret);
        return ret;
    }

    /* We no longer need Server/ClientHello.random values */
    mbedtls_platform_zeroize(ssl->handshake->randbytes,
                             sizeof(ssl->handshake->randbytes));

    /* Allocate compression buffer */
#if defined(MBEDTLS_ZLIB_SUPPORT)
    if (ssl->session_negotiate->compression == MBEDTLS_SSL_COMPRESS_DEFLATE &&
        ssl->compress_buf == NULL) {
        MBEDTLS_SSL_DEBUG_MSG(3, ("Allocating compression buffer"));
        ssl->compress_buf = mbedtls_calloc(1, MBEDTLS_SSL_COMPRESS_BUFFER_LEN);
        if (ssl->compress_buf == NULL) {
            MBEDTLS_SSL_DEBUG_MSG(1, ("alloc(%d bytes) failed",
                                      MBEDTLS_SSL_COMPRESS_BUFFER_LEN));
            return MBEDTLS_ERR_SSL_ALLOC_FAILED;
        }
    }
#endif

    MBEDTLS_SSL_DEBUG_MSG(2, ("<= derive keys"));

    return 0;
}

#if defined(MBEDTLS_SSL_PROTO_SSL3)
void ssl_calc_verify_ssl(const mbedtls_ssl_context *ssl,
                         unsigned char *hash,
                         size_t *hlen)
{
    mbedtls_md5_context md5;
    mbedtls_sha1_context sha1;
    unsigned char pad_1[48];
    unsigned char pad_2[48];

    MBEDTLS_SSL_DEBUG_MSG(2, ("=> calc verify ssl"));

    mbedtls_md5_init(&md5);
    mbedtls_sha1_init(&sha1);

    mbedtls_md5_clone(&md5, &ssl->handshake->fin_md5);
    mbedtls_sha1_clone(&sha1, &ssl->handshake->fin_sha1);

    memset(pad_1, 0x36, 48);
    memset(pad_2, 0x5C, 48);

    mbedtls_md5_update_ret(&md5, ssl->session_negotiate->master, 48);
    mbedtls_md5_update_ret(&md5, pad_1, 48);
    mbedtls_md5_finish_ret(&md5, hash);

    mbedtls_md5_starts_ret(&md5);
    mbedtls_md5_update_ret(&md5, ssl->session_negotiate->master, 48);
    mbedtls_md5_update_ret(&md5, pad_2, 48);
    mbedtls_md5_update_ret(&md5, hash,  16);
    mbedtls_md5_finish_ret(&md5, hash);

    mbedtls_sha1_update_ret(&sha1, ssl->session_negotiate->master, 48);
    mbedtls_sha1_update_ret(&sha1, pad_1, 40);
    mbedtls_sha1_finish_ret(&sha1, hash + 16);

    mbedtls_sha1_starts_ret(&sha1);
    mbedtls_sha1_update_ret(&sha1, ssl->session_negotiate->master, 48);
    mbedtls_sha1_update_ret(&sha1, pad_2, 40);
    mbedtls_sha1_update_ret(&sha1, hash + 16, 20);
    mbedtls_sha1_finish_ret(&sha1, hash + 16);

    *hlen = 36;

    MBEDTLS_SSL_DEBUG_BUF(3, "calculated verify result", hash, *hlen);
    MBEDTLS_SSL_DEBUG_MSG(2, ("<= calc verify"));

    mbedtls_md5_free(&md5);
    mbedtls_sha1_free(&sha1);

    return;
}
#endif /* MBEDTLS_SSL_PROTO_SSL3 */

#if defined(MBEDTLS_SSL_PROTO_TLS1) || defined(MBEDTLS_SSL_PROTO_TLS1_1)
void ssl_calc_verify_tls(const mbedtls_ssl_context *ssl,
                         unsigned char *hash,
                         size_t *hlen)
{
    mbedtls_md5_context md5;
    mbedtls_sha1_context sha1;

    MBEDTLS_SSL_DEBUG_MSG(2, ("=> calc verify tls"));

    mbedtls_md5_init(&md5);
    mbedtls_sha1_init(&sha1);

    mbedtls_md5_clone(&md5, &ssl->handshake->fin_md5);
    mbedtls_sha1_clone(&sha1, &ssl->handshake->fin_sha1);

    mbedtls_md5_finish_ret(&md5,  hash);
    mbedtls_sha1_finish_ret(&sha1, hash + 16);

    *hlen = 36;

    MBEDTLS_SSL_DEBUG_BUF(3, "calculated verify result", hash, *hlen);
    MBEDTLS_SSL_DEBUG_MSG(2, ("<= calc verify"));

    mbedtls_md5_free(&md5);
    mbedtls_sha1_free(&sha1);

    return;
}
#endif /* MBEDTLS_SSL_PROTO_TLS1 || MBEDTLS_SSL_PROTO_TLS1_1 */

#if defined(MBEDTLS_SSL_PROTO_TLS1_2)
#if defined(MBEDTLS_SHA256_C)
void ssl_calc_verify_tls_sha256(const mbedtls_ssl_context *ssl,
                                unsigned char *hash,
                                size_t *hlen)
{
#if defined(MBEDTLS_USE_PSA_CRYPTO)
    size_t hash_size;
    psa_status_t status;
    psa_hash_operation_t sha256_psa = psa_hash_operation_init();

    MBEDTLS_SSL_DEBUG_MSG(2, ("=> PSA calc verify sha256"));
    status = psa_hash_clone(&ssl->handshake->fin_sha256_psa, &sha256_psa);
    if (status != PSA_SUCCESS) {
        MBEDTLS_SSL_DEBUG_MSG(2, ("PSA hash clone failed"));
        return;
    }

    status = psa_hash_finish(&sha256_psa, hash, 32, &hash_size);
    if (status != PSA_SUCCESS) {
        MBEDTLS_SSL_DEBUG_MSG(2, ("PSA hash finish failed"));
        return;
    }

    *hlen = 32;
    MBEDTLS_SSL_DEBUG_BUF(3, "PSA calculated verify result", hash, *hlen);
    MBEDTLS_SSL_DEBUG_MSG(2, ("<= PSA calc verify"));
#else
    mbedtls_sha256_context sha256;

    mbedtls_sha256_init(&sha256);

    MBEDTLS_SSL_DEBUG_MSG(2, ("=> calc verify sha256"));

    mbedtls_sha256_clone(&sha256, &ssl->handshake->fin_sha256);
    mbedtls_sha256_finish_ret(&sha256, hash);

    *hlen = 32;

    MBEDTLS_SSL_DEBUG_BUF(3, "calculated verify result", hash, *hlen);
    MBEDTLS_SSL_DEBUG_MSG(2, ("<= calc verify"));

    mbedtls_sha256_free(&sha256);
#endif /* MBEDTLS_USE_PSA_CRYPTO */
    return;
}
#endif /* MBEDTLS_SHA256_C */

#if defined(MBEDTLS_SHA512_C) && !defined(MBEDTLS_SHA512_NO_SHA384)
void ssl_calc_verify_tls_sha384(const mbedtls_ssl_context *ssl,
                                unsigned char *hash,
                                size_t *hlen)
{
#if defined(MBEDTLS_USE_PSA_CRYPTO)
    size_t hash_size;
    psa_status_t status;
    psa_hash_operation_t sha384_psa = psa_hash_operation_init();

    MBEDTLS_SSL_DEBUG_MSG(2, ("=> PSA calc verify sha384"));
    status = psa_hash_clone(&ssl->handshake->fin_sha384_psa, &sha384_psa);
    if (status != PSA_SUCCESS) {
        MBEDTLS_SSL_DEBUG_MSG(2, ("PSA hash clone failed"));
        return;
    }

    status = psa_hash_finish(&sha384_psa, hash, 48, &hash_size);
    if (status != PSA_SUCCESS) {
        MBEDTLS_SSL_DEBUG_MSG(2, ("PSA hash finish failed"));
        return;
    }

    *hlen = 48;
    MBEDTLS_SSL_DEBUG_BUF(3, "PSA calculated verify result", hash, *hlen);
    MBEDTLS_SSL_DEBUG_MSG(2, ("<= PSA calc verify"));
#else
    mbedtls_sha512_context sha512;

    mbedtls_sha512_init(&sha512);

    MBEDTLS_SSL_DEBUG_MSG(2, ("=> calc verify sha384"));

    mbedtls_sha512_clone(&sha512, &ssl->handshake->fin_sha512);
    mbedtls_sha512_finish_ret(&sha512, hash);

    *hlen = 48;

    MBEDTLS_SSL_DEBUG_BUF(3, "calculated verify result", hash, *hlen);
    MBEDTLS_SSL_DEBUG_MSG(2, ("<= calc verify"));

    mbedtls_sha512_free(&sha512);
#endif /* MBEDTLS_USE_PSA_CRYPTO */
    return;
}
#endif /* MBEDTLS_SHA512_C && !MBEDTLS_SHA512_NO_SHA384 */
#endif /* MBEDTLS_SSL_PROTO_TLS1_2 */

#if defined(MBEDTLS_KEY_EXCHANGE_SOME_PSK_ENABLED)
int mbedtls_ssl_psk_derive_premaster(mbedtls_ssl_context *ssl, mbedtls_key_exchange_type_t key_ex)
{
    unsigned char *p = ssl->handshake->premaster;
    unsigned char *end = p + sizeof(ssl->handshake->premaster);
    const unsigned char *psk = NULL;
    size_t psk_len = 0;

    if (mbedtls_ssl_get_psk(ssl, &psk, &psk_len)
        == MBEDTLS_ERR_SSL_PRIVATE_KEY_REQUIRED) {
        /*
         * This should never happen because the existence of a PSK is always
         * checked before calling this function
         */
        MBEDTLS_SSL_DEBUG_MSG(1, ("should never happen"));
        return MBEDTLS_ERR_SSL_INTERNAL_ERROR;
    }

    /*
     * PMS = struct {
     *     opaque other_secret<0..2^16-1>;
     *     opaque psk<0..2^16-1>;
     * };
     * with "other_secret" depending on the particular key exchange
     */
#if defined(MBEDTLS_KEY_EXCHANGE_PSK_ENABLED)
    if (key_ex == MBEDTLS_KEY_EXCHANGE_PSK) {
        if (end - p < 2) {
            return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
        }

        MBEDTLS_PUT_UINT16_BE(psk_len, p, 0);
        p += 2;

        if (end < p || (size_t) (end - p) < psk_len) {
            return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
        }

        memset(p, 0, psk_len);
        p += psk_len;
    } else
#endif /* MBEDTLS_KEY_EXCHANGE_PSK_ENABLED */
#if defined(MBEDTLS_KEY_EXCHANGE_RSA_PSK_ENABLED)
    if (key_ex == MBEDTLS_KEY_EXCHANGE_RSA_PSK) {
        /*
         * other_secret already set by the ClientKeyExchange message,
         * and is 48 bytes long
         */
        if (end - p < 2) {
            return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
        }

        *p++ = 0;
        *p++ = 48;
        p += 48;
    } else
#endif /* MBEDTLS_KEY_EXCHANGE_RSA_PSK_ENABLED */
#if defined(MBEDTLS_KEY_EXCHANGE_DHE_PSK_ENABLED)
    if (key_ex == MBEDTLS_KEY_EXCHANGE_DHE_PSK) {
        int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
        size_t len;

        /* Write length only when we know the actual value */
        if ((ret = mbedtls_dhm_calc_secret(&ssl->handshake->dhm_ctx,
                                           p + 2, end - (p + 2), &len,
                                           ssl->conf->f_rng, ssl->conf->p_rng)) != 0) {
            MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_dhm_calc_secret", ret);
            return ret;
        }
        MBEDTLS_PUT_UINT16_BE(len, p, 0);
        p += 2 + len;

        MBEDTLS_SSL_DEBUG_MPI(3, "DHM: K ", &ssl->handshake->dhm_ctx.K);
    } else
#endif /* MBEDTLS_KEY_EXCHANGE_DHE_PSK_ENABLED */
#if defined(MBEDTLS_KEY_EXCHANGE_ECDHE_PSK_ENABLED)
    if (key_ex == MBEDTLS_KEY_EXCHANGE_ECDHE_PSK) {
        int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
        size_t zlen;

        if ((ret = mbedtls_ecdh_calc_secret(&ssl->handshake->ecdh_ctx, &zlen,
                                            p + 2, end - (p + 2),
                                            ssl->conf->f_rng, ssl->conf->p_rng)) != 0) {
            MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_ecdh_calc_secret", ret);
            return ret;
        }

        MBEDTLS_PUT_UINT16_BE(zlen, p, 0);
        p += 2 + zlen;

        MBEDTLS_SSL_DEBUG_ECDH(3, &ssl->handshake->ecdh_ctx,
                               MBEDTLS_DEBUG_ECDH_Z);
    } else
#endif /* MBEDTLS_KEY_EXCHANGE_ECDHE_PSK_ENABLED */
    {
        MBEDTLS_SSL_DEBUG_MSG(1, ("should never happen"));
        return MBEDTLS_ERR_SSL_INTERNAL_ERROR;
    }

    /* opaque psk<0..2^16-1>; */
    if (end - p < 2) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    MBEDTLS_PUT_UINT16_BE(psk_len, p, 0);
    p += 2;

    if (end < p || (size_t) (end - p) < psk_len) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    memcpy(p, psk, psk_len);
    p += psk_len;

    ssl->handshake->pmslen = p - ssl->handshake->premaster;

    return 0;
}
#endif /* MBEDTLS_KEY_EXCHANGE_SOME_PSK_ENABLED */

#if defined(MBEDTLS_SSL_SRV_C) && defined(MBEDTLS_SSL_RENEGOTIATION)
MBEDTLS_CHECK_RETURN_CRITICAL
static int ssl_write_hello_request(mbedtls_ssl_context *ssl);

#if defined(MBEDTLS_SSL_PROTO_DTLS)
int mbedtls_ssl_resend_hello_request(mbedtls_ssl_context *ssl)
{
    /* If renegotiation is not enforced, retransmit until we would reach max
     * timeout if we were using the usual handshake doubling scheme */
    if (ssl->conf->renego_max_records < 0) {
        uint32_t ratio = ssl->conf->hs_timeout_max / ssl->conf->hs_timeout_min + 1;
        unsigned char doublings = 1;

        while (ratio != 0) {
            ++doublings;
            ratio >>= 1;
        }

        if (++ssl->renego_records_seen > doublings) {
            MBEDTLS_SSL_DEBUG_MSG(2, ("no longer retransmitting hello request"));
            return 0;
        }
    }

    return ssl_write_hello_request(ssl);
}
#endif
#endif /* MBEDTLS_SSL_SRV_C && MBEDTLS_SSL_RENEGOTIATION */

#if defined(MBEDTLS_X509_CRT_PARSE_C)
static void ssl_clear_peer_cert(mbedtls_ssl_session *session)
{
#if defined(MBEDTLS_SSL_KEEP_PEER_CERTIFICATE)
    if (session->peer_cert != NULL) {
        mbedtls_x509_crt_free(session->peer_cert);
        mbedtls_free(session->peer_cert);
        session->peer_cert = NULL;
    }
#else /* MBEDTLS_SSL_KEEP_PEER_CERTIFICATE */
    if (session->peer_cert_digest != NULL) {
        /* Zeroization is not necessary. */
        mbedtls_free(session->peer_cert_digest);
        session->peer_cert_digest      = NULL;
        session->peer_cert_digest_type = MBEDTLS_MD_NONE;
        session->peer_cert_digest_len  = 0;
    }
#endif /* !MBEDTLS_SSL_KEEP_PEER_CERTIFICATE */
}
#endif /* MBEDTLS_X509_CRT_PARSE_C */

/*
 * Handshake functions
 */
#if !defined(MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED)
/* No certificate support -> dummy functions */
int mbedtls_ssl_write_certificate(mbedtls_ssl_context *ssl)
{
    const mbedtls_ssl_ciphersuite_t *ciphersuite_info =
        ssl->handshake->ciphersuite_info;

    MBEDTLS_SSL_DEBUG_MSG(2, ("=> write certificate"));

    if (!mbedtls_ssl_ciphersuite_uses_srv_cert(ciphersuite_info)) {
        MBEDTLS_SSL_DEBUG_MSG(2, ("<= skip write certificate"));
        ssl->state++;
        return 0;
    }

    MBEDTLS_SSL_DEBUG_MSG(1, ("should never happen"));
    return MBEDTLS_ERR_SSL_INTERNAL_ERROR;
}

int mbedtls_ssl_parse_certificate(mbedtls_ssl_context *ssl)
{
    const mbedtls_ssl_ciphersuite_t *ciphersuite_info =
        ssl->handshake->ciphersuite_info;

    MBEDTLS_SSL_DEBUG_MSG(2, ("=> parse certificate"));

    if (!mbedtls_ssl_ciphersuite_uses_srv_cert(ciphersuite_info)) {
        MBEDTLS_SSL_DEBUG_MSG(2, ("<= skip parse certificate"));
        ssl->state++;
        return 0;
    }

    MBEDTLS_SSL_DEBUG_MSG(1, ("should never happen"));
    return MBEDTLS_ERR_SSL_INTERNAL_ERROR;
}

#else /* MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED */
/* Some certificate support -> implement write and parse */

int mbedtls_ssl_write_certificate(mbedtls_ssl_context *ssl)
{
    int ret = MBEDTLS_ERR_SSL_FEATURE_UNAVAILABLE;
    size_t i, n;
    const mbedtls_x509_crt *crt;
    const mbedtls_ssl_ciphersuite_t *ciphersuite_info =
        ssl->handshake->ciphersuite_info;

    MBEDTLS_SSL_DEBUG_MSG(2, ("=> write certificate"));

    if (!mbedtls_ssl_ciphersuite_uses_srv_cert(ciphersuite_info)) {
        MBEDTLS_SSL_DEBUG_MSG(2, ("<= skip write certificate"));
        ssl->state++;
        return 0;
    }

#if defined(MBEDTLS_SSL_CLI_C)
    if (ssl->conf->endpoint == MBEDTLS_SSL_IS_CLIENT) {
        if (ssl->client_auth == 0) {
            MBEDTLS_SSL_DEBUG_MSG(2, ("<= skip write certificate"));
            ssl->state++;
            return 0;
        }

#if defined(MBEDTLS_SSL_PROTO_SSL3)
        /*
         * If using SSLv3 and got no cert, send an Alert message
         * (otherwise an empty Certificate message will be sent).
         */
        if (mbedtls_ssl_own_cert(ssl)  == NULL &&
            ssl->minor_ver == MBEDTLS_SSL_MINOR_VERSION_0) {
            ssl->out_msglen  = 2;
            ssl->out_msgtype = MBEDTLS_SSL_MSG_ALERT;
            ssl->out_msg[0]  = MBEDTLS_SSL_ALERT_LEVEL_WARNING;
            ssl->out_msg[1]  = MBEDTLS_SSL_ALERT_MSG_NO_CERT;

            MBEDTLS_SSL_DEBUG_MSG(2, ("got no certificate to send"));
            goto write_msg;
        }
#endif /* MBEDTLS_SSL_PROTO_SSL3 */
    }
#endif /* MBEDTLS_SSL_CLI_C */
#if defined(MBEDTLS_SSL_SRV_C)
    if (ssl->conf->endpoint == MBEDTLS_SSL_IS_SERVER) {
        if (mbedtls_ssl_own_cert(ssl) == NULL) {
            MBEDTLS_SSL_DEBUG_MSG(1, ("got no certificate to send"));
            return MBEDTLS_ERR_SSL_CERTIFICATE_REQUIRED;
        }
    }
#endif

    MBEDTLS_SSL_DEBUG_CRT(3, "own certificate", mbedtls_ssl_own_cert(ssl));

    /*
     *     0  .  0    handshake type
     *     1  .  3    handshake length
     *     4  .  6    length of all certs
     *     7  .  9    length of cert. 1
     *    10  . n-1   peer certificate
     *     n  . n+2   length of cert. 2
     *    n+3 . ...   upper level cert, etc.
     */
    i = 7;
    crt = mbedtls_ssl_own_cert(ssl);

    while (crt != NULL) {
        n = crt->raw.len;
        if (n > MBEDTLS_SSL_OUT_CONTENT_LEN - 3 - i) {
            MBEDTLS_SSL_DEBUG_MSG(1, ("certificate too large, %" MBEDTLS_PRINTF_SIZET
                                      " > %" MBEDTLS_PRINTF_SIZET,
                                      i + 3 + n, (size_t) MBEDTLS_SSL_OUT_CONTENT_LEN));
            return MBEDTLS_ERR_SSL_CERTIFICATE_TOO_LARGE;
        }

        ssl->out_msg[i] = MBEDTLS_BYTE_2(n);
        ssl->out_msg[i + 1] = MBEDTLS_BYTE_1(n);
        ssl->out_msg[i + 2] = MBEDTLS_BYTE_0(n);

        i += 3; memcpy(ssl->out_msg + i, crt->raw.p, n);
        i += n; crt = crt->next;
    }

    ssl->out_msg[4]  = MBEDTLS_BYTE_2(i - 7);
    ssl->out_msg[5]  = MBEDTLS_BYTE_1(i - 7);
    ssl->out_msg[6]  = MBEDTLS_BYTE_0(i - 7);

    ssl->out_msglen  = i;
    ssl->out_msgtype = MBEDTLS_SSL_MSG_HANDSHAKE;
    ssl->out_msg[0]  = MBEDTLS_SSL_HS_CERTIFICATE;

#if defined(MBEDTLS_SSL_PROTO_SSL3) && defined(MBEDTLS_SSL_CLI_C)
write_msg:
#endif

    ssl->state++;

    if ((ret = mbedtls_ssl_write_handshake_msg(ssl)) != 0) {
        MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_ssl_write_handshake_msg", ret);
        return ret;
    }

    MBEDTLS_SSL_DEBUG_MSG(2, ("<= write certificate"));

    return ret;
}

#if defined(MBEDTLS_SSL_RENEGOTIATION) && defined(MBEDTLS_SSL_CLI_C)

#if defined(MBEDTLS_SSL_KEEP_PEER_CERTIFICATE)
MBEDTLS_CHECK_RETURN_CRITICAL
static int ssl_check_peer_crt_unchanged(mbedtls_ssl_context *ssl,
                                        unsigned char *crt_buf,
                                        size_t crt_buf_len)
{
    mbedtls_x509_crt const * const peer_crt = ssl->session->peer_cert;

    if (peer_crt == NULL) {
        return -1;
    }

    if (peer_crt->raw.len != crt_buf_len) {
        return -1;
    }

    return memcmp(peer_crt->raw.p, crt_buf, peer_crt->raw.len);
}
#else /* MBEDTLS_SSL_KEEP_PEER_CERTIFICATE */
MBEDTLS_CHECK_RETURN_CRITICAL
static int ssl_check_peer_crt_unchanged(mbedtls_ssl_context *ssl,
                                        unsigned char *crt_buf,
                                        size_t crt_buf_len)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    unsigned char const * const peer_cert_digest =
        ssl->session->peer_cert_digest;
    mbedtls_md_type_t const peer_cert_digest_type =
        ssl->session->peer_cert_digest_type;
    mbedtls_md_info_t const * const digest_info =
        mbedtls_md_info_from_type(peer_cert_digest_type);
    unsigned char tmp_digest[MBEDTLS_SSL_PEER_CERT_DIGEST_MAX_LEN];
    size_t digest_len;

    if (peer_cert_digest == NULL || digest_info == NULL) {
        return -1;
    }

    digest_len = mbedtls_md_get_size(digest_info);
    if (digest_len > MBEDTLS_SSL_PEER_CERT_DIGEST_MAX_LEN) {
        return -1;
    }

    ret = mbedtls_md(digest_info, crt_buf, crt_buf_len, tmp_digest);
    if (ret != 0) {
        return -1;
    }

    return memcmp(tmp_digest, peer_cert_digest, digest_len);
}
#endif /* MBEDTLS_SSL_KEEP_PEER_CERTIFICATE */
#endif /* MBEDTLS_SSL_RENEGOTIATION && MBEDTLS_SSL_CLI_C */

/*
 * Once the certificate message is read, parse it into a cert chain and
 * perform basic checks, but leave actual verification to the caller
 */
MBEDTLS_CHECK_RETURN_CRITICAL
static int ssl_parse_certificate_chain(mbedtls_ssl_context *ssl,
                                       mbedtls_x509_crt *chain)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
#if defined(MBEDTLS_SSL_RENEGOTIATION) && defined(MBEDTLS_SSL_CLI_C)
    int crt_cnt = 0;
#endif
    size_t i, n;
    uint8_t alert;

    if (ssl->in_msgtype != MBEDTLS_SSL_MSG_HANDSHAKE) {
        MBEDTLS_SSL_DEBUG_MSG(1, ("bad certificate message"));
        mbedtls_ssl_send_alert_message(ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                       MBEDTLS_SSL_ALERT_MSG_UNEXPECTED_MESSAGE);
        return MBEDTLS_ERR_SSL_UNEXPECTED_MESSAGE;
    }

    if (ssl->in_msg[0] != MBEDTLS_SSL_HS_CERTIFICATE ||
        ssl->in_hslen < mbedtls_ssl_hs_hdr_len(ssl) + 3 + 3) {
        MBEDTLS_SSL_DEBUG_MSG(1, ("bad certificate message"));
        mbedtls_ssl_send_alert_message(ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                       MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR);
        return MBEDTLS_ERR_SSL_BAD_HS_CERTIFICATE;
    }

    i = mbedtls_ssl_hs_hdr_len(ssl);

    /*
     * Same message structure as in mbedtls_ssl_write_certificate()
     */
    n = (ssl->in_msg[i+1] << 8) | ssl->in_msg[i+2];

    if (ssl->in_msg[i] != 0 ||
        ssl->in_hslen != n + 3 + mbedtls_ssl_hs_hdr_len(ssl)) {
        MBEDTLS_SSL_DEBUG_MSG(1, ("bad certificate message"));
        mbedtls_ssl_send_alert_message(ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                       MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR);
        return MBEDTLS_ERR_SSL_BAD_HS_CERTIFICATE;
    }

    /* Make &ssl->in_msg[i] point to the beginning of the CRT chain. */
    i += 3;

    /* Iterate through and parse the CRTs in the provided chain. */
    while (i < ssl->in_hslen) {
        /* Check that there's room for the next CRT's length fields. */
        if (i + 3 > ssl->in_hslen) {
            MBEDTLS_SSL_DEBUG_MSG(1, ("bad certificate message"));
            mbedtls_ssl_send_alert_message(ssl,
                                           MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                           MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR);
            return MBEDTLS_ERR_SSL_BAD_HS_CERTIFICATE;
        }
        /* In theory, the CRT can be up to 2**24 Bytes, but we don't support
         * anything beyond 2**16 ~ 64K. */
        if (ssl->in_msg[i] != 0) {
            MBEDTLS_SSL_DEBUG_MSG(1, ("bad certificate message"));
            mbedtls_ssl_send_alert_message(ssl,
                                           MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                           MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR);
            return MBEDTLS_ERR_SSL_BAD_HS_CERTIFICATE;
        }

        /* Read length of the next CRT in the chain. */
        n = ((unsigned int) ssl->in_msg[i + 1] << 8)
            | (unsigned int) ssl->in_msg[i + 2];
        i += 3;

        if (n < 128 || i + n > ssl->in_hslen) {
            MBEDTLS_SSL_DEBUG_MSG(1, ("bad certificate message"));
            mbedtls_ssl_send_alert_message(ssl,
                                           MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                           MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR);
            return MBEDTLS_ERR_SSL_BAD_HS_CERTIFICATE;
        }

        /* Check if we're handling the first CRT in the chain. */
#if defined(MBEDTLS_SSL_RENEGOTIATION) && defined(MBEDTLS_SSL_CLI_C)
        if (crt_cnt++ == 0 &&
            ssl->conf->endpoint == MBEDTLS_SSL_IS_CLIENT &&
            ssl->renego_status == MBEDTLS_SSL_RENEGOTIATION_IN_PROGRESS) {
            /* During client-side renegotiation, check that the server's
             * end-CRTs hasn't changed compared to the initial handshake,
             * mitigating the triple handshake attack. On success, reuse
             * the original end-CRT instead of parsing it again. */
            MBEDTLS_SSL_DEBUG_MSG(3, ("Check that peer CRT hasn't changed during renegotiation"));
            if (ssl_check_peer_crt_unchanged(ssl,
                                             &ssl->in_msg[i],
                                             n) != 0) {
                MBEDTLS_SSL_DEBUG_MSG(1, ("new server cert during renegotiation"));
                mbedtls_ssl_send_alert_message(ssl,
                                               MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                               MBEDTLS_SSL_ALERT_MSG_ACCESS_DENIED);
                return MBEDTLS_ERR_SSL_BAD_HS_CERTIFICATE;
            }

            /* Now we can safely free the original chain. */
            ssl_clear_peer_cert(ssl->session);
        }
#endif /* MBEDTLS_SSL_RENEGOTIATION && MBEDTLS_SSL_CLI_C */

        /* Parse the next certificate in the chain. */
#if defined(MBEDTLS_SSL_KEEP_PEER_CERTIFICATE)
        ret = mbedtls_x509_crt_parse_der(chain, ssl->in_msg + i, n);
#else
        /* If we don't need to store the CRT chain permanently, parse
         * it in-place from the input buffer instead of making a copy. */
        ret = mbedtls_x509_crt_parse_der_nocopy(chain, ssl->in_msg + i, n);
#endif /* MBEDTLS_SSL_KEEP_PEER_CERTIFICATE */
        switch (ret) {
            case 0: /*ok*/
            case MBEDTLS_ERR_X509_UNKNOWN_SIG_ALG + MBEDTLS_ERR_OID_NOT_FOUND:
                /* Ignore certificate with an unknown algorithm: maybe a
                   prior certificate was already trusted. */
                break;

            case MBEDTLS_ERR_X509_ALLOC_FAILED:
                alert = MBEDTLS_SSL_ALERT_MSG_INTERNAL_ERROR;
                goto crt_parse_der_failed;

            case MBEDTLS_ERR_X509_UNKNOWN_VERSION:
                alert = MBEDTLS_SSL_ALERT_MSG_UNSUPPORTED_CERT;
                goto crt_parse_der_failed;

            default:
                alert = MBEDTLS_SSL_ALERT_MSG_BAD_CERT;
crt_parse_der_failed:
                mbedtls_ssl_send_alert_message(ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL, alert);
                MBEDTLS_SSL_DEBUG_RET(1, " mbedtls_x509_crt_parse_der", ret);
                return ret;
        }

        i += n;
    }

    MBEDTLS_SSL_DEBUG_CRT(3, "peer certificate", chain);
    return 0;
}

#if defined(MBEDTLS_SSL_SRV_C)
MBEDTLS_CHECK_RETURN_CRITICAL
static int ssl_srv_check_client_no_crt_notification(mbedtls_ssl_context *ssl)
{
    if (ssl->conf->endpoint == MBEDTLS_SSL_IS_CLIENT) {
        return -1;
    }

#if defined(MBEDTLS_SSL_PROTO_SSL3)
    /*
     * Check if the client sent an empty certificate
     */
    if (ssl->minor_ver == MBEDTLS_SSL_MINOR_VERSION_0) {
        if (ssl->in_msglen  == 2                        &&
            ssl->in_msgtype == MBEDTLS_SSL_MSG_ALERT            &&
            ssl->in_msg[0]  == MBEDTLS_SSL_ALERT_LEVEL_WARNING  &&
            ssl->in_msg[1]  == MBEDTLS_SSL_ALERT_MSG_NO_CERT) {
            MBEDTLS_SSL_DEBUG_MSG(1, ("SSLv3 client has no certificate"));
            return 0;
        }

        return -1;
    }
#endif /* MBEDTLS_SSL_PROTO_SSL3 */

#if defined(MBEDTLS_SSL_PROTO_TLS1) || defined(MBEDTLS_SSL_PROTO_TLS1_1) || \
    defined(MBEDTLS_SSL_PROTO_TLS1_2)
    if (ssl->in_hslen   == 3 + mbedtls_ssl_hs_hdr_len(ssl) &&
        ssl->in_msgtype == MBEDTLS_SSL_MSG_HANDSHAKE    &&
        ssl->in_msg[0]  == MBEDTLS_SSL_HS_CERTIFICATE   &&
        memcmp(ssl->in_msg + mbedtls_ssl_hs_hdr_len(ssl), "\0\0\0", 3) == 0) {
        MBEDTLS_SSL_DEBUG_MSG(1, ("TLSv1 client has no certificate"));
        return 0;
    }

    return -1;
#endif /* MBEDTLS_SSL_PROTO_TLS1 || MBEDTLS_SSL_PROTO_TLS1_1 || \
          MBEDTLS_SSL_PROTO_TLS1_2 */
}
#endif /* MBEDTLS_SSL_SRV_C */

/* Check if a certificate message is expected.
 * Return either
 * - SSL_CERTIFICATE_EXPECTED, or
 * - SSL_CERTIFICATE_SKIP
 * indicating whether a Certificate message is expected or not.
 */
#define SSL_CERTIFICATE_EXPECTED 0
#define SSL_CERTIFICATE_SKIP     1
MBEDTLS_CHECK_RETURN_CRITICAL
static int ssl_parse_certificate_coordinate(mbedtls_ssl_context *ssl,
                                            int authmode)
{
    const mbedtls_ssl_ciphersuite_t *ciphersuite_info =
        ssl->handshake->ciphersuite_info;

    if (!mbedtls_ssl_ciphersuite_uses_srv_cert(ciphersuite_info)) {
        return SSL_CERTIFICATE_SKIP;
    }

#if defined(MBEDTLS_SSL_SRV_C)
    if (ssl->conf->endpoint == MBEDTLS_SSL_IS_SERVER) {
        if (ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_RSA_PSK) {
            return SSL_CERTIFICATE_SKIP;
        }

        if (authmode == MBEDTLS_SSL_VERIFY_NONE) {
            ssl->session_negotiate->verify_result =
                MBEDTLS_X509_BADCERT_SKIP_VERIFY;
            return SSL_CERTIFICATE_SKIP;
        }
    }
#else
    ((void) authmode);
#endif /* MBEDTLS_SSL_SRV_C */

    return SSL_CERTIFICATE_EXPECTED;
}

MBEDTLS_CHECK_RETURN_CRITICAL
static int ssl_parse_certificate_verify(mbedtls_ssl_context *ssl,
                                        int authmode,
                                        mbedtls_x509_crt *chain,
                                        void *rs_ctx)
{
    int ret = 0;
    const mbedtls_ssl_ciphersuite_t *ciphersuite_info =
        ssl->handshake->ciphersuite_info;
    int have_ca_chain = 0;

    int (*f_vrfy)(void *, mbedtls_x509_crt *, int, uint32_t *);
    void *p_vrfy;

    if (authmode == MBEDTLS_SSL_VERIFY_NONE) {
        return 0;
    }

    if (ssl->f_vrfy != NULL) {
        MBEDTLS_SSL_DEBUG_MSG(3, ("Use context-specific verification callback"));
        f_vrfy = ssl->f_vrfy;
        p_vrfy = ssl->p_vrfy;
    } else {
        MBEDTLS_SSL_DEBUG_MSG(3, ("Use configuration-specific verification callback"));
        f_vrfy = ssl->conf->f_vrfy;
        p_vrfy = ssl->conf->p_vrfy;
    }

    /*
     * Main check: verify certificate
     */
#if defined(MBEDTLS_X509_TRUSTED_CERTIFICATE_CALLBACK)
    if (ssl->conf->f_ca_cb != NULL) {
        ((void) rs_ctx);
        have_ca_chain = 1;

        MBEDTLS_SSL_DEBUG_MSG(3, ("use CA callback for X.509 CRT verification"));
        ret = mbedtls_x509_crt_verify_with_ca_cb(
            chain,
            ssl->conf->f_ca_cb,
            ssl->conf->p_ca_cb,
            ssl->conf->cert_profile,
            ssl->hostname,
            &ssl->session_negotiate->verify_result,
            f_vrfy, p_vrfy);
    } else
#endif /* MBEDTLS_X509_TRUSTED_CERTIFICATE_CALLBACK */
    {
        mbedtls_x509_crt *ca_chain;
        mbedtls_x509_crl *ca_crl;

#if defined(MBEDTLS_SSL_SERVER_NAME_INDICATION)
        if (ssl->handshake->sni_ca_chain != NULL) {
            ca_chain = ssl->handshake->sni_ca_chain;
            ca_crl   = ssl->handshake->sni_ca_crl;
        } else
#endif
        {
            ca_chain = ssl->conf->ca_chain;
            ca_crl   = ssl->conf->ca_crl;
        }

        if (ca_chain != NULL) {
            have_ca_chain = 1;
        }

        ret = mbedtls_x509_crt_verify_restartable(
            chain,
            ca_chain, ca_crl,
            ssl->conf->cert_profile,
            ssl->hostname,
            &ssl->session_negotiate->verify_result,
            f_vrfy, p_vrfy, rs_ctx);
    }

    if (ret != 0) {
        MBEDTLS_SSL_DEBUG_RET(1, "x509_verify_cert", ret);
    }

#if defined(MBEDTLS_SSL_ECP_RESTARTABLE_ENABLED)
    if (ret == MBEDTLS_ERR_ECP_IN_PROGRESS) {
        return MBEDTLS_ERR_SSL_CRYPTO_IN_PROGRESS;
    }
#endif

    /*
     * Secondary checks: always done, but change 'ret' only if it was 0
     */

#if defined(MBEDTLS_ECP_C)
    {
        const mbedtls_pk_context *pk = &chain->pk;

        /* If certificate uses an EC key, make sure the curve is OK.
         * This is a public key, so it can't be opaque, so can_do() is a good
         * enough check to ensure pk_ec() is safe to use here. */
        if (mbedtls_pk_can_do(pk, MBEDTLS_PK_ECKEY) &&
            mbedtls_ssl_check_curve(ssl, mbedtls_pk_ec(*pk)->grp.id) != 0) {
            ssl->session_negotiate->verify_result |= MBEDTLS_X509_BADCERT_BAD_KEY;

            MBEDTLS_SSL_DEBUG_MSG(1, ("bad certificate (EC key curve)"));
            if (ret == 0) {
                ret = MBEDTLS_ERR_SSL_BAD_HS_CERTIFICATE;
            }
        }
    }
#endif /* MBEDTLS_ECP_C */

    if (mbedtls_ssl_check_cert_usage(chain,
                                     ciphersuite_info,
                                     !ssl->conf->endpoint,
                                     &ssl->session_negotiate->verify_result) != 0) {
        MBEDTLS_SSL_DEBUG_MSG(1, ("bad certificate (usage extensions)"));
        if (ret == 0) {
            ret = MBEDTLS_ERR_SSL_BAD_HS_CERTIFICATE;
        }
    }

    /* mbedtls_x509_crt_verify_with_profile is supposed to report a
     * verification failure through MBEDTLS_ERR_X509_CERT_VERIFY_FAILED,
     * with details encoded in the verification flags. All other kinds
     * of error codes, including those from the user provided f_vrfy
     * functions, are treated as fatal and lead to a failure of
     * ssl_parse_certificate even if verification was optional. */
    if (authmode == MBEDTLS_SSL_VERIFY_OPTIONAL &&
        (ret == MBEDTLS_ERR_X509_CERT_VERIFY_FAILED ||
         ret == MBEDTLS_ERR_SSL_BAD_HS_CERTIFICATE)) {
        ret = 0;
    }

    if (have_ca_chain == 0 && authmode == MBEDTLS_SSL_VERIFY_REQUIRED) {
        MBEDTLS_SSL_DEBUG_MSG(1, ("got no CA chain"));
        ret = MBEDTLS_ERR_SSL_CA_CHAIN_REQUIRED;
    }

    if (ret != 0) {
        uint8_t alert;

        /* The certificate may have been rejected for several reasons.
           Pick one and send the corresponding alert. Which alert to send
           may be a subject of debate in some cases. */
        if (ssl->session_negotiate->verify_result & MBEDTLS_X509_BADCERT_OTHER) {
            alert = MBEDTLS_SSL_ALERT_MSG_ACCESS_DENIED;
        } else if (ssl->session_negotiate->verify_result & MBEDTLS_X509_BADCERT_CN_MISMATCH) {
            alert = MBEDTLS_SSL_ALERT_MSG_BAD_CERT;
        } else if (ssl->session_negotiate->verify_result & MBEDTLS_X509_BADCERT_KEY_USAGE) {
            alert = MBEDTLS_SSL_ALERT_MSG_UNSUPPORTED_CERT;
        } else if (ssl->session_negotiate->verify_result & MBEDTLS_X509_BADCERT_EXT_KEY_USAGE) {
            alert = MBEDTLS_SSL_ALERT_MSG_UNSUPPORTED_CERT;
        } else if (ssl->session_negotiate->verify_result & MBEDTLS_X509_BADCERT_NS_CERT_TYPE) {
            alert = MBEDTLS_SSL_ALERT_MSG_UNSUPPORTED_CERT;
        } else if (ssl->session_negotiate->verify_result & MBEDTLS_X509_BADCERT_BAD_PK) {
            alert = MBEDTLS_SSL_ALERT_MSG_UNSUPPORTED_CERT;
        } else if (ssl->session_negotiate->verify_result & MBEDTLS_X509_BADCERT_BAD_KEY) {
            alert = MBEDTLS_SSL_ALERT_MSG_UNSUPPORTED_CERT;
        } else if (ssl->session_negotiate->verify_result & MBEDTLS_X509_BADCERT_EXPIRED) {
            alert = MBEDTLS_SSL_ALERT_MSG_CERT_EXPIRED;
        } else if (ssl->session_negotiate->verify_result & MBEDTLS_X509_BADCERT_REVOKED) {
            alert = MBEDTLS_SSL_ALERT_MSG_CERT_REVOKED;
        } else if (ssl->session_negotiate->verify_result & MBEDTLS_X509_BADCERT_NOT_TRUSTED) {
            alert = MBEDTLS_SSL_ALERT_MSG_UNKNOWN_CA;
        } else {
            alert = MBEDTLS_SSL_ALERT_MSG_CERT_UNKNOWN;
        }
        mbedtls_ssl_send_alert_message(ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                       alert);
    }

#if defined(MBEDTLS_DEBUG_C)
    if (ssl->session_negotiate->verify_result != 0) {
        MBEDTLS_SSL_DEBUG_MSG(3, ("! Certificate verification flags %08x",
                                  (unsigned int) ssl->session_negotiate->verify_result));
    } else {
        MBEDTLS_SSL_DEBUG_MSG(3, ("Certificate verification flags clear"));
    }
#endif /* MBEDTLS_DEBUG_C */

    return ret;
}

#if !defined(MBEDTLS_SSL_KEEP_PEER_CERTIFICATE)
MBEDTLS_CHECK_RETURN_CRITICAL
static int ssl_remember_peer_crt_digest(mbedtls_ssl_context *ssl,
                                        unsigned char *start, size_t len)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    /* Remember digest of the peer's end-CRT. */
    ssl->session_negotiate->peer_cert_digest =
        mbedtls_calloc(1, MBEDTLS_SSL_PEER_CERT_DIGEST_DFL_LEN);
    if (ssl->session_negotiate->peer_cert_digest == NULL) {
        MBEDTLS_SSL_DEBUG_MSG(1, ("alloc(%d bytes) failed",
                                  MBEDTLS_SSL_PEER_CERT_DIGEST_DFL_LEN));
        mbedtls_ssl_send_alert_message(ssl,
                                       MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                       MBEDTLS_SSL_ALERT_MSG_INTERNAL_ERROR);

        return MBEDTLS_ERR_SSL_ALLOC_FAILED;
    }

    ret = mbedtls_md(mbedtls_md_info_from_type(
                         MBEDTLS_SSL_PEER_CERT_DIGEST_DFL_TYPE),
                     start, len,
                     ssl->session_negotiate->peer_cert_digest);

    ssl->session_negotiate->peer_cert_digest_type =
        MBEDTLS_SSL_PEER_CERT_DIGEST_DFL_TYPE;
    ssl->session_negotiate->peer_cert_digest_len =
        MBEDTLS_SSL_PEER_CERT_DIGEST_DFL_LEN;

    return ret;
}

MBEDTLS_CHECK_RETURN_CRITICAL
static int ssl_remember_peer_pubkey(mbedtls_ssl_context *ssl,
                                    unsigned char *start, size_t len)
{
    unsigned char *end = start + len;
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;

    /* Make a copy of the peer's raw public key. */
    mbedtls_pk_init(&ssl->handshake->peer_pubkey);
    ret = mbedtls_pk_parse_subpubkey(&start, end,
                                     &ssl->handshake->peer_pubkey);
    if (ret != 0) {
        /* We should have parsed the public key before. */
        return MBEDTLS_ERR_SSL_INTERNAL_ERROR;
    }

    return 0;
}
#endif /* !MBEDTLS_SSL_KEEP_PEER_CERTIFICATE */

int mbedtls_ssl_parse_certificate(mbedtls_ssl_context *ssl)
{
    int ret = 0;
    int crt_expected;
#if defined(MBEDTLS_SSL_SRV_C) && defined(MBEDTLS_SSL_SERVER_NAME_INDICATION)
    const int authmode = ssl->handshake->sni_authmode != MBEDTLS_SSL_VERIFY_UNSET
                       ? ssl->handshake->sni_authmode
                       : ssl->conf->authmode;
#else
    const int authmode = ssl->conf->authmode;
#endif
    void *rs_ctx = NULL;
    mbedtls_x509_crt *chain = NULL;

    MBEDTLS_SSL_DEBUG_MSG(2, ("=> parse certificate"));

    crt_expected = ssl_parse_certificate_coordinate(ssl, authmode);
    if (crt_expected == SSL_CERTIFICATE_SKIP) {
        MBEDTLS_SSL_DEBUG_MSG(2, ("<= skip parse certificate"));
        goto exit;
    }

#if defined(MBEDTLS_SSL_ECP_RESTARTABLE_ENABLED)
    if (ssl->handshake->ecrs_enabled &&
        ssl->handshake->ecrs_state == ssl_ecrs_crt_verify) {
        chain = ssl->handshake->ecrs_peer_cert;
        ssl->handshake->ecrs_peer_cert = NULL;
        goto crt_verify;
    }
#endif

    if ((ret = mbedtls_ssl_read_record(ssl, 1)) != 0) {
        /* mbedtls_ssl_read_record may have sent an alert already. We
           let it decide whether to alert. */
        MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_ssl_read_record", ret);
        goto exit;
    }

#if defined(MBEDTLS_SSL_SRV_C)
    if (ssl_srv_check_client_no_crt_notification(ssl) == 0) {
        ssl->session_negotiate->verify_result = MBEDTLS_X509_BADCERT_MISSING;

        if (authmode != MBEDTLS_SSL_VERIFY_OPTIONAL) {
            ret = MBEDTLS_ERR_SSL_NO_CLIENT_CERTIFICATE;
        }

        goto exit;
    }
#endif /* MBEDTLS_SSL_SRV_C */

    /* Clear existing peer CRT structure in case we tried to
     * reuse a session but it failed, and allocate a new one. */
    ssl_clear_peer_cert(ssl->session_negotiate);

    chain = mbedtls_calloc(1, sizeof(mbedtls_x509_crt));
    if (chain == NULL) {
        MBEDTLS_SSL_DEBUG_MSG(1, ("alloc(%" MBEDTLS_PRINTF_SIZET " bytes) failed",
                                  sizeof(mbedtls_x509_crt)));
        mbedtls_ssl_send_alert_message(ssl,
                                       MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                       MBEDTLS_SSL_ALERT_MSG_INTERNAL_ERROR);

        ret = MBEDTLS_ERR_SSL_ALLOC_FAILED;
        goto exit;
    }
    mbedtls_x509_crt_init(chain);

    ret = ssl_parse_certificate_chain(ssl, chain);
    if (ret != 0) {
        goto exit;
    }

#if defined(MBEDTLS_SSL_ECP_RESTARTABLE_ENABLED)
    if (ssl->handshake->ecrs_enabled) {
        ssl->handshake->ecrs_state = ssl_ecrs_crt_verify;
    }

crt_verify:
    if (ssl->handshake->ecrs_enabled) {
        rs_ctx = &ssl->handshake->ecrs_ctx;
    }
#endif

    ret = ssl_parse_certificate_verify(ssl, authmode,
                                       chain, rs_ctx);
    if (ret != 0) {
        goto exit;
    }

#if !defined(MBEDTLS_SSL_KEEP_PEER_CERTIFICATE)
    {
        unsigned char *crt_start, *pk_start;
        size_t crt_len, pk_len;

        /* We parse the CRT chain without copying, so
         * these pointers point into the input buffer,
         * and are hence still valid after freeing the
         * CRT chain. */

        crt_start = chain->raw.p;
        crt_len   = chain->raw.len;

        pk_start = chain->pk_raw.p;
        pk_len   = chain->pk_raw.len;

        /* Free the CRT structures before computing
         * digest and copying the peer's public key. */
        mbedtls_x509_crt_free(chain);
        mbedtls_free(chain);
        chain = NULL;

        ret = ssl_remember_peer_crt_digest(ssl, crt_start, crt_len);
        if (ret != 0) {
            goto exit;
        }

        ret = ssl_remember_peer_pubkey(ssl, pk_start, pk_len);
        if (ret != 0) {
            goto exit;
        }
    }
#else /* !MBEDTLS_SSL_KEEP_PEER_CERTIFICATE */
    /* Pass ownership to session structure. */
    ssl->session_negotiate->peer_cert = chain;
    chain = NULL;
#endif /* MBEDTLS_SSL_KEEP_PEER_CERTIFICATE */

    MBEDTLS_SSL_DEBUG_MSG(2, ("<= parse certificate"));

exit:

    if (ret == 0) {
        ssl->state++;
    }

#if defined(MBEDTLS_SSL_ECP_RESTARTABLE_ENABLED)
    if (ret == MBEDTLS_ERR_SSL_CRYPTO_IN_PROGRESS) {
        ssl->handshake->ecrs_peer_cert = chain;
        chain = NULL;
    }
#endif

    if (chain != NULL) {
        mbedtls_x509_crt_free(chain);
        mbedtls_free(chain);
    }

    return ret;
}
#endif /* MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED */

void mbedtls_ssl_optimize_checksum(mbedtls_ssl_context *ssl,
                                   const mbedtls_ssl_ciphersuite_t *ciphersuite_info)
{
    ((void) ciphersuite_info);

#if defined(MBEDTLS_SSL_PROTO_SSL3) || defined(MBEDTLS_SSL_PROTO_TLS1) || \
    defined(MBEDTLS_SSL_PROTO_TLS1_1)
    if (ssl->minor_ver < MBEDTLS_SSL_MINOR_VERSION_3) {
        ssl->handshake->update_checksum = ssl_update_checksum_md5sha1;
    } else
#endif
#if defined(MBEDTLS_SSL_PROTO_TLS1_2)
#if defined(MBEDTLS_SHA512_C) && !defined(MBEDTLS_SHA512_NO_SHA384)
    if (ciphersuite_info->mac == MBEDTLS_MD_SHA384) {
        ssl->handshake->update_checksum = ssl_update_checksum_sha384;
    } else
#endif
#if defined(MBEDTLS_SHA256_C)
    if (ciphersuite_info->mac != MBEDTLS_MD_SHA384) {
        ssl->handshake->update_checksum = ssl_update_checksum_sha256;
    } else
#endif
#endif /* MBEDTLS_SSL_PROTO_TLS1_2 */
    {
        MBEDTLS_SSL_DEBUG_MSG(1, ("should never happen"));
        return;
    }
}

void mbedtls_ssl_reset_checksum(mbedtls_ssl_context *ssl)
{
#if defined(MBEDTLS_SSL_PROTO_SSL3) || defined(MBEDTLS_SSL_PROTO_TLS1) || \
    defined(MBEDTLS_SSL_PROTO_TLS1_1)
    mbedtls_md5_starts_ret(&ssl->handshake->fin_md5);
    mbedtls_sha1_starts_ret(&ssl->handshake->fin_sha1);
#endif
#if defined(MBEDTLS_SSL_PROTO_TLS1_2)
#if defined(MBEDTLS_SHA256_C)
#if defined(MBEDTLS_USE_PSA_CRYPTO)
    psa_hash_abort(&ssl->handshake->fin_sha256_psa);
    psa_hash_setup(&ssl->handshake->fin_sha256_psa, PSA_ALG_SHA_256);
#else
    mbedtls_sha256_starts_ret(&ssl->handshake->fin_sha256, 0);
#endif
#endif
#if defined(MBEDTLS_SHA512_C) && !defined(MBEDTLS_SHA512_NO_SHA384)
#if defined(MBEDTLS_USE_PSA_CRYPTO)
    psa_hash_abort(&ssl->handshake->fin_sha384_psa);
    psa_hash_setup(&ssl->handshake->fin_sha384_psa, PSA_ALG_SHA_384);
#else
    mbedtls_sha512_starts_ret(&ssl->handshake->fin_sha512, 1);
#endif
#endif
#endif /* MBEDTLS_SSL_PROTO_TLS1_2 */
}

static void ssl_update_checksum_start(mbedtls_ssl_context *ssl,
                                      const unsigned char *buf, size_t len)
{
#if defined(MBEDTLS_SSL_PROTO_SSL3) || defined(MBEDTLS_SSL_PROTO_TLS1) || \
    defined(MBEDTLS_SSL_PROTO_TLS1_1)
    mbedtls_md5_update_ret(&ssl->handshake->fin_md5, buf, len);
    mbedtls_sha1_update_ret(&ssl->handshake->fin_sha1, buf, len);
#endif
#if defined(MBEDTLS_SSL_PROTO_TLS1_2)
#if defined(MBEDTLS_SHA256_C)
#if defined(MBEDTLS_USE_PSA_CRYPTO)
    psa_hash_update(&ssl->handshake->fin_sha256_psa, buf, len);
#else
    mbedtls_sha256_update_ret(&ssl->handshake->fin_sha256, buf, len);
#endif
#endif
#if defined(MBEDTLS_SHA512_C) && !defined(MBEDTLS_SHA512_NO_SHA384)
#if defined(MBEDTLS_USE_PSA_CRYPTO)
    psa_hash_update(&ssl->handshake->fin_sha384_psa, buf, len);
#else
    mbedtls_sha512_update_ret(&ssl->handshake->fin_sha512, buf, len);
#endif
#endif
#endif /* MBEDTLS_SSL_PROTO_TLS1_2 */
}

#if defined(MBEDTLS_SSL_PROTO_SSL3) || defined(MBEDTLS_SSL_PROTO_TLS1) || \
    defined(MBEDTLS_SSL_PROTO_TLS1_1)
static void ssl_update_checksum_md5sha1(mbedtls_ssl_context *ssl,
                                        const unsigned char *buf, size_t len)
{
    mbedtls_md5_update_ret(&ssl->handshake->fin_md5, buf, len);
    mbedtls_sha1_update_ret(&ssl->handshake->fin_sha1, buf, len);
}
#endif

#if defined(MBEDTLS_SSL_PROTO_TLS1_2)
#if defined(MBEDTLS_SHA256_C)
static void ssl_update_checksum_sha256(mbedtls_ssl_context *ssl,
                                       const unsigned char *buf, size_t len)
{
#if defined(MBEDTLS_USE_PSA_CRYPTO)
    psa_hash_update(&ssl->handshake->fin_sha256_psa, buf, len);
#else
    mbedtls_sha256_update_ret(&ssl->handshake->fin_sha256, buf, len);
#endif
}
#endif

#if defined(MBEDTLS_SHA512_C) && !defined(MBEDTLS_SHA512_NO_SHA384)
static void ssl_update_checksum_sha384(mbedtls_ssl_context *ssl,
                                       const unsigned char *buf, size_t len)
{
#if defined(MBEDTLS_USE_PSA_CRYPTO)
    psa_hash_update(&ssl->handshake->fin_sha384_psa, buf, len);
#else
    mbedtls_sha512_update_ret(&ssl->handshake->fin_sha512, buf, len);
#endif
}
#endif
#endif /* MBEDTLS_SSL_PROTO_TLS1_2 */

#if defined(MBEDTLS_SSL_PROTO_SSL3)
static void ssl_calc_finished_ssl(
    mbedtls_ssl_context *ssl, unsigned char *buf, int from)
{
    const char *sender;
    mbedtls_md5_context  md5;
    mbedtls_sha1_context sha1;

    unsigned char padbuf[48];
    unsigned char md5sum[16];
    unsigned char sha1sum[20];

    mbedtls_ssl_session *session = ssl->session_negotiate;
    if (!session) {
        session = ssl->session;
    }

    MBEDTLS_SSL_DEBUG_MSG(2, ("=> calc  finished ssl"));

    mbedtls_md5_init(&md5);
    mbedtls_sha1_init(&sha1);

    mbedtls_md5_clone(&md5, &ssl->handshake->fin_md5);
    mbedtls_sha1_clone(&sha1, &ssl->handshake->fin_sha1);

    /*
     * SSLv3:
     *   hash =
     *      MD5( master + pad2 +
     *          MD5( handshake + sender + master + pad1 ) )
     *   + SHA1( master + pad2 +
     *         SHA1( handshake + sender + master + pad1 ) )
     */

#if !defined(MBEDTLS_MD5_ALT)
    MBEDTLS_SSL_DEBUG_BUF(4, "finished  md5 state", (unsigned char *)
                          md5.state, sizeof(md5.state));
#endif

#if !defined(MBEDTLS_SHA1_ALT)
    MBEDTLS_SSL_DEBUG_BUF(4, "finished sha1 state", (unsigned char *)
                          sha1.state, sizeof(sha1.state));
#endif

    sender = (from == MBEDTLS_SSL_IS_CLIENT) ? "CLNT"
                                       : "SRVR";

    memset(padbuf, 0x36, 48);

    mbedtls_md5_update_ret(&md5, (const unsigned char *) sender, 4);
    mbedtls_md5_update_ret(&md5, session->master, 48);
    mbedtls_md5_update_ret(&md5, padbuf, 48);
    mbedtls_md5_finish_ret(&md5, md5sum);

    mbedtls_sha1_update_ret(&sha1, (const unsigned char *) sender, 4);
    mbedtls_sha1_update_ret(&sha1, session->master, 48);
    mbedtls_sha1_update_ret(&sha1, padbuf, 40);
    mbedtls_sha1_finish_ret(&sha1, sha1sum);

    memset(padbuf, 0x5C, 48);

    mbedtls_md5_starts_ret(&md5);
    mbedtls_md5_update_ret(&md5, session->master, 48);
    mbedtls_md5_update_ret(&md5, padbuf, 48);
    mbedtls_md5_update_ret(&md5, md5sum, 16);
    mbedtls_md5_finish_ret(&md5, buf);

    mbedtls_sha1_starts_ret(&sha1);
    mbedtls_sha1_update_ret(&sha1, session->master, 48);
    mbedtls_sha1_update_ret(&sha1, padbuf, 40);
    mbedtls_sha1_update_ret(&sha1, sha1sum, 20);
    mbedtls_sha1_finish_ret(&sha1, buf + 16);

    MBEDTLS_SSL_DEBUG_BUF(3, "calc finished result", buf, 36);

    mbedtls_md5_free(&md5);
    mbedtls_sha1_free(&sha1);

    mbedtls_platform_zeroize(padbuf, sizeof(padbuf));
    mbedtls_platform_zeroize(md5sum, sizeof(md5sum));
    mbedtls_platform_zeroize(sha1sum, sizeof(sha1sum));

    MBEDTLS_SSL_DEBUG_MSG(2, ("<= calc  finished"));
}
#endif /* MBEDTLS_SSL_PROTO_SSL3 */

#if defined(MBEDTLS_SSL_PROTO_TLS1) || defined(MBEDTLS_SSL_PROTO_TLS1_1)
static void ssl_calc_finished_tls(
    mbedtls_ssl_context *ssl, unsigned char *buf, int from)
{
    int len = 12;
    const char *sender;
    mbedtls_md5_context  md5;
    mbedtls_sha1_context sha1;
    unsigned char padbuf[36];

    mbedtls_ssl_session *session = ssl->session_negotiate;
    if (!session) {
        session = ssl->session;
    }

    MBEDTLS_SSL_DEBUG_MSG(2, ("=> calc  finished tls"));

    mbedtls_md5_init(&md5);
    mbedtls_sha1_init(&sha1);

    mbedtls_md5_clone(&md5, &ssl->handshake->fin_md5);
    mbedtls_sha1_clone(&sha1, &ssl->handshake->fin_sha1);

    /*
     * TLSv1:
     *   hash = PRF( master, finished_label,
     *               MD5( handshake ) + SHA1( handshake ) )[0..11]
     */

#if !defined(MBEDTLS_MD5_ALT)
    MBEDTLS_SSL_DEBUG_BUF(4, "finished  md5 state", (unsigned char *)
                          md5.state, sizeof(md5.state));
#endif

#if !defined(MBEDTLS_SHA1_ALT)
    MBEDTLS_SSL_DEBUG_BUF(4, "finished sha1 state", (unsigned char *)
                          sha1.state, sizeof(sha1.state));
#endif

    sender = (from == MBEDTLS_SSL_IS_CLIENT)
             ? "client finished"
             : "server finished";

    mbedtls_md5_finish_ret(&md5, padbuf);
    mbedtls_sha1_finish_ret(&sha1, padbuf + 16);

    ssl->handshake->tls_prf(session->master, 48, sender,
                            padbuf, 36, buf, len);

    MBEDTLS_SSL_DEBUG_BUF(3, "calc finished result", buf, len);

    mbedtls_md5_free(&md5);
    mbedtls_sha1_free(&sha1);

    mbedtls_platform_zeroize(padbuf, sizeof(padbuf));

    MBEDTLS_SSL_DEBUG_MSG(2, ("<= calc  finished"));
}
#endif /* MBEDTLS_SSL_PROTO_TLS1 || MBEDTLS_SSL_PROTO_TLS1_1 */

#if defined(MBEDTLS_SSL_PROTO_TLS1_2)
#if defined(MBEDTLS_SHA256_C)
static void ssl_calc_finished_tls_sha256(
    mbedtls_ssl_context *ssl, unsigned char *buf, int from)
{
    int len = 12;
    const char *sender;
    unsigned char padbuf[32];
#if defined(MBEDTLS_USE_PSA_CRYPTO)
    size_t hash_size;
    psa_hash_operation_t sha256_psa = PSA_HASH_OPERATION_INIT;
    psa_status_t status;
#else
    mbedtls_sha256_context sha256;
#endif

    mbedtls_ssl_session *session = ssl->session_negotiate;
    if (!session) {
        session = ssl->session;
    }

    sender = (from == MBEDTLS_SSL_IS_CLIENT)
             ? "client finished"
             : "server finished";

#if defined(MBEDTLS_USE_PSA_CRYPTO)
    sha256_psa = psa_hash_operation_init();

    MBEDTLS_SSL_DEBUG_MSG(2, ("=> calc PSA finished tls sha256"));

    status = psa_hash_clone(&ssl->handshake->fin_sha256_psa, &sha256_psa);
    if (status != PSA_SUCCESS) {
        MBEDTLS_SSL_DEBUG_MSG(2, ("PSA hash clone failed"));
        return;
    }

    status = psa_hash_finish(&sha256_psa, padbuf, sizeof(padbuf), &hash_size);
    if (status != PSA_SUCCESS) {
        MBEDTLS_SSL_DEBUG_MSG(2, ("PSA hash finish failed"));
        return;
    }
    MBEDTLS_SSL_DEBUG_BUF(3, "PSA calculated padbuf", padbuf, 32);
#else

    mbedtls_sha256_init(&sha256);

    MBEDTLS_SSL_DEBUG_MSG(2, ("=> calc  finished tls sha256"));

    mbedtls_sha256_clone(&sha256, &ssl->handshake->fin_sha256);

    /*
     * TLSv1.2:
     *   hash = PRF( master, finished_label,
     *               Hash( handshake ) )[0.11]
     */

#if !defined(MBEDTLS_SHA256_ALT)
    MBEDTLS_SSL_DEBUG_BUF(4, "finished sha2 state", (unsigned char *)
                          sha256.state, sizeof(sha256.state));
#endif

    mbedtls_sha256_finish_ret(&sha256, padbuf);
    mbedtls_sha256_free(&sha256);
#endif /* MBEDTLS_USE_PSA_CRYPTO */

    ssl->handshake->tls_prf(session->master, 48, sender,
                            padbuf, 32, buf, len);

    MBEDTLS_SSL_DEBUG_BUF(3, "calc finished result", buf, len);

    mbedtls_platform_zeroize(padbuf, sizeof(padbuf));

    MBEDTLS_SSL_DEBUG_MSG(2, ("<= calc  finished"));
}
#endif /* MBEDTLS_SHA256_C */

#if defined(MBEDTLS_SHA512_C) && !defined(MBEDTLS_SHA512_NO_SHA384)

static void ssl_calc_finished_tls_sha384(
    mbedtls_ssl_context *ssl, unsigned char *buf, int from)
{
    int len = 12;
    const char *sender;
    unsigned char padbuf[48];
#if defined(MBEDTLS_USE_PSA_CRYPTO)
    size_t hash_size;
    psa_hash_operation_t sha384_psa = PSA_HASH_OPERATION_INIT;
    psa_status_t status;
#else
    mbedtls_sha512_context sha512;
#endif

    mbedtls_ssl_session *session = ssl->session_negotiate;
    if (!session) {
        session = ssl->session;
    }

    sender = (from == MBEDTLS_SSL_IS_CLIENT)
                ? "client finished"
                : "server finished";

#if defined(MBEDTLS_USE_PSA_CRYPTO)
    sha384_psa = psa_hash_operation_init();

    MBEDTLS_SSL_DEBUG_MSG(2, ("=> calc PSA finished tls sha384"));

    status = psa_hash_clone(&ssl->handshake->fin_sha384_psa, &sha384_psa);
    if (status != PSA_SUCCESS) {
        MBEDTLS_SSL_DEBUG_MSG(2, ("PSA hash clone failed"));
        return;
    }

    status = psa_hash_finish(&sha384_psa, padbuf, sizeof(padbuf), &hash_size);
    if (status != PSA_SUCCESS) {
        MBEDTLS_SSL_DEBUG_MSG(2, ("PSA hash finish failed"));
        return;
    }
    MBEDTLS_SSL_DEBUG_BUF(3, "PSA calculated padbuf", padbuf, 48);
#else
    mbedtls_sha512_init(&sha512);

    MBEDTLS_SSL_DEBUG_MSG(2, ("=> calc  finished tls sha384"));

    mbedtls_sha512_clone(&sha512, &ssl->handshake->fin_sha512);

    /*
     * TLSv1.2:
     *   hash = PRF( master, finished_label,
     *               Hash( handshake ) )[0.11]
     */

#if !defined(MBEDTLS_SHA512_ALT)
    MBEDTLS_SSL_DEBUG_BUF(4, "finished sha512 state", (unsigned char *)
                          sha512.state, sizeof(sha512.state));
#endif
    /* mbedtls_sha512_finish_ret's output parameter is declared as a
     * 64-byte buffer, but since we're using SHA-384, we know that the
     * output fits in 48 bytes. This is correct C, but GCC 11.1 warns
     * about it.
     */
#if defined(__GNUC__) && __GNUC__ >= 11
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif
    mbedtls_sha512_finish_ret(&sha512, padbuf);
#if defined(__GNUC__) && __GNUC__ >= 11
#pragma GCC diagnostic pop
#endif

    mbedtls_sha512_free(&sha512);
#endif

    ssl->handshake->tls_prf(session->master, 48, sender,
                            padbuf, 48, buf, len);

    MBEDTLS_SSL_DEBUG_BUF(3, "calc finished result", buf, len);

    mbedtls_platform_zeroize(padbuf, sizeof(padbuf));

    MBEDTLS_SSL_DEBUG_MSG(2, ("<= calc  finished"));
}
#endif /* MBEDTLS_SHA512_C && !MBEDTLS_SHA512_NO_SHA384 */
#endif /* MBEDTLS_SSL_PROTO_TLS1_2 */

void mbedtls_ssl_handshake_wrapup_free_hs_transform(mbedtls_ssl_context *ssl)
{
    MBEDTLS_SSL_DEBUG_MSG(3, ("=> handshake wrapup: final free"));

    /*
     * Free our handshake params
     */
    mbedtls_ssl_handshake_free(ssl);
    mbedtls_free(ssl->handshake);
    ssl->handshake = NULL;

    /*
     * Free the previous transform and switch in the current one
     */
    if (ssl->transform) {
        mbedtls_ssl_transform_free(ssl->transform);
        mbedtls_free(ssl->transform);
    }
    ssl->transform = ssl->transform_negotiate;
    ssl->transform_negotiate = NULL;

    MBEDTLS_SSL_DEBUG_MSG(3, ("<= handshake wrapup: final free"));
}

void mbedtls_ssl_handshake_wrapup(mbedtls_ssl_context *ssl)
{
    int resume = ssl->handshake->resume;

    MBEDTLS_SSL_DEBUG_MSG(3, ("=> handshake wrapup"));

#if defined(MBEDTLS_SSL_RENEGOTIATION)
    if (ssl->renego_status == MBEDTLS_SSL_RENEGOTIATION_IN_PROGRESS) {
        ssl->renego_status =  MBEDTLS_SSL_RENEGOTIATION_DONE;
        ssl->renego_records_seen = 0;
    }
#endif

    /*
     * Free the previous session and switch in the current one
     */
    if (ssl->session) {
#if defined(MBEDTLS_SSL_ENCRYPT_THEN_MAC)
        /* RFC 7366 3.1: keep the EtM state */
        ssl->session_negotiate->encrypt_then_mac =
            ssl->session->encrypt_then_mac;
#endif

        mbedtls_ssl_session_free(ssl->session);
        mbedtls_free(ssl->session);
    }
    ssl->session = ssl->session_negotiate;
    ssl->session_negotiate = NULL;

    /*
     * Add cache entry
     */
    if (ssl->conf->f_set_cache != NULL &&
        ssl->session->id_len != 0 &&
        resume == 0) {
        if (ssl->conf->f_set_cache(ssl->conf->p_cache, ssl->session) != 0) {
            MBEDTLS_SSL_DEBUG_MSG(1, ("cache did not store session"));
        }
    }

#if defined(MBEDTLS_SSL_PROTO_DTLS)
    if (ssl->conf->transport == MBEDTLS_SSL_TRANSPORT_DATAGRAM &&
        ssl->handshake->flight != NULL) {
        /* Cancel handshake timer */
        mbedtls_ssl_set_timer(ssl, 0);

        /* Keep last flight around in case we need to resend it:
         * we need the handshake and transform structures for that */
        MBEDTLS_SSL_DEBUG_MSG(3, ("skip freeing handshake and transform"));
    } else
#endif
    mbedtls_ssl_handshake_wrapup_free_hs_transform(ssl);

    ssl->state++;

    MBEDTLS_SSL_DEBUG_MSG(3, ("<= handshake wrapup"));
}

int mbedtls_ssl_write_finished(mbedtls_ssl_context *ssl)
{
    int ret, hash_len;

    MBEDTLS_SSL_DEBUG_MSG(2, ("=> write finished"));

    mbedtls_ssl_update_out_pointers(ssl, ssl->transform_negotiate);

    ssl->handshake->calc_finished(ssl, ssl->out_msg + 4, ssl->conf->endpoint);

    /*
     * RFC 5246 7.4.9 (Page 63) says 12 is the default length and ciphersuites
     * may define some other value. Currently (early 2016), no defined
     * ciphersuite does this (and this is unlikely to change as activity has
     * moved to TLS 1.3 now) so we can keep the hardcoded 12 here.
     */
    hash_len = (ssl->minor_ver == MBEDTLS_SSL_MINOR_VERSION_0) ? 36 : 12;

#if defined(MBEDTLS_SSL_RENEGOTIATION)
    ssl->verify_data_len = hash_len;
    memcpy(ssl->own_verify_data, ssl->out_msg + 4, hash_len);
#endif

    ssl->out_msglen  = 4 + hash_len;
    ssl->out_msgtype = MBEDTLS_SSL_MSG_HANDSHAKE;
    ssl->out_msg[0]  = MBEDTLS_SSL_HS_FINISHED;

    /*
     * In case of session resuming, invert the client and server
     * ChangeCipherSpec messages order.
     */
    if (ssl->handshake->resume != 0) {
#if defined(MBEDTLS_SSL_CLI_C)
        if (ssl->conf->endpoint == MBEDTLS_SSL_IS_CLIENT) {
            ssl->state = MBEDTLS_SSL_HANDSHAKE_WRAPUP;
        }
#endif
#if defined(MBEDTLS_SSL_SRV_C)
        if (ssl->conf->endpoint == MBEDTLS_SSL_IS_SERVER) {
            ssl->state = MBEDTLS_SSL_CLIENT_CHANGE_CIPHER_SPEC;
        }
#endif
    } else {
        ssl->state++;
    }

    /*
     * Switch to our negotiated transform and session parameters for outbound
     * data.
     */
    MBEDTLS_SSL_DEBUG_MSG(3, ("switching to new transform spec for outbound data"));

#if defined(MBEDTLS_SSL_PROTO_DTLS)
    if (ssl->conf->transport == MBEDTLS_SSL_TRANSPORT_DATAGRAM) {
        unsigned char i;

        /* Remember current epoch settings for resending */
        ssl->handshake->alt_transform_out = ssl->transform_out;
        memcpy(ssl->handshake->alt_out_ctr, ssl->cur_out_ctr, 8);

        /* Set sequence_number to zero */
        memset(ssl->cur_out_ctr + 2, 0, 6);

        /* Increment epoch */
        for (i = 2; i > 0; i--) {
            if (++ssl->cur_out_ctr[i - 1] != 0) {
                break;
            }
        }

        /* The loop goes to its end iff the counter is wrapping */
        if (i == 0) {
            MBEDTLS_SSL_DEBUG_MSG(1, ("DTLS epoch would wrap"));
            return MBEDTLS_ERR_SSL_COUNTER_WRAPPING;
        }
    } else
#endif /* MBEDTLS_SSL_PROTO_DTLS */
    memset(ssl->cur_out_ctr, 0, 8);

    ssl->transform_out = ssl->transform_negotiate;
    ssl->session_out = ssl->session_negotiate;

#if defined(MBEDTLS_SSL_HW_RECORD_ACCEL)
    if (mbedtls_ssl_hw_record_activate != NULL) {
        if ((ret = mbedtls_ssl_hw_record_activate(ssl, MBEDTLS_SSL_CHANNEL_OUTBOUND)) != 0) {
            MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_ssl_hw_record_activate", ret);
            return MBEDTLS_ERR_SSL_HW_ACCEL_FAILED;
        }
    }
#endif

#if defined(MBEDTLS_SSL_PROTO_DTLS)
    if (ssl->conf->transport == MBEDTLS_SSL_TRANSPORT_DATAGRAM) {
        mbedtls_ssl_send_flight_completed(ssl);
    }
#endif

    if ((ret = mbedtls_ssl_write_handshake_msg(ssl)) != 0) {
        MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_ssl_write_handshake_msg", ret);
        return ret;
    }

#if defined(MBEDTLS_SSL_PROTO_DTLS)
    if (ssl->conf->transport == MBEDTLS_SSL_TRANSPORT_DATAGRAM &&
        (ret = mbedtls_ssl_flight_transmit(ssl)) != 0) {
        MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_ssl_flight_transmit", ret);
        return ret;
    }
#endif

    MBEDTLS_SSL_DEBUG_MSG(2, ("<= write finished"));

    return 0;
}

#if defined(MBEDTLS_SSL_PROTO_SSL3)
#define SSL_MAX_HASH_LEN 36
#else
#define SSL_MAX_HASH_LEN 12
#endif

int mbedtls_ssl_parse_finished(mbedtls_ssl_context *ssl)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    unsigned int hash_len;
    unsigned char buf[SSL_MAX_HASH_LEN];

    MBEDTLS_SSL_DEBUG_MSG(2, ("=> parse finished"));

    /* There is currently no ciphersuite using another length with TLS 1.2 */
#if defined(MBEDTLS_SSL_PROTO_SSL3)
    if (ssl->minor_ver == MBEDTLS_SSL_MINOR_VERSION_0) {
        hash_len = 36;
    } else
#endif
    hash_len = 12;

    ssl->handshake->calc_finished(ssl, buf, ssl->conf->endpoint ^ 1);

    if ((ret = mbedtls_ssl_read_record(ssl, 1)) != 0) {
        MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_ssl_read_record", ret);
        goto exit;
    }

    if (ssl->in_msgtype != MBEDTLS_SSL_MSG_HANDSHAKE) {
        MBEDTLS_SSL_DEBUG_MSG(1, ("bad finished message"));
        mbedtls_ssl_send_alert_message(ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                       MBEDTLS_SSL_ALERT_MSG_UNEXPECTED_MESSAGE);
        ret = MBEDTLS_ERR_SSL_UNEXPECTED_MESSAGE;
        goto exit;
    }

    if (ssl->in_msg[0] != MBEDTLS_SSL_HS_FINISHED ||
        ssl->in_hslen  != mbedtls_ssl_hs_hdr_len(ssl) + hash_len) {
        MBEDTLS_SSL_DEBUG_MSG(1, ("bad finished message"));
        mbedtls_ssl_send_alert_message(ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                       MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR);
        ret = MBEDTLS_ERR_SSL_BAD_HS_FINISHED;
        goto exit;
    }

    if (mbedtls_ct_memcmp(ssl->in_msg + mbedtls_ssl_hs_hdr_len(ssl),
                          buf, hash_len) != 0) {
        MBEDTLS_SSL_DEBUG_MSG(1, ("bad finished message"));
        mbedtls_ssl_send_alert_message(ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                       MBEDTLS_SSL_ALERT_MSG_DECRYPT_ERROR);
        ret = MBEDTLS_ERR_SSL_BAD_HS_FINISHED;
        goto exit;
    }

#if defined(MBEDTLS_SSL_RENEGOTIATION)
    ssl->verify_data_len = hash_len;
    memcpy(ssl->peer_verify_data, buf, hash_len);
#endif

    if (ssl->handshake->resume != 0) {
#if defined(MBEDTLS_SSL_CLI_C)
        if (ssl->conf->endpoint == MBEDTLS_SSL_IS_CLIENT) {
            ssl->state = MBEDTLS_SSL_CLIENT_CHANGE_CIPHER_SPEC;
        }
#endif
#if defined(MBEDTLS_SSL_SRV_C)
        if (ssl->conf->endpoint == MBEDTLS_SSL_IS_SERVER) {
            ssl->state = MBEDTLS_SSL_HANDSHAKE_WRAPUP;
        }
#endif
    } else {
        ssl->state++;
    }

#if defined(MBEDTLS_SSL_PROTO_DTLS)
    if (ssl->conf->transport == MBEDTLS_SSL_TRANSPORT_DATAGRAM) {
        mbedtls_ssl_recv_flight_completed(ssl);
    }
#endif

    MBEDTLS_SSL_DEBUG_MSG(2, ("<= parse finished"));

exit:
    mbedtls_platform_zeroize(buf, hash_len);
    return ret;
}

static void ssl_handshake_params_init(mbedtls_ssl_handshake_params *handshake)
{
    memset(handshake, 0, sizeof(mbedtls_ssl_handshake_params));

#if defined(MBEDTLS_SSL_PROTO_SSL3) || defined(MBEDTLS_SSL_PROTO_TLS1) || \
    defined(MBEDTLS_SSL_PROTO_TLS1_1)
    mbedtls_md5_init(&handshake->fin_md5);
    mbedtls_sha1_init(&handshake->fin_sha1);
    mbedtls_md5_starts_ret(&handshake->fin_md5);
    mbedtls_sha1_starts_ret(&handshake->fin_sha1);
#endif
#if defined(MBEDTLS_SSL_PROTO_TLS1_2)
#if defined(MBEDTLS_SHA256_C)
#if defined(MBEDTLS_USE_PSA_CRYPTO)
    handshake->fin_sha256_psa = psa_hash_operation_init();
    psa_hash_setup(&handshake->fin_sha256_psa, PSA_ALG_SHA_256);
#else
    mbedtls_sha256_init(&handshake->fin_sha256);
    mbedtls_sha256_starts_ret(&handshake->fin_sha256, 0);
#endif
#endif
#if defined(MBEDTLS_SHA512_C) && !defined(MBEDTLS_SHA512_NO_SHA384)
#if defined(MBEDTLS_USE_PSA_CRYPTO)
    handshake->fin_sha384_psa = psa_hash_operation_init();
    psa_hash_setup(&handshake->fin_sha384_psa, PSA_ALG_SHA_384);
#else
    mbedtls_sha512_init(&handshake->fin_sha512);
    mbedtls_sha512_starts_ret(&handshake->fin_sha512, 1);
#endif
#endif
#endif /* MBEDTLS_SSL_PROTO_TLS1_2 */

    handshake->update_checksum = ssl_update_checksum_start;

#if defined(MBEDTLS_SSL_PROTO_TLS1_2) && \
    defined(MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED)
    mbedtls_ssl_sig_hash_set_init(&handshake->hash_algs);
#endif

#if defined(MBEDTLS_DHM_C)
    mbedtls_dhm_init(&handshake->dhm_ctx);
#endif
#if defined(MBEDTLS_ECDH_C)
    mbedtls_ecdh_init(&handshake->ecdh_ctx);
#endif
#if defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
    mbedtls_ecjpake_init(&handshake->ecjpake_ctx);
#if defined(MBEDTLS_SSL_CLI_C)
    handshake->ecjpake_cache = NULL;
    handshake->ecjpake_cache_len = 0;
#endif
#endif

#if defined(MBEDTLS_SSL_ECP_RESTARTABLE_ENABLED)
    mbedtls_x509_crt_restart_init(&handshake->ecrs_ctx);
#endif

#if defined(MBEDTLS_SSL_SERVER_NAME_INDICATION)
    handshake->sni_authmode = MBEDTLS_SSL_VERIFY_UNSET;
#endif

#if defined(MBEDTLS_X509_CRT_PARSE_C) && \
    !defined(MBEDTLS_SSL_KEEP_PEER_CERTIFICATE)
    mbedtls_pk_init(&handshake->peer_pubkey);
#endif
}

void mbedtls_ssl_transform_init(mbedtls_ssl_transform *transform)
{
    memset(transform, 0, sizeof(mbedtls_ssl_transform));

    mbedtls_cipher_init(&transform->cipher_ctx_enc);
    mbedtls_cipher_init(&transform->cipher_ctx_dec);

#if defined(MBEDTLS_SSL_SOME_MODES_USE_MAC)
    mbedtls_md_init(&transform->md_ctx_enc);
    mbedtls_md_init(&transform->md_ctx_dec);
#endif
}

void mbedtls_ssl_session_init(mbedtls_ssl_session *session)
{
    memset(session, 0, sizeof(mbedtls_ssl_session));
}

MBEDTLS_CHECK_RETURN_CRITICAL
static int ssl_handshake_init(mbedtls_ssl_context *ssl)
{
    /* Clear old handshake information if present */
    if (ssl->transform_negotiate) {
        mbedtls_ssl_transform_free(ssl->transform_negotiate);
    }
    if (ssl->session_negotiate) {
        mbedtls_ssl_session_free(ssl->session_negotiate);
    }
    if (ssl->handshake) {
        mbedtls_ssl_handshake_free(ssl);
    }

    /*
     * Either the pointers are now NULL or cleared properly and can be freed.
     * Now allocate missing structures.
     */
    if (ssl->transform_negotiate == NULL) {
        ssl->transform_negotiate = mbedtls_calloc(1, sizeof(mbedtls_ssl_transform));
    }

    if (ssl->session_negotiate == NULL) {
        ssl->session_negotiate = mbedtls_calloc(1, sizeof(mbedtls_ssl_session));
    }

    if (ssl->handshake == NULL) {
        ssl->handshake = mbedtls_calloc(1, sizeof(mbedtls_ssl_handshake_params));
    }
#if defined(MBEDTLS_SSL_VARIABLE_BUFFER_LENGTH)
    /* If the buffers are too small - reallocate */

    handle_buffer_resizing(ssl, 0, MBEDTLS_SSL_IN_BUFFER_LEN,
                           MBEDTLS_SSL_OUT_BUFFER_LEN);
#endif

    /* All pointers should exist and can be directly freed without issue */
    if (ssl->handshake == NULL ||
        ssl->transform_negotiate == NULL ||
        ssl->session_negotiate == NULL) {
        MBEDTLS_SSL_DEBUG_MSG(1, ("alloc() of ssl sub-contexts failed"));

        mbedtls_free(ssl->handshake);
        mbedtls_free(ssl->transform_negotiate);
        mbedtls_free(ssl->session_negotiate);

        ssl->handshake = NULL;
        ssl->transform_negotiate = NULL;
        ssl->session_negotiate = NULL;

        return MBEDTLS_ERR_SSL_ALLOC_FAILED;
    }

    /* Initialize structures */
    mbedtls_ssl_session_init(ssl->session_negotiate);
    mbedtls_ssl_transform_init(ssl->transform_negotiate);
    ssl_handshake_params_init(ssl->handshake);

#if defined(MBEDTLS_SSL_PROTO_DTLS)
    if (ssl->conf->transport == MBEDTLS_SSL_TRANSPORT_DATAGRAM) {
        ssl->handshake->alt_transform_out = ssl->transform_out;

        if (ssl->conf->endpoint == MBEDTLS_SSL_IS_CLIENT) {
            ssl->handshake->retransmit_state = MBEDTLS_SSL_RETRANS_PREPARING;
        } else {
            ssl->handshake->retransmit_state = MBEDTLS_SSL_RETRANS_WAITING;
        }

        mbedtls_ssl_set_timer(ssl, 0);
    }
#endif

    return 0;
}

#if defined(MBEDTLS_SSL_DTLS_HELLO_VERIFY) && defined(MBEDTLS_SSL_SRV_C)
/* Dummy cookie callbacks for defaults */
MBEDTLS_CHECK_RETURN_CRITICAL
static int ssl_cookie_write_dummy(void *ctx,
                                  unsigned char **p, unsigned char *end,
                                  const unsigned char *cli_id, size_t cli_id_len)
{
    ((void) ctx);
    ((void) p);
    ((void) end);
    ((void) cli_id);
    ((void) cli_id_len);

    return MBEDTLS_ERR_SSL_FEATURE_UNAVAILABLE;
}

MBEDTLS_CHECK_RETURN_CRITICAL
static int ssl_cookie_check_dummy(void *ctx,
                                  const unsigned char *cookie, size_t cookie_len,
                                  const unsigned char *cli_id, size_t cli_id_len)
{
    ((void) ctx);
    ((void) cookie);
    ((void) cookie_len);
    ((void) cli_id);
    ((void) cli_id_len);

    return MBEDTLS_ERR_SSL_FEATURE_UNAVAILABLE;
}
#endif /* MBEDTLS_SSL_DTLS_HELLO_VERIFY && MBEDTLS_SSL_SRV_C */

/*
 * Initialize an SSL context
 */
void mbedtls_ssl_init(mbedtls_ssl_context *ssl)
{
    memset(ssl, 0, sizeof(mbedtls_ssl_context));
}

/*
 * Setup an SSL context
 */

int mbedtls_ssl_setup(mbedtls_ssl_context *ssl,
                      const mbedtls_ssl_config *conf)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    size_t in_buf_len = MBEDTLS_SSL_IN_BUFFER_LEN;
    size_t out_buf_len = MBEDTLS_SSL_OUT_BUFFER_LEN;

    ssl->conf = conf;

    /*
     * Prepare base structures
     */

    /* Set to NULL in case of an error condition */
    ssl->out_buf = NULL;

#if defined(MBEDTLS_SSL_VARIABLE_BUFFER_LENGTH)
    ssl->in_buf_len = in_buf_len;
#endif
    ssl->in_buf = mbedtls_calloc(1, in_buf_len);
    if (ssl->in_buf == NULL) {
        MBEDTLS_SSL_DEBUG_MSG(1, ("alloc(%" MBEDTLS_PRINTF_SIZET " bytes) failed", in_buf_len));
        ret = MBEDTLS_ERR_SSL_ALLOC_FAILED;
        goto error;
    }

#if defined(MBEDTLS_SSL_VARIABLE_BUFFER_LENGTH)
    ssl->out_buf_len = out_buf_len;
#endif
    ssl->out_buf = mbedtls_calloc(1, out_buf_len);
    if (ssl->out_buf == NULL) {
        MBEDTLS_SSL_DEBUG_MSG(1, ("alloc(%" MBEDTLS_PRINTF_SIZET " bytes) failed", out_buf_len));
        ret = MBEDTLS_ERR_SSL_ALLOC_FAILED;
        goto error;
    }

    mbedtls_ssl_reset_in_out_pointers(ssl);

#if defined(MBEDTLS_SSL_DTLS_SRTP)
    memset(&ssl->dtls_srtp_info, 0, sizeof(ssl->dtls_srtp_info));
#endif

    if ((ret = ssl_handshake_init(ssl)) != 0) {
        goto error;
    }

    return 0;

error:
    mbedtls_free(ssl->in_buf);
    mbedtls_free(ssl->out_buf);

    ssl->conf = NULL;

#if defined(MBEDTLS_SSL_VARIABLE_BUFFER_LENGTH)
    ssl->in_buf_len = 0;
    ssl->out_buf_len = 0;
#endif
    ssl->in_buf = NULL;
    ssl->out_buf = NULL;

    ssl->in_hdr = NULL;
    ssl->in_ctr = NULL;
    ssl->in_len = NULL;
    ssl->in_iv = NULL;
    ssl->in_msg = NULL;

    ssl->out_hdr = NULL;
    ssl->out_ctr = NULL;
    ssl->out_len = NULL;
    ssl->out_iv = NULL;
    ssl->out_msg = NULL;

    return ret;
}

/*
 * Reset an initialized and used SSL context for re-use while retaining
 * all application-set variables, function pointers and data.
 *
 * If partial is non-zero, keep data in the input buffer and client ID.
 * (Use when a DTLS client reconnects from the same port.)
 */
int mbedtls_ssl_session_reset_int(mbedtls_ssl_context *ssl, int partial)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
#if defined(MBEDTLS_SSL_VARIABLE_BUFFER_LENGTH)
    size_t in_buf_len = ssl->in_buf_len;
    size_t out_buf_len = ssl->out_buf_len;
#else
    size_t in_buf_len = MBEDTLS_SSL_IN_BUFFER_LEN;
    size_t out_buf_len = MBEDTLS_SSL_OUT_BUFFER_LEN;
#endif

#if !defined(MBEDTLS_SSL_DTLS_CLIENT_PORT_REUSE) ||     \
    !defined(MBEDTLS_SSL_SRV_C)
    ((void) partial);
#endif

    ssl->state = MBEDTLS_SSL_HELLO_REQUEST;

    /* Cancel any possibly running timer */
    mbedtls_ssl_set_timer(ssl, 0);

#if defined(MBEDTLS_SSL_RENEGOTIATION)
    ssl->renego_status = MBEDTLS_SSL_INITIAL_HANDSHAKE;
    ssl->renego_records_seen = 0;

    ssl->verify_data_len = 0;
    memset(ssl->own_verify_data, 0, MBEDTLS_SSL_VERIFY_DATA_MAX_LEN);
    memset(ssl->peer_verify_data, 0, MBEDTLS_SSL_VERIFY_DATA_MAX_LEN);
#endif
    ssl->secure_renegotiation = MBEDTLS_SSL_LEGACY_RENEGOTIATION;

    ssl->in_offt = NULL;
    mbedtls_ssl_reset_in_out_pointers(ssl);

    ssl->in_msgtype = 0;
    ssl->in_msglen = 0;
#if defined(MBEDTLS_SSL_PROTO_DTLS)
    ssl->next_record_offset = 0;
    ssl->in_epoch = 0;
#endif
#if defined(MBEDTLS_SSL_DTLS_ANTI_REPLAY)
    mbedtls_ssl_dtls_replay_reset(ssl);
#endif

    ssl->in_hslen = 0;
    ssl->nb_zero = 0;

    ssl->keep_current_message = 0;

    ssl->out_msgtype = 0;
    ssl->out_msglen = 0;
    ssl->out_left = 0;
#if defined(MBEDTLS_SSL_CBC_RECORD_SPLITTING)
    if (ssl->split_done != MBEDTLS_SSL_CBC_RECORD_SPLITTING_DISABLED) {
        ssl->split_done = 0;
    }
#endif

    memset(ssl->cur_out_ctr, 0, sizeof(ssl->cur_out_ctr));

    ssl->transform_in = NULL;
    ssl->transform_out = NULL;

    ssl->session_in = NULL;
    ssl->session_out = NULL;

    memset(ssl->out_buf, 0, out_buf_len);

    int clear_in_buf = 1;
#if defined(MBEDTLS_SSL_DTLS_CLIENT_PORT_REUSE) && defined(MBEDTLS_SSL_SRV_C)
    if (partial != 0) {
        clear_in_buf = 0;
    }
#endif /* MBEDTLS_SSL_DTLS_CLIENT_PORT_REUSE && MBEDTLS_SSL_SRV_C */
    if (clear_in_buf) {
        ssl->in_left = 0;
        memset(ssl->in_buf, 0, in_buf_len);
    }

#if defined(MBEDTLS_SSL_HW_RECORD_ACCEL)
    if (mbedtls_ssl_hw_record_reset != NULL) {
        MBEDTLS_SSL_DEBUG_MSG(2, ("going for mbedtls_ssl_hw_record_reset()"));
        if ((ret = mbedtls_ssl_hw_record_reset(ssl)) != 0) {
            MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_ssl_hw_record_reset", ret);
            return MBEDTLS_ERR_SSL_HW_ACCEL_FAILED;
        }
    }
#endif

    if (ssl->transform) {
        mbedtls_ssl_transform_free(ssl->transform);
        mbedtls_free(ssl->transform);
        ssl->transform = NULL;
    }

    if (ssl->session) {
        mbedtls_ssl_session_free(ssl->session);
        mbedtls_free(ssl->session);
        ssl->session = NULL;
    }

#if defined(MBEDTLS_SSL_ALPN)
    ssl->alpn_chosen = NULL;
#endif

#if defined(MBEDTLS_SSL_DTLS_HELLO_VERIFY) && defined(MBEDTLS_SSL_SRV_C)
    int free_cli_id = 1;
#if defined(MBEDTLS_SSL_DTLS_CLIENT_PORT_REUSE)
    if (partial != 0) {
        free_cli_id = 0;
    }
#endif
    if (free_cli_id) {
        mbedtls_free(ssl->cli_id);
        ssl->cli_id = NULL;
        ssl->cli_id_len = 0;
    }
#endif

    if ((ret = ssl_handshake_init(ssl)) != 0) {
        return ret;
    }

    return 0;
}

/*
 * Reset an initialized and used SSL context for re-use while retaining
 * all application-set variables, function pointers and data.
 */
int mbedtls_ssl_session_reset(mbedtls_ssl_context *ssl)
{
    return mbedtls_ssl_session_reset_int(ssl, 0);
}

/*
 * SSL set accessors
 */
void mbedtls_ssl_conf_endpoint(mbedtls_ssl_config *conf, int endpoint)
{
    conf->endpoint   = endpoint;
}

void mbedtls_ssl_conf_transport(mbedtls_ssl_config *conf, int transport)
{
    conf->transport = transport;
}

#if defined(MBEDTLS_SSL_DTLS_ANTI_REPLAY)
void mbedtls_ssl_conf_dtls_anti_replay(mbedtls_ssl_config *conf, char mode)
{
    conf->anti_replay = mode;
}
#endif

#if defined(MBEDTLS_SSL_DTLS_BADMAC_LIMIT)
void mbedtls_ssl_conf_dtls_badmac_limit(mbedtls_ssl_config *conf, unsigned limit)
{
    conf->badmac_limit = limit;
}
#endif

#if defined(MBEDTLS_SSL_PROTO_DTLS)

void mbedtls_ssl_set_datagram_packing(mbedtls_ssl_context *ssl,
                                      unsigned allow_packing)
{
    ssl->disable_datagram_packing = !allow_packing;
}

void mbedtls_ssl_conf_handshake_timeout(mbedtls_ssl_config *conf,
                                        uint32_t min, uint32_t max)
{
    conf->hs_timeout_min = min;
    conf->hs_timeout_max = max;
}
#endif

void mbedtls_ssl_conf_authmode(mbedtls_ssl_config *conf, int authmode)
{
    conf->authmode   = authmode;
}

#if defined(MBEDTLS_X509_CRT_PARSE_C)
void mbedtls_ssl_conf_verify(mbedtls_ssl_config *conf,
                             int (*f_vrfy)(void *, mbedtls_x509_crt *, int, uint32_t *),
                             void *p_vrfy)
{
    conf->f_vrfy      = f_vrfy;
    conf->p_vrfy      = p_vrfy;
}
#endif /* MBEDTLS_X509_CRT_PARSE_C */

void mbedtls_ssl_conf_rng(mbedtls_ssl_config *conf,
                          int (*f_rng)(void *, unsigned char *, size_t),
                          void *p_rng)
{
    conf->f_rng      = f_rng;
    conf->p_rng      = p_rng;
}

void mbedtls_ssl_conf_dbg(mbedtls_ssl_config *conf,
                          void (*f_dbg)(void *, int, const char *, int, const char *),
                          void  *p_dbg)
{
    conf->f_dbg      = f_dbg;
    conf->p_dbg      = p_dbg;
}

void mbedtls_ssl_set_bio(mbedtls_ssl_context *ssl,
                         void *p_bio,
                         mbedtls_ssl_send_t *f_send,
                         mbedtls_ssl_recv_t *f_recv,
                         mbedtls_ssl_recv_timeout_t *f_recv_timeout)
{
    ssl->p_bio          = p_bio;
    ssl->f_send         = f_send;
    ssl->f_recv         = f_recv;
    ssl->f_recv_timeout = f_recv_timeout;
}

#if defined(MBEDTLS_SSL_PROTO_DTLS)
void mbedtls_ssl_set_mtu(mbedtls_ssl_context *ssl, uint16_t mtu)
{
    ssl->mtu = mtu;
}
#endif

void mbedtls_ssl_conf_read_timeout(mbedtls_ssl_config *conf, uint32_t timeout)
{
    conf->read_timeout   = timeout;
}

void mbedtls_ssl_set_timer_cb(mbedtls_ssl_context *ssl,
                              void *p_timer,
                              mbedtls_ssl_set_timer_t *f_set_timer,
                              mbedtls_ssl_get_timer_t *f_get_timer)
{
    ssl->p_timer        = p_timer;
    ssl->f_set_timer    = f_set_timer;
    ssl->f_get_timer    = f_get_timer;

    /* Make sure we start with no timer running */
    mbedtls_ssl_set_timer(ssl, 0);
}

#if defined(MBEDTLS_SSL_SRV_C)
void mbedtls_ssl_conf_session_cache(mbedtls_ssl_config *conf,
                                    void *p_cache,
                                    int (*f_get_cache)(void *, mbedtls_ssl_session *),
                                    int (*f_set_cache)(void *, const mbedtls_ssl_session *))
{
    conf->p_cache = p_cache;
    conf->f_get_cache = f_get_cache;
    conf->f_set_cache = f_set_cache;
}
#endif /* MBEDTLS_SSL_SRV_C */

#if defined(MBEDTLS_SSL_CLI_C)
int mbedtls_ssl_set_session(mbedtls_ssl_context *ssl, const mbedtls_ssl_session *session)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;

    if (ssl == NULL ||
        session == NULL ||
        ssl->session_negotiate == NULL ||
        ssl->conf->endpoint != MBEDTLS_SSL_IS_CLIENT) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    if ((ret = mbedtls_ssl_session_copy(ssl->session_negotiate,
                                        session)) != 0) {
        return ret;
    }

    ssl->handshake->resume = 1;

    return 0;
}
#endif /* MBEDTLS_SSL_CLI_C */

void mbedtls_ssl_conf_ciphersuites(mbedtls_ssl_config *conf,
                                   const int *ciphersuites)
{
    conf->ciphersuite_list[MBEDTLS_SSL_MINOR_VERSION_0] = ciphersuites;
    conf->ciphersuite_list[MBEDTLS_SSL_MINOR_VERSION_1] = ciphersuites;
    conf->ciphersuite_list[MBEDTLS_SSL_MINOR_VERSION_2] = ciphersuites;
    conf->ciphersuite_list[MBEDTLS_SSL_MINOR_VERSION_3] = ciphersuites;
}

void mbedtls_ssl_conf_ciphersuites_for_version(mbedtls_ssl_config *conf,
                                               const int *ciphersuites,
                                               int major, int minor)
{
    if (major != MBEDTLS_SSL_MAJOR_VERSION_3) {
        return;
    }

    if (minor < MBEDTLS_SSL_MINOR_VERSION_0 || minor > MBEDTLS_SSL_MINOR_VERSION_3) {
        return;
    }

    conf->ciphersuite_list[minor] = ciphersuites;
}

#if defined(MBEDTLS_X509_CRT_PARSE_C)
void mbedtls_ssl_conf_cert_profile(mbedtls_ssl_config *conf,
                                   const mbedtls_x509_crt_profile *profile)
{
    conf->cert_profile = profile;
}

/* Append a new keycert entry to a (possibly empty) list */
MBEDTLS_CHECK_RETURN_CRITICAL
static int ssl_append_key_cert(mbedtls_ssl_key_cert **head,
                               mbedtls_x509_crt *cert,
                               mbedtls_pk_context *key)
{
    mbedtls_ssl_key_cert *new_cert;

    new_cert = mbedtls_calloc(1, sizeof(mbedtls_ssl_key_cert));
    if (new_cert == NULL) {
        return MBEDTLS_ERR_SSL_ALLOC_FAILED;
    }

    new_cert->cert = cert;
    new_cert->key  = key;
    new_cert->next = NULL;

    /* Update head is the list was null, else add to the end */
    if (*head == NULL) {
        *head = new_cert;
    } else {
        mbedtls_ssl_key_cert *cur = *head;
        while (cur->next != NULL) {
            cur = cur->next;
        }
        cur->next = new_cert;
    }

    return 0;
}

int mbedtls_ssl_conf_own_cert(mbedtls_ssl_config *conf,
                              mbedtls_x509_crt *own_cert,
                              mbedtls_pk_context *pk_key)
{
    return ssl_append_key_cert(&conf->key_cert, own_cert, pk_key);
}

void mbedtls_ssl_conf_ca_chain(mbedtls_ssl_config *conf,
                               mbedtls_x509_crt *ca_chain,
                               mbedtls_x509_crl *ca_crl)
{
    conf->ca_chain   = ca_chain;
    conf->ca_crl     = ca_crl;

#if defined(MBEDTLS_X509_TRUSTED_CERTIFICATE_CALLBACK)
    /* mbedtls_ssl_conf_ca_chain() and mbedtls_ssl_conf_ca_cb()
     * cannot be used together. */
    conf->f_ca_cb = NULL;
    conf->p_ca_cb = NULL;
#endif /* MBEDTLS_X509_TRUSTED_CERTIFICATE_CALLBACK */
}

#if defined(MBEDTLS_X509_TRUSTED_CERTIFICATE_CALLBACK)
void mbedtls_ssl_conf_ca_cb(mbedtls_ssl_config *conf,
                            mbedtls_x509_crt_ca_cb_t f_ca_cb,
                            void *p_ca_cb)
{
    conf->f_ca_cb = f_ca_cb;
    conf->p_ca_cb = p_ca_cb;

    /* mbedtls_ssl_conf_ca_chain() and mbedtls_ssl_conf_ca_cb()
     * cannot be used together. */
    conf->ca_chain   = NULL;
    conf->ca_crl     = NULL;
}
#endif /* MBEDTLS_X509_TRUSTED_CERTIFICATE_CALLBACK */
#endif /* MBEDTLS_X509_CRT_PARSE_C */

#if defined(MBEDTLS_SSL_SERVER_NAME_INDICATION)
int mbedtls_ssl_set_hs_own_cert(mbedtls_ssl_context *ssl,
                                mbedtls_x509_crt *own_cert,
                                mbedtls_pk_context *pk_key)
{
    return ssl_append_key_cert(&ssl->handshake->sni_key_cert,
                               own_cert, pk_key);
}

void mbedtls_ssl_set_hs_ca_chain(mbedtls_ssl_context *ssl,
                                 mbedtls_x509_crt *ca_chain,
                                 mbedtls_x509_crl *ca_crl)
{
    ssl->handshake->sni_ca_chain   = ca_chain;
    ssl->handshake->sni_ca_crl     = ca_crl;
}

void mbedtls_ssl_set_hs_authmode(mbedtls_ssl_context *ssl,
                                 int authmode)
{
    ssl->handshake->sni_authmode = authmode;
}
#endif /* MBEDTLS_SSL_SERVER_NAME_INDICATION */

#if defined(MBEDTLS_X509_CRT_PARSE_C)
void mbedtls_ssl_set_verify(mbedtls_ssl_context *ssl,
                            int (*f_vrfy)(void *, mbedtls_x509_crt *, int, uint32_t *),
                            void *p_vrfy)
{
    ssl->f_vrfy = f_vrfy;
    ssl->p_vrfy = p_vrfy;
}
#endif

#if defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
/*
 * Set EC J-PAKE password for current handshake
 */
int mbedtls_ssl_set_hs_ecjpake_password(mbedtls_ssl_context *ssl,
                                        const unsigned char *pw,
                                        size_t pw_len)
{
    mbedtls_ecjpake_role role;

    if (ssl->handshake == NULL || ssl->conf == NULL) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    if (ssl->conf->endpoint == MBEDTLS_SSL_IS_SERVER) {
        role = MBEDTLS_ECJPAKE_SERVER;
    } else {
        role = MBEDTLS_ECJPAKE_CLIENT;
    }

    return mbedtls_ecjpake_setup(&ssl->handshake->ecjpake_ctx,
                                 role,
                                 MBEDTLS_MD_SHA256,
                                 MBEDTLS_ECP_DP_SECP256R1,
                                 pw, pw_len);
}
#endif /* MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED */

#if defined(MBEDTLS_KEY_EXCHANGE_SOME_PSK_ENABLED)

static void ssl_conf_remove_psk(mbedtls_ssl_config *conf)
{
    /* Remove reference to existing PSK, if any. */
#if defined(MBEDTLS_USE_PSA_CRYPTO)
    if (!mbedtls_svc_key_id_is_null(conf->psk_opaque)) {
        /* The maintenance of the PSK key slot is the
         * user's responsibility. */
        conf->psk_opaque = MBEDTLS_SVC_KEY_ID_INIT;
    }
    /* This and the following branch should never
     * be taken simultaneously as we maintain the
     * invariant that raw and opaque PSKs are never
     * configured simultaneously. As a safeguard,
     * though, `else` is omitted here. */
#endif /* MBEDTLS_USE_PSA_CRYPTO */
    if (conf->psk != NULL) {
        mbedtls_platform_zeroize(conf->psk, conf->psk_len);

        mbedtls_free(conf->psk);
        conf->psk = NULL;
        conf->psk_len = 0;
    }

    /* Remove reference to PSK identity, if any. */
    if (conf->psk_identity != NULL) {
        mbedtls_free(conf->psk_identity);
        conf->psk_identity = NULL;
        conf->psk_identity_len = 0;
    }
}

/* This function assumes that PSK identity in the SSL config is unset.
 * It checks that the provided identity is well-formed and attempts
 * to make a copy of it in the SSL config.
 * On failure, the PSK identity in the config remains unset. */
MBEDTLS_CHECK_RETURN_CRITICAL
static int ssl_conf_set_psk_identity(mbedtls_ssl_config *conf,
                                     unsigned char const *psk_identity,
                                     size_t psk_identity_len)
{
    /* Identity len will be encoded on two bytes */
    if (psk_identity               == NULL ||
        (psk_identity_len >> 16) != 0    ||
        psk_identity_len > MBEDTLS_SSL_OUT_CONTENT_LEN) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    conf->psk_identity = mbedtls_calloc(1, psk_identity_len);
    if (conf->psk_identity == NULL) {
        return MBEDTLS_ERR_SSL_ALLOC_FAILED;
    }

    conf->psk_identity_len = psk_identity_len;
    memcpy(conf->psk_identity, psk_identity, conf->psk_identity_len);

    return 0;
}

int mbedtls_ssl_conf_psk(mbedtls_ssl_config *conf,
                         const unsigned char *psk, size_t psk_len,
                         const unsigned char *psk_identity, size_t psk_identity_len)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    /* Remove opaque/raw PSK + PSK Identity */
    ssl_conf_remove_psk(conf);

    /* Check and set raw PSK */
    if (psk == NULL) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }
    if (psk_len == 0) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }
    if (psk_len > MBEDTLS_PSK_MAX_LEN) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    if ((conf->psk = mbedtls_calloc(1, psk_len)) == NULL) {
        return MBEDTLS_ERR_SSL_ALLOC_FAILED;
    }
    conf->psk_len = psk_len;
    memcpy(conf->psk, psk, conf->psk_len);

    /* Check and set PSK Identity */
    ret = ssl_conf_set_psk_identity(conf, psk_identity, psk_identity_len);
    if (ret != 0) {
        ssl_conf_remove_psk(conf);
    }

    return ret;
}

static void ssl_remove_psk(mbedtls_ssl_context *ssl)
{
#if defined(MBEDTLS_USE_PSA_CRYPTO)
    if (!mbedtls_svc_key_id_is_null(ssl->handshake->psk_opaque)) {
        ssl->handshake->psk_opaque = MBEDTLS_SVC_KEY_ID_INIT;
    } else
#endif /* MBEDTLS_USE_PSA_CRYPTO */
    if (ssl->handshake->psk != NULL) {
        mbedtls_platform_zeroize(ssl->handshake->psk,
                                 ssl->handshake->psk_len);
        mbedtls_free(ssl->handshake->psk);
        ssl->handshake->psk_len = 0;
    }
}

int mbedtls_ssl_set_hs_psk(mbedtls_ssl_context *ssl,
                           const unsigned char *psk, size_t psk_len)
{
    if (psk == NULL || ssl->handshake == NULL) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    if (psk_len > MBEDTLS_PSK_MAX_LEN) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    ssl_remove_psk(ssl);

    if ((ssl->handshake->psk = mbedtls_calloc(1, psk_len)) == NULL) {
        return MBEDTLS_ERR_SSL_ALLOC_FAILED;
    }

    ssl->handshake->psk_len = psk_len;
    memcpy(ssl->handshake->psk, psk, ssl->handshake->psk_len);

    return 0;
}

#if defined(MBEDTLS_USE_PSA_CRYPTO)
int mbedtls_ssl_conf_psk_opaque(mbedtls_ssl_config *conf,
                                psa_key_id_t psk,
                                const unsigned char *psk_identity,
                                size_t psk_identity_len)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    /* Clear opaque/raw PSK + PSK Identity, if present. */
    ssl_conf_remove_psk(conf);

    /* Check and set opaque PSK */
    if (mbedtls_svc_key_id_is_null(psk)) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }
    conf->psk_opaque = psk;

    /* Check and set PSK Identity */
    ret = ssl_conf_set_psk_identity(conf, psk_identity,
                                    psk_identity_len);
    if (ret != 0) {
        ssl_conf_remove_psk(conf);
    }

    return ret;
}

int mbedtls_ssl_set_hs_psk_opaque(mbedtls_ssl_context *ssl,
                                  psa_key_id_t psk)
{
    if ((mbedtls_svc_key_id_is_null(psk)) ||
        (ssl->handshake == NULL)) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    ssl_remove_psk(ssl);
    ssl->handshake->psk_opaque = psk;
    return 0;
}
#endif /* MBEDTLS_USE_PSA_CRYPTO */

void mbedtls_ssl_conf_psk_cb(mbedtls_ssl_config *conf,
                             int (*f_psk)(void *, mbedtls_ssl_context *, const unsigned char *,
                                          size_t),
                             void *p_psk)
{
    conf->f_psk = f_psk;
    conf->p_psk = p_psk;
}
#endif /* MBEDTLS_KEY_EXCHANGE_SOME_PSK_ENABLED */

#if defined(MBEDTLS_DHM_C) && defined(MBEDTLS_SSL_SRV_C)

#if !defined(MBEDTLS_DEPRECATED_REMOVED)
int mbedtls_ssl_conf_dh_param(mbedtls_ssl_config *conf, const char *dhm_P, const char *dhm_G)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;

    if ((ret = mbedtls_mpi_read_string(&conf->dhm_P, 16, dhm_P)) != 0 ||
        (ret = mbedtls_mpi_read_string(&conf->dhm_G, 16, dhm_G)) != 0) {
        mbedtls_mpi_free(&conf->dhm_P);
        mbedtls_mpi_free(&conf->dhm_G);
        return ret;
    }

    return 0;
}
#endif /* MBEDTLS_DEPRECATED_REMOVED */

int mbedtls_ssl_conf_dh_param_bin(mbedtls_ssl_config *conf,
                                  const unsigned char *dhm_P, size_t P_len,
                                  const unsigned char *dhm_G, size_t G_len)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;

    mbedtls_mpi_free(&conf->dhm_P);
    mbedtls_mpi_free(&conf->dhm_G);

    if ((ret = mbedtls_mpi_read_binary(&conf->dhm_P, dhm_P, P_len)) != 0 ||
        (ret = mbedtls_mpi_read_binary(&conf->dhm_G, dhm_G, G_len)) != 0) {
        mbedtls_mpi_free(&conf->dhm_P);
        mbedtls_mpi_free(&conf->dhm_G);
        return ret;
    }

    return 0;
}

int mbedtls_ssl_conf_dh_param_ctx(mbedtls_ssl_config *conf, mbedtls_dhm_context *dhm_ctx)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;

    mbedtls_mpi_free(&conf->dhm_P);
    mbedtls_mpi_free(&conf->dhm_G);

    if ((ret = mbedtls_mpi_copy(&conf->dhm_P, &dhm_ctx->P)) != 0 ||
        (ret = mbedtls_mpi_copy(&conf->dhm_G, &dhm_ctx->G)) != 0) {
        mbedtls_mpi_free(&conf->dhm_P);
        mbedtls_mpi_free(&conf->dhm_G);
        return ret;
    }

    return 0;
}
#endif /* MBEDTLS_DHM_C && MBEDTLS_SSL_SRV_C */

#if defined(MBEDTLS_DHM_C) && defined(MBEDTLS_SSL_CLI_C)
/*
 * Set the minimum length for Diffie-Hellman parameters
 */
void mbedtls_ssl_conf_dhm_min_bitlen(mbedtls_ssl_config *conf,
                                     unsigned int bitlen)
{
    conf->dhm_min_bitlen = bitlen;
}
#endif /* MBEDTLS_DHM_C && MBEDTLS_SSL_CLI_C */

#if defined(MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED)
/*
 * Set allowed/preferred hashes for handshake signatures
 */
void mbedtls_ssl_conf_sig_hashes(mbedtls_ssl_config *conf,
                                 const int *hashes)
{
    conf->sig_hashes = hashes;
}
#endif /* MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED */

#if defined(MBEDTLS_ECP_C)
/*
 * Set the allowed elliptic curves
 */
void mbedtls_ssl_conf_curves(mbedtls_ssl_config *conf,
                             const mbedtls_ecp_group_id *curve_list)
{
    conf->curve_list = curve_list;
}
#endif /* MBEDTLS_ECP_C */

#if defined(MBEDTLS_X509_CRT_PARSE_C)
int mbedtls_ssl_set_hostname(mbedtls_ssl_context *ssl, const char *hostname)
{
    /* Initialize to suppress unnecessary compiler warning */
    size_t hostname_len = 0;

    /* Check if new hostname is valid before
     * making any change to current one */
    if (hostname != NULL) {
        hostname_len = strlen(hostname);

        if (hostname_len > MBEDTLS_SSL_MAX_HOST_NAME_LEN) {
            return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
        }
    }

    /* Now it's clear that we will overwrite the old hostname,
     * so we can free it safely */

    if (ssl->hostname != NULL) {
        mbedtls_platform_zeroize(ssl->hostname, strlen(ssl->hostname));
        mbedtls_free(ssl->hostname);
    }

    /* Passing NULL as hostname shall clear the old one */

    if (hostname == NULL) {
        ssl->hostname = NULL;
    } else {
        ssl->hostname = mbedtls_calloc(1, hostname_len + 1);
        if (ssl->hostname == NULL) {
            return MBEDTLS_ERR_SSL_ALLOC_FAILED;
        }

        memcpy(ssl->hostname, hostname, hostname_len);

        ssl->hostname[hostname_len] = '\0';
    }

    return 0;
}
#endif /* MBEDTLS_X509_CRT_PARSE_C */

#if defined(MBEDTLS_SSL_SERVER_NAME_INDICATION)
void mbedtls_ssl_conf_sni(mbedtls_ssl_config *conf,
                          int (*f_sni)(void *, mbedtls_ssl_context *,
                                       const unsigned char *, size_t),
                          void *p_sni)
{
    conf->f_sni = f_sni;
    conf->p_sni = p_sni;
}
#endif /* MBEDTLS_SSL_SERVER_NAME_INDICATION */

#if defined(MBEDTLS_SSL_ALPN)
int mbedtls_ssl_conf_alpn_protocols(mbedtls_ssl_config *conf, const char **protos)
{
    size_t cur_len, tot_len;
    const char **p;

    /*
     * RFC 7301 3.1: "Empty strings MUST NOT be included and byte strings
     * MUST NOT be truncated."
     * We check lengths now rather than later.
     */
    tot_len = 0;
    for (p = protos; *p != NULL; p++) {
        cur_len = strlen(*p);
        tot_len += cur_len;

        if ((cur_len == 0) ||
            (cur_len > MBEDTLS_SSL_MAX_ALPN_NAME_LEN) ||
            (tot_len > MBEDTLS_SSL_MAX_ALPN_LIST_LEN)) {
            return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
        }
    }

    conf->alpn_list = protos;

    return 0;
}

const char *mbedtls_ssl_get_alpn_protocol(const mbedtls_ssl_context *ssl)
{
    return ssl->alpn_chosen;
}
#endif /* MBEDTLS_SSL_ALPN */

#if defined(MBEDTLS_SSL_DTLS_SRTP)
void mbedtls_ssl_conf_srtp_mki_value_supported(mbedtls_ssl_config *conf,
                                               int support_mki_value)
{
    conf->dtls_srtp_mki_support = support_mki_value;
}

int mbedtls_ssl_dtls_srtp_set_mki_value(mbedtls_ssl_context *ssl,
                                        unsigned char *mki_value,
                                        uint16_t mki_len)
{
    if (mki_len > MBEDTLS_TLS_SRTP_MAX_MKI_LENGTH) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    if (ssl->conf->dtls_srtp_mki_support == MBEDTLS_SSL_DTLS_SRTP_MKI_UNSUPPORTED) {
        return MBEDTLS_ERR_SSL_FEATURE_UNAVAILABLE;
    }

    memcpy(ssl->dtls_srtp_info.mki_value, mki_value, mki_len);
    ssl->dtls_srtp_info.mki_len = mki_len;
    return 0;
}

int mbedtls_ssl_conf_dtls_srtp_protection_profiles(mbedtls_ssl_config *conf,
                                                   const mbedtls_ssl_srtp_profile *profiles)
{
    const mbedtls_ssl_srtp_profile *p;
    size_t list_size = 0;

    /* check the profiles list: all entry must be valid,
     * its size cannot be more than the total number of supported profiles, currently 4 */
    for (p = profiles; *p != MBEDTLS_TLS_SRTP_UNSET &&
         list_size <= MBEDTLS_TLS_SRTP_MAX_PROFILE_LIST_LENGTH;
         p++) {
        if (mbedtls_ssl_check_srtp_profile_value(*p) != MBEDTLS_TLS_SRTP_UNSET) {
            list_size++;
        } else {
            /* unsupported value, stop parsing and set the size to an error value */
            list_size = MBEDTLS_TLS_SRTP_MAX_PROFILE_LIST_LENGTH + 1;
        }
    }

    if (list_size > MBEDTLS_TLS_SRTP_MAX_PROFILE_LIST_LENGTH) {
        conf->dtls_srtp_profile_list = NULL;
        conf->dtls_srtp_profile_list_len = 0;
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    conf->dtls_srtp_profile_list = profiles;
    conf->dtls_srtp_profile_list_len = list_size;

    return 0;
}

void mbedtls_ssl_get_dtls_srtp_negotiation_result(const mbedtls_ssl_context *ssl,
                                                  mbedtls_dtls_srtp_info *dtls_srtp_info)
{
    dtls_srtp_info->chosen_dtls_srtp_profile = ssl->dtls_srtp_info.chosen_dtls_srtp_profile;
    /* do not copy the mki value if there is no chosen profile */
    if (dtls_srtp_info->chosen_dtls_srtp_profile == MBEDTLS_TLS_SRTP_UNSET) {
        dtls_srtp_info->mki_len = 0;
    } else {
        dtls_srtp_info->mki_len = ssl->dtls_srtp_info.mki_len;
        memcpy(dtls_srtp_info->mki_value, ssl->dtls_srtp_info.mki_value,
               ssl->dtls_srtp_info.mki_len);
    }
}
#endif /* MBEDTLS_SSL_DTLS_SRTP */

void mbedtls_ssl_conf_max_version(mbedtls_ssl_config *conf, int major, int minor)
{
    conf->max_major_ver = major;
    conf->max_minor_ver = minor;
}

void mbedtls_ssl_conf_min_version(mbedtls_ssl_config *conf, int major, int minor)
{
    conf->min_major_ver = major;
    conf->min_minor_ver = minor;
}

#if defined(MBEDTLS_SSL_FALLBACK_SCSV) && defined(MBEDTLS_SSL_CLI_C)
void mbedtls_ssl_conf_fallback(mbedtls_ssl_config *conf, char fallback)
{
    conf->fallback = fallback;
}
#endif

#if defined(MBEDTLS_SSL_SRV_C)
void mbedtls_ssl_conf_cert_req_ca_list(mbedtls_ssl_config *conf,
                                       char cert_req_ca_list)
{
    conf->cert_req_ca_list = cert_req_ca_list;
}
#endif

#if defined(MBEDTLS_SSL_ENCRYPT_THEN_MAC)
void mbedtls_ssl_conf_encrypt_then_mac(mbedtls_ssl_config *conf, char etm)
{
    conf->encrypt_then_mac = etm;
}
#endif

#if defined(MBEDTLS_SSL_EXTENDED_MASTER_SECRET)
void mbedtls_ssl_conf_extended_master_secret(mbedtls_ssl_config *conf, char ems)
{
    conf->extended_ms = ems;
}
#endif

#if defined(MBEDTLS_ARC4_C)
void mbedtls_ssl_conf_arc4_support(mbedtls_ssl_config *conf, char arc4)
{
    conf->arc4_disabled = arc4;
}
#endif

#if defined(MBEDTLS_SSL_MAX_FRAGMENT_LENGTH)
int mbedtls_ssl_conf_max_frag_len(mbedtls_ssl_config *conf, unsigned char mfl_code)
{
    if (mfl_code >= MBEDTLS_SSL_MAX_FRAG_LEN_INVALID ||
        ssl_mfl_code_to_length(mfl_code) > MBEDTLS_TLS_EXT_ADV_CONTENT_LEN) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    conf->mfl_code = mfl_code;

    return 0;
}
#endif /* MBEDTLS_SSL_MAX_FRAGMENT_LENGTH */

#if defined(MBEDTLS_SSL_TRUNCATED_HMAC)
void mbedtls_ssl_conf_truncated_hmac(mbedtls_ssl_config *conf, int truncate)
{
    conf->trunc_hmac = truncate;
}
#endif /* MBEDTLS_SSL_TRUNCATED_HMAC */

#if defined(MBEDTLS_SSL_CBC_RECORD_SPLITTING)
void mbedtls_ssl_conf_cbc_record_splitting(mbedtls_ssl_config *conf, char split)
{
    conf->cbc_record_splitting = split;
}
#endif

void mbedtls_ssl_conf_legacy_renegotiation(mbedtls_ssl_config *conf, int allow_legacy)
{
    conf->allow_legacy_renegotiation = allow_legacy;
}

#if defined(MBEDTLS_SSL_RENEGOTIATION)
void mbedtls_ssl_conf_renegotiation(mbedtls_ssl_config *conf, int renegotiation)
{
    conf->disable_renegotiation = renegotiation;
}

void mbedtls_ssl_conf_renegotiation_enforced(mbedtls_ssl_config *conf, int max_records)
{
    conf->renego_max_records = max_records;
}

void mbedtls_ssl_conf_renegotiation_period(mbedtls_ssl_config *conf,
                                           const unsigned char period[8])
{
    memcpy(conf->renego_period, period, 8);
}
#endif /* MBEDTLS_SSL_RENEGOTIATION */

#if defined(MBEDTLS_SSL_SESSION_TICKETS)
#if defined(MBEDTLS_SSL_CLI_C)
void mbedtls_ssl_conf_session_tickets(mbedtls_ssl_config *conf, int use_tickets)
{
    conf->session_tickets = use_tickets;
}
#endif

#if defined(MBEDTLS_SSL_SRV_C)
void mbedtls_ssl_conf_session_tickets_cb(mbedtls_ssl_config *conf,
                                         mbedtls_ssl_ticket_write_t *f_ticket_write,
                                         mbedtls_ssl_ticket_parse_t *f_ticket_parse,
                                         void *p_ticket)
{
    conf->f_ticket_write = f_ticket_write;
    conf->f_ticket_parse = f_ticket_parse;
    conf->p_ticket       = p_ticket;
}
#endif
#endif /* MBEDTLS_SSL_SESSION_TICKETS */

#if defined(MBEDTLS_SSL_EXPORT_KEYS)
void mbedtls_ssl_conf_export_keys_cb(mbedtls_ssl_config *conf,
                                     mbedtls_ssl_export_keys_t *f_export_keys,
                                     void *p_export_keys)
{
    conf->f_export_keys = f_export_keys;
    conf->p_export_keys = p_export_keys;
}

void mbedtls_ssl_conf_export_keys_ext_cb(mbedtls_ssl_config *conf,
                                         mbedtls_ssl_export_keys_ext_t *f_export_keys_ext,
                                         void *p_export_keys)
{
    conf->f_export_keys_ext = f_export_keys_ext;
    conf->p_export_keys = p_export_keys;
}
#endif

#if defined(MBEDTLS_SSL_ASYNC_PRIVATE)
void mbedtls_ssl_conf_async_private_cb(
    mbedtls_ssl_config *conf,
    mbedtls_ssl_async_sign_t *f_async_sign,
    mbedtls_ssl_async_decrypt_t *f_async_decrypt,
    mbedtls_ssl_async_resume_t *f_async_resume,
    mbedtls_ssl_async_cancel_t *f_async_cancel,
    void *async_config_data)
{
    conf->f_async_sign_start = f_async_sign;
    conf->f_async_decrypt_start = f_async_decrypt;
    conf->f_async_resume = f_async_resume;
    conf->f_async_cancel = f_async_cancel;
    conf->p_async_config_data = async_config_data;
}

void *mbedtls_ssl_conf_get_async_config_data(const mbedtls_ssl_config *conf)
{
    return conf->p_async_config_data;
}

void *mbedtls_ssl_get_async_operation_data(const mbedtls_ssl_context *ssl)
{
    if (ssl->handshake == NULL) {
        return NULL;
    } else {
        return ssl->handshake->user_async_ctx;
    }
}

void mbedtls_ssl_set_async_operation_data(mbedtls_ssl_context *ssl,
                                          void *ctx)
{
    if (ssl->handshake != NULL) {
        ssl->handshake->user_async_ctx = ctx;
    }
}
#endif /* MBEDTLS_SSL_ASYNC_PRIVATE */

/*
 * SSL get accessors
 */
uint32_t mbedtls_ssl_get_verify_result(const mbedtls_ssl_context *ssl)
{
    if (ssl->session != NULL) {
        return ssl->session->verify_result;
    }

    if (ssl->session_negotiate != NULL) {
        return ssl->session_negotiate->verify_result;
    }

    return 0xFFFFFFFF;
}

const char *mbedtls_ssl_get_ciphersuite(const mbedtls_ssl_context *ssl)
{
    if (ssl == NULL || ssl->session == NULL) {
        return NULL;
    }

    return mbedtls_ssl_get_ciphersuite_name(ssl->session->ciphersuite);
}

const char *mbedtls_ssl_get_version(const mbedtls_ssl_context *ssl)
{
#if defined(MBEDTLS_SSL_PROTO_DTLS)
    if (ssl->conf->transport == MBEDTLS_SSL_TRANSPORT_DATAGRAM) {
        switch (ssl->minor_ver) {
            case MBEDTLS_SSL_MINOR_VERSION_2:
                return "DTLSv1.0";

            case MBEDTLS_SSL_MINOR_VERSION_3:
                return "DTLSv1.2";

            default:
                return "unknown (DTLS)";
        }
    }
#endif

    switch (ssl->minor_ver) {
        case MBEDTLS_SSL_MINOR_VERSION_0:
            return "SSLv3.0";

        case MBEDTLS_SSL_MINOR_VERSION_1:
            return "TLSv1.0";

        case MBEDTLS_SSL_MINOR_VERSION_2:
            return "TLSv1.1";

        case MBEDTLS_SSL_MINOR_VERSION_3:
            return "TLSv1.2";

        default:
            return "unknown";
    }
}

#if defined(MBEDTLS_SSL_MAX_FRAGMENT_LENGTH)
size_t mbedtls_ssl_get_input_max_frag_len(const mbedtls_ssl_context *ssl)
{
    size_t max_len = MBEDTLS_SSL_MAX_CONTENT_LEN;
    size_t read_mfl;

    /* Use the configured MFL for the client if we're past SERVER_HELLO_DONE */
    if (ssl->conf->endpoint == MBEDTLS_SSL_IS_CLIENT &&
        ssl->state >= MBEDTLS_SSL_SERVER_HELLO_DONE) {
        return ssl_mfl_code_to_length(ssl->conf->mfl_code);
    }

    /* Check if a smaller max length was negotiated */
    if (ssl->session_out != NULL) {
        read_mfl = ssl_mfl_code_to_length(ssl->session_out->mfl_code);
        if (read_mfl < max_len) {
            max_len = read_mfl;
        }
    }

    // During a handshake, use the value being negotiated
    if (ssl->session_negotiate != NULL) {
        read_mfl = ssl_mfl_code_to_length(ssl->session_negotiate->mfl_code);
        if (read_mfl < max_len) {
            max_len = read_mfl;
        }
    }

    return max_len;
}

size_t mbedtls_ssl_get_output_max_frag_len(const mbedtls_ssl_context *ssl)
{
    size_t max_len;

    /*
     * Assume mfl_code is correct since it was checked when set
     */
    max_len = ssl_mfl_code_to_length(ssl->conf->mfl_code);

    /* Check if a smaller max length was negotiated */
    if (ssl->session_out != NULL &&
        ssl_mfl_code_to_length(ssl->session_out->mfl_code) < max_len) {
        max_len = ssl_mfl_code_to_length(ssl->session_out->mfl_code);
    }

    /* During a handshake, use the value being negotiated */
    if (ssl->session_negotiate != NULL &&
        ssl_mfl_code_to_length(ssl->session_negotiate->mfl_code) < max_len) {
        max_len = ssl_mfl_code_to_length(ssl->session_negotiate->mfl_code);
    }

    return max_len;
}

#if !defined(MBEDTLS_DEPRECATED_REMOVED)
size_t mbedtls_ssl_get_max_frag_len(const mbedtls_ssl_context *ssl)
{
    return mbedtls_ssl_get_output_max_frag_len(ssl);
}
#endif /* !MBEDTLS_DEPRECATED_REMOVED */
#endif /* MBEDTLS_SSL_MAX_FRAGMENT_LENGTH */

#if defined(MBEDTLS_SSL_PROTO_DTLS)
size_t mbedtls_ssl_get_current_mtu(const mbedtls_ssl_context *ssl)
{
    /* Return unlimited mtu for client hello messages to avoid fragmentation. */
    if (ssl->conf->endpoint == MBEDTLS_SSL_IS_CLIENT &&
        (ssl->state == MBEDTLS_SSL_CLIENT_HELLO ||
         ssl->state == MBEDTLS_SSL_SERVER_HELLO)) {
        return 0;
    }

    if (ssl->handshake == NULL || ssl->handshake->mtu == 0) {
        return ssl->mtu;
    }

    if (ssl->mtu == 0) {
        return ssl->handshake->mtu;
    }

    return ssl->mtu < ssl->handshake->mtu ?
           ssl->mtu : ssl->handshake->mtu;
}
#endif /* MBEDTLS_SSL_PROTO_DTLS */

int mbedtls_ssl_get_max_out_record_payload(const mbedtls_ssl_context *ssl)
{
    size_t max_len = MBEDTLS_SSL_OUT_CONTENT_LEN;

#if !defined(MBEDTLS_SSL_MAX_FRAGMENT_LENGTH) && \
    !defined(MBEDTLS_SSL_PROTO_DTLS)
    (void) ssl;
#endif

#if defined(MBEDTLS_SSL_MAX_FRAGMENT_LENGTH)
    const size_t mfl = mbedtls_ssl_get_output_max_frag_len(ssl);

    if (max_len > mfl) {
        max_len = mfl;
    }
#endif

#if defined(MBEDTLS_SSL_PROTO_DTLS)
    if (mbedtls_ssl_get_current_mtu(ssl) != 0) {
        const size_t mtu = mbedtls_ssl_get_current_mtu(ssl);
        const int ret = mbedtls_ssl_get_record_expansion(ssl);
        const size_t overhead = (size_t) ret;

        if (ret < 0) {
            return ret;
        }

        if (mtu <= overhead) {
            MBEDTLS_SSL_DEBUG_MSG(1, ("MTU too low for record expansion"));
            return MBEDTLS_ERR_SSL_FEATURE_UNAVAILABLE;
        }

        if (max_len > mtu - overhead) {
            max_len = mtu - overhead;
        }
    }
#endif /* MBEDTLS_SSL_PROTO_DTLS */

#if !defined(MBEDTLS_SSL_MAX_FRAGMENT_LENGTH) &&        \
    !defined(MBEDTLS_SSL_PROTO_DTLS)
    ((void) ssl);
#endif

    return (int) max_len;
}

#if defined(MBEDTLS_X509_CRT_PARSE_C)
const mbedtls_x509_crt *mbedtls_ssl_get_peer_cert(const mbedtls_ssl_context *ssl)
{
    if (ssl == NULL || ssl->session == NULL) {
        return NULL;
    }

#if defined(MBEDTLS_SSL_KEEP_PEER_CERTIFICATE)
    return ssl->session->peer_cert;
#else
    return NULL;
#endif /* MBEDTLS_SSL_KEEP_PEER_CERTIFICATE */
}
#endif /* MBEDTLS_X509_CRT_PARSE_C */

#if defined(MBEDTLS_SSL_CLI_C)
int mbedtls_ssl_get_session(const mbedtls_ssl_context *ssl,
                            mbedtls_ssl_session *dst)
{
    if (ssl == NULL ||
        dst == NULL ||
        ssl->session == NULL ||
        ssl->conf->endpoint != MBEDTLS_SSL_IS_CLIENT) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    return mbedtls_ssl_session_copy(dst, ssl->session);
}
#endif /* MBEDTLS_SSL_CLI_C */

const mbedtls_ssl_session *mbedtls_ssl_get_session_pointer(const mbedtls_ssl_context *ssl)
{
    if (ssl == NULL) {
        return NULL;
    }

    return ssl->session;
}

/*
 * Define ticket header determining Mbed TLS version
 * and structure of the ticket.
 */

/*
 * Define bitflag determining compile-time settings influencing
 * structure of serialized SSL sessions.
 */

#if defined(MBEDTLS_HAVE_TIME)
#define SSL_SERIALIZED_SESSION_CONFIG_TIME 1
#else
#define SSL_SERIALIZED_SESSION_CONFIG_TIME 0
#endif /* MBEDTLS_HAVE_TIME */

#if defined(MBEDTLS_X509_CRT_PARSE_C)
#define SSL_SERIALIZED_SESSION_CONFIG_CRT 1
#else
#define SSL_SERIALIZED_SESSION_CONFIG_CRT 0
#endif /* MBEDTLS_X509_CRT_PARSE_C */

#if defined(MBEDTLS_SSL_CLI_C) && defined(MBEDTLS_SSL_SESSION_TICKETS)
#define SSL_SERIALIZED_SESSION_CONFIG_CLIENT_TICKET 1
#else
#define SSL_SERIALIZED_SESSION_CONFIG_CLIENT_TICKET 0
#endif /* MBEDTLS_SSL_CLI_C && MBEDTLS_SSL_SESSION_TICKETS */

#if defined(MBEDTLS_SSL_MAX_FRAGMENT_LENGTH)
#define SSL_SERIALIZED_SESSION_CONFIG_MFL 1
#else
#define SSL_SERIALIZED_SESSION_CONFIG_MFL 0
#endif /* MBEDTLS_SSL_MAX_FRAGMENT_LENGTH */

#if defined(MBEDTLS_SSL_TRUNCATED_HMAC)
#define SSL_SERIALIZED_SESSION_CONFIG_TRUNC_HMAC 1
#else
#define SSL_SERIALIZED_SESSION_CONFIG_TRUNC_HMAC 0
#endif /* MBEDTLS_SSL_TRUNCATED_HMAC */

#if defined(MBEDTLS_SSL_ENCRYPT_THEN_MAC)
#define SSL_SERIALIZED_SESSION_CONFIG_ETM 1
#else
#define SSL_SERIALIZED_SESSION_CONFIG_ETM 0
#endif /* MBEDTLS_SSL_ENCRYPT_THEN_MAC */

#if defined(MBEDTLS_SSL_SESSION_TICKETS)
#define SSL_SERIALIZED_SESSION_CONFIG_TICKET 1
#else
#define SSL_SERIALIZED_SESSION_CONFIG_TICKET 0
#endif /* MBEDTLS_SSL_SESSION_TICKETS */

#define SSL_SERIALIZED_SESSION_CONFIG_TIME_BIT          0
#define SSL_SERIALIZED_SESSION_CONFIG_CRT_BIT           1
#define SSL_SERIALIZED_SESSION_CONFIG_CLIENT_TICKET_BIT 2
#define SSL_SERIALIZED_SESSION_CONFIG_MFL_BIT           3
#define SSL_SERIALIZED_SESSION_CONFIG_TRUNC_HMAC_BIT    4
#define SSL_SERIALIZED_SESSION_CONFIG_ETM_BIT           5
#define SSL_SERIALIZED_SESSION_CONFIG_TICKET_BIT        6

#define SSL_SERIALIZED_SESSION_CONFIG_BITFLAG                           \
    ((uint16_t) (                                                      \
         (SSL_SERIALIZED_SESSION_CONFIG_TIME << SSL_SERIALIZED_SESSION_CONFIG_TIME_BIT) | \
         (SSL_SERIALIZED_SESSION_CONFIG_CRT << SSL_SERIALIZED_SESSION_CONFIG_CRT_BIT) | \
         (SSL_SERIALIZED_SESSION_CONFIG_CLIENT_TICKET << \
             SSL_SERIALIZED_SESSION_CONFIG_CLIENT_TICKET_BIT) | \
         (SSL_SERIALIZED_SESSION_CONFIG_MFL << SSL_SERIALIZED_SESSION_CONFIG_MFL_BIT) | \
         (SSL_SERIALIZED_SESSION_CONFIG_TRUNC_HMAC << \
             SSL_SERIALIZED_SESSION_CONFIG_TRUNC_HMAC_BIT) | \
         (SSL_SERIALIZED_SESSION_CONFIG_ETM << SSL_SERIALIZED_SESSION_CONFIG_ETM_BIT) | \
         (SSL_SERIALIZED_SESSION_CONFIG_TICKET << SSL_SERIALIZED_SESSION_CONFIG_TICKET_BIT)))

static unsigned char ssl_serialized_session_header[] = {
    MBEDTLS_VERSION_MAJOR,
    MBEDTLS_VERSION_MINOR,
    MBEDTLS_VERSION_PATCH,
    MBEDTLS_BYTE_1(SSL_SERIALIZED_SESSION_CONFIG_BITFLAG),
    MBEDTLS_BYTE_0(SSL_SERIALIZED_SESSION_CONFIG_BITFLAG),
};

/*
 * Serialize a session in the following format:
 * (in the presentation language of TLS, RFC 8446 section 3)
 *
 *  opaque mbedtls_version[3];   // major, minor, patch
 *  opaque session_format[2];    // version-specific 16-bit field determining
 *                               // the format of the remaining
 *                               // serialized data.
 *
 *  Note: When updating the format, remember to keep
 *        these version+format bytes.
 *
 *                               // In this version, `session_format` determines
 *                               // the setting of those compile-time
 *                               // configuration options which influence
 *                               // the structure of mbedtls_ssl_session.
 *  uint64 start_time;
 *  uint8 ciphersuite[2];        // defined by the standard
 *  uint8 compression;           // 0 or 1
 *  uint8 session_id_len;        // at most 32
 *  opaque session_id[32];
 *  opaque master[48];           // fixed length in the standard
 *  uint32 verify_result;
 *  opaque peer_cert<0..2^24-1>; // length 0 means no peer cert
 *  opaque ticket<0..2^24-1>;    // length 0 means no ticket
 *  uint32 ticket_lifetime;
 *  uint8 mfl_code;              // up to 255 according to standard
 *  uint8 trunc_hmac;            // 0 or 1
 *  uint8 encrypt_then_mac;      // 0 or 1
 *
 * The order is the same as in the definition of the structure, except
 * verify_result is put before peer_cert so that all mandatory fields come
 * together in one block.
 */
MBEDTLS_CHECK_RETURN_CRITICAL
static int ssl_session_save(const mbedtls_ssl_session *session,
                            unsigned char omit_header,
                            unsigned char *buf,
                            size_t buf_len,
                            size_t *olen)
{
    unsigned char *p = buf;
    size_t used = 0;
#if defined(MBEDTLS_HAVE_TIME)
    uint64_t start;
#endif
#if defined(MBEDTLS_X509_CRT_PARSE_C)
#if defined(MBEDTLS_SSL_KEEP_PEER_CERTIFICATE)
    size_t cert_len;
#endif /* MBEDTLS_SSL_KEEP_PEER_CERTIFICATE */
#endif /* MBEDTLS_X509_CRT_PARSE_C */


    if (!omit_header) {
        /*
         * Add version identifier
         */

        used += sizeof(ssl_serialized_session_header);

        if (used <= buf_len) {
            memcpy(p, ssl_serialized_session_header,
                   sizeof(ssl_serialized_session_header));
            p += sizeof(ssl_serialized_session_header);
        }
    }

    /*
     * Time
     */
#if defined(MBEDTLS_HAVE_TIME)
    used += 8;

    if (used <= buf_len) {
        start = (uint64_t) session->start;

        MBEDTLS_PUT_UINT64_BE(start, p, 0);
        p += 8;
    }
#endif /* MBEDTLS_HAVE_TIME */

    /*
     * Basic mandatory fields
     */
    used += 2   /* ciphersuite */
            + 1 /* compression */
            + 1 /* id_len */
            + sizeof(session->id)
            + sizeof(session->master)
            + 4; /* verify_result */

    if (used <= buf_len) {
        MBEDTLS_PUT_UINT16_BE(session->ciphersuite, p, 0);
        p += 2;

        *p++ = MBEDTLS_BYTE_0(session->compression);

        *p++ = MBEDTLS_BYTE_0(session->id_len);
        memcpy(p, session->id, 32);
        p += 32;

        memcpy(p, session->master, 48);
        p += 48;

        MBEDTLS_PUT_UINT32_BE(session->verify_result, p, 0);
        p += 4;
    }

    /*
     * Peer's end-entity certificate
     */
#if defined(MBEDTLS_X509_CRT_PARSE_C)
#if defined(MBEDTLS_SSL_KEEP_PEER_CERTIFICATE)
    if (session->peer_cert == NULL) {
        cert_len = 0;
    } else {
        cert_len = session->peer_cert->raw.len;
    }

    used += 3 + cert_len;

    if (used <= buf_len) {
        *p++ = MBEDTLS_BYTE_2(cert_len);
        *p++ = MBEDTLS_BYTE_1(cert_len);
        *p++ = MBEDTLS_BYTE_0(cert_len);

        if (session->peer_cert != NULL) {
            memcpy(p, session->peer_cert->raw.p, cert_len);
            p += cert_len;
        }
    }
#else /* MBEDTLS_SSL_KEEP_PEER_CERTIFICATE */
    if (session->peer_cert_digest != NULL) {
        used += 1 /* type */ + 1 /* length */ + session->peer_cert_digest_len;
        if (used <= buf_len) {
            *p++ = (unsigned char) session->peer_cert_digest_type;
            *p++ = (unsigned char) session->peer_cert_digest_len;
            memcpy(p, session->peer_cert_digest,
                   session->peer_cert_digest_len);
            p += session->peer_cert_digest_len;
        }
    } else {
        used += 2;
        if (used <= buf_len) {
            *p++ = (unsigned char) MBEDTLS_MD_NONE;
            *p++ = 0;
        }
    }
#endif /* !MBEDTLS_SSL_KEEP_PEER_CERTIFICATE */
#endif /* MBEDTLS_X509_CRT_PARSE_C */

    /*
     * Session ticket if any, plus associated data
     */
#if defined(MBEDTLS_SSL_SESSION_TICKETS) && defined(MBEDTLS_SSL_CLI_C)
    used += 3 + session->ticket_len + 4; /* len + ticket + lifetime */

    if (used <= buf_len) {
        *p++ = MBEDTLS_BYTE_2(session->ticket_len);
        *p++ = MBEDTLS_BYTE_1(session->ticket_len);
        *p++ = MBEDTLS_BYTE_0(session->ticket_len);

        if (session->ticket != NULL) {
            memcpy(p, session->ticket, session->ticket_len);
            p += session->ticket_len;
        }

        MBEDTLS_PUT_UINT32_BE(session->ticket_lifetime, p, 0);
        p += 4;
    }
#endif /* MBEDTLS_SSL_SESSION_TICKETS && MBEDTLS_SSL_CLI_C */

    /*
     * Misc extension-related info
     */
#if defined(MBEDTLS_SSL_MAX_FRAGMENT_LENGTH)
    used += 1;

    if (used <= buf_len) {
        *p++ = session->mfl_code;
    }
#endif

#if defined(MBEDTLS_SSL_TRUNCATED_HMAC)
    used += 1;

    if (used <= buf_len) {
        *p++ = (unsigned char) ((session->trunc_hmac) & 0xFF);
    }
#endif

#if defined(MBEDTLS_SSL_ENCRYPT_THEN_MAC)
    used += 1;

    if (used <= buf_len) {
        *p++ = MBEDTLS_BYTE_0(session->encrypt_then_mac);
    }
#endif

    /* Done */
    *olen = used;

    if (used > buf_len) {
        return MBEDTLS_ERR_SSL_BUFFER_TOO_SMALL;
    }

    return 0;
}

/*
 * Public wrapper for ssl_session_save()
 */
int mbedtls_ssl_session_save(const mbedtls_ssl_session *session,
                             unsigned char *buf,
                             size_t buf_len,
                             size_t *olen)
{
    return ssl_session_save(session, 0, buf, buf_len, olen);
}

/*
 * Deserialize session, see mbedtls_ssl_session_save() for format.
 *
 * This internal version is wrapped by a public function that cleans up in
 * case of error, and has an extra option omit_header.
 */
MBEDTLS_CHECK_RETURN_CRITICAL
static int ssl_session_load(mbedtls_ssl_session *session,
                            unsigned char omit_header,
                            const unsigned char *buf,
                            size_t len)
{
    const unsigned char *p = buf;
    const unsigned char * const end = buf + len;
#if defined(MBEDTLS_HAVE_TIME)
    uint64_t start;
#endif
#if defined(MBEDTLS_X509_CRT_PARSE_C)
#if defined(MBEDTLS_SSL_KEEP_PEER_CERTIFICATE)
    size_t cert_len;
#endif /* MBEDTLS_SSL_KEEP_PEER_CERTIFICATE */
#endif /* MBEDTLS_X509_CRT_PARSE_C */

    if (!omit_header) {
        /*
         * Check version identifier
         */

        if ((size_t) (end - p) < sizeof(ssl_serialized_session_header)) {
            return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
        }

        if (memcmp(p, ssl_serialized_session_header,
                   sizeof(ssl_serialized_session_header)) != 0) {
            return MBEDTLS_ERR_SSL_VERSION_MISMATCH;
        }
        p += sizeof(ssl_serialized_session_header);
    }

    /*
     * Time
     */
#if defined(MBEDTLS_HAVE_TIME)
    if (8 > (size_t) (end - p)) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    start = ((uint64_t) p[0] << 56) |
            ((uint64_t) p[1] << 48) |
            ((uint64_t) p[2] << 40) |
            ((uint64_t) p[3] << 32) |
            ((uint64_t) p[4] << 24) |
            ((uint64_t) p[5] << 16) |
            ((uint64_t) p[6] <<  8) |
            ((uint64_t) p[7]);
    p += 8;

    session->start = (time_t) start;
#endif /* MBEDTLS_HAVE_TIME */

    /*
     * Basic mandatory fields
     */
    if (2 + 1 + 1 + 32 + 48 + 4 > (size_t) (end - p)) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    session->ciphersuite = (p[0] << 8) | p[1];
    p += 2;

    session->compression = *p++;

    session->id_len = *p++;
    memcpy(session->id, p, 32);
    p += 32;

    memcpy(session->master, p, 48);
    p += 48;

    session->verify_result = ((uint32_t) p[0] << 24) |
                             ((uint32_t) p[1] << 16) |
                             ((uint32_t) p[2] <<  8) |
                             ((uint32_t) p[3]);
    p += 4;

    /* Immediately clear invalid pointer values that have been read, in case
     * we exit early before we replaced them with valid ones. */
#if defined(MBEDTLS_X509_CRT_PARSE_C)
#if defined(MBEDTLS_SSL_KEEP_PEER_CERTIFICATE)
    session->peer_cert = NULL;
#else
    session->peer_cert_digest = NULL;
#endif /* !MBEDTLS_SSL_KEEP_PEER_CERTIFICATE */
#endif /* MBEDTLS_X509_CRT_PARSE_C */
#if defined(MBEDTLS_SSL_SESSION_TICKETS) && defined(MBEDTLS_SSL_CLI_C)
    session->ticket = NULL;
#endif /* MBEDTLS_SSL_SESSION_TICKETS && MBEDTLS_SSL_CLI_C */

    /*
     * Peer certificate
     */
#if defined(MBEDTLS_X509_CRT_PARSE_C)
#if defined(MBEDTLS_SSL_KEEP_PEER_CERTIFICATE)
    /* Deserialize CRT from the end of the ticket. */
    if (3 > (size_t) (end - p)) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    cert_len = (p[0] << 16) | (p[1] << 8) | p[2];
    p += 3;

    if (cert_len != 0) {
        int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;

        if (cert_len > (size_t) (end - p)) {
            return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
        }

        session->peer_cert = mbedtls_calloc(1, sizeof(mbedtls_x509_crt));

        if (session->peer_cert == NULL) {
            return MBEDTLS_ERR_SSL_ALLOC_FAILED;
        }

        mbedtls_x509_crt_init(session->peer_cert);

        if ((ret = mbedtls_x509_crt_parse_der(session->peer_cert,
                                              p, cert_len)) != 0) {
            mbedtls_x509_crt_free(session->peer_cert);
            mbedtls_free(session->peer_cert);
            session->peer_cert = NULL;
            return ret;
        }

        p += cert_len;
    }
#else /* MBEDTLS_SSL_KEEP_PEER_CERTIFICATE */
    /* Deserialize CRT digest from the end of the ticket. */
    if (2 > (size_t) (end - p)) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    session->peer_cert_digest_type = (mbedtls_md_type_t) *p++;
    session->peer_cert_digest_len  = (size_t) *p++;

    if (session->peer_cert_digest_len != 0) {
        const mbedtls_md_info_t *md_info =
            mbedtls_md_info_from_type(session->peer_cert_digest_type);
        if (md_info == NULL) {
            return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
        }
        if (session->peer_cert_digest_len != mbedtls_md_get_size(md_info)) {
            return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
        }

        if (session->peer_cert_digest_len > (size_t) (end - p)) {
            return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
        }

        session->peer_cert_digest =
            mbedtls_calloc(1, session->peer_cert_digest_len);
        if (session->peer_cert_digest == NULL) {
            return MBEDTLS_ERR_SSL_ALLOC_FAILED;
        }

        memcpy(session->peer_cert_digest, p,
               session->peer_cert_digest_len);
        p += session->peer_cert_digest_len;
    }
#endif /* MBEDTLS_SSL_KEEP_PEER_CERTIFICATE */
#endif /* MBEDTLS_X509_CRT_PARSE_C */

    /*
     * Session ticket and associated data
     */
#if defined(MBEDTLS_SSL_SESSION_TICKETS) && defined(MBEDTLS_SSL_CLI_C)
    if (3 > (size_t) (end - p)) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    session->ticket_len = (p[0] << 16) | (p[1] << 8) | p[2];
    p += 3;

    if (session->ticket_len != 0) {
        if (session->ticket_len > (size_t) (end - p)) {
            return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
        }

        session->ticket = mbedtls_calloc(1, session->ticket_len);
        if (session->ticket == NULL) {
            return MBEDTLS_ERR_SSL_ALLOC_FAILED;
        }

        memcpy(session->ticket, p, session->ticket_len);
        p += session->ticket_len;
    }

    if (4 > (size_t) (end - p)) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    session->ticket_lifetime = ((uint32_t) p[0] << 24) |
                               ((uint32_t) p[1] << 16) |
                               ((uint32_t) p[2] <<  8) |
                               ((uint32_t) p[3]);
    p += 4;
#endif /* MBEDTLS_SSL_SESSION_TICKETS && MBEDTLS_SSL_CLI_C */

    /*
     * Misc extension-related info
     */
#if defined(MBEDTLS_SSL_MAX_FRAGMENT_LENGTH)
    if (1 > (size_t) (end - p)) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    session->mfl_code = *p++;
#endif

#if defined(MBEDTLS_SSL_TRUNCATED_HMAC)
    if (1 > (size_t) (end - p)) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    session->trunc_hmac = *p++;
#endif

#if defined(MBEDTLS_SSL_ENCRYPT_THEN_MAC)
    if (1 > (size_t) (end - p)) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    session->encrypt_then_mac = *p++;
#endif

    /* Done, should have consumed entire buffer */
    if (p != end) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    return 0;
}

/*
 * Deserialize session: public wrapper for error cleaning
 */
int mbedtls_ssl_session_load(mbedtls_ssl_session *session,
                             const unsigned char *buf,
                             size_t len)
{
    int ret = ssl_session_load(session, 0, buf, len);

    if (ret != 0) {
        mbedtls_ssl_session_free(session);
    }

    return ret;
}

/*
 * Perform a single step of the SSL handshake
 */
int mbedtls_ssl_handshake_step(mbedtls_ssl_context *ssl)
{
    int ret = MBEDTLS_ERR_SSL_FEATURE_UNAVAILABLE;

    if (ssl == NULL || ssl->conf == NULL) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

#if defined(MBEDTLS_SSL_CLI_C)
    if (ssl->conf->endpoint == MBEDTLS_SSL_IS_CLIENT) {
        ret = mbedtls_ssl_handshake_client_step(ssl);
    }
#endif
#if defined(MBEDTLS_SSL_SRV_C)
    if (ssl->conf->endpoint == MBEDTLS_SSL_IS_SERVER) {
        ret = mbedtls_ssl_handshake_server_step(ssl);
    }
#endif

    return ret;
}

/*
 * Perform the SSL handshake
 */
int mbedtls_ssl_handshake(mbedtls_ssl_context *ssl)
{
    int ret = 0;

    /* Sanity checks */

    if (ssl == NULL || ssl->conf == NULL) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

#if defined(MBEDTLS_SSL_PROTO_DTLS)
    if (ssl->conf->transport == MBEDTLS_SSL_TRANSPORT_DATAGRAM &&
        (ssl->f_set_timer == NULL || ssl->f_get_timer == NULL)) {
        MBEDTLS_SSL_DEBUG_MSG(1, ("You must use "
                                  "mbedtls_ssl_set_timer_cb() for DTLS"));
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }
#endif /* MBEDTLS_SSL_PROTO_DTLS */

    MBEDTLS_SSL_DEBUG_MSG(2, ("=> handshake"));

    /* Main handshake loop */
    while (ssl->state != MBEDTLS_SSL_HANDSHAKE_OVER) {
        ret = mbedtls_ssl_handshake_step(ssl);

        if (ret != 0) {
            break;
        }
    }

    MBEDTLS_SSL_DEBUG_MSG(2, ("<= handshake"));

    return ret;
}

#if defined(MBEDTLS_SSL_RENEGOTIATION)
#if defined(MBEDTLS_SSL_SRV_C)
/*
 * Write HelloRequest to request renegotiation on server
 */
MBEDTLS_CHECK_RETURN_CRITICAL
static int ssl_write_hello_request(mbedtls_ssl_context *ssl)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;

    MBEDTLS_SSL_DEBUG_MSG(2, ("=> write hello request"));

    ssl->out_msglen  = 4;
    ssl->out_msgtype = MBEDTLS_SSL_MSG_HANDSHAKE;
    ssl->out_msg[0]  = MBEDTLS_SSL_HS_HELLO_REQUEST;

    if ((ret = mbedtls_ssl_write_handshake_msg(ssl)) != 0) {
        MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_ssl_write_handshake_msg", ret);
        return ret;
    }

    MBEDTLS_SSL_DEBUG_MSG(2, ("<= write hello request"));

    return 0;
}
#endif /* MBEDTLS_SSL_SRV_C */

/*
 * Actually renegotiate current connection, triggered by either:
 * - any side: calling mbedtls_ssl_renegotiate(),
 * - client: receiving a HelloRequest during mbedtls_ssl_read(),
 * - server: receiving any handshake message on server during mbedtls_ssl_read() after
 *   the initial handshake is completed.
 * If the handshake doesn't complete due to waiting for I/O, it will continue
 * during the next calls to mbedtls_ssl_renegotiate() or mbedtls_ssl_read() respectively.
 */
int mbedtls_ssl_start_renegotiation(mbedtls_ssl_context *ssl)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;

    MBEDTLS_SSL_DEBUG_MSG(2, ("=> renegotiate"));

    if ((ret = ssl_handshake_init(ssl)) != 0) {
        return ret;
    }

    /* RFC 6347 4.2.2: "[...] the HelloRequest will have message_seq = 0 and
     * the ServerHello will have message_seq = 1" */
#if defined(MBEDTLS_SSL_PROTO_DTLS)
    if (ssl->conf->transport == MBEDTLS_SSL_TRANSPORT_DATAGRAM &&
        ssl->renego_status == MBEDTLS_SSL_RENEGOTIATION_PENDING) {
        if (ssl->conf->endpoint == MBEDTLS_SSL_IS_SERVER) {
            ssl->handshake->out_msg_seq = 1;
        } else {
            ssl->handshake->in_msg_seq = 1;
        }
    }
#endif

    ssl->state = MBEDTLS_SSL_HELLO_REQUEST;
    ssl->renego_status = MBEDTLS_SSL_RENEGOTIATION_IN_PROGRESS;

    if ((ret = mbedtls_ssl_handshake(ssl)) != 0) {
        MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_ssl_handshake", ret);
        return ret;
    }

    MBEDTLS_SSL_DEBUG_MSG(2, ("<= renegotiate"));

    return 0;
}

/*
 * Renegotiate current connection on client,
 * or request renegotiation on server
 */
int mbedtls_ssl_renegotiate(mbedtls_ssl_context *ssl)
{
    int ret = MBEDTLS_ERR_SSL_FEATURE_UNAVAILABLE;

    if (ssl == NULL || ssl->conf == NULL) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

#if defined(MBEDTLS_SSL_SRV_C)
    /* On server, just send the request */
    if (ssl->conf->endpoint == MBEDTLS_SSL_IS_SERVER) {
        if (ssl->state != MBEDTLS_SSL_HANDSHAKE_OVER) {
            return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
        }

        ssl->renego_status = MBEDTLS_SSL_RENEGOTIATION_PENDING;

        /* Did we already try/start sending HelloRequest? */
        if (ssl->out_left != 0) {
            return mbedtls_ssl_flush_output(ssl);
        }

        return ssl_write_hello_request(ssl);
    }
#endif /* MBEDTLS_SSL_SRV_C */

#if defined(MBEDTLS_SSL_CLI_C)
    /*
     * On client, either start the renegotiation process or,
     * if already in progress, continue the handshake
     */
    if (ssl->renego_status != MBEDTLS_SSL_RENEGOTIATION_IN_PROGRESS) {
        if (ssl->state != MBEDTLS_SSL_HANDSHAKE_OVER) {
            return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
        }

        if ((ret = mbedtls_ssl_start_renegotiation(ssl)) != 0) {
            MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_ssl_start_renegotiation", ret);
            return ret;
        }
    } else {
        if ((ret = mbedtls_ssl_handshake(ssl)) != 0) {
            MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_ssl_handshake", ret);
            return ret;
        }
    }
#endif /* MBEDTLS_SSL_CLI_C */

    return ret;
}
#endif /* MBEDTLS_SSL_RENEGOTIATION */

#if defined(MBEDTLS_X509_CRT_PARSE_C)
static void ssl_key_cert_free(mbedtls_ssl_key_cert *key_cert)
{
    mbedtls_ssl_key_cert *cur = key_cert, *next;

    while (cur != NULL) {
        next = cur->next;
        mbedtls_free(cur);
        cur = next;
    }
}
#endif /* MBEDTLS_X509_CRT_PARSE_C */

void mbedtls_ssl_handshake_free(mbedtls_ssl_context *ssl)
{
    mbedtls_ssl_handshake_params *handshake = ssl->handshake;

    if (handshake == NULL) {
        return;
    }

#if defined(MBEDTLS_SSL_ASYNC_PRIVATE)
    if (ssl->conf->f_async_cancel != NULL && handshake->async_in_progress != 0) {
        ssl->conf->f_async_cancel(ssl);
        handshake->async_in_progress = 0;
    }
#endif /* MBEDTLS_SSL_ASYNC_PRIVATE */

#if defined(MBEDTLS_SSL_PROTO_SSL3) || defined(MBEDTLS_SSL_PROTO_TLS1) || \
    defined(MBEDTLS_SSL_PROTO_TLS1_1)
    mbedtls_md5_free(&handshake->fin_md5);
    mbedtls_sha1_free(&handshake->fin_sha1);
#endif
#if defined(MBEDTLS_SSL_PROTO_TLS1_2)
#if defined(MBEDTLS_SHA256_C)
#if defined(MBEDTLS_USE_PSA_CRYPTO)
    psa_hash_abort(&handshake->fin_sha256_psa);
#else
    mbedtls_sha256_free(&handshake->fin_sha256);
#endif
#endif
#if defined(MBEDTLS_SHA512_C) && !defined(MBEDTLS_SHA512_NO_SHA384)
#if defined(MBEDTLS_USE_PSA_CRYPTO)
    psa_hash_abort(&handshake->fin_sha384_psa);
#else
    mbedtls_sha512_free(&handshake->fin_sha512);
#endif
#endif
#endif /* MBEDTLS_SSL_PROTO_TLS1_2 */

#if defined(MBEDTLS_DHM_C)
    mbedtls_dhm_free(&handshake->dhm_ctx);
#endif
#if defined(MBEDTLS_ECDH_C)
    mbedtls_ecdh_free(&handshake->ecdh_ctx);
#endif
#if defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
    mbedtls_ecjpake_free(&handshake->ecjpake_ctx);
#if defined(MBEDTLS_SSL_CLI_C)
    mbedtls_free(handshake->ecjpake_cache);
    handshake->ecjpake_cache = NULL;
    handshake->ecjpake_cache_len = 0;
#endif
#endif

#if defined(MBEDTLS_ECDH_C) || defined(MBEDTLS_ECDSA_C) || \
    defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
    /* explicit void pointer cast for buggy MS compiler */
    mbedtls_free((void *) handshake->curves);
#endif

#if defined(MBEDTLS_KEY_EXCHANGE_SOME_PSK_ENABLED)
    if (handshake->psk != NULL) {
        mbedtls_platform_zeroize(handshake->psk, handshake->psk_len);
        mbedtls_free(handshake->psk);
    }
#endif

#if defined(MBEDTLS_X509_CRT_PARSE_C) && \
    defined(MBEDTLS_SSL_SERVER_NAME_INDICATION)
    /*
     * Free only the linked list wrapper, not the keys themselves
     * since the belong to the SNI callback
     */
    if (handshake->sni_key_cert != NULL) {
        mbedtls_ssl_key_cert *cur = handshake->sni_key_cert, *next;

        while (cur != NULL) {
            next = cur->next;
            mbedtls_free(cur);
            cur = next;
        }
    }
#endif /* MBEDTLS_X509_CRT_PARSE_C && MBEDTLS_SSL_SERVER_NAME_INDICATION */

#if defined(MBEDTLS_SSL_ECP_RESTARTABLE_ENABLED)
    mbedtls_x509_crt_restart_free(&handshake->ecrs_ctx);
    if (handshake->ecrs_peer_cert != NULL) {
        mbedtls_x509_crt_free(handshake->ecrs_peer_cert);
        mbedtls_free(handshake->ecrs_peer_cert);
    }
#endif

#if defined(MBEDTLS_X509_CRT_PARSE_C) &&        \
    !defined(MBEDTLS_SSL_KEEP_PEER_CERTIFICATE)
    mbedtls_pk_free(&handshake->peer_pubkey);
#endif /* MBEDTLS_X509_CRT_PARSE_C && !MBEDTLS_SSL_KEEP_PEER_CERTIFICATE */

#if defined(MBEDTLS_SSL_PROTO_DTLS)
    mbedtls_free(handshake->verify_cookie);
    mbedtls_ssl_flight_free(handshake->flight);
    mbedtls_ssl_buffering_free(ssl);
#endif

#if defined(MBEDTLS_ECDH_C) &&                  \
    defined(MBEDTLS_USE_PSA_CRYPTO)
    psa_destroy_key(handshake->ecdh_psa_privkey);
#endif /* MBEDTLS_ECDH_C && MBEDTLS_USE_PSA_CRYPTO */

    mbedtls_platform_zeroize(handshake,
                             sizeof(mbedtls_ssl_handshake_params));

#if defined(MBEDTLS_SSL_VARIABLE_BUFFER_LENGTH)
    /* If the buffers are too big - reallocate. Because of the way Mbed TLS
     * processes datagrams and the fact that a datagram is allowed to have
     * several records in it, it is possible that the I/O buffers are not
     * empty at this stage */
    handle_buffer_resizing(ssl, 1, mbedtls_ssl_get_input_buflen(ssl),
                           mbedtls_ssl_get_output_buflen(ssl));
#endif
}

void mbedtls_ssl_session_free(mbedtls_ssl_session *session)
{
    if (session == NULL) {
        return;
    }

#if defined(MBEDTLS_X509_CRT_PARSE_C)
    ssl_clear_peer_cert(session);
#endif

#if defined(MBEDTLS_SSL_SESSION_TICKETS) && defined(MBEDTLS_SSL_CLI_C)
    mbedtls_free(session->ticket);
#endif

    mbedtls_platform_zeroize(session, sizeof(mbedtls_ssl_session));
}

#if defined(MBEDTLS_SSL_CONTEXT_SERIALIZATION)

#if defined(MBEDTLS_SSL_DTLS_CONNECTION_ID)
#define SSL_SERIALIZED_CONTEXT_CONFIG_DTLS_CONNECTION_ID 1u
#else
#define SSL_SERIALIZED_CONTEXT_CONFIG_DTLS_CONNECTION_ID 0u
#endif /* MBEDTLS_SSL_DTLS_CONNECTION_ID */

#if defined(MBEDTLS_SSL_DTLS_BADMAC_LIMIT)
#define SSL_SERIALIZED_CONTEXT_CONFIG_DTLS_BADMAC_LIMIT 1u
#else
#define SSL_SERIALIZED_CONTEXT_CONFIG_DTLS_BADMAC_LIMIT 0u
#endif /* MBEDTLS_SSL_DTLS_BADMAC_LIMIT */

#if defined(MBEDTLS_SSL_DTLS_ANTI_REPLAY)
#define SSL_SERIALIZED_CONTEXT_CONFIG_DTLS_ANTI_REPLAY 1u
#else
#define SSL_SERIALIZED_CONTEXT_CONFIG_DTLS_ANTI_REPLAY 0u
#endif /* MBEDTLS_SSL_DTLS_ANTI_REPLAY */

#if defined(MBEDTLS_SSL_ALPN)
#define SSL_SERIALIZED_CONTEXT_CONFIG_ALPN 1u
#else
#define SSL_SERIALIZED_CONTEXT_CONFIG_ALPN 0u
#endif /* MBEDTLS_SSL_ALPN */

#define SSL_SERIALIZED_CONTEXT_CONFIG_DTLS_CONNECTION_ID_BIT    0
#define SSL_SERIALIZED_CONTEXT_CONFIG_DTLS_BADMAC_LIMIT_BIT     1
#define SSL_SERIALIZED_CONTEXT_CONFIG_DTLS_ANTI_REPLAY_BIT      2
#define SSL_SERIALIZED_CONTEXT_CONFIG_ALPN_BIT                  3

#define SSL_SERIALIZED_CONTEXT_CONFIG_BITFLAG   \
    ((uint32_t) (                              \
         (SSL_SERIALIZED_CONTEXT_CONFIG_DTLS_CONNECTION_ID << \
             SSL_SERIALIZED_CONTEXT_CONFIG_DTLS_CONNECTION_ID_BIT) | \
         (SSL_SERIALIZED_CONTEXT_CONFIG_DTLS_BADMAC_LIMIT << \
             SSL_SERIALIZED_CONTEXT_CONFIG_DTLS_BADMAC_LIMIT_BIT) | \
         (SSL_SERIALIZED_CONTEXT_CONFIG_DTLS_ANTI_REPLAY << \
             SSL_SERIALIZED_CONTEXT_CONFIG_DTLS_ANTI_REPLAY_BIT) | \
         (SSL_SERIALIZED_CONTEXT_CONFIG_ALPN << SSL_SERIALIZED_CONTEXT_CONFIG_ALPN_BIT) | \
         0u))

static unsigned char ssl_serialized_context_header[] = {
    MBEDTLS_VERSION_MAJOR,
    MBEDTLS_VERSION_MINOR,
    MBEDTLS_VERSION_PATCH,
    MBEDTLS_BYTE_1(SSL_SERIALIZED_SESSION_CONFIG_BITFLAG),
    MBEDTLS_BYTE_0(SSL_SERIALIZED_SESSION_CONFIG_BITFLAG),
    MBEDTLS_BYTE_2(SSL_SERIALIZED_CONTEXT_CONFIG_BITFLAG),
    MBEDTLS_BYTE_1(SSL_SERIALIZED_CONTEXT_CONFIG_BITFLAG),
    MBEDTLS_BYTE_0(SSL_SERIALIZED_CONTEXT_CONFIG_BITFLAG),
};

/*
 * Serialize a full SSL context
 *
 * The format of the serialized data is:
 * (in the presentation language of TLS, RFC 8446 section 3)
 *
 *  // header
 *  opaque mbedtls_version[3];   // major, minor, patch
 *  opaque context_format[5];    // version-specific field determining
 *                               // the format of the remaining
 *                               // serialized data.
 *  Note: When updating the format, remember to keep these
 *        version+format bytes. (We may make their size part of the API.)
 *
 *  // session sub-structure
 *  opaque session<1..2^32-1>;  // see mbedtls_ssl_session_save()
 *  // transform sub-structure
 *  uint8 random[64];           // ServerHello.random+ClientHello.random
 *  uint8 in_cid<0..2^8-1>      // Connection ID: expected incoming value
 *  uint8 out_cid<0..2^8-1>     // Connection ID: outgoing value to use
 *  // fields from ssl_context
 *  uint32 badmac_seen;         // DTLS: number of records with failing MAC
 *  uint64 in_window_top;       // DTLS: last validated record seq_num
 *  uint64 in_window;           // DTLS: bitmask for replay protection
 *  uint8 disable_datagram_packing; // DTLS: only one record per datagram
 *  uint64 cur_out_ctr;         // Record layer: outgoing sequence number
 *  uint16 mtu;                 // DTLS: path mtu (max outgoing fragment size)
 *  uint8 alpn_chosen<0..2^8-1> // ALPN: negotiated application protocol
 *
 * Note that many fields of the ssl_context or sub-structures are not
 * serialized, as they fall in one of the following categories:
 *
 *  1. forced value (eg in_left must be 0)
 *  2. pointer to dynamically-allocated memory (eg session, transform)
 *  3. value can be re-derived from other data (eg session keys from MS)
 *  4. value was temporary (eg content of input buffer)
 *  5. value will be provided by the user again (eg I/O callbacks and context)
 */
int mbedtls_ssl_context_save(mbedtls_ssl_context *ssl,
                             unsigned char *buf,
                             size_t buf_len,
                             size_t *olen)
{
    unsigned char *p = buf;
    size_t used = 0;
    size_t session_len;
    int ret = 0;

    /*
     * Enforce usage restrictions, see "return BAD_INPUT_DATA" in
     * this function's documentation.
     *
     * These are due to assumptions/limitations in the implementation. Some of
     * them are likely to stay (no handshake in progress) some might go away
     * (only DTLS) but are currently used to simplify the implementation.
     */
    /* The initial handshake must be over */
    if (ssl->state != MBEDTLS_SSL_HANDSHAKE_OVER) {
        MBEDTLS_SSL_DEBUG_MSG(1, ("Initial handshake isn't over"));
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }
    if (ssl->handshake != NULL) {
        MBEDTLS_SSL_DEBUG_MSG(1, ("Handshake isn't completed"));
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }
    /* Double-check that sub-structures are indeed ready */
    if (ssl->transform == NULL || ssl->session == NULL) {
        MBEDTLS_SSL_DEBUG_MSG(1, ("Serialised structures aren't ready"));
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }
    /* There must be no pending incoming or outgoing data */
    if (mbedtls_ssl_check_pending(ssl) != 0) {
        MBEDTLS_SSL_DEBUG_MSG(1, ("There is pending incoming data"));
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }
    if (ssl->out_left != 0) {
        MBEDTLS_SSL_DEBUG_MSG(1, ("There is pending outgoing data"));
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }
    /* Protocol must be DTLS, not TLS */
    if (ssl->conf->transport != MBEDTLS_SSL_TRANSPORT_DATAGRAM) {
        MBEDTLS_SSL_DEBUG_MSG(1, ("Only DTLS is supported"));
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }
    /* Version must be 1.2 */
    if (ssl->major_ver != MBEDTLS_SSL_MAJOR_VERSION_3) {
        MBEDTLS_SSL_DEBUG_MSG(1, ("Only version 1.2 supported"));
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }
    if (ssl->minor_ver != MBEDTLS_SSL_MINOR_VERSION_3) {
        MBEDTLS_SSL_DEBUG_MSG(1, ("Only version 1.2 supported"));
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }
    /* We must be using an AEAD ciphersuite */
    if (mbedtls_ssl_transform_uses_aead(ssl->transform) != 1) {
        MBEDTLS_SSL_DEBUG_MSG(1, ("Only AEAD ciphersuites supported"));
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }
    /* Renegotiation must not be enabled */
#if defined(MBEDTLS_SSL_RENEGOTIATION)
    if (ssl->conf->disable_renegotiation != MBEDTLS_SSL_RENEGOTIATION_DISABLED) {
        MBEDTLS_SSL_DEBUG_MSG(1, ("Renegotiation must not be enabled"));
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }
#endif

    /*
     * Version and format identifier
     */
    used += sizeof(ssl_serialized_context_header);

    if (used <= buf_len) {
        memcpy(p, ssl_serialized_context_header,
               sizeof(ssl_serialized_context_header));
        p += sizeof(ssl_serialized_context_header);
    }

    /*
     * Session (length + data)
     */
    ret = ssl_session_save(ssl->session, 1, NULL, 0, &session_len);
    if (ret != MBEDTLS_ERR_SSL_BUFFER_TOO_SMALL) {
        return ret;
    }

    used += 4 + session_len;
    if (used <= buf_len) {
        MBEDTLS_PUT_UINT32_BE(session_len, p, 0);
        p += 4;

        ret = ssl_session_save(ssl->session, 1,
                               p, session_len, &session_len);
        if (ret != 0) {
            return ret;
        }

        p += session_len;
    }

    /*
     * Transform
     */
    used += sizeof(ssl->transform->randbytes);
    if (used <= buf_len) {
        memcpy(p, ssl->transform->randbytes,
               sizeof(ssl->transform->randbytes));
        p += sizeof(ssl->transform->randbytes);
    }

#if defined(MBEDTLS_SSL_DTLS_CONNECTION_ID)
    used += 2 + ssl->transform->in_cid_len + ssl->transform->out_cid_len;
    if (used <= buf_len) {
        *p++ = ssl->transform->in_cid_len;
        memcpy(p, ssl->transform->in_cid, ssl->transform->in_cid_len);
        p += ssl->transform->in_cid_len;

        *p++ = ssl->transform->out_cid_len;
        memcpy(p, ssl->transform->out_cid, ssl->transform->out_cid_len);
        p += ssl->transform->out_cid_len;
    }
#endif /* MBEDTLS_SSL_DTLS_CONNECTION_ID */

    /*
     * Saved fields from top-level ssl_context structure
     */
#if defined(MBEDTLS_SSL_DTLS_BADMAC_LIMIT)
    used += 4;
    if (used <= buf_len) {
        MBEDTLS_PUT_UINT32_BE(ssl->badmac_seen, p, 0);
        p += 4;
    }
#endif /* MBEDTLS_SSL_DTLS_BADMAC_LIMIT */

#if defined(MBEDTLS_SSL_DTLS_ANTI_REPLAY)
    used += 16;
    if (used <= buf_len) {
        MBEDTLS_PUT_UINT64_BE(ssl->in_window_top, p, 0);
        p += 8;

        MBEDTLS_PUT_UINT64_BE(ssl->in_window, p, 0);
        p += 8;
    }
#endif /* MBEDTLS_SSL_DTLS_ANTI_REPLAY */

#if defined(MBEDTLS_SSL_PROTO_DTLS)
    used += 1;
    if (used <= buf_len) {
        *p++ = ssl->disable_datagram_packing;
    }
#endif /* MBEDTLS_SSL_PROTO_DTLS */

    used += 8;
    if (used <= buf_len) {
        memcpy(p, ssl->cur_out_ctr, 8);
        p += 8;
    }

#if defined(MBEDTLS_SSL_PROTO_DTLS)
    used += 2;
    if (used <= buf_len) {
        MBEDTLS_PUT_UINT16_BE(ssl->mtu, p, 0);
        p += 2;
    }
#endif /* MBEDTLS_SSL_PROTO_DTLS */

#if defined(MBEDTLS_SSL_ALPN)
    {
        const uint8_t alpn_len = ssl->alpn_chosen
                               ? (uint8_t) strlen(ssl->alpn_chosen)
                               : 0;

        used += 1 + alpn_len;
        if (used <= buf_len) {
            *p++ = alpn_len;

            if (ssl->alpn_chosen != NULL) {
                memcpy(p, ssl->alpn_chosen, alpn_len);
                p += alpn_len;
            }
        }
    }
#endif /* MBEDTLS_SSL_ALPN */

    /*
     * Done
     */
    *olen = used;

    if (used > buf_len) {
        return MBEDTLS_ERR_SSL_BUFFER_TOO_SMALL;
    }

    MBEDTLS_SSL_DEBUG_BUF(4, "saved context", buf, used);

    return mbedtls_ssl_session_reset_int(ssl, 0);
}

/*
 * Helper to get TLS 1.2 PRF from ciphersuite
 * (Duplicates bits of logic from ssl_set_handshake_prfs().)
 */
#if defined(MBEDTLS_SHA256_C) || \
    (defined(MBEDTLS_SHA512_C) && !defined(MBEDTLS_SHA512_NO_SHA384))
typedef int (*tls_prf_fn)(const unsigned char *secret, size_t slen,
                          const char *label,
                          const unsigned char *random, size_t rlen,
                          unsigned char *dstbuf, size_t dlen);
static tls_prf_fn ssl_tls12prf_from_cs(int ciphersuite_id)
{
    const mbedtls_ssl_ciphersuite_t * const ciphersuite_info =
        mbedtls_ssl_ciphersuite_from_id(ciphersuite_id);

    if (ciphersuite_info == NULL) {
        return NULL;
    }

#if defined(MBEDTLS_SHA512_C) && !defined(MBEDTLS_SHA512_NO_SHA384)
    if (ciphersuite_info->mac == MBEDTLS_MD_SHA384) {
        return tls_prf_sha384;
    } else
#endif
#if defined(MBEDTLS_SHA256_C)
    {
        if (ciphersuite_info->mac == MBEDTLS_MD_SHA256) {
            return tls_prf_sha256;
        }
    }
#endif
#if !defined(MBEDTLS_SHA256_C) && \
    (!defined(MBEDTLS_SHA512_C) || defined(MBEDTLS_SHA512_NO_SHA384))
    (void) ciphersuite_info;
#endif
    return NULL;
}

#endif /* MBEDTLS_SHA256_C ||
          (MBEDTLS_SHA512_C && !MBEDTLS_SHA512_NO_SHA384) */

/*
 * Deserialize context, see mbedtls_ssl_context_save() for format.
 *
 * This internal version is wrapped by a public function that cleans up in
 * case of error.
 */
MBEDTLS_CHECK_RETURN_CRITICAL
static int ssl_context_load(mbedtls_ssl_context *ssl,
                            const unsigned char *buf,
                            size_t len)
{
    const unsigned char *p = buf;
    const unsigned char * const end = buf + len;
    size_t session_len;
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    tls_prf_fn prf_func = NULL;

    /*
     * The context should have been freshly setup or reset.
     * Give the user an error in case of obvious misuse.
     * (Checking session is useful because it won't be NULL if we're
     * renegotiating, or if the user mistakenly loaded a session first.)
     */
    if (ssl->state != MBEDTLS_SSL_HELLO_REQUEST ||
        ssl->session != NULL) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    /*
     * We can't check that the config matches the initial one, but we can at
     * least check it matches the requirements for serializing.
     */
    if (ssl->conf->transport != MBEDTLS_SSL_TRANSPORT_DATAGRAM ||
        ssl->conf->max_major_ver < MBEDTLS_SSL_MAJOR_VERSION_3 ||
        ssl->conf->min_major_ver > MBEDTLS_SSL_MAJOR_VERSION_3 ||
        ssl->conf->max_minor_ver < MBEDTLS_SSL_MINOR_VERSION_3 ||
        ssl->conf->min_minor_ver > MBEDTLS_SSL_MINOR_VERSION_3 ||
#if defined(MBEDTLS_SSL_RENEGOTIATION)
        ssl->conf->disable_renegotiation != MBEDTLS_SSL_RENEGOTIATION_DISABLED ||
#endif
        0) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    MBEDTLS_SSL_DEBUG_BUF(4, "context to load", buf, len);

    /*
     * Check version identifier
     */
    if ((size_t) (end - p) < sizeof(ssl_serialized_context_header)) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    if (memcmp(p, ssl_serialized_context_header,
               sizeof(ssl_serialized_context_header)) != 0) {
        return MBEDTLS_ERR_SSL_VERSION_MISMATCH;
    }
    p += sizeof(ssl_serialized_context_header);

    /*
     * Session
     */
    if ((size_t) (end - p) < 4) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    session_len = ((size_t) p[0] << 24) |
                  ((size_t) p[1] << 16) |
                  ((size_t) p[2] <<  8) |
                  ((size_t) p[3]);
    p += 4;

    /* This has been allocated by ssl_handshake_init(), called by
     * by either mbedtls_ssl_session_reset_int() or mbedtls_ssl_setup(). */
    ssl->session = ssl->session_negotiate;
    ssl->session_in = ssl->session;
    ssl->session_out = ssl->session;
    ssl->session_negotiate = NULL;

    if ((size_t) (end - p) < session_len) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    ret = ssl_session_load(ssl->session, 1, p, session_len);
    if (ret != 0) {
        mbedtls_ssl_session_free(ssl->session);
        return ret;
    }

    p += session_len;

    /*
     * Transform
     */

    /* This has been allocated by ssl_handshake_init(), called by
     * by either mbedtls_ssl_session_reset_int() or mbedtls_ssl_setup(). */
    ssl->transform = ssl->transform_negotiate;
    ssl->transform_in = ssl->transform;
    ssl->transform_out = ssl->transform;
    ssl->transform_negotiate = NULL;

    prf_func = ssl_tls12prf_from_cs(ssl->session->ciphersuite);
    if (prf_func == NULL) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    /* Read random bytes and populate structure */
    if ((size_t) (end - p) < sizeof(ssl->transform->randbytes)) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    ret = ssl_populate_transform(ssl->transform,
                                 ssl->session->ciphersuite,
                                 ssl->session->master,
#if defined(MBEDTLS_SSL_SOME_MODES_USE_MAC)
#if defined(MBEDTLS_SSL_ENCRYPT_THEN_MAC)
                                 ssl->session->encrypt_then_mac,
#endif
#if defined(MBEDTLS_SSL_TRUNCATED_HMAC)
                                 ssl->session->trunc_hmac,
#endif
#endif /* MBEDTLS_SSL_SOME_MODES_USE_MAC */
#if defined(MBEDTLS_ZLIB_SUPPORT)
                                 ssl->session->compression,
#endif
                                 prf_func,
                                 p, /* currently pointing to randbytes */
                                 MBEDTLS_SSL_MINOR_VERSION_3, /* (D)TLS 1.2 is forced */
                                 ssl->conf->endpoint,
                                 ssl);
    if (ret != 0) {
        return ret;
    }

    p += sizeof(ssl->transform->randbytes);

#if defined(MBEDTLS_SSL_DTLS_CONNECTION_ID)
    /* Read connection IDs and store them */
    if ((size_t) (end - p) < 1) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    ssl->transform->in_cid_len = *p++;

    if ((size_t) (end - p) < ssl->transform->in_cid_len + 1u) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    memcpy(ssl->transform->in_cid, p, ssl->transform->in_cid_len);
    p += ssl->transform->in_cid_len;

    ssl->transform->out_cid_len = *p++;

    if ((size_t) (end - p) < ssl->transform->out_cid_len) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    memcpy(ssl->transform->out_cid, p, ssl->transform->out_cid_len);
    p += ssl->transform->out_cid_len;
#endif /* MBEDTLS_SSL_DTLS_CONNECTION_ID */

    /*
     * Saved fields from top-level ssl_context structure
     */
#if defined(MBEDTLS_SSL_DTLS_BADMAC_LIMIT)
    if ((size_t) (end - p) < 4) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    ssl->badmac_seen = ((uint32_t) p[0] << 24) |
                       ((uint32_t) p[1] << 16) |
                       ((uint32_t) p[2] <<  8) |
                       ((uint32_t) p[3]);
    p += 4;
#endif /* MBEDTLS_SSL_DTLS_BADMAC_LIMIT */

#if defined(MBEDTLS_SSL_DTLS_ANTI_REPLAY)
    if ((size_t) (end - p) < 16) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    ssl->in_window_top = ((uint64_t) p[0] << 56) |
                         ((uint64_t) p[1] << 48) |
                         ((uint64_t) p[2] << 40) |
                         ((uint64_t) p[3] << 32) |
                         ((uint64_t) p[4] << 24) |
                         ((uint64_t) p[5] << 16) |
                         ((uint64_t) p[6] <<  8) |
                         ((uint64_t) p[7]);
    p += 8;

    ssl->in_window = ((uint64_t) p[0] << 56) |
                     ((uint64_t) p[1] << 48) |
                     ((uint64_t) p[2] << 40) |
                     ((uint64_t) p[3] << 32) |
                     ((uint64_t) p[4] << 24) |
                     ((uint64_t) p[5] << 16) |
                     ((uint64_t) p[6] <<  8) |
                     ((uint64_t) p[7]);
    p += 8;
#endif /* MBEDTLS_SSL_DTLS_ANTI_REPLAY */

#if defined(MBEDTLS_SSL_PROTO_DTLS)
    if ((size_t) (end - p) < 1) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    ssl->disable_datagram_packing = *p++;
#endif /* MBEDTLS_SSL_PROTO_DTLS */

    if ((size_t) (end - p) < 8) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    memcpy(ssl->cur_out_ctr, p, 8);
    p += 8;

#if defined(MBEDTLS_SSL_PROTO_DTLS)
    if ((size_t) (end - p) < 2) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    ssl->mtu = (p[0] << 8) | p[1];
    p += 2;
#endif /* MBEDTLS_SSL_PROTO_DTLS */

#if defined(MBEDTLS_SSL_ALPN)
    {
        uint8_t alpn_len;
        const char **cur;

        if ((size_t) (end - p) < 1) {
            return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
        }

        alpn_len = *p++;

        if (alpn_len != 0 && ssl->conf->alpn_list != NULL) {
            /* alpn_chosen should point to an item in the configured list */
            for (cur = ssl->conf->alpn_list; *cur != NULL; cur++) {
                if (strlen(*cur) == alpn_len &&
                    memcmp(p, cur, alpn_len) == 0) {
                    ssl->alpn_chosen = *cur;
                    break;
                }
            }
        }

        /* can only happen on conf mismatch */
        if (alpn_len != 0 && ssl->alpn_chosen == NULL) {
            return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
        }

        p += alpn_len;
    }
#endif /* MBEDTLS_SSL_ALPN */

    /*
     * Forced fields from top-level ssl_context structure
     *
     * Most of them already set to the correct value by mbedtls_ssl_init() and
     * mbedtls_ssl_reset(), so we only need to set the remaining ones.
     */
    ssl->state = MBEDTLS_SSL_HANDSHAKE_OVER;

    ssl->major_ver = MBEDTLS_SSL_MAJOR_VERSION_3;
    ssl->minor_ver = MBEDTLS_SSL_MINOR_VERSION_3;

    /* Adjust pointers for header fields of outgoing records to
     * the given transform, accounting for explicit IV and CID. */
    mbedtls_ssl_update_out_pointers(ssl, ssl->transform);

#if defined(MBEDTLS_SSL_PROTO_DTLS)
    ssl->in_epoch = 1;
#endif

    /* mbedtls_ssl_reset() leaves the handshake sub-structure allocated,
     * which we don't want - otherwise we'd end up freeing the wrong transform
     * by calling mbedtls_ssl_handshake_wrapup_free_hs_transform()
     * inappropriately. */
    if (ssl->handshake != NULL) {
        mbedtls_ssl_handshake_free(ssl);
        mbedtls_free(ssl->handshake);
        ssl->handshake = NULL;
    }

    /*
     * Done - should have consumed entire buffer
     */
    if (p != end) {
        return MBEDTLS_ERR_SSL_BAD_INPUT_DATA;
    }

    return 0;
}

/*
 * Deserialize context: public wrapper for error cleaning
 */
int mbedtls_ssl_context_load(mbedtls_ssl_context *context,
                             const unsigned char *buf,
                             size_t len)
{
    int ret = ssl_context_load(context, buf, len);

    if (ret != 0) {
        mbedtls_ssl_free(context);
    }

    return ret;
}
#endif /* MBEDTLS_SSL_CONTEXT_SERIALIZATION */

/*
 * Free an SSL context
 */
void mbedtls_ssl_free(mbedtls_ssl_context *ssl)
{
    if (ssl == NULL) {
        return;
    }

    MBEDTLS_SSL_DEBUG_MSG(2, ("=> free"));

    if (ssl->out_buf != NULL) {
#if defined(MBEDTLS_SSL_VARIABLE_BUFFER_LENGTH)
        size_t out_buf_len = ssl->out_buf_len;
#else
        size_t out_buf_len = MBEDTLS_SSL_OUT_BUFFER_LEN;
#endif

        mbedtls_platform_zeroize(ssl->out_buf, out_buf_len);
        mbedtls_free(ssl->out_buf);
        ssl->out_buf = NULL;
    }

    if (ssl->in_buf != NULL) {
#if defined(MBEDTLS_SSL_VARIABLE_BUFFER_LENGTH)
        size_t in_buf_len = ssl->in_buf_len;
#else
        size_t in_buf_len = MBEDTLS_SSL_IN_BUFFER_LEN;
#endif

        mbedtls_platform_zeroize(ssl->in_buf, in_buf_len);
        mbedtls_free(ssl->in_buf);
        ssl->in_buf = NULL;
    }

#if defined(MBEDTLS_ZLIB_SUPPORT)
    if (ssl->compress_buf != NULL) {
        mbedtls_platform_zeroize(ssl->compress_buf, MBEDTLS_SSL_COMPRESS_BUFFER_LEN);
        mbedtls_free(ssl->compress_buf);
    }
#endif

    if (ssl->transform) {
        mbedtls_ssl_transform_free(ssl->transform);
        mbedtls_free(ssl->transform);
    }

    if (ssl->handshake) {
        mbedtls_ssl_handshake_free(ssl);
        mbedtls_ssl_transform_free(ssl->transform_negotiate);
        mbedtls_ssl_session_free(ssl->session_negotiate);

        mbedtls_free(ssl->handshake);
        mbedtls_free(ssl->transform_negotiate);
        mbedtls_free(ssl->session_negotiate);
    }

    if (ssl->session) {
        mbedtls_ssl_session_free(ssl->session);
        mbedtls_free(ssl->session);
    }

#if defined(MBEDTLS_X509_CRT_PARSE_C)
    if (ssl->hostname != NULL) {
        mbedtls_platform_zeroize(ssl->hostname, strlen(ssl->hostname));
        mbedtls_free(ssl->hostname);
    }
#endif

#if defined(MBEDTLS_SSL_HW_RECORD_ACCEL)
    if (mbedtls_ssl_hw_record_finish != NULL) {
        MBEDTLS_SSL_DEBUG_MSG(2, ("going for mbedtls_ssl_hw_record_finish()"));
        mbedtls_ssl_hw_record_finish(ssl);
    }
#endif

#if defined(MBEDTLS_SSL_DTLS_HELLO_VERIFY) && defined(MBEDTLS_SSL_SRV_C)
    mbedtls_free(ssl->cli_id);
#endif

    MBEDTLS_SSL_DEBUG_MSG(2, ("<= free"));

    /* Actually clear after last debug message */
    mbedtls_platform_zeroize(ssl, sizeof(mbedtls_ssl_context));
}

/*
 * Initialize mbedtls_ssl_config
 */
void mbedtls_ssl_config_init(mbedtls_ssl_config *conf)
{
    memset(conf, 0, sizeof(mbedtls_ssl_config));
}

#if defined(MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED)
static int ssl_preset_default_hashes[] = {
#if defined(MBEDTLS_SHA512_C)
    MBEDTLS_MD_SHA512,
#endif
#if defined(MBEDTLS_SHA512_C) && !defined(MBEDTLS_SHA512_NO_SHA384)
    MBEDTLS_MD_SHA384,
#endif
#if defined(MBEDTLS_SHA256_C)
    MBEDTLS_MD_SHA256,
    MBEDTLS_MD_SHA224,
#endif
#if defined(MBEDTLS_SHA1_C) && defined(MBEDTLS_TLS_DEFAULT_ALLOW_SHA1_IN_KEY_EXCHANGE)
    MBEDTLS_MD_SHA1,
#endif
    MBEDTLS_MD_NONE
};
#endif

static int ssl_preset_suiteb_ciphersuites[] = {
    MBEDTLS_TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
    MBEDTLS_TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
    0
};

#if defined(MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED)
static int ssl_preset_suiteb_hashes[] = {
    MBEDTLS_MD_SHA256,
    MBEDTLS_MD_SHA384,
    MBEDTLS_MD_NONE
};
#endif

#if defined(MBEDTLS_ECP_C)
static mbedtls_ecp_group_id ssl_preset_suiteb_curves[] = {
#if defined(MBEDTLS_ECP_DP_SECP256R1_ENABLED)
    MBEDTLS_ECP_DP_SECP256R1,
#endif
#if defined(MBEDTLS_ECP_DP_SECP384R1_ENABLED)
    MBEDTLS_ECP_DP_SECP384R1,
#endif
    MBEDTLS_ECP_DP_NONE
};
#endif

/*
 * Load default in mbedtls_ssl_config
 */
int mbedtls_ssl_config_defaults(mbedtls_ssl_config *conf,
                                int endpoint, int transport, int preset)
{
#if defined(MBEDTLS_DHM_C) && defined(MBEDTLS_SSL_SRV_C)
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
#endif

    /* Use the functions here so that they are covered in tests,
     * but otherwise access member directly for efficiency */
    mbedtls_ssl_conf_endpoint(conf, endpoint);
    mbedtls_ssl_conf_transport(conf, transport);

    /*
     * Things that are common to all presets
     */
#if defined(MBEDTLS_SSL_CLI_C)
    if (endpoint == MBEDTLS_SSL_IS_CLIENT) {
        conf->authmode = MBEDTLS_SSL_VERIFY_REQUIRED;
#if defined(MBEDTLS_SSL_SESSION_TICKETS)
        conf->session_tickets = MBEDTLS_SSL_SESSION_TICKETS_ENABLED;
#endif
    }
#endif

#if defined(MBEDTLS_ARC4_C)
    conf->arc4_disabled = MBEDTLS_SSL_ARC4_DISABLED;
#endif

#if defined(MBEDTLS_SSL_ENCRYPT_THEN_MAC)
    conf->encrypt_then_mac = MBEDTLS_SSL_ETM_ENABLED;
#endif

#if defined(MBEDTLS_SSL_EXTENDED_MASTER_SECRET)
    conf->extended_ms = MBEDTLS_SSL_EXTENDED_MS_ENABLED;
#endif

#if defined(MBEDTLS_SSL_CBC_RECORD_SPLITTING)
    conf->cbc_record_splitting = MBEDTLS_SSL_CBC_RECORD_SPLITTING_ENABLED;
#endif

#if defined(MBEDTLS_SSL_DTLS_HELLO_VERIFY) && defined(MBEDTLS_SSL_SRV_C)
    conf->f_cookie_write = ssl_cookie_write_dummy;
    conf->f_cookie_check = ssl_cookie_check_dummy;
#endif

#if defined(MBEDTLS_SSL_DTLS_ANTI_REPLAY)
    conf->anti_replay = MBEDTLS_SSL_ANTI_REPLAY_ENABLED;
#endif

#if defined(MBEDTLS_SSL_SRV_C)
    conf->cert_req_ca_list = MBEDTLS_SSL_CERT_REQ_CA_LIST_ENABLED;
#endif

#if defined(MBEDTLS_SSL_PROTO_DTLS)
    conf->hs_timeout_min = MBEDTLS_SSL_DTLS_TIMEOUT_DFL_MIN;
    conf->hs_timeout_max = MBEDTLS_SSL_DTLS_TIMEOUT_DFL_MAX;
#endif

#if defined(MBEDTLS_SSL_RENEGOTIATION)
    conf->renego_max_records = MBEDTLS_SSL_RENEGO_MAX_RECORDS_DEFAULT;
    memset(conf->renego_period,     0x00, 2);
    memset(conf->renego_period + 2, 0xFF, 6);
#endif

#if defined(MBEDTLS_DHM_C) && defined(MBEDTLS_SSL_SRV_C)
    if (endpoint == MBEDTLS_SSL_IS_SERVER) {
        const unsigned char dhm_p[] =
            MBEDTLS_DHM_RFC3526_MODP_2048_P_BIN;
        const unsigned char dhm_g[] =
            MBEDTLS_DHM_RFC3526_MODP_2048_G_BIN;

        if ((ret = mbedtls_ssl_conf_dh_param_bin(conf,
                                                 dhm_p, sizeof(dhm_p),
                                                 dhm_g, sizeof(dhm_g))) != 0) {
            return ret;
        }
    }
#endif

    /*
     * Preset-specific defaults
     */
    switch (preset) {
        /*
         * NSA Suite B
         */
        case MBEDTLS_SSL_PRESET_SUITEB:
            conf->min_major_ver = MBEDTLS_SSL_MAJOR_VERSION_3;
            conf->min_minor_ver = MBEDTLS_SSL_MINOR_VERSION_3; /* TLS 1.2 */
            conf->max_major_ver = MBEDTLS_SSL_MAX_MAJOR_VERSION;
            conf->max_minor_ver = MBEDTLS_SSL_MAX_MINOR_VERSION;

            conf->ciphersuite_list[MBEDTLS_SSL_MINOR_VERSION_0] =
                conf->ciphersuite_list[MBEDTLS_SSL_MINOR_VERSION_1] =
                    conf->ciphersuite_list[MBEDTLS_SSL_MINOR_VERSION_2] =
                        conf->ciphersuite_list[MBEDTLS_SSL_MINOR_VERSION_3] =
                            ssl_preset_suiteb_ciphersuites;

#if defined(MBEDTLS_X509_CRT_PARSE_C)
            conf->cert_profile = &mbedtls_x509_crt_profile_suiteb;
#endif

#if defined(MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED)
            conf->sig_hashes = ssl_preset_suiteb_hashes;
#endif

#if defined(MBEDTLS_ECP_C)
            conf->curve_list = ssl_preset_suiteb_curves;
#endif
            break;

        /*
         * Default
         */
        default:
            conf->min_major_ver = (MBEDTLS_SSL_MIN_MAJOR_VERSION >
                                   MBEDTLS_SSL_MIN_VALID_MAJOR_VERSION) ?
                                  MBEDTLS_SSL_MIN_MAJOR_VERSION :
                                  MBEDTLS_SSL_MIN_VALID_MAJOR_VERSION;
            conf->min_minor_ver = (MBEDTLS_SSL_MIN_MINOR_VERSION >
                                   MBEDTLS_SSL_MIN_VALID_MINOR_VERSION) ?
                                  MBEDTLS_SSL_MIN_MINOR_VERSION :
                                  MBEDTLS_SSL_MIN_VALID_MINOR_VERSION;
            conf->max_major_ver = MBEDTLS_SSL_MAX_MAJOR_VERSION;
            conf->max_minor_ver = MBEDTLS_SSL_MAX_MINOR_VERSION;

#if defined(MBEDTLS_SSL_PROTO_DTLS)
            if (transport == MBEDTLS_SSL_TRANSPORT_DATAGRAM) {
                conf->min_minor_ver = MBEDTLS_SSL_MINOR_VERSION_2;
            }
#endif

            conf->ciphersuite_list[MBEDTLS_SSL_MINOR_VERSION_0] =
                conf->ciphersuite_list[MBEDTLS_SSL_MINOR_VERSION_1] =
                    conf->ciphersuite_list[MBEDTLS_SSL_MINOR_VERSION_2] =
                        conf->ciphersuite_list[MBEDTLS_SSL_MINOR_VERSION_3] =
                            mbedtls_ssl_list_ciphersuites();

#if defined(MBEDTLS_X509_CRT_PARSE_C)
            conf->cert_profile = &mbedtls_x509_crt_profile_default;
#endif

#if defined(MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED)
            conf->sig_hashes = ssl_preset_default_hashes;
#endif

#if defined(MBEDTLS_ECP_C)
            conf->curve_list = mbedtls_ecp_grp_id_list();
#endif

#if defined(MBEDTLS_DHM_C) && defined(MBEDTLS_SSL_CLI_C)
            conf->dhm_min_bitlen = 1024;
#endif
    }

    return 0;
}

/*
 * Free mbedtls_ssl_config
 */
void mbedtls_ssl_config_free(mbedtls_ssl_config *conf)
{
#if defined(MBEDTLS_DHM_C)
    mbedtls_mpi_free(&conf->dhm_P);
    mbedtls_mpi_free(&conf->dhm_G);
#endif

#if defined(MBEDTLS_KEY_EXCHANGE_SOME_PSK_ENABLED)
    if (conf->psk != NULL) {
        mbedtls_platform_zeroize(conf->psk, conf->psk_len);
        mbedtls_free(conf->psk);
        conf->psk = NULL;
        conf->psk_len = 0;
    }

    if (conf->psk_identity != NULL) {
        mbedtls_platform_zeroize(conf->psk_identity, conf->psk_identity_len);
        mbedtls_free(conf->psk_identity);
        conf->psk_identity = NULL;
        conf->psk_identity_len = 0;
    }
#endif

#if defined(MBEDTLS_X509_CRT_PARSE_C)
    ssl_key_cert_free(conf->key_cert);
#endif

    mbedtls_platform_zeroize(conf, sizeof(mbedtls_ssl_config));
}

#if defined(MBEDTLS_PK_C) && \
    (defined(MBEDTLS_RSA_C) || defined(MBEDTLS_ECDSA_C))
/*
 * Convert between MBEDTLS_PK_XXX and SSL_SIG_XXX
 */
unsigned char mbedtls_ssl_sig_from_pk(mbedtls_pk_context *pk)
{
#if defined(MBEDTLS_RSA_C)
    if (mbedtls_pk_can_do(pk, MBEDTLS_PK_RSA)) {
        return MBEDTLS_SSL_SIG_RSA;
    }
#endif
#if defined(MBEDTLS_ECDSA_C)
    if (mbedtls_pk_can_do(pk, MBEDTLS_PK_ECDSA)) {
        return MBEDTLS_SSL_SIG_ECDSA;
    }
#endif
    return MBEDTLS_SSL_SIG_ANON;
}

unsigned char mbedtls_ssl_sig_from_pk_alg(mbedtls_pk_type_t type)
{
    switch (type) {
        case MBEDTLS_PK_RSA:
            return MBEDTLS_SSL_SIG_RSA;
        case MBEDTLS_PK_ECDSA:
        case MBEDTLS_PK_ECKEY:
            return MBEDTLS_SSL_SIG_ECDSA;
        default:
            return MBEDTLS_SSL_SIG_ANON;
    }
}

mbedtls_pk_type_t mbedtls_ssl_pk_alg_from_sig(unsigned char sig)
{
    switch (sig) {
#if defined(MBEDTLS_RSA_C)
        case MBEDTLS_SSL_SIG_RSA:
            return MBEDTLS_PK_RSA;
#endif
#if defined(MBEDTLS_ECDSA_C)
        case MBEDTLS_SSL_SIG_ECDSA:
            return MBEDTLS_PK_ECDSA;
#endif
        default:
            return MBEDTLS_PK_NONE;
    }
}
#endif /* MBEDTLS_PK_C && ( MBEDTLS_RSA_C || MBEDTLS_ECDSA_C ) */

#if defined(MBEDTLS_SSL_PROTO_TLS1_2) && \
    defined(MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED)

/* Find an entry in a signature-hash set matching a given hash algorithm. */
mbedtls_md_type_t mbedtls_ssl_sig_hash_set_find(mbedtls_ssl_sig_hash_set_t *set,
                                                mbedtls_pk_type_t sig_alg)
{
    switch (sig_alg) {
        case MBEDTLS_PK_RSA:
            return set->rsa;
        case MBEDTLS_PK_ECDSA:
            return set->ecdsa;
        default:
            return MBEDTLS_MD_NONE;
    }
}

/* Add a signature-hash-pair to a signature-hash set */
void mbedtls_ssl_sig_hash_set_add(mbedtls_ssl_sig_hash_set_t *set,
                                  mbedtls_pk_type_t sig_alg,
                                  mbedtls_md_type_t md_alg)
{
    switch (sig_alg) {
        case MBEDTLS_PK_RSA:
            if (set->rsa == MBEDTLS_MD_NONE) {
                set->rsa = md_alg;
            }
            break;

        case MBEDTLS_PK_ECDSA:
            if (set->ecdsa == MBEDTLS_MD_NONE) {
                set->ecdsa = md_alg;
            }
            break;

        default:
            break;
    }
}

/* Allow exactly one hash algorithm for each signature. */
void mbedtls_ssl_sig_hash_set_const_hash(mbedtls_ssl_sig_hash_set_t *set,
                                         mbedtls_md_type_t md_alg)
{
    set->rsa   = md_alg;
    set->ecdsa = md_alg;
}

#endif /* MBEDTLS_SSL_PROTO_TLS1_2) &&
          MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED */

/*
 * Convert from MBEDTLS_SSL_HASH_XXX to MBEDTLS_MD_XXX
 */
mbedtls_md_type_t mbedtls_ssl_md_alg_from_hash(unsigned char hash)
{
    switch (hash) {
#if defined(MBEDTLS_MD5_C)
        case MBEDTLS_SSL_HASH_MD5:
            return MBEDTLS_MD_MD5;
#endif
#if defined(MBEDTLS_SHA1_C)
        case MBEDTLS_SSL_HASH_SHA1:
            return MBEDTLS_MD_SHA1;
#endif
#if defined(MBEDTLS_SHA256_C)
        case MBEDTLS_SSL_HASH_SHA224:
            return MBEDTLS_MD_SHA224;
        case MBEDTLS_SSL_HASH_SHA256:
            return MBEDTLS_MD_SHA256;
#endif
#if defined(MBEDTLS_SHA512_C) && !defined(MBEDTLS_SHA512_NO_SHA384)
        case MBEDTLS_SSL_HASH_SHA384:
            return MBEDTLS_MD_SHA384;
#endif
#if defined(MBEDTLS_SHA512_C)
        case MBEDTLS_SSL_HASH_SHA512:
            return MBEDTLS_MD_SHA512;
#endif
        default:
            return MBEDTLS_MD_NONE;
    }
}

/*
 * Convert from MBEDTLS_MD_XXX to MBEDTLS_SSL_HASH_XXX
 */
unsigned char mbedtls_ssl_hash_from_md_alg(int md)
{
    switch (md) {
#if defined(MBEDTLS_MD5_C)
        case MBEDTLS_MD_MD5:
            return MBEDTLS_SSL_HASH_MD5;
#endif
#if defined(MBEDTLS_SHA1_C)
        case MBEDTLS_MD_SHA1:
            return MBEDTLS_SSL_HASH_SHA1;
#endif
#if defined(MBEDTLS_SHA256_C)
        case MBEDTLS_MD_SHA224:
            return MBEDTLS_SSL_HASH_SHA224;
        case MBEDTLS_MD_SHA256:
            return MBEDTLS_SSL_HASH_SHA256;
#endif
#if defined(MBEDTLS_SHA512_C) && !defined(MBEDTLS_SHA512_NO_SHA384)
        case MBEDTLS_MD_SHA384:
            return MBEDTLS_SSL_HASH_SHA384;
#endif
#if defined(MBEDTLS_SHA512_C)
        case MBEDTLS_MD_SHA512:
            return MBEDTLS_SSL_HASH_SHA512;
#endif
        default:
            return MBEDTLS_SSL_HASH_NONE;
    }
}

#if defined(MBEDTLS_ECP_C)
/*
 * Check if a curve proposed by the peer is in our list.
 * Return 0 if we're willing to use it, -1 otherwise.
 */
int mbedtls_ssl_check_curve(const mbedtls_ssl_context *ssl, mbedtls_ecp_group_id grp_id)
{
    const mbedtls_ecp_group_id *gid;

    if (ssl->conf->curve_list == NULL) {
        return -1;
    }

    for (gid = ssl->conf->curve_list; *gid != MBEDTLS_ECP_DP_NONE; gid++) {
        if (*gid == grp_id) {
            return 0;
        }
    }

    return -1;
}

/*
 * Same as mbedtls_ssl_check_curve() but takes a TLS ID for the curve.
 */
int mbedtls_ssl_check_curve_tls_id(const mbedtls_ssl_context *ssl, uint16_t tls_id)
{
    const mbedtls_ecp_curve_info *curve_info =
        mbedtls_ecp_curve_info_from_tls_id(tls_id);
    if (curve_info == NULL) {
        return -1;
    }
    return mbedtls_ssl_check_curve(ssl, curve_info->grp_id);
}
#endif /* MBEDTLS_ECP_C */

#if defined(MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED)
/*
 * Check if a hash proposed by the peer is in our list.
 * Return 0 if we're willing to use it, -1 otherwise.
 */
int mbedtls_ssl_check_sig_hash(const mbedtls_ssl_context *ssl,
                               mbedtls_md_type_t md)
{
    const int *cur;

    if (ssl->conf->sig_hashes == NULL) {
        return -1;
    }

    for (cur = ssl->conf->sig_hashes; *cur != MBEDTLS_MD_NONE; cur++) {
        if (*cur == (int) md) {
            return 0;
        }
    }

    return -1;
}
#endif /* MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED */

#if defined(MBEDTLS_X509_CRT_PARSE_C)
int mbedtls_ssl_check_cert_usage(const mbedtls_x509_crt *cert,
                                 const mbedtls_ssl_ciphersuite_t *ciphersuite,
                                 int cert_endpoint,
                                 uint32_t *flags)
{
    int ret = 0;
#if defined(MBEDTLS_X509_CHECK_KEY_USAGE)
    int usage = 0;
#endif
#if defined(MBEDTLS_X509_CHECK_EXTENDED_KEY_USAGE)
    const char *ext_oid;
    size_t ext_len;
#endif

#if !defined(MBEDTLS_X509_CHECK_KEY_USAGE) &&          \
    !defined(MBEDTLS_X509_CHECK_EXTENDED_KEY_USAGE)
    ((void) cert);
    ((void) cert_endpoint);
    ((void) flags);
#endif

#if defined(MBEDTLS_X509_CHECK_KEY_USAGE)
    if (cert_endpoint == MBEDTLS_SSL_IS_SERVER) {
        /* Server part of the key exchange */
        switch (ciphersuite->key_exchange) {
            case MBEDTLS_KEY_EXCHANGE_RSA:
            case MBEDTLS_KEY_EXCHANGE_RSA_PSK:
                usage = MBEDTLS_X509_KU_KEY_ENCIPHERMENT;
                break;

            case MBEDTLS_KEY_EXCHANGE_DHE_RSA:
            case MBEDTLS_KEY_EXCHANGE_ECDHE_RSA:
            case MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA:
                usage = MBEDTLS_X509_KU_DIGITAL_SIGNATURE;
                break;

            case MBEDTLS_KEY_EXCHANGE_ECDH_RSA:
            case MBEDTLS_KEY_EXCHANGE_ECDH_ECDSA:
                usage = MBEDTLS_X509_KU_KEY_AGREEMENT;
                break;

            /* Don't use default: we want warnings when adding new values */
            case MBEDTLS_KEY_EXCHANGE_NONE:
            case MBEDTLS_KEY_EXCHANGE_PSK:
            case MBEDTLS_KEY_EXCHANGE_DHE_PSK:
            case MBEDTLS_KEY_EXCHANGE_ECDHE_PSK:
            case MBEDTLS_KEY_EXCHANGE_ECJPAKE:
                usage = 0;
        }
    } else {
        /* Client auth: we only implement rsa_sign and mbedtls_ecdsa_sign for now */
        usage = MBEDTLS_X509_KU_DIGITAL_SIGNATURE;
    }

    if (mbedtls_x509_crt_check_key_usage(cert, usage) != 0) {
        *flags |= MBEDTLS_X509_BADCERT_KEY_USAGE;
        ret = -1;
    }
#else
    ((void) ciphersuite);
#endif /* MBEDTLS_X509_CHECK_KEY_USAGE */

#if defined(MBEDTLS_X509_CHECK_EXTENDED_KEY_USAGE)
    if (cert_endpoint == MBEDTLS_SSL_IS_SERVER) {
        ext_oid = MBEDTLS_OID_SERVER_AUTH;
        ext_len = MBEDTLS_OID_SIZE(MBEDTLS_OID_SERVER_AUTH);
    } else {
        ext_oid = MBEDTLS_OID_CLIENT_AUTH;
        ext_len = MBEDTLS_OID_SIZE(MBEDTLS_OID_CLIENT_AUTH);
    }

    if (mbedtls_x509_crt_check_extended_key_usage(cert, ext_oid, ext_len) != 0) {
        *flags |= MBEDTLS_X509_BADCERT_EXT_KEY_USAGE;
        ret = -1;
    }
#endif /* MBEDTLS_X509_CHECK_EXTENDED_KEY_USAGE */

    return ret;
}
#endif /* MBEDTLS_X509_CRT_PARSE_C */

int mbedtls_ssl_set_calc_verify_md(mbedtls_ssl_context *ssl, int md)
{
#if defined(MBEDTLS_SSL_PROTO_TLS1_2)
    if (ssl->minor_ver != MBEDTLS_SSL_MINOR_VERSION_3) {
        return MBEDTLS_ERR_SSL_INVALID_VERIFY_HASH;
    }

    switch (md) {
#if defined(MBEDTLS_SSL_PROTO_TLS1) || defined(MBEDTLS_SSL_PROTO_TLS1_1)
#if defined(MBEDTLS_MD5_C)
        case MBEDTLS_SSL_HASH_MD5:
            return MBEDTLS_ERR_SSL_INVALID_VERIFY_HASH;
#endif
#if defined(MBEDTLS_SHA1_C)
        case MBEDTLS_SSL_HASH_SHA1:
            ssl->handshake->calc_verify = ssl_calc_verify_tls;
            break;
#endif
#endif /* MBEDTLS_SSL_PROTO_TLS1 || MBEDTLS_SSL_PROTO_TLS1_1 */
#if defined(MBEDTLS_SHA512_C) && !defined(MBEDTLS_SHA512_NO_SHA384)
        case MBEDTLS_SSL_HASH_SHA384:
            ssl->handshake->calc_verify = ssl_calc_verify_tls_sha384;
            break;
#endif
#if defined(MBEDTLS_SHA256_C)
        case MBEDTLS_SSL_HASH_SHA256:
            ssl->handshake->calc_verify = ssl_calc_verify_tls_sha256;
            break;
#endif
        default:
            return MBEDTLS_ERR_SSL_INVALID_VERIFY_HASH;
    }

    return 0;
#else /* !MBEDTLS_SSL_PROTO_TLS1_2 */
    (void) ssl;
    (void) md;

    return MBEDTLS_ERR_SSL_INVALID_VERIFY_HASH;
#endif /* MBEDTLS_SSL_PROTO_TLS1_2 */
}

#if defined(MBEDTLS_SSL_PROTO_SSL3) || defined(MBEDTLS_SSL_PROTO_TLS1) || \
    defined(MBEDTLS_SSL_PROTO_TLS1_1)
int mbedtls_ssl_get_key_exchange_md_ssl_tls(mbedtls_ssl_context *ssl,
                                            unsigned char *output,
                                            unsigned char *data, size_t data_len)
{
    int ret = 0;
    mbedtls_md5_context mbedtls_md5;
    mbedtls_sha1_context mbedtls_sha1;

    mbedtls_md5_init(&mbedtls_md5);
    mbedtls_sha1_init(&mbedtls_sha1);

    /*
     * digitally-signed struct {
     *     opaque md5_hash[16];
     *     opaque sha_hash[20];
     * };
     *
     * md5_hash
     *     MD5(ClientHello.random + ServerHello.random
     *                            + ServerParams);
     * sha_hash
     *     SHA(ClientHello.random + ServerHello.random
     *                            + ServerParams);
     */
    if ((ret = mbedtls_md5_starts_ret(&mbedtls_md5)) != 0) {
        MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_md5_starts_ret", ret);
        goto exit;
    }
    if ((ret = mbedtls_md5_update_ret(&mbedtls_md5,
                                      ssl->handshake->randbytes, 64)) != 0) {
        MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_md5_update_ret", ret);
        goto exit;
    }
    if ((ret = mbedtls_md5_update_ret(&mbedtls_md5, data, data_len)) != 0) {
        MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_md5_update_ret", ret);
        goto exit;
    }
    if ((ret = mbedtls_md5_finish_ret(&mbedtls_md5, output)) != 0) {
        MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_md5_finish_ret", ret);
        goto exit;
    }

    if ((ret = mbedtls_sha1_starts_ret(&mbedtls_sha1)) != 0) {
        MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_sha1_starts_ret", ret);
        goto exit;
    }
    if ((ret = mbedtls_sha1_update_ret(&mbedtls_sha1,
                                       ssl->handshake->randbytes, 64)) != 0) {
        MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_sha1_update_ret", ret);
        goto exit;
    }
    if ((ret = mbedtls_sha1_update_ret(&mbedtls_sha1, data,
                                       data_len)) != 0) {
        MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_sha1_update_ret", ret);
        goto exit;
    }
    if ((ret = mbedtls_sha1_finish_ret(&mbedtls_sha1,
                                       output + 16)) != 0) {
        MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_sha1_finish_ret", ret);
        goto exit;
    }

exit:
    mbedtls_md5_free(&mbedtls_md5);
    mbedtls_sha1_free(&mbedtls_sha1);

    if (ret != 0) {
        mbedtls_ssl_send_alert_message(ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                       MBEDTLS_SSL_ALERT_MSG_INTERNAL_ERROR);
    }

    return ret;

}
#endif /* MBEDTLS_SSL_PROTO_SSL3 || MBEDTLS_SSL_PROTO_TLS1 || \
          MBEDTLS_SSL_PROTO_TLS1_1 */

#if defined(MBEDTLS_SSL_PROTO_TLS1) || defined(MBEDTLS_SSL_PROTO_TLS1_1) || \
    defined(MBEDTLS_SSL_PROTO_TLS1_2)

#if defined(MBEDTLS_USE_PSA_CRYPTO)
int mbedtls_ssl_get_key_exchange_md_tls1_2(mbedtls_ssl_context *ssl,
                                           unsigned char *hash, size_t *hashlen,
                                           unsigned char *data, size_t data_len,
                                           mbedtls_md_type_t md_alg)
{
    psa_status_t status;
    psa_hash_operation_t hash_operation = PSA_HASH_OPERATION_INIT;
    psa_algorithm_t hash_alg = mbedtls_psa_translate_md(md_alg);

    MBEDTLS_SSL_DEBUG_MSG(3, ("Perform PSA-based computation of digest of ServerKeyExchange"));

    if ((status = psa_hash_setup(&hash_operation,
                                 hash_alg)) != PSA_SUCCESS) {
        MBEDTLS_SSL_DEBUG_RET(1, "psa_hash_setup", status);
        goto exit;
    }

    if ((status = psa_hash_update(&hash_operation, ssl->handshake->randbytes,
                                  64)) != PSA_SUCCESS) {
        MBEDTLS_SSL_DEBUG_RET(1, "psa_hash_update", status);
        goto exit;
    }

    if ((status = psa_hash_update(&hash_operation,
                                  data, data_len)) != PSA_SUCCESS) {
        MBEDTLS_SSL_DEBUG_RET(1, "psa_hash_update", status);
        goto exit;
    }

    if ((status = psa_hash_finish(&hash_operation, hash, PSA_HASH_MAX_SIZE,
                                  hashlen)) != PSA_SUCCESS) {
        MBEDTLS_SSL_DEBUG_RET(1, "psa_hash_finish", status);
        goto exit;
    }

exit:
    if (status != PSA_SUCCESS) {
        mbedtls_ssl_send_alert_message(ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                       MBEDTLS_SSL_ALERT_MSG_INTERNAL_ERROR);
        switch (status) {
            case PSA_ERROR_NOT_SUPPORTED:
                return MBEDTLS_ERR_MD_FEATURE_UNAVAILABLE;
            case PSA_ERROR_BAD_STATE: /* Intentional fallthrough */
            case PSA_ERROR_BUFFER_TOO_SMALL:
                return MBEDTLS_ERR_MD_BAD_INPUT_DATA;
            case PSA_ERROR_INSUFFICIENT_MEMORY:
                return MBEDTLS_ERR_MD_ALLOC_FAILED;
            default:
                return MBEDTLS_ERR_MD_HW_ACCEL_FAILED;
        }
    }
    return 0;
}

#else

int mbedtls_ssl_get_key_exchange_md_tls1_2(mbedtls_ssl_context *ssl,
                                           unsigned char *hash, size_t *hashlen,
                                           unsigned char *data, size_t data_len,
                                           mbedtls_md_type_t md_alg)
{
    int ret = 0;
    mbedtls_md_context_t ctx;
    const mbedtls_md_info_t *md_info = mbedtls_md_info_from_type(md_alg);
    *hashlen = mbedtls_md_get_size(md_info);

    MBEDTLS_SSL_DEBUG_MSG(3, ("Perform mbedtls-based computation of digest of ServerKeyExchange"));

    mbedtls_md_init(&ctx);

    /*
     * digitally-signed struct {
     *     opaque client_random[32];
     *     opaque server_random[32];
     *     ServerDHParams params;
     * };
     */
    if ((ret = mbedtls_md_setup(&ctx, md_info, 0)) != 0) {
        MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_md_setup", ret);
        goto exit;
    }
    if ((ret = mbedtls_md_starts(&ctx)) != 0) {
        MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_md_starts", ret);
        goto exit;
    }
    if ((ret = mbedtls_md_update(&ctx, ssl->handshake->randbytes, 64)) != 0) {
        MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_md_update", ret);
        goto exit;
    }
    if ((ret = mbedtls_md_update(&ctx, data, data_len)) != 0) {
        MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_md_update", ret);
        goto exit;
    }
    if ((ret = mbedtls_md_finish(&ctx, hash)) != 0) {
        MBEDTLS_SSL_DEBUG_RET(1, "mbedtls_md_finish", ret);
        goto exit;
    }

exit:
    mbedtls_md_free(&ctx);

    if (ret != 0) {
        mbedtls_ssl_send_alert_message(ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                       MBEDTLS_SSL_ALERT_MSG_INTERNAL_ERROR);
    }

    return ret;
}
#endif /* MBEDTLS_USE_PSA_CRYPTO */

#endif /* MBEDTLS_SSL_PROTO_TLS1 || MBEDTLS_SSL_PROTO_TLS1_1 || \
          MBEDTLS_SSL_PROTO_TLS1_2 */

#endif /* MBEDTLS_SSL_TLS_C */
