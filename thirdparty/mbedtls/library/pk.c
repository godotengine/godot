/*
 *  Public Key abstraction layer
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

#include "common.h"

#if defined(MBEDTLS_PK_C)
#include "mbedtls/pk.h"
#include "mbedtls/pk_internal.h"

#include "mbedtls/platform_util.h"
#include "mbedtls/error.h"

#if defined(MBEDTLS_RSA_C)
#include "mbedtls/rsa.h"
#endif
#if defined(MBEDTLS_ECP_C)
#include "mbedtls/ecp.h"
#endif
#if defined(MBEDTLS_ECDSA_C)
#include "mbedtls/ecdsa.h"
#endif

#if defined(MBEDTLS_USE_PSA_CRYPTO)
#include "mbedtls/psa_util.h"
#endif

#include <limits.h>
#include <stdint.h>

/* Parameter validation macros based on platform_util.h */
#define PK_VALIDATE_RET(cond)    \
    MBEDTLS_INTERNAL_VALIDATE_RET(cond, MBEDTLS_ERR_PK_BAD_INPUT_DATA)
#define PK_VALIDATE(cond)        \
    MBEDTLS_INTERNAL_VALIDATE(cond)

/*
 * Initialise a mbedtls_pk_context
 */
void mbedtls_pk_init(mbedtls_pk_context *ctx)
{
    PK_VALIDATE(ctx != NULL);

    ctx->pk_info = NULL;
    ctx->pk_ctx = NULL;
}

/*
 * Free (the components of) a mbedtls_pk_context
 */
void mbedtls_pk_free(mbedtls_pk_context *ctx)
{
    if (ctx == NULL) {
        return;
    }

    if (ctx->pk_info != NULL) {
        ctx->pk_info->ctx_free_func(ctx->pk_ctx);
    }

    mbedtls_platform_zeroize(ctx, sizeof(mbedtls_pk_context));
}

#if defined(MBEDTLS_ECDSA_C) && defined(MBEDTLS_ECP_RESTARTABLE)
/*
 * Initialize a restart context
 */
void mbedtls_pk_restart_init(mbedtls_pk_restart_ctx *ctx)
{
    PK_VALIDATE(ctx != NULL);
    ctx->pk_info = NULL;
    ctx->rs_ctx = NULL;
}

/*
 * Free the components of a restart context
 */
void mbedtls_pk_restart_free(mbedtls_pk_restart_ctx *ctx)
{
    if (ctx == NULL || ctx->pk_info == NULL ||
        ctx->pk_info->rs_free_func == NULL) {
        return;
    }

    ctx->pk_info->rs_free_func(ctx->rs_ctx);

    ctx->pk_info = NULL;
    ctx->rs_ctx = NULL;
}
#endif /* MBEDTLS_ECDSA_C && MBEDTLS_ECP_RESTARTABLE */

/*
 * Get pk_info structure from type
 */
const mbedtls_pk_info_t *mbedtls_pk_info_from_type(mbedtls_pk_type_t pk_type)
{
    switch (pk_type) {
#if defined(MBEDTLS_RSA_C)
        case MBEDTLS_PK_RSA:
            return &mbedtls_rsa_info;
#endif
#if defined(MBEDTLS_ECP_C)
        case MBEDTLS_PK_ECKEY:
            return &mbedtls_eckey_info;
        case MBEDTLS_PK_ECKEY_DH:
            return &mbedtls_eckeydh_info;
#endif
#if defined(MBEDTLS_ECDSA_C)
        case MBEDTLS_PK_ECDSA:
            return &mbedtls_ecdsa_info;
#endif
        /* MBEDTLS_PK_RSA_ALT omitted on purpose */
        default:
            return NULL;
    }
}

/*
 * Initialise context
 */
int mbedtls_pk_setup(mbedtls_pk_context *ctx, const mbedtls_pk_info_t *info)
{
    PK_VALIDATE_RET(ctx != NULL);
    if (info == NULL || ctx->pk_info != NULL) {
        return MBEDTLS_ERR_PK_BAD_INPUT_DATA;
    }

    if ((ctx->pk_ctx = info->ctx_alloc_func()) == NULL) {
        return MBEDTLS_ERR_PK_ALLOC_FAILED;
    }

    ctx->pk_info = info;

    return 0;
}

#if defined(MBEDTLS_USE_PSA_CRYPTO)
/*
 * Initialise a PSA-wrapping context
 */
int mbedtls_pk_setup_opaque(mbedtls_pk_context *ctx,
                            const psa_key_id_t key)
{
    const mbedtls_pk_info_t * const info = &mbedtls_pk_opaque_info;
    psa_key_attributes_t attributes = PSA_KEY_ATTRIBUTES_INIT;
    psa_key_id_t *pk_ctx;
    psa_key_type_t type;

    if (ctx == NULL || ctx->pk_info != NULL) {
        return MBEDTLS_ERR_PK_BAD_INPUT_DATA;
    }

    if (PSA_SUCCESS != psa_get_key_attributes(key, &attributes)) {
        return MBEDTLS_ERR_PK_BAD_INPUT_DATA;
    }
    type = psa_get_key_type(&attributes);
    psa_reset_key_attributes(&attributes);

    /* Current implementation of can_do() relies on this. */
    if (!PSA_KEY_TYPE_IS_ECC_KEY_PAIR(type)) {
        return MBEDTLS_ERR_PK_FEATURE_UNAVAILABLE;
    }

    if ((ctx->pk_ctx = info->ctx_alloc_func()) == NULL) {
        return MBEDTLS_ERR_PK_ALLOC_FAILED;
    }

    ctx->pk_info = info;

    pk_ctx = (psa_key_id_t *) ctx->pk_ctx;
    *pk_ctx = key;

    return 0;
}
#endif /* MBEDTLS_USE_PSA_CRYPTO */

#if defined(MBEDTLS_PK_RSA_ALT_SUPPORT)
/*
 * Initialize an RSA-alt context
 */
int mbedtls_pk_setup_rsa_alt(mbedtls_pk_context *ctx, void *key,
                             mbedtls_pk_rsa_alt_decrypt_func decrypt_func,
                             mbedtls_pk_rsa_alt_sign_func sign_func,
                             mbedtls_pk_rsa_alt_key_len_func key_len_func)
{
    mbedtls_rsa_alt_context *rsa_alt;
    const mbedtls_pk_info_t *info = &mbedtls_rsa_alt_info;

    PK_VALIDATE_RET(ctx != NULL);
    if (ctx->pk_info != NULL) {
        return MBEDTLS_ERR_PK_BAD_INPUT_DATA;
    }

    if ((ctx->pk_ctx = info->ctx_alloc_func()) == NULL) {
        return MBEDTLS_ERR_PK_ALLOC_FAILED;
    }

    ctx->pk_info = info;

    rsa_alt = (mbedtls_rsa_alt_context *) ctx->pk_ctx;

    rsa_alt->key = key;
    rsa_alt->decrypt_func = decrypt_func;
    rsa_alt->sign_func = sign_func;
    rsa_alt->key_len_func = key_len_func;

    return 0;
}
#endif /* MBEDTLS_PK_RSA_ALT_SUPPORT */

/*
 * Tell if a PK can do the operations of the given type
 */
int mbedtls_pk_can_do(const mbedtls_pk_context *ctx, mbedtls_pk_type_t type)
{
    /* A context with null pk_info is not set up yet and can't do anything.
     * For backward compatibility, also accept NULL instead of a context
     * pointer. */
    if (ctx == NULL || ctx->pk_info == NULL) {
        return 0;
    }

    return ctx->pk_info->can_do(type);
}

/*
 * Helper for mbedtls_pk_sign and mbedtls_pk_verify
 */
static inline int pk_hashlen_helper(mbedtls_md_type_t md_alg, size_t *hash_len)
{
    const mbedtls_md_info_t *md_info;

    if (*hash_len != 0 && md_alg == MBEDTLS_MD_NONE) {
        return 0;
    }

    if ((md_info = mbedtls_md_info_from_type(md_alg)) == NULL) {
        return -1;
    }

    if (*hash_len != 0 && *hash_len != mbedtls_md_get_size(md_info)) {
        return -1;
    }

    *hash_len = mbedtls_md_get_size(md_info);
    return 0;
}

#if defined(MBEDTLS_ECDSA_C) && defined(MBEDTLS_ECP_RESTARTABLE)
/*
 * Helper to set up a restart context if needed
 */
static int pk_restart_setup(mbedtls_pk_restart_ctx *ctx,
                            const mbedtls_pk_info_t *info)
{
    /* Don't do anything if already set up or invalid */
    if (ctx == NULL || ctx->pk_info != NULL) {
        return 0;
    }

    /* Should never happen when we're called */
    if (info->rs_alloc_func == NULL || info->rs_free_func == NULL) {
        return MBEDTLS_ERR_PK_BAD_INPUT_DATA;
    }

    if ((ctx->rs_ctx = info->rs_alloc_func()) == NULL) {
        return MBEDTLS_ERR_PK_ALLOC_FAILED;
    }

    ctx->pk_info = info;

    return 0;
}
#endif /* MBEDTLS_ECDSA_C && MBEDTLS_ECP_RESTARTABLE */

/*
 * Verify a signature (restartable)
 */
int mbedtls_pk_verify_restartable(mbedtls_pk_context *ctx,
                                  mbedtls_md_type_t md_alg,
                                  const unsigned char *hash, size_t hash_len,
                                  const unsigned char *sig, size_t sig_len,
                                  mbedtls_pk_restart_ctx *rs_ctx)
{
    PK_VALIDATE_RET(ctx != NULL);
    PK_VALIDATE_RET((md_alg == MBEDTLS_MD_NONE && hash_len == 0) ||
                    hash != NULL);
    PK_VALIDATE_RET(sig != NULL);

    if (ctx->pk_info == NULL ||
        pk_hashlen_helper(md_alg, &hash_len) != 0) {
        return MBEDTLS_ERR_PK_BAD_INPUT_DATA;
    }

#if defined(MBEDTLS_ECDSA_C) && defined(MBEDTLS_ECP_RESTARTABLE)
    /* optimization: use non-restartable version if restart disabled */
    if (rs_ctx != NULL &&
        mbedtls_ecp_restart_is_enabled() &&
        ctx->pk_info->verify_rs_func != NULL) {
        int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;

        if ((ret = pk_restart_setup(rs_ctx, ctx->pk_info)) != 0) {
            return ret;
        }

        ret = ctx->pk_info->verify_rs_func(ctx->pk_ctx,
                                           md_alg, hash, hash_len, sig, sig_len, rs_ctx->rs_ctx);

        if (ret != MBEDTLS_ERR_ECP_IN_PROGRESS) {
            mbedtls_pk_restart_free(rs_ctx);
        }

        return ret;
    }
#else /* MBEDTLS_ECDSA_C && MBEDTLS_ECP_RESTARTABLE */
    (void) rs_ctx;
#endif /* MBEDTLS_ECDSA_C && MBEDTLS_ECP_RESTARTABLE */

    if (ctx->pk_info->verify_func == NULL) {
        return MBEDTLS_ERR_PK_TYPE_MISMATCH;
    }

    return ctx->pk_info->verify_func(ctx->pk_ctx, md_alg, hash, hash_len,
                                     sig, sig_len);
}

/*
 * Verify a signature
 */
int mbedtls_pk_verify(mbedtls_pk_context *ctx, mbedtls_md_type_t md_alg,
                      const unsigned char *hash, size_t hash_len,
                      const unsigned char *sig, size_t sig_len)
{
    return mbedtls_pk_verify_restartable(ctx, md_alg, hash, hash_len,
                                         sig, sig_len, NULL);
}

/*
 * Verify a signature with options
 */
int mbedtls_pk_verify_ext(mbedtls_pk_type_t type, const void *options,
                          mbedtls_pk_context *ctx, mbedtls_md_type_t md_alg,
                          const unsigned char *hash, size_t hash_len,
                          const unsigned char *sig, size_t sig_len)
{
    PK_VALIDATE_RET(ctx != NULL);
    PK_VALIDATE_RET((md_alg == MBEDTLS_MD_NONE && hash_len == 0) ||
                    hash != NULL);
    PK_VALIDATE_RET(sig != NULL);

    if (ctx->pk_info == NULL) {
        return MBEDTLS_ERR_PK_BAD_INPUT_DATA;
    }

    if (!mbedtls_pk_can_do(ctx, type)) {
        return MBEDTLS_ERR_PK_TYPE_MISMATCH;
    }

    if (type == MBEDTLS_PK_RSASSA_PSS) {
#if defined(MBEDTLS_RSA_C) && defined(MBEDTLS_PKCS1_V21)
        int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
        const mbedtls_pk_rsassa_pss_options *pss_opts;

#if SIZE_MAX > UINT_MAX
        if (md_alg == MBEDTLS_MD_NONE && UINT_MAX < hash_len) {
            return MBEDTLS_ERR_PK_BAD_INPUT_DATA;
        }
#endif /* SIZE_MAX > UINT_MAX */

        if (options == NULL) {
            return MBEDTLS_ERR_PK_BAD_INPUT_DATA;
        }

        pss_opts = (const mbedtls_pk_rsassa_pss_options *) options;

        if (sig_len < mbedtls_pk_get_len(ctx)) {
            return MBEDTLS_ERR_RSA_VERIFY_FAILED;
        }

        ret = mbedtls_rsa_rsassa_pss_verify_ext(mbedtls_pk_rsa(*ctx),
                                                NULL, NULL, MBEDTLS_RSA_PUBLIC,
                                                md_alg, (unsigned int) hash_len, hash,
                                                pss_opts->mgf1_hash_id,
                                                pss_opts->expected_salt_len,
                                                sig);
        if (ret != 0) {
            return ret;
        }

        if (sig_len > mbedtls_pk_get_len(ctx)) {
            return MBEDTLS_ERR_PK_SIG_LEN_MISMATCH;
        }

        return 0;
#else
        return MBEDTLS_ERR_PK_FEATURE_UNAVAILABLE;
#endif /* MBEDTLS_RSA_C && MBEDTLS_PKCS1_V21 */
    }

    /* General case: no options */
    if (options != NULL) {
        return MBEDTLS_ERR_PK_BAD_INPUT_DATA;
    }

    return mbedtls_pk_verify(ctx, md_alg, hash, hash_len, sig, sig_len);
}

/*
 * Make a signature (restartable)
 */
int mbedtls_pk_sign_restartable(mbedtls_pk_context *ctx,
                                mbedtls_md_type_t md_alg,
                                const unsigned char *hash, size_t hash_len,
                                unsigned char *sig, size_t *sig_len,
                                int (*f_rng)(void *, unsigned char *, size_t), void *p_rng,
                                mbedtls_pk_restart_ctx *rs_ctx)
{
    PK_VALIDATE_RET(ctx != NULL);
    PK_VALIDATE_RET((md_alg == MBEDTLS_MD_NONE && hash_len == 0) ||
                    hash != NULL);
    PK_VALIDATE_RET(sig != NULL);

    if (ctx->pk_info == NULL ||
        pk_hashlen_helper(md_alg, &hash_len) != 0) {
        return MBEDTLS_ERR_PK_BAD_INPUT_DATA;
    }

#if defined(MBEDTLS_ECDSA_C) && defined(MBEDTLS_ECP_RESTARTABLE)
    /* optimization: use non-restartable version if restart disabled */
    if (rs_ctx != NULL &&
        mbedtls_ecp_restart_is_enabled() &&
        ctx->pk_info->sign_rs_func != NULL) {
        int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;

        if ((ret = pk_restart_setup(rs_ctx, ctx->pk_info)) != 0) {
            return ret;
        }

        ret = ctx->pk_info->sign_rs_func(ctx->pk_ctx, md_alg,
                                         hash, hash_len, sig, sig_len, f_rng, p_rng,
                                         rs_ctx->rs_ctx);

        if (ret != MBEDTLS_ERR_ECP_IN_PROGRESS) {
            mbedtls_pk_restart_free(rs_ctx);
        }

        return ret;
    }
#else /* MBEDTLS_ECDSA_C && MBEDTLS_ECP_RESTARTABLE */
    (void) rs_ctx;
#endif /* MBEDTLS_ECDSA_C && MBEDTLS_ECP_RESTARTABLE */

    if (ctx->pk_info->sign_func == NULL) {
        return MBEDTLS_ERR_PK_TYPE_MISMATCH;
    }

    return ctx->pk_info->sign_func(ctx->pk_ctx, md_alg, hash, hash_len,
                                   sig, sig_len, f_rng, p_rng);
}

/*
 * Make a signature
 */
int mbedtls_pk_sign(mbedtls_pk_context *ctx, mbedtls_md_type_t md_alg,
                    const unsigned char *hash, size_t hash_len,
                    unsigned char *sig, size_t *sig_len,
                    int (*f_rng)(void *, unsigned char *, size_t), void *p_rng)
{
    return mbedtls_pk_sign_restartable(ctx, md_alg, hash, hash_len,
                                       sig, sig_len, f_rng, p_rng, NULL);
}

/*
 * Decrypt message
 */
int mbedtls_pk_decrypt(mbedtls_pk_context *ctx,
                       const unsigned char *input, size_t ilen,
                       unsigned char *output, size_t *olen, size_t osize,
                       int (*f_rng)(void *, unsigned char *, size_t), void *p_rng)
{
    PK_VALIDATE_RET(ctx != NULL);
    PK_VALIDATE_RET(input != NULL || ilen == 0);
    PK_VALIDATE_RET(output != NULL || osize == 0);
    PK_VALIDATE_RET(olen != NULL);

    if (ctx->pk_info == NULL) {
        return MBEDTLS_ERR_PK_BAD_INPUT_DATA;
    }

    if (ctx->pk_info->decrypt_func == NULL) {
        return MBEDTLS_ERR_PK_TYPE_MISMATCH;
    }

    return ctx->pk_info->decrypt_func(ctx->pk_ctx, input, ilen,
                                      output, olen, osize, f_rng, p_rng);
}

/*
 * Encrypt message
 */
int mbedtls_pk_encrypt(mbedtls_pk_context *ctx,
                       const unsigned char *input, size_t ilen,
                       unsigned char *output, size_t *olen, size_t osize,
                       int (*f_rng)(void *, unsigned char *, size_t), void *p_rng)
{
    PK_VALIDATE_RET(ctx != NULL);
    PK_VALIDATE_RET(input != NULL || ilen == 0);
    PK_VALIDATE_RET(output != NULL || osize == 0);
    PK_VALIDATE_RET(olen != NULL);

    if (ctx->pk_info == NULL) {
        return MBEDTLS_ERR_PK_BAD_INPUT_DATA;
    }

    if (ctx->pk_info->encrypt_func == NULL) {
        return MBEDTLS_ERR_PK_TYPE_MISMATCH;
    }

    return ctx->pk_info->encrypt_func(ctx->pk_ctx, input, ilen,
                                      output, olen, osize, f_rng, p_rng);
}

/*
 * Check public-private key pair
 */
int mbedtls_pk_check_pair(const mbedtls_pk_context *pub, const mbedtls_pk_context *prv)
{
    PK_VALIDATE_RET(pub != NULL);
    PK_VALIDATE_RET(prv != NULL);

    if (pub->pk_info == NULL ||
        prv->pk_info == NULL) {
        return MBEDTLS_ERR_PK_BAD_INPUT_DATA;
    }

    if (prv->pk_info->check_pair_func == NULL) {
        return MBEDTLS_ERR_PK_FEATURE_UNAVAILABLE;
    }

    if (prv->pk_info->type == MBEDTLS_PK_RSA_ALT) {
        if (pub->pk_info->type != MBEDTLS_PK_RSA) {
            return MBEDTLS_ERR_PK_TYPE_MISMATCH;
        }
    } else {
        if (pub->pk_info != prv->pk_info) {
            return MBEDTLS_ERR_PK_TYPE_MISMATCH;
        }
    }

    return prv->pk_info->check_pair_func(pub->pk_ctx, prv->pk_ctx);
}

/*
 * Get key size in bits
 */
size_t mbedtls_pk_get_bitlen(const mbedtls_pk_context *ctx)
{
    /* For backward compatibility, accept NULL or a context that
     * isn't set up yet, and return a fake value that should be safe. */
    if (ctx == NULL || ctx->pk_info == NULL) {
        return 0;
    }

    return ctx->pk_info->get_bitlen(ctx->pk_ctx);
}

/*
 * Export debug information
 */
int mbedtls_pk_debug(const mbedtls_pk_context *ctx, mbedtls_pk_debug_item *items)
{
    PK_VALIDATE_RET(ctx != NULL);
    if (ctx->pk_info == NULL) {
        return MBEDTLS_ERR_PK_BAD_INPUT_DATA;
    }

    if (ctx->pk_info->debug_func == NULL) {
        return MBEDTLS_ERR_PK_TYPE_MISMATCH;
    }

    ctx->pk_info->debug_func(ctx->pk_ctx, items);
    return 0;
}

/*
 * Access the PK type name
 */
const char *mbedtls_pk_get_name(const mbedtls_pk_context *ctx)
{
    if (ctx == NULL || ctx->pk_info == NULL) {
        return "invalid PK";
    }

    return ctx->pk_info->name;
}

/*
 * Access the PK type
 */
mbedtls_pk_type_t mbedtls_pk_get_type(const mbedtls_pk_context *ctx)
{
    if (ctx == NULL || ctx->pk_info == NULL) {
        return MBEDTLS_PK_NONE;
    }

    return ctx->pk_info->type;
}

#if defined(MBEDTLS_USE_PSA_CRYPTO)
/*
 * Load the key to a PSA key slot,
 * then turn the PK context into a wrapper for that key slot.
 *
 * Currently only works for EC private keys.
 */
int mbedtls_pk_wrap_as_opaque(mbedtls_pk_context *pk,
                              psa_key_id_t *key,
                              psa_algorithm_t hash_alg)
{
#if !defined(MBEDTLS_ECP_C)
    ((void) pk);
    ((void) key);
    ((void) hash_alg);
    return MBEDTLS_ERR_PK_TYPE_MISMATCH;
#else
    const mbedtls_ecp_keypair *ec;
    unsigned char d[MBEDTLS_ECP_MAX_BYTES];
    size_t d_len;
    psa_ecc_family_t curve_id;
    psa_key_attributes_t attributes = PSA_KEY_ATTRIBUTES_INIT;
    psa_key_type_t key_type;
    size_t bits;
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    psa_status_t status;

    /* export the private key material in the format PSA wants */
    if (mbedtls_pk_get_type(pk) != MBEDTLS_PK_ECKEY) {
        return MBEDTLS_ERR_PK_TYPE_MISMATCH;
    }

    ec = mbedtls_pk_ec(*pk);
    d_len = (ec->grp.nbits + 7) / 8;
    if ((ret = mbedtls_mpi_write_binary(&ec->d, d, d_len)) != 0) {
        return ret;
    }

    curve_id = mbedtls_ecc_group_to_psa(ec->grp.id, &bits);
    key_type = PSA_KEY_TYPE_ECC_KEY_PAIR(curve_id);

    /* prepare the key attributes */
    psa_set_key_type(&attributes, key_type);
    psa_set_key_bits(&attributes, bits);
    psa_set_key_usage_flags(&attributes, PSA_KEY_USAGE_SIGN_HASH);
    psa_set_key_algorithm(&attributes, PSA_ALG_ECDSA(hash_alg));

    /* import private key into PSA */
    status = psa_import_key(&attributes, d, d_len, key);
    mbedtls_platform_zeroize(d, sizeof(d));
    if (status != PSA_SUCCESS) {
        return MBEDTLS_ERR_PK_HW_ACCEL_FAILED;
    }

    /* make PK context wrap the key slot */
    mbedtls_pk_free(pk);
    mbedtls_pk_init(pk);

    return mbedtls_pk_setup_opaque(pk, *key);
#endif /* MBEDTLS_ECP_C */
}
#endif /* MBEDTLS_USE_PSA_CRYPTO */
#endif /* MBEDTLS_PK_C */
