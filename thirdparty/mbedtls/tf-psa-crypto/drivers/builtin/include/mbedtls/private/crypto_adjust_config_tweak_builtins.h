/**
 * \file mbedtls/private/crypto_adjust_config_tweak_builtins.h
 * \brief Adjust macros used by legacy built-in crypto modules
 *
 * This is an internal header. Do not include it directly.
 *
 * Automatically enable certain parts of the cryptography implementation
 * that are required by other parts. Also define some internal symbols
 * that are derived from public ones. This file is about individual
 * modules that lie below PSA, not about the PSA configuration.
 */
/*
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */

#ifndef TF_PSA_CRYPTO_MBEDTLS_PRIVATE_CRYPTO_ADJUST_CONFIG_TWEAK_BUILTINS_H
#define TF_PSA_CRYPTO_MBEDTLS_PRIVATE_CRYPTO_ADJUST_CONFIG_TWEAK_BUILTINS_H

/**
 * \def MBEDTLS_USE_PSA_CRYPTO
 *
 * Make the X.509 and TLS libraries use PSA for cryptographic operations as
 * much as possible, and enable new APIs for using keys handled by PSA Crypto.
 *
 * \note This is a legacy symbol which still exists for backward compatibility.
 *       Up to Mbed TLS 3.x, it was not enabled by default. Now it is always
 *       enabled, and it will eventually disappear from the code base. This
 *       is not part of the public API of TF-PSA-Crypto or of Mbed TLS >=4.0.
 */
#define MBEDTLS_USE_PSA_CRYPTO

/* Whether any hash based on sha3 is enabled in psa_crypto_hash.c. */
#if defined(MBEDTLS_PSA_BUILTIN_ALG_SHA3_224) || \
    defined(MBEDTLS_PSA_BUILTIN_ALG_SHA3_256) || \
    defined(MBEDTLS_PSA_BUILTIN_ALG_SHA3_384) || \
    defined(MBEDTLS_PSA_BUILTIN_ALG_SHA3_512)
#define MBEDTLS_PSA_BUILTIN_ALG_SHA3_SOME_HASH
#endif

/* Whether any XOF based on sha3 is enabled in psa_crypto_xof.c. */
#if defined(MBEDTLS_PSA_BUILTIN_ALG_SHAKE128) || \
    defined(MBEDTLS_PSA_BUILTIN_ALG_SHAKE256)
#define MBEDTLS_PSA_BUILTIN_ALG_SOME_SHAKE
#endif

/* If a SHAKE variant is enabled in psa_crypto_xof.c, tell sha3.c that we
 * want it.
 *
 * Note that the PSA API (following NIST standards) defines hash algorithms
 * that are SHAKE128 or SHAKE256 with a specific output lengths. From the
 * perspective of sha3.c, these are just users of SHAKE128/SHAKE256, but
 * from the perspective of psa_crypto_hash.c and psa_crypto_xof.c,
 * they are hashes and not XOF. So, for example, if the SHAKE256/512 hash
 * algorithm is enabled in the PSA API (for Ed448ph) but the SHAKE256 XOF
 * algorithm is disabled, then MBEDTLS_PSA_BUILTIN_ALG_SHAKE256 will be
 * disabled but we'll still need to enable MBEDTLS_SHA3_WANT_SHAKE256.
 */
#if defined(MBEDTLS_PSA_BUILTIN_ALG_SHAKE128)
#define MBEDTLS_SHA3_WANT_SHAKE128
#endif
#if defined(MBEDTLS_PSA_BUILTIN_ALG_SHAKE256)
#define MBEDTLS_SHA3_WANT_SHAKE256
#endif

/* Whether any Keccak variant is enabled, i.e. the bulk of sha3.c. */
#if defined(MBEDTLS_PSA_BUILTIN_ALG_SHA3_SOME_HASH) || \
    defined(MBEDTLS_SHA3_WANT_SHAKE128) || defined(MBEDTLS_SHA3_WANT_SHAKE256)
#define MBEDTLS_SHA3_C
#endif

/* Auto-enable CIPHER_C when any of the unauthenticated ciphers is builtin
 * in PSA. */
#if defined(MBEDTLS_PSA_CRYPTO_C) && \
    (defined(MBEDTLS_PSA_BUILTIN_ALG_STREAM_CIPHER) || \
    defined(MBEDTLS_PSA_BUILTIN_ALG_CTR) || \
    defined(MBEDTLS_PSA_BUILTIN_ALG_CFB) || \
    defined(MBEDTLS_PSA_BUILTIN_ALG_OFB) || \
    defined(MBEDTLS_PSA_BUILTIN_ALG_ECB_NO_PADDING) || \
    defined(MBEDTLS_PSA_BUILTIN_ALG_CBC_NO_PADDING) || \
    defined(MBEDTLS_PSA_BUILTIN_ALG_CBC_PKCS7) || \
    defined(MBEDTLS_PSA_BUILTIN_ALG_CCM_STAR_NO_TAG) || \
    defined(MBEDTLS_PSA_BUILTIN_ALG_CMAC))
#define MBEDTLS_CIPHER_C
#endif

/* BLOCK_CIPHER module can dispatch to PSA when:
 * - PSA is enabled and drivers have been initialized
 * - desired key type is supported on the PSA side
 * If the above conditions are not met, but the legacy support is enabled, then
 * BLOCK_CIPHER will dynamically fallback to it.
 *
 * In case BLOCK_CIPHER is defined (see below) the following symbols/helpers
 * can be used to define its capabilities:
 * - MBEDTLS_BLOCK_CIPHER_SOME_PSA: there is at least 1 key type between AES,
 *   ARIA and Camellia which is supported through a driver;
 * - MBEDTLS_BLOCK_CIPHER_xxx_VIA_PSA: xxx key type is supported through a
 *   driver;
 * - MBEDTLS_BLOCK_CIPHER_xxx_VIA_LEGACY: xxx key type is supported through
 *   a legacy module (i.e. MBEDTLS_xxx_C)
 */
#if defined(MBEDTLS_PSA_CRYPTO_C)
#if defined(MBEDTLS_PSA_ACCEL_KEY_TYPE_AES)
#define MBEDTLS_BLOCK_CIPHER_AES_VIA_PSA
#define MBEDTLS_BLOCK_CIPHER_SOME_PSA
#endif
#if defined(MBEDTLS_PSA_ACCEL_KEY_TYPE_ARIA)
#define MBEDTLS_BLOCK_CIPHER_ARIA_VIA_PSA
#define MBEDTLS_BLOCK_CIPHER_SOME_PSA
#endif
#if defined(MBEDTLS_PSA_ACCEL_KEY_TYPE_CAMELLIA)
#define MBEDTLS_BLOCK_CIPHER_CAMELLIA_VIA_PSA
#define MBEDTLS_BLOCK_CIPHER_SOME_PSA
#endif
#endif /* MBEDTLS_PSA_CRYPTO_C */

#if defined(MBEDTLS_AES_C)
#define MBEDTLS_BLOCK_CIPHER_AES_VIA_LEGACY
#endif
#if defined(MBEDTLS_ARIA_C)
#define MBEDTLS_BLOCK_CIPHER_ARIA_VIA_LEGACY
#endif
#if defined(MBEDTLS_CAMELLIA_C)
#define MBEDTLS_BLOCK_CIPHER_CAMELLIA_VIA_LEGACY
#endif

/* Helpers to state that BLOCK_CIPHER module supports AES, ARIA and/or Camellia
 * block ciphers via either PSA or legacy. */
#if defined(MBEDTLS_BLOCK_CIPHER_AES_VIA_PSA) || \
    defined(MBEDTLS_BLOCK_CIPHER_AES_VIA_LEGACY)
#define MBEDTLS_BLOCK_CIPHER_CAN_AES
#endif
#if defined(MBEDTLS_BLOCK_CIPHER_ARIA_VIA_PSA) || \
    defined(MBEDTLS_BLOCK_CIPHER_ARIA_VIA_LEGACY)
#define MBEDTLS_BLOCK_CIPHER_CAN_ARIA
#endif
#if defined(MBEDTLS_BLOCK_CIPHER_CAMELLIA_VIA_PSA) || \
    defined(MBEDTLS_BLOCK_CIPHER_CAMELLIA_VIA_LEGACY)
#define MBEDTLS_BLOCK_CIPHER_CAN_CAMELLIA
#endif

/* GCM_C and CCM_C can either depend on (in order of preference) BLOCK_CIPHER_C
 * or CIPHER_C. The former is auto-enabled when:
 * - CIPHER_C is not defined, which is also the legacy solution;
 * - BLOCK_CIPHER_SOME_PSA because in this case BLOCK_CIPHER can take advantage
 *   of the driver's acceleration.
 */
#if (defined(MBEDTLS_GCM_C) || defined(MBEDTLS_CCM_C)) && \
    (!defined(MBEDTLS_CIPHER_C) || defined(MBEDTLS_BLOCK_CIPHER_SOME_PSA))
#define MBEDTLS_BLOCK_CIPHER_C
#endif

/* Helpers for GCM/CCM capabilities */
#if (defined(MBEDTLS_CIPHER_C) && defined(MBEDTLS_AES_C)) || \
    (defined(MBEDTLS_BLOCK_CIPHER_C) && defined(MBEDTLS_BLOCK_CIPHER_CAN_AES))
#define MBEDTLS_CCM_GCM_CAN_AES
#endif

#if (defined(MBEDTLS_CIPHER_C) && defined(MBEDTLS_ARIA_C)) || \
    (defined(MBEDTLS_BLOCK_CIPHER_C) && defined(MBEDTLS_BLOCK_CIPHER_CAN_ARIA))
#define MBEDTLS_CCM_GCM_CAN_ARIA
#endif

#if (defined(MBEDTLS_CIPHER_C) && defined(MBEDTLS_CAMELLIA_C)) || \
    (defined(MBEDTLS_BLOCK_CIPHER_C) && defined(MBEDTLS_BLOCK_CIPHER_CAN_CAMELLIA))
#define MBEDTLS_CCM_GCM_CAN_CAMELLIA
#endif

/* MBEDTLS_ECP_LIGHT is auto-enabled by the following symbols:
 * - MBEDTLS_ECP_C because now it consists of MBEDTLS_ECP_LIGHT plus functions
 *   for curve arithmetic. As a consequence if MBEDTLS_ECP_C is required for
 *   some reason, then MBEDTLS_ECP_LIGHT should be enabled as well.
 * - MBEDTLS_PK_PARSE_EC_EXTENDED and MBEDTLS_PK_PARSE_EC_COMPRESSED because
 *   these features are not supported in PSA so the only way to have them is
 *   to enable the built-in solution.
 *   Both of them are temporary dependencies:
 *   - PK_PARSE_EC_EXTENDED will be removed after #7779 and #7789
 *   - support for compressed points should also be added to PSA, but in this
 *     case there is no associated issue to track it yet.
 * - PSA_WANT_KEY_TYPE_ECC_KEY_PAIR_DERIVE because Weierstrass key derivation
 *   still depends on ECP_LIGHT.
 */
#if defined(MBEDTLS_ECP_C) || \
    defined(MBEDTLS_PK_PARSE_EC_EXTENDED) || \
    defined(MBEDTLS_PK_PARSE_EC_COMPRESSED) || \
    defined(MBEDTLS_PSA_BUILTIN_KEY_TYPE_ECC_KEY_PAIR_DERIVE)
#define MBEDTLS_ECP_LIGHT
#endif

/* Legacy helper, still used by mbedtls_check_config.h */
#if defined(PSA_WANT_ALG_ECDH)
#define MBEDTLS_CAN_ECDH
#endif

/* Historically pkparse did not check the CBC padding when decrypting
 * a key. This was a bug, which is now fixed. As a consequence, pkparse
 * now needs PKCS7 padding support, but existing configurations might not
 * enable it, so we enable it here. */
#if defined(MBEDTLS_PK_PARSE_C) && defined(MBEDTLS_PKCS5_C) && defined(MBEDTLS_CIPHER_MODE_CBC)
#define MBEDTLS_CIPHER_PADDING_PKCS7
#endif

#endif /* TF_PSA_CRYPTO_MBEDTLS_PRIVATE_CRYPTO_ADJUST_CONFIG_TWEAK_BUILTINS_H */
