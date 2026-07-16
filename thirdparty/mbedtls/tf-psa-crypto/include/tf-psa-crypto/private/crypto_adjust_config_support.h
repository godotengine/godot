/**
 * \file tf-psa-crypto/private/crypto_adjust_config_support.h
 * \brief Adjust TF-PSA-Crypto configuration: support modules
 *
 * This is an internal header. Do not include it directly.
 *
 * Activate parts of support modules, based on the user configuration
 * as well as requirements of generic code and requirements of
 * driver-specific code.
 */
/*
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */

#ifndef TF_PSA_CRYPTO_PRIVATE_CRYPTO_ADJUST_CONFIG_SUPPORT_H
#define TF_PSA_CRYPTO_PRIVATE_CRYPTO_ADJUST_CONFIG_SUPPORT_H

/* Ideally, we'd set those as defaults in crypto_config.h, but
 * putting an #ifdef _WIN32 in crypto_config.h would confuse config.py.
 *
 * So, adjust it here.
 * Not related to crypto, but this is the bottom of the stack. */
#if defined(__MINGW32__) || (defined(_MSC_VER) && _MSC_VER <= 1900)
#if !defined(MBEDTLS_PLATFORM_SNPRINTF_ALT) && \
    !defined(MBEDTLS_PLATFORM_SNPRINTF_MACRO)
#define MBEDTLS_PLATFORM_SNPRINTF_ALT
#endif
#if !defined(MBEDTLS_PLATFORM_VSNPRINTF_ALT) && \
    !defined(MBEDTLS_PLATFORM_VSNPRINTF_MACRO)
#define MBEDTLS_PLATFORM_VSNPRINTF_ALT
#endif
#endif /* _MINGW32__ || (_MSC_VER && (_MSC_VER <= 1900)) */

/* If MBEDTLS_PSA_CRYPTO_C is defined, make sure MBEDTLS_PSA_CRYPTO_CLIENT
 * is defined as well to include all PSA code.
 */
#if defined(MBEDTLS_PSA_CRYPTO_C)
#define MBEDTLS_PSA_CRYPTO_CLIENT
/* Enable  MBEDTLS_ENTROPY_C in not client-only builds without an
 * external entropy source. */
#if !defined(MBEDTLS_PSA_CRYPTO_EXTERNAL_RNG)
#define MBEDTLS_ENTROPY_C
#endif
#endif /* MBEDTLS_PSA_CRYPTO_C */

/* PKCS5 needs MD. */
#if defined(MBEDTLS_PKCS5_C)
#define MBEDTLS_MD_C
#endif

/* Auto-enable MBEDTLS_MD_LIGHT based on MBEDTLS_MD_C.
 * This allows checking for MD_LIGHT rather than MD_LIGHT || MD_C.
 */
#if defined(MBEDTLS_MD_C)
#define MBEDTLS_MD_LIGHT
#endif

/* Auto-enable MBEDTLS_MD_LIGHT if needed by a module that didn't require it
 * in a previous release, to ensure backwards compatibility.
 */
#if defined(MBEDTLS_ECJPAKE_C) || \
    defined(MBEDTLS_PEM_PARSE_C) || \
    defined(MBEDTLS_ENTROPY_C) || \
    defined(MBEDTLS_PK_C) || \
    defined(MBEDTLS_RSA_C)
#define MBEDTLS_MD_LIGHT
#endif

#if defined(MBEDTLS_MD_LIGHT)
/*
 * - MBEDTLS_MD_xxx_VIA_PSA is defined if the md module may perform xxx via PSA
 *   (see below).
 * - MBEDTLS_MD_SOME_PSA is defined if at least one algorithm may be performed
 *   via PSA (see below).
 * - MBEDTLS_MD_SOME_LEGACY is defined if at least one algorithm may be performed
 *   via a direct legacy call (see below).
 *
 * The md module performs an algorithm via PSA if there is a PSA hash
 * accelerator and the PSA driver subsytem is initialized at the time the
 * operation is started, and makes a direct legacy call otherwise.
 */

/* PSA accelerated implementations */
#if defined(MBEDTLS_PSA_CRYPTO_C)

#if defined(MBEDTLS_PSA_ACCEL_ALG_MD5)
#define MBEDTLS_MD_MD5_VIA_PSA
#define MBEDTLS_MD_SOME_PSA
#endif
#if defined(MBEDTLS_PSA_ACCEL_ALG_SHA_1)
#define MBEDTLS_MD_SHA1_VIA_PSA
#define MBEDTLS_MD_SOME_PSA
#endif
#if defined(MBEDTLS_PSA_ACCEL_ALG_SHA_224)
#define MBEDTLS_MD_SHA224_VIA_PSA
#define MBEDTLS_MD_SOME_PSA
#endif
#if defined(MBEDTLS_PSA_ACCEL_ALG_SHA_256)
#define MBEDTLS_MD_SHA256_VIA_PSA
#define MBEDTLS_MD_SOME_PSA
#endif
#if defined(MBEDTLS_PSA_ACCEL_ALG_SHA_384)
#define MBEDTLS_MD_SHA384_VIA_PSA
#define MBEDTLS_MD_SOME_PSA
#endif
#if defined(MBEDTLS_PSA_ACCEL_ALG_SHA_512)
#define MBEDTLS_MD_SHA512_VIA_PSA
#define MBEDTLS_MD_SOME_PSA
#endif
#if defined(MBEDTLS_PSA_ACCEL_ALG_RIPEMD160)
#define MBEDTLS_MD_RIPEMD160_VIA_PSA
#define MBEDTLS_MD_SOME_PSA
#endif
#if defined(MBEDTLS_PSA_ACCEL_ALG_SHA3_224)
#define MBEDTLS_MD_SHA3_224_VIA_PSA
#define MBEDTLS_MD_SOME_PSA
#endif
#if defined(MBEDTLS_PSA_ACCEL_ALG_SHA3_256)
#define MBEDTLS_MD_SHA3_256_VIA_PSA
#define MBEDTLS_MD_SOME_PSA
#endif
#if defined(MBEDTLS_PSA_ACCEL_ALG_SHA3_384)
#define MBEDTLS_MD_SHA3_384_VIA_PSA
#define MBEDTLS_MD_SOME_PSA
#endif
#if defined(MBEDTLS_PSA_ACCEL_ALG_SHA3_512)
#define MBEDTLS_MD_SHA3_512_VIA_PSA
#define MBEDTLS_MD_SOME_PSA
#endif

#elif defined(MBEDTLS_PSA_CRYPTO_CLIENT)

#if defined(PSA_WANT_ALG_MD5)
#define MBEDTLS_MD_MD5_VIA_PSA
#define MBEDTLS_MD_SOME_PSA
#endif
#if defined(PSA_WANT_ALG_SHA_1)
#define MBEDTLS_MD_SHA1_VIA_PSA
#define MBEDTLS_MD_SOME_PSA
#endif
#if defined(PSA_WANT_ALG_SHA_224)
#define MBEDTLS_MD_SHA224_VIA_PSA
#define MBEDTLS_MD_SOME_PSA
#endif
#if defined(PSA_WANT_ALG_SHA_256)
#define MBEDTLS_MD_SHA256_VIA_PSA
#define MBEDTLS_MD_SOME_PSA
#endif
#if defined(PSA_WANT_ALG_SHA_384)
#define MBEDTLS_MD_SHA384_VIA_PSA
#define MBEDTLS_MD_SOME_PSA
#endif
#if defined(PSA_WANT_ALG_SHA_512)
#define MBEDTLS_MD_SHA512_VIA_PSA
#define MBEDTLS_MD_SOME_PSA
#endif
#if defined(PSA_WANT_ALG_RIPEMD160)
#define MBEDTLS_MD_RIPEMD160_VIA_PSA
#define MBEDTLS_MD_SOME_PSA
#endif
#if defined(PSA_WANT_ALG_SHA3_224)
#define MBEDTLS_MD_SHA3_224_VIA_PSA
#define MBEDTLS_MD_SOME_PSA
#endif
#if defined(PSA_WANT_ALG_SHA3_256)
#define MBEDTLS_MD_SHA3_256_VIA_PSA
#define MBEDTLS_MD_SOME_PSA
#endif
#if defined(PSA_WANT_ALG_SHA3_384)
#define MBEDTLS_MD_SHA3_384_VIA_PSA
#define MBEDTLS_MD_SOME_PSA
#endif
#if defined(PSA_WANT_ALG_SHA3_512)
#define MBEDTLS_MD_SHA3_512_VIA_PSA
#define MBEDTLS_MD_SOME_PSA
#endif

#endif /* !MBEDTLS_PSA_CRYPTO_CLIENT && !MBEDTLS_PSA_CRYPTO_C */

/* Built-in implementations */
#if defined(MBEDTLS_MD5_C) || \
    defined(MBEDTLS_SHA1_C) || \
    defined(MBEDTLS_SHA224_C) || \
    defined(MBEDTLS_SHA256_C) || \
    defined(MBEDTLS_SHA384_C) || \
    defined(MBEDTLS_SHA512_C) || \
    defined(MBEDTLS_RIPEMD160_C)
#define MBEDTLS_MD_SOME_LEGACY
#endif

#endif /* MBEDTLS_MD_LIGHT */

/* Backward compatibility: after #8740 the RSA module offers functions to parse
 * and write RSA private/public keys without relying on the PK one. Of course
 * this needs ASN1 support to do so, so we enable it here. */
#if defined(MBEDTLS_RSA_C)
#define MBEDTLS_ASN1_PARSE_C
#define MBEDTLS_ASN1_WRITE_C
#endif

/* MBEDTLS_PK_PARSE_EC_COMPRESSED is introduced in Mbed TLS version 3.5, while
 * in previous version compressed points were automatically supported as long
 * as PK_PARSE_C and ECP_C were enabled. As a consequence, for backward
 * compatibility, we auto-enable PK_PARSE_EC_COMPRESSED when these conditions
 * are met. */
#if defined(MBEDTLS_PK_PARSE_C) && defined(MBEDTLS_ECP_C)
#define MBEDTLS_PK_PARSE_EC_COMPRESSED
#endif

#endif /* TF_PSA_CRYPTO_PRIVATE_CRYPTO_ADJUST_CONFIG_SUPPORT_H */
