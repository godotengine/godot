/**
 *  Macros to express dependencies for code and tests that may use either the
 *  legacy API or PSA in various builds. This whole header file is currently
 *  for internal use only and both the header file and the macros it defines
 *  may change or be removed without notice.
 */
/*
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
 * Note: applications that are targeting a specific configuration do not need
 * to use these macros; instead they should directly use the functions they
 * know are available in their configuration.
 *
 * Note: code that is purely based on PSA Crypto (psa_xxx() functions)
 * does not need to use these macros; instead it should use the relevant
 * PSA_WANT_xxx macros.
 *
 * Note: code that is purely based on the legacy crypto APIs (mbedtls_xxx())
 * does not need to use these macros; instead it should use the relevant
 * MBEDTLS_xxx macros.
 *
 * These macros are for code that wants to use <crypto feature> and will do so
 * using <legacy API> or PSA depending on <condition>, where:
 * - <crypto feature> will generally be an algorithm (SHA-256, ECDH) but may
 *   also be a key type (AES, RSA, EC) or domain parameters (elliptic curve);
 * - <legacy API> will be either:
 *      - low-level module API (aes.h, sha256.h), or
 *      - an abstraction layer (md.h, cipher.h);
 * - <condition> will be either:
 *      - depending on what's available in the build:
 *          legacy API used if available, PSA otherwise
 *          (this is done to ensure backwards compatibility); or
 *      - depending on whether MBEDTLS_USE_PSA_CRYPTO is defined.
 *
 * Examples:
 * - TLS 1.2 will compute hashes using either mbedtls_md_xxx() (and
 *   mbedtls_sha256_xxx()) or psa_aead_xxx() depending on whether
 *   MBEDTLS_USE_PSA_CRYPTO is defined;
 * - RSA PKCS#1 v2.1 will compute hashes (for padding) using either
 *   `mbedtls_md()` if it's available, or `psa_hash_compute()` otherwise;
 * - PEM decoding of PEM-encrypted keys will compute MD5 hashes using either
 *   `mbedtls_md5_xxx()` if it's available, or `psa_hash_xxx()` otherwise.
 *
 * Note: the macros are essential to express test dependencies. Inside code,
 * we could instead just use the equivalent pre-processor condition, but
 * that's not possible in test dependencies where we need a single macro.
 * Hopefully, using these macros in code will also help with consistency.
 *
 * The naming scheme for these macros is:
 *      MBEDTLS_HAS_feature_VIA_legacy_OR_PSA(_condition)
 * where:
 * - feature is expressed the same way as in PSA_WANT_xxx macros, for example:
 *   KEY_TYPE_AES, ALG_SHA_256, ECC_SECP_R1_256;
 * - legacy is either LOWLEVEL or the name of the layer: MD, CIPHER;
 * - condition is omitted if it's based on availability, else it's
 *   BASED_ON_USE_PSA.
 *
 * Coming back to the examples above:
 * - TLS 1.2 will determine if it can use SHA-256 using
 *      MBEDTLS_HAS_ALG_SHA_256_VIA_MD_OR_PSA_BASED_ON_USE_PSA
 *   for the purposes of negotiation, and in test dependencies;
 * - RSA PKCS#1 v2.1 tests that used SHA-256 will depend on
 *      MBEDTLS_HAS_ALG_SHA_256_VIA_MD_OR_PSA
 * - PEM decoding code and its associated tests will depend on
 *      MBEDTLS_HAS_ALG_MD5_VIA_LOWLEVEL_OR_PSA
 *
 * Note: every time it's possible to use, say SHA-256, via the MD API, then
 * it's also possible to use it via the low-level API. So, code that wants to
 * use SHA-256 via both APIs only needs to depend on the MD macro. Also, it
 * just so happens that all the code choosing which API to use based on
 * MBEDTLS_USE_PSA_CRYPTO (X.509, TLS 1.2/shared), always uses the abstraction
 * layer (sometimes in addition to the low-level API), so we don't need the
 * MBEDTLS_HAS_feature_VIA_LOWLEVEL_OR_PSA_BASED_ON_USE_PSA macros.
 * (PK, while obeying MBEDTLS_USE_PSA_CRYPTO, doesn't compute hashes itself,
 * even less makes use of ciphers.)
 *
 * Note: the macros MBEDTLS_HAS_feature_VIA_LOWLEVEL_OR_PSA are the minimal
 * condition for being able to use <feature> at all. As such, they should be
 * used for guarding data about <feature>, such as OIDs or size. For example,
 * OID values related to SHA-256 are only useful when SHA-256 can be used at
 * least in some way.
 */

#ifndef MBEDTLS_OR_PSA_HELPERS_H
#define MBEDTLS_OR_PSA_HELPERS_H

#include "mbedtls/build_info.h"
#if defined(MBEDTLS_PSA_CRYPTO_C)
#include "psa/crypto.h"
#endif /* MBEDTLS_PSA_CRYPTO_C */

/*
 * Hashes
 */

/* Hashes using low-level or PSA based on availability */
#if defined(MBEDTLS_MD5_C) || \
    (defined(MBEDTLS_PSA_CRYPTO_C) && defined(PSA_WANT_ALG_MD5))
#define MBEDTLS_HAS_ALG_MD5_VIA_LOWLEVEL_OR_PSA
#endif
#if defined(MBEDTLS_RIPEMD160_C) || \
    (defined(MBEDTLS_PSA_CRYPTO_C) && defined(PSA_WANT_ALG_RIPEMD160))
#define MBEDTLS_HAS_ALG_RIPEMD160_VIA_LOWLEVEL_OR_PSA
#endif
#if defined(MBEDTLS_SHA1_C) || \
    (defined(MBEDTLS_PSA_CRYPTO_C) && defined(PSA_WANT_ALG_SHA_1))
#define MBEDTLS_HAS_ALG_SHA_1_VIA_LOWLEVEL_OR_PSA
#endif
#if defined(MBEDTLS_SHA224_C) || \
    (defined(MBEDTLS_PSA_CRYPTO_C) && defined(PSA_WANT_ALG_SHA_224))
#define MBEDTLS_HAS_ALG_SHA_224_VIA_LOWLEVEL_OR_PSA
#endif
#if defined(MBEDTLS_SHA256_C) || \
    (defined(MBEDTLS_PSA_CRYPTO_C) && defined(PSA_WANT_ALG_SHA_256))
#define MBEDTLS_HAS_ALG_SHA_256_VIA_LOWLEVEL_OR_PSA
#endif
#if defined(MBEDTLS_SHA384_C) || \
    (defined(MBEDTLS_PSA_CRYPTO_C) && defined(PSA_WANT_ALG_SHA_384))
#define MBEDTLS_HAS_ALG_SHA_384_VIA_LOWLEVEL_OR_PSA
#endif
#if defined(MBEDTLS_SHA512_C) || \
    (defined(MBEDTLS_PSA_CRYPTO_C) && defined(PSA_WANT_ALG_SHA_512))
#define MBEDTLS_HAS_ALG_SHA_512_VIA_LOWLEVEL_OR_PSA
#endif

/* Hashes using MD or PSA based on availability */
#if (defined(MBEDTLS_MD_C) && defined(MBEDTLS_MD5_C)) || \
    (!defined(MBEDTLS_MD_C) && \
    defined(MBEDTLS_PSA_CRYPTO_C) && defined(PSA_WANT_ALG_MD5))
#define MBEDTLS_HAS_ALG_MD5_VIA_MD_OR_PSA
#endif
#if (defined(MBEDTLS_MD_C) && defined(MBEDTLS_RIPEMD160_C)) || \
    (!defined(MBEDTLS_MD_C) && \
    defined(MBEDTLS_PSA_CRYPTO_C) && defined(PSA_WANT_ALG_RIPEMD160))
#define MBEDTLS_HAS_ALG_RIPEMD160_VIA_MD_OR_PSA
#endif
#if (defined(MBEDTLS_MD_C) && defined(MBEDTLS_SHA1_C)) || \
    (!defined(MBEDTLS_MD_C) && \
    defined(MBEDTLS_PSA_CRYPTO_C) && defined(PSA_WANT_ALG_SHA_1))
#define MBEDTLS_HAS_ALG_SHA_1_VIA_MD_OR_PSA
#endif
#if (defined(MBEDTLS_MD_C) && defined(MBEDTLS_SHA224_C)) || \
    (!defined(MBEDTLS_MD_C) && \
    defined(MBEDTLS_PSA_CRYPTO_C) && defined(PSA_WANT_ALG_SHA_224))
#define MBEDTLS_HAS_ALG_SHA_224_VIA_MD_OR_PSA
#endif
#if (defined(MBEDTLS_MD_C) && defined(MBEDTLS_SHA256_C)) || \
    (!defined(MBEDTLS_MD_C) && \
    defined(MBEDTLS_PSA_CRYPTO_C) && defined(PSA_WANT_ALG_SHA_256))
#define MBEDTLS_HAS_ALG_SHA_256_VIA_MD_OR_PSA
#endif
#if (defined(MBEDTLS_MD_C) && defined(MBEDTLS_SHA384_C)) || \
    (!defined(MBEDTLS_MD_C) && \
    defined(MBEDTLS_PSA_CRYPTO_C) && defined(PSA_WANT_ALG_SHA_384))
#define MBEDTLS_HAS_ALG_SHA_384_VIA_MD_OR_PSA
#endif
#if (defined(MBEDTLS_MD_C) && defined(MBEDTLS_SHA512_C)) || \
    (!defined(MBEDTLS_MD_C) && \
    defined(MBEDTLS_PSA_CRYPTO_C) && defined(PSA_WANT_ALG_SHA_512))
#define MBEDTLS_HAS_ALG_SHA_512_VIA_MD_OR_PSA
#endif

/* Hashes using MD or PSA based on MBEDTLS_USE_PSA_CRYPTO */
#if (!defined(MBEDTLS_USE_PSA_CRYPTO) && \
    defined(MBEDTLS_MD_C) && defined(MBEDTLS_MD5_C)) || \
    (defined(MBEDTLS_USE_PSA_CRYPTO) && defined(PSA_WANT_ALG_MD5))
#define MBEDTLS_HAS_ALG_MD5_VIA_MD_OR_PSA_BASED_ON_USE_PSA
#endif
#if (!defined(MBEDTLS_USE_PSA_CRYPTO) && \
    defined(MBEDTLS_MD_C) && defined(MBEDTLS_RIPEMD160_C)) || \
    (defined(MBEDTLS_USE_PSA_CRYPTO) && defined(PSA_WANT_ALG_RIPEMD160))
#define MBEDTLS_HAS_ALG_RIPEMD160_VIA_MD_OR_PSA_BASED_ON_USE_PSA
#endif
#if (!defined(MBEDTLS_USE_PSA_CRYPTO) && \
    defined(MBEDTLS_MD_C) && defined(MBEDTLS_SHA1_C)) || \
    (defined(MBEDTLS_USE_PSA_CRYPTO) && defined(PSA_WANT_ALG_SHA_1))
#define MBEDTLS_HAS_ALG_SHA_1_VIA_MD_OR_PSA_BASED_ON_USE_PSA
#endif
#if (!defined(MBEDTLS_USE_PSA_CRYPTO) && \
    defined(MBEDTLS_MD_C) && defined(MBEDTLS_SHA224_C)) || \
    (defined(MBEDTLS_USE_PSA_CRYPTO) && defined(PSA_WANT_ALG_SHA_224))
#define MBEDTLS_HAS_ALG_SHA_224_VIA_MD_OR_PSA_BASED_ON_USE_PSA
#endif
#if (!defined(MBEDTLS_USE_PSA_CRYPTO) && \
    defined(MBEDTLS_MD_C) && defined(MBEDTLS_SHA256_C)) || \
    (defined(MBEDTLS_USE_PSA_CRYPTO) && defined(PSA_WANT_ALG_SHA_256))
#define MBEDTLS_HAS_ALG_SHA_256_VIA_MD_OR_PSA_BASED_ON_USE_PSA
#endif
#if (!defined(MBEDTLS_USE_PSA_CRYPTO) && \
    defined(MBEDTLS_MD_C) && defined(MBEDTLS_SHA384_C)) || \
    (defined(MBEDTLS_USE_PSA_CRYPTO) && defined(PSA_WANT_ALG_SHA_384))
#define MBEDTLS_HAS_ALG_SHA_384_VIA_MD_OR_PSA_BASED_ON_USE_PSA
#endif
#if (!defined(MBEDTLS_USE_PSA_CRYPTO) && \
    defined(MBEDTLS_MD_C) && defined(MBEDTLS_SHA512_C)) || \
    (defined(MBEDTLS_USE_PSA_CRYPTO) && defined(PSA_WANT_ALG_SHA_512))
#define MBEDTLS_HAS_ALG_SHA_512_VIA_MD_OR_PSA_BASED_ON_USE_PSA
#endif

#endif /* MBEDTLS_OR_PSA_HELPERS_H */
