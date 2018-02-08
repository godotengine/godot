/**
 * \file config.h
 *
 * \brief Configuration options (set of defines)
 *
 *  This set of compile-time options may be used to enable
 *  or disable features selectively, and reduce the global
 *  memory footprint.
 */
/*
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

#ifndef MBEDTLS_CONFIG_H
#define MBEDTLS_CONFIG_H

#if defined(_MSC_VER) && !defined(_CRT_SECURE_NO_DEPRECATE)
#define _CRT_SECURE_NO_DEPRECATE 1
#endif

/**
 * \name SECTION: System support
 *
 * This section sets system specific settings.
 * \{
 */

/**
 * \def MBEDTLS_HAVE_ASM
 *
 * The compiler has support for asm().
 *
 * Requires support for asm() in compiler.
 *
 * Used in:
 *      library/timing.c
 *      library/padlock.c
 *      include/mbedtls/bn_mul.h
 *
 * Comment to disable the use of assembly code.
 */
#define MBEDTLS_HAVE_ASM

/**
 * \def MBEDTLS_NO_UDBL_DIVISION
 *
 * The platform lacks support for double-width integer division (64-bit
 * division on a 32-bit platform, 128-bit division on a 64-bit platform).
 *
 * Used in:
 *      include/mbedtls/bignum.h
 *      library/bignum.c
 *
 * The bignum code uses double-width division to speed up some operations.
 * Double-width division is often implemented in software that needs to
 * be linked with the program. The presence of a double-width integer
 * type is usually detected automatically through preprocessor macros,
 * but the automatic detection cannot know whether the code needs to
 * and can be linked with an implementation of division for that type.
 * By default division is assumed to be usable if the type is present.
 * Uncomment this option to prevent the use of double-width division.
 *
 * Note that division for the native integer type is always required.
 * Furthermore, a 64-bit type is always required even on a 32-bit
 * platform, but it need not support multiplication or division. In some
 * cases it is also desirable to disable some double-width operations. For
 * example, if double-width division is implemented in software, disabling
 * it can reduce code size in some embedded targets.
 */
//#define MBEDTLS_NO_UDBL_DIVISION

/**
 * \def MBEDTLS_HAVE_SSE2
 *
 * CPU supports SSE2 instruction set.
 *
 * Uncomment if the CPU supports SSE2 (IA-32 specific).
 */
//#define MBEDTLS_HAVE_SSE2

/**
 * \def MBEDTLS_HAVE_TIME
 *
 * System has time.h and time().
 * The time does not need to be correct, only time differences are used,
 * by contrast with MBEDTLS_HAVE_TIME_DATE
 *
 * Defining MBEDTLS_HAVE_TIME allows you to specify MBEDTLS_PLATFORM_TIME_ALT,
 * MBEDTLS_PLATFORM_TIME_MACRO, MBEDTLS_PLATFORM_TIME_TYPE_MACRO and
 * MBEDTLS_PLATFORM_STD_TIME.
 *
 * Comment if your system does not support time functions
 */
#define MBEDTLS_HAVE_TIME

/**
 * \def MBEDTLS_HAVE_TIME_DATE
 *
 * System has time.h and time(), gmtime() and the clock is correct.
 * The time needs to be correct (not necesarily very accurate, but at least
 * the date should be correct). This is used to verify the validity period of
 * X.509 certificates.
 *
 * Comment if your system does not have a correct clock.
 */
#define MBEDTLS_HAVE_TIME_DATE

/**
 * \def MBEDTLS_PLATFORM_MEMORY
 *
 * Enable the memory allocation layer.
 *
 * By default mbed TLS uses the system-provided calloc() and free().
 * This allows different allocators (self-implemented or provided) to be
 * provided to the platform abstraction layer.
 *
 * Enabling MBEDTLS_PLATFORM_MEMORY without the
 * MBEDTLS_PLATFORM_{FREE,CALLOC}_MACROs will provide
 * "mbedtls_platform_set_calloc_free()" allowing you to set an alternative calloc() and
 * free() function pointer at runtime.
 *
 * Enabling MBEDTLS_PLATFORM_MEMORY and specifying
 * MBEDTLS_PLATFORM_{CALLOC,FREE}_MACROs will allow you to specify the
 * alternate function at compile time.
 *
 * Requires: MBEDTLS_PLATFORM_C
 *
 * Enable this layer to allow use of alternative memory allocators.
 */
//#define MBEDTLS_PLATFORM_MEMORY

/**
 * \def MBEDTLS_PLATFORM_NO_STD_FUNCTIONS
 *
 * Do not assign standard functions in the platform layer (e.g. calloc() to
 * MBEDTLS_PLATFORM_STD_CALLOC and printf() to MBEDTLS_PLATFORM_STD_PRINTF)
 *
 * This makes sure there are no linking errors on platforms that do not support
 * these functions. You will HAVE to provide alternatives, either at runtime
 * via the platform_set_xxx() functions or at compile time by setting
 * the MBEDTLS_PLATFORM_STD_XXX defines, or enabling a
 * MBEDTLS_PLATFORM_XXX_MACRO.
 *
 * Requires: MBEDTLS_PLATFORM_C
 *
 * Uncomment to prevent default assignment of standard functions in the
 * platform layer.
 */
//#define MBEDTLS_PLATFORM_NO_STD_FUNCTIONS

/**
 * \def MBEDTLS_PLATFORM_EXIT_ALT
 *
 * MBEDTLS_PLATFORM_XXX_ALT: Uncomment a macro to let mbed TLS support the
 * function in the platform abstraction layer.
 *
 * Example: In case you uncomment MBEDTLS_PLATFORM_PRINTF_ALT, mbed TLS will
 * provide a function "mbedtls_platform_set_printf()" that allows you to set an
 * alternative printf function pointer.
 *
 * All these define require MBEDTLS_PLATFORM_C to be defined!
 *
 * \note MBEDTLS_PLATFORM_SNPRINTF_ALT is required on Windows;
 * it will be enabled automatically by check_config.h
 *
 * \warning MBEDTLS_PLATFORM_XXX_ALT cannot be defined at the same time as
 * MBEDTLS_PLATFORM_XXX_MACRO!
 *
 * Requires: MBEDTLS_PLATFORM_TIME_ALT requires MBEDTLS_HAVE_TIME
 *
 * Uncomment a macro to enable alternate implementation of specific base
 * platform function
 */
//#define MBEDTLS_PLATFORM_EXIT_ALT
//#define MBEDTLS_PLATFORM_TIME_ALT
//#define MBEDTLS_PLATFORM_FPRINTF_ALT
//#define MBEDTLS_PLATFORM_PRINTF_ALT
//#define MBEDTLS_PLATFORM_SNPRINTF_ALT
//#define MBEDTLS_PLATFORM_NV_SEED_ALT
//#define MBEDTLS_PLATFORM_SETUP_TEARDOWN_ALT

/**
 * \def MBEDTLS_DEPRECATED_WARNING
 *
 * Mark deprecated functions so that they generate a warning if used.
 * Functions deprecated in one version will usually be removed in the next
 * version. You can enable this to help you prepare the transition to a new
 * major version by making sure your code is not using these functions.
 *
 * This only works with GCC and Clang. With other compilers, you may want to
 * use MBEDTLS_DEPRECATED_REMOVED
 *
 * Uncomment to get warnings on using deprecated functions.
 */
//#define MBEDTLS_DEPRECATED_WARNING

/**
 * \def MBEDTLS_DEPRECATED_REMOVED
 *
 * Remove deprecated functions so that they generate an error if used.
 * Functions deprecated in one version will usually be removed in the next
 * version. You can enable this to help you prepare the transition to a new
 * major version by making sure your code is not using these functions.
 *
 * Uncomment to get errors on using deprecated functions.
 */
//#define MBEDTLS_DEPRECATED_REMOVED

/* \} name SECTION: System support */

/**
 * \name SECTION: mbed TLS feature support
 *
 * This section sets support for features that are or are not needed
 * within the modules that are enabled.
 * \{
 */

/**
 * \def MBEDTLS_TIMING_ALT
 *
 * Uncomment to provide your own alternate implementation for mbedtls_timing_hardclock(),
 * mbedtls_timing_get_timer(), mbedtls_set_alarm(), mbedtls_set/get_delay()
 *
 * Only works if you have MBEDTLS_TIMING_C enabled.
 *
 * You will need to provide a header "timing_alt.h" and an implementation at
 * compile time.
 */
//#define MBEDTLS_TIMING_ALT

/**
 * \def MBEDTLS_AES_ALT
 *
 * MBEDTLS__MODULE_NAME__ALT: Uncomment a macro to let mbed TLS use your
 * alternate core implementation of a symmetric crypto, an arithmetic or hash
 * module (e.g. platform specific assembly optimized implementations). Keep
 * in mind that the function prototypes should remain the same.
 *
 * This replaces the whole module. If you only want to replace one of the
 * functions, use one of the MBEDTLS__FUNCTION_NAME__ALT flags.
 *
 * Example: In case you uncomment MBEDTLS_AES_ALT, mbed TLS will no longer
 * provide the "struct mbedtls_aes_context" definition and omit the base
 * function declarations and implementations. "aes_alt.h" will be included from
 * "aes.h" to include the new function definitions.
 *
 * Uncomment a macro to enable alternate implementation of the corresponding
 * module.
 *
 * \warning   MD2, MD4, MD5, ARC4, DES and SHA-1 are considered weak and their
 *            use constitutes a security risk. If possible, we recommend
 *            avoiding dependencies on them, and considering stronger message
 *            digests and ciphers instead.
 *
 */
//#define MBEDTLS_AES_ALT
//#define MBEDTLS_ARC4_ALT
//#define MBEDTLS_BLOWFISH_ALT
//#define MBEDTLS_CAMELLIA_ALT
//#define MBEDTLS_CCM_ALT
//#define MBEDTLS_CMAC_ALT
//#define MBEDTLS_DES_ALT
//#define MBEDTLS_DHM_ALT
//#define MBEDTLS_ECJPAKE_ALT
//#define MBEDTLS_GCM_ALT
//#define MBEDTLS_MD2_ALT
//#define MBEDTLS_MD4_ALT
//#define MBEDTLS_MD5_ALT
//#define MBEDTLS_RIPEMD160_ALT
//#define MBEDTLS_RSA_ALT
//#define MBEDTLS_SHA1_ALT
//#define MBEDTLS_SHA256_ALT
//#define MBEDTLS_SHA512_ALT
//#define MBEDTLS_XTEA_ALT
/*
 * When replacing the elliptic curve module, pleace consider, that it is
 * implemented with two .c files:
 *      - ecp.c
 *      - ecp_curves.c
 * You can replace them very much like all the other MBEDTLS__MODULE_NAME__ALT
 * macros as described above. The only difference is that you have to make sure
 * that you provide functionality for both .c files.
 */
//#define MBEDTLS_ECP_ALT

/**
 * \def MBEDTLS_MD2_PROCESS_ALT
 *
 * MBEDTLS__FUNCTION_NAME__ALT: Uncomment a macro to let mbed TLS use you
 * alternate core implementation of symmetric crypto or hash function. Keep in
 * mind that function prototypes should remain the same.
 *
 * This replaces only one function. The header file from mbed TLS is still
 * used, in contrast to the MBEDTLS__MODULE_NAME__ALT flags.
 *
 * Example: In case you uncomment MBEDTLS_SHA256_PROCESS_ALT, mbed TLS will
 * no longer provide the mbedtls_sha1_process() function, but it will still provide
 * the other function (using your mbedtls_sha1_process() function) and the definition
 * of mbedtls_sha1_context, so your implementation of mbedtls_sha1_process must be compatible
 * with this definition.
 *
 * \note Because of a signature change, the core AES encryption and decryption routines are
 *       currently named mbedtls_aes_internal_encrypt and mbedtls_aes_internal_decrypt,
 *       respectively. When setting up alternative implementations, these functions should
 *       be overriden, but the wrapper functions mbedtls_aes_decrypt and mbedtls_aes_encrypt
 *       must stay untouched.
 *
 * \note If you use the AES_xxx_ALT macros, then is is recommended to also set
 *       MBEDTLS_AES_ROM_TABLES in order to help the linker garbage-collect the AES
 *       tables.
 *
 * Uncomment a macro to enable alternate implementation of the corresponding
 * function.
 *
 * \warning   MD2, MD4, MD5, DES and SHA-1 are considered weak and their use
 *            constitutes a security risk. If possible, we recommend avoiding
 *            dependencies on them, and considering stronger message digests
 *            and ciphers instead.
 *
 */
//#define MBEDTLS_MD2_PROCESS_ALT
//#define MBEDTLS_MD4_PROCESS_ALT
//#define MBEDTLS_MD5_PROCESS_ALT
//#define MBEDTLS_RIPEMD160_PROCESS_ALT
//#define MBEDTLS_SHA1_PROCESS_ALT
//#define MBEDTLS_SHA256_PROCESS_ALT
//#define MBEDTLS_SHA512_PROCESS_ALT
//#define MBEDTLS_DES_SETKEY_ALT
//#define MBEDTLS_DES_CRYPT_ECB_ALT
//#define MBEDTLS_DES3_CRYPT_ECB_ALT
//#define MBEDTLS_AES_SETKEY_ENC_ALT
//#define MBEDTLS_AES_SETKEY_DEC_ALT
//#define MBEDTLS_AES_ENCRYPT_ALT
//#define MBEDTLS_AES_DECRYPT_ALT
//#define MBEDTLS_ECDH_GEN_PUBLIC_ALT
//#define MBEDTLS_ECDH_COMPUTE_SHARED_ALT
//#define MBEDTLS_ECDSA_VERIFY_ALT
//#define MBEDTLS_ECDSA_SIGN_ALT
//#define MBEDTLS_ECDSA_GENKEY_ALT

/**
 * \def MBEDTLS_ECP_INTERNAL_ALT
 *
 * Expose a part of the internal interface of the Elliptic Curve Point module.
 *
 * MBEDTLS_ECP__FUNCTION_NAME__ALT: Uncomment a macro to let mbed TLS use your
 * alternative core implementation of elliptic curve arithmetic. Keep in mind
 * that function prototypes should remain the same.
 *
 * This partially replaces one function. The header file from mbed TLS is still
 * used, in contrast to the MBEDTLS_ECP_ALT flag. The original implementation
 * is still present and it is used for group structures not supported by the
 * alternative.
 *
 * Any of these options become available by defining MBEDTLS_ECP_INTERNAL_ALT
 * and implementing the following functions:
 *      unsigned char mbedtls_internal_ecp_grp_capable(
 *          const mbedtls_ecp_group *grp )
 *      int  mbedtls_internal_ecp_init( const mbedtls_ecp_group *grp )
 *      void mbedtls_internal_ecp_deinit( const mbedtls_ecp_group *grp )
 * The mbedtls_internal_ecp_grp_capable function should return 1 if the
 * replacement functions implement arithmetic for the given group and 0
 * otherwise.
 * The functions mbedtls_internal_ecp_init and mbedtls_internal_ecp_deinit are
 * called before and after each point operation and provide an opportunity to
 * implement optimized set up and tear down instructions.
 *
 * Example: In case you uncomment MBEDTLS_ECP_INTERNAL_ALT and
 * MBEDTLS_ECP_DOUBLE_JAC_ALT, mbed TLS will still provide the ecp_double_jac
 * function, but will use your mbedtls_internal_ecp_double_jac if the group is
 * supported (your mbedtls_internal_ecp_grp_capable function returns 1 when
 * receives it as an argument). If the group is not supported then the original
 * implementation is used. The other functions and the definition of
 * mbedtls_ecp_group and mbedtls_ecp_point will not change, so your
 * implementation of mbedtls_internal_ecp_double_jac and
 * mbedtls_internal_ecp_grp_capable must be compatible with this definition.
 *
 * Uncomment a macro to enable alternate implementation of the corresponding
 * function.
 */
/* Required for all the functions in this section */
//#define MBEDTLS_ECP_INTERNAL_ALT
/* Support for Weierstrass curves with Jacobi representation */
//#define MBEDTLS_ECP_RANDOMIZE_JAC_ALT
//#define MBEDTLS_ECP_ADD_MIXED_ALT
//#define MBEDTLS_ECP_DOUBLE_JAC_ALT
//#define MBEDTLS_ECP_NORMALIZE_JAC_MANY_ALT
//#define MBEDTLS_ECP_NORMALIZE_JAC_ALT
/* Support for curves with Montgomery arithmetic */
//#define MBEDTLS_ECP_DOUBLE_ADD_MXZ_ALT
//#define MBEDTLS_ECP_RANDOMIZE_MXZ_ALT
//#define MBEDTLS_ECP_NORMALIZE_MXZ_ALT

/**
 * \def MBEDTLS_TEST_NULL_ENTROPY
 *
 * Enables testing and use of mbed TLS without any configured entropy sources.
 * This permits use of the library on platforms before an entropy source has
 * been integrated (see for example the MBEDTLS_ENTROPY_HARDWARE_ALT or the
 * MBEDTLS_ENTROPY_NV_SEED switches).
 *
 * WARNING! This switch MUST be disabled in production builds, and is suitable
 * only for development.
 * Enabling the switch negates any security provided by the library.
 *
 * Requires MBEDTLS_ENTROPY_C, MBEDTLS_NO_DEFAULT_ENTROPY_SOURCES
 *
 */
//#define MBEDTLS_TEST_NULL_ENTROPY

/**
 * \def MBEDTLS_ENTROPY_HARDWARE_ALT
 *
 * Uncomment this macro to let mbed TLS use your own implementation of a
 * hardware entropy collector.
 *
 * Your function must be called \c mbedtls_hardware_poll(), have the same
 * prototype as declared in entropy_poll.h, and accept NULL as first argument.
 *
 * Uncomment to use your own hardware entropy collector.
 */
//#define MBEDTLS_ENTROPY_HARDWARE_ALT

/**
 * \def MBEDTLS_AES_ROM_TABLES
 *
 * Store the AES tables in ROM.
 *
 * Uncomment this macro to store the AES tables in ROM.
 */
//#define MBEDTLS_AES_ROM_TABLES

/**
 * \def MBEDTLS_CAMELLIA_SMALL_MEMORY
 *
 * Use less ROM for the Camellia implementation (saves about 768 bytes).
 *
 * Uncomment this macro to use less memory for Camellia.
 */
//#define MBEDTLS_CAMELLIA_SMALL_MEMORY

/**
 * \def MBEDTLS_CIPHER_MODE_CBC
 *
 * Enable Cipher Block Chaining mode (CBC) for symmetric ciphers.
 */
#define MBEDTLS_CIPHER_MODE_CBC

/**
 * \def MBEDTLS_CIPHER_MODE_CFB
 *
 * Enable Cipher Feedback mode (CFB) for symmetric ciphers.
 */
#define MBEDTLS_CIPHER_MODE_CFB

/**
 * \def MBEDTLS_CIPHER_MODE_CTR
 *
 * Enable Counter Block Cipher mode (CTR) for symmetric ciphers.
 */
#define MBEDTLS_CIPHER_MODE_CTR

/**
 * \def MBEDTLS_CIPHER_NULL_CIPHER
 *
 * Enable NULL cipher.
 * Warning: Only do so when you know what you are doing. This allows for
 * encryption or channels without any security!
 *
 * Requires MBEDTLS_ENABLE_WEAK_CIPHERSUITES as well to enable
 * the following ciphersuites:
 *      MBEDTLS_TLS_ECDH_ECDSA_WITH_NULL_SHA
 *      MBEDTLS_TLS_ECDH_RSA_WITH_NULL_SHA
 *      MBEDTLS_TLS_ECDHE_ECDSA_WITH_NULL_SHA
 *      MBEDTLS_TLS_ECDHE_RSA_WITH_NULL_SHA
 *      MBEDTLS_TLS_ECDHE_PSK_WITH_NULL_SHA384
 *      MBEDTLS_TLS_ECDHE_PSK_WITH_NULL_SHA256
 *      MBEDTLS_TLS_ECDHE_PSK_WITH_NULL_SHA
 *      MBEDTLS_TLS_DHE_PSK_WITH_NULL_SHA384
 *      MBEDTLS_TLS_DHE_PSK_WITH_NULL_SHA256
 *      MBEDTLS_TLS_DHE_PSK_WITH_NULL_SHA
 *      MBEDTLS_TLS_RSA_WITH_NULL_SHA256
 *      MBEDTLS_TLS_RSA_WITH_NULL_SHA
 *      MBEDTLS_TLS_RSA_WITH_NULL_MD5
 *      MBEDTLS_TLS_RSA_PSK_WITH_NULL_SHA384
 *      MBEDTLS_TLS_RSA_PSK_WITH_NULL_SHA256
 *      MBEDTLS_TLS_RSA_PSK_WITH_NULL_SHA
 *      MBEDTLS_TLS_PSK_WITH_NULL_SHA384
 *      MBEDTLS_TLS_PSK_WITH_NULL_SHA256
 *      MBEDTLS_TLS_PSK_WITH_NULL_SHA
 *
 * Uncomment this macro to enable the NULL cipher and ciphersuites
 */
//#define MBEDTLS_CIPHER_NULL_CIPHER

/**
 * \def MBEDTLS_CIPHER_PADDING_PKCS7
 *
 * MBEDTLS_CIPHER_PADDING_XXX: Uncomment or comment macros to add support for
 * specific padding modes in the cipher layer with cipher modes that support
 * padding (e.g. CBC)
 *
 * If you disable all padding modes, only full blocks can be used with CBC.
 *
 * Enable padding modes in the cipher layer.
 */
#define MBEDTLS_CIPHER_PADDING_PKCS7
#define MBEDTLS_CIPHER_PADDING_ONE_AND_ZEROS
#define MBEDTLS_CIPHER_PADDING_ZEROS_AND_LEN
#define MBEDTLS_CIPHER_PADDING_ZEROS

/**
 * \def MBEDTLS_ENABLE_WEAK_CIPHERSUITES
 *
 * Enable weak ciphersuites in SSL / TLS.
 * Warning: Only do so when you know what you are doing. This allows for
 * channels with virtually no security at all!
 *
 * This enables the following ciphersuites:
 *      MBEDTLS_TLS_RSA_WITH_DES_CBC_SHA
 *      MBEDTLS_TLS_DHE_RSA_WITH_DES_CBC_SHA
 *
 * Uncomment this macro to enable weak ciphersuites
 *
 * \warning   DES is considered a weak cipher and its use constitutes a
 *            security risk. We recommend considering stronger ciphers instead.
 */
//#define MBEDTLS_ENABLE_WEAK_CIPHERSUITES

/**
 * \def MBEDTLS_REMOVE_ARC4_CIPHERSUITES
 *
 * Remove RC4 ciphersuites by default in SSL / TLS.
 * This flag removes the ciphersuites based on RC4 from the default list as
 * returned by mbedtls_ssl_list_ciphersuites(). However, it is still possible to
 * enable (some of) them with mbedtls_ssl_conf_ciphersuites() by including them
 * explicitly.
 *
 * Uncomment this macro to remove RC4 ciphersuites by default.
 */
#define MBEDTLS_REMOVE_ARC4_CIPHERSUITES

/**
 * \def MBEDTLS_ECP_DP_SECP192R1_ENABLED
 *
 * MBEDTLS_ECP_XXXX_ENABLED: Enables specific curves within the Elliptic Curve
 * module.  By default all supported curves are enabled.
 *
 * Comment macros to disable the curve and functions for it
 */
#define MBEDTLS_ECP_DP_SECP192R1_ENABLED
#define MBEDTLS_ECP_DP_SECP224R1_ENABLED
#define MBEDTLS_ECP_DP_SECP256R1_ENABLED
#define MBEDTLS_ECP_DP_SECP384R1_ENABLED
#define MBEDTLS_ECP_DP_SECP521R1_ENABLED
#define MBEDTLS_ECP_DP_SECP192K1_ENABLED
#define MBEDTLS_ECP_DP_SECP224K1_ENABLED
#define MBEDTLS_ECP_DP_SECP256K1_ENABLED
#define MBEDTLS_ECP_DP_BP256R1_ENABLED
#define MBEDTLS_ECP_DP_BP384R1_ENABLED
#define MBEDTLS_ECP_DP_BP512R1_ENABLED
#define MBEDTLS_ECP_DP_CURVE25519_ENABLED

/**
 * \def MBEDTLS_ECP_NIST_OPTIM
 *
 * Enable specific 'modulo p' routines for each NIST prime.
 * Depending on the prime and architecture, makes operations 4 to 8 times
 * faster on the corresponding curve.
 *
 * Comment this macro to disable NIST curves optimisation.
 */
#define MBEDTLS_ECP_NIST_OPTIM

/**
 * \def MBEDTLS_ECDSA_DETERMINISTIC
 *
 * Enable deterministic ECDSA (RFC 6979).
 * Standard ECDSA is "fragile" in the sense that lack of entropy when signing
 * may result in a compromise of the long-term signing key. This is avoided by
 * the deterministic variant.
 *
 * Requires: MBEDTLS_HMAC_DRBG_C
 *
 * Comment this macro to disable deterministic ECDSA.
 */
#define MBEDTLS_ECDSA_DETERMINISTIC

/**
 * \def MBEDTLS_KEY_EXCHANGE_PSK_ENABLED
 *
 * Enable the PSK based ciphersuite modes in SSL / TLS.
 *
 * This enables the following ciphersuites (if other requisites are
 * enabled as well):
 *      MBEDTLS_TLS_PSK_WITH_AES_256_GCM_SHA384
 *      MBEDTLS_TLS_PSK_WITH_AES_256_CBC_SHA384
 *      MBEDTLS_TLS_PSK_WITH_AES_256_CBC_SHA
 *      MBEDTLS_TLS_PSK_WITH_CAMELLIA_256_GCM_SHA384
 *      MBEDTLS_TLS_PSK_WITH_CAMELLIA_256_CBC_SHA384
 *      MBEDTLS_TLS_PSK_WITH_AES_128_GCM_SHA256
 *      MBEDTLS_TLS_PSK_WITH_AES_128_CBC_SHA256
 *      MBEDTLS_TLS_PSK_WITH_AES_128_CBC_SHA
 *      MBEDTLS_TLS_PSK_WITH_CAMELLIA_128_GCM_SHA256
 *      MBEDTLS_TLS_PSK_WITH_CAMELLIA_128_CBC_SHA256
 *      MBEDTLS_TLS_PSK_WITH_3DES_EDE_CBC_SHA
 *      MBEDTLS_TLS_PSK_WITH_RC4_128_SHA
 */
#define MBEDTLS_KEY_EXCHANGE_PSK_ENABLED

/**
 * \def MBEDTLS_KEY_EXCHANGE_DHE_PSK_ENABLED
 *
 * Enable the DHE-PSK based ciphersuite modes in SSL / TLS.
 *
 * Requires: MBEDTLS_DHM_C
 *
 * This enables the following ciphersuites (if other requisites are
 * enabled as well):
 *      MBEDTLS_TLS_DHE_PSK_WITH_AES_256_GCM_SHA384
 *      MBEDTLS_TLS_DHE_PSK_WITH_AES_256_CBC_SHA384
 *      MBEDTLS_TLS_DHE_PSK_WITH_AES_256_CBC_SHA
 *      MBEDTLS_TLS_DHE_PSK_WITH_CAMELLIA_256_GCM_SHA384
 *      MBEDTLS_TLS_DHE_PSK_WITH_CAMELLIA_256_CBC_SHA384
 *      MBEDTLS_TLS_DHE_PSK_WITH_AES_128_GCM_SHA256
 *      MBEDTLS_TLS_DHE_PSK_WITH_AES_128_CBC_SHA256
 *      MBEDTLS_TLS_DHE_PSK_WITH_AES_128_CBC_SHA
 *      MBEDTLS_TLS_DHE_PSK_WITH_CAMELLIA_128_GCM_SHA256
 *      MBEDTLS_TLS_DHE_PSK_WITH_CAMELLIA_128_CBC_SHA256
 *      MBEDTLS_TLS_DHE_PSK_WITH_3DES_EDE_CBC_SHA
 *      MBEDTLS_TLS_DHE_PSK_WITH_RC4_128_SHA
 *
 * \warning    Using DHE constitutes a security risk as it
 *             is not possible to validate custom DH parameters.
 *             If possible, it is recommended users should consider
 *             preferring other methods of key exchange.
 *             See dhm.h for more details.
 *
 */
#define MBEDTLS_KEY_EXCHANGE_DHE_PSK_ENABLED

/**
 * \def MBEDTLS_KEY_EXCHANGE_ECDHE_PSK_ENABLED
 *
 * Enable the ECDHE-PSK based ciphersuite modes in SSL / TLS.
 *
 * Requires: MBEDTLS_ECDH_C
 *
 * This enables the following ciphersuites (if other requisites are
 * enabled as well):
 *      MBEDTLS_TLS_ECDHE_PSK_WITH_AES_256_CBC_SHA384
 *      MBEDTLS_TLS_ECDHE_PSK_WITH_AES_256_CBC_SHA
 *      MBEDTLS_TLS_ECDHE_PSK_WITH_CAMELLIA_256_CBC_SHA384
 *      MBEDTLS_TLS_ECDHE_PSK_WITH_AES_128_CBC_SHA256
 *      MBEDTLS_TLS_ECDHE_PSK_WITH_AES_128_CBC_SHA
 *      MBEDTLS_TLS_ECDHE_PSK_WITH_CAMELLIA_128_CBC_SHA256
 *      MBEDTLS_TLS_ECDHE_PSK_WITH_3DES_EDE_CBC_SHA
 *      MBEDTLS_TLS_ECDHE_PSK_WITH_RC4_128_SHA
 */
#define MBEDTLS_KEY_EXCHANGE_ECDHE_PSK_ENABLED

/**
 * \def MBEDTLS_KEY_EXCHANGE_RSA_PSK_ENABLED
 *
 * Enable the RSA-PSK based ciphersuite modes in SSL / TLS.
 *
 * Requires: MBEDTLS_RSA_C, MBEDTLS_PKCS1_V15,
 *           MBEDTLS_X509_CRT_PARSE_C
 *
 * This enables the following ciphersuites (if other requisites are
 * enabled as well):
 *      MBEDTLS_TLS_RSA_PSK_WITH_AES_256_GCM_SHA384
 *      MBEDTLS_TLS_RSA_PSK_WITH_AES_256_CBC_SHA384
 *      MBEDTLS_TLS_RSA_PSK_WITH_AES_256_CBC_SHA
 *      MBEDTLS_TLS_RSA_PSK_WITH_CAMELLIA_256_GCM_SHA384
 *      MBEDTLS_TLS_RSA_PSK_WITH_CAMELLIA_256_CBC_SHA384
 *      MBEDTLS_TLS_RSA_PSK_WITH_AES_128_GCM_SHA256
 *      MBEDTLS_TLS_RSA_PSK_WITH_AES_128_CBC_SHA256
 *      MBEDTLS_TLS_RSA_PSK_WITH_AES_128_CBC_SHA
 *      MBEDTLS_TLS_RSA_PSK_WITH_CAMELLIA_128_GCM_SHA256
 *      MBEDTLS_TLS_RSA_PSK_WITH_CAMELLIA_128_CBC_SHA256
 *      MBEDTLS_TLS_RSA_PSK_WITH_3DES_EDE_CBC_SHA
 *      MBEDTLS_TLS_RSA_PSK_WITH_RC4_128_SHA
 */
#define MBEDTLS_KEY_EXCHANGE_RSA_PSK_ENABLED

/**
 * \def MBEDTLS_KEY_EXCHANGE_RSA_ENABLED
 *
 * Enable the RSA-only based ciphersuite modes in SSL / TLS.
 *
 * Requires: MBEDTLS_RSA_C, MBEDTLS_PKCS1_V15,
 *           MBEDTLS_X509_CRT_PARSE_C
 *
 * This enables the following ciphersuites (if other requisites are
 * enabled as well):
 *      MBEDTLS_TLS_RSA_WITH_AES_256_GCM_SHA384
 *      MBEDTLS_TLS_RSA_WITH_AES_256_CBC_SHA256
 *      MBEDTLS_TLS_RSA_WITH_AES_256_CBC_SHA
 *      MBEDTLS_TLS_RSA_WITH_CAMELLIA_256_GCM_SHA384
 *      MBEDTLS_TLS_RSA_WITH_CAMELLIA_256_CBC_SHA256
 *      MBEDTLS_TLS_RSA_WITH_CAMELLIA_256_CBC_SHA
 *      MBEDTLS_TLS_RSA_WITH_AES_128_GCM_SHA256
 *      MBEDTLS_TLS_RSA_WITH_AES_128_CBC_SHA256
 *      MBEDTLS_TLS_RSA_WITH_AES_128_CBC_SHA
 *      MBEDTLS_TLS_RSA_WITH_CAMELLIA_128_GCM_SHA256
 *      MBEDTLS_TLS_RSA_WITH_CAMELLIA_128_CBC_SHA256
 *      MBEDTLS_TLS_RSA_WITH_CAMELLIA_128_CBC_SHA
 *      MBEDTLS_TLS_RSA_WITH_3DES_EDE_CBC_SHA
 *      MBEDTLS_TLS_RSA_WITH_RC4_128_SHA
 *      MBEDTLS_TLS_RSA_WITH_RC4_128_MD5
 */
#define MBEDTLS_KEY_EXCHANGE_RSA_ENABLED

/**
 * \def MBEDTLS_KEY_EXCHANGE_DHE_RSA_ENABLED
 *
 * Enable the DHE-RSA based ciphersuite modes in SSL / TLS.
 *
 * Requires: MBEDTLS_DHM_C, MBEDTLS_RSA_C, MBEDTLS_PKCS1_V15,
 *           MBEDTLS_X509_CRT_PARSE_C
 *
 * This enables the following ciphersuites (if other requisites are
 * enabled as well):
 *      MBEDTLS_TLS_DHE_RSA_WITH_AES_256_GCM_SHA384
 *      MBEDTLS_TLS_DHE_RSA_WITH_AES_256_CBC_SHA256
 *      MBEDTLS_TLS_DHE_RSA_WITH_AES_256_CBC_SHA
 *      MBEDTLS_TLS_DHE_RSA_WITH_CAMELLIA_256_GCM_SHA384
 *      MBEDTLS_TLS_DHE_RSA_WITH_CAMELLIA_256_CBC_SHA256
 *      MBEDTLS_TLS_DHE_RSA_WITH_CAMELLIA_256_CBC_SHA
 *      MBEDTLS_TLS_DHE_RSA_WITH_AES_128_GCM_SHA256
 *      MBEDTLS_TLS_DHE_RSA_WITH_AES_128_CBC_SHA256
 *      MBEDTLS_TLS_DHE_RSA_WITH_AES_128_CBC_SHA
 *      MBEDTLS_TLS_DHE_RSA_WITH_CAMELLIA_128_GCM_SHA256
 *      MBEDTLS_TLS_DHE_RSA_WITH_CAMELLIA_128_CBC_SHA256
 *      MBEDTLS_TLS_DHE_RSA_WITH_CAMELLIA_128_CBC_SHA
 *      MBEDTLS_TLS_DHE_RSA_WITH_3DES_EDE_CBC_SHA
 *
 * \warning    Using DHE constitutes a security risk as it
 *             is not possible to validate custom DH parameters.
 *             If possible, it is recommended users should consider
 *             preferring other methods of key exchange.
 *             See dhm.h for more details.
 *
 */
#define MBEDTLS_KEY_EXCHANGE_DHE_RSA_ENABLED

/**
 * \def MBEDTLS_KEY_EXCHANGE_ECDHE_RSA_ENABLED
 *
 * Enable the ECDHE-RSA based ciphersuite modes in SSL / TLS.
 *
 * Requires: MBEDTLS_ECDH_C, MBEDTLS_RSA_C, MBEDTLS_PKCS1_V15,
 *           MBEDTLS_X509_CRT_PARSE_C
 *
 * This enables the following ciphersuites (if other requisites are
 * enabled as well):
 *      MBEDTLS_TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384
 *      MBEDTLS_TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384
 *      MBEDTLS_TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA
 *      MBEDTLS_TLS_ECDHE_RSA_WITH_CAMELLIA_256_GCM_SHA384
 *      MBEDTLS_TLS_ECDHE_RSA_WITH_CAMELLIA_256_CBC_SHA384
 *      MBEDTLS_TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
 *      MBEDTLS_TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256
 *      MBEDTLS_TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA
 *      MBEDTLS_TLS_ECDHE_RSA_WITH_CAMELLIA_128_GCM_SHA256
 *      MBEDTLS_TLS_ECDHE_RSA_WITH_CAMELLIA_128_CBC_SHA256
 *      MBEDTLS_TLS_ECDHE_RSA_WITH_3DES_EDE_CBC_SHA
 *      MBEDTLS_TLS_ECDHE_RSA_WITH_RC4_128_SHA
 */
#define MBEDTLS_KEY_EXCHANGE_ECDHE_RSA_ENABLED

/**
 * \def MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA_ENABLED
 *
 * Enable the ECDHE-ECDSA based ciphersuite modes in SSL / TLS.
 *
 * Requires: MBEDTLS_ECDH_C, MBEDTLS_ECDSA_C, MBEDTLS_X509_CRT_PARSE_C,
 *
 * This enables the following ciphersuites (if other requisites are
 * enabled as well):
 *      MBEDTLS_TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384
 *      MBEDTLS_TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA384
 *      MBEDTLS_TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA
 *      MBEDTLS_TLS_ECDHE_ECDSA_WITH_CAMELLIA_256_GCM_SHA384
 *      MBEDTLS_TLS_ECDHE_ECDSA_WITH_CAMELLIA_256_CBC_SHA384
 *      MBEDTLS_TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256
 *      MBEDTLS_TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256
 *      MBEDTLS_TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA
 *      MBEDTLS_TLS_ECDHE_ECDSA_WITH_CAMELLIA_128_GCM_SHA256
 *      MBEDTLS_TLS_ECDHE_ECDSA_WITH_CAMELLIA_128_CBC_SHA256
 *      MBEDTLS_TLS_ECDHE_ECDSA_WITH_3DES_EDE_CBC_SHA
 *      MBEDTLS_TLS_ECDHE_ECDSA_WITH_RC4_128_SHA
 */
#define MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA_ENABLED

/**
 * \def MBEDTLS_KEY_EXCHANGE_ECDH_ECDSA_ENABLED
 *
 * Enable the ECDH-ECDSA based ciphersuite modes in SSL / TLS.
 *
 * Requires: MBEDTLS_ECDH_C, MBEDTLS_X509_CRT_PARSE_C
 *
 * This enables the following ciphersuites (if other requisites are
 * enabled as well):
 *      MBEDTLS_TLS_ECDH_ECDSA_WITH_RC4_128_SHA
 *      MBEDTLS_TLS_ECDH_ECDSA_WITH_3DES_EDE_CBC_SHA
 *      MBEDTLS_TLS_ECDH_ECDSA_WITH_AES_128_CBC_SHA
 *      MBEDTLS_TLS_ECDH_ECDSA_WITH_AES_256_CBC_SHA
 *      MBEDTLS_TLS_ECDH_ECDSA_WITH_AES_128_CBC_SHA256
 *      MBEDTLS_TLS_ECDH_ECDSA_WITH_AES_256_CBC_SHA384
 *      MBEDTLS_TLS_ECDH_ECDSA_WITH_AES_128_GCM_SHA256
 *      MBEDTLS_TLS_ECDH_ECDSA_WITH_AES_256_GCM_SHA384
 *      MBEDTLS_TLS_ECDH_ECDSA_WITH_CAMELLIA_128_CBC_SHA256
 *      MBEDTLS_TLS_ECDH_ECDSA_WITH_CAMELLIA_256_CBC_SHA384
 *      MBEDTLS_TLS_ECDH_ECDSA_WITH_CAMELLIA_128_GCM_SHA256
 *      MBEDTLS_TLS_ECDH_ECDSA_WITH_CAMELLIA_256_GCM_SHA384
 */
#define MBEDTLS_KEY_EXCHANGE_ECDH_ECDSA_ENABLED

/**
 * \def MBEDTLS_KEY_EXCHANGE_ECDH_RSA_ENABLED
 *
 * Enable the ECDH-RSA based ciphersuite modes in SSL / TLS.
 *
 * Requires: MBEDTLS_ECDH_C, MBEDTLS_X509_CRT_PARSE_C
 *
 * This enables the following ciphersuites (if other requisites are
 * enabled as well):
 *      MBEDTLS_TLS_ECDH_RSA_WITH_RC4_128_SHA
 *      MBEDTLS_TLS_ECDH_RSA_WITH_3DES_EDE_CBC_SHA
 *      MBEDTLS_TLS_ECDH_RSA_WITH_AES_128_CBC_SHA
 *      MBEDTLS_TLS_ECDH_RSA_WITH_AES_256_CBC_SHA
 *      MBEDTLS_TLS_ECDH_RSA_WITH_AES_128_CBC_SHA256
 *      MBEDTLS_TLS_ECDH_RSA_WITH_AES_256_CBC_SHA384
 *      MBEDTLS_TLS_ECDH_RSA_WITH_AES_128_GCM_SHA256
 *      MBEDTLS_TLS_ECDH_RSA_WITH_AES_256_GCM_SHA384
 *      MBEDTLS_TLS_ECDH_RSA_WITH_CAMELLIA_128_CBC_SHA256
 *      MBEDTLS_TLS_ECDH_RSA_WITH_CAMELLIA_256_CBC_SHA384
 *      MBEDTLS_TLS_ECDH_RSA_WITH_CAMELLIA_128_GCM_SHA256
 *      MBEDTLS_TLS_ECDH_RSA_WITH_CAMELLIA_256_GCM_SHA384
 */
#define MBEDTLS_KEY_EXCHANGE_ECDH_RSA_ENABLED

/**
 * \def MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED
 *
 * Enable the ECJPAKE based ciphersuite modes in SSL / TLS.
 *
 * \warning This is currently experimental. EC J-PAKE support is based on the
 * Thread v1.0.0 specification; incompatible changes to the specification
 * might still happen. For this reason, this is disabled by default.
 *
 * Requires: MBEDTLS_ECJPAKE_C
 *           MBEDTLS_SHA256_C
 *           MBEDTLS_ECP_DP_SECP256R1_ENABLED
 *
 * This enables the following ciphersuites (if other requisites are
 * enabled as well):
 *      MBEDTLS_TLS_ECJPAKE_WITH_AES_128_CCM_8
 */
//#define MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED

/**
 * \def MBEDTLS_PK_PARSE_EC_EXTENDED
 *
 * Enhance support for reading EC keys using variants of SEC1 not allowed by
 * RFC 5915 and RFC 5480.
 *
 * Currently this means parsing the SpecifiedECDomain choice of EC
 * parameters (only known groups are supported, not arbitrary domains, to
 * avoid validation issues).
 *
 * Disable if you only need to support RFC 5915 + 5480 key formats.
 */
#define MBEDTLS_PK_PARSE_EC_EXTENDED

/**
 * \def MBEDTLS_ERROR_STRERROR_DUMMY
 *
 * Enable a dummy error function to make use of mbedtls_strerror() in
 * third party libraries easier when MBEDTLS_ERROR_C is disabled
 * (no effect when MBEDTLS_ERROR_C is enabled).
 *
 * You can safely disable this if MBEDTLS_ERROR_C is enabled, or if you're
 * not using mbedtls_strerror() or error_strerror() in your application.
 *
 * Disable if you run into name conflicts and want to really remove the
 * mbedtls_strerror()
 */
#define MBEDTLS_ERROR_STRERROR_DUMMY

/**
 * \def MBEDTLS_GENPRIME
 *
 * Enable the prime-number generation code.
 *
 * Requires: MBEDTLS_BIGNUM_C
 */
#define MBEDTLS_GENPRIME

/**
 * \def MBEDTLS_FS_IO
 *
 * Enable functions that use the filesystem.
 */
#define MBEDTLS_FS_IO

/**
 * \def MBEDTLS_NO_DEFAULT_ENTROPY_SOURCES
 *
 * Do not add default entropy sources. These are the platform specific,
 * mbedtls_timing_hardclock and HAVEGE based poll functions.
 *
 * This is useful to have more control over the added entropy sources in an
 * application.
 *
 * Uncomment this macro to prevent loading of default entropy functions.
 */
//#define MBEDTLS_NO_DEFAULT_ENTROPY_SOURCES

/**
 * \def MBEDTLS_NO_PLATFORM_ENTROPY
 *
 * Do not use built-in platform entropy functions.
 * This is useful if your platform does not support
 * standards like the /dev/urandom or Windows CryptoAPI.
 *
 * Uncomment this macro to disable the built-in platform entropy functions.
 */
//#define MBEDTLS_NO_PLATFORM_ENTROPY

/**
 * \def MBEDTLS_ENTROPY_FORCE_SHA256
 *
 * Force the entropy accumulator to use a SHA-256 accumulator instead of the
 * default SHA-512 based one (if both are available).
 *
 * Requires: MBEDTLS_SHA256_C
 *
 * On 32-bit systems SHA-256 can be much faster than SHA-512. Use this option
 * if you have performance concerns.
 *
 * This option is only useful if both MBEDTLS_SHA256_C and
 * MBEDTLS_SHA512_C are defined. Otherwise the available hash module is used.
 */
//#define MBEDTLS_ENTROPY_FORCE_SHA256

/**
 * \def MBEDTLS_ENTROPY_NV_SEED
 *
 * Enable the non-volatile (NV) seed file-based entropy source.
 * (Also enables the NV seed read/write functions in the platform layer)
 *
 * This is crucial (if not required) on systems that do not have a
 * cryptographic entropy source (in hardware or kernel) available.
 *
 * Requires: MBEDTLS_ENTROPY_C, MBEDTLS_PLATFORM_C
 *
 * \note The read/write functions that are used by the entropy source are
 *       determined in the platform layer, and can be modified at runtime and/or
 *       compile-time depending on the flags (MBEDTLS_PLATFORM_NV_SEED_*) used.
 *
 * \note If you use the default implementation functions that read a seedfile
 *       with regular fopen(), please make sure you make a seedfile with the
 *       proper name (defined in MBEDTLS_PLATFORM_STD_NV_SEED_FILE) and at
 *       least MBEDTLS_ENTROPY_BLOCK_SIZE bytes in size that can be read from
 *       and written to or you will get an entropy source error! The default
 *       implementation will only use the first MBEDTLS_ENTROPY_BLOCK_SIZE
 *       bytes from the file.
 *
 * \note The entropy collector will write to the seed file before entropy is
 *       given to an external source, to update it.
 */
//#define MBEDTLS_ENTROPY_NV_SEED

/**
 * \def MBEDTLS_MEMORY_DEBUG
 *
 * Enable debugging of buffer allocator memory issues. Automatically prints
 * (to stderr) all (fatal) messages on memory allocation issues. Enables
 * function for 'debug output' of allocated memory.
 *
 * Requires: MBEDTLS_MEMORY_BUFFER_ALLOC_C
 *
 * Uncomment this macro to let the buffer allocator print out error messages.
 */
//#define MBEDTLS_MEMORY_DEBUG

/**
 * \def MBEDTLS_MEMORY_BACKTRACE
 *
 * Include backtrace information with each allocated block.
 *
 * Requires: MBEDTLS_MEMORY_BUFFER_ALLOC_C
 *           GLIBC-compatible backtrace() an backtrace_symbols() support
 *
 * Uncomment this macro to include backtrace information
 */
//#define MBEDTLS_MEMORY_BACKTRACE

/**
 * \def MBEDTLS_PK_RSA_ALT_SUPPORT
 *
 * Support external private RSA keys (eg from a HSM) in the PK layer.
 *
 * Comment this macro to disable support for external private RSA keys.
 */
#define MBEDTLS_PK_RSA_ALT_SUPPORT

/**
 * \def MBEDTLS_PKCS1_V15
 *
 * Enable support for PKCS#1 v1.5 encoding.
 *
 * Requires: MBEDTLS_RSA_C
 *
 * This enables support for PKCS#1 v1.5 operations.
 */
#define MBEDTLS_PKCS1_V15

/**
 * \def MBEDTLS_PKCS1_V21
 *
 * Enable support for PKCS#1 v2.1 encoding.
 *
 * Requires: MBEDTLS_MD_C, MBEDTLS_RSA_C
 *
 * This enables support for RSAES-OAEP and RSASSA-PSS operations.
 */
#define MBEDTLS_PKCS1_V21

/**
 * \def MBEDTLS_RSA_NO_CRT
 *
 * Do not use the Chinese Remainder Theorem for the RSA private operation.
 *
 * Uncomment this macro to disable the use of CRT in RSA.
 *
 */
//#define MBEDTLS_RSA_NO_CRT

/**
 * \def MBEDTLS_SELF_TEST
 *
 * Enable the checkup functions (*_self_test).
 */
#define MBEDTLS_SELF_TEST

/**
 * \def MBEDTLS_SHA256_SMALLER
 *
 * Enable an implementation of SHA-256 that has lower ROM footprint but also
 * lower performance.
 *
 * The default implementation is meant to be a reasonnable compromise between
 * performance and size. This version optimizes more aggressively for size at
 * the expense of performance. Eg on Cortex-M4 it reduces the size of
 * mbedtls_sha256_process() from ~2KB to ~0.5KB for a performance hit of about
 * 30%.
 *
 * Uncomment to enable the smaller implementation of SHA256.
 */
//#define MBEDTLS_SHA256_SMALLER

/**
 * \def MBEDTLS_SSL_ALL_ALERT_MESSAGES
 *
 * Enable sending of alert messages in case of encountered errors as per RFC.
 * If you choose not to send the alert messages, mbed TLS can still communicate
 * with other servers, only debugging of failures is harder.
 *
 * The advantage of not sending alert messages, is that no information is given
 * about reasons for failures thus preventing adversaries of gaining intel.
 *
 * Enable sending of all alert messages
 */
#define MBEDTLS_SSL_ALL_ALERT_MESSAGES

/**
 * \def MBEDTLS_SSL_DEBUG_ALL
 *
 * Enable the debug messages in SSL module for all issues.
 * Debug messages have been disabled in some places to prevent timing
 * attacks due to (unbalanced) debugging function calls.
 *
 * If you need all error reporting you should enable this during debugging,
 * but remove this for production servers that should log as well.
 *
 * Uncomment this macro to report all debug messages on errors introducing
 * a timing side-channel.
 *
 */
//#define MBEDTLS_SSL_DEBUG_ALL

/** \def MBEDTLS_SSL_ENCRYPT_THEN_MAC
 *
 * Enable support for Encrypt-then-MAC, RFC 7366.
 *
 * This allows peers that both support it to use a more robust protection for
 * ciphersuites using CBC, providing deep resistance against timing attacks
 * on the padding or underlying cipher.
 *
 * This only affects CBC ciphersuites, and is useless if none is defined.
 *
 * Requires: MBEDTLS_SSL_PROTO_TLS1    or
 *           MBEDTLS_SSL_PROTO_TLS1_1  or
 *           MBEDTLS_SSL_PROTO_TLS1_2
 *
 * Comment this macro to disable support for Encrypt-then-MAC
 */
#define MBEDTLS_SSL_ENCRYPT_THEN_MAC

/** \def MBEDTLS_SSL_EXTENDED_MASTER_SECRET
 *
 * Enable support for Extended Master Secret, aka Session Hash
 * (draft-ietf-tls-session-hash-02).
 *
 * This was introduced as "the proper fix" to the Triple Handshake familiy of
 * attacks, but it is recommended to always use it (even if you disable
 * renegotiation), since it actually fixes a more fundamental issue in the
 * original SSL/TLS design, and has implications beyond Triple Handshake.
 *
 * Requires: MBEDTLS_SSL_PROTO_TLS1    or
 *           MBEDTLS_SSL_PROTO_TLS1_1  or
 *           MBEDTLS_SSL_PROTO_TLS1_2
 *
 * Comment this macro to disable support for Extended Master Secret.
 */
#define MBEDTLS_SSL_EXTENDED_MASTER_SECRET

/**
 * \def MBEDTLS_SSL_FALLBACK_SCSV
 *
 * Enable support for FALLBACK_SCSV (draft-ietf-tls-downgrade-scsv-00).
 *
 * For servers, it is recommended to always enable this, unless you support
 * only one version of TLS, or know for sure that none of your clients
 * implements a fallback strategy.
 *
 * For clients, you only need this if you're using a fallback strategy, which
 * is not recommended in the first place, unless you absolutely need it to
 * interoperate with buggy (version-intolerant) servers.
 *
 * Comment this macro to disable support for FALLBACK_SCSV
 */
#define MBEDTLS_SSL_FALLBACK_SCSV

/**
 * \def MBEDTLS_SSL_HW_RECORD_ACCEL
 *
 * Enable hooking functions in SSL module for hardware acceleration of
 * individual records.
 *
 * Uncomment this macro to enable hooking functions.
 */
//#define MBEDTLS_SSL_HW_RECORD_ACCEL

/**
 * \def MBEDTLS_SSL_CBC_RECORD_SPLITTING
 *
 * Enable 1/n-1 record splitting for CBC mode in SSLv3 and TLS 1.0.
 *
 * This is a countermeasure to the BEAST attack, which also minimizes the risk
 * of interoperability issues compared to sending 0-length records.
 *
 * Comment this macro to disable 1/n-1 record splitting.
 */
#define MBEDTLS_SSL_CBC_RECORD_SPLITTING

/**
 * \def MBEDTLS_SSL_RENEGOTIATION
 *
 * Disable support for TLS renegotiation.
 *
 * The two main uses of renegotiation are (1) refresh keys on long-lived
 * connections and (2) client authentication after the initial handshake.
 * If you don't need renegotiation, it's probably better to disable it, since
 * it has been associated with security issues in the past and is easy to
 * misuse/misunderstand.
 *
 * Comment this to disable support for renegotiation.
 *
 * \note   Even if this option is disabled, both client and server are aware
 *         of the Renegotiation Indication Extension (RFC 5746) used to
 *         prevent the SSL renegotiation attack (see RFC 5746 Sect. 1).
 *         (See \c mbedtls_ssl_conf_legacy_renegotiation for the
 *          configuration of this extension).
 *
 */
#define MBEDTLS_SSL_RENEGOTIATION

/**
 * \def MBEDTLS_SSL_SRV_SUPPORT_SSLV2_CLIENT_HELLO
 *
 * Enable support for receiving and parsing SSLv2 Client Hello messages for the
 * SSL Server module (MBEDTLS_SSL_SRV_C).
 *
 * Uncomment this macro to enable support for SSLv2 Client Hello messages.
 */
//#define MBEDTLS_SSL_SRV_SUPPORT_SSLV2_CLIENT_HELLO

/**
 * \def MBEDTLS_SSL_SRV_RESPECT_CLIENT_PREFERENCE
 *
 * Pick the ciphersuite according to the client's preferences rather than ours
 * in the SSL Server module (MBEDTLS_SSL_SRV_C).
 *
 * Uncomment this macro to respect client's ciphersuite order
 */
//#define MBEDTLS_SSL_SRV_RESPECT_CLIENT_PREFERENCE

/**
 * \def MBEDTLS_SSL_MAX_FRAGMENT_LENGTH
 *
 * Enable support for RFC 6066 max_fragment_length extension in SSL.
 *
 * Comment this macro to disable support for the max_fragment_length extension
 */
#define MBEDTLS_SSL_MAX_FRAGMENT_LENGTH

/**
 * \def MBEDTLS_SSL_PROTO_SSL3
 *
 * Enable support for SSL 3.0.
 *
 * Requires: MBEDTLS_MD5_C
 *           MBEDTLS_SHA1_C
 *
 * Comment this macro to disable support for SSL 3.0
 */
//#define MBEDTLS_SSL_PROTO_SSL3

/**
 * \def MBEDTLS_SSL_PROTO_TLS1
 *
 * Enable support for TLS 1.0.
 *
 * Requires: MBEDTLS_MD5_C
 *           MBEDTLS_SHA1_C
 *
 * Comment this macro to disable support for TLS 1.0
 */
#define MBEDTLS_SSL_PROTO_TLS1

/**
 * \def MBEDTLS_SSL_PROTO_TLS1_1
 *
 * Enable support for TLS 1.1 (and DTLS 1.0 if DTLS is enabled).
 *
 * Requires: MBEDTLS_MD5_C
 *           MBEDTLS_SHA1_C
 *
 * Comment this macro to disable support for TLS 1.1 / DTLS 1.0
 */
#define MBEDTLS_SSL_PROTO_TLS1_1

/**
 * \def MBEDTLS_SSL_PROTO_TLS1_2
 *
 * Enable support for TLS 1.2 (and DTLS 1.2 if DTLS is enabled).
 *
 * Requires: MBEDTLS_SHA1_C or MBEDTLS_SHA256_C or MBEDTLS_SHA512_C
 *           (Depends on ciphersuites)
 *
 * Comment this macro to disable support for TLS 1.2 / DTLS 1.2
 */
#define MBEDTLS_SSL_PROTO_TLS1_2

/**
 * \def MBEDTLS_SSL_PROTO_DTLS
 *
 * Enable support for DTLS (all available versions).
 *
 * Enable this and MBEDTLS_SSL_PROTO_TLS1_1 to enable DTLS 1.0,
 * and/or this and MBEDTLS_SSL_PROTO_TLS1_2 to enable DTLS 1.2.
 *
 * Requires: MBEDTLS_SSL_PROTO_TLS1_1
 *        or MBEDTLS_SSL_PROTO_TLS1_2
 *
 * Comment this macro to disable support for DTLS
 */
#define MBEDTLS_SSL_PROTO_DTLS

/**
 * \def MBEDTLS_SSL_ALPN
 *
 * Enable support for RFC 7301 Application Layer Protocol Negotiation.
 *
 * Comment this macro to disable support for ALPN.
 */
#define MBEDTLS_SSL_ALPN

/**
 * \def MBEDTLS_SSL_DTLS_ANTI_REPLAY
 *
 * Enable support for the anti-replay mechanism in DTLS.
 *
 * Requires: MBEDTLS_SSL_TLS_C
 *           MBEDTLS_SSL_PROTO_DTLS
 *
 * \warning Disabling this is often a security risk!
 * See mbedtls_ssl_conf_dtls_anti_replay() for details.
 *
 * Comment this to disable anti-replay in DTLS.
 */
#define MBEDTLS_SSL_DTLS_ANTI_REPLAY

/**
 * \def MBEDTLS_SSL_DTLS_HELLO_VERIFY
 *
 * Enable support for HelloVerifyRequest on DTLS servers.
 *
 * This feature is highly recommended to prevent DTLS servers being used as
 * amplifiers in DoS attacks against other hosts. It should always be enabled
 * unless you know for sure amplification cannot be a problem in the
 * environment in which your server operates.
 *
 * \warning Disabling this can ba a security risk! (see above)
 *
 * Requires: MBEDTLS_SSL_PROTO_DTLS
 *
 * Comment this to disable support for HelloVerifyRequest.
 */
#define MBEDTLS_SSL_DTLS_HELLO_VERIFY

/**
 * \def MBEDTLS_SSL_DTLS_CLIENT_PORT_REUSE
 *
 * Enable server-side support for clients that reconnect from the same port.
 *
 * Some clients unexpectedly close the connection and try to reconnect using the
 * same source port. This needs special support from the server to handle the
 * new connection securely, as described in section 4.2.8 of RFC 6347. This
 * flag enables that support.
 *
 * Requires: MBEDTLS_SSL_DTLS_HELLO_VERIFY
 *
 * Comment this to disable support for clients reusing the source port.
 */
#define MBEDTLS_SSL_DTLS_CLIENT_PORT_REUSE

/**
 * \def MBEDTLS_SSL_DTLS_BADMAC_LIMIT
 *
 * Enable support for a limit of records with bad MAC.
 *
 * See mbedtls_ssl_conf_dtls_badmac_limit().
 *
 * Requires: MBEDTLS_SSL_PROTO_DTLS
 */
#define MBEDTLS_SSL_DTLS_BADMAC_LIMIT

/**
 * \def MBEDTLS_SSL_SESSION_TICKETS
 *
 * Enable support for RFC 5077 session tickets in SSL.
 * Client-side, provides full support for session tickets (maintainance of a
 * session store remains the responsibility of the application, though).
 * Server-side, you also need to provide callbacks for writing and parsing
 * tickets, including authenticated encryption and key management. Example
 * callbacks are provided by MBEDTLS_SSL_TICKET_C.
 *
 * Comment this macro to disable support for SSL session tickets
 */
#define MBEDTLS_SSL_SESSION_TICKETS

/**
 * \def MBEDTLS_SSL_EXPORT_KEYS
 *
 * Enable support for exporting key block and master secret.
 * This is required for certain users of TLS, e.g. EAP-TLS.
 *
 * Comment this macro to disable support for key export
 */
#define MBEDTLS_SSL_EXPORT_KEYS

/**
 * \def MBEDTLS_SSL_SERVER_NAME_INDICATION
 *
 * Enable support for RFC 6066 server name indication (SNI) in SSL.
 *
 * Requires: MBEDTLS_X509_CRT_PARSE_C
 *
 * Comment this macro to disable support for server name indication in SSL
 */
#define MBEDTLS_SSL_SERVER_NAME_INDICATION

/**
 * \def MBEDTLS_SSL_TRUNCATED_HMAC
 *
 * Enable support for RFC 6066 truncated HMAC in SSL.
 *
 * Comment this macro to disable support for truncated HMAC in SSL
 */
#define MBEDTLS_SSL_TRUNCATED_HMAC

/**
 * \def MBEDTLS_THREADING_ALT
 *
 * Provide your own alternate threading implementation.
 *
 * Requires: MBEDTLS_THREADING_C
 *
 * Uncomment this to allow your own alternate threading implementation.
 */
//#define MBEDTLS_THREADING_ALT

/**
 * \def MBEDTLS_THREADING_PTHREAD
 *
 * Enable the pthread wrapper layer for the threading layer.
 *
 * Requires: MBEDTLS_THREADING_C
 *
 * Uncomment this to enable pthread mutexes.
 */
//#define MBEDTLS_THREADING_PTHREAD

/**
 * \def MBEDTLS_VERSION_FEATURES
 *
 * Allow run-time checking of compile-time enabled features. Thus allowing users
 * to check at run-time if the library is for instance compiled with threading
 * support via mbedtls_version_check_feature().
 *
 * Requires: MBEDTLS_VERSION_C
 *
 * Comment this to disable run-time checking and save ROM space
 */
#define MBEDTLS_VERSION_FEATURES

/**
 * \def MBEDTLS_X509_ALLOW_EXTENSIONS_NON_V3
 *
 * If set, the X509 parser will not break-off when parsing an X509 certificate
 * and encountering an extension in a v1 or v2 certificate.
 *
 * Uncomment to prevent an error.
 */
//#define MBEDTLS_X509_ALLOW_EXTENSIONS_NON_V3

/**
 * \def MBEDTLS_X509_ALLOW_UNSUPPORTED_CRITICAL_EXTENSION
 *
 * If set, the X509 parser will not break-off when parsing an X509 certificate
 * and encountering an unknown critical extension.
 *
 * \warning Depending on your PKI use, enabling this can be a security risk!
 *
 * Uncomment to prevent an error.
 */
//#define MBEDTLS_X509_ALLOW_UNSUPPORTED_CRITICAL_EXTENSION

/**
 * \def MBEDTLS_X509_CHECK_KEY_USAGE
 *
 * Enable verification of the keyUsage extension (CA and leaf certificates).
 *
 * Disabling this avoids problems with mis-issued and/or misused
 * (intermediate) CA and leaf certificates.
 *
 * \warning Depending on your PKI use, disabling this can be a security risk!
 *
 * Comment to skip keyUsage checking for both CA and leaf certificates.
 */
#define MBEDTLS_X509_CHECK_KEY_USAGE

/**
 * \def MBEDTLS_X509_CHECK_EXTENDED_KEY_USAGE
 *
 * Enable verification of the extendedKeyUsage extension (leaf certificates).
 *
 * Disabling this avoids problems with mis-issued and/or misused certificates.
 *
 * \warning Depending on your PKI use, disabling this can be a security risk!
 *
 * Comment to skip extendedKeyUsage checking for certificates.
 */
#define MBEDTLS_X509_CHECK_EXTENDED_KEY_USAGE

/**
 * \def MBEDTLS_X509_RSASSA_PSS_SUPPORT
 *
 * Enable parsing and verification of X.509 certificates, CRLs and CSRS
 * signed with RSASSA-PSS (aka PKCS#1 v2.1).
 *
 * Comment this macro to disallow using RSASSA-PSS in certificates.
 */
#define MBEDTLS_X509_RSASSA_PSS_SUPPORT

/**
 * \def MBEDTLS_ZLIB_SUPPORT
 *
 * If set, the SSL/TLS module uses ZLIB to support compression and
 * decompression of packet data.
 *
 * \warning TLS-level compression MAY REDUCE SECURITY! See for example the
 * CRIME attack. Before enabling this option, you should examine with care if
 * CRIME or similar exploits may be a applicable to your use case.
 *
 * \note Currently compression can't be used with DTLS.
 *
 * Used in: library/ssl_tls.c
 *          library/ssl_cli.c
 *          library/ssl_srv.c
 *
 * This feature requires zlib library and headers to be present.
 *
 * Uncomment to enable use of ZLIB
 */
//#define MBEDTLS_ZLIB_SUPPORT
/* \} name SECTION: mbed TLS feature support */

/**
 * \name SECTION: mbed TLS modules
 *
 * This section enables or disables entire modules in mbed TLS
 * \{
 */

/**
 * \def MBEDTLS_AESNI_C
 *
 * Enable AES-NI support on x86-64.
 *
 * Module:  library/aesni.c
 * Caller:  library/aes.c
 *
 * Requires: MBEDTLS_HAVE_ASM
 *
 * This modules adds support for the AES-NI instructions on x86-64
 */
#define MBEDTLS_AESNI_C

/**
 * \def MBEDTLS_AES_C
 *
 * Enable the AES block cipher.
 *
 * Module:  library/aes.c
 * Caller:  library/ssl_tls.c
 *          library/pem.c
 *          library/ctr_drbg.c
 *
 * This module enables the following ciphersuites (if other requisites are
 * enabled as well):
 *      MBEDTLS_TLS_ECDH_ECDSA_WITH_AES_128_CBC_SHA
 *      MBEDTLS_TLS_ECDH_ECDSA_WITH_AES_256_CBC_SHA
 *      MBEDTLS_TLS_ECDH_RSA_WITH_AES_128_CBC_SHA
 *      MBEDTLS_TLS_ECDH_RSA_WITH_AES_256_CBC_SHA
 *      MBEDTLS_TLS_ECDH_ECDSA_WITH_AES_128_CBC_SHA256
 *      MBEDTLS_TLS_ECDH_ECDSA_WITH_AES_256_CBC_SHA384
 *      MBEDTLS_TLS_ECDH_RSA_WITH_AES_128_CBC_SHA256
 *      MBEDTLS_TLS_ECDH_RSA_WITH_AES_256_CBC_SHA384
 *      MBEDTLS_TLS_ECDH_ECDSA_WITH_AES_128_GCM_SHA256
 *      MBEDTLS_TLS_ECDH_ECDSA_WITH_AES_256_GCM_SHA384
 *      MBEDTLS_TLS_ECDH_RSA_WITH_AES_128_GCM_SHA256
 *      MBEDTLS_TLS_ECDH_RSA_WITH_AES_256_GCM_SHA384
 *      MBEDTLS_TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384
 *      MBEDTLS_TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384
 *      MBEDTLS_TLS_DHE_RSA_WITH_AES_256_GCM_SHA384
 *      MBEDTLS_TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA384
 *      MBEDTLS_TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384
 *      MBEDTLS_TLS_DHE_RSA_WITH_AES_256_CBC_SHA256
 *      MBEDTLS_TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA
 *      MBEDTLS_TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA
 *      MBEDTLS_TLS_DHE_RSA_WITH_AES_256_CBC_SHA
 *      MBEDTLS_TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256
 *      MBEDTLS_TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
 *      MBEDTLS_TLS_DHE_RSA_WITH_AES_128_GCM_SHA256
 *      MBEDTLS_TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256
 *      MBEDTLS_TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256
 *      MBEDTLS_TLS_DHE_RSA_WITH_AES_128_CBC_SHA256
 *      MBEDTLS_TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA
 *      MBEDTLS_TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA
 *      MBEDTLS_TLS_DHE_RSA_WITH_AES_128_CBC_SHA
 *      MBEDTLS_TLS_DHE_PSK_WITH_AES_256_GCM_SHA384
 *      MBEDTLS_TLS_ECDHE_PSK_WITH_AES_256_CBC_SHA384
 *      MBEDTLS_TLS_DHE_PSK_WITH_AES_256_CBC_SHA384
 *      MBEDTLS_TLS_ECDHE_PSK_WITH_AES_256_CBC_SHA
 *      MBEDTLS_TLS_DHE_PSK_WITH_AES_256_CBC_SHA
 *      MBEDTLS_TLS_DHE_PSK_WITH_AES_128_GCM_SHA256
 *      MBEDTLS_TLS_ECDHE_PSK_WITH_AES_128_CBC_SHA256
 *      MBEDTLS_TLS_DHE_PSK_WITH_AES_128_CBC_SHA256
 *      MBEDTLS_TLS_ECDHE_PSK_WITH_AES_128_CBC_SHA
 *      MBEDTLS_TLS_DHE_PSK_WITH_AES_128_CBC_SHA
 *      MBEDTLS_TLS_RSA_WITH_AES_256_GCM_SHA384
 *      MBEDTLS_TLS_RSA_WITH_AES_256_CBC_SHA256
 *      MBEDTLS_TLS_RSA_WITH_AES_256_CBC_SHA
 *      MBEDTLS_TLS_RSA_WITH_AES_128_GCM_SHA256
 *      MBEDTLS_TLS_RSA_WITH_AES_128_CBC_SHA256
 *      MBEDTLS_TLS_RSA_WITH_AES_128_CBC_SHA
 *      MBEDTLS_TLS_RSA_PSK_WITH_AES_256_GCM_SHA384
 *      MBEDTLS_TLS_RSA_PSK_WITH_AES_256_CBC_SHA384
 *      MBEDTLS_TLS_RSA_PSK_WITH_AES_256_CBC_SHA
 *      MBEDTLS_TLS_RSA_PSK_WITH_AES_128_GCM_SHA256
 *      MBEDTLS_TLS_RSA_PSK_WITH_AES_128_CBC_SHA256
 *      MBEDTLS_TLS_RSA_PSK_WITH_AES_128_CBC_SHA
 *      MBEDTLS_TLS_PSK_WITH_AES_256_GCM_SHA384
 *      MBEDTLS_TLS_PSK_WITH_AES_256_CBC_SHA384
 *      MBEDTLS_TLS_PSK_WITH_AES_256_CBC_SHA
 *      MBEDTLS_TLS_PSK_WITH_AES_128_GCM_SHA256
 *      MBEDTLS_TLS_PSK_WITH_AES_128_CBC_SHA256
 *      MBEDTLS_TLS_PSK_WITH_AES_128_CBC_SHA
 *
 * PEM_PARSE uses AES for decrypting encrypted keys.
 */
#define MBEDTLS_AES_C

/**
 * \def MBEDTLS_ARC4_C
 *
 * Enable the ARCFOUR stream cipher.
 *
 * Module:  library/arc4.c
 * Caller:  library/ssl_tls.c
 *
 * This module enables the following ciphersuites (if other requisites are
 * enabled as well):
 *      MBEDTLS_TLS_ECDH_ECDSA_WITH_RC4_128_SHA
 *      MBEDTLS_TLS_ECDH_RSA_WITH_RC4_128_SHA
 *      MBEDTLS_TLS_ECDHE_ECDSA_WITH_RC4_128_SHA
 *      MBEDTLS_TLS_ECDHE_RSA_WITH_RC4_128_SHA
 *      MBEDTLS_TLS_ECDHE_PSK_WITH_RC4_128_SHA
 *      MBEDTLS_TLS_DHE_PSK_WITH_RC4_128_SHA
 *      MBEDTLS_TLS_RSA_WITH_RC4_128_SHA
 *      MBEDTLS_TLS_RSA_WITH_RC4_128_MD5
 *      MBEDTLS_TLS_RSA_PSK_WITH_RC4_128_SHA
 *      MBEDTLS_TLS_PSK_WITH_RC4_128_SHA
 *
 * \warning   ARC4 is considered a weak cipher and its use constitutes a
 *            security risk. If possible, we recommend avoidng dependencies on
 *            it, and considering stronger ciphers instead.
 *
 */
#define MBEDTLS_ARC4_C

/**
 * \def MBEDTLS_ASN1_PARSE_C
 *
 * Enable the generic ASN1 parser.
 *
 * Module:  library/asn1.c
 * Caller:  library/x509.c
 *          library/dhm.c
 *          library/pkcs12.c
 *          library/pkcs5.c
 *          library/pkparse.c
 */
#define MBEDTLS_ASN1_PARSE_C

/**
 * \def MBEDTLS_ASN1_WRITE_C
 *
 * Enable the generic ASN1 writer.
 *
 * Module:  library/asn1write.c
 * Caller:  library/ecdsa.c
 *          library/pkwrite.c
 *          library/x509_create.c
 *          library/x509write_crt.c
 *          library/x509write_csr.c
 */
#define MBEDTLS_ASN1_WRITE_C

/**
 * \def MBEDTLS_BASE64_C
 *
 * Enable the Base64 module.
 *
 * Module:  library/base64.c
 * Caller:  library/pem.c
 *
 * This module is required for PEM support (required by X.509).
 */
#define MBEDTLS_BASE64_C

/**
 * \def MBEDTLS_BIGNUM_C
 *
 * Enable the multi-precision integer library.
 *
 * Module:  library/bignum.c
 * Caller:  library/dhm.c
 *          library/ecp.c
 *          library/ecdsa.c
 *          library/rsa.c
 *          library/rsa_internal.c
 *          library/ssl_tls.c
 *
 * This module is required for RSA, DHM and ECC (ECDH, ECDSA) support.
 */
#define MBEDTLS_BIGNUM_C

/**
 * \def MBEDTLS_BLOWFISH_C
 *
 * Enable the Blowfish block cipher.
 *
 * Module:  library/blowfish.c
 */
#define MBEDTLS_BLOWFISH_C

/**
 * \def MBEDTLS_CAMELLIA_C
 *
 * Enable the Camellia block cipher.
 *
 * Module:  library/camellia.c
 * Caller:  library/ssl_tls.c
 *
 * This module enables the following ciphersuites (if other requisites are
 * enabled as well):
 *      MBEDTLS_TLS_ECDH_ECDSA_WITH_CAMELLIA_128_CBC_SHA256
 *      MBEDTLS_TLS_ECDH_ECDSA_WITH_CAMELLIA_256_CBC_SHA384
 *      MBEDTLS_TLS_ECDH_RSA_WITH_CAMELLIA_128_CBC_SHA256
 *      MBEDTLS_TLS_ECDH_RSA_WITH_CAMELLIA_256_CBC_SHA384
 *      MBEDTLS_TLS_ECDH_ECDSA_WITH_CAMELLIA_128_GCM_SHA256
 *      MBEDTLS_TLS_ECDH_ECDSA_WITH_CAMELLIA_256_GCM_SHA384
 *      MBEDTLS_TLS_ECDH_RSA_WITH_CAMELLIA_128_GCM_SHA256
 *      MBEDTLS_TLS_ECDH_RSA_WITH_CAMELLIA_256_GCM_SHA384
 *      MBEDTLS_TLS_ECDHE_ECDSA_WITH_CAMELLIA_256_GCM_SHA384
 *      MBEDTLS_TLS_ECDHE_RSA_WITH_CAMELLIA_256_GCM_SHA384
 *      MBEDTLS_TLS_DHE_RSA_WITH_CAMELLIA_256_GCM_SHA384
 *      MBEDTLS_TLS_ECDHE_ECDSA_WITH_CAMELLIA_256_CBC_SHA384
 *      MBEDTLS_TLS_ECDHE_RSA_WITH_CAMELLIA_256_CBC_SHA384
 *      MBEDTLS_TLS_DHE_RSA_WITH_CAMELLIA_256_CBC_SHA256
 *      MBEDTLS_TLS_DHE_RSA_WITH_CAMELLIA_256_CBC_SHA
 *      MBEDTLS_TLS_ECDHE_ECDSA_WITH_CAMELLIA_128_GCM_SHA256
 *      MBEDTLS_TLS_ECDHE_RSA_WITH_CAMELLIA_128_GCM_SHA256
 *      MBEDTLS_TLS_DHE_RSA_WITH_CAMELLIA_128_GCM_SHA256
 *      MBEDTLS_TLS_ECDHE_ECDSA_WITH_CAMELLIA_128_CBC_SHA256
 *      MBEDTLS_TLS_ECDHE_RSA_WITH_CAMELLIA_128_CBC_SHA256
 *      MBEDTLS_TLS_DHE_RSA_WITH_CAMELLIA_128_CBC_SHA256
 *      MBEDTLS_TLS_DHE_RSA_WITH_CAMELLIA_128_CBC_SHA
 *      MBEDTLS_TLS_DHE_PSK_WITH_CAMELLIA_256_GCM_SHA384
 *      MBEDTLS_TLS_ECDHE_PSK_WITH_CAMELLIA_256_CBC_SHA384
 *      MBEDTLS_TLS_DHE_PSK_WITH_CAMELLIA_256_CBC_SHA384
 *      MBEDTLS_TLS_DHE_PSK_WITH_CAMELLIA_128_GCM_SHA256
 *      MBEDTLS_TLS_DHE_PSK_WITH_CAMELLIA_128_CBC_SHA256
 *      MBEDTLS_TLS_ECDHE_PSK_WITH_CAMELLIA_128_CBC_SHA256
 *      MBEDTLS_TLS_RSA_WITH_CAMELLIA_256_GCM_SHA384
 *      MBEDTLS_TLS_RSA_WITH_CAMELLIA_256_CBC_SHA256
 *      MBEDTLS_TLS_RSA_WITH_CAMELLIA_256_CBC_SHA
 *      MBEDTLS_TLS_RSA_WITH_CAMELLIA_128_GCM_SHA256
 *      MBEDTLS_TLS_RSA_WITH_CAMELLIA_128_CBC_SHA256
 *      MBEDTLS_TLS_RSA_WITH_CAMELLIA_128_CBC_SHA
 *      MBEDTLS_TLS_RSA_PSK_WITH_CAMELLIA_256_GCM_SHA384
 *      MBEDTLS_TLS_RSA_PSK_WITH_CAMELLIA_256_CBC_SHA384
 *      MBEDTLS_TLS_RSA_PSK_WITH_CAMELLIA_128_GCM_SHA256
 *      MBEDTLS_TLS_RSA_PSK_WITH_CAMELLIA_128_CBC_SHA256
 *      MBEDTLS_TLS_PSK_WITH_CAMELLIA_256_GCM_SHA384
 *      MBEDTLS_TLS_PSK_WITH_CAMELLIA_256_CBC_SHA384
 *      MBEDTLS_TLS_PSK_WITH_CAMELLIA_128_GCM_SHA256
 *      MBEDTLS_TLS_PSK_WITH_CAMELLIA_128_CBC_SHA256
 */
#define MBEDTLS_CAMELLIA_C

/**
 * \def MBEDTLS_CCM_C
 *
 * Enable the Counter with CBC-MAC (CCM) mode for 128-bit block cipher.
 *
 * Module:  library/ccm.c
 *
 * Requires: MBEDTLS_AES_C or MBEDTLS_CAMELLIA_C
 *
 * This module enables the AES-CCM ciphersuites, if other requisites are
 * enabled as well.
 */
#define MBEDTLS_CCM_C

/**
 * \def MBEDTLS_CERTS_C
 *
 * Enable the test certificates.
 *
 * Module:  library/certs.c
 * Caller:
 *
 * This module is used for testing (ssl_client/server).
 */
#define MBEDTLS_CERTS_C

/**
 * \def MBEDTLS_CIPHER_C
 *
 * Enable the generic cipher layer.
 *
 * Module:  library/cipher.c
 * Caller:  library/ssl_tls.c
 *
 * Uncomment to enable generic cipher wrappers.
 */
#define MBEDTLS_CIPHER_C

/**
 * \def MBEDTLS_CMAC_C
 *
 * Enable the CMAC (Cipher-based Message Authentication Code) mode for block
 * ciphers.
 *
 * Module:  library/cmac.c
 *
 * Requires: MBEDTLS_AES_C or MBEDTLS_DES_C
 *
 */
//#define MBEDTLS_CMAC_C

/**
 * \def MBEDTLS_CTR_DRBG_C
 *
 * Enable the CTR_DRBG AES-256-based random generator.
 *
 * Module:  library/ctr_drbg.c
 * Caller:
 *
 * Requires: MBEDTLS_AES_C
 *
 * This module provides the CTR_DRBG AES-256 random number generator.
 */
#define MBEDTLS_CTR_DRBG_C

/**
 * \def MBEDTLS_DEBUG_C
 *
 * Enable the debug functions.
 *
 * Module:  library/debug.c
 * Caller:  library/ssl_cli.c
 *          library/ssl_srv.c
 *          library/ssl_tls.c
 *
 * This module provides debugging functions.
 */
#define MBEDTLS_DEBUG_C

/**
 * \def MBEDTLS_DES_C
 *
 * Enable the DES block cipher.
 *
 * Module:  library/des.c
 * Caller:  library/pem.c
 *          library/ssl_tls.c
 *
 * This module enables the following ciphersuites (if other requisites are
 * enabled as well):
 *      MBEDTLS_TLS_ECDH_ECDSA_WITH_3DES_EDE_CBC_SHA
 *      MBEDTLS_TLS_ECDH_RSA_WITH_3DES_EDE_CBC_SHA
 *      MBEDTLS_TLS_ECDHE_ECDSA_WITH_3DES_EDE_CBC_SHA
 *      MBEDTLS_TLS_ECDHE_RSA_WITH_3DES_EDE_CBC_SHA
 *      MBEDTLS_TLS_DHE_RSA_WITH_3DES_EDE_CBC_SHA
 *      MBEDTLS_TLS_ECDHE_PSK_WITH_3DES_EDE_CBC_SHA
 *      MBEDTLS_TLS_DHE_PSK_WITH_3DES_EDE_CBC_SHA
 *      MBEDTLS_TLS_RSA_WITH_3DES_EDE_CBC_SHA
 *      MBEDTLS_TLS_RSA_PSK_WITH_3DES_EDE_CBC_SHA
 *      MBEDTLS_TLS_PSK_WITH_3DES_EDE_CBC_SHA
 *
 * PEM_PARSE uses DES/3DES for decrypting encrypted keys.
 *
 * \warning   DES is considered a weak cipher and its use constitutes a
 *            security risk. We recommend considering stronger ciphers instead.
 */
#define MBEDTLS_DES_C

/**
 * \def MBEDTLS_DHM_C
 *
 * Enable the Diffie-Hellman-Merkle module.
 *
 * Module:  library/dhm.c
 * Caller:  library/ssl_cli.c
 *          library/ssl_srv.c
 *
 * This module is used by the following key exchanges:
 *      DHE-RSA, DHE-PSK
 *
 * \warning    Using DHE constitutes a security risk as it
 *             is not possible to validate custom DH parameters.
 *             If possible, it is recommended users should consider
 *             preferring other methods of key exchange.
 *             See dhm.h for more details.
 *
 */
#define MBEDTLS_DHM_C

/**
 * \def MBEDTLS_ECDH_C
 *
 * Enable the elliptic curve Diffie-Hellman library.
 *
 * Module:  library/ecdh.c
 * Caller:  library/ssl_cli.c
 *          library/ssl_srv.c
 *
 * This module is used by the following key exchanges:
 *      ECDHE-ECDSA, ECDHE-RSA, DHE-PSK
 *
 * Requires: MBEDTLS_ECP_C
 */
#define MBEDTLS_ECDH_C

/**
 * \def MBEDTLS_ECDSA_C
 *
 * Enable the elliptic curve DSA library.
 *
 * Module:  library/ecdsa.c
 * Caller:
 *
 * This module is used by the following key exchanges:
 *      ECDHE-ECDSA
 *
 * Requires: MBEDTLS_ECP_C, MBEDTLS_ASN1_WRITE_C, MBEDTLS_ASN1_PARSE_C
 */
#define MBEDTLS_ECDSA_C

/**
 * \def MBEDTLS_ECJPAKE_C
 *
 * Enable the elliptic curve J-PAKE library.
 *
 * \warning This is currently experimental. EC J-PAKE support is based on the
 * Thread v1.0.0 specification; incompatible changes to the specification
 * might still happen. For this reason, this is disabled by default.
 *
 * Module:  library/ecjpake.c
 * Caller:
 *
 * This module is used by the following key exchanges:
 *      ECJPAKE
 *
 * Requires: MBEDTLS_ECP_C, MBEDTLS_MD_C
 */
//#define MBEDTLS_ECJPAKE_C

/**
 * \def MBEDTLS_ECP_C
 *
 * Enable the elliptic curve over GF(p) library.
 *
 * Module:  library/ecp.c
 * Caller:  library/ecdh.c
 *          library/ecdsa.c
 *          library/ecjpake.c
 *
 * Requires: MBEDTLS_BIGNUM_C and at least one MBEDTLS_ECP_DP_XXX_ENABLED
 */
#define MBEDTLS_ECP_C

/**
 * \def MBEDTLS_ENTROPY_C
 *
 * Enable the platform-specific entropy code.
 *
 * Module:  library/entropy.c
 * Caller:
 *
 * Requires: MBEDTLS_SHA512_C or MBEDTLS_SHA256_C
 *
 * This module provides a generic entropy pool
 */
#define MBEDTLS_ENTROPY_C

/**
 * \def MBEDTLS_ERROR_C
 *
 * Enable error code to error string conversion.
 *
 * Module:  library/error.c
 * Caller:
 *
 * This module enables mbedtls_strerror().
 */
#define MBEDTLS_ERROR_C

/**
 * \def MBEDTLS_GCM_C
 *
 * Enable the Galois/Counter Mode (GCM) for AES.
 *
 * Module:  library/gcm.c
 *
 * Requires: MBEDTLS_AES_C or MBEDTLS_CAMELLIA_C
 *
 * This module enables the AES-GCM and CAMELLIA-GCM ciphersuites, if other
 * requisites are enabled as well.
 */
#define MBEDTLS_GCM_C

/**
 * \def MBEDTLS_HAVEGE_C
 *
 * Enable the HAVEGE random generator.
 *
 * Warning: the HAVEGE random generator is not suitable for virtualized
 *          environments
 *
 * Warning: the HAVEGE random generator is dependent on timing and specific
 *          processor traits. It is therefore not advised to use HAVEGE as
 *          your applications primary random generator or primary entropy pool
 *          input. As a secondary input to your entropy pool, it IS able add
 *          the (limited) extra entropy it provides.
 *
 * Module:  library/havege.c
 * Caller:
 *
 * Requires: MBEDTLS_TIMING_C
 *
 * Uncomment to enable the HAVEGE random generator.
 */
//#define MBEDTLS_HAVEGE_C

/**
 * \def MBEDTLS_HMAC_DRBG_C
 *
 * Enable the HMAC_DRBG random generator.
 *
 * Module:  library/hmac_drbg.c
 * Caller:
 *
 * Requires: MBEDTLS_MD_C
 *
 * Uncomment to enable the HMAC_DRBG random number geerator.
 */
#define MBEDTLS_HMAC_DRBG_C

/**
 * \def MBEDTLS_MD_C
 *
 * Enable the generic message digest layer.
 *
 * Module:  library/md.c
 * Caller:
 *
 * Uncomment to enable generic message digest wrappers.
 */
#define MBEDTLS_MD_C

/**
 * \def MBEDTLS_MD2_C
 *
 * Enable the MD2 hash algorithm.
 *
 * Module:  library/md2.c
 * Caller:
 *
 * Uncomment to enable support for (rare) MD2-signed X.509 certs.
 *
 * \warning   MD2 is considered a weak message digest and its use constitutes a
 *            security risk. If possible, we recommend avoiding dependencies on
 *            it, and considering stronger message digests instead.
 *
 */
//#define MBEDTLS_MD2_C

/**
 * \def MBEDTLS_MD4_C
 *
 * Enable the MD4 hash algorithm.
 *
 * Module:  library/md4.c
 * Caller:
 *
 * Uncomment to enable support for (rare) MD4-signed X.509 certs.
 *
 * \warning   MD4 is considered a weak message digest and its use constitutes a
 *            security risk. If possible, we recommend avoiding dependencies on
 *            it, and considering stronger message digests instead.
 *
 */
//#define MBEDTLS_MD4_C

/**
 * \def MBEDTLS_MD5_C
 *
 * Enable the MD5 hash algorithm.
 *
 * Module:  library/md5.c
 * Caller:  library/md.c
 *          library/pem.c
 *          library/ssl_tls.c
 *
 * This module is required for SSL/TLS up to version 1.1, and for TLS 1.2
 * depending on the handshake parameters. Further, it is used for checking
 * MD5-signed certificates, and for PBKDF1 when decrypting PEM-encoded
 * encrypted keys.
 *
 * \warning   MD5 is considered a weak message digest and its use constitutes a
 *            security risk. If possible, we recommend avoiding dependencies on
 *            it, and considering stronger message digests instead.
 *
 */
#define MBEDTLS_MD5_C

/**
 * \def MBEDTLS_MEMORY_BUFFER_ALLOC_C
 *
 * Enable the buffer allocator implementation that makes use of a (stack)
 * based buffer to 'allocate' dynamic memory. (replaces calloc() and free()
 * calls)
 *
 * Module:  library/memory_buffer_alloc.c
 *
 * Requires: MBEDTLS_PLATFORM_C
 *           MBEDTLS_PLATFORM_MEMORY (to use it within mbed TLS)
 *
 * Enable this module to enable the buffer memory allocator.
 */
//#define MBEDTLS_MEMORY_BUFFER_ALLOC_C

/**
 * \def MBEDTLS_NET_C
 *
 * Enable the TCP and UDP over IPv6/IPv4 networking routines.
 *
 * \note This module only works on POSIX/Unix (including Linux, BSD and OS X)
 * and Windows. For other platforms, you'll want to disable it, and write your
 * own networking callbacks to be passed to \c mbedtls_ssl_set_bio().
 *
 * \note See also our Knowledge Base article about porting to a new
 * environment:
 * https://tls.mbed.org/kb/how-to/how-do-i-port-mbed-tls-to-a-new-environment-OS
 *
 * Module:  library/net_sockets.c
 *
 * This module provides networking routines.
 */
#define MBEDTLS_NET_C

/**
 * \def MBEDTLS_OID_C
 *
 * Enable the OID database.
 *
 * Module:  library/oid.c
 * Caller:  library/asn1write.c
 *          library/pkcs5.c
 *          library/pkparse.c
 *          library/pkwrite.c
 *          library/rsa.c
 *          library/x509.c
 *          library/x509_create.c
 *          library/x509_crl.c
 *          library/x509_crt.c
 *          library/x509_csr.c
 *          library/x509write_crt.c
 *          library/x509write_csr.c
 *
 * This modules translates between OIDs and internal values.
 */
#define MBEDTLS_OID_C

/**
 * \def MBEDTLS_PADLOCK_C
 *
 * Enable VIA Padlock support on x86.
 *
 * Module:  library/padlock.c
 * Caller:  library/aes.c
 *
 * Requires: MBEDTLS_HAVE_ASM
 *
 * This modules adds support for the VIA PadLock on x86.
 */
#define MBEDTLS_PADLOCK_C

/**
 * \def MBEDTLS_PEM_PARSE_C
 *
 * Enable PEM decoding / parsing.
 *
 * Module:  library/pem.c
 * Caller:  library/dhm.c
 *          library/pkparse.c
 *          library/x509_crl.c
 *          library/x509_crt.c
 *          library/x509_csr.c
 *
 * Requires: MBEDTLS_BASE64_C
 *
 * This modules adds support for decoding / parsing PEM files.
 */
#define MBEDTLS_PEM_PARSE_C

/**
 * \def MBEDTLS_PEM_WRITE_C
 *
 * Enable PEM encoding / writing.
 *
 * Module:  library/pem.c
 * Caller:  library/pkwrite.c
 *          library/x509write_crt.c
 *          library/x509write_csr.c
 *
 * Requires: MBEDTLS_BASE64_C
 *
 * This modules adds support for encoding / writing PEM files.
 */
#define MBEDTLS_PEM_WRITE_C

/**
 * \def MBEDTLS_PK_C
 *
 * Enable the generic public (asymetric) key layer.
 *
 * Module:  library/pk.c
 * Caller:  library/ssl_tls.c
 *          library/ssl_cli.c
 *          library/ssl_srv.c
 *
 * Requires: MBEDTLS_RSA_C or MBEDTLS_ECP_C
 *
 * Uncomment to enable generic public key wrappers.
 */
#define MBEDTLS_PK_C

/**
 * \def MBEDTLS_PK_PARSE_C
 *
 * Enable the generic public (asymetric) key parser.
 *
 * Module:  library/pkparse.c
 * Caller:  library/x509_crt.c
 *          library/x509_csr.c
 *
 * Requires: MBEDTLS_PK_C
 *
 * Uncomment to enable generic public key parse functions.
 */
#define MBEDTLS_PK_PARSE_C

/**
 * \def MBEDTLS_PK_WRITE_C
 *
 * Enable the generic public (asymetric) key writer.
 *
 * Module:  library/pkwrite.c
 * Caller:  library/x509write.c
 *
 * Requires: MBEDTLS_PK_C
 *
 * Uncomment to enable generic public key write functions.
 */
#define MBEDTLS_PK_WRITE_C

/**
 * \def MBEDTLS_PKCS5_C
 *
 * Enable PKCS#5 functions.
 *
 * Module:  library/pkcs5.c
 *
 * Requires: MBEDTLS_MD_C
 *
 * This module adds support for the PKCS#5 functions.
 */
#define MBEDTLS_PKCS5_C

/**
 * \def MBEDTLS_PKCS11_C
 *
 * Enable wrapper for PKCS#11 smartcard support.
 *
 * Module:  library/pkcs11.c
 * Caller:  library/pk.c
 *
 * Requires: MBEDTLS_PK_C
 *
 * This module enables SSL/TLS PKCS #11 smartcard support.
 * Requires the presence of the PKCS#11 helper library (libpkcs11-helper)
 */
//#define MBEDTLS_PKCS11_C

/**
 * \def MBEDTLS_PKCS12_C
 *
 * Enable PKCS#12 PBE functions.
 * Adds algorithms for parsing PKCS#8 encrypted private keys
 *
 * Module:  library/pkcs12.c
 * Caller:  library/pkparse.c
 *
 * Requires: MBEDTLS_ASN1_PARSE_C, MBEDTLS_CIPHER_C, MBEDTLS_MD_C
 * Can use:  MBEDTLS_ARC4_C
 *
 * This module enables PKCS#12 functions.
 */
#define MBEDTLS_PKCS12_C

/**
 * \def MBEDTLS_PLATFORM_C
 *
 * Enable the platform abstraction layer that allows you to re-assign
 * functions like calloc(), free(), snprintf(), printf(), fprintf(), exit().
 *
 * Enabling MBEDTLS_PLATFORM_C enables to use of MBEDTLS_PLATFORM_XXX_ALT
 * or MBEDTLS_PLATFORM_XXX_MACRO directives, allowing the functions mentioned
 * above to be specified at runtime or compile time respectively.
 *
 * \note This abstraction layer must be enabled on Windows (including MSYS2)
 * as other module rely on it for a fixed snprintf implementation.
 *
 * Module:  library/platform.c
 * Caller:  Most other .c files
 *
 * This module enables abstraction of common (libc) functions.
 */
#define MBEDTLS_PLATFORM_C

/**
 * \def MBEDTLS_RIPEMD160_C
 *
 * Enable the RIPEMD-160 hash algorithm.
 *
 * Module:  library/ripemd160.c
 * Caller:  library/md.c
 *
 */
#define MBEDTLS_RIPEMD160_C

/**
 * \def MBEDTLS_RSA_C
 *
 * Enable the RSA public-key cryptosystem.
 *
 * Module:  library/rsa.c
 *          library/rsa_internal.c
 * Caller:  library/ssl_cli.c
 *          library/ssl_srv.c
 *          library/ssl_tls.c
 *          library/x509.c
 *
 * This module is used by the following key exchanges:
 *      RSA, DHE-RSA, ECDHE-RSA, RSA-PSK
 *
 * Requires: MBEDTLS_BIGNUM_C, MBEDTLS_OID_C
 */
#define MBEDTLS_RSA_C

/**
 * \def MBEDTLS_SHA1_C
 *
 * Enable the SHA1 cryptographic hash algorithm.
 *
 * Module:  library/sha1.c
 * Caller:  library/md.c
 *          library/ssl_cli.c
 *          library/ssl_srv.c
 *          library/ssl_tls.c
 *          library/x509write_crt.c
 *
 * This module is required for SSL/TLS up to version 1.1, for TLS 1.2
 * depending on the handshake parameters, and for SHA1-signed certificates.
 *
 * \warning   SHA-1 is considered a weak message digest and its use constitutes
 *            a security risk. If possible, we recommend avoiding dependencies
 *            on it, and considering stronger message digests instead.
 *
 */
#define MBEDTLS_SHA1_C

/**
 * \def MBEDTLS_SHA256_C
 *
 * Enable the SHA-224 and SHA-256 cryptographic hash algorithms.
 *
 * Module:  library/sha256.c
 * Caller:  library/entropy.c
 *          library/md.c
 *          library/ssl_cli.c
 *          library/ssl_srv.c
 *          library/ssl_tls.c
 *
 * This module adds support for SHA-224 and SHA-256.
 * This module is required for the SSL/TLS 1.2 PRF function.
 */
#define MBEDTLS_SHA256_C

/**
 * \def MBEDTLS_SHA512_C
 *
 * Enable the SHA-384 and SHA-512 cryptographic hash algorithms.
 *
 * Module:  library/sha512.c
 * Caller:  library/entropy.c
 *          library/md.c
 *          library/ssl_cli.c
 *          library/ssl_srv.c
 *
 * This module adds support for SHA-384 and SHA-512.
 */
#define MBEDTLS_SHA512_C

/**
 * \def MBEDTLS_SSL_CACHE_C
 *
 * Enable simple SSL cache implementation.
 *
 * Module:  library/ssl_cache.c
 * Caller:
 *
 * Requires: MBEDTLS_SSL_CACHE_C
 */
#define MBEDTLS_SSL_CACHE_C

/**
 * \def MBEDTLS_SSL_COOKIE_C
 *
 * Enable basic implementation of DTLS cookies for hello verification.
 *
 * Module:  library/ssl_cookie.c
 * Caller:
 */
#define MBEDTLS_SSL_COOKIE_C

/**
 * \def MBEDTLS_SSL_TICKET_C
 *
 * Enable an implementation of TLS server-side callbacks for session tickets.
 *
 * Module:  library/ssl_ticket.c
 * Caller:
 *
 * Requires: MBEDTLS_CIPHER_C
 */
#define MBEDTLS_SSL_TICKET_C

/**
 * \def MBEDTLS_SSL_CLI_C
 *
 * Enable the SSL/TLS client code.
 *
 * Module:  library/ssl_cli.c
 * Caller:
 *
 * Requires: MBEDTLS_SSL_TLS_C
 *
 * This module is required for SSL/TLS client support.
 */
#define MBEDTLS_SSL_CLI_C

/**
 * \def MBEDTLS_SSL_SRV_C
 *
 * Enable the SSL/TLS server code.
 *
 * Module:  library/ssl_srv.c
 * Caller:
 *
 * Requires: MBEDTLS_SSL_TLS_C
 *
 * This module is required for SSL/TLS server support.
 */
#define MBEDTLS_SSL_SRV_C

/**
 * \def MBEDTLS_SSL_TLS_C
 *
 * Enable the generic SSL/TLS code.
 *
 * Module:  library/ssl_tls.c
 * Caller:  library/ssl_cli.c
 *          library/ssl_srv.c
 *
 * Requires: MBEDTLS_CIPHER_C, MBEDTLS_MD_C
 *           and at least one of the MBEDTLS_SSL_PROTO_XXX defines
 *
 * This module is required for SSL/TLS.
 */
#define MBEDTLS_SSL_TLS_C

/**
 * \def MBEDTLS_THREADING_C
 *
 * Enable the threading abstraction layer.
 * By default mbed TLS assumes it is used in a non-threaded environment or that
 * contexts are not shared between threads. If you do intend to use contexts
 * between threads, you will need to enable this layer to prevent race
 * conditions. See also our Knowledge Base article about threading:
 * https://tls.mbed.org/kb/development/thread-safety-and-multi-threading
 *
 * Module:  library/threading.c
 *
 * This allows different threading implementations (self-implemented or
 * provided).
 *
 * You will have to enable either MBEDTLS_THREADING_ALT or
 * MBEDTLS_THREADING_PTHREAD.
 *
 * Enable this layer to allow use of mutexes within mbed TLS
 */
//#define MBEDTLS_THREADING_C

/**
 * \def MBEDTLS_TIMING_C
 *
 * Enable the semi-portable timing interface.
 *
 * \note The provided implementation only works on POSIX/Unix (including Linux,
 * BSD and OS X) and Windows. On other platforms, you can either disable that
 * module and provide your own implementations of the callbacks needed by
 * \c mbedtls_ssl_set_timer_cb() for DTLS, or leave it enabled and provide
 * your own implementation of the whole module by setting
 * \c MBEDTLS_TIMING_ALT in the current file.
 *
 * \note See also our Knowledge Base article about porting to a new
 * environment:
 * https://tls.mbed.org/kb/how-to/how-do-i-port-mbed-tls-to-a-new-environment-OS
 *
 * Module:  library/timing.c
 * Caller:  library/havege.c
 *
 * This module is used by the HAVEGE random number generator.
 */
#define MBEDTLS_TIMING_C

/**
 * \def MBEDTLS_VERSION_C
 *
 * Enable run-time version information.
 *
 * Module:  library/version.c
 *
 * This module provides run-time version information.
 */
#define MBEDTLS_VERSION_C

/**
 * \def MBEDTLS_X509_USE_C
 *
 * Enable X.509 core for using certificates.
 *
 * Module:  library/x509.c
 * Caller:  library/x509_crl.c
 *          library/x509_crt.c
 *          library/x509_csr.c
 *
 * Requires: MBEDTLS_ASN1_PARSE_C, MBEDTLS_BIGNUM_C, MBEDTLS_OID_C,
 *           MBEDTLS_PK_PARSE_C
 *
 * This module is required for the X.509 parsing modules.
 */
#define MBEDTLS_X509_USE_C

/**
 * \def MBEDTLS_X509_CRT_PARSE_C
 *
 * Enable X.509 certificate parsing.
 *
 * Module:  library/x509_crt.c
 * Caller:  library/ssl_cli.c
 *          library/ssl_srv.c
 *          library/ssl_tls.c
 *
 * Requires: MBEDTLS_X509_USE_C
 *
 * This module is required for X.509 certificate parsing.
 */
#define MBEDTLS_X509_CRT_PARSE_C

/**
 * \def MBEDTLS_X509_CRL_PARSE_C
 *
 * Enable X.509 CRL parsing.
 *
 * Module:  library/x509_crl.c
 * Caller:  library/x509_crt.c
 *
 * Requires: MBEDTLS_X509_USE_C
 *
 * This module is required for X.509 CRL parsing.
 */
#define MBEDTLS_X509_CRL_PARSE_C

/**
 * \def MBEDTLS_X509_CSR_PARSE_C
 *
 * Enable X.509 Certificate Signing Request (CSR) parsing.
 *
 * Module:  library/x509_csr.c
 * Caller:  library/x509_crt_write.c
 *
 * Requires: MBEDTLS_X509_USE_C
 *
 * This module is used for reading X.509 certificate request.
 */
#define MBEDTLS_X509_CSR_PARSE_C

/**
 * \def MBEDTLS_X509_CREATE_C
 *
 * Enable X.509 core for creating certificates.
 *
 * Module:  library/x509_create.c
 *
 * Requires: MBEDTLS_BIGNUM_C, MBEDTLS_OID_C, MBEDTLS_PK_WRITE_C
 *
 * This module is the basis for creating X.509 certificates and CSRs.
 */
#define MBEDTLS_X509_CREATE_C

/**
 * \def MBEDTLS_X509_CRT_WRITE_C
 *
 * Enable creating X.509 certificates.
 *
 * Module:  library/x509_crt_write.c
 *
 * Requires: MBEDTLS_X509_CREATE_C
 *
 * This module is required for X.509 certificate creation.
 */
#define MBEDTLS_X509_CRT_WRITE_C

/**
 * \def MBEDTLS_X509_CSR_WRITE_C
 *
 * Enable creating X.509 Certificate Signing Requests (CSR).
 *
 * Module:  library/x509_csr_write.c
 *
 * Requires: MBEDTLS_X509_CREATE_C
 *
 * This module is required for X.509 certificate request writing.
 */
#define MBEDTLS_X509_CSR_WRITE_C

/**
 * \def MBEDTLS_XTEA_C
 *
 * Enable the XTEA block cipher.
 *
 * Module:  library/xtea.c
 * Caller:
 */
#define MBEDTLS_XTEA_C

/* \} name SECTION: mbed TLS modules */

/**
 * \name SECTION: Module configuration options
 *
 * This section allows for the setting of module specific sizes and
 * configuration options. The default values are already present in the
 * relevant header files and should suffice for the regular use cases.
 *
 * Our advice is to enable options and change their values here
 * only if you have a good reason and know the consequences.
 *
 * Please check the respective header file for documentation on these
 * parameters (to prevent duplicate documentation).
 * \{
 */

/* MPI / BIGNUM options */
//#define MBEDTLS_MPI_WINDOW_SIZE            6 /**< Maximum windows size used. */
//#define MBEDTLS_MPI_MAX_SIZE            1024 /**< Maximum number of bytes for usable MPIs. */

/* CTR_DRBG options */
//#define MBEDTLS_CTR_DRBG_ENTROPY_LEN               48 /**< Amount of entropy used per seed by default (48 with SHA-512, 32 with SHA-256) */
//#define MBEDTLS_CTR_DRBG_RESEED_INTERVAL        10000 /**< Interval before reseed is performed by default */
//#define MBEDTLS_CTR_DRBG_MAX_INPUT                256 /**< Maximum number of additional input bytes */
//#define MBEDTLS_CTR_DRBG_MAX_REQUEST             1024 /**< Maximum number of requested bytes per call */
//#define MBEDTLS_CTR_DRBG_MAX_SEED_INPUT           384 /**< Maximum size of (re)seed buffer */

/* HMAC_DRBG options */
//#define MBEDTLS_HMAC_DRBG_RESEED_INTERVAL   10000 /**< Interval before reseed is performed by default */
//#define MBEDTLS_HMAC_DRBG_MAX_INPUT           256 /**< Maximum number of additional input bytes */
//#define MBEDTLS_HMAC_DRBG_MAX_REQUEST        1024 /**< Maximum number of requested bytes per call */
//#define MBEDTLS_HMAC_DRBG_MAX_SEED_INPUT      384 /**< Maximum size of (re)seed buffer */

/* ECP options */
//#define MBEDTLS_ECP_MAX_BITS             521 /**< Maximum bit size of groups */
//#define MBEDTLS_ECP_WINDOW_SIZE            6 /**< Maximum window size used */
//#define MBEDTLS_ECP_FIXED_POINT_OPTIM      1 /**< Enable fixed-point speed-up */

/* Entropy options */
//#define MBEDTLS_ENTROPY_MAX_SOURCES                20 /**< Maximum number of sources supported */
//#define MBEDTLS_ENTROPY_MAX_GATHER                128 /**< Maximum amount requested from entropy sources */
//#define MBEDTLS_ENTROPY_MIN_HARDWARE               32 /**< Default minimum number of bytes required for the hardware entropy source mbedtls_hardware_poll() before entropy is released */

/* Memory buffer allocator options */
//#define MBEDTLS_MEMORY_ALIGN_MULTIPLE      4 /**< Align on multiples of this value */

/* Platform options */
//#define MBEDTLS_PLATFORM_STD_MEM_HDR   <stdlib.h> /**< Header to include if MBEDTLS_PLATFORM_NO_STD_FUNCTIONS is defined. Don't define if no header is needed. */
//#define MBEDTLS_PLATFORM_STD_CALLOC        calloc /**< Default allocator to use, can be undefined */
//#define MBEDTLS_PLATFORM_STD_FREE            free /**< Default free to use, can be undefined */
//#define MBEDTLS_PLATFORM_STD_EXIT            exit /**< Default exit to use, can be undefined */
//#define MBEDTLS_PLATFORM_STD_TIME            time /**< Default time to use, can be undefined. MBEDTLS_HAVE_TIME must be enabled */
//#define MBEDTLS_PLATFORM_STD_FPRINTF      fprintf /**< Default fprintf to use, can be undefined */
//#define MBEDTLS_PLATFORM_STD_PRINTF        printf /**< Default printf to use, can be undefined */
/* Note: your snprintf must correclty zero-terminate the buffer! */
//#define MBEDTLS_PLATFORM_STD_SNPRINTF    snprintf /**< Default snprintf to use, can be undefined */
//#define MBEDTLS_PLATFORM_STD_EXIT_SUCCESS       0 /**< Default exit value to use, can be undefined */
//#define MBEDTLS_PLATFORM_STD_EXIT_FAILURE       1 /**< Default exit value to use, can be undefined */
//#define MBEDTLS_PLATFORM_STD_NV_SEED_READ   mbedtls_platform_std_nv_seed_read /**< Default nv_seed_read function to use, can be undefined */
//#define MBEDTLS_PLATFORM_STD_NV_SEED_WRITE  mbedtls_platform_std_nv_seed_write /**< Default nv_seed_write function to use, can be undefined */
//#define MBEDTLS_PLATFORM_STD_NV_SEED_FILE  "seedfile" /**< Seed file to read/write with default implementation */

/* To Use Function Macros MBEDTLS_PLATFORM_C must be enabled */
/* MBEDTLS_PLATFORM_XXX_MACRO and MBEDTLS_PLATFORM_XXX_ALT cannot both be defined */
//#define MBEDTLS_PLATFORM_CALLOC_MACRO        calloc /**< Default allocator macro to use, can be undefined */
//#define MBEDTLS_PLATFORM_FREE_MACRO            free /**< Default free macro to use, can be undefined */
//#define MBEDTLS_PLATFORM_EXIT_MACRO            exit /**< Default exit macro to use, can be undefined */
//#define MBEDTLS_PLATFORM_TIME_MACRO            time /**< Default time macro to use, can be undefined. MBEDTLS_HAVE_TIME must be enabled */
//#define MBEDTLS_PLATFORM_TIME_TYPE_MACRO       time_t /**< Default time macro to use, can be undefined. MBEDTLS_HAVE_TIME must be enabled */
//#define MBEDTLS_PLATFORM_FPRINTF_MACRO      fprintf /**< Default fprintf macro to use, can be undefined */
//#define MBEDTLS_PLATFORM_PRINTF_MACRO        printf /**< Default printf macro to use, can be undefined */
/* Note: your snprintf must correclty zero-terminate the buffer! */
//#define MBEDTLS_PLATFORM_SNPRINTF_MACRO    snprintf /**< Default snprintf macro to use, can be undefined */
//#define MBEDTLS_PLATFORM_NV_SEED_READ_MACRO   mbedtls_platform_std_nv_seed_read /**< Default nv_seed_read function to use, can be undefined */
//#define MBEDTLS_PLATFORM_NV_SEED_WRITE_MACRO  mbedtls_platform_std_nv_seed_write /**< Default nv_seed_write function to use, can be undefined */

/* SSL Cache options */
//#define MBEDTLS_SSL_CACHE_DEFAULT_TIMEOUT       86400 /**< 1 day  */
//#define MBEDTLS_SSL_CACHE_DEFAULT_MAX_ENTRIES      50 /**< Maximum entries in cache */

/* SSL options */
//#define MBEDTLS_SSL_MAX_CONTENT_LEN             16384 /**< Maxium fragment length in bytes, determines the size of each of the two internal I/O buffers */
//#define MBEDTLS_SSL_DEFAULT_TICKET_LIFETIME     86400 /**< Lifetime of session tickets (if enabled) */
//#define MBEDTLS_PSK_MAX_LEN               32 /**< Max size of TLS pre-shared keys, in bytes (default 256 bits) */
//#define MBEDTLS_SSL_COOKIE_TIMEOUT        60 /**< Default expiration delay of DTLS cookies, in seconds if HAVE_TIME, or in number of cookies issued */

/**
 * Complete list of ciphersuites to use, in order of preference.
 *
 * \warning No dependency checking is done on that field! This option can only
 * be used to restrict the set of available ciphersuites. It is your
 * responsibility to make sure the needed modules are active.
 *
 * Use this to save a few hundred bytes of ROM (default ordering of all
 * available ciphersuites) and a few to a few hundred bytes of RAM.
 *
 * The value below is only an example, not the default.
 */
//#define MBEDTLS_SSL_CIPHERSUITES MBEDTLS_TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,MBEDTLS_TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256

/* X509 options */
//#define MBEDTLS_X509_MAX_INTERMEDIATE_CA   8   /**< Maximum number of intermediate CAs in a verification chain. */
//#define MBEDTLS_X509_MAX_FILE_PATH_LEN     512 /**< Maximum length of a path/filename string in bytes including the null terminator character ('\0'). */

/**
 * Allow SHA-1 in the default TLS configuration for certificate signing.
 * Without this build-time option, SHA-1 support must be activated explicitly
 * through mbedtls_ssl_conf_cert_profile. Turning on this option is not
 * recommended because of it is possible to generate SHA-1 collisions, however
 * this may be safe for legacy infrastructure where additional controls apply.
 *
 * \warning   SHA-1 is considered a weak message digest and its use constitutes
 *            a security risk. If possible, we recommend avoiding dependencies
 *            on it, and considering stronger message digests instead.
 *
 */
// #define MBEDTLS_TLS_DEFAULT_ALLOW_SHA1_IN_CERTIFICATES

/**
 * Allow SHA-1 in the default TLS configuration for TLS 1.2 handshake
 * signature and ciphersuite selection. Without this build-time option, SHA-1
 * support must be activated explicitly through mbedtls_ssl_conf_sig_hashes.
 * The use of SHA-1 in TLS <= 1.1 and in HMAC-SHA-1 is always allowed by
 * default. At the time of writing, there is no practical attack on the use
 * of SHA-1 in handshake signatures, hence this option is turned on by default
 * to preserve compatibility with existing peers, but the general
 * warning applies nonetheless:
 *
 * \warning   SHA-1 is considered a weak message digest and its use constitutes
 *            a security risk. If possible, we recommend avoiding dependencies
 *            on it, and considering stronger message digests instead.
 *
 */
#define MBEDTLS_TLS_DEFAULT_ALLOW_SHA1_IN_KEY_EXCHANGE

/* \} name SECTION: Customisation configuration options */

/* Target and application specific configurations */
//#define YOTTA_CFG_MBEDTLS_TARGET_CONFIG_FILE "mbedtls/target_config.h"

#if defined(TARGET_LIKE_MBED) && defined(YOTTA_CFG_MBEDTLS_TARGET_CONFIG_FILE)
#include YOTTA_CFG_MBEDTLS_TARGET_CONFIG_FILE
#endif

/*
 * Allow user to override any previous default.
 *
 * Use two macro names for that, as:
 * - with yotta the prefix YOTTA_CFG_ is forced
 * - without yotta is looks weird to have a YOTTA prefix.
 */
#if defined(YOTTA_CFG_MBEDTLS_USER_CONFIG_FILE)
#include YOTTA_CFG_MBEDTLS_USER_CONFIG_FILE
#elif defined(MBEDTLS_USER_CONFIG_FILE)
#include MBEDTLS_USER_CONFIG_FILE
#endif

#include "check_config.h"

#endif /* MBEDTLS_CONFIG_H */
