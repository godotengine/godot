/**************************************************************************/
/*  godot_psa_config.h                                                    */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "platform_config.h"

#ifdef GODOT_PSA_INCLUDE_H

// Allow platforms to customize the mbedTLS configuration.
#include GODOT_PSA_INCLUDE_H

#else

#if !defined(GODOT_MBEDTLS_LIGHT)
// Full module build (include default mbedTLS config).
#include <psa/crypto_config.h>

#if !(defined(__linux__) && defined(__aarch64__))
// ARMv8 hardware AES operations. Detection only possible on linux.
// May technically be supported on some ARM32 arches but doesn't seem
// to be in our current Linux SDK's neon-fp-armv8.
#undef MBEDTLS_AESCE_C
#endif

#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
// MemorySanitizer is incompatible with ASM.
#undef MBEDTLS_HAVE_ASM
#undef MBEDTLS_AESNI_C
#endif
#endif

#else // GODOT_MBEDTLS_LIGHT

// Light build for CryptoCore when module is disabled
#include <limits.h>

// For AES
#define MBEDTLS_AES_C
#define MBEDTLS_BASE64_C
#define MBEDTLS_CTR_DRBG_C
#define MBEDTLS_ENTROPY_C
#define MBEDTLS_MD_C
#define MBEDTLS_MD5_C
#define MBEDTLS_SHA1_C
#define MBEDTLS_SHA256_C
#define MBEDTLS_PLATFORM_ZEROIZE_ALT

#define MBEDTLS_PSA_CRYPTO_C
#define MBEDTLS_PSA_CRYPTO_CLIENT

// Hashing
#define PSA_WANT_ALG_SHA_1
#define PSA_WANT_ALG_SHA_256
#define PSA_WANT_ALG_MD5

// Encryption
#define PSA_WANT_KEY_TYPE_AES
#define PSA_WANT_ALG_CBC_NO_PADDING
#define PSA_WANT_ALG_ECB_NO_PADDING
#define PSA_WANT_ALG_CFB

// This is only to pass a check in the mbedtls check_config.h header, none of
// the files we include as part of the core build uses it anyway, we already
// define MBEDTLS_PLATFORM_ZEROIZE_ALT which is the only relevant function.
#if defined(__MINGW32__)
#define MBEDTLS_PLATFORM_C
#endif

#endif

// Disable deprecated
#define MBEDTLS_DEPRECATED_REMOVED

// Godot mbedTLS platform implementation
#define GODOT_MBEDTLS_PLATFORM

#ifdef MBEDTLS_PSA_BUILTIN_GET_ENTROPY
#undef MBEDTLS_PSA_BUILTIN_GET_ENTROPY
#endif
#define MBEDTLS_PSA_DRIVER_GET_ENTROPY

#if defined(THREADS_ENABLED)
#define MBEDTLS_THREADING_C
#define MBEDTLS_THREADING_ALT
#endif

#ifdef __cplusplus
extern "C" {
#endif
void godot_mbedtls_platform_init();
void godot_mbedtls_platform_free();
#ifdef __cplusplus
};
#endif

#endif // GODOT_PSA_INCLUDE_H
