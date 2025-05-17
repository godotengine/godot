/**************************************************************************/
/*  godot_module_mbedtls_config.h                                         */
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

#ifndef GODOT_MODULE_MBEDTLS_CONFIG_H
#define GODOT_MODULE_MBEDTLS_CONFIG_H

#include "platform_config.h"

#ifdef GODOT_MBEDTLS_INCLUDE_H

// Allow platforms to customize the mbedTLS configuration.
#include GODOT_MBEDTLS_INCLUDE_H

#else

// Include default mbedTLS config.
#include <mbedtls/mbedtls_config.h>

// Disable weak cryptography.
#undef MBEDTLS_KEY_EXCHANGE_DHE_PSK_ENABLED
#undef MBEDTLS_KEY_EXCHANGE_DHE_RSA_ENABLED
#undef MBEDTLS_DES_C
#undef MBEDTLS_DHM_C

#ifdef THREADS_ENABLED
// In mbedTLS 3, the PSA subsystem has an implicit shared context, MBEDTLS_THREADING_C is required to make it thread safe.
#define MBEDTLS_THREADING_C
#define MBEDTLS_THREADING_ALT
#define GODOT_MBEDTLS_THREADING_ALT
#endif

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

// Disable deprecated
#define MBEDTLS_DEPRECATED_REMOVED

#endif // GODOT_MBEDTLS_INCLUDE_H

#endif // GODOT_MODULE_MBEDTLS_CONFIG_H
