/**************************************************************************/
/*  mbedtls_cpp_compat.h                                                  */
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

#include <mbedtls/psa_util.h>

#ifdef MBEDTLS_ECDSA_DER_MAX_LEN
#define GODOT_MBEDTLS_ECDSA_DER_MAX_LEN MBEDTLS_ECDSA_DER_MAX_LEN
#else
#define GODOT_MBEDTLS_ECDSA_DER_MAX_LEN 192
#endif

// The compatibility functions defined in <mbedtls/psa_util.h> are not guarded by
// __cplusplus ifdefs which results in incorrect symbols when included from C++.
// For this reason, (and to avoid problems if this is fixed in future versions of
// the header) we define our own wrappers implemented in pure C.
#ifdef __cplusplus
extern "C" {
#endif
int godot_mbedtls_ecdsa_raw_to_der(size_t bits, const unsigned char *raw, size_t raw_len, unsigned char *der, size_t der_size, size_t *der_len);
int godot_mbedtls_ecdsa_der_to_raw(size_t bits, const unsigned char *der, size_t der_len, unsigned char *raw, size_t raw_size, size_t *raw_len);
#ifdef __cplusplus
}
#endif
