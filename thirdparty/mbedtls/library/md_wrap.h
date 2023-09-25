/**
 * \file md_wrap.h
 *
 * \brief Message digest wrappers.
 *
 * \warning This in an internal header. Do not include directly.
 *
 * \author Adriaan de Jong <dejong@fox-it.com>
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
#ifndef MBEDTLS_MD_WRAP_H
#define MBEDTLS_MD_WRAP_H

#include "mbedtls/build_info.h"

#include "mbedtls/md.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Message digest information.
 * Allows message digest functions to be called in a generic way.
 */
struct mbedtls_md_info_t {
    /** Name of the message digest */
    const char *name;

    /** Digest identifier */
    mbedtls_md_type_t type;

    /** Output length of the digest function in bytes */
    unsigned char size;

    /** Block length of the digest function in bytes */
    unsigned char block_size;
};

#if defined(MBEDTLS_MD5_C)
extern const mbedtls_md_info_t mbedtls_md5_info;
#endif
#if defined(MBEDTLS_RIPEMD160_C)
extern const mbedtls_md_info_t mbedtls_ripemd160_info;
#endif
#if defined(MBEDTLS_SHA1_C)
extern const mbedtls_md_info_t mbedtls_sha1_info;
#endif
#if defined(MBEDTLS_SHA224_C)
extern const mbedtls_md_info_t mbedtls_sha224_info;
#endif
#if defined(MBEDTLS_SHA256_C)
extern const mbedtls_md_info_t mbedtls_sha256_info;
#endif
#if defined(MBEDTLS_SHA384_C)
extern const mbedtls_md_info_t mbedtls_sha384_info;
#endif
#if defined(MBEDTLS_SHA512_C)
extern const mbedtls_md_info_t mbedtls_sha512_info;
#endif

#ifdef __cplusplus
}
#endif

#endif /* MBEDTLS_MD_WRAP_H */
