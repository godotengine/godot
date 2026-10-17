/**
 * \file mbedtls_common.h
 *
 * \brief Utility macros for internal use in the library.
 *
 * This file should be included as the first thing in all library C files
 * (directly, or indirectly via x509_internal.h or ssl_misc.h).
 * It must not be included by sample programs, since sample programs
 * illustrate what you can do without the library sources.
 * It may be included (often indirectly) by test code that isn't purely
 * black-box testing.
 *
 * This file takes care of setting up requirements for platform headers.
 * It includes the library configuration and derived macros.
 * It additionally defines various utility macros and other definitions
 * (but no function declarations).
 */
/*
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */

#ifndef MBEDTLS_MBEDTLS_COMMON_H
#define MBEDTLS_MBEDTLS_COMMON_H

/* Before including any system header, declare some macros to tell system
 * headers what we expect of them.
 *
 * Do this before including any header from TF-PSA-Crypto, since the
 * convention is first-come-first-served (so that users can
 * override some macros on the command line, and individual users can
 * override some macros before including the common header).
 */
#include "mbedtls_platform_requirements.h"

/* Mbed TLS is tightly coupled with TF-PSA-Crypto, and inherits all of
 * its platform requirements because we don't have a clear separation of
 * public vs private platform interfaces. So make sure we declare the
 * TF-PSA-Crypto platform requirements. We need to do that before including
 * any system headers, thus before including the user config file since it
 * may include platform headers. */
#include "tf_psa_crypto_platform_requirements.h"

/* From this point onwards, ensure we have the library configuration and
 * the configuration-derived macros. */
#include <mbedtls/build_info.h>

/* Mbed TLS requires TF-PSA-Crypto internals. */
#include "tf_psa_crypto_common.h"

#endif /* MBEDTLS_MBEDTLS_COMMON_H */
