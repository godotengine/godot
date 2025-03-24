/**
 * \file version.h
 *
 * \brief Run-time version information
 */
/*
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */
/*
 * This set of compile-time defines and run-time variables can be used to
 * determine the version number of the Mbed TLS library used.
 */
#ifndef MBEDTLS_VERSION_H
#define MBEDTLS_VERSION_H

#if !defined(MBEDTLS_CONFIG_FILE)
#include "mbedtls/config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

/**
 * The version number x.y.z is split into three parts.
 * Major, Minor, Patchlevel
 */
#define MBEDTLS_VERSION_MAJOR  2
#define MBEDTLS_VERSION_MINOR  28
#define MBEDTLS_VERSION_PATCH  10

/**
 * The single version number has the following structure:
 *    MMNNPP00
 *    Major version | Minor version | Patch version
 */
#define MBEDTLS_VERSION_NUMBER         0x021C0A00
#define MBEDTLS_VERSION_STRING         "2.28.10"
#define MBEDTLS_VERSION_STRING_FULL    "Mbed TLS 2.28.10"

#if defined(MBEDTLS_VERSION_C)

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Get the version number.
 *
 * \return          The constructed version number in the format
 *                  MMNNPP00 (Major, Minor, Patch).
 */
unsigned int mbedtls_version_get_number(void);

/**
 * Get the version string ("x.y.z").
 *
 * \param string    The string that will receive the value.
 *                  (Should be at least 9 bytes in size)
 */
void mbedtls_version_get_string(char *string);

/**
 * Get the full version string ("Mbed TLS x.y.z").
 *
 * \param string    The string that will receive the value. The Mbed TLS version
 *                  string will use 18 bytes AT MOST including a terminating
 *                  null byte.
 *                  (So the buffer should be at least 18 bytes to receive this
 *                  version string).
 */
void mbedtls_version_get_string_full(char *string);

/**
 * \brief           Check if support for a feature was compiled into this
 *                  Mbed TLS binary. This allows you to see at runtime if the
 *                  library was for instance compiled with or without
 *                  Multi-threading support.
 *
 * \note            only checks against defines in the sections "System
 *                  support", "Mbed TLS modules" and "Mbed TLS feature
 *                  support" in config.h
 *
 * \param feature   The string for the define to check (e.g. "MBEDTLS_AES_C")
 *
 * \return          0 if the feature is present,
 *                  -1 if the feature is not present and
 *                  -2 if support for feature checking as a whole was not
 *                  compiled in.
 */
int mbedtls_version_check_feature(const char *feature);

#ifdef __cplusplus
}
#endif

#endif /* MBEDTLS_VERSION_C */

#endif /* version.h */
