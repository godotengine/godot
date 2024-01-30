/**
 * \file mbedtls/config_adjust_x509.h
 * \brief Adjust X.509 configuration
 *
 * Automatically enable certain dependencies. Generally, MBEDLTS_xxx
 * configurations need to be explicitly enabled by the user: enabling
 * MBEDTLS_xxx_A but not MBEDTLS_xxx_B when A requires B results in a
 * compilation error. However, we do automatically enable certain options
 * in some circumstances. One case is if MBEDTLS_xxx_B is an internal option
 * used to identify parts of a module that are used by other module, and we
 * don't want to make the symbol MBEDTLS_xxx_B part of the public API.
 * Another case is if A didn't depend on B in earlier versions, and we
 * want to use B in A but we need to preserve backward compatibility with
 * configurations that explicitly activate MBEDTLS_xxx_A but not
 * MBEDTLS_xxx_B.
 */
/*
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */

#ifndef MBEDTLS_CONFIG_ADJUST_X509_H
#define MBEDTLS_CONFIG_ADJUST_X509_H

#endif /* MBEDTLS_CONFIG_ADJUST_X509_H */
