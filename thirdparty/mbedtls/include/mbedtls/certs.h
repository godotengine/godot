/**
 * \file certs.h
 *
 * \brief Sample certificates and DHM parameters for testing
 */
/*
 *  Copyright (C) 2006-2015, ARM Limited, All Rights Reserved
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 *
 *  This file is provided under the Apache License 2.0, or the
 *  GNU General Public License v2.0 or later.
 *
 *  **********
 *  Apache License 2.0:
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
 *  **********
 *
 *  **********
 *  GNU General Public License v2.0 or later:
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 *  **********
 *
 *  This file is part of mbed TLS (https://tls.mbed.org)
 */
#ifndef MBEDTLS_CERTS_H
#define MBEDTLS_CERTS_H

#if !defined(MBEDTLS_CONFIG_FILE)
#include "config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* List of all PEM-encoded CA certificates, terminated by NULL;
 * PEM encoded if MBEDTLS_PEM_PARSE_C is enabled, DER encoded
 * otherwise. */
extern const char * mbedtls_test_cas[];
extern const size_t mbedtls_test_cas_len[];

/* List of all DER-encoded CA certificates, terminated by NULL */
extern const unsigned char * mbedtls_test_cas_der[];
extern const size_t mbedtls_test_cas_der_len[];

#if defined(MBEDTLS_PEM_PARSE_C)
/* Concatenation of all CA certificates in PEM format if available */
extern const char   mbedtls_test_cas_pem[];
extern const size_t mbedtls_test_cas_pem_len;
#endif /* MBEDTLS_PEM_PARSE_C */

/*
 * CA test certificates
 */

extern const char mbedtls_test_ca_crt_ec_pem[];
extern const char mbedtls_test_ca_key_ec_pem[];
extern const char mbedtls_test_ca_pwd_ec_pem[];
extern const char mbedtls_test_ca_key_rsa_pem[];
extern const char mbedtls_test_ca_pwd_rsa_pem[];
extern const char mbedtls_test_ca_crt_rsa_sha1_pem[];
extern const char mbedtls_test_ca_crt_rsa_sha256_pem[];

extern const unsigned char mbedtls_test_ca_crt_ec_der[];
extern const unsigned char mbedtls_test_ca_key_ec_der[];
extern const unsigned char mbedtls_test_ca_key_rsa_der[];
extern const unsigned char mbedtls_test_ca_crt_rsa_sha1_der[];
extern const unsigned char mbedtls_test_ca_crt_rsa_sha256_der[];

extern const size_t mbedtls_test_ca_crt_ec_pem_len;
extern const size_t mbedtls_test_ca_key_ec_pem_len;
extern const size_t mbedtls_test_ca_pwd_ec_pem_len;
extern const size_t mbedtls_test_ca_key_rsa_pem_len;
extern const size_t mbedtls_test_ca_pwd_rsa_pem_len;
extern const size_t mbedtls_test_ca_crt_rsa_sha1_pem_len;
extern const size_t mbedtls_test_ca_crt_rsa_sha256_pem_len;

extern const size_t mbedtls_test_ca_crt_ec_der_len;
extern const size_t mbedtls_test_ca_key_ec_der_len;
extern const size_t mbedtls_test_ca_pwd_ec_der_len;
extern const size_t mbedtls_test_ca_key_rsa_der_len;
extern const size_t mbedtls_test_ca_pwd_rsa_der_len;
extern const size_t mbedtls_test_ca_crt_rsa_sha1_der_len;
extern const size_t mbedtls_test_ca_crt_rsa_sha256_der_len;

/* Config-dependent dispatch between PEM and DER encoding
 * (PEM if enabled, otherwise DER) */

extern const char mbedtls_test_ca_crt_ec[];
extern const char mbedtls_test_ca_key_ec[];
extern const char mbedtls_test_ca_pwd_ec[];
extern const char mbedtls_test_ca_key_rsa[];
extern const char mbedtls_test_ca_pwd_rsa[];
extern const char mbedtls_test_ca_crt_rsa_sha1[];
extern const char mbedtls_test_ca_crt_rsa_sha256[];

extern const size_t mbedtls_test_ca_crt_ec_len;
extern const size_t mbedtls_test_ca_key_ec_len;
extern const size_t mbedtls_test_ca_pwd_ec_len;
extern const size_t mbedtls_test_ca_key_rsa_len;
extern const size_t mbedtls_test_ca_pwd_rsa_len;
extern const size_t mbedtls_test_ca_crt_rsa_sha1_len;
extern const size_t mbedtls_test_ca_crt_rsa_sha256_len;

/* Config-dependent dispatch between SHA-1 and SHA-256
 * (SHA-256 if enabled, otherwise SHA-1) */

extern const char mbedtls_test_ca_crt_rsa[];
extern const size_t mbedtls_test_ca_crt_rsa_len;

/* Config-dependent dispatch between EC and RSA
 * (RSA if enabled, otherwise EC) */

extern const char * mbedtls_test_ca_crt;
extern const char * mbedtls_test_ca_key;
extern const char * mbedtls_test_ca_pwd;
extern const size_t mbedtls_test_ca_crt_len;
extern const size_t mbedtls_test_ca_key_len;
extern const size_t mbedtls_test_ca_pwd_len;

/*
 * Server test certificates
 */

extern const char mbedtls_test_srv_crt_ec_pem[];
extern const char mbedtls_test_srv_key_ec_pem[];
extern const char mbedtls_test_srv_pwd_ec_pem[];
extern const char mbedtls_test_srv_key_rsa_pem[];
extern const char mbedtls_test_srv_pwd_rsa_pem[];
extern const char mbedtls_test_srv_crt_rsa_sha1_pem[];
extern const char mbedtls_test_srv_crt_rsa_sha256_pem[];

extern const unsigned char mbedtls_test_srv_crt_ec_der[];
extern const unsigned char mbedtls_test_srv_key_ec_der[];
extern const unsigned char mbedtls_test_srv_key_rsa_der[];
extern const unsigned char mbedtls_test_srv_crt_rsa_sha1_der[];
extern const unsigned char mbedtls_test_srv_crt_rsa_sha256_der[];

extern const size_t mbedtls_test_srv_crt_ec_pem_len;
extern const size_t mbedtls_test_srv_key_ec_pem_len;
extern const size_t mbedtls_test_srv_pwd_ec_pem_len;
extern const size_t mbedtls_test_srv_key_rsa_pem_len;
extern const size_t mbedtls_test_srv_pwd_rsa_pem_len;
extern const size_t mbedtls_test_srv_crt_rsa_sha1_pem_len;
extern const size_t mbedtls_test_srv_crt_rsa_sha256_pem_len;

extern const size_t mbedtls_test_srv_crt_ec_der_len;
extern const size_t mbedtls_test_srv_key_ec_der_len;
extern const size_t mbedtls_test_srv_pwd_ec_der_len;
extern const size_t mbedtls_test_srv_key_rsa_der_len;
extern const size_t mbedtls_test_srv_pwd_rsa_der_len;
extern const size_t mbedtls_test_srv_crt_rsa_sha1_der_len;
extern const size_t mbedtls_test_srv_crt_rsa_sha256_der_len;

/* Config-dependent dispatch between PEM and DER encoding
 * (PEM if enabled, otherwise DER) */

extern const char mbedtls_test_srv_crt_ec[];
extern const char mbedtls_test_srv_key_ec[];
extern const char mbedtls_test_srv_pwd_ec[];
extern const char mbedtls_test_srv_key_rsa[];
extern const char mbedtls_test_srv_pwd_rsa[];
extern const char mbedtls_test_srv_crt_rsa_sha1[];
extern const char mbedtls_test_srv_crt_rsa_sha256[];

extern const size_t mbedtls_test_srv_crt_ec_len;
extern const size_t mbedtls_test_srv_key_ec_len;
extern const size_t mbedtls_test_srv_pwd_ec_len;
extern const size_t mbedtls_test_srv_key_rsa_len;
extern const size_t mbedtls_test_srv_pwd_rsa_len;
extern const size_t mbedtls_test_srv_crt_rsa_sha1_len;
extern const size_t mbedtls_test_srv_crt_rsa_sha256_len;

/* Config-dependent dispatch between SHA-1 and SHA-256
 * (SHA-256 if enabled, otherwise SHA-1) */

extern const char mbedtls_test_srv_crt_rsa[];
extern const size_t mbedtls_test_srv_crt_rsa_len;

/* Config-dependent dispatch between EC and RSA
 * (RSA if enabled, otherwise EC) */

extern const char * mbedtls_test_srv_crt;
extern const char * mbedtls_test_srv_key;
extern const char * mbedtls_test_srv_pwd;
extern const size_t mbedtls_test_srv_crt_len;
extern const size_t mbedtls_test_srv_key_len;
extern const size_t mbedtls_test_srv_pwd_len;

/*
 * Client test certificates
 */

extern const char mbedtls_test_cli_crt_ec_pem[];
extern const char mbedtls_test_cli_key_ec_pem[];
extern const char mbedtls_test_cli_pwd_ec_pem[];
extern const char mbedtls_test_cli_key_rsa_pem[];
extern const char mbedtls_test_cli_pwd_rsa_pem[];
extern const char mbedtls_test_cli_crt_rsa_pem[];

extern const unsigned char mbedtls_test_cli_crt_ec_der[];
extern const unsigned char mbedtls_test_cli_key_ec_der[];
extern const unsigned char mbedtls_test_cli_key_rsa_der[];
extern const unsigned char mbedtls_test_cli_crt_rsa_der[];

extern const size_t mbedtls_test_cli_crt_ec_pem_len;
extern const size_t mbedtls_test_cli_key_ec_pem_len;
extern const size_t mbedtls_test_cli_pwd_ec_pem_len;
extern const size_t mbedtls_test_cli_key_rsa_pem_len;
extern const size_t mbedtls_test_cli_pwd_rsa_pem_len;
extern const size_t mbedtls_test_cli_crt_rsa_pem_len;

extern const size_t mbedtls_test_cli_crt_ec_der_len;
extern const size_t mbedtls_test_cli_key_ec_der_len;
extern const size_t mbedtls_test_cli_key_rsa_der_len;
extern const size_t mbedtls_test_cli_crt_rsa_der_len;

/* Config-dependent dispatch between PEM and DER encoding
 * (PEM if enabled, otherwise DER) */

extern const char mbedtls_test_cli_crt_ec[];
extern const char mbedtls_test_cli_key_ec[];
extern const char mbedtls_test_cli_pwd_ec[];
extern const char mbedtls_test_cli_key_rsa[];
extern const char mbedtls_test_cli_pwd_rsa[];
extern const char mbedtls_test_cli_crt_rsa[];

extern const size_t mbedtls_test_cli_crt_ec_len;
extern const size_t mbedtls_test_cli_key_ec_len;
extern const size_t mbedtls_test_cli_pwd_ec_len;
extern const size_t mbedtls_test_cli_key_rsa_len;
extern const size_t mbedtls_test_cli_pwd_rsa_len;
extern const size_t mbedtls_test_cli_crt_rsa_len;

/* Config-dependent dispatch between EC and RSA
 * (RSA if enabled, otherwise EC) */

extern const char * mbedtls_test_cli_crt;
extern const char * mbedtls_test_cli_key;
extern const char * mbedtls_test_cli_pwd;
extern const size_t mbedtls_test_cli_crt_len;
extern const size_t mbedtls_test_cli_key_len;
extern const size_t mbedtls_test_cli_pwd_len;

#ifdef __cplusplus
}
#endif

#endif /* certs.h */
