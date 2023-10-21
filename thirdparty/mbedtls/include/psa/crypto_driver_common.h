/**
 * \file psa/crypto_driver_common.h
 * \brief Definitions for all PSA crypto drivers
 *
 * This file contains common definitions shared by all PSA crypto drivers.
 * Do not include it directly: instead, include the header file(s) for
 * the type(s) of driver that you are implementing. For example, if
 * you are writing a dynamically registered driver for a secure element,
 * include `psa/crypto_se_driver.h`.
 *
 * This file is part of the PSA Crypto Driver Model, containing functions for
 * driver developers to implement to enable hardware to be called in a
 * standardized way by a PSA Cryptographic API implementation. The functions
 * comprising the driver model, which driver authors implement, are not
 * intended to be called by application developers.
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
#ifndef PSA_CRYPTO_DRIVER_COMMON_H
#define PSA_CRYPTO_DRIVER_COMMON_H

#include <stddef.h>
#include <stdint.h>

/* Include type definitions (psa_status_t, psa_algorithm_t,
 * psa_key_type_t, etc.) and macros to build and analyze values
 * of these types. */
#include "crypto_types.h"
#include "crypto_values.h"
/* Include size definitions which are used to size some arrays in operation
 * structures. */
#include <psa/crypto_sizes.h>

/** For encrypt-decrypt functions, whether the operation is an encryption
 * or a decryption. */
typedef enum {
    PSA_CRYPTO_DRIVER_DECRYPT,
    PSA_CRYPTO_DRIVER_ENCRYPT
} psa_encrypt_or_decrypt_t;

#endif /* PSA_CRYPTO_DRIVER_COMMON_H */
