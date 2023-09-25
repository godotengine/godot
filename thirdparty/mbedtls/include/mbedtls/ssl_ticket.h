/**
 * \file ssl_ticket.h
 *
 * \brief TLS server ticket callbacks implementation
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
#ifndef MBEDTLS_SSL_TICKET_H
#define MBEDTLS_SSL_TICKET_H
#include "mbedtls/private_access.h"

#include "mbedtls/build_info.h"

/*
 * This implementation of the session ticket callbacks includes key
 * management, rotating the keys periodically in order to preserve forward
 * secrecy, when MBEDTLS_HAVE_TIME is defined.
 */

#include "mbedtls/ssl.h"
#include "mbedtls/cipher.h"

#if defined(MBEDTLS_HAVE_TIME)
#include "mbedtls/platform_time.h"
#endif

#if defined(MBEDTLS_USE_PSA_CRYPTO)
#include "psa/crypto.h"
#endif

#if defined(MBEDTLS_THREADING_C)
#include "mbedtls/threading.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define MBEDTLS_SSL_TICKET_MAX_KEY_BYTES 32          /*!< Max supported key length in bytes */
#define MBEDTLS_SSL_TICKET_KEY_NAME_BYTES 4          /*!< key name length in bytes */

/**
 * \brief   Information for session ticket protection
 */
typedef struct mbedtls_ssl_ticket_key {
    unsigned char MBEDTLS_PRIVATE(name)[MBEDTLS_SSL_TICKET_KEY_NAME_BYTES];
    /*!< random key identifier              */
#if defined(MBEDTLS_HAVE_TIME)
    mbedtls_time_t MBEDTLS_PRIVATE(generation_time); /*!< key generation timestamp (seconds) */
#endif
#if !defined(MBEDTLS_USE_PSA_CRYPTO)
    mbedtls_cipher_context_t MBEDTLS_PRIVATE(ctx);   /*!< context for auth enc/decryption    */
#else
    mbedtls_svc_key_id_t MBEDTLS_PRIVATE(key);       /*!< key used for auth enc/decryption   */
    psa_algorithm_t MBEDTLS_PRIVATE(alg);            /*!< algorithm of auth enc/decryption   */
    psa_key_type_t MBEDTLS_PRIVATE(key_type);        /*!< key type                           */
    size_t MBEDTLS_PRIVATE(key_bits);                /*!< key length in bits                 */
#endif
}
mbedtls_ssl_ticket_key;

/**
 * \brief   Context for session ticket handling functions
 */
typedef struct mbedtls_ssl_ticket_context {
    mbedtls_ssl_ticket_key MBEDTLS_PRIVATE(keys)[2]; /*!< ticket protection keys             */
    unsigned char MBEDTLS_PRIVATE(active);           /*!< index of the currently active key  */

    uint32_t MBEDTLS_PRIVATE(ticket_lifetime);       /*!< lifetime of tickets in seconds     */

    /** Callback for getting (pseudo-)random numbers                        */
    int(*MBEDTLS_PRIVATE(f_rng))(void *, unsigned char *, size_t);
    void *MBEDTLS_PRIVATE(p_rng);                    /*!< context for the RNG function       */

#if defined(MBEDTLS_THREADING_C)
    mbedtls_threading_mutex_t MBEDTLS_PRIVATE(mutex);
#endif
}
mbedtls_ssl_ticket_context;

/**
 * \brief           Initialize a ticket context.
 *                  (Just make it ready for mbedtls_ssl_ticket_setup()
 *                  or mbedtls_ssl_ticket_free().)
 *
 * \param ctx       Context to be initialized
 */
void mbedtls_ssl_ticket_init(mbedtls_ssl_ticket_context *ctx);

/**
 * \brief           Prepare context to be actually used
 *
 * \param ctx       Context to be set up
 * \param f_rng     RNG callback function (mandatory)
 * \param p_rng     RNG callback context
 * \param cipher    AEAD cipher to use for ticket protection.
 *                  Recommended value: MBEDTLS_CIPHER_AES_256_GCM.
 * \param lifetime  Tickets lifetime in seconds
 *                  Recommended value: 86400 (one day).
 *
 * \note            It is highly recommended to select a cipher that is at
 *                  least as strong as the strongest ciphersuite
 *                  supported. Usually that means a 256-bit key.
 *
 * \note            The lifetime of the keys is twice the lifetime of tickets.
 *                  It is recommended to pick a reasonable lifetime so as not
 *                  to negate the benefits of forward secrecy.
 *
 * \return          0 if successful,
 *                  or a specific MBEDTLS_ERR_XXX error code
 */
int mbedtls_ssl_ticket_setup(mbedtls_ssl_ticket_context *ctx,
                             int (*f_rng)(void *, unsigned char *, size_t), void *p_rng,
                             mbedtls_cipher_type_t cipher,
                             uint32_t lifetime);

/**
 * \brief           Rotate session ticket encryption key to new specified key.
 *                  Provides for external control of session ticket encryption
 *                  key rotation, e.g. for synchronization between different
 *                  machines.  If this function is not used, or if not called
 *                  before ticket lifetime expires, then a new session ticket
 *                  encryption key is generated internally in order to avoid
 *                  unbounded session ticket encryption key lifetimes.
 *
 * \param ctx       Context to be set up
 * \param name      Session ticket encryption key name
 * \param nlength   Session ticket encryption key name length in bytes
 * \param k         Session ticket encryption key
 * \param klength   Session ticket encryption key length in bytes
 * \param lifetime  Tickets lifetime in seconds
 *                  Recommended value: 86400 (one day).
 *
 * \note            \c name and \c k are recommended to be cryptographically
 *                  random data.
 *
 * \note            \c nlength must match sizeof( ctx->name )
 *
 * \note            \c klength must be sufficient for use by cipher specified
 *                  to \c mbedtls_ssl_ticket_setup
 *
 * \note            The lifetime of the keys is twice the lifetime of tickets.
 *                  It is recommended to pick a reasonable lifetime so as not
 *                  to negate the benefits of forward secrecy.
 *
 * \return          0 if successful,
 *                  or a specific MBEDTLS_ERR_XXX error code
 */
int mbedtls_ssl_ticket_rotate(mbedtls_ssl_ticket_context *ctx,
                              const unsigned char *name, size_t nlength,
                              const unsigned char *k, size_t klength,
                              uint32_t lifetime);

/**
 * \brief           Implementation of the ticket write callback
 *
 * \note            See \c mbedtls_ssl_ticket_write_t for description
 */
mbedtls_ssl_ticket_write_t mbedtls_ssl_ticket_write;

/**
 * \brief           Implementation of the ticket parse callback
 *
 * \note            See \c mbedtls_ssl_ticket_parse_t for description
 */
mbedtls_ssl_ticket_parse_t mbedtls_ssl_ticket_parse;

/**
 * \brief           Free a context's content and zeroize it.
 *
 * \param ctx       Context to be cleaned up
 */
void mbedtls_ssl_ticket_free(mbedtls_ssl_ticket_context *ctx);

#ifdef __cplusplus
}
#endif

#endif /* ssl_ticket.h */
