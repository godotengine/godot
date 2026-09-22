/* -*-c-*-
 *  Version feature information
 *
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */

#include "mbedtls_common.h"

#if defined(MBEDTLS_VERSION_C)

#include "mbedtls/version.h"

#include <string.h>

static const char * const features[] = {
#if defined(MBEDTLS_VERSION_FEATURES)
    #if defined(MBEDTLS_NET_C)
    "NET_C", //no-check-names
#endif /* MBEDTLS_NET_C */
#if defined(MBEDTLS_TIMING_ALT)
    "TIMING_ALT", //no-check-names
#endif /* MBEDTLS_TIMING_ALT */
#if defined(MBEDTLS_TIMING_C)
    "TIMING_C", //no-check-names
#endif /* MBEDTLS_TIMING_C */
#if defined(MBEDTLS_ERROR_C)
    "ERROR_C", //no-check-names
#endif /* MBEDTLS_ERROR_C */
#if defined(MBEDTLS_ERROR_STRERROR_DUMMY)
    "ERROR_STRERROR_DUMMY", //no-check-names
#endif /* MBEDTLS_ERROR_STRERROR_DUMMY */
#if defined(MBEDTLS_VERSION_C)
    "VERSION_C", //no-check-names
#endif /* MBEDTLS_VERSION_C */
#if defined(MBEDTLS_VERSION_FEATURES)
    "VERSION_FEATURES", //no-check-names
#endif /* MBEDTLS_VERSION_FEATURES */
#if defined(MBEDTLS_CONFIG_FILE)
    "CONFIG_FILE", //no-check-names
#endif /* MBEDTLS_CONFIG_FILE */
#if defined(MBEDTLS_USER_CONFIG_FILE)
    "USER_CONFIG_FILE", //no-check-names
#endif /* MBEDTLS_USER_CONFIG_FILE */
#if defined(MBEDTLS_SSL_NULL_CIPHERSUITES)
    "SSL_NULL_CIPHERSUITES", //no-check-names
#endif /* MBEDTLS_SSL_NULL_CIPHERSUITES */
#if defined(MBEDTLS_DEBUG_C)
    "DEBUG_C", //no-check-names
#endif /* MBEDTLS_DEBUG_C */
#if defined(MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA_ENABLED)
    "KEY_EXCHANGE_ECDHE_ECDSA_ENABLED", //no-check-names
#endif /* MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA_ENABLED */
#if defined(MBEDTLS_KEY_EXCHANGE_ECDHE_PSK_ENABLED)
    "KEY_EXCHANGE_ECDHE_PSK_ENABLED", //no-check-names
#endif /* MBEDTLS_KEY_EXCHANGE_ECDHE_PSK_ENABLED */
#if defined(MBEDTLS_KEY_EXCHANGE_ECDHE_RSA_ENABLED)
    "KEY_EXCHANGE_ECDHE_RSA_ENABLED", //no-check-names
#endif /* MBEDTLS_KEY_EXCHANGE_ECDHE_RSA_ENABLED */
#if defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
    "KEY_EXCHANGE_ECJPAKE_ENABLED", //no-check-names
#endif /* MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED */
#if defined(MBEDTLS_KEY_EXCHANGE_PSK_ENABLED)
    "KEY_EXCHANGE_PSK_ENABLED", //no-check-names
#endif /* MBEDTLS_KEY_EXCHANGE_PSK_ENABLED */
#if defined(MBEDTLS_SSL_ALL_ALERT_MESSAGES)
    "SSL_ALL_ALERT_MESSAGES", //no-check-names
#endif /* MBEDTLS_SSL_ALL_ALERT_MESSAGES */
#if defined(MBEDTLS_SSL_ALPN)
    "SSL_ALPN", //no-check-names
#endif /* MBEDTLS_SSL_ALPN */
#if defined(MBEDTLS_SSL_ASYNC_PRIVATE)
    "SSL_ASYNC_PRIVATE", //no-check-names
#endif /* MBEDTLS_SSL_ASYNC_PRIVATE */
#if defined(MBEDTLS_SSL_CACHE_C)
    "SSL_CACHE_C", //no-check-names
#endif /* MBEDTLS_SSL_CACHE_C */
#if defined(MBEDTLS_SSL_CLI_C)
    "SSL_CLI_C", //no-check-names
#endif /* MBEDTLS_SSL_CLI_C */
#if defined(MBEDTLS_SSL_CONTEXT_SERIALIZATION)
    "SSL_CONTEXT_SERIALIZATION", //no-check-names
#endif /* MBEDTLS_SSL_CONTEXT_SERIALIZATION */
#if defined(MBEDTLS_SSL_COOKIE_C)
    "SSL_COOKIE_C", //no-check-names
#endif /* MBEDTLS_SSL_COOKIE_C */
#if defined(MBEDTLS_SSL_DEBUG_ALL)
    "SSL_DEBUG_ALL", //no-check-names
#endif /* MBEDTLS_SSL_DEBUG_ALL */
#if defined(MBEDTLS_SSL_DTLS_ANTI_REPLAY)
    "SSL_DTLS_ANTI_REPLAY", //no-check-names
#endif /* MBEDTLS_SSL_DTLS_ANTI_REPLAY */
#if defined(MBEDTLS_SSL_DTLS_CLIENT_PORT_REUSE)
    "SSL_DTLS_CLIENT_PORT_REUSE", //no-check-names
#endif /* MBEDTLS_SSL_DTLS_CLIENT_PORT_REUSE */
#if defined(MBEDTLS_SSL_DTLS_CONNECTION_ID)
    "SSL_DTLS_CONNECTION_ID", //no-check-names
#endif /* MBEDTLS_SSL_DTLS_CONNECTION_ID */
#if defined(MBEDTLS_SSL_DTLS_HELLO_VERIFY)
    "SSL_DTLS_HELLO_VERIFY", //no-check-names
#endif /* MBEDTLS_SSL_DTLS_HELLO_VERIFY */
#if defined(MBEDTLS_SSL_DTLS_SRTP)
    "SSL_DTLS_SRTP", //no-check-names
#endif /* MBEDTLS_SSL_DTLS_SRTP */
#if defined(MBEDTLS_SSL_EARLY_DATA)
    "SSL_EARLY_DATA", //no-check-names
#endif /* MBEDTLS_SSL_EARLY_DATA */
#if defined(MBEDTLS_SSL_ENCRYPT_THEN_MAC)
    "SSL_ENCRYPT_THEN_MAC", //no-check-names
#endif /* MBEDTLS_SSL_ENCRYPT_THEN_MAC */
#if defined(MBEDTLS_SSL_EXTENDED_MASTER_SECRET)
    "SSL_EXTENDED_MASTER_SECRET", //no-check-names
#endif /* MBEDTLS_SSL_EXTENDED_MASTER_SECRET */
#if defined(MBEDTLS_SSL_KEEP_PEER_CERTIFICATE)
    "SSL_KEEP_PEER_CERTIFICATE", //no-check-names
#endif /* MBEDTLS_SSL_KEEP_PEER_CERTIFICATE */
#if defined(MBEDTLS_SSL_MAX_FRAGMENT_LENGTH)
    "SSL_MAX_FRAGMENT_LENGTH", //no-check-names
#endif /* MBEDTLS_SSL_MAX_FRAGMENT_LENGTH */
#if defined(MBEDTLS_SSL_PROTO_DTLS)
    "SSL_PROTO_DTLS", //no-check-names
#endif /* MBEDTLS_SSL_PROTO_DTLS */
#if defined(MBEDTLS_SSL_PROTO_TLS1_2)
    "SSL_PROTO_TLS1_2", //no-check-names
#endif /* MBEDTLS_SSL_PROTO_TLS1_2 */
#if defined(MBEDTLS_SSL_PROTO_TLS1_3)
    "SSL_PROTO_TLS1_3", //no-check-names
#endif /* MBEDTLS_SSL_PROTO_TLS1_3 */
#if defined(MBEDTLS_SSL_RECORD_SIZE_LIMIT)
    "SSL_RECORD_SIZE_LIMIT", //no-check-names
#endif /* MBEDTLS_SSL_RECORD_SIZE_LIMIT */
#if defined(MBEDTLS_SSL_KEYING_MATERIAL_EXPORT)
    "SSL_KEYING_MATERIAL_EXPORT", //no-check-names
#endif /* MBEDTLS_SSL_KEYING_MATERIAL_EXPORT */
#if defined(MBEDTLS_SSL_RENEGOTIATION)
    "SSL_RENEGOTIATION", //no-check-names
#endif /* MBEDTLS_SSL_RENEGOTIATION */
#if defined(MBEDTLS_SSL_SERVER_NAME_INDICATION)
    "SSL_SERVER_NAME_INDICATION", //no-check-names
#endif /* MBEDTLS_SSL_SERVER_NAME_INDICATION */
#if defined(MBEDTLS_SSL_SESSION_TICKETS)
    "SSL_SESSION_TICKETS", //no-check-names
#endif /* MBEDTLS_SSL_SESSION_TICKETS */
#if defined(MBEDTLS_SSL_SRV_C)
    "SSL_SRV_C", //no-check-names
#endif /* MBEDTLS_SSL_SRV_C */
#if defined(MBEDTLS_SSL_TICKET_C)
    "SSL_TICKET_C", //no-check-names
#endif /* MBEDTLS_SSL_TICKET_C */
#if defined(MBEDTLS_SSL_TLS1_3_COMPATIBILITY_MODE)
    "SSL_TLS1_3_COMPATIBILITY_MODE", //no-check-names
#endif /* MBEDTLS_SSL_TLS1_3_COMPATIBILITY_MODE */
#if defined(MBEDTLS_SSL_TLS1_3_KEY_EXCHANGE_MODE_EPHEMERAL_ENABLED)
    "SSL_TLS1_3_KEY_EXCHANGE_MODE_EPHEMERAL_ENABLED", //no-check-names
#endif /* MBEDTLS_SSL_TLS1_3_KEY_EXCHANGE_MODE_EPHEMERAL_ENABLED */
#if defined(MBEDTLS_SSL_TLS1_3_KEY_EXCHANGE_MODE_PSK_ENABLED)
    "SSL_TLS1_3_KEY_EXCHANGE_MODE_PSK_ENABLED", //no-check-names
#endif /* MBEDTLS_SSL_TLS1_3_KEY_EXCHANGE_MODE_PSK_ENABLED */
#if defined(MBEDTLS_SSL_TLS1_3_KEY_EXCHANGE_MODE_PSK_EPHEMERAL_ENABLED)
    "SSL_TLS1_3_KEY_EXCHANGE_MODE_PSK_EPHEMERAL_ENABLED", //no-check-names
#endif /* MBEDTLS_SSL_TLS1_3_KEY_EXCHANGE_MODE_PSK_EPHEMERAL_ENABLED */
#if defined(MBEDTLS_SSL_TLS_C)
    "SSL_TLS_C", //no-check-names
#endif /* MBEDTLS_SSL_TLS_C */
#if defined(MBEDTLS_SSL_VARIABLE_BUFFER_LENGTH)
    "SSL_VARIABLE_BUFFER_LENGTH", //no-check-names
#endif /* MBEDTLS_SSL_VARIABLE_BUFFER_LENGTH */
#if defined(MBEDTLS_PSK_MAX_LEN)
    "PSK_MAX_LEN", //no-check-names
#endif /* MBEDTLS_PSK_MAX_LEN */
#if defined(MBEDTLS_SSL_CACHE_DEFAULT_MAX_ENTRIES)
    "SSL_CACHE_DEFAULT_MAX_ENTRIES", //no-check-names
#endif /* MBEDTLS_SSL_CACHE_DEFAULT_MAX_ENTRIES */
#if defined(MBEDTLS_SSL_CACHE_DEFAULT_TIMEOUT)
    "SSL_CACHE_DEFAULT_TIMEOUT", //no-check-names
#endif /* MBEDTLS_SSL_CACHE_DEFAULT_TIMEOUT */
#if defined(MBEDTLS_SSL_CID_IN_LEN_MAX)
    "SSL_CID_IN_LEN_MAX", //no-check-names
#endif /* MBEDTLS_SSL_CID_IN_LEN_MAX */
#if defined(MBEDTLS_SSL_CID_OUT_LEN_MAX)
    "SSL_CID_OUT_LEN_MAX", //no-check-names
#endif /* MBEDTLS_SSL_CID_OUT_LEN_MAX */
#if defined(MBEDTLS_SSL_CID_TLS1_3_PADDING_GRANULARITY)
    "SSL_CID_TLS1_3_PADDING_GRANULARITY", //no-check-names
#endif /* MBEDTLS_SSL_CID_TLS1_3_PADDING_GRANULARITY */
#if defined(MBEDTLS_SSL_CIPHERSUITES)
    "SSL_CIPHERSUITES", //no-check-names
#endif /* MBEDTLS_SSL_CIPHERSUITES */
#if defined(MBEDTLS_SSL_COOKIE_TIMEOUT)
    "SSL_COOKIE_TIMEOUT", //no-check-names
#endif /* MBEDTLS_SSL_COOKIE_TIMEOUT */
#if defined(MBEDTLS_SSL_DTLS_MAX_BUFFERING)
    "SSL_DTLS_MAX_BUFFERING", //no-check-names
#endif /* MBEDTLS_SSL_DTLS_MAX_BUFFERING */
#if defined(MBEDTLS_SSL_IN_CONTENT_LEN)
    "SSL_IN_CONTENT_LEN", //no-check-names
#endif /* MBEDTLS_SSL_IN_CONTENT_LEN */
#if defined(MBEDTLS_SSL_MAX_EARLY_DATA_SIZE)
    "SSL_MAX_EARLY_DATA_SIZE", //no-check-names
#endif /* MBEDTLS_SSL_MAX_EARLY_DATA_SIZE */
#if defined(MBEDTLS_SSL_OUT_CONTENT_LEN)
    "SSL_OUT_CONTENT_LEN", //no-check-names
#endif /* MBEDTLS_SSL_OUT_CONTENT_LEN */
#if defined(MBEDTLS_SSL_TLS1_3_DEFAULT_NEW_SESSION_TICKETS)
    "SSL_TLS1_3_DEFAULT_NEW_SESSION_TICKETS", //no-check-names
#endif /* MBEDTLS_SSL_TLS1_3_DEFAULT_NEW_SESSION_TICKETS */
#if defined(MBEDTLS_SSL_TLS1_3_TICKET_AGE_TOLERANCE)
    "SSL_TLS1_3_TICKET_AGE_TOLERANCE", //no-check-names
#endif /* MBEDTLS_SSL_TLS1_3_TICKET_AGE_TOLERANCE */
#if defined(MBEDTLS_SSL_TLS1_3_TICKET_NONCE_LENGTH)
    "SSL_TLS1_3_TICKET_NONCE_LENGTH", //no-check-names
#endif /* MBEDTLS_SSL_TLS1_3_TICKET_NONCE_LENGTH */
#endif /* MBEDTLS_VERSION_FEATURES */
    NULL
};

int mbedtls_version_check_feature(const char *feature)
{
    const char * const *idx = features;

    if (*idx == NULL) {
        return -2;
    }

    if (feature == NULL) {
        return -1;
    }

    if (strncmp(feature, "MBEDTLS_", 8)) {
        return -1;
    }

    feature += 8;

    while (*idx != NULL) {
        if (!strcmp(*idx, feature)) {
            return 0;
        }
        idx++;
    }
    return -1;
}

#endif /* MBEDTLS_VERSION_C */
