/**
 * \file ssl.h
 *
 * \brief SSL/TLS functions.
 */
/*
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */
#ifndef MBEDTLS_SSL_H
#define MBEDTLS_SSL_H

#if !defined(MBEDTLS_CONFIG_FILE)
#include "mbedtls/config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#include "mbedtls/bignum.h"
#include "mbedtls/ecp.h"

#include "mbedtls/ssl_ciphersuites.h"

#if defined(MBEDTLS_X509_CRT_PARSE_C)
#include "mbedtls/x509_crt.h"
#include "mbedtls/x509_crl.h"
#endif

#if defined(MBEDTLS_DHM_C)
#include "mbedtls/dhm.h"
#endif

/* Adding guard for MBEDTLS_ECDSA_C to ensure no compile errors due
 * to guards also being in ssl_srv.c and ssl_cli.c. There is a gap
 * in functionality that access to ecdh_ctx structure is needed for
 * MBEDTLS_ECDSA_C which does not seem correct.
 */
#if defined(MBEDTLS_ECDH_C) || defined(MBEDTLS_ECDSA_C)
#include "mbedtls/ecdh.h"
#endif

#if defined(MBEDTLS_ZLIB_SUPPORT)

#if defined(MBEDTLS_DEPRECATED_WARNING)
#warning \
    "Record compression support via MBEDTLS_ZLIB_SUPPORT is deprecated and will be removed in the next major revision of the library"
#endif

#if defined(MBEDTLS_DEPRECATED_REMOVED)
#error \
    "Record compression support via MBEDTLS_ZLIB_SUPPORT is deprecated and cannot be used if MBEDTLS_DEPRECATED_REMOVED is set"
#endif

#include "zlib.h"
#endif

#if defined(MBEDTLS_HAVE_TIME)
#include "mbedtls/platform_time.h"
#endif

#if defined(MBEDTLS_USE_PSA_CRYPTO)
#include "psa/crypto.h"
#endif /* MBEDTLS_USE_PSA_CRYPTO */

/*
 * SSL Error codes
 */
/** The requested feature is not available. */
#define MBEDTLS_ERR_SSL_FEATURE_UNAVAILABLE               -0x7080
/** Bad input parameters to function. */
#define MBEDTLS_ERR_SSL_BAD_INPUT_DATA                    -0x7100
/** Verification of the message MAC failed. */
#define MBEDTLS_ERR_SSL_INVALID_MAC                       -0x7180
/** An invalid SSL record was received. */
#define MBEDTLS_ERR_SSL_INVALID_RECORD                    -0x7200
/** The connection indicated an EOF. */
#define MBEDTLS_ERR_SSL_CONN_EOF                          -0x7280
/** An unknown cipher was received. */
#define MBEDTLS_ERR_SSL_UNKNOWN_CIPHER                    -0x7300
/** The server has no ciphersuites in common with the client. */
#define MBEDTLS_ERR_SSL_NO_CIPHER_CHOSEN                  -0x7380
/** No RNG was provided to the SSL module. */
#define MBEDTLS_ERR_SSL_NO_RNG                            -0x7400
/** No client certification received from the client, but required by the authentication mode. */
#define MBEDTLS_ERR_SSL_NO_CLIENT_CERTIFICATE             -0x7480
/** Our own certificate(s) is/are too large to send in an SSL message. */
#define MBEDTLS_ERR_SSL_CERTIFICATE_TOO_LARGE             -0x7500
/** The own certificate is not set, but needed by the server. */
#define MBEDTLS_ERR_SSL_CERTIFICATE_REQUIRED              -0x7580
/** The own private key or pre-shared key is not set, but needed. */
#define MBEDTLS_ERR_SSL_PRIVATE_KEY_REQUIRED              -0x7600
/** No CA Chain is set, but required to operate. */
#define MBEDTLS_ERR_SSL_CA_CHAIN_REQUIRED                 -0x7680
/** An unexpected message was received from our peer. */
#define MBEDTLS_ERR_SSL_UNEXPECTED_MESSAGE                -0x7700
/** A fatal alert message was received from our peer. */
#define MBEDTLS_ERR_SSL_FATAL_ALERT_MESSAGE               -0x7780
/** Verification of our peer failed. */
#define MBEDTLS_ERR_SSL_PEER_VERIFY_FAILED                -0x7800
/** The peer notified us that the connection is going to be closed. */
#define MBEDTLS_ERR_SSL_PEER_CLOSE_NOTIFY                 -0x7880
/** Processing of the ClientHello handshake message failed. */
#define MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO               -0x7900
/** Processing of the ServerHello handshake message failed. */
#define MBEDTLS_ERR_SSL_BAD_HS_SERVER_HELLO               -0x7980
/** Processing of the Certificate handshake message failed. */
#define MBEDTLS_ERR_SSL_BAD_HS_CERTIFICATE                -0x7A00
/** Processing of the CertificateRequest handshake message failed. */
#define MBEDTLS_ERR_SSL_BAD_HS_CERTIFICATE_REQUEST        -0x7A80
/** Processing of the ServerKeyExchange handshake message failed. */
#define MBEDTLS_ERR_SSL_BAD_HS_SERVER_KEY_EXCHANGE        -0x7B00
/** Processing of the ServerHelloDone handshake message failed. */
#define MBEDTLS_ERR_SSL_BAD_HS_SERVER_HELLO_DONE          -0x7B80
/** Processing of the ClientKeyExchange handshake message failed. */
#define MBEDTLS_ERR_SSL_BAD_HS_CLIENT_KEY_EXCHANGE        -0x7C00
/** Processing of the ClientKeyExchange handshake message failed in DHM / ECDH Read Public. */
#define MBEDTLS_ERR_SSL_BAD_HS_CLIENT_KEY_EXCHANGE_RP     -0x7C80
/** Processing of the ClientKeyExchange handshake message failed in DHM / ECDH Calculate Secret. */
#define MBEDTLS_ERR_SSL_BAD_HS_CLIENT_KEY_EXCHANGE_CS     -0x7D00
/** Processing of the CertificateVerify handshake message failed. */
#define MBEDTLS_ERR_SSL_BAD_HS_CERTIFICATE_VERIFY         -0x7D80
/** Processing of the ChangeCipherSpec handshake message failed. */
#define MBEDTLS_ERR_SSL_BAD_HS_CHANGE_CIPHER_SPEC         -0x7E00
/** Processing of the Finished handshake message failed. */
#define MBEDTLS_ERR_SSL_BAD_HS_FINISHED                   -0x7E80
/** Memory allocation failed */
#define MBEDTLS_ERR_SSL_ALLOC_FAILED                      -0x7F00
/** Hardware acceleration function returned with error */
#define MBEDTLS_ERR_SSL_HW_ACCEL_FAILED                   -0x7F80
/** Hardware acceleration function skipped / left alone data */
#define MBEDTLS_ERR_SSL_HW_ACCEL_FALLTHROUGH              -0x6F80
/** Processing of the compression / decompression failed */
#define MBEDTLS_ERR_SSL_COMPRESSION_FAILED                -0x6F00
/** Handshake protocol not within min/max boundaries */
#define MBEDTLS_ERR_SSL_BAD_HS_PROTOCOL_VERSION           -0x6E80
/** Processing of the NewSessionTicket handshake message failed. */
#define MBEDTLS_ERR_SSL_BAD_HS_NEW_SESSION_TICKET         -0x6E00
/** Session ticket has expired. */
#define MBEDTLS_ERR_SSL_SESSION_TICKET_EXPIRED            -0x6D80
/** Public key type mismatch (eg, asked for RSA key exchange and presented EC key) */
#define MBEDTLS_ERR_SSL_PK_TYPE_MISMATCH                  -0x6D00
/** Unknown identity received (eg, PSK identity) */
#define MBEDTLS_ERR_SSL_UNKNOWN_IDENTITY                  -0x6C80
/** Internal error (eg, unexpected failure in lower-level module) */
#define MBEDTLS_ERR_SSL_INTERNAL_ERROR                    -0x6C00
/** A counter would wrap (eg, too many messages exchanged). */
#define MBEDTLS_ERR_SSL_COUNTER_WRAPPING                  -0x6B80
/** Unexpected message at ServerHello in renegotiation. */
#define MBEDTLS_ERR_SSL_WAITING_SERVER_HELLO_RENEGO       -0x6B00
/** DTLS client must retry for hello verification */
#define MBEDTLS_ERR_SSL_HELLO_VERIFY_REQUIRED             -0x6A80
/** A buffer is too small to receive or write a message */
#define MBEDTLS_ERR_SSL_BUFFER_TOO_SMALL                  -0x6A00
/** None of the common ciphersuites is usable (eg, no suitable certificate, see debug messages). */
#define MBEDTLS_ERR_SSL_NO_USABLE_CIPHERSUITE             -0x6980
/** No data of requested type currently available on underlying transport. */
#define MBEDTLS_ERR_SSL_WANT_READ                         -0x6900
/** Connection requires a write call. */
#define MBEDTLS_ERR_SSL_WANT_WRITE                        -0x6880
/** The operation timed out. */
#define MBEDTLS_ERR_SSL_TIMEOUT                           -0x6800
/** The client initiated a reconnect from the same port. */
#define MBEDTLS_ERR_SSL_CLIENT_RECONNECT                  -0x6780
/** Record header looks valid but is not expected. */
#define MBEDTLS_ERR_SSL_UNEXPECTED_RECORD                 -0x6700
/** The alert message received indicates a non-fatal error. */
#define MBEDTLS_ERR_SSL_NON_FATAL                         -0x6680
/** Couldn't set the hash for verifying CertificateVerify */
#define MBEDTLS_ERR_SSL_INVALID_VERIFY_HASH               -0x6600
/** Internal-only message signaling that further message-processing should be done */
#define MBEDTLS_ERR_SSL_CONTINUE_PROCESSING               -0x6580
/** The asynchronous operation is not completed yet. */
#define MBEDTLS_ERR_SSL_ASYNC_IN_PROGRESS                 -0x6500
/** Internal-only message signaling that a message arrived early. */
#define MBEDTLS_ERR_SSL_EARLY_MESSAGE                     -0x6480
/** An encrypted DTLS-frame with an unexpected CID was received. */
#define MBEDTLS_ERR_SSL_UNEXPECTED_CID                    -0x6000
/** An operation failed due to an unexpected version or configuration. */
#define MBEDTLS_ERR_SSL_VERSION_MISMATCH                  -0x5F00
/** A cryptographic operation is in progress. Try again later. */
#define MBEDTLS_ERR_SSL_CRYPTO_IN_PROGRESS                -0x7000
/** Invalid value in SSL config */
#define MBEDTLS_ERR_SSL_BAD_CONFIG                        -0x5E80
/** Cache entry not found */
#define MBEDTLS_ERR_SSL_CACHE_ENTRY_NOT_FOUND             -0x5E00

/*
 * Various constants
 */
#define MBEDTLS_SSL_MAJOR_VERSION_3             3
#define MBEDTLS_SSL_MINOR_VERSION_0             0   /*!< SSL v3.0 */
#define MBEDTLS_SSL_MINOR_VERSION_1             1   /*!< TLS v1.0 */
#define MBEDTLS_SSL_MINOR_VERSION_2             2   /*!< TLS v1.1 */
#define MBEDTLS_SSL_MINOR_VERSION_3             3   /*!< TLS v1.2 */
#define MBEDTLS_SSL_MINOR_VERSION_4             4   /*!< TLS v1.3 (experimental) */

#define MBEDTLS_SSL_TRANSPORT_STREAM            0   /*!< TLS      */
#define MBEDTLS_SSL_TRANSPORT_DATAGRAM          1   /*!< DTLS     */

#define MBEDTLS_SSL_MAX_HOST_NAME_LEN           255 /*!< Maximum host name defined in RFC 1035 */
#define MBEDTLS_SSL_MAX_ALPN_NAME_LEN           255 /*!< Maximum size in bytes of a protocol name in alpn ext., RFC 7301 */

#define MBEDTLS_SSL_MAX_ALPN_LIST_LEN           65535 /*!< Maximum size in bytes of list in alpn ext., RFC 7301          */

/* RFC 6066 section 4, see also mfl_code_to_length in ssl_tls.c
 * NONE must be zero so that memset()ing structure to zero works */
#define MBEDTLS_SSL_MAX_FRAG_LEN_NONE           0   /*!< don't use this extension   */
#define MBEDTLS_SSL_MAX_FRAG_LEN_512            1   /*!< MaxFragmentLength 2^9      */
#define MBEDTLS_SSL_MAX_FRAG_LEN_1024           2   /*!< MaxFragmentLength 2^10     */
#define MBEDTLS_SSL_MAX_FRAG_LEN_2048           3   /*!< MaxFragmentLength 2^11     */
#define MBEDTLS_SSL_MAX_FRAG_LEN_4096           4   /*!< MaxFragmentLength 2^12     */
#define MBEDTLS_SSL_MAX_FRAG_LEN_INVALID        5   /*!< first invalid value        */

#define MBEDTLS_SSL_IS_CLIENT                   0
#define MBEDTLS_SSL_IS_SERVER                   1

#define MBEDTLS_SSL_IS_NOT_FALLBACK             0
#define MBEDTLS_SSL_IS_FALLBACK                 1

#define MBEDTLS_SSL_EXTENDED_MS_DISABLED        0
#define MBEDTLS_SSL_EXTENDED_MS_ENABLED         1

#define MBEDTLS_SSL_CID_DISABLED                0
#define MBEDTLS_SSL_CID_ENABLED                 1

#define MBEDTLS_SSL_ETM_DISABLED                0
#define MBEDTLS_SSL_ETM_ENABLED                 1

#define MBEDTLS_SSL_COMPRESS_NULL               0
#define MBEDTLS_SSL_COMPRESS_DEFLATE            1

#define MBEDTLS_SSL_VERIFY_NONE                 0
#define MBEDTLS_SSL_VERIFY_OPTIONAL             1
#define MBEDTLS_SSL_VERIFY_REQUIRED             2
#define MBEDTLS_SSL_VERIFY_UNSET                3 /* Used only for sni_authmode */

#define MBEDTLS_SSL_LEGACY_RENEGOTIATION        0
#define MBEDTLS_SSL_SECURE_RENEGOTIATION        1

#define MBEDTLS_SSL_RENEGOTIATION_DISABLED      0
#define MBEDTLS_SSL_RENEGOTIATION_ENABLED       1

#define MBEDTLS_SSL_ANTI_REPLAY_DISABLED        0
#define MBEDTLS_SSL_ANTI_REPLAY_ENABLED         1

#define MBEDTLS_SSL_RENEGOTIATION_NOT_ENFORCED  -1
#define MBEDTLS_SSL_RENEGO_MAX_RECORDS_DEFAULT  16

#define MBEDTLS_SSL_LEGACY_NO_RENEGOTIATION     0
#define MBEDTLS_SSL_LEGACY_ALLOW_RENEGOTIATION  1
#define MBEDTLS_SSL_LEGACY_BREAK_HANDSHAKE      2

#define MBEDTLS_SSL_TRUNC_HMAC_DISABLED         0
#define MBEDTLS_SSL_TRUNC_HMAC_ENABLED          1
#define MBEDTLS_SSL_TRUNCATED_HMAC_LEN          10  /* 80 bits, rfc 6066 section 7 */

#define MBEDTLS_SSL_SESSION_TICKETS_DISABLED     0
#define MBEDTLS_SSL_SESSION_TICKETS_ENABLED      1

#define MBEDTLS_SSL_CBC_RECORD_SPLITTING_DISABLED    0
#define MBEDTLS_SSL_CBC_RECORD_SPLITTING_ENABLED     1

#define MBEDTLS_SSL_ARC4_ENABLED                0
#define MBEDTLS_SSL_ARC4_DISABLED               1

#define MBEDTLS_SSL_PRESET_DEFAULT              0
#define MBEDTLS_SSL_PRESET_SUITEB               2

#define MBEDTLS_SSL_CERT_REQ_CA_LIST_ENABLED       1
#define MBEDTLS_SSL_CERT_REQ_CA_LIST_DISABLED      0

#define MBEDTLS_SSL_DTLS_SRTP_MKI_UNSUPPORTED    0
#define MBEDTLS_SSL_DTLS_SRTP_MKI_SUPPORTED      1

/*
 * Default range for DTLS retransmission timer value, in milliseconds.
 * RFC 6347 4.2.4.1 says from 1 second to 60 seconds.
 */
#define MBEDTLS_SSL_DTLS_TIMEOUT_DFL_MIN    1000
#define MBEDTLS_SSL_DTLS_TIMEOUT_DFL_MAX   60000

/**
 * \name SECTION: Module settings
 *
 * The configuration options you can set for this module are in this section.
 * Either change them in config.h or define them on the compiler command line.
 * \{
 */

#if !defined(MBEDTLS_SSL_DEFAULT_TICKET_LIFETIME)
#define MBEDTLS_SSL_DEFAULT_TICKET_LIFETIME     86400 /**< Lifetime of session tickets (if enabled) */
#endif

/*
 * Maximum fragment length in bytes,
 * determines the size of each of the two internal I/O buffers.
 *
 * Note: the RFC defines the default size of SSL / TLS messages. If you
 * change the value here, other clients / servers may not be able to
 * communicate with you anymore. Only change this value if you control
 * both sides of the connection and have it reduced at both sides, or
 * if you're using the Max Fragment Length extension and you know all your
 * peers are using it too!
 */
#if !defined(MBEDTLS_SSL_MAX_CONTENT_LEN)
#define MBEDTLS_SSL_MAX_CONTENT_LEN         16384   /**< Size of the input / output buffer */
#endif

#if !defined(MBEDTLS_SSL_IN_CONTENT_LEN)
#define MBEDTLS_SSL_IN_CONTENT_LEN MBEDTLS_SSL_MAX_CONTENT_LEN
#endif

#if !defined(MBEDTLS_SSL_OUT_CONTENT_LEN)
#define MBEDTLS_SSL_OUT_CONTENT_LEN MBEDTLS_SSL_MAX_CONTENT_LEN
#endif

/*
 * Maximum number of heap-allocated bytes for the purpose of
 * DTLS handshake message reassembly and future message buffering.
 */
#if !defined(MBEDTLS_SSL_DTLS_MAX_BUFFERING)
#define MBEDTLS_SSL_DTLS_MAX_BUFFERING 32768
#endif

/*
 * Maximum length of CIDs for incoming and outgoing messages.
 */
#if !defined(MBEDTLS_SSL_CID_IN_LEN_MAX)
#define MBEDTLS_SSL_CID_IN_LEN_MAX          32
#endif

#if !defined(MBEDTLS_SSL_CID_OUT_LEN_MAX)
#define MBEDTLS_SSL_CID_OUT_LEN_MAX         32
#endif

#if !defined(MBEDTLS_SSL_CID_PADDING_GRANULARITY)
#define MBEDTLS_SSL_CID_PADDING_GRANULARITY 16
#endif

#if !defined(MBEDTLS_SSL_TLS1_3_PADDING_GRANULARITY)
#define MBEDTLS_SSL_TLS1_3_PADDING_GRANULARITY 1
#endif

/** \} name SECTION: Module settings */

/*
 * Length of the verify data for secure renegotiation
 */
#if defined(MBEDTLS_SSL_PROTO_SSL3)
#define MBEDTLS_SSL_VERIFY_DATA_MAX_LEN 36
#else
#define MBEDTLS_SSL_VERIFY_DATA_MAX_LEN 12
#endif

/*
 * Signaling ciphersuite values (SCSV)
 */
#define MBEDTLS_SSL_EMPTY_RENEGOTIATION_INFO    0xFF   /**< renegotiation info ext */
#define MBEDTLS_SSL_FALLBACK_SCSV_VALUE         0x5600 /**< RFC 7507 section 2 */

/*
 * Supported Signature and Hash algorithms (For TLS 1.2)
 * RFC 5246 section 7.4.1.4.1
 */
#define MBEDTLS_SSL_HASH_NONE                0
#define MBEDTLS_SSL_HASH_MD5                 1
#define MBEDTLS_SSL_HASH_SHA1                2
#define MBEDTLS_SSL_HASH_SHA224              3
#define MBEDTLS_SSL_HASH_SHA256              4
#define MBEDTLS_SSL_HASH_SHA384              5
#define MBEDTLS_SSL_HASH_SHA512              6

#define MBEDTLS_SSL_SIG_ANON                 0
#define MBEDTLS_SSL_SIG_RSA                  1
#define MBEDTLS_SSL_SIG_ECDSA                3

/*
 * Client Certificate Types
 * RFC 5246 section 7.4.4 plus RFC 4492 section 5.5
 */
#define MBEDTLS_SSL_CERT_TYPE_RSA_SIGN       1
#define MBEDTLS_SSL_CERT_TYPE_ECDSA_SIGN    64

/*
 * Message, alert and handshake types
 */
#define MBEDTLS_SSL_MSG_CHANGE_CIPHER_SPEC     20
#define MBEDTLS_SSL_MSG_ALERT                  21
#define MBEDTLS_SSL_MSG_HANDSHAKE              22
#define MBEDTLS_SSL_MSG_APPLICATION_DATA       23
#define MBEDTLS_SSL_MSG_CID                    25

#define MBEDTLS_SSL_ALERT_LEVEL_WARNING         1
#define MBEDTLS_SSL_ALERT_LEVEL_FATAL           2

#define MBEDTLS_SSL_ALERT_MSG_CLOSE_NOTIFY           0  /* 0x00 */
#define MBEDTLS_SSL_ALERT_MSG_UNEXPECTED_MESSAGE    10  /* 0x0A */
#define MBEDTLS_SSL_ALERT_MSG_BAD_RECORD_MAC        20  /* 0x14 */
#define MBEDTLS_SSL_ALERT_MSG_DECRYPTION_FAILED     21  /* 0x15 */
#define MBEDTLS_SSL_ALERT_MSG_RECORD_OVERFLOW       22  /* 0x16 */
#define MBEDTLS_SSL_ALERT_MSG_DECOMPRESSION_FAILURE 30  /* 0x1E */
#define MBEDTLS_SSL_ALERT_MSG_HANDSHAKE_FAILURE     40  /* 0x28 */
#define MBEDTLS_SSL_ALERT_MSG_NO_CERT               41  /* 0x29 */
#define MBEDTLS_SSL_ALERT_MSG_BAD_CERT              42  /* 0x2A */
#define MBEDTLS_SSL_ALERT_MSG_UNSUPPORTED_CERT      43  /* 0x2B */
#define MBEDTLS_SSL_ALERT_MSG_CERT_REVOKED          44  /* 0x2C */
#define MBEDTLS_SSL_ALERT_MSG_CERT_EXPIRED          45  /* 0x2D */
#define MBEDTLS_SSL_ALERT_MSG_CERT_UNKNOWN          46  /* 0x2E */
#define MBEDTLS_SSL_ALERT_MSG_ILLEGAL_PARAMETER     47  /* 0x2F */
#define MBEDTLS_SSL_ALERT_MSG_UNKNOWN_CA            48  /* 0x30 */
#define MBEDTLS_SSL_ALERT_MSG_ACCESS_DENIED         49  /* 0x31 */
#define MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR          50  /* 0x32 */
#define MBEDTLS_SSL_ALERT_MSG_DECRYPT_ERROR         51  /* 0x33 */
#define MBEDTLS_SSL_ALERT_MSG_EXPORT_RESTRICTION    60  /* 0x3C */
#define MBEDTLS_SSL_ALERT_MSG_PROTOCOL_VERSION      70  /* 0x46 */
#define MBEDTLS_SSL_ALERT_MSG_INSUFFICIENT_SECURITY 71  /* 0x47 */
#define MBEDTLS_SSL_ALERT_MSG_INTERNAL_ERROR        80  /* 0x50 */
#define MBEDTLS_SSL_ALERT_MSG_INAPROPRIATE_FALLBACK 86  /* 0x56 */
#define MBEDTLS_SSL_ALERT_MSG_USER_CANCELED         90  /* 0x5A */
#define MBEDTLS_SSL_ALERT_MSG_NO_RENEGOTIATION     100  /* 0x64 */
#define MBEDTLS_SSL_ALERT_MSG_UNSUPPORTED_EXT      110  /* 0x6E */
#define MBEDTLS_SSL_ALERT_MSG_UNRECOGNIZED_NAME    112  /* 0x70 */
#define MBEDTLS_SSL_ALERT_MSG_UNKNOWN_PSK_IDENTITY 115  /* 0x73 */
#define MBEDTLS_SSL_ALERT_MSG_NO_APPLICATION_PROTOCOL 120 /* 0x78 */

#define MBEDTLS_SSL_HS_HELLO_REQUEST            0
#define MBEDTLS_SSL_HS_CLIENT_HELLO             1
#define MBEDTLS_SSL_HS_SERVER_HELLO             2
#define MBEDTLS_SSL_HS_HELLO_VERIFY_REQUEST     3
#define MBEDTLS_SSL_HS_NEW_SESSION_TICKET       4
#define MBEDTLS_SSL_HS_CERTIFICATE             11
#define MBEDTLS_SSL_HS_SERVER_KEY_EXCHANGE     12
#define MBEDTLS_SSL_HS_CERTIFICATE_REQUEST     13
#define MBEDTLS_SSL_HS_SERVER_HELLO_DONE       14
#define MBEDTLS_SSL_HS_CERTIFICATE_VERIFY      15
#define MBEDTLS_SSL_HS_CLIENT_KEY_EXCHANGE     16
#define MBEDTLS_SSL_HS_FINISHED                20

/*
 * TLS extensions
 */
#define MBEDTLS_TLS_EXT_SERVERNAME                   0
#define MBEDTLS_TLS_EXT_SERVERNAME_HOSTNAME          0

#define MBEDTLS_TLS_EXT_MAX_FRAGMENT_LENGTH          1

#define MBEDTLS_TLS_EXT_TRUNCATED_HMAC               4

#define MBEDTLS_TLS_EXT_SUPPORTED_ELLIPTIC_CURVES   10
#define MBEDTLS_TLS_EXT_SUPPORTED_POINT_FORMATS     11

#define MBEDTLS_TLS_EXT_SIG_ALG                     13

#define MBEDTLS_TLS_EXT_USE_SRTP                    14

#define MBEDTLS_TLS_EXT_ALPN                        16

#define MBEDTLS_TLS_EXT_ENCRYPT_THEN_MAC            22 /* 0x16 */
#define MBEDTLS_TLS_EXT_EXTENDED_MASTER_SECRET  0x0017 /* 23 */

#define MBEDTLS_TLS_EXT_SESSION_TICKET              35

/* The value of the CID extension is still TBD as of
 * draft-ietf-tls-dtls-connection-id-05
 * (https://tools.ietf.org/html/draft-ietf-tls-dtls-connection-id-05).
 *
 * A future minor revision of Mbed TLS may change the default value of
 * this option to match evolving standards and usage.
 */
#if !defined(MBEDTLS_TLS_EXT_CID)
#define MBEDTLS_TLS_EXT_CID                        254 /* TBD */
#endif

#define MBEDTLS_TLS_EXT_ECJPAKE_KKPP               256 /* experimental */

#define MBEDTLS_TLS_EXT_RENEGOTIATION_INFO      0xFF01

/*
 * Size defines
 */
#if !defined(MBEDTLS_PSK_MAX_LEN)
#define MBEDTLS_PSK_MAX_LEN            32 /* 256 bits */
#endif

/* Dummy type used only for its size */
union mbedtls_ssl_premaster_secret {
    unsigned char dummy; /* Make the union non-empty even with SSL disabled */
#if defined(MBEDTLS_KEY_EXCHANGE_RSA_ENABLED)
    unsigned char _pms_rsa[48];                         /* RFC 5246 8.1.1 */
#endif
#if defined(MBEDTLS_KEY_EXCHANGE_DHE_RSA_ENABLED)
    unsigned char _pms_dhm[MBEDTLS_MPI_MAX_SIZE];      /* RFC 5246 8.1.2 */
#endif
#if defined(MBEDTLS_KEY_EXCHANGE_ECDHE_RSA_ENABLED)    || \
    defined(MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA_ENABLED)  || \
    defined(MBEDTLS_KEY_EXCHANGE_ECDH_RSA_ENABLED)     || \
    defined(MBEDTLS_KEY_EXCHANGE_ECDH_ECDSA_ENABLED)
    unsigned char _pms_ecdh[MBEDTLS_ECP_MAX_BYTES];    /* RFC 4492 5.10 */
#endif
#if defined(MBEDTLS_KEY_EXCHANGE_PSK_ENABLED)
    unsigned char _pms_psk[4 + 2 * MBEDTLS_PSK_MAX_LEN];       /* RFC 4279 2 */
#endif
#if defined(MBEDTLS_KEY_EXCHANGE_DHE_PSK_ENABLED)
    unsigned char _pms_dhe_psk[4 + MBEDTLS_MPI_MAX_SIZE
                               + MBEDTLS_PSK_MAX_LEN];         /* RFC 4279 3 */
#endif
#if defined(MBEDTLS_KEY_EXCHANGE_RSA_PSK_ENABLED)
    unsigned char _pms_rsa_psk[52 + MBEDTLS_PSK_MAX_LEN];      /* RFC 4279 4 */
#endif
#if defined(MBEDTLS_KEY_EXCHANGE_ECDHE_PSK_ENABLED)
    unsigned char _pms_ecdhe_psk[4 + MBEDTLS_ECP_MAX_BYTES
                                 + MBEDTLS_PSK_MAX_LEN];       /* RFC 5489 2 */
#endif
#if defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
    unsigned char _pms_ecjpake[32];     /* Thread spec: SHA-256 output */
#endif
};

#define MBEDTLS_PREMASTER_SIZE     sizeof(union mbedtls_ssl_premaster_secret)

#ifdef __cplusplus
extern "C" {
#endif

/*
 * SSL state machine
 */
typedef enum {
    MBEDTLS_SSL_HELLO_REQUEST,
    MBEDTLS_SSL_CLIENT_HELLO,
    MBEDTLS_SSL_SERVER_HELLO,
    MBEDTLS_SSL_SERVER_CERTIFICATE,
    MBEDTLS_SSL_SERVER_KEY_EXCHANGE,
    MBEDTLS_SSL_CERTIFICATE_REQUEST,
    MBEDTLS_SSL_SERVER_HELLO_DONE,
    MBEDTLS_SSL_CLIENT_CERTIFICATE,
    MBEDTLS_SSL_CLIENT_KEY_EXCHANGE,
    MBEDTLS_SSL_CERTIFICATE_VERIFY,
    MBEDTLS_SSL_CLIENT_CHANGE_CIPHER_SPEC,
    MBEDTLS_SSL_CLIENT_FINISHED,
    MBEDTLS_SSL_SERVER_CHANGE_CIPHER_SPEC,
    MBEDTLS_SSL_SERVER_FINISHED,
    MBEDTLS_SSL_FLUSH_BUFFERS,
    MBEDTLS_SSL_HANDSHAKE_WRAPUP,
    MBEDTLS_SSL_HANDSHAKE_OVER,
    MBEDTLS_SSL_SERVER_NEW_SESSION_TICKET,
    MBEDTLS_SSL_SERVER_HELLO_VERIFY_REQUEST_SENT,
}
mbedtls_ssl_states;

/*
 * The tls_prf function types.
 */
typedef enum {
    MBEDTLS_SSL_TLS_PRF_NONE,
    MBEDTLS_SSL_TLS_PRF_SSL3,
    MBEDTLS_SSL_TLS_PRF_TLS1,
    MBEDTLS_SSL_TLS_PRF_SHA384,
    MBEDTLS_SSL_TLS_PRF_SHA256
}
mbedtls_tls_prf_types;
/**
 * \brief          Callback type: send data on the network.
 *
 * \note           That callback may be either blocking or non-blocking.
 *
 * \param ctx      Context for the send callback (typically a file descriptor)
 * \param buf      Buffer holding the data to send
 * \param len      Length of the data to send
 *
 * \return         The callback must return the number of bytes sent if any,
 *                 or a non-zero error code.
 *                 If performing non-blocking I/O, \c MBEDTLS_ERR_SSL_WANT_WRITE
 *                 must be returned when the operation would block.
 *
 * \note           The callback is allowed to send fewer bytes than requested.
 *                 It must always return the number of bytes actually sent.
 */
typedef int mbedtls_ssl_send_t(void *ctx,
                               const unsigned char *buf,
                               size_t len);

/**
 * \brief          Callback type: receive data from the network.
 *
 * \note           That callback may be either blocking or non-blocking.
 *
 * \param ctx      Context for the receive callback (typically a file
 *                 descriptor)
 * \param buf      Buffer to write the received data to
 * \param len      Length of the receive buffer
 *
 * \returns        If data has been received, the positive number of bytes received.
 * \returns        \c 0 if the connection has been closed.
 * \returns        If performing non-blocking I/O, \c MBEDTLS_ERR_SSL_WANT_READ
 *                 must be returned when the operation would block.
 * \returns        Another negative error code on other kinds of failures.
 *
 * \note           The callback may receive fewer bytes than the length of the
 *                 buffer. It must always return the number of bytes actually
 *                 received and written to the buffer.
 */
typedef int mbedtls_ssl_recv_t(void *ctx,
                               unsigned char *buf,
                               size_t len);

/**
 * \brief          Callback type: receive data from the network, with timeout
 *
 * \note           That callback must block until data is received, or the
 *                 timeout delay expires, or the operation is interrupted by a
 *                 signal.
 *
 * \param ctx      Context for the receive callback (typically a file descriptor)
 * \param buf      Buffer to write the received data to
 * \param len      Length of the receive buffer
 * \param timeout  Maximum number of milliseconds to wait for data
 *                 0 means no timeout (potentially waiting forever)
 *
 * \return         The callback must return the number of bytes received,
 *                 or a non-zero error code:
 *                 \c MBEDTLS_ERR_SSL_TIMEOUT if the operation timed out,
 *                 \c MBEDTLS_ERR_SSL_WANT_READ if interrupted by a signal.
 *
 * \note           The callback may receive fewer bytes than the length of the
 *                 buffer. It must always return the number of bytes actually
 *                 received and written to the buffer.
 */
typedef int mbedtls_ssl_recv_timeout_t(void *ctx,
                                       unsigned char *buf,
                                       size_t len,
                                       uint32_t timeout);
/**
 * \brief          Callback type: set a pair of timers/delays to watch
 *
 * \param ctx      Context pointer
 * \param int_ms   Intermediate delay in milliseconds
 * \param fin_ms   Final delay in milliseconds
 *                 0 cancels the current timer.
 *
 * \note           This callback must at least store the necessary information
 *                 for the associated \c mbedtls_ssl_get_timer_t callback to
 *                 return correct information.
 *
 * \note           If using an event-driven style of programming, an event must
 *                 be generated when the final delay is passed. The event must
 *                 cause a call to \c mbedtls_ssl_handshake() with the proper
 *                 SSL context to be scheduled. Care must be taken to ensure
 *                 that at most one such call happens at a time.
 *
 * \note           Only one timer at a time must be running. Calling this
 *                 function while a timer is running must cancel it. Cancelled
 *                 timers must not generate any event.
 */
typedef void mbedtls_ssl_set_timer_t(void *ctx,
                                     uint32_t int_ms,
                                     uint32_t fin_ms);

/**
 * \brief          Callback type: get status of timers/delays
 *
 * \param ctx      Context pointer
 *
 * \return         This callback must return:
 *                 -1 if cancelled (fin_ms == 0),
 *                  0 if none of the delays have passed,
 *                  1 if only the intermediate delay has passed,
 *                  2 if the final delay has passed.
 */
typedef int mbedtls_ssl_get_timer_t(void *ctx);

/* Defined below */
typedef struct mbedtls_ssl_session mbedtls_ssl_session;
typedef struct mbedtls_ssl_context mbedtls_ssl_context;
typedef struct mbedtls_ssl_config  mbedtls_ssl_config;

/* Defined in ssl_internal.h */
typedef struct mbedtls_ssl_transform mbedtls_ssl_transform;
typedef struct mbedtls_ssl_handshake_params mbedtls_ssl_handshake_params;
typedef struct mbedtls_ssl_sig_hash_set_t mbedtls_ssl_sig_hash_set_t;
#if defined(MBEDTLS_X509_CRT_PARSE_C)
typedef struct mbedtls_ssl_key_cert mbedtls_ssl_key_cert;
#endif
#if defined(MBEDTLS_SSL_PROTO_DTLS)
typedef struct mbedtls_ssl_flight_item mbedtls_ssl_flight_item;
#endif

#if defined(MBEDTLS_SSL_ASYNC_PRIVATE)
#if defined(MBEDTLS_X509_CRT_PARSE_C)
/**
 * \brief           Callback type: start external signature operation.
 *
 *                  This callback is called during an SSL handshake to start
 *                  a signature decryption operation using an
 *                  external processor. The parameter \p cert contains
 *                  the public key; it is up to the callback function to
 *                  determine how to access the associated private key.
 *
 *                  This function typically sends or enqueues a request, and
 *                  does not wait for the operation to complete. This allows
 *                  the handshake step to be non-blocking.
 *
 *                  The parameters \p ssl and \p cert are guaranteed to remain
 *                  valid throughout the handshake. On the other hand, this
 *                  function must save the contents of \p hash if the value
 *                  is needed for later processing, because the \p hash buffer
 *                  is no longer valid after this function returns.
 *
 *                  This function may call mbedtls_ssl_set_async_operation_data()
 *                  to store an operation context for later retrieval
 *                  by the resume or cancel callback.
 *
 * \note            For RSA signatures, this function must produce output
 *                  that is consistent with PKCS#1 v1.5 in the same way as
 *                  mbedtls_rsa_pkcs1_sign(). Before the private key operation,
 *                  apply the padding steps described in RFC 8017, section 9.2
 *                  "EMSA-PKCS1-v1_5" as follows.
 *                  - If \p md_alg is #MBEDTLS_MD_NONE, apply the PKCS#1 v1.5
 *                    encoding, treating \p hash as the DigestInfo to be
 *                    padded. In other words, apply EMSA-PKCS1-v1_5 starting
 *                    from step 3, with `T = hash` and `tLen = hash_len`.
 *                  - If `md_alg != MBEDTLS_MD_NONE`, apply the PKCS#1 v1.5
 *                    encoding, treating \p hash as the hash to be encoded and
 *                    padded. In other words, apply EMSA-PKCS1-v1_5 starting
 *                    from step 2, with `digestAlgorithm` obtained by calling
 *                    mbedtls_oid_get_oid_by_md() on \p md_alg.
 *
 * \note            For ECDSA signatures, the output format is the DER encoding
 *                  `Ecdsa-Sig-Value` defined in
 *                  [RFC 4492 section 5.4](https://tools.ietf.org/html/rfc4492#section-5.4).
 *
 * \param ssl             The SSL connection instance. It should not be
 *                        modified other than via
 *                        mbedtls_ssl_set_async_operation_data().
 * \param cert            Certificate containing the public key.
 *                        In simple cases, this is one of the pointers passed to
 *                        mbedtls_ssl_conf_own_cert() when configuring the SSL
 *                        connection. However, if other callbacks are used, this
 *                        property may not hold. For example, if an SNI callback
 *                        is registered with mbedtls_ssl_conf_sni(), then
 *                        this callback determines what certificate is used.
 * \param md_alg          Hash algorithm.
 * \param hash            Buffer containing the hash. This buffer is
 *                        no longer valid when the function returns.
 * \param hash_len        Size of the \c hash buffer in bytes.
 *
 * \return          0 if the operation was started successfully and the SSL
 *                  stack should call the resume callback immediately.
 * \return          #MBEDTLS_ERR_SSL_ASYNC_IN_PROGRESS if the operation
 *                  was started successfully and the SSL stack should return
 *                  immediately without calling the resume callback yet.
 * \return          #MBEDTLS_ERR_SSL_HW_ACCEL_FALLTHROUGH if the external
 *                  processor does not support this key. The SSL stack will
 *                  use the private key object instead.
 * \return          Any other error indicates a fatal failure and is
 *                  propagated up the call chain. The callback should
 *                  use \c MBEDTLS_ERR_PK_xxx error codes, and <b>must not</b>
 *                  use \c MBEDTLS_ERR_SSL_xxx error codes except as
 *                  directed in the documentation of this callback.
 */
typedef int mbedtls_ssl_async_sign_t(mbedtls_ssl_context *ssl,
                                     mbedtls_x509_crt *cert,
                                     mbedtls_md_type_t md_alg,
                                     const unsigned char *hash,
                                     size_t hash_len);

/**
 * \brief           Callback type: start external decryption operation.
 *
 *                  This callback is called during an SSL handshake to start
 *                  an RSA decryption operation using an
 *                  external processor. The parameter \p cert contains
 *                  the public key; it is up to the callback function to
 *                  determine how to access the associated private key.
 *
 *                  This function typically sends or enqueues a request, and
 *                  does not wait for the operation to complete. This allows
 *                  the handshake step to be non-blocking.
 *
 *                  The parameters \p ssl and \p cert are guaranteed to remain
 *                  valid throughout the handshake. On the other hand, this
 *                  function must save the contents of \p input if the value
 *                  is needed for later processing, because the \p input buffer
 *                  is no longer valid after this function returns.
 *
 *                  This function may call mbedtls_ssl_set_async_operation_data()
 *                  to store an operation context for later retrieval
 *                  by the resume or cancel callback.
 *
 * \warning         RSA decryption as used in TLS is subject to a potential
 *                  timing side channel attack first discovered by Bleichenbacher
 *                  in 1998. This attack can be remotely exploitable
 *                  in practice. To avoid this attack, you must ensure that
 *                  if the callback performs an RSA decryption, the time it
 *                  takes to execute and return the result does not depend
 *                  on whether the RSA decryption succeeded or reported
 *                  invalid padding.
 *
 * \param ssl             The SSL connection instance. It should not be
 *                        modified other than via
 *                        mbedtls_ssl_set_async_operation_data().
 * \param cert            Certificate containing the public key.
 *                        In simple cases, this is one of the pointers passed to
 *                        mbedtls_ssl_conf_own_cert() when configuring the SSL
 *                        connection. However, if other callbacks are used, this
 *                        property may not hold. For example, if an SNI callback
 *                        is registered with mbedtls_ssl_conf_sni(), then
 *                        this callback determines what certificate is used.
 * \param input           Buffer containing the input ciphertext. This buffer
 *                        is no longer valid when the function returns.
 * \param input_len       Size of the \p input buffer in bytes.
 *
 * \return          0 if the operation was started successfully and the SSL
 *                  stack should call the resume callback immediately.
 * \return          #MBEDTLS_ERR_SSL_ASYNC_IN_PROGRESS if the operation
 *                  was started successfully and the SSL stack should return
 *                  immediately without calling the resume callback yet.
 * \return          #MBEDTLS_ERR_SSL_HW_ACCEL_FALLTHROUGH if the external
 *                  processor does not support this key. The SSL stack will
 *                  use the private key object instead.
 * \return          Any other error indicates a fatal failure and is
 *                  propagated up the call chain. The callback should
 *                  use \c MBEDTLS_ERR_PK_xxx error codes, and <b>must not</b>
 *                  use \c MBEDTLS_ERR_SSL_xxx error codes except as
 *                  directed in the documentation of this callback.
 */
typedef int mbedtls_ssl_async_decrypt_t(mbedtls_ssl_context *ssl,
                                        mbedtls_x509_crt *cert,
                                        const unsigned char *input,
                                        size_t input_len);
#endif /* MBEDTLS_X509_CRT_PARSE_C */

/**
 * \brief           Callback type: resume external operation.
 *
 *                  This callback is called during an SSL handshake to resume
 *                  an external operation started by the
 *                  ::mbedtls_ssl_async_sign_t or
 *                  ::mbedtls_ssl_async_decrypt_t callback.
 *
 *                  This function typically checks the status of a pending
 *                  request or causes the request queue to make progress, and
 *                  does not wait for the operation to complete. This allows
 *                  the handshake step to be non-blocking.
 *
 *                  This function may call mbedtls_ssl_get_async_operation_data()
 *                  to retrieve an operation context set by the start callback.
 *                  It may call mbedtls_ssl_set_async_operation_data() to modify
 *                  this context.
 *
 *                  Note that when this function returns a status other than
 *                  #MBEDTLS_ERR_SSL_ASYNC_IN_PROGRESS, it must free any
 *                  resources associated with the operation.
 *
 * \param ssl             The SSL connection instance. It should not be
 *                        modified other than via
 *                        mbedtls_ssl_set_async_operation_data().
 * \param output          Buffer containing the output (signature or decrypted
 *                        data) on success.
 * \param output_len      On success, number of bytes written to \p output.
 * \param output_size     Size of the \p output buffer in bytes.
 *
 * \return          0 if output of the operation is available in the
 *                  \p output buffer.
 * \return          #MBEDTLS_ERR_SSL_ASYNC_IN_PROGRESS if the operation
 *                  is still in progress. Subsequent requests for progress
 *                  on the SSL connection will call the resume callback
 *                  again.
 * \return          Any other error means that the operation is aborted.
 *                  The SSL handshake is aborted. The callback should
 *                  use \c MBEDTLS_ERR_PK_xxx error codes, and <b>must not</b>
 *                  use \c MBEDTLS_ERR_SSL_xxx error codes except as
 *                  directed in the documentation of this callback.
 */
typedef int mbedtls_ssl_async_resume_t(mbedtls_ssl_context *ssl,
                                       unsigned char *output,
                                       size_t *output_len,
                                       size_t output_size);

/**
 * \brief           Callback type: cancel external operation.
 *
 *                  This callback is called if an SSL connection is closed
 *                  while an asynchronous operation is in progress. Note that
 *                  this callback is not called if the
 *                  ::mbedtls_ssl_async_resume_t callback has run and has
 *                  returned a value other than
 *                  #MBEDTLS_ERR_SSL_ASYNC_IN_PROGRESS, since in that case
 *                  the asynchronous operation has already completed.
 *
 *                  This function may call mbedtls_ssl_get_async_operation_data()
 *                  to retrieve an operation context set by the start callback.
 *
 * \param ssl             The SSL connection instance. It should not be
 *                        modified.
 */
typedef void mbedtls_ssl_async_cancel_t(mbedtls_ssl_context *ssl);
#endif /* MBEDTLS_SSL_ASYNC_PRIVATE */

#if defined(MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED) &&        \
    !defined(MBEDTLS_SSL_KEEP_PEER_CERTIFICATE)
#define MBEDTLS_SSL_PEER_CERT_DIGEST_MAX_LEN  48
#if defined(MBEDTLS_SHA256_C)
#define MBEDTLS_SSL_PEER_CERT_DIGEST_DFL_TYPE MBEDTLS_MD_SHA256
#define MBEDTLS_SSL_PEER_CERT_DIGEST_DFL_LEN  32
#elif defined(MBEDTLS_SHA512_C)
#define MBEDTLS_SSL_PEER_CERT_DIGEST_DFL_TYPE MBEDTLS_MD_SHA384
#define MBEDTLS_SSL_PEER_CERT_DIGEST_DFL_LEN  48
#elif defined(MBEDTLS_SHA1_C)
#define MBEDTLS_SSL_PEER_CERT_DIGEST_DFL_TYPE MBEDTLS_MD_SHA1
#define MBEDTLS_SSL_PEER_CERT_DIGEST_DFL_LEN  20
#else
/* This is already checked in check_config.h, but be sure. */
#error "Bad configuration - need SHA-1, SHA-256 or SHA-512 enabled to compute digest of peer CRT."
#endif
#endif /* MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED &&
          !MBEDTLS_SSL_KEEP_PEER_CERTIFICATE */

#if defined(MBEDTLS_SSL_DTLS_SRTP)

#define MBEDTLS_TLS_SRTP_MAX_MKI_LENGTH             255
#define MBEDTLS_TLS_SRTP_MAX_PROFILE_LIST_LENGTH    4
/*
 * For code readability use a typedef for DTLS-SRTP profiles
 *
 * Use_srtp extension protection profiles values as defined in
 * http://www.iana.org/assignments/srtp-protection/srtp-protection.xhtml
 *
 * Reminder: if this list is expanded mbedtls_ssl_check_srtp_profile_value
 * must be updated too.
 */
#define MBEDTLS_TLS_SRTP_AES128_CM_HMAC_SHA1_80     ((uint16_t) 0x0001)
#define MBEDTLS_TLS_SRTP_AES128_CM_HMAC_SHA1_32     ((uint16_t) 0x0002)
#define MBEDTLS_TLS_SRTP_NULL_HMAC_SHA1_80          ((uint16_t) 0x0005)
#define MBEDTLS_TLS_SRTP_NULL_HMAC_SHA1_32          ((uint16_t) 0x0006)
/* This one is not iana defined, but for code readability. */
#define MBEDTLS_TLS_SRTP_UNSET                      ((uint16_t) 0x0000)

typedef uint16_t mbedtls_ssl_srtp_profile;

typedef struct mbedtls_dtls_srtp_info_t {
    /*! The SRTP profile that was negotiated. */
    mbedtls_ssl_srtp_profile chosen_dtls_srtp_profile;
    /*! The length of mki_value. */
    uint16_t mki_len;
    /*! The mki_value used, with max size of 256 bytes. */
    unsigned char mki_value[MBEDTLS_TLS_SRTP_MAX_MKI_LENGTH];
}
mbedtls_dtls_srtp_info;

#endif /* MBEDTLS_SSL_DTLS_SRTP */

/*
 * This structure is used for storing current session data.
 *
 * Note: when changing this definition, we need to check and update:
 *  - in tests/suites/test_suite_ssl.function:
 *      ssl_populate_session() and ssl_serialize_session_save_load()
 *  - in library/ssl_tls.c:
 *      mbedtls_ssl_session_init() and mbedtls_ssl_session_free()
 *      mbedtls_ssl_session_save() and ssl_session_load()
 *      ssl_session_copy()
 */
struct mbedtls_ssl_session {
#if defined(MBEDTLS_SSL_MAX_FRAGMENT_LENGTH)
    unsigned char mfl_code;     /*!< MaxFragmentLength negotiated by peer */
#endif /* MBEDTLS_SSL_MAX_FRAGMENT_LENGTH */

#if defined(MBEDTLS_HAVE_TIME)
    mbedtls_time_t start;       /*!< starting time      */
#endif
    int ciphersuite;            /*!< chosen ciphersuite */
    int compression;            /*!< chosen compression */
    size_t id_len;              /*!< session id length  */
    unsigned char id[32];       /*!< session identifier */
    unsigned char master[48];   /*!< the master secret  */

#if defined(MBEDTLS_X509_CRT_PARSE_C)
#if defined(MBEDTLS_SSL_KEEP_PEER_CERTIFICATE)
    mbedtls_x509_crt *peer_cert;       /*!< peer X.509 cert chain */
#else /* MBEDTLS_SSL_KEEP_PEER_CERTIFICATE */
    /*! The digest of the peer's end-CRT. This must be kept to detect CRT
     *  changes during renegotiation, mitigating the triple handshake attack. */
    unsigned char *peer_cert_digest;
    size_t peer_cert_digest_len;
    mbedtls_md_type_t peer_cert_digest_type;
#endif /* !MBEDTLS_SSL_KEEP_PEER_CERTIFICATE */
#endif /* MBEDTLS_X509_CRT_PARSE_C */
    uint32_t verify_result;          /*!<  verification result     */

#if defined(MBEDTLS_SSL_SESSION_TICKETS) && defined(MBEDTLS_SSL_CLI_C)
    unsigned char *ticket;      /*!< RFC 5077 session ticket */
    size_t ticket_len;          /*!< session ticket length   */
    uint32_t ticket_lifetime;   /*!< ticket lifetime hint    */
#endif /* MBEDTLS_SSL_SESSION_TICKETS && MBEDTLS_SSL_CLI_C */

#if defined(MBEDTLS_SSL_TRUNCATED_HMAC)
    int trunc_hmac;             /*!< flag for truncated hmac activation   */
#endif /* MBEDTLS_SSL_TRUNCATED_HMAC */

#if defined(MBEDTLS_SSL_ENCRYPT_THEN_MAC)
    int encrypt_then_mac;       /*!< flag for EtM activation                */
#endif
};

/**
 * SSL/TLS configuration to be shared between mbedtls_ssl_context structures.
 */
struct mbedtls_ssl_config {
    /* Group items by size and reorder them to maximize usage of immediate offset access.    */

    /*
     * Numerical settings (char)
     */

    unsigned char max_major_ver;    /*!< max. major version used            */
    unsigned char max_minor_ver;    /*!< max. minor version used            */
    unsigned char min_major_ver;    /*!< min. major version used            */
    unsigned char min_minor_ver;    /*!< min. minor version used            */

    /*
     * Flags (could be bit-fields to save RAM, but separate bytes make
     * the code smaller on architectures with an instruction for direct
     * byte access).
     */

    uint8_t endpoint /*bool*/;      /*!< 0: client, 1: server               */
    uint8_t transport /*bool*/;     /*!< stream (TLS) or datagram (DTLS)    */
    uint8_t authmode /*2 bits*/;    /*!< MBEDTLS_SSL_VERIFY_XXX             */
    /* needed even with renego disabled for LEGACY_BREAK_HANDSHAKE          */
    uint8_t allow_legacy_renegotiation /*2 bits*/; /*!< MBEDTLS_LEGACY_XXX  */
#if defined(MBEDTLS_ARC4_C)
    uint8_t arc4_disabled /*bool*/; /*!< blacklist RC4 ciphersuites?        */
#endif
#if defined(MBEDTLS_SSL_MAX_FRAGMENT_LENGTH)
    uint8_t mfl_code /*3 bits*/;    /*!< desired fragment length            */
#endif
#if defined(MBEDTLS_SSL_ENCRYPT_THEN_MAC)
    uint8_t encrypt_then_mac /*bool*/;  /*!< negotiate encrypt-then-mac?    */
#endif
#if defined(MBEDTLS_SSL_EXTENDED_MASTER_SECRET)
    uint8_t extended_ms /*bool*/;   /*!< negotiate extended master secret?  */
#endif
#if defined(MBEDTLS_SSL_DTLS_ANTI_REPLAY)
    uint8_t anti_replay /*bool*/;   /*!< detect and prevent replay?         */
#endif
#if defined(MBEDTLS_SSL_CBC_RECORD_SPLITTING)
    uint8_t cbc_record_splitting /*bool*/;  /*!< do cbc record splitting    */
#endif
#if defined(MBEDTLS_SSL_RENEGOTIATION)
    uint8_t disable_renegotiation /*bool*/; /*!< disable renegotiation?     */
#endif
#if defined(MBEDTLS_SSL_TRUNCATED_HMAC)
    uint8_t trunc_hmac /*bool*/;    /*!< negotiate truncated hmac?          */
#endif
#if defined(MBEDTLS_SSL_SESSION_TICKETS)
    uint8_t session_tickets /*bool*/;   /*!< use session tickets?           */
#endif
#if defined(MBEDTLS_SSL_FALLBACK_SCSV) && defined(MBEDTLS_SSL_CLI_C)
    uint8_t fallback /*bool*/;      /*!< is this a fallback?                */
#endif
#if defined(MBEDTLS_SSL_SRV_C)
    uint8_t cert_req_ca_list /*bool*/;  /*!< enable sending CA list in
                                           Certificate Request messages?     */
#endif
#if defined(MBEDTLS_SSL_DTLS_CONNECTION_ID)
    uint8_t ignore_unexpected_cid /*bool*/; /*!< Determines whether DTLS
                                             *   record with unexpected CID
                                             *   should lead to failure.    */
#endif /* MBEDTLS_SSL_DTLS_CONNECTION_ID */
#if defined(MBEDTLS_SSL_DTLS_SRTP)
    uint8_t dtls_srtp_mki_support /*bool*/; /*!< support having mki_value
                                                 in the use_srtp extension? */
#endif

    /*
     * Numerical settings (int or larger)
     */

    uint32_t read_timeout;          /*!< timeout for mbedtls_ssl_read (ms)  */

#if defined(MBEDTLS_SSL_PROTO_DTLS)
    uint32_t hs_timeout_min;        /*!< initial value of the handshake
                                         retransmission timeout (ms)        */
    uint32_t hs_timeout_max;        /*!< maximum value of the handshake
                                         retransmission timeout (ms)        */
#endif

#if defined(MBEDTLS_SSL_RENEGOTIATION)
    int renego_max_records;         /*!< grace period for renegotiation     */
    unsigned char renego_period[8]; /*!< value of the record counters
                                         that triggers renegotiation        */
#endif

#if defined(MBEDTLS_SSL_DTLS_BADMAC_LIMIT)
    unsigned int badmac_limit;      /*!< limit of records with a bad MAC    */
#endif

#if defined(MBEDTLS_DHM_C) && defined(MBEDTLS_SSL_CLI_C)
    unsigned int dhm_min_bitlen;    /*!< min. bit length of the DHM prime   */
#endif

    /*
     * Pointers
     */

    const int *ciphersuite_list[4]; /*!< allowed ciphersuites per version   */

    /** Callback for printing debug output                                  */
    void (*f_dbg)(void *, int, const char *, int, const char *);
    void *p_dbg;                    /*!< context for the debug function     */

    /** Callback for getting (pseudo-)random numbers                        */
    int  (*f_rng)(void *, unsigned char *, size_t);
    void *p_rng;                    /*!< context for the RNG function       */

    /** Callback to retrieve a session from the cache                       */
    int (*f_get_cache)(void *, mbedtls_ssl_session *);
    /** Callback to store a session into the cache                          */
    int (*f_set_cache)(void *, const mbedtls_ssl_session *);
    void *p_cache;                  /*!< context for cache callbacks        */

#if defined(MBEDTLS_SSL_SERVER_NAME_INDICATION)
    /** Callback for setting cert according to SNI extension                */
    int (*f_sni)(void *, mbedtls_ssl_context *, const unsigned char *, size_t);
    void *p_sni;                    /*!< context for SNI callback           */
#endif

#if defined(MBEDTLS_X509_CRT_PARSE_C)
    /** Callback to customize X.509 certificate chain verification          */
    int (*f_vrfy)(void *, mbedtls_x509_crt *, int, uint32_t *);
    void *p_vrfy;                   /*!< context for X.509 verify calllback */
#endif

#if defined(MBEDTLS_KEY_EXCHANGE_SOME_PSK_ENABLED)
    /** Callback to retrieve PSK key from identity                          */
    int (*f_psk)(void *, mbedtls_ssl_context *, const unsigned char *, size_t);
    void *p_psk;                    /*!< context for PSK callback           */
#endif

#if defined(MBEDTLS_SSL_DTLS_HELLO_VERIFY) && defined(MBEDTLS_SSL_SRV_C)
    /** Callback to create & write a cookie for ClientHello verification    */
    int (*f_cookie_write)(void *, unsigned char **, unsigned char *,
                          const unsigned char *, size_t);
    /** Callback to verify validity of a ClientHello cookie                 */
    int (*f_cookie_check)(void *, const unsigned char *, size_t,
                          const unsigned char *, size_t);
    void *p_cookie;                 /*!< context for the cookie callbacks   */
#endif

#if defined(MBEDTLS_SSL_SESSION_TICKETS) && defined(MBEDTLS_SSL_SRV_C)
    /** Callback to create & write a session ticket                         */
    int (*f_ticket_write)(void *, const mbedtls_ssl_session *,
                          unsigned char *, const unsigned char *, size_t *, uint32_t *);
    /** Callback to parse a session ticket into a session structure         */
    int (*f_ticket_parse)(void *, mbedtls_ssl_session *, unsigned char *, size_t);
    void *p_ticket;                 /*!< context for the ticket callbacks   */
#endif /* MBEDTLS_SSL_SESSION_TICKETS && MBEDTLS_SSL_SRV_C */

#if defined(MBEDTLS_SSL_EXPORT_KEYS)
    /** Callback to export key block and master secret                      */
    int (*f_export_keys)(void *, const unsigned char *,
                         const unsigned char *, size_t, size_t, size_t);
    /** Callback to export key block, master secret,
     *  tls_prf and random bytes. Should replace f_export_keys    */
    int (*f_export_keys_ext)(void *, const unsigned char *,
                             const unsigned char *, size_t, size_t, size_t,
                             const unsigned char[32], const unsigned char[32],
                             mbedtls_tls_prf_types);
    void *p_export_keys;            /*!< context for key export callback    */
#endif

#if defined(MBEDTLS_SSL_DTLS_CONNECTION_ID)
    size_t cid_len; /*!< The length of CIDs for incoming DTLS records.      */
#endif /* MBEDTLS_SSL_DTLS_CONNECTION_ID */

#if defined(MBEDTLS_X509_CRT_PARSE_C)
    const mbedtls_x509_crt_profile *cert_profile; /*!< verification profile */
    mbedtls_ssl_key_cert *key_cert; /*!< own certificate/key pair(s)        */
    mbedtls_x509_crt *ca_chain;     /*!< trusted CAs                        */
    mbedtls_x509_crl *ca_crl;       /*!< trusted CAs CRLs                   */
#if defined(MBEDTLS_X509_TRUSTED_CERTIFICATE_CALLBACK)
    mbedtls_x509_crt_ca_cb_t f_ca_cb;
    void *p_ca_cb;
#endif /* MBEDTLS_X509_TRUSTED_CERTIFICATE_CALLBACK */
#endif /* MBEDTLS_X509_CRT_PARSE_C */

#if defined(MBEDTLS_SSL_ASYNC_PRIVATE)
#if defined(MBEDTLS_X509_CRT_PARSE_C)
    mbedtls_ssl_async_sign_t *f_async_sign_start; /*!< start asynchronous signature operation */
    mbedtls_ssl_async_decrypt_t *f_async_decrypt_start; /*!< start asynchronous decryption operation */
#endif /* MBEDTLS_X509_CRT_PARSE_C */
    mbedtls_ssl_async_resume_t *f_async_resume; /*!< resume asynchronous operation */
    mbedtls_ssl_async_cancel_t *f_async_cancel; /*!< cancel asynchronous operation */
    void *p_async_config_data; /*!< Configuration data set by mbedtls_ssl_conf_async_private_cb(). */
#endif /* MBEDTLS_SSL_ASYNC_PRIVATE */

#if defined(MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED)
    const int *sig_hashes;          /*!< allowed signature hashes           */
#endif

#if defined(MBEDTLS_ECP_C)
    const mbedtls_ecp_group_id *curve_list; /*!< allowed curves             */
#endif

#if defined(MBEDTLS_DHM_C)
    mbedtls_mpi dhm_P;              /*!< prime modulus for DHM              */
    mbedtls_mpi dhm_G;              /*!< generator for DHM                  */
#endif

#if defined(MBEDTLS_KEY_EXCHANGE_SOME_PSK_ENABLED)

#if defined(MBEDTLS_USE_PSA_CRYPTO)
    psa_key_id_t psk_opaque; /*!< PSA key slot holding opaque PSK. This field
                              *   should only be set via
                              *   mbedtls_ssl_conf_psk_opaque().
                              *   If either no PSK or a raw PSK have been
                              *   configured, this has value \c 0.
                              */
#endif /* MBEDTLS_USE_PSA_CRYPTO */

    unsigned char *psk;      /*!< The raw pre-shared key. This field should
                              *   only be set via mbedtls_ssl_conf_psk().
                              *   If either no PSK or an opaque PSK
                              *   have been configured, this has value NULL. */
    size_t         psk_len;  /*!< The length of the raw pre-shared key.
                              *   This field should only be set via
                              *   mbedtls_ssl_conf_psk().
                              *   Its value is non-zero if and only if
                              *   \c psk is not \c NULL. */

    unsigned char *psk_identity;    /*!< The PSK identity for PSK negotiation.
                                     *   This field should only be set via
                                     *   mbedtls_ssl_conf_psk().
                                     *   This is set if and only if either
                                     *   \c psk or \c psk_opaque are set. */
    size_t         psk_identity_len;/*!< The length of PSK identity.
                                     *   This field should only be set via
                                     *   mbedtls_ssl_conf_psk().
                                     *   Its value is non-zero if and only if
                                     *   \c psk is not \c NULL or \c psk_opaque
                                     *   is not \c 0. */
#endif /* MBEDTLS_KEY_EXCHANGE_SOME_PSK_ENABLED */

#if defined(MBEDTLS_SSL_ALPN)
    const char **alpn_list;         /*!< ordered list of protocols          */
#endif

#if defined(MBEDTLS_SSL_DTLS_SRTP)
    /*! ordered list of supported srtp profile */
    const mbedtls_ssl_srtp_profile *dtls_srtp_profile_list;
    /*! number of supported profiles */
    size_t dtls_srtp_profile_list_len;
#endif /* MBEDTLS_SSL_DTLS_SRTP */
};

struct mbedtls_ssl_context {
    const mbedtls_ssl_config *conf; /*!< configuration information          */

    /*
     * Miscellaneous
     */
    int state;                  /*!< SSL handshake: current state     */
#if defined(MBEDTLS_SSL_RENEGOTIATION)
    int renego_status;          /*!< Initial, in progress, pending?   */
    int renego_records_seen;    /*!< Records since renego request, or with DTLS,
                                   number of retransmissions of request if
                                   renego_max_records is < 0           */
#endif /* MBEDTLS_SSL_RENEGOTIATION */

    int major_ver;              /*!< equal to  MBEDTLS_SSL_MAJOR_VERSION_3    */
    int minor_ver;              /*!< either 0 (SSL3) or 1 (TLS1.0)    */

#if defined(MBEDTLS_SSL_DTLS_BADMAC_LIMIT)
    unsigned badmac_seen;       /*!< records with a bad MAC received    */
#endif /* MBEDTLS_SSL_DTLS_BADMAC_LIMIT */

#if defined(MBEDTLS_X509_CRT_PARSE_C)
    /** Callback to customize X.509 certificate chain verification          */
    int (*f_vrfy)(void *, mbedtls_x509_crt *, int, uint32_t *);
    void *p_vrfy;                   /*!< context for X.509 verify callback */
#endif

    mbedtls_ssl_send_t *f_send; /*!< Callback for network send */
    mbedtls_ssl_recv_t *f_recv; /*!< Callback for network receive */
    mbedtls_ssl_recv_timeout_t *f_recv_timeout;
    /*!< Callback for network receive with timeout */

    void *p_bio;                /*!< context for I/O operations   */

    /*
     * Session layer
     */
    mbedtls_ssl_session *session_in;            /*!<  current session data (in)   */
    mbedtls_ssl_session *session_out;           /*!<  current session data (out)  */
    mbedtls_ssl_session *session;               /*!<  negotiated session data     */
    mbedtls_ssl_session *session_negotiate;     /*!<  session data in negotiation */

    mbedtls_ssl_handshake_params *handshake;    /*!<  params required only during
                                                   the handshake process        */

    /*
     * Record layer transformations
     */
    mbedtls_ssl_transform *transform_in;        /*!<  current transform params (in)   */
    mbedtls_ssl_transform *transform_out;       /*!<  current transform params (in)   */
    mbedtls_ssl_transform *transform;           /*!<  negotiated transform params     */
    mbedtls_ssl_transform *transform_negotiate; /*!<  transform params in negotiation */

    /*
     * Timers
     */
    void *p_timer;              /*!< context for the timer callbacks */

    mbedtls_ssl_set_timer_t *f_set_timer;       /*!< set timer callback */
    mbedtls_ssl_get_timer_t *f_get_timer;       /*!< get timer callback */

    /*
     * Record layer (incoming data)
     */
    unsigned char *in_buf;      /*!< input buffer                     */
    unsigned char *in_ctr;      /*!< 64-bit incoming message counter
                                     TLS: maintained by us
                                     DTLS: read from peer             */
    unsigned char *in_hdr;      /*!< start of record header           */
#if defined(MBEDTLS_SSL_DTLS_CONNECTION_ID)
    unsigned char *in_cid;      /*!< The start of the CID;
                                 *   (the end is marked by in_len).   */
#endif /* MBEDTLS_SSL_DTLS_CONNECTION_ID */
    unsigned char *in_len;      /*!< two-bytes message length field   */
    unsigned char *in_iv;       /*!< ivlen-byte IV                    */
    unsigned char *in_msg;      /*!< message contents (in_iv+ivlen)   */
    unsigned char *in_offt;     /*!< read offset in application data  */

    int in_msgtype;             /*!< record header: message type      */
    size_t in_msglen;           /*!< record header: message length    */
    size_t in_left;             /*!< amount of data read so far       */
#if defined(MBEDTLS_SSL_VARIABLE_BUFFER_LENGTH)
    size_t in_buf_len;          /*!< length of input buffer           */
#endif
#if defined(MBEDTLS_SSL_PROTO_DTLS)
    uint16_t in_epoch;          /*!< DTLS epoch for incoming records  */
    size_t next_record_offset;  /*!< offset of the next record in datagram
                                     (equal to in_left if none)       */
#endif /* MBEDTLS_SSL_PROTO_DTLS */
#if defined(MBEDTLS_SSL_DTLS_ANTI_REPLAY)
    uint64_t in_window_top;     /*!< last validated record seq_num    */
    uint64_t in_window;         /*!< bitmask for replay detection     */
#endif /* MBEDTLS_SSL_DTLS_ANTI_REPLAY */

    size_t in_hslen;            /*!< current handshake message length,
                                     including the handshake header   */
    int nb_zero;                /*!< # of 0-length encrypted messages */

    int keep_current_message;   /*!< drop or reuse current message
                                     on next call to record layer? */

#if defined(MBEDTLS_SSL_PROTO_DTLS)
    uint8_t disable_datagram_packing;  /*!< Disable packing multiple records
                                        *   within a single datagram.  */
#endif /* MBEDTLS_SSL_PROTO_DTLS */

    /*
     * Record layer (outgoing data)
     */
    unsigned char *out_buf;     /*!< output buffer                    */
    unsigned char *out_ctr;     /*!< 64-bit outgoing message counter  */
    unsigned char *out_hdr;     /*!< start of record header           */
#if defined(MBEDTLS_SSL_DTLS_CONNECTION_ID)
    unsigned char *out_cid;     /*!< The start of the CID;
                                 *   (the end is marked by in_len).   */
#endif /* MBEDTLS_SSL_DTLS_CONNECTION_ID */
    unsigned char *out_len;     /*!< two-bytes message length field   */
    unsigned char *out_iv;      /*!< ivlen-byte IV                    */
    unsigned char *out_msg;     /*!< message contents (out_iv+ivlen)  */

    int out_msgtype;            /*!< record header: message type      */
    size_t out_msglen;          /*!< record header: message length    */
    size_t out_left;            /*!< amount of data not yet written   */
#if defined(MBEDTLS_SSL_VARIABLE_BUFFER_LENGTH)
    size_t out_buf_len;         /*!< length of output buffer          */
#endif

    unsigned char cur_out_ctr[8]; /*!<  Outgoing record sequence  number. */

#if defined(MBEDTLS_SSL_PROTO_DTLS)
    uint16_t mtu;               /*!< path mtu, used to fragment outgoing messages */
#endif /* MBEDTLS_SSL_PROTO_DTLS */

#if defined(MBEDTLS_ZLIB_SUPPORT)
    unsigned char *compress_buf;        /*!<  zlib data buffer        */
#endif /* MBEDTLS_ZLIB_SUPPORT */
#if defined(MBEDTLS_SSL_CBC_RECORD_SPLITTING)
    signed char split_done;     /*!< current record already split? */
#endif /* MBEDTLS_SSL_CBC_RECORD_SPLITTING */

    /*
     * PKI layer
     */
    int client_auth;                    /*!<  flag for client auth.   */

    /*
     * User settings
     */
#if defined(MBEDTLS_X509_CRT_PARSE_C)
    char *hostname;             /*!< expected peer CN for verification
                                     (and SNI if available)                 */
#endif /* MBEDTLS_X509_CRT_PARSE_C */

#if defined(MBEDTLS_SSL_ALPN)
    const char *alpn_chosen;    /*!<  negotiated protocol                   */
#endif /* MBEDTLS_SSL_ALPN */

#if defined(MBEDTLS_SSL_DTLS_SRTP)
    /*
     * use_srtp extension
     */
    mbedtls_dtls_srtp_info dtls_srtp_info;
#endif /* MBEDTLS_SSL_DTLS_SRTP */

    /*
     * Information for DTLS hello verify
     */
#if defined(MBEDTLS_SSL_DTLS_HELLO_VERIFY) && defined(MBEDTLS_SSL_SRV_C)
    unsigned char  *cli_id;         /*!<  transport-level ID of the client  */
    size_t          cli_id_len;     /*!<  length of cli_id                  */
#endif /* MBEDTLS_SSL_DTLS_HELLO_VERIFY && MBEDTLS_SSL_SRV_C */

    /*
     * Secure renegotiation
     */
    /* needed to know when to send extension on server */
    int secure_renegotiation;           /*!<  does peer support legacy or
                                              secure renegotiation           */
#if defined(MBEDTLS_SSL_RENEGOTIATION)
    size_t verify_data_len;             /*!<  length of verify data stored   */
    char own_verify_data[MBEDTLS_SSL_VERIFY_DATA_MAX_LEN]; /*!<  previous handshake verify data */
    char peer_verify_data[MBEDTLS_SSL_VERIFY_DATA_MAX_LEN]; /*!<  previous handshake verify data */
#endif /* MBEDTLS_SSL_RENEGOTIATION */

#if defined(MBEDTLS_SSL_DTLS_CONNECTION_ID)
    /* CID configuration to use in subsequent handshakes. */

    /*! The next incoming CID, chosen by the user and applying to
     *  all subsequent handshakes. This may be different from the
     *  CID currently used in case the user has re-configured the CID
     *  after an initial handshake. */
    unsigned char own_cid[MBEDTLS_SSL_CID_IN_LEN_MAX];
    uint8_t own_cid_len;   /*!< The length of \c own_cid. */
    uint8_t negotiate_cid; /*!< This indicates whether the CID extension should
                            *   be negotiated in the next handshake or not.
                            *   Possible values are #MBEDTLS_SSL_CID_ENABLED
                            *   and #MBEDTLS_SSL_CID_DISABLED. */
#endif /* MBEDTLS_SSL_DTLS_CONNECTION_ID */
};

#if defined(MBEDTLS_SSL_HW_RECORD_ACCEL)

#if !defined(MBEDTLS_DEPRECATED_REMOVED)

#define MBEDTLS_SSL_CHANNEL_OUTBOUND   MBEDTLS_DEPRECATED_NUMERIC_CONSTANT(0)
#define MBEDTLS_SSL_CHANNEL_INBOUND    MBEDTLS_DEPRECATED_NUMERIC_CONSTANT(1)

#if defined(MBEDTLS_DEPRECATED_WARNING)
#define MBEDTLS_DEPRECATED      __attribute__((deprecated))
#else
#define MBEDTLS_DEPRECATED
#endif /* MBEDTLS_DEPRECATED_WARNING */

MBEDTLS_DEPRECATED extern int (*mbedtls_ssl_hw_record_init)(
    mbedtls_ssl_context *ssl,
    const unsigned char *key_enc, const unsigned char *key_dec,
    size_t keylen,
    const unsigned char *iv_enc,  const unsigned char *iv_dec,
    size_t ivlen,
    const unsigned char *mac_enc, const unsigned char *mac_dec,
    size_t maclen);
MBEDTLS_DEPRECATED extern int (*mbedtls_ssl_hw_record_activate)(
    mbedtls_ssl_context *ssl,
    int direction);
MBEDTLS_DEPRECATED extern int (*mbedtls_ssl_hw_record_reset)(
    mbedtls_ssl_context *ssl);
MBEDTLS_DEPRECATED extern int (*mbedtls_ssl_hw_record_write)(
    mbedtls_ssl_context *ssl);
MBEDTLS_DEPRECATED extern int (*mbedtls_ssl_hw_record_read)(
    mbedtls_ssl_context *ssl);
MBEDTLS_DEPRECATED extern int (*mbedtls_ssl_hw_record_finish)(
    mbedtls_ssl_context *ssl);

#undef MBEDTLS_DEPRECATED
#endif /* !MBEDTLS_DEPRECATED_REMOVED */

#endif /* MBEDTLS_SSL_HW_RECORD_ACCEL */

/**
 * \brief               Return the name of the ciphersuite associated with the
 *                      given ID
 *
 * \param ciphersuite_id SSL ciphersuite ID
 *
 * \return              a string containing the ciphersuite name
 */
const char *mbedtls_ssl_get_ciphersuite_name(const int ciphersuite_id);

/**
 * \brief               Return the ID of the ciphersuite associated with the
 *                      given name
 *
 * \param ciphersuite_name SSL ciphersuite name
 *
 * \return              the ID with the ciphersuite or 0 if not found
 */
int mbedtls_ssl_get_ciphersuite_id(const char *ciphersuite_name);

/**
 * \brief          Initialize an SSL context
 *                 Just makes the context ready for mbedtls_ssl_setup() or
 *                 mbedtls_ssl_free()
 *
 * \param ssl      SSL context
 */
void mbedtls_ssl_init(mbedtls_ssl_context *ssl);

/**
 * \brief          Set up an SSL context for use
 *
 * \note           No copy of the configuration context is made, it can be
 *                 shared by many mbedtls_ssl_context structures.
 *
 * \warning        The conf structure will be accessed during the session.
 *                 It must not be modified or freed as long as the session
 *                 is active.
 *
 * \warning        This function must be called exactly once per context.
 *                 Calling mbedtls_ssl_setup again is not supported, even
 *                 if no session is active.
 *
 * \note           If #MBEDTLS_USE_PSA_CRYPTO is enabled, the PSA crypto
 *                 subsystem must have been initialized by calling
 *                 psa_crypto_init() before calling this function.
 *
 * \param ssl      SSL context
 * \param conf     SSL configuration to use
 *
 * \return         0 if successful, or MBEDTLS_ERR_SSL_ALLOC_FAILED if
 *                 memory allocation failed
 */
int mbedtls_ssl_setup(mbedtls_ssl_context *ssl,
                      const mbedtls_ssl_config *conf);

/**
 * \brief          Reset an already initialized SSL context for re-use
 *                 while retaining application-set variables, function
 *                 pointers and data.
 *
 * \param ssl      SSL context
 * \return         0 if successful, or MBEDTLS_ERR_SSL_ALLOC_FAILED,
                   MBEDTLS_ERR_SSL_HW_ACCEL_FAILED or
 *                 MBEDTLS_ERR_SSL_COMPRESSION_FAILED
 */
int mbedtls_ssl_session_reset(mbedtls_ssl_context *ssl);

/**
 * \brief          Set the current endpoint type
 *
 * \param conf     SSL configuration
 * \param endpoint must be MBEDTLS_SSL_IS_CLIENT or MBEDTLS_SSL_IS_SERVER
 */
void mbedtls_ssl_conf_endpoint(mbedtls_ssl_config *conf, int endpoint);

/**
 * \brief           Set the transport type (TLS or DTLS).
 *                  Default: TLS
 *
 * \note            For DTLS, you must either provide a recv callback that
 *                  doesn't block, or one that handles timeouts, see
 *                  \c mbedtls_ssl_set_bio(). You also need to provide timer
 *                  callbacks with \c mbedtls_ssl_set_timer_cb().
 *
 * \param conf      SSL configuration
 * \param transport transport type:
 *                  MBEDTLS_SSL_TRANSPORT_STREAM for TLS,
 *                  MBEDTLS_SSL_TRANSPORT_DATAGRAM for DTLS.
 */
void mbedtls_ssl_conf_transport(mbedtls_ssl_config *conf, int transport);

/**
 * \brief          Set the certificate verification mode
 *                 Default: NONE on server, REQUIRED on client
 *
 * \param conf     SSL configuration
 * \param authmode can be:
 *
 *  MBEDTLS_SSL_VERIFY_NONE:      peer certificate is not checked
 *                        (default on server)
 *                        (insecure on client)
 *
 *  MBEDTLS_SSL_VERIFY_OPTIONAL:  peer certificate is checked, however the
 *                        handshake continues even if verification failed;
 *                        mbedtls_ssl_get_verify_result() can be called after the
 *                        handshake is complete.
 *
 *  MBEDTLS_SSL_VERIFY_REQUIRED:  peer *must* present a valid certificate,
 *                        handshake is aborted if verification failed.
 *                        (default on client)
 *
 * \note On client, MBEDTLS_SSL_VERIFY_REQUIRED is the recommended mode.
 * With MBEDTLS_SSL_VERIFY_OPTIONAL, the user needs to call mbedtls_ssl_get_verify_result() at
 * the right time(s), which may not be obvious, while REQUIRED always perform
 * the verification as soon as possible. For example, REQUIRED was protecting
 * against the "triple handshake" attack even before it was found.
 */
void mbedtls_ssl_conf_authmode(mbedtls_ssl_config *conf, int authmode);

#if defined(MBEDTLS_X509_CRT_PARSE_C)
/**
 * \brief          Set the verification callback (Optional).
 *
 *                 If set, the provided verify callback is called for each
 *                 certificate in the peer's CRT chain, including the trusted
 *                 root. For more information, please see the documentation of
 *                 \c mbedtls_x509_crt_verify().
 *
 * \note           For per context callbacks and contexts, please use
 *                 mbedtls_ssl_set_verify() instead.
 *
 * \param conf     The SSL configuration to use.
 * \param f_vrfy   The verification callback to use during CRT verification.
 * \param p_vrfy   The opaque context to be passed to the callback.
 */
void mbedtls_ssl_conf_verify(mbedtls_ssl_config *conf,
                             int (*f_vrfy)(void *, mbedtls_x509_crt *, int, uint32_t *),
                             void *p_vrfy);
#endif /* MBEDTLS_X509_CRT_PARSE_C */

/**
 * \brief          Set the random number generator callback
 *
 * \param conf     SSL configuration
 * \param f_rng    RNG function
 * \param p_rng    RNG parameter
 */
void mbedtls_ssl_conf_rng(mbedtls_ssl_config *conf,
                          int (*f_rng)(void *, unsigned char *, size_t),
                          void *p_rng);

/**
 * \brief          Set the debug callback
 *
 *                 The callback has the following argument:
 *                 void *           opaque context for the callback
 *                 int              debug level
 *                 const char *     file name
 *                 int              line number
 *                 const char *     message
 *
 * \param conf     SSL configuration
 * \param f_dbg    debug function
 * \param p_dbg    debug parameter
 */
void mbedtls_ssl_conf_dbg(mbedtls_ssl_config *conf,
                          void (*f_dbg)(void *, int, const char *, int, const char *),
                          void  *p_dbg);

/**
 * \brief          Set the underlying BIO callbacks for write, read and
 *                 read-with-timeout.
 *
 * \param ssl      SSL context
 * \param p_bio    parameter (context) shared by BIO callbacks
 * \param f_send   write callback
 * \param f_recv   read callback
 * \param f_recv_timeout blocking read callback with timeout.
 *
 * \note           One of f_recv or f_recv_timeout can be NULL, in which case
 *                 the other is used. If both are non-NULL, f_recv_timeout is
 *                 used and f_recv is ignored (as if it were NULL).
 *
 * \note           The two most common use cases are:
 *                 - non-blocking I/O, f_recv != NULL, f_recv_timeout == NULL
 *                 - blocking I/O, f_recv == NULL, f_recv_timeout != NULL
 *
 * \note           For DTLS, you need to provide either a non-NULL
 *                 f_recv_timeout callback, or a f_recv that doesn't block.
 *
 * \note           See the documentations of \c mbedtls_ssl_send_t,
 *                 \c mbedtls_ssl_recv_t and \c mbedtls_ssl_recv_timeout_t for
 *                 the conventions those callbacks must follow.
 *
 * \note           On some platforms, net_sockets.c provides
 *                 \c mbedtls_net_send(), \c mbedtls_net_recv() and
 *                 \c mbedtls_net_recv_timeout() that are suitable to be used
 *                 here.
 */
void mbedtls_ssl_set_bio(mbedtls_ssl_context *ssl,
                         void *p_bio,
                         mbedtls_ssl_send_t *f_send,
                         mbedtls_ssl_recv_t *f_recv,
                         mbedtls_ssl_recv_timeout_t *f_recv_timeout);

#if defined(MBEDTLS_SSL_PROTO_DTLS)

#if defined(MBEDTLS_SSL_DTLS_CONNECTION_ID)


/**
 * \brief             Configure the use of the Connection ID (CID)
 *                    extension in the next handshake.
 *
 *                    Reference: draft-ietf-tls-dtls-connection-id-05
 *                    https://tools.ietf.org/html/draft-ietf-tls-dtls-connection-id-05
 *
 *                    The DTLS CID extension allows the reliable association of
 *                    DTLS records to DTLS connections across changes in the
 *                    underlying transport (changed IP and Port metadata) by
 *                    adding explicit connection identifiers (CIDs) to the
 *                    headers of encrypted DTLS records. The desired CIDs are
 *                    configured by the application layer and are exchanged in
 *                    new `ClientHello` / `ServerHello` extensions during the
 *                    handshake, where each side indicates the CID it wants the
 *                    peer to use when writing encrypted messages. The CIDs are
 *                    put to use once records get encrypted: the stack discards
 *                    any incoming records that don't include the configured CID
 *                    in their header, and adds the peer's requested CID to the
 *                    headers of outgoing messages.
 *
 *                    This API enables or disables the use of the CID extension
 *                    in the next handshake and sets the value of the CID to
 *                    be used for incoming messages.
 *
 * \param ssl         The SSL context to configure. This must be initialized.
 * \param enable      This value determines whether the CID extension should
 *                    be used or not. Possible values are:
 *                    - MBEDTLS_SSL_CID_ENABLED to enable the use of the CID.
 *                    - MBEDTLS_SSL_CID_DISABLED (default) to disable the use
 *                      of the CID.
 * \param own_cid     The address of the readable buffer holding the CID we want
 *                    the peer to use when sending encrypted messages to us.
 *                    This may be \c NULL if \p own_cid_len is \c 0.
 *                    This parameter is unused if \p enable is set to
 *                    MBEDTLS_SSL_CID_DISABLED.
 * \param own_cid_len The length of \p own_cid.
 *                    This parameter is unused if \p enable is set to
 *                    MBEDTLS_SSL_CID_DISABLED.
 *
 * \note              The value of \p own_cid_len must match the value of the
 *                    \c len parameter passed to mbedtls_ssl_conf_cid()
 *                    when configuring the ::mbedtls_ssl_config that \p ssl
 *                    is bound to.
 *
 * \note              This CID configuration applies to subsequent handshakes
 *                    performed on the SSL context \p ssl, but does not trigger
 *                    one. You still have to call `mbedtls_ssl_handshake()`
 *                    (for the initial handshake) or `mbedtls_ssl_renegotiate()`
 *                    (for a renegotiation handshake) explicitly after a
 *                    successful call to this function to run the handshake.
 *
 * \note              This call cannot guarantee that the use of the CID
 *                    will be successfully negotiated in the next handshake,
 *                    because the peer might not support it. Specifically:
 *                    - On the Client, enabling the use of the CID through
 *                      this call implies that the `ClientHello` in the next
 *                      handshake will include the CID extension, thereby
 *                      offering the use of the CID to the server. Only if
 *                      the `ServerHello` contains the CID extension, too,
 *                      the CID extension will actually be put to use.
 *                    - On the Server, enabling the use of the CID through
 *                      this call implies that that the server will look for
 *                      the CID extension in a `ClientHello` from the client,
 *                      and, if present, reply with a CID extension in its
 *                      `ServerHello`.
 *
 * \note              To check whether the use of the CID was negotiated
 *                    after the subsequent handshake has completed, please
 *                    use the API mbedtls_ssl_get_peer_cid().
 *
 * \warning           If the use of the CID extension is enabled in this call
 *                    and the subsequent handshake negotiates its use, Mbed TLS
 *                    will silently drop every packet whose CID does not match
 *                    the CID configured in \p own_cid. It is the responsibility
 *                    of the user to adapt the underlying transport to take care
 *                    of CID-based demultiplexing before handing datagrams to
 *                    Mbed TLS.
 *
 * \return            \c 0 on success. In this case, the CID configuration
 *                    applies to the next handshake.
 * \return            A negative error code on failure.
 */
int mbedtls_ssl_set_cid(mbedtls_ssl_context *ssl,
                        int enable,
                        unsigned char const *own_cid,
                        size_t own_cid_len);

/**
 * \brief              Get information about the use of the CID extension
 *                     in the current connection.
 *
 * \param ssl          The SSL context to query.
 * \param enabled      The address at which to store whether the CID extension
 *                     is currently in use or not. If the CID is in use,
 *                     `*enabled` is set to MBEDTLS_SSL_CID_ENABLED;
 *                     otherwise, it is set to MBEDTLS_SSL_CID_DISABLED.
 * \param peer_cid     The address of the buffer in which to store the CID
 *                     chosen by the peer (if the CID extension is used).
 *                     This may be \c NULL in case the value of peer CID
 *                     isn't needed. If it is not \c NULL, \p peer_cid_len
 *                     must not be \c NULL.
 * \param peer_cid_len The address at which to store the size of the CID
 *                     chosen by the peer (if the CID extension is used).
 *                     This is also the number of Bytes in \p peer_cid that
 *                     have been written.
 *                     This may be \c NULL in case the length of the peer CID
 *                     isn't needed. If it is \c NULL, \p peer_cid must be
 *                     \c NULL, too.
 *
 * \note               This applies to the state of the CID negotiated in
 *                     the last complete handshake. If a handshake is in
 *                     progress, this function will attempt to complete
 *                     the handshake first.
 *
 * \note               If CID extensions have been exchanged but both client
 *                     and server chose to use an empty CID, this function
 *                     sets `*enabled` to #MBEDTLS_SSL_CID_DISABLED
 *                     (the rationale for this is that the resulting
 *                     communication is the same as if the CID extensions
 *                     hadn't been used).
 *
 * \return            \c 0 on success.
 * \return            A negative error code on failure.
 */
int mbedtls_ssl_get_peer_cid(mbedtls_ssl_context *ssl,
                             int *enabled,
                             unsigned char peer_cid[MBEDTLS_SSL_CID_OUT_LEN_MAX],
                             size_t *peer_cid_len);

#endif /* MBEDTLS_SSL_DTLS_CONNECTION_ID */

/**
 * \brief          Set the Maximum Transport Unit (MTU).
 *                 Special value: 0 means unset (no limit).
 *                 This represents the maximum size of a datagram payload
 *                 handled by the transport layer (usually UDP) as determined
 *                 by the network link and stack. In practice, this controls
 *                 the maximum size datagram the DTLS layer will pass to the
 *                 \c f_send() callback set using \c mbedtls_ssl_set_bio().
 *
 * \note           The limit on datagram size is converted to a limit on
 *                 record payload by subtracting the current overhead of
 *                 encapsulation and encryption/authentication if any.
 *
 * \note           This can be called at any point during the connection, for
 *                 example when a Path Maximum Transfer Unit (PMTU)
 *                 estimate becomes available from other sources,
 *                 such as lower (or higher) protocol layers.
 *
 * \note           This setting only controls the size of the packets we send,
 *                 and does not restrict the size of the datagrams we're
 *                 willing to receive. Client-side, you can request the
 *                 server to use smaller records with \c
 *                 mbedtls_ssl_conf_max_frag_len().
 *
 * \note           If both a MTU and a maximum fragment length have been
 *                 configured (or negotiated with the peer), the resulting
 *                 lower limit on record payload (see first note) is used.
 *
 * \note           This can only be used to decrease the maximum size
 *                 of datagrams (hence records, see first note) sent. It
 *                 cannot be used to increase the maximum size of records over
 *                 the limit set by #MBEDTLS_SSL_OUT_CONTENT_LEN.
 *
 * \note           Values lower than the current record layer expansion will
 *                 result in an error when trying to send data.
 *
 * \note           Using record compression together with a non-zero MTU value
 *                 will result in an error when trying to send data.
 *
 * \param ssl      SSL context
 * \param mtu      Value of the path MTU in bytes
 */
void mbedtls_ssl_set_mtu(mbedtls_ssl_context *ssl, uint16_t mtu);
#endif /* MBEDTLS_SSL_PROTO_DTLS */

#if defined(MBEDTLS_X509_CRT_PARSE_C)
/**
 * \brief          Set a connection-specific verification callback (optional).
 *
 *                 If set, the provided verify callback is called for each
 *                 certificate in the peer's CRT chain, including the trusted
 *                 root. For more information, please see the documentation of
 *                 \c mbedtls_x509_crt_verify().
 *
 * \note           This call is analogous to mbedtls_ssl_conf_verify() but
 *                 binds the verification callback and context to an SSL context
 *                 as opposed to an SSL configuration.
 *                 If mbedtls_ssl_conf_verify() and mbedtls_ssl_set_verify()
 *                 are both used, mbedtls_ssl_set_verify() takes precedence.
 *
 * \param ssl      The SSL context to use.
 * \param f_vrfy   The verification callback to use during CRT verification.
 * \param p_vrfy   The opaque context to be passed to the callback.
 */
void mbedtls_ssl_set_verify(mbedtls_ssl_context *ssl,
                            int (*f_vrfy)(void *, mbedtls_x509_crt *, int, uint32_t *),
                            void *p_vrfy);
#endif /* MBEDTLS_X509_CRT_PARSE_C */

/**
 * \brief          Set the timeout period for mbedtls_ssl_read()
 *                 (Default: no timeout.)
 *
 * \param conf     SSL configuration context
 * \param timeout  Timeout value in milliseconds.
 *                 Use 0 for no timeout (default).
 *
 * \note           With blocking I/O, this will only work if a non-NULL
 *                 \c f_recv_timeout was set with \c mbedtls_ssl_set_bio().
 *                 With non-blocking I/O, this will only work if timer
 *                 callbacks were set with \c mbedtls_ssl_set_timer_cb().
 *
 * \note           With non-blocking I/O, you may also skip this function
 *                 altogether and handle timeouts at the application layer.
 */
void mbedtls_ssl_conf_read_timeout(mbedtls_ssl_config *conf, uint32_t timeout);

#if defined(MBEDTLS_SSL_RECORD_CHECKING)
/**
 * \brief          Check whether a buffer contains a valid and authentic record
 *                 that has not been seen before. (DTLS only).
 *
 *                 This function does not change the user-visible state
 *                 of the SSL context. Its sole purpose is to provide
 *                 an indication of the legitimacy of an incoming record.
 *
 *                 This can be useful e.g. in distributed server environments
 *                 using the DTLS Connection ID feature, in which connections
 *                 might need to be passed between service instances on a change
 *                 of peer address, but where such disruptive operations should
 *                 only happen after the validity of incoming records has been
 *                 confirmed.
 *
 * \param ssl      The SSL context to use.
 * \param buf      The address of the buffer holding the record to be checked.
 *                 This must be a read/write buffer of length \p buflen Bytes.
 * \param buflen   The length of \p buf in Bytes.
 *
 * \note           This routine only checks whether the provided buffer begins
 *                 with a valid and authentic record that has not been seen
 *                 before, but does not check potential data following the
 *                 initial record. In particular, it is possible to pass DTLS
 *                 datagrams containing multiple records, in which case only
 *                 the first record is checked.
 *
 * \note           This function modifies the input buffer \p buf. If you need
 *                 to preserve the original record, you have to maintain a copy.
 *
 * \return         \c 0 if the record is valid and authentic and has not been
 *                 seen before.
 * \return         MBEDTLS_ERR_SSL_INVALID_MAC if the check completed
 *                 successfully but the record was found to be not authentic.
 * \return         MBEDTLS_ERR_SSL_INVALID_RECORD if the check completed
 *                 successfully but the record was found to be invalid for
 *                 a reason different from authenticity checking.
 * \return         MBEDTLS_ERR_SSL_UNEXPECTED_RECORD if the check completed
 *                 successfully but the record was found to be unexpected
 *                 in the state of the SSL context, including replayed records.
 * \return         Another negative error code on different kinds of failure.
 *                 In this case, the SSL context becomes unusable and needs
 *                 to be freed or reset before reuse.
 */
int mbedtls_ssl_check_record(mbedtls_ssl_context const *ssl,
                             unsigned char *buf,
                             size_t buflen);
#endif /* MBEDTLS_SSL_RECORD_CHECKING */

/**
 * \brief          Set the timer callbacks (Mandatory for DTLS.)
 *
 * \param ssl      SSL context
 * \param p_timer  parameter (context) shared by timer callbacks
 * \param f_set_timer   set timer callback
 * \param f_get_timer   get timer callback. Must return:
 *
 * \note           See the documentation of \c mbedtls_ssl_set_timer_t and
 *                 \c mbedtls_ssl_get_timer_t for the conventions this pair of
 *                 callbacks must follow.
 *
 * \note           On some platforms, timing.c provides
 *                 \c mbedtls_timing_set_delay() and
 *                 \c mbedtls_timing_get_delay() that are suitable for using
 *                 here, except if using an event-driven style.
 *
 * \note           See also the "DTLS tutorial" article in our knowledge base.
 *                 https://mbed-tls.readthedocs.io/en/latest/kb/how-to/dtls-tutorial
 */
void mbedtls_ssl_set_timer_cb(mbedtls_ssl_context *ssl,
                              void *p_timer,
                              mbedtls_ssl_set_timer_t *f_set_timer,
                              mbedtls_ssl_get_timer_t *f_get_timer);

/**
 * \brief           Callback type: generate and write session ticket
 *
 * \note            This describes what a callback implementation should do.
 *                  This callback should generate an encrypted and
 *                  authenticated ticket for the session and write it to the
 *                  output buffer. Here, ticket means the opaque ticket part
 *                  of the NewSessionTicket structure of RFC 5077.
 *
 * \param p_ticket  Context for the callback
 * \param session   SSL session to be written in the ticket
 * \param start     Start of the output buffer
 * \param end       End of the output buffer
 * \param tlen      On exit, holds the length written
 * \param lifetime  On exit, holds the lifetime of the ticket in seconds
 *
 * \return          0 if successful, or
 *                  a specific MBEDTLS_ERR_XXX code.
 */
typedef int mbedtls_ssl_ticket_write_t(void *p_ticket,
                                       const mbedtls_ssl_session *session,
                                       unsigned char *start,
                                       const unsigned char *end,
                                       size_t *tlen,
                                       uint32_t *lifetime);

#if defined(MBEDTLS_SSL_EXPORT_KEYS)
/**
 * \brief           Callback type: Export key block and master secret
 *
 * \note            This is required for certain uses of TLS, e.g. EAP-TLS
 *                  (RFC 5216) and Thread. The key pointers are ephemeral and
 *                  therefore must not be stored. The master secret and keys
 *                  should not be used directly except as an input to a key
 *                  derivation function.
 *
 * \param p_expkey  Context for the callback
 * \param ms        Pointer to master secret (fixed length: 48 bytes)
 * \param kb        Pointer to key block, see RFC 5246 section 6.3
 *                  (variable length: 2 * maclen + 2 * keylen + 2 * ivlen).
 * \param maclen    MAC length
 * \param keylen    Key length
 * \param ivlen     IV length
 *
 * \return          0 if successful, or
 *                  a specific MBEDTLS_ERR_XXX code.
 */
typedef int mbedtls_ssl_export_keys_t(void *p_expkey,
                                      const unsigned char *ms,
                                      const unsigned char *kb,
                                      size_t maclen,
                                      size_t keylen,
                                      size_t ivlen);

/**
 * \brief           Callback type: Export key block, master secret,
 *                                 handshake randbytes and the tls_prf function
 *                                 used to derive keys.
 *
 * \note            This is required for certain uses of TLS, e.g. EAP-TLS
 *                  (RFC 5216) and Thread. The key pointers are ephemeral and
 *                  therefore must not be stored. The master secret and keys
 *                  should not be used directly except as an input to a key
 *                  derivation function.
 *
 * \param p_expkey  Context for the callback.
 * \param ms        Pointer to master secret (fixed length: 48 bytes).
 * \param kb            Pointer to key block, see RFC 5246 section 6.3.
 *                      (variable length: 2 * maclen + 2 * keylen + 2 * ivlen).
 * \param maclen        MAC length.
 * \param keylen        Key length.
 * \param ivlen         IV length.
 * \param client_random The client random bytes.
 * \param server_random The server random bytes.
 * \param tls_prf_type The tls_prf enum type.
 *
 * \return          0 if successful, or
 *                  a specific MBEDTLS_ERR_XXX code.
 */
typedef int mbedtls_ssl_export_keys_ext_t(void *p_expkey,
                                          const unsigned char *ms,
                                          const unsigned char *kb,
                                          size_t maclen,
                                          size_t keylen,
                                          size_t ivlen,
                                          const unsigned char client_random[32],
                                          const unsigned char server_random[32],
                                          mbedtls_tls_prf_types tls_prf_type);
#endif /* MBEDTLS_SSL_EXPORT_KEYS */

/**
 * \brief           Callback type: parse and load session ticket
 *
 * \note            This describes what a callback implementation should do.
 *                  This callback should parse a session ticket as generated
 *                  by the corresponding mbedtls_ssl_ticket_write_t function,
 *                  and, if the ticket is authentic and valid, load the
 *                  session.
 *
 * \note            The implementation is allowed to modify the first len
 *                  bytes of the input buffer, eg to use it as a temporary
 *                  area for the decrypted ticket contents.
 *
 * \param p_ticket  Context for the callback
 * \param session   SSL session to be loaded
 * \param buf       Start of the buffer containing the ticket
 * \param len       Length of the ticket.
 *
 * \return          0 if successful, or
 *                  MBEDTLS_ERR_SSL_INVALID_MAC if not authentic, or
 *                  MBEDTLS_ERR_SSL_SESSION_TICKET_EXPIRED if expired, or
 *                  any other non-zero code for other failures.
 */
typedef int mbedtls_ssl_ticket_parse_t(void *p_ticket,
                                       mbedtls_ssl_session *session,
                                       unsigned char *buf,
                                       size_t len);

#if defined(MBEDTLS_SSL_SESSION_TICKETS) && defined(MBEDTLS_SSL_SRV_C)
/**
 * \brief           Configure SSL session ticket callbacks (server only).
 *                  (Default: none.)
 *
 * \note            On server, session tickets are enabled by providing
 *                  non-NULL callbacks.
 *
 * \note            On client, use \c mbedtls_ssl_conf_session_tickets().
 *
 * \param conf      SSL configuration context
 * \param f_ticket_write    Callback for writing a ticket
 * \param f_ticket_parse    Callback for parsing a ticket
 * \param p_ticket          Context shared by the two callbacks
 */
void mbedtls_ssl_conf_session_tickets_cb(mbedtls_ssl_config *conf,
                                         mbedtls_ssl_ticket_write_t *f_ticket_write,
                                         mbedtls_ssl_ticket_parse_t *f_ticket_parse,
                                         void *p_ticket);
#endif /* MBEDTLS_SSL_SESSION_TICKETS && MBEDTLS_SSL_SRV_C */

#if defined(MBEDTLS_SSL_EXPORT_KEYS)
/**
 * \brief           Configure key export callback.
 *                  (Default: none.)
 *
 * \note            See \c mbedtls_ssl_export_keys_t.
 *
 * \param conf      SSL configuration context
 * \param f_export_keys     Callback for exporting keys
 * \param p_export_keys     Context for the callback
 */
void mbedtls_ssl_conf_export_keys_cb(mbedtls_ssl_config *conf,
                                     mbedtls_ssl_export_keys_t *f_export_keys,
                                     void *p_export_keys);

/**
 * \brief           Configure extended key export callback.
 *                  (Default: none.)
 *
 * \note            See \c mbedtls_ssl_export_keys_ext_t.
 * \warning         Exported key material must not be used for any purpose
 *                  before the (D)TLS handshake is completed
 *
 * \param conf      SSL configuration context
 * \param f_export_keys_ext Callback for exporting keys
 * \param p_export_keys     Context for the callback
 */
void mbedtls_ssl_conf_export_keys_ext_cb(mbedtls_ssl_config *conf,
                                         mbedtls_ssl_export_keys_ext_t *f_export_keys_ext,
                                         void *p_export_keys);
#endif /* MBEDTLS_SSL_EXPORT_KEYS */

#if defined(MBEDTLS_SSL_ASYNC_PRIVATE)
/**
 * \brief           Configure asynchronous private key operation callbacks.
 *
 * \param conf              SSL configuration context
 * \param f_async_sign      Callback to start a signature operation. See
 *                          the description of ::mbedtls_ssl_async_sign_t
 *                          for more information. This may be \c NULL if the
 *                          external processor does not support any signature
 *                          operation; in this case the private key object
 *                          associated with the certificate will be used.
 * \param f_async_decrypt   Callback to start a decryption operation. See
 *                          the description of ::mbedtls_ssl_async_decrypt_t
 *                          for more information. This may be \c NULL if the
 *                          external processor does not support any decryption
 *                          operation; in this case the private key object
 *                          associated with the certificate will be used.
 * \param f_async_resume    Callback to resume an asynchronous operation. See
 *                          the description of ::mbedtls_ssl_async_resume_t
 *                          for more information. This may not be \c NULL unless
 *                          \p f_async_sign and \p f_async_decrypt are both
 *                          \c NULL.
 * \param f_async_cancel    Callback to cancel an asynchronous operation. See
 *                          the description of ::mbedtls_ssl_async_cancel_t
 *                          for more information. This may be \c NULL if
 *                          no cleanup is needed.
 * \param config_data       A pointer to configuration data which can be
 *                          retrieved with
 *                          mbedtls_ssl_conf_get_async_config_data(). The
 *                          library stores this value without dereferencing it.
 */
void mbedtls_ssl_conf_async_private_cb(mbedtls_ssl_config *conf,
                                       mbedtls_ssl_async_sign_t *f_async_sign,
                                       mbedtls_ssl_async_decrypt_t *f_async_decrypt,
                                       mbedtls_ssl_async_resume_t *f_async_resume,
                                       mbedtls_ssl_async_cancel_t *f_async_cancel,
                                       void *config_data);

/**
 * \brief           Retrieve the configuration data set by
 *                  mbedtls_ssl_conf_async_private_cb().
 *
 * \param conf      SSL configuration context
 * \return          The configuration data set by
 *                  mbedtls_ssl_conf_async_private_cb().
 */
void *mbedtls_ssl_conf_get_async_config_data(const mbedtls_ssl_config *conf);

/**
 * \brief           Retrieve the asynchronous operation user context.
 *
 * \note            This function may only be called while a handshake
 *                  is in progress.
 *
 * \param ssl       The SSL context to access.
 *
 * \return          The asynchronous operation user context that was last
 *                  set during the current handshake. If
 *                  mbedtls_ssl_set_async_operation_data() has not yet been
 *                  called during the current handshake, this function returns
 *                  \c NULL.
 */
void *mbedtls_ssl_get_async_operation_data(const mbedtls_ssl_context *ssl);

/**
 * \brief           Retrieve the asynchronous operation user context.
 *
 * \note            This function may only be called while a handshake
 *                  is in progress.
 *
 * \param ssl       The SSL context to access.
 * \param ctx       The new value of the asynchronous operation user context.
 *                  Call mbedtls_ssl_get_async_operation_data() later during the
 *                  same handshake to retrieve this value.
 */
void mbedtls_ssl_set_async_operation_data(mbedtls_ssl_context *ssl,
                                          void *ctx);
#endif /* MBEDTLS_SSL_ASYNC_PRIVATE */

/**
 * \brief          Callback type: generate a cookie
 *
 * \param ctx      Context for the callback
 * \param p        Buffer to write to,
 *                 must be updated to point right after the cookie
 * \param end      Pointer to one past the end of the output buffer
 * \param info     Client ID info that was passed to
 *                 \c mbedtls_ssl_set_client_transport_id()
 * \param ilen     Length of info in bytes
 *
 * \return         The callback must return 0 on success,
 *                 or a negative error code.
 */
typedef int mbedtls_ssl_cookie_write_t(void *ctx,
                                       unsigned char **p, unsigned char *end,
                                       const unsigned char *info, size_t ilen);

/**
 * \brief          Callback type: verify a cookie
 *
 * \param ctx      Context for the callback
 * \param cookie   Cookie to verify
 * \param clen     Length of cookie
 * \param info     Client ID info that was passed to
 *                 \c mbedtls_ssl_set_client_transport_id()
 * \param ilen     Length of info in bytes
 *
 * \return         The callback must return 0 if cookie is valid,
 *                 or a negative error code.
 */
typedef int mbedtls_ssl_cookie_check_t(void *ctx,
                                       const unsigned char *cookie, size_t clen,
                                       const unsigned char *info, size_t ilen);

#if defined(MBEDTLS_SSL_DTLS_HELLO_VERIFY) && defined(MBEDTLS_SSL_SRV_C)
/**
 * \brief           Register callbacks for DTLS cookies
 *                  (Server only. DTLS only.)
 *
 *                  Default: dummy callbacks that fail, in order to force you to
 *                  register working callbacks (and initialize their context).
 *
 *                  To disable HelloVerifyRequest, register NULL callbacks.
 *
 * \warning         Disabling hello verification allows your server to be used
 *                  for amplification in DoS attacks against other hosts.
 *                  Only disable if you known this can't happen in your
 *                  particular environment.
 *
 * \note            See comments on \c mbedtls_ssl_handshake() about handling
 *                  the MBEDTLS_ERR_SSL_HELLO_VERIFY_REQUIRED that is expected
 *                  on the first handshake attempt when this is enabled.
 *
 * \note            This is also necessary to handle client reconnection from
 *                  the same port as described in RFC 6347 section 4.2.8 (only
 *                  the variant with cookies is supported currently). See
 *                  comments on \c mbedtls_ssl_read() for details.
 *
 * \param conf              SSL configuration
 * \param f_cookie_write    Cookie write callback
 * \param f_cookie_check    Cookie check callback
 * \param p_cookie          Context for both callbacks
 */
void mbedtls_ssl_conf_dtls_cookies(mbedtls_ssl_config *conf,
                                   mbedtls_ssl_cookie_write_t *f_cookie_write,
                                   mbedtls_ssl_cookie_check_t *f_cookie_check,
                                   void *p_cookie);

/**
 * \brief          Set client's transport-level identification info.
 *                 (Server only. DTLS only.)
 *
 *                 This is usually the IP address (and port), but could be
 *                 anything identify the client depending on the underlying
 *                 network stack. Used for HelloVerifyRequest with DTLS.
 *                 This is *not* used to route the actual packets.
 *
 * \param ssl      SSL context
 * \param info     Transport-level info identifying the client (eg IP + port)
 * \param ilen     Length of info in bytes
 *
 * \note           An internal copy is made, so the info buffer can be reused.
 *
 * \return         0 on success,
 *                 MBEDTLS_ERR_SSL_BAD_INPUT_DATA if used on client,
 *                 MBEDTLS_ERR_SSL_ALLOC_FAILED if out of memory.
 */
int mbedtls_ssl_set_client_transport_id(mbedtls_ssl_context *ssl,
                                        const unsigned char *info,
                                        size_t ilen);

#endif /* MBEDTLS_SSL_DTLS_HELLO_VERIFY && MBEDTLS_SSL_SRV_C */

#if defined(MBEDTLS_SSL_DTLS_ANTI_REPLAY)
/**
 * \brief          Enable or disable anti-replay protection for DTLS.
 *                 (DTLS only, no effect on TLS.)
 *                 Default: enabled.
 *
 * \param conf     SSL configuration
 * \param mode     MBEDTLS_SSL_ANTI_REPLAY_ENABLED or MBEDTLS_SSL_ANTI_REPLAY_DISABLED.
 *
 * \warning        Disabling this is a security risk unless the application
 *                 protocol handles duplicated packets in a safe way. You
 *                 should not disable this without careful consideration.
 *                 However, if your application already detects duplicated
 *                 packets and needs information about them to adjust its
 *                 transmission strategy, then you'll want to disable this.
 */
void mbedtls_ssl_conf_dtls_anti_replay(mbedtls_ssl_config *conf, char mode);
#endif /* MBEDTLS_SSL_DTLS_ANTI_REPLAY */

#if defined(MBEDTLS_SSL_DTLS_BADMAC_LIMIT)
/**
 * \brief          Set a limit on the number of records with a bad MAC
 *                 before terminating the connection.
 *                 (DTLS only, no effect on TLS.)
 *                 Default: 0 (disabled).
 *
 * \param conf     SSL configuration
 * \param limit    Limit, or 0 to disable.
 *
 * \note           If the limit is N, then the connection is terminated when
 *                 the Nth non-authentic record is seen.
 *
 * \note           Records with an invalid header are not counted, only the
 *                 ones going through the authentication-decryption phase.
 *
 * \note           This is a security trade-off related to the fact that it's
 *                 often relatively easy for an active attacker to inject UDP
 *                 datagrams. On one hand, setting a low limit here makes it
 *                 easier for such an attacker to forcibly terminated a
 *                 connection. On the other hand, a high limit or no limit
 *                 might make us waste resources checking authentication on
 *                 many bogus packets.
 */
void mbedtls_ssl_conf_dtls_badmac_limit(mbedtls_ssl_config *conf, unsigned limit);
#endif /* MBEDTLS_SSL_DTLS_BADMAC_LIMIT */

#if defined(MBEDTLS_SSL_PROTO_DTLS)

/**
 * \brief          Allow or disallow packing of multiple handshake records
 *                 within a single datagram.
 *
 * \param ssl           The SSL context to configure.
 * \param allow_packing This determines whether datagram packing may
 *                      be used or not. A value of \c 0 means that every
 *                      record will be sent in a separate datagram; a
 *                      value of \c 1 means that, if space permits,
 *                      multiple handshake messages (including CCS) belonging to
 *                      a single flight may be packed within a single datagram.
 *
 * \note           This is enabled by default and should only be disabled
 *                 for test purposes, or if datagram packing causes
 *                 interoperability issues with peers that don't support it.
 *
 * \note           Allowing datagram packing reduces the network load since
 *                 there's less overhead if multiple messages share the same
 *                 datagram. Also, it increases the handshake efficiency
 *                 since messages belonging to a single datagram will not
 *                 be reordered in transit, and so future message buffering
 *                 or flight retransmission (if no buffering is used) as
 *                 means to deal with reordering are needed less frequently.
 *
 * \note           Application records are not affected by this option and
 *                 are currently always sent in separate datagrams.
 *
 */
void mbedtls_ssl_set_datagram_packing(mbedtls_ssl_context *ssl,
                                      unsigned allow_packing);

/**
 * \brief          Set retransmit timeout values for the DTLS handshake.
 *                 (DTLS only, no effect on TLS.)
 *
 * \param conf     SSL configuration
 * \param min      Initial timeout value in milliseconds.
 *                 Default: 1000 (1 second).
 * \param max      Maximum timeout value in milliseconds.
 *                 Default: 60000 (60 seconds).
 *
 * \note           Default values are from RFC 6347 section 4.2.4.1.
 *
 * \note           The 'min' value should typically be slightly above the
 *                 expected round-trip time to your peer, plus whatever time
 *                 it takes for the peer to process the message. For example,
 *                 if your RTT is about 600ms and you peer needs up to 1s to
 *                 do the cryptographic operations in the handshake, then you
 *                 should set 'min' slightly above 1600. Lower values of 'min'
 *                 might cause spurious resends which waste network resources,
 *                 while larger value of 'min' will increase overall latency
 *                 on unreliable network links.
 *
 * \note           The more unreliable your network connection is, the larger
 *                 your max / min ratio needs to be in order to achieve
 *                 reliable handshakes.
 *
 * \note           Messages are retransmitted up to log2(ceil(max/min)) times.
 *                 For example, if min = 1s and max = 5s, the retransmit plan
 *                 goes: send ... 1s -> resend ... 2s -> resend ... 4s ->
 *                 resend ... 5s -> give up and return a timeout error.
 */
void mbedtls_ssl_conf_handshake_timeout(mbedtls_ssl_config *conf, uint32_t min, uint32_t max);
#endif /* MBEDTLS_SSL_PROTO_DTLS */

#if defined(MBEDTLS_SSL_SRV_C)
/**
 * \brief          Set the session cache callbacks (server-side only)
 *                 If not set, no session resuming is done (except if session
 *                 tickets are enabled too).
 *
 *                 The session cache has the responsibility to check for stale
 *                 entries based on timeout. See RFC 5246 for recommendations.
 *
 *                 Warning: session.peer_cert is cleared by the SSL/TLS layer on
 *                 connection shutdown, so do not cache the pointer! Either set
 *                 it to NULL or make a full copy of the certificate.
 *
 *                 The get callback is called once during the initial handshake
 *                 to enable session resuming. The get function has the
 *                 following parameters: (void *parameter, mbedtls_ssl_session *session)
 *                 If a valid entry is found, it should fill the master of
 *                 the session object with the cached values and return 0,
 *                 return 1 otherwise. Optionally peer_cert can be set as well
 *                 if it is properly present in cache entry.
 *
 *                 The set callback is called once during the initial handshake
 *                 to enable session resuming after the entire handshake has
 *                 been finished. The set function has the following parameters:
 *                 (void *parameter, const mbedtls_ssl_session *session). The function
 *                 should create a cache entry for future retrieval based on
 *                 the data in the session structure and should keep in mind
 *                 that the mbedtls_ssl_session object presented (and all its referenced
 *                 data) is cleared by the SSL/TLS layer when the connection is
 *                 terminated. It is recommended to add metadata to determine if
 *                 an entry is still valid in the future. Return 0 if
 *                 successfully cached, return 1 otherwise.
 *
 * \param conf           SSL configuration
 * \param p_cache        parameter (context) for both callbacks
 * \param f_get_cache    session get callback
 * \param f_set_cache    session set callback
 */
void mbedtls_ssl_conf_session_cache(mbedtls_ssl_config *conf,
                                    void *p_cache,
                                    int (*f_get_cache)(void *, mbedtls_ssl_session *),
                                    int (*f_set_cache)(void *, const mbedtls_ssl_session *));
#endif /* MBEDTLS_SSL_SRV_C */

#if defined(MBEDTLS_SSL_CLI_C)
/**
 * \brief          Request resumption of session (client-side only)
 *                 Session data is copied from presented session structure.
 *
 * \param ssl      SSL context
 * \param session  session context
 *
 * \return         0 if successful,
 *                 MBEDTLS_ERR_SSL_ALLOC_FAILED if memory allocation failed,
 *                 MBEDTLS_ERR_SSL_BAD_INPUT_DATA if used server-side or
 *                 arguments are otherwise invalid
 *
 * \sa             mbedtls_ssl_get_session()
 */
int mbedtls_ssl_set_session(mbedtls_ssl_context *ssl, const mbedtls_ssl_session *session);
#endif /* MBEDTLS_SSL_CLI_C */

/**
 * \brief          Load serialized session data into a session structure.
 *                 On client, this can be used for loading saved sessions
 *                 before resuming them with mbedtls_ssl_set_session().
 *                 On server, this can be used for alternative implementations
 *                 of session cache or session tickets.
 *
 * \warning        If a peer certificate chain is associated with the session,
 *                 the serialized state will only contain the peer's
 *                 end-entity certificate and the result of the chain
 *                 verification (unless verification was disabled), but not
 *                 the rest of the chain.
 *
 * \see            mbedtls_ssl_session_save()
 * \see            mbedtls_ssl_set_session()
 *
 * \param session  The session structure to be populated. It must have been
 *                 initialised with mbedtls_ssl_session_init() but not
 *                 populated yet.
 * \param buf      The buffer holding the serialized session data. It must be a
 *                 readable buffer of at least \p len bytes.
 * \param len      The size of the serialized data in bytes.
 *
 * \return         \c 0 if successful.
 * \return         #MBEDTLS_ERR_SSL_ALLOC_FAILED if memory allocation failed.
 * \return         #MBEDTLS_ERR_SSL_BAD_INPUT_DATA if input data is invalid.
 * \return         #MBEDTLS_ERR_SSL_VERSION_MISMATCH if the serialized data
 *                 was generated in a different version or configuration of
 *                 Mbed TLS.
 * \return         Another negative value for other kinds of errors (for
 *                 example, unsupported features in the embedded certificate).
 */
int mbedtls_ssl_session_load(mbedtls_ssl_session *session,
                             const unsigned char *buf,
                             size_t len);

/**
 * \brief          Save session structure as serialized data in a buffer.
 *                 On client, this can be used for saving session data,
 *                 potentially in non-volatile storage, for resuming later.
 *                 On server, this can be used for alternative implementations
 *                 of session cache or session tickets.
 *
 * \see            mbedtls_ssl_session_load()
 * \see            mbedtls_ssl_get_session_pointer()
 *
 * \param session  The session structure to be saved.
 * \param buf      The buffer to write the serialized data to. It must be a
 *                 writeable buffer of at least \p buf_len bytes, or may be \c
 *                 NULL if \p buf_len is \c 0.
 * \param buf_len  The number of bytes available for writing in \p buf.
 * \param olen     The size in bytes of the data that has been or would have
 *                 been written. It must point to a valid \c size_t.
 *
 * \note           \p olen is updated to the correct value regardless of
 *                 whether \p buf_len was large enough. This makes it possible
 *                 to determine the necessary size by calling this function
 *                 with \p buf set to \c NULL and \p buf_len to \c 0.
 *
 * \return         \c 0 if successful.
 * \return         #MBEDTLS_ERR_SSL_BUFFER_TOO_SMALL if \p buf is too small.
 */
int mbedtls_ssl_session_save(const mbedtls_ssl_session *session,
                             unsigned char *buf,
                             size_t buf_len,
                             size_t *olen);

/**
 * \brief          Get a pointer to the current session structure, for example
 *                 to serialize it.
 *
 * \warning        Ownership of the session remains with the SSL context, and
 *                 the returned pointer is only guaranteed to be valid until
 *                 the next API call operating on the same \p ssl context.
 *
 * \see            mbedtls_ssl_session_save()
 *
 * \param ssl      The SSL context.
 *
 * \return         A pointer to the current session if successful.
 * \return         \c NULL if no session is active.
 */
const mbedtls_ssl_session *mbedtls_ssl_get_session_pointer(const mbedtls_ssl_context *ssl);

/**
 * \brief               Set the list of allowed ciphersuites and the preference
 *                      order. First in the list has the highest preference.
 *                      (Overrides all version-specific lists)
 *
 *                      The ciphersuites array is not copied, and must remain
 *                      valid for the lifetime of the ssl_config.
 *
 *                      Note: The server uses its own preferences
 *                      over the preference of the client unless
 *                      MBEDTLS_SSL_SRV_RESPECT_CLIENT_PREFERENCE is defined!
 *
 * \param conf          SSL configuration
 * \param ciphersuites  0-terminated list of allowed ciphersuites
 */
void mbedtls_ssl_conf_ciphersuites(mbedtls_ssl_config *conf,
                                   const int *ciphersuites);

#if defined(MBEDTLS_SSL_DTLS_CONNECTION_ID)
#define MBEDTLS_SSL_UNEXPECTED_CID_IGNORE 0
#define MBEDTLS_SSL_UNEXPECTED_CID_FAIL   1
/**
 * \brief               Specify the length of Connection IDs for incoming
 *                      encrypted DTLS records, as well as the behaviour
 *                      on unexpected CIDs.
 *
 *                      By default, the CID length is set to \c 0,
 *                      and unexpected CIDs are silently ignored.
 *
 * \param conf          The SSL configuration to modify.
 * \param len           The length in Bytes of the CID fields in encrypted
 *                      DTLS records using the CID mechanism. This must
 *                      not be larger than #MBEDTLS_SSL_CID_OUT_LEN_MAX.
 * \param ignore_other_cids This determines the stack's behaviour when
 *                          receiving a record with an unexpected CID.
 *                          Possible values are:
 *                          - #MBEDTLS_SSL_UNEXPECTED_CID_IGNORE
 *                            In this case, the record is silently ignored.
 *                          - #MBEDTLS_SSL_UNEXPECTED_CID_FAIL
 *                            In this case, the stack fails with the specific
 *                            error code #MBEDTLS_ERR_SSL_UNEXPECTED_CID.
 *
 * \note                The CID specification allows implementations to either
 *                      use a common length for all incoming connection IDs or
 *                      allow variable-length incoming IDs. Mbed TLS currently
 *                      requires a common length for all connections sharing the
 *                      same SSL configuration; this allows simpler parsing of
 *                      record headers.
 *
 * \return              \c 0 on success.
 * \return              #MBEDTLS_ERR_SSL_BAD_INPUT_DATA if \p len
 *                      is too large.
 */
int mbedtls_ssl_conf_cid(mbedtls_ssl_config *conf, size_t len,
                         int ignore_other_cids);
#endif /* MBEDTLS_SSL_DTLS_CONNECTION_ID */

/**
 * \brief               Set the list of allowed ciphersuites and the
 *                      preference order for a specific version of the protocol.
 *                      (Only useful on the server side)
 *
 *                      The ciphersuites array is not copied, and must remain
 *                      valid for the lifetime of the ssl_config.
 *
 * \param conf          SSL configuration
 * \param ciphersuites  0-terminated list of allowed ciphersuites
 * \param major         Major version number (only MBEDTLS_SSL_MAJOR_VERSION_3
 *                      supported)
 * \param minor         Minor version number (MBEDTLS_SSL_MINOR_VERSION_0,
 *                      MBEDTLS_SSL_MINOR_VERSION_1 and MBEDTLS_SSL_MINOR_VERSION_2,
 *                      MBEDTLS_SSL_MINOR_VERSION_3 supported)
 *
 * \note                With DTLS, use MBEDTLS_SSL_MINOR_VERSION_2 for DTLS 1.0
 *                      and MBEDTLS_SSL_MINOR_VERSION_3 for DTLS 1.2
 */
void mbedtls_ssl_conf_ciphersuites_for_version(mbedtls_ssl_config *conf,
                                               const int *ciphersuites,
                                               int major, int minor);

#if defined(MBEDTLS_X509_CRT_PARSE_C)
/**
 * \brief          Set the X.509 security profile used for verification
 *
 * \note           The restrictions are enforced for all certificates in the
 *                 chain. However, signatures in the handshake are not covered
 *                 by this setting but by \b mbedtls_ssl_conf_sig_hashes().
 *
 * \param conf     SSL configuration
 * \param profile  Profile to use
 */
void mbedtls_ssl_conf_cert_profile(mbedtls_ssl_config *conf,
                                   const mbedtls_x509_crt_profile *profile);

/**
 * \brief          Set the data required to verify peer certificate
 *
 * \note           See \c mbedtls_x509_crt_verify() for notes regarding the
 *                 parameters ca_chain (maps to trust_ca for that function)
 *                 and ca_crl.
 *
 * \param conf     SSL configuration
 * \param ca_chain trusted CA chain (meaning all fully trusted top-level CAs)
 * \param ca_crl   trusted CA CRLs
 */
void mbedtls_ssl_conf_ca_chain(mbedtls_ssl_config *conf,
                               mbedtls_x509_crt *ca_chain,
                               mbedtls_x509_crl *ca_crl);

#if defined(MBEDTLS_X509_TRUSTED_CERTIFICATE_CALLBACK)
/**
 * \brief          Set the trusted certificate callback.
 *
 *                 This API allows to register the set of trusted certificates
 *                 through a callback, instead of a linked list as configured
 *                 by mbedtls_ssl_conf_ca_chain().
 *
 *                 This is useful for example in contexts where a large number
 *                 of CAs are used, and the inefficiency of maintaining them
 *                 in a linked list cannot be tolerated. It is also useful when
 *                 the set of trusted CAs needs to be modified frequently.
 *
 *                 See the documentation of `mbedtls_x509_crt_ca_cb_t` for
 *                 more information.
 *
 * \param conf     The SSL configuration to register the callback with.
 * \param f_ca_cb  The trusted certificate callback to use when verifying
 *                 certificate chains.
 * \param p_ca_cb  The context to be passed to \p f_ca_cb (for example,
 *                 a reference to a trusted CA database).
 *
 * \note           This API is incompatible with mbedtls_ssl_conf_ca_chain():
 *                 Any call to this function overwrites the values set through
 *                 earlier calls to mbedtls_ssl_conf_ca_chain() or
 *                 mbedtls_ssl_conf_ca_cb().
 *
 * \note           This API is incompatible with CA indication in
 *                 CertificateRequest messages: A server-side SSL context which
 *                 is bound to an SSL configuration that uses a CA callback
 *                 configured via mbedtls_ssl_conf_ca_cb(), and which requires
 *                 client authentication, will send an empty CA list in the
 *                 corresponding CertificateRequest message.
 *
 * \note           This API is incompatible with mbedtls_ssl_set_hs_ca_chain():
 *                 If an SSL context is bound to an SSL configuration which uses
 *                 CA callbacks configured via mbedtls_ssl_conf_ca_cb(), then
 *                 calls to mbedtls_ssl_set_hs_ca_chain() have no effect.
 *
 * \note           The use of this API disables the use of restartable ECC
 *                 during X.509 CRT signature verification (but doesn't affect
 *                 other uses).
 *
 * \warning        This API is incompatible with the use of CRLs. Any call to
 *                 mbedtls_ssl_conf_ca_cb() unsets CRLs configured through
 *                 earlier calls to mbedtls_ssl_conf_ca_chain().
 *
 * \warning        In multi-threaded environments, the callback \p f_ca_cb
 *                 must be thread-safe, and it is the user's responsibility
 *                 to guarantee this (for example through a mutex
 *                 contained in the callback context pointed to by \p p_ca_cb).
 */
void mbedtls_ssl_conf_ca_cb(mbedtls_ssl_config *conf,
                            mbedtls_x509_crt_ca_cb_t f_ca_cb,
                            void *p_ca_cb);
#endif /* MBEDTLS_X509_TRUSTED_CERTIFICATE_CALLBACK */

/**
 * \brief          Set own certificate chain and private key
 *
 * \note           own_cert should contain in order from the bottom up your
 *                 certificate chain. The top certificate (self-signed)
 *                 can be omitted.
 *
 * \note           On server, this function can be called multiple times to
 *                 provision more than one cert/key pair (eg one ECDSA, one
 *                 RSA with SHA-256, one RSA with SHA-1). An adequate
 *                 certificate will be selected according to the client's
 *                 advertised capabilities. In case multiple certificates are
 *                 adequate, preference is given to the one set by the first
 *                 call to this function, then second, etc.
 *
 * \note           On client, only the first call has any effect. That is,
 *                 only one client certificate can be provisioned. The
 *                 server's preferences in its CertificateRequest message will
 *                 be ignored and our only cert will be sent regardless of
 *                 whether it matches those preferences - the server can then
 *                 decide what it wants to do with it.
 *
 * \note           The provided \p pk_key needs to match the public key in the
 *                 first certificate in \p own_cert, or all handshakes using
 *                 that certificate will fail. It is your responsibility
 *                 to ensure that; this function will not perform any check.
 *                 You may use mbedtls_pk_check_pair() in order to perform
 *                 this check yourself, but be aware that this function can
 *                 be computationally expensive on some key types.
 *
 * \param conf     SSL configuration
 * \param own_cert own public certificate chain
 * \param pk_key   own private key
 *
 * \return         0 on success or MBEDTLS_ERR_SSL_ALLOC_FAILED
 */
int mbedtls_ssl_conf_own_cert(mbedtls_ssl_config *conf,
                              mbedtls_x509_crt *own_cert,
                              mbedtls_pk_context *pk_key);
#endif /* MBEDTLS_X509_CRT_PARSE_C */

#if defined(MBEDTLS_KEY_EXCHANGE_SOME_PSK_ENABLED)
/**
 * \brief          Configure a pre-shared key (PSK) and identity
 *                 to be used in PSK-based ciphersuites.
 *
 * \note           This is mainly useful for clients. Servers will usually
 *                 want to use \c mbedtls_ssl_conf_psk_cb() instead.
 *
 * \note           A PSK set by \c mbedtls_ssl_set_hs_psk() in the PSK callback
 *                 takes precedence over a PSK configured by this function.
 *
 * \warning        Currently, clients can only register a single pre-shared key.
 *                 Calling this function or mbedtls_ssl_conf_psk_opaque() more
 *                 than once will overwrite values configured in previous calls.
 *                 Support for setting multiple PSKs on clients and selecting
 *                 one based on the identity hint is not a planned feature,
 *                 but feedback is welcomed.
 *
 * \param conf     The SSL configuration to register the PSK with.
 * \param psk      The pointer to the pre-shared key to use.
 * \param psk_len  The length of the pre-shared key in bytes.
 * \param psk_identity      The pointer to the pre-shared key identity.
 * \param psk_identity_len  The length of the pre-shared key identity
 *                          in bytes.
 *
 * \note           The PSK and its identity are copied internally and
 *                 hence need not be preserved by the caller for the lifetime
 *                 of the SSL configuration.
 *
 * \return         \c 0 if successful.
 * \return         An \c MBEDTLS_ERR_SSL_XXX error code on failure.
 */
int mbedtls_ssl_conf_psk(mbedtls_ssl_config *conf,
                         const unsigned char *psk, size_t psk_len,
                         const unsigned char *psk_identity, size_t psk_identity_len);

#if defined(MBEDTLS_USE_PSA_CRYPTO)
/**
 * \brief          Configure an opaque pre-shared key (PSK) and identity
 *                 to be used in PSK-based ciphersuites.
 *
 * \note           This is mainly useful for clients. Servers will usually
 *                 want to use \c mbedtls_ssl_conf_psk_cb() instead.
 *
 * \note           An opaque PSK set by \c mbedtls_ssl_set_hs_psk_opaque() in
 *                 the PSK callback takes precedence over an opaque PSK
 *                 configured by this function.
 *
 * \warning        Currently, clients can only register a single pre-shared key.
 *                 Calling this function or mbedtls_ssl_conf_psk() more than
 *                 once will overwrite values configured in previous calls.
 *                 Support for setting multiple PSKs on clients and selecting
 *                 one based on the identity hint is not a planned feature,
 *                 but feedback is welcomed.
 *
 * \param conf     The SSL configuration to register the PSK with.
 * \param psk      The identifier of the key slot holding the PSK.
 *                 Until \p conf is destroyed or this function is successfully
 *                 called again, the key slot \p psk must be populated with a
 *                 key of type PSA_ALG_CATEGORY_KEY_DERIVATION whose policy
 *                 allows its use for the key derivation algorithm applied
 *                 in the handshake.
 * \param psk_identity      The pointer to the pre-shared key identity.
 * \param psk_identity_len  The length of the pre-shared key identity
 *                          in bytes.
 *
 * \note           The PSK identity hint is copied internally and hence need
 *                 not be preserved by the caller for the lifetime of the
 *                 SSL configuration.
 *
 * \return         \c 0 if successful.
 * \return         An \c MBEDTLS_ERR_SSL_XXX error code on failure.
 */
int mbedtls_ssl_conf_psk_opaque(mbedtls_ssl_config *conf,
                                psa_key_id_t psk,
                                const unsigned char *psk_identity,
                                size_t psk_identity_len);
#endif /* MBEDTLS_USE_PSA_CRYPTO */

/**
 * \brief          Set the pre-shared Key (PSK) for the current handshake.
 *
 * \note           This should only be called inside the PSK callback,
 *                 i.e. the function passed to \c mbedtls_ssl_conf_psk_cb().
 *
 * \note           A PSK set by this function takes precedence over a PSK
 *                 configured by \c mbedtls_ssl_conf_psk().
 *
 * \param ssl      The SSL context to configure a PSK for.
 * \param psk      The pointer to the pre-shared key.
 * \param psk_len  The length of the pre-shared key in bytes.
 *
 * \return         \c 0 if successful.
 * \return         An \c MBEDTLS_ERR_SSL_XXX error code on failure.
 */
int mbedtls_ssl_set_hs_psk(mbedtls_ssl_context *ssl,
                           const unsigned char *psk, size_t psk_len);

#if defined(MBEDTLS_USE_PSA_CRYPTO)
/**
 * \brief          Set an opaque pre-shared Key (PSK) for the current handshake.
 *
 * \note           This should only be called inside the PSK callback,
 *                 i.e. the function passed to \c mbedtls_ssl_conf_psk_cb().
 *
 * \note           An opaque PSK set by this function takes precedence over an
 *                 opaque PSK configured by \c mbedtls_ssl_conf_psk_opaque().
 *
 * \param ssl      The SSL context to configure a PSK for.
 * \param psk      The identifier of the key slot holding the PSK.
 *                 For the duration of the current handshake, the key slot
 *                 must be populated with a key of type
 *                 PSA_ALG_CATEGORY_KEY_DERIVATION whose policy allows its
 *                 use for the key derivation algorithm
 *                 applied in the handshake.
 *
 * \return         \c 0 if successful.
 * \return         An \c MBEDTLS_ERR_SSL_XXX error code on failure.
 */
int mbedtls_ssl_set_hs_psk_opaque(mbedtls_ssl_context *ssl,
                                  psa_key_id_t psk);
#endif /* MBEDTLS_USE_PSA_CRYPTO */

/**
 * \brief          Set the PSK callback (server-side only).
 *
 *                 If set, the PSK callback is called for each
 *                 handshake where a PSK-based ciphersuite was negotiated.
 *                 The caller provides the identity received and wants to
 *                 receive the actual PSK data and length.
 *
 *                 The callback has the following parameters:
 *                 - \c void*: The opaque pointer \p p_psk.
 *                 - \c mbedtls_ssl_context*: The SSL context to which
 *                                            the operation applies.
 *                 - \c const unsigned char*: The PSK identity
 *                                            selected by the client.
 *                 - \c size_t: The length of the PSK identity
 *                              selected by the client.
 *
 *                 If a valid PSK identity is found, the callback should use
 *                 \c mbedtls_ssl_set_hs_psk() or
 *                 \c mbedtls_ssl_set_hs_psk_opaque()
 *                 on the SSL context to set the correct PSK and return \c 0.
 *                 Any other return value will result in a denied PSK identity.
 *
 * \note           A dynamic PSK (i.e. set by the PSK callback) takes
 *                 precedence over a static PSK (i.e. set by
 *                 \c mbedtls_ssl_conf_psk() or
 *                 \c mbedtls_ssl_conf_psk_opaque()).
 *                 This means that if you set a PSK callback using this
 *                 function, you don't need to set a PSK using
 *                 \c mbedtls_ssl_conf_psk() or
 *                 \c mbedtls_ssl_conf_psk_opaque()).
 *
 * \param conf     The SSL configuration to register the callback with.
 * \param f_psk    The callback for selecting and setting the PSK based
 *                 in the PSK identity chosen by the client.
 * \param p_psk    A pointer to an opaque structure to be passed to
 *                 the callback, for example a PSK store.
 */
void mbedtls_ssl_conf_psk_cb(mbedtls_ssl_config *conf,
                             int (*f_psk)(void *, mbedtls_ssl_context *, const unsigned char *,
                                          size_t),
                             void *p_psk);
#endif /* MBEDTLS_KEY_EXCHANGE_SOME_PSK_ENABLED */

#if defined(MBEDTLS_DHM_C) && defined(MBEDTLS_SSL_SRV_C)

#if !defined(MBEDTLS_DEPRECATED_REMOVED)

#if defined(MBEDTLS_DEPRECATED_WARNING)
#define MBEDTLS_DEPRECATED    __attribute__((deprecated))
#else
#define MBEDTLS_DEPRECATED
#endif

/**
 * \brief          Set the Diffie-Hellman public P and G values,
 *                 read as hexadecimal strings (server-side only)
 *                 (Default values: MBEDTLS_DHM_RFC3526_MODP_2048_[PG])
 *
 * \param conf     SSL configuration
 * \param dhm_P    Diffie-Hellman-Merkle modulus
 * \param dhm_G    Diffie-Hellman-Merkle generator
 *
 * \deprecated     Superseded by \c mbedtls_ssl_conf_dh_param_bin.
 *
 * \return         0 if successful
 */
MBEDTLS_DEPRECATED int mbedtls_ssl_conf_dh_param(mbedtls_ssl_config *conf,
                                                 const char *dhm_P,
                                                 const char *dhm_G);

#endif /* MBEDTLS_DEPRECATED_REMOVED */

/**
 * \brief          Set the Diffie-Hellman public P and G values
 *                 from big-endian binary presentations.
 *                 (Default values: MBEDTLS_DHM_RFC3526_MODP_2048_[PG]_BIN)
 *
 * \param conf     SSL configuration
 * \param dhm_P    Diffie-Hellman-Merkle modulus in big-endian binary form
 * \param P_len    Length of DHM modulus
 * \param dhm_G    Diffie-Hellman-Merkle generator in big-endian binary form
 * \param G_len    Length of DHM generator
 *
 * \return         0 if successful
 */
int mbedtls_ssl_conf_dh_param_bin(mbedtls_ssl_config *conf,
                                  const unsigned char *dhm_P, size_t P_len,
                                  const unsigned char *dhm_G,  size_t G_len);

/**
 * \brief          Set the Diffie-Hellman public P and G values,
 *                 read from existing context (server-side only)
 *
 * \param conf     SSL configuration
 * \param dhm_ctx  Diffie-Hellman-Merkle context
 *
 * \return         0 if successful
 */
int mbedtls_ssl_conf_dh_param_ctx(mbedtls_ssl_config *conf, mbedtls_dhm_context *dhm_ctx);
#endif /* MBEDTLS_DHM_C && defined(MBEDTLS_SSL_SRV_C) */

#if defined(MBEDTLS_DHM_C) && defined(MBEDTLS_SSL_CLI_C)
/**
 * \brief          Set the minimum length for Diffie-Hellman parameters.
 *                 (Client-side only.)
 *                 (Default: 1024 bits.)
 *
 * \param conf     SSL configuration
 * \param bitlen   Minimum bit length of the DHM prime
 */
void mbedtls_ssl_conf_dhm_min_bitlen(mbedtls_ssl_config *conf,
                                     unsigned int bitlen);
#endif /* MBEDTLS_DHM_C && MBEDTLS_SSL_CLI_C */

#if defined(MBEDTLS_ECP_C)
/**
 * \brief          Set the allowed curves in order of preference.
 *                 (Default: all defined curves in order of decreasing size,
 *                 except that Montgomery curves come last. This order
 *                 is likely to change in a future version.)
 *
 *                 On server: this only affects selection of the ECDHE curve;
 *                 the curves used for ECDH and ECDSA are determined by the
 *                 list of available certificates instead.
 *
 *                 On client: this affects the list of curves offered for any
 *                 use. The server can override our preference order.
 *
 *                 Both sides: limits the set of curves accepted for use in
 *                 ECDHE and in the peer's end-entity certificate.
 *
 * \note           This has no influence on which curves are allowed inside the
 *                 certificate chains, see \c mbedtls_ssl_conf_cert_profile()
 *                 for that. For the end-entity certificate however, the key
 *                 will be accepted only if it is allowed both by this list
 *                 and by the cert profile.
 *
 * \note           This list should be ordered by decreasing preference
 *                 (preferred curve first).
 *
 * \param conf     SSL configuration
 * \param curves   Ordered list of allowed curves,
 *                 terminated by MBEDTLS_ECP_DP_NONE.
 */
void mbedtls_ssl_conf_curves(mbedtls_ssl_config *conf,
                             const mbedtls_ecp_group_id *curves);
#endif /* MBEDTLS_ECP_C */

#if defined(MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED)
/**
 * \brief          Set the allowed hashes for signatures during the handshake.
 *                 (Default: all SHA-2 hashes, largest first. Also SHA-1 if
 *                 the compile-time option
 *                 `MBEDTLS_TLS_DEFAULT_ALLOW_SHA1_IN_KEY_EXCHANGE` is enabled.)
 *
 * \note           This only affects which hashes are offered and can be used
 *                 for signatures during the handshake. Hashes for message
 *                 authentication and the TLS PRF are controlled by the
 *                 ciphersuite, see \c mbedtls_ssl_conf_ciphersuites(). Hashes
 *                 used for certificate signature are controlled by the
 *                 verification profile, see \c mbedtls_ssl_conf_cert_profile().
 *
 * \note           This list should be ordered by decreasing preference
 *                 (preferred hash first).
 *
 * \param conf     SSL configuration
 * \param hashes   Ordered list of allowed signature hashes,
 *                 terminated by \c MBEDTLS_MD_NONE.
 */
void mbedtls_ssl_conf_sig_hashes(mbedtls_ssl_config *conf,
                                 const int *hashes);
#endif /* MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED */

#if defined(MBEDTLS_X509_CRT_PARSE_C)
/**
 * \brief          Set or reset the hostname to check against the received
 *                 server certificate. It sets the ServerName TLS extension,
 *                 too, if that extension is enabled. (client-side only)
 *
 * \param ssl      SSL context
 * \param hostname the server hostname, may be NULL to clear hostname

 * \note           Maximum hostname length MBEDTLS_SSL_MAX_HOST_NAME_LEN.
 *
 * \return         0 if successful, MBEDTLS_ERR_SSL_ALLOC_FAILED on
 *                 allocation failure, MBEDTLS_ERR_SSL_BAD_INPUT_DATA on
 *                 too long input hostname.
 *
 *                 Hostname set to the one provided on success (cleared
 *                 when NULL). On allocation failure hostname is cleared.
 *                 On too long input failure, old hostname is unchanged.
 */
int mbedtls_ssl_set_hostname(mbedtls_ssl_context *ssl, const char *hostname);
#endif /* MBEDTLS_X509_CRT_PARSE_C */

#if defined(MBEDTLS_SSL_SERVER_NAME_INDICATION)
/**
 * \brief          Set own certificate and key for the current handshake
 *
 * \note           Same as \c mbedtls_ssl_conf_own_cert() but for use within
 *                 the SNI callback.
 *
 * \param ssl      SSL context
 * \param own_cert own public certificate chain
 * \param pk_key   own private key
 *
 * \return         0 on success or MBEDTLS_ERR_SSL_ALLOC_FAILED
 */
int mbedtls_ssl_set_hs_own_cert(mbedtls_ssl_context *ssl,
                                mbedtls_x509_crt *own_cert,
                                mbedtls_pk_context *pk_key);

/**
 * \brief          Set the data required to verify peer certificate for the
 *                 current handshake
 *
 * \note           Same as \c mbedtls_ssl_conf_ca_chain() but for use within
 *                 the SNI callback.
 *
 * \param ssl      SSL context
 * \param ca_chain trusted CA chain (meaning all fully trusted top-level CAs)
 * \param ca_crl   trusted CA CRLs
 */
void mbedtls_ssl_set_hs_ca_chain(mbedtls_ssl_context *ssl,
                                 mbedtls_x509_crt *ca_chain,
                                 mbedtls_x509_crl *ca_crl);

/**
 * \brief          Set authmode for the current handshake.
 *
 * \note           Same as \c mbedtls_ssl_conf_authmode() but for use within
 *                 the SNI callback.
 *
 * \param ssl      SSL context
 * \param authmode MBEDTLS_SSL_VERIFY_NONE, MBEDTLS_SSL_VERIFY_OPTIONAL or
 *                 MBEDTLS_SSL_VERIFY_REQUIRED
 */
void mbedtls_ssl_set_hs_authmode(mbedtls_ssl_context *ssl,
                                 int authmode);

/**
 * \brief          Set server side ServerName TLS extension callback
 *                 (optional, server-side only).
 *
 *                 If set, the ServerName callback is called whenever the
 *                 server receives a ServerName TLS extension from the client
 *                 during a handshake. The ServerName callback has the
 *                 following parameters: (void *parameter, mbedtls_ssl_context *ssl,
 *                 const unsigned char *hostname, size_t len). If a suitable
 *                 certificate is found, the callback must set the
 *                 certificate(s) and key(s) to use with \c
 *                 mbedtls_ssl_set_hs_own_cert() (can be called repeatedly),
 *                 and may optionally adjust the CA and associated CRL with \c
 *                 mbedtls_ssl_set_hs_ca_chain() as well as the client
 *                 authentication mode with \c mbedtls_ssl_set_hs_authmode(),
 *                 then must return 0. If no matching name is found, the
 *                 callback must either set a default cert, or
 *                 return non-zero to abort the handshake at this point.
 *
 * \param conf     SSL configuration
 * \param f_sni    verification function
 * \param p_sni    verification parameter
 */
void mbedtls_ssl_conf_sni(mbedtls_ssl_config *conf,
                          int (*f_sni)(void *, mbedtls_ssl_context *, const unsigned char *,
                                       size_t),
                          void *p_sni);
#endif /* MBEDTLS_SSL_SERVER_NAME_INDICATION */

#if defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
/**
 * \brief          Set the EC J-PAKE password for current handshake.
 *
 * \note           An internal copy is made, and destroyed as soon as the
 *                 handshake is completed, or when the SSL context is reset or
 *                 freed.
 *
 * \note           The SSL context needs to be already set up. The right place
 *                 to call this function is between \c mbedtls_ssl_setup() or
 *                 \c mbedtls_ssl_reset() and \c mbedtls_ssl_handshake().
 *
 * \param ssl      SSL context
 * \param pw       EC J-PAKE password (pre-shared secret)
 * \param pw_len   length of pw in bytes
 *
 * \return         0 on success, or a negative error code.
 */
int mbedtls_ssl_set_hs_ecjpake_password(mbedtls_ssl_context *ssl,
                                        const unsigned char *pw,
                                        size_t pw_len);
#endif /*MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED */

#if defined(MBEDTLS_SSL_ALPN)
/**
 * \brief          Set the supported Application Layer Protocols.
 *
 * \param conf     SSL configuration
 * \param protos   Pointer to a NULL-terminated list of supported protocols,
 *                 in decreasing preference order. The pointer to the list is
 *                 recorded by the library for later reference as required, so
 *                 the lifetime of the table must be at least as long as the
 *                 lifetime of the SSL configuration structure.
 *
 * \return         0 on success, or MBEDTLS_ERR_SSL_BAD_INPUT_DATA.
 */
int mbedtls_ssl_conf_alpn_protocols(mbedtls_ssl_config *conf, const char **protos);

/**
 * \brief          Get the name of the negotiated Application Layer Protocol.
 *                 This function should be called after the handshake is
 *                 completed.
 *
 * \param ssl      SSL context
 *
 * \return         Protocol name, or NULL if no protocol was negotiated.
 */
const char *mbedtls_ssl_get_alpn_protocol(const mbedtls_ssl_context *ssl);
#endif /* MBEDTLS_SSL_ALPN */

#if defined(MBEDTLS_SSL_DTLS_SRTP)
#if defined(MBEDTLS_DEBUG_C)
static inline const char *mbedtls_ssl_get_srtp_profile_as_string(mbedtls_ssl_srtp_profile profile)
{
    switch (profile) {
        case MBEDTLS_TLS_SRTP_AES128_CM_HMAC_SHA1_80:
            return "MBEDTLS_TLS_SRTP_AES128_CM_HMAC_SHA1_80";
        case MBEDTLS_TLS_SRTP_AES128_CM_HMAC_SHA1_32:
            return "MBEDTLS_TLS_SRTP_AES128_CM_HMAC_SHA1_32";
        case MBEDTLS_TLS_SRTP_NULL_HMAC_SHA1_80:
            return "MBEDTLS_TLS_SRTP_NULL_HMAC_SHA1_80";
        case MBEDTLS_TLS_SRTP_NULL_HMAC_SHA1_32:
            return "MBEDTLS_TLS_SRTP_NULL_HMAC_SHA1_32";
        default: break;
    }
    return "";
}
#endif /* MBEDTLS_DEBUG_C */
/**
 * \brief                   Manage support for mki(master key id) value
 *                          in use_srtp extension.
 *                          MKI is an optional part of SRTP used for key management
 *                          and re-keying. See RFC3711 section 3.1 for details.
 *                          The default value is
 *                          #MBEDTLS_SSL_DTLS_SRTP_MKI_UNSUPPORTED.
 *
 * \param conf              The SSL configuration to manage mki support.
 * \param support_mki_value Enable or disable mki usage. Values are
 *                          #MBEDTLS_SSL_DTLS_SRTP_MKI_UNSUPPORTED
 *                          or #MBEDTLS_SSL_DTLS_SRTP_MKI_SUPPORTED.
 */
void mbedtls_ssl_conf_srtp_mki_value_supported(mbedtls_ssl_config *conf,
                                               int support_mki_value);

/**
 * \brief                   Set the supported DTLS-SRTP protection profiles.
 *
 * \param conf              SSL configuration
 * \param profiles          Pointer to a List of MBEDTLS_TLS_SRTP_UNSET terminated
 *                          supported protection profiles
 *                          in decreasing preference order.
 *                          The pointer to the list is recorded by the library
 *                          for later reference as required, so the lifetime
 *                          of the table must be at least as long as the lifetime
 *                          of the SSL configuration structure.
 *                          The list must not hold more than
 *                          MBEDTLS_TLS_SRTP_MAX_PROFILE_LIST_LENGTH elements
 *                          (excluding the terminating MBEDTLS_TLS_SRTP_UNSET).
 *
 * \return                  0 on success
 * \return                  #MBEDTLS_ERR_SSL_BAD_INPUT_DATA when the list of
 *                          protection profiles is incorrect.
 */
int mbedtls_ssl_conf_dtls_srtp_protection_profiles
    (mbedtls_ssl_config *conf,
    const mbedtls_ssl_srtp_profile *profiles);

/**
 * \brief                  Set the mki_value for the current DTLS-SRTP session.
 *
 * \param ssl              SSL context to use.
 * \param mki_value        The MKI value to set.
 * \param mki_len          The length of the MKI value.
 *
 * \note                   This function is relevant on client side only.
 *                         The server discovers the mki value during handshake.
 *                         A mki value set on server side using this function
 *                         is ignored.
 *
 * \return                 0 on success
 * \return                 #MBEDTLS_ERR_SSL_BAD_INPUT_DATA
 * \return                 #MBEDTLS_ERR_SSL_FEATURE_UNAVAILABLE
 */
int mbedtls_ssl_dtls_srtp_set_mki_value(mbedtls_ssl_context *ssl,
                                        unsigned char *mki_value,
                                        uint16_t mki_len);
/**
 * \brief                  Get the negotiated DTLS-SRTP information:
 *                         Protection profile and MKI value.
 *
 * \warning                This function must be called after the handshake is
 *                         completed. The value returned by this function must
 *                         not be trusted or acted upon before the handshake completes.
 *
 * \param ssl              The SSL context to query.
 * \param dtls_srtp_info   The negotiated DTLS-SRTP information:
 *                         - Protection profile in use.
 *                         A direct mapping of the iana defined value for protection
 *                         profile on an uint16_t.
                   http://www.iana.org/assignments/srtp-protection/srtp-protection.xhtml
 *                         #MBEDTLS_TLS_SRTP_UNSET if the use of SRTP was not negotiated
 *                         or peer's Hello packet was not parsed yet.
 *                         - mki size and value( if size is > 0 ).
 */
void mbedtls_ssl_get_dtls_srtp_negotiation_result(const mbedtls_ssl_context *ssl,
                                                  mbedtls_dtls_srtp_info *dtls_srtp_info);
#endif /* MBEDTLS_SSL_DTLS_SRTP */

/**
 * \brief          Set the maximum supported version sent from the client side
 *                 and/or accepted at the server side
 *                 (Default: MBEDTLS_SSL_MAX_MAJOR_VERSION, MBEDTLS_SSL_MAX_MINOR_VERSION)
 *
 * \note           This ignores ciphersuites from higher versions.
 *
 * \note           With DTLS, use MBEDTLS_SSL_MINOR_VERSION_2 for DTLS 1.0 and
 *                 MBEDTLS_SSL_MINOR_VERSION_3 for DTLS 1.2
 *
 * \param conf     SSL configuration
 * \param major    Major version number (only MBEDTLS_SSL_MAJOR_VERSION_3 supported)
 * \param minor    Minor version number (MBEDTLS_SSL_MINOR_VERSION_0,
 *                 MBEDTLS_SSL_MINOR_VERSION_1 and MBEDTLS_SSL_MINOR_VERSION_2,
 *                 MBEDTLS_SSL_MINOR_VERSION_3 supported)
 */
void mbedtls_ssl_conf_max_version(mbedtls_ssl_config *conf, int major, int minor);

/**
 * \brief          Set the minimum accepted SSL/TLS protocol version
 *                 (Default: TLS 1.0)
 *
 * \note           Input outside of the SSL_MAX_XXXXX_VERSION and
 *                 SSL_MIN_XXXXX_VERSION range is ignored.
 *
 * \note           MBEDTLS_SSL_MINOR_VERSION_0 (SSL v3) should be avoided.
 *
 * \note           With DTLS, use MBEDTLS_SSL_MINOR_VERSION_2 for DTLS 1.0 and
 *                 MBEDTLS_SSL_MINOR_VERSION_3 for DTLS 1.2
 *
 * \param conf     SSL configuration
 * \param major    Major version number (only MBEDTLS_SSL_MAJOR_VERSION_3 supported)
 * \param minor    Minor version number (MBEDTLS_SSL_MINOR_VERSION_0,
 *                 MBEDTLS_SSL_MINOR_VERSION_1 and MBEDTLS_SSL_MINOR_VERSION_2,
 *                 MBEDTLS_SSL_MINOR_VERSION_3 supported)
 */
void mbedtls_ssl_conf_min_version(mbedtls_ssl_config *conf, int major, int minor);

#if defined(MBEDTLS_SSL_FALLBACK_SCSV) && defined(MBEDTLS_SSL_CLI_C)
/**
 * \brief          Set the fallback flag (client-side only).
 *                 (Default: MBEDTLS_SSL_IS_NOT_FALLBACK).
 *
 * \note           Set to MBEDTLS_SSL_IS_FALLBACK when preparing a fallback
 *                 connection, that is a connection with max_version set to a
 *                 lower value than the value you're willing to use. Such
 *                 fallback connections are not recommended but are sometimes
 *                 necessary to interoperate with buggy (version-intolerant)
 *                 servers.
 *
 * \warning        You should NOT set this to MBEDTLS_SSL_IS_FALLBACK for
 *                 non-fallback connections! This would appear to work for a
 *                 while, then cause failures when the server is upgraded to
 *                 support a newer TLS version.
 *
 * \param conf     SSL configuration
 * \param fallback MBEDTLS_SSL_IS_NOT_FALLBACK or MBEDTLS_SSL_IS_FALLBACK
 */
void mbedtls_ssl_conf_fallback(mbedtls_ssl_config *conf, char fallback);
#endif /* MBEDTLS_SSL_FALLBACK_SCSV && MBEDTLS_SSL_CLI_C */

#if defined(MBEDTLS_SSL_ENCRYPT_THEN_MAC)
/**
 * \brief           Enable or disable Encrypt-then-MAC
 *                  (Default: MBEDTLS_SSL_ETM_ENABLED)
 *
 * \note            This should always be enabled, it is a security
 *                  improvement, and should not cause any interoperability
 *                  issue (used only if the peer supports it too).
 *
 * \param conf      SSL configuration
 * \param etm       MBEDTLS_SSL_ETM_ENABLED or MBEDTLS_SSL_ETM_DISABLED
 */
void mbedtls_ssl_conf_encrypt_then_mac(mbedtls_ssl_config *conf, char etm);
#endif /* MBEDTLS_SSL_ENCRYPT_THEN_MAC */

#if defined(MBEDTLS_SSL_EXTENDED_MASTER_SECRET)
/**
 * \brief           Enable or disable Extended Master Secret negotiation.
 *                  (Default: MBEDTLS_SSL_EXTENDED_MS_ENABLED)
 *
 * \note            This should always be enabled, it is a security fix to the
 *                  protocol, and should not cause any interoperability issue
 *                  (used only if the peer supports it too).
 *
 * \param conf      SSL configuration
 * \param ems       MBEDTLS_SSL_EXTENDED_MS_ENABLED or MBEDTLS_SSL_EXTENDED_MS_DISABLED
 */
void mbedtls_ssl_conf_extended_master_secret(mbedtls_ssl_config *conf, char ems);
#endif /* MBEDTLS_SSL_EXTENDED_MASTER_SECRET */

#if defined(MBEDTLS_ARC4_C)
/**
 * \brief          Disable or enable support for RC4
 *                 (Default: MBEDTLS_SSL_ARC4_DISABLED)
 *
 * \warning        Use of RC4 in DTLS/TLS has been prohibited by RFC 7465
 *                 for security reasons. Use at your own risk.
 *
 * \note           This function is deprecated and will be removed in
 *                 a future version of the library.
 *                 RC4 is disabled by default at compile time and needs to be
 *                 actively enabled for use with legacy systems.
 *
 * \param conf     SSL configuration
 * \param arc4     MBEDTLS_SSL_ARC4_ENABLED or MBEDTLS_SSL_ARC4_DISABLED
 */
void mbedtls_ssl_conf_arc4_support(mbedtls_ssl_config *conf, char arc4);
#endif /* MBEDTLS_ARC4_C */

#if defined(MBEDTLS_SSL_SRV_C)
/**
 * \brief          Whether to send a list of acceptable CAs in
 *                 CertificateRequest messages.
 *                 (Default: do send)
 *
 * \param conf     SSL configuration
 * \param cert_req_ca_list   MBEDTLS_SSL_CERT_REQ_CA_LIST_ENABLED or
 *                          MBEDTLS_SSL_CERT_REQ_CA_LIST_DISABLED
 */
void mbedtls_ssl_conf_cert_req_ca_list(mbedtls_ssl_config *conf,
                                       char cert_req_ca_list);
#endif /* MBEDTLS_SSL_SRV_C */

#if defined(MBEDTLS_SSL_MAX_FRAGMENT_LENGTH)
/**
 * \brief          Set the maximum fragment length to emit and/or negotiate.
 *                 (Typical: the smaller of #MBEDTLS_SSL_IN_CONTENT_LEN and
 *                 #MBEDTLS_SSL_OUT_CONTENT_LEN, usually `2^14` bytes)
 *                 (Server: set maximum fragment length to emit,
 *                 usually negotiated by the client during handshake)
 *                 (Client: set maximum fragment length to emit *and*
 *                 negotiate with the server during handshake)
 *                 (Default: #MBEDTLS_SSL_MAX_FRAG_LEN_NONE)
 *
 * \note           On the client side, the maximum fragment length extension
 *                 *will not* be used, unless the maximum fragment length has
 *                 been set via this function to a value different than
 *                 #MBEDTLS_SSL_MAX_FRAG_LEN_NONE.
 *
 * \note           With TLS, this currently only affects ApplicationData (sent
 *                 with \c mbedtls_ssl_read()), not handshake messages.
 *                 With DTLS, this affects both ApplicationData and handshake.
 *
 * \note           This sets the maximum length for a record's payload,
 *                 excluding record overhead that will be added to it, see
 *                 \c mbedtls_ssl_get_record_expansion().
 *
 * \note           For DTLS, it is also possible to set a limit for the total
 *                 size of datagrams passed to the transport layer, including
 *                 record overhead, see \c mbedtls_ssl_set_mtu().
 *
 * \param conf     SSL configuration
 * \param mfl_code Code for maximum fragment length (allowed values:
 *                 MBEDTLS_SSL_MAX_FRAG_LEN_512,  MBEDTLS_SSL_MAX_FRAG_LEN_1024,
 *                 MBEDTLS_SSL_MAX_FRAG_LEN_2048, MBEDTLS_SSL_MAX_FRAG_LEN_4096)
 *
 * \return         0 if successful or MBEDTLS_ERR_SSL_BAD_INPUT_DATA
 */
int mbedtls_ssl_conf_max_frag_len(mbedtls_ssl_config *conf, unsigned char mfl_code);
#endif /* MBEDTLS_SSL_MAX_FRAGMENT_LENGTH */

#if defined(MBEDTLS_SSL_TRUNCATED_HMAC)
/**
 * \brief          Activate negotiation of truncated HMAC
 *                 (Default: MBEDTLS_SSL_TRUNC_HMAC_DISABLED)
 *
 * \param conf     SSL configuration
 * \param truncate Enable or disable (MBEDTLS_SSL_TRUNC_HMAC_ENABLED or
 *                                    MBEDTLS_SSL_TRUNC_HMAC_DISABLED)
 */
void mbedtls_ssl_conf_truncated_hmac(mbedtls_ssl_config *conf, int truncate);
#endif /* MBEDTLS_SSL_TRUNCATED_HMAC */

#if defined(MBEDTLS_SSL_CBC_RECORD_SPLITTING)
/**
 * \brief          Enable / Disable 1/n-1 record splitting
 *                 (Default: MBEDTLS_SSL_CBC_RECORD_SPLITTING_ENABLED)
 *
 * \note           Only affects SSLv3 and TLS 1.0, not higher versions.
 *                 Does not affect non-CBC ciphersuites in any version.
 *
 * \param conf     SSL configuration
 * \param split    MBEDTLS_SSL_CBC_RECORD_SPLITTING_ENABLED or
 *                 MBEDTLS_SSL_CBC_RECORD_SPLITTING_DISABLED
 */
void mbedtls_ssl_conf_cbc_record_splitting(mbedtls_ssl_config *conf, char split);
#endif /* MBEDTLS_SSL_CBC_RECORD_SPLITTING */

#if defined(MBEDTLS_SSL_SESSION_TICKETS) && defined(MBEDTLS_SSL_CLI_C)
/**
 * \brief          Enable / Disable session tickets (client only).
 *                 (Default: MBEDTLS_SSL_SESSION_TICKETS_ENABLED.)
 *
 * \note           On server, use \c mbedtls_ssl_conf_session_tickets_cb().
 *
 * \param conf     SSL configuration
 * \param use_tickets   Enable or disable (MBEDTLS_SSL_SESSION_TICKETS_ENABLED or
 *                                         MBEDTLS_SSL_SESSION_TICKETS_DISABLED)
 */
void mbedtls_ssl_conf_session_tickets(mbedtls_ssl_config *conf, int use_tickets);
#endif /* MBEDTLS_SSL_SESSION_TICKETS && MBEDTLS_SSL_CLI_C */

#if defined(MBEDTLS_SSL_RENEGOTIATION)
/**
 * \brief          Enable / Disable renegotiation support for connection when
 *                 initiated by peer
 *                 (Default: MBEDTLS_SSL_RENEGOTIATION_DISABLED)
 *
 * \warning        It is recommended to always disable renegotiation unless you
 *                 know you need it and you know what you're doing. In the
 *                 past, there have been several issues associated with
 *                 renegotiation or a poor understanding of its properties.
 *
 * \note           Server-side, enabling renegotiation also makes the server
 *                 susceptible to a resource DoS by a malicious client.
 *
 * \param conf    SSL configuration
 * \param renegotiation     Enable or disable (MBEDTLS_SSL_RENEGOTIATION_ENABLED or
 *                                             MBEDTLS_SSL_RENEGOTIATION_DISABLED)
 */
void mbedtls_ssl_conf_renegotiation(mbedtls_ssl_config *conf, int renegotiation);
#endif /* MBEDTLS_SSL_RENEGOTIATION */

/**
 * \brief          Prevent or allow legacy renegotiation.
 *                 (Default: MBEDTLS_SSL_LEGACY_NO_RENEGOTIATION)
 *
 *                 MBEDTLS_SSL_LEGACY_NO_RENEGOTIATION allows connections to
 *                 be established even if the peer does not support
 *                 secure renegotiation, but does not allow renegotiation
 *                 to take place if not secure.
 *                 (Interoperable and secure option)
 *
 *                 MBEDTLS_SSL_LEGACY_ALLOW_RENEGOTIATION allows renegotiations
 *                 with non-upgraded peers. Allowing legacy renegotiation
 *                 makes the connection vulnerable to specific man in the
 *                 middle attacks. (See RFC 5746)
 *                 (Most interoperable and least secure option)
 *
 *                 MBEDTLS_SSL_LEGACY_BREAK_HANDSHAKE breaks off connections
 *                 if peer does not support secure renegotiation. Results
 *                 in interoperability issues with non-upgraded peers
 *                 that do not support renegotiation altogether.
 *                 (Most secure option, interoperability issues)
 *
 * \param conf     SSL configuration
 * \param allow_legacy  Prevent or allow (SSL_NO_LEGACY_RENEGOTIATION,
 *                                        SSL_ALLOW_LEGACY_RENEGOTIATION or
 *                                        MBEDTLS_SSL_LEGACY_BREAK_HANDSHAKE)
 */
void mbedtls_ssl_conf_legacy_renegotiation(mbedtls_ssl_config *conf, int allow_legacy);

#if defined(MBEDTLS_SSL_RENEGOTIATION)
/**
 * \brief          Enforce renegotiation requests.
 *                 (Default: enforced, max_records = 16)
 *
 *                 When we request a renegotiation, the peer can comply or
 *                 ignore the request. This function allows us to decide
 *                 whether to enforce our renegotiation requests by closing
 *                 the connection if the peer doesn't comply.
 *
 *                 However, records could already be in transit from the peer
 *                 when the request is emitted. In order to increase
 *                 reliability, we can accept a number of records before the
 *                 expected handshake records.
 *
 *                 The optimal value is highly dependent on the specific usage
 *                 scenario.
 *
 * \note           With DTLS and server-initiated renegotiation, the
 *                 HelloRequest is retransmitted every time mbedtls_ssl_read() times
 *                 out or receives Application Data, until:
 *                 - max_records records have beens seen, if it is >= 0, or
 *                 - the number of retransmits that would happen during an
 *                 actual handshake has been reached.
 *                 Please remember the request might be lost a few times
 *                 if you consider setting max_records to a really low value.
 *
 * \warning        On client, the grace period can only happen during
 *                 mbedtls_ssl_read(), as opposed to mbedtls_ssl_write() and mbedtls_ssl_renegotiate()
 *                 which always behave as if max_record was 0. The reason is,
 *                 if we receive application data from the server, we need a
 *                 place to write it, which only happens during mbedtls_ssl_read().
 *
 * \param conf     SSL configuration
 * \param max_records Use MBEDTLS_SSL_RENEGOTIATION_NOT_ENFORCED if you don't want to
 *                 enforce renegotiation, or a non-negative value to enforce
 *                 it but allow for a grace period of max_records records.
 */
void mbedtls_ssl_conf_renegotiation_enforced(mbedtls_ssl_config *conf, int max_records);

/**
 * \brief          Set record counter threshold for periodic renegotiation.
 *                 (Default: 2^48 - 1)
 *
 *                 Renegotiation is automatically triggered when a record
 *                 counter (outgoing or incoming) crosses the defined
 *                 threshold. The default value is meant to prevent the
 *                 connection from being closed when the counter is about to
 *                 reached its maximal value (it is not allowed to wrap).
 *
 *                 Lower values can be used to enforce policies such as "keys
 *                 must be refreshed every N packets with cipher X".
 *
 *                 The renegotiation period can be disabled by setting
 *                 conf->disable_renegotiation to
 *                 MBEDTLS_SSL_RENEGOTIATION_DISABLED.
 *
 * \note           When the configured transport is
 *                 MBEDTLS_SSL_TRANSPORT_DATAGRAM the maximum renegotiation
 *                 period is 2^48 - 1, and for MBEDTLS_SSL_TRANSPORT_STREAM,
 *                 the maximum renegotiation period is 2^64 - 1.
 *
 * \param conf     SSL configuration
 * \param period   The threshold value: a big-endian 64-bit number.
 */
void mbedtls_ssl_conf_renegotiation_period(mbedtls_ssl_config *conf,
                                           const unsigned char period[8]);
#endif /* MBEDTLS_SSL_RENEGOTIATION */

/**
 * \brief          Check if there is data already read from the
 *                 underlying transport but not yet processed.
 *
 * \param ssl      SSL context
 *
 * \return         0 if nothing's pending, 1 otherwise.
 *
 * \note           This is different in purpose and behaviour from
 *                 \c mbedtls_ssl_get_bytes_avail in that it considers
 *                 any kind of unprocessed data, not only unread
 *                 application data. If \c mbedtls_ssl_get_bytes
 *                 returns a non-zero value, this function will
 *                 also signal pending data, but the converse does
 *                 not hold. For example, in DTLS there might be
 *                 further records waiting to be processed from
 *                 the current underlying transport's datagram.
 *
 * \note           If this function returns 1 (data pending), this
 *                 does not imply that a subsequent call to
 *                 \c mbedtls_ssl_read will provide any data;
 *                 e.g., the unprocessed data might turn out
 *                 to be an alert or a handshake message.
 *
 * \note           This function is useful in the following situation:
 *                 If the SSL/TLS module successfully returns from an
 *                 operation - e.g. a handshake or an application record
 *                 read - and you're awaiting incoming data next, you
 *                 must not immediately idle on the underlying transport
 *                 to have data ready, but you need to check the value
 *                 of this function first. The reason is that the desired
 *                 data might already be read but not yet processed.
 *                 If, in contrast, a previous call to the SSL/TLS module
 *                 returned MBEDTLS_ERR_SSL_WANT_READ, it is not necessary
 *                 to call this function, as the latter error code entails
 *                 that all internal data has been processed.
 *
 */
int mbedtls_ssl_check_pending(const mbedtls_ssl_context *ssl);

/**
 * \brief          Return the number of application data bytes
 *                 remaining to be read from the current record.
 *
 * \param ssl      SSL context
 *
 * \return         How many bytes are available in the application
 *                 data record read buffer.
 *
 * \note           When working over a datagram transport, this is
 *                 useful to detect the current datagram's boundary
 *                 in case \c mbedtls_ssl_read has written the maximal
 *                 amount of data fitting into the input buffer.
 *
 */
size_t mbedtls_ssl_get_bytes_avail(const mbedtls_ssl_context *ssl);

/**
 * \brief          Return the result of the certificate verification
 *
 * \param ssl      The SSL context to use.
 *
 * \return         \c 0 if the certificate verification was successful.
 * \return         \c -1u if the result is not available. This may happen
 *                 e.g. if the handshake aborts early, or a verification
 *                 callback returned a fatal error.
 * \return         A bitwise combination of \c MBEDTLS_X509_BADCERT_XXX
 *                 and \c MBEDTLS_X509_BADCRL_XXX failure flags; see x509.h.
 */
uint32_t mbedtls_ssl_get_verify_result(const mbedtls_ssl_context *ssl);

/**
 * \brief          Return the name of the current ciphersuite
 *
 * \param ssl      SSL context
 *
 * \return         a string containing the ciphersuite name
 */
const char *mbedtls_ssl_get_ciphersuite(const mbedtls_ssl_context *ssl);

/**
 * \brief          Return the current SSL version (SSLv3/TLSv1/etc)
 *
 * \param ssl      SSL context
 *
 * \return         a string containing the SSL version
 */
const char *mbedtls_ssl_get_version(const mbedtls_ssl_context *ssl);

/**
 * \brief          Return the (maximum) number of bytes added by the record
 *                 layer: header + encryption/MAC overhead (inc. padding)
 *
 * \note           This function is not available (always returns an error)
 *                 when record compression is enabled.
 *
 * \param ssl      SSL context
 *
 * \return         Current maximum record expansion in bytes, or
 *                 MBEDTLS_ERR_SSL_FEATURE_UNAVAILABLE if compression is
 *                 enabled, which makes expansion much less predictable
 */
int mbedtls_ssl_get_record_expansion(const mbedtls_ssl_context *ssl);

#if defined(MBEDTLS_SSL_MAX_FRAGMENT_LENGTH)
/**
 * \brief          Return the maximum fragment length (payload, in bytes) for
 *                 the output buffer. For the client, this is the configured
 *                 value. For the server, it is the minimum of two - the
 *                 configured value and the negotiated one.
 *
 * \sa             mbedtls_ssl_conf_max_frag_len()
 * \sa             mbedtls_ssl_get_max_record_payload()
 *
 * \param ssl      SSL context
 *
 * \return         Current maximum fragment length for the output buffer.
 */
size_t mbedtls_ssl_get_output_max_frag_len(const mbedtls_ssl_context *ssl);

/**
 * \brief          Return the maximum fragment length (payload, in bytes) for
 *                 the input buffer. This is the negotiated maximum fragment
 *                 length, or, if there is none, MBEDTLS_SSL_MAX_CONTENT_LEN.
 *                 If it is not defined either, the value is 2^14. This function
 *                 works as its predecessor, \c mbedtls_ssl_get_max_frag_len().
 *
 * \sa             mbedtls_ssl_conf_max_frag_len()
 * \sa             mbedtls_ssl_get_max_record_payload()
 *
 * \param ssl      SSL context
 *
 * \return         Current maximum fragment length for the output buffer.
 */
size_t mbedtls_ssl_get_input_max_frag_len(const mbedtls_ssl_context *ssl);

#if !defined(MBEDTLS_DEPRECATED_REMOVED)

#if defined(MBEDTLS_DEPRECATED_WARNING)
#define MBEDTLS_DEPRECATED    __attribute__((deprecated))
#else
#define MBEDTLS_DEPRECATED
#endif

/**
 * \brief          This function is a deprecated approach to getting the max
 *                 fragment length. Its an alias for
 *                 \c mbedtls_ssl_get_output_max_frag_len(), as the behaviour
 *                 is the same. See \c mbedtls_ssl_get_output_max_frag_len() for
 *                 more detail.
 *
 * \sa             mbedtls_ssl_get_input_max_frag_len()
 * \sa             mbedtls_ssl_get_output_max_frag_len()
 *
 * \param ssl      SSL context
 *
 * \return         Current maximum fragment length for the output buffer.
 */
MBEDTLS_DEPRECATED size_t mbedtls_ssl_get_max_frag_len(
    const mbedtls_ssl_context *ssl);
#endif /* MBEDTLS_DEPRECATED_REMOVED */
#endif /* MBEDTLS_SSL_MAX_FRAGMENT_LENGTH */

/**
 * \brief          Return the current maximum outgoing record payload in bytes.
 *                 This takes into account the config.h setting \c
 *                 MBEDTLS_SSL_OUT_CONTENT_LEN, the configured and negotiated
 *                 max fragment length extension if used, and for DTLS the
 *                 path MTU as configured and current record expansion.
 *
 * \note           With DTLS, \c mbedtls_ssl_write() will return an error if
 *                 called with a larger length value.
 *                 With TLS, \c mbedtls_ssl_write() will fragment the input if
 *                 necessary and return the number of bytes written; it is up
 *                 to the caller to call \c mbedtls_ssl_write() again in
 *                 order to send the remaining bytes if any.
 *
 * \note           This function is not available (always returns an error)
 *                 when record compression is enabled.
 *
 * \sa             mbedtls_ssl_set_mtu()
 * \sa             mbedtls_ssl_get_output_max_frag_len()
 * \sa             mbedtls_ssl_get_input_max_frag_len()
 * \sa             mbedtls_ssl_get_record_expansion()
 *
 * \param ssl      SSL context
 *
 * \return         Current maximum payload for an outgoing record,
 *                 or a negative error code.
 */
int mbedtls_ssl_get_max_out_record_payload(const mbedtls_ssl_context *ssl);

#if defined(MBEDTLS_X509_CRT_PARSE_C)
/**
 * \brief          Return the peer certificate from the current connection.
 *
 * \param  ssl     The SSL context to use. This must be initialized and setup.
 *
 * \return         The current peer certificate, if available.
 *                 The returned certificate is owned by the SSL context and
 *                 is valid only until the next call to the SSL API.
 * \return         \c NULL if no peer certificate is available. This might
 *                 be because the chosen ciphersuite doesn't use CRTs
 *                 (PSK-based ciphersuites, for example), or because
 *                 #MBEDTLS_SSL_KEEP_PEER_CERTIFICATE has been disabled,
 *                 allowing the stack to free the peer's CRT to save memory.
 *
 * \note           For one-time inspection of the peer's certificate during
 *                 the handshake, consider registering an X.509 CRT verification
 *                 callback through mbedtls_ssl_conf_verify() instead of calling
 *                 this function. Using mbedtls_ssl_conf_verify() also comes at
 *                 the benefit of allowing you to influence the verification
 *                 process, for example by masking expected and tolerated
 *                 verification failures.
 *
 * \warning        You must not use the pointer returned by this function
 *                 after any further call to the SSL API, including
 *                 mbedtls_ssl_read() and mbedtls_ssl_write(); this is
 *                 because the pointer might change during renegotiation,
 *                 which happens transparently to the user.
 *                 If you want to use the certificate across API calls,
 *                 you must make a copy.
 */
const mbedtls_x509_crt *mbedtls_ssl_get_peer_cert(const mbedtls_ssl_context *ssl);
#endif /* MBEDTLS_X509_CRT_PARSE_C */

#if defined(MBEDTLS_SSL_CLI_C)
/**
 * \brief          Save session in order to resume it later (client-side only)
 *                 Session data is copied to presented session structure.
 *
 *
 * \param ssl      SSL context
 * \param session  session context
 *
 * \return         0 if successful,
 *                 MBEDTLS_ERR_SSL_ALLOC_FAILED if memory allocation failed,
 *                 MBEDTLS_ERR_SSL_BAD_INPUT_DATA if used server-side or
 *                 arguments are otherwise invalid.
 *
 * \note           Only the server certificate is copied, and not the full chain,
 *                 so you should not attempt to validate the certificate again
 *                 by calling \c mbedtls_x509_crt_verify() on it.
 *                 Instead, you should use the results from the verification
 *                 in the original handshake by calling \c mbedtls_ssl_get_verify_result()
 *                 after loading the session again into a new SSL context
 *                 using \c mbedtls_ssl_set_session().
 *
 * \note           Once the session object is not needed anymore, you should
 *                 free it by calling \c mbedtls_ssl_session_free().
 *
 * \sa             mbedtls_ssl_set_session()
 */
int mbedtls_ssl_get_session(const mbedtls_ssl_context *ssl, mbedtls_ssl_session *session);
#endif /* MBEDTLS_SSL_CLI_C */

/**
 * \brief          Perform the SSL handshake
 *
 * \param ssl      SSL context
 *
 * \return         \c 0 if successful.
 * \return         #MBEDTLS_ERR_SSL_WANT_READ or #MBEDTLS_ERR_SSL_WANT_WRITE
 *                 if the handshake is incomplete and waiting for data to
 *                 be available for reading from or writing to the underlying
 *                 transport - in this case you must call this function again
 *                 when the underlying transport is ready for the operation.
 * \return         #MBEDTLS_ERR_SSL_ASYNC_IN_PROGRESS if an asynchronous
 *                 operation is in progress (see
 *                 mbedtls_ssl_conf_async_private_cb()) - in this case you
 *                 must call this function again when the operation is ready.
 * \return         #MBEDTLS_ERR_SSL_CRYPTO_IN_PROGRESS if a cryptographic
 *                 operation is in progress (see mbedtls_ecp_set_max_ops()) -
 *                 in this case you must call this function again to complete
 *                 the handshake when you're done attending other tasks.
 * \return         #MBEDTLS_ERR_SSL_HELLO_VERIFY_REQUIRED if DTLS is in use
 *                 and the client did not demonstrate reachability yet - in
 *                 this case you must stop using the context (see below).
 * \return         Another SSL error code - in this case you must stop using
 *                 the context (see below).
 *
 * \warning        If this function returns something other than
 *                 \c 0,
 *                 #MBEDTLS_ERR_SSL_WANT_READ,
 *                 #MBEDTLS_ERR_SSL_WANT_WRITE,
 *                 #MBEDTLS_ERR_SSL_ASYNC_IN_PROGRESS or
 *                 #MBEDTLS_ERR_SSL_CRYPTO_IN_PROGRESS,
 *                 you must stop using the SSL context for reading or writing,
 *                 and either free it or call \c mbedtls_ssl_session_reset()
 *                 on it before re-using it for a new connection; the current
 *                 connection must be closed.
 *
 * \note           If DTLS is in use, then you may choose to handle
 *                 #MBEDTLS_ERR_SSL_HELLO_VERIFY_REQUIRED specially for logging
 *                 purposes, as it is an expected return value rather than an
 *                 actual error, but you still need to reset/free the context.
 *
 * \note           Remarks regarding event-driven DTLS:
 *                 If the function returns #MBEDTLS_ERR_SSL_WANT_READ, no datagram
 *                 from the underlying transport layer is currently being processed,
 *                 and it is safe to idle until the timer or the underlying transport
 *                 signal a new event. This is not true for a successful handshake,
 *                 in which case the datagram of the underlying transport that is
 *                 currently being processed might or might not contain further
 *                 DTLS records.
 *
 * \note           If #MBEDTLS_USE_PSA_CRYPTO is enabled, the PSA crypto
 *                 subsystem must have been initialized by calling
 *                 psa_crypto_init() before calling this function.
 */
int mbedtls_ssl_handshake(mbedtls_ssl_context *ssl);

/**
 * \brief          Perform a single step of the SSL handshake
 *
 * \note           The state of the context (ssl->state) will be at
 *                 the next state after this function returns \c 0. Do not
 *                 call this function if state is MBEDTLS_SSL_HANDSHAKE_OVER.
 *
 * \param ssl      SSL context
 *
 * \return         See mbedtls_ssl_handshake().
 *
 * \warning        If this function returns something other than \c 0,
 *                 #MBEDTLS_ERR_SSL_WANT_READ, #MBEDTLS_ERR_SSL_WANT_WRITE,
 *                 #MBEDTLS_ERR_SSL_ASYNC_IN_PROGRESS or
 *                 #MBEDTLS_ERR_SSL_CRYPTO_IN_PROGRESS, you must stop using
 *                 the SSL context for reading or writing, and either free it
 *                 or call \c mbedtls_ssl_session_reset() on it before
 *                 re-using it for a new connection; the current connection
 *                 must be closed.
 */
int mbedtls_ssl_handshake_step(mbedtls_ssl_context *ssl);

#if defined(MBEDTLS_SSL_RENEGOTIATION)
/**
 * \brief          Initiate an SSL renegotiation on the running connection.
 *                 Client: perform the renegotiation right now.
 *                 Server: request renegotiation, which will be performed
 *                 during the next call to mbedtls_ssl_read() if honored by
 *                 client.
 *
 * \param ssl      SSL context
 *
 * \return         0 if successful, or any mbedtls_ssl_handshake() return
 *                 value except #MBEDTLS_ERR_SSL_CLIENT_RECONNECT that can't
 *                 happen during a renegotiation.
 *
 * \warning        If this function returns something other than \c 0,
 *                 #MBEDTLS_ERR_SSL_WANT_READ, #MBEDTLS_ERR_SSL_WANT_WRITE,
 *                 #MBEDTLS_ERR_SSL_ASYNC_IN_PROGRESS or
 *                 #MBEDTLS_ERR_SSL_CRYPTO_IN_PROGRESS, you must stop using
 *                 the SSL context for reading or writing, and either free it
 *                 or call \c mbedtls_ssl_session_reset() on it before
 *                 re-using it for a new connection; the current connection
 *                 must be closed.
 *
 */
int mbedtls_ssl_renegotiate(mbedtls_ssl_context *ssl);
#endif /* MBEDTLS_SSL_RENEGOTIATION */

/**
 * \brief          Read at most 'len' application data bytes
 *
 * \param ssl      SSL context
 * \param buf      buffer that will hold the data
 * \param len      maximum number of bytes to read
 *
 * \return         The (positive) number of bytes read if successful.
 * \return         \c 0 if the read end of the underlying transport was closed
 *                 without sending a CloseNotify beforehand, which might happen
 *                 because of various reasons (internal error of an underlying
 *                 stack, non-conformant peer not sending a CloseNotify and
 *                 such) - in this case you must stop using the context
 *                 (see below).
 * \return         #MBEDTLS_ERR_SSL_PEER_CLOSE_NOTIFY if the underlying
 *                 transport is still functional, but the peer has
 *                 acknowledged to not send anything anymore.
 * \return         #MBEDTLS_ERR_SSL_WANT_READ or #MBEDTLS_ERR_SSL_WANT_WRITE
 *                 if the handshake is incomplete and waiting for data to
 *                 be available for reading from or writing to the underlying
 *                 transport - in this case you must call this function again
 *                 when the underlying transport is ready for the operation.
 * \return         #MBEDTLS_ERR_SSL_ASYNC_IN_PROGRESS if an asynchronous
 *                 operation is in progress (see
 *                 mbedtls_ssl_conf_async_private_cb()) - in this case you
 *                 must call this function again when the operation is ready.
 * \return         #MBEDTLS_ERR_SSL_CRYPTO_IN_PROGRESS if a cryptographic
 *                 operation is in progress (see mbedtls_ecp_set_max_ops()) -
 *                 in this case you must call this function again to complete
 *                 the handshake when you're done attending other tasks.
 * \return         #MBEDTLS_ERR_SSL_CLIENT_RECONNECT if we're at the server
 *                 side of a DTLS connection and the client is initiating a
 *                 new connection using the same source port. See below.
 * \return         Another SSL error code - in this case you must stop using
 *                 the context (see below).
 *
 * \warning        If this function returns something other than
 *                 a positive value,
 *                 #MBEDTLS_ERR_SSL_WANT_READ,
 *                 #MBEDTLS_ERR_SSL_WANT_WRITE,
 *                 #MBEDTLS_ERR_SSL_ASYNC_IN_PROGRESS,
 *                 #MBEDTLS_ERR_SSL_CRYPTO_IN_PROGRESS or
 *                 #MBEDTLS_ERR_SSL_CLIENT_RECONNECT,
 *                 you must stop using the SSL context for reading or writing,
 *                 and either free it or call \c mbedtls_ssl_session_reset()
 *                 on it before re-using it for a new connection; the current
 *                 connection must be closed.
 *
 * \note           When this function returns #MBEDTLS_ERR_SSL_CLIENT_RECONNECT
 *                 (which can only happen server-side), it means that a client
 *                 is initiating a new connection using the same source port.
 *                 You can either treat that as a connection close and wait
 *                 for the client to resend a ClientHello, or directly
 *                 continue with \c mbedtls_ssl_handshake() with the same
 *                 context (as it has been reset internally). Either way, you
 *                 must make sure this is seen by the application as a new
 *                 connection: application state, if any, should be reset, and
 *                 most importantly the identity of the client must be checked
 *                 again. WARNING: not validating the identity of the client
 *                 again, or not transmitting the new identity to the
 *                 application layer, would allow authentication bypass!
 *
 * \note           Remarks regarding event-driven DTLS:
 *                 - If the function returns #MBEDTLS_ERR_SSL_WANT_READ, no datagram
 *                   from the underlying transport layer is currently being processed,
 *                   and it is safe to idle until the timer or the underlying transport
 *                   signal a new event.
 *                 - This function may return MBEDTLS_ERR_SSL_WANT_READ even if data was
 *                   initially available on the underlying transport, as this data may have
 *                   been only e.g. duplicated messages or a renegotiation request.
 *                   Therefore, you must be prepared to receive MBEDTLS_ERR_SSL_WANT_READ even
 *                   when reacting to an incoming-data event from the underlying transport.
 *                 - On success, the datagram of the underlying transport that is currently
 *                   being processed may contain further DTLS records. You should call
 *                   \c mbedtls_ssl_check_pending to check for remaining records.
 *
 */
int mbedtls_ssl_read(mbedtls_ssl_context *ssl, unsigned char *buf, size_t len);

/**
 * \brief          Try to write exactly 'len' application data bytes
 *
 * \warning        This function will do partial writes in some cases. If the
 *                 return value is non-negative but less than length, the
 *                 function must be called again with updated arguments:
 *                 buf + ret, len - ret (if ret is the return value) until
 *                 it returns a value equal to the last 'len' argument.
 *
 * \param ssl      SSL context
 * \param buf      buffer holding the data
 * \param len      how many bytes must be written
 *
 * \return         The (non-negative) number of bytes actually written if
 *                 successful (may be less than \p len).
 * \return         #MBEDTLS_ERR_SSL_WANT_READ or #MBEDTLS_ERR_SSL_WANT_WRITE
 *                 if the handshake is incomplete and waiting for data to
 *                 be available for reading from or writing to the underlying
 *                 transport - in this case you must call this function again
 *                 when the underlying transport is ready for the operation.
 * \return         #MBEDTLS_ERR_SSL_ASYNC_IN_PROGRESS if an asynchronous
 *                 operation is in progress (see
 *                 mbedtls_ssl_conf_async_private_cb()) - in this case you
 *                 must call this function again when the operation is ready.
 * \return         #MBEDTLS_ERR_SSL_CRYPTO_IN_PROGRESS if a cryptographic
 *                 operation is in progress (see mbedtls_ecp_set_max_ops()) -
 *                 in this case you must call this function again to complete
 *                 the handshake when you're done attending other tasks.
 * \return         Another SSL error code - in this case you must stop using
 *                 the context (see below).
 *
 * \warning        If this function returns something other than
 *                 a non-negative value,
 *                 #MBEDTLS_ERR_SSL_WANT_READ,
 *                 #MBEDTLS_ERR_SSL_WANT_WRITE,
 *                 #MBEDTLS_ERR_SSL_ASYNC_IN_PROGRESS or
 *                 #MBEDTLS_ERR_SSL_CRYPTO_IN_PROGRESS,
 *                 you must stop using the SSL context for reading or writing,
 *                 and either free it or call \c mbedtls_ssl_session_reset()
 *                 on it before re-using it for a new connection; the current
 *                 connection must be closed.
 *
 * \note           When this function returns #MBEDTLS_ERR_SSL_WANT_WRITE/READ,
 *                 it must be called later with the *same* arguments,
 *                 until it returns a value greater that or equal to 0. When
 *                 the function returns #MBEDTLS_ERR_SSL_WANT_WRITE there may be
 *                 some partial data in the output buffer, however this is not
 *                 yet sent.
 *
 * \note           If the requested length is greater than the maximum
 *                 fragment length (either the built-in limit or the one set
 *                 or negotiated with the peer), then:
 *                 - with TLS, less bytes than requested are written.
 *                 - with DTLS, MBEDTLS_ERR_SSL_BAD_INPUT_DATA is returned.
 *                 \c mbedtls_ssl_get_output_max_frag_len() may be used to
 *                 query the active maximum fragment length.
 *
 * \note           Attempting to write 0 bytes will result in an empty TLS
 *                 application record being sent.
 */
int mbedtls_ssl_write(mbedtls_ssl_context *ssl, const unsigned char *buf, size_t len);

/**
 * \brief           Send an alert message
 *
 * \param ssl       SSL context
 * \param level     The alert level of the message
 *                  (MBEDTLS_SSL_ALERT_LEVEL_WARNING or MBEDTLS_SSL_ALERT_LEVEL_FATAL)
 * \param message   The alert message (SSL_ALERT_MSG_*)
 *
 * \return          0 if successful, or a specific SSL error code.
 *
 * \note           If this function returns something other than 0 or
 *                 MBEDTLS_ERR_SSL_WANT_READ/WRITE, you must stop using
 *                 the SSL context for reading or writing, and either free it or
 *                 call \c mbedtls_ssl_session_reset() on it before re-using it
 *                 for a new connection; the current connection must be closed.
 */
int mbedtls_ssl_send_alert_message(mbedtls_ssl_context *ssl,
                                   unsigned char level,
                                   unsigned char message);
/**
 * \brief          Notify the peer that the connection is being closed
 *
 * \param ssl      SSL context
 *
 * \return          0 if successful, or a specific SSL error code.
 *
 * \note           If this function returns something other than 0 or
 *                 MBEDTLS_ERR_SSL_WANT_READ/WRITE, you must stop using
 *                 the SSL context for reading or writing, and either free it or
 *                 call \c mbedtls_ssl_session_reset() on it before re-using it
 *                 for a new connection; the current connection must be closed.
 */
int mbedtls_ssl_close_notify(mbedtls_ssl_context *ssl);

/**
 * \brief          Free referenced items in an SSL context and clear memory
 *
 * \param ssl      SSL context
 */
void mbedtls_ssl_free(mbedtls_ssl_context *ssl);

#if defined(MBEDTLS_SSL_CONTEXT_SERIALIZATION)
/**
 * \brief          Save an active connection as serialized data in a buffer.
 *                 This allows the freeing or re-using of the SSL context
 *                 while still picking up the connection later in a way that
 *                 it entirely transparent to the peer.
 *
 * \see            mbedtls_ssl_context_load()
 *
 * \note           This feature is currently only available under certain
 *                 conditions, see the documentation of the return value
 *                 #MBEDTLS_ERR_SSL_BAD_INPUT_DATA for details.
 *
 * \note           When this function succeeds, it calls
 *                 mbedtls_ssl_session_reset() on \p ssl which as a result is
 *                 no longer associated with the connection that has been
 *                 serialized. This avoids creating copies of the connection
 *                 state. You're then free to either re-use the context
 *                 structure for a different connection, or call
 *                 mbedtls_ssl_free() on it. See the documentation of
 *                 mbedtls_ssl_session_reset() for more details.
 *
 * \param ssl      The SSL context to save. On success, it is no longer
 *                 associated with the connection that has been serialized.
 * \param buf      The buffer to write the serialized data to. It must be a
 *                 writeable buffer of at least \p buf_len bytes, or may be \c
 *                 NULL if \p buf_len is \c 0.
 * \param buf_len  The number of bytes available for writing in \p buf.
 * \param olen     The size in bytes of the data that has been or would have
 *                 been written. It must point to a valid \c size_t.
 *
 * \note           \p olen is updated to the correct value regardless of
 *                 whether \p buf_len was large enough. This makes it possible
 *                 to determine the necessary size by calling this function
 *                 with \p buf set to \c NULL and \p buf_len to \c 0. However,
 *                 the value of \p olen is only guaranteed to be correct when
 *                 the function returns #MBEDTLS_ERR_SSL_BUFFER_TOO_SMALL or
 *                 \c 0. If the return value is different, then the value of
 *                 \p olen is undefined.
 *
 * \return         \c 0 if successful.
 * \return         #MBEDTLS_ERR_SSL_BUFFER_TOO_SMALL if \p buf is too small.
 * \return         #MBEDTLS_ERR_SSL_ALLOC_FAILED if memory allocation failed
 *                 while resetting the context.
 * \return         #MBEDTLS_ERR_SSL_BAD_INPUT_DATA if a handshake is in
 *                 progress, or there is pending data for reading or sending,
 *                 or the connection does not use DTLS 1.2 with an AEAD
 *                 ciphersuite, or renegotiation is enabled.
 */
int mbedtls_ssl_context_save(mbedtls_ssl_context *ssl,
                             unsigned char *buf,
                             size_t buf_len,
                             size_t *olen);

/**
 * \brief          Load serialized connection data to an SSL context.
 *
 * \see            mbedtls_ssl_context_save()
 *
 * \warning        The same serialized data must never be loaded into more
 *                 that one context. In order to ensure that, after
 *                 successfully loading serialized data to an SSL context, you
 *                 should immediately destroy or invalidate all copies of the
 *                 serialized data that was loaded. Loading the same data in
 *                 more than one context would cause severe security failures
 *                 including but not limited to loss of confidentiality.
 *
 * \note           Before calling this function, the SSL context must be
 *                 prepared in one of the two following ways. The first way is
 *                 to take a context freshly initialised with
 *                 mbedtls_ssl_init() and call mbedtls_ssl_setup() on it with
 *                 the same ::mbedtls_ssl_config structure that was used in
 *                 the original connection. The second way is to
 *                 call mbedtls_ssl_session_reset() on a context that was
 *                 previously prepared as above but used in the meantime.
 *                 Either way, you must not use the context to perform a
 *                 handshake between calling mbedtls_ssl_setup() or
 *                 mbedtls_ssl_session_reset() and calling this function. You
 *                 may however call other setter functions in that time frame
 *                 as indicated in the note below.
 *
 * \note           Before or after calling this function successfully, you
 *                 also need to configure some connection-specific callbacks
 *                 and settings before you can use the connection again
 *                 (unless they were already set before calling
 *                 mbedtls_ssl_session_reset() and the values are suitable for
 *                 the present connection). Specifically, you want to call
 *                 at least mbedtls_ssl_set_bio() and
 *                 mbedtls_ssl_set_timer_cb(). All other SSL setter functions
 *                 are not necessary to call, either because they're only used
 *                 in handshakes, or because the setting is already saved. You
 *                 might choose to call them anyway, for example in order to
 *                 share code between the cases of establishing a new
 *                 connection and the case of loading an already-established
 *                 connection.
 *
 * \note           If you have new information about the path MTU, you want to
 *                 call mbedtls_ssl_set_mtu() after calling this function, as
 *                 otherwise this function would overwrite your
 *                 newly-configured value with the value that was active when
 *                 the context was saved.
 *
 * \note           When this function returns an error code, it calls
 *                 mbedtls_ssl_free() on \p ssl. In this case, you need to
 *                 prepare the context with the usual sequence starting with a
 *                 call to mbedtls_ssl_init() if you want to use it again.
 *
 * \param ssl      The SSL context structure to be populated. It must have
 *                 been prepared as described in the note above.
 * \param buf      The buffer holding the serialized connection data. It must
 *                 be a readable buffer of at least \p len bytes.
 * \param len      The size of the serialized data in bytes.
 *
 * \return         \c 0 if successful.
 * \return         #MBEDTLS_ERR_SSL_ALLOC_FAILED if memory allocation failed.
 * \return         #MBEDTLS_ERR_SSL_VERSION_MISMATCH if the serialized data
 *                 comes from a different Mbed TLS version or build.
 * \return         #MBEDTLS_ERR_SSL_BAD_INPUT_DATA if input data is invalid.
 */
int mbedtls_ssl_context_load(mbedtls_ssl_context *ssl,
                             const unsigned char *buf,
                             size_t len);
#endif /* MBEDTLS_SSL_CONTEXT_SERIALIZATION */

/**
 * \brief          Initialize an SSL configuration context
 *                 Just makes the context ready for
 *                 mbedtls_ssl_config_defaults() or mbedtls_ssl_config_free().
 *
 * \note           You need to call mbedtls_ssl_config_defaults() unless you
 *                 manually set all of the relevant fields yourself.
 *
 * \param conf     SSL configuration context
 */
void mbedtls_ssl_config_init(mbedtls_ssl_config *conf);

/**
 * \brief          Load reasonable default SSL configuration values.
 *                 (You need to call mbedtls_ssl_config_init() first.)
 *
 * \param conf     SSL configuration context
 * \param endpoint MBEDTLS_SSL_IS_CLIENT or MBEDTLS_SSL_IS_SERVER
 * \param transport MBEDTLS_SSL_TRANSPORT_STREAM for TLS, or
 *                  MBEDTLS_SSL_TRANSPORT_DATAGRAM for DTLS
 * \param preset   a MBEDTLS_SSL_PRESET_XXX value
 *
 * \note           See \c mbedtls_ssl_conf_transport() for notes on DTLS.
 *
 * \return         0 if successful, or
 *                 MBEDTLS_ERR_XXX_ALLOC_FAILED on memory allocation error.
 */
int mbedtls_ssl_config_defaults(mbedtls_ssl_config *conf,
                                int endpoint, int transport, int preset);

/**
 * \brief          Free an SSL configuration context
 *
 * \param conf     SSL configuration context
 */
void mbedtls_ssl_config_free(mbedtls_ssl_config *conf);

/**
 * \brief          Initialize SSL session structure
 *
 * \param session  SSL session
 */
void mbedtls_ssl_session_init(mbedtls_ssl_session *session);

/**
 * \brief          Free referenced items in an SSL session including the
 *                 peer certificate and clear memory
 *
 * \note           A session object can be freed even if the SSL context
 *                 that was used to retrieve the session is still in use.
 *
 * \param session  SSL session
 */
void mbedtls_ssl_session_free(mbedtls_ssl_session *session);

/**
 * \brief          TLS-PRF function for key derivation.
 *
 * \param prf      The tls_prf type function type to be used.
 * \param secret   Secret for the key derivation function.
 * \param slen     Length of the secret.
 * \param label    String label for the key derivation function,
 *                 terminated with null character.
 * \param random   Random bytes.
 * \param rlen     Length of the random bytes buffer.
 * \param dstbuf   The buffer holding the derived key.
 * \param dlen     Length of the output buffer.
 *
 * \return         0 on success. An SSL specific error on failure.
 */
int  mbedtls_ssl_tls_prf(const mbedtls_tls_prf_types prf,
                         const unsigned char *secret, size_t slen,
                         const char *label,
                         const unsigned char *random, size_t rlen,
                         unsigned char *dstbuf, size_t dlen);

#ifdef __cplusplus
}
#endif

#endif /* ssl.h */
