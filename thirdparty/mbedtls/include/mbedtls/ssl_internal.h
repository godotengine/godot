/**
 * \file ssl_internal.h
 *
 * \brief Internal functions shared by the SSL modules
 */
/*
 *  Copyright The Mbed TLS Contributors
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
 */
#ifndef MBEDTLS_SSL_INTERNAL_H
#define MBEDTLS_SSL_INTERNAL_H

#if !defined(MBEDTLS_CONFIG_FILE)
#include "config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#include "ssl.h"
#include "cipher.h"

#if defined(MBEDTLS_MD5_C)
#include "md5.h"
#endif

#if defined(MBEDTLS_SHA1_C)
#include "sha1.h"
#endif

#if defined(MBEDTLS_SHA256_C)
#include "sha256.h"
#endif

#if defined(MBEDTLS_SHA512_C)
#include "sha512.h"
#endif

#if defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
#include "ecjpake.h"
#endif

#if ( defined(__ARMCC_VERSION) || defined(_MSC_VER) ) && \
    !defined(inline) && !defined(__cplusplus)
#define inline __inline
#endif

/* Determine minimum supported version */
#define MBEDTLS_SSL_MIN_MAJOR_VERSION           MBEDTLS_SSL_MAJOR_VERSION_3

#if defined(MBEDTLS_SSL_PROTO_SSL3)
#define MBEDTLS_SSL_MIN_MINOR_VERSION           MBEDTLS_SSL_MINOR_VERSION_0
#else
#if defined(MBEDTLS_SSL_PROTO_TLS1)
#define MBEDTLS_SSL_MIN_MINOR_VERSION           MBEDTLS_SSL_MINOR_VERSION_1
#else
#if defined(MBEDTLS_SSL_PROTO_TLS1_1)
#define MBEDTLS_SSL_MIN_MINOR_VERSION           MBEDTLS_SSL_MINOR_VERSION_2
#else
#if defined(MBEDTLS_SSL_PROTO_TLS1_2)
#define MBEDTLS_SSL_MIN_MINOR_VERSION           MBEDTLS_SSL_MINOR_VERSION_3
#endif /* MBEDTLS_SSL_PROTO_TLS1_2 */
#endif /* MBEDTLS_SSL_PROTO_TLS1_1 */
#endif /* MBEDTLS_SSL_PROTO_TLS1   */
#endif /* MBEDTLS_SSL_PROTO_SSL3   */

#define MBEDTLS_SSL_MIN_VALID_MINOR_VERSION MBEDTLS_SSL_MINOR_VERSION_1
#define MBEDTLS_SSL_MIN_VALID_MAJOR_VERSION MBEDTLS_SSL_MAJOR_VERSION_3

/* Determine maximum supported version */
#define MBEDTLS_SSL_MAX_MAJOR_VERSION           MBEDTLS_SSL_MAJOR_VERSION_3

#if defined(MBEDTLS_SSL_PROTO_TLS1_2)
#define MBEDTLS_SSL_MAX_MINOR_VERSION           MBEDTLS_SSL_MINOR_VERSION_3
#else
#if defined(MBEDTLS_SSL_PROTO_TLS1_1)
#define MBEDTLS_SSL_MAX_MINOR_VERSION           MBEDTLS_SSL_MINOR_VERSION_2
#else
#if defined(MBEDTLS_SSL_PROTO_TLS1)
#define MBEDTLS_SSL_MAX_MINOR_VERSION           MBEDTLS_SSL_MINOR_VERSION_1
#else
#if defined(MBEDTLS_SSL_PROTO_SSL3)
#define MBEDTLS_SSL_MAX_MINOR_VERSION           MBEDTLS_SSL_MINOR_VERSION_0
#endif /* MBEDTLS_SSL_PROTO_SSL3   */
#endif /* MBEDTLS_SSL_PROTO_TLS1   */
#endif /* MBEDTLS_SSL_PROTO_TLS1_1 */
#endif /* MBEDTLS_SSL_PROTO_TLS1_2 */

/* Shorthand for restartable ECC */
#if defined(MBEDTLS_ECP_RESTARTABLE) && \
    defined(MBEDTLS_SSL_CLI_C) && \
    defined(MBEDTLS_SSL_PROTO_TLS1_2) && \
    defined(MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA_ENABLED)
#define MBEDTLS_SSL__ECP_RESTARTABLE
#endif

#define MBEDTLS_SSL_INITIAL_HANDSHAKE           0
#define MBEDTLS_SSL_RENEGOTIATION_IN_PROGRESS   1   /* In progress */
#define MBEDTLS_SSL_RENEGOTIATION_DONE          2   /* Done or aborted */
#define MBEDTLS_SSL_RENEGOTIATION_PENDING       3   /* Requested (server only) */

/*
 * DTLS retransmission states, see RFC 6347 4.2.4
 *
 * The SENDING state is merged in PREPARING for initial sends,
 * but is distinct for resends.
 *
 * Note: initial state is wrong for server, but is not used anyway.
 */
#define MBEDTLS_SSL_RETRANS_PREPARING       0
#define MBEDTLS_SSL_RETRANS_SENDING         1
#define MBEDTLS_SSL_RETRANS_WAITING         2
#define MBEDTLS_SSL_RETRANS_FINISHED        3

/* This macro determines whether CBC is supported. */
#if defined(MBEDTLS_CIPHER_MODE_CBC) &&                               \
    ( defined(MBEDTLS_AES_C)      ||                                  \
      defined(MBEDTLS_CAMELLIA_C) ||                                  \
      defined(MBEDTLS_ARIA_C)     ||                                  \
      defined(MBEDTLS_DES_C) )
#define MBEDTLS_SSL_SOME_SUITES_USE_CBC
#endif

/* This macro determines whether the CBC construct used in TLS 1.0-1.2 (as
 * opposed to the very different CBC construct used in SSLv3) is supported. */
#if defined(MBEDTLS_SSL_SOME_SUITES_USE_CBC) && \
    ( defined(MBEDTLS_SSL_PROTO_TLS1) ||        \
      defined(MBEDTLS_SSL_PROTO_TLS1_1) ||      \
      defined(MBEDTLS_SSL_PROTO_TLS1_2) )
#define MBEDTLS_SSL_SOME_SUITES_USE_TLS_CBC
#endif

/*
 * Allow extra bytes for record, authentication and encryption overhead:
 * counter (8) + header (5) + IV(16) + MAC (16-48) + padding (0-256)
 * and allow for a maximum of 1024 of compression expansion if
 * enabled.
 */
#if defined(MBEDTLS_ZLIB_SUPPORT)
#define MBEDTLS_SSL_COMPRESSION_ADD          1024
#else
#define MBEDTLS_SSL_COMPRESSION_ADD             0
#endif

#if defined(MBEDTLS_ARC4_C) || defined(MBEDTLS_CIPHER_MODE_CBC)
/* Ciphersuites using HMAC */
#if defined(MBEDTLS_SHA512_C)
#define MBEDTLS_SSL_MAC_ADD                 48  /* SHA-384 used for HMAC */
#elif defined(MBEDTLS_SHA256_C)
#define MBEDTLS_SSL_MAC_ADD                 32  /* SHA-256 used for HMAC */
#else
#define MBEDTLS_SSL_MAC_ADD                 20  /* SHA-1   used for HMAC */
#endif
#else
/* AEAD ciphersuites: GCM and CCM use a 128 bits tag */
#define MBEDTLS_SSL_MAC_ADD                 16
#endif

#if defined(MBEDTLS_CIPHER_MODE_CBC)
#define MBEDTLS_SSL_PADDING_ADD            256
#else
#define MBEDTLS_SSL_PADDING_ADD              0
#endif

#define MBEDTLS_SSL_PAYLOAD_OVERHEAD ( MBEDTLS_SSL_COMPRESSION_ADD +    \
                                       MBEDTLS_MAX_IV_LENGTH +          \
                                       MBEDTLS_SSL_MAC_ADD +            \
                                       MBEDTLS_SSL_PADDING_ADD          \
                                       )

#define MBEDTLS_SSL_IN_PAYLOAD_LEN ( MBEDTLS_SSL_PAYLOAD_OVERHEAD + \
                                     ( MBEDTLS_SSL_IN_CONTENT_LEN ) )

#define MBEDTLS_SSL_OUT_PAYLOAD_LEN ( MBEDTLS_SSL_PAYLOAD_OVERHEAD + \
                                      ( MBEDTLS_SSL_OUT_CONTENT_LEN ) )

/* The maximum number of buffered handshake messages. */
#define MBEDTLS_SSL_MAX_BUFFERED_HS 4

/* Maximum length we can advertise as our max content length for
   RFC 6066 max_fragment_length extension negotiation purposes
   (the lesser of both sizes, if they are unequal.)
 */
#define MBEDTLS_TLS_EXT_ADV_CONTENT_LEN (                            \
        (MBEDTLS_SSL_IN_CONTENT_LEN > MBEDTLS_SSL_OUT_CONTENT_LEN)   \
        ? ( MBEDTLS_SSL_OUT_CONTENT_LEN )                            \
        : ( MBEDTLS_SSL_IN_CONTENT_LEN )                             \
        )

/* Maximum size in bytes of list in sig-hash algorithm ext., RFC 5246 */
#define MBEDTLS_SSL_MAX_SIG_HASH_ALG_LIST_LEN  65534

/* Maximum size in bytes of list in supported elliptic curve ext., RFC 4492 */
#define MBEDTLS_SSL_MAX_CURVE_LIST_LEN         65535

/*
 * Check that we obey the standard's message size bounds
 */

#if MBEDTLS_SSL_MAX_CONTENT_LEN > 16384
#error "Bad configuration - record content too large."
#endif

#if MBEDTLS_SSL_IN_CONTENT_LEN > MBEDTLS_SSL_MAX_CONTENT_LEN
#error "Bad configuration - incoming record content should not be larger than MBEDTLS_SSL_MAX_CONTENT_LEN."
#endif

#if MBEDTLS_SSL_OUT_CONTENT_LEN > MBEDTLS_SSL_MAX_CONTENT_LEN
#error "Bad configuration - outgoing record content should not be larger than MBEDTLS_SSL_MAX_CONTENT_LEN."
#endif

#if MBEDTLS_SSL_IN_PAYLOAD_LEN > MBEDTLS_SSL_MAX_CONTENT_LEN + 2048
#error "Bad configuration - incoming protected record payload too large."
#endif

#if MBEDTLS_SSL_OUT_PAYLOAD_LEN > MBEDTLS_SSL_MAX_CONTENT_LEN + 2048
#error "Bad configuration - outgoing protected record payload too large."
#endif

/* Calculate buffer sizes */

/* Note: Even though the TLS record header is only 5 bytes
   long, we're internally using 8 bytes to store the
   implicit sequence number. */
#define MBEDTLS_SSL_HEADER_LEN 13

#define MBEDTLS_SSL_IN_BUFFER_LEN  \
    ( ( MBEDTLS_SSL_HEADER_LEN ) + ( MBEDTLS_SSL_IN_PAYLOAD_LEN ) )

#define MBEDTLS_SSL_OUT_BUFFER_LEN  \
    ( ( MBEDTLS_SSL_HEADER_LEN ) + ( MBEDTLS_SSL_OUT_PAYLOAD_LEN ) )

#ifdef MBEDTLS_ZLIB_SUPPORT
/* Compression buffer holds both IN and OUT buffers, so should be size of the larger */
#define MBEDTLS_SSL_COMPRESS_BUFFER_LEN (                               \
        ( MBEDTLS_SSL_IN_BUFFER_LEN > MBEDTLS_SSL_OUT_BUFFER_LEN )      \
        ? MBEDTLS_SSL_IN_BUFFER_LEN                                     \
        : MBEDTLS_SSL_OUT_BUFFER_LEN                                    \
        )
#endif

/*
 * TLS extension flags (for extensions with outgoing ServerHello content
 * that need it (e.g. for RENEGOTIATION_INFO the server already knows because
 * of state of the renegotiation flag, so no indicator is required)
 */
#define MBEDTLS_TLS_EXT_SUPPORTED_POINT_FORMATS_PRESENT (1 << 0)
#define MBEDTLS_TLS_EXT_ECJPAKE_KKPP_OK                 (1 << 1)

/**
 * \brief        This function checks if the remaining size in a buffer is
 *               greater or equal than a needed space.
 *
 * \param cur    Pointer to the current position in the buffer.
 * \param end    Pointer to one past the end of the buffer.
 * \param need   Needed space in bytes.
 *
 * \return       Zero if the needed space is available in the buffer, non-zero
 *               otherwise.
 */
static inline int mbedtls_ssl_chk_buf_ptr( const uint8_t *cur,
                                           const uint8_t *end, size_t need )
{
    return( ( cur > end ) || ( need > (size_t)( end - cur ) ) );
}

/**
 * \brief        This macro checks if the remaining size in a buffer is
 *               greater or equal than a needed space. If it is not the case,
 *               it returns an SSL_BUFFER_TOO_SMALL error.
 *
 * \param cur    Pointer to the current position in the buffer.
 * \param end    Pointer to one past the end of the buffer.
 * \param need   Needed space in bytes.
 *
 */
#define MBEDTLS_SSL_CHK_BUF_PTR( cur, end, need )                        \
    do {                                                                 \
        if( mbedtls_ssl_chk_buf_ptr( ( cur ), ( end ), ( need ) ) != 0 ) \
        {                                                                \
            return( MBEDTLS_ERR_SSL_BUFFER_TOO_SMALL );                  \
        }                                                                \
    } while( 0 )

#ifdef __cplusplus
extern "C" {
#endif

#if defined(MBEDTLS_SSL_PROTO_TLS1_2) && \
    defined(MBEDTLS_KEY_EXCHANGE__WITH_CERT__ENABLED)
/*
 * Abstraction for a grid of allowed signature-hash-algorithm pairs.
 */
struct mbedtls_ssl_sig_hash_set_t
{
    /* At the moment, we only need to remember a single suitable
     * hash algorithm per signature algorithm. As long as that's
     * the case - and we don't need a general lookup function -
     * we can implement the sig-hash-set as a map from signatures
     * to hash algorithms. */
    mbedtls_md_type_t rsa;
    mbedtls_md_type_t ecdsa;
};
#endif /* MBEDTLS_SSL_PROTO_TLS1_2 &&
          MBEDTLS_KEY_EXCHANGE__WITH_CERT__ENABLED */

/*
 * This structure contains the parameters only needed during handshake.
 */
struct mbedtls_ssl_handshake_params
{
    /*
     * Handshake specific crypto variables
     */

#if defined(MBEDTLS_SSL_PROTO_TLS1_2) && \
    defined(MBEDTLS_KEY_EXCHANGE__WITH_CERT__ENABLED)
    mbedtls_ssl_sig_hash_set_t hash_algs;             /*!<  Set of suitable sig-hash pairs */
#endif
#if defined(MBEDTLS_DHM_C)
    mbedtls_dhm_context dhm_ctx;                /*!<  DHM key exchange        */
#endif
#if defined(MBEDTLS_ECDH_C)
    mbedtls_ecdh_context ecdh_ctx;              /*!<  ECDH key exchange       */
#endif
#if defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
    mbedtls_ecjpake_context ecjpake_ctx;        /*!< EC J-PAKE key exchange */
#if defined(MBEDTLS_SSL_CLI_C)
    unsigned char *ecjpake_cache;               /*!< Cache for ClientHello ext */
    size_t ecjpake_cache_len;                   /*!< Length of cached data */
#endif
#endif /* MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED */
#if defined(MBEDTLS_ECDH_C) || defined(MBEDTLS_ECDSA_C) || \
    defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
    const mbedtls_ecp_curve_info **curves;      /*!<  Supported elliptic curves */
#endif
#if defined(MBEDTLS_KEY_EXCHANGE__SOME__PSK_ENABLED)
    unsigned char *psk;                 /*!<  PSK from the callback         */
    size_t psk_len;                     /*!<  Length of PSK from callback   */
#endif
#if defined(MBEDTLS_X509_CRT_PARSE_C)
    mbedtls_ssl_key_cert *key_cert;     /*!< chosen key/cert pair (server)  */
#if defined(MBEDTLS_SSL_SERVER_NAME_INDICATION)
    int sni_authmode;                   /*!< authmode from SNI callback     */
    mbedtls_ssl_key_cert *sni_key_cert; /*!< key/cert list from SNI         */
    mbedtls_x509_crt *sni_ca_chain;     /*!< trusted CAs from SNI callback  */
    mbedtls_x509_crl *sni_ca_crl;       /*!< trusted CAs CRLs from SNI      */
#endif /* MBEDTLS_SSL_SERVER_NAME_INDICATION */
#endif /* MBEDTLS_X509_CRT_PARSE_C */
#if defined(MBEDTLS_SSL__ECP_RESTARTABLE)
    int ecrs_enabled;                   /*!< Handshake supports EC restart? */
    mbedtls_x509_crt_restart_ctx ecrs_ctx;  /*!< restart context            */
    enum { /* this complements ssl->state with info on intra-state operations */
        ssl_ecrs_none = 0,              /*!< nothing going on (yet)         */
        ssl_ecrs_crt_verify,            /*!< Certificate: crt_verify()      */
        ssl_ecrs_ske_start_processing,  /*!< ServerKeyExchange: pk_verify() */
        ssl_ecrs_cke_ecdh_calc_secret,  /*!< ClientKeyExchange: ECDH step 2 */
        ssl_ecrs_crt_vrfy_sign,         /*!< CertificateVerify: pk_sign()   */
    } ecrs_state;                       /*!< current (or last) operation    */
    size_t ecrs_n;                      /*!< place for saving a length      */
#endif
#if defined(MBEDTLS_SSL_PROTO_DTLS)
    unsigned int out_msg_seq;           /*!<  Outgoing handshake sequence number */
    unsigned int in_msg_seq;            /*!<  Incoming handshake sequence number */

    unsigned char *verify_cookie;       /*!<  Cli: HelloVerifyRequest cookie
                                              Srv: unused                    */
    unsigned char verify_cookie_len;    /*!<  Cli: cookie length
                                              Srv: flag for sending a cookie */

    uint32_t retransmit_timeout;        /*!<  Current value of timeout       */
    unsigned char retransmit_state;     /*!<  Retransmission state           */
    mbedtls_ssl_flight_item *flight;    /*!<  Current outgoing flight        */
    mbedtls_ssl_flight_item *cur_msg;   /*!<  Current message in flight      */
    unsigned char *cur_msg_p;           /*!<  Position in current message    */
    unsigned int in_flight_start_seq;   /*!<  Minimum message sequence in the
                                              flight being received          */
    mbedtls_ssl_transform *alt_transform_out;   /*!<  Alternative transform for
                                              resending messages             */
    unsigned char alt_out_ctr[8];       /*!<  Alternative record epoch/counter
                                              for resending messages         */

    struct
    {
        size_t total_bytes_buffered; /*!< Cumulative size of heap allocated
                                      *   buffers used for message buffering. */

        uint8_t seen_ccs;               /*!< Indicates if a CCS message has
                                         *   been seen in the current flight. */

        struct mbedtls_ssl_hs_buffer
        {
            unsigned is_valid      : 1;
            unsigned is_fragmented : 1;
            unsigned is_complete   : 1;
            unsigned char *data;
            size_t data_len;
        } hs[MBEDTLS_SSL_MAX_BUFFERED_HS];

        struct
        {
            unsigned char *data;
            size_t len;
            unsigned epoch;
        } future_record;

    } buffering;

    uint16_t mtu;                       /*!<  Handshake mtu, used to fragment outgoing messages */
#endif /* MBEDTLS_SSL_PROTO_DTLS */

    /*
     * Checksum contexts
     */
#if defined(MBEDTLS_SSL_PROTO_SSL3) || defined(MBEDTLS_SSL_PROTO_TLS1) || \
    defined(MBEDTLS_SSL_PROTO_TLS1_1)
       mbedtls_md5_context fin_md5;
      mbedtls_sha1_context fin_sha1;
#endif
#if defined(MBEDTLS_SSL_PROTO_TLS1_2)
#if defined(MBEDTLS_SHA256_C)
    mbedtls_sha256_context fin_sha256;
#endif
#if defined(MBEDTLS_SHA512_C)
    mbedtls_sha512_context fin_sha512;
#endif
#endif /* MBEDTLS_SSL_PROTO_TLS1_2 */

    void (*update_checksum)(mbedtls_ssl_context *, const unsigned char *, size_t);
    void (*calc_verify)(mbedtls_ssl_context *, unsigned char *);
    void (*calc_finished)(mbedtls_ssl_context *, unsigned char *, int);
    int  (*tls_prf)(const unsigned char *, size_t, const char *,
                    const unsigned char *, size_t,
                    unsigned char *, size_t);

    size_t pmslen;                      /*!<  premaster length        */

    unsigned char randbytes[64];        /*!<  random bytes            */
    unsigned char premaster[MBEDTLS_PREMASTER_SIZE];
                                        /*!<  premaster secret        */

    int resume;                         /*!<  session resume indicator*/
    int max_major_ver;                  /*!< max. major version client*/
    int max_minor_ver;                  /*!< max. minor version client*/
    int cli_exts;                       /*!< client extension presence*/

#if defined(MBEDTLS_SSL_SESSION_TICKETS)
    int new_session_ticket;             /*!< use NewSessionTicket?    */
#endif /* MBEDTLS_SSL_SESSION_TICKETS */
#if defined(MBEDTLS_SSL_EXTENDED_MASTER_SECRET)
    int extended_ms;                    /*!< use Extended Master Secret? */
#endif

#if defined(MBEDTLS_SSL_ASYNC_PRIVATE)
    unsigned int async_in_progress : 1; /*!< an asynchronous operation is in progress */
#endif /* MBEDTLS_SSL_ASYNC_PRIVATE */

#if defined(MBEDTLS_SSL_ASYNC_PRIVATE)
    /** Asynchronous operation context. This field is meant for use by the
     * asynchronous operation callbacks (mbedtls_ssl_config::f_async_sign_start,
     * mbedtls_ssl_config::f_async_decrypt_start,
     * mbedtls_ssl_config::f_async_resume, mbedtls_ssl_config::f_async_cancel).
     * The library does not use it internally. */
    void *user_async_ctx;
#endif /* MBEDTLS_SSL_ASYNC_PRIVATE */
};

typedef struct mbedtls_ssl_hs_buffer mbedtls_ssl_hs_buffer;

/*
 * This structure contains a full set of runtime transform parameters
 * either in negotiation or active.
 */
struct mbedtls_ssl_transform
{
    /*
     * Session specific crypto layer
     */
    const mbedtls_ssl_ciphersuite_t *ciphersuite_info;
                                        /*!<  Chosen cipersuite_info  */
    unsigned int keylen;                /*!<  symmetric key length (bytes)  */
    size_t minlen;                      /*!<  min. ciphertext length  */
    size_t ivlen;                       /*!<  IV length               */
    size_t fixed_ivlen;                 /*!<  Fixed part of IV (AEAD) */
    size_t maclen;                      /*!<  MAC length              */

    unsigned char iv_enc[16];           /*!<  IV (encryption)         */
    unsigned char iv_dec[16];           /*!<  IV (decryption)         */

#if defined(MBEDTLS_SSL_PROTO_SSL3)
    /* Needed only for SSL v3.0 secret */
    unsigned char mac_enc[20];          /*!<  SSL v3.0 secret (enc)   */
    unsigned char mac_dec[20];          /*!<  SSL v3.0 secret (dec)   */
#endif /* MBEDTLS_SSL_PROTO_SSL3 */

    mbedtls_md_context_t md_ctx_enc;            /*!<  MAC (encryption)        */
    mbedtls_md_context_t md_ctx_dec;            /*!<  MAC (decryption)        */

    mbedtls_cipher_context_t cipher_ctx_enc;    /*!<  encryption context      */
    mbedtls_cipher_context_t cipher_ctx_dec;    /*!<  decryption context      */

    /*
     * Session specific compression layer
     */
#if defined(MBEDTLS_ZLIB_SUPPORT)
    z_stream ctx_deflate;               /*!<  compression context     */
    z_stream ctx_inflate;               /*!<  decompression context   */
#endif
};

#if defined(MBEDTLS_X509_CRT_PARSE_C)
/*
 * List of certificate + private key pairs
 */
struct mbedtls_ssl_key_cert
{
    mbedtls_x509_crt *cert;                 /*!< cert                       */
    mbedtls_pk_context *key;                /*!< private key                */
    mbedtls_ssl_key_cert *next;             /*!< next key/cert pair         */
};
#endif /* MBEDTLS_X509_CRT_PARSE_C */

#if defined(MBEDTLS_SSL_PROTO_DTLS)
/*
 * List of handshake messages kept around for resending
 */
struct mbedtls_ssl_flight_item
{
    unsigned char *p;       /*!< message, including handshake headers   */
    size_t len;             /*!< length of p                            */
    unsigned char type;     /*!< type of the message: handshake or CCS  */
    mbedtls_ssl_flight_item *next;  /*!< next handshake message(s)              */
};
#endif /* MBEDTLS_SSL_PROTO_DTLS */

#if defined(MBEDTLS_SSL_PROTO_TLS1_2) && \
    defined(MBEDTLS_KEY_EXCHANGE__WITH_CERT__ENABLED)

/* Find an entry in a signature-hash set matching a given hash algorithm. */
mbedtls_md_type_t mbedtls_ssl_sig_hash_set_find( mbedtls_ssl_sig_hash_set_t *set,
                                                 mbedtls_pk_type_t sig_alg );
/* Add a signature-hash-pair to a signature-hash set */
void mbedtls_ssl_sig_hash_set_add( mbedtls_ssl_sig_hash_set_t *set,
                                   mbedtls_pk_type_t sig_alg,
                                   mbedtls_md_type_t md_alg );
/* Allow exactly one hash algorithm for each signature. */
void mbedtls_ssl_sig_hash_set_const_hash( mbedtls_ssl_sig_hash_set_t *set,
                                          mbedtls_md_type_t md_alg );

/* Setup an empty signature-hash set */
static inline void mbedtls_ssl_sig_hash_set_init( mbedtls_ssl_sig_hash_set_t *set )
{
    mbedtls_ssl_sig_hash_set_const_hash( set, MBEDTLS_MD_NONE );
}

#endif /* MBEDTLS_SSL_PROTO_TLS1_2) &&
          MBEDTLS_KEY_EXCHANGE__WITH_CERT__ENABLED */

/**
 * \brief           Free referenced items in an SSL transform context and clear
 *                  memory
 *
 * \param transform SSL transform context
 */
void mbedtls_ssl_transform_free( mbedtls_ssl_transform *transform );

/**
 * \brief           Free referenced items in an SSL handshake context and clear
 *                  memory
 *
 * \param ssl       SSL context
 */
void mbedtls_ssl_handshake_free( mbedtls_ssl_context *ssl );

int mbedtls_ssl_handshake_client_step( mbedtls_ssl_context *ssl );
int mbedtls_ssl_handshake_server_step( mbedtls_ssl_context *ssl );
void mbedtls_ssl_handshake_wrapup( mbedtls_ssl_context *ssl );

int mbedtls_ssl_send_fatal_handshake_failure( mbedtls_ssl_context *ssl );

void mbedtls_ssl_reset_checksum( mbedtls_ssl_context *ssl );
int mbedtls_ssl_derive_keys( mbedtls_ssl_context *ssl );

int mbedtls_ssl_handle_message_type( mbedtls_ssl_context *ssl );
int mbedtls_ssl_prepare_handshake_record( mbedtls_ssl_context *ssl );
void mbedtls_ssl_update_handshake_status( mbedtls_ssl_context *ssl );

/**
 * \brief       Update record layer
 *
 *              This function roughly separates the implementation
 *              of the logic of (D)TLS from the implementation
 *              of the secure transport.
 *
 * \param  ssl              The SSL context to use.
 * \param  update_hs_digest This indicates if the handshake digest
 *                          should be automatically updated in case
 *                          a handshake message is found.
 *
 * \return      0 or non-zero error code.
 *
 * \note        A clarification on what is called 'record layer' here
 *              is in order, as many sensible definitions are possible:
 *
 *              The record layer takes as input an untrusted underlying
 *              transport (stream or datagram) and transforms it into
 *              a serially multiplexed, secure transport, which
 *              conceptually provides the following:
 *
 *              (1) Three datagram based, content-agnostic transports
 *                  for handshake, alert and CCS messages.
 *              (2) One stream- or datagram-based transport
 *                  for application data.
 *              (3) Functionality for changing the underlying transform
 *                  securing the contents.
 *
 *              The interface to this functionality is given as follows:
 *
 *              a Updating
 *                [Currently implemented by mbedtls_ssl_read_record]
 *
 *                Check if and on which of the four 'ports' data is pending:
 *                Nothing, a controlling datagram of type (1), or application
 *                data (2). In any case data is present, internal buffers
 *                provide access to the data for the user to process it.
 *                Consumption of type (1) datagrams is done automatically
 *                on the next update, invalidating that the internal buffers
 *                for previous datagrams, while consumption of application
 *                data (2) is user-controlled.
 *
 *              b Reading of application data
 *                [Currently manual adaption of ssl->in_offt pointer]
 *
 *                As mentioned in the last paragraph, consumption of data
 *                is different from the automatic consumption of control
 *                datagrams (1) because application data is treated as a stream.
 *
 *              c Tracking availability of application data
 *                [Currently manually through decreasing ssl->in_msglen]
 *
 *                For efficiency and to retain datagram semantics for
 *                application data in case of DTLS, the record layer
 *                provides functionality for checking how much application
 *                data is still available in the internal buffer.
 *
 *              d Changing the transformation securing the communication.
 *
 *              Given an opaque implementation of the record layer in the
 *              above sense, it should be possible to implement the logic
 *              of (D)TLS on top of it without the need to know anything
 *              about the record layer's internals. This is done e.g.
 *              in all the handshake handling functions, and in the
 *              application data reading function mbedtls_ssl_read.
 *
 * \note        The above tries to give a conceptual picture of the
 *              record layer, but the current implementation deviates
 *              from it in some places. For example, our implementation of
 *              the update functionality through mbedtls_ssl_read_record
 *              discards datagrams depending on the current state, which
 *              wouldn't fall under the record layer's responsibility
 *              following the above definition.
 *
 */
int mbedtls_ssl_read_record( mbedtls_ssl_context *ssl,
                             unsigned update_hs_digest );
int mbedtls_ssl_fetch_input( mbedtls_ssl_context *ssl, size_t nb_want );

int mbedtls_ssl_write_handshake_msg( mbedtls_ssl_context *ssl );
int mbedtls_ssl_write_record( mbedtls_ssl_context *ssl, uint8_t force_flush );
int mbedtls_ssl_flush_output( mbedtls_ssl_context *ssl );

int mbedtls_ssl_parse_certificate( mbedtls_ssl_context *ssl );
int mbedtls_ssl_write_certificate( mbedtls_ssl_context *ssl );

int mbedtls_ssl_parse_change_cipher_spec( mbedtls_ssl_context *ssl );
int mbedtls_ssl_write_change_cipher_spec( mbedtls_ssl_context *ssl );

int mbedtls_ssl_parse_finished( mbedtls_ssl_context *ssl );
int mbedtls_ssl_write_finished( mbedtls_ssl_context *ssl );

void mbedtls_ssl_optimize_checksum( mbedtls_ssl_context *ssl,
                            const mbedtls_ssl_ciphersuite_t *ciphersuite_info );

#if defined(MBEDTLS_KEY_EXCHANGE__SOME__PSK_ENABLED)
int mbedtls_ssl_psk_derive_premaster( mbedtls_ssl_context *ssl, mbedtls_key_exchange_type_t key_ex );
#endif

#if defined(MBEDTLS_PK_C)
unsigned char mbedtls_ssl_sig_from_pk( mbedtls_pk_context *pk );
unsigned char mbedtls_ssl_sig_from_pk_alg( mbedtls_pk_type_t type );
mbedtls_pk_type_t mbedtls_ssl_pk_alg_from_sig( unsigned char sig );
#endif

mbedtls_md_type_t mbedtls_ssl_md_alg_from_hash( unsigned char hash );
unsigned char mbedtls_ssl_hash_from_md_alg( int md );
int mbedtls_ssl_set_calc_verify_md( mbedtls_ssl_context *ssl, int md );

#if defined(MBEDTLS_ECP_C)
int mbedtls_ssl_check_curve( const mbedtls_ssl_context *ssl, mbedtls_ecp_group_id grp_id );
#endif

#if defined(MBEDTLS_KEY_EXCHANGE__WITH_CERT__ENABLED)
int mbedtls_ssl_check_sig_hash( const mbedtls_ssl_context *ssl,
                                mbedtls_md_type_t md );
#endif

#if defined(MBEDTLS_X509_CRT_PARSE_C)
static inline mbedtls_pk_context *mbedtls_ssl_own_key( mbedtls_ssl_context *ssl )
{
    mbedtls_ssl_key_cert *key_cert;

    if( ssl->handshake != NULL && ssl->handshake->key_cert != NULL )
        key_cert = ssl->handshake->key_cert;
    else
        key_cert = ssl->conf->key_cert;

    return( key_cert == NULL ? NULL : key_cert->key );
}

static inline mbedtls_x509_crt *mbedtls_ssl_own_cert( mbedtls_ssl_context *ssl )
{
    mbedtls_ssl_key_cert *key_cert;

    if( ssl->handshake != NULL && ssl->handshake->key_cert != NULL )
        key_cert = ssl->handshake->key_cert;
    else
        key_cert = ssl->conf->key_cert;

    return( key_cert == NULL ? NULL : key_cert->cert );
}

/*
 * Check usage of a certificate wrt extensions:
 * keyUsage, extendedKeyUsage (later), and nSCertType (later).
 *
 * Warning: cert_endpoint is the endpoint of the cert (ie, of our peer when we
 * check a cert we received from them)!
 *
 * Return 0 if everything is OK, -1 if not.
 */
int mbedtls_ssl_check_cert_usage( const mbedtls_x509_crt *cert,
                          const mbedtls_ssl_ciphersuite_t *ciphersuite,
                          int cert_endpoint,
                          uint32_t *flags );
#endif /* MBEDTLS_X509_CRT_PARSE_C */

void mbedtls_ssl_write_version( int major, int minor, int transport,
                        unsigned char ver[2] );
void mbedtls_ssl_read_version( int *major, int *minor, int transport,
                       const unsigned char ver[2] );

static inline size_t mbedtls_ssl_hdr_len( const mbedtls_ssl_context *ssl )
{
#if defined(MBEDTLS_SSL_PROTO_DTLS)
    if( ssl->conf->transport == MBEDTLS_SSL_TRANSPORT_DATAGRAM )
        return( 13 );
#else
    ((void) ssl);
#endif
    return( 5 );
}

static inline size_t mbedtls_ssl_hs_hdr_len( const mbedtls_ssl_context *ssl )
{
#if defined(MBEDTLS_SSL_PROTO_DTLS)
    if( ssl->conf->transport == MBEDTLS_SSL_TRANSPORT_DATAGRAM )
        return( 12 );
#else
    ((void) ssl);
#endif
    return( 4 );
}

#if defined(MBEDTLS_SSL_PROTO_DTLS)
void mbedtls_ssl_send_flight_completed( mbedtls_ssl_context *ssl );
void mbedtls_ssl_recv_flight_completed( mbedtls_ssl_context *ssl );
int mbedtls_ssl_resend( mbedtls_ssl_context *ssl );
int mbedtls_ssl_flight_transmit( mbedtls_ssl_context *ssl );
#endif

/* Visible for testing purposes only */
#if defined(MBEDTLS_SSL_DTLS_ANTI_REPLAY)
int mbedtls_ssl_dtls_replay_check( mbedtls_ssl_context *ssl );
void mbedtls_ssl_dtls_replay_update( mbedtls_ssl_context *ssl );
#endif

/* constant-time buffer comparison */
static inline int mbedtls_ssl_safer_memcmp( const void *a, const void *b, size_t n )
{
    size_t i;
    volatile const unsigned char *A = (volatile const unsigned char *) a;
    volatile const unsigned char *B = (volatile const unsigned char *) b;
    volatile unsigned char diff = 0;

    for( i = 0; i < n; i++ )
    {
        /* Read volatile data in order before computing diff.
         * This avoids IAR compiler warning:
         * 'the order of volatile accesses is undefined ..' */
        unsigned char x = A[i], y = B[i];
        diff |= x ^ y;
    }

    return( diff );
}

#if defined(MBEDTLS_SSL_PROTO_SSL3) || defined(MBEDTLS_SSL_PROTO_TLS1) || \
    defined(MBEDTLS_SSL_PROTO_TLS1_1)
int mbedtls_ssl_get_key_exchange_md_ssl_tls( mbedtls_ssl_context *ssl,
                                        unsigned char *output,
                                        unsigned char *data, size_t data_len );
#endif /* MBEDTLS_SSL_PROTO_SSL3 || MBEDTLS_SSL_PROTO_TLS1 || \
          MBEDTLS_SSL_PROTO_TLS1_1 */

#if defined(MBEDTLS_SSL_PROTO_TLS1) || defined(MBEDTLS_SSL_PROTO_TLS1_1) || \
    defined(MBEDTLS_SSL_PROTO_TLS1_2)
int mbedtls_ssl_get_key_exchange_md_tls1_2( mbedtls_ssl_context *ssl,
                                            unsigned char *hash, size_t *hashlen,
                                            unsigned char *data, size_t data_len,
                                            mbedtls_md_type_t md_alg );
#endif /* MBEDTLS_SSL_PROTO_TLS1 || MBEDTLS_SSL_PROTO_TLS1_1 || \
          MBEDTLS_SSL_PROTO_TLS1_2 */

#if defined(MBEDTLS_SSL_SOME_SUITES_USE_TLS_CBC)
/** \brief Compute the HMAC of variable-length data with constant flow.
 *
 * This function computes the HMAC of the concatenation of \p add_data and \p
 * data, and does with a code flow and memory access pattern that does not
 * depend on \p data_len_secret, but only on \p min_data_len and \p
 * max_data_len. In particular, this function always reads exactly \p
 * max_data_len bytes from \p data.
 *
 * \param ctx               The HMAC context. It must have keys configured
 *                          with mbedtls_md_hmac_starts() and use one of the
 *                          following hashes: SHA-384, SHA-256, SHA-1 or MD-5.
 *                          It is reset using mbedtls_md_hmac_reset() after
 *                          the computation is complete to prepare for the
 *                          next computation.
 * \param add_data          The additional data prepended to \p data. This
 *                          must point to a readable buffer of \p add_data_len
 *                          bytes.
 * \param add_data_len      The length of \p add_data in bytes.
 * \param data              The data appended to \p add_data. This must point
 *                          to a readable buffer of \p max_data_len bytes.
 * \param data_len_secret   The length of the data to process in \p data.
 *                          This must be no less than \p min_data_len and no
 *                          greater than \p max_data_len.
 * \param min_data_len      The minimal length of \p data in bytes.
 * \param max_data_len      The maximal length of \p data in bytes.
 * \param output            The HMAC will be written here. This must point to
 *                          a writable buffer of sufficient size to hold the
 *                          HMAC value.
 *
 * \retval 0
 *         Success.
 * \retval MBEDTLS_ERR_PLATFORM_HW_ACCEL_FAILED
 *         The hardware accelerator failed.
 */
int mbedtls_ssl_cf_hmac(
        mbedtls_md_context_t *ctx,
        const unsigned char *add_data, size_t add_data_len,
        const unsigned char *data, size_t data_len_secret,
        size_t min_data_len, size_t max_data_len,
        unsigned char *output );

/** \brief Copy data from a secret position with constant flow.
 *
 * This function copies \p len bytes from \p src_base + \p offset_secret to \p
 * dst, with a code flow and memory access pattern that does not depend on \p
 * offset_secret, but only on \p offset_min, \p offset_max and \p len.
 *
 * \param dst           The destination buffer. This must point to a writable
 *                      buffer of at least \p len bytes.
 * \param src_base      The base of the source buffer. This must point to a
 *                      readable buffer of at least \p offset_max + \p len
 *                      bytes.
 * \param offset_secret The offset in the source buffer from which to copy.
 *                      This must be no less than \p offset_min and no greater
 *                      than \p offset_max.
 * \param offset_min    The minimal value of \p offset_secret.
 * \param offset_max    The maximal value of \p offset_secret.
 * \param len           The number of bytes to copy.
 */
void mbedtls_ssl_cf_memcpy_offset( unsigned char *dst,
                                   const unsigned char *src_base,
                                   size_t offset_secret,
                                   size_t offset_min, size_t offset_max,
                                   size_t len );
#endif /* MBEDTLS_SSL_SOME_SUITES_USE_TLS_CBC */

#ifdef __cplusplus
}
#endif

#endif /* ssl_internal.h */
