/*
 *  SSLv3/TLSv1 server-side functions
 *
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

#include "common.h"

#if defined(MBEDTLS_SSL_SRV_C)

#if defined(MBEDTLS_PLATFORM_C)
#include "mbedtls/platform.h"
#else
#include <stdlib.h>
#define mbedtls_calloc    calloc
#define mbedtls_free      free
#endif

#include "mbedtls/ssl.h"
#include "mbedtls/ssl_internal.h"
#include "mbedtls/debug.h"
#include "mbedtls/error.h"
#include "mbedtls/platform_util.h"
#include "constant_time_internal.h"
#include "mbedtls/constant_time.h"

#include <string.h>

#if defined(MBEDTLS_ECP_C)
#include "mbedtls/ecp.h"
#endif

#if defined(MBEDTLS_HAVE_TIME)
#include "mbedtls/platform_time.h"
#endif

#if defined(MBEDTLS_SSL_DTLS_HELLO_VERIFY)
int mbedtls_ssl_set_client_transport_id( mbedtls_ssl_context *ssl,
                                 const unsigned char *info,
                                 size_t ilen )
{
    if( ssl->conf->endpoint != MBEDTLS_SSL_IS_SERVER )
        return( MBEDTLS_ERR_SSL_BAD_INPUT_DATA );

    mbedtls_free( ssl->cli_id );

    if( ( ssl->cli_id = mbedtls_calloc( 1, ilen ) ) == NULL )
        return( MBEDTLS_ERR_SSL_ALLOC_FAILED );

    memcpy( ssl->cli_id, info, ilen );
    ssl->cli_id_len = ilen;

    return( 0 );
}

void mbedtls_ssl_conf_dtls_cookies( mbedtls_ssl_config *conf,
                           mbedtls_ssl_cookie_write_t *f_cookie_write,
                           mbedtls_ssl_cookie_check_t *f_cookie_check,
                           void *p_cookie )
{
    conf->f_cookie_write = f_cookie_write;
    conf->f_cookie_check = f_cookie_check;
    conf->p_cookie       = p_cookie;
}
#endif /* MBEDTLS_SSL_DTLS_HELLO_VERIFY */

#if defined(MBEDTLS_SSL_SERVER_NAME_INDICATION)
static int ssl_parse_servername_ext( mbedtls_ssl_context *ssl,
                                     const unsigned char *buf,
                                     size_t len )
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    size_t servername_list_size, hostname_len;
    const unsigned char *p;

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "parse ServerName extension" ) );

    if( len < 2 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                       MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }
    servername_list_size = ( ( buf[0] << 8 ) | ( buf[1] ) );
    if( servername_list_size + 2 != len )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

    p = buf + 2;
    while( servername_list_size > 2 )
    {
        hostname_len = ( ( p[1] << 8 ) | p[2] );
        if( hostname_len + 3 > servername_list_size )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
            mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                            MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
            return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
        }

        if( p[0] == MBEDTLS_TLS_EXT_SERVERNAME_HOSTNAME )
        {
            ret = ssl->conf->f_sni( ssl->conf->p_sni,
                                    ssl, p + 3, hostname_len );
            if( ret != 0 )
            {
                MBEDTLS_SSL_DEBUG_RET( 1, "ssl_sni_wrapper", ret );
                mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                        MBEDTLS_SSL_ALERT_MSG_UNRECOGNIZED_NAME );
                return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
            }
            return( 0 );
        }

        servername_list_size -= hostname_len + 3;
        p += hostname_len + 3;
    }

    if( servername_list_size != 0 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_ILLEGAL_PARAMETER );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

    return( 0 );
}
#endif /* MBEDTLS_SSL_SERVER_NAME_INDICATION */

#if defined(MBEDTLS_KEY_EXCHANGE_SOME_PSK_ENABLED)
static int ssl_conf_has_psk_or_cb( mbedtls_ssl_config const *conf )
{
    if( conf->f_psk != NULL )
        return( 1 );

    if( conf->psk_identity_len == 0 || conf->psk_identity == NULL )
        return( 0 );

    if( conf->psk != NULL && conf->psk_len != 0 )
        return( 1 );

#if defined(MBEDTLS_USE_PSA_CRYPTO)
    if( ! mbedtls_svc_key_id_is_null( conf->psk_opaque ) )
        return( 1 );
#endif /* MBEDTLS_USE_PSA_CRYPTO */

    return( 0 );
}

#if defined(MBEDTLS_USE_PSA_CRYPTO)
static int ssl_use_opaque_psk( mbedtls_ssl_context const *ssl )
{
    if( ssl->conf->f_psk != NULL )
    {
        /* If we've used a callback to select the PSK,
         * the static configuration is irrelevant. */

        if( ! mbedtls_svc_key_id_is_null( ssl->handshake->psk_opaque ) )
            return( 1 );

        return( 0 );
    }

    if( ! mbedtls_svc_key_id_is_null( ssl->conf->psk_opaque ) )
        return( 1 );

    return( 0 );
}
#endif /* MBEDTLS_USE_PSA_CRYPTO */
#endif /* MBEDTLS_KEY_EXCHANGE_SOME_PSK_ENABLED */

static int ssl_parse_renegotiation_info( mbedtls_ssl_context *ssl,
                                         const unsigned char *buf,
                                         size_t len )
{
#if defined(MBEDTLS_SSL_RENEGOTIATION)
    if( ssl->renego_status != MBEDTLS_SSL_INITIAL_HANDSHAKE )
    {
        /* Check verify-data in constant-time. The length OTOH is no secret */
        if( len    != 1 + ssl->verify_data_len ||
            buf[0] !=     ssl->verify_data_len ||
            mbedtls_ct_memcmp( buf + 1, ssl->peer_verify_data,
                          ssl->verify_data_len ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "non-matching renegotiation info" ) );
            mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                            MBEDTLS_SSL_ALERT_MSG_HANDSHAKE_FAILURE );
            return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
        }
    }
    else
#endif /* MBEDTLS_SSL_RENEGOTIATION */
    {
        if( len != 1 || buf[0] != 0x0 )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "non-zero length renegotiation info" ) );
            mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                            MBEDTLS_SSL_ALERT_MSG_HANDSHAKE_FAILURE );
            return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
        }

        ssl->secure_renegotiation = MBEDTLS_SSL_SECURE_RENEGOTIATION;
    }

    return( 0 );
}

#if defined(MBEDTLS_SSL_PROTO_TLS1_2) && \
    defined(MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED)

/*
 * Status of the implementation of signature-algorithms extension:
 *
 * Currently, we are only considering the signature-algorithm extension
 * to pick a ciphersuite which allows us to send the ServerKeyExchange
 * message with a signature-hash combination that the user allows.
 *
 * We do *not* check whether all certificates in our certificate
 * chain are signed with an allowed signature-hash pair.
 * This needs to be done at a later stage.
 *
 */
static int ssl_parse_signature_algorithms_ext( mbedtls_ssl_context *ssl,
                                               const unsigned char *buf,
                                               size_t len )
{
    size_t sig_alg_list_size;

    const unsigned char *p;
    const unsigned char *end = buf + len;

    mbedtls_md_type_t md_cur;
    mbedtls_pk_type_t sig_cur;

    if ( len < 2 ) {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                       MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }
    sig_alg_list_size = ( ( buf[0] << 8 ) | ( buf[1] ) );
    if( sig_alg_list_size + 2 != len ||
        sig_alg_list_size % 2 != 0 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

    /* Currently we only guarantee signing the ServerKeyExchange message according
     * to the constraints specified in this extension (see above), so it suffices
     * to remember only one suitable hash for each possible signature algorithm.
     *
     * This will change when we also consider certificate signatures,
     * in which case we will need to remember the whole signature-hash
     * pair list from the extension.
     */

    for( p = buf + 2; p < end; p += 2 )
    {
        /* Silently ignore unknown signature or hash algorithms. */

        if( ( sig_cur = mbedtls_ssl_pk_alg_from_sig( p[1] ) ) == MBEDTLS_PK_NONE )
        {
            MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello v3, signature_algorithm ext"
                                        " unknown sig alg encoding %d", p[1] ) );
            continue;
        }

        /* Check if we support the hash the user proposes */
        md_cur = mbedtls_ssl_md_alg_from_hash( p[0] );
        if( md_cur == MBEDTLS_MD_NONE )
        {
            MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello v3, signature_algorithm ext:"
                                        " unknown hash alg encoding %d", p[0] ) );
            continue;
        }

        if( mbedtls_ssl_check_sig_hash( ssl, md_cur ) == 0 )
        {
            mbedtls_ssl_sig_hash_set_add( &ssl->handshake->hash_algs, sig_cur, md_cur );
            MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello v3, signature_algorithm ext:"
                                        " match sig %u and hash %u",
                                        (unsigned) sig_cur, (unsigned) md_cur ) );
        }
        else
        {
            MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello v3, signature_algorithm ext: "
                                        "hash alg %u not supported", (unsigned) md_cur ) );
        }
    }

    return( 0 );
}
#endif /* MBEDTLS_SSL_PROTO_TLS1_2 &&
          MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED */

#if defined(MBEDTLS_ECDH_C) || defined(MBEDTLS_ECDSA_C) || \
    defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
static int ssl_parse_supported_elliptic_curves( mbedtls_ssl_context *ssl,
                                                const unsigned char *buf,
                                                size_t len )
{
    size_t list_size, our_size;
    const unsigned char *p;
    const mbedtls_ecp_curve_info *curve_info, **curves;

    if ( len < 2 ) {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                       MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }
    list_size = ( ( buf[0] << 8 ) | ( buf[1] ) );
    if( list_size + 2 != len ||
        list_size % 2 != 0 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

    /* Should never happen unless client duplicates the extension */
    if( ssl->handshake->curves != NULL )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

    /* Don't allow our peer to make us allocate too much memory,
     * and leave room for a final 0 */
    our_size = list_size / 2 + 1;
    if( our_size > MBEDTLS_ECP_DP_MAX )
        our_size = MBEDTLS_ECP_DP_MAX;

    if( ( curves = mbedtls_calloc( our_size, sizeof( *curves ) ) ) == NULL )
    {
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_INTERNAL_ERROR );
        return( MBEDTLS_ERR_SSL_ALLOC_FAILED );
    }

    ssl->handshake->curves = curves;

    p = buf + 2;
    while( list_size > 0 && our_size > 1 )
    {
        curve_info = mbedtls_ecp_curve_info_from_tls_id( ( p[0] << 8 ) | p[1] );

        if( curve_info != NULL )
        {
            *curves++ = curve_info;
            our_size--;
        }

        list_size -= 2;
        p += 2;
    }

    return( 0 );
}

static int ssl_parse_supported_point_formats( mbedtls_ssl_context *ssl,
                                              const unsigned char *buf,
                                              size_t len )
{
    size_t list_size;
    const unsigned char *p;

    if( len == 0 || (size_t)( buf[0] + 1 ) != len )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }
    list_size = buf[0];

    p = buf + 1;
    while( list_size > 0 )
    {
        if( p[0] == MBEDTLS_ECP_PF_UNCOMPRESSED ||
            p[0] == MBEDTLS_ECP_PF_COMPRESSED )
        {
#if defined(MBEDTLS_ECDH_C) || defined(MBEDTLS_ECDSA_C)
            ssl->handshake->ecdh_ctx.point_format = p[0];
#endif
#if defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
            ssl->handshake->ecjpake_ctx.point_format = p[0];
#endif
            MBEDTLS_SSL_DEBUG_MSG( 4, ( "point format selected: %d", p[0] ) );
            return( 0 );
        }

        list_size--;
        p++;
    }

    return( 0 );
}
#endif /* MBEDTLS_ECDH_C || MBEDTLS_ECDSA_C ||
          MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED */

#if defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
static int ssl_parse_ecjpake_kkpp( mbedtls_ssl_context *ssl,
                                   const unsigned char *buf,
                                   size_t len )
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;

    if( mbedtls_ecjpake_check( &ssl->handshake->ecjpake_ctx ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 3, ( "skip ecjpake kkpp extension" ) );
        return( 0 );
    }

    if( ( ret = mbedtls_ecjpake_read_round_one( &ssl->handshake->ecjpake_ctx,
                                                buf, len ) ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ecjpake_read_round_one", ret );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_ILLEGAL_PARAMETER );
        return( ret );
    }

    /* Only mark the extension as OK when we're sure it is */
    ssl->handshake->cli_exts |= MBEDTLS_TLS_EXT_ECJPAKE_KKPP_OK;

    return( 0 );
}
#endif /* MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED */

#if defined(MBEDTLS_SSL_MAX_FRAGMENT_LENGTH)
static int ssl_parse_max_fragment_length_ext( mbedtls_ssl_context *ssl,
                                              const unsigned char *buf,
                                              size_t len )
{
    if( len != 1 || buf[0] >= MBEDTLS_SSL_MAX_FRAG_LEN_INVALID )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_ILLEGAL_PARAMETER );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

    ssl->session_negotiate->mfl_code = buf[0];

    return( 0 );
}
#endif /* MBEDTLS_SSL_MAX_FRAGMENT_LENGTH */

#if defined(MBEDTLS_SSL_DTLS_CONNECTION_ID)
static int ssl_parse_cid_ext( mbedtls_ssl_context *ssl,
                              const unsigned char *buf,
                              size_t len )
{
    size_t peer_cid_len;

    /* CID extension only makes sense in DTLS */
    if( ssl->conf->transport != MBEDTLS_SSL_TRANSPORT_DATAGRAM )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_ILLEGAL_PARAMETER );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

    /*
     * Quoting draft-ietf-tls-dtls-connection-id-05
     * https://tools.ietf.org/html/draft-ietf-tls-dtls-connection-id-05
     *
     *   struct {
     *      opaque cid<0..2^8-1>;
     *   } ConnectionId;
    */

    if( len < 1 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_ILLEGAL_PARAMETER );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

    peer_cid_len = *buf++;
    len--;

    if( len != peer_cid_len )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_ILLEGAL_PARAMETER );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

    /* Ignore CID if the user has disabled its use. */
    if( ssl->negotiate_cid == MBEDTLS_SSL_CID_DISABLED )
    {
        /* Leave ssl->handshake->cid_in_use in its default
         * value of MBEDTLS_SSL_CID_DISABLED. */
        MBEDTLS_SSL_DEBUG_MSG( 3, ( "Client sent CID extension, but CID disabled" ) );
        return( 0 );
    }

    if( peer_cid_len > MBEDTLS_SSL_CID_OUT_LEN_MAX )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_ILLEGAL_PARAMETER );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

    ssl->handshake->cid_in_use = MBEDTLS_SSL_CID_ENABLED;
    ssl->handshake->peer_cid_len = (uint8_t) peer_cid_len;
    memcpy( ssl->handshake->peer_cid, buf, peer_cid_len );

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "Use of CID extension negotiated" ) );
    MBEDTLS_SSL_DEBUG_BUF( 3, "Client CID", buf, peer_cid_len );

    return( 0 );
}
#endif /* MBEDTLS_SSL_DTLS_CONNECTION_ID */

#if defined(MBEDTLS_SSL_TRUNCATED_HMAC)
static int ssl_parse_truncated_hmac_ext( mbedtls_ssl_context *ssl,
                                         const unsigned char *buf,
                                         size_t len )
{
    if( len != 0 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

    ((void) buf);

    if( ssl->conf->trunc_hmac == MBEDTLS_SSL_TRUNC_HMAC_ENABLED )
        ssl->session_negotiate->trunc_hmac = MBEDTLS_SSL_TRUNC_HMAC_ENABLED;

    return( 0 );
}
#endif /* MBEDTLS_SSL_TRUNCATED_HMAC */

#if defined(MBEDTLS_SSL_ENCRYPT_THEN_MAC)
static int ssl_parse_encrypt_then_mac_ext( mbedtls_ssl_context *ssl,
                                      const unsigned char *buf,
                                      size_t len )
{
    if( len != 0 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

    ((void) buf);

    if( ssl->conf->encrypt_then_mac == MBEDTLS_SSL_ETM_ENABLED &&
        ssl->minor_ver != MBEDTLS_SSL_MINOR_VERSION_0 )
    {
        ssl->session_negotiate->encrypt_then_mac = MBEDTLS_SSL_ETM_ENABLED;
    }

    return( 0 );
}
#endif /* MBEDTLS_SSL_ENCRYPT_THEN_MAC */

#if defined(MBEDTLS_SSL_EXTENDED_MASTER_SECRET)
static int ssl_parse_extended_ms_ext( mbedtls_ssl_context *ssl,
                                      const unsigned char *buf,
                                      size_t len )
{
    if( len != 0 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

    ((void) buf);

    if( ssl->conf->extended_ms == MBEDTLS_SSL_EXTENDED_MS_ENABLED &&
        ssl->minor_ver != MBEDTLS_SSL_MINOR_VERSION_0 )
    {
        ssl->handshake->extended_ms = MBEDTLS_SSL_EXTENDED_MS_ENABLED;
    }

    return( 0 );
}
#endif /* MBEDTLS_SSL_EXTENDED_MASTER_SECRET */

#if defined(MBEDTLS_SSL_SESSION_TICKETS)
static int ssl_parse_session_ticket_ext( mbedtls_ssl_context *ssl,
                                         unsigned char *buf,
                                         size_t len )
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    mbedtls_ssl_session session;

    mbedtls_ssl_session_init( &session );

    if( ssl->conf->f_ticket_parse == NULL ||
        ssl->conf->f_ticket_write == NULL )
    {
        return( 0 );
    }

    /* Remember the client asked us to send a new ticket */
    ssl->handshake->new_session_ticket = 1;

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "ticket length: %" MBEDTLS_PRINTF_SIZET, len ) );

    if( len == 0 )
        return( 0 );

#if defined(MBEDTLS_SSL_RENEGOTIATION)
    if( ssl->renego_status != MBEDTLS_SSL_INITIAL_HANDSHAKE )
    {
        MBEDTLS_SSL_DEBUG_MSG( 3, ( "ticket rejected: renegotiating" ) );
        return( 0 );
    }
#endif /* MBEDTLS_SSL_RENEGOTIATION */

    /*
     * Failures are ok: just ignore the ticket and proceed.
     */
    if( ( ret = ssl->conf->f_ticket_parse( ssl->conf->p_ticket, &session,
                                           buf, len ) ) != 0 )
    {
        mbedtls_ssl_session_free( &session );

        if( ret == MBEDTLS_ERR_SSL_INVALID_MAC )
            MBEDTLS_SSL_DEBUG_MSG( 3, ( "ticket is not authentic" ) );
        else if( ret == MBEDTLS_ERR_SSL_SESSION_TICKET_EXPIRED )
            MBEDTLS_SSL_DEBUG_MSG( 3, ( "ticket is expired" ) );
        else
            MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ssl_ticket_parse", ret );

        return( 0 );
    }

    /*
     * Keep the session ID sent by the client, since we MUST send it back to
     * inform them we're accepting the ticket  (RFC 5077 section 3.4)
     */
    session.id_len = ssl->session_negotiate->id_len;
    memcpy( &session.id, ssl->session_negotiate->id, session.id_len );

    mbedtls_ssl_session_free( ssl->session_negotiate );
    memcpy( ssl->session_negotiate, &session, sizeof( mbedtls_ssl_session ) );

    /* Zeroize instead of free as we copied the content */
    mbedtls_platform_zeroize( &session, sizeof( mbedtls_ssl_session ) );

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "session successfully restored from ticket" ) );

    ssl->handshake->resume = 1;

    /* Don't send a new ticket after all, this one is OK */
    ssl->handshake->new_session_ticket = 0;

    return( 0 );
}
#endif /* MBEDTLS_SSL_SESSION_TICKETS */

#if defined(MBEDTLS_SSL_ALPN)
static int ssl_parse_alpn_ext( mbedtls_ssl_context *ssl,
                               const unsigned char *buf, size_t len )
{
    size_t list_len, cur_len, ours_len;
    const unsigned char *theirs, *start, *end;
    const char **ours;

    /* If ALPN not configured, just ignore the extension */
    if( ssl->conf->alpn_list == NULL )
        return( 0 );

    /*
     * opaque ProtocolName<1..2^8-1>;
     *
     * struct {
     *     ProtocolName protocol_name_list<2..2^16-1>
     * } ProtocolNameList;
     */

    /* Min length is 2 (list_len) + 1 (name_len) + 1 (name) */
    if( len < 4 )
    {
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

    list_len = ( buf[0] << 8 ) | buf[1];
    if( list_len != len - 2 )
    {
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

    /*
     * Validate peer's list (lengths)
     */
    start = buf + 2;
    end = buf + len;
    for( theirs = start; theirs != end; theirs += cur_len )
    {
        cur_len = *theirs++;

        /* Current identifier must fit in list */
        if( cur_len > (size_t)( end - theirs ) )
        {
            mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                            MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
            return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
        }

        /* Empty strings MUST NOT be included */
        if( cur_len == 0 )
        {
            mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                            MBEDTLS_SSL_ALERT_MSG_ILLEGAL_PARAMETER );
            return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
        }
    }

    /*
     * Use our order of preference
     */
    for( ours = ssl->conf->alpn_list; *ours != NULL; ours++ )
    {
        ours_len = strlen( *ours );
        for( theirs = start; theirs != end; theirs += cur_len )
        {
            cur_len = *theirs++;

            if( cur_len == ours_len &&
                memcmp( theirs, *ours, cur_len ) == 0 )
            {
                ssl->alpn_chosen = *ours;
                return( 0 );
            }
        }
    }

    /* If we get there, no match was found */
    mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                            MBEDTLS_SSL_ALERT_MSG_NO_APPLICATION_PROTOCOL );
    return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
}
#endif /* MBEDTLS_SSL_ALPN */

#if defined(MBEDTLS_SSL_DTLS_SRTP)
static int ssl_parse_use_srtp_ext( mbedtls_ssl_context *ssl,
                                   const unsigned char *buf,
                                   size_t len )
{
    mbedtls_ssl_srtp_profile client_protection = MBEDTLS_TLS_SRTP_UNSET;
    size_t i,j;
    size_t profile_length;
    uint16_t mki_length;
    /*! 2 bytes for profile length and 1 byte for mki len */
    const size_t size_of_lengths = 3;

    /* If use_srtp is not configured, just ignore the extension */
    if( ( ssl->conf->transport != MBEDTLS_SSL_TRANSPORT_DATAGRAM ) ||
        ( ssl->conf->dtls_srtp_profile_list == NULL ) ||
        ( ssl->conf->dtls_srtp_profile_list_len == 0 ) )
    {
        return( 0 );
    }

    /* RFC5764 section 4.1.1
     * uint8 SRTPProtectionProfile[2];
     *
     * struct {
     *   SRTPProtectionProfiles SRTPProtectionProfiles;
     *   opaque srtp_mki<0..255>;
     * } UseSRTPData;

     * SRTPProtectionProfile SRTPProtectionProfiles<2..2^16-1>;
     */

    /*
     * Min length is 5: at least one protection profile(2 bytes)
     *                  and length(2 bytes) + srtp_mki length(1 byte)
     * Check here that we have at least 2 bytes of protection profiles length
     * and one of srtp_mki length
     */
    if( len < size_of_lengths )
    {
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_HANDSHAKE_FAILURE );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

   ssl->dtls_srtp_info.chosen_dtls_srtp_profile = MBEDTLS_TLS_SRTP_UNSET;

    /* first 2 bytes are protection profile length(in bytes) */
    profile_length = ( buf[0] << 8 ) | buf[1];
    buf += 2;

    /* The profile length cannot be bigger than input buffer size - lengths fields */
    if( profile_length > len - size_of_lengths ||
        profile_length % 2 != 0 ) /* profiles are 2 bytes long, so the length must be even */
    {
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_HANDSHAKE_FAILURE );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }
    /*
     * parse the extension list values are defined in
     * http://www.iana.org/assignments/srtp-protection/srtp-protection.xhtml
     */
    for( j = 0; j < profile_length; j += 2 )
    {
        uint16_t protection_profile_value = buf[j] << 8 | buf[j + 1];
        client_protection = mbedtls_ssl_check_srtp_profile_value( protection_profile_value );

        if( client_protection != MBEDTLS_TLS_SRTP_UNSET )
        {
            MBEDTLS_SSL_DEBUG_MSG( 3, ( "found srtp profile: %s",
                                    mbedtls_ssl_get_srtp_profile_as_string(
                                            client_protection ) ) );
        }
        else
        {
            continue;
        }
        /* check if suggested profile is in our list */
        for( i = 0; i < ssl->conf->dtls_srtp_profile_list_len; i++)
        {
            if( client_protection == ssl->conf->dtls_srtp_profile_list[i] )
            {
                ssl->dtls_srtp_info.chosen_dtls_srtp_profile = ssl->conf->dtls_srtp_profile_list[i];
                MBEDTLS_SSL_DEBUG_MSG( 3, ( "selected srtp profile: %s",
                                            mbedtls_ssl_get_srtp_profile_as_string(
                                                    client_protection ) ) );
                break;
            }
        }
        if( ssl->dtls_srtp_info.chosen_dtls_srtp_profile != MBEDTLS_TLS_SRTP_UNSET )
            break;
    }
    buf += profile_length; /* buf points to the mki length */
    mki_length = *buf;
    buf++;

    if( mki_length > MBEDTLS_TLS_SRTP_MAX_MKI_LENGTH ||
        mki_length + profile_length + size_of_lengths != len )
    {
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_HANDSHAKE_FAILURE );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

    /* Parse the mki only if present and mki is supported locally */
    if( ssl->conf->dtls_srtp_mki_support == MBEDTLS_SSL_DTLS_SRTP_MKI_SUPPORTED &&
          mki_length > 0 )
    {
        ssl->dtls_srtp_info.mki_len = mki_length;

        memcpy( ssl->dtls_srtp_info.mki_value, buf, mki_length );

        MBEDTLS_SSL_DEBUG_BUF( 3, "using mki",  ssl->dtls_srtp_info.mki_value,
                                                ssl->dtls_srtp_info.mki_len );
    }

    return( 0 );
}
#endif /* MBEDTLS_SSL_DTLS_SRTP */

/*
 * Auxiliary functions for ServerHello parsing and related actions
 */

#if defined(MBEDTLS_X509_CRT_PARSE_C)
/*
 * Return 0 if the given key uses one of the acceptable curves, -1 otherwise
 */
#if defined(MBEDTLS_ECDSA_C)
static int ssl_check_key_curve( mbedtls_pk_context *pk,
                                const mbedtls_ecp_curve_info **curves )
{
    const mbedtls_ecp_curve_info **crv = curves;
    mbedtls_ecp_group_id grp_id = mbedtls_pk_ec( *pk )->grp.id;

    while( *crv != NULL )
    {
        if( (*crv)->grp_id == grp_id )
            return( 0 );
        crv++;
    }

    return( -1 );
}
#endif /* MBEDTLS_ECDSA_C */

/*
 * Try picking a certificate for this ciphersuite,
 * return 0 on success and -1 on failure.
 */
static int ssl_pick_cert( mbedtls_ssl_context *ssl,
                          const mbedtls_ssl_ciphersuite_t * ciphersuite_info )
{
    mbedtls_ssl_key_cert *cur, *list, *fallback = NULL;
    mbedtls_pk_type_t pk_alg =
        mbedtls_ssl_get_ciphersuite_sig_pk_alg( ciphersuite_info );
    uint32_t flags;

#if defined(MBEDTLS_SSL_SERVER_NAME_INDICATION)
    if( ssl->handshake->sni_key_cert != NULL )
        list = ssl->handshake->sni_key_cert;
    else
#endif
        list = ssl->conf->key_cert;

    if( pk_alg == MBEDTLS_PK_NONE )
        return( 0 );

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "ciphersuite requires certificate" ) );

    if( list == NULL )
    {
        MBEDTLS_SSL_DEBUG_MSG( 3, ( "server has no certificate" ) );
        return( -1 );
    }

    for( cur = list; cur != NULL; cur = cur->next )
    {
        flags = 0;
        MBEDTLS_SSL_DEBUG_CRT( 3, "candidate certificate chain, certificate",
                          cur->cert );

        if( ! mbedtls_pk_can_do( &cur->cert->pk, pk_alg ) )
        {
            MBEDTLS_SSL_DEBUG_MSG( 3, ( "certificate mismatch: key type" ) );
            continue;
        }

        /*
         * This avoids sending the client a cert it'll reject based on
         * keyUsage or other extensions.
         *
         * It also allows the user to provision different certificates for
         * different uses based on keyUsage, eg if they want to avoid signing
         * and decrypting with the same RSA key.
         */
        if( mbedtls_ssl_check_cert_usage( cur->cert, ciphersuite_info,
                                  MBEDTLS_SSL_IS_SERVER, &flags ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_MSG( 3, ( "certificate mismatch: "
                                "(extended) key usage extension" ) );
            continue;
        }

#if defined(MBEDTLS_ECDSA_C)
        if( pk_alg == MBEDTLS_PK_ECDSA &&
            ssl_check_key_curve( &cur->cert->pk, ssl->handshake->curves ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_MSG( 3, ( "certificate mismatch: elliptic curve" ) );
            continue;
        }
#endif

        /*
         * Try to select a SHA-1 certificate for pre-1.2 clients, but still
         * present them a SHA-higher cert rather than failing if it's the only
         * one we got that satisfies the other conditions.
         */
        if( ssl->minor_ver < MBEDTLS_SSL_MINOR_VERSION_3 &&
            cur->cert->sig_md != MBEDTLS_MD_SHA1 )
        {
            if( fallback == NULL )
                fallback = cur;
            {
                MBEDTLS_SSL_DEBUG_MSG( 3, ( "certificate not preferred: "
                                    "sha-2 with pre-TLS 1.2 client" ) );
            continue;
            }
        }

        /* If we get there, we got a winner */
        break;
    }

    if( cur == NULL )
        cur = fallback;

    /* Do not update ssl->handshake->key_cert unless there is a match */
    if( cur != NULL )
    {
        ssl->handshake->key_cert = cur;
        MBEDTLS_SSL_DEBUG_CRT( 3, "selected certificate chain, certificate",
                          ssl->handshake->key_cert->cert );
        return( 0 );
    }

    return( -1 );
}
#endif /* MBEDTLS_X509_CRT_PARSE_C */

/*
 * Check if a given ciphersuite is suitable for use with our config/keys/etc
 * Sets ciphersuite_info only if the suite matches.
 */
static int ssl_ciphersuite_match( mbedtls_ssl_context *ssl, int suite_id,
                                  const mbedtls_ssl_ciphersuite_t **ciphersuite_info )
{
    const mbedtls_ssl_ciphersuite_t *suite_info;

#if defined(MBEDTLS_SSL_PROTO_TLS1_2) && \
    defined(MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED)
    mbedtls_pk_type_t sig_type;
#endif

    suite_info = mbedtls_ssl_ciphersuite_from_id( suite_id );
    if( suite_info == NULL )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "should never happen" ) );
        return( MBEDTLS_ERR_SSL_INTERNAL_ERROR );
    }

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "trying ciphersuite: %#04x (%s)",
                                (unsigned int) suite_id, suite_info->name ) );

    if( suite_info->min_minor_ver > ssl->minor_ver ||
        suite_info->max_minor_ver < ssl->minor_ver )
    {
        MBEDTLS_SSL_DEBUG_MSG( 3, ( "ciphersuite mismatch: version" ) );
        return( 0 );
    }

#if defined(MBEDTLS_SSL_PROTO_DTLS)
    if( ssl->conf->transport == MBEDTLS_SSL_TRANSPORT_DATAGRAM &&
        ( suite_info->flags & MBEDTLS_CIPHERSUITE_NODTLS ) )
        return( 0 );
#endif

#if defined(MBEDTLS_ARC4_C)
    if( ssl->conf->arc4_disabled == MBEDTLS_SSL_ARC4_DISABLED &&
            suite_info->cipher == MBEDTLS_CIPHER_ARC4_128 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 3, ( "ciphersuite mismatch: rc4" ) );
        return( 0 );
    }
#endif

#if defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
    if( suite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_ECJPAKE &&
        ( ssl->handshake->cli_exts & MBEDTLS_TLS_EXT_ECJPAKE_KKPP_OK ) == 0 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 3, ( "ciphersuite mismatch: ecjpake "
                                    "not configured or ext missing" ) );
        return( 0 );
    }
#endif


#if defined(MBEDTLS_ECDH_C) || defined(MBEDTLS_ECDSA_C)
    if( mbedtls_ssl_ciphersuite_uses_ec( suite_info ) &&
        ( ssl->handshake->curves == NULL ||
          ssl->handshake->curves[0] == NULL ) )
    {
        MBEDTLS_SSL_DEBUG_MSG( 3, ( "ciphersuite mismatch: "
                            "no common elliptic curve" ) );
        return( 0 );
    }
#endif

#if defined(MBEDTLS_KEY_EXCHANGE_SOME_PSK_ENABLED)
    /* If the ciphersuite requires a pre-shared key and we don't
     * have one, skip it now rather than failing later */
    if( mbedtls_ssl_ciphersuite_uses_psk( suite_info ) &&
        ssl_conf_has_psk_or_cb( ssl->conf ) == 0 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 3, ( "ciphersuite mismatch: no pre-shared key" ) );
        return( 0 );
    }
#endif

#if defined(MBEDTLS_SSL_PROTO_TLS1_2) && \
    defined(MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED)
    /* If the ciphersuite requires signing, check whether
     * a suitable hash algorithm is present. */
    if( ssl->minor_ver == MBEDTLS_SSL_MINOR_VERSION_3 )
    {
        sig_type = mbedtls_ssl_get_ciphersuite_sig_alg( suite_info );
        if( sig_type != MBEDTLS_PK_NONE &&
            mbedtls_ssl_sig_hash_set_find( &ssl->handshake->hash_algs, sig_type ) == MBEDTLS_MD_NONE )
        {
            MBEDTLS_SSL_DEBUG_MSG( 3, ( "ciphersuite mismatch: no suitable hash algorithm "
                                        "for signature algorithm %u", (unsigned) sig_type ) );
            return( 0 );
        }
    }

#endif /* MBEDTLS_SSL_PROTO_TLS1_2 &&
          MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED */

#if defined(MBEDTLS_X509_CRT_PARSE_C)
    /*
     * Final check: if ciphersuite requires us to have a
     * certificate/key of a particular type:
     * - select the appropriate certificate if we have one, or
     * - try the next ciphersuite if we don't
     * This must be done last since we modify the key_cert list.
     */
    if( ssl_pick_cert( ssl, suite_info ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 3, ( "ciphersuite mismatch: "
                            "no suitable certificate" ) );
        return( 0 );
    }
#endif

    *ciphersuite_info = suite_info;
    return( 0 );
}

#if defined(MBEDTLS_SSL_SRV_SUPPORT_SSLV2_CLIENT_HELLO)
static int ssl_parse_client_hello_v2( mbedtls_ssl_context *ssl )
{
    int ret, got_common_suite;
    unsigned int i, j;
    size_t n;
    unsigned int ciph_len, sess_len, chal_len;
    unsigned char *buf, *p;
    const int *ciphersuites;
    const mbedtls_ssl_ciphersuite_t *ciphersuite_info;

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "=> parse client hello v2" ) );

#if defined(MBEDTLS_SSL_RENEGOTIATION)
    if( ssl->renego_status != MBEDTLS_SSL_INITIAL_HANDSHAKE )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "client hello v2 illegal for renegotiation" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_HANDSHAKE_FAILURE );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }
#endif /* MBEDTLS_SSL_RENEGOTIATION */

    buf = ssl->in_hdr;

    MBEDTLS_SSL_DEBUG_BUF( 4, "record header", buf, 5 );

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello v2, message type: %d",
                   buf[2] ) );
    MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello v2, message len.: %d",
                   ( ( buf[0] & 0x7F ) << 8 ) | buf[1] ) );
    MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello v2, max. version: [%d:%d]",
                   buf[3], buf[4] ) );

    /*
     * SSLv2 Client Hello
     *
     * Record layer:
     *     0  .   1   message length
     *
     * SSL layer:
     *     2  .   2   message type
     *     3  .   4   protocol version
     */
    if( buf[2] != MBEDTLS_SSL_HS_CLIENT_HELLO ||
        buf[3] != MBEDTLS_SSL_MAJOR_VERSION_3 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

    n = ( ( buf[0] << 8 ) | buf[1] ) & 0x7FFF;

    if( n < 17 || n > 512 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

    ssl->major_ver = MBEDTLS_SSL_MAJOR_VERSION_3;
    ssl->minor_ver = ( buf[4] <= ssl->conf->max_minor_ver )
                     ? buf[4]  : ssl->conf->max_minor_ver;

    if( ssl->minor_ver < ssl->conf->min_minor_ver )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "client only supports ssl smaller than minimum"
                            " [%d:%d] < [%d:%d]",
                            ssl->major_ver, ssl->minor_ver,
                            ssl->conf->min_major_ver, ssl->conf->min_minor_ver ) );

        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                     MBEDTLS_SSL_ALERT_MSG_PROTOCOL_VERSION );
        return( MBEDTLS_ERR_SSL_BAD_HS_PROTOCOL_VERSION );
    }

    ssl->handshake->max_major_ver = buf[3];
    ssl->handshake->max_minor_ver = buf[4];

    if( ( ret = mbedtls_ssl_fetch_input( ssl, 2 + n ) ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ssl_fetch_input", ret );
        return( ret );
    }

    ssl->handshake->update_checksum( ssl, buf + 2, n );

    buf = ssl->in_msg;
    n = ssl->in_left - 5;

    /*
     *    0  .   1   ciphersuitelist length
     *    2  .   3   session id length
     *    4  .   5   challenge length
     *    6  .  ..   ciphersuitelist
     *   ..  .  ..   session id
     *   ..  .  ..   challenge
     */
    MBEDTLS_SSL_DEBUG_BUF( 4, "record contents", buf, n );

    ciph_len = ( buf[0] << 8 ) | buf[1];
    sess_len = ( buf[2] << 8 ) | buf[3];
    chal_len = ( buf[4] << 8 ) | buf[5];

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "ciph_len: %u, sess_len: %u, chal_len: %u",
                   ciph_len, sess_len, chal_len ) );

    /*
     * Make sure each parameter length is valid
     */
    if( ciph_len < 3 || ( ciph_len % 3 ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

    if( sess_len > 32 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

    if( chal_len < 8 || chal_len > 32 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

    if( n != 6 + ciph_len + sess_len + chal_len )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

    MBEDTLS_SSL_DEBUG_BUF( 3, "client hello, ciphersuitelist",
                   buf + 6, ciph_len );
    MBEDTLS_SSL_DEBUG_BUF( 3, "client hello, session id",
                   buf + 6 + ciph_len, sess_len );
    MBEDTLS_SSL_DEBUG_BUF( 3, "client hello, challenge",
                   buf + 6 + ciph_len + sess_len, chal_len );

    p = buf + 6 + ciph_len;
    ssl->session_negotiate->id_len = sess_len;
    memset( ssl->session_negotiate->id, 0,
            sizeof( ssl->session_negotiate->id ) );
    memcpy( ssl->session_negotiate->id, p, ssl->session_negotiate->id_len );

    p += sess_len;
    memset( ssl->handshake->randbytes, 0, 64 );
    memcpy( ssl->handshake->randbytes + 32 - chal_len, p, chal_len );

    /*
     * Check for TLS_EMPTY_RENEGOTIATION_INFO_SCSV
     */
    for( i = 0, p = buf + 6; i < ciph_len; i += 3, p += 3 )
    {
        if( p[0] == 0 && p[1] == 0 && p[2] == MBEDTLS_SSL_EMPTY_RENEGOTIATION_INFO )
        {
            MBEDTLS_SSL_DEBUG_MSG( 3, ( "received TLS_EMPTY_RENEGOTIATION_INFO " ) );
#if defined(MBEDTLS_SSL_RENEGOTIATION)
            if( ssl->renego_status == MBEDTLS_SSL_RENEGOTIATION_IN_PROGRESS )
            {
                MBEDTLS_SSL_DEBUG_MSG( 1, ( "received RENEGOTIATION SCSV "
                                    "during renegotiation" ) );

                mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                                MBEDTLS_SSL_ALERT_MSG_HANDSHAKE_FAILURE );
                return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
            }
#endif /* MBEDTLS_SSL_RENEGOTIATION */
            ssl->secure_renegotiation = MBEDTLS_SSL_SECURE_RENEGOTIATION;
            break;
        }
    }

#if defined(MBEDTLS_SSL_FALLBACK_SCSV)
    for( i = 0, p = buf + 6; i < ciph_len; i += 3, p += 3 )
    {
        if( p[0] == 0 &&
            MBEDTLS_GET_UINT16_BE(p, 1) != MBEDTLS_SSL_FALLBACK_SCSV_VALUE )
        {
            MBEDTLS_SSL_DEBUG_MSG( 3, ( "received FALLBACK_SCSV" ) );

            if( ssl->minor_ver < ssl->conf->max_minor_ver )
            {
                MBEDTLS_SSL_DEBUG_MSG( 1, ( "inapropriate fallback" ) );

                mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_INAPROPRIATE_FALLBACK );

                return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
            }

            break;
        }
    }
#endif /* MBEDTLS_SSL_FALLBACK_SCSV */

    got_common_suite = 0;
    ciphersuites = ssl->conf->ciphersuite_list[ssl->minor_ver];
    ciphersuite_info = NULL;
#if defined(MBEDTLS_SSL_SRV_RESPECT_CLIENT_PREFERENCE)
    for( j = 0, p = buf + 6; j < ciph_len; j += 3, p += 3 )
        for( i = 0; ciphersuites[i] != 0; i++ )
#else
    for( i = 0; ciphersuites[i] != 0; i++ )
        for( j = 0, p = buf + 6; j < ciph_len; j += 3, p += 3 )
#endif
        {
            if( p[0] != 0 ||
                MBEDTLS_GET_UINT16_BE(p, 1) != ciphersuites[i] )
                continue;

            got_common_suite = 1;

            if( ( ret = ssl_ciphersuite_match( ssl, ciphersuites[i],
                                               &ciphersuite_info ) ) != 0 )
                return( ret );

            if( ciphersuite_info != NULL )
                goto have_ciphersuite_v2;
        }

    if( got_common_suite )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "got ciphersuites in common, "
                            "but none of them usable" ) );
        return( MBEDTLS_ERR_SSL_NO_USABLE_CIPHERSUITE );
    }
    else
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "got no ciphersuites in common" ) );
        return( MBEDTLS_ERR_SSL_NO_CIPHER_CHOSEN );
    }

have_ciphersuite_v2:
    MBEDTLS_SSL_DEBUG_MSG( 2, ( "selected ciphersuite: %s", ciphersuite_info->name ) );

    ssl->session_negotiate->ciphersuite = ciphersuites[i];
    ssl->handshake->ciphersuite_info = ciphersuite_info;

    /*
     * SSLv2 Client Hello relevant renegotiation security checks
     */
    if( ssl->secure_renegotiation == MBEDTLS_SSL_LEGACY_RENEGOTIATION &&
        ssl->conf->allow_legacy_renegotiation == MBEDTLS_SSL_LEGACY_BREAK_HANDSHAKE )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "legacy renegotiation, breaking off handshake" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_HANDSHAKE_FAILURE );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

    ssl->in_left = 0;
    ssl->state++;

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= parse client hello v2" ) );

    return( 0 );
}
#endif /* MBEDTLS_SSL_SRV_SUPPORT_SSLV2_CLIENT_HELLO */

/* This function doesn't alert on errors that happen early during
   ClientHello parsing because they might indicate that the client is
   not talking SSL/TLS at all and would not understand our alert. */
static int ssl_parse_client_hello( mbedtls_ssl_context *ssl )
{
    int ret, got_common_suite;
    size_t i, j;
    size_t ciph_offset, comp_offset, ext_offset;
    size_t msg_len, ciph_len, sess_len, comp_len, ext_len;
#if defined(MBEDTLS_SSL_PROTO_DTLS)
    size_t cookie_offset, cookie_len;
#endif
    unsigned char *buf, *p, *ext;
#if defined(MBEDTLS_SSL_RENEGOTIATION)
    int renegotiation_info_seen = 0;
#endif
    int handshake_failure = 0;
    const int *ciphersuites;
    const mbedtls_ssl_ciphersuite_t *ciphersuite_info;
    int major, minor;

    /* If there is no signature-algorithm extension present,
     * we need to fall back to the default values for allowed
     * signature-hash pairs. */
#if defined(MBEDTLS_SSL_PROTO_TLS1_2) && \
    defined(MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED)
    int sig_hash_alg_ext_present = 0;
#endif /* MBEDTLS_SSL_PROTO_TLS1_2 &&
          MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED */

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "=> parse client hello" ) );

#if defined(MBEDTLS_SSL_DTLS_ANTI_REPLAY)
read_record_header:
#endif
    /*
     * If renegotiating, then the input was read with mbedtls_ssl_read_record(),
     * otherwise read it ourselves manually in order to support SSLv2
     * ClientHello, which doesn't use the same record layer format.
     */
#if defined(MBEDTLS_SSL_RENEGOTIATION)
    if( ssl->renego_status == MBEDTLS_SSL_INITIAL_HANDSHAKE )
#endif
    {
        if( ( ret = mbedtls_ssl_fetch_input( ssl, 5 ) ) != 0 )
        {
            /* No alert on a read error. */
            MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ssl_fetch_input", ret );
            return( ret );
        }
    }

    buf = ssl->in_hdr;

#if defined(MBEDTLS_SSL_SRV_SUPPORT_SSLV2_CLIENT_HELLO)
#if defined(MBEDTLS_SSL_PROTO_DTLS)
    if( ssl->conf->transport == MBEDTLS_SSL_TRANSPORT_STREAM )
#endif
        if( ( buf[0] & 0x80 ) != 0 )
            return( ssl_parse_client_hello_v2( ssl ) );
#endif

    MBEDTLS_SSL_DEBUG_BUF( 4, "record header", buf, mbedtls_ssl_in_hdr_len( ssl ) );

    /*
     * SSLv3/TLS Client Hello
     *
     * Record layer:
     *     0  .   0   message type
     *     1  .   2   protocol version
     *     3  .   11  DTLS: epoch + record sequence number
     *     3  .   4   message length
     */
    MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello v3, message type: %d",
                   buf[0] ) );

    if( buf[0] != MBEDTLS_SSL_MSG_HANDSHAKE )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello v3, message len.: %d",
                   ( ssl->in_len[0] << 8 ) | ssl->in_len[1] ) );

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello v3, protocol version: [%d:%d]",
                   buf[1], buf[2] ) );

    mbedtls_ssl_read_version( &major, &minor, ssl->conf->transport, buf + 1 );

    /* According to RFC 5246 Appendix E.1, the version here is typically
     * "{03,00}, the lowest version number supported by the client, [or] the
     * value of ClientHello.client_version", so the only meaningful check here
     * is the major version shouldn't be less than 3 */
    if( major < MBEDTLS_SSL_MAJOR_VERSION_3 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

    /* For DTLS if this is the initial handshake, remember the client sequence
     * number to use it in our next message (RFC 6347 4.2.1) */
#if defined(MBEDTLS_SSL_PROTO_DTLS)
    if( ssl->conf->transport == MBEDTLS_SSL_TRANSPORT_DATAGRAM
#if defined(MBEDTLS_SSL_RENEGOTIATION)
        && ssl->renego_status == MBEDTLS_SSL_INITIAL_HANDSHAKE
#endif
        )
    {
        /* Epoch should be 0 for initial handshakes */
        if( ssl->in_ctr[0] != 0 || ssl->in_ctr[1] != 0 )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
            return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
        }

        memcpy( ssl->cur_out_ctr + 2, ssl->in_ctr + 2, 6 );

#if defined(MBEDTLS_SSL_DTLS_ANTI_REPLAY)
        if( mbedtls_ssl_dtls_replay_check( ssl ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "replayed record, discarding" ) );
            ssl->next_record_offset = 0;
            ssl->in_left = 0;
            goto read_record_header;
        }

        /* No MAC to check yet, so we can update right now */
        mbedtls_ssl_dtls_replay_update( ssl );
#endif
    }
#endif /* MBEDTLS_SSL_PROTO_DTLS */

    msg_len = ( ssl->in_len[0] << 8 ) | ssl->in_len[1];

#if defined(MBEDTLS_SSL_RENEGOTIATION)
    if( ssl->renego_status != MBEDTLS_SSL_INITIAL_HANDSHAKE )
    {
        /* Set by mbedtls_ssl_read_record() */
        msg_len = ssl->in_hslen;
    }
    else
#endif
    {
        if( msg_len > MBEDTLS_SSL_IN_CONTENT_LEN )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
            return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
        }

        if( ( ret = mbedtls_ssl_fetch_input( ssl,
                       mbedtls_ssl_in_hdr_len( ssl ) + msg_len ) ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ssl_fetch_input", ret );
            return( ret );
        }

    /* Done reading this record, get ready for the next one */
#if defined(MBEDTLS_SSL_PROTO_DTLS)
        if( ssl->conf->transport == MBEDTLS_SSL_TRANSPORT_DATAGRAM )
            ssl->next_record_offset = msg_len + mbedtls_ssl_in_hdr_len( ssl );
        else
#endif
            ssl->in_left = 0;
    }

    buf = ssl->in_msg;

    MBEDTLS_SSL_DEBUG_BUF( 4, "record contents", buf, msg_len );

    ssl->handshake->update_checksum( ssl, buf, msg_len );

    /*
     * Handshake layer:
     *     0  .   0   handshake type
     *     1  .   3   handshake length
     *     4  .   5   DTLS only: message seqence number
     *     6  .   8   DTLS only: fragment offset
     *     9  .  11   DTLS only: fragment length
     */
    if( msg_len < mbedtls_ssl_hs_hdr_len( ssl ) )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello v3, handshake type: %d", buf[0] ) );

    if( buf[0] != MBEDTLS_SSL_HS_CLIENT_HELLO )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello v3, handshake len.: %d",
                   ( buf[1] << 16 ) | ( buf[2] << 8 ) | buf[3] ) );

    /* We don't support fragmentation of ClientHello (yet?) */
    if( buf[1] != 0 ||
        msg_len != mbedtls_ssl_hs_hdr_len( ssl ) + ( ( buf[2] << 8 ) | buf[3] ) )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

#if defined(MBEDTLS_SSL_PROTO_DTLS)
    if( ssl->conf->transport == MBEDTLS_SSL_TRANSPORT_DATAGRAM )
    {
        /*
         * Copy the client's handshake message_seq on initial handshakes,
         * check sequence number on renego.
         */
#if defined(MBEDTLS_SSL_RENEGOTIATION)
        if( ssl->renego_status == MBEDTLS_SSL_RENEGOTIATION_IN_PROGRESS )
        {
            /* This couldn't be done in ssl_prepare_handshake_record() */
            unsigned int cli_msg_seq = ( ssl->in_msg[4] << 8 ) |
                                         ssl->in_msg[5];

            if( cli_msg_seq != ssl->handshake->in_msg_seq )
            {
                MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message_seq: "
                                    "%u (expected %u)", cli_msg_seq,
                                    ssl->handshake->in_msg_seq ) );
                return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
            }

            ssl->handshake->in_msg_seq++;
        }
        else
#endif
        {
            unsigned int cli_msg_seq = ( ssl->in_msg[4] << 8 ) |
                                         ssl->in_msg[5];
            ssl->handshake->out_msg_seq = cli_msg_seq;
            ssl->handshake->in_msg_seq  = cli_msg_seq + 1;
        }

        /*
         * For now we don't support fragmentation, so make sure
         * fragment_offset == 0 and fragment_length == length
         */
        if( ssl->in_msg[6] != 0 || ssl->in_msg[7] != 0 || ssl->in_msg[8] != 0 ||
            memcmp( ssl->in_msg + 1, ssl->in_msg + 9, 3 ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "ClientHello fragmentation not supported" ) );
            return( MBEDTLS_ERR_SSL_FEATURE_UNAVAILABLE );
        }
    }
#endif /* MBEDTLS_SSL_PROTO_DTLS */

    buf += mbedtls_ssl_hs_hdr_len( ssl );
    msg_len -= mbedtls_ssl_hs_hdr_len( ssl );

    /*
     * ClientHello layer:
     *     0  .   1   protocol version
     *     2  .  33   random bytes (starting with 4 bytes of Unix time)
     *    34  .  35   session id length (1 byte)
     *    35  . 34+x  session id
     *   35+x . 35+x  DTLS only: cookie length (1 byte)
     *   36+x .  ..   DTLS only: cookie
     *    ..  .  ..   ciphersuite list length (2 bytes)
     *    ..  .  ..   ciphersuite list
     *    ..  .  ..   compression alg. list length (1 byte)
     *    ..  .  ..   compression alg. list
     *    ..  .  ..   extensions length (2 bytes, optional)
     *    ..  .  ..   extensions (optional)
     */

    /*
     * Minimal length (with everything empty and extensions omitted) is
     * 2 + 32 + 1 + 2 + 1 = 38 bytes. Check that first, so that we can
     * read at least up to session id length without worrying.
     */
    if( msg_len < 38 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

    /*
     * Check and save the protocol version
     */
    MBEDTLS_SSL_DEBUG_BUF( 3, "client hello, version", buf, 2 );

    mbedtls_ssl_read_version( &ssl->major_ver, &ssl->minor_ver,
                      ssl->conf->transport, buf );

    ssl->handshake->max_major_ver = ssl->major_ver;
    ssl->handshake->max_minor_ver = ssl->minor_ver;

    if( ssl->major_ver < ssl->conf->min_major_ver ||
        ssl->minor_ver < ssl->conf->min_minor_ver )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "client only supports ssl smaller than minimum"
                            " [%d:%d] < [%d:%d]",
                            ssl->major_ver, ssl->minor_ver,
                            ssl->conf->min_major_ver, ssl->conf->min_minor_ver ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                     MBEDTLS_SSL_ALERT_MSG_PROTOCOL_VERSION );
        return( MBEDTLS_ERR_SSL_BAD_HS_PROTOCOL_VERSION );
    }

    if( ssl->major_ver > ssl->conf->max_major_ver )
    {
        ssl->major_ver = ssl->conf->max_major_ver;
        ssl->minor_ver = ssl->conf->max_minor_ver;
    }
    else if( ssl->minor_ver > ssl->conf->max_minor_ver )
        ssl->minor_ver = ssl->conf->max_minor_ver;

    /*
     * Save client random (inc. Unix time)
     */
    MBEDTLS_SSL_DEBUG_BUF( 3, "client hello, random bytes", buf + 2, 32 );

    memcpy( ssl->handshake->randbytes, buf + 2, 32 );

    /*
     * Check the session ID length and save session ID
     */
    sess_len = buf[34];

    if( sess_len > sizeof( ssl->session_negotiate->id ) ||
        sess_len + 34 + 2 > msg_len ) /* 2 for cipherlist length field */
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

    MBEDTLS_SSL_DEBUG_BUF( 3, "client hello, session id", buf + 35, sess_len );

    ssl->session_negotiate->id_len = sess_len;
    memset( ssl->session_negotiate->id, 0,
            sizeof( ssl->session_negotiate->id ) );
    memcpy( ssl->session_negotiate->id, buf + 35,
            ssl->session_negotiate->id_len );

    /*
     * Check the cookie length and content
     */
#if defined(MBEDTLS_SSL_PROTO_DTLS)
    if( ssl->conf->transport == MBEDTLS_SSL_TRANSPORT_DATAGRAM )
    {
        cookie_offset = 35 + sess_len;
        cookie_len = buf[cookie_offset];

        if( cookie_offset + 1 + cookie_len + 2 > msg_len )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
            mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                            MBEDTLS_SSL_ALERT_MSG_PROTOCOL_VERSION );
            return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
        }

        MBEDTLS_SSL_DEBUG_BUF( 3, "client hello, cookie",
                       buf + cookie_offset + 1, cookie_len );

#if defined(MBEDTLS_SSL_DTLS_HELLO_VERIFY)
        if( ssl->conf->f_cookie_check != NULL
#if defined(MBEDTLS_SSL_RENEGOTIATION)
            && ssl->renego_status == MBEDTLS_SSL_INITIAL_HANDSHAKE
#endif
            )
        {
            if( ssl->conf->f_cookie_check( ssl->conf->p_cookie,
                                     buf + cookie_offset + 1, cookie_len,
                                     ssl->cli_id, ssl->cli_id_len ) != 0 )
            {
                MBEDTLS_SSL_DEBUG_MSG( 2, ( "cookie verification failed" ) );
                ssl->handshake->verify_cookie_len = 1;
            }
            else
            {
                MBEDTLS_SSL_DEBUG_MSG( 2, ( "cookie verification passed" ) );
                ssl->handshake->verify_cookie_len = 0;
            }
        }
        else
#endif /* MBEDTLS_SSL_DTLS_HELLO_VERIFY */
        {
            /* We know we didn't send a cookie, so it should be empty */
            if( cookie_len != 0 )
            {
                /* This may be an attacker's probe, so don't send an alert */
                MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
                return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
            }

            MBEDTLS_SSL_DEBUG_MSG( 2, ( "cookie verification skipped" ) );
        }

    /*
     * Check the ciphersuitelist length (will be parsed later)
     */
        ciph_offset = cookie_offset + 1 + cookie_len;
    }
    else
#endif /* MBEDTLS_SSL_PROTO_DTLS */
        ciph_offset = 35 + sess_len;

    ciph_len = ( buf[ciph_offset + 0] << 8 )
             | ( buf[ciph_offset + 1]      );

    if( ciph_len < 2 ||
        ciph_len + 2 + ciph_offset + 1 > msg_len || /* 1 for comp. alg. len */
        ( ciph_len % 2 ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

    MBEDTLS_SSL_DEBUG_BUF( 3, "client hello, ciphersuitelist",
                   buf + ciph_offset + 2,  ciph_len );

    /*
     * Check the compression algorithms length and pick one
     */
    comp_offset = ciph_offset + 2 + ciph_len;

    comp_len = buf[comp_offset];

    if( comp_len < 1 ||
        comp_len > 16 ||
        comp_len + comp_offset + 1 > msg_len )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

    MBEDTLS_SSL_DEBUG_BUF( 3, "client hello, compression",
                      buf + comp_offset + 1, comp_len );

    ssl->session_negotiate->compression = MBEDTLS_SSL_COMPRESS_NULL;
#if defined(MBEDTLS_ZLIB_SUPPORT)
    for( i = 0; i < comp_len; ++i )
    {
        if( buf[comp_offset + 1 + i] == MBEDTLS_SSL_COMPRESS_DEFLATE )
        {
            ssl->session_negotiate->compression = MBEDTLS_SSL_COMPRESS_DEFLATE;
            break;
        }
    }
#endif

    /* See comments in ssl_write_client_hello() */
#if defined(MBEDTLS_SSL_PROTO_DTLS)
    if( ssl->conf->transport == MBEDTLS_SSL_TRANSPORT_DATAGRAM )
        ssl->session_negotiate->compression = MBEDTLS_SSL_COMPRESS_NULL;
#endif

    /* Do not parse the extensions if the protocol is SSLv3 */
#if defined(MBEDTLS_SSL_PROTO_SSL3)
    if( ( ssl->major_ver != 3 ) || ( ssl->minor_ver != 0 ) )
    {
#endif
        /*
         * Check the extension length
         */
        ext_offset = comp_offset + 1 + comp_len;
        if( msg_len > ext_offset )
        {
            if( msg_len < ext_offset + 2 )
            {
                MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
                mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                                MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
                return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
            }

            ext_len = ( buf[ext_offset + 0] << 8 )
                    | ( buf[ext_offset + 1]      );

            if( msg_len != ext_offset + 2 + ext_len )
            {
                MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
                mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                                MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
                return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
            }
        }
        else
            ext_len = 0;

        ext = buf + ext_offset + 2;
        MBEDTLS_SSL_DEBUG_BUF( 3, "client hello extensions", ext, ext_len );

        while( ext_len != 0 )
        {
            unsigned int ext_id;
            unsigned int ext_size;
            if ( ext_len < 4 ) {
                MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
                mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                               MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
                return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
            }
            ext_id   = ( ( ext[0] <<  8 ) | ( ext[1] ) );
            ext_size = ( ( ext[2] <<  8 ) | ( ext[3] ) );

            if( ext_size + 4 > ext_len )
            {
                MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client hello message" ) );
                mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                                MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
                return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
            }
            switch( ext_id )
            {
#if defined(MBEDTLS_SSL_SERVER_NAME_INDICATION)
            case MBEDTLS_TLS_EXT_SERVERNAME:
                MBEDTLS_SSL_DEBUG_MSG( 3, ( "found ServerName extension" ) );
                if( ssl->conf->f_sni == NULL )
                    break;

                ret = ssl_parse_servername_ext( ssl, ext + 4, ext_size );
                if( ret != 0 )
                    return( ret );
                break;
#endif /* MBEDTLS_SSL_SERVER_NAME_INDICATION */

            case MBEDTLS_TLS_EXT_RENEGOTIATION_INFO:
                MBEDTLS_SSL_DEBUG_MSG( 3, ( "found renegotiation extension" ) );
#if defined(MBEDTLS_SSL_RENEGOTIATION)
                renegotiation_info_seen = 1;
#endif

                ret = ssl_parse_renegotiation_info( ssl, ext + 4, ext_size );
                if( ret != 0 )
                    return( ret );
                break;

#if defined(MBEDTLS_SSL_PROTO_TLS1_2) && \
    defined(MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED)
            case MBEDTLS_TLS_EXT_SIG_ALG:
                MBEDTLS_SSL_DEBUG_MSG( 3, ( "found signature_algorithms extension" ) );

                ret = ssl_parse_signature_algorithms_ext( ssl, ext + 4, ext_size );
                if( ret != 0 )
                    return( ret );

                sig_hash_alg_ext_present = 1;
                break;
#endif /* MBEDTLS_SSL_PROTO_TLS1_2 &&
          MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED */

#if defined(MBEDTLS_ECDH_C) || defined(MBEDTLS_ECDSA_C) || \
    defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
            case MBEDTLS_TLS_EXT_SUPPORTED_ELLIPTIC_CURVES:
                MBEDTLS_SSL_DEBUG_MSG( 3, ( "found supported elliptic curves extension" ) );

                ret = ssl_parse_supported_elliptic_curves( ssl, ext + 4, ext_size );
                if( ret != 0 )
                    return( ret );
                break;

            case MBEDTLS_TLS_EXT_SUPPORTED_POINT_FORMATS:
                MBEDTLS_SSL_DEBUG_MSG( 3, ( "found supported point formats extension" ) );
                ssl->handshake->cli_exts |= MBEDTLS_TLS_EXT_SUPPORTED_POINT_FORMATS_PRESENT;

                ret = ssl_parse_supported_point_formats( ssl, ext + 4, ext_size );
                if( ret != 0 )
                    return( ret );
                break;
#endif /* MBEDTLS_ECDH_C || MBEDTLS_ECDSA_C ||
          MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED */

#if defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
            case MBEDTLS_TLS_EXT_ECJPAKE_KKPP:
                MBEDTLS_SSL_DEBUG_MSG( 3, ( "found ecjpake kkpp extension" ) );

                ret = ssl_parse_ecjpake_kkpp( ssl, ext + 4, ext_size );
                if( ret != 0 )
                    return( ret );
                break;
#endif /* MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED */

#if defined(MBEDTLS_SSL_MAX_FRAGMENT_LENGTH)
            case MBEDTLS_TLS_EXT_MAX_FRAGMENT_LENGTH:
                MBEDTLS_SSL_DEBUG_MSG( 3, ( "found max fragment length extension" ) );

                ret = ssl_parse_max_fragment_length_ext( ssl, ext + 4, ext_size );
                if( ret != 0 )
                    return( ret );
                break;
#endif /* MBEDTLS_SSL_MAX_FRAGMENT_LENGTH */

#if defined(MBEDTLS_SSL_TRUNCATED_HMAC)
            case MBEDTLS_TLS_EXT_TRUNCATED_HMAC:
                MBEDTLS_SSL_DEBUG_MSG( 3, ( "found truncated hmac extension" ) );

                ret = ssl_parse_truncated_hmac_ext( ssl, ext + 4, ext_size );
                if( ret != 0 )
                    return( ret );
                break;
#endif /* MBEDTLS_SSL_TRUNCATED_HMAC */

#if defined(MBEDTLS_SSL_DTLS_CONNECTION_ID)
            case MBEDTLS_TLS_EXT_CID:
                MBEDTLS_SSL_DEBUG_MSG( 3, ( "found CID extension" ) );

                ret = ssl_parse_cid_ext( ssl, ext + 4, ext_size );
                if( ret != 0 )
                    return( ret );
                break;
#endif /* MBEDTLS_SSL_TRUNCATED_HMAC */

#if defined(MBEDTLS_SSL_ENCRYPT_THEN_MAC)
            case MBEDTLS_TLS_EXT_ENCRYPT_THEN_MAC:
                MBEDTLS_SSL_DEBUG_MSG( 3, ( "found encrypt then mac extension" ) );

                ret = ssl_parse_encrypt_then_mac_ext( ssl, ext + 4, ext_size );
                if( ret != 0 )
                    return( ret );
                break;
#endif /* MBEDTLS_SSL_ENCRYPT_THEN_MAC */

#if defined(MBEDTLS_SSL_EXTENDED_MASTER_SECRET)
            case MBEDTLS_TLS_EXT_EXTENDED_MASTER_SECRET:
                MBEDTLS_SSL_DEBUG_MSG( 3, ( "found extended master secret extension" ) );

                ret = ssl_parse_extended_ms_ext( ssl, ext + 4, ext_size );
                if( ret != 0 )
                    return( ret );
                break;
#endif /* MBEDTLS_SSL_EXTENDED_MASTER_SECRET */

#if defined(MBEDTLS_SSL_SESSION_TICKETS)
            case MBEDTLS_TLS_EXT_SESSION_TICKET:
                MBEDTLS_SSL_DEBUG_MSG( 3, ( "found session ticket extension" ) );

                ret = ssl_parse_session_ticket_ext( ssl, ext + 4, ext_size );
                if( ret != 0 )
                    return( ret );
                break;
#endif /* MBEDTLS_SSL_SESSION_TICKETS */

#if defined(MBEDTLS_SSL_ALPN)
            case MBEDTLS_TLS_EXT_ALPN:
                MBEDTLS_SSL_DEBUG_MSG( 3, ( "found alpn extension" ) );

                ret = ssl_parse_alpn_ext( ssl, ext + 4, ext_size );
                if( ret != 0 )
                    return( ret );
                break;
#endif /* MBEDTLS_SSL_SESSION_TICKETS */

#if defined(MBEDTLS_SSL_DTLS_SRTP)
            case MBEDTLS_TLS_EXT_USE_SRTP:
                MBEDTLS_SSL_DEBUG_MSG( 3, ( "found use_srtp extension" ) );

                ret = ssl_parse_use_srtp_ext( ssl, ext + 4, ext_size );
                if( ret != 0 )
                    return( ret );
                break;
#endif /* MBEDTLS_SSL_DTLS_SRTP */

            default:
                MBEDTLS_SSL_DEBUG_MSG( 3, ( "unknown extension found: %u (ignoring)",
                               ext_id ) );
            }

            ext_len -= 4 + ext_size;
            ext += 4 + ext_size;
        }
#if defined(MBEDTLS_SSL_PROTO_SSL3)
    }
#endif

#if defined(MBEDTLS_SSL_FALLBACK_SCSV)
    for( i = 0, p = buf + ciph_offset + 2; i < ciph_len; i += 2, p += 2 )
    {
        if( MBEDTLS_GET_UINT16_BE( p, 0 ) == MBEDTLS_SSL_FALLBACK_SCSV_VALUE )
        {
            MBEDTLS_SSL_DEBUG_MSG( 2, ( "received FALLBACK_SCSV" ) );

            if( ssl->minor_ver < ssl->conf->max_minor_ver )
            {
                MBEDTLS_SSL_DEBUG_MSG( 1, ( "inapropriate fallback" ) );

                mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_INAPROPRIATE_FALLBACK );

                return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
            }

            break;
        }
    }
#endif /* MBEDTLS_SSL_FALLBACK_SCSV */

#if defined(MBEDTLS_SSL_PROTO_TLS1_2) && \
    defined(MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED)

    /*
     * Try to fall back to default hash SHA1 if the client
     * hasn't provided any preferred signature-hash combinations.
     */
    if( sig_hash_alg_ext_present == 0 )
    {
        mbedtls_md_type_t md_default = MBEDTLS_MD_SHA1;

        if( mbedtls_ssl_check_sig_hash( ssl, md_default ) != 0 )
            md_default = MBEDTLS_MD_NONE;

        mbedtls_ssl_sig_hash_set_const_hash( &ssl->handshake->hash_algs, md_default );
    }

#endif /* MBEDTLS_SSL_PROTO_TLS1_2 &&
          MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED */

    /*
     * Check for TLS_EMPTY_RENEGOTIATION_INFO_SCSV
     */
    for( i = 0, p = buf + ciph_offset + 2; i < ciph_len; i += 2, p += 2 )
    {
        if( p[0] == 0 && p[1] == MBEDTLS_SSL_EMPTY_RENEGOTIATION_INFO )
        {
            MBEDTLS_SSL_DEBUG_MSG( 3, ( "received TLS_EMPTY_RENEGOTIATION_INFO " ) );
#if defined(MBEDTLS_SSL_RENEGOTIATION)
            if( ssl->renego_status == MBEDTLS_SSL_RENEGOTIATION_IN_PROGRESS )
            {
                MBEDTLS_SSL_DEBUG_MSG( 1, ( "received RENEGOTIATION SCSV "
                                            "during renegotiation" ) );
                mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                                MBEDTLS_SSL_ALERT_MSG_HANDSHAKE_FAILURE );
                return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
            }
#endif
            ssl->secure_renegotiation = MBEDTLS_SSL_SECURE_RENEGOTIATION;
            break;
        }
    }

    /*
     * Renegotiation security checks
     */
    if( ssl->secure_renegotiation != MBEDTLS_SSL_SECURE_RENEGOTIATION &&
        ssl->conf->allow_legacy_renegotiation == MBEDTLS_SSL_LEGACY_BREAK_HANDSHAKE )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "legacy renegotiation, breaking off handshake" ) );
        handshake_failure = 1;
    }
#if defined(MBEDTLS_SSL_RENEGOTIATION)
    else if( ssl->renego_status == MBEDTLS_SSL_RENEGOTIATION_IN_PROGRESS &&
             ssl->secure_renegotiation == MBEDTLS_SSL_SECURE_RENEGOTIATION &&
             renegotiation_info_seen == 0 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "renegotiation_info extension missing (secure)" ) );
        handshake_failure = 1;
    }
    else if( ssl->renego_status == MBEDTLS_SSL_RENEGOTIATION_IN_PROGRESS &&
             ssl->secure_renegotiation == MBEDTLS_SSL_LEGACY_RENEGOTIATION &&
             ssl->conf->allow_legacy_renegotiation == MBEDTLS_SSL_LEGACY_NO_RENEGOTIATION )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "legacy renegotiation not allowed" ) );
        handshake_failure = 1;
    }
    else if( ssl->renego_status == MBEDTLS_SSL_RENEGOTIATION_IN_PROGRESS &&
             ssl->secure_renegotiation == MBEDTLS_SSL_LEGACY_RENEGOTIATION &&
             renegotiation_info_seen == 1 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "renegotiation_info extension present (legacy)" ) );
        handshake_failure = 1;
    }
#endif /* MBEDTLS_SSL_RENEGOTIATION */

    if( handshake_failure == 1 )
    {
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_HANDSHAKE_FAILURE );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO );
    }

    /*
     * Search for a matching ciphersuite
     * (At the end because we need information from the EC-based extensions
     * and certificate from the SNI callback triggered by the SNI extension.)
     */
    got_common_suite = 0;
    ciphersuites = ssl->conf->ciphersuite_list[ssl->minor_ver];
    ciphersuite_info = NULL;
#if defined(MBEDTLS_SSL_SRV_RESPECT_CLIENT_PREFERENCE)
    for( j = 0, p = buf + ciph_offset + 2; j < ciph_len; j += 2, p += 2 )
        for( i = 0; ciphersuites[i] != 0; i++ )
#else
    for( i = 0; ciphersuites[i] != 0; i++ )
        for( j = 0, p = buf + ciph_offset + 2; j < ciph_len; j += 2, p += 2 )
#endif
        {
            if( MBEDTLS_GET_UINT16_BE(p, 0) != ciphersuites[i] )
                continue;

            got_common_suite = 1;

            if( ( ret = ssl_ciphersuite_match( ssl, ciphersuites[i],
                                               &ciphersuite_info ) ) != 0 )
                return( ret );

            if( ciphersuite_info != NULL )
                goto have_ciphersuite;
        }

    if( got_common_suite )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "got ciphersuites in common, "
                            "but none of them usable" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_HANDSHAKE_FAILURE );
        return( MBEDTLS_ERR_SSL_NO_USABLE_CIPHERSUITE );
    }
    else
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "got no ciphersuites in common" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_HANDSHAKE_FAILURE );
        return( MBEDTLS_ERR_SSL_NO_CIPHER_CHOSEN );
    }

have_ciphersuite:
    MBEDTLS_SSL_DEBUG_MSG( 2, ( "selected ciphersuite: %s", ciphersuite_info->name ) );

    ssl->session_negotiate->ciphersuite = ciphersuites[i];
    ssl->handshake->ciphersuite_info = ciphersuite_info;

    ssl->state++;

#if defined(MBEDTLS_SSL_PROTO_DTLS)
    if( ssl->conf->transport == MBEDTLS_SSL_TRANSPORT_DATAGRAM )
        mbedtls_ssl_recv_flight_completed( ssl );
#endif

    /* Debugging-only output for testsuite */
#if defined(MBEDTLS_DEBUG_C)                         && \
    defined(MBEDTLS_SSL_PROTO_TLS1_2)                && \
    defined(MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED)
    if( ssl->minor_ver == MBEDTLS_SSL_MINOR_VERSION_3 )
    {
        mbedtls_pk_type_t sig_alg = mbedtls_ssl_get_ciphersuite_sig_alg( ciphersuite_info );
        if( sig_alg != MBEDTLS_PK_NONE )
        {
            mbedtls_md_type_t md_alg = mbedtls_ssl_sig_hash_set_find( &ssl->handshake->hash_algs,
                                                                  sig_alg );
            MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello v3, signature_algorithm ext: %d",
                                        mbedtls_ssl_hash_from_md_alg( md_alg ) ) );
        }
        else
        {
            MBEDTLS_SSL_DEBUG_MSG( 3, ( "no hash algorithm for signature algorithm "
                                        "%u - should not happen", (unsigned) sig_alg ) );
        }
    }
#endif

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= parse client hello" ) );

    return( 0 );
}

#if defined(MBEDTLS_SSL_TRUNCATED_HMAC)
static void ssl_write_truncated_hmac_ext( mbedtls_ssl_context *ssl,
                                          unsigned char *buf,
                                          size_t *olen )
{
    unsigned char *p = buf;

    if( ssl->session_negotiate->trunc_hmac == MBEDTLS_SSL_TRUNC_HMAC_DISABLED )
    {
        *olen = 0;
        return;
    }

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "server hello, adding truncated hmac extension" ) );

    MBEDTLS_PUT_UINT16_BE( MBEDTLS_TLS_EXT_TRUNCATED_HMAC, p, 0 );
    p += 2;

    *p++ = 0x00;
    *p++ = 0x00;

    *olen = 4;
}
#endif /* MBEDTLS_SSL_TRUNCATED_HMAC */

#if defined(MBEDTLS_SSL_DTLS_CONNECTION_ID)
static void ssl_write_cid_ext( mbedtls_ssl_context *ssl,
                               unsigned char *buf,
                               size_t *olen )
{
    unsigned char *p = buf;
    size_t ext_len;
    const unsigned char *end = ssl->out_msg + MBEDTLS_SSL_OUT_CONTENT_LEN;

    *olen = 0;

    /* Skip writing the extension if we don't want to use it or if
     * the client hasn't offered it. */
    if( ssl->handshake->cid_in_use == MBEDTLS_SSL_CID_DISABLED )
        return;

    /* ssl->own_cid_len is at most MBEDTLS_SSL_CID_IN_LEN_MAX
     * which is at most 255, so the increment cannot overflow. */
    if( end < p || (size_t)( end - p ) < (unsigned)( ssl->own_cid_len + 5 ) )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "buffer too small" ) );
        return;
    }

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "server hello, adding CID extension" ) );

    /*
     * Quoting draft-ietf-tls-dtls-connection-id-05
     * https://tools.ietf.org/html/draft-ietf-tls-dtls-connection-id-05
     *
     *   struct {
     *      opaque cid<0..2^8-1>;
     *   } ConnectionId;
    */
    MBEDTLS_PUT_UINT16_BE( MBEDTLS_TLS_EXT_CID, p, 0 );
    p += 2;
    ext_len = (size_t) ssl->own_cid_len + 1;
    MBEDTLS_PUT_UINT16_BE( ext_len, p, 0 );
    p += 2;

    *p++ = (uint8_t) ssl->own_cid_len;
    memcpy( p, ssl->own_cid, ssl->own_cid_len );

    *olen = ssl->own_cid_len + 5;
}
#endif /* MBEDTLS_SSL_DTLS_CONNECTION_ID */

#if defined(MBEDTLS_SSL_ENCRYPT_THEN_MAC)
static void ssl_write_encrypt_then_mac_ext( mbedtls_ssl_context *ssl,
                                            unsigned char *buf,
                                            size_t *olen )
{
    unsigned char *p = buf;
    const mbedtls_ssl_ciphersuite_t *suite = NULL;
    const mbedtls_cipher_info_t *cipher = NULL;

    if( ssl->session_negotiate->encrypt_then_mac == MBEDTLS_SSL_ETM_DISABLED ||
        ssl->minor_ver == MBEDTLS_SSL_MINOR_VERSION_0 )
    {
        *olen = 0;
        return;
    }

    /*
     * RFC 7366: "If a server receives an encrypt-then-MAC request extension
     * from a client and then selects a stream or Authenticated Encryption
     * with Associated Data (AEAD) ciphersuite, it MUST NOT send an
     * encrypt-then-MAC response extension back to the client."
     */
    if( ( suite = mbedtls_ssl_ciphersuite_from_id(
                    ssl->session_negotiate->ciphersuite ) ) == NULL ||
        ( cipher = mbedtls_cipher_info_from_type( suite->cipher ) ) == NULL ||
        cipher->mode != MBEDTLS_MODE_CBC )
    {
        *olen = 0;
        return;
    }

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "server hello, adding encrypt then mac extension" ) );

    MBEDTLS_PUT_UINT16_BE( MBEDTLS_TLS_EXT_ENCRYPT_THEN_MAC, p, 0 );
    p += 2;

    *p++ = 0x00;
    *p++ = 0x00;

    *olen = 4;
}
#endif /* MBEDTLS_SSL_ENCRYPT_THEN_MAC */

#if defined(MBEDTLS_SSL_EXTENDED_MASTER_SECRET)
static void ssl_write_extended_ms_ext( mbedtls_ssl_context *ssl,
                                       unsigned char *buf,
                                       size_t *olen )
{
    unsigned char *p = buf;

    if( ssl->handshake->extended_ms == MBEDTLS_SSL_EXTENDED_MS_DISABLED ||
        ssl->minor_ver == MBEDTLS_SSL_MINOR_VERSION_0 )
    {
        *olen = 0;
        return;
    }

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "server hello, adding extended master secret "
                        "extension" ) );

    MBEDTLS_PUT_UINT16_BE( MBEDTLS_TLS_EXT_EXTENDED_MASTER_SECRET, p, 0 );
    p += 2;

    *p++ = 0x00;
    *p++ = 0x00;

    *olen = 4;
}
#endif /* MBEDTLS_SSL_EXTENDED_MASTER_SECRET */

#if defined(MBEDTLS_SSL_SESSION_TICKETS)
static void ssl_write_session_ticket_ext( mbedtls_ssl_context *ssl,
                                          unsigned char *buf,
                                          size_t *olen )
{
    unsigned char *p = buf;

    if( ssl->handshake->new_session_ticket == 0 )
    {
        *olen = 0;
        return;
    }

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "server hello, adding session ticket extension" ) );

    MBEDTLS_PUT_UINT16_BE( MBEDTLS_TLS_EXT_SESSION_TICKET, p, 0 );
    p += 2;

    *p++ = 0x00;
    *p++ = 0x00;

    *olen = 4;
}
#endif /* MBEDTLS_SSL_SESSION_TICKETS */

static void ssl_write_renegotiation_ext( mbedtls_ssl_context *ssl,
                                         unsigned char *buf,
                                         size_t *olen )
{
    unsigned char *p = buf;

    if( ssl->secure_renegotiation != MBEDTLS_SSL_SECURE_RENEGOTIATION )
    {
        *olen = 0;
        return;
    }

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "server hello, secure renegotiation extension" ) );

    MBEDTLS_PUT_UINT16_BE( MBEDTLS_TLS_EXT_RENEGOTIATION_INFO, p, 0 );
    p += 2;

#if defined(MBEDTLS_SSL_RENEGOTIATION)
    if( ssl->renego_status != MBEDTLS_SSL_INITIAL_HANDSHAKE )
    {
        *p++ = 0x00;
        *p++ = ( ssl->verify_data_len * 2 + 1 ) & 0xFF;
        *p++ = ssl->verify_data_len * 2 & 0xFF;

        memcpy( p, ssl->peer_verify_data, ssl->verify_data_len );
        p += ssl->verify_data_len;
        memcpy( p, ssl->own_verify_data, ssl->verify_data_len );
        p += ssl->verify_data_len;
    }
    else
#endif /* MBEDTLS_SSL_RENEGOTIATION */
    {
        *p++ = 0x00;
        *p++ = 0x01;
        *p++ = 0x00;
    }

    *olen = p - buf;
}

#if defined(MBEDTLS_SSL_MAX_FRAGMENT_LENGTH)
static void ssl_write_max_fragment_length_ext( mbedtls_ssl_context *ssl,
                                               unsigned char *buf,
                                               size_t *olen )
{
    unsigned char *p = buf;

    if( ssl->session_negotiate->mfl_code == MBEDTLS_SSL_MAX_FRAG_LEN_NONE )
    {
        *olen = 0;
        return;
    }

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "server hello, max_fragment_length extension" ) );

    MBEDTLS_PUT_UINT16_BE( MBEDTLS_TLS_EXT_MAX_FRAGMENT_LENGTH, p, 0 );
    p += 2;

    *p++ = 0x00;
    *p++ = 1;

    *p++ = ssl->session_negotiate->mfl_code;

    *olen = 5;
}
#endif /* MBEDTLS_SSL_MAX_FRAGMENT_LENGTH */

#if defined(MBEDTLS_ECDH_C) || defined(MBEDTLS_ECDSA_C) || \
    defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
static void ssl_write_supported_point_formats_ext( mbedtls_ssl_context *ssl,
                                                   unsigned char *buf,
                                                   size_t *olen )
{
    unsigned char *p = buf;
    ((void) ssl);

    if( ( ssl->handshake->cli_exts &
          MBEDTLS_TLS_EXT_SUPPORTED_POINT_FORMATS_PRESENT ) == 0 )
    {
        *olen = 0;
        return;
    }

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "server hello, supported_point_formats extension" ) );

    MBEDTLS_PUT_UINT16_BE( MBEDTLS_TLS_EXT_SUPPORTED_POINT_FORMATS, p, 0 );
    p += 2;

    *p++ = 0x00;
    *p++ = 2;

    *p++ = 1;
    *p++ = MBEDTLS_ECP_PF_UNCOMPRESSED;

    *olen = 6;
}
#endif /* MBEDTLS_ECDH_C || MBEDTLS_ECDSA_C || MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED */

#if defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
static void ssl_write_ecjpake_kkpp_ext( mbedtls_ssl_context *ssl,
                                        unsigned char *buf,
                                        size_t *olen )
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    unsigned char *p = buf;
    const unsigned char *end = ssl->out_msg + MBEDTLS_SSL_OUT_CONTENT_LEN;
    size_t kkpp_len;

    *olen = 0;

    /* Skip costly computation if not needed */
    if( ssl->handshake->ciphersuite_info->key_exchange !=
        MBEDTLS_KEY_EXCHANGE_ECJPAKE )
        return;

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "server hello, ecjpake kkpp extension" ) );

    if( end - p < 4 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "buffer too small" ) );
        return;
    }

    MBEDTLS_PUT_UINT16_BE( MBEDTLS_TLS_EXT_ECJPAKE_KKPP, p, 0 );
    p += 2;

    ret = mbedtls_ecjpake_write_round_one( &ssl->handshake->ecjpake_ctx,
                                        p + 2, end - p - 2, &kkpp_len,
                                        ssl->conf->f_rng, ssl->conf->p_rng );
    if( ret != 0 )
    {
        MBEDTLS_SSL_DEBUG_RET( 1 , "mbedtls_ecjpake_write_round_one", ret );
        return;
    }

    MBEDTLS_PUT_UINT16_BE( kkpp_len, p, 0 );
    p += 2;

    *olen = kkpp_len + 4;
}
#endif /* MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED */

#if defined(MBEDTLS_SSL_ALPN )
static void ssl_write_alpn_ext( mbedtls_ssl_context *ssl,
                                unsigned char *buf, size_t *olen )
{
    if( ssl->alpn_chosen == NULL )
    {
        *olen = 0;
        return;
    }

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "server hello, adding alpn extension" ) );

    /*
     * 0 . 1    ext identifier
     * 2 . 3    ext length
     * 4 . 5    protocol list length
     * 6 . 6    protocol name length
     * 7 . 7+n  protocol name
     */
    MBEDTLS_PUT_UINT16_BE( MBEDTLS_TLS_EXT_ALPN, buf, 0);

    *olen = 7 + strlen( ssl->alpn_chosen );

    MBEDTLS_PUT_UINT16_BE( *olen - 4, buf, 2 );

    MBEDTLS_PUT_UINT16_BE( *olen - 6, buf, 4 );

    buf[6] = MBEDTLS_BYTE_0( *olen - 7 );

    memcpy( buf + 7, ssl->alpn_chosen, *olen - 7 );
}
#endif /* MBEDTLS_ECDH_C || MBEDTLS_ECDSA_C */

#if defined(MBEDTLS_SSL_DTLS_SRTP ) && defined(MBEDTLS_SSL_PROTO_DTLS)
static void ssl_write_use_srtp_ext( mbedtls_ssl_context *ssl,
                                    unsigned char *buf,
                                    size_t *olen )
{
    size_t mki_len = 0, ext_len = 0;
    uint16_t profile_value = 0;
    const unsigned char *end = ssl->out_msg + MBEDTLS_SSL_OUT_CONTENT_LEN;

    *olen = 0;

    if( ( ssl->conf->transport != MBEDTLS_SSL_TRANSPORT_DATAGRAM ) ||
        ( ssl->dtls_srtp_info.chosen_dtls_srtp_profile == MBEDTLS_TLS_SRTP_UNSET ) )
    {
        return;
    }

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "server hello, adding use_srtp extension" ) );

    if( ssl->conf->dtls_srtp_mki_support == MBEDTLS_SSL_DTLS_SRTP_MKI_SUPPORTED )
    {
        mki_len = ssl->dtls_srtp_info.mki_len;
    }

    /* The extension total size is 9 bytes :
     * - 2 bytes for the extension tag
     * - 2 bytes for the total size
     * - 2 bytes for the protection profile length
     * - 2 bytes for the protection profile
     * - 1 byte for the mki length
     * +  the actual mki length
     * Check we have enough room in the output buffer */
    if( (size_t)( end - buf ) < mki_len + 9 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "buffer too small" ) );
        return;
    }

    /* extension */
    MBEDTLS_PUT_UINT16_BE(MBEDTLS_TLS_EXT_USE_SRTP, buf, 0 );
    /*
     * total length 5 and mki value: only one profile(2 bytes)
     *              and length(2 bytes) and srtp_mki  )
     */
    ext_len = 5 + mki_len;
    MBEDTLS_PUT_UINT16_BE( ext_len, buf, 2 );

    /* protection profile length: 2 */
    buf[4] = 0x00;
    buf[5] = 0x02;
    profile_value = mbedtls_ssl_check_srtp_profile_value(
                                ssl->dtls_srtp_info.chosen_dtls_srtp_profile );
    if( profile_value != MBEDTLS_TLS_SRTP_UNSET )
    {
        MBEDTLS_PUT_UINT16_BE( profile_value, buf, 6 );
    }
    else
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "use_srtp extension invalid profile" ) );
        return;
    }

    buf[8] = mki_len & 0xFF;
    memcpy( &buf[9], ssl->dtls_srtp_info.mki_value, mki_len );

    *olen = 9 + mki_len;
}
#endif /* MBEDTLS_SSL_DTLS_SRTP */

#if defined(MBEDTLS_SSL_DTLS_HELLO_VERIFY)
static int ssl_write_hello_verify_request( mbedtls_ssl_context *ssl )
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    unsigned char *p = ssl->out_msg + 4;
    unsigned char *cookie_len_byte;

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "=> write hello verify request" ) );

    /*
     * struct {
     *   ProtocolVersion server_version;
     *   opaque cookie<0..2^8-1>;
     * } HelloVerifyRequest;
     */

    /* The RFC is not clear on this point, but sending the actual negotiated
     * version looks like the most interoperable thing to do. */
    mbedtls_ssl_write_version( ssl->major_ver, ssl->minor_ver,
                       ssl->conf->transport, p );
    MBEDTLS_SSL_DEBUG_BUF( 3, "server version", p, 2 );
    p += 2;

    /* If we get here, f_cookie_check is not null */
    if( ssl->conf->f_cookie_write == NULL )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "inconsistent cookie callbacks" ) );
        return( MBEDTLS_ERR_SSL_INTERNAL_ERROR );
    }

    /* Skip length byte until we know the length */
    cookie_len_byte = p++;

    if( ( ret = ssl->conf->f_cookie_write( ssl->conf->p_cookie,
                                     &p, ssl->out_buf + MBEDTLS_SSL_OUT_BUFFER_LEN,
                                     ssl->cli_id, ssl->cli_id_len ) ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_RET( 1, "f_cookie_write", ret );
        return( ret );
    }

    *cookie_len_byte = (unsigned char)( p - ( cookie_len_byte + 1 ) );

    MBEDTLS_SSL_DEBUG_BUF( 3, "cookie sent", cookie_len_byte + 1, *cookie_len_byte );

    ssl->out_msglen  = p - ssl->out_msg;
    ssl->out_msgtype = MBEDTLS_SSL_MSG_HANDSHAKE;
    ssl->out_msg[0]  = MBEDTLS_SSL_HS_HELLO_VERIFY_REQUEST;

    ssl->state = MBEDTLS_SSL_SERVER_HELLO_VERIFY_REQUEST_SENT;

    if( ( ret = mbedtls_ssl_write_handshake_msg( ssl ) ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ssl_write_handshake_msg", ret );
        return( ret );
    }

#if defined(MBEDTLS_SSL_PROTO_DTLS)
    if( ssl->conf->transport == MBEDTLS_SSL_TRANSPORT_DATAGRAM &&
        ( ret = mbedtls_ssl_flight_transmit( ssl ) ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ssl_flight_transmit", ret );
        return( ret );
    }
#endif /* MBEDTLS_SSL_PROTO_DTLS */

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= write hello verify request" ) );

    return( 0 );
}
#endif /* MBEDTLS_SSL_DTLS_HELLO_VERIFY */

static void ssl_handle_id_based_session_resumption( mbedtls_ssl_context *ssl )
{
    int ret;
    mbedtls_ssl_session session_tmp;
    mbedtls_ssl_session * const session = ssl->session_negotiate;

    /* Resume is 0  by default, see ssl_handshake_init().
     * It may be already set to 1 by ssl_parse_session_ticket_ext(). */
    if( ssl->handshake->resume == 1 )
        return;
    if( session->id_len == 0 )
        return;
    if( ssl->conf->f_get_cache == NULL )
        return;
#if defined(MBEDTLS_SSL_RENEGOTIATION)
    if( ssl->renego_status != MBEDTLS_SSL_INITIAL_HANDSHAKE )
        return;
#endif

    mbedtls_ssl_session_init( &session_tmp );

    session_tmp.id_len = session->id_len;
    memcpy( session_tmp.id, session->id, session->id_len );

    ret = ssl->conf->f_get_cache( ssl->conf->p_cache,
                                  &session_tmp );
    if( ret != 0 )
        goto exit;

    if( session->ciphersuite != session_tmp.ciphersuite ||
        session->compression != session_tmp.compression )
    {
        /* Mismatch between cached and negotiated session */
        goto exit;
    }

    /* Move semantics */
    mbedtls_ssl_session_free( session );
    *session = session_tmp;
    memset( &session_tmp, 0, sizeof( session_tmp ) );

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "session successfully restored from cache" ) );
    ssl->handshake->resume = 1;

exit:

    mbedtls_ssl_session_free( &session_tmp );
}

static int ssl_write_server_hello( mbedtls_ssl_context *ssl )
{
#if defined(MBEDTLS_HAVE_TIME)
    mbedtls_time_t t;
#endif
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    size_t olen, ext_len = 0, n;
    unsigned char *buf, *p;

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "=> write server hello" ) );

#if defined(MBEDTLS_SSL_DTLS_HELLO_VERIFY)
    if( ssl->conf->transport == MBEDTLS_SSL_TRANSPORT_DATAGRAM &&
        ssl->handshake->verify_cookie_len != 0 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 2, ( "client hello was not authenticated" ) );
        MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= write server hello" ) );

        return( ssl_write_hello_verify_request( ssl ) );
    }
#endif /* MBEDTLS_SSL_DTLS_HELLO_VERIFY */

    if( ssl->conf->f_rng == NULL )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "no RNG provided") );
        return( MBEDTLS_ERR_SSL_NO_RNG );
    }

    /*
     *     0  .   0   handshake type
     *     1  .   3   handshake length
     *     4  .   5   protocol version
     *     6  .   9   UNIX time()
     *    10  .  37   random bytes
     */
    buf = ssl->out_msg;
    p = buf + 4;

    mbedtls_ssl_write_version( ssl->major_ver, ssl->minor_ver,
                       ssl->conf->transport, p );
    p += 2;

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "server hello, chosen version: [%d:%d]",
                        buf[4], buf[5] ) );

#if defined(MBEDTLS_HAVE_TIME)
    t = mbedtls_time( NULL );
    MBEDTLS_PUT_UINT32_BE( t, p, 0 );
    p += 4;

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "server hello, current time: %" MBEDTLS_PRINTF_LONGLONG,
                                (long long) t ) );
#else
    if( ( ret = ssl->conf->f_rng( ssl->conf->p_rng, p, 4 ) ) != 0 )
        return( ret );

    p += 4;
#endif /* MBEDTLS_HAVE_TIME */

    if( ( ret = ssl->conf->f_rng( ssl->conf->p_rng, p, 28 ) ) != 0 )
        return( ret );

    p += 28;

    memcpy( ssl->handshake->randbytes + 32, buf + 6, 32 );

    MBEDTLS_SSL_DEBUG_BUF( 3, "server hello, random bytes", buf + 6, 32 );

    ssl_handle_id_based_session_resumption( ssl );

    if( ssl->handshake->resume == 0 )
    {
        /*
         * New session, create a new session id,
         * unless we're about to issue a session ticket
         */
        ssl->state++;

#if defined(MBEDTLS_HAVE_TIME)
        ssl->session_negotiate->start = mbedtls_time( NULL );
#endif

#if defined(MBEDTLS_SSL_SESSION_TICKETS)
        if( ssl->handshake->new_session_ticket != 0 )
        {
            ssl->session_negotiate->id_len = n = 0;
            memset( ssl->session_negotiate->id, 0, 32 );
        }
        else
#endif /* MBEDTLS_SSL_SESSION_TICKETS */
        {
            ssl->session_negotiate->id_len = n = 32;
            if( ( ret = ssl->conf->f_rng( ssl->conf->p_rng, ssl->session_negotiate->id,
                                    n ) ) != 0 )
                return( ret );
        }
    }
    else
    {
        /*
         * Resuming a session
         */
        n = ssl->session_negotiate->id_len;
        ssl->state = MBEDTLS_SSL_SERVER_CHANGE_CIPHER_SPEC;

        if( ( ret = mbedtls_ssl_derive_keys( ssl ) ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ssl_derive_keys", ret );
            return( ret );
        }
    }

    /*
     *    38  .  38     session id length
     *    39  . 38+n    session id
     *   39+n . 40+n    chosen ciphersuite
     *   41+n . 41+n    chosen compression alg.
     *   42+n . 43+n    extensions length
     *   44+n . 43+n+m  extensions
     */
    *p++ = (unsigned char) ssl->session_negotiate->id_len;
    memcpy( p, ssl->session_negotiate->id, ssl->session_negotiate->id_len );
    p += ssl->session_negotiate->id_len;

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "server hello, session id len.: %" MBEDTLS_PRINTF_SIZET, n ) );
    MBEDTLS_SSL_DEBUG_BUF( 3,   "server hello, session id", buf + 39, n );
    MBEDTLS_SSL_DEBUG_MSG( 3, ( "%s session has been resumed",
                   ssl->handshake->resume ? "a" : "no" ) );

    MBEDTLS_PUT_UINT16_BE( ssl->session_negotiate->ciphersuite, p, 0 );
    p += 2;
    *p++ = MBEDTLS_BYTE_0( ssl->session_negotiate->compression );

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "server hello, chosen ciphersuite: %s",
           mbedtls_ssl_get_ciphersuite_name( ssl->session_negotiate->ciphersuite ) ) );
    MBEDTLS_SSL_DEBUG_MSG( 3, ( "server hello, compress alg.: 0x%02X",
                   (unsigned int) ssl->session_negotiate->compression ) );

    /* Do not write the extensions if the protocol is SSLv3 */
#if defined(MBEDTLS_SSL_PROTO_SSL3)
    if( ( ssl->major_ver != 3 ) || ( ssl->minor_ver != 0 ) )
    {
#endif

    /*
     *  First write extensions, then the total length
     */
    ssl_write_renegotiation_ext( ssl, p + 2 + ext_len, &olen );
    ext_len += olen;

#if defined(MBEDTLS_SSL_MAX_FRAGMENT_LENGTH)
    ssl_write_max_fragment_length_ext( ssl, p + 2 + ext_len, &olen );
    ext_len += olen;
#endif

#if defined(MBEDTLS_SSL_TRUNCATED_HMAC)
    ssl_write_truncated_hmac_ext( ssl, p + 2 + ext_len, &olen );
    ext_len += olen;
#endif

#if defined(MBEDTLS_SSL_DTLS_CONNECTION_ID)
    ssl_write_cid_ext( ssl, p + 2 + ext_len, &olen );
    ext_len += olen;
#endif

#if defined(MBEDTLS_SSL_ENCRYPT_THEN_MAC)
    ssl_write_encrypt_then_mac_ext( ssl, p + 2 + ext_len, &olen );
    ext_len += olen;
#endif

#if defined(MBEDTLS_SSL_EXTENDED_MASTER_SECRET)
    ssl_write_extended_ms_ext( ssl, p + 2 + ext_len, &olen );
    ext_len += olen;
#endif

#if defined(MBEDTLS_SSL_SESSION_TICKETS)
    ssl_write_session_ticket_ext( ssl, p + 2 + ext_len, &olen );
    ext_len += olen;
#endif

#if defined(MBEDTLS_ECDH_C) || defined(MBEDTLS_ECDSA_C) || \
    defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
    if ( mbedtls_ssl_ciphersuite_uses_ec(
         mbedtls_ssl_ciphersuite_from_id( ssl->session_negotiate->ciphersuite ) ) )
    {
        ssl_write_supported_point_formats_ext( ssl, p + 2 + ext_len, &olen );
        ext_len += olen;
    }
#endif

#if defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
    ssl_write_ecjpake_kkpp_ext( ssl, p + 2 + ext_len, &olen );
    ext_len += olen;
#endif

#if defined(MBEDTLS_SSL_ALPN)
    ssl_write_alpn_ext( ssl, p + 2 + ext_len, &olen );
    ext_len += olen;
#endif

#if defined(MBEDTLS_SSL_DTLS_SRTP)
    ssl_write_use_srtp_ext( ssl, p + 2 + ext_len, &olen );
    ext_len += olen;
#endif

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "server hello, total extension length: %" MBEDTLS_PRINTF_SIZET,
                                ext_len ) );

    if( ext_len > 0 )
    {
        MBEDTLS_PUT_UINT16_BE( ext_len, p, 0 );
        p += 2 + ext_len;
    }

#if defined(MBEDTLS_SSL_PROTO_SSL3)
    }
#endif

    ssl->out_msglen  = p - buf;
    ssl->out_msgtype = MBEDTLS_SSL_MSG_HANDSHAKE;
    ssl->out_msg[0]  = MBEDTLS_SSL_HS_SERVER_HELLO;

    ret = mbedtls_ssl_write_handshake_msg( ssl );

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= write server hello" ) );

    return( ret );
}

#if !defined(MBEDTLS_KEY_EXCHANGE_CERT_REQ_ALLOWED_ENABLED)
static int ssl_write_certificate_request( mbedtls_ssl_context *ssl )
{
    const mbedtls_ssl_ciphersuite_t *ciphersuite_info =
        ssl->handshake->ciphersuite_info;

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "=> write certificate request" ) );

    if( !mbedtls_ssl_ciphersuite_cert_req_allowed( ciphersuite_info ) )
    {
        MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= skip write certificate request" ) );
        ssl->state++;
        return( 0 );
    }

    MBEDTLS_SSL_DEBUG_MSG( 1, ( "should never happen" ) );
    return( MBEDTLS_ERR_SSL_INTERNAL_ERROR );
}
#else /* !MBEDTLS_KEY_EXCHANGE_CERT_REQ_ALLOWED_ENABLED */
static int ssl_write_certificate_request( mbedtls_ssl_context *ssl )
{
    int ret = MBEDTLS_ERR_SSL_FEATURE_UNAVAILABLE;
    const mbedtls_ssl_ciphersuite_t *ciphersuite_info =
        ssl->handshake->ciphersuite_info;
    uint16_t dn_size, total_dn_size; /* excluding length bytes */
    size_t ct_len, sa_len; /* including length bytes */
    unsigned char *buf, *p;
    const unsigned char * const end = ssl->out_msg + MBEDTLS_SSL_OUT_CONTENT_LEN;
    const mbedtls_x509_crt *crt;
    int authmode;

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "=> write certificate request" ) );

    ssl->state++;

#if defined(MBEDTLS_SSL_SERVER_NAME_INDICATION)
    if( ssl->handshake->sni_authmode != MBEDTLS_SSL_VERIFY_UNSET )
        authmode = ssl->handshake->sni_authmode;
    else
#endif
        authmode = ssl->conf->authmode;

    if( !mbedtls_ssl_ciphersuite_cert_req_allowed( ciphersuite_info ) ||
        authmode == MBEDTLS_SSL_VERIFY_NONE )
    {
        MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= skip write certificate request" ) );
        return( 0 );
    }

    /*
     *     0  .   0   handshake type
     *     1  .   3   handshake length
     *     4  .   4   cert type count
     *     5  .. m-1  cert types
     *     m  .. m+1  sig alg length (TLS 1.2 only)
     *    m+1 .. n-1  SignatureAndHashAlgorithms (TLS 1.2 only)
     *     n  .. n+1  length of all DNs
     *    n+2 .. n+3  length of DN 1
     *    n+4 .. ...  Distinguished Name #1
     *    ... .. ...  length of DN 2, etc.
     */
    buf = ssl->out_msg;
    p = buf + 4;

    /*
     * Supported certificate types
     *
     *     ClientCertificateType certificate_types<1..2^8-1>;
     *     enum { (255) } ClientCertificateType;
     */
    ct_len = 0;

#if defined(MBEDTLS_RSA_C)
    p[1 + ct_len++] = MBEDTLS_SSL_CERT_TYPE_RSA_SIGN;
#endif
#if defined(MBEDTLS_ECDSA_C)
    p[1 + ct_len++] = MBEDTLS_SSL_CERT_TYPE_ECDSA_SIGN;
#endif

    p[0] = (unsigned char) ct_len++;
    p += ct_len;

    sa_len = 0;
#if defined(MBEDTLS_SSL_PROTO_TLS1_2)
    /*
     * Add signature_algorithms for verify (TLS 1.2)
     *
     *     SignatureAndHashAlgorithm supported_signature_algorithms<2..2^16-2>;
     *
     *     struct {
     *           HashAlgorithm hash;
     *           SignatureAlgorithm signature;
     *     } SignatureAndHashAlgorithm;
     *
     *     enum { (255) } HashAlgorithm;
     *     enum { (255) } SignatureAlgorithm;
     */
    if( ssl->minor_ver == MBEDTLS_SSL_MINOR_VERSION_3 )
    {
        const int *cur;

        /*
         * Supported signature algorithms
         */
        for( cur = ssl->conf->sig_hashes; *cur != MBEDTLS_MD_NONE; cur++ )
        {
            unsigned char hash = mbedtls_ssl_hash_from_md_alg( *cur );

            if( MBEDTLS_SSL_HASH_NONE == hash || mbedtls_ssl_set_calc_verify_md( ssl, hash ) )
                continue;

#if defined(MBEDTLS_RSA_C)
            p[2 + sa_len++] = hash;
            p[2 + sa_len++] = MBEDTLS_SSL_SIG_RSA;
#endif
#if defined(MBEDTLS_ECDSA_C)
            p[2 + sa_len++] = hash;
            p[2 + sa_len++] = MBEDTLS_SSL_SIG_ECDSA;
#endif
        }

        MBEDTLS_PUT_UINT16_BE( sa_len, p, 0 );
        sa_len += 2;
        p += sa_len;
    }
#endif /* MBEDTLS_SSL_PROTO_TLS1_2 */

    /*
     * DistinguishedName certificate_authorities<0..2^16-1>;
     * opaque DistinguishedName<1..2^16-1>;
     */
    p += 2;

    total_dn_size = 0;

    if( ssl->conf->cert_req_ca_list ==  MBEDTLS_SSL_CERT_REQ_CA_LIST_ENABLED )
    {
        /* NOTE: If trusted certificates are provisioned
         *       via a CA callback (configured through
         *       `mbedtls_ssl_conf_ca_cb()`, then the
         *       CertificateRequest is currently left empty. */

#if defined(MBEDTLS_SSL_SERVER_NAME_INDICATION)
        if( ssl->handshake->sni_ca_chain != NULL )
            crt = ssl->handshake->sni_ca_chain;
        else
#endif
            crt = ssl->conf->ca_chain;

        while( crt != NULL && crt->version != 0 )
        {
            /* It follows from RFC 5280 A.1 that this length
             * can be represented in at most 11 bits. */
            dn_size = (uint16_t) crt->subject_raw.len;

            if( end < p || (size_t)( end - p ) < 2 + (size_t) dn_size )
            {
                MBEDTLS_SSL_DEBUG_MSG( 1, ( "skipping CAs: buffer too short" ) );
                break;
            }

            MBEDTLS_PUT_UINT16_BE( dn_size, p, 0 );
            p += 2;
            memcpy( p, crt->subject_raw.p, dn_size );
            p += dn_size;

            MBEDTLS_SSL_DEBUG_BUF( 3, "requested DN", p - dn_size, dn_size );

            total_dn_size += 2 + dn_size;
            crt = crt->next;
        }
    }

    ssl->out_msglen  = p - buf;
    ssl->out_msgtype = MBEDTLS_SSL_MSG_HANDSHAKE;
    ssl->out_msg[0]  = MBEDTLS_SSL_HS_CERTIFICATE_REQUEST;
    MBEDTLS_PUT_UINT16_BE( total_dn_size, ssl->out_msg, 4 + ct_len + sa_len );

    ret = mbedtls_ssl_write_handshake_msg( ssl );

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= write certificate request" ) );

    return( ret );
}
#endif /* MBEDTLS_KEY_EXCHANGE_CERT_REQ_ALLOWED_ENABLED */

#if defined(MBEDTLS_KEY_EXCHANGE_ECDH_RSA_ENABLED) || \
    defined(MBEDTLS_KEY_EXCHANGE_ECDH_ECDSA_ENABLED)
static int ssl_get_ecdh_params_from_cert( mbedtls_ssl_context *ssl )
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;

    if( ! mbedtls_pk_can_do( mbedtls_ssl_own_key( ssl ), MBEDTLS_PK_ECKEY ) )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "server key not ECDH capable" ) );
        return( MBEDTLS_ERR_SSL_PK_TYPE_MISMATCH );
    }

    if( ( ret = mbedtls_ecdh_get_params( &ssl->handshake->ecdh_ctx,
                                 mbedtls_pk_ec( *mbedtls_ssl_own_key( ssl ) ),
                                 MBEDTLS_ECDH_OURS ) ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_RET( 1, ( "mbedtls_ecdh_get_params" ), ret );
        return( ret );
    }

    return( 0 );
}
#endif /* MBEDTLS_KEY_EXCHANGE_ECDH_RSA_ENABLED) ||
          MBEDTLS_KEY_EXCHANGE_ECDH_ECDSA_ENABLED */

#if defined(MBEDTLS_KEY_EXCHANGE_WITH_SERVER_SIGNATURE_ENABLED) && \
    defined(MBEDTLS_SSL_ASYNC_PRIVATE)
static int ssl_resume_server_key_exchange( mbedtls_ssl_context *ssl,
                                           size_t *signature_len )
{
    /* Append the signature to ssl->out_msg, leaving 2 bytes for the
     * signature length which will be added in ssl_write_server_key_exchange
     * after the call to ssl_prepare_server_key_exchange.
     * ssl_write_server_key_exchange also takes care of incrementing
     * ssl->out_msglen. */
    unsigned char *sig_start = ssl->out_msg + ssl->out_msglen + 2;
    size_t sig_max_len = ( ssl->out_buf + MBEDTLS_SSL_OUT_CONTENT_LEN
                           - sig_start );
    int ret = ssl->conf->f_async_resume( ssl,
                                         sig_start, signature_len, sig_max_len );
    if( ret != MBEDTLS_ERR_SSL_ASYNC_IN_PROGRESS )
    {
        ssl->handshake->async_in_progress = 0;
        mbedtls_ssl_set_async_operation_data( ssl, NULL );
    }
    MBEDTLS_SSL_DEBUG_RET( 2, "ssl_resume_server_key_exchange", ret );
    return( ret );
}
#endif /* defined(MBEDTLS_KEY_EXCHANGE_WITH_SERVER_SIGNATURE_ENABLED) &&
          defined(MBEDTLS_SSL_ASYNC_PRIVATE) */

/* Prepare the ServerKeyExchange message, up to and including
 * calculating the signature if any, but excluding formatting the
 * signature and sending the message. */
static int ssl_prepare_server_key_exchange( mbedtls_ssl_context *ssl,
                                            size_t *signature_len )
{
    const mbedtls_ssl_ciphersuite_t *ciphersuite_info =
        ssl->handshake->ciphersuite_info;

#if defined(MBEDTLS_KEY_EXCHANGE_SOME_PFS_ENABLED)
#if defined(MBEDTLS_KEY_EXCHANGE_WITH_SERVER_SIGNATURE_ENABLED)
    unsigned char *dig_signed = NULL;
#endif /* MBEDTLS_KEY_EXCHANGE_WITH_SERVER_SIGNATURE_ENABLED */
#endif /* MBEDTLS_KEY_EXCHANGE_SOME_PFS_ENABLED */

    (void) ciphersuite_info; /* unused in some configurations */
#if !defined(MBEDTLS_KEY_EXCHANGE_WITH_SERVER_SIGNATURE_ENABLED)
    (void) signature_len;
#endif /* MBEDTLS_KEY_EXCHANGE_WITH_SERVER_SIGNATURE_ENABLED */

    ssl->out_msglen = 4; /* header (type:1, length:3) to be written later */

    /*
     *
     * Part 1: Provide key exchange parameters for chosen ciphersuite.
     *
     */

    /*
     * - ECJPAKE key exchanges
     */
#if defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
    if( ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_ECJPAKE )
    {
        int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
        size_t len = 0;

        ret = mbedtls_ecjpake_write_round_two(
            &ssl->handshake->ecjpake_ctx,
            ssl->out_msg + ssl->out_msglen,
            MBEDTLS_SSL_OUT_CONTENT_LEN - ssl->out_msglen, &len,
            ssl->conf->f_rng, ssl->conf->p_rng );
        if( ret != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ecjpake_write_round_two", ret );
            return( ret );
        }

        ssl->out_msglen += len;
    }
#endif /* MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED */

    /*
     * For (EC)DHE key exchanges with PSK, parameters are prefixed by support
     * identity hint (RFC 4279, Sec. 3). Until someone needs this feature,
     * we use empty support identity hints here.
     **/
#if defined(MBEDTLS_KEY_EXCHANGE_DHE_PSK_ENABLED)   || \
    defined(MBEDTLS_KEY_EXCHANGE_ECDHE_PSK_ENABLED)
    if( ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_DHE_PSK ||
        ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_ECDHE_PSK )
    {
        ssl->out_msg[ssl->out_msglen++] = 0x00;
        ssl->out_msg[ssl->out_msglen++] = 0x00;
    }
#endif /* MBEDTLS_KEY_EXCHANGE_DHE_PSK_ENABLED ||
          MBEDTLS_KEY_EXCHANGE_ECDHE_PSK_ENABLED */

    /*
     * - DHE key exchanges
     */
#if defined(MBEDTLS_KEY_EXCHANGE_SOME_DHE_ENABLED)
    if( mbedtls_ssl_ciphersuite_uses_dhe( ciphersuite_info ) )
    {
        int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
        size_t len = 0;

        if( ssl->conf->dhm_P.p == NULL || ssl->conf->dhm_G.p == NULL )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "no DH parameters set" ) );
            return( MBEDTLS_ERR_SSL_BAD_INPUT_DATA );
        }

        /*
         * Ephemeral DH parameters:
         *
         * struct {
         *     opaque dh_p<1..2^16-1>;
         *     opaque dh_g<1..2^16-1>;
         *     opaque dh_Ys<1..2^16-1>;
         * } ServerDHParams;
         */
        if( ( ret = mbedtls_dhm_set_group( &ssl->handshake->dhm_ctx,
                                           &ssl->conf->dhm_P,
                                           &ssl->conf->dhm_G ) ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_dhm_set_group", ret );
            return( ret );
        }

        if( ( ret = mbedtls_dhm_make_params(
                  &ssl->handshake->dhm_ctx,
                  (int) mbedtls_mpi_size( &ssl->handshake->dhm_ctx.P ),
                  ssl->out_msg + ssl->out_msglen, &len,
                  ssl->conf->f_rng, ssl->conf->p_rng ) ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_dhm_make_params", ret );
            return( ret );
        }

#if defined(MBEDTLS_KEY_EXCHANGE_WITH_SERVER_SIGNATURE_ENABLED)
        dig_signed = ssl->out_msg + ssl->out_msglen;
#endif

        ssl->out_msglen += len;

        MBEDTLS_SSL_DEBUG_MPI( 3, "DHM: X ", &ssl->handshake->dhm_ctx.X  );
        MBEDTLS_SSL_DEBUG_MPI( 3, "DHM: P ", &ssl->handshake->dhm_ctx.P  );
        MBEDTLS_SSL_DEBUG_MPI( 3, "DHM: G ", &ssl->handshake->dhm_ctx.G  );
        MBEDTLS_SSL_DEBUG_MPI( 3, "DHM: GX", &ssl->handshake->dhm_ctx.GX );
    }
#endif /* MBEDTLS_KEY_EXCHANGE_SOME_DHE_ENABLED */

    /*
     * - ECDHE key exchanges
     */
#if defined(MBEDTLS_KEY_EXCHANGE_SOME_ECDHE_ENABLED)
    if( mbedtls_ssl_ciphersuite_uses_ecdhe( ciphersuite_info ) )
    {
        /*
         * Ephemeral ECDH parameters:
         *
         * struct {
         *     ECParameters curve_params;
         *     ECPoint      public;
         * } ServerECDHParams;
         */
        const mbedtls_ecp_curve_info **curve = NULL;
        const mbedtls_ecp_group_id *gid;
        int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
        size_t len = 0;

        /* Match our preference list against the offered curves */
        for( gid = ssl->conf->curve_list; *gid != MBEDTLS_ECP_DP_NONE; gid++ )
            for( curve = ssl->handshake->curves; *curve != NULL; curve++ )
                if( (*curve)->grp_id == *gid )
                    goto curve_matching_done;

curve_matching_done:
        if( curve == NULL || *curve == NULL )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "no matching curve for ECDHE" ) );
            return( MBEDTLS_ERR_SSL_NO_CIPHER_CHOSEN );
        }

        MBEDTLS_SSL_DEBUG_MSG( 2, ( "ECDHE curve: %s", (*curve)->name ) );

        if( ( ret = mbedtls_ecdh_setup( &ssl->handshake->ecdh_ctx,
                                        (*curve)->grp_id ) ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ecp_group_load", ret );
            return( ret );
        }

        if( ( ret = mbedtls_ecdh_make_params(
                  &ssl->handshake->ecdh_ctx, &len,
                  ssl->out_msg + ssl->out_msglen,
                  MBEDTLS_SSL_OUT_CONTENT_LEN - ssl->out_msglen,
                  ssl->conf->f_rng, ssl->conf->p_rng ) ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ecdh_make_params", ret );
            return( ret );
        }

#if defined(MBEDTLS_KEY_EXCHANGE_WITH_SERVER_SIGNATURE_ENABLED)
        dig_signed = ssl->out_msg + ssl->out_msglen;
#endif

        ssl->out_msglen += len;

        MBEDTLS_SSL_DEBUG_ECDH( 3, &ssl->handshake->ecdh_ctx,
                                MBEDTLS_DEBUG_ECDH_Q );
    }
#endif /* MBEDTLS_KEY_EXCHANGE_SOME_ECDHE_ENABLED */

    /*
     *
     * Part 2: For key exchanges involving the server signing the
     *         exchange parameters, compute and add the signature here.
     *
     */
#if defined(MBEDTLS_KEY_EXCHANGE_WITH_SERVER_SIGNATURE_ENABLED)
    if( mbedtls_ssl_ciphersuite_uses_server_signature( ciphersuite_info ) )
    {
        size_t dig_signed_len = ssl->out_msg + ssl->out_msglen - dig_signed;
        size_t hashlen = 0;
#if defined(MBEDTLS_USE_PSA_CRYPTO)
        unsigned char hash[PSA_HASH_MAX_SIZE];
#else
        unsigned char hash[MBEDTLS_MD_MAX_SIZE];
#endif
        int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;

        /*
         * 2.1: Choose hash algorithm:
         * A: For TLS 1.2, obey signature-hash-algorithm extension
         *    to choose appropriate hash.
         * B: For SSL3, TLS1.0, TLS1.1 and ECDHE_ECDSA, use SHA1
         *    (RFC 4492, Sec. 5.4)
         * C: Otherwise, use MD5 + SHA1 (RFC 4346, Sec. 7.4.3)
         */

        mbedtls_md_type_t md_alg;

#if defined(MBEDTLS_SSL_PROTO_TLS1_2)
        mbedtls_pk_type_t sig_alg =
            mbedtls_ssl_get_ciphersuite_sig_pk_alg( ciphersuite_info );
        if( ssl->minor_ver == MBEDTLS_SSL_MINOR_VERSION_3 )
        {
            /* A: For TLS 1.2, obey signature-hash-algorithm extension
             *    (RFC 5246, Sec. 7.4.1.4.1). */
            if( sig_alg == MBEDTLS_PK_NONE ||
                ( md_alg = mbedtls_ssl_sig_hash_set_find( &ssl->handshake->hash_algs,
                                                          sig_alg ) ) == MBEDTLS_MD_NONE )
            {
                MBEDTLS_SSL_DEBUG_MSG( 1, ( "should never happen" ) );
                /* (... because we choose a cipher suite
                 *      only if there is a matching hash.) */
                return( MBEDTLS_ERR_SSL_INTERNAL_ERROR );
            }
        }
        else
#endif /* MBEDTLS_SSL_PROTO_TLS1_2 */
#if defined(MBEDTLS_SSL_PROTO_SSL3) || defined(MBEDTLS_SSL_PROTO_TLS1) || \
    defined(MBEDTLS_SSL_PROTO_TLS1_1)
        if( ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA )
        {
            /* B: Default hash SHA1 */
            md_alg = MBEDTLS_MD_SHA1;
        }
        else
#endif /* MBEDTLS_SSL_PROTO_SSL3 || MBEDTLS_SSL_PROTO_TLS1 || \
          MBEDTLS_SSL_PROTO_TLS1_1 */
        {
            /* C: MD5 + SHA1 */
            md_alg = MBEDTLS_MD_NONE;
        }

        MBEDTLS_SSL_DEBUG_MSG( 3, ( "pick hash algorithm %u for signing", (unsigned) md_alg ) );

        /*
         * 2.2: Compute the hash to be signed
         */
#if defined(MBEDTLS_SSL_PROTO_SSL3) || defined(MBEDTLS_SSL_PROTO_TLS1) || \
    defined(MBEDTLS_SSL_PROTO_TLS1_1)
        if( md_alg == MBEDTLS_MD_NONE )
        {
            hashlen = 36;
            ret = mbedtls_ssl_get_key_exchange_md_ssl_tls( ssl, hash,
                                                           dig_signed,
                                                           dig_signed_len );
            if( ret != 0 )
                return( ret );
        }
        else
#endif /* MBEDTLS_SSL_PROTO_SSL3 || MBEDTLS_SSL_PROTO_TLS1 || \
          MBEDTLS_SSL_PROTO_TLS1_1 */
#if defined(MBEDTLS_SSL_PROTO_TLS1) || defined(MBEDTLS_SSL_PROTO_TLS1_1) || \
    defined(MBEDTLS_SSL_PROTO_TLS1_2)
        if( md_alg != MBEDTLS_MD_NONE )
        {
            ret = mbedtls_ssl_get_key_exchange_md_tls1_2( ssl, hash, &hashlen,
                                                          dig_signed,
                                                          dig_signed_len,
                                                          md_alg );
            if( ret != 0 )
                return( ret );
        }
        else
#endif /* MBEDTLS_SSL_PROTO_TLS1 || MBEDTLS_SSL_PROTO_TLS1_1 || \
          MBEDTLS_SSL_PROTO_TLS1_2 */
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "should never happen" ) );
            return( MBEDTLS_ERR_SSL_INTERNAL_ERROR );
        }

        MBEDTLS_SSL_DEBUG_BUF( 3, "parameters hash", hash, hashlen );

        /*
         * 2.3: Compute and add the signature
         */
#if defined(MBEDTLS_SSL_PROTO_TLS1_2)
        if( ssl->minor_ver == MBEDTLS_SSL_MINOR_VERSION_3 )
        {
            /*
             * For TLS 1.2, we need to specify signature and hash algorithm
             * explicitly through a prefix to the signature.
             *
             * struct {
             *    HashAlgorithm hash;
             *    SignatureAlgorithm signature;
             * } SignatureAndHashAlgorithm;
             *
             * struct {
             *    SignatureAndHashAlgorithm algorithm;
             *    opaque signature<0..2^16-1>;
             * } DigitallySigned;
             *
             */

            ssl->out_msg[ssl->out_msglen++] =
                mbedtls_ssl_hash_from_md_alg( md_alg );
            ssl->out_msg[ssl->out_msglen++] =
                mbedtls_ssl_sig_from_pk_alg( sig_alg );
        }
#endif /* MBEDTLS_SSL_PROTO_TLS1_2 */

#if defined(MBEDTLS_SSL_ASYNC_PRIVATE)
        if( ssl->conf->f_async_sign_start != NULL )
        {
            ret = ssl->conf->f_async_sign_start( ssl,
                                                 mbedtls_ssl_own_cert( ssl ),
                                                 md_alg, hash, hashlen );
            switch( ret )
            {
            case MBEDTLS_ERR_SSL_HW_ACCEL_FALLTHROUGH:
                /* act as if f_async_sign was null */
                break;
            case 0:
                ssl->handshake->async_in_progress = 1;
                return( ssl_resume_server_key_exchange( ssl, signature_len ) );
            case MBEDTLS_ERR_SSL_ASYNC_IN_PROGRESS:
                ssl->handshake->async_in_progress = 1;
                return( MBEDTLS_ERR_SSL_ASYNC_IN_PROGRESS );
            default:
                MBEDTLS_SSL_DEBUG_RET( 1, "f_async_sign_start", ret );
                return( ret );
            }
        }
#endif /* MBEDTLS_SSL_ASYNC_PRIVATE */

        if( mbedtls_ssl_own_key( ssl ) == NULL )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "got no private key" ) );
            return( MBEDTLS_ERR_SSL_PRIVATE_KEY_REQUIRED );
        }

        /* Append the signature to ssl->out_msg, leaving 2 bytes for the
         * signature length which will be added in ssl_write_server_key_exchange
         * after the call to ssl_prepare_server_key_exchange.
         * ssl_write_server_key_exchange also takes care of incrementing
         * ssl->out_msglen. */
        if( ( ret = mbedtls_pk_sign( mbedtls_ssl_own_key( ssl ),
                                     md_alg, hash, hashlen,
                                     ssl->out_msg + ssl->out_msglen + 2,
                                     signature_len,
                                     ssl->conf->f_rng,
                                     ssl->conf->p_rng ) ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_pk_sign", ret );
            return( ret );
        }
    }
#endif /* MBEDTLS_KEY_EXCHANGE_WITH_SERVER_SIGNATURE_ENABLED */

    return( 0 );
}

/* Prepare the ServerKeyExchange message and send it. For ciphersuites
 * that do not include a ServerKeyExchange message, do nothing. Either
 * way, if successful, move on to the next step in the SSL state
 * machine. */
static int ssl_write_server_key_exchange( mbedtls_ssl_context *ssl )
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    size_t signature_len = 0;
#if defined(MBEDTLS_KEY_EXCHANGE_SOME_NON_PFS_ENABLED)
    const mbedtls_ssl_ciphersuite_t *ciphersuite_info =
                            ssl->handshake->ciphersuite_info;
#endif /* MBEDTLS_KEY_EXCHANGE_SOME_NON_PFS_ENABLED */

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "=> write server key exchange" ) );

#if defined(MBEDTLS_KEY_EXCHANGE_SOME_NON_PFS_ENABLED)
    /* Extract static ECDH parameters and abort if ServerKeyExchange
     * is not needed. */
    if( mbedtls_ssl_ciphersuite_no_pfs( ciphersuite_info ) )
    {
        /* For suites involving ECDH, extract DH parameters
         * from certificate at this point. */
#if defined(MBEDTLS_KEY_EXCHANGE_SOME_ECDH_ENABLED)
        if( mbedtls_ssl_ciphersuite_uses_ecdh( ciphersuite_info ) )
        {
            ssl_get_ecdh_params_from_cert( ssl );
        }
#endif /* MBEDTLS_KEY_EXCHANGE_SOME_ECDH_ENABLED */

        /* Key exchanges not involving ephemeral keys don't use
         * ServerKeyExchange, so end here. */
        MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= skip write server key exchange" ) );
        ssl->state++;
        return( 0 );
    }
#endif /* MBEDTLS_KEY_EXCHANGE_SOME_NON_PFS_ENABLED */

#if defined(MBEDTLS_KEY_EXCHANGE_WITH_SERVER_SIGNATURE_ENABLED) && \
    defined(MBEDTLS_SSL_ASYNC_PRIVATE)
    /* If we have already prepared the message and there is an ongoing
     * signature operation, resume signing. */
    if( ssl->handshake->async_in_progress != 0 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 2, ( "resuming signature operation" ) );
        ret = ssl_resume_server_key_exchange( ssl, &signature_len );
    }
    else
#endif /* defined(MBEDTLS_KEY_EXCHANGE_WITH_SERVER_SIGNATURE_ENABLED) &&
          defined(MBEDTLS_SSL_ASYNC_PRIVATE) */
    {
        /* ServerKeyExchange is needed. Prepare the message. */
        ret = ssl_prepare_server_key_exchange( ssl, &signature_len );
    }

    if( ret != 0 )
    {
        /* If we're starting to write a new message, set ssl->out_msglen
         * to 0. But if we're resuming after an asynchronous message,
         * out_msglen is the amount of data written so far and mst be
         * preserved. */
        if( ret == MBEDTLS_ERR_SSL_ASYNC_IN_PROGRESS )
            MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= write server key exchange (pending)" ) );
        else
            ssl->out_msglen = 0;
        return( ret );
    }

    /* If there is a signature, write its length.
     * ssl_prepare_server_key_exchange already wrote the signature
     * itself at its proper place in the output buffer. */
#if defined(MBEDTLS_KEY_EXCHANGE_WITH_SERVER_SIGNATURE_ENABLED)
    if( signature_len != 0 )
    {
        ssl->out_msg[ssl->out_msglen++] = MBEDTLS_BYTE_1( signature_len );
        ssl->out_msg[ssl->out_msglen++] = MBEDTLS_BYTE_0( signature_len );

        MBEDTLS_SSL_DEBUG_BUF( 3, "my signature",
                               ssl->out_msg + ssl->out_msglen,
                               signature_len );

        /* Skip over the already-written signature */
        ssl->out_msglen += signature_len;
    }
#endif /* MBEDTLS_KEY_EXCHANGE_WITH_SERVER_SIGNATURE_ENABLED */

    /* Add header and send. */
    ssl->out_msgtype = MBEDTLS_SSL_MSG_HANDSHAKE;
    ssl->out_msg[0]  = MBEDTLS_SSL_HS_SERVER_KEY_EXCHANGE;

    ssl->state++;

    if( ( ret = mbedtls_ssl_write_handshake_msg( ssl ) ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ssl_write_handshake_msg", ret );
        return( ret );
    }

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= write server key exchange" ) );
    return( 0 );
}

static int ssl_write_server_hello_done( mbedtls_ssl_context *ssl )
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "=> write server hello done" ) );

    ssl->out_msglen  = 4;
    ssl->out_msgtype = MBEDTLS_SSL_MSG_HANDSHAKE;
    ssl->out_msg[0]  = MBEDTLS_SSL_HS_SERVER_HELLO_DONE;

    ssl->state++;

#if defined(MBEDTLS_SSL_PROTO_DTLS)
    if( ssl->conf->transport == MBEDTLS_SSL_TRANSPORT_DATAGRAM )
        mbedtls_ssl_send_flight_completed( ssl );
#endif

    if( ( ret = mbedtls_ssl_write_handshake_msg( ssl ) ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ssl_write_handshake_msg", ret );
        return( ret );
    }

#if defined(MBEDTLS_SSL_PROTO_DTLS)
    if( ssl->conf->transport == MBEDTLS_SSL_TRANSPORT_DATAGRAM &&
        ( ret = mbedtls_ssl_flight_transmit( ssl ) ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ssl_flight_transmit", ret );
        return( ret );
    }
#endif /* MBEDTLS_SSL_PROTO_DTLS */

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= write server hello done" ) );

    return( 0 );
}

#if defined(MBEDTLS_KEY_EXCHANGE_DHE_RSA_ENABLED) ||                       \
    defined(MBEDTLS_KEY_EXCHANGE_DHE_PSK_ENABLED)
static int ssl_parse_client_dh_public( mbedtls_ssl_context *ssl, unsigned char **p,
                                       const unsigned char *end )
{
    int ret = MBEDTLS_ERR_SSL_FEATURE_UNAVAILABLE;
    size_t n;

    /*
     * Receive G^Y mod P, premaster = (G^Y)^X mod P
     */
    if( *p + 2 > end )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client key exchange message" ) );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_KEY_EXCHANGE );
    }

    n = ( (*p)[0] << 8 ) | (*p)[1];
    *p += 2;

    if( *p + n > end )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client key exchange message" ) );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_KEY_EXCHANGE );
    }

    if( ( ret = mbedtls_dhm_read_public( &ssl->handshake->dhm_ctx, *p, n ) ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_dhm_read_public", ret );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_KEY_EXCHANGE_RP );
    }

    *p += n;

    MBEDTLS_SSL_DEBUG_MPI( 3, "DHM: GY", &ssl->handshake->dhm_ctx.GY );

    return( ret );
}
#endif /* MBEDTLS_KEY_EXCHANGE_DHE_RSA_ENABLED ||
          MBEDTLS_KEY_EXCHANGE_DHE_PSK_ENABLED */

#if defined(MBEDTLS_KEY_EXCHANGE_RSA_ENABLED) ||                           \
    defined(MBEDTLS_KEY_EXCHANGE_RSA_PSK_ENABLED)

#if defined(MBEDTLS_SSL_ASYNC_PRIVATE)
static int ssl_resume_decrypt_pms( mbedtls_ssl_context *ssl,
                                   unsigned char *peer_pms,
                                   size_t *peer_pmslen,
                                   size_t peer_pmssize )
{
    int ret = ssl->conf->f_async_resume( ssl,
                                         peer_pms, peer_pmslen, peer_pmssize );
    if( ret != MBEDTLS_ERR_SSL_ASYNC_IN_PROGRESS )
    {
        ssl->handshake->async_in_progress = 0;
        mbedtls_ssl_set_async_operation_data( ssl, NULL );
    }
    MBEDTLS_SSL_DEBUG_RET( 2, "ssl_decrypt_encrypted_pms", ret );
    return( ret );
}
#endif /* MBEDTLS_SSL_ASYNC_PRIVATE */

static int ssl_decrypt_encrypted_pms( mbedtls_ssl_context *ssl,
                                      const unsigned char *p,
                                      const unsigned char *end,
                                      unsigned char *peer_pms,
                                      size_t *peer_pmslen,
                                      size_t peer_pmssize )
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    mbedtls_pk_context *private_key = mbedtls_ssl_own_key( ssl );
    mbedtls_pk_context *public_key = &mbedtls_ssl_own_cert( ssl )->pk;
    size_t len = mbedtls_pk_get_len( public_key );

#if defined(MBEDTLS_SSL_ASYNC_PRIVATE)
    /* If we have already started decoding the message and there is an ongoing
     * decryption operation, resume signing. */
    if( ssl->handshake->async_in_progress != 0 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 2, ( "resuming decryption operation" ) );
        return( ssl_resume_decrypt_pms( ssl,
                                        peer_pms, peer_pmslen, peer_pmssize ) );
    }
#endif /* MBEDTLS_SSL_ASYNC_PRIVATE */

    /*
     * Prepare to decrypt the premaster using own private RSA key
     */
#if defined(MBEDTLS_SSL_PROTO_TLS1) || defined(MBEDTLS_SSL_PROTO_TLS1_1) || \
    defined(MBEDTLS_SSL_PROTO_TLS1_2)
    if( ssl->minor_ver != MBEDTLS_SSL_MINOR_VERSION_0 )
    {
        if ( p + 2 > end )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client key exchange message" ) );
            return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_KEY_EXCHANGE );
        }
        if( *p++ != MBEDTLS_BYTE_1( len ) ||
            *p++ != MBEDTLS_BYTE_0( len ) )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client key exchange message" ) );
            return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_KEY_EXCHANGE );
        }
    }
#endif

    if( p + len != end )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client key exchange message" ) );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_KEY_EXCHANGE );
    }

    /*
     * Decrypt the premaster secret
     */
#if defined(MBEDTLS_SSL_ASYNC_PRIVATE)
    if( ssl->conf->f_async_decrypt_start != NULL )
    {
        ret = ssl->conf->f_async_decrypt_start( ssl,
                                                mbedtls_ssl_own_cert( ssl ),
                                                p, len );
        switch( ret )
        {
        case MBEDTLS_ERR_SSL_HW_ACCEL_FALLTHROUGH:
            /* act as if f_async_decrypt_start was null */
            break;
        case 0:
            ssl->handshake->async_in_progress = 1;
            return( ssl_resume_decrypt_pms( ssl,
                                            peer_pms,
                                            peer_pmslen,
                                            peer_pmssize ) );
        case MBEDTLS_ERR_SSL_ASYNC_IN_PROGRESS:
            ssl->handshake->async_in_progress = 1;
            return( MBEDTLS_ERR_SSL_ASYNC_IN_PROGRESS );
        default:
            MBEDTLS_SSL_DEBUG_RET( 1, "f_async_decrypt_start", ret );
            return( ret );
        }
    }
#endif /* MBEDTLS_SSL_ASYNC_PRIVATE */

    if( ! mbedtls_pk_can_do( private_key, MBEDTLS_PK_RSA ) )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "got no RSA private key" ) );
        return( MBEDTLS_ERR_SSL_PRIVATE_KEY_REQUIRED );
    }

    ret = mbedtls_pk_decrypt( private_key, p, len,
                              peer_pms, peer_pmslen, peer_pmssize,
                              ssl->conf->f_rng, ssl->conf->p_rng );
    return( ret );
}

static int ssl_parse_encrypted_pms( mbedtls_ssl_context *ssl,
                                    const unsigned char *p,
                                    const unsigned char *end,
                                    size_t pms_offset )
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    unsigned char *pms = ssl->handshake->premaster + pms_offset;
    unsigned char ver[2];
    unsigned char fake_pms[48], peer_pms[48];
    unsigned char mask;
    size_t i, peer_pmslen;
    unsigned int diff;

    /* In case of a failure in decryption, the decryption may write less than
     * 2 bytes of output, but we always read the first two bytes. It doesn't
     * matter in the end because diff will be nonzero in that case due to
     * ret being nonzero, and we only care whether diff is 0.
     * But do initialize peer_pms and peer_pmslen for robustness anyway. This
     * also makes memory analyzers happy (don't access uninitialized memory,
     * even if it's an unsigned char). */
    peer_pms[0] = peer_pms[1] = ~0;
    peer_pmslen = 0;

    ret = ssl_decrypt_encrypted_pms( ssl, p, end,
                                     peer_pms,
                                     &peer_pmslen,
                                     sizeof( peer_pms ) );

#if defined(MBEDTLS_SSL_ASYNC_PRIVATE)
    if ( ret == MBEDTLS_ERR_SSL_ASYNC_IN_PROGRESS )
        return( ret );
#endif /* MBEDTLS_SSL_ASYNC_PRIVATE */

    mbedtls_ssl_write_version( ssl->handshake->max_major_ver,
                               ssl->handshake->max_minor_ver,
                               ssl->conf->transport, ver );

    /* Avoid data-dependent branches while checking for invalid
     * padding, to protect against timing-based Bleichenbacher-type
     * attacks. */
    diff  = (unsigned int) ret;
    diff |= peer_pmslen ^ 48;
    diff |= peer_pms[0] ^ ver[0];
    diff |= peer_pms[1] ^ ver[1];

    /* mask = diff ? 0xff : 0x00 using bit operations to avoid branches */
    mask = mbedtls_ct_uint_mask( diff );

    /*
     * Protection against Bleichenbacher's attack: invalid PKCS#1 v1.5 padding
     * must not cause the connection to end immediately; instead, send a
     * bad_record_mac later in the handshake.
     * To protect against timing-based variants of the attack, we must
     * not have any branch that depends on whether the decryption was
     * successful. In particular, always generate the fake premaster secret,
     * regardless of whether it will ultimately influence the output or not.
     */
    ret = ssl->conf->f_rng( ssl->conf->p_rng, fake_pms, sizeof( fake_pms ) );
    if( ret != 0 )
    {
        /* It's ok to abort on an RNG failure, since this does not reveal
         * anything about the RSA decryption. */
        return( ret );
    }

#if defined(MBEDTLS_SSL_DEBUG_ALL)
    if( diff != 0 )
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client key exchange message" ) );
#endif

    if( sizeof( ssl->handshake->premaster ) < pms_offset ||
        sizeof( ssl->handshake->premaster ) - pms_offset < 48 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "should never happen" ) );
        return( MBEDTLS_ERR_SSL_INTERNAL_ERROR );
    }
    ssl->handshake->pmslen = 48;

    /* Set pms to either the true or the fake PMS, without
     * data-dependent branches. */
    for( i = 0; i < ssl->handshake->pmslen; i++ )
        pms[i] = ( mask & fake_pms[i] ) | ( (~mask) & peer_pms[i] );

    return( 0 );
}
#endif /* MBEDTLS_KEY_EXCHANGE_RSA_ENABLED ||
          MBEDTLS_KEY_EXCHANGE_RSA_PSK_ENABLED */

#if defined(MBEDTLS_KEY_EXCHANGE_SOME_PSK_ENABLED)
static int ssl_parse_client_psk_identity( mbedtls_ssl_context *ssl, unsigned char **p,
                                          const unsigned char *end )
{
    int ret = 0;
    uint16_t n;

    if( ssl_conf_has_psk_or_cb( ssl->conf ) == 0 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "got no pre-shared key" ) );
        return( MBEDTLS_ERR_SSL_PRIVATE_KEY_REQUIRED );
    }

    /*
     * Receive client pre-shared key identity name
     */
    if( end - *p < 2 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client key exchange message" ) );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_KEY_EXCHANGE );
    }

    n = ( (*p)[0] << 8 ) | (*p)[1];
    *p += 2;

    if( n == 0 || n > end - *p )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client key exchange message" ) );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_KEY_EXCHANGE );
    }

    if( ssl->conf->f_psk != NULL )
    {
        if( ssl->conf->f_psk( ssl->conf->p_psk, ssl, *p, n ) != 0 )
            ret = MBEDTLS_ERR_SSL_UNKNOWN_IDENTITY;
    }
    else
    {
        /* Identity is not a big secret since clients send it in the clear,
         * but treat it carefully anyway, just in case */
        if( n != ssl->conf->psk_identity_len ||
            mbedtls_ct_memcmp( ssl->conf->psk_identity, *p, n ) != 0 )
        {
            ret = MBEDTLS_ERR_SSL_UNKNOWN_IDENTITY;
        }
    }

    if( ret == MBEDTLS_ERR_SSL_UNKNOWN_IDENTITY )
    {
        MBEDTLS_SSL_DEBUG_BUF( 3, "Unknown PSK identity", *p, n );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_UNKNOWN_PSK_IDENTITY );
        return( MBEDTLS_ERR_SSL_UNKNOWN_IDENTITY );
    }

    *p += n;

    return( 0 );
}
#endif /* MBEDTLS_KEY_EXCHANGE_SOME_PSK_ENABLED */

static int ssl_parse_client_key_exchange( mbedtls_ssl_context *ssl )
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    const mbedtls_ssl_ciphersuite_t *ciphersuite_info;
    unsigned char *p, *end;

    ciphersuite_info = ssl->handshake->ciphersuite_info;

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "=> parse client key exchange" ) );

#if defined(MBEDTLS_SSL_ASYNC_PRIVATE) && \
    ( defined(MBEDTLS_KEY_EXCHANGE_RSA_ENABLED) || \
      defined(MBEDTLS_KEY_EXCHANGE_RSA_PSK_ENABLED) )
    if( ( ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_RSA_PSK ||
          ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_RSA ) &&
        ( ssl->handshake->async_in_progress != 0 ) )
    {
        /* We've already read a record and there is an asynchronous
         * operation in progress to decrypt it. So skip reading the
         * record. */
        MBEDTLS_SSL_DEBUG_MSG( 3, ( "will resume decryption of previously-read record" ) );
    }
    else
#endif
    if( ( ret = mbedtls_ssl_read_record( ssl, 1 ) ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ssl_read_record", ret );
        return( ret );
    }

    p = ssl->in_msg + mbedtls_ssl_hs_hdr_len( ssl );
    end = ssl->in_msg + ssl->in_hslen;

    if( ssl->in_msgtype != MBEDTLS_SSL_MSG_HANDSHAKE )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client key exchange message" ) );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_KEY_EXCHANGE );
    }

    if( ssl->in_msg[0] != MBEDTLS_SSL_HS_CLIENT_KEY_EXCHANGE )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client key exchange message" ) );
        return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_KEY_EXCHANGE );
    }

#if defined(MBEDTLS_KEY_EXCHANGE_DHE_RSA_ENABLED)
    if( ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_DHE_RSA )
    {
        if( ( ret = ssl_parse_client_dh_public( ssl, &p, end ) ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, ( "ssl_parse_client_dh_public" ), ret );
            return( ret );
        }

        if( p != end )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client key exchange" ) );
            return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_KEY_EXCHANGE );
        }

        if( ( ret = mbedtls_dhm_calc_secret( &ssl->handshake->dhm_ctx,
                                      ssl->handshake->premaster,
                                      MBEDTLS_PREMASTER_SIZE,
                                     &ssl->handshake->pmslen,
                                      ssl->conf->f_rng, ssl->conf->p_rng ) ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_dhm_calc_secret", ret );
            return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_KEY_EXCHANGE_CS );
        }

        MBEDTLS_SSL_DEBUG_MPI( 3, "DHM: K ", &ssl->handshake->dhm_ctx.K  );
    }
    else
#endif /* MBEDTLS_KEY_EXCHANGE_DHE_RSA_ENABLED */
#if defined(MBEDTLS_KEY_EXCHANGE_ECDHE_RSA_ENABLED) ||                     \
    defined(MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA_ENABLED) ||                   \
    defined(MBEDTLS_KEY_EXCHANGE_ECDH_RSA_ENABLED) ||                      \
    defined(MBEDTLS_KEY_EXCHANGE_ECDH_ECDSA_ENABLED)
    if( ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_ECDHE_RSA ||
        ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA ||
        ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_ECDH_RSA ||
        ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_ECDH_ECDSA )
    {
        if( ( ret = mbedtls_ecdh_read_public( &ssl->handshake->ecdh_ctx,
                                      p, end - p) ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ecdh_read_public", ret );
            return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_KEY_EXCHANGE_RP );
        }

        MBEDTLS_SSL_DEBUG_ECDH( 3, &ssl->handshake->ecdh_ctx,
                                MBEDTLS_DEBUG_ECDH_QP );

        if( ( ret = mbedtls_ecdh_calc_secret( &ssl->handshake->ecdh_ctx,
                                      &ssl->handshake->pmslen,
                                       ssl->handshake->premaster,
                                       MBEDTLS_MPI_MAX_SIZE,
                                       ssl->conf->f_rng, ssl->conf->p_rng ) ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ecdh_calc_secret", ret );
            return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_KEY_EXCHANGE_CS );
        }

        MBEDTLS_SSL_DEBUG_ECDH( 3, &ssl->handshake->ecdh_ctx,
                                MBEDTLS_DEBUG_ECDH_Z );
    }
    else
#endif /* MBEDTLS_KEY_EXCHANGE_ECDHE_RSA_ENABLED ||
          MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA_ENABLED ||
          MBEDTLS_KEY_EXCHANGE_ECDH_RSA_ENABLED ||
          MBEDTLS_KEY_EXCHANGE_ECDH_ECDSA_ENABLED */
#if defined(MBEDTLS_KEY_EXCHANGE_PSK_ENABLED)
    if( ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_PSK )
    {
        if( ( ret = ssl_parse_client_psk_identity( ssl, &p, end ) ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, ( "ssl_parse_client_psk_identity" ), ret );
            return( ret );
        }

        if( p != end )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client key exchange" ) );
            return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_KEY_EXCHANGE );
        }

#if defined(MBEDTLS_USE_PSA_CRYPTO)
        /* For opaque PSKs, we perform the PSK-to-MS derivation atomatically
         * and skip the intermediate PMS. */
        if( ssl_use_opaque_psk( ssl ) == 1 )
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "skip PMS generation for opaque PSK" ) );
        else
#endif /* MBEDTLS_USE_PSA_CRYPTO */
        if( ( ret = mbedtls_ssl_psk_derive_premaster( ssl,
                        ciphersuite_info->key_exchange ) ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ssl_psk_derive_premaster", ret );
            return( ret );
        }
    }
    else
#endif /* MBEDTLS_KEY_EXCHANGE_PSK_ENABLED */
#if defined(MBEDTLS_KEY_EXCHANGE_RSA_PSK_ENABLED)
    if( ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_RSA_PSK )
    {
#if defined(MBEDTLS_SSL_ASYNC_PRIVATE)
        if ( ssl->handshake->async_in_progress != 0 )
        {
            /* There is an asynchronous operation in progress to
             * decrypt the encrypted premaster secret, so skip
             * directly to resuming this operation. */
            MBEDTLS_SSL_DEBUG_MSG( 3, ( "PSK identity already parsed" ) );
            /* Update p to skip the PSK identity. ssl_parse_encrypted_pms
             * won't actually use it, but maintain p anyway for robustness. */
            p += ssl->conf->psk_identity_len + 2;
        }
        else
#endif /* MBEDTLS_SSL_ASYNC_PRIVATE */
        if( ( ret = ssl_parse_client_psk_identity( ssl, &p, end ) ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, ( "ssl_parse_client_psk_identity" ), ret );
            return( ret );
        }

#if defined(MBEDTLS_USE_PSA_CRYPTO)
        /* Opaque PSKs are currently only supported for PSK-only. */
        if( ssl_use_opaque_psk( ssl ) == 1 )
            return( MBEDTLS_ERR_SSL_FEATURE_UNAVAILABLE );
#endif

        if( ( ret = ssl_parse_encrypted_pms( ssl, p, end, 2 ) ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, ( "ssl_parse_encrypted_pms" ), ret );
            return( ret );
        }

        if( ( ret = mbedtls_ssl_psk_derive_premaster( ssl,
                        ciphersuite_info->key_exchange ) ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ssl_psk_derive_premaster", ret );
            return( ret );
        }
    }
    else
#endif /* MBEDTLS_KEY_EXCHANGE_RSA_PSK_ENABLED */
#if defined(MBEDTLS_KEY_EXCHANGE_DHE_PSK_ENABLED)
    if( ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_DHE_PSK )
    {
        if( ( ret = ssl_parse_client_psk_identity( ssl, &p, end ) ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, ( "ssl_parse_client_psk_identity" ), ret );
            return( ret );
        }
        if( ( ret = ssl_parse_client_dh_public( ssl, &p, end ) ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, ( "ssl_parse_client_dh_public" ), ret );
            return( ret );
        }

#if defined(MBEDTLS_USE_PSA_CRYPTO)
        /* Opaque PSKs are currently only supported for PSK-only. */
        if( ssl_use_opaque_psk( ssl ) == 1 )
            return( MBEDTLS_ERR_SSL_FEATURE_UNAVAILABLE );
#endif

        if( p != end )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad client key exchange" ) );
            return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_KEY_EXCHANGE );
        }

        if( ( ret = mbedtls_ssl_psk_derive_premaster( ssl,
                        ciphersuite_info->key_exchange ) ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ssl_psk_derive_premaster", ret );
            return( ret );
        }
    }
    else
#endif /* MBEDTLS_KEY_EXCHANGE_DHE_PSK_ENABLED */
#if defined(MBEDTLS_KEY_EXCHANGE_ECDHE_PSK_ENABLED)
    if( ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_ECDHE_PSK )
    {
        if( ( ret = ssl_parse_client_psk_identity( ssl, &p, end ) ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, ( "ssl_parse_client_psk_identity" ), ret );
            return( ret );
        }

        if( ( ret = mbedtls_ecdh_read_public( &ssl->handshake->ecdh_ctx,
                                       p, end - p ) ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ecdh_read_public", ret );
            return( MBEDTLS_ERR_SSL_BAD_HS_CLIENT_KEY_EXCHANGE_RP );
        }

#if defined(MBEDTLS_USE_PSA_CRYPTO)
        /* Opaque PSKs are currently only supported for PSK-only. */
        if( ssl_use_opaque_psk( ssl ) == 1 )
            return( MBEDTLS_ERR_SSL_FEATURE_UNAVAILABLE );
#endif

        MBEDTLS_SSL_DEBUG_ECDH( 3, &ssl->handshake->ecdh_ctx,
                                MBEDTLS_DEBUG_ECDH_QP );

        if( ( ret = mbedtls_ssl_psk_derive_premaster( ssl,
                        ciphersuite_info->key_exchange ) ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ssl_psk_derive_premaster", ret );
            return( ret );
        }
    }
    else
#endif /* MBEDTLS_KEY_EXCHANGE_ECDHE_PSK_ENABLED */
#if defined(MBEDTLS_KEY_EXCHANGE_RSA_ENABLED)
    if( ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_RSA )
    {
        if( ( ret = ssl_parse_encrypted_pms( ssl, p, end, 0 ) ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, ( "ssl_parse_parse_encrypted_pms_secret" ), ret );
            return( ret );
        }
    }
    else
#endif /* MBEDTLS_KEY_EXCHANGE_RSA_ENABLED */
#if defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
    if( ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_ECJPAKE )
    {
        ret = mbedtls_ecjpake_read_round_two( &ssl->handshake->ecjpake_ctx,
                                              p, end - p );
        if( ret != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ecjpake_read_round_two", ret );
            return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_KEY_EXCHANGE );
        }

        ret = mbedtls_ecjpake_derive_secret( &ssl->handshake->ecjpake_ctx,
                ssl->handshake->premaster, 32, &ssl->handshake->pmslen,
                ssl->conf->f_rng, ssl->conf->p_rng );
        if( ret != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ecjpake_derive_secret", ret );
            return( ret );
        }
    }
    else
#endif /* MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED */
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "should never happen" ) );
        return( MBEDTLS_ERR_SSL_INTERNAL_ERROR );
    }

    if( ( ret = mbedtls_ssl_derive_keys( ssl ) ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ssl_derive_keys", ret );
        return( ret );
    }

    ssl->state++;

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= parse client key exchange" ) );

    return( 0 );
}

#if !defined(MBEDTLS_KEY_EXCHANGE_CERT_REQ_ALLOWED_ENABLED)
static int ssl_parse_certificate_verify( mbedtls_ssl_context *ssl )
{
    const mbedtls_ssl_ciphersuite_t *ciphersuite_info =
        ssl->handshake->ciphersuite_info;

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "=> parse certificate verify" ) );

    if( !mbedtls_ssl_ciphersuite_cert_req_allowed( ciphersuite_info ) )
    {
        MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= skip parse certificate verify" ) );
        ssl->state++;
        return( 0 );
    }

    MBEDTLS_SSL_DEBUG_MSG( 1, ( "should never happen" ) );
    return( MBEDTLS_ERR_SSL_INTERNAL_ERROR );
}
#else /* !MBEDTLS_KEY_EXCHANGE_CERT_REQ_ALLOWED_ENABLED */
static int ssl_parse_certificate_verify( mbedtls_ssl_context *ssl )
{
    int ret = MBEDTLS_ERR_SSL_FEATURE_UNAVAILABLE;
    size_t i, sig_len;
    unsigned char hash[48];
    unsigned char *hash_start = hash;
    size_t hashlen;
#if defined(MBEDTLS_SSL_PROTO_TLS1_2)
    mbedtls_pk_type_t pk_alg;
#endif
    mbedtls_md_type_t md_alg;
    const mbedtls_ssl_ciphersuite_t *ciphersuite_info =
        ssl->handshake->ciphersuite_info;
    mbedtls_pk_context * peer_pk;

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "=> parse certificate verify" ) );

    if( !mbedtls_ssl_ciphersuite_cert_req_allowed( ciphersuite_info ) )
    {
        MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= skip parse certificate verify" ) );
        ssl->state++;
        return( 0 );
    }

#if defined(MBEDTLS_SSL_KEEP_PEER_CERTIFICATE)
    if( ssl->session_negotiate->peer_cert == NULL )
    {
        MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= skip parse certificate verify" ) );
        ssl->state++;
        return( 0 );
    }
#else /* MBEDTLS_SSL_KEEP_PEER_CERTIFICATE */
    if( ssl->session_negotiate->peer_cert_digest == NULL )
    {
        MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= skip parse certificate verify" ) );
        ssl->state++;
        return( 0 );
    }
#endif /* !MBEDTLS_SSL_KEEP_PEER_CERTIFICATE */

    /* Read the message without adding it to the checksum */
    ret = mbedtls_ssl_read_record( ssl, 0 /* no checksum update */ );
    if( 0 != ret )
    {
        MBEDTLS_SSL_DEBUG_RET( 1, ( "mbedtls_ssl_read_record" ), ret );
        return( ret );
    }

    ssl->state++;

    /* Process the message contents */
    if( ssl->in_msgtype != MBEDTLS_SSL_MSG_HANDSHAKE ||
        ssl->in_msg[0] != MBEDTLS_SSL_HS_CERTIFICATE_VERIFY )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad certificate verify message" ) );
        return( MBEDTLS_ERR_SSL_BAD_HS_CERTIFICATE_VERIFY );
    }

    i = mbedtls_ssl_hs_hdr_len( ssl );

#if !defined(MBEDTLS_SSL_KEEP_PEER_CERTIFICATE)
    peer_pk = &ssl->handshake->peer_pubkey;
#else /* !MBEDTLS_SSL_KEEP_PEER_CERTIFICATE */
    if( ssl->session_negotiate->peer_cert == NULL )
    {
        /* Should never happen */
        return( MBEDTLS_ERR_SSL_INTERNAL_ERROR );
    }
    peer_pk = &ssl->session_negotiate->peer_cert->pk;
#endif /* MBEDTLS_SSL_KEEP_PEER_CERTIFICATE */

    /*
     *  struct {
     *     SignatureAndHashAlgorithm algorithm; -- TLS 1.2 only
     *     opaque signature<0..2^16-1>;
     *  } DigitallySigned;
     */
#if defined(MBEDTLS_SSL_PROTO_SSL3) || defined(MBEDTLS_SSL_PROTO_TLS1) || \
    defined(MBEDTLS_SSL_PROTO_TLS1_1)
    if( ssl->minor_ver != MBEDTLS_SSL_MINOR_VERSION_3 )
    {
        md_alg = MBEDTLS_MD_NONE;
        hashlen = 36;

        /* For ECDSA, use SHA-1, not MD-5 + SHA-1 */
        if( mbedtls_pk_can_do( peer_pk, MBEDTLS_PK_ECDSA ) )
        {
            hash_start += 16;
            hashlen -= 16;
            md_alg = MBEDTLS_MD_SHA1;
        }
    }
    else
#endif /* MBEDTLS_SSL_PROTO_SSL3 || MBEDTLS_SSL_PROTO_TLS1 ||
          MBEDTLS_SSL_PROTO_TLS1_1 */
#if defined(MBEDTLS_SSL_PROTO_TLS1_2)
    if( ssl->minor_ver == MBEDTLS_SSL_MINOR_VERSION_3 )
    {
        if( i + 2 > ssl->in_hslen )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad certificate verify message" ) );
            return( MBEDTLS_ERR_SSL_BAD_HS_CERTIFICATE_VERIFY );
        }

        /*
         * Hash
         */
        md_alg = mbedtls_ssl_md_alg_from_hash( ssl->in_msg[i] );

        if( md_alg == MBEDTLS_MD_NONE || mbedtls_ssl_set_calc_verify_md( ssl, ssl->in_msg[i] ) )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "peer not adhering to requested sig_alg"
                                " for verify message" ) );
            return( MBEDTLS_ERR_SSL_BAD_HS_CERTIFICATE_VERIFY );
        }

#if !defined(MBEDTLS_MD_SHA1)
        if( MBEDTLS_MD_SHA1 == md_alg )
            hash_start += 16;
#endif

        /* Info from md_alg will be used instead */
        hashlen = 0;

        i++;

        /*
         * Signature
         */
        if( ( pk_alg = mbedtls_ssl_pk_alg_from_sig( ssl->in_msg[i] ) )
                        == MBEDTLS_PK_NONE )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "peer not adhering to requested sig_alg"
                                " for verify message" ) );
            return( MBEDTLS_ERR_SSL_BAD_HS_CERTIFICATE_VERIFY );
        }

        /*
         * Check the certificate's key type matches the signature alg
         */
        if( !mbedtls_pk_can_do( peer_pk, pk_alg ) )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "sig_alg doesn't match cert key" ) );
            return( MBEDTLS_ERR_SSL_BAD_HS_CERTIFICATE_VERIFY );
        }

        i++;
    }
    else
#endif /* MBEDTLS_SSL_PROTO_TLS1_2 */
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "should never happen" ) );
        return( MBEDTLS_ERR_SSL_INTERNAL_ERROR );
    }

    if( i + 2 > ssl->in_hslen )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad certificate verify message" ) );
        return( MBEDTLS_ERR_SSL_BAD_HS_CERTIFICATE_VERIFY );
    }

    sig_len = ( ssl->in_msg[i] << 8 ) | ssl->in_msg[i+1];
    i += 2;

    if( i + sig_len != ssl->in_hslen )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad certificate verify message" ) );
        return( MBEDTLS_ERR_SSL_BAD_HS_CERTIFICATE_VERIFY );
    }

    /* Calculate hash and verify signature */
    {
        size_t dummy_hlen;
        ssl->handshake->calc_verify( ssl, hash, &dummy_hlen );
    }

    if( ( ret = mbedtls_pk_verify( peer_pk,
                           md_alg, hash_start, hashlen,
                           ssl->in_msg + i, sig_len ) ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_pk_verify", ret );
        return( ret );
    }

    mbedtls_ssl_update_handshake_status( ssl );

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= parse certificate verify" ) );

    return( ret );
}
#endif /* MBEDTLS_KEY_EXCHANGE_CERT_REQ_ALLOWED_ENABLED */

#if defined(MBEDTLS_SSL_SESSION_TICKETS)
static int ssl_write_new_session_ticket( mbedtls_ssl_context *ssl )
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    size_t tlen;
    uint32_t lifetime;

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "=> write new session ticket" ) );

    ssl->out_msgtype = MBEDTLS_SSL_MSG_HANDSHAKE;
    ssl->out_msg[0]  = MBEDTLS_SSL_HS_NEW_SESSION_TICKET;

    /*
     * struct {
     *     uint32 ticket_lifetime_hint;
     *     opaque ticket<0..2^16-1>;
     * } NewSessionTicket;
     *
     * 4  .  7   ticket_lifetime_hint (0 = unspecified)
     * 8  .  9   ticket_len (n)
     * 10 .  9+n ticket content
     */

    if( ( ret = ssl->conf->f_ticket_write( ssl->conf->p_ticket,
                                ssl->session_negotiate,
                                ssl->out_msg + 10,
                                ssl->out_msg + MBEDTLS_SSL_OUT_CONTENT_LEN,
                                &tlen, &lifetime ) ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ssl_ticket_write", ret );
        tlen = 0;
    }

    MBEDTLS_PUT_UINT32_BE( lifetime, ssl->out_msg, 4 );
    MBEDTLS_PUT_UINT16_BE( tlen, ssl->out_msg, 8 );
    ssl->out_msglen = 10 + tlen;

    /*
     * Morally equivalent to updating ssl->state, but NewSessionTicket and
     * ChangeCipherSpec share the same state.
     */
    ssl->handshake->new_session_ticket = 0;

    if( ( ret = mbedtls_ssl_write_handshake_msg( ssl ) ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ssl_write_handshake_msg", ret );
        return( ret );
    }

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= write new session ticket" ) );

    return( 0 );
}
#endif /* MBEDTLS_SSL_SESSION_TICKETS */

/*
 * SSL handshake -- server side -- single step
 */
int mbedtls_ssl_handshake_server_step( mbedtls_ssl_context *ssl )
{
    int ret = 0;

    if( ssl->state == MBEDTLS_SSL_HANDSHAKE_OVER || ssl->handshake == NULL )
        return( MBEDTLS_ERR_SSL_BAD_INPUT_DATA );

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "server state: %d", ssl->state ) );

    if( ( ret = mbedtls_ssl_flush_output( ssl ) ) != 0 )
        return( ret );

#if defined(MBEDTLS_SSL_PROTO_DTLS)
    if( ssl->conf->transport == MBEDTLS_SSL_TRANSPORT_DATAGRAM &&
        ssl->handshake->retransmit_state == MBEDTLS_SSL_RETRANS_SENDING )
    {
        if( ( ret = mbedtls_ssl_flight_transmit( ssl ) ) != 0 )
            return( ret );
    }
#endif /* MBEDTLS_SSL_PROTO_DTLS */

    switch( ssl->state )
    {
        case MBEDTLS_SSL_HELLO_REQUEST:
            ssl->state = MBEDTLS_SSL_CLIENT_HELLO;
            break;

        /*
         *  <==   ClientHello
         */
        case MBEDTLS_SSL_CLIENT_HELLO:
            ret = ssl_parse_client_hello( ssl );
            break;

#if defined(MBEDTLS_SSL_PROTO_DTLS)
        case MBEDTLS_SSL_SERVER_HELLO_VERIFY_REQUEST_SENT:
            return( MBEDTLS_ERR_SSL_HELLO_VERIFY_REQUIRED );
#endif

        /*
         *  ==>   ServerHello
         *        Certificate
         *      ( ServerKeyExchange  )
         *      ( CertificateRequest )
         *        ServerHelloDone
         */
        case MBEDTLS_SSL_SERVER_HELLO:
            ret = ssl_write_server_hello( ssl );
            break;

        case MBEDTLS_SSL_SERVER_CERTIFICATE:
            ret = mbedtls_ssl_write_certificate( ssl );
            break;

        case MBEDTLS_SSL_SERVER_KEY_EXCHANGE:
            ret = ssl_write_server_key_exchange( ssl );
            break;

        case MBEDTLS_SSL_CERTIFICATE_REQUEST:
            ret = ssl_write_certificate_request( ssl );
            break;

        case MBEDTLS_SSL_SERVER_HELLO_DONE:
            ret = ssl_write_server_hello_done( ssl );
            break;

        /*
         *  <== ( Certificate/Alert  )
         *        ClientKeyExchange
         *      ( CertificateVerify  )
         *        ChangeCipherSpec
         *        Finished
         */
        case MBEDTLS_SSL_CLIENT_CERTIFICATE:
            ret = mbedtls_ssl_parse_certificate( ssl );
            break;

        case MBEDTLS_SSL_CLIENT_KEY_EXCHANGE:
            ret = ssl_parse_client_key_exchange( ssl );
            break;

        case MBEDTLS_SSL_CERTIFICATE_VERIFY:
            ret = ssl_parse_certificate_verify( ssl );
            break;

        case MBEDTLS_SSL_CLIENT_CHANGE_CIPHER_SPEC:
            ret = mbedtls_ssl_parse_change_cipher_spec( ssl );
            break;

        case MBEDTLS_SSL_CLIENT_FINISHED:
            ret = mbedtls_ssl_parse_finished( ssl );
            break;

        /*
         *  ==> ( NewSessionTicket )
         *        ChangeCipherSpec
         *        Finished
         */
        case MBEDTLS_SSL_SERVER_CHANGE_CIPHER_SPEC:
#if defined(MBEDTLS_SSL_SESSION_TICKETS)
            if( ssl->handshake->new_session_ticket != 0 )
                ret = ssl_write_new_session_ticket( ssl );
            else
#endif
                ret = mbedtls_ssl_write_change_cipher_spec( ssl );
            break;

        case MBEDTLS_SSL_SERVER_FINISHED:
            ret = mbedtls_ssl_write_finished( ssl );
            break;

        case MBEDTLS_SSL_FLUSH_BUFFERS:
            MBEDTLS_SSL_DEBUG_MSG( 2, ( "handshake: done" ) );
            ssl->state = MBEDTLS_SSL_HANDSHAKE_WRAPUP;
            break;

        case MBEDTLS_SSL_HANDSHAKE_WRAPUP:
            mbedtls_ssl_handshake_wrapup( ssl );
            break;

        default:
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "invalid state %d", ssl->state ) );
            return( MBEDTLS_ERR_SSL_BAD_INPUT_DATA );
    }

    return( ret );
}
#endif /* MBEDTLS_SSL_SRV_C */
