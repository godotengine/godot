/*
 *  SSLv3/TLSv1 client-side functions
 *
 *  Copyright (C) 2006-2015, ARM Limited, All Rights Reserved
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
 *
 *  This file is part of mbed TLS (https://tls.mbed.org)
 */

#if !defined(MBEDTLS_CONFIG_FILE)
#include "mbedtls/config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#if defined(MBEDTLS_SSL_CLI_C)

#if defined(MBEDTLS_PLATFORM_C)
#include "mbedtls/platform.h"
#else
#include <stdlib.h>
#define mbedtls_calloc    calloc
#define mbedtls_free      free
#endif

#include "mbedtls/debug.h"
#include "mbedtls/ssl.h"
#include "mbedtls/ssl_internal.h"

#include <string.h>

#include <stdint.h>

#if defined(MBEDTLS_HAVE_TIME)
#include "mbedtls/platform_time.h"
#endif

#if defined(MBEDTLS_SSL_SESSION_TICKETS)
#include "mbedtls/platform_util.h"
#endif

#if defined(MBEDTLS_SSL_SERVER_NAME_INDICATION)
static void ssl_write_hostname_ext( mbedtls_ssl_context *ssl,
                                    unsigned char *buf,
                                    size_t *olen )
{
    unsigned char *p = buf;
    const unsigned char *end = ssl->out_msg + MBEDTLS_SSL_OUT_CONTENT_LEN;
    size_t hostname_len;

    *olen = 0;

    if( ssl->hostname == NULL )
        return;

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello, adding server name extension: %s",
                   ssl->hostname ) );

    hostname_len = strlen( ssl->hostname );

    if( end < p || (size_t)( end - p ) < hostname_len + 9 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "buffer too small" ) );
        return;
    }

    /*
     * Sect. 3, RFC 6066 (TLS Extensions Definitions)
     *
     * In order to provide any of the server names, clients MAY include an
     * extension of type "server_name" in the (extended) client hello. The
     * "extension_data" field of this extension SHALL contain
     * "ServerNameList" where:
     *
     * struct {
     *     NameType name_type;
     *     select (name_type) {
     *         case host_name: HostName;
     *     } name;
     * } ServerName;
     *
     * enum {
     *     host_name(0), (255)
     * } NameType;
     *
     * opaque HostName<1..2^16-1>;
     *
     * struct {
     *     ServerName server_name_list<1..2^16-1>
     * } ServerNameList;
     *
     */
    *p++ = (unsigned char)( ( MBEDTLS_TLS_EXT_SERVERNAME >> 8 ) & 0xFF );
    *p++ = (unsigned char)( ( MBEDTLS_TLS_EXT_SERVERNAME      ) & 0xFF );

    *p++ = (unsigned char)( ( (hostname_len + 5) >> 8 ) & 0xFF );
    *p++ = (unsigned char)( ( (hostname_len + 5)      ) & 0xFF );

    *p++ = (unsigned char)( ( (hostname_len + 3) >> 8 ) & 0xFF );
    *p++ = (unsigned char)( ( (hostname_len + 3)      ) & 0xFF );

    *p++ = (unsigned char)( ( MBEDTLS_TLS_EXT_SERVERNAME_HOSTNAME ) & 0xFF );
    *p++ = (unsigned char)( ( hostname_len >> 8 ) & 0xFF );
    *p++ = (unsigned char)( ( hostname_len      ) & 0xFF );

    memcpy( p, ssl->hostname, hostname_len );

    *olen = hostname_len + 9;
}
#endif /* MBEDTLS_SSL_SERVER_NAME_INDICATION */

#if defined(MBEDTLS_SSL_RENEGOTIATION)
static void ssl_write_renegotiation_ext( mbedtls_ssl_context *ssl,
                                         unsigned char *buf,
                                         size_t *olen )
{
    unsigned char *p = buf;
    const unsigned char *end = ssl->out_msg + MBEDTLS_SSL_OUT_CONTENT_LEN;

    *olen = 0;

    /* We're always including an TLS_EMPTY_RENEGOTIATION_INFO_SCSV in the
     * initial ClientHello, in which case also adding the renegotiation
     * info extension is NOT RECOMMENDED as per RFC 5746 Section 3.4. */
    if( ssl->renego_status != MBEDTLS_SSL_RENEGOTIATION_IN_PROGRESS )
        return;

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello, adding renegotiation extension" ) );

    if( end < p || (size_t)( end - p ) < 5 + ssl->verify_data_len )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "buffer too small" ) );
        return;
    }

    /*
     * Secure renegotiation
     */
    *p++ = (unsigned char)( ( MBEDTLS_TLS_EXT_RENEGOTIATION_INFO >> 8 ) & 0xFF );
    *p++ = (unsigned char)( ( MBEDTLS_TLS_EXT_RENEGOTIATION_INFO      ) & 0xFF );

    *p++ = 0x00;
    *p++ = ( ssl->verify_data_len + 1 ) & 0xFF;
    *p++ = ssl->verify_data_len & 0xFF;

    memcpy( p, ssl->own_verify_data, ssl->verify_data_len );

    *olen = 5 + ssl->verify_data_len;
}
#endif /* MBEDTLS_SSL_RENEGOTIATION */

/*
 * Only if we handle at least one key exchange that needs signatures.
 */
#if defined(MBEDTLS_SSL_PROTO_TLS1_2) && \
    defined(MBEDTLS_KEY_EXCHANGE__WITH_CERT__ENABLED)
static void ssl_write_signature_algorithms_ext( mbedtls_ssl_context *ssl,
                                                unsigned char *buf,
                                                size_t *olen )
{
    unsigned char *p = buf;
    const unsigned char *end = ssl->out_msg + MBEDTLS_SSL_OUT_CONTENT_LEN;
    size_t sig_alg_len = 0;
    const int *md;
#if defined(MBEDTLS_RSA_C) || defined(MBEDTLS_ECDSA_C)
    unsigned char *sig_alg_list = buf + 6;
#endif

    *olen = 0;

    if( ssl->conf->max_minor_ver != MBEDTLS_SSL_MINOR_VERSION_3 )
        return;

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello, adding signature_algorithms extension" ) );

    for( md = ssl->conf->sig_hashes; *md != MBEDTLS_MD_NONE; md++ )
    {
#if defined(MBEDTLS_ECDSA_C)
        sig_alg_len += 2;
#endif
#if defined(MBEDTLS_RSA_C)
        sig_alg_len += 2;
#endif
    }

    if( end < p || (size_t)( end - p ) < sig_alg_len + 6 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "buffer too small" ) );
        return;
    }

    /*
     * Prepare signature_algorithms extension (TLS 1.2)
     */
    sig_alg_len = 0;

    for( md = ssl->conf->sig_hashes; *md != MBEDTLS_MD_NONE; md++ )
    {
#if defined(MBEDTLS_ECDSA_C)
        sig_alg_list[sig_alg_len++] = mbedtls_ssl_hash_from_md_alg( *md );
        sig_alg_list[sig_alg_len++] = MBEDTLS_SSL_SIG_ECDSA;
#endif
#if defined(MBEDTLS_RSA_C)
        sig_alg_list[sig_alg_len++] = mbedtls_ssl_hash_from_md_alg( *md );
        sig_alg_list[sig_alg_len++] = MBEDTLS_SSL_SIG_RSA;
#endif
    }

    /*
     * enum {
     *     none(0), md5(1), sha1(2), sha224(3), sha256(4), sha384(5),
     *     sha512(6), (255)
     * } HashAlgorithm;
     *
     * enum { anonymous(0), rsa(1), dsa(2), ecdsa(3), (255) }
     *   SignatureAlgorithm;
     *
     * struct {
     *     HashAlgorithm hash;
     *     SignatureAlgorithm signature;
     * } SignatureAndHashAlgorithm;
     *
     * SignatureAndHashAlgorithm
     *   supported_signature_algorithms<2..2^16-2>;
     */
    *p++ = (unsigned char)( ( MBEDTLS_TLS_EXT_SIG_ALG >> 8 ) & 0xFF );
    *p++ = (unsigned char)( ( MBEDTLS_TLS_EXT_SIG_ALG      ) & 0xFF );

    *p++ = (unsigned char)( ( ( sig_alg_len + 2 ) >> 8 ) & 0xFF );
    *p++ = (unsigned char)( ( ( sig_alg_len + 2 )      ) & 0xFF );

    *p++ = (unsigned char)( ( sig_alg_len >> 8 ) & 0xFF );
    *p++ = (unsigned char)( ( sig_alg_len      ) & 0xFF );

    *olen = 6 + sig_alg_len;
}
#endif /* MBEDTLS_SSL_PROTO_TLS1_2 &&
          MBEDTLS_KEY_EXCHANGE__WITH_CERT__ENABLED */

#if defined(MBEDTLS_ECDH_C) || defined(MBEDTLS_ECDSA_C) || \
    defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
static void ssl_write_supported_elliptic_curves_ext( mbedtls_ssl_context *ssl,
                                                     unsigned char *buf,
                                                     size_t *olen )
{
    unsigned char *p = buf;
    const unsigned char *end = ssl->out_msg + MBEDTLS_SSL_OUT_CONTENT_LEN;
    unsigned char *elliptic_curve_list = p + 6;
    size_t elliptic_curve_len = 0;
    const mbedtls_ecp_curve_info *info;
#if defined(MBEDTLS_ECP_C)
    const mbedtls_ecp_group_id *grp_id;
#else
    ((void) ssl);
#endif

    *olen = 0;

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello, adding supported_elliptic_curves extension" ) );

#if defined(MBEDTLS_ECP_C)
    for( grp_id = ssl->conf->curve_list; *grp_id != MBEDTLS_ECP_DP_NONE; grp_id++ )
#else
    for( info = mbedtls_ecp_curve_list(); info->grp_id != MBEDTLS_ECP_DP_NONE; info++ )
#endif
    {
#if defined(MBEDTLS_ECP_C)
        info = mbedtls_ecp_curve_info_from_grp_id( *grp_id );
#endif
        if( info == NULL )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "invalid curve in ssl configuration" ) );
            return;
        }

        elliptic_curve_len += 2;
    }

    if( end < p || (size_t)( end - p ) < 6 + elliptic_curve_len )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "buffer too small" ) );
        return;
    }

    elliptic_curve_len = 0;

#if defined(MBEDTLS_ECP_C)
    for( grp_id = ssl->conf->curve_list; *grp_id != MBEDTLS_ECP_DP_NONE; grp_id++ )
#else
    for( info = mbedtls_ecp_curve_list(); info->grp_id != MBEDTLS_ECP_DP_NONE; info++ )
#endif
    {
#if defined(MBEDTLS_ECP_C)
        info = mbedtls_ecp_curve_info_from_grp_id( *grp_id );
#endif
        elliptic_curve_list[elliptic_curve_len++] = info->tls_id >> 8;
        elliptic_curve_list[elliptic_curve_len++] = info->tls_id & 0xFF;
    }

    if( elliptic_curve_len == 0 )
        return;

    *p++ = (unsigned char)( ( MBEDTLS_TLS_EXT_SUPPORTED_ELLIPTIC_CURVES >> 8 ) & 0xFF );
    *p++ = (unsigned char)( ( MBEDTLS_TLS_EXT_SUPPORTED_ELLIPTIC_CURVES      ) & 0xFF );

    *p++ = (unsigned char)( ( ( elliptic_curve_len + 2 ) >> 8 ) & 0xFF );
    *p++ = (unsigned char)( ( ( elliptic_curve_len + 2 )      ) & 0xFF );

    *p++ = (unsigned char)( ( ( elliptic_curve_len     ) >> 8 ) & 0xFF );
    *p++ = (unsigned char)( ( ( elliptic_curve_len     )      ) & 0xFF );

    *olen = 6 + elliptic_curve_len;
}

static void ssl_write_supported_point_formats_ext( mbedtls_ssl_context *ssl,
                                                   unsigned char *buf,
                                                   size_t *olen )
{
    unsigned char *p = buf;
    const unsigned char *end = ssl->out_msg + MBEDTLS_SSL_OUT_CONTENT_LEN;

    *olen = 0;

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello, adding supported_point_formats extension" ) );

    if( end < p || (size_t)( end - p ) < 6 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "buffer too small" ) );
        return;
    }

    *p++ = (unsigned char)( ( MBEDTLS_TLS_EXT_SUPPORTED_POINT_FORMATS >> 8 ) & 0xFF );
    *p++ = (unsigned char)( ( MBEDTLS_TLS_EXT_SUPPORTED_POINT_FORMATS      ) & 0xFF );

    *p++ = 0x00;
    *p++ = 2;

    *p++ = 1;
    *p++ = MBEDTLS_ECP_PF_UNCOMPRESSED;

    *olen = 6;
}
#endif /* MBEDTLS_ECDH_C || MBEDTLS_ECDSA_C ||
          MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED */

#if defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
static void ssl_write_ecjpake_kkpp_ext( mbedtls_ssl_context *ssl,
                                        unsigned char *buf,
                                        size_t *olen )
{
    int ret;
    unsigned char *p = buf;
    const unsigned char *end = ssl->out_msg + MBEDTLS_SSL_OUT_CONTENT_LEN;
    size_t kkpp_len;

    *olen = 0;

    /* Skip costly extension if we can't use EC J-PAKE anyway */
    if( mbedtls_ecjpake_check( &ssl->handshake->ecjpake_ctx ) != 0 )
        return;

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello, adding ecjpake_kkpp extension" ) );

    if( end - p < 4 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "buffer too small" ) );
        return;
    }

    *p++ = (unsigned char)( ( MBEDTLS_TLS_EXT_ECJPAKE_KKPP >> 8 ) & 0xFF );
    *p++ = (unsigned char)( ( MBEDTLS_TLS_EXT_ECJPAKE_KKPP      ) & 0xFF );

    /*
     * We may need to send ClientHello multiple times for Hello verification.
     * We don't want to compute fresh values every time (both for performance
     * and consistency reasons), so cache the extension content.
     */
    if( ssl->handshake->ecjpake_cache == NULL ||
        ssl->handshake->ecjpake_cache_len == 0 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 3, ( "generating new ecjpake parameters" ) );

        ret = mbedtls_ecjpake_write_round_one( &ssl->handshake->ecjpake_ctx,
                                        p + 2, end - p - 2, &kkpp_len,
                                        ssl->conf->f_rng, ssl->conf->p_rng );
        if( ret != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1 , "mbedtls_ecjpake_write_round_one", ret );
            return;
        }

        ssl->handshake->ecjpake_cache = mbedtls_calloc( 1, kkpp_len );
        if( ssl->handshake->ecjpake_cache == NULL )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "allocation failed" ) );
            return;
        }

        memcpy( ssl->handshake->ecjpake_cache, p + 2, kkpp_len );
        ssl->handshake->ecjpake_cache_len = kkpp_len;
    }
    else
    {
        MBEDTLS_SSL_DEBUG_MSG( 3, ( "re-using cached ecjpake parameters" ) );

        kkpp_len = ssl->handshake->ecjpake_cache_len;

        if( (size_t)( end - p - 2 ) < kkpp_len )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "buffer too small" ) );
            return;
        }

        memcpy( p + 2, ssl->handshake->ecjpake_cache, kkpp_len );
    }

    *p++ = (unsigned char)( ( kkpp_len >> 8 ) & 0xFF );
    *p++ = (unsigned char)( ( kkpp_len      ) & 0xFF );

    *olen = kkpp_len + 4;
}
#endif /* MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED */

#if defined(MBEDTLS_SSL_MAX_FRAGMENT_LENGTH)
static void ssl_write_max_fragment_length_ext( mbedtls_ssl_context *ssl,
                                               unsigned char *buf,
                                               size_t *olen )
{
    unsigned char *p = buf;
    const unsigned char *end = ssl->out_msg + MBEDTLS_SSL_OUT_CONTENT_LEN;

    *olen = 0;

    if( ssl->conf->mfl_code == MBEDTLS_SSL_MAX_FRAG_LEN_NONE ) {
        return;
    }

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello, adding max_fragment_length extension" ) );

    if( end < p || (size_t)( end - p ) < 5 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "buffer too small" ) );
        return;
    }

    *p++ = (unsigned char)( ( MBEDTLS_TLS_EXT_MAX_FRAGMENT_LENGTH >> 8 ) & 0xFF );
    *p++ = (unsigned char)( ( MBEDTLS_TLS_EXT_MAX_FRAGMENT_LENGTH      ) & 0xFF );

    *p++ = 0x00;
    *p++ = 1;

    *p++ = ssl->conf->mfl_code;

    *olen = 5;
}
#endif /* MBEDTLS_SSL_MAX_FRAGMENT_LENGTH */

#if defined(MBEDTLS_SSL_TRUNCATED_HMAC)
static void ssl_write_truncated_hmac_ext( mbedtls_ssl_context *ssl,
                                          unsigned char *buf, size_t *olen )
{
    unsigned char *p = buf;
    const unsigned char *end = ssl->out_msg + MBEDTLS_SSL_OUT_CONTENT_LEN;

    *olen = 0;

    if( ssl->conf->trunc_hmac == MBEDTLS_SSL_TRUNC_HMAC_DISABLED )
    {
        return;
    }

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello, adding truncated_hmac extension" ) );

    if( end < p || (size_t)( end - p ) < 4 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "buffer too small" ) );
        return;
    }

    *p++ = (unsigned char)( ( MBEDTLS_TLS_EXT_TRUNCATED_HMAC >> 8 ) & 0xFF );
    *p++ = (unsigned char)( ( MBEDTLS_TLS_EXT_TRUNCATED_HMAC      ) & 0xFF );

    *p++ = 0x00;
    *p++ = 0x00;

    *olen = 4;
}
#endif /* MBEDTLS_SSL_TRUNCATED_HMAC */

#if defined(MBEDTLS_SSL_ENCRYPT_THEN_MAC)
static void ssl_write_encrypt_then_mac_ext( mbedtls_ssl_context *ssl,
                                       unsigned char *buf, size_t *olen )
{
    unsigned char *p = buf;
    const unsigned char *end = ssl->out_msg + MBEDTLS_SSL_OUT_CONTENT_LEN;

    *olen = 0;

    if( ssl->conf->encrypt_then_mac == MBEDTLS_SSL_ETM_DISABLED ||
        ssl->conf->max_minor_ver == MBEDTLS_SSL_MINOR_VERSION_0 )
    {
        return;
    }

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello, adding encrypt_then_mac "
                        "extension" ) );

    if( end < p || (size_t)( end - p ) < 4 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "buffer too small" ) );
        return;
    }

    *p++ = (unsigned char)( ( MBEDTLS_TLS_EXT_ENCRYPT_THEN_MAC >> 8 ) & 0xFF );
    *p++ = (unsigned char)( ( MBEDTLS_TLS_EXT_ENCRYPT_THEN_MAC      ) & 0xFF );

    *p++ = 0x00;
    *p++ = 0x00;

    *olen = 4;
}
#endif /* MBEDTLS_SSL_ENCRYPT_THEN_MAC */

#if defined(MBEDTLS_SSL_EXTENDED_MASTER_SECRET)
static void ssl_write_extended_ms_ext( mbedtls_ssl_context *ssl,
                                       unsigned char *buf, size_t *olen )
{
    unsigned char *p = buf;
    const unsigned char *end = ssl->out_msg + MBEDTLS_SSL_OUT_CONTENT_LEN;

    *olen = 0;

    if( ssl->conf->extended_ms == MBEDTLS_SSL_EXTENDED_MS_DISABLED ||
        ssl->conf->max_minor_ver == MBEDTLS_SSL_MINOR_VERSION_0 )
    {
        return;
    }

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello, adding extended_master_secret "
                        "extension" ) );

    if( end < p || (size_t)( end - p ) < 4 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "buffer too small" ) );
        return;
    }

    *p++ = (unsigned char)( ( MBEDTLS_TLS_EXT_EXTENDED_MASTER_SECRET >> 8 ) & 0xFF );
    *p++ = (unsigned char)( ( MBEDTLS_TLS_EXT_EXTENDED_MASTER_SECRET      ) & 0xFF );

    *p++ = 0x00;
    *p++ = 0x00;

    *olen = 4;
}
#endif /* MBEDTLS_SSL_EXTENDED_MASTER_SECRET */

#if defined(MBEDTLS_SSL_SESSION_TICKETS)
static void ssl_write_session_ticket_ext( mbedtls_ssl_context *ssl,
                                          unsigned char *buf, size_t *olen )
{
    unsigned char *p = buf;
    const unsigned char *end = ssl->out_msg + MBEDTLS_SSL_OUT_CONTENT_LEN;
    size_t tlen = ssl->session_negotiate->ticket_len;

    *olen = 0;

    if( ssl->conf->session_tickets == MBEDTLS_SSL_SESSION_TICKETS_DISABLED )
    {
        return;
    }

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello, adding session ticket extension" ) );

    if( end < p || (size_t)( end - p ) < 4 + tlen )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "buffer too small" ) );
        return;
    }

    *p++ = (unsigned char)( ( MBEDTLS_TLS_EXT_SESSION_TICKET >> 8 ) & 0xFF );
    *p++ = (unsigned char)( ( MBEDTLS_TLS_EXT_SESSION_TICKET      ) & 0xFF );

    *p++ = (unsigned char)( ( tlen >> 8 ) & 0xFF );
    *p++ = (unsigned char)( ( tlen      ) & 0xFF );

    *olen = 4;

    if( ssl->session_negotiate->ticket == NULL || tlen == 0 )
    {
        return;
    }

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "sending session ticket of length %d", tlen ) );

    memcpy( p, ssl->session_negotiate->ticket, tlen );

    *olen += tlen;
}
#endif /* MBEDTLS_SSL_SESSION_TICKETS */

#if defined(MBEDTLS_SSL_ALPN)
static void ssl_write_alpn_ext( mbedtls_ssl_context *ssl,
                                unsigned char *buf, size_t *olen )
{
    unsigned char *p = buf;
    const unsigned char *end = ssl->out_msg + MBEDTLS_SSL_OUT_CONTENT_LEN;
    size_t alpnlen = 0;
    const char **cur;

    *olen = 0;

    if( ssl->conf->alpn_list == NULL )
    {
        return;
    }

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello, adding alpn extension" ) );

    for( cur = ssl->conf->alpn_list; *cur != NULL; cur++ )
        alpnlen += (unsigned char)( strlen( *cur ) & 0xFF ) + 1;

    if( end < p || (size_t)( end - p ) < 6 + alpnlen )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "buffer too small" ) );
        return;
    }

    *p++ = (unsigned char)( ( MBEDTLS_TLS_EXT_ALPN >> 8 ) & 0xFF );
    *p++ = (unsigned char)( ( MBEDTLS_TLS_EXT_ALPN      ) & 0xFF );

    /*
     * opaque ProtocolName<1..2^8-1>;
     *
     * struct {
     *     ProtocolName protocol_name_list<2..2^16-1>
     * } ProtocolNameList;
     */

    /* Skip writing extension and list length for now */
    p += 4;

    for( cur = ssl->conf->alpn_list; *cur != NULL; cur++ )
    {
        *p = (unsigned char)( strlen( *cur ) & 0xFF );
        memcpy( p + 1, *cur, *p );
        p += 1 + *p;
    }

    *olen = p - buf;

    /* List length = olen - 2 (ext_type) - 2 (ext_len) - 2 (list_len) */
    buf[4] = (unsigned char)( ( ( *olen - 6 ) >> 8 ) & 0xFF );
    buf[5] = (unsigned char)( ( ( *olen - 6 )      ) & 0xFF );

    /* Extension length = olen - 2 (ext_type) - 2 (ext_len) */
    buf[2] = (unsigned char)( ( ( *olen - 4 ) >> 8 ) & 0xFF );
    buf[3] = (unsigned char)( ( ( *olen - 4 )      ) & 0xFF );
}
#endif /* MBEDTLS_SSL_ALPN */

/*
 * Generate random bytes for ClientHello
 */
static int ssl_generate_random( mbedtls_ssl_context *ssl )
{
    int ret;
    unsigned char *p = ssl->handshake->randbytes;
#if defined(MBEDTLS_HAVE_TIME)
    mbedtls_time_t t;
#endif

    /*
     * When responding to a verify request, MUST reuse random (RFC 6347 4.2.1)
     */
#if defined(MBEDTLS_SSL_PROTO_DTLS)
    if( ssl->conf->transport == MBEDTLS_SSL_TRANSPORT_DATAGRAM &&
        ssl->handshake->verify_cookie != NULL )
    {
        return( 0 );
    }
#endif

#if defined(MBEDTLS_HAVE_TIME)
    t = mbedtls_time( NULL );
    *p++ = (unsigned char)( t >> 24 );
    *p++ = (unsigned char)( t >> 16 );
    *p++ = (unsigned char)( t >>  8 );
    *p++ = (unsigned char)( t       );

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello, current time: %lu", t ) );
#else
    if( ( ret = ssl->conf->f_rng( ssl->conf->p_rng, p, 4 ) ) != 0 )
        return( ret );

    p += 4;
#endif /* MBEDTLS_HAVE_TIME */

    if( ( ret = ssl->conf->f_rng( ssl->conf->p_rng, p, 28 ) ) != 0 )
        return( ret );

    return( 0 );
}

/**
 * \brief           Validate cipher suite against config in SSL context.
 *
 * \param suite_info    cipher suite to validate
 * \param ssl           SSL context
 * \param min_minor_ver Minimal minor version to accept a cipher suite
 * \param max_minor_ver Maximal minor version to accept a cipher suite
 *
 * \return          0 if valid, else 1
 */
static int ssl_validate_ciphersuite( const mbedtls_ssl_ciphersuite_t * suite_info,
                                     const mbedtls_ssl_context * ssl,
                                     int min_minor_ver, int max_minor_ver )
{
    (void) ssl;
    if( suite_info == NULL )
        return( 1 );

    if( suite_info->min_minor_ver > max_minor_ver ||
            suite_info->max_minor_ver < min_minor_ver )
        return( 1 );

#if defined(MBEDTLS_SSL_PROTO_DTLS)
    if( ssl->conf->transport == MBEDTLS_SSL_TRANSPORT_DATAGRAM &&
            ( suite_info->flags & MBEDTLS_CIPHERSUITE_NODTLS ) )
        return( 1 );
#endif

#if defined(MBEDTLS_ARC4_C)
    if( ssl->conf->arc4_disabled == MBEDTLS_SSL_ARC4_DISABLED &&
            suite_info->cipher == MBEDTLS_CIPHER_ARC4_128 )
        return( 1 );
#endif

#if defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
    if( suite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_ECJPAKE &&
            mbedtls_ecjpake_check( &ssl->handshake->ecjpake_ctx ) != 0 )
        return( 1 );
#endif

    return( 0 );
}

static int ssl_write_client_hello( mbedtls_ssl_context *ssl )
{
    int ret;
    size_t i, n, olen, ext_len = 0;
    unsigned char *buf;
    unsigned char *p, *q;
    unsigned char offer_compress;
    const int *ciphersuites;
    const mbedtls_ssl_ciphersuite_t *ciphersuite_info;
#if defined(MBEDTLS_ECDH_C) || defined(MBEDTLS_ECDSA_C) || \
    defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
    int uses_ec = 0;
#endif

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "=> write client hello" ) );

    if( ssl->conf->f_rng == NULL )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "no RNG provided") );
        return( MBEDTLS_ERR_SSL_NO_RNG );
    }

#if defined(MBEDTLS_SSL_RENEGOTIATION)
    if( ssl->renego_status == MBEDTLS_SSL_INITIAL_HANDSHAKE )
#endif
    {
        ssl->major_ver = ssl->conf->min_major_ver;
        ssl->minor_ver = ssl->conf->min_minor_ver;
    }

    if( ssl->conf->max_major_ver == 0 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "configured max major version is invalid, "
                            "consider using mbedtls_ssl_config_defaults()" ) );
        return( MBEDTLS_ERR_SSL_BAD_INPUT_DATA );
    }

    /*
     *     0  .   0   handshake type
     *     1  .   3   handshake length
     *     4  .   5   highest version supported
     *     6  .   9   current UNIX time
     *    10  .  37   random bytes
     */
    buf = ssl->out_msg;
    p = buf + 4;

    mbedtls_ssl_write_version( ssl->conf->max_major_ver, ssl->conf->max_minor_ver,
                       ssl->conf->transport, p );
    p += 2;

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello, max version: [%d:%d]",
                   buf[4], buf[5] ) );

    if( ( ret = ssl_generate_random( ssl ) ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_RET( 1, "ssl_generate_random", ret );
        return( ret );
    }

    memcpy( p, ssl->handshake->randbytes, 32 );
    MBEDTLS_SSL_DEBUG_BUF( 3, "client hello, random bytes", p, 32 );
    p += 32;

    /*
     *    38  .  38   session id length
     *    39  . 39+n  session id
     *   39+n . 39+n  DTLS only: cookie length (1 byte)
     *   40+n .  ..   DTSL only: cookie
     *   ..   . ..    ciphersuitelist length (2 bytes)
     *   ..   . ..    ciphersuitelist
     *   ..   . ..    compression methods length (1 byte)
     *   ..   . ..    compression methods
     *   ..   . ..    extensions length (2 bytes)
     *   ..   . ..    extensions
     */
    n = ssl->session_negotiate->id_len;

    if( n < 16 || n > 32 ||
#if defined(MBEDTLS_SSL_RENEGOTIATION)
        ssl->renego_status != MBEDTLS_SSL_INITIAL_HANDSHAKE ||
#endif
        ssl->handshake->resume == 0 )
    {
        n = 0;
    }

#if defined(MBEDTLS_SSL_SESSION_TICKETS)
    /*
     * RFC 5077 section 3.4: "When presenting a ticket, the client MAY
     * generate and include a Session ID in the TLS ClientHello."
     */
#if defined(MBEDTLS_SSL_RENEGOTIATION)
    if( ssl->renego_status == MBEDTLS_SSL_INITIAL_HANDSHAKE )
#endif
    {
        if( ssl->session_negotiate->ticket != NULL &&
                ssl->session_negotiate->ticket_len != 0 )
        {
            ret = ssl->conf->f_rng( ssl->conf->p_rng, ssl->session_negotiate->id, 32 );

            if( ret != 0 )
                return( ret );

            ssl->session_negotiate->id_len = n = 32;
        }
    }
#endif /* MBEDTLS_SSL_SESSION_TICKETS */

    *p++ = (unsigned char) n;

    for( i = 0; i < n; i++ )
        *p++ = ssl->session_negotiate->id[i];

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello, session id len.: %d", n ) );
    MBEDTLS_SSL_DEBUG_BUF( 3,   "client hello, session id", buf + 39, n );

    /*
     * DTLS cookie
     */
#if defined(MBEDTLS_SSL_PROTO_DTLS)
    if( ssl->conf->transport == MBEDTLS_SSL_TRANSPORT_DATAGRAM )
    {
        if( ssl->handshake->verify_cookie == NULL )
        {
            MBEDTLS_SSL_DEBUG_MSG( 3, ( "no verify cookie to send" ) );
            *p++ = 0;
        }
        else
        {
            MBEDTLS_SSL_DEBUG_BUF( 3, "client hello, cookie",
                              ssl->handshake->verify_cookie,
                              ssl->handshake->verify_cookie_len );

            *p++ = ssl->handshake->verify_cookie_len;
            memcpy( p, ssl->handshake->verify_cookie,
                       ssl->handshake->verify_cookie_len );
            p += ssl->handshake->verify_cookie_len;
        }
    }
#endif

    /*
     * Ciphersuite list
     */
    ciphersuites = ssl->conf->ciphersuite_list[ssl->minor_ver];

    /* Skip writing ciphersuite length for now */
    n = 0;
    q = p;
    p += 2;

    for( i = 0; ciphersuites[i] != 0; i++ )
    {
        ciphersuite_info = mbedtls_ssl_ciphersuite_from_id( ciphersuites[i] );

        if( ssl_validate_ciphersuite( ciphersuite_info, ssl,
                                      ssl->conf->min_minor_ver,
                                      ssl->conf->max_minor_ver ) != 0 )
            continue;

        MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello, add ciphersuite: %04x",
                                    ciphersuites[i] ) );

#if defined(MBEDTLS_ECDH_C) || defined(MBEDTLS_ECDSA_C) || \
    defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
        uses_ec |= mbedtls_ssl_ciphersuite_uses_ec( ciphersuite_info );
#endif

        n++;
        *p++ = (unsigned char)( ciphersuites[i] >> 8 );
        *p++ = (unsigned char)( ciphersuites[i]      );
    }

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello, got %d ciphersuites (excluding SCSVs)", n ) );

    /*
     * Add TLS_EMPTY_RENEGOTIATION_INFO_SCSV
     */
#if defined(MBEDTLS_SSL_RENEGOTIATION)
    if( ssl->renego_status == MBEDTLS_SSL_INITIAL_HANDSHAKE )
#endif
    {
        MBEDTLS_SSL_DEBUG_MSG( 3, ( "adding EMPTY_RENEGOTIATION_INFO_SCSV" ) );
        *p++ = (unsigned char)( MBEDTLS_SSL_EMPTY_RENEGOTIATION_INFO >> 8 );
        *p++ = (unsigned char)( MBEDTLS_SSL_EMPTY_RENEGOTIATION_INFO      );
        n++;
    }

    /* Some versions of OpenSSL don't handle it correctly if not at end */
#if defined(MBEDTLS_SSL_FALLBACK_SCSV)
    if( ssl->conf->fallback == MBEDTLS_SSL_IS_FALLBACK )
    {
        MBEDTLS_SSL_DEBUG_MSG( 3, ( "adding FALLBACK_SCSV" ) );
        *p++ = (unsigned char)( MBEDTLS_SSL_FALLBACK_SCSV_VALUE >> 8 );
        *p++ = (unsigned char)( MBEDTLS_SSL_FALLBACK_SCSV_VALUE      );
        n++;
    }
#endif

    *q++ = (unsigned char)( n >> 7 );
    *q++ = (unsigned char)( n << 1 );

#if defined(MBEDTLS_ZLIB_SUPPORT)
    offer_compress = 1;
#else
    offer_compress = 0;
#endif

    /*
     * We don't support compression with DTLS right now: if many records come
     * in the same datagram, uncompressing one could overwrite the next one.
     * We don't want to add complexity for handling that case unless there is
     * an actual need for it.
     */
#if defined(MBEDTLS_SSL_PROTO_DTLS)
    if( ssl->conf->transport == MBEDTLS_SSL_TRANSPORT_DATAGRAM )
        offer_compress = 0;
#endif

    if( offer_compress )
    {
        MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello, compress len.: %d", 2 ) );
        MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello, compress alg.: %d %d",
                            MBEDTLS_SSL_COMPRESS_DEFLATE, MBEDTLS_SSL_COMPRESS_NULL ) );

        *p++ = 2;
        *p++ = MBEDTLS_SSL_COMPRESS_DEFLATE;
        *p++ = MBEDTLS_SSL_COMPRESS_NULL;
    }
    else
    {
        MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello, compress len.: %d", 1 ) );
        MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello, compress alg.: %d",
                            MBEDTLS_SSL_COMPRESS_NULL ) );

        *p++ = 1;
        *p++ = MBEDTLS_SSL_COMPRESS_NULL;
    }

    // First write extensions, then the total length
    //
#if defined(MBEDTLS_SSL_SERVER_NAME_INDICATION)
    ssl_write_hostname_ext( ssl, p + 2 + ext_len, &olen );
    ext_len += olen;
#endif

    /* Note that TLS_EMPTY_RENEGOTIATION_INFO_SCSV is always added
     * even if MBEDTLS_SSL_RENEGOTIATION is not defined. */
#if defined(MBEDTLS_SSL_RENEGOTIATION)
    ssl_write_renegotiation_ext( ssl, p + 2 + ext_len, &olen );
    ext_len += olen;
#endif

#if defined(MBEDTLS_SSL_PROTO_TLS1_2) && \
    defined(MBEDTLS_KEY_EXCHANGE__WITH_CERT__ENABLED)
    ssl_write_signature_algorithms_ext( ssl, p + 2 + ext_len, &olen );
    ext_len += olen;
#endif

#if defined(MBEDTLS_ECDH_C) || defined(MBEDTLS_ECDSA_C) || \
    defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
    if( uses_ec )
    {
        ssl_write_supported_elliptic_curves_ext( ssl, p + 2 + ext_len, &olen );
        ext_len += olen;

        ssl_write_supported_point_formats_ext( ssl, p + 2 + ext_len, &olen );
        ext_len += olen;
    }
#endif

#if defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
    ssl_write_ecjpake_kkpp_ext( ssl, p + 2 + ext_len, &olen );
    ext_len += olen;
#endif

#if defined(MBEDTLS_SSL_MAX_FRAGMENT_LENGTH)
    ssl_write_max_fragment_length_ext( ssl, p + 2 + ext_len, &olen );
    ext_len += olen;
#endif

#if defined(MBEDTLS_SSL_TRUNCATED_HMAC)
    ssl_write_truncated_hmac_ext( ssl, p + 2 + ext_len, &olen );
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

#if defined(MBEDTLS_SSL_ALPN)
    ssl_write_alpn_ext( ssl, p + 2 + ext_len, &olen );
    ext_len += olen;
#endif

#if defined(MBEDTLS_SSL_SESSION_TICKETS)
    ssl_write_session_ticket_ext( ssl, p + 2 + ext_len, &olen );
    ext_len += olen;
#endif

    /* olen unused if all extensions are disabled */
    ((void) olen);

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "client hello, total extension length: %d",
                   ext_len ) );

    if( ext_len > 0 )
    {
        *p++ = (unsigned char)( ( ext_len >> 8 ) & 0xFF );
        *p++ = (unsigned char)( ( ext_len      ) & 0xFF );
        p += ext_len;
    }

    ssl->out_msglen  = p - buf;
    ssl->out_msgtype = MBEDTLS_SSL_MSG_HANDSHAKE;
    ssl->out_msg[0]  = MBEDTLS_SSL_HS_CLIENT_HELLO;

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

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= write client hello" ) );

    return( 0 );
}

static int ssl_parse_renegotiation_info( mbedtls_ssl_context *ssl,
                                         const unsigned char *buf,
                                         size_t len )
{
#if defined(MBEDTLS_SSL_RENEGOTIATION)
    if( ssl->renego_status != MBEDTLS_SSL_INITIAL_HANDSHAKE )
    {
        /* Check verify-data in constant-time. The length OTOH is no secret */
        if( len    != 1 + ssl->verify_data_len * 2 ||
            buf[0] !=     ssl->verify_data_len * 2 ||
            mbedtls_ssl_safer_memcmp( buf + 1,
                          ssl->own_verify_data, ssl->verify_data_len ) != 0 ||
            mbedtls_ssl_safer_memcmp( buf + 1 + ssl->verify_data_len,
                          ssl->peer_verify_data, ssl->verify_data_len ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "non-matching renegotiation info" ) );
            mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                            MBEDTLS_SSL_ALERT_MSG_HANDSHAKE_FAILURE );
            return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_HELLO );
        }
    }
    else
#endif /* MBEDTLS_SSL_RENEGOTIATION */
    {
        if( len != 1 || buf[0] != 0x00 )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "non-zero length renegotiation info" ) );
            mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                            MBEDTLS_SSL_ALERT_MSG_HANDSHAKE_FAILURE );
            return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_HELLO );
        }

        ssl->secure_renegotiation = MBEDTLS_SSL_SECURE_RENEGOTIATION;
    }

    return( 0 );
}

#if defined(MBEDTLS_SSL_MAX_FRAGMENT_LENGTH)
static int ssl_parse_max_fragment_length_ext( mbedtls_ssl_context *ssl,
                                              const unsigned char *buf,
                                              size_t len )
{
    /*
     * server should use the extension only if we did,
     * and if so the server's value should match ours (and len is always 1)
     */
    if( ssl->conf->mfl_code == MBEDTLS_SSL_MAX_FRAG_LEN_NONE ||
        len != 1 ||
        buf[0] != ssl->conf->mfl_code )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "non-matching max fragment length extension" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_HANDSHAKE_FAILURE );
        return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_HELLO );
    }

    return( 0 );
}
#endif /* MBEDTLS_SSL_MAX_FRAGMENT_LENGTH */

#if defined(MBEDTLS_SSL_TRUNCATED_HMAC)
static int ssl_parse_truncated_hmac_ext( mbedtls_ssl_context *ssl,
                                         const unsigned char *buf,
                                         size_t len )
{
    if( ssl->conf->trunc_hmac == MBEDTLS_SSL_TRUNC_HMAC_DISABLED ||
        len != 0 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "non-matching truncated HMAC extension" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_HANDSHAKE_FAILURE );
        return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_HELLO );
    }

    ((void) buf);

    ssl->session_negotiate->trunc_hmac = MBEDTLS_SSL_TRUNC_HMAC_ENABLED;

    return( 0 );
}
#endif /* MBEDTLS_SSL_TRUNCATED_HMAC */

#if defined(MBEDTLS_SSL_ENCRYPT_THEN_MAC)
static int ssl_parse_encrypt_then_mac_ext( mbedtls_ssl_context *ssl,
                                         const unsigned char *buf,
                                         size_t len )
{
    if( ssl->conf->encrypt_then_mac == MBEDTLS_SSL_ETM_DISABLED ||
        ssl->minor_ver == MBEDTLS_SSL_MINOR_VERSION_0 ||
        len != 0 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "non-matching encrypt-then-MAC extension" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_HANDSHAKE_FAILURE );
        return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_HELLO );
    }

    ((void) buf);

    ssl->session_negotiate->encrypt_then_mac = MBEDTLS_SSL_ETM_ENABLED;

    return( 0 );
}
#endif /* MBEDTLS_SSL_ENCRYPT_THEN_MAC */

#if defined(MBEDTLS_SSL_EXTENDED_MASTER_SECRET)
static int ssl_parse_extended_ms_ext( mbedtls_ssl_context *ssl,
                                         const unsigned char *buf,
                                         size_t len )
{
    if( ssl->conf->extended_ms == MBEDTLS_SSL_EXTENDED_MS_DISABLED ||
        ssl->minor_ver == MBEDTLS_SSL_MINOR_VERSION_0 ||
        len != 0 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "non-matching extended master secret extension" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_HANDSHAKE_FAILURE );
        return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_HELLO );
    }

    ((void) buf);

    ssl->handshake->extended_ms = MBEDTLS_SSL_EXTENDED_MS_ENABLED;

    return( 0 );
}
#endif /* MBEDTLS_SSL_EXTENDED_MASTER_SECRET */

#if defined(MBEDTLS_SSL_SESSION_TICKETS)
static int ssl_parse_session_ticket_ext( mbedtls_ssl_context *ssl,
                                         const unsigned char *buf,
                                         size_t len )
{
    if( ssl->conf->session_tickets == MBEDTLS_SSL_SESSION_TICKETS_DISABLED ||
        len != 0 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "non-matching session ticket extension" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_HANDSHAKE_FAILURE );
        return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_HELLO );
    }

    ((void) buf);

    ssl->handshake->new_session_ticket = 1;

    return( 0 );
}
#endif /* MBEDTLS_SSL_SESSION_TICKETS */

#if defined(MBEDTLS_ECDH_C) || defined(MBEDTLS_ECDSA_C) || \
    defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
static int ssl_parse_supported_point_formats_ext( mbedtls_ssl_context *ssl,
                                                  const unsigned char *buf,
                                                  size_t len )
{
    size_t list_size;
    const unsigned char *p;

    if( len == 0 || (size_t)( buf[0] + 1 ) != len )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad server hello message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
        return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_HELLO );
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

    MBEDTLS_SSL_DEBUG_MSG( 1, ( "no point format in common" ) );
    mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                    MBEDTLS_SSL_ALERT_MSG_HANDSHAKE_FAILURE );
    return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_HELLO );
}
#endif /* MBEDTLS_ECDH_C || MBEDTLS_ECDSA_C ||
          MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED */

#if defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
static int ssl_parse_ecjpake_kkpp( mbedtls_ssl_context *ssl,
                                   const unsigned char *buf,
                                   size_t len )
{
    int ret;

    if( ssl->transform_negotiate->ciphersuite_info->key_exchange !=
        MBEDTLS_KEY_EXCHANGE_ECJPAKE )
    {
        MBEDTLS_SSL_DEBUG_MSG( 3, ( "skip ecjpake kkpp extension" ) );
        return( 0 );
    }

    /* If we got here, we no longer need our cached extension */
    mbedtls_free( ssl->handshake->ecjpake_cache );
    ssl->handshake->ecjpake_cache = NULL;
    ssl->handshake->ecjpake_cache_len = 0;

    if( ( ret = mbedtls_ecjpake_read_round_one( &ssl->handshake->ecjpake_ctx,
                                                buf, len ) ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ecjpake_read_round_one", ret );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_HANDSHAKE_FAILURE );
        return( ret );
    }

    return( 0 );
}
#endif /* MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED */

#if defined(MBEDTLS_SSL_ALPN)
static int ssl_parse_alpn_ext( mbedtls_ssl_context *ssl,
                               const unsigned char *buf, size_t len )
{
    size_t list_len, name_len;
    const char **p;

    /* If we didn't send it, the server shouldn't send it */
    if( ssl->conf->alpn_list == NULL )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "non-matching ALPN extension" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_HANDSHAKE_FAILURE );
        return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_HELLO );
    }

    /*
     * opaque ProtocolName<1..2^8-1>;
     *
     * struct {
     *     ProtocolName protocol_name_list<2..2^16-1>
     * } ProtocolNameList;
     *
     * the "ProtocolNameList" MUST contain exactly one "ProtocolName"
     */

    /* Min length is 2 (list_len) + 1 (name_len) + 1 (name) */
    if( len < 4 )
    {
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
        return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_HELLO );
    }

    list_len = ( buf[0] << 8 ) | buf[1];
    if( list_len != len - 2 )
    {
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
        return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_HELLO );
    }

    name_len = buf[2];
    if( name_len != list_len - 1 )
    {
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
        return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_HELLO );
    }

    /* Check that the server chosen protocol was in our list and save it */
    for( p = ssl->conf->alpn_list; *p != NULL; p++ )
    {
        if( name_len == strlen( *p ) &&
            memcmp( buf + 3, *p, name_len ) == 0 )
        {
            ssl->alpn_chosen = *p;
            return( 0 );
        }
    }

    MBEDTLS_SSL_DEBUG_MSG( 1, ( "ALPN extension: no matching protocol" ) );
    mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                    MBEDTLS_SSL_ALERT_MSG_HANDSHAKE_FAILURE );
    return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_HELLO );
}
#endif /* MBEDTLS_SSL_ALPN */

/*
 * Parse HelloVerifyRequest.  Only called after verifying the HS type.
 */
#if defined(MBEDTLS_SSL_PROTO_DTLS)
static int ssl_parse_hello_verify_request( mbedtls_ssl_context *ssl )
{
    const unsigned char *p = ssl->in_msg + mbedtls_ssl_hs_hdr_len( ssl );
    int major_ver, minor_ver;
    unsigned char cookie_len;

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "=> parse hello verify request" ) );

    /*
     * struct {
     *   ProtocolVersion server_version;
     *   opaque cookie<0..2^8-1>;
     * } HelloVerifyRequest;
     */
    MBEDTLS_SSL_DEBUG_BUF( 3, "server version", p, 2 );
    mbedtls_ssl_read_version( &major_ver, &minor_ver, ssl->conf->transport, p );
    p += 2;

    /*
     * Since the RFC is not clear on this point, accept DTLS 1.0 (TLS 1.1)
     * even is lower than our min version.
     */
    if( major_ver < MBEDTLS_SSL_MAJOR_VERSION_3 ||
        minor_ver < MBEDTLS_SSL_MINOR_VERSION_2 ||
        major_ver > ssl->conf->max_major_ver  ||
        minor_ver > ssl->conf->max_minor_ver  )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad server version" ) );

        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                     MBEDTLS_SSL_ALERT_MSG_PROTOCOL_VERSION );

        return( MBEDTLS_ERR_SSL_BAD_HS_PROTOCOL_VERSION );
    }

    cookie_len = *p++;
    MBEDTLS_SSL_DEBUG_BUF( 3, "cookie", p, cookie_len );

    if( ( ssl->in_msg + ssl->in_msglen ) - p < cookie_len )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1,
            ( "cookie length does not match incoming message size" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                    MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
        return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_HELLO );
    }

    mbedtls_free( ssl->handshake->verify_cookie );

    ssl->handshake->verify_cookie = mbedtls_calloc( 1, cookie_len );
    if( ssl->handshake->verify_cookie  == NULL )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "alloc failed (%d bytes)", cookie_len ) );
        return( MBEDTLS_ERR_SSL_ALLOC_FAILED );
    }

    memcpy( ssl->handshake->verify_cookie, p, cookie_len );
    ssl->handshake->verify_cookie_len = cookie_len;

    /* Start over at ClientHello */
    ssl->state = MBEDTLS_SSL_CLIENT_HELLO;
    mbedtls_ssl_reset_checksum( ssl );

    mbedtls_ssl_recv_flight_completed( ssl );

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= parse hello verify request" ) );

    return( 0 );
}
#endif /* MBEDTLS_SSL_PROTO_DTLS */

static int ssl_parse_server_hello( mbedtls_ssl_context *ssl )
{
    int ret, i;
    size_t n;
    size_t ext_len;
    unsigned char *buf, *ext;
    unsigned char comp;
#if defined(MBEDTLS_ZLIB_SUPPORT)
    int accept_comp;
#endif
#if defined(MBEDTLS_SSL_RENEGOTIATION)
    int renegotiation_info_seen = 0;
#endif
    int handshake_failure = 0;
    const mbedtls_ssl_ciphersuite_t *suite_info;

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "=> parse server hello" ) );

    buf = ssl->in_msg;

    if( ( ret = mbedtls_ssl_read_record( ssl, 1 ) ) != 0 )
    {
        /* No alert on a read error. */
        MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ssl_read_record", ret );
        return( ret );
    }

    if( ssl->in_msgtype != MBEDTLS_SSL_MSG_HANDSHAKE )
    {
#if defined(MBEDTLS_SSL_RENEGOTIATION)
        if( ssl->renego_status == MBEDTLS_SSL_RENEGOTIATION_IN_PROGRESS )
        {
            ssl->renego_records_seen++;

            if( ssl->conf->renego_max_records >= 0 &&
                ssl->renego_records_seen > ssl->conf->renego_max_records )
            {
                MBEDTLS_SSL_DEBUG_MSG( 1, ( "renegotiation requested, "
                                    "but not honored by server" ) );
                return( MBEDTLS_ERR_SSL_UNEXPECTED_MESSAGE );
            }

            MBEDTLS_SSL_DEBUG_MSG( 1, ( "non-handshake message during renego" ) );

            ssl->keep_current_message = 1;
            return( MBEDTLS_ERR_SSL_WAITING_SERVER_HELLO_RENEGO );
        }
#endif /* MBEDTLS_SSL_RENEGOTIATION */

        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad server hello message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_UNEXPECTED_MESSAGE );
        return( MBEDTLS_ERR_SSL_UNEXPECTED_MESSAGE );
    }

#if defined(MBEDTLS_SSL_PROTO_DTLS)
    if( ssl->conf->transport == MBEDTLS_SSL_TRANSPORT_DATAGRAM )
    {
        if( buf[0] == MBEDTLS_SSL_HS_HELLO_VERIFY_REQUEST )
        {
            MBEDTLS_SSL_DEBUG_MSG( 2, ( "received hello verify request" ) );
            MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= parse server hello" ) );
            return( ssl_parse_hello_verify_request( ssl ) );
        }
        else
        {
            /* We made it through the verification process */
            mbedtls_free( ssl->handshake->verify_cookie );
            ssl->handshake->verify_cookie = NULL;
            ssl->handshake->verify_cookie_len = 0;
        }
    }
#endif /* MBEDTLS_SSL_PROTO_DTLS */

    if( ssl->in_hslen < 38 + mbedtls_ssl_hs_hdr_len( ssl ) ||
        buf[0] != MBEDTLS_SSL_HS_SERVER_HELLO )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad server hello message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
        return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_HELLO );
    }

    /*
     *  0   .  1    server_version
     *  2   . 33    random (maybe including 4 bytes of Unix time)
     * 34   . 34    session_id length = n
     * 35   . 34+n  session_id
     * 35+n . 36+n  cipher_suite
     * 37+n . 37+n  compression_method
     *
     * 38+n . 39+n  extensions length (optional)
     * 40+n .  ..   extensions
     */
    buf += mbedtls_ssl_hs_hdr_len( ssl );

    MBEDTLS_SSL_DEBUG_BUF( 3, "server hello, version", buf + 0, 2 );
    mbedtls_ssl_read_version( &ssl->major_ver, &ssl->minor_ver,
                      ssl->conf->transport, buf + 0 );

    if( ssl->major_ver < ssl->conf->min_major_ver ||
        ssl->minor_ver < ssl->conf->min_minor_ver ||
        ssl->major_ver > ssl->conf->max_major_ver ||
        ssl->minor_ver > ssl->conf->max_minor_ver )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "server version out of bounds - "
                            " min: [%d:%d], server: [%d:%d], max: [%d:%d]",
                            ssl->conf->min_major_ver, ssl->conf->min_minor_ver,
                            ssl->major_ver, ssl->minor_ver,
                            ssl->conf->max_major_ver, ssl->conf->max_minor_ver ) );

        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                     MBEDTLS_SSL_ALERT_MSG_PROTOCOL_VERSION );

        return( MBEDTLS_ERR_SSL_BAD_HS_PROTOCOL_VERSION );
    }

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "server hello, current time: %lu",
                           ( (uint32_t) buf[2] << 24 ) |
                           ( (uint32_t) buf[3] << 16 ) |
                           ( (uint32_t) buf[4] <<  8 ) |
                           ( (uint32_t) buf[5]       ) ) );

    memcpy( ssl->handshake->randbytes + 32, buf + 2, 32 );

    n = buf[34];

    MBEDTLS_SSL_DEBUG_BUF( 3,   "server hello, random bytes", buf + 2, 32 );

    if( n > 32 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad server hello message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
        return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_HELLO );
    }

    if( ssl->in_hslen > mbedtls_ssl_hs_hdr_len( ssl ) + 39 + n )
    {
        ext_len = ( ( buf[38 + n] <<  8 )
                  | ( buf[39 + n]       ) );

        if( ( ext_len > 0 && ext_len < 4 ) ||
            ssl->in_hslen != mbedtls_ssl_hs_hdr_len( ssl ) + 40 + n + ext_len )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad server hello message" ) );
            mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                            MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
            return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_HELLO );
        }
    }
    else if( ssl->in_hslen == mbedtls_ssl_hs_hdr_len( ssl ) + 38 + n )
    {
        ext_len = 0;
    }
    else
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad server hello message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
        return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_HELLO );
    }

    /* ciphersuite (used later) */
    i = ( buf[35 + n] << 8 ) | buf[36 + n];

    /*
     * Read and check compression
     */
    comp = buf[37 + n];

#if defined(MBEDTLS_ZLIB_SUPPORT)
    /* See comments in ssl_write_client_hello() */
#if defined(MBEDTLS_SSL_PROTO_DTLS)
    if( ssl->conf->transport == MBEDTLS_SSL_TRANSPORT_DATAGRAM )
        accept_comp = 0;
    else
#endif
        accept_comp = 1;

    if( comp != MBEDTLS_SSL_COMPRESS_NULL &&
        ( comp != MBEDTLS_SSL_COMPRESS_DEFLATE || accept_comp == 0 ) )
#else /* MBEDTLS_ZLIB_SUPPORT */
    if( comp != MBEDTLS_SSL_COMPRESS_NULL )
#endif/* MBEDTLS_ZLIB_SUPPORT */
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "server hello, bad compression: %d", comp ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_ILLEGAL_PARAMETER );
        return( MBEDTLS_ERR_SSL_FEATURE_UNAVAILABLE );
    }

    /*
     * Initialize update checksum functions
     */
    ssl->transform_negotiate->ciphersuite_info = mbedtls_ssl_ciphersuite_from_id( i );

    if( ssl->transform_negotiate->ciphersuite_info == NULL )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "ciphersuite info for %04x not found", i ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_INTERNAL_ERROR );
        return( MBEDTLS_ERR_SSL_BAD_INPUT_DATA );
    }

    mbedtls_ssl_optimize_checksum( ssl, ssl->transform_negotiate->ciphersuite_info );

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "server hello, session id len.: %d", n ) );
    MBEDTLS_SSL_DEBUG_BUF( 3,   "server hello, session id", buf + 35, n );

    /*
     * Check if the session can be resumed
     */
    if( ssl->handshake->resume == 0 || n == 0 ||
#if defined(MBEDTLS_SSL_RENEGOTIATION)
        ssl->renego_status != MBEDTLS_SSL_INITIAL_HANDSHAKE ||
#endif
        ssl->session_negotiate->ciphersuite != i ||
        ssl->session_negotiate->compression != comp ||
        ssl->session_negotiate->id_len != n ||
        memcmp( ssl->session_negotiate->id, buf + 35, n ) != 0 )
    {
        ssl->state++;
        ssl->handshake->resume = 0;
#if defined(MBEDTLS_HAVE_TIME)
        ssl->session_negotiate->start = mbedtls_time( NULL );
#endif
        ssl->session_negotiate->ciphersuite = i;
        ssl->session_negotiate->compression = comp;
        ssl->session_negotiate->id_len = n;
        memcpy( ssl->session_negotiate->id, buf + 35, n );
    }
    else
    {
        ssl->state = MBEDTLS_SSL_SERVER_CHANGE_CIPHER_SPEC;

        if( ( ret = mbedtls_ssl_derive_keys( ssl ) ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ssl_derive_keys", ret );
            mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                            MBEDTLS_SSL_ALERT_MSG_INTERNAL_ERROR );
            return( ret );
        }
    }

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "%s session has been resumed",
                   ssl->handshake->resume ? "a" : "no" ) );

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "server hello, chosen ciphersuite: %04x", i ) );
    MBEDTLS_SSL_DEBUG_MSG( 3, ( "server hello, compress alg.: %d", buf[37 + n] ) );

    /*
     * Perform cipher suite validation in same way as in ssl_write_client_hello.
     */
    i = 0;
    while( 1 )
    {
        if( ssl->conf->ciphersuite_list[ssl->minor_ver][i] == 0 )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad server hello message" ) );
            mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                            MBEDTLS_SSL_ALERT_MSG_ILLEGAL_PARAMETER );
            return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_HELLO );
        }

        if( ssl->conf->ciphersuite_list[ssl->minor_ver][i++] ==
            ssl->session_negotiate->ciphersuite )
        {
            break;
        }
    }

    suite_info = mbedtls_ssl_ciphersuite_from_id( ssl->session_negotiate->ciphersuite );
    if( ssl_validate_ciphersuite( suite_info, ssl, ssl->minor_ver, ssl->minor_ver ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad server hello message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_ILLEGAL_PARAMETER );
        return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_HELLO );
    }

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "server hello, chosen ciphersuite: %s", suite_info->name ) );

#if defined(MBEDTLS_SSL__ECP_RESTARTABLE)
    if( suite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA &&
        ssl->minor_ver == MBEDTLS_SSL_MINOR_VERSION_3 )
    {
        ssl->handshake->ecrs_enabled = 1;
    }
#endif

    if( comp != MBEDTLS_SSL_COMPRESS_NULL
#if defined(MBEDTLS_ZLIB_SUPPORT)
        && comp != MBEDTLS_SSL_COMPRESS_DEFLATE
#endif
      )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad server hello message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_ILLEGAL_PARAMETER );
        return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_HELLO );
    }
    ssl->session_negotiate->compression = comp;

    ext = buf + 40 + n;

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "server hello, total extension length: %d", ext_len ) );

    while( ext_len )
    {
        unsigned int ext_id   = ( ( ext[0] <<  8 )
                                | ( ext[1]       ) );
        unsigned int ext_size = ( ( ext[2] <<  8 )
                                | ( ext[3]       ) );

        if( ext_size + 4 > ext_len )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad server hello message" ) );
            mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                            MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
            return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_HELLO );
        }

        switch( ext_id )
        {
        case MBEDTLS_TLS_EXT_RENEGOTIATION_INFO:
            MBEDTLS_SSL_DEBUG_MSG( 3, ( "found renegotiation extension" ) );
#if defined(MBEDTLS_SSL_RENEGOTIATION)
            renegotiation_info_seen = 1;
#endif

            if( ( ret = ssl_parse_renegotiation_info( ssl, ext + 4,
                                                      ext_size ) ) != 0 )
                return( ret );

            break;

#if defined(MBEDTLS_SSL_MAX_FRAGMENT_LENGTH)
        case MBEDTLS_TLS_EXT_MAX_FRAGMENT_LENGTH:
            MBEDTLS_SSL_DEBUG_MSG( 3, ( "found max_fragment_length extension" ) );

            if( ( ret = ssl_parse_max_fragment_length_ext( ssl,
                            ext + 4, ext_size ) ) != 0 )
            {
                return( ret );
            }

            break;
#endif /* MBEDTLS_SSL_MAX_FRAGMENT_LENGTH */

#if defined(MBEDTLS_SSL_TRUNCATED_HMAC)
        case MBEDTLS_TLS_EXT_TRUNCATED_HMAC:
            MBEDTLS_SSL_DEBUG_MSG( 3, ( "found truncated_hmac extension" ) );

            if( ( ret = ssl_parse_truncated_hmac_ext( ssl,
                            ext + 4, ext_size ) ) != 0 )
            {
                return( ret );
            }

            break;
#endif /* MBEDTLS_SSL_TRUNCATED_HMAC */

#if defined(MBEDTLS_SSL_ENCRYPT_THEN_MAC)
        case MBEDTLS_TLS_EXT_ENCRYPT_THEN_MAC:
            MBEDTLS_SSL_DEBUG_MSG( 3, ( "found encrypt_then_mac extension" ) );

            if( ( ret = ssl_parse_encrypt_then_mac_ext( ssl,
                            ext + 4, ext_size ) ) != 0 )
            {
                return( ret );
            }

            break;
#endif /* MBEDTLS_SSL_ENCRYPT_THEN_MAC */

#if defined(MBEDTLS_SSL_EXTENDED_MASTER_SECRET)
        case MBEDTLS_TLS_EXT_EXTENDED_MASTER_SECRET:
            MBEDTLS_SSL_DEBUG_MSG( 3, ( "found extended_master_secret extension" ) );

            if( ( ret = ssl_parse_extended_ms_ext( ssl,
                            ext + 4, ext_size ) ) != 0 )
            {
                return( ret );
            }

            break;
#endif /* MBEDTLS_SSL_EXTENDED_MASTER_SECRET */

#if defined(MBEDTLS_SSL_SESSION_TICKETS)
        case MBEDTLS_TLS_EXT_SESSION_TICKET:
            MBEDTLS_SSL_DEBUG_MSG( 3, ( "found session_ticket extension" ) );

            if( ( ret = ssl_parse_session_ticket_ext( ssl,
                            ext + 4, ext_size ) ) != 0 )
            {
                return( ret );
            }

            break;
#endif /* MBEDTLS_SSL_SESSION_TICKETS */

#if defined(MBEDTLS_ECDH_C) || defined(MBEDTLS_ECDSA_C) || \
    defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
        case MBEDTLS_TLS_EXT_SUPPORTED_POINT_FORMATS:
            MBEDTLS_SSL_DEBUG_MSG( 3, ( "found supported_point_formats extension" ) );

            if( ( ret = ssl_parse_supported_point_formats_ext( ssl,
                            ext + 4, ext_size ) ) != 0 )
            {
                return( ret );
            }

            break;
#endif /* MBEDTLS_ECDH_C || MBEDTLS_ECDSA_C ||
          MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED */

#if defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
        case MBEDTLS_TLS_EXT_ECJPAKE_KKPP:
            MBEDTLS_SSL_DEBUG_MSG( 3, ( "found ecjpake_kkpp extension" ) );

            if( ( ret = ssl_parse_ecjpake_kkpp( ssl,
                            ext + 4, ext_size ) ) != 0 )
            {
                return( ret );
            }

            break;
#endif /* MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED */

#if defined(MBEDTLS_SSL_ALPN)
        case MBEDTLS_TLS_EXT_ALPN:
            MBEDTLS_SSL_DEBUG_MSG( 3, ( "found alpn extension" ) );

            if( ( ret = ssl_parse_alpn_ext( ssl, ext + 4, ext_size ) ) != 0 )
                return( ret );

            break;
#endif /* MBEDTLS_SSL_ALPN */

        default:
            MBEDTLS_SSL_DEBUG_MSG( 3, ( "unknown extension found: %d (ignoring)",
                           ext_id ) );
        }

        ext_len -= 4 + ext_size;
        ext += 4 + ext_size;

        if( ext_len > 0 && ext_len < 4 )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad server hello message" ) );
            return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_HELLO );
        }
    }

    /*
     * Renegotiation security checks
     */
    if( ssl->secure_renegotiation == MBEDTLS_SSL_LEGACY_RENEGOTIATION &&
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
        return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_HELLO );
    }

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= parse server hello" ) );

    return( 0 );
}

#if defined(MBEDTLS_KEY_EXCHANGE_DHE_RSA_ENABLED) ||                       \
    defined(MBEDTLS_KEY_EXCHANGE_DHE_PSK_ENABLED)
static int ssl_parse_server_dh_params( mbedtls_ssl_context *ssl, unsigned char **p,
                                       unsigned char *end )
{
    int ret = MBEDTLS_ERR_SSL_FEATURE_UNAVAILABLE;

    /*
     * Ephemeral DH parameters:
     *
     * struct {
     *     opaque dh_p<1..2^16-1>;
     *     opaque dh_g<1..2^16-1>;
     *     opaque dh_Ys<1..2^16-1>;
     * } ServerDHParams;
     */
    if( ( ret = mbedtls_dhm_read_params( &ssl->handshake->dhm_ctx, p, end ) ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_RET( 2, ( "mbedtls_dhm_read_params" ), ret );
        return( ret );
    }

    if( ssl->handshake->dhm_ctx.len * 8 < ssl->conf->dhm_min_bitlen )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "DHM prime too short: %d < %d",
                                    ssl->handshake->dhm_ctx.len * 8,
                                    ssl->conf->dhm_min_bitlen ) );
        return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_KEY_EXCHANGE );
    }

    MBEDTLS_SSL_DEBUG_MPI( 3, "DHM: P ", &ssl->handshake->dhm_ctx.P  );
    MBEDTLS_SSL_DEBUG_MPI( 3, "DHM: G ", &ssl->handshake->dhm_ctx.G  );
    MBEDTLS_SSL_DEBUG_MPI( 3, "DHM: GY", &ssl->handshake->dhm_ctx.GY );

    return( ret );
}
#endif /* MBEDTLS_KEY_EXCHANGE_DHE_RSA_ENABLED ||
          MBEDTLS_KEY_EXCHANGE_DHE_PSK_ENABLED */

#if defined(MBEDTLS_KEY_EXCHANGE_ECDHE_RSA_ENABLED) ||                     \
    defined(MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA_ENABLED) ||                   \
    defined(MBEDTLS_KEY_EXCHANGE_ECDHE_PSK_ENABLED) ||                     \
    defined(MBEDTLS_KEY_EXCHANGE_ECDH_RSA_ENABLED) ||                      \
    defined(MBEDTLS_KEY_EXCHANGE_ECDH_ECDSA_ENABLED)
static int ssl_check_server_ecdh_params( const mbedtls_ssl_context *ssl )
{
    const mbedtls_ecp_curve_info *curve_info;
    mbedtls_ecp_group_id grp_id;
#if defined(MBEDTLS_ECDH_LEGACY_CONTEXT)
    grp_id = ssl->handshake->ecdh_ctx.grp.id;
#else
    grp_id = ssl->handshake->ecdh_ctx.grp_id;
#endif

    curve_info = mbedtls_ecp_curve_info_from_grp_id( grp_id );
    if( curve_info == NULL )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "should never happen" ) );
        return( MBEDTLS_ERR_SSL_INTERNAL_ERROR );
    }

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "ECDH curve: %s", curve_info->name ) );

#if defined(MBEDTLS_ECP_C)
    if( mbedtls_ssl_check_curve( ssl, grp_id ) != 0 )
#else
    if( ssl->handshake->ecdh_ctx.grp.nbits < 163 ||
        ssl->handshake->ecdh_ctx.grp.nbits > 521 )
#endif
        return( -1 );

    MBEDTLS_SSL_DEBUG_ECDH( 3, &ssl->handshake->ecdh_ctx,
                            MBEDTLS_DEBUG_ECDH_QP );

    return( 0 );
}
#endif /* MBEDTLS_KEY_EXCHANGE_ECDHE_RSA_ENABLED ||
          MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA_ENABLED ||
          MBEDTLS_KEY_EXCHANGE_ECDHE_PSK_ENABLED ||
          MBEDTLS_KEY_EXCHANGE_ECDH_RSA_ENABLED ||
          MBEDTLS_KEY_EXCHANGE_ECDH_ECDSA_ENABLED */

#if defined(MBEDTLS_KEY_EXCHANGE_ECDHE_RSA_ENABLED) ||                     \
    defined(MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA_ENABLED) ||                   \
    defined(MBEDTLS_KEY_EXCHANGE_ECDHE_PSK_ENABLED)
static int ssl_parse_server_ecdh_params( mbedtls_ssl_context *ssl,
                                         unsigned char **p,
                                         unsigned char *end )
{
    int ret = MBEDTLS_ERR_SSL_FEATURE_UNAVAILABLE;

    /*
     * Ephemeral ECDH parameters:
     *
     * struct {
     *     ECParameters curve_params;
     *     ECPoint      public;
     * } ServerECDHParams;
     */
    if( ( ret = mbedtls_ecdh_read_params( &ssl->handshake->ecdh_ctx,
                                  (const unsigned char **) p, end ) ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_RET( 1, ( "mbedtls_ecdh_read_params" ), ret );
#if defined(MBEDTLS_SSL__ECP_RESTARTABLE)
        if( ret == MBEDTLS_ERR_ECP_IN_PROGRESS )
            ret = MBEDTLS_ERR_SSL_CRYPTO_IN_PROGRESS;
#endif
        return( ret );
    }

    if( ssl_check_server_ecdh_params( ssl ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad server key exchange message (ECDHE curve)" ) );
        return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_KEY_EXCHANGE );
    }

    return( ret );
}
#endif /* MBEDTLS_KEY_EXCHANGE_ECDHE_RSA_ENABLED ||
          MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA_ENABLED ||
          MBEDTLS_KEY_EXCHANGE_ECDHE_PSK_ENABLED */

#if defined(MBEDTLS_KEY_EXCHANGE__SOME__PSK_ENABLED)
static int ssl_parse_server_psk_hint( mbedtls_ssl_context *ssl,
                                      unsigned char **p,
                                      unsigned char *end )
{
    int ret = MBEDTLS_ERR_SSL_FEATURE_UNAVAILABLE;
    size_t  len;
    ((void) ssl);

    /*
     * PSK parameters:
     *
     * opaque psk_identity_hint<0..2^16-1>;
     */
    if( end - (*p) < 2 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad server key exchange message "
                                    "(psk_identity_hint length)" ) );
        return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_KEY_EXCHANGE );
    }
    len = (*p)[0] << 8 | (*p)[1];
    *p += 2;

    if( end - (*p) < (int) len )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad server key exchange message "
                                    "(psk_identity_hint length)" ) );
        return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_KEY_EXCHANGE );
    }

    /*
     * Note: we currently ignore the PKS identity hint, as we only allow one
     * PSK to be provisionned on the client. This could be changed later if
     * someone needs that feature.
     */
    *p += len;
    ret = 0;

    return( ret );
}
#endif /* MBEDTLS_KEY_EXCHANGE__SOME__PSK_ENABLED */

#if defined(MBEDTLS_KEY_EXCHANGE_RSA_ENABLED) ||                           \
    defined(MBEDTLS_KEY_EXCHANGE_RSA_PSK_ENABLED)
/*
 * Generate a pre-master secret and encrypt it with the server's RSA key
 */
static int ssl_write_encrypted_pms( mbedtls_ssl_context *ssl,
                                    size_t offset, size_t *olen,
                                    size_t pms_offset )
{
    int ret;
    size_t len_bytes = ssl->minor_ver == MBEDTLS_SSL_MINOR_VERSION_0 ? 0 : 2;
    unsigned char *p = ssl->handshake->premaster + pms_offset;

    if( offset + len_bytes > MBEDTLS_SSL_OUT_CONTENT_LEN )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "buffer too small for encrypted pms" ) );
        return( MBEDTLS_ERR_SSL_BUFFER_TOO_SMALL );
    }

    /*
     * Generate (part of) the pre-master as
     *  struct {
     *      ProtocolVersion client_version;
     *      opaque random[46];
     *  } PreMasterSecret;
     */
    mbedtls_ssl_write_version( ssl->conf->max_major_ver, ssl->conf->max_minor_ver,
                       ssl->conf->transport, p );

    if( ( ret = ssl->conf->f_rng( ssl->conf->p_rng, p + 2, 46 ) ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_RET( 1, "f_rng", ret );
        return( ret );
    }

    ssl->handshake->pmslen = 48;

    if( ssl->session_negotiate->peer_cert == NULL )
    {
        MBEDTLS_SSL_DEBUG_MSG( 2, ( "certificate required" ) );
        return( MBEDTLS_ERR_SSL_UNEXPECTED_MESSAGE );
    }

    /*
     * Now write it out, encrypted
     */
    if( ! mbedtls_pk_can_do( &ssl->session_negotiate->peer_cert->pk,
                MBEDTLS_PK_RSA ) )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "certificate key type mismatch" ) );
        return( MBEDTLS_ERR_SSL_PK_TYPE_MISMATCH );
    }

    if( ( ret = mbedtls_pk_encrypt( &ssl->session_negotiate->peer_cert->pk,
                            p, ssl->handshake->pmslen,
                            ssl->out_msg + offset + len_bytes, olen,
                            MBEDTLS_SSL_OUT_CONTENT_LEN - offset - len_bytes,
                            ssl->conf->f_rng, ssl->conf->p_rng ) ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_rsa_pkcs1_encrypt", ret );
        return( ret );
    }

#if defined(MBEDTLS_SSL_PROTO_TLS1) || defined(MBEDTLS_SSL_PROTO_TLS1_1) || \
    defined(MBEDTLS_SSL_PROTO_TLS1_2)
    if( len_bytes == 2 )
    {
        ssl->out_msg[offset+0] = (unsigned char)( *olen >> 8 );
        ssl->out_msg[offset+1] = (unsigned char)( *olen      );
        *olen += 2;
    }
#endif

    return( 0 );
}
#endif /* MBEDTLS_KEY_EXCHANGE_RSA_ENABLED ||
          MBEDTLS_KEY_EXCHANGE_RSA_PSK_ENABLED */

#if defined(MBEDTLS_SSL_PROTO_TLS1_2)
#if defined(MBEDTLS_KEY_EXCHANGE_DHE_RSA_ENABLED) ||                       \
    defined(MBEDTLS_KEY_EXCHANGE_ECDHE_RSA_ENABLED) ||                     \
    defined(MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA_ENABLED)
static int ssl_parse_signature_algorithm( mbedtls_ssl_context *ssl,
                                          unsigned char **p,
                                          unsigned char *end,
                                          mbedtls_md_type_t *md_alg,
                                          mbedtls_pk_type_t *pk_alg )
{
    ((void) ssl);
    *md_alg = MBEDTLS_MD_NONE;
    *pk_alg = MBEDTLS_PK_NONE;

    /* Only in TLS 1.2 */
    if( ssl->minor_ver != MBEDTLS_SSL_MINOR_VERSION_3 )
    {
        return( 0 );
    }

    if( (*p) + 2 > end )
        return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_KEY_EXCHANGE );

    /*
     * Get hash algorithm
     */
    if( ( *md_alg = mbedtls_ssl_md_alg_from_hash( (*p)[0] ) ) == MBEDTLS_MD_NONE )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "Server used unsupported "
                            "HashAlgorithm %d", *(p)[0] ) );
        return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_KEY_EXCHANGE );
    }

    /*
     * Get signature algorithm
     */
    if( ( *pk_alg = mbedtls_ssl_pk_alg_from_sig( (*p)[1] ) ) == MBEDTLS_PK_NONE )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "server used unsupported "
                            "SignatureAlgorithm %d", (*p)[1] ) );
        return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_KEY_EXCHANGE );
    }

    /*
     * Check if the hash is acceptable
     */
    if( mbedtls_ssl_check_sig_hash( ssl, *md_alg ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "server used HashAlgorithm %d that was not offered",
                                    *(p)[0] ) );
        return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_KEY_EXCHANGE );
    }

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "Server used SignatureAlgorithm %d", (*p)[1] ) );
    MBEDTLS_SSL_DEBUG_MSG( 2, ( "Server used HashAlgorithm %d", (*p)[0] ) );
    *p += 2;

    return( 0 );
}
#endif /* MBEDTLS_KEY_EXCHANGE_DHE_RSA_ENABLED ||
          MBEDTLS_KEY_EXCHANGE_ECDHE_RSA_ENABLED ||
          MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA_ENABLED */
#endif /* MBEDTLS_SSL_PROTO_TLS1_2 */

#if defined(MBEDTLS_KEY_EXCHANGE_ECDH_RSA_ENABLED) || \
    defined(MBEDTLS_KEY_EXCHANGE_ECDH_ECDSA_ENABLED)
static int ssl_get_ecdh_params_from_cert( mbedtls_ssl_context *ssl )
{
    int ret;
    const mbedtls_ecp_keypair *peer_key;

    if( ssl->session_negotiate->peer_cert == NULL )
    {
        MBEDTLS_SSL_DEBUG_MSG( 2, ( "certificate required" ) );
        return( MBEDTLS_ERR_SSL_UNEXPECTED_MESSAGE );
    }

    if( ! mbedtls_pk_can_do( &ssl->session_negotiate->peer_cert->pk,
                     MBEDTLS_PK_ECKEY ) )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "server key not ECDH capable" ) );
        return( MBEDTLS_ERR_SSL_PK_TYPE_MISMATCH );
    }

    peer_key = mbedtls_pk_ec( ssl->session_negotiate->peer_cert->pk );

    if( ( ret = mbedtls_ecdh_get_params( &ssl->handshake->ecdh_ctx, peer_key,
                                 MBEDTLS_ECDH_THEIRS ) ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_RET( 1, ( "mbedtls_ecdh_get_params" ), ret );
        return( ret );
    }

    if( ssl_check_server_ecdh_params( ssl ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad server certificate (ECDH curve)" ) );
        return( MBEDTLS_ERR_SSL_BAD_HS_CERTIFICATE );
    }

    return( ret );
}
#endif /* MBEDTLS_KEY_EXCHANGE_ECDH_RSA_ENABLED) ||
          MBEDTLS_KEY_EXCHANGE_ECDH_ECDSA_ENABLED */

static int ssl_parse_server_key_exchange( mbedtls_ssl_context *ssl )
{
    int ret;
    const mbedtls_ssl_ciphersuite_t *ciphersuite_info =
        ssl->transform_negotiate->ciphersuite_info;
    unsigned char *p = NULL, *end = NULL;

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "=> parse server key exchange" ) );

#if defined(MBEDTLS_KEY_EXCHANGE_RSA_ENABLED)
    if( ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_RSA )
    {
        MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= skip parse server key exchange" ) );
        ssl->state++;
        return( 0 );
    }
    ((void) p);
    ((void) end);
#endif

#if defined(MBEDTLS_KEY_EXCHANGE_ECDH_RSA_ENABLED) || \
    defined(MBEDTLS_KEY_EXCHANGE_ECDH_ECDSA_ENABLED)
    if( ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_ECDH_RSA ||
        ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_ECDH_ECDSA )
    {
        if( ( ret = ssl_get_ecdh_params_from_cert( ssl ) ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, "ssl_get_ecdh_params_from_cert", ret );
            mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                            MBEDTLS_SSL_ALERT_MSG_HANDSHAKE_FAILURE );
            return( ret );
        }

        MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= skip parse server key exchange" ) );
        ssl->state++;
        return( 0 );
    }
    ((void) p);
    ((void) end);
#endif /* MBEDTLS_KEY_EXCHANGE_ECDH_RSA_ENABLED ||
          MBEDTLS_KEY_EXCHANGE_ECDH_ECDSA_ENABLED */

#if defined(MBEDTLS_SSL__ECP_RESTARTABLE)
    if( ssl->handshake->ecrs_enabled &&
        ssl->handshake->ecrs_state == ssl_ecrs_ske_start_processing )
    {
        goto start_processing;
    }
#endif

    if( ( ret = mbedtls_ssl_read_record( ssl, 1 ) ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ssl_read_record", ret );
        return( ret );
    }

    if( ssl->in_msgtype != MBEDTLS_SSL_MSG_HANDSHAKE )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad server key exchange message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_UNEXPECTED_MESSAGE );
        return( MBEDTLS_ERR_SSL_UNEXPECTED_MESSAGE );
    }

    /*
     * ServerKeyExchange may be skipped with PSK and RSA-PSK when the server
     * doesn't use a psk_identity_hint
     */
    if( ssl->in_msg[0] != MBEDTLS_SSL_HS_SERVER_KEY_EXCHANGE )
    {
        if( ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_PSK ||
            ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_RSA_PSK )
        {
            /* Current message is probably either
             * CertificateRequest or ServerHelloDone */
            ssl->keep_current_message = 1;
            goto exit;
        }

        MBEDTLS_SSL_DEBUG_MSG( 1, ( "server key exchange message must "
                                    "not be skipped" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_UNEXPECTED_MESSAGE );

        return( MBEDTLS_ERR_SSL_UNEXPECTED_MESSAGE );
    }

#if defined(MBEDTLS_SSL__ECP_RESTARTABLE)
    if( ssl->handshake->ecrs_enabled )
        ssl->handshake->ecrs_state = ssl_ecrs_ske_start_processing;

start_processing:
#endif
    p   = ssl->in_msg + mbedtls_ssl_hs_hdr_len( ssl );
    end = ssl->in_msg + ssl->in_hslen;
    MBEDTLS_SSL_DEBUG_BUF( 3,   "server key exchange", p, end - p );

#if defined(MBEDTLS_KEY_EXCHANGE__SOME__PSK_ENABLED)
    if( ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_PSK ||
        ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_RSA_PSK ||
        ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_DHE_PSK ||
        ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_ECDHE_PSK )
    {
        if( ssl_parse_server_psk_hint( ssl, &p, end ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad server key exchange message" ) );
            mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                            MBEDTLS_SSL_ALERT_MSG_ILLEGAL_PARAMETER );
            return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_KEY_EXCHANGE );
        }
    } /* FALLTROUGH */
#endif /* MBEDTLS_KEY_EXCHANGE__SOME__PSK_ENABLED */

#if defined(MBEDTLS_KEY_EXCHANGE_PSK_ENABLED) ||                       \
    defined(MBEDTLS_KEY_EXCHANGE_RSA_PSK_ENABLED)
    if( ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_PSK ||
        ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_RSA_PSK )
        ; /* nothing more to do */
    else
#endif /* MBEDTLS_KEY_EXCHANGE_PSK_ENABLED ||
          MBEDTLS_KEY_EXCHANGE_RSA_PSK_ENABLED */
#if defined(MBEDTLS_KEY_EXCHANGE_DHE_RSA_ENABLED) ||                       \
    defined(MBEDTLS_KEY_EXCHANGE_DHE_PSK_ENABLED)
    if( ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_DHE_RSA ||
        ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_DHE_PSK )
    {
        if( ssl_parse_server_dh_params( ssl, &p, end ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad server key exchange message" ) );
            mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                            MBEDTLS_SSL_ALERT_MSG_ILLEGAL_PARAMETER );
            return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_KEY_EXCHANGE );
        }
    }
    else
#endif /* MBEDTLS_KEY_EXCHANGE_DHE_RSA_ENABLED ||
          MBEDTLS_KEY_EXCHANGE_DHE_PSK_ENABLED */
#if defined(MBEDTLS_KEY_EXCHANGE_ECDHE_RSA_ENABLED) ||                     \
    defined(MBEDTLS_KEY_EXCHANGE_ECDHE_PSK_ENABLED) ||                     \
    defined(MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA_ENABLED)
    if( ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_ECDHE_RSA ||
        ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_ECDHE_PSK ||
        ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA )
    {
        if( ssl_parse_server_ecdh_params( ssl, &p, end ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad server key exchange message" ) );
            mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                            MBEDTLS_SSL_ALERT_MSG_ILLEGAL_PARAMETER );
            return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_KEY_EXCHANGE );
        }
    }
    else
#endif /* MBEDTLS_KEY_EXCHANGE_ECDHE_RSA_ENABLED ||
          MBEDTLS_KEY_EXCHANGE_ECDHE_PSK_ENABLED ||
          MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA_ENABLED */
#if defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
    if( ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_ECJPAKE )
    {
        ret = mbedtls_ecjpake_read_round_two( &ssl->handshake->ecjpake_ctx,
                                              p, end - p );
        if( ret != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ecjpake_read_round_two", ret );
            mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                            MBEDTLS_SSL_ALERT_MSG_ILLEGAL_PARAMETER );
            return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_KEY_EXCHANGE );
        }
    }
    else
#endif /* MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED */
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "should never happen" ) );
        return( MBEDTLS_ERR_SSL_INTERNAL_ERROR );
    }

#if defined(MBEDTLS_KEY_EXCHANGE__WITH_SERVER_SIGNATURE__ENABLED)
    if( mbedtls_ssl_ciphersuite_uses_server_signature( ciphersuite_info ) )
    {
        size_t sig_len, hashlen;
        unsigned char hash[64];
        mbedtls_md_type_t md_alg = MBEDTLS_MD_NONE;
        mbedtls_pk_type_t pk_alg = MBEDTLS_PK_NONE;
        unsigned char *params = ssl->in_msg + mbedtls_ssl_hs_hdr_len( ssl );
        size_t params_len = p - params;
        void *rs_ctx = NULL;

        /*
         * Handle the digitally-signed structure
         */
#if defined(MBEDTLS_SSL_PROTO_TLS1_2)
        if( ssl->minor_ver == MBEDTLS_SSL_MINOR_VERSION_3 )
        {
            if( ssl_parse_signature_algorithm( ssl, &p, end,
                                               &md_alg, &pk_alg ) != 0 )
            {
                MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad server key exchange message" ) );
                mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                                MBEDTLS_SSL_ALERT_MSG_ILLEGAL_PARAMETER );
                return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_KEY_EXCHANGE );
            }

            if( pk_alg != mbedtls_ssl_get_ciphersuite_sig_pk_alg( ciphersuite_info ) )
            {
                MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad server key exchange message" ) );
                mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                                MBEDTLS_SSL_ALERT_MSG_ILLEGAL_PARAMETER );
                return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_KEY_EXCHANGE );
            }
        }
        else
#endif /* MBEDTLS_SSL_PROTO_TLS1_2 */
#if defined(MBEDTLS_SSL_PROTO_SSL3) || defined(MBEDTLS_SSL_PROTO_TLS1) || \
    defined(MBEDTLS_SSL_PROTO_TLS1_1)
        if( ssl->minor_ver < MBEDTLS_SSL_MINOR_VERSION_3 )
        {
            pk_alg = mbedtls_ssl_get_ciphersuite_sig_pk_alg( ciphersuite_info );

            /* Default hash for ECDSA is SHA-1 */
            if( pk_alg == MBEDTLS_PK_ECDSA && md_alg == MBEDTLS_MD_NONE )
                md_alg = MBEDTLS_MD_SHA1;
        }
        else
#endif
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "should never happen" ) );
            return( MBEDTLS_ERR_SSL_INTERNAL_ERROR );
        }

        /*
         * Read signature
         */

        if( p > end - 2 )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad server key exchange message" ) );
            mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                            MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
            return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_KEY_EXCHANGE );
        }
        sig_len = ( p[0] << 8 ) | p[1];
        p += 2;

        if( p != end - sig_len )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad server key exchange message" ) );
            mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                            MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
            return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_KEY_EXCHANGE );
        }

        MBEDTLS_SSL_DEBUG_BUF( 3, "signature", p, sig_len );

        /*
         * Compute the hash that has been signed
         */
#if defined(MBEDTLS_SSL_PROTO_SSL3) || defined(MBEDTLS_SSL_PROTO_TLS1) || \
    defined(MBEDTLS_SSL_PROTO_TLS1_1)
        if( md_alg == MBEDTLS_MD_NONE )
        {
            hashlen = 36;
            ret = mbedtls_ssl_get_key_exchange_md_ssl_tls( ssl, hash, params,
                                                           params_len );
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
                                                          params, params_len,
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

        if( ssl->session_negotiate->peer_cert == NULL )
        {
            MBEDTLS_SSL_DEBUG_MSG( 2, ( "certificate required" ) );
            mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                            MBEDTLS_SSL_ALERT_MSG_HANDSHAKE_FAILURE );
            return( MBEDTLS_ERR_SSL_UNEXPECTED_MESSAGE );
        }

        /*
         * Verify signature
         */
        if( ! mbedtls_pk_can_do( &ssl->session_negotiate->peer_cert->pk, pk_alg ) )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad server key exchange message" ) );
            mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                            MBEDTLS_SSL_ALERT_MSG_HANDSHAKE_FAILURE );
            return( MBEDTLS_ERR_SSL_PK_TYPE_MISMATCH );
        }

#if defined(MBEDTLS_SSL__ECP_RESTARTABLE)
        if( ssl->handshake->ecrs_enabled )
            rs_ctx = &ssl->handshake->ecrs_ctx.pk;
#endif

        if( ( ret = mbedtls_pk_verify_restartable(
                        &ssl->session_negotiate->peer_cert->pk,
                        md_alg, hash, hashlen, p, sig_len, rs_ctx ) ) != 0 )
        {
#if defined(MBEDTLS_SSL__ECP_RESTARTABLE)
            if( ret != MBEDTLS_ERR_ECP_IN_PROGRESS )
#endif
                mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                                MBEDTLS_SSL_ALERT_MSG_DECRYPT_ERROR );
            MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_pk_verify", ret );
#if defined(MBEDTLS_SSL__ECP_RESTARTABLE)
            if( ret == MBEDTLS_ERR_ECP_IN_PROGRESS )
                ret = MBEDTLS_ERR_SSL_CRYPTO_IN_PROGRESS;
#endif
            return( ret );
        }
    }
#endif /* MBEDTLS_KEY_EXCHANGE__WITH_SERVER_SIGNATURE__ENABLED */

exit:
    ssl->state++;

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= parse server key exchange" ) );

    return( 0 );
}

#if ! defined(MBEDTLS_KEY_EXCHANGE__CERT_REQ_ALLOWED__ENABLED)
static int ssl_parse_certificate_request( mbedtls_ssl_context *ssl )
{
    const mbedtls_ssl_ciphersuite_t *ciphersuite_info =
        ssl->transform_negotiate->ciphersuite_info;

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "=> parse certificate request" ) );

    if( ! mbedtls_ssl_ciphersuite_cert_req_allowed( ciphersuite_info ) )
    {
        MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= skip parse certificate request" ) );
        ssl->state++;
        return( 0 );
    }

    MBEDTLS_SSL_DEBUG_MSG( 1, ( "should never happen" ) );
    return( MBEDTLS_ERR_SSL_INTERNAL_ERROR );
}
#else /* MBEDTLS_KEY_EXCHANGE__CERT_REQ_ALLOWED__ENABLED */
static int ssl_parse_certificate_request( mbedtls_ssl_context *ssl )
{
    int ret;
    unsigned char *buf;
    size_t n = 0;
    size_t cert_type_len = 0, dn_len = 0;
    const mbedtls_ssl_ciphersuite_t *ciphersuite_info =
        ssl->transform_negotiate->ciphersuite_info;

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "=> parse certificate request" ) );

    if( ! mbedtls_ssl_ciphersuite_cert_req_allowed( ciphersuite_info ) )
    {
        MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= skip parse certificate request" ) );
        ssl->state++;
        return( 0 );
    }

    if( ( ret = mbedtls_ssl_read_record( ssl, 1 ) ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ssl_read_record", ret );
        return( ret );
    }

    if( ssl->in_msgtype != MBEDTLS_SSL_MSG_HANDSHAKE )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad certificate request message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_UNEXPECTED_MESSAGE );
        return( MBEDTLS_ERR_SSL_UNEXPECTED_MESSAGE );
    }

    ssl->state++;
    ssl->client_auth = ( ssl->in_msg[0] == MBEDTLS_SSL_HS_CERTIFICATE_REQUEST );

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "got %s certificate request",
                        ssl->client_auth ? "a" : "no" ) );

    if( ssl->client_auth == 0 )
    {
        /* Current message is probably the ServerHelloDone */
        ssl->keep_current_message = 1;
        goto exit;
    }

    /*
     *  struct {
     *      ClientCertificateType certificate_types<1..2^8-1>;
     *      SignatureAndHashAlgorithm
     *        supported_signature_algorithms<2^16-1>; -- TLS 1.2 only
     *      DistinguishedName certificate_authorities<0..2^16-1>;
     *  } CertificateRequest;
     *
     *  Since we only support a single certificate on clients, let's just
     *  ignore all the information that's supposed to help us pick a
     *  certificate.
     *
     *  We could check that our certificate matches the request, and bail out
     *  if it doesn't, but it's simpler to just send the certificate anyway,
     *  and give the server the opportunity to decide if it should terminate
     *  the connection when it doesn't like our certificate.
     *
     *  Same goes for the hash in TLS 1.2's signature_algorithms: at this
     *  point we only have one hash available (see comments in
     *  write_certificate_verify), so let's just use what we have.
     *
     *  However, we still minimally parse the message to check it is at least
     *  superficially sane.
     */
    buf = ssl->in_msg;

    /* certificate_types */
    if( ssl->in_hslen <= mbedtls_ssl_hs_hdr_len( ssl ) )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad certificate request message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
        return( MBEDTLS_ERR_SSL_BAD_HS_CERTIFICATE_REQUEST );
    }
    cert_type_len = buf[mbedtls_ssl_hs_hdr_len( ssl )];
    n = cert_type_len;

    /*
     * In the subsequent code there are two paths that read from buf:
     *     * the length of the signature algorithms field (if minor version of
     *       SSL is 3),
     *     * distinguished name length otherwise.
     * Both reach at most the index:
     *    ...hdr_len + 2 + n,
     * therefore the buffer length at this point must be greater than that
     * regardless of the actual code path.
     */
    if( ssl->in_hslen <= mbedtls_ssl_hs_hdr_len( ssl ) + 2 + n )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad certificate request message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
        return( MBEDTLS_ERR_SSL_BAD_HS_CERTIFICATE_REQUEST );
    }

    /* supported_signature_algorithms */
#if defined(MBEDTLS_SSL_PROTO_TLS1_2)
    if( ssl->minor_ver == MBEDTLS_SSL_MINOR_VERSION_3 )
    {
        size_t sig_alg_len = ( ( buf[mbedtls_ssl_hs_hdr_len( ssl ) + 1 + n] <<  8 )
                             | ( buf[mbedtls_ssl_hs_hdr_len( ssl ) + 2 + n]       ) );
#if defined(MBEDTLS_DEBUG_C)
        unsigned char* sig_alg;
        size_t i;
#endif

        /*
         * The furthest access in buf is in the loop few lines below:
         *     sig_alg[i + 1],
         * where:
         *     sig_alg = buf + ...hdr_len + 3 + n,
         *     max(i) = sig_alg_len - 1.
         * Therefore the furthest access is:
         *     buf[...hdr_len + 3 + n + sig_alg_len - 1 + 1],
         * which reduces to:
         *     buf[...hdr_len + 3 + n + sig_alg_len],
         * which is one less than we need the buf to be.
         */
        if( ssl->in_hslen <= mbedtls_ssl_hs_hdr_len( ssl ) + 3 + n + sig_alg_len )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad certificate request message" ) );
            mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                            MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
            return( MBEDTLS_ERR_SSL_BAD_HS_CERTIFICATE_REQUEST );
        }

#if defined(MBEDTLS_DEBUG_C)
        sig_alg = buf + mbedtls_ssl_hs_hdr_len( ssl ) + 3 + n;
        for( i = 0; i < sig_alg_len; i += 2 )
        {
            MBEDTLS_SSL_DEBUG_MSG( 3, ( "Supported Signature Algorithm found: %d"
                                        ",%d", sig_alg[i], sig_alg[i + 1]  ) );
        }
#endif

        n += 2 + sig_alg_len;
    }
#endif /* MBEDTLS_SSL_PROTO_TLS1_2 */

    /* certificate_authorities */
    dn_len = ( ( buf[mbedtls_ssl_hs_hdr_len( ssl ) + 1 + n] <<  8 )
             | ( buf[mbedtls_ssl_hs_hdr_len( ssl ) + 2 + n]       ) );

    n += dn_len;
    if( ssl->in_hslen != mbedtls_ssl_hs_hdr_len( ssl ) + 3 + n )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad certificate request message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
        return( MBEDTLS_ERR_SSL_BAD_HS_CERTIFICATE_REQUEST );
    }

exit:
    MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= parse certificate request" ) );

    return( 0 );
}
#endif /* MBEDTLS_KEY_EXCHANGE__CERT_REQ_ALLOWED__ENABLED */

static int ssl_parse_server_hello_done( mbedtls_ssl_context *ssl )
{
    int ret;

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "=> parse server hello done" ) );

    if( ( ret = mbedtls_ssl_read_record( ssl, 1 ) ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ssl_read_record", ret );
        return( ret );
    }

    if( ssl->in_msgtype != MBEDTLS_SSL_MSG_HANDSHAKE )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad server hello done message" ) );
        return( MBEDTLS_ERR_SSL_UNEXPECTED_MESSAGE );
    }

    if( ssl->in_hslen  != mbedtls_ssl_hs_hdr_len( ssl ) ||
        ssl->in_msg[0] != MBEDTLS_SSL_HS_SERVER_HELLO_DONE )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad server hello done message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
        return( MBEDTLS_ERR_SSL_BAD_HS_SERVER_HELLO_DONE );
    }

    ssl->state++;

#if defined(MBEDTLS_SSL_PROTO_DTLS)
    if( ssl->conf->transport == MBEDTLS_SSL_TRANSPORT_DATAGRAM )
        mbedtls_ssl_recv_flight_completed( ssl );
#endif

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= parse server hello done" ) );

    return( 0 );
}

static int ssl_write_client_key_exchange( mbedtls_ssl_context *ssl )
{
    int ret;
    size_t i, n;
    const mbedtls_ssl_ciphersuite_t *ciphersuite_info =
        ssl->transform_negotiate->ciphersuite_info;

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "=> write client key exchange" ) );

#if defined(MBEDTLS_KEY_EXCHANGE_DHE_RSA_ENABLED)
    if( ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_DHE_RSA )
    {
        /*
         * DHM key exchange -- send G^X mod P
         */
        n = ssl->handshake->dhm_ctx.len;

        ssl->out_msg[4] = (unsigned char)( n >> 8 );
        ssl->out_msg[5] = (unsigned char)( n      );
        i = 6;

        ret = mbedtls_dhm_make_public( &ssl->handshake->dhm_ctx,
                                (int) mbedtls_mpi_size( &ssl->handshake->dhm_ctx.P ),
                               &ssl->out_msg[i], n,
                                ssl->conf->f_rng, ssl->conf->p_rng );
        if( ret != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_dhm_make_public", ret );
            return( ret );
        }

        MBEDTLS_SSL_DEBUG_MPI( 3, "DHM: X ", &ssl->handshake->dhm_ctx.X  );
        MBEDTLS_SSL_DEBUG_MPI( 3, "DHM: GX", &ssl->handshake->dhm_ctx.GX );

        if( ( ret = mbedtls_dhm_calc_secret( &ssl->handshake->dhm_ctx,
                                      ssl->handshake->premaster,
                                      MBEDTLS_PREMASTER_SIZE,
                                     &ssl->handshake->pmslen,
                                      ssl->conf->f_rng, ssl->conf->p_rng ) ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_dhm_calc_secret", ret );
            return( ret );
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
        /*
         * ECDH key exchange -- send client public value
         */
        i = 4;

#if defined(MBEDTLS_SSL__ECP_RESTARTABLE)
        if( ssl->handshake->ecrs_enabled )
        {
            if( ssl->handshake->ecrs_state == ssl_ecrs_cke_ecdh_calc_secret )
                goto ecdh_calc_secret;

            mbedtls_ecdh_enable_restart( &ssl->handshake->ecdh_ctx );
        }
#endif

        ret = mbedtls_ecdh_make_public( &ssl->handshake->ecdh_ctx,
                                &n,
                                &ssl->out_msg[i], 1000,
                                ssl->conf->f_rng, ssl->conf->p_rng );
        if( ret != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ecdh_make_public", ret );
#if defined(MBEDTLS_SSL__ECP_RESTARTABLE)
            if( ret == MBEDTLS_ERR_ECP_IN_PROGRESS )
                ret = MBEDTLS_ERR_SSL_CRYPTO_IN_PROGRESS;
#endif
            return( ret );
        }

        MBEDTLS_SSL_DEBUG_ECDH( 3, &ssl->handshake->ecdh_ctx,
                                MBEDTLS_DEBUG_ECDH_Q );

#if defined(MBEDTLS_SSL__ECP_RESTARTABLE)
        if( ssl->handshake->ecrs_enabled )
        {
            ssl->handshake->ecrs_n = n;
            ssl->handshake->ecrs_state = ssl_ecrs_cke_ecdh_calc_secret;
        }

ecdh_calc_secret:
        if( ssl->handshake->ecrs_enabled )
            n = ssl->handshake->ecrs_n;
#endif
        if( ( ret = mbedtls_ecdh_calc_secret( &ssl->handshake->ecdh_ctx,
                                      &ssl->handshake->pmslen,
                                       ssl->handshake->premaster,
                                       MBEDTLS_MPI_MAX_SIZE,
                                       ssl->conf->f_rng, ssl->conf->p_rng ) ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ecdh_calc_secret", ret );
#if defined(MBEDTLS_SSL__ECP_RESTARTABLE)
            if( ret == MBEDTLS_ERR_ECP_IN_PROGRESS )
                ret = MBEDTLS_ERR_SSL_CRYPTO_IN_PROGRESS;
#endif
            return( ret );
        }

        MBEDTLS_SSL_DEBUG_ECDH( 3, &ssl->handshake->ecdh_ctx,
                                MBEDTLS_DEBUG_ECDH_Z );
    }
    else
#endif /* MBEDTLS_KEY_EXCHANGE_ECDHE_RSA_ENABLED ||
          MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA_ENABLED ||
          MBEDTLS_KEY_EXCHANGE_ECDH_RSA_ENABLED ||
          MBEDTLS_KEY_EXCHANGE_ECDH_ECDSA_ENABLED */
#if defined(MBEDTLS_KEY_EXCHANGE__SOME__PSK_ENABLED)
    if( mbedtls_ssl_ciphersuite_uses_psk( ciphersuite_info ) )
    {
        /*
         * opaque psk_identity<0..2^16-1>;
         */
        if( ssl->conf->psk == NULL || ssl->conf->psk_identity == NULL )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "got no private key for PSK" ) );
            return( MBEDTLS_ERR_SSL_PRIVATE_KEY_REQUIRED );
        }

        i = 4;
        n = ssl->conf->psk_identity_len;

        if( i + 2 + n > MBEDTLS_SSL_OUT_CONTENT_LEN )
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "psk identity too long or "
                                        "SSL buffer too short" ) );
            return( MBEDTLS_ERR_SSL_BUFFER_TOO_SMALL );
        }

        ssl->out_msg[i++] = (unsigned char)( n >> 8 );
        ssl->out_msg[i++] = (unsigned char)( n      );

        memcpy( ssl->out_msg + i, ssl->conf->psk_identity, ssl->conf->psk_identity_len );
        i += ssl->conf->psk_identity_len;

#if defined(MBEDTLS_KEY_EXCHANGE_PSK_ENABLED)
        if( ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_PSK )
        {
            n = 0;
        }
        else
#endif
#if defined(MBEDTLS_KEY_EXCHANGE_RSA_PSK_ENABLED)
        if( ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_RSA_PSK )
        {
            if( ( ret = ssl_write_encrypted_pms( ssl, i, &n, 2 ) ) != 0 )
                return( ret );
        }
        else
#endif
#if defined(MBEDTLS_KEY_EXCHANGE_DHE_PSK_ENABLED)
        if( ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_DHE_PSK )
        {
            /*
             * ClientDiffieHellmanPublic public (DHM send G^X mod P)
             */
            n = ssl->handshake->dhm_ctx.len;

            if( i + 2 + n > MBEDTLS_SSL_OUT_CONTENT_LEN )
            {
                MBEDTLS_SSL_DEBUG_MSG( 1, ( "psk identity or DHM size too long"
                                            " or SSL buffer too short" ) );
                return( MBEDTLS_ERR_SSL_BUFFER_TOO_SMALL );
            }

            ssl->out_msg[i++] = (unsigned char)( n >> 8 );
            ssl->out_msg[i++] = (unsigned char)( n      );

            ret = mbedtls_dhm_make_public( &ssl->handshake->dhm_ctx,
                    (int) mbedtls_mpi_size( &ssl->handshake->dhm_ctx.P ),
                    &ssl->out_msg[i], n,
                    ssl->conf->f_rng, ssl->conf->p_rng );
            if( ret != 0 )
            {
                MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_dhm_make_public", ret );
                return( ret );
            }
        }
        else
#endif /* MBEDTLS_KEY_EXCHANGE_DHE_PSK_ENABLED */
#if defined(MBEDTLS_KEY_EXCHANGE_ECDHE_PSK_ENABLED)
        if( ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_ECDHE_PSK )
        {
            /*
             * ClientECDiffieHellmanPublic public;
             */
            ret = mbedtls_ecdh_make_public( &ssl->handshake->ecdh_ctx, &n,
                    &ssl->out_msg[i], MBEDTLS_SSL_OUT_CONTENT_LEN - i,
                    ssl->conf->f_rng, ssl->conf->p_rng );
            if( ret != 0 )
            {
                MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ecdh_make_public", ret );
                return( ret );
            }

            MBEDTLS_SSL_DEBUG_ECDH( 3, &ssl->handshake->ecdh_ctx,
                                    MBEDTLS_DEBUG_ECDH_Q );
        }
        else
#endif /* MBEDTLS_KEY_EXCHANGE_ECDHE_PSK_ENABLED */
        {
            MBEDTLS_SSL_DEBUG_MSG( 1, ( "should never happen" ) );
            return( MBEDTLS_ERR_SSL_INTERNAL_ERROR );
        }

        if( ( ret = mbedtls_ssl_psk_derive_premaster( ssl,
                        ciphersuite_info->key_exchange ) ) != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ssl_psk_derive_premaster", ret );
            return( ret );
        }
    }
    else
#endif /* MBEDTLS_KEY_EXCHANGE__SOME__PSK_ENABLED */
#if defined(MBEDTLS_KEY_EXCHANGE_RSA_ENABLED)
    if( ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_RSA )
    {
        i = 4;
        if( ( ret = ssl_write_encrypted_pms( ssl, i, &n, 0 ) ) != 0 )
            return( ret );
    }
    else
#endif /* MBEDTLS_KEY_EXCHANGE_RSA_ENABLED */
#if defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
    if( ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_ECJPAKE )
    {
        i = 4;

        ret = mbedtls_ecjpake_write_round_two( &ssl->handshake->ecjpake_ctx,
                ssl->out_msg + i, MBEDTLS_SSL_OUT_CONTENT_LEN - i, &n,
                ssl->conf->f_rng, ssl->conf->p_rng );
        if( ret != 0 )
        {
            MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ecjpake_write_round_two", ret );
            return( ret );
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
#endif /* MBEDTLS_KEY_EXCHANGE_RSA_ENABLED */
    {
        ((void) ciphersuite_info);
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "should never happen" ) );
        return( MBEDTLS_ERR_SSL_INTERNAL_ERROR );
    }

    ssl->out_msglen  = i + n;
    ssl->out_msgtype = MBEDTLS_SSL_MSG_HANDSHAKE;
    ssl->out_msg[0]  = MBEDTLS_SSL_HS_CLIENT_KEY_EXCHANGE;

    ssl->state++;

    if( ( ret = mbedtls_ssl_write_handshake_msg( ssl ) ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ssl_write_handshake_msg", ret );
        return( ret );
    }

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= write client key exchange" ) );

    return( 0 );
}

#if !defined(MBEDTLS_KEY_EXCHANGE_RSA_ENABLED)       && \
    !defined(MBEDTLS_KEY_EXCHANGE_DHE_RSA_ENABLED)   && \
    !defined(MBEDTLS_KEY_EXCHANGE_ECDH_RSA_ENABLED)  && \
    !defined(MBEDTLS_KEY_EXCHANGE_ECDHE_RSA_ENABLED) && \
    !defined(MBEDTLS_KEY_EXCHANGE_ECDH_ECDSA_ENABLED)&& \
    !defined(MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA_ENABLED)
static int ssl_write_certificate_verify( mbedtls_ssl_context *ssl )
{
    const mbedtls_ssl_ciphersuite_t *ciphersuite_info =
        ssl->transform_negotiate->ciphersuite_info;
    int ret;

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "=> write certificate verify" ) );

    if( ( ret = mbedtls_ssl_derive_keys( ssl ) ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ssl_derive_keys", ret );
        return( ret );
    }

    if( ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_PSK ||
        ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_RSA_PSK ||
        ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_ECDHE_PSK ||
        ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_DHE_PSK ||
        ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_ECJPAKE )
    {
        MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= skip write certificate verify" ) );
        ssl->state++;
        return( 0 );
    }

    MBEDTLS_SSL_DEBUG_MSG( 1, ( "should never happen" ) );
    return( MBEDTLS_ERR_SSL_INTERNAL_ERROR );
}
#else
static int ssl_write_certificate_verify( mbedtls_ssl_context *ssl )
{
    int ret = MBEDTLS_ERR_SSL_FEATURE_UNAVAILABLE;
    const mbedtls_ssl_ciphersuite_t *ciphersuite_info =
        ssl->transform_negotiate->ciphersuite_info;
    size_t n = 0, offset = 0;
    unsigned char hash[48];
    unsigned char *hash_start = hash;
    mbedtls_md_type_t md_alg = MBEDTLS_MD_NONE;
    unsigned int hashlen;
    void *rs_ctx = NULL;

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "=> write certificate verify" ) );

#if defined(MBEDTLS_SSL__ECP_RESTARTABLE)
    if( ssl->handshake->ecrs_enabled &&
        ssl->handshake->ecrs_state == ssl_ecrs_crt_vrfy_sign )
    {
        goto sign;
    }
#endif

    if( ( ret = mbedtls_ssl_derive_keys( ssl ) ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ssl_derive_keys", ret );
        return( ret );
    }

    if( ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_PSK ||
        ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_RSA_PSK ||
        ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_ECDHE_PSK ||
        ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_DHE_PSK ||
        ciphersuite_info->key_exchange == MBEDTLS_KEY_EXCHANGE_ECJPAKE )
    {
        MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= skip write certificate verify" ) );
        ssl->state++;
        return( 0 );
    }

    if( ssl->client_auth == 0 || mbedtls_ssl_own_cert( ssl ) == NULL )
    {
        MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= skip write certificate verify" ) );
        ssl->state++;
        return( 0 );
    }

    if( mbedtls_ssl_own_key( ssl ) == NULL )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "got no private key for certificate" ) );
        return( MBEDTLS_ERR_SSL_PRIVATE_KEY_REQUIRED );
    }

    /*
     * Make a signature of the handshake digests
     */
#if defined(MBEDTLS_SSL__ECP_RESTARTABLE)
    if( ssl->handshake->ecrs_enabled )
        ssl->handshake->ecrs_state = ssl_ecrs_crt_vrfy_sign;

sign:
#endif

    ssl->handshake->calc_verify( ssl, hash );

#if defined(MBEDTLS_SSL_PROTO_SSL3) || defined(MBEDTLS_SSL_PROTO_TLS1) || \
    defined(MBEDTLS_SSL_PROTO_TLS1_1)
    if( ssl->minor_ver != MBEDTLS_SSL_MINOR_VERSION_3 )
    {
        /*
         * digitally-signed struct {
         *     opaque md5_hash[16];
         *     opaque sha_hash[20];
         * };
         *
         * md5_hash
         *     MD5(handshake_messages);
         *
         * sha_hash
         *     SHA(handshake_messages);
         */
        hashlen = 36;
        md_alg = MBEDTLS_MD_NONE;

        /*
         * For ECDSA, default hash is SHA-1 only
         */
        if( mbedtls_pk_can_do( mbedtls_ssl_own_key( ssl ), MBEDTLS_PK_ECDSA ) )
        {
            hash_start += 16;
            hashlen -= 16;
            md_alg = MBEDTLS_MD_SHA1;
        }
    }
    else
#endif /* MBEDTLS_SSL_PROTO_SSL3 || MBEDTLS_SSL_PROTO_TLS1 || \
          MBEDTLS_SSL_PROTO_TLS1_1 */
#if defined(MBEDTLS_SSL_PROTO_TLS1_2)
    if( ssl->minor_ver == MBEDTLS_SSL_MINOR_VERSION_3 )
    {
        /*
         * digitally-signed struct {
         *     opaque handshake_messages[handshake_messages_length];
         * };
         *
         * Taking shortcut here. We assume that the server always allows the
         * PRF Hash function and has sent it in the allowed signature
         * algorithms list received in the Certificate Request message.
         *
         * Until we encounter a server that does not, we will take this
         * shortcut.
         *
         * Reason: Otherwise we should have running hashes for SHA512 and SHA224
         *         in order to satisfy 'weird' needs from the server side.
         */
        if( ssl->transform_negotiate->ciphersuite_info->mac ==
            MBEDTLS_MD_SHA384 )
        {
            md_alg = MBEDTLS_MD_SHA384;
            ssl->out_msg[4] = MBEDTLS_SSL_HASH_SHA384;
        }
        else
        {
            md_alg = MBEDTLS_MD_SHA256;
            ssl->out_msg[4] = MBEDTLS_SSL_HASH_SHA256;
        }
        ssl->out_msg[5] = mbedtls_ssl_sig_from_pk( mbedtls_ssl_own_key( ssl ) );

        /* Info from md_alg will be used instead */
        hashlen = 0;
        offset = 2;
    }
    else
#endif /* MBEDTLS_SSL_PROTO_TLS1_2 */
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "should never happen" ) );
        return( MBEDTLS_ERR_SSL_INTERNAL_ERROR );
    }

#if defined(MBEDTLS_SSL__ECP_RESTARTABLE)
    if( ssl->handshake->ecrs_enabled )
        rs_ctx = &ssl->handshake->ecrs_ctx.pk;
#endif

    if( ( ret = mbedtls_pk_sign_restartable( mbedtls_ssl_own_key( ssl ),
                         md_alg, hash_start, hashlen,
                         ssl->out_msg + 6 + offset, &n,
                         ssl->conf->f_rng, ssl->conf->p_rng, rs_ctx ) ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_pk_sign", ret );
#if defined(MBEDTLS_SSL__ECP_RESTARTABLE)
        if( ret == MBEDTLS_ERR_ECP_IN_PROGRESS )
            ret = MBEDTLS_ERR_SSL_CRYPTO_IN_PROGRESS;
#endif
        return( ret );
    }

    ssl->out_msg[4 + offset] = (unsigned char)( n >> 8 );
    ssl->out_msg[5 + offset] = (unsigned char)( n      );

    ssl->out_msglen  = 6 + n + offset;
    ssl->out_msgtype = MBEDTLS_SSL_MSG_HANDSHAKE;
    ssl->out_msg[0]  = MBEDTLS_SSL_HS_CERTIFICATE_VERIFY;

    ssl->state++;

    if( ( ret = mbedtls_ssl_write_handshake_msg( ssl ) ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ssl_write_handshake_msg", ret );
        return( ret );
    }

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= write certificate verify" ) );

    return( ret );
}
#endif /* !MBEDTLS_KEY_EXCHANGE_RSA_ENABLED &&
          !MBEDTLS_KEY_EXCHANGE_DHE_RSA_ENABLED &&
          !MBEDTLS_KEY_EXCHANGE_ECDH_RSA_ENABLED &&
          !MBEDTLS_KEY_EXCHANGE_ECDHE_RSA_ENABLED &&
          !MBEDTLS_KEY_EXCHANGE_ECDH_ECDSA_ENABLED &&
          !MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA_ENABLED */

#if defined(MBEDTLS_SSL_SESSION_TICKETS)
static int ssl_parse_new_session_ticket( mbedtls_ssl_context *ssl )
{
    int ret;
    uint32_t lifetime;
    size_t ticket_len;
    unsigned char *ticket;
    const unsigned char *msg;

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "=> parse new session ticket" ) );

    if( ( ret = mbedtls_ssl_read_record( ssl, 1 ) ) != 0 )
    {
        MBEDTLS_SSL_DEBUG_RET( 1, "mbedtls_ssl_read_record", ret );
        return( ret );
    }

    if( ssl->in_msgtype != MBEDTLS_SSL_MSG_HANDSHAKE )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad new session ticket message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_UNEXPECTED_MESSAGE );
        return( MBEDTLS_ERR_SSL_UNEXPECTED_MESSAGE );
    }

    /*
     * struct {
     *     uint32 ticket_lifetime_hint;
     *     opaque ticket<0..2^16-1>;
     * } NewSessionTicket;
     *
     * 0  .  3   ticket_lifetime_hint
     * 4  .  5   ticket_len (n)
     * 6  .  5+n ticket content
     */
    if( ssl->in_msg[0] != MBEDTLS_SSL_HS_NEW_SESSION_TICKET ||
        ssl->in_hslen < 6 + mbedtls_ssl_hs_hdr_len( ssl ) )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad new session ticket message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
        return( MBEDTLS_ERR_SSL_BAD_HS_NEW_SESSION_TICKET );
    }

    msg = ssl->in_msg + mbedtls_ssl_hs_hdr_len( ssl );

    lifetime = ( ((uint32_t) msg[0]) << 24 ) | ( msg[1] << 16 ) |
               ( msg[2] << 8 ) | ( msg[3] );

    ticket_len = ( msg[4] << 8 ) | ( msg[5] );

    if( ticket_len + 6 + mbedtls_ssl_hs_hdr_len( ssl ) != ssl->in_hslen )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "bad new session ticket message" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR );
        return( MBEDTLS_ERR_SSL_BAD_HS_NEW_SESSION_TICKET );
    }

    MBEDTLS_SSL_DEBUG_MSG( 3, ( "ticket length: %d", ticket_len ) );

    /* We're not waiting for a NewSessionTicket message any more */
    ssl->handshake->new_session_ticket = 0;
    ssl->state = MBEDTLS_SSL_SERVER_CHANGE_CIPHER_SPEC;

    /*
     * Zero-length ticket means the server changed his mind and doesn't want
     * to send a ticket after all, so just forget it
     */
    if( ticket_len == 0 )
        return( 0 );

    mbedtls_platform_zeroize( ssl->session_negotiate->ticket,
                              ssl->session_negotiate->ticket_len );
    mbedtls_free( ssl->session_negotiate->ticket );
    ssl->session_negotiate->ticket = NULL;
    ssl->session_negotiate->ticket_len = 0;

    if( ( ticket = mbedtls_calloc( 1, ticket_len ) ) == NULL )
    {
        MBEDTLS_SSL_DEBUG_MSG( 1, ( "ticket alloc failed" ) );
        mbedtls_ssl_send_alert_message( ssl, MBEDTLS_SSL_ALERT_LEVEL_FATAL,
                                        MBEDTLS_SSL_ALERT_MSG_INTERNAL_ERROR );
        return( MBEDTLS_ERR_SSL_ALLOC_FAILED );
    }

    memcpy( ticket, msg + 6, ticket_len );

    ssl->session_negotiate->ticket = ticket;
    ssl->session_negotiate->ticket_len = ticket_len;
    ssl->session_negotiate->ticket_lifetime = lifetime;

    /*
     * RFC 5077 section 3.4:
     * "If the client receives a session ticket from the server, then it
     * discards any Session ID that was sent in the ServerHello."
     */
    MBEDTLS_SSL_DEBUG_MSG( 3, ( "ticket in use, discarding session id" ) );
    ssl->session_negotiate->id_len = 0;

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "<= parse new session ticket" ) );

    return( 0 );
}
#endif /* MBEDTLS_SSL_SESSION_TICKETS */

/*
 * SSL handshake -- client side -- single step
 */
int mbedtls_ssl_handshake_client_step( mbedtls_ssl_context *ssl )
{
    int ret = 0;

    if( ssl->state == MBEDTLS_SSL_HANDSHAKE_OVER || ssl->handshake == NULL )
        return( MBEDTLS_ERR_SSL_BAD_INPUT_DATA );

    MBEDTLS_SSL_DEBUG_MSG( 2, ( "client state: %d", ssl->state ) );

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

    /* Change state now, so that it is right in mbedtls_ssl_read_record(), used
     * by DTLS for dropping out-of-sequence ChangeCipherSpec records */
#if defined(MBEDTLS_SSL_SESSION_TICKETS)
    if( ssl->state == MBEDTLS_SSL_SERVER_CHANGE_CIPHER_SPEC &&
        ssl->handshake->new_session_ticket != 0 )
    {
        ssl->state = MBEDTLS_SSL_SERVER_NEW_SESSION_TICKET;
    }
#endif

    switch( ssl->state )
    {
        case MBEDTLS_SSL_HELLO_REQUEST:
            ssl->state = MBEDTLS_SSL_CLIENT_HELLO;
            break;

       /*
        *  ==>   ClientHello
        */
       case MBEDTLS_SSL_CLIENT_HELLO:
           ret = ssl_write_client_hello( ssl );
           break;

       /*
        *  <==   ServerHello
        *        Certificate
        *      ( ServerKeyExchange  )
        *      ( CertificateRequest )
        *        ServerHelloDone
        */
       case MBEDTLS_SSL_SERVER_HELLO:
           ret = ssl_parse_server_hello( ssl );
           break;

       case MBEDTLS_SSL_SERVER_CERTIFICATE:
           ret = mbedtls_ssl_parse_certificate( ssl );
           break;

       case MBEDTLS_SSL_SERVER_KEY_EXCHANGE:
           ret = ssl_parse_server_key_exchange( ssl );
           break;

       case MBEDTLS_SSL_CERTIFICATE_REQUEST:
           ret = ssl_parse_certificate_request( ssl );
           break;

       case MBEDTLS_SSL_SERVER_HELLO_DONE:
           ret = ssl_parse_server_hello_done( ssl );
           break;

       /*
        *  ==> ( Certificate/Alert  )
        *        ClientKeyExchange
        *      ( CertificateVerify  )
        *        ChangeCipherSpec
        *        Finished
        */
       case MBEDTLS_SSL_CLIENT_CERTIFICATE:
           ret = mbedtls_ssl_write_certificate( ssl );
           break;

       case MBEDTLS_SSL_CLIENT_KEY_EXCHANGE:
           ret = ssl_write_client_key_exchange( ssl );
           break;

       case MBEDTLS_SSL_CERTIFICATE_VERIFY:
           ret = ssl_write_certificate_verify( ssl );
           break;

       case MBEDTLS_SSL_CLIENT_CHANGE_CIPHER_SPEC:
           ret = mbedtls_ssl_write_change_cipher_spec( ssl );
           break;

       case MBEDTLS_SSL_CLIENT_FINISHED:
           ret = mbedtls_ssl_write_finished( ssl );
           break;

       /*
        *  <==   ( NewSessionTicket )
        *        ChangeCipherSpec
        *        Finished
        */
#if defined(MBEDTLS_SSL_SESSION_TICKETS)
       case MBEDTLS_SSL_SERVER_NEW_SESSION_TICKET:
           ret = ssl_parse_new_session_ticket( ssl );
           break;
#endif

       case MBEDTLS_SSL_SERVER_CHANGE_CIPHER_SPEC:
           ret = mbedtls_ssl_parse_change_cipher_spec( ssl );
           break;

       case MBEDTLS_SSL_SERVER_FINISHED:
           ret = mbedtls_ssl_parse_finished( ssl );
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
#endif /* MBEDTLS_SSL_CLI_C */
