#ifndef HEADER_CURL_GSSAPI_H
#define HEADER_CURL_GSSAPI_H
/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 2011 - 2020, Daniel Stenberg, <daniel@haxx.se>, et al.
 *
 * This software is licensed as described in the file COPYING, which
 * you should have received as part of this distribution. The terms
 * are also available at https://curl.se/docs/copyright.html.
 *
 * You may opt to use, copy, modify, merge, publish, distribute and/or sell
 * copies of the Software, and permit persons to whom the Software is
 * furnished to do so, under the terms of the COPYING file.
 *
 * This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
 * KIND, either express or implied.
 *
 ***************************************************************************/

#include "curl_setup.h"
#include "urldata.h"

#ifdef HAVE_GSSAPI
extern gss_OID_desc Curl_spnego_mech_oid;
extern gss_OID_desc Curl_krb5_mech_oid;

/* Common method for using GSS-API */
OM_uint32 Curl_gss_init_sec_context(
    struct Curl_easy *data,
    OM_uint32 *minor_status,
    gss_ctx_id_t *context,
    gss_name_t target_name,
    gss_OID mech_type,
    gss_channel_bindings_t input_chan_bindings,
    gss_buffer_t input_token,
    gss_buffer_t output_token,
    const bool mutual_auth,
    OM_uint32 *ret_flags);

/* Helper to log a GSS-API error status */
void Curl_gss_log_error(struct Curl_easy *data, const char *prefix,
                        OM_uint32 major, OM_uint32 minor);

/* Provide some definitions missing in old headers */
#ifdef HAVE_OLD_GSSMIT
#define GSS_C_NT_HOSTBASED_SERVICE gss_nt_service_name
#define NCOMPAT 1
#endif

/* Define our privacy and integrity protection values */
#define GSSAUTH_P_NONE      1
#define GSSAUTH_P_INTEGRITY 2
#define GSSAUTH_P_PRIVACY   4

#endif /* HAVE_GSSAPI */
#endif /* HEADER_CURL_GSSAPI_H */
