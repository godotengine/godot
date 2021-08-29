#ifndef HEADER_CURL_SETUP_OS400_H
#define HEADER_CURL_SETUP_OS400_H
/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 1998 - 2020, Daniel Stenberg, <daniel@haxx.se>, et al.
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


/* OS/400 netdb.h does not define NI_MAXHOST. */
#define NI_MAXHOST      1025

/* OS/400 netdb.h does not define NI_MAXSERV. */
#define NI_MAXSERV      32

/* No OS/400 header file defines u_int32_t. */
typedef unsigned long   u_int32_t;


/* System API wrapper prototypes & definitions to support ASCII parameters. */

#include <sys/socket.h>
#include <netdb.h>
#include <gskssl.h>
#include <qsoasync.h>
#include <gssapi.h>

extern int Curl_getaddrinfo_a(const char *nodename,
                              const char *servname,
                              const struct addrinfo *hints,
                              struct addrinfo **res);
#define getaddrinfo             Curl_getaddrinfo_a


extern int Curl_getnameinfo_a(const struct sockaddr *sa,
                              curl_socklen_t salen,
                              char *nodename, curl_socklen_t nodenamelen,
                              char *servname, curl_socklen_t servnamelen,
                              int flags);
#define getnameinfo             Curl_getnameinfo_a


/* GSKit wrappers. */

extern int      Curl_gsk_environment_open(gsk_handle * my_env_handle);
#define gsk_environment_open    Curl_gsk_environment_open

extern int      Curl_gsk_secure_soc_open(gsk_handle my_env_handle,
                                         gsk_handle * my_session_handle);
#define gsk_secure_soc_open     Curl_gsk_secure_soc_open

extern int      Curl_gsk_environment_close(gsk_handle * my_env_handle);
#define gsk_environment_close   Curl_gsk_environment_close

extern int      Curl_gsk_secure_soc_close(gsk_handle * my_session_handle);
#define gsk_secure_soc_close    Curl_gsk_secure_soc_close

extern int      Curl_gsk_environment_init(gsk_handle my_env_handle);
#define gsk_environment_init    Curl_gsk_environment_init

extern int      Curl_gsk_secure_soc_init(gsk_handle my_session_handle);
#define gsk_secure_soc_init     Curl_gsk_secure_soc_init

extern int      Curl_gsk_attribute_set_buffer_a(gsk_handle my_gsk_handle,
                                                GSK_BUF_ID bufID,
                                                const char *buffer,
                                                int bufSize);
#define gsk_attribute_set_buffer        Curl_gsk_attribute_set_buffer_a

extern int      Curl_gsk_attribute_set_enum(gsk_handle my_gsk_handle,
                                            GSK_ENUM_ID enumID,
                                            GSK_ENUM_VALUE enumValue);
#define gsk_attribute_set_enum  Curl_gsk_attribute_set_enum

extern int      Curl_gsk_attribute_set_numeric_value(gsk_handle my_gsk_handle,
                                                     GSK_NUM_ID numID,
                                                     int numValue);
#define gsk_attribute_set_numeric_value Curl_gsk_attribute_set_numeric_value

extern int      Curl_gsk_attribute_set_callback(gsk_handle my_gsk_handle,
                                                GSK_CALLBACK_ID callBackID,
                                                void *callBackAreaPtr);
#define gsk_attribute_set_callback      Curl_gsk_attribute_set_callback

extern int      Curl_gsk_attribute_get_buffer_a(gsk_handle my_gsk_handle,
                                                GSK_BUF_ID bufID,
                                                const char **buffer,
                                                int *bufSize);
#define gsk_attribute_get_buffer        Curl_gsk_attribute_get_buffer_a

extern int      Curl_gsk_attribute_get_enum(gsk_handle my_gsk_handle,
                                            GSK_ENUM_ID enumID,
                                            GSK_ENUM_VALUE *enumValue);
#define gsk_attribute_get_enum  Curl_gsk_attribute_get_enum

extern int      Curl_gsk_attribute_get_numeric_value(gsk_handle my_gsk_handle,
                                                     GSK_NUM_ID numID,
                                                     int *numValue);
#define gsk_attribute_get_numeric_value Curl_gsk_attribute_get_numeric_value

extern int      Curl_gsk_attribute_get_cert_info(gsk_handle my_gsk_handle,
                                 GSK_CERT_ID certID,
                                 const gsk_cert_data_elem **certDataElem,
                                 int *certDataElementCount);
#define gsk_attribute_get_cert_info     Curl_gsk_attribute_get_cert_info

extern int      Curl_gsk_secure_soc_misc(gsk_handle my_session_handle,
                                         GSK_MISC_ID miscID);
#define gsk_secure_soc_misc     Curl_gsk_secure_soc_misc

extern int      Curl_gsk_secure_soc_read(gsk_handle my_session_handle,
                                         char *readBuffer,
                                         int readBufSize, int *amtRead);
#define gsk_secure_soc_read     Curl_gsk_secure_soc_read

extern int      Curl_gsk_secure_soc_write(gsk_handle my_session_handle,
                                          char *writeBuffer,
                                          int writeBufSize, int *amtWritten);
#define gsk_secure_soc_write    Curl_gsk_secure_soc_write

extern const char *     Curl_gsk_strerror_a(int gsk_return_value);
#define gsk_strerror    Curl_gsk_strerror_a

extern int      Curl_gsk_secure_soc_startInit(gsk_handle my_session_handle,
                                      int IOCompletionPort,
                                      Qso_OverlappedIO_t * communicationsArea);
#define gsk_secure_soc_startInit        Curl_gsk_secure_soc_startInit


/* GSSAPI wrappers. */

extern OM_uint32 Curl_gss_import_name_a(OM_uint32 * minor_status,
                                        gss_buffer_t in_name,
                                        gss_OID in_name_type,
                                        gss_name_t * out_name);
#define gss_import_name         Curl_gss_import_name_a


extern OM_uint32 Curl_gss_display_status_a(OM_uint32 * minor_status,
                                           OM_uint32 status_value,
                                           int status_type, gss_OID mech_type,
                                           gss_msg_ctx_t * message_context,
                                           gss_buffer_t status_string);
#define gss_display_status      Curl_gss_display_status_a


extern OM_uint32 Curl_gss_init_sec_context_a(OM_uint32 * minor_status,
                                             gss_cred_id_t cred_handle,
                                             gss_ctx_id_t * context_handle,
                                             gss_name_t target_name,
                                             gss_OID mech_type,
                                             gss_flags_t req_flags,
                                             OM_uint32 time_req,
                                             gss_channel_bindings_t
                                             input_chan_bindings,
                                             gss_buffer_t input_token,
                                             gss_OID * actual_mech_type,
                                             gss_buffer_t output_token,
                                             gss_flags_t * ret_flags,
                                             OM_uint32 * time_rec);
#define gss_init_sec_context    Curl_gss_init_sec_context_a


extern OM_uint32 Curl_gss_delete_sec_context_a(OM_uint32 * minor_status,
                                               gss_ctx_id_t * context_handle,
                                               gss_buffer_t output_token);
#define gss_delete_sec_context  Curl_gss_delete_sec_context_a


/* LDAP wrappers. */

#define BerValue                struct berval

#define ldap_url_parse          ldap_url_parse_utf8
#define ldap_init               Curl_ldap_init_a
#define ldap_simple_bind_s      Curl_ldap_simple_bind_s_a
#define ldap_search_s           Curl_ldap_search_s_a
#define ldap_get_values_len     Curl_ldap_get_values_len_a
#define ldap_err2string         Curl_ldap_err2string_a
#define ldap_get_dn             Curl_ldap_get_dn_a
#define ldap_first_attribute    Curl_ldap_first_attribute_a
#define ldap_next_attribute     Curl_ldap_next_attribute_a

/* Some socket functions must be wrapped to process textual addresses
   like AF_UNIX. */

extern int Curl_os400_connect(int sd, struct sockaddr *destaddr, int addrlen);
extern int Curl_os400_bind(int sd, struct sockaddr *localaddr, int addrlen);
extern int Curl_os400_sendto(int sd, char *buffer, int buflen, int flags,
                             struct sockaddr *dstaddr, int addrlen);
extern int Curl_os400_recvfrom(int sd, char *buffer, int buflen, int flags,
                               struct sockaddr *fromaddr, int *addrlen);
extern int Curl_os400_getpeername(int sd, struct sockaddr *addr, int *addrlen);
extern int Curl_os400_getsockname(int sd, struct sockaddr *addr, int *addrlen);

#define connect                 Curl_os400_connect
#define bind                    Curl_os400_bind
#define sendto                  Curl_os400_sendto
#define recvfrom                Curl_os400_recvfrom
#define getpeername             Curl_os400_getpeername
#define getsockname             Curl_os400_getsockname

#ifdef HAVE_LIBZ
#define zlibVersion             Curl_os400_zlibVersion
#define inflateInit_            Curl_os400_inflateInit_
#define inflateInit2_           Curl_os400_inflateInit2_
#define inflate                 Curl_os400_inflate
#define inflateEnd              Curl_os400_inflateEnd
#endif

#endif /* HEADER_CURL_SETUP_OS400_H */
