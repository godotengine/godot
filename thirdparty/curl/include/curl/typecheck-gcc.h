#ifndef CURLINC_TYPECHECK_GCC_H
#define CURLINC_TYPECHECK_GCC_H
/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 1998 - 2021, Daniel Stenberg, <daniel@haxx.se>, et al.
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

/* wraps curl_easy_setopt() with typechecking */

/* To add a new kind of warning, add an
 *   if(curlcheck_sometype_option(_curl_opt))
 *     if(!curlcheck_sometype(value))
 *       _curl_easy_setopt_err_sometype();
 * block and define curlcheck_sometype_option, curlcheck_sometype and
 * _curl_easy_setopt_err_sometype below
 *
 * NOTE: We use two nested 'if' statements here instead of the && operator, in
 *       order to work around gcc bug #32061.  It affects only gcc 4.3.x/4.4.x
 *       when compiling with -Wlogical-op.
 *
 * To add an option that uses the same type as an existing option, you'll just
 * need to extend the appropriate _curl_*_option macro
 */
#define curl_easy_setopt(handle, option, value)                         \
  __extension__({                                                       \
      __typeof__(option) _curl_opt = option;                            \
      if(__builtin_constant_p(_curl_opt)) {                             \
        if(curlcheck_long_option(_curl_opt))                            \
          if(!curlcheck_long(value))                                    \
            _curl_easy_setopt_err_long();                               \
        if(curlcheck_off_t_option(_curl_opt))                           \
          if(!curlcheck_off_t(value))                                   \
            _curl_easy_setopt_err_curl_off_t();                         \
        if(curlcheck_string_option(_curl_opt))                          \
          if(!curlcheck_string(value))                                  \
            _curl_easy_setopt_err_string();                             \
        if(curlcheck_write_cb_option(_curl_opt))                        \
          if(!curlcheck_write_cb(value))                                \
            _curl_easy_setopt_err_write_callback();                     \
        if((_curl_opt) == CURLOPT_RESOLVER_START_FUNCTION)              \
          if(!curlcheck_resolver_start_callback(value))                 \
            _curl_easy_setopt_err_resolver_start_callback();            \
        if((_curl_opt) == CURLOPT_READFUNCTION)                         \
          if(!curlcheck_read_cb(value))                                 \
            _curl_easy_setopt_err_read_cb();                            \
        if((_curl_opt) == CURLOPT_IOCTLFUNCTION)                        \
          if(!curlcheck_ioctl_cb(value))                                \
            _curl_easy_setopt_err_ioctl_cb();                           \
        if((_curl_opt) == CURLOPT_SOCKOPTFUNCTION)                      \
          if(!curlcheck_sockopt_cb(value))                              \
            _curl_easy_setopt_err_sockopt_cb();                         \
        if((_curl_opt) == CURLOPT_OPENSOCKETFUNCTION)                   \
          if(!curlcheck_opensocket_cb(value))                           \
            _curl_easy_setopt_err_opensocket_cb();                      \
        if((_curl_opt) == CURLOPT_PROGRESSFUNCTION)                     \
          if(!curlcheck_progress_cb(value))                             \
            _curl_easy_setopt_err_progress_cb();                        \
        if((_curl_opt) == CURLOPT_DEBUGFUNCTION)                        \
          if(!curlcheck_debug_cb(value))                                \
            _curl_easy_setopt_err_debug_cb();                           \
        if((_curl_opt) == CURLOPT_SSL_CTX_FUNCTION)                     \
          if(!curlcheck_ssl_ctx_cb(value))                              \
            _curl_easy_setopt_err_ssl_ctx_cb();                         \
        if(curlcheck_conv_cb_option(_curl_opt))                         \
          if(!curlcheck_conv_cb(value))                                 \
            _curl_easy_setopt_err_conv_cb();                            \
        if((_curl_opt) == CURLOPT_SEEKFUNCTION)                         \
          if(!curlcheck_seek_cb(value))                                 \
            _curl_easy_setopt_err_seek_cb();                            \
        if(curlcheck_cb_data_option(_curl_opt))                         \
          if(!curlcheck_cb_data(value))                                 \
            _curl_easy_setopt_err_cb_data();                            \
        if((_curl_opt) == CURLOPT_ERRORBUFFER)                          \
          if(!curlcheck_error_buffer(value))                            \
            _curl_easy_setopt_err_error_buffer();                       \
        if((_curl_opt) == CURLOPT_STDERR)                               \
          if(!curlcheck_FILE(value))                                    \
            _curl_easy_setopt_err_FILE();                               \
        if(curlcheck_postfields_option(_curl_opt))                      \
          if(!curlcheck_postfields(value))                              \
            _curl_easy_setopt_err_postfields();                         \
        if((_curl_opt) == CURLOPT_HTTPPOST)                             \
          if(!curlcheck_arr((value), struct curl_httppost))             \
            _curl_easy_setopt_err_curl_httpost();                       \
        if((_curl_opt) == CURLOPT_MIMEPOST)                             \
          if(!curlcheck_ptr((value), curl_mime))                        \
            _curl_easy_setopt_err_curl_mimepost();                      \
        if(curlcheck_slist_option(_curl_opt))                           \
          if(!curlcheck_arr((value), struct curl_slist))                \
            _curl_easy_setopt_err_curl_slist();                         \
        if((_curl_opt) == CURLOPT_SHARE)                                \
          if(!curlcheck_ptr((value), CURLSH))                           \
            _curl_easy_setopt_err_CURLSH();                             \
      }                                                                 \
      curl_easy_setopt(handle, _curl_opt, value);                       \
    })

/* wraps curl_easy_getinfo() with typechecking */
#define curl_easy_getinfo(handle, info, arg)                            \
  __extension__({                                                      \
      __typeof__(info) _curl_info = info;                               \
      if(__builtin_constant_p(_curl_info)) {                            \
        if(curlcheck_string_info(_curl_info))                           \
          if(!curlcheck_arr((arg), char *))                             \
            _curl_easy_getinfo_err_string();                            \
        if(curlcheck_long_info(_curl_info))                             \
          if(!curlcheck_arr((arg), long))                               \
            _curl_easy_getinfo_err_long();                              \
        if(curlcheck_double_info(_curl_info))                           \
          if(!curlcheck_arr((arg), double))                             \
            _curl_easy_getinfo_err_double();                            \
        if(curlcheck_slist_info(_curl_info))                            \
          if(!curlcheck_arr((arg), struct curl_slist *))                \
            _curl_easy_getinfo_err_curl_slist();                        \
        if(curlcheck_tlssessioninfo_info(_curl_info))                   \
          if(!curlcheck_arr((arg), struct curl_tlssessioninfo *))       \
            _curl_easy_getinfo_err_curl_tlssesssioninfo();              \
        if(curlcheck_certinfo_info(_curl_info))                         \
          if(!curlcheck_arr((arg), struct curl_certinfo *))             \
            _curl_easy_getinfo_err_curl_certinfo();                     \
        if(curlcheck_socket_info(_curl_info))                           \
          if(!curlcheck_arr((arg), curl_socket_t))                      \
            _curl_easy_getinfo_err_curl_socket();                       \
        if(curlcheck_off_t_info(_curl_info))                            \
          if(!curlcheck_arr((arg), curl_off_t))                         \
            _curl_easy_getinfo_err_curl_off_t();                        \
      }                                                                 \
      curl_easy_getinfo(handle, _curl_info, arg);                       \
    })

/*
 * For now, just make sure that the functions are called with three arguments
 */
#define curl_share_setopt(share,opt,param) curl_share_setopt(share,opt,param)
#define curl_multi_setopt(handle,opt,param) curl_multi_setopt(handle,opt,param)


/* the actual warnings, triggered by calling the _curl_easy_setopt_err*
 * functions */

/* To define a new warning, use _CURL_WARNING(identifier, "message") */
#define CURLWARNING(id, message)                                        \
  static void __attribute__((__warning__(message)))                     \
  __attribute__((__unused__)) __attribute__((__noinline__))             \
  id(void) { __asm__(""); }

CURLWARNING(_curl_easy_setopt_err_long,
  "curl_easy_setopt expects a long argument for this option")
CURLWARNING(_curl_easy_setopt_err_curl_off_t,
  "curl_easy_setopt expects a curl_off_t argument for this option")
CURLWARNING(_curl_easy_setopt_err_string,
              "curl_easy_setopt expects a "
              "string ('char *' or char[]) argument for this option"
  )
CURLWARNING(_curl_easy_setopt_err_write_callback,
  "curl_easy_setopt expects a curl_write_callback argument for this option")
CURLWARNING(_curl_easy_setopt_err_resolver_start_callback,
              "curl_easy_setopt expects a "
              "curl_resolver_start_callback argument for this option"
  )
CURLWARNING(_curl_easy_setopt_err_read_cb,
  "curl_easy_setopt expects a curl_read_callback argument for this option")
CURLWARNING(_curl_easy_setopt_err_ioctl_cb,
  "curl_easy_setopt expects a curl_ioctl_callback argument for this option")
CURLWARNING(_curl_easy_setopt_err_sockopt_cb,
  "curl_easy_setopt expects a curl_sockopt_callback argument for this option")
CURLWARNING(_curl_easy_setopt_err_opensocket_cb,
              "curl_easy_setopt expects a "
              "curl_opensocket_callback argument for this option"
  )
CURLWARNING(_curl_easy_setopt_err_progress_cb,
  "curl_easy_setopt expects a curl_progress_callback argument for this option")
CURLWARNING(_curl_easy_setopt_err_debug_cb,
  "curl_easy_setopt expects a curl_debug_callback argument for this option")
CURLWARNING(_curl_easy_setopt_err_ssl_ctx_cb,
  "curl_easy_setopt expects a curl_ssl_ctx_callback argument for this option")
CURLWARNING(_curl_easy_setopt_err_conv_cb,
  "curl_easy_setopt expects a curl_conv_callback argument for this option")
CURLWARNING(_curl_easy_setopt_err_seek_cb,
  "curl_easy_setopt expects a curl_seek_callback argument for this option")
CURLWARNING(_curl_easy_setopt_err_cb_data,
              "curl_easy_setopt expects a "
              "private data pointer as argument for this option")
CURLWARNING(_curl_easy_setopt_err_error_buffer,
              "curl_easy_setopt expects a "
              "char buffer of CURL_ERROR_SIZE as argument for this option")
CURLWARNING(_curl_easy_setopt_err_FILE,
  "curl_easy_setopt expects a 'FILE *' argument for this option")
CURLWARNING(_curl_easy_setopt_err_postfields,
  "curl_easy_setopt expects a 'void *' or 'char *' argument for this option")
CURLWARNING(_curl_easy_setopt_err_curl_httpost,
              "curl_easy_setopt expects a 'struct curl_httppost *' "
              "argument for this option")
CURLWARNING(_curl_easy_setopt_err_curl_mimepost,
              "curl_easy_setopt expects a 'curl_mime *' "
              "argument for this option")
CURLWARNING(_curl_easy_setopt_err_curl_slist,
  "curl_easy_setopt expects a 'struct curl_slist *' argument for this option")
CURLWARNING(_curl_easy_setopt_err_CURLSH,
  "curl_easy_setopt expects a CURLSH* argument for this option")

CURLWARNING(_curl_easy_getinfo_err_string,
  "curl_easy_getinfo expects a pointer to 'char *' for this info")
CURLWARNING(_curl_easy_getinfo_err_long,
  "curl_easy_getinfo expects a pointer to long for this info")
CURLWARNING(_curl_easy_getinfo_err_double,
  "curl_easy_getinfo expects a pointer to double for this info")
CURLWARNING(_curl_easy_getinfo_err_curl_slist,
  "curl_easy_getinfo expects a pointer to 'struct curl_slist *' for this info")
CURLWARNING(_curl_easy_getinfo_err_curl_tlssesssioninfo,
              "curl_easy_getinfo expects a pointer to "
              "'struct curl_tlssessioninfo *' for this info")
CURLWARNING(_curl_easy_getinfo_err_curl_certinfo,
              "curl_easy_getinfo expects a pointer to "
              "'struct curl_certinfo *' for this info")
CURLWARNING(_curl_easy_getinfo_err_curl_socket,
  "curl_easy_getinfo expects a pointer to curl_socket_t for this info")
CURLWARNING(_curl_easy_getinfo_err_curl_off_t,
  "curl_easy_getinfo expects a pointer to curl_off_t for this info")

/* groups of curl_easy_setops options that take the same type of argument */

/* To add a new option to one of the groups, just add
 *   (option) == CURLOPT_SOMETHING
 * to the or-expression. If the option takes a long or curl_off_t, you don't
 * have to do anything
 */

/* evaluates to true if option takes a long argument */
#define curlcheck_long_option(option)                   \
  (0 < (option) && (option) < CURLOPTTYPE_OBJECTPOINT)

#define curlcheck_off_t_option(option)          \
  (((option) > CURLOPTTYPE_OFF_T) && ((option) < CURLOPTTYPE_BLOB))

/* evaluates to true if option takes a char* argument */
#define curlcheck_string_option(option)                                       \
  ((option) == CURLOPT_ABSTRACT_UNIX_SOCKET ||                                \
   (option) == CURLOPT_ACCEPT_ENCODING ||                                     \
   (option) == CURLOPT_ALTSVC ||                                              \
   (option) == CURLOPT_CAINFO ||                                              \
   (option) == CURLOPT_CAPATH ||                                              \
   (option) == CURLOPT_COOKIE ||                                              \
   (option) == CURLOPT_COOKIEFILE ||                                          \
   (option) == CURLOPT_COOKIEJAR ||                                           \
   (option) == CURLOPT_COOKIELIST ||                                          \
   (option) == CURLOPT_CRLFILE ||                                             \
   (option) == CURLOPT_CUSTOMREQUEST ||                                       \
   (option) == CURLOPT_DEFAULT_PROTOCOL ||                                    \
   (option) == CURLOPT_DNS_INTERFACE ||                                       \
   (option) == CURLOPT_DNS_LOCAL_IP4 ||                                       \
   (option) == CURLOPT_DNS_LOCAL_IP6 ||                                       \
   (option) == CURLOPT_DNS_SERVERS ||                                         \
   (option) == CURLOPT_DOH_URL ||                                             \
   (option) == CURLOPT_EGDSOCKET ||                                           \
   (option) == CURLOPT_FTPPORT ||                                             \
   (option) == CURLOPT_FTP_ACCOUNT ||                                         \
   (option) == CURLOPT_FTP_ALTERNATIVE_TO_USER ||                             \
   (option) == CURLOPT_HSTS ||                                                \
   (option) == CURLOPT_INTERFACE ||                                           \
   (option) == CURLOPT_ISSUERCERT ||                                          \
   (option) == CURLOPT_KEYPASSWD ||                                           \
   (option) == CURLOPT_KRBLEVEL ||                                            \
   (option) == CURLOPT_LOGIN_OPTIONS ||                                       \
   (option) == CURLOPT_MAIL_AUTH ||                                           \
   (option) == CURLOPT_MAIL_FROM ||                                           \
   (option) == CURLOPT_NETRC_FILE ||                                          \
   (option) == CURLOPT_NOPROXY ||                                             \
   (option) == CURLOPT_PASSWORD ||                                            \
   (option) == CURLOPT_PINNEDPUBLICKEY ||                                     \
   (option) == CURLOPT_PRE_PROXY ||                                           \
   (option) == CURLOPT_PROXY ||                                               \
   (option) == CURLOPT_PROXYPASSWORD ||                                       \
   (option) == CURLOPT_PROXYUSERNAME ||                                       \
   (option) == CURLOPT_PROXYUSERPWD ||                                        \
   (option) == CURLOPT_PROXY_CAINFO ||                                        \
   (option) == CURLOPT_PROXY_CAPATH ||                                        \
   (option) == CURLOPT_PROXY_CRLFILE ||                                       \
   (option) == CURLOPT_PROXY_ISSUERCERT ||                                    \
   (option) == CURLOPT_PROXY_KEYPASSWD ||                                     \
   (option) == CURLOPT_PROXY_PINNEDPUBLICKEY ||                               \
   (option) == CURLOPT_PROXY_SERVICE_NAME ||                                  \
   (option) == CURLOPT_PROXY_SSLCERT ||                                       \
   (option) == CURLOPT_PROXY_SSLCERTTYPE ||                                   \
   (option) == CURLOPT_PROXY_SSLKEY ||                                        \
   (option) == CURLOPT_PROXY_SSLKEYTYPE ||                                    \
   (option) == CURLOPT_PROXY_SSL_CIPHER_LIST ||                               \
   (option) == CURLOPT_PROXY_TLS13_CIPHERS ||                                 \
   (option) == CURLOPT_PROXY_TLSAUTH_PASSWORD ||                              \
   (option) == CURLOPT_PROXY_TLSAUTH_TYPE ||                                  \
   (option) == CURLOPT_PROXY_TLSAUTH_USERNAME ||                              \
   (option) == CURLOPT_RANDOM_FILE ||                                         \
   (option) == CURLOPT_RANGE ||                                               \
   (option) == CURLOPT_REFERER ||                                             \
   (option) == CURLOPT_REQUEST_TARGET ||                                      \
   (option) == CURLOPT_RTSP_SESSION_ID ||                                     \
   (option) == CURLOPT_RTSP_STREAM_URI ||                                     \
   (option) == CURLOPT_RTSP_TRANSPORT ||                                      \
   (option) == CURLOPT_SASL_AUTHZID ||                                        \
   (option) == CURLOPT_SERVICE_NAME ||                                        \
   (option) == CURLOPT_SOCKS5_GSSAPI_SERVICE ||                               \
   (option) == CURLOPT_SSH_HOST_PUBLIC_KEY_MD5 ||                             \
   (option) == CURLOPT_SSH_HOST_PUBLIC_KEY_SHA256 ||                          \
   (option) == CURLOPT_SSH_KNOWNHOSTS ||                                      \
   (option) == CURLOPT_SSH_PRIVATE_KEYFILE ||                                 \
   (option) == CURLOPT_SSH_PUBLIC_KEYFILE ||                                  \
   (option) == CURLOPT_SSLCERT ||                                             \
   (option) == CURLOPT_SSLCERTTYPE ||                                         \
   (option) == CURLOPT_SSLENGINE ||                                           \
   (option) == CURLOPT_SSLKEY ||                                              \
   (option) == CURLOPT_SSLKEYTYPE ||                                          \
   (option) == CURLOPT_SSL_CIPHER_LIST ||                                     \
   (option) == CURLOPT_TLS13_CIPHERS ||                                       \
   (option) == CURLOPT_TLSAUTH_PASSWORD ||                                    \
   (option) == CURLOPT_TLSAUTH_TYPE ||                                        \
   (option) == CURLOPT_TLSAUTH_USERNAME ||                                    \
   (option) == CURLOPT_UNIX_SOCKET_PATH ||                                    \
   (option) == CURLOPT_URL ||                                                 \
   (option) == CURLOPT_USERAGENT ||                                           \
   (option) == CURLOPT_USERNAME ||                                            \
   (option) == CURLOPT_AWS_SIGV4 ||                                           \
   (option) == CURLOPT_USERPWD ||                                             \
   (option) == CURLOPT_XOAUTH2_BEARER ||                                      \
   (option) == CURLOPT_SSL_EC_CURVES ||                                       \
   0)

/* evaluates to true if option takes a curl_write_callback argument */
#define curlcheck_write_cb_option(option)                               \
  ((option) == CURLOPT_HEADERFUNCTION ||                                \
   (option) == CURLOPT_WRITEFUNCTION)

/* evaluates to true if option takes a curl_conv_callback argument */
#define curlcheck_conv_cb_option(option)                                \
  ((option) == CURLOPT_CONV_TO_NETWORK_FUNCTION ||                      \
   (option) == CURLOPT_CONV_FROM_NETWORK_FUNCTION ||                    \
   (option) == CURLOPT_CONV_FROM_UTF8_FUNCTION)

/* evaluates to true if option takes a data argument to pass to a callback */
#define curlcheck_cb_data_option(option)                                      \
  ((option) == CURLOPT_CHUNK_DATA ||                                          \
   (option) == CURLOPT_CLOSESOCKETDATA ||                                     \
   (option) == CURLOPT_DEBUGDATA ||                                           \
   (option) == CURLOPT_FNMATCH_DATA ||                                        \
   (option) == CURLOPT_HEADERDATA ||                                          \
   (option) == CURLOPT_HSTSREADDATA ||                                        \
   (option) == CURLOPT_HSTSWRITEDATA ||                                       \
   (option) == CURLOPT_INTERLEAVEDATA ||                                      \
   (option) == CURLOPT_IOCTLDATA ||                                           \
   (option) == CURLOPT_OPENSOCKETDATA ||                                      \
   (option) == CURLOPT_PREREQDATA ||                                          \
   (option) == CURLOPT_PROGRESSDATA ||                                        \
   (option) == CURLOPT_READDATA ||                                            \
   (option) == CURLOPT_SEEKDATA ||                                            \
   (option) == CURLOPT_SOCKOPTDATA ||                                         \
   (option) == CURLOPT_SSH_KEYDATA ||                                         \
   (option) == CURLOPT_SSL_CTX_DATA ||                                        \
   (option) == CURLOPT_WRITEDATA ||                                           \
   (option) == CURLOPT_RESOLVER_START_DATA ||                                 \
   (option) == CURLOPT_TRAILERDATA ||                                         \
   0)

/* evaluates to true if option takes a POST data argument (void* or char*) */
#define curlcheck_postfields_option(option)                                   \
  ((option) == CURLOPT_POSTFIELDS ||                                          \
   (option) == CURLOPT_COPYPOSTFIELDS ||                                      \
   0)

/* evaluates to true if option takes a struct curl_slist * argument */
#define curlcheck_slist_option(option)                                        \
  ((option) == CURLOPT_HTTP200ALIASES ||                                      \
   (option) == CURLOPT_HTTPHEADER ||                                          \
   (option) == CURLOPT_MAIL_RCPT ||                                           \
   (option) == CURLOPT_POSTQUOTE ||                                           \
   (option) == CURLOPT_PREQUOTE ||                                            \
   (option) == CURLOPT_PROXYHEADER ||                                         \
   (option) == CURLOPT_QUOTE ||                                               \
   (option) == CURLOPT_RESOLVE ||                                             \
   (option) == CURLOPT_TELNETOPTIONS ||                                       \
   (option) == CURLOPT_CONNECT_TO ||                                          \
   0)

/* groups of curl_easy_getinfo infos that take the same type of argument */

/* evaluates to true if info expects a pointer to char * argument */
#define curlcheck_string_info(info)                             \
  (CURLINFO_STRING < (info) && (info) < CURLINFO_LONG &&        \
   (info) != CURLINFO_PRIVATE)

/* evaluates to true if info expects a pointer to long argument */
#define curlcheck_long_info(info)                       \
  (CURLINFO_LONG < (info) && (info) < CURLINFO_DOUBLE)

/* evaluates to true if info expects a pointer to double argument */
#define curlcheck_double_info(info)                     \
  (CURLINFO_DOUBLE < (info) && (info) < CURLINFO_SLIST)

/* true if info expects a pointer to struct curl_slist * argument */
#define curlcheck_slist_info(info)                                      \
  (((info) == CURLINFO_SSL_ENGINES) || ((info) == CURLINFO_COOKIELIST))

/* true if info expects a pointer to struct curl_tlssessioninfo * argument */
#define curlcheck_tlssessioninfo_info(info)                              \
  (((info) == CURLINFO_TLS_SSL_PTR) || ((info) == CURLINFO_TLS_SESSION))

/* true if info expects a pointer to struct curl_certinfo * argument */
#define curlcheck_certinfo_info(info) ((info) == CURLINFO_CERTINFO)

/* true if info expects a pointer to struct curl_socket_t argument */
#define curlcheck_socket_info(info)                     \
  (CURLINFO_SOCKET < (info) && (info) < CURLINFO_OFF_T)

/* true if info expects a pointer to curl_off_t argument */
#define curlcheck_off_t_info(info)              \
  (CURLINFO_OFF_T < (info))


/* typecheck helpers -- check whether given expression has requested type*/

/* For pointers, you can use the curlcheck_ptr/curlcheck_arr macros,
 * otherwise define a new macro. Search for __builtin_types_compatible_p
 * in the GCC manual.
 * NOTE: these macros MUST NOT EVALUATE their arguments! The argument is
 * the actual expression passed to the curl_easy_setopt macro. This
 * means that you can only apply the sizeof and __typeof__ operators, no
 * == or whatsoever.
 */

/* XXX: should evaluate to true if expr is a pointer */
#define curlcheck_any_ptr(expr)                 \
  (sizeof(expr) == sizeof(void *))

/* evaluates to true if expr is NULL */
/* XXX: must not evaluate expr, so this check is not accurate */
#define curlcheck_NULL(expr)                                            \
  (__builtin_types_compatible_p(__typeof__(expr), __typeof__(NULL)))

/* evaluates to true if expr is type*, const type* or NULL */
#define curlcheck_ptr(expr, type)                                       \
  (curlcheck_NULL(expr) ||                                              \
   __builtin_types_compatible_p(__typeof__(expr), type *) ||            \
   __builtin_types_compatible_p(__typeof__(expr), const type *))

/* evaluates to true if expr is one of type[], type*, NULL or const type* */
#define curlcheck_arr(expr, type)                                       \
  (curlcheck_ptr((expr), type) ||                                       \
   __builtin_types_compatible_p(__typeof__(expr), type []))

/* evaluates to true if expr is a string */
#define curlcheck_string(expr)                                          \
  (curlcheck_arr((expr), char) ||                                       \
   curlcheck_arr((expr), signed char) ||                                \
   curlcheck_arr((expr), unsigned char))

/* evaluates to true if expr is a long (no matter the signedness)
 * XXX: for now, int is also accepted (and therefore short and char, which
 * are promoted to int when passed to a variadic function) */
#define curlcheck_long(expr)                                                  \
  (__builtin_types_compatible_p(__typeof__(expr), long) ||                    \
   __builtin_types_compatible_p(__typeof__(expr), signed long) ||             \
   __builtin_types_compatible_p(__typeof__(expr), unsigned long) ||           \
   __builtin_types_compatible_p(__typeof__(expr), int) ||                     \
   __builtin_types_compatible_p(__typeof__(expr), signed int) ||              \
   __builtin_types_compatible_p(__typeof__(expr), unsigned int) ||            \
   __builtin_types_compatible_p(__typeof__(expr), short) ||                   \
   __builtin_types_compatible_p(__typeof__(expr), signed short) ||            \
   __builtin_types_compatible_p(__typeof__(expr), unsigned short) ||          \
   __builtin_types_compatible_p(__typeof__(expr), char) ||                    \
   __builtin_types_compatible_p(__typeof__(expr), signed char) ||             \
   __builtin_types_compatible_p(__typeof__(expr), unsigned char))

/* evaluates to true if expr is of type curl_off_t */
#define curlcheck_off_t(expr)                                   \
  (__builtin_types_compatible_p(__typeof__(expr), curl_off_t))

/* evaluates to true if expr is abuffer suitable for CURLOPT_ERRORBUFFER */
/* XXX: also check size of an char[] array? */
#define curlcheck_error_buffer(expr)                                    \
  (curlcheck_NULL(expr) ||                                              \
   __builtin_types_compatible_p(__typeof__(expr), char *) ||            \
   __builtin_types_compatible_p(__typeof__(expr), char[]))

/* evaluates to true if expr is of type (const) void* or (const) FILE* */
#if 0
#define curlcheck_cb_data(expr)                                         \
  (curlcheck_ptr((expr), void) ||                                       \
   curlcheck_ptr((expr), FILE))
#else /* be less strict */
#define curlcheck_cb_data(expr)                 \
  curlcheck_any_ptr(expr)
#endif

/* evaluates to true if expr is of type FILE* */
#define curlcheck_FILE(expr)                                            \
  (curlcheck_NULL(expr) ||                                              \
   (__builtin_types_compatible_p(__typeof__(expr), FILE *)))

/* evaluates to true if expr can be passed as POST data (void* or char*) */
#define curlcheck_postfields(expr)                                      \
  (curlcheck_ptr((expr), void) ||                                       \
   curlcheck_arr((expr), char) ||                                       \
   curlcheck_arr((expr), unsigned char))

/* helper: __builtin_types_compatible_p distinguishes between functions and
 * function pointers, hide it */
#define curlcheck_cb_compatible(func, type)                             \
  (__builtin_types_compatible_p(__typeof__(func), type) ||              \
   __builtin_types_compatible_p(__typeof__(func) *, type))

/* evaluates to true if expr is of type curl_resolver_start_callback */
#define curlcheck_resolver_start_callback(expr)       \
  (curlcheck_NULL(expr) || \
   curlcheck_cb_compatible((expr), curl_resolver_start_callback))

/* evaluates to true if expr is of type curl_read_callback or "similar" */
#define curlcheck_read_cb(expr)                                         \
  (curlcheck_NULL(expr) ||                                              \
   curlcheck_cb_compatible((expr), __typeof__(fread) *) ||              \
   curlcheck_cb_compatible((expr), curl_read_callback) ||               \
   curlcheck_cb_compatible((expr), _curl_read_callback1) ||             \
   curlcheck_cb_compatible((expr), _curl_read_callback2) ||             \
   curlcheck_cb_compatible((expr), _curl_read_callback3) ||             \
   curlcheck_cb_compatible((expr), _curl_read_callback4) ||             \
   curlcheck_cb_compatible((expr), _curl_read_callback5) ||             \
   curlcheck_cb_compatible((expr), _curl_read_callback6))
typedef size_t (*_curl_read_callback1)(char *, size_t, size_t, void *);
typedef size_t (*_curl_read_callback2)(char *, size_t, size_t, const void *);
typedef size_t (*_curl_read_callback3)(char *, size_t, size_t, FILE *);
typedef size_t (*_curl_read_callback4)(void *, size_t, size_t, void *);
typedef size_t (*_curl_read_callback5)(void *, size_t, size_t, const void *);
typedef size_t (*_curl_read_callback6)(void *, size_t, size_t, FILE *);

/* evaluates to true if expr is of type curl_write_callback or "similar" */
#define curlcheck_write_cb(expr)                                        \
  (curlcheck_read_cb(expr) ||                                           \
   curlcheck_cb_compatible((expr), __typeof__(fwrite) *) ||             \
   curlcheck_cb_compatible((expr), curl_write_callback) ||              \
   curlcheck_cb_compatible((expr), _curl_write_callback1) ||            \
   curlcheck_cb_compatible((expr), _curl_write_callback2) ||            \
   curlcheck_cb_compatible((expr), _curl_write_callback3) ||            \
   curlcheck_cb_compatible((expr), _curl_write_callback4) ||            \
   curlcheck_cb_compatible((expr), _curl_write_callback5) ||            \
   curlcheck_cb_compatible((expr), _curl_write_callback6))
typedef size_t (*_curl_write_callback1)(const char *, size_t, size_t, void *);
typedef size_t (*_curl_write_callback2)(const char *, size_t, size_t,
                                       const void *);
typedef size_t (*_curl_write_callback3)(const char *, size_t, size_t, FILE *);
typedef size_t (*_curl_write_callback4)(const void *, size_t, size_t, void *);
typedef size_t (*_curl_write_callback5)(const void *, size_t, size_t,
                                       const void *);
typedef size_t (*_curl_write_callback6)(const void *, size_t, size_t, FILE *);

/* evaluates to true if expr is of type curl_ioctl_callback or "similar" */
#define curlcheck_ioctl_cb(expr)                                        \
  (curlcheck_NULL(expr) ||                                              \
   curlcheck_cb_compatible((expr), curl_ioctl_callback) ||              \
   curlcheck_cb_compatible((expr), _curl_ioctl_callback1) ||            \
   curlcheck_cb_compatible((expr), _curl_ioctl_callback2) ||            \
   curlcheck_cb_compatible((expr), _curl_ioctl_callback3) ||            \
   curlcheck_cb_compatible((expr), _curl_ioctl_callback4))
typedef curlioerr (*_curl_ioctl_callback1)(CURL *, int, void *);
typedef curlioerr (*_curl_ioctl_callback2)(CURL *, int, const void *);
typedef curlioerr (*_curl_ioctl_callback3)(CURL *, curliocmd, void *);
typedef curlioerr (*_curl_ioctl_callback4)(CURL *, curliocmd, const void *);

/* evaluates to true if expr is of type curl_sockopt_callback or "similar" */
#define curlcheck_sockopt_cb(expr)                                      \
  (curlcheck_NULL(expr) ||                                              \
   curlcheck_cb_compatible((expr), curl_sockopt_callback) ||            \
   curlcheck_cb_compatible((expr), _curl_sockopt_callback1) ||          \
   curlcheck_cb_compatible((expr), _curl_sockopt_callback2))
typedef int (*_curl_sockopt_callback1)(void *, curl_socket_t, curlsocktype);
typedef int (*_curl_sockopt_callback2)(const void *, curl_socket_t,
                                      curlsocktype);

/* evaluates to true if expr is of type curl_opensocket_callback or
   "similar" */
#define curlcheck_opensocket_cb(expr)                                   \
  (curlcheck_NULL(expr) ||                                              \
   curlcheck_cb_compatible((expr), curl_opensocket_callback) ||         \
   curlcheck_cb_compatible((expr), _curl_opensocket_callback1) ||       \
   curlcheck_cb_compatible((expr), _curl_opensocket_callback2) ||       \
   curlcheck_cb_compatible((expr), _curl_opensocket_callback3) ||       \
   curlcheck_cb_compatible((expr), _curl_opensocket_callback4))
typedef curl_socket_t (*_curl_opensocket_callback1)
  (void *, curlsocktype, struct curl_sockaddr *);
typedef curl_socket_t (*_curl_opensocket_callback2)
  (void *, curlsocktype, const struct curl_sockaddr *);
typedef curl_socket_t (*_curl_opensocket_callback3)
  (const void *, curlsocktype, struct curl_sockaddr *);
typedef curl_socket_t (*_curl_opensocket_callback4)
  (const void *, curlsocktype, const struct curl_sockaddr *);

/* evaluates to true if expr is of type curl_progress_callback or "similar" */
#define curlcheck_progress_cb(expr)                                     \
  (curlcheck_NULL(expr) ||                                              \
   curlcheck_cb_compatible((expr), curl_progress_callback) ||           \
   curlcheck_cb_compatible((expr), _curl_progress_callback1) ||         \
   curlcheck_cb_compatible((expr), _curl_progress_callback2))
typedef int (*_curl_progress_callback1)(void *,
    double, double, double, double);
typedef int (*_curl_progress_callback2)(const void *,
    double, double, double, double);

/* evaluates to true if expr is of type curl_debug_callback or "similar" */
#define curlcheck_debug_cb(expr)                                        \
  (curlcheck_NULL(expr) ||                                              \
   curlcheck_cb_compatible((expr), curl_debug_callback) ||              \
   curlcheck_cb_compatible((expr), _curl_debug_callback1) ||            \
   curlcheck_cb_compatible((expr), _curl_debug_callback2) ||            \
   curlcheck_cb_compatible((expr), _curl_debug_callback3) ||            \
   curlcheck_cb_compatible((expr), _curl_debug_callback4) ||            \
   curlcheck_cb_compatible((expr), _curl_debug_callback5) ||            \
   curlcheck_cb_compatible((expr), _curl_debug_callback6) ||            \
   curlcheck_cb_compatible((expr), _curl_debug_callback7) ||            \
   curlcheck_cb_compatible((expr), _curl_debug_callback8))
typedef int (*_curl_debug_callback1) (CURL *,
    curl_infotype, char *, size_t, void *);
typedef int (*_curl_debug_callback2) (CURL *,
    curl_infotype, char *, size_t, const void *);
typedef int (*_curl_debug_callback3) (CURL *,
    curl_infotype, const char *, size_t, void *);
typedef int (*_curl_debug_callback4) (CURL *,
    curl_infotype, const char *, size_t, const void *);
typedef int (*_curl_debug_callback5) (CURL *,
    curl_infotype, unsigned char *, size_t, void *);
typedef int (*_curl_debug_callback6) (CURL *,
    curl_infotype, unsigned char *, size_t, const void *);
typedef int (*_curl_debug_callback7) (CURL *,
    curl_infotype, const unsigned char *, size_t, void *);
typedef int (*_curl_debug_callback8) (CURL *,
    curl_infotype, const unsigned char *, size_t, const void *);

/* evaluates to true if expr is of type curl_ssl_ctx_callback or "similar" */
/* this is getting even messier... */
#define curlcheck_ssl_ctx_cb(expr)                                      \
  (curlcheck_NULL(expr) ||                                              \
   curlcheck_cb_compatible((expr), curl_ssl_ctx_callback) ||            \
   curlcheck_cb_compatible((expr), _curl_ssl_ctx_callback1) ||          \
   curlcheck_cb_compatible((expr), _curl_ssl_ctx_callback2) ||          \
   curlcheck_cb_compatible((expr), _curl_ssl_ctx_callback3) ||          \
   curlcheck_cb_compatible((expr), _curl_ssl_ctx_callback4) ||          \
   curlcheck_cb_compatible((expr), _curl_ssl_ctx_callback5) ||          \
   curlcheck_cb_compatible((expr), _curl_ssl_ctx_callback6) ||          \
   curlcheck_cb_compatible((expr), _curl_ssl_ctx_callback7) ||          \
   curlcheck_cb_compatible((expr), _curl_ssl_ctx_callback8))
typedef CURLcode (*_curl_ssl_ctx_callback1)(CURL *, void *, void *);
typedef CURLcode (*_curl_ssl_ctx_callback2)(CURL *, void *, const void *);
typedef CURLcode (*_curl_ssl_ctx_callback3)(CURL *, const void *, void *);
typedef CURLcode (*_curl_ssl_ctx_callback4)(CURL *, const void *,
                                            const void *);
#ifdef HEADER_SSL_H
/* hack: if we included OpenSSL's ssl.h, we know about SSL_CTX
 * this will of course break if we're included before OpenSSL headers...
 */
typedef CURLcode (*_curl_ssl_ctx_callback5)(CURL *, SSL_CTX *, void *);
typedef CURLcode (*_curl_ssl_ctx_callback6)(CURL *, SSL_CTX *, const void *);
typedef CURLcode (*_curl_ssl_ctx_callback7)(CURL *, const SSL_CTX *, void *);
typedef CURLcode (*_curl_ssl_ctx_callback8)(CURL *, const SSL_CTX *,
                                            const void *);
#else
typedef _curl_ssl_ctx_callback1 _curl_ssl_ctx_callback5;
typedef _curl_ssl_ctx_callback1 _curl_ssl_ctx_callback6;
typedef _curl_ssl_ctx_callback1 _curl_ssl_ctx_callback7;
typedef _curl_ssl_ctx_callback1 _curl_ssl_ctx_callback8;
#endif

/* evaluates to true if expr is of type curl_conv_callback or "similar" */
#define curlcheck_conv_cb(expr)                                         \
  (curlcheck_NULL(expr) ||                                              \
   curlcheck_cb_compatible((expr), curl_conv_callback) ||               \
   curlcheck_cb_compatible((expr), _curl_conv_callback1) ||             \
   curlcheck_cb_compatible((expr), _curl_conv_callback2) ||             \
   curlcheck_cb_compatible((expr), _curl_conv_callback3) ||             \
   curlcheck_cb_compatible((expr), _curl_conv_callback4))
typedef CURLcode (*_curl_conv_callback1)(char *, size_t length);
typedef CURLcode (*_curl_conv_callback2)(const char *, size_t length);
typedef CURLcode (*_curl_conv_callback3)(void *, size_t length);
typedef CURLcode (*_curl_conv_callback4)(const void *, size_t length);

/* evaluates to true if expr is of type curl_seek_callback or "similar" */
#define curlcheck_seek_cb(expr)                                         \
  (curlcheck_NULL(expr) ||                                              \
   curlcheck_cb_compatible((expr), curl_seek_callback) ||               \
   curlcheck_cb_compatible((expr), _curl_seek_callback1) ||             \
   curlcheck_cb_compatible((expr), _curl_seek_callback2))
typedef CURLcode (*_curl_seek_callback1)(void *, curl_off_t, int);
typedef CURLcode (*_curl_seek_callback2)(const void *, curl_off_t, int);


#endif /* CURLINC_TYPECHECK_GCC_H */
