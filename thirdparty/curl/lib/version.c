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

#include "curl_setup.h"

#include <curl/curl.h>
#include "urldata.h"
#include "vtls/vtls.h"
#include "http2.h"
#include "vssh/ssh.h"
#include "quic.h"
#include "curl_printf.h"

#ifdef USE_ARES
#  if defined(CURL_STATICLIB) && !defined(CARES_STATICLIB) &&   \
  defined(WIN32)
#    define CARES_STATICLIB
#  endif
#  include <ares.h>
#endif

#ifdef USE_LIBIDN2
#include <idn2.h>
#endif

#ifdef USE_LIBPSL
#include <libpsl.h>
#endif

#if defined(HAVE_ICONV) && defined(CURL_DOES_CONVERSIONS)
#include <iconv.h>
#endif

#ifdef USE_LIBRTMP
#include <librtmp/rtmp.h>
#endif

#ifdef HAVE_ZLIB_H
#include <zlib.h>
#endif

#ifdef HAVE_BROTLI
#include <brotli/decode.h>
#endif

#ifdef HAVE_ZSTD
#include <zstd.h>
#endif

#ifdef USE_GSASL
#include <gsasl.h>
#endif

#ifdef USE_OPENLDAP
#include <ldap.h>
#endif

#ifdef HAVE_BROTLI
static void brotli_version(char *buf, size_t bufsz)
{
  uint32_t brotli_version = BrotliDecoderVersion();
  unsigned int major = brotli_version >> 24;
  unsigned int minor = (brotli_version & 0x00FFFFFF) >> 12;
  unsigned int patch = brotli_version & 0x00000FFF;
  (void)msnprintf(buf, bufsz, "%u.%u.%u", major, minor, patch);
}
#endif

#ifdef HAVE_ZSTD
static void zstd_version(char *buf, size_t bufsz)
{
  unsigned long zstd_version = (unsigned long)ZSTD_versionNumber();
  unsigned int major = (unsigned int)(zstd_version / (100 * 100));
  unsigned int minor = (unsigned int)((zstd_version -
                                       (major * 100 * 100)) / 100);
  unsigned int patch = (unsigned int)(zstd_version -
                                      (major * 100 * 100) - (minor * 100));
  (void)msnprintf(buf, bufsz, "%u.%u.%u", major, minor, patch);
}
#endif

/*
 * curl_version() returns a pointer to a static buffer.
 *
 * It is implemented to work multi-threaded by making sure repeated invokes
 * generate the exact same string and never write any temporary data like
 * zeros in the data.
 */

#define VERSION_PARTS 17 /* number of substrings we can concatenate */

char *curl_version(void)
{
  static char out[300];
  char *outp;
  size_t outlen;
  const char *src[VERSION_PARTS];
#ifdef USE_SSL
  char ssl_version[200];
#endif
#ifdef HAVE_LIBZ
  char z_version[40];
#endif
#ifdef HAVE_BROTLI
  char br_version[40] = "brotli/";
#endif
#ifdef HAVE_ZSTD
  char zst_version[40] = "zstd/";
#endif
#ifdef USE_ARES
  char cares_version[40];
#endif
#if defined(USE_LIBIDN2)
  char idn_version[40];
#endif
#ifdef USE_LIBPSL
  char psl_version[40];
#endif
#if defined(HAVE_ICONV) && defined(CURL_DOES_CONVERSIONS)
  char iconv_version[40]="iconv";
#endif
#ifdef USE_SSH
  char ssh_version[40];
#endif
#ifdef USE_NGHTTP2
  char h2_version[40];
#endif
#ifdef ENABLE_QUIC
  char h3_version[40];
#endif
#ifdef USE_LIBRTMP
  char rtmp_version[40];
#endif
#ifdef USE_HYPER
  char hyper_buf[30];
#endif
#ifdef USE_GSASL
  char gsasl_buf[30];
#endif
#ifdef USE_OPENLDAP
  char ldap_buf[30];
#endif
  int i = 0;
  int j;

#ifdef DEBUGBUILD
  /* Override version string when environment variable CURL_VERSION is set */
  const char *debugversion = getenv("CURL_VERSION");
  if(debugversion) {
    strncpy(out, debugversion, sizeof(out)-1);
    out[sizeof(out)-1] = '\0';
    return out;
  }
#endif

  src[i++] = LIBCURL_NAME "/" LIBCURL_VERSION;
#ifdef USE_SSL
  Curl_ssl_version(ssl_version, sizeof(ssl_version));
  src[i++] = ssl_version;
#endif
#ifdef HAVE_LIBZ
  msnprintf(z_version, sizeof(z_version), "zlib/%s", zlibVersion());
  src[i++] = z_version;
#endif
#ifdef HAVE_BROTLI
  brotli_version(&br_version[7], sizeof(br_version) - 7);
  src[i++] = br_version;
#endif
#ifdef HAVE_ZSTD
  zstd_version(&zst_version[5], sizeof(zst_version) - 5);
  src[i++] = zst_version;
#endif
#ifdef USE_ARES
  msnprintf(cares_version, sizeof(cares_version),
            "c-ares/%s", ares_version(NULL));
  src[i++] = cares_version;
#endif
#ifdef USE_LIBIDN2
  msnprintf(idn_version, sizeof(idn_version),
            "libidn2/%s", idn2_check_version(NULL));
  src[i++] = idn_version;
#elif defined(USE_WIN32_IDN)
  src[i++] = (char *)"WinIDN";
#endif

#ifdef USE_LIBPSL
  msnprintf(psl_version, sizeof(psl_version), "libpsl/%s", psl_get_version());
  src[i++] = psl_version;
#endif
#if defined(HAVE_ICONV) && defined(CURL_DOES_CONVERSIONS)
#ifdef _LIBICONV_VERSION
  msnprintf(iconv_version, sizeof(iconv_version), "iconv/%d.%d",
            _LIBICONV_VERSION >> 8, _LIBICONV_VERSION & 255);
#else
  /* version unknown, let the default stand */
#endif /* _LIBICONV_VERSION */
  src[i++] = iconv_version;
#endif
#ifdef USE_SSH
  Curl_ssh_version(ssh_version, sizeof(ssh_version));
  src[i++] = ssh_version;
#endif
#ifdef USE_NGHTTP2
  Curl_http2_ver(h2_version, sizeof(h2_version));
  src[i++] = h2_version;
#endif
#ifdef ENABLE_QUIC
  Curl_quic_ver(h3_version, sizeof(h3_version));
  src[i++] = h3_version;
#endif
#ifdef USE_LIBRTMP
  {
    char suff[2];
    if(RTMP_LIB_VERSION & 0xff) {
      suff[0] = (RTMP_LIB_VERSION & 0xff) + 'a' - 1;
      suff[1] = '\0';
    }
    else
      suff[0] = '\0';

    msnprintf(rtmp_version, sizeof(rtmp_version), "librtmp/%d.%d%s",
              RTMP_LIB_VERSION >> 16, (RTMP_LIB_VERSION >> 8) & 0xff,
              suff);
    src[i++] = rtmp_version;
  }
#endif
#ifdef USE_HYPER
  msnprintf(hyper_buf, sizeof(hyper_buf), "Hyper/%s", hyper_version());
  src[i++] = hyper_buf;
#endif
#ifdef USE_GSASL
  msnprintf(gsasl_buf, sizeof(gsasl_buf), "libgsasl/%s",
            gsasl_check_version(NULL));
  src[i++] = gsasl_buf;
#endif
#ifdef USE_OPENLDAP
  {
    LDAPAPIInfo api;
    api.ldapai_info_version = LDAP_API_INFO_VERSION;

    if(ldap_get_option(NULL, LDAP_OPT_API_INFO, &api) == LDAP_OPT_SUCCESS) {
      unsigned int patch = api.ldapai_vendor_version % 100;
      unsigned int major = api.ldapai_vendor_version / 10000;
      unsigned int minor =
        ((api.ldapai_vendor_version - major * 10000) - patch) / 100;
      msnprintf(ldap_buf, sizeof(ldap_buf), "%s/%u.%u.%u",
                api.ldapai_vendor_name, major, minor, patch);
      src[i++] = ldap_buf;
      ldap_memfree(api.ldapai_vendor_name);
      ber_memvfree((void **)api.ldapai_extensions);
    }
  }
#endif

  DEBUGASSERT(i <= VERSION_PARTS);

  outp = &out[0];
  outlen = sizeof(out);
  for(j = 0; j < i; j++) {
    size_t n = strlen(src[j]);
    /* we need room for a space, the string and the final zero */
    if(outlen <= (n + 2))
      break;
    if(j) {
      /* prepend a space if not the first */
      *outp++ = ' ';
      outlen--;
    }
    memcpy(outp, src[j], n);
    outp += n;
    outlen -= n;
  }
  *outp = 0;

  return out;
}

/* data for curl_version_info

   Keep the list sorted alphabetically. It is also written so that each
   protocol line has its own #if line to make things easier on the eye.
 */

static const char * const protocols[] = {
#ifndef CURL_DISABLE_DICT
  "dict",
#endif
#ifndef CURL_DISABLE_FILE
  "file",
#endif
#ifndef CURL_DISABLE_FTP
  "ftp",
#endif
#if defined(USE_SSL) && !defined(CURL_DISABLE_FTP)
  "ftps",
#endif
#ifndef CURL_DISABLE_GOPHER
  "gopher",
#endif
#if defined(USE_SSL) && !defined(CURL_DISABLE_GOPHER)
  "gophers",
#endif
#ifndef CURL_DISABLE_HTTP
  "http",
#endif
#if defined(USE_SSL) && !defined(CURL_DISABLE_HTTP)
  "https",
#endif
#ifndef CURL_DISABLE_IMAP
  "imap",
#endif
#if defined(USE_SSL) && !defined(CURL_DISABLE_IMAP)
  "imaps",
#endif
#ifndef CURL_DISABLE_LDAP
  "ldap",
#if !defined(CURL_DISABLE_LDAPS) && \
    ((defined(USE_OPENLDAP) && defined(USE_SSL)) || \
     (!defined(USE_OPENLDAP) && defined(HAVE_LDAP_SSL)))
  "ldaps",
#endif
#endif
#ifndef CURL_DISABLE_MQTT
  "mqtt",
#endif
#ifndef CURL_DISABLE_POP3
  "pop3",
#endif
#if defined(USE_SSL) && !defined(CURL_DISABLE_POP3)
  "pop3s",
#endif
#ifdef USE_LIBRTMP
  "rtmp",
#endif
#ifndef CURL_DISABLE_RTSP
  "rtsp",
#endif
#if defined(USE_SSH) && !defined(USE_WOLFSSH)
  "scp",
#endif
#ifdef USE_SSH
  "sftp",
#endif
#if !defined(CURL_DISABLE_SMB) && defined(USE_CURL_NTLM_CORE) && \
   (SIZEOF_CURL_OFF_T > 4)
  "smb",
#  ifdef USE_SSL
  "smbs",
#  endif
#endif
#ifndef CURL_DISABLE_SMTP
  "smtp",
#endif
#if defined(USE_SSL) && !defined(CURL_DISABLE_SMTP)
  "smtps",
#endif
#ifndef CURL_DISABLE_TELNET
  "telnet",
#endif
#ifndef CURL_DISABLE_TFTP
  "tftp",
#endif

  NULL
};

static curl_version_info_data version_info = {
  CURLVERSION_NOW,
  LIBCURL_VERSION,
  LIBCURL_VERSION_NUM,
  OS, /* as found by configure or set by hand at build-time */
  0 /* features is 0 by default */
#ifdef ENABLE_IPV6
  | CURL_VERSION_IPV6
#endif
#ifdef USE_SSL
  | CURL_VERSION_SSL
#endif
#ifdef USE_NTLM
  | CURL_VERSION_NTLM
#endif
#if !defined(CURL_DISABLE_HTTP) && defined(USE_NTLM) && \
  defined(NTLM_WB_ENABLED)
  | CURL_VERSION_NTLM_WB
#endif
#ifdef USE_SPNEGO
  | CURL_VERSION_SPNEGO
#endif
#ifdef USE_KERBEROS5
  | CURL_VERSION_KERBEROS5
#endif
#ifdef HAVE_GSSAPI
  | CURL_VERSION_GSSAPI
#endif
#ifdef USE_WINDOWS_SSPI
  | CURL_VERSION_SSPI
#endif
#ifdef HAVE_LIBZ
  | CURL_VERSION_LIBZ
#endif
#ifdef DEBUGBUILD
  | CURL_VERSION_DEBUG
#endif
#ifdef CURLDEBUG
  | CURL_VERSION_CURLDEBUG
#endif
#ifdef CURLRES_ASYNCH
  | CURL_VERSION_ASYNCHDNS
#endif
#if (SIZEOF_CURL_OFF_T > 4) && \
    ( (SIZEOF_OFF_T > 4) || defined(USE_WIN32_LARGE_FILES) )
  | CURL_VERSION_LARGEFILE
#endif
#if defined(WIN32) && defined(UNICODE) && defined(_UNICODE)
  | CURL_VERSION_UNICODE
#endif
#if defined(CURL_DOES_CONVERSIONS)
  | CURL_VERSION_CONV
#endif
#if defined(USE_TLS_SRP)
  | CURL_VERSION_TLSAUTH_SRP
#endif
#if defined(USE_NGHTTP2) || defined(USE_HYPER)
  | CURL_VERSION_HTTP2
#endif
#if defined(ENABLE_QUIC)
  | CURL_VERSION_HTTP3
#endif
#if defined(USE_UNIX_SOCKETS)
  | CURL_VERSION_UNIX_SOCKETS
#endif
#if defined(USE_LIBPSL)
  | CURL_VERSION_PSL
#endif
#if defined(CURL_WITH_MULTI_SSL)
  | CURL_VERSION_MULTI_SSL
#endif
#if defined(HAVE_BROTLI)
  | CURL_VERSION_BROTLI
#endif
#if defined(HAVE_ZSTD)
  | CURL_VERSION_ZSTD
#endif
#ifndef CURL_DISABLE_ALTSVC
  | CURL_VERSION_ALTSVC
#endif
#ifndef CURL_DISABLE_HSTS
  | CURL_VERSION_HSTS
#endif
#if defined(USE_GSASL)
  | CURL_VERSION_GSASL
#endif
  ,
  NULL, /* ssl_version */
  0,    /* ssl_version_num, this is kept at zero */
  NULL, /* zlib_version */
  protocols,
  NULL, /* c-ares version */
  0,    /* c-ares version numerical */
  NULL, /* libidn version */
  0,    /* iconv version */
  NULL, /* ssh lib version */
  0,    /* brotli_ver_num */
  NULL, /* brotli version */
  0,    /* nghttp2 version number */
  NULL, /* nghttp2 version string */
  NULL, /* quic library string */
#ifdef CURL_CA_BUNDLE
  CURL_CA_BUNDLE, /* cainfo */
#else
  NULL,
#endif
#ifdef CURL_CA_PATH
  CURL_CA_PATH,  /* capath */
#else
  NULL,
#endif
  0,    /* zstd_ver_num */
  NULL, /* zstd version */
  NULL, /* Hyper version */
  NULL  /* gsasl version */
};

curl_version_info_data *curl_version_info(CURLversion stamp)
{
#if defined(USE_SSH)
  static char ssh_buffer[80];
#endif
#ifdef USE_SSL
#ifdef CURL_WITH_MULTI_SSL
  static char ssl_buffer[200];
#else
  static char ssl_buffer[80];
#endif
#endif
#ifdef HAVE_BROTLI
  static char brotli_buffer[80];
#endif
#ifdef HAVE_ZSTD
  static char zstd_buffer[80];
#endif

#ifdef USE_SSL
  Curl_ssl_version(ssl_buffer, sizeof(ssl_buffer));
  version_info.ssl_version = ssl_buffer;
#ifndef CURL_DISABLE_PROXY
  if(Curl_ssl->supports & SSLSUPP_HTTPS_PROXY)
    version_info.features |= CURL_VERSION_HTTPS_PROXY;
  else
    version_info.features &= ~CURL_VERSION_HTTPS_PROXY;
#endif
#endif

#ifdef HAVE_LIBZ
  version_info.libz_version = zlibVersion();
  /* libz left NULL if non-existing */
#endif
#ifdef USE_ARES
  {
    int aresnum;
    version_info.ares = ares_version(&aresnum);
    version_info.ares_num = aresnum;
  }
#endif
#ifdef USE_LIBIDN2
  /* This returns a version string if we use the given version or later,
     otherwise it returns NULL */
  version_info.libidn = idn2_check_version(IDN2_VERSION);
  if(version_info.libidn)
    version_info.features |= CURL_VERSION_IDN;
#elif defined(USE_WIN32_IDN)
  version_info.features |= CURL_VERSION_IDN;
#endif

#if defined(HAVE_ICONV) && defined(CURL_DOES_CONVERSIONS)
#ifdef _LIBICONV_VERSION
  version_info.iconv_ver_num = _LIBICONV_VERSION;
#else
  /* version unknown */
  version_info.iconv_ver_num = -1;
#endif /* _LIBICONV_VERSION */
#endif

#if defined(USE_SSH)
  Curl_ssh_version(ssh_buffer, sizeof(ssh_buffer));
  version_info.libssh_version = ssh_buffer;
#endif

#ifdef HAVE_BROTLI
  version_info.brotli_ver_num = BrotliDecoderVersion();
  brotli_version(brotli_buffer, sizeof(brotli_buffer));
  version_info.brotli_version = brotli_buffer;
#endif

#ifdef HAVE_ZSTD
  version_info.zstd_ver_num = (unsigned int)ZSTD_versionNumber();
  zstd_version(zstd_buffer, sizeof(zstd_buffer));
  version_info.zstd_version = zstd_buffer;
#endif

#ifdef USE_NGHTTP2
  {
    nghttp2_info *h2 = nghttp2_version(0);
    version_info.nghttp2_ver_num = h2->version_num;
    version_info.nghttp2_version = h2->version_str;
  }
#endif

#ifdef ENABLE_QUIC
  {
    static char quicbuffer[80];
    Curl_quic_ver(quicbuffer, sizeof(quicbuffer));
    version_info.quic_version = quicbuffer;
  }
#endif

#ifdef USE_HYPER
  {
    static char hyper_buffer[30];
    msnprintf(hyper_buffer, sizeof(hyper_buffer), "Hyper/%s", hyper_version());
    version_info.hyper_version = hyper_buffer;
  }
#endif

#ifdef USE_GSASL
  {
    version_info.gsasl_version = gsasl_check_version(NULL);
  }
#endif

  (void)stamp; /* avoid compiler warnings, we don't use this */
  return &version_info;
}
