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
#include "tool_setup.h"

#ifndef CURL_DISABLE_LIBCURL_OPTION

#define ENABLE_CURLX_PRINTF
/* use our own printf() functions */
#include "curlx.h"

#include "tool_cfgable.h"
#include "tool_easysrc.h"
#include "tool_setopt.h"
#include "tool_convert.h"
#include "tool_msgs.h"

#include "memdebug.h" /* keep this as LAST include */

/* Lookup tables for converting setopt values back to symbols */
/* For enums, values may be in any order. */
/* For bit masks, put combinations first, then single bits, */
/* and finally any "NONE" value. */

#define NV(e) {#e, e}
#define NV1(e, v) {#e, (v)}
#define NVEND {NULL, 0}         /* sentinel to mark end of list */

const struct NameValue setopt_nv_CURLPROXY[] = {
  NV(CURLPROXY_HTTP),
  NV(CURLPROXY_HTTP_1_0),
  NV(CURLPROXY_HTTPS),
  NV(CURLPROXY_SOCKS4),
  NV(CURLPROXY_SOCKS5),
  NV(CURLPROXY_SOCKS4A),
  NV(CURLPROXY_SOCKS5_HOSTNAME),
  NVEND,
};

const struct NameValue setopt_nv_CURL_SOCKS_PROXY[] = {
  NV(CURLPROXY_SOCKS4),
  NV(CURLPROXY_SOCKS5),
  NV(CURLPROXY_SOCKS4A),
  NV(CURLPROXY_SOCKS5_HOSTNAME),
  NVEND,
};

const struct NameValueUnsigned setopt_nv_CURLHSTS[] = {
  NV(CURLHSTS_ENABLE),
  NVEND,
};

const struct NameValueUnsigned setopt_nv_CURLAUTH[] = {
  NV(CURLAUTH_ANY),             /* combination */
  NV(CURLAUTH_ANYSAFE),         /* combination */
  NV(CURLAUTH_BASIC),
  NV(CURLAUTH_DIGEST),
  NV(CURLAUTH_GSSNEGOTIATE),
  NV(CURLAUTH_NTLM),
  NV(CURLAUTH_DIGEST_IE),
  NV(CURLAUTH_NTLM_WB),
  NV(CURLAUTH_ONLY),
  NV(CURLAUTH_NONE),
  NVEND,
};

const struct NameValue setopt_nv_CURL_HTTP_VERSION[] = {
  NV(CURL_HTTP_VERSION_NONE),
  NV(CURL_HTTP_VERSION_1_0),
  NV(CURL_HTTP_VERSION_1_1),
  NV(CURL_HTTP_VERSION_2_0),
  NV(CURL_HTTP_VERSION_2TLS),
  NV(CURL_HTTP_VERSION_3),
  NVEND,
};

const struct NameValue setopt_nv_CURL_SSLVERSION[] = {
  NV(CURL_SSLVERSION_DEFAULT),
  NV(CURL_SSLVERSION_TLSv1),
  NV(CURL_SSLVERSION_SSLv2),
  NV(CURL_SSLVERSION_SSLv3),
  NV(CURL_SSLVERSION_TLSv1_0),
  NV(CURL_SSLVERSION_TLSv1_1),
  NV(CURL_SSLVERSION_TLSv1_2),
  NV(CURL_SSLVERSION_TLSv1_3),
  NVEND,
};

const struct NameValue setopt_nv_CURL_TIMECOND[] = {
  NV(CURL_TIMECOND_IFMODSINCE),
  NV(CURL_TIMECOND_IFUNMODSINCE),
  NV(CURL_TIMECOND_LASTMOD),
  NV(CURL_TIMECOND_NONE),
  NVEND,
};

const struct NameValue setopt_nv_CURLFTPSSL_CCC[] = {
  NV(CURLFTPSSL_CCC_NONE),
  NV(CURLFTPSSL_CCC_PASSIVE),
  NV(CURLFTPSSL_CCC_ACTIVE),
  NVEND,
};

const struct NameValue setopt_nv_CURLUSESSL[] = {
  NV(CURLUSESSL_NONE),
  NV(CURLUSESSL_TRY),
  NV(CURLUSESSL_CONTROL),
  NV(CURLUSESSL_ALL),
  NVEND,
};

const struct NameValueUnsigned setopt_nv_CURLSSLOPT[] = {
  NV(CURLSSLOPT_ALLOW_BEAST),
  NV(CURLSSLOPT_NO_REVOKE),
  NV(CURLSSLOPT_NO_PARTIALCHAIN),
  NV(CURLSSLOPT_REVOKE_BEST_EFFORT),
  NV(CURLSSLOPT_NATIVE_CA),
  NV(CURLSSLOPT_AUTO_CLIENT_CERT),
  NVEND,
};

const struct NameValue setopt_nv_CURL_NETRC[] = {
  NV(CURL_NETRC_IGNORED),
  NV(CURL_NETRC_OPTIONAL),
  NV(CURL_NETRC_REQUIRED),
  NVEND,
};

/* These mappings essentially triplicated - see
 * tool_libinfo.c and tool_paramhlp.c */
const struct NameValue setopt_nv_CURLPROTO[] = {
  NV(CURLPROTO_ALL),            /* combination */
  NV(CURLPROTO_DICT),
  NV(CURLPROTO_FILE),
  NV(CURLPROTO_FTP),
  NV(CURLPROTO_FTPS),
  NV(CURLPROTO_GOPHER),
  NV(CURLPROTO_HTTP),
  NV(CURLPROTO_HTTPS),
  NV(CURLPROTO_IMAP),
  NV(CURLPROTO_IMAPS),
  NV(CURLPROTO_LDAP),
  NV(CURLPROTO_LDAPS),
  NV(CURLPROTO_POP3),
  NV(CURLPROTO_POP3S),
  NV(CURLPROTO_RTSP),
  NV(CURLPROTO_SCP),
  NV(CURLPROTO_SFTP),
  NV(CURLPROTO_SMB),
  NV(CURLPROTO_SMBS),
  NV(CURLPROTO_SMTP),
  NV(CURLPROTO_SMTPS),
  NV(CURLPROTO_TELNET),
  NV(CURLPROTO_TFTP),
  NVEND,
};

/* These options have non-zero default values. */
static const struct NameValue setopt_nv_CURLNONZERODEFAULTS[] = {
  NV1(CURLOPT_SSL_VERIFYPEER, 1),
  NV1(CURLOPT_SSL_VERIFYHOST, 1),
  NV1(CURLOPT_SSL_ENABLE_NPN, 1),
  NV1(CURLOPT_SSL_ENABLE_ALPN, 1),
  NV1(CURLOPT_TCP_NODELAY, 1),
  NV1(CURLOPT_PROXY_SSL_VERIFYPEER, 1),
  NV1(CURLOPT_PROXY_SSL_VERIFYHOST, 1),
  NV1(CURLOPT_SOCKS5_AUTH, 1),
  NVEND
};

/* Format and add code; jump to nomem on malloc error */
#define ADD(args) do { \
  ret = easysrc_add args; \
  if(ret) \
    goto nomem; \
} while(0)
#define ADDF(args) do { \
  ret = easysrc_addf args; \
  if(ret) \
    goto nomem; \
} while(0)
#define NULL_CHECK(p) do { \
  if(!p) { \
    ret = CURLE_OUT_OF_MEMORY; \
    goto nomem; \
  } \
} while(0)

#define DECL0(s) ADD((&easysrc_decl, s))
#define DECL1(f,a) ADDF((&easysrc_decl, f,a))

#define DATA0(s) ADD((&easysrc_data, s))
#define DATA1(f,a) ADDF((&easysrc_data, f,a))
#define DATA2(f,a,b) ADDF((&easysrc_data, f,a,b))
#define DATA3(f,a,b,c) ADDF((&easysrc_data, f,a,b,c))

#define CODE0(s) ADD((&easysrc_code, s))
#define CODE1(f,a) ADDF((&easysrc_code, f,a))
#define CODE2(f,a,b) ADDF((&easysrc_code, f,a,b))
#define CODE3(f,a,b,c) ADDF((&easysrc_code, f,a,b,c))

#define CLEAN0(s) ADD((&easysrc_clean, s))
#define CLEAN1(f,a) ADDF((&easysrc_clean, f,a))

#define REM0(s) ADD((&easysrc_toohard, s))
#define REM1(f,a) ADDF((&easysrc_toohard, f,a))
#define REM2(f,a,b) ADDF((&easysrc_toohard, f,a,b))

/* Escape string to C string syntax.  Return NULL if out of memory.
 * Is this correct for those wacky EBCDIC guys? */

#define MAX_STRING_LENGTH_OUTPUT 2000
#define ZERO_TERMINATED -1

static char *c_escape(const char *str, curl_off_t len)
{
  const char *s;
  unsigned char c;
  char *escaped, *e;
  unsigned int cutoff = 0;

  if(len == ZERO_TERMINATED)
    len = strlen(str);

  if(len > MAX_STRING_LENGTH_OUTPUT) {
    /* cap ridiculously long strings */
    len = MAX_STRING_LENGTH_OUTPUT;
    cutoff = 3;
  }

  /* Allocate space based on worst-case */
  escaped = malloc(4 * (size_t)len + 1 + cutoff);
  if(!escaped)
    return NULL;

  e = escaped;
  for(s = str; len; s++, len--) {
    c = *s;
    if(c == '\n') {
      strcpy(e, "\\n");
      e += 2;
    }
    else if(c == '\r') {
      strcpy(e, "\\r");
      e += 2;
    }
    else if(c == '\t') {
      strcpy(e, "\\t");
      e += 2;
    }
    else if(c == '\\') {
      strcpy(e, "\\\\");
      e += 2;
    }
    else if(c == '"') {
      strcpy(e, "\\\"");
      e += 2;
    }
    else if(!isprint(c)) {
      msnprintf(e, 5, "\\x%02x", (unsigned)c);
      e += 4;
    }
    else
      *e++ = c;
  }
  while(cutoff--)
    *e++ = '.';
  *e = '\0';
  return escaped;
}

/* setopt wrapper for enum types */
CURLcode tool_setopt_enum(CURL *curl, struct GlobalConfig *config,
                          const char *name, CURLoption tag,
                          const struct NameValue *nvlist, long lval)
{
  CURLcode ret = CURLE_OK;
  bool skip = FALSE;

  ret = curl_easy_setopt(curl, tag, lval);
  if(!lval)
    skip = TRUE;

  if(config->libcurl && !skip && !ret) {
    /* we only use this for real if --libcurl was used */
    const struct NameValue *nv = NULL;
    for(nv = nvlist; nv->name; nv++) {
      if(nv->value == lval)
        break; /* found it */
    }
    if(!nv->name) {
      /* If no definition was found, output an explicit value.
       * This could happen if new values are defined and used
       * but the NameValue list is not updated. */
      CODE2("curl_easy_setopt(hnd, %s, %ldL);", name, lval);
    }
    else {
      CODE2("curl_easy_setopt(hnd, %s, (long)%s);", name, nv->name);
    }
  }

#ifdef DEBUGBUILD
  if(ret)
    warnf(config, "option %s returned error (%d)\n", name, (int)ret);
#endif
  nomem:
  return ret;
}

/* setopt wrapper for flags */
CURLcode tool_setopt_flags(CURL *curl, struct GlobalConfig *config,
                           const char *name, CURLoption tag,
                           const struct NameValue *nvlist, long lval)
{
  CURLcode ret = CURLE_OK;
  bool skip = FALSE;

  ret = curl_easy_setopt(curl, tag, lval);
  if(!lval)
    skip = TRUE;

  if(config->libcurl && !skip && !ret) {
    /* we only use this for real if --libcurl was used */
    char preamble[80];          /* should accommodate any symbol name */
    long rest = lval;           /* bits not handled yet */
    const struct NameValue *nv = NULL;
    msnprintf(preamble, sizeof(preamble),
              "curl_easy_setopt(hnd, %s, ", name);
    for(nv = nvlist; nv->name; nv++) {
      if((nv->value & ~ rest) == 0) {
        /* all value flags contained in rest */
        rest &= ~ nv->value;    /* remove bits handled here */
        CODE3("%s(long)%s%s",
              preamble, nv->name, rest ? " |" : ");");
        if(!rest)
          break;                /* handled them all */
        /* replace with all spaces for continuation line */
        msnprintf(preamble, sizeof(preamble), "%*s", strlen(preamble), "");
      }
    }
    /* If any bits have no definition, output an explicit value.
     * This could happen if new bits are defined and used
     * but the NameValue list is not updated. */
    if(rest)
      CODE2("%s%ldL);", preamble, rest);
  }

 nomem:
  return ret;
}

/* setopt wrapper for bitmasks */
CURLcode tool_setopt_bitmask(CURL *curl, struct GlobalConfig *config,
                             const char *name, CURLoption tag,
                             const struct NameValueUnsigned *nvlist,
                             long lval)
{
  CURLcode ret = CURLE_OK;
  bool skip = FALSE;

  ret = curl_easy_setopt(curl, tag, lval);
  if(!lval)
    skip = TRUE;

  if(config->libcurl && !skip && !ret) {
    /* we only use this for real if --libcurl was used */
    char preamble[80];
    unsigned long rest = (unsigned long)lval;
    const struct NameValueUnsigned *nv = NULL;
    msnprintf(preamble, sizeof(preamble),
              "curl_easy_setopt(hnd, %s, ", name);
    for(nv = nvlist; nv->name; nv++) {
      if((nv->value & ~ rest) == 0) {
        /* all value flags contained in rest */
        rest &= ~ nv->value;    /* remove bits handled here */
        CODE3("%s(long)%s%s",
              preamble, nv->name, rest ? " |" : ");");
        if(!rest)
          break;                /* handled them all */
        /* replace with all spaces for continuation line */
        msnprintf(preamble, sizeof(preamble), "%*s", strlen(preamble), "");
      }
    }
    /* If any bits have no definition, output an explicit value.
     * This could happen if new bits are defined and used
     * but the NameValue list is not updated. */
    if(rest)
      CODE2("%s%luUL);", preamble, rest);
  }

 nomem:
  return ret;
}

/* Generate code for a struct curl_slist. */
static CURLcode libcurl_generate_slist(struct curl_slist *slist, int *slistno)
{
  CURLcode ret = CURLE_OK;
  char *escaped = NULL;

  /* May need several slist variables, so invent name */
  *slistno = ++easysrc_slist_count;

  DECL1("struct curl_slist *slist%d;", *slistno);
  DATA1("slist%d = NULL;", *slistno);
  CLEAN1("curl_slist_free_all(slist%d);", *slistno);
  CLEAN1("slist%d = NULL;", *slistno);
  for(; slist; slist = slist->next) {
    Curl_safefree(escaped);
    escaped = c_escape(slist->data, ZERO_TERMINATED);
    if(!escaped)
      return CURLE_OUT_OF_MEMORY;
    DATA3("slist%d = curl_slist_append(slist%d, \"%s\");",
                                       *slistno, *slistno, escaped);
  }

 nomem:
  Curl_safefree(escaped);
  return ret;
}

static CURLcode libcurl_generate_mime(CURL *curl,
                                      struct GlobalConfig *config,
                                      struct tool_mime *toolmime,
                                      int *mimeno);     /* Forward. */

/* Wrapper to generate source code for a mime part. */
static CURLcode libcurl_generate_mime_part(CURL *curl,
                                           struct GlobalConfig *config,
                                           struct tool_mime *part,
                                           int mimeno)
{
  CURLcode ret = CURLE_OK;
  int submimeno = 0;
  char *escaped = NULL;
  const char *data = NULL;
  const char *filename = part->filename;

  /* Parts are linked in reverse order. */
  if(part->prev) {
    ret = libcurl_generate_mime_part(curl, config, part->prev, mimeno);
    if(ret)
      return ret;
  }

  /* Create the part. */
  CODE2("part%d = curl_mime_addpart(mime%d);", mimeno, mimeno);

  switch(part->kind) {
  case TOOLMIME_PARTS:
    ret = libcurl_generate_mime(curl, config, part, &submimeno);
    if(!ret) {
      CODE2("curl_mime_subparts(part%d, mime%d);", mimeno, submimeno);
      CODE1("mime%d = NULL;", submimeno);   /* Avoid freeing in CLEAN. */
    }
    break;

  case TOOLMIME_DATA:
#ifdef CURL_DOES_CONVERSIONS
    /* Data will be set in ASCII, thus issue a comment with clear text. */
    escaped = c_escape(part->data, ZERO_TERMINATED);
    NULL_CHECK(escaped);
    CODE1("/* \"%s\" */", escaped);

    /* Our data is always textual: convert it to ASCII. */
    {
      size_t size = strlen(part->data);
      char *cp = malloc(size + 1);

      NULL_CHECK(cp);
      memcpy(cp, part->data, size + 1);
      ret = convert_to_network(cp, size);
      data = cp;
    }
#else
    data = part->data;
#endif
    if(!ret) {
      Curl_safefree(escaped);
      escaped = c_escape(data, ZERO_TERMINATED);
      NULL_CHECK(escaped);
      CODE2("curl_mime_data(part%d, \"%s\", CURL_ZERO_TERMINATED);",
                            mimeno, escaped);
    }
    break;

  case TOOLMIME_FILE:
  case TOOLMIME_FILEDATA:
    escaped = c_escape(part->data, ZERO_TERMINATED);
    NULL_CHECK(escaped);
    CODE2("curl_mime_filedata(part%d, \"%s\");", mimeno, escaped);
    if(part->kind == TOOLMIME_FILEDATA && !filename) {
      CODE1("curl_mime_filename(part%d, NULL);", mimeno);
    }
    break;

  case TOOLMIME_STDIN:
    if(!filename)
      filename = "-";
    /* FALLTHROUGH */
  case TOOLMIME_STDINDATA:
    /* Can only be reading stdin in the current context. */
    CODE1("curl_mime_data_cb(part%d, -1, (curl_read_callback) fread, \\",
          mimeno);
    CODE0("                  (curl_seek_callback) fseek, NULL, stdin);");
    break;
  default:
    /* Other cases not possible in this context. */
    break;
  }

  if(!ret && part->encoder) {
    Curl_safefree(escaped);
    escaped = c_escape(part->encoder, ZERO_TERMINATED);
    NULL_CHECK(escaped);
    CODE2("curl_mime_encoder(part%d, \"%s\");", mimeno, escaped);
  }

  if(!ret && filename) {
    Curl_safefree(escaped);
    escaped = c_escape(filename, ZERO_TERMINATED);
    NULL_CHECK(escaped);
    CODE2("curl_mime_filename(part%d, \"%s\");", mimeno, escaped);
  }

  if(!ret && part->name) {
    Curl_safefree(escaped);
    escaped = c_escape(part->name, ZERO_TERMINATED);
    NULL_CHECK(escaped);
    CODE2("curl_mime_name(part%d, \"%s\");", mimeno, escaped);
  }

  if(!ret && part->type) {
    Curl_safefree(escaped);
    escaped = c_escape(part->type, ZERO_TERMINATED);
    NULL_CHECK(escaped);
    CODE2("curl_mime_type(part%d, \"%s\");", mimeno, escaped);
  }

  if(!ret && part->headers) {
    int slistno;

    ret = libcurl_generate_slist(part->headers, &slistno);
    if(!ret) {
      CODE2("curl_mime_headers(part%d, slist%d, 1);", mimeno, slistno);
      CODE1("slist%d = NULL;", slistno); /* Prevent CLEANing. */
    }
  }

nomem:
#ifdef CURL_DOES_CONVERSIONS
  if(data)
    free((char *) data);
#endif

  Curl_safefree(escaped);
  return ret;
}

/* Wrapper to generate source code for a mime structure. */
static CURLcode libcurl_generate_mime(CURL *curl,
                                      struct GlobalConfig *config,
                                      struct tool_mime *toolmime,
                                      int *mimeno)
{
  CURLcode ret = CURLE_OK;

  /* May need several mime variables, so invent name. */
  *mimeno = ++easysrc_mime_count;
  DECL1("curl_mime *mime%d;", *mimeno);
  DATA1("mime%d = NULL;", *mimeno);
  CODE1("mime%d = curl_mime_init(hnd);", *mimeno);
  CLEAN1("curl_mime_free(mime%d);", *mimeno);
  CLEAN1("mime%d = NULL;", *mimeno);

  if(toolmime->subparts) {
    DECL1("curl_mimepart *part%d;", *mimeno);
    ret = libcurl_generate_mime_part(curl, config,
                                     toolmime->subparts, *mimeno);
  }

nomem:
  return ret;
}

/* setopt wrapper for CURLOPT_MIMEPOST */
CURLcode tool_setopt_mimepost(CURL *curl, struct GlobalConfig *config,
                              const char *name, CURLoption tag,
                              curl_mime *mimepost)
{
  CURLcode ret = curl_easy_setopt(curl, tag, mimepost);
  int mimeno = 0;

  if(!ret && config->libcurl) {
    ret = libcurl_generate_mime(curl, config,
                                config->current->mimeroot, &mimeno);

    if(!ret)
      CODE2("curl_easy_setopt(hnd, %s, mime%d);", name, mimeno);
  }

nomem:
  return ret;
}

/* setopt wrapper for curl_slist options */
CURLcode tool_setopt_slist(CURL *curl, struct GlobalConfig *config,
                           const char *name, CURLoption tag,
                           struct curl_slist *list)
{
  CURLcode ret = CURLE_OK;

  ret = curl_easy_setopt(curl, tag, list);

  if(config->libcurl && list && !ret) {
    int i;

    ret = libcurl_generate_slist(list, &i);
    if(!ret)
      CODE2("curl_easy_setopt(hnd, %s, slist%d);", name, i);
  }

 nomem:
  return ret;
}

/* generic setopt wrapper for all other options.
 * Some type information is encoded in the tag value. */
CURLcode tool_setopt(CURL *curl, bool str, struct GlobalConfig *global,
                     struct OperationConfig *config,
                     const char *name, CURLoption tag, ...)
{
  va_list arg;
  char buf[256];
  const char *value = NULL;
  bool remark = FALSE;
  bool skip = FALSE;
  bool escape = FALSE;
  char *escaped = NULL;
  CURLcode ret = CURLE_OK;

  va_start(arg, tag);

  if(tag < CURLOPTTYPE_OBJECTPOINT) {
    /* Value is expected to be a long */
    long lval = va_arg(arg, long);
    long defval = 0L;
    const struct NameValue *nv = NULL;
    for(nv = setopt_nv_CURLNONZERODEFAULTS; nv->name; nv++) {
      if(!strcmp(name, nv->name)) {
        defval = nv->value;
        break; /* found it */
      }
    }

    msnprintf(buf, sizeof(buf), "%ldL", lval);
    value = buf;
    ret = curl_easy_setopt(curl, tag, lval);
    if(lval == defval)
      skip = TRUE;
  }
  else if(tag < CURLOPTTYPE_OFF_T) {
    /* Value is some sort of object pointer */
    void *pval = va_arg(arg, void *);

    /* function pointers are never printable */
    if(tag >= CURLOPTTYPE_FUNCTIONPOINT) {
      if(pval) {
        value = "functionpointer";
        remark = TRUE;
      }
      else
        skip = TRUE;
    }

    else if(pval && str) {
      value = (char *)pval;
      escape = TRUE;
    }
    else if(pval) {
      value = "objectpointer";
      remark = TRUE;
    }
    else
      skip = TRUE;

    ret = curl_easy_setopt(curl, tag, pval);

  }
  else if(tag < CURLOPTTYPE_BLOB) {
    /* Value is expected to be curl_off_t */
    curl_off_t oval = va_arg(arg, curl_off_t);
    msnprintf(buf, sizeof(buf),
              "(curl_off_t)%" CURL_FORMAT_CURL_OFF_T, oval);
    value = buf;
    ret = curl_easy_setopt(curl, tag, oval);

    if(!oval)
      skip = TRUE;
  }
  else {
    /* Value is a blob */
    void *pblob = va_arg(arg, void *);

    /* blobs are never printable */
    if(pblob) {
      value = "blobpointer";
      remark = TRUE;
    }
    else
      skip = TRUE;

    ret = curl_easy_setopt(curl, tag, pblob);
  }

  va_end(arg);

  if(global->libcurl && !skip && !ret) {
    /* we only use this for real if --libcurl was used */

    if(remark)
      REM2("%s set to a %s", name, value);
    else {
      if(escape) {
        curl_off_t len = ZERO_TERMINATED;
        if(tag == CURLOPT_POSTFIELDS)
          len = config->postfieldsize;
        escaped = c_escape(value, len);
        NULL_CHECK(escaped);
        CODE2("curl_easy_setopt(hnd, %s, \"%s\");", name, escaped);
      }
      else
        CODE2("curl_easy_setopt(hnd, %s, %s);", name, value);
    }
  }

 nomem:
  Curl_safefree(escaped);
  return ret;
}

#else /* CURL_DISABLE_LIBCURL_OPTION */

#include "tool_cfgable.h"
#include "tool_setopt.h"

#endif /* CURL_DISABLE_LIBCURL_OPTION */

/*
 * tool_setopt_skip() allows the curl tool code to avoid setopt options that
 * are explicitly disabled in the build.
 */
bool tool_setopt_skip(CURLoption tag)
{
#ifdef CURL_DISABLE_PROXY
#define USED_TAG
  switch(tag) {
  case CURLOPT_HAPROXYPROTOCOL:
  case CURLOPT_HTTPPROXYTUNNEL:
  case CURLOPT_NOPROXY:
  case CURLOPT_PRE_PROXY:
  case CURLOPT_PROXY:
  case CURLOPT_PROXYAUTH:
  case CURLOPT_PROXY_CAINFO:
  case CURLOPT_PROXY_CAPATH:
  case CURLOPT_PROXY_CRLFILE:
  case CURLOPT_PROXYHEADER:
  case CURLOPT_PROXY_KEYPASSWD:
  case CURLOPT_PROXYPASSWORD:
  case CURLOPT_PROXY_PINNEDPUBLICKEY:
  case CURLOPT_PROXYPORT:
  case CURLOPT_PROXY_SERVICE_NAME:
  case CURLOPT_PROXY_SSLCERT:
  case CURLOPT_PROXY_SSLCERTTYPE:
  case CURLOPT_PROXY_SSL_CIPHER_LIST:
  case CURLOPT_PROXY_SSLKEY:
  case CURLOPT_PROXY_SSLKEYTYPE:
  case CURLOPT_PROXY_SSL_OPTIONS:
  case CURLOPT_PROXY_SSL_VERIFYHOST:
  case CURLOPT_PROXY_SSL_VERIFYPEER:
  case CURLOPT_PROXY_SSLVERSION:
  case CURLOPT_PROXY_TLS13_CIPHERS:
  case CURLOPT_PROXY_TLSAUTH_PASSWORD:
  case CURLOPT_PROXY_TLSAUTH_TYPE:
  case CURLOPT_PROXY_TLSAUTH_USERNAME:
  case CURLOPT_PROXY_TRANSFER_MODE:
  case CURLOPT_PROXYTYPE:
  case CURLOPT_PROXYUSERNAME:
  case CURLOPT_PROXYUSERPWD:
    return TRUE;
  default:
    break;
  }
#endif
#ifdef CURL_DISABLE_FTP
#define USED_TAG
  switch(tag) {
  case CURLOPT_FTPPORT:
  case CURLOPT_FTP_ACCOUNT:
  case CURLOPT_FTP_ALTERNATIVE_TO_USER:
  case CURLOPT_FTP_FILEMETHOD:
  case CURLOPT_FTP_SKIP_PASV_IP:
  case CURLOPT_FTP_USE_EPRT:
  case CURLOPT_FTP_USE_EPSV:
  case CURLOPT_FTP_USE_PRET:
  case CURLOPT_KRBLEVEL:
    return TRUE;
  default:
    break;
  }
#endif
#ifdef CURL_DISABLE_RTSP
#define USED_TAG
  switch(tag) {
  case CURLOPT_INTERLEAVEDATA:
    return TRUE;
  default:
    break;
  }
#endif
#if defined(CURL_DISABLE_HTTP) || defined(CURL_DISABLE_COOKIES)
#define USED_TAG
  switch(tag) {
  case CURLOPT_COOKIE:
  case CURLOPT_COOKIEFILE:
  case CURLOPT_COOKIEJAR:
  case CURLOPT_COOKIESESSION:
    return TRUE;
  default:
    break;
  }
#endif
#if defined(CURL_DISABLE_TELNET)
#define USED_TAG
  switch(tag) {
  case CURLOPT_TELNETOPTIONS:
    return TRUE;
  default:
    break;
  }
#endif
#ifdef CURL_DISABLE_TFTP
#define USED_TAG
  switch(tag) {
  case CURLOPT_TFTP_BLKSIZE:
  case CURLOPT_TFTP_NO_OPTIONS:
    return TRUE;
  default:
    break;
  }
#endif
#ifdef CURL_DISABLE_NETRC
#define USED_TAG
  switch(tag) {
  case CURLOPT_NETRC:
  case CURLOPT_NETRC_FILE:
    return TRUE;
  default:
    break;
  }
#endif

#ifndef USED_TAG
  (void)tag;
#endif
  return FALSE;
}
