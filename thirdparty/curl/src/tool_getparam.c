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

#include "strcase.h"

#define ENABLE_CURLX_PRINTF
/* use our own printf() functions */
#include "curlx.h"

#include "tool_binmode.h"
#include "tool_cfgable.h"
#include "tool_cb_prg.h"
#include "tool_convert.h"
#include "tool_filetime.h"
#include "tool_formparse.h"
#include "tool_getparam.h"
#include "tool_helpers.h"
#include "tool_libinfo.h"
#include "tool_msgs.h"
#include "tool_paramhlp.h"
#include "tool_parsecfg.h"
#include "tool_main.h"

#include "memdebug.h" /* keep this as LAST include */

#ifdef MSDOS
#  define USE_WATT32
#endif

#define GetStr(str,val) do { \
  if(*(str)) { \
    free(*(str)); \
    *(str) = NULL; \
  } \
  if((val)) {              \
    *(str) = strdup((val)); \
    if(!(*(str)))          \
      return PARAM_NO_MEM; \
  } \
} while(0)

struct LongShort {
  const char *letter; /* short name option */
  const char *lname;  /* long name option */
  enum {
    ARG_NONE,   /* stand-alone but not a boolean */
    ARG_BOOL,   /* accepts a --no-[name] prefix */
    ARG_STRING, /* requires an argument */
    ARG_FILENAME /* requires an argument, usually a file name */
  } desc;
};

static const struct LongShort aliases[]= {
  /* 'letter' strings with more than one character have *no* short option to
     mention. */
  {"*@", "url",                      ARG_STRING},
  {"*4", "dns-ipv4-addr",            ARG_STRING},
  {"*6", "dns-ipv6-addr",            ARG_STRING},
  {"*a", "random-file",              ARG_FILENAME},
  {"*b", "egd-file",                 ARG_STRING},
  {"*B", "oauth2-bearer",            ARG_STRING},
  {"*c", "connect-timeout",          ARG_STRING},
  {"*C", "doh-url"        ,          ARG_STRING},
  {"*d", "ciphers",                  ARG_STRING},
  {"*D", "dns-interface",            ARG_STRING},
  {"*e", "disable-epsv",             ARG_BOOL},
  {"*f", "disallow-username-in-url", ARG_BOOL},
  {"*E", "epsv",                     ARG_BOOL},
         /* 'epsv' made like this to make --no-epsv and --epsv to work
             although --disable-epsv is the documented option */
  {"*F", "dns-servers",              ARG_STRING},
  {"*g", "trace",                    ARG_FILENAME},
  {"*G", "npn",                      ARG_BOOL},
  {"*h", "trace-ascii",              ARG_FILENAME},
  {"*H", "alpn",                     ARG_BOOL},
  {"*i", "limit-rate",               ARG_STRING},
  {"*j", "compressed",               ARG_BOOL},
  {"*J", "tr-encoding",              ARG_BOOL},
  {"*k", "digest",                   ARG_BOOL},
  {"*l", "negotiate",                ARG_BOOL},
  {"*m", "ntlm",                     ARG_BOOL},
  {"*M", "ntlm-wb",                  ARG_BOOL},
  {"*n", "basic",                    ARG_BOOL},
  {"*o", "anyauth",                  ARG_BOOL},
#ifdef USE_WATT32
  {"*p", "wdebug",                   ARG_BOOL},
#endif
  {"*q", "ftp-create-dirs",          ARG_BOOL},
  {"*r", "create-dirs",              ARG_BOOL},
  {"*R", "create-file-mode",         ARG_STRING},
  {"*s", "max-redirs",               ARG_STRING},
  {"*t", "proxy-ntlm",               ARG_BOOL},
  {"*u", "crlf",                     ARG_BOOL},
  {"*v", "stderr",                   ARG_FILENAME},
  {"*V", "aws-sigv4",                ARG_STRING},
  {"*w", "interface",                ARG_STRING},
  {"*x", "krb",                      ARG_STRING},
  {"*x", "krb4",                     ARG_STRING},
         /* 'krb4' is the previous name */
  {"*X", "haproxy-protocol",         ARG_BOOL},
  {"*y", "max-filesize",             ARG_STRING},
  {"*z", "disable-eprt",             ARG_BOOL},
  {"*Z", "eprt",                     ARG_BOOL},
         /* 'eprt' made like this to make --no-eprt and --eprt to work
             although --disable-eprt is the documented option */
  {"*~", "xattr",                    ARG_BOOL},
  {"$a", "ftp-ssl",                  ARG_BOOL},
         /* 'ftp-ssl' deprecated name since 7.20.0 */
  {"$a", "ssl",                      ARG_BOOL},
         /* 'ssl' new option name in 7.20.0, previously this was ftp-ssl */
  {"$b", "ftp-pasv",                 ARG_BOOL},
  {"$c", "socks5",                   ARG_STRING},
  {"$d", "tcp-nodelay",              ARG_BOOL},
  {"$e", "proxy-digest",             ARG_BOOL},
  {"$f", "proxy-basic",              ARG_BOOL},
  {"$g", "retry",                    ARG_STRING},
  {"$V", "retry-connrefused",        ARG_BOOL},
  {"$h", "retry-delay",              ARG_STRING},
  {"$i", "retry-max-time",           ARG_STRING},
  {"$k", "proxy-negotiate",          ARG_BOOL},
  {"$l", "form-escape",              ARG_BOOL},
  {"$m", "ftp-account",              ARG_STRING},
  {"$n", "proxy-anyauth",            ARG_BOOL},
  {"$o", "trace-time",               ARG_BOOL},
  {"$p", "ignore-content-length",    ARG_BOOL},
  {"$q", "ftp-skip-pasv-ip",         ARG_BOOL},
  {"$r", "ftp-method",               ARG_STRING},
  {"$s", "local-port",               ARG_STRING},
  {"$t", "socks4",                   ARG_STRING},
  {"$T", "socks4a",                  ARG_STRING},
  {"$u", "ftp-alternative-to-user",  ARG_STRING},
  {"$v", "ftp-ssl-reqd",             ARG_BOOL},
         /* 'ftp-ssl-reqd' deprecated name since 7.20.0 */
  {"$v", "ssl-reqd",                 ARG_BOOL},
         /* 'ssl-reqd' new in 7.20.0, previously this was ftp-ssl-reqd */
  {"$w", "sessionid",                ARG_BOOL},
         /* 'sessionid' listed as --no-sessionid in the help */
  {"$x", "ftp-ssl-control",          ARG_BOOL},
  {"$y", "ftp-ssl-ccc",              ARG_BOOL},
  {"$j", "ftp-ssl-ccc-mode",         ARG_STRING},
  {"$z", "libcurl",                  ARG_STRING},
  {"$#", "raw",                      ARG_BOOL},
  {"$0", "post301",                  ARG_BOOL},
  {"$1", "keepalive",                ARG_BOOL},
         /* 'keepalive' listed as --no-keepalive in the help */
  {"$2", "socks5-hostname",          ARG_STRING},
  {"$3", "keepalive-time",           ARG_STRING},
  {"$4", "post302",                  ARG_BOOL},
  {"$5", "noproxy",                  ARG_STRING},
  {"$7", "socks5-gssapi-nec",        ARG_BOOL},
  {"$8", "proxy1.0",                 ARG_STRING},
  {"$9", "tftp-blksize",             ARG_STRING},
  {"$A", "mail-from",                ARG_STRING},
  {"$B", "mail-rcpt",                ARG_STRING},
  {"$C", "ftp-pret",                 ARG_BOOL},
  {"$D", "proto",                    ARG_STRING},
  {"$E", "proto-redir",              ARG_STRING},
  {"$F", "resolve",                  ARG_STRING},
  {"$G", "delegation",               ARG_STRING},
  {"$H", "mail-auth",                ARG_STRING},
  {"$I", "post303",                  ARG_BOOL},
  {"$J", "metalink",                 ARG_BOOL},
  {"$6", "sasl-authzid",             ARG_STRING},
  {"$K", "sasl-ir",                  ARG_BOOL },
  {"$L", "test-event",               ARG_BOOL},
  {"$M", "unix-socket",              ARG_FILENAME},
  {"$N", "path-as-is",               ARG_BOOL},
  {"$O", "socks5-gssapi-service",    ARG_STRING},
         /* 'socks5-gssapi-service' merged with'proxy-service-name' and
            deprecated since 7.49.0 */
  {"$O", "proxy-service-name",       ARG_STRING},
  {"$P", "service-name",             ARG_STRING},
  {"$Q", "proto-default",            ARG_STRING},
  {"$R", "expect100-timeout",        ARG_STRING},
  {"$S", "tftp-no-options",          ARG_BOOL},
  {"$U", "connect-to",               ARG_STRING},
  {"$W", "abstract-unix-socket",     ARG_FILENAME},
  {"$X", "tls-max",                  ARG_STRING},
  {"$Y", "suppress-connect-headers", ARG_BOOL},
  {"$Z", "compressed-ssh",           ARG_BOOL},
  {"$~", "happy-eyeballs-timeout-ms", ARG_STRING},
  {"$!", "retry-all-errors",         ARG_BOOL},
  {"0",   "http1.0",                 ARG_NONE},
  {"01",  "http1.1",                 ARG_NONE},
  {"02",  "http2",                   ARG_NONE},
  {"03",  "http2-prior-knowledge",   ARG_NONE},
  {"04",  "http3",                   ARG_NONE},
  {"09",  "http0.9",                 ARG_BOOL},
  {"1",  "tlsv1",                    ARG_NONE},
  {"10",  "tlsv1.0",                 ARG_NONE},
  {"11",  "tlsv1.1",                 ARG_NONE},
  {"12",  "tlsv1.2",                 ARG_NONE},
  {"13",  "tlsv1.3",                 ARG_NONE},
  {"1A", "tls13-ciphers",            ARG_STRING},
  {"1B", "proxy-tls13-ciphers",      ARG_STRING},
  {"2",  "sslv2",                    ARG_NONE},
  {"3",  "sslv3",                    ARG_NONE},
  {"4",  "ipv4",                     ARG_NONE},
  {"6",  "ipv6",                     ARG_NONE},
  {"a",  "append",                   ARG_BOOL},
  {"A",  "user-agent",               ARG_STRING},
  {"b",  "cookie",                   ARG_STRING},
  {"ba", "alt-svc",                  ARG_STRING},
  {"bb", "hsts",                     ARG_STRING},
  {"B",  "use-ascii",                ARG_BOOL},
  {"c",  "cookie-jar",               ARG_STRING},
  {"C",  "continue-at",              ARG_STRING},
  {"d",  "data",                     ARG_STRING},
  {"dr", "data-raw",                 ARG_STRING},
  {"da", "data-ascii",               ARG_STRING},
  {"db", "data-binary",              ARG_STRING},
  {"de", "data-urlencode",           ARG_STRING},
  {"D",  "dump-header",              ARG_FILENAME},
  {"e",  "referer",                  ARG_STRING},
  {"E",  "cert",                     ARG_FILENAME},
  {"Ea", "cacert",                   ARG_FILENAME},
  {"Eb", "cert-type",                ARG_STRING},
  {"Ec", "key",                      ARG_FILENAME},
  {"Ed", "key-type",                 ARG_STRING},
  {"Ee", "pass",                     ARG_STRING},
  {"Ef", "engine",                   ARG_STRING},
  {"Eg", "capath",                   ARG_FILENAME},
  {"Eh", "pubkey",                   ARG_STRING},
  {"Ei", "hostpubmd5",               ARG_STRING},
  {"EF", "hostpubsha256",            ARG_STRING},
  {"Ej", "crlfile",                  ARG_FILENAME},
  {"Ek", "tlsuser",                  ARG_STRING},
  {"El", "tlspassword",              ARG_STRING},
  {"Em", "tlsauthtype",              ARG_STRING},
  {"En", "ssl-allow-beast",          ARG_BOOL},
  {"Eo", "ssl-auto-client-cert",     ARG_BOOL},
  {"EO", "proxy-ssl-auto-client-cert", ARG_BOOL},
  {"Ep", "pinnedpubkey",             ARG_STRING},
  {"EP", "proxy-pinnedpubkey",       ARG_STRING},
  {"Eq", "cert-status",              ARG_BOOL},
  {"EQ", "doh-cert-status",          ARG_BOOL},
  {"Er", "false-start",              ARG_BOOL},
  {"Es", "ssl-no-revoke",            ARG_BOOL},
  {"ES", "ssl-revoke-best-effort",   ARG_BOOL},
  {"Et", "tcp-fastopen",             ARG_BOOL},
  {"Eu", "proxy-tlsuser",            ARG_STRING},
  {"Ev", "proxy-tlspassword",        ARG_STRING},
  {"Ew", "proxy-tlsauthtype",        ARG_STRING},
  {"Ex", "proxy-cert",               ARG_FILENAME},
  {"Ey", "proxy-cert-type",          ARG_STRING},
  {"Ez", "proxy-key",                ARG_FILENAME},
  {"E0", "proxy-key-type",           ARG_STRING},
  {"E1", "proxy-pass",               ARG_STRING},
  {"E2", "proxy-ciphers",            ARG_STRING},
  {"E3", "proxy-crlfile",            ARG_FILENAME},
  {"E4", "proxy-ssl-allow-beast",    ARG_BOOL},
  {"E5", "login-options",            ARG_STRING},
  {"E6", "proxy-cacert",             ARG_FILENAME},
  {"E7", "proxy-capath",             ARG_FILENAME},
  {"E8", "proxy-insecure",           ARG_BOOL},
  {"E9", "proxy-tlsv1",              ARG_NONE},
  {"EA", "socks5-basic",             ARG_BOOL},
  {"EB", "socks5-gssapi",            ARG_BOOL},
  {"EC", "etag-save",                ARG_FILENAME},
  {"ED", "etag-compare",             ARG_FILENAME},
  {"EE", "curves",                   ARG_STRING},
  {"f",  "fail",                     ARG_BOOL},
  {"fa", "fail-early",               ARG_BOOL},
  {"fb", "styled-output",            ARG_BOOL},
  {"fc", "mail-rcpt-allowfails",     ARG_BOOL},
  {"fd", "fail-with-body",           ARG_BOOL},
  {"F",  "form",                     ARG_STRING},
  {"Fs", "form-string",              ARG_STRING},
  {"g",  "globoff",                  ARG_BOOL},
  {"G",  "get",                      ARG_NONE},
  {"Ga", "request-target",           ARG_STRING},
  {"h",  "help",                     ARG_BOOL},
  {"H",  "header",                   ARG_STRING},
  {"Hp", "proxy-header",             ARG_STRING},
  {"i",  "include",                  ARG_BOOL},
  {"I",  "head",                     ARG_BOOL},
  {"j",  "junk-session-cookies",     ARG_BOOL},
  {"J",  "remote-header-name",       ARG_BOOL},
  {"k",  "insecure",                 ARG_BOOL},
  {"kd", "doh-insecure",             ARG_BOOL},
  {"K",  "config",                   ARG_FILENAME},
  {"l",  "list-only",                ARG_BOOL},
  {"L",  "location",                 ARG_BOOL},
  {"Lt", "location-trusted",         ARG_BOOL},
  {"m",  "max-time",                 ARG_STRING},
  {"M",  "manual",                   ARG_BOOL},
  {"n",  "netrc",                    ARG_BOOL},
  {"no", "netrc-optional",           ARG_BOOL},
  {"ne", "netrc-file",               ARG_FILENAME},
  {"N",  "buffer",                   ARG_BOOL},
         /* 'buffer' listed as --no-buffer in the help */
  {"o",  "output",                   ARG_FILENAME},
  {"O",  "remote-name",              ARG_NONE},
  {"Oa", "remote-name-all",          ARG_BOOL},
  {"Ob", "output-dir",               ARG_STRING},
  {"p",  "proxytunnel",              ARG_BOOL},
  {"P",  "ftp-port",                 ARG_STRING},
  {"q",  "disable",                  ARG_BOOL},
  {"Q",  "quote",                    ARG_STRING},
  {"r",  "range",                    ARG_STRING},
  {"R",  "remote-time",              ARG_BOOL},
  {"s",  "silent",                   ARG_BOOL},
  {"S",  "show-error",               ARG_BOOL},
  {"t",  "telnet-option",            ARG_STRING},
  {"T",  "upload-file",              ARG_FILENAME},
  {"u",  "user",                     ARG_STRING},
  {"U",  "proxy-user",               ARG_STRING},
  {"v",  "verbose",                  ARG_BOOL},
  {"V",  "version",                  ARG_BOOL},
  {"w",  "write-out",                ARG_STRING},
  {"x",  "proxy",                    ARG_STRING},
  {"xa", "preproxy",                 ARG_STRING},
  {"X",  "request",                  ARG_STRING},
  {"Y",  "speed-limit",              ARG_STRING},
  {"y",  "speed-time",               ARG_STRING},
  {"z",  "time-cond",                ARG_STRING},
  {"Z",  "parallel",                 ARG_BOOL},
  {"Zb", "parallel-max",             ARG_STRING},
  {"Zc", "parallel-immediate",       ARG_BOOL},
  {"#",  "progress-bar",             ARG_BOOL},
  {"#m", "progress-meter",           ARG_BOOL},
  {":",  "next",                     ARG_NONE},
};

/* Split the argument of -E to 'certname' and 'passphrase' separated by colon.
 * We allow ':' and '\' to be escaped by '\' so that we can use certificate
 * nicknames containing ':'.  See <https://sourceforge.net/p/curl/bugs/1196/>
 * for details. */
#ifndef UNITTESTS
static
#endif
void parse_cert_parameter(const char *cert_parameter,
                          char **certname,
                          char **passphrase)
{
  size_t param_length = strlen(cert_parameter);
  size_t span;
  const char *param_place = NULL;
  char *certname_place = NULL;
  *certname = NULL;
  *passphrase = NULL;

  /* most trivial assumption: cert_parameter is empty */
  if(param_length == 0)
    return;

  /* next less trivial: cert_parameter starts 'pkcs11:' and thus
   * looks like a RFC7512 PKCS#11 URI which can be used as-is.
   * Also if cert_parameter contains no colon nor backslash, this
   * means no passphrase was given and no characters escaped */
  if(curl_strnequal(cert_parameter, "pkcs11:", 7) ||
     !strpbrk(cert_parameter, ":\\")) {
    *certname = strdup(cert_parameter);
    return;
  }
  /* deal with escaped chars; find unescaped colon if it exists */
  certname_place = malloc(param_length + 1);
  if(!certname_place)
    return;

  *certname = certname_place;
  param_place = cert_parameter;
  while(*param_place) {
    span = strcspn(param_place, ":\\");
    strncpy(certname_place, param_place, span);
    param_place += span;
    certname_place += span;
    /* we just ate all the non-special chars. now we're on either a special
     * char or the end of the string. */
    switch(*param_place) {
    case '\0':
      break;
    case '\\':
      param_place++;
      switch(*param_place) {
        case '\0':
          *certname_place++ = '\\';
          break;
        case '\\':
          *certname_place++ = '\\';
          param_place++;
          break;
        case ':':
          *certname_place++ = ':';
          param_place++;
          break;
        default:
          *certname_place++ = '\\';
          *certname_place++ = *param_place;
          param_place++;
          break;
      }
      break;
    case ':':
      /* Since we live in a world of weirdness and confusion, the win32
         dudes can use : when using drive letters and thus c:\file:password
         needs to work. In order not to break compatibility, we still use : as
         separator, but we try to detect when it is used for a file name! On
         windows. */
#ifdef WIN32
      if(param_place &&
          (param_place == &cert_parameter[1]) &&
          (cert_parameter[2] == '\\' || cert_parameter[2] == '/') &&
          (ISALPHA(cert_parameter[0])) ) {
        /* colon in the second column, followed by a backslash, and the
           first character is an alphabetic letter:

           this is a drive letter colon */
        *certname_place++ = ':';
        param_place++;
        break;
      }
#endif
      /* escaped colons and Windows drive letter colons were handled
       * above; if we're still here, this is a separating colon */
      param_place++;
      if(*param_place) {
        *passphrase = strdup(param_place);
      }
      goto done;
    }
  }
done:
  *certname_place = '\0';
}

/* Replace (in-place) '%20' by '+' according to RFC1866 */
static size_t replace_url_encoded_space_by_plus(char *url)
{
  size_t orig_len = strlen(url);
  size_t orig_index = 0;
  size_t new_index = 0;

  while(orig_index < orig_len) {
    if((url[orig_index] == '%') &&
       (url[orig_index + 1] == '2') &&
       (url[orig_index + 2] == '0')) {
      url[new_index] = '+';
      orig_index += 3;
    }
    else{
      if(new_index != orig_index) {
        url[new_index] = url[orig_index];
      }
      orig_index++;
    }
    new_index++;
  }

  url[new_index] = 0; /* terminate string */

  return new_index; /* new size */
}

static void
GetFileAndPassword(char *nextarg, char **file, char **password)
{
  char *certname, *passphrase;
  parse_cert_parameter(nextarg, &certname, &passphrase);
  Curl_safefree(*file);
  *file = certname;
  if(passphrase) {
    Curl_safefree(*password);
    *password = passphrase;
  }
  cleanarg(nextarg);
}

/* Get a size parameter for '--limit-rate' or '--max-filesize'.
 * We support a 'G', 'M' or 'K' suffix too.
  */
static ParameterError GetSizeParameter(struct GlobalConfig *global,
                                       const char *arg,
                                       const char *which,
                                       curl_off_t *value_out)
{
  char *unit;
  curl_off_t value;

  if(curlx_strtoofft(arg, &unit, 0, &value)) {
    warnf(global, "invalid number specified for %s\n", which);
    return PARAM_BAD_USE;
  }

  if(!*unit)
    unit = (char *)"b";
  else if(strlen(unit) > 1)
    unit = (char *)"w"; /* unsupported */

  switch(*unit) {
  case 'G':
  case 'g':
    if(value > (CURL_OFF_T_MAX / (1024*1024*1024)))
      return PARAM_NUMBER_TOO_LARGE;
    value *= 1024*1024*1024;
    break;
  case 'M':
  case 'm':
    if(value > (CURL_OFF_T_MAX / (1024*1024)))
      return PARAM_NUMBER_TOO_LARGE;
    value *= 1024*1024;
    break;
  case 'K':
  case 'k':
    if(value > (CURL_OFF_T_MAX / 1024))
      return PARAM_NUMBER_TOO_LARGE;
    value *= 1024;
    break;
  case 'b':
  case 'B':
    /* for plain bytes, leave as-is */
    break;
  default:
    warnf(global, "unsupported %s unit. Use G, M, K or B!\n", which);
    return PARAM_BAD_USE;
  }
  *value_out = value;
  return PARAM_OK;
}

ParameterError getparameter(const char *flag, /* f or -long-flag */
                            char *nextarg,    /* NULL if unset */
                            bool *usedarg,    /* set to TRUE if the arg
                                                 has been used */
                            struct GlobalConfig *global,
                            struct OperationConfig *config)
{
  char letter;
  char subletter = '\0'; /* subletters can only occur on long options */
  int rc;
  const char *parse = NULL;
  unsigned int j;
  time_t now;
  int hit = -1;
  bool longopt = FALSE;
  bool singleopt = FALSE; /* when true means '-o foo' used '-ofoo' */
  ParameterError err;
  bool toggle = TRUE; /* how to switch boolean options, on or off. Controlled
                         by using --OPTION or --no-OPTION */

  *usedarg = FALSE; /* default is that we don't use the arg */

  if(('-' != flag[0]) || ('-' == flag[1])) {
    /* this should be a long name */
    const char *word = ('-' == flag[0]) ? flag + 2 : flag;
    size_t fnam = strlen(word);
    int numhits = 0;
    bool noflagged = FALSE;

    if(!strncmp(word, "no-", 3)) {
      /* disable this option but ignore the "no-" part when looking for it */
      word += 3;
      toggle = FALSE;
      noflagged = TRUE;
    }

    for(j = 0; j < sizeof(aliases)/sizeof(aliases[0]); j++) {
      if(curl_strnequal(aliases[j].lname, word, fnam)) {
        longopt = TRUE;
        numhits++;
        if(curl_strequal(aliases[j].lname, word)) {
          parse = aliases[j].letter;
          hit = j;
          numhits = 1; /* a single unique hit */
          break;
        }
        parse = aliases[j].letter;
        hit = j;
      }
    }
    if(numhits > 1) {
      /* this is at least the second match! */
      return PARAM_OPTION_AMBIGUOUS;
    }
    if(hit < 0) {
      return PARAM_OPTION_UNKNOWN;
    }
    if(noflagged && (aliases[hit].desc != ARG_BOOL))
      /* --no- prefixed an option that isn't boolean! */
      return PARAM_NO_NOT_BOOLEAN;
  }
  else {
    flag++; /* prefixed with one dash, pass it */
    hit = -1;
    parse = flag;
  }

  do {
    /* we can loop here if we have multiple single-letters */

    if(!longopt) {
      letter = (char)*parse;
      subletter = '\0';
    }
    else {
      letter = parse[0];
      subletter = parse[1];
    }

    if(hit < 0) {
      for(j = 0; j < sizeof(aliases)/sizeof(aliases[0]); j++) {
        if(letter == aliases[j].letter[0]) {
          hit = j;
          break;
        }
      }
      if(hit < 0) {
        return PARAM_OPTION_UNKNOWN;
      }
    }

    if(aliases[hit].desc >= ARG_STRING) {
      /* this option requires an extra parameter */
      if(!longopt && parse[1]) {
        nextarg = (char *)&parse[1]; /* this is the actual extra parameter */
        singleopt = TRUE;   /* don't loop anymore after this */
      }
      else if(!nextarg)
        return PARAM_REQUIRES_PARAMETER;
      else
        *usedarg = TRUE; /* mark it as used */

      if((aliases[hit].desc == ARG_FILENAME) &&
         (nextarg[0] == '-') && nextarg[1]) {
        /* if the file name looks like a command line option */
        warnf(global, "The file name argument '%s' looks like a flag.\n",
              nextarg);
      }
    }
    else if((aliases[hit].desc == ARG_NONE) && !toggle)
      return PARAM_NO_PREFIX;

    switch(letter) {
    case '*': /* options without a short option */
      switch(subletter) {
      case '4': /* --dns-ipv4-addr */
        /* addr in dot notation */
        GetStr(&config->dns_ipv4_addr, nextarg);
        break;
      case '6': /* --dns-ipv6-addr */
        /* addr in dot notation */
        GetStr(&config->dns_ipv6_addr, nextarg);
        break;
      case 'a': /* random-file */
        GetStr(&config->random_file, nextarg);
        break;
      case 'b': /* egd-file */
        GetStr(&config->egd_file, nextarg);
        break;
      case 'B': /* OAuth 2.0 bearer token */
        GetStr(&config->oauth_bearer, nextarg);
        config->authtype |= CURLAUTH_BEARER;
        break;
      case 'c': /* connect-timeout */
        err = str2udouble(&config->connecttimeout, nextarg,
                          LONG_MAX/1000);
        if(err)
          return err;
        break;
      case 'C': /* doh-url */
        GetStr(&config->doh_url, nextarg);
        break;
      case 'd': /* ciphers */
        GetStr(&config->cipher_list, nextarg);
        break;
      case 'D': /* --dns-interface */
        /* interface name */
        GetStr(&config->dns_interface, nextarg);
        break;
      case 'e': /* --disable-epsv */
        config->disable_epsv = toggle;
        break;
      case 'f': /* --disallow-username-in-url */
        config->disallow_username_in_url = toggle;
        break;
      case 'E': /* --epsv */
        config->disable_epsv = (!toggle)?TRUE:FALSE;
        break;
      case 'F': /* --dns-servers */
        /* IP addrs of DNS servers */
        GetStr(&config->dns_servers, nextarg);
        break;
      case 'g': /* --trace */
        GetStr(&global->trace_dump, nextarg);
        if(global->tracetype && (global->tracetype != TRACE_BIN))
          warnf(global, "--trace overrides an earlier trace/verbose option\n");
        global->tracetype = TRACE_BIN;
        break;
      case 'G': /* --npn */
        config->nonpn = (!toggle)?TRUE:FALSE;
        break;
      case 'h': /* --trace-ascii */
        GetStr(&global->trace_dump, nextarg);
        if(global->tracetype && (global->tracetype != TRACE_ASCII))
          warnf(global,
                "--trace-ascii overrides an earlier trace/verbose option\n");
        global->tracetype = TRACE_ASCII;
        break;
      case 'H': /* --alpn */
        config->noalpn = (!toggle)?TRUE:FALSE;
        break;
      case 'i': /* --limit-rate */
      {
        curl_off_t value;
        ParameterError pe = GetSizeParameter(global, nextarg, "rate", &value);

        if(pe != PARAM_OK)
           return pe;
        config->recvpersecond = value;
        config->sendpersecond = value;
      }
      break;

      case 'j': /* --compressed */
        if(toggle &&
           !(curlinfo->features & (CURL_VERSION_LIBZ |
                                   CURL_VERSION_BROTLI | CURL_VERSION_ZSTD)))
          return PARAM_LIBCURL_DOESNT_SUPPORT;
        config->encoding = toggle;
        break;

      case 'J': /* --tr-encoding */
        config->tr_encoding = toggle;
        break;

      case 'k': /* --digest */
        if(toggle)
          config->authtype |= CURLAUTH_DIGEST;
        else
          config->authtype &= ~CURLAUTH_DIGEST;
        break;

      case 'l': /* --negotiate */
        if(toggle) {
          if(curlinfo->features & CURL_VERSION_SPNEGO)
            config->authtype |= CURLAUTH_NEGOTIATE;
          else
            return PARAM_LIBCURL_DOESNT_SUPPORT;
        }
        else
          config->authtype &= ~CURLAUTH_NEGOTIATE;
        break;

      case 'm': /* --ntlm */
        if(toggle) {
          if(curlinfo->features & CURL_VERSION_NTLM)
            config->authtype |= CURLAUTH_NTLM;
          else
            return PARAM_LIBCURL_DOESNT_SUPPORT;
        }
        else
          config->authtype &= ~CURLAUTH_NTLM;
        break;

      case 'M': /* --ntlm-wb */
        if(toggle) {
          if(curlinfo->features & CURL_VERSION_NTLM_WB)
            config->authtype |= CURLAUTH_NTLM_WB;
          else
            return PARAM_LIBCURL_DOESNT_SUPPORT;
        }
        else
          config->authtype &= ~CURLAUTH_NTLM_WB;
        break;

      case 'n': /* --basic for completeness */
        if(toggle)
          config->authtype |= CURLAUTH_BASIC;
        else
          config->authtype &= ~CURLAUTH_BASIC;
        break;

      case 'o': /* --anyauth, let libcurl pick it */
        if(toggle)
          config->authtype = CURLAUTH_ANY;
        /* --no-anyauth simply doesn't touch it */
        break;

#ifdef USE_WATT32
      case 'p': /* --wdebug */
        dbug_init();
        break;
#endif
      case 'q': /* --ftp-create-dirs */
        config->ftp_create_dirs = toggle;
        break;

      case 'r': /* --create-dirs */
        config->create_dirs = toggle;
        break;

      case 'R': /* --create-file-mode */
        err = oct2nummax(&config->create_file_mode, nextarg, 0777);
        if(err)
          return err;
        break;

      case 's': /* --max-redirs */
        /* specified max no of redirects (http(s)), this accepts -1 as a
           special condition */
        err = str2num(&config->maxredirs, nextarg);
        if(err)
          return err;
        if(config->maxredirs < -1)
          return PARAM_BAD_NUMERIC;
        break;

      case 't': /* --proxy-ntlm */
        if(curlinfo->features & CURL_VERSION_NTLM)
          config->proxyntlm = toggle;
        else
          return PARAM_LIBCURL_DOESNT_SUPPORT;
        break;

      case 'u': /* --crlf */
        /* LF -> CRLF conversion? */
        config->crlf = toggle;
        break;

      case 'V': /* --aws-sigv4 */
        config->authtype |= CURLAUTH_AWS_SIGV4;
        GetStr(&config->aws_sigv4, nextarg);
        break;

      case 'v': /* --stderr */
        if(strcmp(nextarg, "-")) {
          FILE *newfile = fopen(nextarg, FOPEN_WRITETEXT);
          if(!newfile)
            warnf(global, "Failed to open %s!\n", nextarg);
          else {
            if(global->errors_fopened)
              fclose(global->errors);
            global->errors = newfile;
            global->errors_fopened = TRUE;
          }
        }
        else
          global->errors = stdout;
        break;
      case 'w': /* --interface */
        /* interface */
        GetStr(&config->iface, nextarg);
        break;
      case 'x': /* --krb */
        /* kerberos level string */
        if(curlinfo->features & CURL_VERSION_SPNEGO)
          GetStr(&config->krblevel, nextarg);
        else
          return PARAM_LIBCURL_DOESNT_SUPPORT;
        break;
      case 'X': /* --haproxy-protocol */
        config->haproxy_protocol = toggle;
        break;
      case 'y': /* --max-filesize */
        {
          curl_off_t value;
          ParameterError pe =
            GetSizeParameter(global, nextarg, "max-filesize", &value);

          if(pe != PARAM_OK)
             return pe;
          config->max_filesize = value;
        }
        break;
      case 'z': /* --disable-eprt */
        config->disable_eprt = toggle;
        break;
      case 'Z': /* --eprt */
        config->disable_eprt = (!toggle)?TRUE:FALSE;
        break;
      case '~': /* --xattr */
        config->xattr = toggle;
        break;
      case '@': /* the URL! */
      {
        struct getout *url;

        if(!config->url_get)
          config->url_get = config->url_list;

        if(config->url_get) {
          /* there's a node here, if it already is filled-in continue to find
             an "empty" node */
          while(config->url_get && (config->url_get->flags & GETOUT_URL))
            config->url_get = config->url_get->next;
        }

        /* now there might or might not be an available node to fill in! */

        if(config->url_get)
          /* existing node */
          url = config->url_get;
        else
          /* there was no free node, create one! */
          config->url_get = url = new_getout(config);

        if(!url)
          return PARAM_NO_MEM;

        /* fill in the URL */
        GetStr(&url->url, nextarg);
        url->flags |= GETOUT_URL;
      }
      }
      break;
    case '$': /* more options without a short option */
      switch(subletter) {
      case 'a': /* --ssl */
        if(toggle && !(curlinfo->features & CURL_VERSION_SSL))
          return PARAM_LIBCURL_DOESNT_SUPPORT;
        config->ftp_ssl = toggle;
        break;
      case 'b': /* --ftp-pasv */
        Curl_safefree(config->ftpport);
        break;
      case 'c': /* --socks5 specifies a socks5 proxy to use, and resolves
                   the name locally and passes on the resolved address */
        GetStr(&config->proxy, nextarg);
        config->proxyver = CURLPROXY_SOCKS5;
        break;
      case 't': /* --socks4 specifies a socks4 proxy to use */
        GetStr(&config->proxy, nextarg);
        config->proxyver = CURLPROXY_SOCKS4;
        break;
      case 'T': /* --socks4a specifies a socks4a proxy to use */
        GetStr(&config->proxy, nextarg);
        config->proxyver = CURLPROXY_SOCKS4A;
        break;
      case '2': /* --socks5-hostname specifies a socks5 proxy and enables name
                   resolving with the proxy */
        GetStr(&config->proxy, nextarg);
        config->proxyver = CURLPROXY_SOCKS5_HOSTNAME;
        break;
      case 'd': /* --tcp-nodelay option */
        config->tcp_nodelay = toggle;
        break;
      case 'e': /* --proxy-digest */
        config->proxydigest = toggle;
        break;
      case 'f': /* --proxy-basic */
        config->proxybasic = toggle;
        break;
      case 'g': /* --retry */
        err = str2unum(&config->req_retry, nextarg);
        if(err)
          return err;
        break;
      case 'V': /* --retry-connrefused */
        config->retry_connrefused = toggle;
        break;
      case 'h': /* --retry-delay */
        err = str2unummax(&config->retry_delay, nextarg, LONG_MAX/1000);
        if(err)
          return err;
        break;
      case 'i': /* --retry-max-time */
        err = str2unummax(&config->retry_maxtime, nextarg, LONG_MAX/1000);
        if(err)
          return err;
        break;
      case '!': /* --retry-all-errors */
        config->retry_all_errors = toggle;
        break;

      case 'k': /* --proxy-negotiate */
        if(curlinfo->features & CURL_VERSION_SPNEGO)
          config->proxynegotiate = toggle;
        else
          return PARAM_LIBCURL_DOESNT_SUPPORT;
        break;

      case 'l': /* --form-escape */
        config->mime_options &= ~CURLMIMEOPT_FORMESCAPE;
        if(toggle)
          config->mime_options |= CURLMIMEOPT_FORMESCAPE;
        break;

      case 'm': /* --ftp-account */
        GetStr(&config->ftp_account, nextarg);
        break;
      case 'n': /* --proxy-anyauth */
        config->proxyanyauth = toggle;
        break;
      case 'o': /* --trace-time */
        global->tracetime = toggle;
        break;
      case 'p': /* --ignore-content-length */
        config->ignorecl = toggle;
        break;
      case 'q': /* --ftp-skip-pasv-ip */
        config->ftp_skip_ip = toggle;
        break;
      case 'r': /* --ftp-method (undocumented at this point) */
        config->ftp_filemethod = ftpfilemethod(config, nextarg);
        break;
      case 's': { /* --local-port */
        /* 16bit base 10 is 5 digits, but we allow 6 so that this catches
           overflows, not just truncates */
        char lrange[7]="";
        char *p = nextarg;
        while(ISDIGIT(*p))
          p++;
        if(*p) {
          /* if there's anything more than a plain decimal number */
          rc = sscanf(p, " - %6s", lrange);
          *p = 0; /* null-terminate to make str2unum() work below */
        }
        else
          rc = 0;

        err = str2unum(&config->localport, nextarg);
        if(err || (config->localport > 65535))
          return PARAM_BAD_USE;
        if(!rc)
          config->localportrange = 1; /* default number of ports to try */
        else {
          err = str2unum(&config->localportrange, lrange);
          if(err || (config->localportrange > 65535))
            return PARAM_BAD_USE;
          config->localportrange -= (config->localport-1);
          if(config->localportrange < 1)
            return PARAM_BAD_USE;
        }
        break;
      }
      case 'u': /* --ftp-alternative-to-user */
        GetStr(&config->ftp_alternative_to_user, nextarg);
        break;
      case 'v': /* --ssl-reqd */
        if(toggle && !(curlinfo->features & CURL_VERSION_SSL))
          return PARAM_LIBCURL_DOESNT_SUPPORT;
        config->ftp_ssl_reqd = toggle;
        break;
      case 'w': /* --no-sessionid */
        config->disable_sessionid = (!toggle)?TRUE:FALSE;
        break;
      case 'x': /* --ftp-ssl-control */
        if(toggle && !(curlinfo->features & CURL_VERSION_SSL))
          return PARAM_LIBCURL_DOESNT_SUPPORT;
        config->ftp_ssl_control = toggle;
        break;
      case 'y': /* --ftp-ssl-ccc */
        config->ftp_ssl_ccc = toggle;
        if(!config->ftp_ssl_ccc_mode)
          config->ftp_ssl_ccc_mode = CURLFTPSSL_CCC_PASSIVE;
        break;
      case 'j': /* --ftp-ssl-ccc-mode */
        config->ftp_ssl_ccc = TRUE;
        config->ftp_ssl_ccc_mode = ftpcccmethod(config, nextarg);
        break;
      case 'z': /* --libcurl */
#ifdef CURL_DISABLE_LIBCURL_OPTION
        warnf(global,
              "--libcurl option was disabled at build-time!\n");
        return PARAM_OPTION_UNKNOWN;
#else
        GetStr(&global->libcurl, nextarg);
        break;
#endif
      case '#': /* --raw */
        config->raw = toggle;
        break;
      case '0': /* --post301 */
        config->post301 = toggle;
        break;
      case '1': /* --no-keepalive */
        config->nokeepalive = (!toggle)?TRUE:FALSE;
        break;
      case '3': /* --keepalive-time */
        err = str2unum(&config->alivetime, nextarg);
        if(err)
          return err;
        break;
      case '4': /* --post302 */
        config->post302 = toggle;
        break;
      case 'I': /* --post303 */
        config->post303 = toggle;
        break;
      case '5': /* --noproxy */
        /* This specifies the noproxy list */
        GetStr(&config->noproxy, nextarg);
        break;
       case '7': /* --socks5-gssapi-nec*/
        config->socks5_gssapi_nec = toggle;
        break;
      case '8': /* --proxy1.0 */
        /* http 1.0 proxy */
        GetStr(&config->proxy, nextarg);
        config->proxyver = CURLPROXY_HTTP_1_0;
        break;
      case '9': /* --tftp-blksize */
        err = str2unum(&config->tftp_blksize, nextarg);
        if(err)
          return err;
        break;
      case 'A': /* --mail-from */
        GetStr(&config->mail_from, nextarg);
        break;
      case 'B': /* --mail-rcpt */
        /* append receiver to a list */
        err = add2list(&config->mail_rcpt, nextarg);
        if(err)
          return err;
        break;
      case 'C': /* --ftp-pret */
        config->ftp_pret = toggle;
        break;
      case 'D': /* --proto */
        config->proto_present = TRUE;
        if(proto2num(config, &config->proto, nextarg))
          return PARAM_BAD_USE;
        break;
      case 'E': /* --proto-redir */
        config->proto_redir_present = TRUE;
        if(proto2num(config, &config->proto_redir, nextarg))
          return PARAM_BAD_USE;
        break;
      case 'F': /* --resolve */
        err = add2list(&config->resolve, nextarg);
        if(err)
          return err;
        break;
      case 'G': /* --delegation LEVEL */
        config->gssapi_delegation = delegation(config, nextarg);
        break;
      case 'H': /* --mail-auth */
        GetStr(&config->mail_auth, nextarg);
        break;
      case 'J': /* --metalink */
        errorf(global, "--metalink is disabled\n");
        return PARAM_BAD_USE;
      case '6': /* --sasl-authzid */
        GetStr(&config->sasl_authzid, nextarg);
        break;
      case 'K': /* --sasl-ir */
        config->sasl_ir = toggle;
        break;
      case 'L': /* --test-event */
#ifdef CURLDEBUG
        global->test_event_based = toggle;
#else
        warnf(global, "--test-event is ignored unless a debug build!\n");
#endif
        break;
      case 'M': /* --unix-socket */
        config->abstract_unix_socket = FALSE;
        GetStr(&config->unix_socket_path, nextarg);
        break;
      case 'N': /* --path-as-is */
        config->path_as_is = toggle;
        break;
      case 'O': /* --proxy-service-name */
        GetStr(&config->proxy_service_name, nextarg);
        break;
      case 'P': /* --service-name */
        GetStr(&config->service_name, nextarg);
        break;
      case 'Q': /* --proto-default */
        GetStr(&config->proto_default, nextarg);
        err = check_protocol(config->proto_default);
        if(err)
          return err;
        break;
      case 'R': /* --expect100-timeout */
        err = str2udouble(&config->expect100timeout, nextarg, LONG_MAX/1000);
        if(err)
          return err;
        break;
      case 'S': /* --tftp-no-options */
        config->tftp_no_options = toggle;
        break;
      case 'U': /* --connect-to */
        err = add2list(&config->connect_to, nextarg);
        if(err)
          return err;
        break;
      case 'W': /* --abstract-unix-socket */
        config->abstract_unix_socket = TRUE;
        GetStr(&config->unix_socket_path, nextarg);
        break;
      case 'X': /* --tls-max */
        err = str2tls_max(&config->ssl_version_max, nextarg);
        if(err)
          return err;
        break;
      case 'Y': /* --suppress-connect-headers */
        config->suppress_connect_headers = toggle;
        break;
      case 'Z': /* --compressed-ssh */
        config->ssh_compression = toggle;
        break;
      case '~': /* --happy-eyeballs-timeout-ms */
        err = str2unum(&config->happy_eyeballs_timeout_ms, nextarg);
        if(err)
          return err;
        /* 0 is a valid value for this timeout */
        break;
      }
      break;
    case '#':
      switch(subletter) {
      case 'm': /* --progress-meter */
        global->noprogress = !toggle;
        break;
      default:  /* --progress-bar */
        global->progressmode =
          toggle ? CURL_PROGRESS_BAR : CURL_PROGRESS_STATS;
        break;
      }
      break;
    case ':': /* --next */
      return PARAM_NEXT_OPERATION;
    case '0': /* --http* options */
      switch(subletter) {
      case '\0':
        /* HTTP version 1.0 */
        config->httpversion = CURL_HTTP_VERSION_1_0;
        break;
      case '1':
        /* HTTP version 1.1 */
        config->httpversion = CURL_HTTP_VERSION_1_1;
        break;
      case '2':
        /* HTTP version 2.0 */
        config->httpversion = CURL_HTTP_VERSION_2_0;
        break;
      case '3': /* --http2-prior-knowledge */
        /* HTTP version 2.0 over clean TCP*/
        config->httpversion = CURL_HTTP_VERSION_2_PRIOR_KNOWLEDGE;
        break;
      case '4': /* --http3 */
        /* HTTP version 3 go over QUIC - at once */
        if(curlinfo->features & CURL_VERSION_HTTP3)
          config->httpversion = CURL_HTTP_VERSION_3;
        else
          return PARAM_LIBCURL_DOESNT_SUPPORT;
        break;
      case '9':
        /* Allow HTTP/0.9 responses! */
        config->http09_allowed = toggle;
        break;
      }
      break;
    case '1': /* --tlsv1* options */
      switch(subletter) {
      case '\0':
        /* TLS version 1.x */
        config->ssl_version = CURL_SSLVERSION_TLSv1;
        break;
      case '0':
        /* TLS version 1.0 */
        config->ssl_version = CURL_SSLVERSION_TLSv1_0;
        break;
      case '1':
        /* TLS version 1.1 */
        config->ssl_version = CURL_SSLVERSION_TLSv1_1;
        break;
      case '2':
        /* TLS version 1.2 */
        config->ssl_version = CURL_SSLVERSION_TLSv1_2;
        break;
      case '3':
        /* TLS version 1.3 */
        config->ssl_version = CURL_SSLVERSION_TLSv1_3;
        break;
      case 'A': /* --tls13-ciphers */
        GetStr(&config->cipher13_list, nextarg);
        break;
      case 'B': /* --proxy-tls13-ciphers */
        GetStr(&config->proxy_cipher13_list, nextarg);
        break;
      }
      break;
    case '2':
      /* SSL version 2 */
      warnf(global, "Ignores instruction to use SSLv2\n");
      break;
    case '3':
      /* SSL version 3 */
      warnf(global, "Ignores instruction to use SSLv3\n");
      break;
    case '4':
      /* IPv4 */
      config->ip_version = CURL_IPRESOLVE_V4;
      break;
    case '6':
      /* IPv6 */
      config->ip_version = CURL_IPRESOLVE_V6;
      break;
    case 'a':
      /* This makes the FTP sessions use APPE instead of STOR */
      config->ftp_append = toggle;
      break;
    case 'A':
      /* This specifies the User-Agent name */
      GetStr(&config->useragent, nextarg);
      break;
    case 'b':
      switch(subletter) {
      case 'a': /* --alt-svc */
        if(curlinfo->features & CURL_VERSION_ALTSVC)
          GetStr(&config->altsvc, nextarg);
        else
          return PARAM_LIBCURL_DOESNT_SUPPORT;
        break;
      case 'b': /* --hsts */
        if(curlinfo->features & CURL_VERSION_HSTS)
          GetStr(&config->hsts, nextarg);
        else
          return PARAM_LIBCURL_DOESNT_SUPPORT;
        break;
      default:  /* --cookie string coming up: */
        if(nextarg[0] == '@') {
          nextarg++;
        }
        else if(strchr(nextarg, '=')) {
          /* A cookie string must have a =-letter */
          err = add2list(&config->cookies, nextarg);
          if(err)
            return err;
          break;
        }
        /* We have a cookie file to read from! */
        err = add2list(&config->cookiefiles, nextarg);
        if(err)
          return err;
      }
      break;
    case 'B':
      /* use ASCII/text when transferring */
      config->use_ascii = toggle;
      break;
    case 'c':
      /* get the file name to dump all cookies in */
      GetStr(&config->cookiejar, nextarg);
      break;
    case 'C':
      /* This makes us continue an ftp transfer at given position */
      if(strcmp(nextarg, "-")) {
        err = str2offset(&config->resume_from, nextarg);
        if(err)
          return err;
        config->resume_from_current = FALSE;
      }
      else {
        config->resume_from_current = TRUE;
        config->resume_from = 0;
      }
      config->use_resume = TRUE;
      break;
    case 'd':
      /* postfield data */
    {
      char *postdata = NULL;
      FILE *file;
      size_t size = 0;
      bool raw_mode = (subletter == 'r');

      if(subletter == 'e') { /* --data-urlencode*/
        /* [name]=[content], we encode the content part only
         * [name]@[file name]
         *
         * Case 2: we first load the file using that name and then encode
         * the content.
         */
        const char *p = strchr(nextarg, '=');
        size_t nlen;
        char is_file;
        if(!p)
          /* there was no '=' letter, check for a '@' instead */
          p = strchr(nextarg, '@');
        if(p) {
          nlen = p - nextarg; /* length of the name part */
          is_file = *p++; /* pass the separator */
        }
        else {
          /* neither @ nor =, so no name and it isn't a file */
          nlen = is_file = 0;
          p = nextarg;
        }
        if('@' == is_file) {
          /* a '@' letter, it means that a file name or - (stdin) follows */
          if(!strcmp("-", p)) {
            file = stdin;
            set_binmode(stdin);
          }
          else {
            file = fopen(p, "rb");
            if(!file)
              warnf(global,
                    "Couldn't read data from file \"%s\", this makes "
                    "an empty POST.\n", nextarg);
          }

          err = file2memory(&postdata, &size, file);

          if(file && (file != stdin))
            fclose(file);
          if(err)
            return err;
        }
        else {
          GetStr(&postdata, p);
          if(postdata)
            size = strlen(postdata);
        }

        if(!postdata) {
          /* no data from the file, point to a zero byte string to make this
             get sent as a POST anyway */
          postdata = strdup("");
          if(!postdata)
            return PARAM_NO_MEM;
          size = 0;
        }
        else {
          char *enc = curl_easy_escape(NULL, postdata, (int)size);
          Curl_safefree(postdata); /* no matter if it worked or not */
          if(enc) {
            /* replace (in-place) '%20' by '+' according to RFC1866 */
            size_t enclen = replace_url_encoded_space_by_plus(enc);
            /* now make a string with the name from above and append the
               encoded string */
            size_t outlen = nlen + enclen + 2;
            char *n = malloc(outlen);
            if(!n) {
              curl_free(enc);
              return PARAM_NO_MEM;
            }
            if(nlen > 0) { /* only append '=' if we have a name */
              msnprintf(n, outlen, "%.*s=%s", nlen, nextarg, enc);
              size = outlen-1;
            }
            else {
              strcpy(n, enc);
              size = outlen-2; /* since no '=' was inserted */
            }
            curl_free(enc);
            postdata = n;
          }
          else
            return PARAM_NO_MEM;
        }
      }
      else if('@' == *nextarg && !raw_mode) {
        /* the data begins with a '@' letter, it means that a file name
           or - (stdin) follows */
        nextarg++; /* pass the @ */

        if(!strcmp("-", nextarg)) {
          file = stdin;
          if(subletter == 'b') /* forced data-binary */
            set_binmode(stdin);
        }
        else {
          file = fopen(nextarg, "rb");
          if(!file)
            warnf(global, "Couldn't read data from file \"%s\", this makes "
                  "an empty POST.\n", nextarg);
        }

        if(subletter == 'b')
          /* forced binary */
          err = file2memory(&postdata, &size, file);
        else {
          err = file2string(&postdata, file);
          if(postdata)
            size = strlen(postdata);
        }

        if(file && (file != stdin))
          fclose(file);
        if(err)
          return err;

        if(!postdata) {
          /* no data from the file, point to a zero byte string to make this
             get sent as a POST anyway */
          postdata = strdup("");
          if(!postdata)
            return PARAM_NO_MEM;
        }
      }
      else {
        GetStr(&postdata, nextarg);
        if(postdata)
          size = strlen(postdata);
      }

#ifdef CURL_DOES_CONVERSIONS
      if(subletter != 'b') {
        /* NOT forced binary, convert to ASCII */
        if(convert_to_network(postdata, strlen(postdata))) {
          Curl_safefree(postdata);
          return PARAM_NO_MEM;
        }
      }
#endif

      if(config->postfields) {
        /* we already have a string, we append this one with a separating
           &-letter */
        char *oldpost = config->postfields;
        curl_off_t oldlen = config->postfieldsize;
        curl_off_t newlen = oldlen + curlx_uztoso(size) + 2;
        config->postfields = malloc((size_t)newlen);
        if(!config->postfields) {
          Curl_safefree(oldpost);
          Curl_safefree(postdata);
          return PARAM_NO_MEM;
        }
        memcpy(config->postfields, oldpost, (size_t)oldlen);
        /* use byte value 0x26 for '&' to accommodate non-ASCII platforms */
        config->postfields[oldlen] = '\x26';
        memcpy(&config->postfields[oldlen + 1], postdata, size);
        config->postfields[oldlen + 1 + size] = '\0';
        Curl_safefree(oldpost);
        Curl_safefree(postdata);
        config->postfieldsize += size + 1;
      }
      else {
        config->postfields = postdata;
        config->postfieldsize = curlx_uztoso(size);
      }
    }
    /*
      We can't set the request type here, as this data might be used in
      a simple GET if -G is used. Already or soon.

      if(SetHTTPrequest(HTTPREQ_SIMPLEPOST, &config->httpreq)) {
        Curl_safefree(postdata);
        return PARAM_BAD_USE;
      }
    */
    break;
    case 'D':
      /* dump-header to given file name */
      GetStr(&config->headerfile, nextarg);
      break;
    case 'e':
    {
      char *ptr = strstr(nextarg, ";auto");
      if(ptr) {
        /* Automatic referer requested, this may be combined with a
           set initial one */
        config->autoreferer = TRUE;
        *ptr = 0; /* null-terminate here */
      }
      else
        config->autoreferer = FALSE;
      ptr = *nextarg ? nextarg : NULL;
      GetStr(&config->referer, ptr);
    }
    break;
    case 'E':
      switch(subletter) {
      case '\0': /* certificate file */
        GetFileAndPassword(nextarg, &config->cert, &config->key_passwd);
        break;
      case 'a': /* CA info PEM file */
        GetStr(&config->cacert, nextarg);
        break;
      case 'b': /* cert file type */
        GetStr(&config->cert_type, nextarg);
        break;
      case 'c': /* private key file */
        GetStr(&config->key, nextarg);
        break;
      case 'd': /* private key file type */
        GetStr(&config->key_type, nextarg);
        break;
      case 'e': /* private key passphrase */
        GetStr(&config->key_passwd, nextarg);
        cleanarg(nextarg);
        break;
      case 'f': /* crypto engine */
        GetStr(&config->engine, nextarg);
        if(config->engine && curl_strequal(config->engine, "list"))
          return PARAM_ENGINES_REQUESTED;
        break;
      case 'g': /* CA cert directory */
        GetStr(&config->capath, nextarg);
        break;
      case 'h': /* --pubkey public key file */
        GetStr(&config->pubkey, nextarg);
        break;
      case 'i': /* --hostpubmd5 md5 of the host public key */
        GetStr(&config->hostpubmd5, nextarg);
        if(!config->hostpubmd5 || strlen(config->hostpubmd5) != 32)
          return PARAM_BAD_USE;
        break;
      case 'F': /* --hostpubsha256 sha256 of the host public key */
        GetStr(&config->hostpubsha256, nextarg);
        break;
      case 'j': /* CRL file */
        GetStr(&config->crlfile, nextarg);
        break;
      case 'k': /* TLS username */
        if(curlinfo->features & CURL_VERSION_TLSAUTH_SRP)
          GetStr(&config->tls_username, nextarg);
        else
          return PARAM_LIBCURL_DOESNT_SUPPORT;
        break;
      case 'l': /* TLS password */
        if(curlinfo->features & CURL_VERSION_TLSAUTH_SRP)
          GetStr(&config->tls_password, nextarg);
        else
          return PARAM_LIBCURL_DOESNT_SUPPORT;
        break;
      case 'm': /* TLS authentication type */
        if(curlinfo->features & CURL_VERSION_TLSAUTH_SRP) {
          GetStr(&config->tls_authtype, nextarg);
          if(!curl_strequal(config->tls_authtype, "SRP"))
            return PARAM_LIBCURL_DOESNT_SUPPORT; /* only support TLS-SRP */
        }
        else
          return PARAM_LIBCURL_DOESNT_SUPPORT;
        break;
      case 'n': /* no empty SSL fragments, --ssl-allow-beast */
        if(curlinfo->features & CURL_VERSION_SSL)
          config->ssl_allow_beast = toggle;
        break;

      case 'o': /* --ssl-auto-client-cert */
        if(curlinfo->features & CURL_VERSION_SSL)
          config->ssl_auto_client_cert = toggle;
        break;

      case 'O': /* --proxy-ssl-auto-client-cert */
        if(curlinfo->features & CURL_VERSION_SSL)
          config->proxy_ssl_auto_client_cert = toggle;
        break;

      case 'p': /* Pinned public key DER file */
        GetStr(&config->pinnedpubkey, nextarg);
        break;

      case 'P': /* proxy pinned public key */
        GetStr(&config->proxy_pinnedpubkey, nextarg);
        break;

      case 'q': /* --cert-status */
        config->verifystatus = TRUE;
        break;

      case 'Q': /* --doh-cert-status */
        config->doh_verifystatus = TRUE;
        break;

      case 'r': /* --false-start */
        config->falsestart = TRUE;
        break;

      case 's': /* --ssl-no-revoke */
        if(curlinfo->features & CURL_VERSION_SSL)
          config->ssl_no_revoke = TRUE;
        break;

      case 'S': /* --ssl-revoke-best-effort */
        if(curlinfo->features & CURL_VERSION_SSL)
          config->ssl_revoke_best_effort = TRUE;
        break;

      case 't': /* --tcp-fastopen */
        config->tcp_fastopen = TRUE;
        break;

      case 'u': /* TLS username for proxy */
        if(curlinfo->features & CURL_VERSION_TLSAUTH_SRP)
          GetStr(&config->proxy_tls_username, nextarg);
        else
          return PARAM_LIBCURL_DOESNT_SUPPORT;
        break;

      case 'v': /* TLS password for proxy */
        if(curlinfo->features & CURL_VERSION_TLSAUTH_SRP)
          GetStr(&config->proxy_tls_password, nextarg);
        else
          return PARAM_LIBCURL_DOESNT_SUPPORT;
        break;

      case 'w': /* TLS authentication type for proxy */
        if(curlinfo->features & CURL_VERSION_TLSAUTH_SRP) {
          GetStr(&config->proxy_tls_authtype, nextarg);
          if(!curl_strequal(config->proxy_tls_authtype, "SRP"))
            return PARAM_LIBCURL_DOESNT_SUPPORT; /* only support TLS-SRP */
        }
        else
          return PARAM_LIBCURL_DOESNT_SUPPORT;
        break;

      case 'x': /* certificate file for proxy */
        GetFileAndPassword(nextarg, &config->proxy_cert,
                           &config->proxy_key_passwd);
        break;

      case 'y': /* cert file type for proxy */
        GetStr(&config->proxy_cert_type, nextarg);
        break;

      case 'z': /* private key file for proxy */
        GetStr(&config->proxy_key, nextarg);
        break;

      case '0': /* private key file type for proxy */
        GetStr(&config->proxy_key_type, nextarg);
        break;

      case '1': /* private key passphrase for proxy */
        GetStr(&config->proxy_key_passwd, nextarg);
        cleanarg(nextarg);
        break;

      case '2': /* ciphers for proxy */
        GetStr(&config->proxy_cipher_list, nextarg);
        break;

      case '3': /* CRL file for proxy */
        GetStr(&config->proxy_crlfile, nextarg);
        break;

      case '4': /* no empty SSL fragments for proxy */
        if(curlinfo->features & CURL_VERSION_SSL)
          config->proxy_ssl_allow_beast = toggle;
        break;

      case '5': /* --login-options */
        GetStr(&config->login_options, nextarg);
        break;

      case '6': /* CA info PEM file for proxy */
        GetStr(&config->proxy_cacert, nextarg);
        break;

      case '7': /* CA cert directory for proxy */
        GetStr(&config->proxy_capath, nextarg);
        break;

      case '8': /* allow insecure SSL connects for proxy */
        config->proxy_insecure_ok = toggle;
        break;

      case '9': /* --proxy-tlsv1 */
        /* TLS version 1 for proxy */
        config->proxy_ssl_version = CURL_SSLVERSION_TLSv1;
        break;

      case 'A':
        /* --socks5-basic */
        if(toggle)
          config->socks5_auth |= CURLAUTH_BASIC;
        else
          config->socks5_auth &= ~CURLAUTH_BASIC;
        break;

      case 'B':
        /* --socks5-gssapi */
        if(toggle)
          config->socks5_auth |= CURLAUTH_GSSAPI;
        else
          config->socks5_auth &= ~CURLAUTH_GSSAPI;
        break;

      case 'C':
        GetStr(&config->etag_save_file, nextarg);
        break;

      case 'D':
        GetStr(&config->etag_compare_file, nextarg);
        break;

      case 'E':
        GetStr(&config->ssl_ec_curves, nextarg);
        break;

      default: /* unknown flag */
        return PARAM_OPTION_UNKNOWN;
      }
      break;
    case 'f':
      switch(subletter) {
      case 'a': /* --fail-early */
        global->fail_early = toggle;
        break;
      case 'b': /* --styled-output */
        global->styled_output = toggle;
        break;
      case 'c': /* --mail-rcpt-allowfails */
        config->mail_rcpt_allowfails = toggle;
        break;
      case 'd': /* --fail-with-body */
        config->failwithbody = toggle;
        break;
      default: /* --fail (hard on errors)  */
        config->failonerror = toggle;
        break;
      }
      if(config->failonerror && config->failwithbody) {
        errorf(config->global, "You must select either --fail or "
               "--fail-with-body, not both.\n");
        return PARAM_BAD_USE;
      }
      break;
    case 'F':
      /* "form data" simulation, this is a little advanced so lets do our best
         to sort this out slowly and carefully */
      if(formparse(config,
                   nextarg,
                   &config->mimeroot,
                   &config->mimecurrent,
                   (subletter == 's')?TRUE:FALSE)) /* 's' is literal string */
        return PARAM_BAD_USE;
      if(SetHTTPrequest(config, HTTPREQ_MIMEPOST, &config->httpreq))
        return PARAM_BAD_USE;
      break;

    case 'g': /* g disables URLglobbing */
      config->globoff = toggle;
      break;

    case 'G': /* HTTP GET */
      if(subletter == 'a') { /* --request-target */
        GetStr(&config->request_target, nextarg);
      }
      else
        config->use_httpget = TRUE;
      break;

    case 'h': /* h for help */
      if(toggle) {
        if(nextarg) {
          global->help_category = strdup(nextarg);
          if(!global->help_category)
            return PARAM_NO_MEM;
        }
        return PARAM_HELP_REQUESTED;
      }
      /* we now actually support --no-help too! */
      break;
    case 'H':
      /* A custom header to append to a list */
      if(nextarg[0] == '@') {
        /* read many headers from a file or stdin */
        char *string;
        size_t len;
        bool use_stdin = !strcmp(&nextarg[1], "-");
        FILE *file = use_stdin?stdin:fopen(&nextarg[1], FOPEN_READTEXT);
        if(!file)
          warnf(global, "Failed to open %s!\n", &nextarg[1]);
        else {
          err = file2memory(&string, &len, file);
          if(!err && string) {
            /* Allow strtok() here since this isn't used threaded */
            /* !checksrc! disable BANNEDFUNC 2 */
            char *h = strtok(string, "\r\n");
            while(h) {
              if(subletter == 'p') /* --proxy-header */
                err = add2list(&config->proxyheaders, h);
              else
                err = add2list(&config->headers, h);
              if(err)
                break;
              h = strtok(NULL, "\r\n");
            }
            free(string);
          }
          if(!use_stdin)
            fclose(file);
          if(err)
            return err;
        }
      }
      else {
        if(subletter == 'p') /* --proxy-header */
          err = add2list(&config->proxyheaders, nextarg);
        else
          err = add2list(&config->headers, nextarg);
        if(err)
          return err;
      }
      break;
    case 'i':
      config->show_headers = toggle; /* show the headers as well in the
                                        general output stream */
      break;
    case 'j':
      config->cookiesession = toggle;
      break;
    case 'I': /* --head */
      config->no_body = toggle;
      config->show_headers = toggle;
      if(SetHTTPrequest(config,
                        (config->no_body)?HTTPREQ_HEAD:HTTPREQ_GET,
                        &config->httpreq))
        return PARAM_BAD_USE;
      break;
    case 'J': /* --remote-header-name */
      config->content_disposition = toggle;
      break;
    case 'k': /* allow insecure SSL connects */
      if(subletter == 'd') /* --doh-insecure */
        config->doh_insecure_ok = toggle;
      else
        config->insecure_ok = toggle;
      break;
    case 'K': /* parse config file */
      if(parseconfig(nextarg, global))
        warnf(global, "error trying read config from the '%s' file\n",
              nextarg);
      break;
    case 'l':
      config->dirlistonly = toggle; /* only list the names of the FTP dir */
      break;
    case 'L':
      config->followlocation = toggle; /* Follow Location: HTTP headers */
      switch(subletter) {
      case 't':
        /* Continue to send authentication (user+password) when following
         * locations, even when hostname changed */
        config->unrestricted_auth = toggle;
        break;
      }
      break;
    case 'm':
      /* specified max time */
      err = str2udouble(&config->timeout, nextarg, LONG_MAX/1000);
      if(err)
        return err;
      break;
    case 'M': /* M for manual, huge help */
      if(toggle) { /* --no-manual shows no manual... */
#ifdef USE_MANUAL
        return PARAM_MANUAL_REQUESTED;
#else
        warnf(global,
              "built-in manual was disabled at build-time!\n");
        return PARAM_OPTION_UNKNOWN;
#endif
      }
      break;
    case 'n':
      switch(subletter) {
      case 'o': /* use .netrc or URL */
        config->netrc_opt = toggle;
        break;
      case 'e': /* netrc-file */
        GetStr(&config->netrc_file, nextarg);
        break;
      default:
        /* pick info from .netrc, if this is used for http, curl will
           automatically enforce user+password with the request */
        config->netrc = toggle;
        break;
      }
      break;
    case 'N':
      /* disable the output I/O buffering. note that the option is called
         --buffer but is mostly used in the negative form: --no-buffer */
      if(longopt)
        config->nobuffer = (!toggle)?TRUE:FALSE;
      else
        config->nobuffer = toggle;
      break;
    case 'O': /* --remote-name */
      if(subletter == 'a') { /* --remote-name-all */
        config->default_node_flags = toggle?GETOUT_USEREMOTE:0;
        break;
      }
      else if(subletter == 'b') { /* --output-dir */
        GetStr(&config->output_dir, nextarg);
        break;
      }
      /* FALLTHROUGH */
    case 'o': /* --output */
      /* output file */
    {
      struct getout *url;
      if(!config->url_out)
        config->url_out = config->url_list;
      if(config->url_out) {
        /* there's a node here, if it already is filled-in continue to find
           an "empty" node */
        while(config->url_out && (config->url_out->flags & GETOUT_OUTFILE))
          config->url_out = config->url_out->next;
      }

      /* now there might or might not be an available node to fill in! */

      if(config->url_out)
        /* existing node */
        url = config->url_out;
      else
        /* there was no free node, create one! */
        config->url_out = url = new_getout(config);

      if(!url)
        return PARAM_NO_MEM;

      /* fill in the outfile */
      if('o' == letter) {
        GetStr(&url->outfile, nextarg);
        url->flags &= ~GETOUT_USEREMOTE; /* switch off */
      }
      else {
        url->outfile = NULL; /* leave it */
        if(toggle)
          url->flags |= GETOUT_USEREMOTE;  /* switch on */
        else
          url->flags &= ~GETOUT_USEREMOTE; /* switch off */
      }
      url->flags |= GETOUT_OUTFILE;
    }
    break;
    case 'P':
      /* This makes the FTP sessions use PORT instead of PASV */
      /* use <eth0> or <192.168.10.10> style addresses. Anything except
         this will make us try to get the "default" address.
         NOTE: this is a changed behavior since the released 4.1!
      */
      GetStr(&config->ftpport, nextarg);
      break;
    case 'p':
      /* proxy tunnel for non-http protocols */
      config->proxytunnel = toggle;
      break;

    case 'q': /* if used first, already taken care of, we do it like
                 this so we don't cause an error! */
      break;
    case 'Q':
      /* QUOTE command to send to FTP server */
      switch(nextarg[0]) {
      case '-':
        /* prefixed with a dash makes it a POST TRANSFER one */
        nextarg++;
        err = add2list(&config->postquote, nextarg);
        break;
      case '+':
        /* prefixed with a plus makes it a just-before-transfer one */
        nextarg++;
        err = add2list(&config->prequote, nextarg);
        break;
      default:
        err = add2list(&config->quote, nextarg);
        break;
      }
      if(err)
        return err;
      break;
    case 'r':
      /* Specifying a range WITHOUT A DASH will create an illegal HTTP range
         (and won't actually be range by definition). The man page previously
         claimed that to be a good way, why this code is added to work-around
         it. */
      if(ISDIGIT(*nextarg) && !strchr(nextarg, '-')) {
        char buffer[32];
        curl_off_t off;
        if(curlx_strtoofft(nextarg, NULL, 10, &off)) {
          warnf(global, "unsupported range point\n");
          return PARAM_BAD_USE;
        }
        warnf(global,
              "A specified range MUST include at least one dash (-). "
              "Appending one for you!\n");
        msnprintf(buffer, sizeof(buffer), "%" CURL_FORMAT_CURL_OFF_T "-", off);
        Curl_safefree(config->range);
        config->range = strdup(buffer);
        if(!config->range)
          return PARAM_NO_MEM;
      }
      else {
        /* byte range requested */
        const char *tmp_range = nextarg;
        while(*tmp_range != '\0') {
          if(!ISDIGIT(*tmp_range) && *tmp_range != '-' && *tmp_range != ',') {
            warnf(global, "Invalid character is found in given range. "
                  "A specified range MUST have only digits in "
                  "\'start\'-\'stop\'. The server's response to this "
                  "request is uncertain.\n");
            break;
          }
          tmp_range++;
        }
        GetStr(&config->range, nextarg);
      }
      break;
    case 'R':
      /* use remote file's time */
      config->remote_time = toggle;
      break;
    case 's':
      /* don't show progress meter, don't show errors : */
      if(toggle)
        global->mute = global->noprogress = TRUE;
      else
        global->mute = global->noprogress = FALSE;
      if(global->showerror < 0)
        /* if still on the default value, set showerror to the reverse of
           toggle. This is to allow -S and -s to be used in an independent
           order but still have the same effect. */
        global->showerror = (!toggle)?TRUE:FALSE; /* toggle off */
      break;
    case 'S':
      /* show errors */
      global->showerror = toggle?1:0; /* toggle on if used with -s */
      break;
    case 't':
      /* Telnet options */
      err = add2list(&config->telnet_options, nextarg);
      if(err)
        return err;
      break;
    case 'T':
      /* we are uploading */
    {
      struct getout *url;
      if(!config->url_ul)
        config->url_ul = config->url_list;
      if(config->url_ul) {
        /* there's a node here, if it already is filled-in continue to find
           an "empty" node */
        while(config->url_ul && (config->url_ul->flags & GETOUT_UPLOAD))
          config->url_ul = config->url_ul->next;
      }

      /* now there might or might not be an available node to fill in! */

      if(config->url_ul)
        /* existing node */
        url = config->url_ul;
      else
        /* there was no free node, create one! */
        config->url_ul = url = new_getout(config);

      if(!url)
        return PARAM_NO_MEM;

      url->flags |= GETOUT_UPLOAD; /* mark -T used */
      if(!*nextarg)
        url->flags |= GETOUT_NOUPLOAD;
      else {
        /* "-" equals stdin, but keep the string around for now */
        GetStr(&url->infile, nextarg);
      }
    }
    break;
    case 'u':
      /* user:password  */
      GetStr(&config->userpwd, nextarg);
      cleanarg(nextarg);
      break;
    case 'U':
      /* Proxy user:password  */
      GetStr(&config->proxyuserpwd, nextarg);
      cleanarg(nextarg);
      break;
    case 'v':
      if(toggle) {
        /* the '%' thing here will cause the trace get sent to stderr */
        Curl_safefree(global->trace_dump);
        global->trace_dump = strdup("%");
        if(!global->trace_dump)
          return PARAM_NO_MEM;
        if(global->tracetype && (global->tracetype != TRACE_PLAIN))
          warnf(global,
                "-v, --verbose overrides an earlier trace/verbose option\n");
        global->tracetype = TRACE_PLAIN;
      }
      else
        /* verbose is disabled here */
        global->tracetype = TRACE_NONE;
      break;
    case 'V':
      if(toggle)    /* --no-version yields no output! */
        return PARAM_VERSION_INFO_REQUESTED;
      break;

    case 'w':
      /* get the output string */
      if('@' == *nextarg) {
        /* the data begins with a '@' letter, it means that a file name
           or - (stdin) follows */
        FILE *file;
        const char *fname;
        nextarg++; /* pass the @ */
        if(!strcmp("-", nextarg)) {
          fname = "<stdin>";
          file = stdin;
        }
        else {
          fname = nextarg;
          file = fopen(nextarg, FOPEN_READTEXT);
        }
        Curl_safefree(config->writeout);
        err = file2string(&config->writeout, file);
        if(file && (file != stdin))
          fclose(file);
        if(err)
          return err;
        if(!config->writeout)
          warnf(global, "Failed to read %s", fname);
      }
      else
        GetStr(&config->writeout, nextarg);
      break;
    case 'x':
      switch(subletter) {
      case 'a': /* --preproxy */
        GetStr(&config->preproxy, nextarg);
        break;
      default:
        /* --proxy */
        GetStr(&config->proxy, nextarg);
        config->proxyver = CURLPROXY_HTTP;
        break;
      }
      break;
    case 'X':
      /* set custom request */
      GetStr(&config->customrequest, nextarg);
      break;
    case 'y':
      /* low speed time */
      err = str2unum(&config->low_speed_time, nextarg);
      if(err)
        return err;
      if(!config->low_speed_limit)
        config->low_speed_limit = 1;
      break;
    case 'Y':
      /* low speed limit */
      err = str2unum(&config->low_speed_limit, nextarg);
      if(err)
        return err;
      if(!config->low_speed_time)
        config->low_speed_time = 30;
      break;
    case 'Z':
      switch(subletter) {
      case '\0':  /* --parallel */
        global->parallel = toggle;
        break;
      case 'b':   /* --parallel-max */
        err = str2unum(&global->parallel_max, nextarg);
        if(err)
          return err;
        if((global->parallel_max > MAX_PARALLEL) ||
           (global->parallel_max < 1))
          global->parallel_max = PARALLEL_DEFAULT;
        break;
      case 'c':   /* --parallel-connect */
        global->parallel_connect = toggle;
        break;
      }
      break;
    case 'z': /* time condition coming up */
      switch(*nextarg) {
      case '+':
        nextarg++;
        /* FALLTHROUGH */
      default:
        /* If-Modified-Since: (section 14.28 in RFC2068) */
        config->timecond = CURL_TIMECOND_IFMODSINCE;
        break;
      case '-':
        /* If-Unmodified-Since:  (section 14.24 in RFC2068) */
        config->timecond = CURL_TIMECOND_IFUNMODSINCE;
        nextarg++;
        break;
      case '=':
        /* Last-Modified:  (section 14.29 in RFC2068) */
        config->timecond = CURL_TIMECOND_LASTMOD;
        nextarg++;
        break;
      }
      now = time(NULL);
      config->condtime = (curl_off_t)curl_getdate(nextarg, &now);
      if(-1 == config->condtime) {
        /* now let's see if it is a file name to get the time from instead! */
        curl_off_t filetime = getfiletime(nextarg, global);
        if(filetime >= 0) {
          /* pull the time out from the file */
          config->condtime = filetime;
        }
        else {
          /* failed, remove time condition */
          config->timecond = CURL_TIMECOND_NONE;
          warnf(global,
                "Illegal date format for -z, --time-cond (and not "
                "a file name). Disabling time condition. "
                "See curl_getdate(3) for valid date syntax.\n");
        }
      }
      break;
    default: /* unknown flag */
      return PARAM_OPTION_UNKNOWN;
    }
    hit = -1;

  } while(!longopt && !singleopt && *++parse && !*usedarg);

  return PARAM_OK;
}

ParameterError parse_args(struct GlobalConfig *global, int argc,
                          argv_item_t argv[])
{
  int i;
  bool stillflags;
  char *orig_opt = NULL;
  ParameterError result = PARAM_OK;
  struct OperationConfig *config = global->first;

  for(i = 1, stillflags = TRUE; i < argc && !result; i++) {
    orig_opt = curlx_convert_tchar_to_UTF8(argv[i]);
    if(!orig_opt)
      return PARAM_NO_MEM;

    if(stillflags && ('-' == orig_opt[0])) {
      bool passarg;

      if(!strcmp("--", orig_opt))
        /* This indicates the end of the flags and thus enables the
           following (URL) argument to start with -. */
        stillflags = FALSE;
      else {
        char *nextarg = (i < (argc - 1))
          ? curlx_convert_tchar_to_UTF8(argv[i + 1])
          : NULL;

        result = getparameter(orig_opt, nextarg, &passarg, global, config);
        curlx_unicodefree(nextarg);
        config = global->last;
        if(result == PARAM_NEXT_OPERATION) {
          /* Reset result as PARAM_NEXT_OPERATION is only used here and not
             returned from this function */
          result = PARAM_OK;

          if(config->url_list && config->url_list->url) {
            /* Allocate the next config */
            config->next = malloc(sizeof(struct OperationConfig));
            if(config->next) {
              /* Initialise the newly created config */
              config_init(config->next);

              /* Set the global config pointer */
              config->next->global = global;

              /* Update the last config pointer */
              global->last = config->next;

              /* Move onto the new config */
              config->next->prev = config;
              config = config->next;
            }
            else
              result = PARAM_NO_MEM;
          }
        }
        else if(!result && passarg)
          i++; /* we're supposed to skip this */
      }
    }
    else {
      bool used;

      /* Just add the URL please */
      result = getparameter("--url", orig_opt, &used, global,
                            config);
    }

    if(!result)
      curlx_unicodefree(orig_opt);
  }

  if(!result && config->content_disposition) {
    if(config->show_headers)
      result = PARAM_CONTDISP_SHOW_HEADER;
    else if(config->resume_from_current)
      result = PARAM_CONTDISP_RESUME_FROM;
  }

  if(result && result != PARAM_HELP_REQUESTED &&
     result != PARAM_MANUAL_REQUESTED &&
     result != PARAM_VERSION_INFO_REQUESTED &&
     result != PARAM_ENGINES_REQUESTED) {
    const char *reason = param2text(result);

    if(orig_opt && strcmp(":", orig_opt))
      helpf(global->errors, "option %s: %s\n", orig_opt, reason);
    else
      helpf(global->errors, "%s\n", reason);
  }

  curlx_unicodefree(orig_opt);
  return result;
}
