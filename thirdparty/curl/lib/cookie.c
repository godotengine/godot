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

/***


RECEIVING COOKIE INFORMATION
============================

struct CookieInfo *Curl_cookie_init(struct Curl_easy *data,
                    const char *file, struct CookieInfo *inc, bool newsession);

        Inits a cookie struct to store data in a local file. This is always
        called before any cookies are set.

struct Cookie *Curl_cookie_add(struct Curl_easy *data,
                 struct CookieInfo *c, bool httpheader, char *lineptr,
                 const char *domain, const char *path);

        The 'lineptr' parameter is a full "Set-cookie:" line as
        received from a server.

        The function need to replace previously stored lines that this new
        line supersedes.

        It may remove lines that are expired.

        It should return an indication of success/error.


SENDING COOKIE INFORMATION
==========================

struct Cookies *Curl_cookie_getlist(struct CookieInfo *cookie,
                                    char *host, char *path, bool secure);

        For a given host and path, return a linked list of cookies that
        the client should send to the server if used now. The secure
        boolean informs the cookie if a secure connection is achieved or
        not.

        It shall only return cookies that haven't expired.


Example set of cookies:

    Set-cookie: PRODUCTINFO=webxpress; domain=.fidelity.com; path=/; secure
    Set-cookie: PERSONALIZE=none;expires=Monday, 13-Jun-1988 03:04:55 GMT;
    domain=.fidelity.com; path=/ftgw; secure
    Set-cookie: FidHist=none;expires=Monday, 13-Jun-1988 03:04:55 GMT;
    domain=.fidelity.com; path=/; secure
    Set-cookie: FidOrder=none;expires=Monday, 13-Jun-1988 03:04:55 GMT;
    domain=.fidelity.com; path=/; secure
    Set-cookie: DisPend=none;expires=Monday, 13-Jun-1988 03:04:55 GMT;
    domain=.fidelity.com; path=/; secure
    Set-cookie: FidDis=none;expires=Monday, 13-Jun-1988 03:04:55 GMT;
    domain=.fidelity.com; path=/; secure
    Set-cookie:
    Session_Key@6791a9e0-901a-11d0-a1c8-9b012c88aa77=none;expires=Monday,
    13-Jun-1988 03:04:55 GMT; domain=.fidelity.com; path=/; secure
****/


#include "curl_setup.h"

#if !defined(CURL_DISABLE_HTTP) && !defined(CURL_DISABLE_COOKIES)

#include "urldata.h"
#include "cookie.h"
#include "psl.h"
#include "strtok.h"
#include "sendf.h"
#include "slist.h"
#include "share.h"
#include "strtoofft.h"
#include "strcase.h"
#include "curl_get_line.h"
#include "curl_memrchr.h"
#include "parsedate.h"
#include "rand.h"
#include "rename.h"

/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"

static void strstore(char **str, const char *newstr);

static void freecookie(struct Cookie *co)
{
  free(co->expirestr);
  free(co->domain);
  free(co->path);
  free(co->spath);
  free(co->name);
  free(co->value);
  free(co->maxage);
  free(co->version);
  free(co);
}

static bool tailmatch(const char *cooke_domain, const char *hostname)
{
  size_t cookie_domain_len = strlen(cooke_domain);
  size_t hostname_len = strlen(hostname);

  if(hostname_len < cookie_domain_len)
    return FALSE;

  if(!strcasecompare(cooke_domain, hostname + hostname_len-cookie_domain_len))
    return FALSE;

  /*
   * A lead char of cookie_domain is not '.'.
   * RFC6265 4.1.2.3. The Domain Attribute says:
   * For example, if the value of the Domain attribute is
   * "example.com", the user agent will include the cookie in the Cookie
   * header when making HTTP requests to example.com, www.example.com, and
   * www.corp.example.com.
   */
  if(hostname_len == cookie_domain_len)
    return TRUE;
  if('.' == *(hostname + hostname_len - cookie_domain_len - 1))
    return TRUE;
  return FALSE;
}

/*
 * matching cookie path and url path
 * RFC6265 5.1.4 Paths and Path-Match
 */
static bool pathmatch(const char *cookie_path, const char *request_uri)
{
  size_t cookie_path_len;
  size_t uri_path_len;
  char *uri_path = NULL;
  char *pos;
  bool ret = FALSE;

  /* cookie_path must not have last '/' separator. ex: /sample */
  cookie_path_len = strlen(cookie_path);
  if(1 == cookie_path_len) {
    /* cookie_path must be '/' */
    return TRUE;
  }

  uri_path = strdup(request_uri);
  if(!uri_path)
    return FALSE;
  pos = strchr(uri_path, '?');
  if(pos)
    *pos = 0x0;

  /* #-fragments are already cut off! */
  if(0 == strlen(uri_path) || uri_path[0] != '/') {
    strstore(&uri_path, "/");
    if(!uri_path)
      return FALSE;
  }

  /*
   * here, RFC6265 5.1.4 says
   *  4. Output the characters of the uri-path from the first character up
   *     to, but not including, the right-most %x2F ("/").
   *  but URL path /hoge?fuga=xxx means /hoge/index.cgi?fuga=xxx in some site
   *  without redirect.
   *  Ignore this algorithm because /hoge is uri path for this case
   *  (uri path is not /).
   */

  uri_path_len = strlen(uri_path);

  if(uri_path_len < cookie_path_len) {
    ret = FALSE;
    goto pathmatched;
  }

  /* not using checkprefix() because matching should be case-sensitive */
  if(strncmp(cookie_path, uri_path, cookie_path_len)) {
    ret = FALSE;
    goto pathmatched;
  }

  /* The cookie-path and the uri-path are identical. */
  if(cookie_path_len == uri_path_len) {
    ret = TRUE;
    goto pathmatched;
  }

  /* here, cookie_path_len < uri_path_len */
  if(uri_path[cookie_path_len] == '/') {
    ret = TRUE;
    goto pathmatched;
  }

  ret = FALSE;

pathmatched:
  free(uri_path);
  return ret;
}

/*
 * Return the top-level domain, for optimal hashing.
 */
static const char *get_top_domain(const char * const domain, size_t *outlen)
{
  size_t len = 0;
  const char *first = NULL, *last;

  if(domain) {
    len = strlen(domain);
    last = memrchr(domain, '.', len);
    if(last) {
      first = memrchr(domain, '.', (last - domain));
      if(first)
        len -= (++first - domain);
    }
  }

  if(outlen)
    *outlen = len;

  return first? first: domain;
}

/* Avoid C1001, an "internal error" with MSVC14 */
#if defined(_MSC_VER) && (_MSC_VER == 1900)
#pragma optimize("", off)
#endif

/*
 * A case-insensitive hash for the cookie domains.
 */
static size_t cookie_hash_domain(const char *domain, const size_t len)
{
  const char *end = domain + len;
  size_t h = 5381;

  while(domain < end) {
    h += h << 5;
    h ^= Curl_raw_toupper(*domain++);
  }

  return (h % COOKIE_HASH_SIZE);
}

#if defined(_MSC_VER) && (_MSC_VER == 1900)
#pragma optimize("", on)
#endif

/*
 * Hash this domain.
 */
static size_t cookiehash(const char * const domain)
{
  const char *top;
  size_t len;

  if(!domain || Curl_host_is_ipnum(domain))
    return 0;

  top = get_top_domain(domain, &len);
  return cookie_hash_domain(top, len);
}

/*
 * cookie path sanitize
 */
static char *sanitize_cookie_path(const char *cookie_path)
{
  size_t len;
  char *new_path = strdup(cookie_path);
  if(!new_path)
    return NULL;

  /* some stupid site sends path attribute with '"'. */
  len = strlen(new_path);
  if(new_path[0] == '\"') {
    memmove((void *)new_path, (const void *)(new_path + 1), len);
    len--;
  }
  if(len && (new_path[len - 1] == '\"')) {
    new_path[len - 1] = 0x0;
    len--;
  }

  /* RFC6265 5.2.4 The Path Attribute */
  if(new_path[0] != '/') {
    /* Let cookie-path be the default-path. */
    strstore(&new_path, "/");
    return new_path;
  }

  /* convert /hoge/ to /hoge */
  if(len && new_path[len - 1] == '/') {
    new_path[len - 1] = 0x0;
  }

  return new_path;
}

/*
 * Load cookies from all given cookie files (CURLOPT_COOKIEFILE).
 *
 * NOTE: OOM or cookie parsing failures are ignored.
 */
void Curl_cookie_loadfiles(struct Curl_easy *data)
{
  struct curl_slist *list = data->state.cookielist;
  if(list) {
    Curl_share_lock(data, CURL_LOCK_DATA_COOKIE, CURL_LOCK_ACCESS_SINGLE);
    while(list) {
      struct CookieInfo *newcookies = Curl_cookie_init(data,
                                        list->data,
                                        data->cookies,
                                        data->set.cookiesession);
      if(!newcookies)
        /*
         * Failure may be due to OOM or a bad cookie; both are ignored
         * but only the first should be
         */
        infof(data, "ignoring failed cookie_init for %s", list->data);
      else
        data->cookies = newcookies;
      list = list->next;
    }
    curl_slist_free_all(data->state.cookielist); /* clean up list */
    data->state.cookielist = NULL; /* don't do this again! */
    Curl_share_unlock(data, CURL_LOCK_DATA_COOKIE);
  }
}

/*
 * strstore
 *
 * A thin wrapper around strdup which ensures that any memory allocated at
 * *str will be freed before the string allocated by strdup is stored there.
 * The intended usecase is repeated assignments to the same variable during
 * parsing in a last-wins scenario. The caller is responsible for checking
 * for OOM errors.
 */
static void strstore(char **str, const char *newstr)
{
  free(*str);
  *str = strdup(newstr);
}

/*
 * remove_expired
 *
 * Remove expired cookies from the hash by inspecting the expires timestamp on
 * each cookie in the hash, freeing and deleting any where the timestamp is in
 * the past.  If the cookiejar has recorded the next timestamp at which one or
 * more cookies expire, then processing will exit early in case this timestamp
 * is in the future.
 */
static void remove_expired(struct CookieInfo *cookies)
{
  struct Cookie *co, *nx;
  curl_off_t now = (curl_off_t)time(NULL);
  unsigned int i;

  /*
   * If the earliest expiration timestamp in the jar is in the future we can
   * skip scanning the whole jar and instead exit early as there won't be any
   * cookies to evict.  If we need to evict however, reset the next_expiration
   * counter in order to track the next one. In case the recorded first
   * expiration is the max offset, then perform the safe fallback of checking
   * all cookies.
   */
  if(now < cookies->next_expiration &&
      cookies->next_expiration != CURL_OFF_T_MAX)
    return;
  else
    cookies->next_expiration = CURL_OFF_T_MAX;

  for(i = 0; i < COOKIE_HASH_SIZE; i++) {
    struct Cookie *pv = NULL;
    co = cookies->cookies[i];
    while(co) {
      nx = co->next;
      if(co->expires && co->expires < now) {
        if(!pv) {
          cookies->cookies[i] = co->next;
        }
        else {
          pv->next = co->next;
        }
        cookies->numcookies--;
        freecookie(co);
      }
      else {
        /*
         * If this cookie has an expiration timestamp earlier than what we've
         * seen so far then record it for the next round of expirations.
         */
        if(co->expires && co->expires < cookies->next_expiration)
          cookies->next_expiration = co->expires;
        pv = co;
      }
      co = nx;
    }
  }
}

/* Make sure domain contains a dot or is localhost. */
static bool bad_domain(const char *domain)
{
  return !strchr(domain, '.') && !strcasecompare(domain, "localhost");
}

/*
 * Curl_cookie_add
 *
 * Add a single cookie line to the cookie keeping object. Be aware that
 * sometimes we get an IP-only host name, and that might also be a numerical
 * IPv6 address.
 *
 * Returns NULL on out of memory or invalid cookie. This is suboptimal,
 * as they should be treated separately.
 */
struct Cookie *
Curl_cookie_add(struct Curl_easy *data,
                /*
                 * The 'data' pointer here may be NULL at times, and thus
                 * must only be used very carefully for things that can deal
                 * with data being NULL. Such as infof() and similar
                 */
                struct CookieInfo *c,
                bool httpheader, /* TRUE if HTTP header-style line */
                bool noexpire, /* if TRUE, skip remove_expired() */
                char *lineptr,   /* first character of the line */
                const char *domain, /* default domain */
                const char *path,   /* full path used when this cookie is set,
                                       used to get default path for the cookie
                                       unless set */
                bool secure)  /* TRUE if connection is over secure origin */
{
  struct Cookie *clist;
  struct Cookie *co;
  struct Cookie *lastc = NULL;
  time_t now = time(NULL);
  bool replace_old = FALSE;
  bool badcookie = FALSE; /* cookies are good by default. mmmmm yummy */
  size_t myhash;

#ifdef CURL_DISABLE_VERBOSE_STRINGS
  (void)data;
#endif

  /* First, alloc and init a new struct for it */
  co = calloc(1, sizeof(struct Cookie));
  if(!co)
    return NULL; /* bail out if we're this low on memory */

  if(httpheader) {
    /* This line was read off a HTTP-header */
    char name[MAX_NAME];
    char what[MAX_NAME];
    const char *ptr;
    const char *semiptr;

    size_t linelength = strlen(lineptr);
    if(linelength > MAX_COOKIE_LINE) {
      /* discard overly long lines at once */
      free(co);
      return NULL;
    }

    semiptr = strchr(lineptr, ';'); /* first, find a semicolon */

    while(*lineptr && ISBLANK(*lineptr))
      lineptr++;

    ptr = lineptr;
    do {
      /* we have a <what>=<this> pair or a stand-alone word here */
      name[0] = what[0] = 0; /* init the buffers */
      if(1 <= sscanf(ptr, "%" MAX_NAME_TXT "[^;\r\n=] =%"
                     MAX_NAME_TXT "[^;\r\n]",
                     name, what)) {
        /*
         * Use strstore() below to properly deal with received cookie
         * headers that have the same string property set more than once,
         * and then we use the last one.
         */
        const char *whatptr;
        bool done = FALSE;
        bool sep;
        size_t len = strlen(what);
        size_t nlen = strlen(name);
        const char *endofn = &ptr[ nlen ];

        /*
         * Check for too long individual name or contents, or too long
         * combination of name + contents. Chrome and Firefox support 4095 or
         * 4096 bytes combo
         */
        if(nlen >= (MAX_NAME-1) || len >= (MAX_NAME-1) ||
           ((nlen + len) > MAX_NAME)) {
          freecookie(co);
          infof(data, "oversized cookie dropped, name/val %zu + %zu bytes",
                nlen, len);
          return NULL;
        }

        /* name ends with a '=' ? */
        sep = (*endofn == '=')?TRUE:FALSE;

        if(nlen) {
          endofn--; /* move to the last character */
          if(ISBLANK(*endofn)) {
            /* skip trailing spaces in name */
            while(*endofn && ISBLANK(*endofn) && nlen) {
              endofn--;
              nlen--;
            }
            name[nlen] = 0; /* new end of name */
          }
        }

        /* Strip off trailing whitespace from the 'what' */
        while(len && ISBLANK(what[len-1])) {
          what[len-1] = 0;
          len--;
        }

        /* Skip leading whitespace from the 'what' */
        whatptr = what;
        while(*whatptr && ISBLANK(*whatptr))
          whatptr++;

        /*
         * Check if we have a reserved prefix set before anything else, as we
         * otherwise have to test for the prefix in both the cookie name and
         * "the rest". Prefixes must start with '__' and end with a '-', so
         * only test for names where that can possibly be true.
         */
        if(nlen > 3 && name[0] == '_' && name[1] == '_') {
          if(!strncmp("__Secure-", name, 9))
            co->prefix |= COOKIE_PREFIX__SECURE;
          else if(!strncmp("__Host-", name, 7))
            co->prefix |= COOKIE_PREFIX__HOST;
        }

        if(!co->name) {
          /* The very first name/value pair is the actual cookie name */
          if(!sep) {
            /* Bad name/value pair. */
            badcookie = TRUE;
            break;
          }
          co->name = strdup(name);
          co->value = strdup(whatptr);
          done = TRUE;
          if(!co->name || !co->value) {
            badcookie = TRUE;
            break;
          }
        }
        else if(!len) {
          /*
           * this was a "<name>=" with no content, and we must allow
           * 'secure' and 'httponly' specified this weirdly
           */
          done = TRUE;
          /*
           * secure cookies are only allowed to be set when the connection is
           * using a secure protocol, or when the cookie is being set by
           * reading from file
           */
          if(strcasecompare("secure", name)) {
            if(secure || !c->running) {
              co->secure = TRUE;
            }
            else {
              badcookie = TRUE;
              break;
            }
          }
          else if(strcasecompare("httponly", name))
            co->httponly = TRUE;
          else if(sep)
            /* there was a '=' so we're not done parsing this field */
            done = FALSE;
        }
        if(done)
          ;
        else if(strcasecompare("path", name)) {
          strstore(&co->path, whatptr);
          if(!co->path) {
            badcookie = TRUE; /* out of memory bad */
            break;
          }
          free(co->spath); /* if this is set again */
          co->spath = sanitize_cookie_path(co->path);
          if(!co->spath) {
            badcookie = TRUE; /* out of memory bad */
            break;
          }
        }
        else if(strcasecompare("domain", name)) {
          bool is_ip;

          /*
           * Now, we make sure that our host is within the given domain, or
           * the given domain is not valid and thus cannot be set.
           */

          if('.' == whatptr[0])
            whatptr++; /* ignore preceding dot */

#ifndef USE_LIBPSL
          /*
           * Without PSL we don't know when the incoming cookie is set on a
           * TLD or otherwise "protected" suffix. To reduce risk, we require a
           * dot OR the exact host name being "localhost".
           */
          if(bad_domain(whatptr))
            domain = ":";
#endif

          is_ip = Curl_host_is_ipnum(domain ? domain : whatptr);

          if(!domain
             || (is_ip && !strcmp(whatptr, domain))
             || (!is_ip && tailmatch(whatptr, domain))) {
            strstore(&co->domain, whatptr);
            if(!co->domain) {
              badcookie = TRUE;
              break;
            }
            if(!is_ip)
              co->tailmatch = TRUE; /* we always do that if the domain name was
                                       given */
          }
          else {
            /*
             * We did not get a tailmatch and then the attempted set domain is
             * not a domain to which the current host belongs. Mark as bad.
             */
            badcookie = TRUE;
            infof(data, "skipped cookie with bad tailmatch domain: %s",
                  whatptr);
          }
        }
        else if(strcasecompare("version", name)) {
          strstore(&co->version, whatptr);
          if(!co->version) {
            badcookie = TRUE;
            break;
          }
        }
        else if(strcasecompare("max-age", name)) {
          /*
           * Defined in RFC2109:
           *
           * Optional.  The Max-Age attribute defines the lifetime of the
           * cookie, in seconds.  The delta-seconds value is a decimal non-
           * negative integer.  After delta-seconds seconds elapse, the
           * client should discard the cookie.  A value of zero means the
           * cookie should be discarded immediately.
           */
          strstore(&co->maxage, whatptr);
          if(!co->maxage) {
            badcookie = TRUE;
            break;
          }
        }
        else if(strcasecompare("expires", name)) {
          strstore(&co->expirestr, whatptr);
          if(!co->expirestr) {
            badcookie = TRUE;
            break;
          }
        }

        /*
         * Else, this is the second (or more) name we don't know about!
         */
      }
      else {
        /* this is an "illegal" <what>=<this> pair */
      }

      if(!semiptr || !*semiptr) {
        /* we already know there are no more cookies */
        semiptr = NULL;
        continue;
      }

      ptr = semiptr + 1;
      while(*ptr && ISBLANK(*ptr))
        ptr++;
      semiptr = strchr(ptr, ';'); /* now, find the next semicolon */

      if(!semiptr && *ptr)
        /*
         * There are no more semicolons, but there's a final name=value pair
         * coming up
         */
        semiptr = strchr(ptr, '\0');
    } while(semiptr);

    if(co->maxage) {
      CURLofft offt;
      offt = curlx_strtoofft((*co->maxage == '\"')?
                             &co->maxage[1]:&co->maxage[0], NULL, 10,
                             &co->expires);
      if(offt == CURL_OFFT_FLOW)
        /* overflow, used max value */
        co->expires = CURL_OFF_T_MAX;
      else if(!offt) {
        if(!co->expires)
          /* already expired */
          co->expires = 1;
        else if(CURL_OFF_T_MAX - now < co->expires)
          /* would overflow */
          co->expires = CURL_OFF_T_MAX;
        else
          co->expires += now;
      }
    }
    else if(co->expirestr) {
      /*
       * Note that if the date couldn't get parsed for whatever reason, the
       * cookie will be treated as a session cookie
       */
      co->expires = Curl_getdate_capped(co->expirestr);

      /*
       * Session cookies have expires set to 0 so if we get that back from the
       * date parser let's add a second to make it a non-session cookie
       */
      if(co->expires == 0)
        co->expires = 1;
      else if(co->expires < 0)
        co->expires = 0;
    }

    if(!badcookie && !co->domain) {
      if(domain) {
        /* no domain was given in the header line, set the default */
        co->domain = strdup(domain);
        if(!co->domain)
          badcookie = TRUE;
      }
    }

    if(!badcookie && !co->path && path) {
      /*
       * No path was given in the header line, set the default.  Note that the
       * passed-in path to this function MAY have a '?' and following part that
       * MUST NOT be stored as part of the path.
       */
      char *queryp = strchr(path, '?');

      /*
       * queryp is where the interesting part of the path ends, so now we
       * want to the find the last
       */
      char *endslash;
      if(!queryp)
        endslash = strrchr(path, '/');
      else
        endslash = memrchr(path, '/', (queryp - path));
      if(endslash) {
        size_t pathlen = (endslash-path + 1); /* include end slash */
        co->path = malloc(pathlen + 1); /* one extra for the zero byte */
        if(co->path) {
          memcpy(co->path, path, pathlen);
          co->path[pathlen] = 0; /* null-terminate */
          co->spath = sanitize_cookie_path(co->path);
          if(!co->spath)
            badcookie = TRUE; /* out of memory bad */
        }
        else
          badcookie = TRUE;
      }
    }

    /*
     * If we didn't get a cookie name, or a bad one, the this is an illegal
     * line so bail out.
     */
    if(badcookie || !co->name) {
      freecookie(co);
      return NULL;
    }

  }
  else {
    /*
     * This line is NOT a HTTP header style line, we do offer support for
     * reading the odd netscape cookies-file format here
     */
    char *ptr;
    char *firstptr;
    char *tok_buf = NULL;
    int fields;

    /*
     * IE introduced HTTP-only cookies to prevent XSS attacks. Cookies marked
     * with httpOnly after the domain name are not accessible from javascripts,
     * but since curl does not operate at javascript level, we include them
     * anyway. In Firefox's cookie files, these lines are preceded with
     * #HttpOnly_ and then everything is as usual, so we skip 10 characters of
     * the line..
     */
    if(strncmp(lineptr, "#HttpOnly_", 10) == 0) {
      lineptr += 10;
      co->httponly = TRUE;
    }

    if(lineptr[0]=='#') {
      /* don't even try the comments */
      free(co);
      return NULL;
    }
    /* strip off the possible end-of-line characters */
    ptr = strchr(lineptr, '\r');
    if(ptr)
      *ptr = 0; /* clear it */
    ptr = strchr(lineptr, '\n');
    if(ptr)
      *ptr = 0; /* clear it */

    firstptr = strtok_r(lineptr, "\t", &tok_buf); /* tokenize it on the TAB */

    /*
     * Now loop through the fields and init the struct we already have
     * allocated
     */
    for(ptr = firstptr, fields = 0; ptr && !badcookie;
        ptr = strtok_r(NULL, "\t", &tok_buf), fields++) {
      switch(fields) {
      case 0:
        if(ptr[0]=='.') /* skip preceding dots */
          ptr++;
        co->domain = strdup(ptr);
        if(!co->domain)
          badcookie = TRUE;
        break;
      case 1:
        /*
         * flag: A TRUE/FALSE value indicating if all machines within a given
         * domain can access the variable. Set TRUE when the cookie says
         * .domain.com and to false when the domain is complete www.domain.com
         */
        co->tailmatch = strcasecompare(ptr, "TRUE")?TRUE:FALSE;
        break;
      case 2:
        /* The file format allows the path field to remain not filled in */
        if(strcmp("TRUE", ptr) && strcmp("FALSE", ptr)) {
          /* only if the path doesn't look like a boolean option! */
          co->path = strdup(ptr);
          if(!co->path)
            badcookie = TRUE;
          else {
            co->spath = sanitize_cookie_path(co->path);
            if(!co->spath) {
              badcookie = TRUE; /* out of memory bad */
            }
          }
          break;
        }
        /* this doesn't look like a path, make one up! */
        co->path = strdup("/");
        if(!co->path)
          badcookie = TRUE;
        co->spath = strdup("/");
        if(!co->spath)
          badcookie = TRUE;
        fields++; /* add a field and fall down to secure */
        /* FALLTHROUGH */
      case 3:
        co->secure = FALSE;
        if(strcasecompare(ptr, "TRUE")) {
          if(secure || c->running)
            co->secure = TRUE;
          else
            badcookie = TRUE;
        }
        break;
      case 4:
        if(curlx_strtoofft(ptr, NULL, 10, &co->expires))
          badcookie = TRUE;
        break;
      case 5:
        co->name = strdup(ptr);
        if(!co->name)
          badcookie = TRUE;
        else {
          /* For Netscape file format cookies we check prefix on the name */
          if(strncasecompare("__Secure-", co->name, 9))
            co->prefix |= COOKIE_PREFIX__SECURE;
          else if(strncasecompare("__Host-", co->name, 7))
            co->prefix |= COOKIE_PREFIX__HOST;
        }
        break;
      case 6:
        co->value = strdup(ptr);
        if(!co->value)
          badcookie = TRUE;
        break;
      }
    }
    if(6 == fields) {
      /* we got a cookie with blank contents, fix it */
      co->value = strdup("");
      if(!co->value)
        badcookie = TRUE;
      else
        fields++;
    }

    if(!badcookie && (7 != fields))
      /* we did not find the sufficient number of fields */
      badcookie = TRUE;

    if(badcookie) {
      freecookie(co);
      return NULL;
    }

  }

  if(co->prefix & COOKIE_PREFIX__SECURE) {
    /* The __Secure- prefix only requires that the cookie be set secure */
    if(!co->secure) {
      freecookie(co);
      return NULL;
    }
  }
  if(co->prefix & COOKIE_PREFIX__HOST) {
    /*
     * The __Host- prefix requires the cookie to be secure, have a "/" path
     * and not have a domain set.
     */
    if(co->secure && co->path && strcmp(co->path, "/") == 0 && !co->tailmatch)
      ;
    else {
      freecookie(co);
      return NULL;
    }
  }

  if(!c->running &&    /* read from a file */
     c->newsession &&  /* clean session cookies */
     !co->expires) {   /* this is a session cookie since it doesn't expire! */
    freecookie(co);
    return NULL;
  }

  co->livecookie = c->running;
  co->creationtime = ++c->lastct;

  /*
   * Now we have parsed the incoming line, we must now check if this supersedes
   * an already existing cookie, which it may if the previous have the same
   * domain and path as this.
   */

  /* at first, remove expired cookies */
  if(!noexpire)
    remove_expired(c);

#ifdef USE_LIBPSL
  /*
   * Check if the domain is a Public Suffix and if yes, ignore the cookie. We
   * must also check that the data handle isn't NULL since the psl code will
   * dereference it.
   */
  if(data && (domain && co->domain && !Curl_host_is_ipnum(co->domain))) {
    const psl_ctx_t *psl = Curl_psl_use(data);
    int acceptable;

    if(psl) {
      acceptable = psl_is_cookie_domain_acceptable(psl, domain, co->domain);
      Curl_psl_release(data);
    }
    else
      acceptable = !bad_domain(domain);

    if(!acceptable) {
      infof(data, "cookie '%s' dropped, domain '%s' must not "
                  "set cookies for '%s'", co->name, domain, co->domain);
      freecookie(co);
      return NULL;
    }
  }
#endif

  myhash = cookiehash(co->domain);
  clist = c->cookies[myhash];
  replace_old = FALSE;
  while(clist) {
    if(strcasecompare(clist->name, co->name)) {
      /* the names are identical */

      if(clist->domain && co->domain) {
        if(strcasecompare(clist->domain, co->domain) &&
          (clist->tailmatch == co->tailmatch))
          /* The domains are identical */
          replace_old = TRUE;
      }
      else if(!clist->domain && !co->domain)
        replace_old = TRUE;

      if(replace_old) {
        /* the domains were identical */

        if(clist->spath && co->spath) {
          if(clist->secure && !co->secure && !secure) {
            size_t cllen;
            const char *sep;

            /*
             * A non-secure cookie may not overlay an existing secure cookie.
             * For an existing cookie "a" with path "/login", refuse a new
             * cookie "a" with for example path "/login/en", while the path
             * "/loginhelper" is ok.
             */

            sep = strchr(clist->spath + 1, '/');

            if(sep)
              cllen = sep - clist->spath;
            else
              cllen = strlen(clist->spath);

            if(strncasecompare(clist->spath, co->spath, cllen)) {
              freecookie(co);
              return NULL;
            }
          }
          else if(strcasecompare(clist->spath, co->spath))
            replace_old = TRUE;
          else
            replace_old = FALSE;
        }
        else if(!clist->spath && !co->spath)
          replace_old = TRUE;
        else
          replace_old = FALSE;

      }

      if(replace_old && !co->livecookie && clist->livecookie) {
        /*
         * Both cookies matched fine, except that the already present cookie is
         * "live", which means it was set from a header, while the new one was
         * read from a file and thus isn't "live". "live" cookies are preferred
         * so the new cookie is freed.
         */
        freecookie(co);
        return NULL;
      }

      if(replace_old) {
        co->next = clist->next; /* get the next-pointer first */

        /* when replacing, creationtime is kept from old */
        co->creationtime = clist->creationtime;

        /* then free all the old pointers */
        free(clist->name);
        free(clist->value);
        free(clist->domain);
        free(clist->path);
        free(clist->spath);
        free(clist->expirestr);
        free(clist->version);
        free(clist->maxage);

        *clist = *co;  /* then store all the new data */

        free(co);   /* free the newly allocated memory */
        co = clist; /* point to the previous struct instead */

        /*
         * We have replaced a cookie, now skip the rest of the list but make
         * sure the 'lastc' pointer is properly set
         */
        do {
          lastc = clist;
          clist = clist->next;
        } while(clist);
        break;
      }
    }
    lastc = clist;
    clist = clist->next;
  }

  if(c->running)
    /* Only show this when NOT reading the cookies from a file */
    infof(data, "%s cookie %s=\"%s\" for domain %s, path %s, "
          "expire %" CURL_FORMAT_CURL_OFF_T,
          replace_old?"Replaced":"Added", co->name, co->value,
          co->domain, co->path, co->expires);

  if(!replace_old) {
    /* then make the last item point on this new one */
    if(lastc)
      lastc->next = co;
    else
      c->cookies[myhash] = co;
    c->numcookies++; /* one more cookie in the jar */
  }

  /*
   * Now that we've added a new cookie to the jar, update the expiration
   * tracker in case it is the next one to expire.
   */
  if(co->expires && (co->expires < c->next_expiration))
    c->next_expiration = co->expires;

  return co;
}


/*
 * Curl_cookie_init()
 *
 * Inits a cookie struct to read data from a local file. This is always
 * called before any cookies are set. File may be NULL in which case only the
 * struct is initialized. Is file is "-" then STDIN is read.
 *
 * If 'newsession' is TRUE, discard all "session cookies" on read from file.
 *
 * Note that 'data' might be called as NULL pointer.
 *
 * Returns NULL on out of memory. Invalid cookies are ignored.
 */
struct CookieInfo *Curl_cookie_init(struct Curl_easy *data,
                                    const char *file,
                                    struct CookieInfo *inc,
                                    bool newsession)
{
  struct CookieInfo *c;
  FILE *fp = NULL;
  bool fromfile = TRUE;
  char *line = NULL;

  if(NULL == inc) {
    /* we didn't get a struct, create one */
    c = calloc(1, sizeof(struct CookieInfo));
    if(!c)
      return NULL; /* failed to get memory */
    c->filename = strdup(file?file:"none"); /* copy the name just in case */
    if(!c->filename)
      goto fail; /* failed to get memory */
    /*
     * Initialize the next_expiration time to signal that we don't have enough
     * information yet.
     */
    c->next_expiration = CURL_OFF_T_MAX;
  }
  else {
    /* we got an already existing one, use that */
    c = inc;
  }
  c->running = FALSE; /* this is not running, this is init */

  if(file && !strcmp(file, "-")) {
    fp = stdin;
    fromfile = FALSE;
  }
  else if(file && !*file) {
    /* points to a "" string */
    fp = NULL;
  }
  else
    fp = file?fopen(file, FOPEN_READTEXT):NULL;

  c->newsession = newsession; /* new session? */

  if(fp) {
    char *lineptr;
    bool headerline;

    line = malloc(MAX_COOKIE_LINE);
    if(!line)
      goto fail;
    while(Curl_get_line(line, MAX_COOKIE_LINE, fp)) {
      if(checkprefix("Set-Cookie:", line)) {
        /* This is a cookie line, get it! */
        lineptr = &line[11];
        headerline = TRUE;
      }
      else {
        lineptr = line;
        headerline = FALSE;
      }
      while(*lineptr && ISBLANK(*lineptr))
        lineptr++;

      Curl_cookie_add(data, c, headerline, TRUE, lineptr, NULL, NULL, TRUE);
    }
    free(line); /* free the line buffer */

    /*
     * Remove expired cookies from the hash. We must make sure to run this
     * after reading the file, and not on every cookie.
     */
    remove_expired(c);

    if(fromfile)
      fclose(fp);
  }

  c->running = TRUE;          /* now, we're running */
  if(data)
    data->state.cookie_engine = TRUE;

  return c;

fail:
  free(line);
  /*
   * Only clean up if we allocated it here, as the original could still be in
   * use by a share handle.
   */
  if(!inc)
    Curl_cookie_cleanup(c);
  if(fromfile && fp)
    fclose(fp);
  return NULL; /* out of memory */
}

/*
 * cookie_sort
 *
 * Helper function to sort cookies such that the longest path gets before the
 * shorter path. Path, domain and name lengths are considered in that order,
 * with the creationtime as the tiebreaker. The creationtime is guaranteed to
 * be unique per cookie, so we know we will get an ordering at that point.
 */
static int cookie_sort(const void *p1, const void *p2)
{
  struct Cookie *c1 = *(struct Cookie **)p1;
  struct Cookie *c2 = *(struct Cookie **)p2;
  size_t l1, l2;

  /* 1 - compare cookie path lengths */
  l1 = c1->path ? strlen(c1->path) : 0;
  l2 = c2->path ? strlen(c2->path) : 0;

  if(l1 != l2)
    return (l2 > l1) ? 1 : -1 ; /* avoid size_t <=> int conversions */

  /* 2 - compare cookie domain lengths */
  l1 = c1->domain ? strlen(c1->domain) : 0;
  l2 = c2->domain ? strlen(c2->domain) : 0;

  if(l1 != l2)
    return (l2 > l1) ? 1 : -1 ;  /* avoid size_t <=> int conversions */

  /* 3 - compare cookie name lengths */
  l1 = c1->name ? strlen(c1->name) : 0;
  l2 = c2->name ? strlen(c2->name) : 0;

  if(l1 != l2)
    return (l2 > l1) ? 1 : -1;

  /* 4 - compare cookie creation time */
  return (c2->creationtime > c1->creationtime) ? 1 : -1;
}

/*
 * cookie_sort_ct
 *
 * Helper function to sort cookies according to creation time.
 */
static int cookie_sort_ct(const void *p1, const void *p2)
{
  struct Cookie *c1 = *(struct Cookie **)p1;
  struct Cookie *c2 = *(struct Cookie **)p2;

  return (c2->creationtime > c1->creationtime) ? 1 : -1;
}

#define CLONE(field)                     \
  do {                                   \
    if(src->field) {                     \
      d->field = strdup(src->field);     \
      if(!d->field)                      \
        goto fail;                       \
    }                                    \
  } while(0)

static struct Cookie *dup_cookie(struct Cookie *src)
{
  struct Cookie *d = calloc(sizeof(struct Cookie), 1);
  if(d) {
    CLONE(expirestr);
    CLONE(domain);
    CLONE(path);
    CLONE(spath);
    CLONE(name);
    CLONE(value);
    CLONE(maxage);
    CLONE(version);
    d->expires = src->expires;
    d->tailmatch = src->tailmatch;
    d->secure = src->secure;
    d->livecookie = src->livecookie;
    d->httponly = src->httponly;
    d->creationtime = src->creationtime;
  }
  return d;

  fail:
  freecookie(d);
  return NULL;
}

/*
 * Curl_cookie_getlist
 *
 * For a given host and path, return a linked list of cookies that the client
 * should send to the server if used now. The secure boolean informs the cookie
 * if a secure connection is achieved or not.
 *
 * It shall only return cookies that haven't expired.
 */
struct Cookie *Curl_cookie_getlist(struct CookieInfo *c,
                                   const char *host, const char *path,
                                   bool secure)
{
  struct Cookie *newco;
  struct Cookie *co;
  struct Cookie *mainco = NULL;
  size_t matches = 0;
  bool is_ip;
  const size_t myhash = cookiehash(host);

  if(!c || !c->cookies[myhash])
    return NULL; /* no cookie struct or no cookies in the struct */

  /* at first, remove expired cookies */
  remove_expired(c);

  /* check if host is an IP(v4|v6) address */
  is_ip = Curl_host_is_ipnum(host);

  co = c->cookies[myhash];

  while(co) {
    /* if the cookie requires we're secure we must only continue if we are! */
    if(co->secure?secure:TRUE) {

      /* now check if the domain is correct */
      if(!co->domain ||
         (co->tailmatch && !is_ip && tailmatch(co->domain, host)) ||
         ((!co->tailmatch || is_ip) && strcasecompare(host, co->domain)) ) {
        /*
         * the right part of the host matches the domain stuff in the
         * cookie data
         */

        /*
         * now check the left part of the path with the cookies path
         * requirement
         */
        if(!co->spath || pathmatch(co->spath, path) ) {

          /*
           * and now, we know this is a match and we should create an
           * entry for the return-linked-list
           */

          newco = dup_cookie(co);
          if(newco) {
            /* then modify our next */
            newco->next = mainco;

            /* point the main to us */
            mainco = newco;

            matches++;
          }
          else
            goto fail;
        }
      }
    }
    co = co->next;
  }

  if(matches) {
    /*
     * Now we need to make sure that if there is a name appearing more than
     * once, the longest specified path version comes first. To make this
     * the swiftest way, we just sort them all based on path length.
     */
    struct Cookie **array;
    size_t i;

    /* alloc an array and store all cookie pointers */
    array = malloc(sizeof(struct Cookie *) * matches);
    if(!array)
      goto fail;

    co = mainco;

    for(i = 0; co; co = co->next)
      array[i++] = co;

    /* now sort the cookie pointers in path length order */
    qsort(array, matches, sizeof(struct Cookie *), cookie_sort);

    /* remake the linked list order according to the new order */

    mainco = array[0]; /* start here */
    for(i = 0; i<matches-1; i++)
      array[i]->next = array[i + 1];
    array[matches-1]->next = NULL; /* terminate the list */

    free(array); /* remove the temporary data again */
  }

  return mainco; /* return the new list */

fail:
  /* failure, clear up the allocated chain and return NULL */
  Curl_cookie_freelist(mainco);
  return NULL;
}

/*
 * Curl_cookie_clearall
 *
 * Clear all existing cookies and reset the counter.
 */
void Curl_cookie_clearall(struct CookieInfo *cookies)
{
  if(cookies) {
    unsigned int i;
    for(i = 0; i < COOKIE_HASH_SIZE; i++) {
      Curl_cookie_freelist(cookies->cookies[i]);
      cookies->cookies[i] = NULL;
    }
    cookies->numcookies = 0;
  }
}

/*
 * Curl_cookie_freelist
 *
 * Free a list of cookies previously returned by Curl_cookie_getlist();
 */
void Curl_cookie_freelist(struct Cookie *co)
{
  struct Cookie *next;
  while(co) {
    next = co->next;
    freecookie(co);
    co = next;
  }
}

/*
 * Curl_cookie_clearsess
 *
 * Free all session cookies in the cookies list.
 */
void Curl_cookie_clearsess(struct CookieInfo *cookies)
{
  struct Cookie *first, *curr, *next, *prev = NULL;
  unsigned int i;

  if(!cookies)
    return;

  for(i = 0; i < COOKIE_HASH_SIZE; i++) {
    if(!cookies->cookies[i])
      continue;

    first = curr = prev = cookies->cookies[i];

    for(; curr; curr = next) {
      next = curr->next;
      if(!curr->expires) {
        if(first == curr)
          first = next;

        if(prev == curr)
          prev = next;
        else
          prev->next = next;

        freecookie(curr);
        cookies->numcookies--;
      }
      else
        prev = curr;
    }

    cookies->cookies[i] = first;
  }
}

/*
 * Curl_cookie_cleanup()
 *
 * Free a "cookie object" previous created with Curl_cookie_init().
 */
void Curl_cookie_cleanup(struct CookieInfo *c)
{
  if(c) {
    unsigned int i;
    free(c->filename);
    for(i = 0; i < COOKIE_HASH_SIZE; i++)
      Curl_cookie_freelist(c->cookies[i]);
    free(c); /* free the base struct as well */
  }
}

/*
 * get_netscape_format()
 *
 * Formats a string for Netscape output file, w/o a newline at the end.
 * Function returns a char * to a formatted line. The caller is responsible
 * for freeing the returned pointer.
 */
static char *get_netscape_format(const struct Cookie *co)
{
  return aprintf(
    "%s"     /* httponly preamble */
    "%s%s\t" /* domain */
    "%s\t"   /* tailmatch */
    "%s\t"   /* path */
    "%s\t"   /* secure */
    "%" CURL_FORMAT_CURL_OFF_T "\t"   /* expires */
    "%s\t"   /* name */
    "%s",    /* value */
    co->httponly?"#HttpOnly_":"",
    /*
     * Make sure all domains are prefixed with a dot if they allow
     * tailmatching. This is Mozilla-style.
     */
    (co->tailmatch && co->domain && co->domain[0] != '.')? ".":"",
    co->domain?co->domain:"unknown",
    co->tailmatch?"TRUE":"FALSE",
    co->path?co->path:"/",
    co->secure?"TRUE":"FALSE",
    co->expires,
    co->name,
    co->value?co->value:"");
}

/*
 * cookie_output()
 *
 * Writes all internally known cookies to the specified file. Specify
 * "-" as file name to write to stdout.
 *
 * The function returns non-zero on write failure.
 */
static CURLcode cookie_output(struct Curl_easy *data,
                              struct CookieInfo *c, const char *filename)
{
  struct Cookie *co;
  FILE *out = NULL;
  bool use_stdout = FALSE;
  char *tempstore = NULL;
  CURLcode error = CURLE_OK;

  if(!c)
    /* no cookie engine alive */
    return CURLE_OK;

  /* at first, remove expired cookies */
  remove_expired(c);

  if(!strcmp("-", filename)) {
    /* use stdout */
    out = stdout;
    use_stdout = TRUE;
  }
  else {
    unsigned char randsuffix[9];

    if(Curl_rand_hex(data, randsuffix, sizeof(randsuffix)))
      return 2;

    tempstore = aprintf("%s.%s.tmp", filename, randsuffix);
    if(!tempstore)
      return CURLE_OUT_OF_MEMORY;

    out = fopen(tempstore, FOPEN_WRITETEXT);
    if(!out) {
      error = CURLE_WRITE_ERROR;
      goto error;
    }
  }

  fputs("# Netscape HTTP Cookie File\n"
        "# https://curl.se/docs/http-cookies.html\n"
        "# This file was generated by libcurl! Edit at your own risk.\n\n",
        out);

  if(c->numcookies) {
    unsigned int i;
    size_t nvalid = 0;
    struct Cookie **array;

    array = calloc(1, sizeof(struct Cookie *) * c->numcookies);
    if(!array) {
      error = CURLE_OUT_OF_MEMORY;
      goto error;
    }

    /* only sort the cookies with a domain property */
    for(i = 0; i < COOKIE_HASH_SIZE; i++) {
      for(co = c->cookies[i]; co; co = co->next) {
        if(!co->domain)
          continue;
        array[nvalid++] = co;
      }
    }

    qsort(array, nvalid, sizeof(struct Cookie *), cookie_sort_ct);

    for(i = 0; i < nvalid; i++) {
      char *format_ptr = get_netscape_format(array[i]);
      if(!format_ptr) {
        free(array);
        error = CURLE_OUT_OF_MEMORY;
        goto error;
      }
      fprintf(out, "%s\n", format_ptr);
      free(format_ptr);
    }

    free(array);
  }

  if(!use_stdout) {
    fclose(out);
    out = NULL;
    if(Curl_rename(tempstore, filename)) {
      unlink(tempstore);
      error = CURLE_WRITE_ERROR;
      goto error;
    }
  }

  /*
   * If we reach here we have successfully written a cookie file so theree is
   * no need to inspect the error, any error case should have jumped into the
   * error block below.
   */
  free(tempstore);
  return CURLE_OK;

error:
  if(out && !use_stdout)
    fclose(out);
  free(tempstore);
  return error;
}

static struct curl_slist *cookie_list(struct Curl_easy *data)
{
  struct curl_slist *list = NULL;
  struct curl_slist *beg;
  struct Cookie *c;
  char *line;
  unsigned int i;

  if(!data->cookies || (data->cookies->numcookies == 0))
    return NULL;

  for(i = 0; i < COOKIE_HASH_SIZE; i++) {
    for(c = data->cookies->cookies[i]; c; c = c->next) {
      if(!c->domain)
        continue;
      line = get_netscape_format(c);
      if(!line) {
        curl_slist_free_all(list);
        return NULL;
      }
      beg = Curl_slist_append_nodup(list, line);
      if(!beg) {
        free(line);
        curl_slist_free_all(list);
        return NULL;
      }
      list = beg;
    }
  }

  return list;
}

struct curl_slist *Curl_cookie_list(struct Curl_easy *data)
{
  struct curl_slist *list;
  Curl_share_lock(data, CURL_LOCK_DATA_COOKIE, CURL_LOCK_ACCESS_SINGLE);
  list = cookie_list(data);
  Curl_share_unlock(data, CURL_LOCK_DATA_COOKIE);
  return list;
}

void Curl_flush_cookies(struct Curl_easy *data, bool cleanup)
{
  CURLcode res;

  if(data->set.str[STRING_COOKIEJAR]) {
    if(data->state.cookielist) {
      /* If there is a list of cookie files to read, do it first so that
         we have all the told files read before we write the new jar.
         Curl_cookie_loadfiles() LOCKS and UNLOCKS the share itself! */
      Curl_cookie_loadfiles(data);
    }

    Curl_share_lock(data, CURL_LOCK_DATA_COOKIE, CURL_LOCK_ACCESS_SINGLE);

    /* if we have a destination file for all the cookies to get dumped to */
    res = cookie_output(data, data->cookies, data->set.str[STRING_COOKIEJAR]);
    if(res)
      infof(data, "WARNING: failed to save cookies in %s: %s",
            data->set.str[STRING_COOKIEJAR], curl_easy_strerror(res));
  }
  else {
    if(cleanup && data->state.cookielist) {
      /* since nothing is written, we can just free the list of cookie file
         names */
      curl_slist_free_all(data->state.cookielist); /* clean up list */
      data->state.cookielist = NULL;
    }
    Curl_share_lock(data, CURL_LOCK_DATA_COOKIE, CURL_LOCK_ACCESS_SINGLE);
  }

  if(cleanup && (!data->share || (data->cookies != data->share->cookies))) {
    Curl_cookie_cleanup(data->cookies);
    data->cookies = NULL;
  }
  Curl_share_unlock(data, CURL_LOCK_DATA_COOKIE);
}

#endif /* CURL_DISABLE_HTTP || CURL_DISABLE_COOKIES */
