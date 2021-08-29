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

#include "urldata.h"
#include "urlapi-int.h"
#include "strcase.h"
#include "dotdot.h"
#include "url.h"
#include "escape.h"
#include "curl_ctype.h"
#include "inet_pton.h"

/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"

  /* MSDOS/Windows style drive prefix, eg c: in c:foo */
#define STARTS_WITH_DRIVE_PREFIX(str) \
  ((('a' <= str[0] && str[0] <= 'z') || \
    ('A' <= str[0] && str[0] <= 'Z')) && \
   (str[1] == ':'))

  /* MSDOS/Windows style drive prefix, optionally with
   * a '|' instead of ':', followed by a slash or NUL */
#define STARTS_WITH_URL_DRIVE_PREFIX(str) \
  ((('a' <= (str)[0] && (str)[0] <= 'z') || \
    ('A' <= (str)[0] && (str)[0] <= 'Z')) && \
   ((str)[1] == ':' || (str)[1] == '|') && \
   ((str)[2] == '/' || (str)[2] == '\\' || (str)[2] == 0))

/* scheme is not URL encoded, the longest libcurl supported ones are... */
#define MAX_SCHEME_LEN 40

/* Internal representation of CURLU. Point to URL-encoded strings. */
struct Curl_URL {
  char *scheme;
  char *user;
  char *password;
  char *options; /* IMAP only? */
  char *host;
  char *zoneid; /* for numerical IPv6 addresses */
  char *port;
  char *path;
  char *query;
  char *fragment;

  char *scratch; /* temporary scratch area */
  char *temppath; /* temporary path pointer */
  long portnum; /* the numerical version */
};

#define DEFAULT_SCHEME "https"

static void free_urlhandle(struct Curl_URL *u)
{
  free(u->scheme);
  free(u->user);
  free(u->password);
  free(u->options);
  free(u->host);
  free(u->zoneid);
  free(u->port);
  free(u->path);
  free(u->query);
  free(u->fragment);
  free(u->scratch);
  free(u->temppath);
}

/* move the full contents of one handle onto another and
   free the original */
static void mv_urlhandle(struct Curl_URL *from,
                         struct Curl_URL *to)
{
  free_urlhandle(to);
  *to = *from;
  free(from);
}

/*
 * Find the separator at the end of the host name, or the '?' in cases like
 * http://www.url.com?id=2380
 */
static const char *find_host_sep(const char *url)
{
  const char *sep;
  const char *query;

  /* Find the start of the hostname */
  sep = strstr(url, "//");
  if(!sep)
    sep = url;
  else
    sep += 2;

  query = strchr(sep, '?');
  sep = strchr(sep, '/');

  if(!sep)
    sep = url + strlen(url);

  if(!query)
    query = url + strlen(url);

  return sep < query ? sep : query;
}

/*
 * Decide in an encoding-independent manner whether a character in an
 * URL must be escaped. The same criterion must be used in strlen_url()
 * and strcpy_url().
 */
static bool urlchar_needs_escaping(int c)
{
  return !(ISCNTRL(c) || ISSPACE(c) || ISGRAPH(c));
}

/*
 * strlen_url() returns the length of the given URL if the spaces within the
 * URL were properly URL encoded.
 * URL encoding should be skipped for host names, otherwise IDN resolution
 * will fail.
 */
static size_t strlen_url(const char *url, bool relative)
{
  const unsigned char *ptr;
  size_t newlen = 0;
  bool left = TRUE; /* left side of the ? */
  const unsigned char *host_sep = (const unsigned char *) url;

  if(!relative)
    host_sep = (const unsigned char *) find_host_sep(url);

  for(ptr = (unsigned char *)url; *ptr; ptr++) {

    if(ptr < host_sep) {
      ++newlen;
      continue;
    }

    if(*ptr == ' ') {
      if(left)
        newlen += 3;
      else
        newlen++;
      continue;
    }

    if (*ptr == '?')
      left = FALSE;

    if(urlchar_needs_escaping(*ptr))
      newlen += 2;

    newlen++;
  }

  return newlen;
}

/* strcpy_url() copies a url to a output buffer and URL-encodes the spaces in
 * the source URL accordingly.
 * URL encoding should be skipped for host names, otherwise IDN resolution
 * will fail.
 */
static void strcpy_url(char *output, const char *url, bool relative)
{
  /* we must add this with whitespace-replacing */
  bool left = TRUE;
  const unsigned char *iptr;
  char *optr = output;
  const unsigned char *host_sep = (const unsigned char *) url;

  if(!relative)
    host_sep = (const unsigned char *) find_host_sep(url);

  for(iptr = (unsigned char *)url;    /* read from here */
      *iptr;         /* until zero byte */
      iptr++) {

    if(iptr < host_sep) {
      *optr++ = *iptr;
      continue;
    }

    if(*iptr == ' ') {
      if(left) {
        *optr++='%'; /* add a '%' */
        *optr++='2'; /* add a '2' */
        *optr++='0'; /* add a '0' */
      }
      else
        *optr++='+'; /* add a '+' here */
      continue;
    }

    if(*iptr == '?')
      left = FALSE;

    if(urlchar_needs_escaping(*iptr)) {
      msnprintf(optr, 4, "%%%02x", *iptr);
      optr += 3;
    }
    else
      *optr++ = *iptr;
  }
  *optr = 0; /* null-terminate output buffer */

}

/*
 * Returns true if the given URL is absolute (as opposed to relative). Returns
 * the scheme in the buffer if TRUE and 'buf' is non-NULL. The buflen must
 * be larger than MAX_SCHEME_LEN if buf is set.
 */
bool Curl_is_absolute_url(const char *url, char *buf, size_t buflen)
{
  size_t i;
  DEBUGASSERT(!buf || (buflen > MAX_SCHEME_LEN));
  (void)buflen; /* only used in debug-builds */
  if(buf)
    buf[0] = 0; /* always leave a defined value in buf */
#ifdef WIN32
  if(STARTS_WITH_DRIVE_PREFIX(url))
    return FALSE;
#endif
  for(i = 0; i < MAX_SCHEME_LEN; ++i) {
    char s = url[i];
    if(s && (ISALNUM(s) || (s == '+') || (s == '-') || (s == '.') )) {
      /* RFC 3986 3.1 explains:
        scheme      = ALPHA *( ALPHA / DIGIT / "+" / "-" / "." )
      */
    }
    else {
      break;
    }
  }
  if(i && (url[i] == ':') && (url[i + 1] == '/')) {
    if(buf) {
      buf[i] = 0;
      while(i--) {
        buf[i] = (char)TOLOWER(url[i]);
      }
    }
    return TRUE;
  }
  return FALSE;
}

/*
 * Concatenate a relative URL to a base URL making it absolute.
 * URL-encodes any spaces.
 * The returned pointer must be freed by the caller unless NULL
 * (returns NULL on out of memory).
 */
static char *concat_url(const char *base, const char *relurl)
{
  /***
   TRY to append this new path to the old URL
   to the right of the host part. Oh crap, this is doomed to cause
   problems in the future...
  */
  char *newest;
  char *protsep;
  char *pathsep;
  size_t newlen;
  bool host_changed = FALSE;

  const char *useurl = relurl;
  size_t urllen;

  /* we must make our own copy of the URL to play with, as it may
     point to read-only data */
  char *url_clone = strdup(base);

  if(!url_clone)
    return NULL; /* skip out of this NOW */

  /* protsep points to the start of the host name */
  protsep = strstr(url_clone, "//");
  if(!protsep)
    protsep = url_clone;
  else
    protsep += 2; /* pass the slashes */

  if('/' != relurl[0]) {
    int level = 0;

    /* First we need to find out if there's a ?-letter in the URL,
       and cut it and the right-side of that off */
    pathsep = strchr(protsep, '?');
    if(pathsep)
      *pathsep = 0;

    /* we have a relative path to append to the last slash if there's one
       available, or if the new URL is just a query string (starts with a
       '?')  we append the new one at the end of the entire currently worked
       out URL */
    if(useurl[0] != '?') {
      pathsep = strrchr(protsep, '/');
      if(pathsep)
        *pathsep = 0;
    }

    /* Check if there's any slash after the host name, and if so, remember
       that position instead */
    pathsep = strchr(protsep, '/');
    if(pathsep)
      protsep = pathsep + 1;
    else
      protsep = NULL;

    /* now deal with one "./" or any amount of "../" in the newurl
       and act accordingly */

    if((useurl[0] == '.') && (useurl[1] == '/'))
      useurl += 2; /* just skip the "./" */

    while((useurl[0] == '.') &&
          (useurl[1] == '.') &&
          (useurl[2] == '/')) {
      level++;
      useurl += 3; /* pass the "../" */
    }

    if(protsep) {
      while(level--) {
        /* cut off one more level from the right of the original URL */
        pathsep = strrchr(protsep, '/');
        if(pathsep)
          *pathsep = 0;
        else {
          *protsep = 0;
          break;
        }
      }
    }
  }
  else {
    /* We got a new absolute path for this server */

    if(relurl[1] == '/') {
      /* the new URL starts with //, just keep the protocol part from the
         original one */
      *protsep = 0;
      useurl = &relurl[2]; /* we keep the slashes from the original, so we
                              skip the new ones */
      host_changed = TRUE;
    }
    else {
      /* cut off the original URL from the first slash, or deal with URLs
         without slash */
      pathsep = strchr(protsep, '/');
      if(pathsep) {
        /* When people use badly formatted URLs, such as
           "http://www.url.com?dir=/home/daniel" we must not use the first
           slash, if there's a ?-letter before it! */
        char *sep = strchr(protsep, '?');
        if(sep && (sep < pathsep))
          pathsep = sep;
        *pathsep = 0;
      }
      else {
        /* There was no slash. Now, since we might be operating on a badly
           formatted URL, such as "http://www.url.com?id=2380" which doesn't
           use a slash separator as it is supposed to, we need to check for a
           ?-letter as well! */
        pathsep = strchr(protsep, '?');
        if(pathsep)
          *pathsep = 0;
      }
    }
  }

  /* If the new part contains a space, this is a mighty stupid redirect
     but we still make an effort to do "right". To the left of a '?'
     letter we replace each space with %20 while it is replaced with '+'
     on the right side of the '?' letter.
  */
  newlen = strlen_url(useurl, !host_changed);

  urllen = strlen(url_clone);

  newest = malloc(urllen + 1 + /* possible slash */
                  newlen + 1 /* zero byte */);

  if(!newest) {
    free(url_clone); /* don't leak this */
    return NULL;
  }

  /* copy over the root url part */
  memcpy(newest, url_clone, urllen);

  /* check if we need to append a slash */
  if(('/' == useurl[0]) || (protsep && !*protsep) || ('?' == useurl[0]))
    ;
  else
    newest[urllen++]='/';

  /* then append the new piece on the right side */
  strcpy_url(&newest[urllen], useurl, !host_changed);

  free(url_clone);

  return newest;
}

/* scan for byte values < 31 or 127 */
static bool junkscan(const char *part, unsigned int flags)
{
  if(part) {
    static const char badbytes[]={
      /* */ 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
      0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
      0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
      0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
      0x7f, 0x00 /* null-terminate */
    };
    size_t n = strlen(part);
    size_t nfine = strcspn(part, badbytes);
    if(nfine != n)
      /* since we don't know which part is scanned, return a generic error
         code */
      return TRUE;
    if(!(flags & CURLU_ALLOW_SPACE) && strchr(part, ' '))
      return TRUE;
  }
  return FALSE;
}

/*
 * parse_hostname_login()
 *
 * Parse the login details (user name, password and options) from the URL and
 * strip them out of the host name
 *
 */
static CURLUcode parse_hostname_login(struct Curl_URL *u,
                                      char **hostname,
                                      unsigned int flags)
{
  CURLUcode result = CURLUE_OK;
  CURLcode ccode;
  char *userp = NULL;
  char *passwdp = NULL;
  char *optionsp = NULL;
  const struct Curl_handler *h = NULL;

  /* At this point, we're hoping all the other special cases have
   * been taken care of, so conn->host.name is at most
   *    [user[:password][;options]]@]hostname
   *
   * We need somewhere to put the embedded details, so do that first.
   */

  char *ptr = strchr(*hostname, '@');
  char *login = *hostname;

  if(!ptr)
    goto out;

  /* We will now try to extract the
   * possible login information in a string like:
   * ftp://user:password@ftp.my.site:8021/README */
  *hostname = ++ptr;

  /* if this is a known scheme, get some details */
  if(u->scheme)
    h = Curl_builtin_scheme(u->scheme);

  /* We could use the login information in the URL so extract it. Only parse
     options if the handler says we should. Note that 'h' might be NULL! */
  ccode = Curl_parse_login_details(login, ptr - login - 1,
                                   &userp, &passwdp,
                                   (h && (h->flags & PROTOPT_URLOPTIONS)) ?
                                   &optionsp:NULL);
  if(ccode) {
    result = CURLUE_BAD_LOGIN;
    goto out;
  }

  if(userp) {
    if(flags & CURLU_DISALLOW_USER) {
      /* Option DISALLOW_USER is set and url contains username. */
      result = CURLUE_USER_NOT_ALLOWED;
      goto out;
    }
    if(junkscan(userp, flags)) {
      result = CURLUE_BAD_USER;
      goto out;
    }
    u->user = userp;
  }

  if(passwdp) {
    if(junkscan(passwdp, flags)) {
      result = CURLUE_BAD_PASSWORD;
      goto out;
    }
    u->password = passwdp;
  }

  if(optionsp) {
    if(junkscan(optionsp, flags)) {
      result = CURLUE_BAD_LOGIN;
      goto out;
    }
    u->options = optionsp;
  }

  return CURLUE_OK;
  out:

  free(userp);
  free(passwdp);
  free(optionsp);
  u->user = NULL;
  u->password = NULL;
  u->options = NULL;

  return result;
}

UNITTEST CURLUcode Curl_parse_port(struct Curl_URL *u, char *hostname,
                                   bool has_scheme)
{
  char *portptr = NULL;
  char endbracket;
  int len;

  /*
   * Find the end of an IPv6 address, either on the ']' ending bracket or
   * a percent-encoded zone index.
   */
  if(1 == sscanf(hostname, "[%*45[0123456789abcdefABCDEF:.]%c%n",
                 &endbracket, &len)) {
    if(']' == endbracket)
      portptr = &hostname[len];
    else if('%' == endbracket) {
      int zonelen = len;
      if(1 == sscanf(hostname + zonelen, "%*[^]]%c%n", &endbracket, &len)) {
        if(']' != endbracket)
          return CURLUE_BAD_IPV6;
        portptr = &hostname[--zonelen + len + 1];
      }
      else
        return CURLUE_BAD_IPV6;
    }
    else
      return CURLUE_BAD_IPV6;

    /* this is a RFC2732-style specified IP-address */
    if(portptr && *portptr) {
      if(*portptr != ':')
        return CURLUE_BAD_IPV6;
    }
    else
      portptr = NULL;
  }
  else
    portptr = strchr(hostname, ':');

  if(portptr) {
    char *rest;
    long port;
    char portbuf[7];

    /* Browser behavior adaptation. If there's a colon with no digits after,
       just cut off the name there which makes us ignore the colon and just
       use the default port. Firefox, Chrome and Safari all do that.

       Don't do it if the URL has no scheme, to make something that looks like
       a scheme not work!
    */
    if(!portptr[1]) {
      *portptr = '\0';
      return has_scheme ? CURLUE_OK : CURLUE_BAD_PORT_NUMBER;
    }

    if(!ISDIGIT(portptr[1]))
      return CURLUE_BAD_PORT_NUMBER;

    port = strtol(portptr + 1, &rest, 10);  /* Port number must be decimal */

    if(port > 0xffff)
      return CURLUE_BAD_PORT_NUMBER;

    if(rest[0])
      return CURLUE_BAD_PORT_NUMBER;

    *portptr++ = '\0'; /* cut off the name there */
    *rest = 0;
    /* generate a new port number string to get rid of leading zeroes etc */
    msnprintf(portbuf, sizeof(portbuf), "%ld", port);
    u->portnum = port;
    u->port = strdup(portbuf);
    if(!u->port)
      return CURLUE_OUT_OF_MEMORY;
  }

  return CURLUE_OK;
}

static CURLUcode hostname_check(struct Curl_URL *u, char *hostname)
{
  size_t len;
  size_t hlen = strlen(hostname);

  if(hostname[0] == '[') {
#ifdef ENABLE_IPV6
    char dest[16]; /* fits a binary IPv6 address */
#endif
    const char *l = "0123456789abcdefABCDEF:.";
    if(hlen < 4) /* '[::]' is the shortest possible valid string */
      return CURLUE_BAD_IPV6;
    hostname++;
    hlen -= 2;

    if(hostname[hlen] != ']')
      return CURLUE_BAD_IPV6;

    /* only valid letters are ok */
    len = strspn(hostname, l);
    if(hlen != len) {
      hlen = len;
      if(hostname[len] == '%') {
        /* this could now be '%[zone id]' */
        char zoneid[16];
        int i = 0;
        char *h = &hostname[len + 1];
        /* pass '25' if present and is a url encoded percent sign */
        if(!strncmp(h, "25", 2) && h[2] && (h[2] != ']'))
          h += 2;
        while(*h && (*h != ']') && (i < 15))
          zoneid[i++] = *h++;
        if(!i || (']' != *h))
          /* impossible to reach? */
          return CURLUE_MALFORMED_INPUT;
        zoneid[i] = 0;
        u->zoneid = strdup(zoneid);
        if(!u->zoneid)
          return CURLUE_OUT_OF_MEMORY;
        hostname[len] = ']'; /* insert end bracket */
        hostname[len + 1] = 0; /* terminate the hostname */
      }
      else
        return CURLUE_BAD_IPV6;
      /* hostname is fine */
    }
#ifdef ENABLE_IPV6
    hostname[hlen] = 0; /* end the address there */
    if(1 != Curl_inet_pton(AF_INET6, hostname, dest))
      return CURLUE_BAD_IPV6;
    hostname[hlen] = ']'; /* restore ending bracket */
#endif
  }
  else {
    /* letters from the second string is not ok */
    len = strcspn(hostname, " \r\n");
    if(hlen != len)
      /* hostname with bad content */
      return CURLUE_BAD_HOSTNAME;
  }
  if(!hostname[0])
    return CURLUE_NO_HOST;
  return CURLUE_OK;
}

#define HOSTNAME_END(x) (((x) == '/') || ((x) == '?') || ((x) == '#'))

/*
 * Handle partial IPv4 numerical addresses and different bases, like
 * '16843009', '0x7f', '0x7f.1' '0177.1.1.1' etc.
 *
 * If the given input string is syntactically wrong or any part for example is
 * too big, this function returns FALSE and doesn't create any output.
 *
 * Output the "normalized" version of that input string in plain quad decimal
 * integers and return TRUE.
 */
static bool ipv4_normalize(const char *hostname, char *outp, size_t olen)
{
  bool done = FALSE;
  int n = 0;
  const char *c = hostname;
  unsigned long parts[4] = {0, 0, 0, 0};

  while(!done) {
    char *endp;
    unsigned long l;
    if((*c < '0') || (*c > '9'))
      /* most importantly this doesn't allow a leading plus or minus */
      return FALSE;
    l = strtoul(c, &endp, 0);

    /* overflow or nothing parsed at all */
    if(((l == ULONG_MAX) && (errno == ERANGE)) ||  (endp == c))
      return FALSE;

#if SIZEOF_LONG > 4
    /* a value larger than 32 bits */
    if(l > UINT_MAX)
      return FALSE;
#endif

    parts[n] = l;
    c = endp;

    switch (*c) {
    case '.' :
      if(n == 3)
        return FALSE;
      n++;
      c++;
      break;

    case '\0':
      done = TRUE;
      break;

    default:
      return FALSE;
    }
  }

  /* this is deemed a valid IPv4 numerical address */

  switch(n) {
  case 0: /* a -- 32 bits */
    msnprintf(outp, olen, "%u.%u.%u.%u",
              parts[0] >> 24, (parts[0] >> 16) & 0xff,
              (parts[0] >> 8) & 0xff, parts[0] & 0xff);
    break;
  case 1: /* a.b -- 8.24 bits */
    if((parts[0] > 0xff) || (parts[1] > 0xffffff))
      return FALSE;
    msnprintf(outp, olen, "%u.%u.%u.%u",
              parts[0], (parts[1] >> 16) & 0xff,
              (parts[1] >> 8) & 0xff, parts[1] & 0xff);
    break;
  case 2: /* a.b.c -- 8.8.16 bits */
    if((parts[0] > 0xff) || (parts[1] > 0xff) || (parts[2] > 0xffff))
      return FALSE;
    msnprintf(outp, olen, "%u.%u.%u.%u",
              parts[0], parts[1], (parts[2] >> 8) & 0xff,
              parts[2] & 0xff);
    break;
  case 3: /* a.b.c.d -- 8.8.8.8 bits */
    if((parts[0] > 0xff) || (parts[1] > 0xff) || (parts[2] > 0xff) ||
       (parts[3] > 0xff))
      return FALSE;
    msnprintf(outp, olen, "%u.%u.%u.%u",
              parts[0], parts[1], parts[2], parts[3]);
    break;
  }
  return TRUE;
}

/* return strdup'ed version in 'outp', possibly percent decoded */
static CURLUcode decode_host(char *hostname, char **outp)
{
  char *per = NULL;
  if(hostname[0] != '[')
    /* only decode if not an ipv6 numerical */
    per = strchr(hostname, '%');
  if(!per) {
    *outp = strdup(hostname);
    if(!*outp)
      return CURLUE_OUT_OF_MEMORY;
  }
  else {
    /* might be encoded */
    size_t dlen;
    CURLcode result = Curl_urldecode(NULL, hostname, 0,
                                     outp, &dlen, REJECT_CTRL);
    if(result)
      return CURLUE_BAD_HOSTNAME;
  }

  return CURLUE_OK;
}

static CURLUcode seturl(const char *url, CURLU *u, unsigned int flags)
{
  char *path;
  bool path_alloced = FALSE;
  bool uncpath = FALSE;
  char *hostname;
  char *query = NULL;
  char *fragment = NULL;
  CURLUcode result;
  bool url_has_scheme = FALSE;
  char schemebuf[MAX_SCHEME_LEN + 1];
  const char *schemep = NULL;
  size_t schemelen = 0;
  size_t urllen;

  DEBUGASSERT(url);

  /*************************************************************
   * Parse the URL.
   ************************************************************/
  /* allocate scratch area */
  urllen = strlen(url);
  if(urllen > CURL_MAX_INPUT_LENGTH)
    /* excessive input length */
    return CURLUE_MALFORMED_INPUT;

  path = u->scratch = malloc(urllen * 2 + 2);
  if(!path)
    return CURLUE_OUT_OF_MEMORY;

  hostname = &path[urllen + 1];
  hostname[0] = 0;

  if(Curl_is_absolute_url(url, schemebuf, sizeof(schemebuf))) {
    url_has_scheme = TRUE;
    schemelen = strlen(schemebuf);
  }

  /* handle the file: scheme */
  if(url_has_scheme && !strcmp(schemebuf, "file")) {
    if(urllen <= 6)
      /* file:/ is not enough to actually be a complete file: URL */
      return CURLUE_BAD_FILE_URL;

    /* path has been allocated large enough to hold this */
    strcpy(path, &url[5]);

    u->scheme = strdup("file");
    if(!u->scheme)
      return CURLUE_OUT_OF_MEMORY;

    /* Extra handling URLs with an authority component (i.e. that start with
     * "file://")
     *
     * We allow omitted hostname (e.g. file:/<path>) -- valid according to
     * RFC 8089, but not the (current) WHAT-WG URL spec.
     */
    if(path[0] == '/' && path[1] == '/') {
      /* swallow the two slashes */
      char *ptr = &path[2];

      /*
       * According to RFC 8089, a file: URL can be reliably dereferenced if:
       *
       *  o it has no/blank hostname, or
       *
       *  o the hostname matches "localhost" (case-insensitively), or
       *
       *  o the hostname is a FQDN that resolves to this machine, or
       *
       *  o it is an UNC String transformed to an URI (Windows only, RFC 8089
       *    Appendix E.3).
       *
       * For brevity, we only consider URLs with empty, "localhost", or
       * "127.0.0.1" hostnames as local, otherwise as an UNC String.
       *
       * Additionally, there is an exception for URLs with a Windows drive
       * letter in the authority (which was accidentally omitted from RFC 8089
       * Appendix E, but believe me, it was meant to be there. --MK)
       */
      if(ptr[0] != '/' && !STARTS_WITH_URL_DRIVE_PREFIX(ptr)) {
        /* the URL includes a host name, it must match "localhost" or
           "127.0.0.1" to be valid */
        if(checkprefix("localhost/", ptr) ||
           checkprefix("127.0.0.1/", ptr)) {
          ptr += 9; /* now points to the slash after the host */
        }
        else {
#if defined(WIN32)
          size_t len;

          /* the host name, NetBIOS computer name, can not contain disallowed
             chars, and the delimiting slash character must be appended to the
             host name */
          path = strpbrk(ptr, "/\\:*?\"<>|");
          if(!path || *path != '/')
            return CURLUE_BAD_FILE_URL;

          len = path - ptr;
          if(len) {
            memcpy(hostname, ptr, len);
            hostname[len] = 0;
            uncpath = TRUE;
          }

          ptr -= 2; /* now points to the // before the host in UNC */
#else
          /* Invalid file://hostname/, expected localhost or 127.0.0.1 or
             none */
          return CURLUE_BAD_FILE_URL;
#endif
        }
      }

      path = ptr;
    }

    if(!uncpath)
        hostname = NULL; /* no host for file: URLs by default */

#if !defined(MSDOS) && !defined(WIN32) && !defined(__CYGWIN__)
    /* Don't allow Windows drive letters when not in Windows.
     * This catches both "file:/c:" and "file:c:" */
    if(('/' == path[0] && STARTS_WITH_URL_DRIVE_PREFIX(&path[1])) ||
       STARTS_WITH_URL_DRIVE_PREFIX(path)) {
      /* File drive letters are only accepted in MSDOS/Windows */
      return CURLUE_BAD_FILE_URL;
    }
#else
    /* If the path starts with a slash and a drive letter, ditch the slash */
    if('/' == path[0] && STARTS_WITH_URL_DRIVE_PREFIX(&path[1])) {
      /* This cannot be done with strcpy, as the memory chunks overlap! */
      memmove(path, &path[1], strlen(&path[1]) + 1);
    }
#endif

  }
  else {
    /* clear path */
    const char *p;
    const char *hostp;
    size_t len;
    path[0] = 0;

    if(url_has_scheme) {
      int i = 0;
      p = &url[schemelen + 1];
      while(p && (*p == '/') && (i < 4)) {
        p++;
        i++;
      }
      if((i < 1) || (i>3))
        /* less than one or more than three slashes */
        return CURLUE_BAD_SLASHES;

      schemep = schemebuf;
      if(!Curl_builtin_scheme(schemep) &&
         !(flags & CURLU_NON_SUPPORT_SCHEME))
        return CURLUE_UNSUPPORTED_SCHEME;

      if(junkscan(schemep, flags))
        return CURLUE_BAD_SCHEME;
    }
    else {
      /* no scheme! */

      if(!(flags & (CURLU_DEFAULT_SCHEME|CURLU_GUESS_SCHEME)))
        return CURLUE_BAD_SCHEME;
      if(flags & CURLU_DEFAULT_SCHEME)
        schemep = DEFAULT_SCHEME;

      /*
       * The URL was badly formatted, let's try without scheme specified.
       */
      p = url;
    }
    hostp = p; /* host name starts here */

    /* find the end of the host name + port number */
    while(*p && !HOSTNAME_END(*p))
      p++;

    len = p - hostp;
    if(len) {
      memcpy(hostname, hostp, len);
      hostname[len] = 0;
    }
    else {
      if(!(flags & CURLU_NO_AUTHORITY))
        return CURLUE_NO_HOST;
    }

    len = strlen(p);
    memcpy(path, p, len);
    path[len] = 0;

    if(schemep) {
      u->scheme = strdup(schemep);
      if(!u->scheme)
        return CURLUE_OUT_OF_MEMORY;
    }
  }

  if((flags & CURLU_URLENCODE) && path[0]) {
    /* worst case output length is 3x the original! */
    char *newp = malloc(strlen(path) * 3);
    if(!newp)
      return CURLUE_OUT_OF_MEMORY;
    path_alloced = TRUE;
    strcpy_url(newp, path, TRUE); /* consider it relative */
    u->temppath = path = newp;
  }

  fragment = strchr(path, '#');
  if(fragment) {
    *fragment++ = 0;
    if(junkscan(fragment, flags))
      return CURLUE_BAD_FRAGMENT;
    if(fragment[0]) {
      u->fragment = strdup(fragment);
      if(!u->fragment)
        return CURLUE_OUT_OF_MEMORY;
    }
  }

  query = strchr(path, '?');
  if(query) {
    *query++ = 0;
    if(junkscan(query, flags))
      return CURLUE_BAD_QUERY;
    /* done even if the query part is a blank string */
    u->query = strdup(query);
    if(!u->query)
      return CURLUE_OUT_OF_MEMORY;
  }

  if(junkscan(path, flags))
    return CURLUE_BAD_PATH;

  if(!path[0])
    /* if there's no path left set, unset */
    path = NULL;
  else {
    if(!(flags & CURLU_PATH_AS_IS)) {
      /* remove ../ and ./ sequences according to RFC3986 */
      char *newp = Curl_dedotdotify(path);
      if(!newp)
        return CURLUE_OUT_OF_MEMORY;

      if(strcmp(newp, path)) {
        /* if we got a new version */
        if(path_alloced)
          Curl_safefree(u->temppath);
        u->temppath = path = newp;
        path_alloced = TRUE;
      }
      else
        free(newp);
    }

    u->path = path_alloced?path:strdup(path);
    if(!u->path)
      return CURLUE_OUT_OF_MEMORY;
    u->temppath = NULL; /* used now */
  }

  if(hostname) {
    char normalized_ipv4[sizeof("255.255.255.255") + 1];

    /*
     * Parse the login details and strip them out of the host name.
     */
    result = parse_hostname_login(u, &hostname, flags);
    if(result)
      return result;

    result = Curl_parse_port(u, hostname, url_has_scheme);
    if(result)
      return result;

    if(junkscan(hostname, flags))
      return CURLUE_BAD_HOSTNAME;

    if(0 == strlen(hostname) && (flags & CURLU_NO_AUTHORITY)) {
      /* Skip hostname check, it's allowed to be empty. */
      u->host = strdup("");
    }
    else {
      if(ipv4_normalize(hostname, normalized_ipv4, sizeof(normalized_ipv4)))
        u->host = strdup(normalized_ipv4);
      else {
        result = decode_host(hostname, &u->host);
        if(result)
          return result;
        result = hostname_check(u, u->host);
        if(result)
          return result;
      }
    }
    if(!u->host)
      return CURLUE_OUT_OF_MEMORY;
    if((flags & CURLU_GUESS_SCHEME) && !schemep) {
      /* legacy curl-style guess based on host name */
      if(checkprefix("ftp.", hostname))
        schemep = "ftp";
      else if(checkprefix("dict.", hostname))
        schemep = "dict";
      else if(checkprefix("ldap.", hostname))
        schemep = "ldap";
      else if(checkprefix("imap.", hostname))
        schemep = "imap";
      else if(checkprefix("smtp.", hostname))
        schemep = "smtp";
      else if(checkprefix("pop3.", hostname))
        schemep = "pop3";
      else
        schemep = "http";

      u->scheme = strdup(schemep);
      if(!u->scheme)
        return CURLUE_OUT_OF_MEMORY;
    }
  }

  Curl_safefree(u->scratch);
  Curl_safefree(u->temppath);

  return CURLUE_OK;
}

/*
 * Parse the URL and set the relevant members of the Curl_URL struct.
 */
static CURLUcode parseurl(const char *url, CURLU *u, unsigned int flags)
{
  CURLUcode result = seturl(url, u, flags);
  if(result) {
    free_urlhandle(u);
    memset(u, 0, sizeof(struct Curl_URL));
  }
  return result;
}

/*
 */
CURLU *curl_url(void)
{
  return calloc(sizeof(struct Curl_URL), 1);
}

void curl_url_cleanup(CURLU *u)
{
  if(u) {
    free_urlhandle(u);
    free(u);
  }
}

#define DUP(dest, src, name)                    \
  do {                                          \
    if(src->name) {                             \
      dest->name = strdup(src->name);           \
      if(!dest->name)                           \
        goto fail;                              \
    }                                           \
  } while(0)

CURLU *curl_url_dup(CURLU *in)
{
  struct Curl_URL *u = calloc(sizeof(struct Curl_URL), 1);
  if(u) {
    DUP(u, in, scheme);
    DUP(u, in, user);
    DUP(u, in, password);
    DUP(u, in, options);
    DUP(u, in, host);
    DUP(u, in, port);
    DUP(u, in, path);
    DUP(u, in, query);
    DUP(u, in, fragment);
    u->portnum = in->portnum;
  }
  return u;
  fail:
  curl_url_cleanup(u);
  return NULL;
}

CURLUcode curl_url_get(CURLU *u, CURLUPart what,
                       char **part, unsigned int flags)
{
  char *ptr;
  CURLUcode ifmissing = CURLUE_UNKNOWN_PART;
  char portbuf[7];
  bool urldecode = (flags & CURLU_URLDECODE)?1:0;
  bool urlencode = (flags & CURLU_URLENCODE)?1:0;
  bool plusdecode = FALSE;
  (void)flags;
  if(!u)
    return CURLUE_BAD_HANDLE;
  if(!part)
    return CURLUE_BAD_PARTPOINTER;
  *part = NULL;

  switch(what) {
  case CURLUPART_SCHEME:
    ptr = u->scheme;
    ifmissing = CURLUE_NO_SCHEME;
    urldecode = FALSE; /* never for schemes */
    break;
  case CURLUPART_USER:
    ptr = u->user;
    ifmissing = CURLUE_NO_USER;
    break;
  case CURLUPART_PASSWORD:
    ptr = u->password;
    ifmissing = CURLUE_NO_PASSWORD;
    break;
  case CURLUPART_OPTIONS:
    ptr = u->options;
    ifmissing = CURLUE_NO_OPTIONS;
    break;
  case CURLUPART_HOST:
    ptr = u->host;
    ifmissing = CURLUE_NO_HOST;
    break;
  case CURLUPART_ZONEID:
    ptr = u->zoneid;
    ifmissing = CURLUE_NO_ZONEID;
    break;
  case CURLUPART_PORT:
    ptr = u->port;
    ifmissing = CURLUE_NO_PORT;
    urldecode = FALSE; /* never for port */
    if(!ptr && (flags & CURLU_DEFAULT_PORT) && u->scheme) {
      /* there's no stored port number, but asked to deliver
         a default one for the scheme */
      const struct Curl_handler *h =
        Curl_builtin_scheme(u->scheme);
      if(h) {
        msnprintf(portbuf, sizeof(portbuf), "%u", h->defport);
        ptr = portbuf;
      }
    }
    else if(ptr && u->scheme) {
      /* there is a stored port number, but ask to inhibit if
         it matches the default one for the scheme */
      const struct Curl_handler *h =
        Curl_builtin_scheme(u->scheme);
      if(h && (h->defport == u->portnum) &&
         (flags & CURLU_NO_DEFAULT_PORT))
        ptr = NULL;
    }
    break;
  case CURLUPART_PATH:
    ptr = u->path;
    if(!ptr) {
      ptr = u->path = strdup("/");
      if(!u->path)
        return CURLUE_OUT_OF_MEMORY;
    }
    break;
  case CURLUPART_QUERY:
    ptr = u->query;
    ifmissing = CURLUE_NO_QUERY;
    plusdecode = urldecode;
    break;
  case CURLUPART_FRAGMENT:
    ptr = u->fragment;
    ifmissing = CURLUE_NO_FRAGMENT;
    break;
  case CURLUPART_URL: {
    char *url;
    char *scheme;
    char *options = u->options;
    char *port = u->port;
    char *allochost = NULL;
    if(u->scheme && strcasecompare("file", u->scheme)) {
      url = aprintf("file://%s%s%s",
                    u->path,
                    u->fragment? "#": "",
                    u->fragment? u->fragment : "");
    }
    else if(!u->host)
      return CURLUE_NO_HOST;
    else {
      const struct Curl_handler *h = NULL;
      if(u->scheme)
        scheme = u->scheme;
      else if(flags & CURLU_DEFAULT_SCHEME)
        scheme = (char *) DEFAULT_SCHEME;
      else
        return CURLUE_NO_SCHEME;

      h = Curl_builtin_scheme(scheme);
      if(!port && (flags & CURLU_DEFAULT_PORT)) {
        /* there's no stored port number, but asked to deliver
           a default one for the scheme */
        if(h) {
          msnprintf(portbuf, sizeof(portbuf), "%u", h->defport);
          port = portbuf;
        }
      }
      else if(port) {
        /* there is a stored port number, but asked to inhibit if it matches
           the default one for the scheme */
        if(h && (h->defport == u->portnum) &&
           (flags & CURLU_NO_DEFAULT_PORT))
          port = NULL;
      }

      if(h && !(h->flags & PROTOPT_URLOPTIONS))
        options = NULL;

      if(u->host[0] == '[') {
        if(u->zoneid) {
          /* make it '[ host %25 zoneid ]' */
          size_t hostlen = strlen(u->host);
          size_t alen = hostlen + 3 + strlen(u->zoneid) + 1;
          allochost = malloc(alen);
          if(!allochost)
            return CURLUE_OUT_OF_MEMORY;
          memcpy(allochost, u->host, hostlen - 1);
          msnprintf(&allochost[hostlen - 1], alen - hostlen + 1,
                    "%%25%s]", u->zoneid);
        }
      }
      else if(urlencode) {
        allochost = curl_easy_escape(NULL, u->host, 0);
        if(!allochost)
          return CURLUE_OUT_OF_MEMORY;
      }
      else {
        /* only encode '%' in output host name */
        char *host = u->host;
        size_t pcount = 0;
        /* first, count number of percents present in the name */
        while(*host) {
          if(*host == '%')
            pcount++;
          host++;
        }
        /* if there were percents, encode the host name */
        if(pcount) {
          size_t hostlen = strlen(u->host);
          size_t alen = hostlen + 2 * pcount + 1;
          char *o = allochost = malloc(alen);
          if(!allochost)
            return CURLUE_OUT_OF_MEMORY;

          host = u->host;
          while(*host) {
            if(*host == '%') {
              memcpy(o, "%25", 3);
              o += 3;
              host++;
              continue;
            }
            *o++ = *host++;
          }
          *o = '\0';
        }
      }

      url = aprintf("%s://%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s",
                    scheme,
                    u->user ? u->user : "",
                    u->password ? ":": "",
                    u->password ? u->password : "",
                    options ? ";" : "",
                    options ? options : "",
                    (u->user || u->password || options) ? "@": "",
                    allochost ? allochost : u->host,
                    port ? ":": "",
                    port ? port : "",
                    (u->path && (u->path[0] != '/')) ? "/": "",
                    u->path ? u->path : "/",
                    (u->query && u->query[0]) ? "?": "",
                    (u->query && u->query[0]) ? u->query : "",
                    u->fragment? "#": "",
                    u->fragment? u->fragment : "");
      free(allochost);
    }
    if(!url)
      return CURLUE_OUT_OF_MEMORY;
    *part = url;
    return CURLUE_OK;
  }
  default:
    ptr = NULL;
    break;
  }
  if(ptr) {
    *part = strdup(ptr);
    if(!*part)
      return CURLUE_OUT_OF_MEMORY;
    if(plusdecode) {
      /* convert + to space */
      char *plus;
      for(plus = *part; *plus; ++plus) {
        if(*plus == '+')
          *plus = ' ';
      }
    }
    if(urldecode) {
      char *decoded;
      size_t dlen;
      /* this unconditional rejection of control bytes is documented
         API behavior */
      CURLcode res = Curl_urldecode(NULL, *part, 0, &decoded, &dlen,
                                    REJECT_CTRL);
      free(*part);
      if(res) {
        *part = NULL;
        return CURLUE_URLDECODE;
      }
      *part = decoded;
    }
    return CURLUE_OK;
  }
  else
    return ifmissing;
}

CURLUcode curl_url_set(CURLU *u, CURLUPart what,
                       const char *part, unsigned int flags)
{
  char **storep = NULL;
  long port = 0;
  bool urlencode = (flags & CURLU_URLENCODE)? 1 : 0;
  bool plusencode = FALSE;
  bool urlskipslash = FALSE;
  bool appendquery = FALSE;
  bool equalsencode = FALSE;

  if(!u)
    return CURLUE_BAD_HANDLE;
  if(!part) {
    /* setting a part to NULL clears it */
    switch(what) {
    case CURLUPART_URL:
      break;
    case CURLUPART_SCHEME:
      storep = &u->scheme;
      break;
    case CURLUPART_USER:
      storep = &u->user;
      break;
    case CURLUPART_PASSWORD:
      storep = &u->password;
      break;
    case CURLUPART_OPTIONS:
      storep = &u->options;
      break;
    case CURLUPART_HOST:
      storep = &u->host;
      break;
    case CURLUPART_ZONEID:
      storep = &u->zoneid;
      break;
    case CURLUPART_PORT:
      u->portnum = 0;
      storep = &u->port;
      break;
    case CURLUPART_PATH:
      storep = &u->path;
      break;
    case CURLUPART_QUERY:
      storep = &u->query;
      break;
    case CURLUPART_FRAGMENT:
      storep = &u->fragment;
      break;
    default:
      return CURLUE_UNKNOWN_PART;
    }
    if(storep && *storep) {
      Curl_safefree(*storep);
    }
    return CURLUE_OK;
  }

  switch(what) {
  case CURLUPART_SCHEME:
    if(strlen(part) > MAX_SCHEME_LEN)
      /* too long */
      return CURLUE_BAD_SCHEME;
    if(!(flags & CURLU_NON_SUPPORT_SCHEME) &&
       /* verify that it is a fine scheme */
       !Curl_builtin_scheme(part))
      return CURLUE_UNSUPPORTED_SCHEME;
    storep = &u->scheme;
    urlencode = FALSE; /* never */
    break;
  case CURLUPART_USER:
    storep = &u->user;
    break;
  case CURLUPART_PASSWORD:
    storep = &u->password;
    break;
  case CURLUPART_OPTIONS:
    storep = &u->options;
    break;
  case CURLUPART_HOST: {
    size_t len = strcspn(part, " \r\n");
    if(strlen(part) != len)
      /* hostname with bad content */
      return CURLUE_BAD_HOSTNAME;
    storep = &u->host;
    Curl_safefree(u->zoneid);
    break;
  }
  case CURLUPART_ZONEID:
    storep = &u->zoneid;
    break;
  case CURLUPART_PORT:
  {
    char *endp;
    urlencode = FALSE; /* never */
    port = strtol(part, &endp, 10);  /* Port number must be decimal */
    if((port <= 0) || (port > 0xffff))
      return CURLUE_BAD_PORT_NUMBER;
    if(*endp)
      /* weirdly provided number, not good! */
      return CURLUE_BAD_PORT_NUMBER;
    storep = &u->port;
  }
  break;
  case CURLUPART_PATH:
    urlskipslash = TRUE;
    storep = &u->path;
    break;
  case CURLUPART_QUERY:
    plusencode = urlencode;
    appendquery = (flags & CURLU_APPENDQUERY)?1:0;
    equalsencode = appendquery;
    storep = &u->query;
    break;
  case CURLUPART_FRAGMENT:
    storep = &u->fragment;
    break;
  case CURLUPART_URL: {
    /*
     * Allow a new URL to replace the existing (if any) contents.
     *
     * If the existing contents is enough for a URL, allow a relative URL to
     * replace it.
     */
    CURLUcode result;
    char *oldurl;
    char *redired_url;
    CURLU *handle2;

    if(Curl_is_absolute_url(part, NULL, 0)) {
      handle2 = curl_url();
      if(!handle2)
        return CURLUE_OUT_OF_MEMORY;
      result = parseurl(part, handle2, flags);
      if(!result)
        mv_urlhandle(handle2, u);
      else
        curl_url_cleanup(handle2);
      return result;
    }
    /* extract the full "old" URL to do the redirect on */
    result = curl_url_get(u, CURLUPART_URL, &oldurl, flags);
    if(result) {
      /* couldn't get the old URL, just use the new! */
      handle2 = curl_url();
      if(!handle2)
        return CURLUE_OUT_OF_MEMORY;
      result = parseurl(part, handle2, flags);
      if(!result)
        mv_urlhandle(handle2, u);
      else
        curl_url_cleanup(handle2);
      return result;
    }

    /* apply the relative part to create a new URL */
    redired_url = concat_url(oldurl, part);
    free(oldurl);
    if(!redired_url)
      return CURLUE_OUT_OF_MEMORY;

    /* now parse the new URL */
    handle2 = curl_url();
    if(!handle2) {
      free(redired_url);
      return CURLUE_OUT_OF_MEMORY;
    }
    result = parseurl(redired_url, handle2, flags);
    free(redired_url);
    if(!result)
      mv_urlhandle(handle2, u);
    else
      curl_url_cleanup(handle2);
    return result;
  }
  default:
    return CURLUE_UNKNOWN_PART;
  }
  DEBUGASSERT(storep);
  {
    const char *newp = part;
    size_t nalloc = strlen(part);

    if(nalloc > CURL_MAX_INPUT_LENGTH)
      /* excessive input length */
      return CURLUE_MALFORMED_INPUT;

    if(urlencode) {
      const unsigned char *i;
      char *o;
      char *enc = malloc(nalloc * 3 + 1); /* for worst case! */
      if(!enc)
        return CURLUE_OUT_OF_MEMORY;
      for(i = (const unsigned char *)part, o = enc; *i; i++) {
        if((*i == ' ') && plusencode) {
          *o = '+';
          o++;
        }
        else if(Curl_isunreserved(*i) ||
                ((*i == '/') && urlskipslash) ||
                ((*i == '=') && equalsencode)) {
          if((*i == '=') && equalsencode)
            /* only skip the first equals sign */
            equalsencode = FALSE;
          *o = *i;
          o++;
        }
        else {
          msnprintf(o, 4, "%%%02x", *i);
          o += 3;
        }
      }
      *o = 0; /* null-terminate */
      newp = enc;
    }
    else {
      char *p;
      newp = strdup(part);
      if(!newp)
        return CURLUE_OUT_OF_MEMORY;
      p = (char *)newp;
      while(*p) {
        /* make sure percent encoded are lower case */
        if((*p == '%') && ISXDIGIT(p[1]) && ISXDIGIT(p[2]) &&
           (ISUPPER(p[1]) || ISUPPER(p[2]))) {
          p[1] = (char)TOLOWER(p[1]);
          p[2] = (char)TOLOWER(p[2]);
          p += 3;
        }
        else
          p++;
      }
    }

    if(appendquery) {
      /* Append the string onto the old query. Add a '&' separator if none is
         present at the end of the exsting query already */
      size_t querylen = u->query ? strlen(u->query) : 0;
      bool addamperand = querylen && (u->query[querylen -1] != '&');
      if(querylen) {
        size_t newplen = strlen(newp);
        char *p = malloc(querylen + addamperand + newplen + 1);
        if(!p) {
          free((char *)newp);
          return CURLUE_OUT_OF_MEMORY;
        }
        strcpy(p, u->query); /* original query */
        if(addamperand)
          p[querylen] = '&'; /* ampersand */
        strcpy(&p[querylen + addamperand], newp); /* new suffix */
        free((char *)newp);
        free(*storep);
        *storep = p;
        return CURLUE_OK;
      }
    }

    if(what == CURLUPART_HOST) {
      if(0 == strlen(newp) && (flags & CURLU_NO_AUTHORITY)) {
        /* Skip hostname check, it's allowed to be empty. */
      }
      else {
        if(hostname_check(u, (char *)newp)) {
          free((char *)newp);
          return CURLUE_BAD_HOSTNAME;
        }
      }
    }

    free(*storep);
    *storep = (char *)newp;
  }
  /* set after the string, to make it not assigned if the allocation above
     fails */
  if(port)
    u->portnum = port;
  return CURLUE_OK;
}
