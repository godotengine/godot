/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 2019 - 2021, Daniel Stenberg, <daniel@haxx.se>, et al.
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
/*
 * The Alt-Svc: header is defined in RFC 7838:
 * https://tools.ietf.org/html/rfc7838
 */
#include "curl_setup.h"

#if !defined(CURL_DISABLE_HTTP) && !defined(CURL_DISABLE_ALTSVC)
#include <curl/curl.h>
#include "urldata.h"
#include "altsvc.h"
#include "curl_get_line.h"
#include "strcase.h"
#include "parsedate.h"
#include "sendf.h"
#include "warnless.h"
#include "rand.h"
#include "rename.h"

/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"

#define MAX_ALTSVC_LINE 4095
#define MAX_ALTSVC_DATELENSTR "64"
#define MAX_ALTSVC_DATELEN 64
#define MAX_ALTSVC_HOSTLENSTR "512"
#define MAX_ALTSVC_HOSTLEN 512
#define MAX_ALTSVC_ALPNLENSTR "10"
#define MAX_ALTSVC_ALPNLEN 10

#if defined(USE_QUICHE) && !defined(UNITTESTS)
#define H3VERSION "h3-29"
#elif defined(USE_NGTCP2) && !defined(UNITTESTS)
#define H3VERSION "h3-29"
#else
#define H3VERSION "h3"
#endif

static enum alpnid alpn2alpnid(char *name)
{
  if(strcasecompare(name, "h1"))
    return ALPN_h1;
  if(strcasecompare(name, "h2"))
    return ALPN_h2;
  if(strcasecompare(name, H3VERSION))
    return ALPN_h3;
  return ALPN_none; /* unknown, probably rubbish input */
}

/* Given the ALPN ID, return the name */
const char *Curl_alpnid2str(enum alpnid id)
{
  switch(id) {
  case ALPN_h1:
    return "h1";
  case ALPN_h2:
    return "h2";
  case ALPN_h3:
    return H3VERSION;
  default:
    return ""; /* bad */
  }
}


static void altsvc_free(struct altsvc *as)
{
  free(as->src.host);
  free(as->dst.host);
  free(as);
}

static struct altsvc *altsvc_createid(const char *srchost,
                                      const char *dsthost,
                                      enum alpnid srcalpnid,
                                      enum alpnid dstalpnid,
                                      unsigned int srcport,
                                      unsigned int dstport)
{
  struct altsvc *as = calloc(sizeof(struct altsvc), 1);
  if(!as)
    return NULL;

  as->src.host = strdup(srchost);
  if(!as->src.host)
    goto error;
  as->dst.host = strdup(dsthost);
  if(!as->dst.host)
    goto error;

  as->src.alpnid = srcalpnid;
  as->dst.alpnid = dstalpnid;
  as->src.port = curlx_ultous(srcport);
  as->dst.port = curlx_ultous(dstport);

  return as;
  error:
  altsvc_free(as);
  return NULL;
}

static struct altsvc *altsvc_create(char *srchost,
                                    char *dsthost,
                                    char *srcalpn,
                                    char *dstalpn,
                                    unsigned int srcport,
                                    unsigned int dstport)
{
  enum alpnid dstalpnid = alpn2alpnid(dstalpn);
  enum alpnid srcalpnid = alpn2alpnid(srcalpn);
  if(!srcalpnid || !dstalpnid)
    return NULL;
  return altsvc_createid(srchost, dsthost, srcalpnid, dstalpnid,
                         srcport, dstport);
}

/* only returns SERIOUS errors */
static CURLcode altsvc_add(struct altsvcinfo *asi, char *line)
{
  /* Example line:
     h2 example.com 443 h3 shiny.example.com 8443 "20191231 10:00:00" 1
   */
  char srchost[MAX_ALTSVC_HOSTLEN + 1];
  char dsthost[MAX_ALTSVC_HOSTLEN + 1];
  char srcalpn[MAX_ALTSVC_ALPNLEN + 1];
  char dstalpn[MAX_ALTSVC_ALPNLEN + 1];
  char date[MAX_ALTSVC_DATELEN + 1];
  unsigned int srcport;
  unsigned int dstport;
  unsigned int prio;
  unsigned int persist;
  int rc;

  rc = sscanf(line,
              "%" MAX_ALTSVC_ALPNLENSTR "s %" MAX_ALTSVC_HOSTLENSTR "s %u "
              "%" MAX_ALTSVC_ALPNLENSTR "s %" MAX_ALTSVC_HOSTLENSTR "s %u "
              "\"%" MAX_ALTSVC_DATELENSTR "[^\"]\" %u %u",
              srcalpn, srchost, &srcport,
              dstalpn, dsthost, &dstport,
              date, &persist, &prio);
  if(9 == rc) {
    struct altsvc *as;
    time_t expires = Curl_getdate_capped(date);
    as = altsvc_create(srchost, dsthost, srcalpn, dstalpn, srcport, dstport);
    if(as) {
      as->expires = expires;
      as->prio = prio;
      as->persist = persist ? 1 : 0;
      Curl_llist_insert_next(&asi->list, asi->list.tail, as, &as->node);
    }
  }

  return CURLE_OK;
}

/*
 * Load alt-svc entries from the given file. The text based line-oriented file
 * format is documented here:
 * https://github.com/curl/curl/wiki/QUIC-implementation
 *
 * This function only returns error on major problems that prevents alt-svc
 * handling to work completely. It will ignore individual syntactical errors
 * etc.
 */
static CURLcode altsvc_load(struct altsvcinfo *asi, const char *file)
{
  CURLcode result = CURLE_OK;
  char *line = NULL;
  FILE *fp;

  /* we need a private copy of the file name so that the altsvc cache file
     name survives an easy handle reset */
  free(asi->filename);
  asi->filename = strdup(file);
  if(!asi->filename)
    return CURLE_OUT_OF_MEMORY;

  fp = fopen(file, FOPEN_READTEXT);
  if(fp) {
    line = malloc(MAX_ALTSVC_LINE);
    if(!line)
      goto fail;
    while(Curl_get_line(line, MAX_ALTSVC_LINE, fp)) {
      char *lineptr = line;
      while(*lineptr && ISBLANK(*lineptr))
        lineptr++;
      if(*lineptr == '#')
        /* skip commented lines */
        continue;

      altsvc_add(asi, lineptr);
    }
    free(line); /* free the line buffer */
    fclose(fp);
  }
  return result;

  fail:
  Curl_safefree(asi->filename);
  free(line);
  fclose(fp);
  return CURLE_OUT_OF_MEMORY;
}

/*
 * Write this single altsvc entry to a single output line
 */

static CURLcode altsvc_out(struct altsvc *as, FILE *fp)
{
  struct tm stamp;
  CURLcode result = Curl_gmtime(as->expires, &stamp);
  if(result)
    return result;

  fprintf(fp,
          "%s %s %u "
          "%s %s %u "
          "\"%d%02d%02d "
          "%02d:%02d:%02d\" "
          "%u %d\n",
          Curl_alpnid2str(as->src.alpnid), as->src.host, as->src.port,
          Curl_alpnid2str(as->dst.alpnid), as->dst.host, as->dst.port,
          stamp.tm_year + 1900, stamp.tm_mon + 1, stamp.tm_mday,
          stamp.tm_hour, stamp.tm_min, stamp.tm_sec,
          as->persist, as->prio);
  return CURLE_OK;
}

/* ---- library-wide functions below ---- */

/*
 * Curl_altsvc_init() creates a new altsvc cache.
 * It returns the new instance or NULL if something goes wrong.
 */
struct altsvcinfo *Curl_altsvc_init(void)
{
  struct altsvcinfo *asi = calloc(sizeof(struct altsvcinfo), 1);
  if(!asi)
    return NULL;
  Curl_llist_init(&asi->list, NULL);

  /* set default behavior */
  asi->flags = CURLALTSVC_H1
#ifdef USE_NGHTTP2
    | CURLALTSVC_H2
#endif
#ifdef ENABLE_QUIC
    | CURLALTSVC_H3
#endif
    ;
  return asi;
}

/*
 * Curl_altsvc_load() loads alt-svc from file.
 */
CURLcode Curl_altsvc_load(struct altsvcinfo *asi, const char *file)
{
  CURLcode result;
  DEBUGASSERT(asi);
  result = altsvc_load(asi, file);
  return result;
}

/*
 * Curl_altsvc_ctrl() passes on the external bitmask.
 */
CURLcode Curl_altsvc_ctrl(struct altsvcinfo *asi, const long ctrl)
{
  DEBUGASSERT(asi);
  if(!ctrl)
    /* unexpected */
    return CURLE_BAD_FUNCTION_ARGUMENT;
  asi->flags = ctrl;
  return CURLE_OK;
}

/*
 * Curl_altsvc_cleanup() frees an altsvc cache instance and all associated
 * resources.
 */
void Curl_altsvc_cleanup(struct altsvcinfo **altsvcp)
{
  struct Curl_llist_element *e;
  struct Curl_llist_element *n;
  if(*altsvcp) {
    struct altsvcinfo *altsvc = *altsvcp;
    for(e = altsvc->list.head; e; e = n) {
      struct altsvc *as = e->ptr;
      n = e->next;
      altsvc_free(as);
    }
    free(altsvc->filename);
    free(altsvc);
    *altsvcp = NULL; /* clear the pointer */
  }
}

/*
 * Curl_altsvc_save() writes the altsvc cache to a file.
 */
CURLcode Curl_altsvc_save(struct Curl_easy *data,
                          struct altsvcinfo *altsvc, const char *file)
{
  struct Curl_llist_element *e;
  struct Curl_llist_element *n;
  CURLcode result = CURLE_OK;
  FILE *out;
  char *tempstore;
  unsigned char randsuffix[9];

  if(!altsvc)
    /* no cache activated */
    return CURLE_OK;

  /* if not new name is given, use the one we stored from the load */
  if(!file && altsvc->filename)
    file = altsvc->filename;

  if((altsvc->flags & CURLALTSVC_READONLYFILE) || !file || !file[0])
    /* marked as read-only, no file or zero length file name */
    return CURLE_OK;

  if(Curl_rand_hex(data, randsuffix, sizeof(randsuffix)))
    return CURLE_FAILED_INIT;

  tempstore = aprintf("%s.%s.tmp", file, randsuffix);
  if(!tempstore)
    return CURLE_OUT_OF_MEMORY;

  out = fopen(tempstore, FOPEN_WRITETEXT);
  if(!out)
    result = CURLE_WRITE_ERROR;
  else {
    fputs("# Your alt-svc cache. https://curl.se/docs/alt-svc.html\n"
          "# This file was generated by libcurl! Edit at your own risk.\n",
          out);
    for(e = altsvc->list.head; e; e = n) {
      struct altsvc *as = e->ptr;
      n = e->next;
      result = altsvc_out(as, out);
      if(result)
        break;
    }
    fclose(out);
    if(!result && Curl_rename(tempstore, file))
      result = CURLE_WRITE_ERROR;

    if(result)
      unlink(tempstore);
  }
  free(tempstore);
  return result;
}

static CURLcode getalnum(const char **ptr, char *alpnbuf, size_t buflen)
{
  size_t len;
  const char *protop;
  const char *p = *ptr;
  while(*p && ISBLANK(*p))
    p++;
  protop = p;
  while(*p && !ISBLANK(*p) && (*p != ';') && (*p != '='))
    p++;
  len = p - protop;
  *ptr = p;

  if(!len || (len >= buflen))
    return CURLE_BAD_FUNCTION_ARGUMENT;
  memcpy(alpnbuf, protop, len);
  alpnbuf[len] = 0;
  return CURLE_OK;
}

/* altsvc_flush() removes all alternatives for this source origin from the
   list */
static void altsvc_flush(struct altsvcinfo *asi, enum alpnid srcalpnid,
                         const char *srchost, unsigned short srcport)
{
  struct Curl_llist_element *e;
  struct Curl_llist_element *n;
  for(e = asi->list.head; e; e = n) {
    struct altsvc *as = e->ptr;
    n = e->next;
    if((srcalpnid == as->src.alpnid) &&
       (srcport == as->src.port) &&
       strcasecompare(srchost, as->src.host)) {
      Curl_llist_remove(&asi->list, e, NULL);
      altsvc_free(as);
    }
  }
}

#ifdef DEBUGBUILD
/* to play well with debug builds, we can *set* a fixed time this will
   return */
static time_t debugtime(void *unused)
{
  char *timestr = getenv("CURL_TIME");
  (void)unused;
  if(timestr) {
    unsigned long val = strtol(timestr, NULL, 10);
    return (time_t)val;
  }
  return time(NULL);
}
#define time(x) debugtime(x)
#endif

#define ISNEWLINE(x) (((x) == '\n') || (x) == '\r')

/*
 * Curl_altsvc_parse() takes an incoming alt-svc response header and stores
 * the data correctly in the cache.
 *
 * 'value' points to the header *value*. That's contents to the right of the
 * header name.
 *
 * Currently this function rejects invalid data without returning an error.
 * Invalid host name, port number will result in the specific alternative
 * being rejected. Unknown protocols are skipped.
 */
CURLcode Curl_altsvc_parse(struct Curl_easy *data,
                           struct altsvcinfo *asi, const char *value,
                           enum alpnid srcalpnid, const char *srchost,
                           unsigned short srcport)
{
  const char *p = value;
  size_t len;
  char namebuf[MAX_ALTSVC_HOSTLEN] = "";
  char alpnbuf[MAX_ALTSVC_ALPNLEN] = "";
  struct altsvc *as;
  unsigned short dstport = srcport; /* the same by default */
  CURLcode result = getalnum(&p, alpnbuf, sizeof(alpnbuf));
#ifdef CURL_DISABLE_VERBOSE_STRINGS
  (void)data;
#endif
  if(result) {
    infof(data, "Excessive alt-svc header, ignoring.");
    return CURLE_OK;
  }

  DEBUGASSERT(asi);

  /* Flush all cached alternatives for this source origin, if any */
  altsvc_flush(asi, srcalpnid, srchost, srcport);

  /* "clear" is a magic keyword */
  if(strcasecompare(alpnbuf, "clear")) {
    return CURLE_OK;
  }

  do {
    if(*p == '=') {
      /* [protocol]="[host][:port]" */
      enum alpnid dstalpnid = alpn2alpnid(alpnbuf); /* the same by default */
      p++;
      if(*p == '\"') {
        const char *dsthost = "";
        const char *value_ptr;
        char option[32];
        unsigned long num;
        char *end_ptr;
        bool quoted = FALSE;
        time_t maxage = 24 * 3600; /* default is 24 hours */
        bool persist = FALSE;
        p++;
        if(*p != ':') {
          /* host name starts here */
          const char *hostp = p;
          while(*p && (ISALNUM(*p) || (*p == '.') || (*p == '-')))
            p++;
          len = p - hostp;
          if(!len || (len >= MAX_ALTSVC_HOSTLEN)) {
            infof(data, "Excessive alt-svc host name, ignoring.");
            dstalpnid = ALPN_none;
          }
          else {
            memcpy(namebuf, hostp, len);
            namebuf[len] = 0;
            dsthost = namebuf;
          }
        }
        else {
          /* no destination name, use source host */
          dsthost = srchost;
        }
        if(*p == ':') {
          /* a port number */
          unsigned long port = strtoul(++p, &end_ptr, 10);
          if(port > USHRT_MAX || end_ptr == p || *end_ptr != '\"') {
            infof(data, "Unknown alt-svc port number, ignoring.");
            dstalpnid = ALPN_none;
          }
          p = end_ptr;
          dstport = curlx_ultous(port);
        }
        if(*p++ != '\"')
          break;
        /* Handle the optional 'ma' and 'persist' flags. Unknown flags
           are skipped. */
        for(;;) {
          while(ISBLANK(*p))
            p++;
          if(*p != ';')
            break;
          p++; /* pass the semicolon */
          if(!*p || ISNEWLINE(*p))
            break;
          result = getalnum(&p, option, sizeof(option));
          if(result) {
            /* skip option if name is too long */
            option[0] = '\0';
          }
          while(*p && ISBLANK(*p))
            p++;
          if(*p != '=')
            return CURLE_OK;
          p++;
          while(*p && ISBLANK(*p))
            p++;
          if(!*p)
            return CURLE_OK;
          if(*p == '\"') {
            /* quoted value */
            p++;
            quoted = TRUE;
          }
          value_ptr = p;
          if(quoted) {
            while(*p && *p != '\"')
              p++;
            if(!*p++)
              return CURLE_OK;
          }
          else {
            while(*p && !ISBLANK(*p) && *p!= ';' && *p != ',')
              p++;
          }
          num = strtoul(value_ptr, &end_ptr, 10);
          if((end_ptr != value_ptr) && (num < ULONG_MAX)) {
            if(strcasecompare("ma", option))
              maxage = num;
            else if(strcasecompare("persist", option) && (num == 1))
              persist = TRUE;
          }
        }
        if(dstalpnid) {
          as = altsvc_createid(srchost, dsthost,
                               srcalpnid, dstalpnid,
                               srcport, dstport);
          if(as) {
            /* The expires time also needs to take the Age: value (if any) into
               account. [See RFC 7838 section 3.1] */
            as->expires = maxage + time(NULL);
            as->persist = persist;
            Curl_llist_insert_next(&asi->list, asi->list.tail, as, &as->node);
            infof(data, "Added alt-svc: %s:%d over %s", dsthost, dstport,
                  Curl_alpnid2str(dstalpnid));
          }
        }
        else {
          infof(data, "Unknown alt-svc protocol \"%s\", skipping.",
                alpnbuf);
        }
      }
      else
        break;
      /* after the double quote there can be a comma if there's another
         string or a semicolon if no more */
      if(*p == ',') {
        /* comma means another alternative is presented */
        p++;
        result = getalnum(&p, alpnbuf, sizeof(alpnbuf));
        if(result)
          break;
      }
    }
    else
      break;
  } while(*p && (*p != ';') && (*p != '\n') && (*p != '\r'));

  return CURLE_OK;
}

/*
 * Return TRUE on a match
 */
bool Curl_altsvc_lookup(struct altsvcinfo *asi,
                        enum alpnid srcalpnid, const char *srchost,
                        int srcport,
                        struct altsvc **dstentry,
                        const int versions) /* one or more bits */
{
  struct Curl_llist_element *e;
  struct Curl_llist_element *n;
  time_t now = time(NULL);
  DEBUGASSERT(asi);
  DEBUGASSERT(srchost);
  DEBUGASSERT(dstentry);

  for(e = asi->list.head; e; e = n) {
    struct altsvc *as = e->ptr;
    n = e->next;
    if(as->expires < now) {
      /* an expired entry, remove */
      Curl_llist_remove(&asi->list, e, NULL);
      altsvc_free(as);
      continue;
    }
    if((as->src.alpnid == srcalpnid) &&
       strcasecompare(as->src.host, srchost) &&
       (as->src.port == srcport) &&
       (versions & as->dst.alpnid)) {
      /* match */
      *dstentry = as;
      return TRUE;
    }
  }
  return FALSE;
}

#endif /* !CURL_DISABLE_HTTP && !CURL_DISABLE_ALTSVC */
