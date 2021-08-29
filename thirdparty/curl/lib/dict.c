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

#ifndef CURL_DISABLE_DICT

#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif
#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif
#ifdef HAVE_ARPA_INET_H
#include <arpa/inet.h>
#endif
#ifdef HAVE_NET_IF_H
#include <net/if.h>
#endif
#ifdef HAVE_SYS_IOCTL_H
#include <sys/ioctl.h>
#endif

#ifdef HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif

#ifdef HAVE_SYS_SELECT_H
#include <sys/select.h>
#elif defined(HAVE_UNISTD_H)
#include <unistd.h>
#endif

#include "urldata.h"
#include <curl/curl.h>
#include "transfer.h"
#include "sendf.h"
#include "escape.h"
#include "progress.h"
#include "dict.h"
#include "curl_printf.h"
#include "strcase.h"
#include "curl_memory.h"
/* The last #include file should be: */
#include "memdebug.h"

/*
 * Forward declarations.
 */

static CURLcode dict_do(struct Curl_easy *data, bool *done);

/*
 * DICT protocol handler.
 */

const struct Curl_handler Curl_handler_dict = {
  "DICT",                               /* scheme */
  ZERO_NULL,                            /* setup_connection */
  dict_do,                              /* do_it */
  ZERO_NULL,                            /* done */
  ZERO_NULL,                            /* do_more */
  ZERO_NULL,                            /* connect_it */
  ZERO_NULL,                            /* connecting */
  ZERO_NULL,                            /* doing */
  ZERO_NULL,                            /* proto_getsock */
  ZERO_NULL,                            /* doing_getsock */
  ZERO_NULL,                            /* domore_getsock */
  ZERO_NULL,                            /* perform_getsock */
  ZERO_NULL,                            /* disconnect */
  ZERO_NULL,                            /* readwrite */
  ZERO_NULL,                            /* connection_check */
  ZERO_NULL,                            /* attach connection */
  PORT_DICT,                            /* defport */
  CURLPROTO_DICT,                       /* protocol */
  CURLPROTO_DICT,                       /* family */
  PROTOPT_NONE | PROTOPT_NOURLQUERY     /* flags */
};

static char *unescape_word(struct Curl_easy *data, const char *inputbuff)
{
  char *newp = NULL;
  char *dictp;
  size_t len;

  CURLcode result = Curl_urldecode(data, inputbuff, 0, &newp, &len,
                                   REJECT_NADA);
  if(!newp || result)
    return NULL;

  dictp = malloc(len*2 + 1); /* add one for terminating zero */
  if(dictp) {
    char *ptr;
    char ch;
    int olen = 0;
    /* According to RFC2229 section 2.2, these letters need to be escaped with
       \[letter] */
    for(ptr = newp;
        (ch = *ptr) != 0;
        ptr++) {
      if((ch <= 32) || (ch == 127) ||
          (ch == '\'') || (ch == '\"') || (ch == '\\')) {
        dictp[olen++] = '\\';
      }
      dictp[olen++] = ch;
    }
    dictp[olen] = 0;
  }
  free(newp);
  return dictp;
}

/* sendf() sends formatted data to the server */
static CURLcode sendf(curl_socket_t sockfd, struct Curl_easy *data,
                      const char *fmt, ...)
{
  ssize_t bytes_written;
  size_t write_len;
  CURLcode result = CURLE_OK;
  char *s;
  char *sptr;
  va_list ap;
  va_start(ap, fmt);
  s = vaprintf(fmt, ap); /* returns an allocated string */
  va_end(ap);
  if(!s)
    return CURLE_OUT_OF_MEMORY; /* failure */

  bytes_written = 0;
  write_len = strlen(s);
  sptr = s;

  for(;;) {
    /* Write the buffer to the socket */
    result = Curl_write(data, sockfd, sptr, write_len, &bytes_written);

    if(result)
      break;

    Curl_debug(data, CURLINFO_DATA_OUT, sptr, (size_t)bytes_written);

    if((size_t)bytes_written != write_len) {
      /* if not all was written at once, we must advance the pointer, decrease
         the size left and try again! */
      write_len -= bytes_written;
      sptr += bytes_written;
    }
    else
      break;
  }

  free(s); /* free the output string */

  return result;
}

static CURLcode dict_do(struct Curl_easy *data, bool *done)
{
  char *word;
  char *eword;
  char *ppath;
  char *database = NULL;
  char *strategy = NULL;
  char *nthdef = NULL; /* This is not part of the protocol, but required
                          by RFC 2229 */
  CURLcode result = CURLE_OK;
  struct connectdata *conn = data->conn;
  curl_socket_t sockfd = conn->sock[FIRSTSOCKET];

  char *path = data->state.up.path;

  *done = TRUE; /* unconditionally */

  if(conn->bits.user_passwd) {
    /* AUTH is missing */
  }

  if(strncasecompare(path, DICT_MATCH, sizeof(DICT_MATCH)-1) ||
     strncasecompare(path, DICT_MATCH2, sizeof(DICT_MATCH2)-1) ||
     strncasecompare(path, DICT_MATCH3, sizeof(DICT_MATCH3)-1)) {

    word = strchr(path, ':');
    if(word) {
      word++;
      database = strchr(word, ':');
      if(database) {
        *database++ = (char)0;
        strategy = strchr(database, ':');
        if(strategy) {
          *strategy++ = (char)0;
          nthdef = strchr(strategy, ':');
          if(nthdef) {
            *nthdef = (char)0;
          }
        }
      }
    }

    if(!word || (*word == (char)0)) {
      infof(data, "lookup word is missing");
      word = (char *)"default";
    }
    if(!database || (*database == (char)0)) {
      database = (char *)"!";
    }
    if(!strategy || (*strategy == (char)0)) {
      strategy = (char *)".";
    }

    eword = unescape_word(data, word);
    if(!eword)
      return CURLE_OUT_OF_MEMORY;

    result = sendf(sockfd, data,
                   "CLIENT " LIBCURL_NAME " " LIBCURL_VERSION "\r\n"
                   "MATCH "
                   "%s "    /* database */
                   "%s "    /* strategy */
                   "%s\r\n" /* word */
                   "QUIT\r\n",
                   database,
                   strategy,
                   eword);

    free(eword);

    if(result) {
      failf(data, "Failed sending DICT request");
      return result;
    }
    Curl_setup_transfer(data, FIRSTSOCKET, -1, FALSE, -1); /* no upload */
  }
  else if(strncasecompare(path, DICT_DEFINE, sizeof(DICT_DEFINE)-1) ||
          strncasecompare(path, DICT_DEFINE2, sizeof(DICT_DEFINE2)-1) ||
          strncasecompare(path, DICT_DEFINE3, sizeof(DICT_DEFINE3)-1)) {

    word = strchr(path, ':');
    if(word) {
      word++;
      database = strchr(word, ':');
      if(database) {
        *database++ = (char)0;
        nthdef = strchr(database, ':');
        if(nthdef) {
          *nthdef = (char)0;
        }
      }
    }

    if(!word || (*word == (char)0)) {
      infof(data, "lookup word is missing");
      word = (char *)"default";
    }
    if(!database || (*database == (char)0)) {
      database = (char *)"!";
    }

    eword = unescape_word(data, word);
    if(!eword)
      return CURLE_OUT_OF_MEMORY;

    result = sendf(sockfd, data,
                   "CLIENT " LIBCURL_NAME " " LIBCURL_VERSION "\r\n"
                   "DEFINE "
                   "%s "     /* database */
                   "%s\r\n"  /* word */
                   "QUIT\r\n",
                   database,
                   eword);

    free(eword);

    if(result) {
      failf(data, "Failed sending DICT request");
      return result;
    }
    Curl_setup_transfer(data, FIRSTSOCKET, -1, FALSE, -1);
  }
  else {

    ppath = strchr(path, '/');
    if(ppath) {
      int i;

      ppath++;
      for(i = 0; ppath[i]; i++) {
        if(ppath[i] == ':')
          ppath[i] = ' ';
      }
      result = sendf(sockfd, data,
                     "CLIENT " LIBCURL_NAME " " LIBCURL_VERSION "\r\n"
                     "%s\r\n"
                     "QUIT\r\n", ppath);
      if(result) {
        failf(data, "Failed sending DICT request");
        return result;
      }

      Curl_setup_transfer(data, FIRSTSOCKET, -1, FALSE, -1);
    }
  }

  return CURLE_OK;
}
#endif /*CURL_DISABLE_DICT*/
