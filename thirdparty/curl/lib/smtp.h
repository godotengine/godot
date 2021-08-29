#ifndef HEADER_CURL_SMTP_H
#define HEADER_CURL_SMTP_H
/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 2009 - 2021, Daniel Stenberg, <daniel@haxx.se>, et al.
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

#include "pingpong.h"
#include "curl_sasl.h"

/****************************************************************************
 * SMTP unique setup
 ***************************************************************************/
typedef enum {
  SMTP_STOP,        /* do nothing state, stops the state machine */
  SMTP_SERVERGREET, /* waiting for the initial greeting immediately after
                       a connect */
  SMTP_EHLO,
  SMTP_HELO,
  SMTP_STARTTLS,
  SMTP_UPGRADETLS,  /* asynchronously upgrade the connection to SSL/TLS
                       (multi mode only) */
  SMTP_AUTH,
  SMTP_COMMAND,     /* VRFY, EXPN, NOOP, RSET and HELP */
  SMTP_MAIL,        /* MAIL FROM */
  SMTP_RCPT,        /* RCPT TO */
  SMTP_DATA,
  SMTP_POSTDATA,
  SMTP_QUIT,
  SMTP_LAST         /* never used */
} smtpstate;

/* This SMTP struct is used in the Curl_easy. All SMTP data that is
   connection-oriented must be in smtp_conn to properly deal with the fact that
   perhaps the Curl_easy is changed between the times the connection is
   used. */
struct SMTP {
  curl_pp_transfer transfer;
  char *custom;            /* Custom Request */
  struct curl_slist *rcpt; /* Recipient list */
  bool rcpt_had_ok;        /* Whether any of RCPT TO commands (depends on
                              total number of recipients) succeeded so far */
  bool trailing_crlf;      /* Specifies if the trailing CRLF is present */
  int rcpt_last_error;     /* The last error received for RCPT TO command */
  size_t eob;              /* Number of bytes of the EOB (End Of Body) that
                              have been received so far */
};

/* smtp_conn is used for struct connection-oriented data in the connectdata
   struct */
struct smtp_conn {
  struct pingpong pp;
  smtpstate state;         /* Always use smtp.c:state() to change state! */
  bool ssldone;            /* Is connect() over SSL done? */
  char *domain;            /* Client address/name to send in the EHLO */
  struct SASL sasl;        /* SASL-related storage */
  bool tls_supported;      /* StartTLS capability supported by server */
  bool size_supported;     /* If server supports SIZE extension according to
                              RFC 1870 */
  bool utf8_supported;     /* If server supports SMTPUTF8 extension according
                              to RFC 6531 */
  bool auth_supported;     /* AUTH capability supported by server */
};

extern const struct Curl_handler Curl_handler_smtp;
extern const struct Curl_handler Curl_handler_smtps;

/* this is the 5-bytes End-Of-Body marker for SMTP */
#define SMTP_EOB "\x0d\x0a\x2e\x0d\x0a"
#define SMTP_EOB_LEN 5
#define SMTP_EOB_FIND_LEN 3

/* if found in data, replace it with this string instead */
#define SMTP_EOB_REPL "\x0d\x0a\x2e\x2e"
#define SMTP_EOB_REPL_LEN 4

CURLcode Curl_smtp_escape_eob(struct Curl_easy *data, const ssize_t nread);

#endif /* HEADER_CURL_SMTP_H */
