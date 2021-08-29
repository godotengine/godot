#ifndef HEADER_CURL_IMAP_H
#define HEADER_CURL_IMAP_H
/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 2009 - 2020, Daniel Stenberg, <daniel@haxx.se>, et al.
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
 * IMAP unique setup
 ***************************************************************************/
typedef enum {
  IMAP_STOP,         /* do nothing state, stops the state machine */
  IMAP_SERVERGREET,  /* waiting for the initial greeting immediately after
                        a connect */
  IMAP_CAPABILITY,
  IMAP_STARTTLS,
  IMAP_UPGRADETLS,   /* asynchronously upgrade the connection to SSL/TLS
                       (multi mode only) */
  IMAP_AUTHENTICATE,
  IMAP_LOGIN,
  IMAP_LIST,
  IMAP_SELECT,
  IMAP_FETCH,
  IMAP_FETCH_FINAL,
  IMAP_APPEND,
  IMAP_APPEND_FINAL,
  IMAP_SEARCH,
  IMAP_LOGOUT,
  IMAP_LAST          /* never used */
} imapstate;

/* This IMAP struct is used in the Curl_easy. All IMAP data that is
   connection-oriented must be in imap_conn to properly deal with the fact that
   perhaps the Curl_easy is changed between the times the connection is
   used. */
struct IMAP {
  curl_pp_transfer transfer;
  char *mailbox;          /* Mailbox to select */
  char *uidvalidity;      /* UIDVALIDITY to check in select */
  char *uid;              /* Message UID to fetch */
  char *mindex;           /* Index in mail box of mail to fetch */
  char *section;          /* Message SECTION to fetch */
  char *partial;          /* Message PARTIAL to fetch */
  char *query;            /* Query to search for */
  char *custom;           /* Custom request */
  char *custom_params;    /* Parameters for the custom request */
};

/* imap_conn is used for struct connection-oriented data in the connectdata
   struct */
struct imap_conn {
  struct pingpong pp;
  imapstate state;            /* Always use imap.c:state() to change state! */
  bool ssldone;               /* Is connect() over SSL done? */
  bool preauth;               /* Is this connection PREAUTH? */
  struct SASL sasl;           /* SASL-related parameters */
  unsigned int preftype;      /* Preferred authentication type */
  unsigned int cmdid;         /* Last used command ID */
  char resptag[5];            /* Response tag to wait for */
  bool tls_supported;         /* StartTLS capability supported by server */
  bool login_disabled;        /* LOGIN command disabled by server */
  bool ir_supported;          /* Initial response supported by server */
  char *mailbox;              /* The last selected mailbox */
  char *mailbox_uidvalidity;  /* UIDVALIDITY parsed from select response */
  struct dynbuf dyn;          /* for the IMAP commands */
};

extern const struct Curl_handler Curl_handler_imap;
extern const struct Curl_handler Curl_handler_imaps;

/* Authentication type flags */
#define IMAP_TYPE_CLEARTEXT (1 << 0)
#define IMAP_TYPE_SASL      (1 << 1)

/* Authentication type values */
#define IMAP_TYPE_NONE      0
#define IMAP_TYPE_ANY       ~0U

#endif /* HEADER_CURL_IMAP_H */
