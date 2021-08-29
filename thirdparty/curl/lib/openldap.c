/***************************************************************************
 *                      _   _ ____  _
 *  Project         ___| | | |  _ \| |
 *                 / __| | | | |_) | |
 *                | (__| |_| |  _ <| |___
 *                 \___|\___/|_| \_\_____|
 *
 * Copyright (C) 2011 - 2021, Daniel Stenberg, <daniel@haxx.se>, et al.
 * Copyright (C) 2010, Howard Chu, <hyc@openldap.org>
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

#if !defined(CURL_DISABLE_LDAP) && defined(USE_OPENLDAP)

/*
 * Notice that USE_OPENLDAP is only a source code selection switch. When
 * libcurl is built with USE_OPENLDAP defined the libcurl source code that
 * gets compiled is the code from openldap.c, otherwise the code that gets
 * compiled is the code from ldap.c.
 *
 * When USE_OPENLDAP is defined a recent version of the OpenLDAP library
 * might be required for compilation and runtime. In order to use ancient
 * OpenLDAP library versions, USE_OPENLDAP shall not be defined.
 */

#include <ldap.h>

#include "urldata.h"
#include <curl/curl.h>
#include "sendf.h"
#include "vtls/vtls.h"
#include "transfer.h"
#include "curl_ldap.h"
#include "curl_base64.h"
#include "connect.h"
/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"

/*
 * Uncommenting this will enable the built-in debug logging of the openldap
 * library. The debug log level can be set using the CURL_OPENLDAP_TRACE
 * environment variable. The debug output is written to stderr.
 *
 * The library supports the following debug flags:
 * LDAP_DEBUG_NONE         0x0000
 * LDAP_DEBUG_TRACE        0x0001
 * LDAP_DEBUG_CONSTRUCT    0x0002
 * LDAP_DEBUG_DESTROY      0x0004
 * LDAP_DEBUG_PARAMETER    0x0008
 * LDAP_DEBUG_ANY          0xffff
 *
 * For example, use CURL_OPENLDAP_TRACE=0 for no debug,
 * CURL_OPENLDAP_TRACE=2 for LDAP_DEBUG_CONSTRUCT messages only,
 * CURL_OPENLDAP_TRACE=65535 for all debug message levels.
 */
/* #define CURL_OPENLDAP_DEBUG */

/* Machine states. */
typedef enum {
  OLDAP_STOP,           /* Do nothing state, stops the state machine */
  OLDAP_SSL,            /* Performing SSL handshake. */
  OLDAP_STARTTLS,       /* STARTTLS request sent. */
  OLDAP_TLS,            /* Performing TLS handshake. */
  OLDAP_BIND,           /* Simple bind reply. */
  OLDAP_BINDV2,         /* Simple bind reply in protocol version 2. */
  OLDAP_LAST            /* Never used */
} ldapstate;

#ifndef _LDAP_PVT_H
extern int ldap_pvt_url_scheme2proto(const char *);
extern int ldap_init_fd(ber_socket_t fd, int proto, const char *url,
                        LDAP **ld);
#endif

static CURLcode oldap_setup_connection(struct Curl_easy *data,
                                       struct connectdata *conn);
static CURLcode oldap_do(struct Curl_easy *data, bool *done);
static CURLcode oldap_done(struct Curl_easy *data, CURLcode, bool);
static CURLcode oldap_connect(struct Curl_easy *data, bool *done);
static CURLcode oldap_connecting(struct Curl_easy *data, bool *done);
static CURLcode oldap_disconnect(struct Curl_easy *data,
                                 struct connectdata *conn, bool dead);

static Curl_recv oldap_recv;

/*
 * LDAP protocol handler.
 */

const struct Curl_handler Curl_handler_ldap = {
  "LDAP",                               /* scheme */
  oldap_setup_connection,               /* setup_connection */
  oldap_do,                             /* do_it */
  oldap_done,                           /* done */
  ZERO_NULL,                            /* do_more */
  oldap_connect,                        /* connect_it */
  oldap_connecting,                     /* connecting */
  ZERO_NULL,                            /* doing */
  ZERO_NULL,                            /* proto_getsock */
  ZERO_NULL,                            /* doing_getsock */
  ZERO_NULL,                            /* domore_getsock */
  ZERO_NULL,                            /* perform_getsock */
  oldap_disconnect,                     /* disconnect */
  ZERO_NULL,                            /* readwrite */
  ZERO_NULL,                            /* connection_check */
  ZERO_NULL,                            /* attach connection */
  PORT_LDAP,                            /* defport */
  CURLPROTO_LDAP,                       /* protocol */
  CURLPROTO_LDAP,                       /* family */
  PROTOPT_NONE                          /* flags */
};

#ifdef USE_SSL
/*
 * LDAPS protocol handler.
 */

const struct Curl_handler Curl_handler_ldaps = {
  "LDAPS",                              /* scheme */
  oldap_setup_connection,               /* setup_connection */
  oldap_do,                             /* do_it */
  oldap_done,                           /* done */
  ZERO_NULL,                            /* do_more */
  oldap_connect,                        /* connect_it */
  oldap_connecting,                     /* connecting */
  ZERO_NULL,                            /* doing */
  ZERO_NULL,                            /* proto_getsock */
  ZERO_NULL,                            /* doing_getsock */
  ZERO_NULL,                            /* domore_getsock */
  ZERO_NULL,                            /* perform_getsock */
  oldap_disconnect,                     /* disconnect */
  ZERO_NULL,                            /* readwrite */
  ZERO_NULL,                            /* connection_check */
  ZERO_NULL,                            /* attach connection */
  PORT_LDAPS,                           /* defport */
  CURLPROTO_LDAPS,                      /* protocol */
  CURLPROTO_LDAP,                       /* family */
  PROTOPT_SSL                           /* flags */
};
#endif

static const char *url_errs[] = {
  "success",
  "out of memory",
  "bad parameter",
  "unrecognized scheme",
  "unbalanced delimiter",
  "bad URL",
  "bad host or port",
  "bad or missing attributes",
  "bad or missing scope",
  "bad or missing filter",
  "bad or missing extensions"
};

struct ldapconninfo {
  LDAP *ld;                  /* Openldap connection handle. */
  Curl_recv *recv;           /* For stacking SSL handler */
  Curl_send *send;
  ldapstate state;           /* Current machine state. */
  int proto;                 /* LDAP_PROTO_TCP/LDAP_PROTO_UDP/LDAP_PROTO_IPC */
  int msgid;                 /* Current message id. */
};

struct ldapreqinfo {
  int msgid;
  int nument;
};

/*
 * state()
 *
 * This is the ONLY way to change LDAP state!
 */
static void state(struct Curl_easy *data, ldapstate newstate)
{
  struct ldapconninfo *ldapc = data->conn->proto.ldapc;

#if defined(DEBUGBUILD) && !defined(CURL_DISABLE_VERBOSE_STRINGS)
  /* for debug purposes */
  static const char * const names[] = {
    "STOP",
    "SSL",
    "STARTTLS",
    "TLS",
    "BIND",
    "BINDV2",
    /* LAST */
  };

  if(ldapc->state != newstate)
    infof(data, "LDAP %p state change from %s to %s",
          (void *)ldapc, names[ldapc->state], names[newstate]);
#endif

  ldapc->state = newstate;
}

/* Map some particular LDAP error codes to CURLcode values. */
static CURLcode oldap_map_error(int rc, CURLcode result)
{
  switch(rc) {
  case LDAP_NO_MEMORY:
    result = CURLE_OUT_OF_MEMORY;
    break;
  case LDAP_INVALID_CREDENTIALS:
    result = CURLE_LOGIN_DENIED;
    break;
  case LDAP_PROTOCOL_ERROR:
    result = CURLE_UNSUPPORTED_PROTOCOL;
    break;
  case LDAP_INSUFFICIENT_ACCESS:
    result = CURLE_REMOTE_ACCESS_DENIED;
    break;
  }
  return result;
}

static CURLcode oldap_setup_connection(struct Curl_easy *data,
                                       struct connectdata *conn)
{
  struct ldapconninfo *li;
  LDAPURLDesc *lud;
  int rc, proto;
  CURLcode status;

  rc = ldap_url_parse(data->state.url, &lud);
  if(rc != LDAP_URL_SUCCESS) {
    const char *msg = "url parsing problem";
    status = CURLE_URL_MALFORMAT;
    if(rc > LDAP_URL_SUCCESS && rc <= LDAP_URL_ERR_BADEXTS) {
      if(rc == LDAP_URL_ERR_MEM)
        status = CURLE_OUT_OF_MEMORY;
      msg = url_errs[rc];
    }
    failf(data, "LDAP local: %s", msg);
    return status;
  }
  proto = ldap_pvt_url_scheme2proto(lud->lud_scheme);
  ldap_free_urldesc(lud);

  li = calloc(1, sizeof(struct ldapconninfo));
  if(!li)
    return CURLE_OUT_OF_MEMORY;
  li->proto = proto;
  conn->proto.ldapc = li;
  connkeep(conn, "OpenLDAP default");

  /* Clear the TLS upgraded flag */
  conn->bits.tls_upgraded = FALSE;

  return CURLE_OK;
}

/* Starts LDAP simple bind. */
static CURLcode oldap_perform_bind(struct Curl_easy *data, ldapstate newstate)
{
  CURLcode result = CURLE_OK;
  struct connectdata *conn = data->conn;
  struct ldapconninfo *li = conn->proto.ldapc;
  char *binddn = NULL;
  struct berval passwd;
  int rc;

  passwd.bv_val = NULL;
  passwd.bv_len = 0;

  if(conn->bits.user_passwd) {
    binddn = conn->user;
    passwd.bv_val = conn->passwd;
    passwd.bv_len = strlen(passwd.bv_val);
  }

  rc = ldap_sasl_bind(li->ld, binddn, LDAP_SASL_SIMPLE, &passwd,
                      NULL, NULL, &li->msgid);
  if(rc == LDAP_SUCCESS)
    state(data, newstate);
  else
    result = oldap_map_error(rc,
                             conn->bits.user_passwd?
                             CURLE_LOGIN_DENIED: CURLE_LDAP_CANNOT_BIND);
  return result;
}

#ifdef USE_SSL
static Sockbuf_IO ldapsb_tls;

static bool ssl_installed(struct connectdata *conn)
{
  return conn->proto.ldapc->recv != NULL;
}

static CURLcode oldap_ssl_connect(struct Curl_easy *data, ldapstate newstate)
{
  CURLcode result = CURLE_OK;
  struct connectdata *conn = data->conn;
  struct ldapconninfo *li = conn->proto.ldapc;
  bool ssldone = 0;

  result = Curl_ssl_connect_nonblocking(data, conn, FALSE,
                                        FIRSTSOCKET, &ssldone);
  if(!result) {
    state(data, newstate);

    if(ssldone) {
      Sockbuf *sb;

      /* Install the libcurl SSL handlers into the sockbuf. */
      ldap_get_option(li->ld, LDAP_OPT_SOCKBUF, &sb);
      ber_sockbuf_add_io(sb, &ldapsb_tls, LBER_SBIOD_LEVEL_TRANSPORT, data);
      li->recv = conn->recv[FIRSTSOCKET];
      li->send = conn->send[FIRSTSOCKET];
    }
  }

  return result;
}

/* Send the STARTTLS request */
static CURLcode oldap_perform_starttls(struct Curl_easy *data)
{
  CURLcode result = CURLE_OK;
  struct ldapconninfo *li = data->conn->proto.ldapc;
  int rc = ldap_start_tls(li->ld, NULL, NULL, &li->msgid);

  if(rc == LDAP_SUCCESS)
    state(data, OLDAP_STARTTLS);
  else
    result = oldap_map_error(rc, CURLE_USE_SSL_FAILED);
  return result;
}
#endif

static CURLcode oldap_connect(struct Curl_easy *data, bool *done)
{
  struct connectdata *conn = data->conn;
  struct ldapconninfo *li = conn->proto.ldapc;
  int rc, proto = LDAP_VERSION3;
  char hosturl[1024];
  char *ptr;
#ifdef CURL_OPENLDAP_DEBUG
  static int do_trace = -1;
#endif

  (void)done;

  strcpy(hosturl, "ldap");
  ptr = hosturl + 4;
  if(conn->handler->flags & PROTOPT_SSL)
    *ptr++ = 's';
  msnprintf(ptr, sizeof(hosturl)-(ptr-hosturl), "://%s:%d",
            conn->host.name, conn->remote_port);

  rc = ldap_init_fd(conn->sock[FIRSTSOCKET], li->proto, hosturl, &li->ld);
  if(rc) {
    failf(data, "LDAP local: Cannot connect to %s, %s",
          hosturl, ldap_err2string(rc));
    return CURLE_COULDNT_CONNECT;
  }

#ifdef CURL_OPENLDAP_DEBUG
  if(do_trace < 0) {
    const char *env = getenv("CURL_OPENLDAP_TRACE");
    do_trace = (env && strtol(env, NULL, 10) > 0);
  }
  if(do_trace)
    ldap_set_option(li->ld, LDAP_OPT_DEBUG_LEVEL, &do_trace);
#endif

  ldap_set_option(li->ld, LDAP_OPT_PROTOCOL_VERSION, &proto);

#ifdef USE_SSL
  if(conn->handler->flags & PROTOPT_SSL)
    return oldap_ssl_connect(data, OLDAP_SSL);

  if(data->set.use_ssl) {
    CURLcode result = oldap_perform_starttls(data);

    if(!result || data->set.use_ssl != CURLUSESSL_TRY)
      return result;
  }
#endif

  /* Force bind even if anonymous bind is not needed in protocol version 3
     to detect missing version 3 support. */
  return oldap_perform_bind(data, OLDAP_BIND);
}

/* Handle a simple bind response. */
static CURLcode oldap_state_bind_resp(struct Curl_easy *data, LDAPMessage *msg,
                                      int code)
{
  struct connectdata *conn = data->conn;
  struct ldapconninfo *li = conn->proto.ldapc;
  CURLcode result = CURLE_OK;
  struct berval *bv = NULL;
  int rc;

  if(code != LDAP_SUCCESS)
    return oldap_map_error(code, CURLE_LDAP_CANNOT_BIND);

  rc = ldap_parse_sasl_bind_result(li->ld, msg, &bv, 0);
  if(rc != LDAP_SUCCESS) {
    failf(data, "LDAP local: bind ldap_parse_sasl_bind_result %s",
          ldap_err2string(rc));
    result = oldap_map_error(rc, CURLE_LDAP_CANNOT_BIND);
  }
  else
    state(data, OLDAP_STOP);

  if(bv)
    ber_bvfree(bv);
  return result;
}

static CURLcode oldap_connecting(struct Curl_easy *data, bool *done)
{
  CURLcode result = CURLE_OK;
  struct connectdata *conn = data->conn;
  struct ldapconninfo *li = conn->proto.ldapc;
  LDAPMessage *msg = NULL;
  struct timeval tv = {0, 0};
  int code = LDAP_SUCCESS;
  int rc;

  if(li->state != OLDAP_SSL && li->state != OLDAP_TLS) {
    /* Get response to last command. */
    rc = ldap_result(li->ld, li->msgid, LDAP_MSG_ONE, &tv, &msg);
    if(!rc)
      return CURLE_OK;                    /* Timed out. */
    if(rc < 0) {
      failf(data, "LDAP local: connecting ldap_result %s",
            ldap_err2string(rc));
      return oldap_map_error(rc, CURLE_COULDNT_CONNECT);
    }

    /* Get error code from message. */
    rc = ldap_parse_result(li->ld, msg, &code, NULL, NULL, NULL, NULL, 0);
    if(rc)
      code = rc;

    /* If protocol version 3 is not supported, fallback to version 2. */
    if(code == LDAP_PROTOCOL_ERROR && li->state != OLDAP_BINDV2
#ifdef USE_SSL
       && (ssl_installed(conn) || data->set.use_ssl <= CURLUSESSL_TRY)
#endif
       ) {
      static const int version = LDAP_VERSION2;

      ldap_set_option(li->ld, LDAP_OPT_PROTOCOL_VERSION, &version);
      ldap_msgfree(msg);
      return oldap_perform_bind(data, OLDAP_BINDV2);
    }
  }

  /* Handle response message according to current state. */
  switch(li->state) {

#ifdef USE_SSL
  case OLDAP_SSL:
    result = oldap_ssl_connect(data, OLDAP_SSL);
    if(!result && ssl_installed(conn))
      result = oldap_perform_bind(data, OLDAP_BIND);
    break;
  case OLDAP_STARTTLS:
    if(code != LDAP_SUCCESS) {
      if(data->set.use_ssl != CURLUSESSL_TRY)
        result = oldap_map_error(code, CURLE_USE_SSL_FAILED);
      else
        result = oldap_perform_bind(data, OLDAP_BIND);
      break;
    }
    /* FALLTHROUGH */
  case OLDAP_TLS:
    result = oldap_ssl_connect(data, OLDAP_TLS);
    if(result && data->set.use_ssl != CURLUSESSL_TRY)
      result = oldap_map_error(code, CURLE_USE_SSL_FAILED);
    else if(ssl_installed(conn)) {
      conn->bits.tls_upgraded = TRUE;
      if(conn->bits.user_passwd)
        result = oldap_perform_bind(data, OLDAP_BIND);
      else {
        state(data, OLDAP_STOP); /* Version 3 supported: no bind required */
        result = CURLE_OK;
      }
    }
    break;
#endif

  case OLDAP_BIND:
  case OLDAP_BINDV2:
    result = oldap_state_bind_resp(data, msg, code);
    break;
  default:
    /* internal error */
    result = CURLE_COULDNT_CONNECT;
    break;
  }

  ldap_msgfree(msg);

  *done = li->state == OLDAP_STOP;
  if(*done)
    conn->recv[FIRSTSOCKET] = oldap_recv;

  return result;
}

static CURLcode oldap_disconnect(struct Curl_easy *data,
                                 struct connectdata *conn,
                                 bool dead_connection)
{
  struct ldapconninfo *li = conn->proto.ldapc;
  (void) dead_connection;

  if(li) {
    if(li->ld) {
#ifdef USE_SSL
      if(ssl_installed(conn)) {
        Sockbuf *sb;
        ldap_get_option(li->ld, LDAP_OPT_SOCKBUF, &sb);
        ber_sockbuf_add_io(sb, &ldapsb_tls, LBER_SBIOD_LEVEL_TRANSPORT, data);
      }
#endif
      ldap_unbind_ext(li->ld, NULL, NULL);
      li->ld = NULL;
    }
    conn->proto.ldapc = NULL;
    free(li);
  }
  return CURLE_OK;
}

static CURLcode oldap_do(struct Curl_easy *data, bool *done)
{
  struct connectdata *conn = data->conn;
  struct ldapconninfo *li = conn->proto.ldapc;
  struct ldapreqinfo *lr;
  CURLcode status = CURLE_OK;
  int rc = 0;
  LDAPURLDesc *ludp = NULL;
  int msgid;

  connkeep(conn, "OpenLDAP do");

  infof(data, "LDAP local: %s", data->state.url);

  rc = ldap_url_parse(data->state.url, &ludp);
  if(rc != LDAP_URL_SUCCESS) {
    const char *msg = "url parsing problem";
    status = CURLE_URL_MALFORMAT;
    if(rc > LDAP_URL_SUCCESS && rc <= LDAP_URL_ERR_BADEXTS) {
      if(rc == LDAP_URL_ERR_MEM)
        status = CURLE_OUT_OF_MEMORY;
      msg = url_errs[rc];
    }
    failf(data, "LDAP local: %s", msg);
    return status;
  }

  rc = ldap_search_ext(li->ld, ludp->lud_dn, ludp->lud_scope,
                       ludp->lud_filter, ludp->lud_attrs, 0,
                       NULL, NULL, NULL, 0, &msgid);
  ldap_free_urldesc(ludp);
  if(rc != LDAP_SUCCESS) {
    failf(data, "LDAP local: ldap_search_ext %s", ldap_err2string(rc));
    return CURLE_LDAP_SEARCH_FAILED;
  }
  lr = calloc(1, sizeof(struct ldapreqinfo));
  if(!lr)
    return CURLE_OUT_OF_MEMORY;
  lr->msgid = msgid;
  data->req.p.ldap = lr;
  Curl_setup_transfer(data, FIRSTSOCKET, -1, FALSE, -1);
  *done = TRUE;
  return CURLE_OK;
}

static CURLcode oldap_done(struct Curl_easy *data, CURLcode res,
                           bool premature)
{
  struct connectdata *conn = data->conn;
  struct ldapreqinfo *lr = data->req.p.ldap;

  (void)res;
  (void)premature;

  if(lr) {
    /* if there was a search in progress, abandon it */
    if(lr->msgid) {
      struct ldapconninfo *li = conn->proto.ldapc;
      ldap_abandon_ext(li->ld, lr->msgid, NULL, NULL);
      lr->msgid = 0;
    }
    data->req.p.ldap = NULL;
    free(lr);
  }

  return CURLE_OK;
}

static ssize_t oldap_recv(struct Curl_easy *data, int sockindex, char *buf,
                          size_t len, CURLcode *err)
{
  struct connectdata *conn = data->conn;
  struct ldapconninfo *li = conn->proto.ldapc;
  struct ldapreqinfo *lr = data->req.p.ldap;
  int rc, ret;
  LDAPMessage *msg = NULL;
  LDAPMessage *ent;
  BerElement *ber = NULL;
  struct timeval tv = {0, 1};

  (void)len;
  (void)buf;
  (void)sockindex;

  rc = ldap_result(li->ld, lr->msgid, LDAP_MSG_RECEIVED, &tv, &msg);
  if(rc < 0) {
    failf(data, "LDAP local: search ldap_result %s", ldap_err2string(rc));
    *err = CURLE_RECV_ERROR;
    return -1;
  }

  *err = CURLE_AGAIN;
  ret = -1;

  /* timed out */
  if(!msg)
    return ret;

  for(ent = ldap_first_message(li->ld, msg); ent;
      ent = ldap_next_message(li->ld, ent)) {
    struct berval bv, *bvals;
    int binary = 0, msgtype;
    CURLcode writeerr;

    msgtype = ldap_msgtype(ent);
    if(msgtype == LDAP_RES_SEARCH_RESULT) {
      int code;
      char *info = NULL;
      rc = ldap_parse_result(li->ld, ent, &code, NULL, &info, NULL, NULL, 0);
      if(rc) {
        failf(data, "LDAP local: search ldap_parse_result %s",
              ldap_err2string(rc));
        *err = CURLE_LDAP_SEARCH_FAILED;
      }
      else if(code && code != LDAP_SIZELIMIT_EXCEEDED) {
        failf(data, "LDAP remote: search failed %s %s", ldap_err2string(rc),
              info ? info : "");
        *err = CURLE_LDAP_SEARCH_FAILED;
      }
      else {
        /* successful */
        if(code == LDAP_SIZELIMIT_EXCEEDED)
          infof(data, "There are more than %d entries", lr->nument);
        data->req.size = data->req.bytecount;
        *err = CURLE_OK;
        ret = 0;
      }
      lr->msgid = 0;
      ldap_memfree(info);
      break;
    }
    else if(msgtype != LDAP_RES_SEARCH_ENTRY)
      continue;

    lr->nument++;
    rc = ldap_get_dn_ber(li->ld, ent, &ber, &bv);
    if(rc < 0) {
      *err = CURLE_RECV_ERROR;
      return -1;
    }
    writeerr = Curl_client_write(data, CLIENTWRITE_BODY, (char *)"DN: ", 4);
    if(writeerr) {
      *err = writeerr;
      return -1;
    }

    writeerr = Curl_client_write(data, CLIENTWRITE_BODY, (char *)bv.bv_val,
                                 bv.bv_len);
    if(writeerr) {
      *err = writeerr;
      return -1;
    }

    writeerr = Curl_client_write(data, CLIENTWRITE_BODY, (char *)"\n", 1);
    if(writeerr) {
      *err = writeerr;
      return -1;
    }
    data->req.bytecount += bv.bv_len + 5;

    for(rc = ldap_get_attribute_ber(li->ld, ent, ber, &bv, &bvals);
        rc == LDAP_SUCCESS;
        rc = ldap_get_attribute_ber(li->ld, ent, ber, &bv, &bvals)) {
      int i;

      if(!bv.bv_val)
        break;

      if(bv.bv_len > 7 && !strncmp(bv.bv_val + bv.bv_len - 7, ";binary", 7))
        binary = 1;
      else
        binary = 0;

      if(!bvals) {
        writeerr = Curl_client_write(data, CLIENTWRITE_BODY, (char *)"\t", 1);
        if(writeerr) {
          *err = writeerr;
          return -1;
        }
        writeerr = Curl_client_write(data, CLIENTWRITE_BODY, (char *)bv.bv_val,
                                     bv.bv_len);
        if(writeerr) {
          *err = writeerr;
          return -1;
        }
        writeerr = Curl_client_write(data, CLIENTWRITE_BODY, (char *)":\n", 2);
        if(writeerr) {
          *err = writeerr;
          return -1;
        }
        data->req.bytecount += bv.bv_len + 3;
        continue;
      }

      for(i = 0; bvals[i].bv_val != NULL; i++) {
        int binval = 0;
        writeerr = Curl_client_write(data, CLIENTWRITE_BODY, (char *)"\t", 1);
        if(writeerr) {
          *err = writeerr;
          return -1;
        }

        writeerr = Curl_client_write(data, CLIENTWRITE_BODY, (char *)bv.bv_val,
                                     bv.bv_len);
        if(writeerr) {
          *err = writeerr;
          return -1;
        }

        writeerr = Curl_client_write(data, CLIENTWRITE_BODY, (char *)":", 1);
        if(writeerr) {
          *err = writeerr;
          return -1;
        }
        data->req.bytecount += bv.bv_len + 2;

        if(!binary) {
          /* check for leading or trailing whitespace */
          if(ISSPACE(bvals[i].bv_val[0]) ||
             ISSPACE(bvals[i].bv_val[bvals[i].bv_len-1]))
            binval = 1;
          else {
            /* check for unprintable characters */
            unsigned int j;
            for(j = 0; j<bvals[i].bv_len; j++)
              if(!ISPRINT(bvals[i].bv_val[j])) {
                binval = 1;
                break;
              }
          }
        }
        if(binary || binval) {
          char *val_b64 = NULL;
          size_t val_b64_sz = 0;
          /* Binary value, encode to base64. */
          CURLcode error = Curl_base64_encode(data,
                                              bvals[i].bv_val,
                                              bvals[i].bv_len,
                                              &val_b64,
                                              &val_b64_sz);
          if(error) {
            ber_memfree(bvals);
            ber_free(ber, 0);
            ldap_msgfree(msg);
            *err = error;
            return -1;
          }
          writeerr = Curl_client_write(data, CLIENTWRITE_BODY,
                                       (char *)": ", 2);
          if(writeerr) {
            *err = writeerr;
            return -1;
          }

          data->req.bytecount += 2;
          if(val_b64_sz > 0) {
            writeerr = Curl_client_write(data, CLIENTWRITE_BODY, val_b64,
                                         val_b64_sz);
            if(writeerr) {
              *err = writeerr;
              return -1;
            }
            free(val_b64);
            data->req.bytecount += val_b64_sz;
          }
        }
        else {
          writeerr = Curl_client_write(data, CLIENTWRITE_BODY, (char *)" ", 1);
          if(writeerr) {
            *err = writeerr;
            return -1;
          }

          writeerr = Curl_client_write(data, CLIENTWRITE_BODY, bvals[i].bv_val,
                                       bvals[i].bv_len);
          if(writeerr) {
            *err = writeerr;
            return -1;
          }

          data->req.bytecount += bvals[i].bv_len + 1;
        }
        writeerr = Curl_client_write(data, CLIENTWRITE_BODY, (char *)"\n", 1);
        if(writeerr) {
          *err = writeerr;
          return -1;
        }

        data->req.bytecount++;
      }
      ber_memfree(bvals);
      writeerr = Curl_client_write(data, CLIENTWRITE_BODY, (char *)"\n", 1);
      if(writeerr) {
        *err = writeerr;
        return -1;
      }
      data->req.bytecount++;
    }
    writeerr = Curl_client_write(data, CLIENTWRITE_BODY, (char *)"\n", 1);
    if(writeerr) {
      *err = writeerr;
      return -1;
    }
    data->req.bytecount++;
    ber_free(ber, 0);
  }
  ldap_msgfree(msg);
  return ret;
}

#ifdef USE_SSL
static int
ldapsb_tls_setup(Sockbuf_IO_Desc *sbiod, void *arg)
{
  sbiod->sbiod_pvt = arg;
  return 0;
}

static int
ldapsb_tls_remove(Sockbuf_IO_Desc *sbiod)
{
  sbiod->sbiod_pvt = NULL;
  return 0;
}

/* We don't need to do anything because libcurl does it already */
static int
ldapsb_tls_close(Sockbuf_IO_Desc *sbiod)
{
  (void)sbiod;
  return 0;
}

static int
ldapsb_tls_ctrl(Sockbuf_IO_Desc *sbiod, int opt, void *arg)
{
  (void)arg;
  if(opt == LBER_SB_OPT_DATA_READY) {
    struct Curl_easy *data = sbiod->sbiod_pvt;
    return Curl_ssl_data_pending(data->conn, FIRSTSOCKET);
  }
  return 0;
}

static ber_slen_t
ldapsb_tls_read(Sockbuf_IO_Desc *sbiod, void *buf, ber_len_t len)
{
  struct Curl_easy *data = sbiod->sbiod_pvt;
  ber_slen_t ret = 0;
  if(data) {
    struct connectdata *conn = data->conn;
    if(conn) {
      struct ldapconninfo *li = conn->proto.ldapc;
      CURLcode err = CURLE_RECV_ERROR;

      ret = (li->recv)(data, FIRSTSOCKET, buf, len, &err);
      if(ret < 0 && err == CURLE_AGAIN) {
        SET_SOCKERRNO(EWOULDBLOCK);
      }
    }
  }
  return ret;
}

static ber_slen_t
ldapsb_tls_write(Sockbuf_IO_Desc *sbiod, void *buf, ber_len_t len)
{
  struct Curl_easy *data = sbiod->sbiod_pvt;
  ber_slen_t ret = 0;
  if(data) {
    struct connectdata *conn = data->conn;
    if(conn) {
      struct ldapconninfo *li = conn->proto.ldapc;
      CURLcode err = CURLE_SEND_ERROR;
      ret = (li->send)(data, FIRSTSOCKET, buf, len, &err);
      if(ret < 0 && err == CURLE_AGAIN) {
        SET_SOCKERRNO(EWOULDBLOCK);
      }
    }
  }
  return ret;
}

static Sockbuf_IO ldapsb_tls =
{
  ldapsb_tls_setup,
  ldapsb_tls_remove,
  ldapsb_tls_ctrl,
  ldapsb_tls_read,
  ldapsb_tls_write,
  ldapsb_tls_close
};
#endif /* USE_SSL */

#endif /* !CURL_DISABLE_LDAP && USE_OPENLDAP */
