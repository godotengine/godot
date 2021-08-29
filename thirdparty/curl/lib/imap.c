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
 * RFC2195 CRAM-MD5 authentication
 * RFC2595 Using TLS with IMAP, POP3 and ACAP
 * RFC2831 DIGEST-MD5 authentication
 * RFC3501 IMAPv4 protocol
 * RFC4422 Simple Authentication and Security Layer (SASL)
 * RFC4616 PLAIN authentication
 * RFC4752 The Kerberos V5 ("GSSAPI") SASL Mechanism
 * RFC4959 IMAP Extension for SASL Initial Client Response
 * RFC5092 IMAP URL Scheme
 * RFC6749 OAuth 2.0 Authorization Framework
 * RFC8314 Use of TLS for Email Submission and Access
 * Draft   LOGIN SASL Mechanism <draft-murchison-sasl-login-00.txt>
 *
 ***************************************************************************/

#include "curl_setup.h"

#ifndef CURL_DISABLE_IMAP

#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif
#ifdef HAVE_ARPA_INET_H
#include <arpa/inet.h>
#endif
#ifdef HAVE_UTSNAME_H
#include <sys/utsname.h>
#endif
#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif
#ifdef __VMS
#include <in.h>
#include <inet.h>
#endif

#if (defined(NETWARE) && defined(__NOVELL_LIBC__))
#undef in_addr_t
#define in_addr_t unsigned long
#endif

#include <curl/curl.h>
#include "urldata.h"
#include "sendf.h"
#include "hostip.h"
#include "progress.h"
#include "transfer.h"
#include "escape.h"
#include "http.h" /* for HTTP proxy tunnel stuff */
#include "socks.h"
#include "imap.h"
#include "mime.h"
#include "strtoofft.h"
#include "strcase.h"
#include "vtls/vtls.h"
#include "connect.h"
#include "select.h"
#include "multiif.h"
#include "url.h"
#include "strcase.h"
#include "bufref.h"
#include "curl_sasl.h"
#include "warnless.h"

/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"

/* Local API functions */
static CURLcode imap_regular_transfer(struct Curl_easy *data, bool *done);
static CURLcode imap_do(struct Curl_easy *data, bool *done);
static CURLcode imap_done(struct Curl_easy *data, CURLcode status,
                          bool premature);
static CURLcode imap_connect(struct Curl_easy *data, bool *done);
static CURLcode imap_disconnect(struct Curl_easy *data,
                                struct connectdata *conn, bool dead);
static CURLcode imap_multi_statemach(struct Curl_easy *data, bool *done);
static int imap_getsock(struct Curl_easy *data, struct connectdata *conn,
                        curl_socket_t *socks);
static CURLcode imap_doing(struct Curl_easy *data, bool *dophase_done);
static CURLcode imap_setup_connection(struct Curl_easy *data,
                                      struct connectdata *conn);
static char *imap_atom(const char *str, bool escape_only);
static CURLcode imap_sendf(struct Curl_easy *data, const char *fmt, ...);
static CURLcode imap_parse_url_options(struct connectdata *conn);
static CURLcode imap_parse_url_path(struct Curl_easy *data);
static CURLcode imap_parse_custom_request(struct Curl_easy *data);
static CURLcode imap_perform_authenticate(struct Curl_easy *data,
                                          const char *mech,
                                          const struct bufref *initresp);
static CURLcode imap_continue_authenticate(struct Curl_easy *data,
                                           const char *mech,
                                           const struct bufref *resp);
static CURLcode imap_cancel_authenticate(struct Curl_easy *data,
                                         const char *mech);
static CURLcode imap_get_message(struct Curl_easy *data, struct bufref *out);

/*
 * IMAP protocol handler.
 */

const struct Curl_handler Curl_handler_imap = {
  "IMAP",                           /* scheme */
  imap_setup_connection,            /* setup_connection */
  imap_do,                          /* do_it */
  imap_done,                        /* done */
  ZERO_NULL,                        /* do_more */
  imap_connect,                     /* connect_it */
  imap_multi_statemach,             /* connecting */
  imap_doing,                       /* doing */
  imap_getsock,                     /* proto_getsock */
  imap_getsock,                     /* doing_getsock */
  ZERO_NULL,                        /* domore_getsock */
  ZERO_NULL,                        /* perform_getsock */
  imap_disconnect,                  /* disconnect */
  ZERO_NULL,                        /* readwrite */
  ZERO_NULL,                        /* connection_check */
  ZERO_NULL,                        /* attach connection */
  PORT_IMAP,                        /* defport */
  CURLPROTO_IMAP,                   /* protocol */
  CURLPROTO_IMAP,                   /* family */
  PROTOPT_CLOSEACTION|              /* flags */
  PROTOPT_URLOPTIONS
};

#ifdef USE_SSL
/*
 * IMAPS protocol handler.
 */

const struct Curl_handler Curl_handler_imaps = {
  "IMAPS",                          /* scheme */
  imap_setup_connection,            /* setup_connection */
  imap_do,                          /* do_it */
  imap_done,                        /* done */
  ZERO_NULL,                        /* do_more */
  imap_connect,                     /* connect_it */
  imap_multi_statemach,             /* connecting */
  imap_doing,                       /* doing */
  imap_getsock,                     /* proto_getsock */
  imap_getsock,                     /* doing_getsock */
  ZERO_NULL,                        /* domore_getsock */
  ZERO_NULL,                        /* perform_getsock */
  imap_disconnect,                  /* disconnect */
  ZERO_NULL,                        /* readwrite */
  ZERO_NULL,                        /* connection_check */
  ZERO_NULL,                        /* attach connection */
  PORT_IMAPS,                       /* defport */
  CURLPROTO_IMAPS,                  /* protocol */
  CURLPROTO_IMAP,                   /* family */
  PROTOPT_CLOSEACTION | PROTOPT_SSL | /* flags */
  PROTOPT_URLOPTIONS
};
#endif

#define IMAP_RESP_OK       1
#define IMAP_RESP_NOT_OK   2
#define IMAP_RESP_PREAUTH  3

/* SASL parameters for the imap protocol */
static const struct SASLproto saslimap = {
  "imap",                     /* The service name */
  imap_perform_authenticate,  /* Send authentication command */
  imap_continue_authenticate, /* Send authentication continuation */
  imap_cancel_authenticate,   /* Send authentication cancellation */
  imap_get_message,           /* Get SASL response message */
  0,                          /* No maximum initial response length */
  '+',                        /* Code received when continuation is expected */
  IMAP_RESP_OK,               /* Code to receive upon authentication success */
  SASL_AUTH_DEFAULT,          /* Default mechanisms */
  SASL_FLAG_BASE64            /* Configuration flags */
};


#ifdef USE_SSL
static void imap_to_imaps(struct connectdata *conn)
{
  /* Change the connection handler */
  conn->handler = &Curl_handler_imaps;

  /* Set the connection's upgraded to TLS flag */
  conn->bits.tls_upgraded = TRUE;
}
#else
#define imap_to_imaps(x) Curl_nop_stmt
#endif

/***********************************************************************
 *
 * imap_matchresp()
 *
 * Determines whether the untagged response is related to the specified
 * command by checking if it is in format "* <command-name> ..." or
 * "* <number> <command-name> ...".
 *
 * The "* " marker is assumed to have already been checked by the caller.
 */
static bool imap_matchresp(const char *line, size_t len, const char *cmd)
{
  const char *end = line + len;
  size_t cmd_len = strlen(cmd);

  /* Skip the untagged response marker */
  line += 2;

  /* Do we have a number after the marker? */
  if(line < end && ISDIGIT(*line)) {
    /* Skip the number */
    do
      line++;
    while(line < end && ISDIGIT(*line));

    /* Do we have the space character? */
    if(line == end || *line != ' ')
      return FALSE;

    line++;
  }

  /* Does the command name match and is it followed by a space character or at
     the end of line? */
  if(line + cmd_len <= end && strncasecompare(line, cmd, cmd_len) &&
     (line[cmd_len] == ' ' || line + cmd_len + 2 == end))
    return TRUE;

  return FALSE;
}

/***********************************************************************
 *
 * imap_endofresp()
 *
 * Checks whether the given string is a valid tagged, untagged or continuation
 * response which can be processed by the response handler.
 */
static bool imap_endofresp(struct Curl_easy *data, struct connectdata *conn,
                           char *line, size_t len, int *resp)
{
  struct IMAP *imap = data->req.p.imap;
  struct imap_conn *imapc = &conn->proto.imapc;
  const char *id = imapc->resptag;
  size_t id_len = strlen(id);

  /* Do we have a tagged command response? */
  if(len >= id_len + 1 && !memcmp(id, line, id_len) && line[id_len] == ' ') {
    line += id_len + 1;
    len -= id_len + 1;

    if(len >= 2 && !memcmp(line, "OK", 2))
      *resp = IMAP_RESP_OK;
    else if(len >= 7 && !memcmp(line, "PREAUTH", 7))
      *resp = IMAP_RESP_PREAUTH;
    else
      *resp = IMAP_RESP_NOT_OK;

    return TRUE;
  }

  /* Do we have an untagged command response? */
  if(len >= 2 && !memcmp("* ", line, 2)) {
    switch(imapc->state) {
      /* States which are interested in untagged responses */
      case IMAP_CAPABILITY:
        if(!imap_matchresp(line, len, "CAPABILITY"))
          return FALSE;
        break;

      case IMAP_LIST:
        if((!imap->custom && !imap_matchresp(line, len, "LIST")) ||
          (imap->custom && !imap_matchresp(line, len, imap->custom) &&
           (!strcasecompare(imap->custom, "STORE") ||
            !imap_matchresp(line, len, "FETCH")) &&
           !strcasecompare(imap->custom, "SELECT") &&
           !strcasecompare(imap->custom, "EXAMINE") &&
           !strcasecompare(imap->custom, "SEARCH") &&
           !strcasecompare(imap->custom, "EXPUNGE") &&
           !strcasecompare(imap->custom, "LSUB") &&
           !strcasecompare(imap->custom, "UID") &&
           !strcasecompare(imap->custom, "GETQUOTAROOT") &&
           !strcasecompare(imap->custom, "NOOP")))
          return FALSE;
        break;

      case IMAP_SELECT:
        /* SELECT is special in that its untagged responses do not have a
           common prefix so accept anything! */
        break;

      case IMAP_FETCH:
        if(!imap_matchresp(line, len, "FETCH"))
          return FALSE;
        break;

      case IMAP_SEARCH:
        if(!imap_matchresp(line, len, "SEARCH"))
          return FALSE;
        break;

      /* Ignore other untagged responses */
      default:
        return FALSE;
    }

    *resp = '*';
    return TRUE;
  }

  /* Do we have a continuation response? This should be a + symbol followed by
     a space and optionally some text as per RFC-3501 for the AUTHENTICATE and
     APPEND commands and as outlined in Section 4. Examples of RFC-4959 but
     some e-mail servers ignore this and only send a single + instead. */
  if(imap && !imap->custom && ((len == 3 && line[0] == '+') ||
     (len >= 2 && !memcmp("+ ", line, 2)))) {
    switch(imapc->state) {
      /* States which are interested in continuation responses */
      case IMAP_AUTHENTICATE:
      case IMAP_APPEND:
        *resp = '+';
        break;

      default:
        failf(data, "Unexpected continuation response");
        *resp = -1;
        break;
    }

    return TRUE;
  }

  return FALSE; /* Nothing for us */
}

/***********************************************************************
 *
 * imap_get_message()
 *
 * Gets the authentication message from the response buffer.
 */
static CURLcode imap_get_message(struct Curl_easy *data, struct bufref *out)
{
  char *message = data->state.buffer;
  size_t len = strlen(message);

  if(len > 2) {
    /* Find the start of the message */
    len -= 2;
    for(message += 2; *message == ' ' || *message == '\t'; message++, len--)
      ;

    /* Find the end of the message */
    while(len--)
      if(message[len] != '\r' && message[len] != '\n' && message[len] != ' ' &&
         message[len] != '\t')
        break;

    /* Terminate the message */
    message[++len] = '\0';
    Curl_bufref_set(out, message, len, NULL);
  }
  else
    /* junk input => zero length output */
    Curl_bufref_set(out, "", 0, NULL);

  return CURLE_OK;
}

/***********************************************************************
 *
 * state()
 *
 * This is the ONLY way to change IMAP state!
 */
static void state(struct Curl_easy *data, imapstate newstate)
{
  struct imap_conn *imapc = &data->conn->proto.imapc;
#if defined(DEBUGBUILD) && !defined(CURL_DISABLE_VERBOSE_STRINGS)
  /* for debug purposes */
  static const char * const names[]={
    "STOP",
    "SERVERGREET",
    "CAPABILITY",
    "STARTTLS",
    "UPGRADETLS",
    "AUTHENTICATE",
    "LOGIN",
    "LIST",
    "SELECT",
    "FETCH",
    "FETCH_FINAL",
    "APPEND",
    "APPEND_FINAL",
    "SEARCH",
    "LOGOUT",
    /* LAST */
  };

  if(imapc->state != newstate)
    infof(data, "IMAP %p state change from %s to %s",
          (void *)imapc, names[imapc->state], names[newstate]);
#endif

  imapc->state = newstate;
}

/***********************************************************************
 *
 * imap_perform_capability()
 *
 * Sends the CAPABILITY command in order to obtain a list of server side
 * supported capabilities.
 */
static CURLcode imap_perform_capability(struct Curl_easy *data,
                                        struct connectdata *conn)
{
  CURLcode result = CURLE_OK;
  struct imap_conn *imapc = &conn->proto.imapc;
  imapc->sasl.authmechs = SASL_AUTH_NONE; /* No known auth. mechanisms yet */
  imapc->sasl.authused = SASL_AUTH_NONE;  /* Clear the auth. mechanism used */
  imapc->tls_supported = FALSE;           /* Clear the TLS capability */

  /* Send the CAPABILITY command */
  result = imap_sendf(data, "CAPABILITY");

  if(!result)
    state(data, IMAP_CAPABILITY);

  return result;
}

/***********************************************************************
 *
 * imap_perform_starttls()
 *
 * Sends the STARTTLS command to start the upgrade to TLS.
 */
static CURLcode imap_perform_starttls(struct Curl_easy *data)
{
  /* Send the STARTTLS command */
  CURLcode result = imap_sendf(data, "STARTTLS");

  if(!result)
    state(data, IMAP_STARTTLS);

  return result;
}

/***********************************************************************
 *
 * imap_perform_upgrade_tls()
 *
 * Performs the upgrade to TLS.
 */
static CURLcode imap_perform_upgrade_tls(struct Curl_easy *data,
                                         struct connectdata *conn)
{
  /* Start the SSL connection */
  struct imap_conn *imapc = &conn->proto.imapc;
  CURLcode result = Curl_ssl_connect_nonblocking(data, conn, FALSE,
                                                 FIRSTSOCKET, &imapc->ssldone);

  if(!result) {
    if(imapc->state != IMAP_UPGRADETLS)
      state(data, IMAP_UPGRADETLS);

    if(imapc->ssldone) {
      imap_to_imaps(conn);
      result = imap_perform_capability(data, conn);
    }
  }

  return result;
}

/***********************************************************************
 *
 * imap_perform_login()
 *
 * Sends a clear text LOGIN command to authenticate with.
 */
static CURLcode imap_perform_login(struct Curl_easy *data,
                                   struct connectdata *conn)
{
  CURLcode result = CURLE_OK;
  char *user;
  char *passwd;

  /* Check we have a username and password to authenticate with and end the
     connect phase if we don't */
  if(!conn->bits.user_passwd) {
    state(data, IMAP_STOP);

    return result;
  }

  /* Make sure the username and password are in the correct atom format */
  user = imap_atom(conn->user, false);
  passwd = imap_atom(conn->passwd, false);

  /* Send the LOGIN command */
  result = imap_sendf(data, "LOGIN %s %s", user ? user : "",
                      passwd ? passwd : "");

  free(user);
  free(passwd);

  if(!result)
    state(data, IMAP_LOGIN);

  return result;
}

/***********************************************************************
 *
 * imap_perform_authenticate()
 *
 * Sends an AUTHENTICATE command allowing the client to login with the given
 * SASL authentication mechanism.
 */
static CURLcode imap_perform_authenticate(struct Curl_easy *data,
                                          const char *mech,
                                          const struct bufref *initresp)
{
  CURLcode result = CURLE_OK;
  const char *ir = (const char *) Curl_bufref_ptr(initresp);

  if(ir) {
    /* Send the AUTHENTICATE command with the initial response */
    result = imap_sendf(data, "AUTHENTICATE %s %s", mech, ir);
  }
  else {
    /* Send the AUTHENTICATE command */
    result = imap_sendf(data, "AUTHENTICATE %s", mech);
  }

  return result;
}

/***********************************************************************
 *
 * imap_continue_authenticate()
 *
 * Sends SASL continuation data.
 */
static CURLcode imap_continue_authenticate(struct Curl_easy *data,
                                           const char *mech,
                                           const struct bufref *resp)
{
  struct imap_conn *imapc = &data->conn->proto.imapc;

  (void)mech;

  return Curl_pp_sendf(data, &imapc->pp,
                       "%s", (const char *) Curl_bufref_ptr(resp));
}

/***********************************************************************
 *
 * imap_cancel_authenticate()
 *
 * Sends SASL cancellation.
 */
static CURLcode imap_cancel_authenticate(struct Curl_easy *data,
                                         const char *mech)
{
  struct imap_conn *imapc = &data->conn->proto.imapc;

  (void)mech;

  return Curl_pp_sendf(data, &imapc->pp, "*");
}

/***********************************************************************
 *
 * imap_perform_authentication()
 *
 * Initiates the authentication sequence, with the appropriate SASL
 * authentication mechanism, falling back to clear text should a common
 * mechanism not be available between the client and server.
 */
static CURLcode imap_perform_authentication(struct Curl_easy *data,
                                            struct connectdata *conn)
{
  CURLcode result = CURLE_OK;
  struct imap_conn *imapc = &conn->proto.imapc;
  saslprogress progress;

  /* Check if already authenticated OR if there is enough data to authenticate
     with and end the connect phase if we don't */
  if(imapc->preauth ||
     !Curl_sasl_can_authenticate(&imapc->sasl, conn)) {
    state(data, IMAP_STOP);
    return result;
  }

  /* Calculate the SASL login details */
  result = Curl_sasl_start(&imapc->sasl, data, imapc->ir_supported, &progress);

  if(!result) {
    if(progress == SASL_INPROGRESS)
      state(data, IMAP_AUTHENTICATE);
    else if(!imapc->login_disabled && (imapc->preftype & IMAP_TYPE_CLEARTEXT))
      /* Perform clear text authentication */
      result = imap_perform_login(data, conn);
    else {
      /* Other mechanisms not supported */
      infof(data, "No known authentication mechanisms supported!");
      result = CURLE_LOGIN_DENIED;
    }
  }

  return result;
}

/***********************************************************************
 *
 * imap_perform_list()
 *
 * Sends a LIST command or an alternative custom request.
 */
static CURLcode imap_perform_list(struct Curl_easy *data)
{
  CURLcode result = CURLE_OK;
  struct IMAP *imap = data->req.p.imap;

  if(imap->custom)
    /* Send the custom request */
    result = imap_sendf(data, "%s%s", imap->custom,
                        imap->custom_params ? imap->custom_params : "");
  else {
    /* Make sure the mailbox is in the correct atom format if necessary */
    char *mailbox = imap->mailbox ? imap_atom(imap->mailbox, true)
                                  : strdup("");
    if(!mailbox)
      return CURLE_OUT_OF_MEMORY;

    /* Send the LIST command */
    result = imap_sendf(data, "LIST \"%s\" *", mailbox);

    free(mailbox);
  }

  if(!result)
    state(data, IMAP_LIST);

  return result;
}

/***********************************************************************
 *
 * imap_perform_select()
 *
 * Sends a SELECT command to ask the server to change the selected mailbox.
 */
static CURLcode imap_perform_select(struct Curl_easy *data)
{
  CURLcode result = CURLE_OK;
  struct connectdata *conn = data->conn;
  struct IMAP *imap = data->req.p.imap;
  struct imap_conn *imapc = &conn->proto.imapc;
  char *mailbox;

  /* Invalidate old information as we are switching mailboxes */
  Curl_safefree(imapc->mailbox);
  Curl_safefree(imapc->mailbox_uidvalidity);

  /* Check we have a mailbox */
  if(!imap->mailbox) {
    failf(data, "Cannot SELECT without a mailbox.");
    return CURLE_URL_MALFORMAT;
  }

  /* Make sure the mailbox is in the correct atom format */
  mailbox = imap_atom(imap->mailbox, false);
  if(!mailbox)
    return CURLE_OUT_OF_MEMORY;

  /* Send the SELECT command */
  result = imap_sendf(data, "SELECT %s", mailbox);

  free(mailbox);

  if(!result)
    state(data, IMAP_SELECT);

  return result;
}

/***********************************************************************
 *
 * imap_perform_fetch()
 *
 * Sends a FETCH command to initiate the download of a message.
 */
static CURLcode imap_perform_fetch(struct Curl_easy *data)
{
  CURLcode result = CURLE_OK;
  struct IMAP *imap = data->req.p.imap;
  /* Check we have a UID */
  if(imap->uid) {

    /* Send the FETCH command */
    if(imap->partial)
      result = imap_sendf(data, "UID FETCH %s BODY[%s]<%s>",
                          imap->uid, imap->section ? imap->section : "",
                          imap->partial);
    else
      result = imap_sendf(data, "UID FETCH %s BODY[%s]",
                          imap->uid, imap->section ? imap->section : "");
  }
  else if(imap->mindex) {
    /* Send the FETCH command */
    if(imap->partial)
      result = imap_sendf(data, "FETCH %s BODY[%s]<%s>",
                          imap->mindex, imap->section ? imap->section : "",
                          imap->partial);
    else
      result = imap_sendf(data, "FETCH %s BODY[%s]",
                          imap->mindex, imap->section ? imap->section : "");
  }
  else {
    failf(data, "Cannot FETCH without a UID.");
    return CURLE_URL_MALFORMAT;
  }
  if(!result)
    state(data, IMAP_FETCH);

  return result;
}

/***********************************************************************
 *
 * imap_perform_append()
 *
 * Sends an APPEND command to initiate the upload of a message.
 */
static CURLcode imap_perform_append(struct Curl_easy *data)
{
  CURLcode result = CURLE_OK;
  struct IMAP *imap = data->req.p.imap;
  char *mailbox;

  /* Check we have a mailbox */
  if(!imap->mailbox) {
    failf(data, "Cannot APPEND without a mailbox.");
    return CURLE_URL_MALFORMAT;
  }

  /* Prepare the mime data if some. */
  if(data->set.mimepost.kind != MIMEKIND_NONE) {
    /* Use the whole structure as data. */
    data->set.mimepost.flags &= ~MIME_BODY_ONLY;

    /* Add external headers and mime version. */
    curl_mime_headers(&data->set.mimepost, data->set.headers, 0);
    result = Curl_mime_prepare_headers(&data->set.mimepost, NULL,
                                       NULL, MIMESTRATEGY_MAIL);

    if(!result)
      if(!Curl_checkheaders(data, "Mime-Version"))
        result = Curl_mime_add_header(&data->set.mimepost.curlheaders,
                                      "Mime-Version: 1.0");

    /* Make sure we will read the entire mime structure. */
    if(!result)
      result = Curl_mime_rewind(&data->set.mimepost);

    if(result)
      return result;

    data->state.infilesize = Curl_mime_size(&data->set.mimepost);

    /* Read from mime structure. */
    data->state.fread_func = (curl_read_callback) Curl_mime_read;
    data->state.in = (void *) &data->set.mimepost;
  }

  /* Check we know the size of the upload */
  if(data->state.infilesize < 0) {
    failf(data, "Cannot APPEND with unknown input file size");
    return CURLE_UPLOAD_FAILED;
  }

  /* Make sure the mailbox is in the correct atom format */
  mailbox = imap_atom(imap->mailbox, false);
  if(!mailbox)
    return CURLE_OUT_OF_MEMORY;

  /* Send the APPEND command */
  result = imap_sendf(data,
                      "APPEND %s (\\Seen) {%" CURL_FORMAT_CURL_OFF_T "}",
                      mailbox, data->state.infilesize);

  free(mailbox);

  if(!result)
    state(data, IMAP_APPEND);

  return result;
}

/***********************************************************************
 *
 * imap_perform_search()
 *
 * Sends a SEARCH command.
 */
static CURLcode imap_perform_search(struct Curl_easy *data)
{
  CURLcode result = CURLE_OK;
  struct IMAP *imap = data->req.p.imap;

  /* Check we have a query string */
  if(!imap->query) {
    failf(data, "Cannot SEARCH without a query string.");
    return CURLE_URL_MALFORMAT;
  }

  /* Send the SEARCH command */
  result = imap_sendf(data, "SEARCH %s", imap->query);

  if(!result)
    state(data, IMAP_SEARCH);

  return result;
}

/***********************************************************************
 *
 * imap_perform_logout()
 *
 * Performs the logout action prior to sclose() being called.
 */
static CURLcode imap_perform_logout(struct Curl_easy *data)
{
  /* Send the LOGOUT command */
  CURLcode result = imap_sendf(data, "LOGOUT");

  if(!result)
    state(data, IMAP_LOGOUT);

  return result;
}

/* For the initial server greeting */
static CURLcode imap_state_servergreet_resp(struct Curl_easy *data,
                                            int imapcode,
                                            imapstate instate)
{
  struct connectdata *conn = data->conn;
  (void)instate; /* no use for this yet */

  if(imapcode == IMAP_RESP_PREAUTH) {
    /* PREAUTH */
    struct imap_conn *imapc = &conn->proto.imapc;
    imapc->preauth = TRUE;
    infof(data, "PREAUTH connection, already authenticated!");
  }
  else if(imapcode != IMAP_RESP_OK) {
    failf(data, "Got unexpected imap-server response");
    return CURLE_WEIRD_SERVER_REPLY;
  }

  return imap_perform_capability(data, conn);
}

/* For CAPABILITY responses */
static CURLcode imap_state_capability_resp(struct Curl_easy *data,
                                           int imapcode,
                                           imapstate instate)
{
  CURLcode result = CURLE_OK;
  struct connectdata *conn = data->conn;
  struct imap_conn *imapc = &conn->proto.imapc;
  const char *line = data->state.buffer;

  (void)instate; /* no use for this yet */

  /* Do we have a untagged response? */
  if(imapcode == '*') {
    line += 2;

    /* Loop through the data line */
    for(;;) {
      size_t wordlen;
      while(*line &&
            (*line == ' ' || *line == '\t' ||
              *line == '\r' || *line == '\n')) {

        line++;
      }

      if(!*line)
        break;

      /* Extract the word */
      for(wordlen = 0; line[wordlen] && line[wordlen] != ' ' &&
            line[wordlen] != '\t' && line[wordlen] != '\r' &&
            line[wordlen] != '\n';)
        wordlen++;

      /* Does the server support the STARTTLS capability? */
      if(wordlen == 8 && !memcmp(line, "STARTTLS", 8))
        imapc->tls_supported = TRUE;

      /* Has the server explicitly disabled clear text authentication? */
      else if(wordlen == 13 && !memcmp(line, "LOGINDISABLED", 13))
        imapc->login_disabled = TRUE;

      /* Does the server support the SASL-IR capability? */
      else if(wordlen == 7 && !memcmp(line, "SASL-IR", 7))
        imapc->ir_supported = TRUE;

      /* Do we have a SASL based authentication mechanism? */
      else if(wordlen > 5 && !memcmp(line, "AUTH=", 5)) {
        size_t llen;
        unsigned short mechbit;

        line += 5;
        wordlen -= 5;

        /* Test the word for a matching authentication mechanism */
        mechbit = Curl_sasl_decode_mech(line, wordlen, &llen);
        if(mechbit && llen == wordlen)
          imapc->sasl.authmechs |= mechbit;
      }

      line += wordlen;
    }
  }
  else if(data->set.use_ssl && !conn->ssl[FIRSTSOCKET].use) {
    /* PREAUTH is not compatible with STARTTLS. */
    if(imapcode == IMAP_RESP_OK && imapc->tls_supported && !imapc->preauth) {
      /* Switch to TLS connection now */
      result = imap_perform_starttls(data);
    }
    else if(data->set.use_ssl <= CURLUSESSL_TRY)
      result = imap_perform_authentication(data, conn);
    else {
      failf(data, "STARTTLS not available.");
      result = CURLE_USE_SSL_FAILED;
    }
  }
  else
    result = imap_perform_authentication(data, conn);

  return result;
}

/* For STARTTLS responses */
static CURLcode imap_state_starttls_resp(struct Curl_easy *data,
                                         int imapcode,
                                         imapstate instate)
{
  CURLcode result = CURLE_OK;
  struct connectdata *conn = data->conn;

  (void)instate; /* no use for this yet */

  /* Pipelining in response is forbidden. */
  if(data->conn->proto.imapc.pp.cache_size)
    return CURLE_WEIRD_SERVER_REPLY;

  if(imapcode != IMAP_RESP_OK) {
    if(data->set.use_ssl != CURLUSESSL_TRY) {
      failf(data, "STARTTLS denied");
      result = CURLE_USE_SSL_FAILED;
    }
    else
      result = imap_perform_authentication(data, conn);
  }
  else
    result = imap_perform_upgrade_tls(data, conn);

  return result;
}

/* For SASL authentication responses */
static CURLcode imap_state_auth_resp(struct Curl_easy *data,
                                     struct connectdata *conn,
                                     int imapcode,
                                     imapstate instate)
{
  CURLcode result = CURLE_OK;
  struct imap_conn *imapc = &conn->proto.imapc;
  saslprogress progress;

  (void)instate; /* no use for this yet */

  result = Curl_sasl_continue(&imapc->sasl, data, imapcode, &progress);
  if(!result)
    switch(progress) {
    case SASL_DONE:
      state(data, IMAP_STOP);  /* Authenticated */
      break;
    case SASL_IDLE:            /* No mechanism left after cancellation */
      if((!imapc->login_disabled) && (imapc->preftype & IMAP_TYPE_CLEARTEXT))
        /* Perform clear text authentication */
        result = imap_perform_login(data, conn);
      else {
        failf(data, "Authentication cancelled");
        result = CURLE_LOGIN_DENIED;
      }
      break;
    default:
      break;
    }

  return result;
}

/* For LOGIN responses */
static CURLcode imap_state_login_resp(struct Curl_easy *data,
                                      int imapcode,
                                      imapstate instate)
{
  CURLcode result = CURLE_OK;
  (void)instate; /* no use for this yet */

  if(imapcode != IMAP_RESP_OK) {
    failf(data, "Access denied. %c", imapcode);
    result = CURLE_LOGIN_DENIED;
  }
  else
    /* End of connect phase */
    state(data, IMAP_STOP);

  return result;
}

/* For LIST and SEARCH responses */
static CURLcode imap_state_listsearch_resp(struct Curl_easy *data,
                                           int imapcode,
                                           imapstate instate)
{
  CURLcode result = CURLE_OK;
  char *line = data->state.buffer;
  size_t len = strlen(line);

  (void)instate; /* No use for this yet */

  if(imapcode == '*') {
    /* Temporarily add the LF character back and send as body to the client */
    line[len] = '\n';
    result = Curl_client_write(data, CLIENTWRITE_BODY, line, len + 1);
    line[len] = '\0';
  }
  else if(imapcode != IMAP_RESP_OK)
    result = CURLE_QUOTE_ERROR;
  else
    /* End of DO phase */
    state(data, IMAP_STOP);

  return result;
}

/* For SELECT responses */
static CURLcode imap_state_select_resp(struct Curl_easy *data, int imapcode,
                                       imapstate instate)
{
  CURLcode result = CURLE_OK;
  struct connectdata *conn = data->conn;
  struct IMAP *imap = data->req.p.imap;
  struct imap_conn *imapc = &conn->proto.imapc;
  const char *line = data->state.buffer;

  (void)instate; /* no use for this yet */

  if(imapcode == '*') {
    /* See if this is an UIDVALIDITY response */
    char tmp[20];
    if(sscanf(line + 2, "OK [UIDVALIDITY %19[0123456789]]", tmp) == 1) {
      Curl_safefree(imapc->mailbox_uidvalidity);
      imapc->mailbox_uidvalidity = strdup(tmp);
    }
  }
  else if(imapcode == IMAP_RESP_OK) {
    /* Check if the UIDVALIDITY has been specified and matches */
    if(imap->uidvalidity && imapc->mailbox_uidvalidity &&
       !strcasecompare(imap->uidvalidity, imapc->mailbox_uidvalidity)) {
      failf(data, "Mailbox UIDVALIDITY has changed");
      result = CURLE_REMOTE_FILE_NOT_FOUND;
    }
    else {
      /* Note the currently opened mailbox on this connection */
      imapc->mailbox = strdup(imap->mailbox);

      if(imap->custom)
        result = imap_perform_list(data);
      else if(imap->query)
        result = imap_perform_search(data);
      else
        result = imap_perform_fetch(data);
    }
  }
  else {
    failf(data, "Select failed");
    result = CURLE_LOGIN_DENIED;
  }

  return result;
}

/* For the (first line of the) FETCH responses */
static CURLcode imap_state_fetch_resp(struct Curl_easy *data,
                                      struct connectdata *conn, int imapcode,
                                      imapstate instate)
{
  CURLcode result = CURLE_OK;
  struct imap_conn *imapc = &conn->proto.imapc;
  struct pingpong *pp = &imapc->pp;
  const char *ptr = data->state.buffer;
  bool parsed = FALSE;
  curl_off_t size = 0;

  (void)instate; /* no use for this yet */

  if(imapcode != '*') {
    Curl_pgrsSetDownloadSize(data, -1);
    state(data, IMAP_STOP);
    return CURLE_REMOTE_FILE_NOT_FOUND;
  }

  /* Something like this is received "* 1 FETCH (BODY[TEXT] {2021}\r" so parse
     the continuation data contained within the curly brackets */
  while(*ptr && (*ptr != '{'))
    ptr++;

  if(*ptr == '{') {
    char *endptr;
    if(!curlx_strtoofft(ptr + 1, &endptr, 10, &size)) {
      if(endptr - ptr > 1 && endptr[0] == '}' &&
         endptr[1] == '\r' && endptr[2] == '\0')
        parsed = TRUE;
    }
  }

  if(parsed) {
    infof(data, "Found %" CURL_FORMAT_CURL_OFF_T " bytes to download",
          size);
    Curl_pgrsSetDownloadSize(data, size);

    if(pp->cache) {
      /* At this point there is a bunch of data in the header "cache" that is
         actually body content, send it as body and then skip it. Do note
         that there may even be additional "headers" after the body. */
      size_t chunk = pp->cache_size;

      if(chunk > (size_t)size)
        /* The conversion from curl_off_t to size_t is always fine here */
        chunk = (size_t)size;

      if(!chunk) {
        /* no size, we're done with the data */
        state(data, IMAP_STOP);
        return CURLE_OK;
      }
      result = Curl_client_write(data, CLIENTWRITE_BODY, pp->cache, chunk);
      if(result)
        return result;

      data->req.bytecount += chunk;

      infof(data, "Written %zu bytes, %" CURL_FORMAT_CURL_OFF_TU
            " bytes are left for transfer", chunk, size - chunk);

      /* Have we used the entire cache or just part of it?*/
      if(pp->cache_size > chunk) {
        /* Only part of it so shrink the cache to fit the trailing data */
        memmove(pp->cache, pp->cache + chunk, pp->cache_size - chunk);
        pp->cache_size -= chunk;
      }
      else {
        /* Free the cache */
        Curl_safefree(pp->cache);

        /* Reset the cache size */
        pp->cache_size = 0;
      }
    }

    if(data->req.bytecount == size)
      /* The entire data is already transferred! */
      Curl_setup_transfer(data, -1, -1, FALSE, -1);
    else {
      /* IMAP download */
      data->req.maxdownload = size;
      /* force a recv/send check of this connection, as the data might've been
       read off the socket already */
      data->conn->cselect_bits = CURL_CSELECT_IN;
      Curl_setup_transfer(data, FIRSTSOCKET, size, FALSE, -1);
    }
  }
  else {
    /* We don't know how to parse this line */
    failf(data, "Failed to parse FETCH response.");
    result = CURLE_WEIRD_SERVER_REPLY;
  }

  /* End of DO phase */
  state(data, IMAP_STOP);

  return result;
}

/* For final FETCH responses performed after the download */
static CURLcode imap_state_fetch_final_resp(struct Curl_easy *data,
                                            int imapcode,
                                            imapstate instate)
{
  CURLcode result = CURLE_OK;

  (void)instate; /* No use for this yet */

  if(imapcode != IMAP_RESP_OK)
    result = CURLE_WEIRD_SERVER_REPLY;
  else
    /* End of DONE phase */
    state(data, IMAP_STOP);

  return result;
}

/* For APPEND responses */
static CURLcode imap_state_append_resp(struct Curl_easy *data, int imapcode,
                                       imapstate instate)
{
  CURLcode result = CURLE_OK;
  (void)instate; /* No use for this yet */

  if(imapcode != '+') {
    result = CURLE_UPLOAD_FAILED;
  }
  else {
    /* Set the progress upload size */
    Curl_pgrsSetUploadSize(data, data->state.infilesize);

    /* IMAP upload */
    Curl_setup_transfer(data, -1, -1, FALSE, FIRSTSOCKET);

    /* End of DO phase */
    state(data, IMAP_STOP);
  }

  return result;
}

/* For final APPEND responses performed after the upload */
static CURLcode imap_state_append_final_resp(struct Curl_easy *data,
                                             int imapcode,
                                             imapstate instate)
{
  CURLcode result = CURLE_OK;

  (void)instate; /* No use for this yet */

  if(imapcode != IMAP_RESP_OK)
    result = CURLE_UPLOAD_FAILED;
  else
    /* End of DONE phase */
    state(data, IMAP_STOP);

  return result;
}

static CURLcode imap_statemachine(struct Curl_easy *data,
                                  struct connectdata *conn)
{
  CURLcode result = CURLE_OK;
  curl_socket_t sock = conn->sock[FIRSTSOCKET];
  int imapcode;
  struct imap_conn *imapc = &conn->proto.imapc;
  struct pingpong *pp = &imapc->pp;
  size_t nread = 0;
  (void)data;

  /* Busy upgrading the connection; right now all I/O is SSL/TLS, not IMAP */
  if(imapc->state == IMAP_UPGRADETLS)
    return imap_perform_upgrade_tls(data, conn);

  /* Flush any data that needs to be sent */
  if(pp->sendleft)
    return Curl_pp_flushsend(data, pp);

  do {
    /* Read the response from the server */
    result = Curl_pp_readresp(data, sock, pp, &imapcode, &nread);
    if(result)
      return result;

    /* Was there an error parsing the response line? */
    if(imapcode == -1)
      return CURLE_WEIRD_SERVER_REPLY;

    if(!imapcode)
      break;

    /* We have now received a full IMAP server response */
    switch(imapc->state) {
    case IMAP_SERVERGREET:
      result = imap_state_servergreet_resp(data, imapcode, imapc->state);
      break;

    case IMAP_CAPABILITY:
      result = imap_state_capability_resp(data, imapcode, imapc->state);
      break;

    case IMAP_STARTTLS:
      result = imap_state_starttls_resp(data, imapcode, imapc->state);
      break;

    case IMAP_AUTHENTICATE:
      result = imap_state_auth_resp(data, conn, imapcode, imapc->state);
      break;

    case IMAP_LOGIN:
      result = imap_state_login_resp(data, imapcode, imapc->state);
      break;

    case IMAP_LIST:
    case IMAP_SEARCH:
      result = imap_state_listsearch_resp(data, imapcode, imapc->state);
      break;

    case IMAP_SELECT:
      result = imap_state_select_resp(data, imapcode, imapc->state);
      break;

    case IMAP_FETCH:
      result = imap_state_fetch_resp(data, conn, imapcode, imapc->state);
      break;

    case IMAP_FETCH_FINAL:
      result = imap_state_fetch_final_resp(data, imapcode, imapc->state);
      break;

    case IMAP_APPEND:
      result = imap_state_append_resp(data, imapcode, imapc->state);
      break;

    case IMAP_APPEND_FINAL:
      result = imap_state_append_final_resp(data, imapcode, imapc->state);
      break;

    case IMAP_LOGOUT:
      /* fallthrough, just stop! */
    default:
      /* internal error */
      state(data, IMAP_STOP);
      break;
    }
  } while(!result && imapc->state != IMAP_STOP && Curl_pp_moredata(pp));

  return result;
}

/* Called repeatedly until done from multi.c */
static CURLcode imap_multi_statemach(struct Curl_easy *data, bool *done)
{
  CURLcode result = CURLE_OK;
  struct connectdata *conn = data->conn;
  struct imap_conn *imapc = &conn->proto.imapc;

  if((conn->handler->flags & PROTOPT_SSL) && !imapc->ssldone) {
    result = Curl_ssl_connect_nonblocking(data, conn, FALSE,
                                          FIRSTSOCKET, &imapc->ssldone);
    if(result || !imapc->ssldone)
      return result;
  }

  result = Curl_pp_statemach(data, &imapc->pp, FALSE, FALSE);
  *done = (imapc->state == IMAP_STOP) ? TRUE : FALSE;

  return result;
}

static CURLcode imap_block_statemach(struct Curl_easy *data,
                                     struct connectdata *conn,
                                     bool disconnecting)
{
  CURLcode result = CURLE_OK;
  struct imap_conn *imapc = &conn->proto.imapc;

  while(imapc->state != IMAP_STOP && !result)
    result = Curl_pp_statemach(data, &imapc->pp, TRUE, disconnecting);

  return result;
}

/* Allocate and initialize the struct IMAP for the current Curl_easy if
   required */
static CURLcode imap_init(struct Curl_easy *data)
{
  CURLcode result = CURLE_OK;
  struct IMAP *imap;

  imap = data->req.p.imap = calloc(sizeof(struct IMAP), 1);
  if(!imap)
    result = CURLE_OUT_OF_MEMORY;

  return result;
}

/* For the IMAP "protocol connect" and "doing" phases only */
static int imap_getsock(struct Curl_easy *data,
                        struct connectdata *conn,
                        curl_socket_t *socks)
{
  return Curl_pp_getsock(data, &conn->proto.imapc.pp, socks);
}

/***********************************************************************
 *
 * imap_connect()
 *
 * This function should do everything that is to be considered a part of the
 * connection phase.
 *
 * The variable 'done' points to will be TRUE if the protocol-layer connect
 * phase is done when this function returns, or FALSE if not.
 */
static CURLcode imap_connect(struct Curl_easy *data, bool *done)
{
  CURLcode result = CURLE_OK;
  struct connectdata *conn = data->conn;
  struct imap_conn *imapc = &conn->proto.imapc;
  struct pingpong *pp = &imapc->pp;

  *done = FALSE; /* default to not done yet */

  /* We always support persistent connections in IMAP */
  connkeep(conn, "IMAP default");

  PINGPONG_SETUP(pp, imap_statemachine, imap_endofresp);

  /* Set the default preferred authentication type and mechanism */
  imapc->preftype = IMAP_TYPE_ANY;
  Curl_sasl_init(&imapc->sasl, data, &saslimap);

  Curl_dyn_init(&imapc->dyn, DYN_IMAP_CMD);
  /* Initialise the pingpong layer */
  Curl_pp_setup(pp);
  Curl_pp_init(data, pp);

  /* Parse the URL options */
  result = imap_parse_url_options(conn);
  if(result)
    return result;

  /* Start off waiting for the server greeting response */
  state(data, IMAP_SERVERGREET);

  /* Start off with an response id of '*' */
  strcpy(imapc->resptag, "*");

  result = imap_multi_statemach(data, done);

  return result;
}

/***********************************************************************
 *
 * imap_done()
 *
 * The DONE function. This does what needs to be done after a single DO has
 * performed.
 *
 * Input argument is already checked for validity.
 */
static CURLcode imap_done(struct Curl_easy *data, CURLcode status,
                          bool premature)
{
  CURLcode result = CURLE_OK;
  struct connectdata *conn = data->conn;
  struct IMAP *imap = data->req.p.imap;

  (void)premature;

  if(!imap)
    return CURLE_OK;

  if(status) {
    connclose(conn, "IMAP done with bad status"); /* marked for closure */
    result = status;         /* use the already set error code */
  }
  else if(!data->set.connect_only && !imap->custom &&
          (imap->uid || imap->mindex || data->set.upload ||
          data->set.mimepost.kind != MIMEKIND_NONE)) {
    /* Handle responses after FETCH or APPEND transfer has finished */

    if(!data->set.upload && data->set.mimepost.kind == MIMEKIND_NONE)
      state(data, IMAP_FETCH_FINAL);
    else {
      /* End the APPEND command first by sending an empty line */
      result = Curl_pp_sendf(data, &conn->proto.imapc.pp, "%s", "");
      if(!result)
        state(data, IMAP_APPEND_FINAL);
    }

    /* Run the state-machine */
    if(!result)
      result = imap_block_statemach(data, conn, FALSE);
  }

  /* Cleanup our per-request based variables */
  Curl_safefree(imap->mailbox);
  Curl_safefree(imap->uidvalidity);
  Curl_safefree(imap->uid);
  Curl_safefree(imap->mindex);
  Curl_safefree(imap->section);
  Curl_safefree(imap->partial);
  Curl_safefree(imap->query);
  Curl_safefree(imap->custom);
  Curl_safefree(imap->custom_params);

  /* Clear the transfer mode for the next request */
  imap->transfer = PPTRANSFER_BODY;

  return result;
}

/***********************************************************************
 *
 * imap_perform()
 *
 * This is the actual DO function for IMAP. Fetch or append a message, or do
 * other things according to the options previously setup.
 */
static CURLcode imap_perform(struct Curl_easy *data, bool *connected,
                             bool *dophase_done)
{
  /* This is IMAP and no proxy */
  CURLcode result = CURLE_OK;
  struct connectdata *conn = data->conn;
  struct IMAP *imap = data->req.p.imap;
  struct imap_conn *imapc = &conn->proto.imapc;
  bool selected = FALSE;

  DEBUGF(infof(data, "DO phase starts"));

  if(data->set.opt_no_body) {
    /* Requested no body means no transfer */
    imap->transfer = PPTRANSFER_INFO;
  }

  *dophase_done = FALSE; /* not done yet */

  /* Determine if the requested mailbox (with the same UIDVALIDITY if set)
     has already been selected on this connection */
  if(imap->mailbox && imapc->mailbox &&
     strcasecompare(imap->mailbox, imapc->mailbox) &&
     (!imap->uidvalidity || !imapc->mailbox_uidvalidity ||
      strcasecompare(imap->uidvalidity, imapc->mailbox_uidvalidity)))
    selected = TRUE;

  /* Start the first command in the DO phase */
  if(data->set.upload || data->set.mimepost.kind != MIMEKIND_NONE)
    /* APPEND can be executed directly */
    result = imap_perform_append(data);
  else if(imap->custom && (selected || !imap->mailbox))
    /* Custom command using the same mailbox or no mailbox */
    result = imap_perform_list(data);
  else if(!imap->custom && selected && (imap->uid || imap->mindex))
    /* FETCH from the same mailbox */
    result = imap_perform_fetch(data);
  else if(!imap->custom && selected && imap->query)
    /* SEARCH the current mailbox */
    result = imap_perform_search(data);
  else if(imap->mailbox && !selected &&
         (imap->custom || imap->uid || imap->mindex || imap->query))
    /* SELECT the mailbox */
    result = imap_perform_select(data);
  else
    /* LIST */
    result = imap_perform_list(data);

  if(result)
    return result;

  /* Run the state-machine */
  result = imap_multi_statemach(data, dophase_done);

  *connected = conn->bits.tcpconnect[FIRSTSOCKET];

  if(*dophase_done)
    DEBUGF(infof(data, "DO phase is complete"));

  return result;
}

/***********************************************************************
 *
 * imap_do()
 *
 * This function is registered as 'curl_do' function. It decodes the path
 * parts etc as a wrapper to the actual DO function (imap_perform).
 *
 * The input argument is already checked for validity.
 */
static CURLcode imap_do(struct Curl_easy *data, bool *done)
{
  CURLcode result = CURLE_OK;
  *done = FALSE; /* default to false */

  /* Parse the URL path */
  result = imap_parse_url_path(data);
  if(result)
    return result;

  /* Parse the custom request */
  result = imap_parse_custom_request(data);
  if(result)
    return result;

  result = imap_regular_transfer(data, done);

  return result;
}

/***********************************************************************
 *
 * imap_disconnect()
 *
 * Disconnect from an IMAP server. Cleanup protocol-specific per-connection
 * resources. BLOCKING.
 */
static CURLcode imap_disconnect(struct Curl_easy *data,
                                struct connectdata *conn, bool dead_connection)
{
  struct imap_conn *imapc = &conn->proto.imapc;
  (void)data;

  /* We cannot send quit unconditionally. If this connection is stale or
     bad in any way, sending quit and waiting around here will make the
     disconnect wait in vain and cause more problems than we need to. */

  /* The IMAP session may or may not have been allocated/setup at this
     point! */
  if(!dead_connection && conn->bits.protoconnstart) {
    if(!imap_perform_logout(data))
      (void)imap_block_statemach(data, conn, TRUE); /* ignore errors */
  }

  /* Disconnect from the server */
  Curl_pp_disconnect(&imapc->pp);
  Curl_dyn_free(&imapc->dyn);

  /* Cleanup the SASL module */
  Curl_sasl_cleanup(conn, imapc->sasl.authused);

  /* Cleanup our connection based variables */
  Curl_safefree(imapc->mailbox);
  Curl_safefree(imapc->mailbox_uidvalidity);

  return CURLE_OK;
}

/* Call this when the DO phase has completed */
static CURLcode imap_dophase_done(struct Curl_easy *data, bool connected)
{
  struct IMAP *imap = data->req.p.imap;

  (void)connected;

  if(imap->transfer != PPTRANSFER_BODY)
    /* no data to transfer */
    Curl_setup_transfer(data, -1, -1, FALSE, -1);

  return CURLE_OK;
}

/* Called from multi.c while DOing */
static CURLcode imap_doing(struct Curl_easy *data, bool *dophase_done)
{
  CURLcode result = imap_multi_statemach(data, dophase_done);

  if(result)
    DEBUGF(infof(data, "DO phase failed"));
  else if(*dophase_done) {
    result = imap_dophase_done(data, FALSE /* not connected */);

    DEBUGF(infof(data, "DO phase is complete"));
  }

  return result;
}

/***********************************************************************
 *
 * imap_regular_transfer()
 *
 * The input argument is already checked for validity.
 *
 * Performs all commands done before a regular transfer between a local and a
 * remote host.
 */
static CURLcode imap_regular_transfer(struct Curl_easy *data,
                                      bool *dophase_done)
{
  CURLcode result = CURLE_OK;
  bool connected = FALSE;

  /* Make sure size is unknown at this point */
  data->req.size = -1;

  /* Set the progress data */
  Curl_pgrsSetUploadCounter(data, 0);
  Curl_pgrsSetDownloadCounter(data, 0);
  Curl_pgrsSetUploadSize(data, -1);
  Curl_pgrsSetDownloadSize(data, -1);

  /* Carry out the perform */
  result = imap_perform(data, &connected, dophase_done);

  /* Perform post DO phase operations if necessary */
  if(!result && *dophase_done)
    result = imap_dophase_done(data, connected);

  return result;
}

static CURLcode imap_setup_connection(struct Curl_easy *data,
                                      struct connectdata *conn)
{
  /* Initialise the IMAP layer */
  CURLcode result = imap_init(data);
  if(result)
    return result;

  /* Clear the TLS upgraded flag */
  conn->bits.tls_upgraded = FALSE;

  return CURLE_OK;
}

/***********************************************************************
 *
 * imap_sendf()
 *
 * Sends the formatted string as an IMAP command to the server.
 *
 * Designed to never block.
 */
static CURLcode imap_sendf(struct Curl_easy *data, const char *fmt, ...)
{
  CURLcode result = CURLE_OK;
  struct imap_conn *imapc = &data->conn->proto.imapc;

  DEBUGASSERT(fmt);

  /* Calculate the tag based on the connection ID and command ID */
  msnprintf(imapc->resptag, sizeof(imapc->resptag), "%c%03d",
            'A' + curlx_sltosi(data->conn->connection_id % 26),
            (++imapc->cmdid)%1000);

  /* start with a blank buffer */
  Curl_dyn_reset(&imapc->dyn);

  /* append tag + space + fmt */
  result = Curl_dyn_addf(&imapc->dyn, "%s %s", imapc->resptag, fmt);
  if(!result) {
    va_list ap;
    va_start(ap, fmt);
    result = Curl_pp_vsendf(data, &imapc->pp, Curl_dyn_ptr(&imapc->dyn), ap);
    va_end(ap);
  }
  return result;
}

/***********************************************************************
 *
 * imap_atom()
 *
 * Checks the input string for characters that need escaping and returns an
 * atom ready for sending to the server.
 *
 * The returned string needs to be freed.
 *
 */
static char *imap_atom(const char *str, bool escape_only)
{
  /* !checksrc! disable PARENBRACE 1 */
  const char atom_specials[] = "(){ %*]";
  const char *p1;
  char *p2;
  size_t backsp_count = 0;
  size_t quote_count = 0;
  bool others_exists = FALSE;
  size_t newlen = 0;
  char *newstr = NULL;

  if(!str)
    return NULL;

  /* Look for "atom-specials", counting the backslash and quote characters as
     these will need escaping */
  p1 = str;
  while(*p1) {
    if(*p1 == '\\')
      backsp_count++;
    else if(*p1 == '"')
      quote_count++;
    else if(!escape_only) {
      const char *p3 = atom_specials;

      while(*p3 && !others_exists) {
        if(*p1 == *p3)
          others_exists = TRUE;

        p3++;
      }
    }

    p1++;
  }

  /* Does the input contain any "atom-special" characters? */
  if(!backsp_count && !quote_count && !others_exists)
    return strdup(str);

  /* Calculate the new string length */
  newlen = strlen(str) + backsp_count + quote_count + (escape_only ? 0 : 2);

  /* Allocate the new string */
  newstr = (char *) malloc((newlen + 1) * sizeof(char));
  if(!newstr)
    return NULL;

  /* Surround the string in quotes if necessary */
  p2 = newstr;
  if(!escape_only) {
    newstr[0] = '"';
    newstr[newlen - 1] = '"';
    p2++;
  }

  /* Copy the string, escaping backslash and quote characters along the way */
  p1 = str;
  while(*p1) {
    if(*p1 == '\\' || *p1 == '"') {
      *p2 = '\\';
      p2++;
    }

   *p2 = *p1;

    p1++;
    p2++;
  }

  /* Terminate the string */
  newstr[newlen] = '\0';

  return newstr;
}

/***********************************************************************
 *
 * imap_is_bchar()
 *
 * Portable test of whether the specified char is a "bchar" as defined in the
 * grammar of RFC-5092.
 */
static bool imap_is_bchar(char ch)
{
  switch(ch) {
    /* bchar */
    case ':': case '@': case '/':
    /* bchar -> achar */
    case '&': case '=':
    /* bchar -> achar -> uchar -> unreserved */
    case '0': case '1': case '2': case '3': case '4': case '5': case '6':
    case '7': case '8': case '9':
    case 'A': case 'B': case 'C': case 'D': case 'E': case 'F': case 'G':
    case 'H': case 'I': case 'J': case 'K': case 'L': case 'M': case 'N':
    case 'O': case 'P': case 'Q': case 'R': case 'S': case 'T': case 'U':
    case 'V': case 'W': case 'X': case 'Y': case 'Z':
    case 'a': case 'b': case 'c': case 'd': case 'e': case 'f': case 'g':
    case 'h': case 'i': case 'j': case 'k': case 'l': case 'm': case 'n':
    case 'o': case 'p': case 'q': case 'r': case 's': case 't': case 'u':
    case 'v': case 'w': case 'x': case 'y': case 'z':
    case '-': case '.': case '_': case '~':
    /* bchar -> achar -> uchar -> sub-delims-sh */
    case '!': case '$': case '\'': case '(': case ')': case '*':
    case '+': case ',':
    /* bchar -> achar -> uchar -> pct-encoded */
    case '%': /* HEXDIG chars are already included above */
      return true;

    default:
      return false;
  }
}

/***********************************************************************
 *
 * imap_parse_url_options()
 *
 * Parse the URL login options.
 */
static CURLcode imap_parse_url_options(struct connectdata *conn)
{
  CURLcode result = CURLE_OK;
  struct imap_conn *imapc = &conn->proto.imapc;
  const char *ptr = conn->options;

  while(!result && ptr && *ptr) {
    const char *key = ptr;
    const char *value;

    while(*ptr && *ptr != '=')
        ptr++;

    value = ptr + 1;

    while(*ptr && *ptr != ';')
      ptr++;

    if(strncasecompare(key, "AUTH=", 5))
      result = Curl_sasl_parse_url_auth_option(&imapc->sasl,
                                               value, ptr - value);
    else
      result = CURLE_URL_MALFORMAT;

    if(*ptr == ';')
      ptr++;
  }

  switch(imapc->sasl.prefmech) {
  case SASL_AUTH_NONE:
    imapc->preftype = IMAP_TYPE_NONE;
    break;
  case SASL_AUTH_DEFAULT:
    imapc->preftype = IMAP_TYPE_ANY;
    break;
  default:
    imapc->preftype = IMAP_TYPE_SASL;
    break;
  }

  return result;
}

/***********************************************************************
 *
 * imap_parse_url_path()
 *
 * Parse the URL path into separate path components.
 *
 */
static CURLcode imap_parse_url_path(struct Curl_easy *data)
{
  /* The imap struct is already initialised in imap_connect() */
  CURLcode result = CURLE_OK;
  struct IMAP *imap = data->req.p.imap;
  const char *begin = &data->state.up.path[1]; /* skip leading slash */
  const char *ptr = begin;

  /* See how much of the URL is a valid path and decode it */
  while(imap_is_bchar(*ptr))
    ptr++;

  if(ptr != begin) {
    /* Remove the trailing slash if present */
    const char *end = ptr;
    if(end > begin && end[-1] == '/')
      end--;

    result = Curl_urldecode(data, begin, end - begin, &imap->mailbox, NULL,
                            REJECT_CTRL);
    if(result)
      return result;
  }
  else
    imap->mailbox = NULL;

  /* There can be any number of parameters in the form ";NAME=VALUE" */
  while(*ptr == ';') {
    char *name;
    char *value;
    size_t valuelen;

    /* Find the length of the name parameter */
    begin = ++ptr;
    while(*ptr && *ptr != '=')
      ptr++;

    if(!*ptr)
      return CURLE_URL_MALFORMAT;

    /* Decode the name parameter */
    result = Curl_urldecode(data, begin, ptr - begin, &name, NULL,
                            REJECT_CTRL);
    if(result)
      return result;

    /* Find the length of the value parameter */
    begin = ++ptr;
    while(imap_is_bchar(*ptr))
      ptr++;

    /* Decode the value parameter */
    result = Curl_urldecode(data, begin, ptr - begin, &value, &valuelen,
                            REJECT_CTRL);
    if(result) {
      free(name);
      return result;
    }

    DEBUGF(infof(data, "IMAP URL parameter '%s' = '%s'", name, value));

    /* Process the known hierarchical parameters (UIDVALIDITY, UID, SECTION and
       PARTIAL) stripping of the trailing slash character if it is present.

       Note: Unknown parameters trigger a URL_MALFORMAT error. */
    if(strcasecompare(name, "UIDVALIDITY") && !imap->uidvalidity) {
      if(valuelen > 0 && value[valuelen - 1] == '/')
        value[valuelen - 1] = '\0';

      imap->uidvalidity = value;
      value = NULL;
    }
    else if(strcasecompare(name, "UID") && !imap->uid) {
      if(valuelen > 0 && value[valuelen - 1] == '/')
        value[valuelen - 1] = '\0';

      imap->uid = value;
      value = NULL;
    }
    else if(strcasecompare(name, "MAILINDEX") && !imap->mindex) {
      if(valuelen > 0 && value[valuelen - 1] == '/')
        value[valuelen - 1] = '\0';

      imap->mindex = value;
      value = NULL;
    }
    else if(strcasecompare(name, "SECTION") && !imap->section) {
      if(valuelen > 0 && value[valuelen - 1] == '/')
        value[valuelen - 1] = '\0';

      imap->section = value;
      value = NULL;
    }
    else if(strcasecompare(name, "PARTIAL") && !imap->partial) {
      if(valuelen > 0 && value[valuelen - 1] == '/')
        value[valuelen - 1] = '\0';

      imap->partial = value;
      value = NULL;
    }
    else {
      free(name);
      free(value);

      return CURLE_URL_MALFORMAT;
    }

    free(name);
    free(value);
  }

  /* Does the URL contain a query parameter? Only valid when we have a mailbox
     and no UID as per RFC-5092 */
  if(imap->mailbox && !imap->uid && !imap->mindex) {
    /* Get the query parameter, URL decoded */
    (void)curl_url_get(data->state.uh, CURLUPART_QUERY, &imap->query,
                       CURLU_URLDECODE);
  }

  /* Any extra stuff at the end of the URL is an error */
  if(*ptr)
    return CURLE_URL_MALFORMAT;

  return CURLE_OK;
}

/***********************************************************************
 *
 * imap_parse_custom_request()
 *
 * Parse the custom request.
 */
static CURLcode imap_parse_custom_request(struct Curl_easy *data)
{
  CURLcode result = CURLE_OK;
  struct IMAP *imap = data->req.p.imap;
  const char *custom = data->set.str[STRING_CUSTOMREQUEST];

  if(custom) {
    /* URL decode the custom request */
    result = Curl_urldecode(data, custom, 0, &imap->custom, NULL, REJECT_CTRL);

    /* Extract the parameters if specified */
    if(!result) {
      const char *params = imap->custom;

      while(*params && *params != ' ')
        params++;

      if(*params) {
        imap->custom_params = strdup(params);
        imap->custom[params - imap->custom] = '\0';

        if(!imap->custom_params)
          result = CURLE_OUT_OF_MEMORY;
      }
    }
  }

  return result;
}

#endif /* CURL_DISABLE_IMAP */
