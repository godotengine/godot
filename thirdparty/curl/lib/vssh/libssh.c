/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 2017 - 2021 Red Hat, Inc.
 *
 * Authors: Nikos Mavrogiannopoulos, Tomas Mraz, Stanislav Zidek,
 *          Robert Kolcun, Andreas Schneider
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

#ifdef USE_LIBSSH

#include <limits.h>

#include <libssh/libssh.h>
#include <libssh/sftp.h>

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif

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
#include "http.h"               /* for HTTP proxy tunnel stuff */
#include "ssh.h"
#include "url.h"
#include "speedcheck.h"
#include "getinfo.h"
#include "strdup.h"
#include "strcase.h"
#include "vtls/vtls.h"
#include "connect.h"
#include "inet_ntop.h"
#include "parsedate.h"          /* for the week day and month names */
#include "sockaddr.h"           /* required for Curl_sockaddr_storage */
#include "strtoofft.h"
#include "multiif.h"
#include "select.h"
#include "warnless.h"

/* for permission and open flags */
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"
#include "curl_path.h"

/* A recent macro provided by libssh. Or make our own. */
#ifndef SSH_STRING_FREE_CHAR
#define SSH_STRING_FREE_CHAR(x)                 \
  do {                                          \
    if(x) {                                     \
      ssh_string_free_char(x);                  \
      x = NULL;                                 \
    }                                           \
  } while(0)
#endif

/* Local functions: */
static CURLcode myssh_connect(struct Curl_easy *data, bool *done);
static CURLcode myssh_multi_statemach(struct Curl_easy *data,
                                      bool *done);
static CURLcode myssh_do_it(struct Curl_easy *data, bool *done);

static CURLcode scp_done(struct Curl_easy *data,
                         CURLcode, bool premature);
static CURLcode scp_doing(struct Curl_easy *data, bool *dophase_done);
static CURLcode scp_disconnect(struct Curl_easy *data,
                               struct connectdata *conn,
                               bool dead_connection);

static CURLcode sftp_done(struct Curl_easy *data,
                          CURLcode, bool premature);
static CURLcode sftp_doing(struct Curl_easy *data,
                           bool *dophase_done);
static CURLcode sftp_disconnect(struct Curl_easy *data,
                                struct connectdata *conn,
                                bool dead);
static
CURLcode sftp_perform(struct Curl_easy *data,
                      bool *connected,
                      bool *dophase_done);

static void sftp_quote(struct Curl_easy *data);
static void sftp_quote_stat(struct Curl_easy *data);
static int myssh_getsock(struct Curl_easy *data,
                         struct connectdata *conn, curl_socket_t *sock);

static CURLcode myssh_setup_connection(struct Curl_easy *data,
                                       struct connectdata *conn);

/*
 * SCP protocol handler.
 */

const struct Curl_handler Curl_handler_scp = {
  "SCP",                        /* scheme */
  myssh_setup_connection,       /* setup_connection */
  myssh_do_it,                  /* do_it */
  scp_done,                     /* done */
  ZERO_NULL,                    /* do_more */
  myssh_connect,                /* connect_it */
  myssh_multi_statemach,        /* connecting */
  scp_doing,                    /* doing */
  myssh_getsock,                /* proto_getsock */
  myssh_getsock,                /* doing_getsock */
  ZERO_NULL,                    /* domore_getsock */
  myssh_getsock,                /* perform_getsock */
  scp_disconnect,               /* disconnect */
  ZERO_NULL,                    /* readwrite */
  ZERO_NULL,                    /* connection_check */
  ZERO_NULL,                    /* attach connection */
  PORT_SSH,                     /* defport */
  CURLPROTO_SCP,                /* protocol */
  CURLPROTO_SCP,                /* family */
  PROTOPT_DIRLOCK | PROTOPT_CLOSEACTION | PROTOPT_NOURLQUERY    /* flags */
};

/*
 * SFTP protocol handler.
 */

const struct Curl_handler Curl_handler_sftp = {
  "SFTP",                               /* scheme */
  myssh_setup_connection,               /* setup_connection */
  myssh_do_it,                          /* do_it */
  sftp_done,                            /* done */
  ZERO_NULL,                            /* do_more */
  myssh_connect,                        /* connect_it */
  myssh_multi_statemach,                /* connecting */
  sftp_doing,                           /* doing */
  myssh_getsock,                        /* proto_getsock */
  myssh_getsock,                        /* doing_getsock */
  ZERO_NULL,                            /* domore_getsock */
  myssh_getsock,                        /* perform_getsock */
  sftp_disconnect,                      /* disconnect */
  ZERO_NULL,                            /* readwrite */
  ZERO_NULL,                            /* connection_check */
  ZERO_NULL,                            /* attach connection */
  PORT_SSH,                             /* defport */
  CURLPROTO_SFTP,                       /* protocol */
  CURLPROTO_SFTP,                       /* family */
  PROTOPT_DIRLOCK | PROTOPT_CLOSEACTION
  | PROTOPT_NOURLQUERY                  /* flags */
};

static CURLcode sftp_error_to_CURLE(int err)
{
  switch(err) {
    case SSH_FX_OK:
      return CURLE_OK;

    case SSH_FX_NO_SUCH_FILE:
    case SSH_FX_NO_SUCH_PATH:
      return CURLE_REMOTE_FILE_NOT_FOUND;

    case SSH_FX_PERMISSION_DENIED:
    case SSH_FX_WRITE_PROTECT:
      return CURLE_REMOTE_ACCESS_DENIED;

    case SSH_FX_FILE_ALREADY_EXISTS:
      return CURLE_REMOTE_FILE_EXISTS;

    default:
      break;
  }

  return CURLE_SSH;
}

#ifndef DEBUGBUILD
#define state(x,y) mystate(x,y)
#else
#define state(x,y) mystate(x,y, __LINE__)
#endif

/*
 * SSH State machine related code
 */
/* This is the ONLY way to change SSH state! */
static void mystate(struct Curl_easy *data, sshstate nowstate
#ifdef DEBUGBUILD
                    , int lineno
#endif
  )
{
  struct connectdata *conn = data->conn;
  struct ssh_conn *sshc = &conn->proto.sshc;
#if defined(DEBUGBUILD) && !defined(CURL_DISABLE_VERBOSE_STRINGS)
  /* for debug purposes */
  static const char *const names[] = {
    "SSH_STOP",
    "SSH_INIT",
    "SSH_S_STARTUP",
    "SSH_HOSTKEY",
    "SSH_AUTHLIST",
    "SSH_AUTH_PKEY_INIT",
    "SSH_AUTH_PKEY",
    "SSH_AUTH_PASS_INIT",
    "SSH_AUTH_PASS",
    "SSH_AUTH_AGENT_INIT",
    "SSH_AUTH_AGENT_LIST",
    "SSH_AUTH_AGENT",
    "SSH_AUTH_HOST_INIT",
    "SSH_AUTH_HOST",
    "SSH_AUTH_KEY_INIT",
    "SSH_AUTH_KEY",
    "SSH_AUTH_GSSAPI",
    "SSH_AUTH_DONE",
    "SSH_SFTP_INIT",
    "SSH_SFTP_REALPATH",
    "SSH_SFTP_QUOTE_INIT",
    "SSH_SFTP_POSTQUOTE_INIT",
    "SSH_SFTP_QUOTE",
    "SSH_SFTP_NEXT_QUOTE",
    "SSH_SFTP_QUOTE_STAT",
    "SSH_SFTP_QUOTE_SETSTAT",
    "SSH_SFTP_QUOTE_SYMLINK",
    "SSH_SFTP_QUOTE_MKDIR",
    "SSH_SFTP_QUOTE_RENAME",
    "SSH_SFTP_QUOTE_RMDIR",
    "SSH_SFTP_QUOTE_UNLINK",
    "SSH_SFTP_QUOTE_STATVFS",
    "SSH_SFTP_GETINFO",
    "SSH_SFTP_FILETIME",
    "SSH_SFTP_TRANS_INIT",
    "SSH_SFTP_UPLOAD_INIT",
    "SSH_SFTP_CREATE_DIRS_INIT",
    "SSH_SFTP_CREATE_DIRS",
    "SSH_SFTP_CREATE_DIRS_MKDIR",
    "SSH_SFTP_READDIR_INIT",
    "SSH_SFTP_READDIR",
    "SSH_SFTP_READDIR_LINK",
    "SSH_SFTP_READDIR_BOTTOM",
    "SSH_SFTP_READDIR_DONE",
    "SSH_SFTP_DOWNLOAD_INIT",
    "SSH_SFTP_DOWNLOAD_STAT",
    "SSH_SFTP_CLOSE",
    "SSH_SFTP_SHUTDOWN",
    "SSH_SCP_TRANS_INIT",
    "SSH_SCP_UPLOAD_INIT",
    "SSH_SCP_DOWNLOAD_INIT",
    "SSH_SCP_DOWNLOAD",
    "SSH_SCP_DONE",
    "SSH_SCP_SEND_EOF",
    "SSH_SCP_WAIT_EOF",
    "SSH_SCP_WAIT_CLOSE",
    "SSH_SCP_CHANNEL_FREE",
    "SSH_SESSION_DISCONNECT",
    "SSH_SESSION_FREE",
    "QUIT"
  };


  if(sshc->state != nowstate) {
    infof(data, "SSH %p state change from %s to %s (line %d)",
          (void *) sshc, names[sshc->state], names[nowstate],
          lineno);
  }
#endif

  sshc->state = nowstate;
}

/* Multiple options:
 * 1. data->set.str[STRING_SSH_HOST_PUBLIC_KEY_MD5] is set with an MD5
 *    hash (90s style auth, not sure we should have it here)
 * 2. data->set.ssh_keyfunc callback is set. Then we do trust on first
 *    use. We even save on knownhosts if CURLKHSTAT_FINE_ADD_TO_FILE
 *    is returned by it.
 * 3. none of the above. We only accept if it is present on known hosts.
 *
 * Returns SSH_OK or SSH_ERROR.
 */
static int myssh_is_known(struct Curl_easy *data)
{
  int rc;
  struct connectdata *conn = data->conn;
  struct ssh_conn *sshc = &conn->proto.sshc;
  ssh_key pubkey;
  size_t hlen;
  unsigned char *hash = NULL;
  char *found_base64 = NULL;
  char *known_base64 = NULL;
  int vstate;
  enum curl_khmatch keymatch;
  struct curl_khkey foundkey;
  struct curl_khkey *knownkeyp = NULL;
  curl_sshkeycallback func =
    data->set.ssh_keyfunc;

#if LIBSSH_VERSION_INT >= SSH_VERSION_INT(0,9,0)
  struct ssh_knownhosts_entry *knownhostsentry = NULL;
  struct curl_khkey knownkey;
#endif

#if LIBSSH_VERSION_INT >= SSH_VERSION_INT(0,8,0)
  rc = ssh_get_server_publickey(sshc->ssh_session, &pubkey);
#else
  rc = ssh_get_publickey(sshc->ssh_session, &pubkey);
#endif
  if(rc != SSH_OK)
    return rc;

  if(data->set.str[STRING_SSH_HOST_PUBLIC_KEY_MD5]) {
    int i;
    char md5buffer[33];
    const char *pubkey_md5 = data->set.str[STRING_SSH_HOST_PUBLIC_KEY_MD5];

    rc = ssh_get_publickey_hash(pubkey, SSH_PUBLICKEY_HASH_MD5,
                                &hash, &hlen);
    if(rc != SSH_OK || hlen != 16) {
      failf(data,
            "Denied establishing ssh session: md5 fingerprint not available");
      goto cleanup;
    }

    for(i = 0; i < 16; i++)
      msnprintf(&md5buffer[i*2], 3, "%02x", (unsigned char)hash[i]);

    infof(data, "SSH MD5 fingerprint: %s", md5buffer);

    if(!strcasecompare(md5buffer, pubkey_md5)) {
      failf(data,
            "Denied establishing ssh session: mismatch md5 fingerprint. "
            "Remote %s is not equal to %s", md5buffer, pubkey_md5);
      rc = SSH_ERROR;
      goto cleanup;
    }

    rc = SSH_OK;
    goto cleanup;
  }

  if(data->set.ssl.primary.verifyhost != TRUE) {
    rc = SSH_OK;
    goto cleanup;
  }

#if LIBSSH_VERSION_INT >= SSH_VERSION_INT(0,9,0)
  /* Get the known_key from the known hosts file */
  vstate = ssh_session_get_known_hosts_entry(sshc->ssh_session,
                                             &knownhostsentry);

  /* Case an entry was found in a known hosts file */
  if(knownhostsentry) {
    if(knownhostsentry->publickey) {
      rc = ssh_pki_export_pubkey_base64(knownhostsentry->publickey,
                                        &known_base64);
      if(rc != SSH_OK) {
        goto cleanup;
      }
      knownkey.key = known_base64;
      knownkey.len = strlen(known_base64);

      switch(ssh_key_type(knownhostsentry->publickey)) {
        case SSH_KEYTYPE_RSA:
          knownkey.keytype = CURLKHTYPE_RSA;
          break;
        case SSH_KEYTYPE_RSA1:
          knownkey.keytype = CURLKHTYPE_RSA1;
          break;
        case SSH_KEYTYPE_ECDSA:
        case SSH_KEYTYPE_ECDSA_P256:
        case SSH_KEYTYPE_ECDSA_P384:
        case SSH_KEYTYPE_ECDSA_P521:
          knownkey.keytype = CURLKHTYPE_ECDSA;
          break;
        case SSH_KEYTYPE_ED25519:
          knownkey.keytype = CURLKHTYPE_ED25519;
          break;
        case SSH_KEYTYPE_DSS:
          knownkey.keytype = CURLKHTYPE_DSS;
          break;
        default:
          rc = SSH_ERROR;
          goto cleanup;
      }
      knownkeyp = &knownkey;
    }
  }

  switch(vstate) {
    case SSH_KNOWN_HOSTS_OK:
      keymatch = CURLKHMATCH_OK;
      break;
    case SSH_KNOWN_HOSTS_OTHER:
      /* fallthrough */
    case SSH_KNOWN_HOSTS_NOT_FOUND:
      /* fallthrough */
    case SSH_KNOWN_HOSTS_UNKNOWN:
      /* fallthrough */
    case SSH_KNOWN_HOSTS_ERROR:
      keymatch = CURLKHMATCH_MISSING;
      break;
  default:
      keymatch = CURLKHMATCH_MISMATCH;
      break;
  }

#else
  vstate = ssh_is_server_known(sshc->ssh_session);
  switch(vstate) {
    case SSH_SERVER_KNOWN_OK:
      keymatch = CURLKHMATCH_OK;
      break;
    case SSH_SERVER_FILE_NOT_FOUND:
      /* fallthrough */
    case SSH_SERVER_NOT_KNOWN:
      keymatch = CURLKHMATCH_MISSING;
      break;
  default:
      keymatch = CURLKHMATCH_MISMATCH;
      break;
  }
#endif

  if(func) { /* use callback to determine action */
    rc = ssh_pki_export_pubkey_base64(pubkey, &found_base64);
    if(rc != SSH_OK)
      goto cleanup;

    foundkey.key = found_base64;
    foundkey.len = strlen(found_base64);

    switch(ssh_key_type(pubkey)) {
      case SSH_KEYTYPE_RSA:
        foundkey.keytype = CURLKHTYPE_RSA;
        break;
      case SSH_KEYTYPE_RSA1:
        foundkey.keytype = CURLKHTYPE_RSA1;
        break;
      case SSH_KEYTYPE_ECDSA:
#if LIBSSH_VERSION_INT >= SSH_VERSION_INT(0,9,0)
      case SSH_KEYTYPE_ECDSA_P256:
      case SSH_KEYTYPE_ECDSA_P384:
      case SSH_KEYTYPE_ECDSA_P521:
#endif
        foundkey.keytype = CURLKHTYPE_ECDSA;
        break;
#if LIBSSH_VERSION_INT >= SSH_VERSION_INT(0,7,0)
      case SSH_KEYTYPE_ED25519:
        foundkey.keytype = CURLKHTYPE_ED25519;
        break;
#endif
      case SSH_KEYTYPE_DSS:
        foundkey.keytype = CURLKHTYPE_DSS;
        break;
      default:
        rc = SSH_ERROR;
        goto cleanup;
    }

    Curl_set_in_callback(data, true);
    rc = func(data, knownkeyp, /* from the knownhosts file */
              &foundkey, /* from the remote host */
              keymatch, data->set.ssh_keyfunc_userp);
    Curl_set_in_callback(data, false);

    switch(rc) {
      case CURLKHSTAT_FINE_ADD_TO_FILE:
#if LIBSSH_VERSION_INT >= SSH_VERSION_INT(0,8,0)
        rc = ssh_session_update_known_hosts(sshc->ssh_session);
#else
        rc = ssh_write_knownhost(sshc->ssh_session);
#endif
        if(rc != SSH_OK) {
          goto cleanup;
        }
        break;
      case CURLKHSTAT_FINE:
        break;
      default: /* REJECT/DEFER */
        rc = SSH_ERROR;
        goto cleanup;
    }
  }
  else {
    if(keymatch != CURLKHMATCH_OK) {
      rc = SSH_ERROR;
      goto cleanup;
    }
  }
  rc = SSH_OK;

cleanup:
  if(found_base64) {
    (free)(found_base64);
  }
  if(known_base64) {
    (free)(known_base64);
  }
  if(hash)
    ssh_clean_pubkey_hash(&hash);
  ssh_key_free(pubkey);
#if LIBSSH_VERSION_INT >= SSH_VERSION_INT(0,9,0)
  if(knownhostsentry) {
    ssh_knownhosts_entry_free(knownhostsentry);
  }
#endif
  return rc;
}

#define MOVE_TO_ERROR_STATE(_r) do {            \
    state(data, SSH_SESSION_DISCONNECT);        \
    sshc->actualcode = _r;                      \
    rc = SSH_ERROR;                             \
  } while(0)

#define MOVE_TO_SFTP_CLOSE_STATE() do {                         \
    state(data, SSH_SFTP_CLOSE);                                \
    sshc->actualcode =                                          \
      sftp_error_to_CURLE(sftp_get_error(sshc->sftp_session));  \
    rc = SSH_ERROR;                                             \
  } while(0)

#define MOVE_TO_LAST_AUTH do {                          \
    if(sshc->auth_methods & SSH_AUTH_METHOD_PASSWORD) { \
      rc = SSH_OK;                                      \
      state(data, SSH_AUTH_PASS_INIT);                  \
    }                                                   \
    else {                                              \
      MOVE_TO_ERROR_STATE(CURLE_LOGIN_DENIED);          \
    }                                                   \
  } while(0)

#define MOVE_TO_TERTIARY_AUTH do {                              \
    if(sshc->auth_methods & SSH_AUTH_METHOD_INTERACTIVE) {      \
      rc = SSH_OK;                                              \
      state(data, SSH_AUTH_KEY_INIT);                           \
    }                                                           \
    else {                                                      \
      MOVE_TO_LAST_AUTH;                                        \
    }                                                           \
  } while(0)

#define MOVE_TO_SECONDARY_AUTH do {                             \
    if(sshc->auth_methods & SSH_AUTH_METHOD_GSSAPI_MIC) {       \
      rc = SSH_OK;                                              \
      state(data, SSH_AUTH_GSSAPI);                             \
    }                                                           \
    else {                                                      \
      MOVE_TO_TERTIARY_AUTH;                                    \
    }                                                           \
  } while(0)

static
int myssh_auth_interactive(struct connectdata *conn)
{
  int rc;
  struct ssh_conn *sshc = &conn->proto.sshc;
  int nprompts;

restart:
  switch(sshc->kbd_state) {
    case 0:
      rc = ssh_userauth_kbdint(sshc->ssh_session, NULL, NULL);
      if(rc == SSH_AUTH_AGAIN)
        return SSH_AGAIN;

      if(rc != SSH_AUTH_INFO)
        return SSH_ERROR;

      nprompts = ssh_userauth_kbdint_getnprompts(sshc->ssh_session);
      if(nprompts != 1)
        return SSH_ERROR;

      rc = ssh_userauth_kbdint_setanswer(sshc->ssh_session, 0, conn->passwd);
      if(rc < 0)
        return SSH_ERROR;

    /* FALLTHROUGH */
    case 1:
      sshc->kbd_state = 1;

      rc = ssh_userauth_kbdint(sshc->ssh_session, NULL, NULL);
      if(rc == SSH_AUTH_AGAIN)
        return SSH_AGAIN;
      else if(rc == SSH_AUTH_SUCCESS)
        rc = SSH_OK;
      else if(rc == SSH_AUTH_INFO) {
        nprompts = ssh_userauth_kbdint_getnprompts(sshc->ssh_session);
        if(nprompts)
          return SSH_ERROR;

        sshc->kbd_state = 2;
        goto restart;
      }
      else
        rc = SSH_ERROR;
      break;
    case 2:
      sshc->kbd_state = 2;

      rc = ssh_userauth_kbdint(sshc->ssh_session, NULL, NULL);
      if(rc == SSH_AUTH_AGAIN)
        return SSH_AGAIN;
      else if(rc == SSH_AUTH_SUCCESS)
        rc = SSH_OK;
      else
        rc = SSH_ERROR;

      break;
    default:
      return SSH_ERROR;
  }

  sshc->kbd_state = 0;
  return rc;
}

/*
 * ssh_statemach_act() runs the SSH state machine as far as it can without
 * blocking and without reaching the end.  The data the pointer 'block' points
 * to will be set to TRUE if the libssh function returns SSH_AGAIN
 * meaning it wants to be called again when the socket is ready
 */
static CURLcode myssh_statemach_act(struct Curl_easy *data, bool *block)
{
  CURLcode result = CURLE_OK;
  struct connectdata *conn = data->conn;
  struct SSHPROTO *protop = data->req.p.ssh;
  struct ssh_conn *sshc = &conn->proto.sshc;
  curl_socket_t sock = conn->sock[FIRSTSOCKET];
  int rc = SSH_NO_ERROR, err;
  char *new_readdir_line;
  int seekerr = CURL_SEEKFUNC_OK;
  const char *err_msg;
  *block = 0;                   /* we're not blocking by default */

  do {

    switch(sshc->state) {
    case SSH_INIT:
      sshc->secondCreateDirs = 0;
      sshc->nextstate = SSH_NO_STATE;
      sshc->actualcode = CURLE_OK;

#if 0
      ssh_set_log_level(SSH_LOG_PROTOCOL);
#endif

      /* Set libssh to non-blocking, since everything internally is
         non-blocking */
      ssh_set_blocking(sshc->ssh_session, 0);

      state(data, SSH_S_STARTUP);
      /* FALLTHROUGH */

    case SSH_S_STARTUP:
      rc = ssh_connect(sshc->ssh_session);
      if(rc == SSH_AGAIN)
        break;

      if(rc != SSH_OK) {
        failf(data, "Failure establishing ssh session");
        MOVE_TO_ERROR_STATE(CURLE_FAILED_INIT);
        break;
      }

      state(data, SSH_HOSTKEY);

      /* FALLTHROUGH */
    case SSH_HOSTKEY:

      rc = myssh_is_known(data);
      if(rc != SSH_OK) {
        MOVE_TO_ERROR_STATE(CURLE_PEER_FAILED_VERIFICATION);
        break;
      }

      state(data, SSH_AUTHLIST);
      /* FALLTHROUGH */
    case SSH_AUTHLIST:{
        sshc->authed = FALSE;

        rc = ssh_userauth_none(sshc->ssh_session, NULL);
        if(rc == SSH_AUTH_AGAIN) {
          rc = SSH_AGAIN;
          break;
        }

        if(rc == SSH_AUTH_SUCCESS) {
          sshc->authed = TRUE;
          infof(data, "Authenticated with none");
          state(data, SSH_AUTH_DONE);
          break;
        }
        else if(rc == SSH_AUTH_ERROR) {
          MOVE_TO_ERROR_STATE(CURLE_LOGIN_DENIED);
          break;
        }

        sshc->auth_methods = ssh_userauth_list(sshc->ssh_session, NULL);
        if(sshc->auth_methods & SSH_AUTH_METHOD_PUBLICKEY) {
          state(data, SSH_AUTH_PKEY_INIT);
          infof(data, "Authentication using SSH public key file");
        }
        else if(sshc->auth_methods & SSH_AUTH_METHOD_GSSAPI_MIC) {
          state(data, SSH_AUTH_GSSAPI);
        }
        else if(sshc->auth_methods & SSH_AUTH_METHOD_INTERACTIVE) {
          state(data, SSH_AUTH_KEY_INIT);
        }
        else if(sshc->auth_methods & SSH_AUTH_METHOD_PASSWORD) {
          state(data, SSH_AUTH_PASS_INIT);
        }
        else {                  /* unsupported authentication method */
          MOVE_TO_ERROR_STATE(CURLE_LOGIN_DENIED);
          break;
        }

        break;
      }
    case SSH_AUTH_PKEY_INIT:
      if(!(data->set.ssh_auth_types & CURLSSH_AUTH_PUBLICKEY)) {
        MOVE_TO_SECONDARY_AUTH;
        break;
      }

      /* Two choices, (1) private key was given on CMD,
       * (2) use the "default" keys. */
      if(data->set.str[STRING_SSH_PRIVATE_KEY]) {
        if(sshc->pubkey && !data->set.ssl.key_passwd) {
          rc = ssh_userauth_try_publickey(sshc->ssh_session, NULL,
                                          sshc->pubkey);
          if(rc == SSH_AUTH_AGAIN) {
            rc = SSH_AGAIN;
            break;
          }

          if(rc != SSH_OK) {
            MOVE_TO_SECONDARY_AUTH;
            break;
          }
        }

        rc = ssh_pki_import_privkey_file(data->
                                         set.str[STRING_SSH_PRIVATE_KEY],
                                         data->set.ssl.key_passwd, NULL,
                                         NULL, &sshc->privkey);
        if(rc != SSH_OK) {
          failf(data, "Could not load private key file %s",
                data->set.str[STRING_SSH_PRIVATE_KEY]);
          MOVE_TO_ERROR_STATE(CURLE_LOGIN_DENIED);
          break;
        }

        state(data, SSH_AUTH_PKEY);
        break;

      }
      else {
        rc = ssh_userauth_publickey_auto(sshc->ssh_session, NULL,
                                         data->set.ssl.key_passwd);
        if(rc == SSH_AUTH_AGAIN) {
          rc = SSH_AGAIN;
          break;
        }
        if(rc == SSH_AUTH_SUCCESS) {
          rc = SSH_OK;
          sshc->authed = TRUE;
          infof(data, "Completed public key authentication");
          state(data, SSH_AUTH_DONE);
          break;
        }

        MOVE_TO_SECONDARY_AUTH;
      }
      break;
    case SSH_AUTH_PKEY:
      rc = ssh_userauth_publickey(sshc->ssh_session, NULL, sshc->privkey);
      if(rc == SSH_AUTH_AGAIN) {
        rc = SSH_AGAIN;
        break;
      }

      if(rc == SSH_AUTH_SUCCESS) {
        sshc->authed = TRUE;
        infof(data, "Completed public key authentication");
        state(data, SSH_AUTH_DONE);
        break;
      }
      else {
        infof(data, "Failed public key authentication (rc: %d)", rc);
        MOVE_TO_SECONDARY_AUTH;
      }
      break;

    case SSH_AUTH_GSSAPI:
      if(!(data->set.ssh_auth_types & CURLSSH_AUTH_GSSAPI)) {
        MOVE_TO_TERTIARY_AUTH;
        break;
      }

      rc = ssh_userauth_gssapi(sshc->ssh_session);
      if(rc == SSH_AUTH_AGAIN) {
        rc = SSH_AGAIN;
        break;
      }

      if(rc == SSH_AUTH_SUCCESS) {
        rc = SSH_OK;
        sshc->authed = TRUE;
        infof(data, "Completed gssapi authentication");
        state(data, SSH_AUTH_DONE);
        break;
      }

      MOVE_TO_TERTIARY_AUTH;
      break;

    case SSH_AUTH_KEY_INIT:
      if(data->set.ssh_auth_types & CURLSSH_AUTH_KEYBOARD) {
        state(data, SSH_AUTH_KEY);
      }
      else {
        MOVE_TO_LAST_AUTH;
      }
      break;

    case SSH_AUTH_KEY:

      /* Authentication failed. Continue with keyboard-interactive now. */
      rc = myssh_auth_interactive(conn);
      if(rc == SSH_AGAIN) {
        break;
      }
      if(rc == SSH_OK) {
        sshc->authed = TRUE;
        infof(data, "completed keyboard interactive authentication");
      }
      state(data, SSH_AUTH_DONE);
      break;

    case SSH_AUTH_PASS_INIT:
      if(!(data->set.ssh_auth_types & CURLSSH_AUTH_PASSWORD)) {
        /* Host key authentication is intentionally not implemented */
        MOVE_TO_ERROR_STATE(CURLE_LOGIN_DENIED);
        break;
      }
      state(data, SSH_AUTH_PASS);
      /* FALLTHROUGH */

    case SSH_AUTH_PASS:
      rc = ssh_userauth_password(sshc->ssh_session, NULL, conn->passwd);
      if(rc == SSH_AUTH_AGAIN) {
        rc = SSH_AGAIN;
        break;
      }

      if(rc == SSH_AUTH_SUCCESS) {
        sshc->authed = TRUE;
        infof(data, "Completed password authentication");
        state(data, SSH_AUTH_DONE);
      }
      else {
        MOVE_TO_ERROR_STATE(CURLE_LOGIN_DENIED);
      }
      break;

    case SSH_AUTH_DONE:
      if(!sshc->authed) {
        failf(data, "Authentication failure");
        MOVE_TO_ERROR_STATE(CURLE_LOGIN_DENIED);
        break;
      }

      /*
       * At this point we have an authenticated ssh session.
       */
      infof(data, "Authentication complete");

      Curl_pgrsTime(data, TIMER_APPCONNECT);      /* SSH is connected */

      conn->sockfd = sock;
      conn->writesockfd = CURL_SOCKET_BAD;

      if(conn->handler->protocol == CURLPROTO_SFTP) {
        state(data, SSH_SFTP_INIT);
        break;
      }
      infof(data, "SSH CONNECT phase done");
      state(data, SSH_STOP);
      break;

    case SSH_SFTP_INIT:
      ssh_set_blocking(sshc->ssh_session, 1);

      sshc->sftp_session = sftp_new(sshc->ssh_session);
      if(!sshc->sftp_session) {
        failf(data, "Failure initializing sftp session: %s",
              ssh_get_error(sshc->ssh_session));
        MOVE_TO_ERROR_STATE(CURLE_COULDNT_CONNECT);
        break;
      }

      rc = sftp_init(sshc->sftp_session);
      if(rc != SSH_OK) {
        rc = sftp_get_error(sshc->sftp_session);
        failf(data, "Failure initializing sftp session: %s",
              ssh_get_error(sshc->ssh_session));
        MOVE_TO_ERROR_STATE(sftp_error_to_CURLE(rc));
        break;
      }
      state(data, SSH_SFTP_REALPATH);
      /* FALLTHROUGH */
    case SSH_SFTP_REALPATH:
      /*
       * Get the "home" directory
       */
      sshc->homedir = sftp_canonicalize_path(sshc->sftp_session, ".");
      if(!sshc->homedir) {
        MOVE_TO_ERROR_STATE(CURLE_COULDNT_CONNECT);
        break;
      }
      data->state.most_recent_ftp_entrypath = sshc->homedir;

      /* This is the last step in the SFTP connect phase. Do note that while
         we get the homedir here, we get the "workingpath" in the DO action
         since the homedir will remain the same between request but the
         working path will not. */
      DEBUGF(infof(data, "SSH CONNECT phase done"));
      state(data, SSH_STOP);
      break;

    case SSH_SFTP_QUOTE_INIT:
      result = Curl_getworkingpath(data, sshc->homedir, &protop->path);
      if(result) {
        sshc->actualcode = result;
        state(data, SSH_STOP);
        break;
      }

      if(data->set.quote) {
        infof(data, "Sending quote commands");
        sshc->quote_item = data->set.quote;
        state(data, SSH_SFTP_QUOTE);
      }
      else {
        state(data, SSH_SFTP_GETINFO);
      }
      break;

    case SSH_SFTP_POSTQUOTE_INIT:
      if(data->set.postquote) {
        infof(data, "Sending quote commands");
        sshc->quote_item = data->set.postquote;
        state(data, SSH_SFTP_QUOTE);
      }
      else {
        state(data, SSH_STOP);
      }
      break;

    case SSH_SFTP_QUOTE:
      /* Send any quote commands */
      sftp_quote(data);
      break;

    case SSH_SFTP_NEXT_QUOTE:
      Curl_safefree(sshc->quote_path1);
      Curl_safefree(sshc->quote_path2);

      sshc->quote_item = sshc->quote_item->next;

      if(sshc->quote_item) {
        state(data, SSH_SFTP_QUOTE);
      }
      else {
        if(sshc->nextstate != SSH_NO_STATE) {
          state(data, sshc->nextstate);
          sshc->nextstate = SSH_NO_STATE;
        }
        else {
          state(data, SSH_SFTP_GETINFO);
        }
      }
      break;

    case SSH_SFTP_QUOTE_STAT:
      sftp_quote_stat(data);
      break;

    case SSH_SFTP_QUOTE_SETSTAT:
      rc = sftp_setstat(sshc->sftp_session, sshc->quote_path2,
                        sshc->quote_attrs);
      if(rc && !sshc->acceptfail) {
        Curl_safefree(sshc->quote_path1);
        Curl_safefree(sshc->quote_path2);
        failf(data, "Attempt to set SFTP stats failed: %s",
              ssh_get_error(sshc->ssh_session));
        state(data, SSH_SFTP_CLOSE);
        sshc->nextstate = SSH_NO_STATE;
        sshc->actualcode = CURLE_QUOTE_ERROR;
        /* sshc->actualcode = sftp_error_to_CURLE(err);
         * we do not send the actual error; we return
         * the error the libssh2 backend is returning */
        break;
      }
      state(data, SSH_SFTP_NEXT_QUOTE);
      break;

    case SSH_SFTP_QUOTE_SYMLINK:
      rc = sftp_symlink(sshc->sftp_session, sshc->quote_path2,
                        sshc->quote_path1);
      if(rc && !sshc->acceptfail) {
        Curl_safefree(sshc->quote_path1);
        Curl_safefree(sshc->quote_path2);
        failf(data, "symlink command failed: %s",
              ssh_get_error(sshc->ssh_session));
        state(data, SSH_SFTP_CLOSE);
        sshc->nextstate = SSH_NO_STATE;
        sshc->actualcode = CURLE_QUOTE_ERROR;
        break;
      }
      state(data, SSH_SFTP_NEXT_QUOTE);
      break;

    case SSH_SFTP_QUOTE_MKDIR:
      rc = sftp_mkdir(sshc->sftp_session, sshc->quote_path1,
                      (mode_t)data->set.new_directory_perms);
      if(rc && !sshc->acceptfail) {
        Curl_safefree(sshc->quote_path1);
        failf(data, "mkdir command failed: %s",
              ssh_get_error(sshc->ssh_session));
        state(data, SSH_SFTP_CLOSE);
        sshc->nextstate = SSH_NO_STATE;
        sshc->actualcode = CURLE_QUOTE_ERROR;
        break;
      }
      state(data, SSH_SFTP_NEXT_QUOTE);
      break;

    case SSH_SFTP_QUOTE_RENAME:
      rc = sftp_rename(sshc->sftp_session, sshc->quote_path1,
                       sshc->quote_path2);
      if(rc && !sshc->acceptfail) {
        Curl_safefree(sshc->quote_path1);
        Curl_safefree(sshc->quote_path2);
        failf(data, "rename command failed: %s",
              ssh_get_error(sshc->ssh_session));
        state(data, SSH_SFTP_CLOSE);
        sshc->nextstate = SSH_NO_STATE;
        sshc->actualcode = CURLE_QUOTE_ERROR;
        break;
      }
      state(data, SSH_SFTP_NEXT_QUOTE);
      break;

    case SSH_SFTP_QUOTE_RMDIR:
      rc = sftp_rmdir(sshc->sftp_session, sshc->quote_path1);
      if(rc && !sshc->acceptfail) {
        Curl_safefree(sshc->quote_path1);
        failf(data, "rmdir command failed: %s",
              ssh_get_error(sshc->ssh_session));
        state(data, SSH_SFTP_CLOSE);
        sshc->nextstate = SSH_NO_STATE;
        sshc->actualcode = CURLE_QUOTE_ERROR;
        break;
      }
      state(data, SSH_SFTP_NEXT_QUOTE);
      break;

    case SSH_SFTP_QUOTE_UNLINK:
      rc = sftp_unlink(sshc->sftp_session, sshc->quote_path1);
      if(rc && !sshc->acceptfail) {
        Curl_safefree(sshc->quote_path1);
        failf(data, "rm command failed: %s",
              ssh_get_error(sshc->ssh_session));
        state(data, SSH_SFTP_CLOSE);
        sshc->nextstate = SSH_NO_STATE;
        sshc->actualcode = CURLE_QUOTE_ERROR;
        break;
      }
      state(data, SSH_SFTP_NEXT_QUOTE);
      break;

    case SSH_SFTP_QUOTE_STATVFS:
    {
      sftp_statvfs_t statvfs;

      statvfs = sftp_statvfs(sshc->sftp_session, sshc->quote_path1);
      if(!statvfs && !sshc->acceptfail) {
        Curl_safefree(sshc->quote_path1);
        failf(data, "statvfs command failed: %s",
              ssh_get_error(sshc->ssh_session));
        state(data, SSH_SFTP_CLOSE);
        sshc->nextstate = SSH_NO_STATE;
        sshc->actualcode = CURLE_QUOTE_ERROR;
        break;
      }
      else if(statvfs) {
        char *tmp = aprintf("statvfs:\n"
                            "f_bsize: %llu\n" "f_frsize: %llu\n"
                            "f_blocks: %llu\n" "f_bfree: %llu\n"
                            "f_bavail: %llu\n" "f_files: %llu\n"
                            "f_ffree: %llu\n" "f_favail: %llu\n"
                            "f_fsid: %llu\n" "f_flag: %llu\n"
                            "f_namemax: %llu\n",
                            statvfs->f_bsize, statvfs->f_frsize,
                            statvfs->f_blocks, statvfs->f_bfree,
                            statvfs->f_bavail, statvfs->f_files,
                            statvfs->f_ffree, statvfs->f_favail,
                            statvfs->f_fsid, statvfs->f_flag,
                            statvfs->f_namemax);
        sftp_statvfs_free(statvfs);

        if(!tmp) {
          result = CURLE_OUT_OF_MEMORY;
          state(data, SSH_SFTP_CLOSE);
          sshc->nextstate = SSH_NO_STATE;
          break;
        }

        result = Curl_client_write(data, CLIENTWRITE_HEADER, tmp, strlen(tmp));
        free(tmp);
        if(result) {
          state(data, SSH_SFTP_CLOSE);
          sshc->nextstate = SSH_NO_STATE;
          sshc->actualcode = result;
        }
      }
      state(data, SSH_SFTP_NEXT_QUOTE);
      break;
    }

    case SSH_SFTP_GETINFO:
      if(data->set.get_filetime) {
        state(data, SSH_SFTP_FILETIME);
      }
      else {
        state(data, SSH_SFTP_TRANS_INIT);
      }
      break;

    case SSH_SFTP_FILETIME:
    {
      sftp_attributes attrs;

      attrs = sftp_stat(sshc->sftp_session, protop->path);
      if(attrs) {
        data->info.filetime = attrs->mtime;
        sftp_attributes_free(attrs);
      }

      state(data, SSH_SFTP_TRANS_INIT);
      break;
    }

    case SSH_SFTP_TRANS_INIT:
      if(data->set.upload)
        state(data, SSH_SFTP_UPLOAD_INIT);
      else {
        if(protop->path[strlen(protop->path)-1] == '/')
          state(data, SSH_SFTP_READDIR_INIT);
        else
          state(data, SSH_SFTP_DOWNLOAD_INIT);
      }
      break;

    case SSH_SFTP_UPLOAD_INIT:
    {
      int flags;

      if(data->state.resume_from) {
        sftp_attributes attrs;

        if(data->state.resume_from < 0) {
          attrs = sftp_stat(sshc->sftp_session, protop->path);
          if(attrs) {
            curl_off_t size = attrs->size;
            if(size < 0) {
              failf(data, "Bad file size (%" CURL_FORMAT_CURL_OFF_T ")", size);
              MOVE_TO_ERROR_STATE(CURLE_BAD_DOWNLOAD_RESUME);
              break;
            }
            data->state.resume_from = attrs->size;

            sftp_attributes_free(attrs);
          }
          else {
            data->state.resume_from = 0;
          }
        }
      }

      if(data->set.remote_append)
        /* Try to open for append, but create if nonexisting */
        flags = O_WRONLY|O_CREAT|O_APPEND;
      else if(data->state.resume_from > 0)
        /* If we have restart position then open for append */
        flags = O_WRONLY|O_APPEND;
      else
        /* Clear file before writing (normal behavior) */
        flags = O_WRONLY|O_CREAT|O_TRUNC;

      if(sshc->sftp_file)
        sftp_close(sshc->sftp_file);
      sshc->sftp_file =
        sftp_open(sshc->sftp_session, protop->path,
                  flags, (mode_t)data->set.new_file_perms);
      if(!sshc->sftp_file) {
        err = sftp_get_error(sshc->sftp_session);

        if(((err == SSH_FX_NO_SUCH_FILE || err == SSH_FX_FAILURE ||
             err == SSH_FX_NO_SUCH_PATH)) &&
             (data->set.ftp_create_missing_dirs &&
             (strlen(protop->path) > 1))) {
               /* try to create the path remotely */
               rc = 0;
               sshc->secondCreateDirs = 1;
               state(data, SSH_SFTP_CREATE_DIRS_INIT);
               break;
        }
        else {
          MOVE_TO_SFTP_CLOSE_STATE();
          break;
        }
      }

      /* If we have a restart point then we need to seek to the correct
         position. */
      if(data->state.resume_from > 0) {
        /* Let's read off the proper amount of bytes from the input. */
        if(conn->seek_func) {
          Curl_set_in_callback(data, true);
          seekerr = conn->seek_func(conn->seek_client, data->state.resume_from,
                                    SEEK_SET);
          Curl_set_in_callback(data, false);
        }

        if(seekerr != CURL_SEEKFUNC_OK) {
          curl_off_t passed = 0;

          if(seekerr != CURL_SEEKFUNC_CANTSEEK) {
            failf(data, "Could not seek stream");
            return CURLE_FTP_COULDNT_USE_REST;
          }
          /* seekerr == CURL_SEEKFUNC_CANTSEEK (can't seek to offset) */
          do {
            size_t readthisamountnow =
              (data->state.resume_from - passed > data->set.buffer_size) ?
              (size_t)data->set.buffer_size :
              curlx_sotouz(data->state.resume_from - passed);

            size_t actuallyread =
              data->state.fread_func(data->state.buffer, 1,
                                     readthisamountnow, data->state.in);

            passed += actuallyread;
            if((actuallyread == 0) || (actuallyread > readthisamountnow)) {
              /* this checks for greater-than only to make sure that the
                 CURL_READFUNC_ABORT return code still aborts */
              failf(data, "Failed to read data");
              MOVE_TO_ERROR_STATE(CURLE_FTP_COULDNT_USE_REST);
              break;
            }
          } while(passed < data->state.resume_from);
          if(rc)
            break;
        }

        /* now, decrease the size of the read */
        if(data->state.infilesize > 0) {
          data->state.infilesize -= data->state.resume_from;
          data->req.size = data->state.infilesize;
          Curl_pgrsSetUploadSize(data, data->state.infilesize);
        }

        rc = sftp_seek64(sshc->sftp_file, data->state.resume_from);
        if(rc) {
          MOVE_TO_SFTP_CLOSE_STATE();
          break;
        }
      }
      if(data->state.infilesize > 0) {
        data->req.size = data->state.infilesize;
        Curl_pgrsSetUploadSize(data, data->state.infilesize);
      }
      /* upload data */
      Curl_setup_transfer(data, -1, -1, FALSE, FIRSTSOCKET);

      /* not set by Curl_setup_transfer to preserve keepon bits */
      conn->sockfd = conn->writesockfd;

      /* store this original bitmask setup to use later on if we can't
         figure out a "real" bitmask */
      sshc->orig_waitfor = data->req.keepon;

      /* we want to use the _sending_ function even when the socket turns
         out readable as the underlying libssh sftp send function will deal
         with both accordingly */
      conn->cselect_bits = CURL_CSELECT_OUT;

      /* since we don't really wait for anything at this point, we want the
         state machine to move on as soon as possible so we set a very short
         timeout here */
      Curl_expire(data, 0, EXPIRE_RUN_NOW);

      state(data, SSH_STOP);
      break;
    }

    case SSH_SFTP_CREATE_DIRS_INIT:
      if(strlen(protop->path) > 1) {
        sshc->slash_pos = protop->path + 1; /* ignore the leading '/' */
        state(data, SSH_SFTP_CREATE_DIRS);
      }
      else {
        state(data, SSH_SFTP_UPLOAD_INIT);
      }
      break;

    case SSH_SFTP_CREATE_DIRS:
      sshc->slash_pos = strchr(sshc->slash_pos, '/');
      if(sshc->slash_pos) {
        *sshc->slash_pos = 0;

        infof(data, "Creating directory '%s'", protop->path);
        state(data, SSH_SFTP_CREATE_DIRS_MKDIR);
        break;
      }
      state(data, SSH_SFTP_UPLOAD_INIT);
      break;

    case SSH_SFTP_CREATE_DIRS_MKDIR:
      /* 'mode' - parameter is preliminary - default to 0644 */
      rc = sftp_mkdir(sshc->sftp_session, protop->path,
                      (mode_t)data->set.new_directory_perms);
      *sshc->slash_pos = '/';
      ++sshc->slash_pos;
      if(rc < 0) {
        /*
         * Abort if failure wasn't that the dir already exists or the
         * permission was denied (creation might succeed further down the
         * path) - retry on unspecific FAILURE also
         */
        err = sftp_get_error(sshc->sftp_session);
        if((err != SSH_FX_FILE_ALREADY_EXISTS) &&
           (err != SSH_FX_FAILURE) &&
           (err != SSH_FX_PERMISSION_DENIED)) {
          MOVE_TO_SFTP_CLOSE_STATE();
          break;
        }
        rc = 0; /* clear rc and continue */
      }
      state(data, SSH_SFTP_CREATE_DIRS);
      break;

    case SSH_SFTP_READDIR_INIT:
      Curl_pgrsSetDownloadSize(data, -1);
      if(data->set.opt_no_body) {
        state(data, SSH_STOP);
        break;
      }

      /*
       * This is a directory that we are trying to get, so produce a directory
       * listing
       */
      sshc->sftp_dir = sftp_opendir(sshc->sftp_session,
                                    protop->path);
      if(!sshc->sftp_dir) {
        failf(data, "Could not open directory for reading: %s",
              ssh_get_error(sshc->ssh_session));
        MOVE_TO_SFTP_CLOSE_STATE();
        break;
      }
      state(data, SSH_SFTP_READDIR);
      break;

    case SSH_SFTP_READDIR:

      if(sshc->readdir_attrs)
        sftp_attributes_free(sshc->readdir_attrs);

      sshc->readdir_attrs = sftp_readdir(sshc->sftp_session, sshc->sftp_dir);
      if(sshc->readdir_attrs) {
        sshc->readdir_filename = sshc->readdir_attrs->name;
        sshc->readdir_longentry = sshc->readdir_attrs->longname;
        sshc->readdir_len = strlen(sshc->readdir_filename);

        if(data->set.list_only) {
          char *tmpLine;

          tmpLine = aprintf("%s\n", sshc->readdir_filename);
          if(!tmpLine) {
            state(data, SSH_SFTP_CLOSE);
            sshc->actualcode = CURLE_OUT_OF_MEMORY;
            break;
          }
          result = Curl_client_write(data, CLIENTWRITE_BODY,
                                     tmpLine, sshc->readdir_len + 1);
          free(tmpLine);

          if(result) {
            state(data, SSH_STOP);
            break;
          }
          /* since this counts what we send to the client, we include the
             newline in this counter */
          data->req.bytecount += sshc->readdir_len + 1;

          /* output debug output if that is requested */
          Curl_debug(data, CURLINFO_DATA_OUT, (char *)sshc->readdir_filename,
                     sshc->readdir_len);
        }
        else {
          sshc->readdir_currLen = strlen(sshc->readdir_longentry);
          sshc->readdir_totalLen = 80 + sshc->readdir_currLen;
          sshc->readdir_line = calloc(sshc->readdir_totalLen, 1);
          if(!sshc->readdir_line) {
            state(data, SSH_SFTP_CLOSE);
            sshc->actualcode = CURLE_OUT_OF_MEMORY;
            break;
          }

          memcpy(sshc->readdir_line, sshc->readdir_longentry,
                 sshc->readdir_currLen);
          if((sshc->readdir_attrs->flags & SSH_FILEXFER_ATTR_PERMISSIONS) &&
             ((sshc->readdir_attrs->permissions & S_IFMT) ==
              S_IFLNK)) {
            sshc->readdir_linkPath = aprintf("%s%s", protop->path,
                                             sshc->readdir_filename);

            if(!sshc->readdir_linkPath) {
              state(data, SSH_SFTP_CLOSE);
              sshc->actualcode = CURLE_OUT_OF_MEMORY;
              break;
            }

            state(data, SSH_SFTP_READDIR_LINK);
            break;
          }
          state(data, SSH_SFTP_READDIR_BOTTOM);
          break;
        }
      }
      else if(sftp_dir_eof(sshc->sftp_dir)) {
        state(data, SSH_SFTP_READDIR_DONE);
        break;
      }
      else {
        failf(data, "Could not open remote file for reading: %s",
              ssh_get_error(sshc->ssh_session));
        MOVE_TO_SFTP_CLOSE_STATE();
        break;
      }
      break;

    case SSH_SFTP_READDIR_LINK:
      if(sshc->readdir_link_attrs)
        sftp_attributes_free(sshc->readdir_link_attrs);

      sshc->readdir_link_attrs = sftp_lstat(sshc->sftp_session,
                                            sshc->readdir_linkPath);
      if(sshc->readdir_link_attrs == 0) {
        failf(data, "Could not read symlink for reading: %s",
              ssh_get_error(sshc->ssh_session));
        MOVE_TO_SFTP_CLOSE_STATE();
        break;
      }

      if(!sshc->readdir_link_attrs->name) {
        sshc->readdir_tmp = sftp_readlink(sshc->sftp_session,
                                          sshc->readdir_linkPath);
        if(!sshc->readdir_filename)
          sshc->readdir_len = 0;
        else
          sshc->readdir_len = strlen(sshc->readdir_tmp);
        sshc->readdir_longentry = NULL;
        sshc->readdir_filename = sshc->readdir_tmp;
      }
      else {
        sshc->readdir_len = strlen(sshc->readdir_link_attrs->name);
        sshc->readdir_filename = sshc->readdir_link_attrs->name;
        sshc->readdir_longentry = sshc->readdir_link_attrs->longname;
      }

      Curl_safefree(sshc->readdir_linkPath);

      /* get room for the filename and extra output */
      sshc->readdir_totalLen += 4 + sshc->readdir_len;
      new_readdir_line = Curl_saferealloc(sshc->readdir_line,
                                          sshc->readdir_totalLen);
      if(!new_readdir_line) {
        sshc->readdir_line = NULL;
        state(data, SSH_SFTP_CLOSE);
        sshc->actualcode = CURLE_OUT_OF_MEMORY;
        break;
      }
      sshc->readdir_line = new_readdir_line;

      sshc->readdir_currLen += msnprintf(sshc->readdir_line +
                                         sshc->readdir_currLen,
                                         sshc->readdir_totalLen -
                                         sshc->readdir_currLen,
                                         " -> %s",
                                         sshc->readdir_filename);

      sftp_attributes_free(sshc->readdir_link_attrs);
      sshc->readdir_link_attrs = NULL;
      sshc->readdir_filename = NULL;
      sshc->readdir_longentry = NULL;

      state(data, SSH_SFTP_READDIR_BOTTOM);
      /* FALLTHROUGH */
    case SSH_SFTP_READDIR_BOTTOM:
      sshc->readdir_currLen += msnprintf(sshc->readdir_line +
                                         sshc->readdir_currLen,
                                         sshc->readdir_totalLen -
                                         sshc->readdir_currLen, "\n");
      result = Curl_client_write(data, CLIENTWRITE_BODY,
                                 sshc->readdir_line,
                                 sshc->readdir_currLen);

      if(!result) {
        /* output debug output if that is requested */
        Curl_debug(data, CURLINFO_DATA_OUT, sshc->readdir_line,
                   sshc->readdir_currLen);
        data->req.bytecount += sshc->readdir_currLen;
      }
      Curl_safefree(sshc->readdir_line);
      ssh_string_free_char(sshc->readdir_tmp);
      sshc->readdir_tmp = NULL;

      if(result) {
        state(data, SSH_STOP);
      }
      else
        state(data, SSH_SFTP_READDIR);
      break;

    case SSH_SFTP_READDIR_DONE:
      sftp_closedir(sshc->sftp_dir);
      sshc->sftp_dir = NULL;

      /* no data to transfer */
      Curl_setup_transfer(data, -1, -1, FALSE, -1);
      state(data, SSH_STOP);
      break;

    case SSH_SFTP_DOWNLOAD_INIT:
      /*
       * Work on getting the specified file
       */
      if(sshc->sftp_file)
        sftp_close(sshc->sftp_file);

      sshc->sftp_file = sftp_open(sshc->sftp_session, protop->path,
                                  O_RDONLY, (mode_t)data->set.new_file_perms);
      if(!sshc->sftp_file) {
        failf(data, "Could not open remote file for reading: %s",
              ssh_get_error(sshc->ssh_session));

        MOVE_TO_SFTP_CLOSE_STATE();
        break;
      }

      state(data, SSH_SFTP_DOWNLOAD_STAT);
      break;

    case SSH_SFTP_DOWNLOAD_STAT:
    {
      sftp_attributes attrs;
      curl_off_t size;

      attrs = sftp_fstat(sshc->sftp_file);
      if(!attrs ||
              !(attrs->flags & SSH_FILEXFER_ATTR_SIZE) ||
              (attrs->size == 0)) {
        /*
         * sftp_fstat didn't return an error, so maybe the server
         * just doesn't support stat()
         * OR the server doesn't return a file size with a stat()
         * OR file size is 0
         */
        data->req.size = -1;
        data->req.maxdownload = -1;
        Curl_pgrsSetDownloadSize(data, -1);
        size = 0;
      }
      else {
        size = attrs->size;

        sftp_attributes_free(attrs);

        if(size < 0) {
          failf(data, "Bad file size (%" CURL_FORMAT_CURL_OFF_T ")", size);
          return CURLE_BAD_DOWNLOAD_RESUME;
        }
        if(data->state.use_range) {
          curl_off_t from, to;
          char *ptr;
          char *ptr2;
          CURLofft to_t;
          CURLofft from_t;

          from_t = curlx_strtoofft(data->state.range, &ptr, 0, &from);
          if(from_t == CURL_OFFT_FLOW) {
            return CURLE_RANGE_ERROR;
          }
          while(*ptr && (ISSPACE(*ptr) || (*ptr == '-')))
            ptr++;
          to_t = curlx_strtoofft(ptr, &ptr2, 0, &to);
          if(to_t == CURL_OFFT_FLOW) {
            return CURLE_RANGE_ERROR;
          }
          if((to_t == CURL_OFFT_INVAL) /* no "to" value given */
             || (to >= size)) {
            to = size - 1;
          }
          if(from_t) {
            /* from is relative to end of file */
            from = size - to;
            to = size - 1;
          }
          if(from > size) {
            failf(data, "Offset (%"
                  CURL_FORMAT_CURL_OFF_T ") was beyond file size (%"
                  CURL_FORMAT_CURL_OFF_T ")", from, size);
            return CURLE_BAD_DOWNLOAD_RESUME;
          }
          if(from > to) {
            from = to;
            size = 0;
          }
          else {
            size = to - from + 1;
          }

          rc = sftp_seek64(sshc->sftp_file, from);
          if(rc) {
            MOVE_TO_SFTP_CLOSE_STATE();
            break;
          }
        }
        data->req.size = size;
        data->req.maxdownload = size;
        Curl_pgrsSetDownloadSize(data, size);
      }

      /* We can resume if we can seek to the resume position */
      if(data->state.resume_from) {
        if(data->state.resume_from < 0) {
          /* We're supposed to download the last abs(from) bytes */
          if((curl_off_t)size < -data->state.resume_from) {
            failf(data, "Offset (%"
                  CURL_FORMAT_CURL_OFF_T ") was beyond file size (%"
                  CURL_FORMAT_CURL_OFF_T ")",
                  data->state.resume_from, size);
            return CURLE_BAD_DOWNLOAD_RESUME;
          }
          /* download from where? */
          data->state.resume_from += size;
        }
        else {
          if((curl_off_t)size < data->state.resume_from) {
            failf(data, "Offset (%" CURL_FORMAT_CURL_OFF_T
                  ") was beyond file size (%" CURL_FORMAT_CURL_OFF_T ")",
                  data->state.resume_from, size);
            return CURLE_BAD_DOWNLOAD_RESUME;
          }
        }
        /* Now store the number of bytes we are expected to download */
        data->req.size = size - data->state.resume_from;
        data->req.maxdownload = size - data->state.resume_from;
        Curl_pgrsSetDownloadSize(data,
                                 size - data->state.resume_from);

        rc = sftp_seek64(sshc->sftp_file, data->state.resume_from);
        if(rc) {
          MOVE_TO_SFTP_CLOSE_STATE();
          break;
        }
      }
    }

    /* Setup the actual download */
    if(data->req.size == 0) {
      /* no data to transfer */
      Curl_setup_transfer(data, -1, -1, FALSE, -1);
      infof(data, "File already completely downloaded");
      state(data, SSH_STOP);
      break;
    }
    Curl_setup_transfer(data, FIRSTSOCKET, data->req.size, FALSE, -1);

    /* not set by Curl_setup_transfer to preserve keepon bits */
    conn->writesockfd = conn->sockfd;

    /* we want to use the _receiving_ function even when the socket turns
       out writableable as the underlying libssh recv function will deal
       with both accordingly */
    conn->cselect_bits = CURL_CSELECT_IN;

    if(result) {
      /* this should never occur; the close state should be entered
         at the time the error occurs */
      state(data, SSH_SFTP_CLOSE);
      sshc->actualcode = result;
    }
    else {
      sshc->sftp_recv_state = 0;
      state(data, SSH_STOP);
    }
    break;

    case SSH_SFTP_CLOSE:
      if(sshc->sftp_file) {
        sftp_close(sshc->sftp_file);
        sshc->sftp_file = NULL;
      }
      Curl_safefree(protop->path);

      DEBUGF(infof(data, "SFTP DONE done"));

      /* Check if nextstate is set and move .nextstate could be POSTQUOTE_INIT
         After nextstate is executed, the control should come back to
         SSH_SFTP_CLOSE to pass the correct result back  */
      if(sshc->nextstate != SSH_NO_STATE &&
         sshc->nextstate != SSH_SFTP_CLOSE) {
        state(data, sshc->nextstate);
        sshc->nextstate = SSH_SFTP_CLOSE;
      }
      else {
        state(data, SSH_STOP);
        result = sshc->actualcode;
      }
      break;

    case SSH_SFTP_SHUTDOWN:
      /* during times we get here due to a broken transfer and then the
         sftp_handle might not have been taken down so make sure that is done
         before we proceed */

      if(sshc->sftp_file) {
        sftp_close(sshc->sftp_file);
        sshc->sftp_file = NULL;
      }

      if(sshc->sftp_session) {
        sftp_free(sshc->sftp_session);
        sshc->sftp_session = NULL;
      }

      SSH_STRING_FREE_CHAR(sshc->homedir);
      data->state.most_recent_ftp_entrypath = NULL;

      state(data, SSH_SESSION_DISCONNECT);
      break;

    case SSH_SCP_TRANS_INIT:
      result = Curl_getworkingpath(data, sshc->homedir, &protop->path);
      if(result) {
        sshc->actualcode = result;
        state(data, SSH_STOP);
        break;
      }

      /* Functions from the SCP subsystem cannot handle/return SSH_AGAIN */
      ssh_set_blocking(sshc->ssh_session, 1);

      if(data->set.upload) {
        if(data->state.infilesize < 0) {
          failf(data, "SCP requires a known file size for upload");
          sshc->actualcode = CURLE_UPLOAD_FAILED;
          MOVE_TO_ERROR_STATE(CURLE_UPLOAD_FAILED);
          break;
        }

        sshc->scp_session =
          ssh_scp_new(sshc->ssh_session, SSH_SCP_WRITE, protop->path);
        state(data, SSH_SCP_UPLOAD_INIT);
      }
      else {
        sshc->scp_session =
          ssh_scp_new(sshc->ssh_session, SSH_SCP_READ, protop->path);
        state(data, SSH_SCP_DOWNLOAD_INIT);
      }

      if(!sshc->scp_session) {
        err_msg = ssh_get_error(sshc->ssh_session);
        failf(data, "%s", err_msg);
        MOVE_TO_ERROR_STATE(CURLE_UPLOAD_FAILED);
      }

      break;

    case SSH_SCP_UPLOAD_INIT:

      rc = ssh_scp_init(sshc->scp_session);
      if(rc != SSH_OK) {
        err_msg = ssh_get_error(sshc->ssh_session);
        failf(data, "%s", err_msg);
        MOVE_TO_ERROR_STATE(CURLE_UPLOAD_FAILED);
        break;
      }

      rc = ssh_scp_push_file(sshc->scp_session, protop->path,
                             data->state.infilesize,
                             (int)data->set.new_file_perms);
      if(rc != SSH_OK) {
        err_msg = ssh_get_error(sshc->ssh_session);
        failf(data, "%s", err_msg);
        MOVE_TO_ERROR_STATE(CURLE_UPLOAD_FAILED);
        break;
      }

      /* upload data */
      Curl_setup_transfer(data, -1, data->req.size, FALSE, FIRSTSOCKET);

      /* not set by Curl_setup_transfer to preserve keepon bits */
      conn->sockfd = conn->writesockfd;

      /* store this original bitmask setup to use later on if we can't
         figure out a "real" bitmask */
      sshc->orig_waitfor = data->req.keepon;

      /* we want to use the _sending_ function even when the socket turns
         out readable as the underlying libssh scp send function will deal
         with both accordingly */
      conn->cselect_bits = CURL_CSELECT_OUT;

      state(data, SSH_STOP);

      break;

    case SSH_SCP_DOWNLOAD_INIT:

      rc = ssh_scp_init(sshc->scp_session);
      if(rc != SSH_OK) {
        err_msg = ssh_get_error(sshc->ssh_session);
        failf(data, "%s", err_msg);
        MOVE_TO_ERROR_STATE(CURLE_COULDNT_CONNECT);
        break;
      }
      state(data, SSH_SCP_DOWNLOAD);
      /* FALLTHROUGH */

    case SSH_SCP_DOWNLOAD:{
        curl_off_t bytecount;

        rc = ssh_scp_pull_request(sshc->scp_session);
        if(rc != SSH_SCP_REQUEST_NEWFILE) {
          err_msg = ssh_get_error(sshc->ssh_session);
          failf(data, "%s", err_msg);
          MOVE_TO_ERROR_STATE(CURLE_REMOTE_FILE_NOT_FOUND);
          break;
        }

        /* download data */
        bytecount = ssh_scp_request_get_size(sshc->scp_session);
        data->req.maxdownload = (curl_off_t) bytecount;
        Curl_setup_transfer(data, FIRSTSOCKET, bytecount, FALSE, -1);

        /* not set by Curl_setup_transfer to preserve keepon bits */
        conn->writesockfd = conn->sockfd;

        /* we want to use the _receiving_ function even when the socket turns
           out writableable as the underlying libssh recv function will deal
           with both accordingly */
        conn->cselect_bits = CURL_CSELECT_IN;

        state(data, SSH_STOP);
        break;
      }
    case SSH_SCP_DONE:
      if(data->set.upload)
        state(data, SSH_SCP_SEND_EOF);
      else
        state(data, SSH_SCP_CHANNEL_FREE);
      break;

    case SSH_SCP_SEND_EOF:
      if(sshc->scp_session) {
        rc = ssh_scp_close(sshc->scp_session);
        if(rc == SSH_AGAIN) {
          /* Currently the ssh_scp_close handles waiting for EOF in
           * blocking way.
           */
          break;
        }
        if(rc != SSH_OK) {
          infof(data, "Failed to close libssh scp channel: %s",
                ssh_get_error(sshc->ssh_session));
        }
      }

      state(data, SSH_SCP_CHANNEL_FREE);
      break;

    case SSH_SCP_CHANNEL_FREE:
      if(sshc->scp_session) {
        ssh_scp_free(sshc->scp_session);
        sshc->scp_session = NULL;
      }
      DEBUGF(infof(data, "SCP DONE phase complete"));

      ssh_set_blocking(sshc->ssh_session, 0);

      state(data, SSH_SESSION_DISCONNECT);
      /* FALLTHROUGH */

    case SSH_SESSION_DISCONNECT:
      /* during weird times when we've been prematurely aborted, the channel
         is still alive when we reach this state and we MUST kill the channel
         properly first */
      if(sshc->scp_session) {
        ssh_scp_free(sshc->scp_session);
        sshc->scp_session = NULL;
      }

      ssh_disconnect(sshc->ssh_session);

      SSH_STRING_FREE_CHAR(sshc->homedir);
      data->state.most_recent_ftp_entrypath = NULL;

      state(data, SSH_SESSION_FREE);
      /* FALLTHROUGH */
    case SSH_SESSION_FREE:
      if(sshc->ssh_session) {
        ssh_free(sshc->ssh_session);
        sshc->ssh_session = NULL;
      }

      /* worst-case scenario cleanup */

      DEBUGASSERT(sshc->ssh_session == NULL);
      DEBUGASSERT(sshc->scp_session == NULL);

      if(sshc->readdir_tmp) {
        ssh_string_free_char(sshc->readdir_tmp);
        sshc->readdir_tmp = NULL;
      }

      if(sshc->quote_attrs)
        sftp_attributes_free(sshc->quote_attrs);

      if(sshc->readdir_attrs)
        sftp_attributes_free(sshc->readdir_attrs);

      if(sshc->readdir_link_attrs)
        sftp_attributes_free(sshc->readdir_link_attrs);

      if(sshc->privkey)
        ssh_key_free(sshc->privkey);
      if(sshc->pubkey)
        ssh_key_free(sshc->pubkey);

      Curl_safefree(sshc->rsa_pub);
      Curl_safefree(sshc->rsa);
      Curl_safefree(sshc->quote_path1);
      Curl_safefree(sshc->quote_path2);
      Curl_safefree(sshc->readdir_line);
      Curl_safefree(sshc->readdir_linkPath);
      SSH_STRING_FREE_CHAR(sshc->homedir);

      /* the code we are about to return */
      result = sshc->actualcode;

      memset(sshc, 0, sizeof(struct ssh_conn));

      connclose(conn, "SSH session free");
      sshc->state = SSH_SESSION_FREE;   /* current */
      sshc->nextstate = SSH_NO_STATE;
      state(data, SSH_STOP);
      break;

    case SSH_QUIT:
      /* fallthrough, just stop! */
    default:
      /* internal error */
      sshc->nextstate = SSH_NO_STATE;
      state(data, SSH_STOP);
      break;

    }
  } while(!rc && (sshc->state != SSH_STOP));


  if(rc == SSH_AGAIN) {
    /* we would block, we need to wait for the socket to be ready (in the
       right direction too)! */
    *block = TRUE;
  }

  return result;
}


/* called by the multi interface to figure out what socket(s) to wait for and
   for what actions in the DO_DONE, PERFORM and WAITPERFORM states */
static int myssh_getsock(struct Curl_easy *data,
                         struct connectdata *conn,
                         curl_socket_t *sock)
{
  int bitmap = GETSOCK_BLANK;
  (void)data;
  sock[0] = conn->sock[FIRSTSOCKET];

  if(conn->waitfor & KEEP_RECV)
    bitmap |= GETSOCK_READSOCK(FIRSTSOCKET);

  if(conn->waitfor & KEEP_SEND)
    bitmap |= GETSOCK_WRITESOCK(FIRSTSOCKET);

  return bitmap;
}

static void myssh_block2waitfor(struct connectdata *conn, bool block)
{
  struct ssh_conn *sshc = &conn->proto.sshc;

  /* If it didn't block, or nothing was returned by ssh_get_poll_flags
   * have the original set */
  conn->waitfor = sshc->orig_waitfor;

  if(block) {
    int dir = ssh_get_poll_flags(sshc->ssh_session);
    if(dir & SSH_READ_PENDING) {
      /* translate the libssh define bits into our own bit defines */
      conn->waitfor = KEEP_RECV;
    }
    else if(dir & SSH_WRITE_PENDING) {
      conn->waitfor = KEEP_SEND;
    }
  }
}

/* called repeatedly until done from multi.c */
static CURLcode myssh_multi_statemach(struct Curl_easy *data,
                                      bool *done)
{
  struct connectdata *conn = data->conn;
  struct ssh_conn *sshc = &conn->proto.sshc;
  bool block;    /* we store the status and use that to provide a ssh_getsock()
                    implementation */
  CURLcode result = myssh_statemach_act(data, &block);

  *done = (sshc->state == SSH_STOP) ? TRUE : FALSE;
  myssh_block2waitfor(conn, block);

  return result;
}

static CURLcode myssh_block_statemach(struct Curl_easy *data,
                                      bool disconnect)
{
  struct connectdata *conn = data->conn;
  struct ssh_conn *sshc = &conn->proto.sshc;
  CURLcode result = CURLE_OK;

  while((sshc->state != SSH_STOP) && !result) {
    bool block;
    timediff_t left = 1000;
    struct curltime now = Curl_now();

    result = myssh_statemach_act(data, &block);
    if(result)
      break;

    if(!disconnect) {
      if(Curl_pgrsUpdate(data))
        return CURLE_ABORTED_BY_CALLBACK;

      result = Curl_speedcheck(data, now);
      if(result)
        break;

      left = Curl_timeleft(data, NULL, FALSE);
      if(left < 0) {
        failf(data, "Operation timed out");
        return CURLE_OPERATION_TIMEDOUT;
      }
    }

    if(block) {
      curl_socket_t fd_read = conn->sock[FIRSTSOCKET];
      /* wait for the socket to become ready */
      (void) Curl_socket_check(fd_read, CURL_SOCKET_BAD,
                               CURL_SOCKET_BAD, left > 1000 ? 1000 : left);
    }

  }

  return result;
}

/*
 * SSH setup connection
 */
static CURLcode myssh_setup_connection(struct Curl_easy *data,
                                       struct connectdata *conn)
{
  struct SSHPROTO *ssh;
  (void)conn;

  data->req.p.ssh = ssh = calloc(1, sizeof(struct SSHPROTO));
  if(!ssh)
    return CURLE_OUT_OF_MEMORY;

  return CURLE_OK;
}

static Curl_recv scp_recv, sftp_recv;
static Curl_send scp_send, sftp_send;

/*
 * Curl_ssh_connect() gets called from Curl_protocol_connect() to allow us to
 * do protocol-specific actions at connect-time.
 */
static CURLcode myssh_connect(struct Curl_easy *data, bool *done)
{
  struct ssh_conn *ssh;
  CURLcode result;
  struct connectdata *conn = data->conn;
  curl_socket_t sock = conn->sock[FIRSTSOCKET];
  int rc;

  /* initialize per-handle data if not already */
  if(!data->req.p.ssh)
    myssh_setup_connection(data, conn);

  /* We default to persistent connections. We set this already in this connect
     function to make the re-use checks properly be able to check this bit. */
  connkeep(conn, "SSH default");

  if(conn->handler->protocol & CURLPROTO_SCP) {
    conn->recv[FIRSTSOCKET] = scp_recv;
    conn->send[FIRSTSOCKET] = scp_send;
  }
  else {
    conn->recv[FIRSTSOCKET] = sftp_recv;
    conn->send[FIRSTSOCKET] = sftp_send;
  }

  ssh = &conn->proto.sshc;

  ssh->ssh_session = ssh_new();
  if(!ssh->ssh_session) {
    failf(data, "Failure initialising ssh session");
    return CURLE_FAILED_INIT;
  }

  rc = ssh_options_set(ssh->ssh_session, SSH_OPTIONS_HOST, conn->host.name);
  if(rc != SSH_OK) {
    failf(data, "Could not set remote host");
    return CURLE_FAILED_INIT;
  }

  rc = ssh_options_parse_config(ssh->ssh_session, NULL);
  if(rc != SSH_OK) {
    infof(data, "Could not parse SSH configuration files");
    /* ignore */
  }

  rc = ssh_options_set(ssh->ssh_session, SSH_OPTIONS_FD, &sock);
  if(rc != SSH_OK) {
    failf(data, "Could not set socket");
    return CURLE_FAILED_INIT;
  }

  if(conn->user && conn->user[0] != '\0') {
    infof(data, "User: %s", conn->user);
    rc = ssh_options_set(ssh->ssh_session, SSH_OPTIONS_USER, conn->user);
    if(rc != SSH_OK) {
      failf(data, "Could not set user");
      return CURLE_FAILED_INIT;
    }
  }

  if(data->set.str[STRING_SSH_KNOWNHOSTS]) {
    infof(data, "Known hosts: %s", data->set.str[STRING_SSH_KNOWNHOSTS]);
    rc = ssh_options_set(ssh->ssh_session, SSH_OPTIONS_KNOWNHOSTS,
                         data->set.str[STRING_SSH_KNOWNHOSTS]);
    if(rc != SSH_OK) {
      failf(data, "Could not set known hosts file path");
      return CURLE_FAILED_INIT;
    }
  }

  if(conn->remote_port) {
    rc = ssh_options_set(ssh->ssh_session, SSH_OPTIONS_PORT,
                         &conn->remote_port);
    if(rc != SSH_OK) {
      failf(data, "Could not set remote port");
      return CURLE_FAILED_INIT;
    }
  }

  if(data->set.ssh_compression) {
    rc = ssh_options_set(ssh->ssh_session, SSH_OPTIONS_COMPRESSION,
                         "zlib,zlib@openssh.com,none");
    if(rc != SSH_OK) {
      failf(data, "Could not set compression");
      return CURLE_FAILED_INIT;
    }
  }

  ssh->privkey = NULL;
  ssh->pubkey = NULL;

  if(data->set.str[STRING_SSH_PUBLIC_KEY]) {
    rc = ssh_pki_import_pubkey_file(data->set.str[STRING_SSH_PUBLIC_KEY],
                                    &ssh->pubkey);
    if(rc != SSH_OK) {
      failf(data, "Could not load public key file");
      return CURLE_FAILED_INIT;
    }
  }

  /* we do not verify here, we do it at the state machine,
   * after connection */

  state(data, SSH_INIT);

  result = myssh_multi_statemach(data, done);

  return result;
}

/* called from multi.c while DOing */
static CURLcode scp_doing(struct Curl_easy *data, bool *dophase_done)
{
  CURLcode result;

  result = myssh_multi_statemach(data, dophase_done);

  if(*dophase_done) {
    DEBUGF(infof(data, "DO phase is complete"));
  }
  return result;
}

/*
 ***********************************************************************
 *
 * scp_perform()
 *
 * This is the actual DO function for SCP. Get a file according to
 * the options previously setup.
 */

static
CURLcode scp_perform(struct Curl_easy *data,
                     bool *connected, bool *dophase_done)
{
  CURLcode result = CURLE_OK;
  struct connectdata *conn = data->conn;

  DEBUGF(infof(data, "DO phase starts"));

  *dophase_done = FALSE;        /* not done yet */

  /* start the first command in the DO phase */
  state(data, SSH_SCP_TRANS_INIT);

  result = myssh_multi_statemach(data, dophase_done);

  *connected = conn->bits.tcpconnect[FIRSTSOCKET];

  if(*dophase_done) {
    DEBUGF(infof(data, "DO phase is complete"));
  }

  return result;
}

static CURLcode myssh_do_it(struct Curl_easy *data, bool *done)
{
  CURLcode result;
  bool connected = 0;
  struct connectdata *conn = data->conn;
  struct ssh_conn *sshc = &conn->proto.sshc;

  *done = FALSE;                /* default to false */

  data->req.size = -1;          /* make sure this is unknown at this point */

  sshc->actualcode = CURLE_OK;  /* reset error code */
  sshc->secondCreateDirs = 0;   /* reset the create dir attempt state
                                   variable */

  Curl_pgrsSetUploadCounter(data, 0);
  Curl_pgrsSetDownloadCounter(data, 0);
  Curl_pgrsSetUploadSize(data, -1);
  Curl_pgrsSetDownloadSize(data, -1);

  if(conn->handler->protocol & CURLPROTO_SCP)
    result = scp_perform(data, &connected, done);
  else
    result = sftp_perform(data, &connected, done);

  return result;
}

/* BLOCKING, but the function is using the state machine so the only reason
   this is still blocking is that the multi interface code has no support for
   disconnecting operations that takes a while */
static CURLcode scp_disconnect(struct Curl_easy *data,
                               struct connectdata *conn,
                               bool dead_connection)
{
  CURLcode result = CURLE_OK;
  struct ssh_conn *ssh = &conn->proto.sshc;
  (void) dead_connection;

  if(ssh->ssh_session) {
    /* only if there's a session still around to use! */

    state(data, SSH_SESSION_DISCONNECT);

    result = myssh_block_statemach(data, TRUE);
  }

  return result;
}

/* generic done function for both SCP and SFTP called from their specific
   done functions */
static CURLcode myssh_done(struct Curl_easy *data, CURLcode status)
{
  CURLcode result = CURLE_OK;
  struct SSHPROTO *protop = data->req.p.ssh;

  if(!status) {
    /* run the state-machine */
    result = myssh_block_statemach(data, FALSE);
  }
  else
    result = status;

  if(protop)
    Curl_safefree(protop->path);
  if(Curl_pgrsDone(data))
    return CURLE_ABORTED_BY_CALLBACK;

  data->req.keepon = 0;   /* clear all bits */
  return result;
}


static CURLcode scp_done(struct Curl_easy *data, CURLcode status,
                         bool premature)
{
  (void) premature;             /* not used */

  if(!status)
    state(data, SSH_SCP_DONE);

  return myssh_done(data, status);

}

static ssize_t scp_send(struct Curl_easy *data, int sockindex,
                        const void *mem, size_t len, CURLcode *err)
{
  int rc;
  struct connectdata *conn = data->conn;
  (void) sockindex; /* we only support SCP on the fixed known primary socket */
  (void) err;

  rc = ssh_scp_write(conn->proto.sshc.scp_session, mem, len);

#if 0
  /* The following code is misleading, mostly added as wishful thinking
   * that libssh at some point will implement non-blocking ssh_scp_write/read.
   * Currently rc can only be number of bytes read or SSH_ERROR. */
  myssh_block2waitfor(conn, (rc == SSH_AGAIN) ? TRUE : FALSE);

  if(rc == SSH_AGAIN) {
    *err = CURLE_AGAIN;
    return 0;
  }
  else
#endif
  if(rc != SSH_OK) {
    *err = CURLE_SSH;
    return -1;
  }

  return len;
}

static ssize_t scp_recv(struct Curl_easy *data, int sockindex,
                        char *mem, size_t len, CURLcode *err)
{
  ssize_t nread;
  struct connectdata *conn = data->conn;
  (void) err;
  (void) sockindex; /* we only support SCP on the fixed known primary socket */

  /* libssh returns int */
  nread = ssh_scp_read(conn->proto.sshc.scp_session, mem, len);

#if 0
  /* The following code is misleading, mostly added as wishful thinking
   * that libssh at some point will implement non-blocking ssh_scp_write/read.
   * Currently rc can only be SSH_OK or SSH_ERROR. */

  myssh_block2waitfor(conn, (nread == SSH_AGAIN) ? TRUE : FALSE);
  if(nread == SSH_AGAIN) {
    *err = CURLE_AGAIN;
    nread = -1;
  }
#endif

  return nread;
}

/*
 * =============== SFTP ===============
 */

/*
 ***********************************************************************
 *
 * sftp_perform()
 *
 * This is the actual DO function for SFTP. Get a file/directory according to
 * the options previously setup.
 */

static
CURLcode sftp_perform(struct Curl_easy *data,
                      bool *connected,
                      bool *dophase_done)
{
  CURLcode result = CURLE_OK;
  struct connectdata *conn = data->conn;

  DEBUGF(infof(data, "DO phase starts"));

  *dophase_done = FALSE; /* not done yet */

  /* start the first command in the DO phase */
  state(data, SSH_SFTP_QUOTE_INIT);

  /* run the state-machine */
  result = myssh_multi_statemach(data, dophase_done);

  *connected = conn->bits.tcpconnect[FIRSTSOCKET];

  if(*dophase_done) {
    DEBUGF(infof(data, "DO phase is complete"));
  }

  return result;
}

/* called from multi.c while DOing */
static CURLcode sftp_doing(struct Curl_easy *data,
                           bool *dophase_done)
{
  CURLcode result = myssh_multi_statemach(data, dophase_done);
  if(*dophase_done) {
    DEBUGF(infof(data, "DO phase is complete"));
  }
  return result;
}

/* BLOCKING, but the function is using the state machine so the only reason
   this is still blocking is that the multi interface code has no support for
   disconnecting operations that takes a while */
static CURLcode sftp_disconnect(struct Curl_easy *data,
                                struct connectdata *conn,
                                bool dead_connection)
{
  CURLcode result = CURLE_OK;
  (void) dead_connection;

  DEBUGF(infof(data, "SSH DISCONNECT starts now"));

  if(conn->proto.sshc.ssh_session) {
    /* only if there's a session still around to use! */
    state(data, SSH_SFTP_SHUTDOWN);
    result = myssh_block_statemach(data, TRUE);
  }

  DEBUGF(infof(data, "SSH DISCONNECT is done"));

  return result;

}

static CURLcode sftp_done(struct Curl_easy *data, CURLcode status,
                          bool premature)
{
  struct connectdata *conn = data->conn;
  struct ssh_conn *sshc = &conn->proto.sshc;

  if(!status) {
    /* Post quote commands are executed after the SFTP_CLOSE state to avoid
       errors that could happen due to open file handles during POSTQUOTE
       operation */
    if(!premature && data->set.postquote && !conn->bits.retry)
      sshc->nextstate = SSH_SFTP_POSTQUOTE_INIT;
    state(data, SSH_SFTP_CLOSE);
  }
  return myssh_done(data, status);
}

/* return number of sent bytes */
static ssize_t sftp_send(struct Curl_easy *data, int sockindex,
                         const void *mem, size_t len, CURLcode *err)
{
  ssize_t nwrite;
  struct connectdata *conn = data->conn;
  (void)sockindex;

  nwrite = sftp_write(conn->proto.sshc.sftp_file, mem, len);

  myssh_block2waitfor(conn, FALSE);

#if 0 /* not returned by libssh on write */
  if(nwrite == SSH_AGAIN) {
    *err = CURLE_AGAIN;
    nwrite = 0;
  }
  else
#endif
  if(nwrite < 0) {
    *err = CURLE_SSH;
    nwrite = -1;
  }

  return nwrite;
}

/*
 * Return number of received (decrypted) bytes
 * or <0 on error
 */
static ssize_t sftp_recv(struct Curl_easy *data, int sockindex,
                         char *mem, size_t len, CURLcode *err)
{
  ssize_t nread;
  struct connectdata *conn = data->conn;
  (void)sockindex;

  DEBUGASSERT(len < CURL_MAX_READ_SIZE);

  switch(conn->proto.sshc.sftp_recv_state) {
    case 0:
      conn->proto.sshc.sftp_file_index =
        sftp_async_read_begin(conn->proto.sshc.sftp_file,
                              (uint32_t)len);
      if(conn->proto.sshc.sftp_file_index < 0) {
        *err = CURLE_RECV_ERROR;
        return -1;
      }

      /* FALLTHROUGH */
    case 1:
      conn->proto.sshc.sftp_recv_state = 1;

      nread = sftp_async_read(conn->proto.sshc.sftp_file,
                              mem, (uint32_t)len,
                              conn->proto.sshc.sftp_file_index);

      myssh_block2waitfor(conn, (nread == SSH_AGAIN)?TRUE:FALSE);

      if(nread == SSH_AGAIN) {
        *err = CURLE_AGAIN;
        return -1;
      }
      else if(nread < 0) {
        *err = CURLE_RECV_ERROR;
        return -1;
      }

      conn->proto.sshc.sftp_recv_state = 0;
      return nread;

    default:
      /* we never reach here */
      return -1;
  }
}

static void sftp_quote(struct Curl_easy *data)
{
  const char *cp;
  struct connectdata *conn = data->conn;
  struct SSHPROTO *protop = data->req.p.ssh;
  struct ssh_conn *sshc = &conn->proto.sshc;
  CURLcode result;

  /*
   * Support some of the "FTP" commands
   */
  char *cmd = sshc->quote_item->data;
  sshc->acceptfail = FALSE;

  /* if a command starts with an asterisk, which a legal SFTP command never
     can, the command will be allowed to fail without it causing any
     aborts or cancels etc. It will cause libcurl to act as if the command
     is successful, whatever the server reponds. */

  if(cmd[0] == '*') {
    cmd++;
    sshc->acceptfail = TRUE;
  }

  if(strcasecompare("pwd", cmd)) {
    /* output debug output if that is requested */
    char *tmp = aprintf("257 \"%s\" is current directory.\n",
                        protop->path);
    if(!tmp) {
      sshc->actualcode = CURLE_OUT_OF_MEMORY;
      state(data, SSH_SFTP_CLOSE);
      sshc->nextstate = SSH_NO_STATE;
      return;
    }
    Curl_debug(data, CURLINFO_HEADER_OUT, (char *) "PWD\n", 4);
    Curl_debug(data, CURLINFO_HEADER_IN, tmp, strlen(tmp));

    /* this sends an FTP-like "header" to the header callback so that the
       current directory can be read very similar to how it is read when
       using ordinary FTP. */
    result = Curl_client_write(data, CLIENTWRITE_HEADER, tmp, strlen(tmp));
    free(tmp);
    if(result) {
      state(data, SSH_SFTP_CLOSE);
      sshc->nextstate = SSH_NO_STATE;
      sshc->actualcode = result;
    }
    else
      state(data, SSH_SFTP_NEXT_QUOTE);
    return;
  }

  /*
   * the arguments following the command must be separated from the
   * command with a space so we can check for it unconditionally
   */
  cp = strchr(cmd, ' ');
  if(!cp) {
    failf(data, "Syntax error in SFTP command. Supply parameter(s)!");
    state(data, SSH_SFTP_CLOSE);
    sshc->nextstate = SSH_NO_STATE;
    sshc->actualcode = CURLE_QUOTE_ERROR;
    return;
  }

  /*
   * also, every command takes at least one argument so we get that
   * first argument right now
   */
  result = Curl_get_pathname(&cp, &sshc->quote_path1, sshc->homedir);
  if(result) {
    if(result == CURLE_OUT_OF_MEMORY)
      failf(data, "Out of memory");
    else
      failf(data, "Syntax error: Bad first parameter");
    state(data, SSH_SFTP_CLOSE);
    sshc->nextstate = SSH_NO_STATE;
    sshc->actualcode = result;
    return;
  }

  /*
   * SFTP is a binary protocol, so we don't send text commands
   * to the server. Instead, we scan for commands used by
   * OpenSSH's sftp program and call the appropriate libssh
   * functions.
   */
  if(strncasecompare(cmd, "chgrp ", 6) ||
     strncasecompare(cmd, "chmod ", 6) ||
     strncasecompare(cmd, "chown ", 6) ||
     strncasecompare(cmd, "atime ", 6) ||
     strncasecompare(cmd, "mtime ", 6)) {
    /* attribute change */

    /* sshc->quote_path1 contains the mode to set */
    /* get the destination */
    result = Curl_get_pathname(&cp, &sshc->quote_path2, sshc->homedir);
    if(result) {
      if(result == CURLE_OUT_OF_MEMORY)
        failf(data, "Out of memory");
      else
        failf(data, "Syntax error in chgrp/chmod/chown/atime/mtime: "
              "Bad second parameter");
      Curl_safefree(sshc->quote_path1);
      state(data, SSH_SFTP_CLOSE);
      sshc->nextstate = SSH_NO_STATE;
      sshc->actualcode = result;
      return;
    }
    sshc->quote_attrs = NULL;
    state(data, SSH_SFTP_QUOTE_STAT);
    return;
  }
  if(strncasecompare(cmd, "ln ", 3) ||
     strncasecompare(cmd, "symlink ", 8)) {
    /* symbolic linking */
    /* sshc->quote_path1 is the source */
    /* get the destination */
    result = Curl_get_pathname(&cp, &sshc->quote_path2, sshc->homedir);
    if(result) {
      if(result == CURLE_OUT_OF_MEMORY)
        failf(data, "Out of memory");
      else
        failf(data, "Syntax error in ln/symlink: Bad second parameter");
      Curl_safefree(sshc->quote_path1);
      state(data, SSH_SFTP_CLOSE);
      sshc->nextstate = SSH_NO_STATE;
      sshc->actualcode = result;
      return;
    }
    state(data, SSH_SFTP_QUOTE_SYMLINK);
    return;
  }
  else if(strncasecompare(cmd, "mkdir ", 6)) {
    /* create dir */
    state(data, SSH_SFTP_QUOTE_MKDIR);
    return;
  }
  else if(strncasecompare(cmd, "rename ", 7)) {
    /* rename file */
    /* first param is the source path */
    /* second param is the dest. path */
    result = Curl_get_pathname(&cp, &sshc->quote_path2, sshc->homedir);
    if(result) {
      if(result == CURLE_OUT_OF_MEMORY)
        failf(data, "Out of memory");
      else
        failf(data, "Syntax error in rename: Bad second parameter");
      Curl_safefree(sshc->quote_path1);
      state(data, SSH_SFTP_CLOSE);
      sshc->nextstate = SSH_NO_STATE;
      sshc->actualcode = result;
      return;
    }
    state(data, SSH_SFTP_QUOTE_RENAME);
    return;
  }
  else if(strncasecompare(cmd, "rmdir ", 6)) {
    /* delete dir */
    state(data, SSH_SFTP_QUOTE_RMDIR);
    return;
  }
  else if(strncasecompare(cmd, "rm ", 3)) {
    state(data, SSH_SFTP_QUOTE_UNLINK);
    return;
  }
#ifdef HAS_STATVFS_SUPPORT
  else if(strncasecompare(cmd, "statvfs ", 8)) {
    state(data, SSH_SFTP_QUOTE_STATVFS);
    return;
  }
#endif

  failf(data, "Unknown SFTP command");
  Curl_safefree(sshc->quote_path1);
  Curl_safefree(sshc->quote_path2);
  state(data, SSH_SFTP_CLOSE);
  sshc->nextstate = SSH_NO_STATE;
  sshc->actualcode = CURLE_QUOTE_ERROR;
}

static void sftp_quote_stat(struct Curl_easy *data)
{
  struct connectdata *conn = data->conn;
  struct ssh_conn *sshc = &conn->proto.sshc;
  char *cmd = sshc->quote_item->data;
  sshc->acceptfail = FALSE;

  /* if a command starts with an asterisk, which a legal SFTP command never
     can, the command will be allowed to fail without it causing any
     aborts or cancels etc. It will cause libcurl to act as if the command
     is successful, whatever the server reponds. */

  if(cmd[0] == '*') {
    cmd++;
    sshc->acceptfail = TRUE;
  }

  /* We read the file attributes, store them in sshc->quote_attrs
   * and modify them accordingly to command. Then we switch to
   * QUOTE_SETSTAT state to write new ones.
   */

  if(sshc->quote_attrs)
    sftp_attributes_free(sshc->quote_attrs);
  sshc->quote_attrs = sftp_stat(sshc->sftp_session, sshc->quote_path2);
  if(!sshc->quote_attrs) {
    Curl_safefree(sshc->quote_path1);
    Curl_safefree(sshc->quote_path2);
    failf(data, "Attempt to get SFTP stats failed: %d",
          sftp_get_error(sshc->sftp_session));
    state(data, SSH_SFTP_CLOSE);
    sshc->nextstate = SSH_NO_STATE;
    sshc->actualcode = CURLE_QUOTE_ERROR;
    return;
  }

  /* Now set the new attributes... */
  if(strncasecompare(cmd, "chgrp", 5)) {
    sshc->quote_attrs->gid = (uint32_t)strtoul(sshc->quote_path1, NULL, 10);
    if(sshc->quote_attrs->gid == 0 && !ISDIGIT(sshc->quote_path1[0]) &&
        !sshc->acceptfail) {
      Curl_safefree(sshc->quote_path1);
      Curl_safefree(sshc->quote_path2);
      failf(data, "Syntax error: chgrp gid not a number");
      state(data, SSH_SFTP_CLOSE);
      sshc->nextstate = SSH_NO_STATE;
      sshc->actualcode = CURLE_QUOTE_ERROR;
      return;
    }
    sshc->quote_attrs->flags |= SSH_FILEXFER_ATTR_UIDGID;
  }
  else if(strncasecompare(cmd, "chmod", 5)) {
    mode_t perms;
    perms = (mode_t)strtoul(sshc->quote_path1, NULL, 8);
    /* permissions are octal */
    if(perms == 0 && !ISDIGIT(sshc->quote_path1[0])) {
      Curl_safefree(sshc->quote_path1);
      Curl_safefree(sshc->quote_path2);
      failf(data, "Syntax error: chmod permissions not a number");
      state(data, SSH_SFTP_CLOSE);
      sshc->nextstate = SSH_NO_STATE;
      sshc->actualcode = CURLE_QUOTE_ERROR;
      return;
    }
    sshc->quote_attrs->permissions = perms;
    sshc->quote_attrs->flags |= SSH_FILEXFER_ATTR_PERMISSIONS;
  }
  else if(strncasecompare(cmd, "chown", 5)) {
    sshc->quote_attrs->uid = (uint32_t)strtoul(sshc->quote_path1, NULL, 10);
    if(sshc->quote_attrs->uid == 0 && !ISDIGIT(sshc->quote_path1[0]) &&
        !sshc->acceptfail) {
      Curl_safefree(sshc->quote_path1);
      Curl_safefree(sshc->quote_path2);
      failf(data, "Syntax error: chown uid not a number");
      state(data, SSH_SFTP_CLOSE);
      sshc->nextstate = SSH_NO_STATE;
      sshc->actualcode = CURLE_QUOTE_ERROR;
      return;
    }
    sshc->quote_attrs->flags |= SSH_FILEXFER_ATTR_UIDGID;
  }
  else if(strncasecompare(cmd, "atime", 5)) {
    time_t date = Curl_getdate_capped(sshc->quote_path1);
    if(date == -1) {
      Curl_safefree(sshc->quote_path1);
      Curl_safefree(sshc->quote_path2);
      failf(data, "Syntax error: incorrect access date format");
      state(data, SSH_SFTP_CLOSE);
      sshc->nextstate = SSH_NO_STATE;
      sshc->actualcode = CURLE_QUOTE_ERROR;
      return;
    }
    sshc->quote_attrs->atime = (uint32_t)date;
    sshc->quote_attrs->flags |= SSH_FILEXFER_ATTR_ACMODTIME;
  }
  else if(strncasecompare(cmd, "mtime", 5)) {
    time_t date = Curl_getdate_capped(sshc->quote_path1);
    if(date == -1) {
      Curl_safefree(sshc->quote_path1);
      Curl_safefree(sshc->quote_path2);
      failf(data, "Syntax error: incorrect modification date format");
      state(data, SSH_SFTP_CLOSE);
      sshc->nextstate = SSH_NO_STATE;
      sshc->actualcode = CURLE_QUOTE_ERROR;
      return;
    }
    sshc->quote_attrs->mtime = (uint32_t)date;
    sshc->quote_attrs->flags |= SSH_FILEXFER_ATTR_ACMODTIME;
  }

  /* Now send the completed structure... */
  state(data, SSH_SFTP_QUOTE_SETSTAT);
  return;
}

CURLcode Curl_ssh_init(void)
{
  if(ssh_init()) {
    DEBUGF(fprintf(stderr, "Error: libssh_init failed\n"));
    return CURLE_FAILED_INIT;
  }
  return CURLE_OK;
}

void Curl_ssh_cleanup(void)
{
  (void)ssh_finalize();
}

void Curl_ssh_version(char *buffer, size_t buflen)
{
  (void)msnprintf(buffer, buflen, "libssh/%s", CURL_LIBSSH_VERSION);
}

#endif                          /* USE_LIBSSH */
