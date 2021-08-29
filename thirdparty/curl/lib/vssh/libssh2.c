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

/* #define CURL_LIBSSH2_DEBUG */

#include "curl_setup.h"

#ifdef USE_LIBSSH2

#include <limits.h>

#include <libssh2.h>
#include <libssh2_sftp.h>

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
#include "http.h" /* for HTTP proxy tunnel stuff */
#include "ssh.h"
#include "url.h"
#include "speedcheck.h"
#include "getinfo.h"
#include "strdup.h"
#include "strcase.h"
#include "vtls/vtls.h"
#include "connect.h"
#include "inet_ntop.h"
#include "parsedate.h" /* for the week day and month names */
#include "sockaddr.h" /* required for Curl_sockaddr_storage */
#include "strtoofft.h"
#include "multiif.h"
#include "select.h"
#include "warnless.h"
#include "curl_path.h"
#include "strcase.h"

#include <curl_base64.h> /* for base64 encoding/decoding */
#include <curl_sha256.h>


/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"

#if LIBSSH2_VERSION_NUM >= 0x010206
/* libssh2_sftp_statvfs and friends were added in 1.2.6 */
#define HAS_STATVFS_SUPPORT 1
#endif

#define sftp_libssh2_realpath(s,p,t,m)                          \
  libssh2_sftp_symlink_ex((s), (p), curlx_uztoui(strlen(p)),    \
                          (t), (m), LIBSSH2_SFTP_REALPATH)

/* Local functions: */
static const char *sftp_libssh2_strerror(unsigned long err);
static LIBSSH2_ALLOC_FUNC(my_libssh2_malloc);
static LIBSSH2_REALLOC_FUNC(my_libssh2_realloc);
static LIBSSH2_FREE_FUNC(my_libssh2_free);

static CURLcode ssh_force_knownhost_key_type(struct Curl_easy *data);
static CURLcode ssh_connect(struct Curl_easy *data, bool *done);
static CURLcode ssh_multi_statemach(struct Curl_easy *data, bool *done);
static CURLcode ssh_do(struct Curl_easy *data, bool *done);
static CURLcode scp_done(struct Curl_easy *data, CURLcode c, bool premature);
static CURLcode scp_doing(struct Curl_easy *data, bool *dophase_done);
static CURLcode scp_disconnect(struct Curl_easy *data,
                               struct connectdata *conn, bool dead_connection);
static CURLcode sftp_done(struct Curl_easy *data, CURLcode, bool premature);
static CURLcode sftp_doing(struct Curl_easy *data, bool *dophase_done);
static CURLcode sftp_disconnect(struct Curl_easy *data,
                                struct connectdata *conn, bool dead);
static CURLcode sftp_perform(struct Curl_easy *data, bool *connected,
                             bool *dophase_done);
static int ssh_getsock(struct Curl_easy *data, struct connectdata *conn,
                       curl_socket_t *sock);
static CURLcode ssh_setup_connection(struct Curl_easy *data,
                                     struct connectdata *conn);
static void ssh_attach(struct Curl_easy *data, struct connectdata *conn);

/*
 * SCP protocol handler.
 */

const struct Curl_handler Curl_handler_scp = {
  "SCP",                                /* scheme */
  ssh_setup_connection,                 /* setup_connection */
  ssh_do,                               /* do_it */
  scp_done,                             /* done */
  ZERO_NULL,                            /* do_more */
  ssh_connect,                          /* connect_it */
  ssh_multi_statemach,                  /* connecting */
  scp_doing,                            /* doing */
  ssh_getsock,                          /* proto_getsock */
  ssh_getsock,                          /* doing_getsock */
  ZERO_NULL,                            /* domore_getsock */
  ssh_getsock,                          /* perform_getsock */
  scp_disconnect,                       /* disconnect */
  ZERO_NULL,                            /* readwrite */
  ZERO_NULL,                            /* connection_check */
  ssh_attach,
  PORT_SSH,                             /* defport */
  CURLPROTO_SCP,                        /* protocol */
  CURLPROTO_SCP,                        /* family */
  PROTOPT_DIRLOCK | PROTOPT_CLOSEACTION
  | PROTOPT_NOURLQUERY                  /* flags */
};


/*
 * SFTP protocol handler.
 */

const struct Curl_handler Curl_handler_sftp = {
  "SFTP",                               /* scheme */
  ssh_setup_connection,                 /* setup_connection */
  ssh_do,                               /* do_it */
  sftp_done,                            /* done */
  ZERO_NULL,                            /* do_more */
  ssh_connect,                          /* connect_it */
  ssh_multi_statemach,                  /* connecting */
  sftp_doing,                           /* doing */
  ssh_getsock,                          /* proto_getsock */
  ssh_getsock,                          /* doing_getsock */
  ZERO_NULL,                            /* domore_getsock */
  ssh_getsock,                          /* perform_getsock */
  sftp_disconnect,                      /* disconnect */
  ZERO_NULL,                            /* readwrite */
  ZERO_NULL,                            /* connection_check */
  ssh_attach,
  PORT_SSH,                             /* defport */
  CURLPROTO_SFTP,                       /* protocol */
  CURLPROTO_SFTP,                       /* family */
  PROTOPT_DIRLOCK | PROTOPT_CLOSEACTION
  | PROTOPT_NOURLQUERY                  /* flags */
};

static void
kbd_callback(const char *name, int name_len, const char *instruction,
             int instruction_len, int num_prompts,
             const LIBSSH2_USERAUTH_KBDINT_PROMPT *prompts,
             LIBSSH2_USERAUTH_KBDINT_RESPONSE *responses,
             void **abstract)
{
  struct Curl_easy *data = (struct Curl_easy *)*abstract;

#ifdef CURL_LIBSSH2_DEBUG
  fprintf(stderr, "name=%s\n", name);
  fprintf(stderr, "name_len=%d\n", name_len);
  fprintf(stderr, "instruction=%s\n", instruction);
  fprintf(stderr, "instruction_len=%d\n", instruction_len);
  fprintf(stderr, "num_prompts=%d\n", num_prompts);
#else
  (void)name;
  (void)name_len;
  (void)instruction;
  (void)instruction_len;
#endif  /* CURL_LIBSSH2_DEBUG */
  if(num_prompts == 1) {
    struct connectdata *conn = data->conn;
    responses[0].text = strdup(conn->passwd);
    responses[0].length = curlx_uztoui(strlen(conn->passwd));
  }
  (void)prompts;
} /* kbd_callback */

static CURLcode sftp_libssh2_error_to_CURLE(unsigned long err)
{
  switch(err) {
    case LIBSSH2_FX_OK:
      return CURLE_OK;

    case LIBSSH2_FX_NO_SUCH_FILE:
    case LIBSSH2_FX_NO_SUCH_PATH:
      return CURLE_REMOTE_FILE_NOT_FOUND;

    case LIBSSH2_FX_PERMISSION_DENIED:
    case LIBSSH2_FX_WRITE_PROTECT:
    case LIBSSH2_FX_LOCK_CONFlICT:
      return CURLE_REMOTE_ACCESS_DENIED;

    case LIBSSH2_FX_NO_SPACE_ON_FILESYSTEM:
    case LIBSSH2_FX_QUOTA_EXCEEDED:
      return CURLE_REMOTE_DISK_FULL;

    case LIBSSH2_FX_FILE_ALREADY_EXISTS:
      return CURLE_REMOTE_FILE_EXISTS;

    case LIBSSH2_FX_DIR_NOT_EMPTY:
      return CURLE_QUOTE_ERROR;

    default:
      break;
  }

  return CURLE_SSH;
}

static CURLcode libssh2_session_error_to_CURLE(int err)
{
  switch(err) {
    /* Ordered by order of appearance in libssh2.h */
    case LIBSSH2_ERROR_NONE:
      return CURLE_OK;

    /* This is the error returned by libssh2_scp_recv2
     * on unknown file */
    case LIBSSH2_ERROR_SCP_PROTOCOL:
      return CURLE_REMOTE_FILE_NOT_FOUND;

    case LIBSSH2_ERROR_SOCKET_NONE:
      return CURLE_COULDNT_CONNECT;

    case LIBSSH2_ERROR_ALLOC:
      return CURLE_OUT_OF_MEMORY;

    case LIBSSH2_ERROR_SOCKET_SEND:
      return CURLE_SEND_ERROR;

    case LIBSSH2_ERROR_HOSTKEY_INIT:
    case LIBSSH2_ERROR_HOSTKEY_SIGN:
    case LIBSSH2_ERROR_PUBLICKEY_UNRECOGNIZED:
    case LIBSSH2_ERROR_PUBLICKEY_UNVERIFIED:
      return CURLE_PEER_FAILED_VERIFICATION;

    case LIBSSH2_ERROR_PASSWORD_EXPIRED:
      return CURLE_LOGIN_DENIED;

    case LIBSSH2_ERROR_SOCKET_TIMEOUT:
    case LIBSSH2_ERROR_TIMEOUT:
      return CURLE_OPERATION_TIMEDOUT;

    case LIBSSH2_ERROR_EAGAIN:
      return CURLE_AGAIN;
  }

  return CURLE_SSH;
}

static LIBSSH2_ALLOC_FUNC(my_libssh2_malloc)
{
  (void)abstract; /* arg not used */
  return malloc(count);
}

static LIBSSH2_REALLOC_FUNC(my_libssh2_realloc)
{
  (void)abstract; /* arg not used */
  return realloc(ptr, count);
}

static LIBSSH2_FREE_FUNC(my_libssh2_free)
{
  (void)abstract; /* arg not used */
  if(ptr) /* ssh2 agent sometimes call free with null ptr */
    free(ptr);
}

/*
 * SSH State machine related code
 */
/* This is the ONLY way to change SSH state! */
static void state(struct Curl_easy *data, sshstate nowstate)
{
  struct connectdata *conn = data->conn;
  struct ssh_conn *sshc = &conn->proto.sshc;
#if defined(DEBUGBUILD) && !defined(CURL_DISABLE_VERBOSE_STRINGS)
  /* for debug purposes */
  static const char * const names[] = {
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

  /* a precaution to make sure the lists are in sync */
  DEBUGASSERT(sizeof(names)/sizeof(names[0]) == SSH_LAST);

  if(sshc->state != nowstate) {
    infof(data, "SFTP %p state change from %s to %s",
          (void *)sshc, names[sshc->state], names[nowstate]);
  }
#endif

  sshc->state = nowstate;
}


#ifdef HAVE_LIBSSH2_KNOWNHOST_API
static int sshkeycallback(struct Curl_easy *easy,
                          const struct curl_khkey *knownkey, /* known */
                          const struct curl_khkey *foundkey, /* found */
                          enum curl_khmatch match,
                          void *clientp)
{
  (void)easy;
  (void)knownkey;
  (void)foundkey;
  (void)clientp;

  /* we only allow perfect matches, and we reject everything else */
  return (match != CURLKHMATCH_OK)?CURLKHSTAT_REJECT:CURLKHSTAT_FINE;
}
#endif

/*
 * Earlier libssh2 versions didn't have the ability to seek to 64bit positions
 * with 32bit size_t.
 */
#ifdef HAVE_LIBSSH2_SFTP_SEEK64
#define SFTP_SEEK(x,y) libssh2_sftp_seek64(x, (libssh2_uint64_t)y)
#else
#define SFTP_SEEK(x,y) libssh2_sftp_seek(x, (size_t)y)
#endif

/*
 * Earlier libssh2 versions didn't do SCP properly beyond 32bit sizes on 32bit
 * architectures so we check of the necessary function is present.
 */
#ifndef HAVE_LIBSSH2_SCP_SEND64
#define SCP_SEND(a,b,c,d) libssh2_scp_send_ex(a, b, (int)(c), (size_t)d, 0, 0)
#else
#define SCP_SEND(a,b,c,d) libssh2_scp_send64(a, b, (int)(c),            \
                                             (libssh2_uint64_t)d, 0, 0)
#endif

/*
 * libssh2 1.2.8 fixed the problem with 32bit ints used for sockets on win64.
 */
#ifdef HAVE_LIBSSH2_SESSION_HANDSHAKE
#define libssh2_session_startup(x,y) libssh2_session_handshake(x,y)
#endif

static CURLcode ssh_knownhost(struct Curl_easy *data)
{
  CURLcode result = CURLE_OK;

#ifdef HAVE_LIBSSH2_KNOWNHOST_API
  if(data->set.str[STRING_SSH_KNOWNHOSTS]) {
    /* we're asked to verify the host against a file */
    struct connectdata *conn = data->conn;
    struct ssh_conn *sshc = &conn->proto.sshc;
    struct libssh2_knownhost *host = NULL;
    int rc;
    int keytype;
    size_t keylen;
    const char *remotekey = libssh2_session_hostkey(sshc->ssh_session,
                                                    &keylen, &keytype);
    int keycheck = LIBSSH2_KNOWNHOST_CHECK_FAILURE;
    int keybit = 0;

    if(remotekey) {
      /*
       * A subject to figure out is what host name we need to pass in here.
       * What host name does OpenSSH store in its file if an IDN name is
       * used?
       */
      enum curl_khmatch keymatch;
      curl_sshkeycallback func =
        data->set.ssh_keyfunc?data->set.ssh_keyfunc:sshkeycallback;
      struct curl_khkey knownkey;
      struct curl_khkey *knownkeyp = NULL;
      struct curl_khkey foundkey;

      switch(keytype) {
      case LIBSSH2_HOSTKEY_TYPE_RSA:
        keybit = LIBSSH2_KNOWNHOST_KEY_SSHRSA;
        break;
      case LIBSSH2_HOSTKEY_TYPE_DSS:
        keybit = LIBSSH2_KNOWNHOST_KEY_SSHDSS;
        break;
#ifdef LIBSSH2_HOSTKEY_TYPE_ECDSA_256
      case LIBSSH2_HOSTKEY_TYPE_ECDSA_256:
        keybit = LIBSSH2_KNOWNHOST_KEY_ECDSA_256;
        break;
#endif
#ifdef LIBSSH2_HOSTKEY_TYPE_ECDSA_384
      case LIBSSH2_HOSTKEY_TYPE_ECDSA_384:
        keybit = LIBSSH2_KNOWNHOST_KEY_ECDSA_384;
        break;
#endif
#ifdef LIBSSH2_HOSTKEY_TYPE_ECDSA_521
      case LIBSSH2_HOSTKEY_TYPE_ECDSA_521:
        keybit = LIBSSH2_KNOWNHOST_KEY_ECDSA_521;
        break;
#endif
#ifdef LIBSSH2_HOSTKEY_TYPE_ED25519
      case LIBSSH2_HOSTKEY_TYPE_ED25519:
        keybit = LIBSSH2_KNOWNHOST_KEY_ED25519;
        break;
#endif
      default:
        infof(data, "unsupported key type, can't check knownhosts!");
        keybit = 0;
        break;
      }
      if(!keybit)
        /* no check means failure! */
        rc = CURLKHSTAT_REJECT;
      else {
#ifdef HAVE_LIBSSH2_KNOWNHOST_CHECKP
        keycheck = libssh2_knownhost_checkp(sshc->kh,
                                            conn->host.name,
                                            (conn->remote_port != PORT_SSH)?
                                            conn->remote_port:-1,
                                            remotekey, keylen,
                                            LIBSSH2_KNOWNHOST_TYPE_PLAIN|
                                            LIBSSH2_KNOWNHOST_KEYENC_RAW|
                                            keybit,
                                            &host);
#else
        keycheck = libssh2_knownhost_check(sshc->kh,
                                           conn->host.name,
                                           remotekey, keylen,
                                           LIBSSH2_KNOWNHOST_TYPE_PLAIN|
                                           LIBSSH2_KNOWNHOST_KEYENC_RAW|
                                           keybit,
                                           &host);
#endif

        infof(data, "SSH host check: %d, key: %s", keycheck,
              (keycheck <= LIBSSH2_KNOWNHOST_CHECK_MISMATCH)?
              host->key:"<none>");

        /* setup 'knownkey' */
        if(keycheck <= LIBSSH2_KNOWNHOST_CHECK_MISMATCH) {
          knownkey.key = host->key;
          knownkey.len = 0;
          knownkey.keytype = (keytype == LIBSSH2_HOSTKEY_TYPE_RSA)?
            CURLKHTYPE_RSA : CURLKHTYPE_DSS;
          knownkeyp = &knownkey;
        }

        /* setup 'foundkey' */
        foundkey.key = remotekey;
        foundkey.len = keylen;
        foundkey.keytype = (keytype == LIBSSH2_HOSTKEY_TYPE_RSA)?
          CURLKHTYPE_RSA : CURLKHTYPE_DSS;

        /*
         * if any of the LIBSSH2_KNOWNHOST_CHECK_* defines and the
         * curl_khmatch enum are ever modified, we need to introduce a
         * translation table here!
         */
        keymatch = (enum curl_khmatch)keycheck;

        /* Ask the callback how to behave */
        Curl_set_in_callback(data, true);
        rc = func(data, knownkeyp, /* from the knownhosts file */
                  &foundkey, /* from the remote host */
                  keymatch, data->set.ssh_keyfunc_userp);
        Curl_set_in_callback(data, false);
      }
    }
    else
      /* no remotekey means failure! */
      rc = CURLKHSTAT_REJECT;

    switch(rc) {
    default: /* unknown return codes will equal reject */
      /* FALLTHROUGH */
    case CURLKHSTAT_REJECT:
      state(data, SSH_SESSION_FREE);
      /* FALLTHROUGH */
    case CURLKHSTAT_DEFER:
      /* DEFER means bail out but keep the SSH_HOSTKEY state */
      result = sshc->actualcode = CURLE_PEER_FAILED_VERIFICATION;
      break;
    case CURLKHSTAT_FINE_REPLACE:
      /* remove old host+key that doesn't match */
      if(host)
        libssh2_knownhost_del(sshc->kh, host);
        /*FALLTHROUGH*/
    case CURLKHSTAT_FINE:
        /*FALLTHROUGH*/
    case CURLKHSTAT_FINE_ADD_TO_FILE:
      /* proceed */
      if(keycheck != LIBSSH2_KNOWNHOST_CHECK_MATCH) {
        /* the found host+key didn't match but has been told to be fine
           anyway so we add it in memory */
        int addrc = libssh2_knownhost_add(sshc->kh,
                                          conn->host.name, NULL,
                                          remotekey, keylen,
                                          LIBSSH2_KNOWNHOST_TYPE_PLAIN|
                                          LIBSSH2_KNOWNHOST_KEYENC_RAW|
                                          keybit, NULL);
        if(addrc)
          infof(data, "Warning adding the known host %s failed!",
                conn->host.name);
        else if(rc == CURLKHSTAT_FINE_ADD_TO_FILE ||
                rc == CURLKHSTAT_FINE_REPLACE) {
          /* now we write the entire in-memory list of known hosts to the
             known_hosts file */
          int wrc =
            libssh2_knownhost_writefile(sshc->kh,
                                        data->set.str[STRING_SSH_KNOWNHOSTS],
                                        LIBSSH2_KNOWNHOST_FILE_OPENSSH);
          if(wrc) {
            infof(data, "Warning, writing %s failed!",
                  data->set.str[STRING_SSH_KNOWNHOSTS]);
          }
        }
      }
      break;
    }
  }
#else /* HAVE_LIBSSH2_KNOWNHOST_API */
  (void)data;
#endif
  return result;
}

static CURLcode ssh_check_fingerprint(struct Curl_easy *data)
{
  struct connectdata *conn = data->conn;
  struct ssh_conn *sshc = &conn->proto.sshc;
  const char *pubkey_md5 = data->set.str[STRING_SSH_HOST_PUBLIC_KEY_MD5];
  const char *pubkey_sha256 = data->set.str[STRING_SSH_HOST_PUBLIC_KEY_SHA256];

  infof(data, "SSH MD5 public key: %s",
    pubkey_md5 != NULL ? pubkey_md5 : "NULL");
  infof(data, "SSH SHA256 public key: %s",
      pubkey_sha256 != NULL ? pubkey_sha256 : "NULL");

  if(pubkey_sha256) {
    const char *fingerprint = NULL;
    char *fingerprint_b64 = NULL;
    size_t fingerprint_b64_len;
    size_t pub_pos = 0;
    size_t b64_pos = 0;

#ifdef LIBSSH2_HOSTKEY_HASH_SHA256
    /* The fingerprint points to static storage (!), don't free() it. */
    fingerprint = libssh2_hostkey_hash(sshc->ssh_session,
        LIBSSH2_HOSTKEY_HASH_SHA256);
#else
    const char *hostkey;
    size_t len = 0;
    unsigned char hash[32];

    hostkey = libssh2_session_hostkey(sshc->ssh_session, &len, NULL);
    if(hostkey) {
      Curl_sha256it(hash, (const unsigned char *) hostkey, len);
      fingerprint = (char *) hash;
    }
#endif

    if(!fingerprint) {
      failf(data,
          "Denied establishing ssh session: sha256 fingerprint "
          "not available");
      state(data, SSH_SESSION_FREE);
      sshc->actualcode = CURLE_PEER_FAILED_VERIFICATION;
      return sshc->actualcode;
    }

    /* The length of fingerprint is 32 bytes for SHA256.
     * See libssh2_hostkey_hash documentation. */
    if(Curl_base64_encode (data, fingerprint, 32, &fingerprint_b64,
        &fingerprint_b64_len) != CURLE_OK) {
      state(data, SSH_SESSION_FREE);
      sshc->actualcode = CURLE_PEER_FAILED_VERIFICATION;
      return sshc->actualcode;
    }

    if(!fingerprint_b64) {
      failf(data,
          "sha256 fingerprint could not be encoded");
      state(data, SSH_SESSION_FREE);
      sshc->actualcode = CURLE_PEER_FAILED_VERIFICATION;
      return sshc->actualcode;
    }

    infof(data, "SSH SHA256 fingerprint: %s", fingerprint_b64);

    /* Find the position of any = padding characters in the public key */
    while((pubkey_sha256[pub_pos] != '=') && pubkey_sha256[pub_pos]) {
      pub_pos++;
    }

    /* Find the position of any = padding characters in the base64 coded
     * hostkey fingerprint */
    while((fingerprint_b64[b64_pos] != '=') && fingerprint_b64[b64_pos]) {
      b64_pos++;
    }

    /* Before we authenticate we check the hostkey's sha256 fingerprint
     * against a known fingerprint, if available.
     */
    if((pub_pos != b64_pos) ||
        Curl_strncasecompare(fingerprint_b64, pubkey_sha256, pub_pos) != 1) {
      free(fingerprint_b64);

      failf(data,
          "Denied establishing ssh session: mismatch sha256 fingerprint. "
          "Remote %s is not equal to %s", fingerprint_b64, pubkey_sha256);
      state(data, SSH_SESSION_FREE);
      sshc->actualcode = CURLE_PEER_FAILED_VERIFICATION;
      return sshc->actualcode;
    }

    free(fingerprint_b64);

    infof(data, "SHA256 checksum match!");
  }

  if(pubkey_md5) {
    char md5buffer[33];
    const char *fingerprint = NULL;

    fingerprint = libssh2_hostkey_hash(sshc->ssh_session,
        LIBSSH2_HOSTKEY_HASH_MD5);

    if(fingerprint) {
      /* The fingerprint points to static storage (!), don't free() it. */
      int i;
      for(i = 0; i < 16; i++) {
        msnprintf(&md5buffer[i*2], 3, "%02x", (unsigned char) fingerprint[i]);
      }

      infof(data, "SSH MD5 fingerprint: %s", md5buffer);
    }

    /* Before we authenticate we check the hostkey's MD5 fingerprint
     * against a known fingerprint, if available.
     */
    if(pubkey_md5 && strlen(pubkey_md5) == 32) {
      if(!fingerprint || !strcasecompare(md5buffer, pubkey_md5)) {
        if(fingerprint) {
          failf(data,
              "Denied establishing ssh session: mismatch md5 fingerprint. "
              "Remote %s is not equal to %s", md5buffer, pubkey_md5);
        }
        else {
          failf(data,
              "Denied establishing ssh session: md5 fingerprint "
              "not available");
        }
        state(data, SSH_SESSION_FREE);
        sshc->actualcode = CURLE_PEER_FAILED_VERIFICATION;
        return sshc->actualcode;
      }
      infof(data, "MD5 checksum match!");
    }
  }

  if(!pubkey_md5 && !pubkey_sha256) {
    return ssh_knownhost(data);
  }
  else {
    /* as we already matched, we skip the check for known hosts */
    return CURLE_OK;
  }
}

/*
 * ssh_force_knownhost_key_type() will check the known hosts file and try to
 * force a specific public key type from the server if an entry is found.
 */
static CURLcode ssh_force_knownhost_key_type(struct Curl_easy *data)
{
  CURLcode result = CURLE_OK;

#ifdef HAVE_LIBSSH2_KNOWNHOST_API

#ifdef LIBSSH2_KNOWNHOST_KEY_ED25519
  static const char * const hostkey_method_ssh_ed25519
    = "ssh-ed25519";
#endif
#ifdef LIBSSH2_KNOWNHOST_KEY_ECDSA_521
  static const char * const hostkey_method_ssh_ecdsa_521
    = "ecdsa-sha2-nistp521";
#endif
#ifdef LIBSSH2_KNOWNHOST_KEY_ECDSA_384
  static const char * const hostkey_method_ssh_ecdsa_384
    = "ecdsa-sha2-nistp384";
#endif
#ifdef LIBSSH2_KNOWNHOST_KEY_ECDSA_256
  static const char * const hostkey_method_ssh_ecdsa_256
    = "ecdsa-sha2-nistp256";
#endif
  static const char * const hostkey_method_ssh_rsa
    = "ssh-rsa";
  static const char * const hostkey_method_ssh_dss
    = "ssh-dss";

  const char *hostkey_method = NULL;
  struct connectdata *conn = data->conn;
  struct ssh_conn *sshc = &conn->proto.sshc;
  struct libssh2_knownhost* store = NULL;
  const char *kh_name_end = NULL;
  size_t kh_name_size = 0;
  int port = 0;
  bool found = false;

  if(sshc->kh && !data->set.str[STRING_SSH_HOST_PUBLIC_KEY_MD5]) {
    /* lets try to find our host in the known hosts file */
    while(!libssh2_knownhost_get(sshc->kh, &store, store)) {
      /* For non-standard ports, the name will be enclosed in */
      /* square brackets, followed by a colon and the port */
      if(store) {
        if(store->name) {
          if(store->name[0] == '[') {
            kh_name_end = strstr(store->name, "]:");
            if(!kh_name_end) {
              infof(data, "Invalid host pattern %s in %s",
                    store->name, data->set.str[STRING_SSH_KNOWNHOSTS]);
              continue;
            }
            port = atoi(kh_name_end + 2);
            if(kh_name_end && (port == conn->remote_port)) {
              kh_name_size = strlen(store->name) - 1 - strlen(kh_name_end);
              if(strncmp(store->name + 1,
                 conn->host.name, kh_name_size) == 0) {
                found = true;
                break;
              }
            }
          }
          else if(strcmp(store->name, conn->host.name) == 0) {
            found = true;
            break;
          }
        }
        else {
          found = true;
          break;
        }
      }
    }

    if(found) {
      infof(data, "Found host %s in %s",
            conn->host.name, data->set.str[STRING_SSH_KNOWNHOSTS]);

      switch(store->typemask & LIBSSH2_KNOWNHOST_KEY_MASK) {
#ifdef LIBSSH2_KNOWNHOST_KEY_ED25519
      case LIBSSH2_KNOWNHOST_KEY_ED25519:
        hostkey_method = hostkey_method_ssh_ed25519;
        break;
#endif
#ifdef LIBSSH2_KNOWNHOST_KEY_ECDSA_521
      case LIBSSH2_KNOWNHOST_KEY_ECDSA_521:
        hostkey_method = hostkey_method_ssh_ecdsa_521;
        break;
#endif
#ifdef LIBSSH2_KNOWNHOST_KEY_ECDSA_384
      case LIBSSH2_KNOWNHOST_KEY_ECDSA_384:
        hostkey_method = hostkey_method_ssh_ecdsa_384;
        break;
#endif
#ifdef LIBSSH2_KNOWNHOST_KEY_ECDSA_256
      case LIBSSH2_KNOWNHOST_KEY_ECDSA_256:
        hostkey_method = hostkey_method_ssh_ecdsa_256;
        break;
#endif
      case LIBSSH2_KNOWNHOST_KEY_SSHRSA:
        hostkey_method = hostkey_method_ssh_rsa;
        break;
      case LIBSSH2_KNOWNHOST_KEY_SSHDSS:
        hostkey_method = hostkey_method_ssh_dss;
        break;
      case LIBSSH2_KNOWNHOST_KEY_RSA1:
        failf(data, "Found host key type RSA1 which is not supported");
        return CURLE_SSH;
      default:
        failf(data, "Unknown host key type: %i",
              (store->typemask & LIBSSH2_KNOWNHOST_KEY_MASK));
        return CURLE_SSH;
      }

      infof(data, "Set \"%s\" as SSH hostkey type", hostkey_method);
      result = libssh2_session_error_to_CURLE(
        libssh2_session_method_pref(
          sshc->ssh_session, LIBSSH2_METHOD_HOSTKEY, hostkey_method));
    }
    else {
      infof(data, "Did not find host %s in %s",
            conn->host.name, data->set.str[STRING_SSH_KNOWNHOSTS]);
    }
  }

#endif /* HAVE_LIBSSH2_KNOWNHOST_API */

  return result;
}

/*
 * ssh_statemach_act() runs the SSH state machine as far as it can without
 * blocking and without reaching the end.  The data the pointer 'block' points
 * to will be set to TRUE if the libssh2 function returns LIBSSH2_ERROR_EAGAIN
 * meaning it wants to be called again when the socket is ready
 */

static CURLcode ssh_statemach_act(struct Curl_easy *data, bool *block)
{
  CURLcode result = CURLE_OK;
  struct connectdata *conn = data->conn;
  struct SSHPROTO *sshp = data->req.p.ssh;
  struct ssh_conn *sshc = &conn->proto.sshc;
  curl_socket_t sock = conn->sock[FIRSTSOCKET];
  int rc = LIBSSH2_ERROR_NONE;
  int ssherr;
  unsigned long sftperr;
  int seekerr = CURL_SEEKFUNC_OK;
  size_t readdir_len;
  *block = 0; /* we're not blocking by default */

  do {
    switch(sshc->state) {
    case SSH_INIT:
      sshc->secondCreateDirs = 0;
      sshc->nextstate = SSH_NO_STATE;
      sshc->actualcode = CURLE_OK;

      /* Set libssh2 to non-blocking, since everything internally is
         non-blocking */
      libssh2_session_set_blocking(sshc->ssh_session, 0);

      result = ssh_force_knownhost_key_type(data);
      if(result) {
        state(data, SSH_SESSION_FREE);
        sshc->actualcode = result;
        break;
      }

      state(data, SSH_S_STARTUP);
      /* FALLTHROUGH */

    case SSH_S_STARTUP:
      rc = libssh2_session_startup(sshc->ssh_session, (int)sock);
      if(rc == LIBSSH2_ERROR_EAGAIN) {
        break;
      }
      if(rc) {
        char *err_msg = NULL;
        (void)libssh2_session_last_error(sshc->ssh_session, &err_msg, NULL, 0);
        failf(data, "Failure establishing ssh session: %d, %s", rc, err_msg);

        state(data, SSH_SESSION_FREE);
        sshc->actualcode = CURLE_FAILED_INIT;
        break;
      }

      state(data, SSH_HOSTKEY);

      /* FALLTHROUGH */
    case SSH_HOSTKEY:
      /*
       * Before we authenticate we should check the hostkey's fingerprint
       * against our known hosts. How that is handled (reading from file,
       * whatever) is up to us.
       */
      result = ssh_check_fingerprint(data);
      if(!result)
        state(data, SSH_AUTHLIST);
      /* ssh_check_fingerprint sets state appropriately on error */
      break;

    case SSH_AUTHLIST:
      /*
       * Figure out authentication methods
       * NB: As soon as we have provided a username to an openssh server we
       * must never change it later. Thus, always specify the correct username
       * here, even though the libssh2 docs kind of indicate that it should be
       * possible to get a 'generic' list (not user-specific) of authentication
       * methods, presumably with a blank username. That won't work in my
       * experience.
       * So always specify it here.
       */
      sshc->authlist = libssh2_userauth_list(sshc->ssh_session,
                                             conn->user,
                                             curlx_uztoui(strlen(conn->user)));

      if(!sshc->authlist) {
        if(libssh2_userauth_authenticated(sshc->ssh_session)) {
          sshc->authed = TRUE;
          infof(data, "SSH user accepted with no authentication");
          state(data, SSH_AUTH_DONE);
          break;
        }
        ssherr = libssh2_session_last_errno(sshc->ssh_session);
        if(ssherr == LIBSSH2_ERROR_EAGAIN)
          rc = LIBSSH2_ERROR_EAGAIN;
        else {
          state(data, SSH_SESSION_FREE);
          sshc->actualcode = libssh2_session_error_to_CURLE(ssherr);
        }
        break;
      }
      infof(data, "SSH authentication methods available: %s",
            sshc->authlist);

      state(data, SSH_AUTH_PKEY_INIT);
      break;

    case SSH_AUTH_PKEY_INIT:
      /*
       * Check the supported auth types in the order I feel is most secure
       * with the requested type of authentication
       */
      sshc->authed = FALSE;

      if((data->set.ssh_auth_types & CURLSSH_AUTH_PUBLICKEY) &&
         (strstr(sshc->authlist, "publickey") != NULL)) {
        bool out_of_memory = FALSE;

        sshc->rsa_pub = sshc->rsa = NULL;

        if(data->set.str[STRING_SSH_PRIVATE_KEY])
          sshc->rsa = strdup(data->set.str[STRING_SSH_PRIVATE_KEY]);
        else {
          /* To ponder about: should really the lib be messing about with the
             HOME environment variable etc? */
          char *home = curl_getenv("HOME");

          /* If no private key file is specified, try some common paths. */
          if(home) {
            /* Try ~/.ssh first. */
            sshc->rsa = aprintf("%s/.ssh/id_rsa", home);
            if(!sshc->rsa)
              out_of_memory = TRUE;
            else if(access(sshc->rsa, R_OK) != 0) {
              Curl_safefree(sshc->rsa);
              sshc->rsa = aprintf("%s/.ssh/id_dsa", home);
              if(!sshc->rsa)
                out_of_memory = TRUE;
              else if(access(sshc->rsa, R_OK) != 0) {
                Curl_safefree(sshc->rsa);
              }
            }
            free(home);
          }
          if(!out_of_memory && !sshc->rsa) {
            /* Nothing found; try the current dir. */
            sshc->rsa = strdup("id_rsa");
            if(sshc->rsa && access(sshc->rsa, R_OK) != 0) {
              Curl_safefree(sshc->rsa);
              sshc->rsa = strdup("id_dsa");
              if(sshc->rsa && access(sshc->rsa, R_OK) != 0) {
                Curl_safefree(sshc->rsa);
                /* Out of guesses. Set to the empty string to avoid
                 * surprising info messages. */
                sshc->rsa = strdup("");
              }
            }
          }
        }

        /*
         * Unless the user explicitly specifies a public key file, let
         * libssh2 extract the public key from the private key file.
         * This is done by simply passing sshc->rsa_pub = NULL.
         */
        if(data->set.str[STRING_SSH_PUBLIC_KEY]
           /* treat empty string the same way as NULL */
           && data->set.str[STRING_SSH_PUBLIC_KEY][0]) {
          sshc->rsa_pub = strdup(data->set.str[STRING_SSH_PUBLIC_KEY]);
          if(!sshc->rsa_pub)
            out_of_memory = TRUE;
        }

        if(out_of_memory || !sshc->rsa) {
          Curl_safefree(sshc->rsa);
          Curl_safefree(sshc->rsa_pub);
          state(data, SSH_SESSION_FREE);
          sshc->actualcode = CURLE_OUT_OF_MEMORY;
          break;
        }

        sshc->passphrase = data->set.ssl.key_passwd;
        if(!sshc->passphrase)
          sshc->passphrase = "";

        if(sshc->rsa_pub)
          infof(data, "Using SSH public key file '%s'", sshc->rsa_pub);
        infof(data, "Using SSH private key file '%s'", sshc->rsa);

        state(data, SSH_AUTH_PKEY);
      }
      else {
        state(data, SSH_AUTH_PASS_INIT);
      }
      break;

    case SSH_AUTH_PKEY:
      /* The function below checks if the files exists, no need to stat() here.
       */
      rc = libssh2_userauth_publickey_fromfile_ex(sshc->ssh_session,
                                                  conn->user,
                                                  curlx_uztoui(
                                                    strlen(conn->user)),
                                                  sshc->rsa_pub,
                                                  sshc->rsa, sshc->passphrase);
      if(rc == LIBSSH2_ERROR_EAGAIN) {
        break;
      }

      Curl_safefree(sshc->rsa_pub);
      Curl_safefree(sshc->rsa);

      if(rc == 0) {
        sshc->authed = TRUE;
        infof(data, "Initialized SSH public key authentication");
        state(data, SSH_AUTH_DONE);
      }
      else {
        char *err_msg = NULL;
        (void)libssh2_session_last_error(sshc->ssh_session,
                                         &err_msg, NULL, 0);
        infof(data, "SSH public key authentication failed: %s", err_msg);
        state(data, SSH_AUTH_PASS_INIT);
        rc = 0; /* clear rc and continue */
      }
      break;

    case SSH_AUTH_PASS_INIT:
      if((data->set.ssh_auth_types & CURLSSH_AUTH_PASSWORD) &&
         (strstr(sshc->authlist, "password") != NULL)) {
        state(data, SSH_AUTH_PASS);
      }
      else {
        state(data, SSH_AUTH_HOST_INIT);
        rc = 0; /* clear rc and continue */
      }
      break;

    case SSH_AUTH_PASS:
      rc = libssh2_userauth_password_ex(sshc->ssh_session, conn->user,
                                        curlx_uztoui(strlen(conn->user)),
                                        conn->passwd,
                                        curlx_uztoui(strlen(conn->passwd)),
                                        NULL);
      if(rc == LIBSSH2_ERROR_EAGAIN) {
        break;
      }
      if(rc == 0) {
        sshc->authed = TRUE;
        infof(data, "Initialized password authentication");
        state(data, SSH_AUTH_DONE);
      }
      else {
        state(data, SSH_AUTH_HOST_INIT);
        rc = 0; /* clear rc and continue */
      }
      break;

    case SSH_AUTH_HOST_INIT:
      if((data->set.ssh_auth_types & CURLSSH_AUTH_HOST) &&
         (strstr(sshc->authlist, "hostbased") != NULL)) {
        state(data, SSH_AUTH_HOST);
      }
      else {
        state(data, SSH_AUTH_AGENT_INIT);
      }
      break;

    case SSH_AUTH_HOST:
      state(data, SSH_AUTH_AGENT_INIT);
      break;

    case SSH_AUTH_AGENT_INIT:
#ifdef HAVE_LIBSSH2_AGENT_API
      if((data->set.ssh_auth_types & CURLSSH_AUTH_AGENT)
         && (strstr(sshc->authlist, "publickey") != NULL)) {

        /* Connect to the ssh-agent */
        /* The agent could be shared by a curl thread i believe
           but nothing obvious as keys can be added/removed at any time */
        if(!sshc->ssh_agent) {
          sshc->ssh_agent = libssh2_agent_init(sshc->ssh_session);
          if(!sshc->ssh_agent) {
            infof(data, "Could not create agent object");

            state(data, SSH_AUTH_KEY_INIT);
            break;
          }
        }

        rc = libssh2_agent_connect(sshc->ssh_agent);
        if(rc == LIBSSH2_ERROR_EAGAIN)
          break;
        if(rc < 0) {
          infof(data, "Failure connecting to agent");
          state(data, SSH_AUTH_KEY_INIT);
          rc = 0; /* clear rc and continue */
        }
        else {
          state(data, SSH_AUTH_AGENT_LIST);
        }
      }
      else
#endif /* HAVE_LIBSSH2_AGENT_API */
        state(data, SSH_AUTH_KEY_INIT);
      break;

    case SSH_AUTH_AGENT_LIST:
#ifdef HAVE_LIBSSH2_AGENT_API
      rc = libssh2_agent_list_identities(sshc->ssh_agent);

      if(rc == LIBSSH2_ERROR_EAGAIN)
        break;
      if(rc < 0) {
        infof(data, "Failure requesting identities to agent");
        state(data, SSH_AUTH_KEY_INIT);
        rc = 0; /* clear rc and continue */
      }
      else {
        state(data, SSH_AUTH_AGENT);
        sshc->sshagent_prev_identity = NULL;
      }
#endif
      break;

    case SSH_AUTH_AGENT:
#ifdef HAVE_LIBSSH2_AGENT_API
      /* as prev_identity evolves only after an identity user auth finished we
         can safely request it again as long as EAGAIN is returned here or by
         libssh2_agent_userauth */
      rc = libssh2_agent_get_identity(sshc->ssh_agent,
                                      &sshc->sshagent_identity,
                                      sshc->sshagent_prev_identity);
      if(rc == LIBSSH2_ERROR_EAGAIN)
        break;

      if(rc == 0) {
        rc = libssh2_agent_userauth(sshc->ssh_agent, conn->user,
                                    sshc->sshagent_identity);

        if(rc < 0) {
          if(rc != LIBSSH2_ERROR_EAGAIN) {
            /* tried and failed? go to next identity */
            sshc->sshagent_prev_identity = sshc->sshagent_identity;
          }
          break;
        }
      }

      if(rc < 0)
        infof(data, "Failure requesting identities to agent");
      else if(rc == 1)
        infof(data, "No identity would match");

      if(rc == LIBSSH2_ERROR_NONE) {
        sshc->authed = TRUE;
        infof(data, "Agent based authentication successful");
        state(data, SSH_AUTH_DONE);
      }
      else {
        state(data, SSH_AUTH_KEY_INIT);
        rc = 0; /* clear rc and continue */
      }
#endif
      break;

    case SSH_AUTH_KEY_INIT:
      if((data->set.ssh_auth_types & CURLSSH_AUTH_KEYBOARD)
         && (strstr(sshc->authlist, "keyboard-interactive") != NULL)) {
        state(data, SSH_AUTH_KEY);
      }
      else {
        state(data, SSH_AUTH_DONE);
      }
      break;

    case SSH_AUTH_KEY:
      /* Authentication failed. Continue with keyboard-interactive now. */
      rc = libssh2_userauth_keyboard_interactive_ex(sshc->ssh_session,
                                                    conn->user,
                                                    curlx_uztoui(
                                                      strlen(conn->user)),
                                                    &kbd_callback);
      if(rc == LIBSSH2_ERROR_EAGAIN) {
        break;
      }
      if(rc == 0) {
        sshc->authed = TRUE;
        infof(data, "Initialized keyboard interactive authentication");
      }
      state(data, SSH_AUTH_DONE);
      break;

    case SSH_AUTH_DONE:
      if(!sshc->authed) {
        failf(data, "Authentication failure");
        state(data, SSH_SESSION_FREE);
        sshc->actualcode = CURLE_LOGIN_DENIED;
        break;
      }

      /*
       * At this point we have an authenticated ssh session.
       */
      infof(data, "Authentication complete");

      Curl_pgrsTime(data, TIMER_APPCONNECT); /* SSH is connected */

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
      /*
       * Start the libssh2 sftp session
       */
      sshc->sftp_session = libssh2_sftp_init(sshc->ssh_session);
      if(!sshc->sftp_session) {
        char *err_msg = NULL;
        if(libssh2_session_last_errno(sshc->ssh_session) ==
           LIBSSH2_ERROR_EAGAIN) {
          rc = LIBSSH2_ERROR_EAGAIN;
          break;
        }

        (void)libssh2_session_last_error(sshc->ssh_session,
                                         &err_msg, NULL, 0);
        failf(data, "Failure initializing sftp session: %s", err_msg);
        state(data, SSH_SESSION_FREE);
        sshc->actualcode = CURLE_FAILED_INIT;
        break;
      }
      state(data, SSH_SFTP_REALPATH);
      break;

    case SSH_SFTP_REALPATH:
    {
      char tempHome[PATH_MAX];

      /*
       * Get the "home" directory
       */
      rc = sftp_libssh2_realpath(sshc->sftp_session, ".",
                                 tempHome, PATH_MAX-1);
      if(rc == LIBSSH2_ERROR_EAGAIN) {
        break;
      }
      if(rc > 0) {
        /* It seems that this string is not always NULL terminated */
        tempHome[rc] = '\0';
        sshc->homedir = strdup(tempHome);
        if(!sshc->homedir) {
          state(data, SSH_SFTP_CLOSE);
          sshc->actualcode = CURLE_OUT_OF_MEMORY;
          break;
        }
        data->state.most_recent_ftp_entrypath = sshc->homedir;
      }
      else {
        /* Return the error type */
        sftperr = libssh2_sftp_last_error(sshc->sftp_session);
        if(sftperr)
          result = sftp_libssh2_error_to_CURLE(sftperr);
        else
          /* in this case, the error wasn't in the SFTP level but for example
             a time-out or similar */
          result = CURLE_SSH;
        sshc->actualcode = result;
        DEBUGF(infof(data, "error = %lu makes libcurl = %d",
                     sftperr, (int)result));
        state(data, SSH_STOP);
        break;
      }
    }
    /* This is the last step in the SFTP connect phase. Do note that while
       we get the homedir here, we get the "workingpath" in the DO action
       since the homedir will remain the same between request but the
       working path will not. */
    DEBUGF(infof(data, "SSH CONNECT phase done"));
    state(data, SSH_STOP);
    break;

    case SSH_SFTP_QUOTE_INIT:

      result = Curl_getworkingpath(data, sshc->homedir, &sshp->path);
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
    {
      const char *cp;

      /*
       * Support some of the "FTP" commands
       *
       * 'sshc->quote_item' is already verified to be non-NULL before it
       * switched to this state.
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
                            sshp->path);
        if(!tmp) {
          result = CURLE_OUT_OF_MEMORY;
          state(data, SSH_SFTP_CLOSE);
          sshc->nextstate = SSH_NO_STATE;
          break;
        }
        Curl_debug(data, CURLINFO_HEADER_OUT, (char *)"PWD\n", 4);
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
        break;
      }
      {
        /*
         * the arguments following the command must be separated from the
         * command with a space so we can check for it unconditionally
         */
        cp = strchr(cmd, ' ');
        if(!cp) {
          failf(data, "Syntax error command '%s'. Missing parameter!",
                cmd);
          state(data, SSH_SFTP_CLOSE);
          sshc->nextstate = SSH_NO_STATE;
          sshc->actualcode = CURLE_QUOTE_ERROR;
          break;
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
            failf(data, "Syntax error: Bad first parameter to '%s'", cmd);
          state(data, SSH_SFTP_CLOSE);
          sshc->nextstate = SSH_NO_STATE;
          sshc->actualcode = result;
          break;
        }

        /*
         * SFTP is a binary protocol, so we don't send text commands
         * to the server. Instead, we scan for commands used by
         * OpenSSH's sftp program and call the appropriate libssh2
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
              failf(data, "Syntax error in %s: Bad second parameter", cmd);
            Curl_safefree(sshc->quote_path1);
            state(data, SSH_SFTP_CLOSE);
            sshc->nextstate = SSH_NO_STATE;
            sshc->actualcode = result;
            break;
          }
          memset(&sshp->quote_attrs, 0, sizeof(LIBSSH2_SFTP_ATTRIBUTES));
          state(data, SSH_SFTP_QUOTE_STAT);
          break;
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
              failf(data,
                    "Syntax error in ln/symlink: Bad second parameter");
            Curl_safefree(sshc->quote_path1);
            state(data, SSH_SFTP_CLOSE);
            sshc->nextstate = SSH_NO_STATE;
            sshc->actualcode = result;
            break;
          }
          state(data, SSH_SFTP_QUOTE_SYMLINK);
          break;
        }
        else if(strncasecompare(cmd, "mkdir ", 6)) {
          /* create dir */
          state(data, SSH_SFTP_QUOTE_MKDIR);
          break;
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
            break;
          }
          state(data, SSH_SFTP_QUOTE_RENAME);
          break;
        }
        else if(strncasecompare(cmd, "rmdir ", 6)) {
          /* delete dir */
          state(data, SSH_SFTP_QUOTE_RMDIR);
          break;
        }
        else if(strncasecompare(cmd, "rm ", 3)) {
          state(data, SSH_SFTP_QUOTE_UNLINK);
          break;
        }
#ifdef HAS_STATVFS_SUPPORT
        else if(strncasecompare(cmd, "statvfs ", 8)) {
          state(data, SSH_SFTP_QUOTE_STATVFS);
          break;
        }
#endif

        failf(data, "Unknown SFTP command");
        Curl_safefree(sshc->quote_path1);
        Curl_safefree(sshc->quote_path2);
        state(data, SSH_SFTP_CLOSE);
        sshc->nextstate = SSH_NO_STATE;
        sshc->actualcode = CURLE_QUOTE_ERROR;
        break;
      }
    }
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
    {
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

      if(!strncasecompare(cmd, "chmod", 5)) {
        /* Since chown and chgrp only set owner OR group but libssh2 wants to
         * set them both at once, we need to obtain the current ownership
         * first.  This takes an extra protocol round trip.
         */
        rc = libssh2_sftp_stat_ex(sshc->sftp_session, sshc->quote_path2,
                                  curlx_uztoui(strlen(sshc->quote_path2)),
                                  LIBSSH2_SFTP_STAT,
                                  &sshp->quote_attrs);
        if(rc == LIBSSH2_ERROR_EAGAIN) {
          break;
        }
        if(rc && !sshc->acceptfail) { /* get those attributes */
          sftperr = libssh2_sftp_last_error(sshc->sftp_session);
          Curl_safefree(sshc->quote_path1);
          Curl_safefree(sshc->quote_path2);
          failf(data, "Attempt to get SFTP stats failed: %s",
                sftp_libssh2_strerror(sftperr));
          state(data, SSH_SFTP_CLOSE);
          sshc->nextstate = SSH_NO_STATE;
          sshc->actualcode = CURLE_QUOTE_ERROR;
          break;
        }
      }

      /* Now set the new attributes... */
      if(strncasecompare(cmd, "chgrp", 5)) {
        sshp->quote_attrs.gid = strtoul(sshc->quote_path1, NULL, 10);
        sshp->quote_attrs.flags = LIBSSH2_SFTP_ATTR_UIDGID;
        if(sshp->quote_attrs.gid == 0 && !ISDIGIT(sshc->quote_path1[0]) &&
           !sshc->acceptfail) {
          Curl_safefree(sshc->quote_path1);
          Curl_safefree(sshc->quote_path2);
          failf(data, "Syntax error: chgrp gid not a number");
          state(data, SSH_SFTP_CLOSE);
          sshc->nextstate = SSH_NO_STATE;
          sshc->actualcode = CURLE_QUOTE_ERROR;
          break;
        }
      }
      else if(strncasecompare(cmd, "chmod", 5)) {
        sshp->quote_attrs.permissions = strtoul(sshc->quote_path1, NULL, 8);
        sshp->quote_attrs.flags = LIBSSH2_SFTP_ATTR_PERMISSIONS;
        /* permissions are octal */
        if(sshp->quote_attrs.permissions == 0 &&
           !ISDIGIT(sshc->quote_path1[0])) {
          Curl_safefree(sshc->quote_path1);
          Curl_safefree(sshc->quote_path2);
          failf(data, "Syntax error: chmod permissions not a number");
          state(data, SSH_SFTP_CLOSE);
          sshc->nextstate = SSH_NO_STATE;
          sshc->actualcode = CURLE_QUOTE_ERROR;
          break;
        }
      }
      else if(strncasecompare(cmd, "chown", 5)) {
        sshp->quote_attrs.uid = strtoul(sshc->quote_path1, NULL, 10);
        sshp->quote_attrs.flags = LIBSSH2_SFTP_ATTR_UIDGID;
        if(sshp->quote_attrs.uid == 0 && !ISDIGIT(sshc->quote_path1[0]) &&
           !sshc->acceptfail) {
          Curl_safefree(sshc->quote_path1);
          Curl_safefree(sshc->quote_path2);
          failf(data, "Syntax error: chown uid not a number");
          state(data, SSH_SFTP_CLOSE);
          sshc->nextstate = SSH_NO_STATE;
          sshc->actualcode = CURLE_QUOTE_ERROR;
          break;
        }
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
          break;
        }
        sshp->quote_attrs.atime = (unsigned long)date;
        sshp->quote_attrs.flags = LIBSSH2_SFTP_ATTR_ACMODTIME;
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
          break;
        }
        sshp->quote_attrs.mtime = (unsigned long)date;
        sshp->quote_attrs.flags = LIBSSH2_SFTP_ATTR_ACMODTIME;
      }

      /* Now send the completed structure... */
      state(data, SSH_SFTP_QUOTE_SETSTAT);
      break;
    }

    case SSH_SFTP_QUOTE_SETSTAT:
      rc = libssh2_sftp_stat_ex(sshc->sftp_session, sshc->quote_path2,
                                curlx_uztoui(strlen(sshc->quote_path2)),
                                LIBSSH2_SFTP_SETSTAT,
                                &sshp->quote_attrs);
      if(rc == LIBSSH2_ERROR_EAGAIN) {
        break;
      }
      if(rc && !sshc->acceptfail) {
        sftperr = libssh2_sftp_last_error(sshc->sftp_session);
        Curl_safefree(sshc->quote_path1);
        Curl_safefree(sshc->quote_path2);
        failf(data, "Attempt to set SFTP stats failed: %s",
              sftp_libssh2_strerror(sftperr));
        state(data, SSH_SFTP_CLOSE);
        sshc->nextstate = SSH_NO_STATE;
        sshc->actualcode = CURLE_QUOTE_ERROR;
        break;
      }
      state(data, SSH_SFTP_NEXT_QUOTE);
      break;

    case SSH_SFTP_QUOTE_SYMLINK:
      rc = libssh2_sftp_symlink_ex(sshc->sftp_session, sshc->quote_path1,
                                   curlx_uztoui(strlen(sshc->quote_path1)),
                                   sshc->quote_path2,
                                   curlx_uztoui(strlen(sshc->quote_path2)),
                                   LIBSSH2_SFTP_SYMLINK);
      if(rc == LIBSSH2_ERROR_EAGAIN) {
        break;
      }
      if(rc && !sshc->acceptfail) {
        sftperr = libssh2_sftp_last_error(sshc->sftp_session);
        Curl_safefree(sshc->quote_path1);
        Curl_safefree(sshc->quote_path2);
        failf(data, "symlink command failed: %s",
              sftp_libssh2_strerror(sftperr));
        state(data, SSH_SFTP_CLOSE);
        sshc->nextstate = SSH_NO_STATE;
        sshc->actualcode = CURLE_QUOTE_ERROR;
        break;
      }
      state(data, SSH_SFTP_NEXT_QUOTE);
      break;

    case SSH_SFTP_QUOTE_MKDIR:
      rc = libssh2_sftp_mkdir_ex(sshc->sftp_session, sshc->quote_path1,
                                 curlx_uztoui(strlen(sshc->quote_path1)),
                                 data->set.new_directory_perms);
      if(rc == LIBSSH2_ERROR_EAGAIN) {
        break;
      }
      if(rc && !sshc->acceptfail) {
        sftperr = libssh2_sftp_last_error(sshc->sftp_session);
        Curl_safefree(sshc->quote_path1);
        failf(data, "mkdir command failed: %s",
              sftp_libssh2_strerror(sftperr));
        state(data, SSH_SFTP_CLOSE);
        sshc->nextstate = SSH_NO_STATE;
        sshc->actualcode = CURLE_QUOTE_ERROR;
        break;
      }
      state(data, SSH_SFTP_NEXT_QUOTE);
      break;

    case SSH_SFTP_QUOTE_RENAME:
      rc = libssh2_sftp_rename_ex(sshc->sftp_session, sshc->quote_path1,
                                  curlx_uztoui(strlen(sshc->quote_path1)),
                                  sshc->quote_path2,
                                  curlx_uztoui(strlen(sshc->quote_path2)),
                                  LIBSSH2_SFTP_RENAME_OVERWRITE |
                                  LIBSSH2_SFTP_RENAME_ATOMIC |
                                  LIBSSH2_SFTP_RENAME_NATIVE);

      if(rc == LIBSSH2_ERROR_EAGAIN) {
        break;
      }
      if(rc && !sshc->acceptfail) {
        sftperr = libssh2_sftp_last_error(sshc->sftp_session);
        Curl_safefree(sshc->quote_path1);
        Curl_safefree(sshc->quote_path2);
        failf(data, "rename command failed: %s",
              sftp_libssh2_strerror(sftperr));
        state(data, SSH_SFTP_CLOSE);
        sshc->nextstate = SSH_NO_STATE;
        sshc->actualcode = CURLE_QUOTE_ERROR;
        break;
      }
      state(data, SSH_SFTP_NEXT_QUOTE);
      break;

    case SSH_SFTP_QUOTE_RMDIR:
      rc = libssh2_sftp_rmdir_ex(sshc->sftp_session, sshc->quote_path1,
                                 curlx_uztoui(strlen(sshc->quote_path1)));
      if(rc == LIBSSH2_ERROR_EAGAIN) {
        break;
      }
      if(rc && !sshc->acceptfail) {
        sftperr = libssh2_sftp_last_error(sshc->sftp_session);
        Curl_safefree(sshc->quote_path1);
        failf(data, "rmdir command failed: %s",
              sftp_libssh2_strerror(sftperr));
        state(data, SSH_SFTP_CLOSE);
        sshc->nextstate = SSH_NO_STATE;
        sshc->actualcode = CURLE_QUOTE_ERROR;
        break;
      }
      state(data, SSH_SFTP_NEXT_QUOTE);
      break;

    case SSH_SFTP_QUOTE_UNLINK:
      rc = libssh2_sftp_unlink_ex(sshc->sftp_session, sshc->quote_path1,
                                  curlx_uztoui(strlen(sshc->quote_path1)));
      if(rc == LIBSSH2_ERROR_EAGAIN) {
        break;
      }
      if(rc && !sshc->acceptfail) {
        sftperr = libssh2_sftp_last_error(sshc->sftp_session);
        Curl_safefree(sshc->quote_path1);
        failf(data, "rm command failed: %s", sftp_libssh2_strerror(sftperr));
        state(data, SSH_SFTP_CLOSE);
        sshc->nextstate = SSH_NO_STATE;
        sshc->actualcode = CURLE_QUOTE_ERROR;
        break;
      }
      state(data, SSH_SFTP_NEXT_QUOTE);
      break;

#ifdef HAS_STATVFS_SUPPORT
    case SSH_SFTP_QUOTE_STATVFS:
    {
      LIBSSH2_SFTP_STATVFS statvfs;
      rc = libssh2_sftp_statvfs(sshc->sftp_session, sshc->quote_path1,
                                curlx_uztoui(strlen(sshc->quote_path1)),
                                &statvfs);

      if(rc == LIBSSH2_ERROR_EAGAIN) {
        break;
      }
      if(rc && !sshc->acceptfail) {
        sftperr = libssh2_sftp_last_error(sshc->sftp_session);
        Curl_safefree(sshc->quote_path1);
        failf(data, "statvfs command failed: %s",
              sftp_libssh2_strerror(sftperr));
        state(data, SSH_SFTP_CLOSE);
        sshc->nextstate = SSH_NO_STATE;
        sshc->actualcode = CURLE_QUOTE_ERROR;
        break;
      }
      else if(rc == 0) {
        char *tmp = aprintf("statvfs:\n"
                            "f_bsize: %llu\n" "f_frsize: %llu\n"
                            "f_blocks: %llu\n" "f_bfree: %llu\n"
                            "f_bavail: %llu\n" "f_files: %llu\n"
                            "f_ffree: %llu\n" "f_favail: %llu\n"
                            "f_fsid: %llu\n" "f_flag: %llu\n"
                            "f_namemax: %llu\n",
                            statvfs.f_bsize, statvfs.f_frsize,
                            statvfs.f_blocks, statvfs.f_bfree,
                            statvfs.f_bavail, statvfs.f_files,
                            statvfs.f_ffree, statvfs.f_favail,
                            statvfs.f_fsid, statvfs.f_flag,
                            statvfs.f_namemax);
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
#endif
    case SSH_SFTP_GETINFO:
    {
      if(data->set.get_filetime) {
        state(data, SSH_SFTP_FILETIME);
      }
      else {
        state(data, SSH_SFTP_TRANS_INIT);
      }
      break;
    }

    case SSH_SFTP_FILETIME:
    {
      LIBSSH2_SFTP_ATTRIBUTES attrs;

      rc = libssh2_sftp_stat_ex(sshc->sftp_session, sshp->path,
                                curlx_uztoui(strlen(sshp->path)),
                                LIBSSH2_SFTP_STAT, &attrs);
      if(rc == LIBSSH2_ERROR_EAGAIN) {
        break;
      }
      if(rc == 0) {
        data->info.filetime = attrs.mtime;
      }

      state(data, SSH_SFTP_TRANS_INIT);
      break;
    }

    case SSH_SFTP_TRANS_INIT:
      if(data->set.upload)
        state(data, SSH_SFTP_UPLOAD_INIT);
      else {
        if(sshp->path[strlen(sshp->path)-1] == '/')
          state(data, SSH_SFTP_READDIR_INIT);
        else
          state(data, SSH_SFTP_DOWNLOAD_INIT);
      }
      break;

    case SSH_SFTP_UPLOAD_INIT:
    {
      unsigned long flags;
      /*
       * NOTE!!!  libssh2 requires that the destination path is a full path
       *          that includes the destination file and name OR ends in a "/"
       *          If this is not done the destination file will be named the
       *          same name as the last directory in the path.
       */

      if(data->state.resume_from) {
        LIBSSH2_SFTP_ATTRIBUTES attrs;
        if(data->state.resume_from < 0) {
          rc = libssh2_sftp_stat_ex(sshc->sftp_session, sshp->path,
                                    curlx_uztoui(strlen(sshp->path)),
                                    LIBSSH2_SFTP_STAT, &attrs);
          if(rc == LIBSSH2_ERROR_EAGAIN) {
            break;
          }
          if(rc) {
            data->state.resume_from = 0;
          }
          else {
            curl_off_t size = attrs.filesize;
            if(size < 0) {
              failf(data, "Bad file size (%" CURL_FORMAT_CURL_OFF_T ")", size);
              return CURLE_BAD_DOWNLOAD_RESUME;
            }
            data->state.resume_from = attrs.filesize;
          }
        }
      }

      if(data->set.remote_append)
        /* Try to open for append, but create if nonexisting */
        flags = LIBSSH2_FXF_WRITE|LIBSSH2_FXF_CREAT|LIBSSH2_FXF_APPEND;
      else if(data->state.resume_from > 0)
        /* If we have restart position then open for append */
        flags = LIBSSH2_FXF_WRITE|LIBSSH2_FXF_APPEND;
      else
        /* Clear file before writing (normal behavior) */
        flags = LIBSSH2_FXF_WRITE|LIBSSH2_FXF_CREAT|LIBSSH2_FXF_TRUNC;

      sshc->sftp_handle =
        libssh2_sftp_open_ex(sshc->sftp_session, sshp->path,
                             curlx_uztoui(strlen(sshp->path)),
                             flags, data->set.new_file_perms,
                             LIBSSH2_SFTP_OPENFILE);

      if(!sshc->sftp_handle) {
        rc = libssh2_session_last_errno(sshc->ssh_session);

        if(LIBSSH2_ERROR_EAGAIN == rc)
          break;

        if(LIBSSH2_ERROR_SFTP_PROTOCOL == rc)
          /* only when there was an SFTP protocol error can we extract
             the sftp error! */
          sftperr = libssh2_sftp_last_error(sshc->sftp_session);
        else
          sftperr = LIBSSH2_FX_OK; /* not an sftp error at all */

        if(sshc->secondCreateDirs) {
          state(data, SSH_SFTP_CLOSE);
          sshc->actualcode = sftperr != LIBSSH2_FX_OK ?
            sftp_libssh2_error_to_CURLE(sftperr):CURLE_SSH;
          failf(data, "Creating the dir/file failed: %s",
                sftp_libssh2_strerror(sftperr));
          break;
        }
        if(((sftperr == LIBSSH2_FX_NO_SUCH_FILE) ||
            (sftperr == LIBSSH2_FX_FAILURE) ||
            (sftperr == LIBSSH2_FX_NO_SUCH_PATH)) &&
           (data->set.ftp_create_missing_dirs &&
            (strlen(sshp->path) > 1))) {
          /* try to create the path remotely */
          rc = 0; /* clear rc and continue */
          sshc->secondCreateDirs = 1;
          state(data, SSH_SFTP_CREATE_DIRS_INIT);
          break;
        }
        state(data, SSH_SFTP_CLOSE);
        sshc->actualcode = sftperr != LIBSSH2_FX_OK ?
          sftp_libssh2_error_to_CURLE(sftperr):CURLE_SSH;
        if(!sshc->actualcode) {
          /* Sometimes, for some reason libssh2_sftp_last_error() returns zero
             even though libssh2_sftp_open() failed previously! We need to
             work around that! */
          sshc->actualcode = CURLE_SSH;
          sftperr = LIBSSH2_FX_OK;
        }
        failf(data, "Upload failed: %s (%lu/%d)",
              sftperr != LIBSSH2_FX_OK ?
              sftp_libssh2_strerror(sftperr):"ssh error",
              sftperr, rc);
        break;
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

            size_t actuallyread;
            Curl_set_in_callback(data, true);
            actuallyread = data->state.fread_func(data->state.buffer, 1,
                                                  readthisamountnow,
                                                  data->state.in);
            Curl_set_in_callback(data, false);

            passed += actuallyread;
            if((actuallyread == 0) || (actuallyread > readthisamountnow)) {
              /* this checks for greater-than only to make sure that the
                 CURL_READFUNC_ABORT return code still aborts */
              failf(data, "Failed to read data");
              return CURLE_FTP_COULDNT_USE_REST;
            }
          } while(passed < data->state.resume_from);
        }

        /* now, decrease the size of the read */
        if(data->state.infilesize > 0) {
          data->state.infilesize -= data->state.resume_from;
          data->req.size = data->state.infilesize;
          Curl_pgrsSetUploadSize(data, data->state.infilesize);
        }

        SFTP_SEEK(sshc->sftp_handle, data->state.resume_from);
      }
      if(data->state.infilesize > 0) {
        data->req.size = data->state.infilesize;
        Curl_pgrsSetUploadSize(data, data->state.infilesize);
      }
      /* upload data */
      Curl_setup_transfer(data, -1, -1, FALSE, FIRSTSOCKET);

      /* not set by Curl_setup_transfer to preserve keepon bits */
      conn->sockfd = conn->writesockfd;

      if(result) {
        state(data, SSH_SFTP_CLOSE);
        sshc->actualcode = result;
      }
      else {
        /* store this original bitmask setup to use later on if we can't
           figure out a "real" bitmask */
        sshc->orig_waitfor = data->req.keepon;

        /* we want to use the _sending_ function even when the socket turns
           out readable as the underlying libssh2 sftp send function will deal
           with both accordingly */
        conn->cselect_bits = CURL_CSELECT_OUT;

        /* since we don't really wait for anything at this point, we want the
           state machine to move on as soon as possible so we set a very short
           timeout here */
        Curl_expire(data, 0, EXPIRE_RUN_NOW);

        state(data, SSH_STOP);
      }
      break;
    }

    case SSH_SFTP_CREATE_DIRS_INIT:
      if(strlen(sshp->path) > 1) {
        sshc->slash_pos = sshp->path + 1; /* ignore the leading '/' */
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

        infof(data, "Creating directory '%s'", sshp->path);
        state(data, SSH_SFTP_CREATE_DIRS_MKDIR);
        break;
      }
      state(data, SSH_SFTP_UPLOAD_INIT);
      break;

    case SSH_SFTP_CREATE_DIRS_MKDIR:
      /* 'mode' - parameter is preliminary - default to 0644 */
      rc = libssh2_sftp_mkdir_ex(sshc->sftp_session, sshp->path,
                                 curlx_uztoui(strlen(sshp->path)),
                                 data->set.new_directory_perms);
      if(rc == LIBSSH2_ERROR_EAGAIN) {
        break;
      }
      *sshc->slash_pos = '/';
      ++sshc->slash_pos;
      if(rc < 0) {
        /*
         * Abort if failure wasn't that the dir already exists or the
         * permission was denied (creation might succeed further down the
         * path) - retry on unspecific FAILURE also
         */
        sftperr = libssh2_sftp_last_error(sshc->sftp_session);
        if((sftperr != LIBSSH2_FX_FILE_ALREADY_EXISTS) &&
           (sftperr != LIBSSH2_FX_FAILURE) &&
           (sftperr != LIBSSH2_FX_PERMISSION_DENIED)) {
          result = sftp_libssh2_error_to_CURLE(sftperr);
          state(data, SSH_SFTP_CLOSE);
          sshc->actualcode = result?result:CURLE_SSH;
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
      sshc->sftp_handle = libssh2_sftp_open_ex(sshc->sftp_session,
                                               sshp->path,
                                               curlx_uztoui(
                                                 strlen(sshp->path)),
                                               0, 0, LIBSSH2_SFTP_OPENDIR);
      if(!sshc->sftp_handle) {
        if(libssh2_session_last_errno(sshc->ssh_session) ==
           LIBSSH2_ERROR_EAGAIN) {
          rc = LIBSSH2_ERROR_EAGAIN;
          break;
        }
        sftperr = libssh2_sftp_last_error(sshc->sftp_session);
        failf(data, "Could not open directory for reading: %s",
              sftp_libssh2_strerror(sftperr));
        state(data, SSH_SFTP_CLOSE);
        result = sftp_libssh2_error_to_CURLE(sftperr);
        sshc->actualcode = result?result:CURLE_SSH;
        break;
      }
      sshp->readdir_filename = malloc(PATH_MAX + 1);
      if(!sshp->readdir_filename) {
        state(data, SSH_SFTP_CLOSE);
        sshc->actualcode = CURLE_OUT_OF_MEMORY;
        break;
      }
      sshp->readdir_longentry = malloc(PATH_MAX + 1);
      if(!sshp->readdir_longentry) {
        Curl_safefree(sshp->readdir_filename);
        state(data, SSH_SFTP_CLOSE);
        sshc->actualcode = CURLE_OUT_OF_MEMORY;
        break;
      }
      Curl_dyn_init(&sshp->readdir, PATH_MAX * 2);
      state(data, SSH_SFTP_READDIR);
      break;

    case SSH_SFTP_READDIR:
      rc = libssh2_sftp_readdir_ex(sshc->sftp_handle,
                                   sshp->readdir_filename,
                                   PATH_MAX,
                                   sshp->readdir_longentry,
                                   PATH_MAX,
                                   &sshp->readdir_attrs);
      if(rc == LIBSSH2_ERROR_EAGAIN) {
        break;
      }
      if(rc > 0) {
        readdir_len = (size_t) rc;
        sshp->readdir_filename[readdir_len] = '\0';

        if(data->set.list_only) {
          result = Curl_client_write(data, CLIENTWRITE_BODY,
                                     sshp->readdir_filename,
                                     readdir_len);
          if(!result)
            result = Curl_client_write(data, CLIENTWRITE_BODY,
                                       (char *)"\n", 1);
          if(result) {
            state(data, SSH_STOP);
            break;
          }
          /* since this counts what we send to the client, we include the
             newline in this counter */
          data->req.bytecount += readdir_len + 1;

          /* output debug output if that is requested */
          Curl_debug(data, CURLINFO_DATA_IN, sshp->readdir_filename,
                     readdir_len);
          Curl_debug(data, CURLINFO_DATA_IN, (char *)"\n", 1);
        }
        else {
          result = Curl_dyn_add(&sshp->readdir, sshp->readdir_longentry);

          if(!result) {
            if((sshp->readdir_attrs.flags & LIBSSH2_SFTP_ATTR_PERMISSIONS) &&
               ((sshp->readdir_attrs.permissions & LIBSSH2_SFTP_S_IFMT) ==
                LIBSSH2_SFTP_S_IFLNK)) {
              Curl_dyn_init(&sshp->readdir_link, PATH_MAX);
              result = Curl_dyn_add(&sshp->readdir_link, sshp->path);
              state(data, SSH_SFTP_READDIR_LINK);
              if(!result)
                break;
            }
            else {
              state(data, SSH_SFTP_READDIR_BOTTOM);
              break;
            }
          }
          sshc->actualcode = result;
          state(data, SSH_SFTP_CLOSE);
          break;
        }
      }
      else if(rc == 0) {
        Curl_safefree(sshp->readdir_filename);
        Curl_safefree(sshp->readdir_longentry);
        state(data, SSH_SFTP_READDIR_DONE);
        break;
      }
      else if(rc < 0) {
        sftperr = libssh2_sftp_last_error(sshc->sftp_session);
        result = sftp_libssh2_error_to_CURLE(sftperr);
        sshc->actualcode = result?result:CURLE_SSH;
        failf(data, "Could not open remote file for reading: %s :: %d",
              sftp_libssh2_strerror(sftperr),
              libssh2_session_last_errno(sshc->ssh_session));
        Curl_safefree(sshp->readdir_filename);
        Curl_safefree(sshp->readdir_longentry);
        state(data, SSH_SFTP_CLOSE);
        break;
      }
      break;

    case SSH_SFTP_READDIR_LINK:
      rc =
        libssh2_sftp_symlink_ex(sshc->sftp_session,
                                Curl_dyn_ptr(&sshp->readdir_link),
                                (int)Curl_dyn_len(&sshp->readdir_link),
                                sshp->readdir_filename,
                                PATH_MAX, LIBSSH2_SFTP_READLINK);
      if(rc == LIBSSH2_ERROR_EAGAIN) {
        break;
      }
      Curl_dyn_free(&sshp->readdir_link);

      /* append filename and extra output */
      result = Curl_dyn_addf(&sshp->readdir, " -> %s", sshp->readdir_filename);

      if(result) {
        sshc->readdir_line = NULL;
        Curl_safefree(sshp->readdir_filename);
        Curl_safefree(sshp->readdir_longentry);
        state(data, SSH_SFTP_CLOSE);
        sshc->actualcode = result;
        break;
      }

      state(data, SSH_SFTP_READDIR_BOTTOM);
      break;

    case SSH_SFTP_READDIR_BOTTOM:
      result = Curl_dyn_addn(&sshp->readdir, "\n", 1);
      if(!result)
        result = Curl_client_write(data, CLIENTWRITE_BODY,
                                   Curl_dyn_ptr(&sshp->readdir),
                                   Curl_dyn_len(&sshp->readdir));

      if(!result) {
        /* output debug output if that is requested */
        Curl_debug(data, CURLINFO_DATA_IN,
                   Curl_dyn_ptr(&sshp->readdir),
                   Curl_dyn_len(&sshp->readdir));
        data->req.bytecount += Curl_dyn_len(&sshp->readdir);
      }
      if(result) {
        Curl_dyn_free(&sshp->readdir);
        state(data, SSH_STOP);
      }
      else {
        Curl_dyn_reset(&sshp->readdir);
        state(data, SSH_SFTP_READDIR);
      }
      break;

    case SSH_SFTP_READDIR_DONE:
      if(libssh2_sftp_closedir(sshc->sftp_handle) ==
         LIBSSH2_ERROR_EAGAIN) {
        rc = LIBSSH2_ERROR_EAGAIN;
        break;
      }
      sshc->sftp_handle = NULL;
      Curl_safefree(sshp->readdir_filename);
      Curl_safefree(sshp->readdir_longentry);

      /* no data to transfer */
      Curl_setup_transfer(data, -1, -1, FALSE, -1);
      state(data, SSH_STOP);
      break;

    case SSH_SFTP_DOWNLOAD_INIT:
      /*
       * Work on getting the specified file
       */
      sshc->sftp_handle =
        libssh2_sftp_open_ex(sshc->sftp_session, sshp->path,
                             curlx_uztoui(strlen(sshp->path)),
                             LIBSSH2_FXF_READ, data->set.new_file_perms,
                             LIBSSH2_SFTP_OPENFILE);
      if(!sshc->sftp_handle) {
        if(libssh2_session_last_errno(sshc->ssh_session) ==
           LIBSSH2_ERROR_EAGAIN) {
          rc = LIBSSH2_ERROR_EAGAIN;
          break;
        }
        sftperr = libssh2_sftp_last_error(sshc->sftp_session);
        failf(data, "Could not open remote file for reading: %s",
              sftp_libssh2_strerror(sftperr));
        state(data, SSH_SFTP_CLOSE);
        result = sftp_libssh2_error_to_CURLE(sftperr);
        sshc->actualcode = result?result:CURLE_SSH;
        break;
      }
      state(data, SSH_SFTP_DOWNLOAD_STAT);
      break;

    case SSH_SFTP_DOWNLOAD_STAT:
    {
      LIBSSH2_SFTP_ATTRIBUTES attrs;

      rc = libssh2_sftp_stat_ex(sshc->sftp_session, sshp->path,
                                curlx_uztoui(strlen(sshp->path)),
                                LIBSSH2_SFTP_STAT, &attrs);
      if(rc == LIBSSH2_ERROR_EAGAIN) {
        break;
      }
      if(rc ||
         !(attrs.flags & LIBSSH2_SFTP_ATTR_SIZE) ||
         (attrs.filesize == 0)) {
        /*
         * libssh2_sftp_open() didn't return an error, so maybe the server
         * just doesn't support stat()
         * OR the server doesn't return a file size with a stat()
         * OR file size is 0
         */
        data->req.size = -1;
        data->req.maxdownload = -1;
        Curl_pgrsSetDownloadSize(data, -1);
      }
      else {
        curl_off_t size = attrs.filesize;

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
          if(from_t == CURL_OFFT_FLOW)
            return CURLE_RANGE_ERROR;
          while(*ptr && (ISSPACE(*ptr) || (*ptr == '-')))
            ptr++;
          to_t = curlx_strtoofft(ptr, &ptr2, 0, &to);
          if(to_t == CURL_OFFT_FLOW)
            return CURLE_RANGE_ERROR;
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
                  CURL_FORMAT_CURL_OFF_T ")", from, attrs.filesize);
            return CURLE_BAD_DOWNLOAD_RESUME;
          }
          if(from > to) {
            from = to;
            size = 0;
          }
          else {
            size = to - from + 1;
          }

          SFTP_SEEK(sshc->sftp_handle, from);
        }
        data->req.size = size;
        data->req.maxdownload = size;
        Curl_pgrsSetDownloadSize(data, size);
      }

      /* We can resume if we can seek to the resume position */
      if(data->state.resume_from) {
        if(data->state.resume_from < 0) {
          /* We're supposed to download the last abs(from) bytes */
          if((curl_off_t)attrs.filesize < -data->state.resume_from) {
            failf(data, "Offset (%"
                  CURL_FORMAT_CURL_OFF_T ") was beyond file size (%"
                  CURL_FORMAT_CURL_OFF_T ")",
                  data->state.resume_from, attrs.filesize);
            return CURLE_BAD_DOWNLOAD_RESUME;
          }
          /* download from where? */
          data->state.resume_from += attrs.filesize;
        }
        else {
          if((curl_off_t)attrs.filesize < data->state.resume_from) {
            failf(data, "Offset (%" CURL_FORMAT_CURL_OFF_T
                  ") was beyond file size (%" CURL_FORMAT_CURL_OFF_T ")",
                  data->state.resume_from, attrs.filesize);
            return CURLE_BAD_DOWNLOAD_RESUME;
          }
        }
        /* Now store the number of bytes we are expected to download */
        data->req.size = attrs.filesize - data->state.resume_from;
        data->req.maxdownload = attrs.filesize - data->state.resume_from;
        Curl_pgrsSetDownloadSize(data,
                                 attrs.filesize - data->state.resume_from);
        SFTP_SEEK(sshc->sftp_handle, data->state.resume_from);
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
       out writableable as the underlying libssh2 recv function will deal
       with both accordingly */
    conn->cselect_bits = CURL_CSELECT_IN;

    if(result) {
      /* this should never occur; the close state should be entered
         at the time the error occurs */
      state(data, SSH_SFTP_CLOSE);
      sshc->actualcode = result;
    }
    else {
      state(data, SSH_STOP);
    }
    break;

    case SSH_SFTP_CLOSE:
      if(sshc->sftp_handle) {
        rc = libssh2_sftp_close(sshc->sftp_handle);
        if(rc == LIBSSH2_ERROR_EAGAIN) {
          break;
        }
        if(rc < 0) {
          char *err_msg = NULL;
          (void)libssh2_session_last_error(sshc->ssh_session,
                                           &err_msg, NULL, 0);
          infof(data, "Failed to close libssh2 file: %d %s", rc, err_msg);
        }
        sshc->sftp_handle = NULL;
      }

      Curl_safefree(sshp->path);

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

      if(sshc->sftp_handle) {
        rc = libssh2_sftp_close(sshc->sftp_handle);
        if(rc == LIBSSH2_ERROR_EAGAIN) {
          break;
        }
        if(rc < 0) {
          char *err_msg = NULL;
          (void)libssh2_session_last_error(sshc->ssh_session, &err_msg,
                                           NULL, 0);
          infof(data, "Failed to close libssh2 file: %d %s", rc, err_msg);
        }
        sshc->sftp_handle = NULL;
      }
      if(sshc->sftp_session) {
        rc = libssh2_sftp_shutdown(sshc->sftp_session);
        if(rc == LIBSSH2_ERROR_EAGAIN) {
          break;
        }
        if(rc < 0) {
          infof(data, "Failed to stop libssh2 sftp subsystem");
        }
        sshc->sftp_session = NULL;
      }

      Curl_safefree(sshc->homedir);
      data->state.most_recent_ftp_entrypath = NULL;

      state(data, SSH_SESSION_DISCONNECT);
      break;

    case SSH_SCP_TRANS_INIT:
      result = Curl_getworkingpath(data, sshc->homedir, &sshp->path);
      if(result) {
        sshc->actualcode = result;
        state(data, SSH_STOP);
        break;
      }

      if(data->set.upload) {
        if(data->state.infilesize < 0) {
          failf(data, "SCP requires a known file size for upload");
          sshc->actualcode = CURLE_UPLOAD_FAILED;
          state(data, SSH_SCP_CHANNEL_FREE);
          break;
        }
        state(data, SSH_SCP_UPLOAD_INIT);
      }
      else {
        state(data, SSH_SCP_DOWNLOAD_INIT);
      }
      break;

    case SSH_SCP_UPLOAD_INIT:
      /*
       * libssh2 requires that the destination path is a full path that
       * includes the destination file and name OR ends in a "/" .  If this is
       * not done the destination file will be named the same name as the last
       * directory in the path.
       */
      sshc->ssh_channel =
        SCP_SEND(sshc->ssh_session, sshp->path, data->set.new_file_perms,
                 data->state.infilesize);
      if(!sshc->ssh_channel) {
        int ssh_err;
        char *err_msg = NULL;

        if(libssh2_session_last_errno(sshc->ssh_session) ==
           LIBSSH2_ERROR_EAGAIN) {
          rc = LIBSSH2_ERROR_EAGAIN;
          break;
        }

        ssh_err = (int)(libssh2_session_last_error(sshc->ssh_session,
                                                   &err_msg, NULL, 0));
        failf(data, "%s", err_msg);
        state(data, SSH_SCP_CHANNEL_FREE);
        sshc->actualcode = libssh2_session_error_to_CURLE(ssh_err);
        /* Map generic errors to upload failed */
        if(sshc->actualcode == CURLE_SSH ||
           sshc->actualcode == CURLE_REMOTE_FILE_NOT_FOUND)
          sshc->actualcode = CURLE_UPLOAD_FAILED;
        break;
      }

      /* upload data */
      data->req.size = data->state.infilesize;
      Curl_pgrsSetUploadSize(data, data->state.infilesize);
      Curl_setup_transfer(data, -1, -1, FALSE, FIRSTSOCKET);

      /* not set by Curl_setup_transfer to preserve keepon bits */
      conn->sockfd = conn->writesockfd;

      if(result) {
        state(data, SSH_SCP_CHANNEL_FREE);
        sshc->actualcode = result;
      }
      else {
        /* store this original bitmask setup to use later on if we can't
           figure out a "real" bitmask */
        sshc->orig_waitfor = data->req.keepon;

        /* we want to use the _sending_ function even when the socket turns
           out readable as the underlying libssh2 scp send function will deal
           with both accordingly */
        conn->cselect_bits = CURL_CSELECT_OUT;

        state(data, SSH_STOP);
      }
      break;

    case SSH_SCP_DOWNLOAD_INIT:
    {
      curl_off_t bytecount;

      /*
       * We must check the remote file; if it is a directory no values will
       * be set in sb
       */

      /*
       * If support for >2GB files exists, use it.
       */

      /* get a fresh new channel from the ssh layer */
#if LIBSSH2_VERSION_NUM < 0x010700
      struct stat sb;
      memset(&sb, 0, sizeof(struct stat));
      sshc->ssh_channel = libssh2_scp_recv(sshc->ssh_session,
                                           sshp->path, &sb);
#else
      libssh2_struct_stat sb;
      memset(&sb, 0, sizeof(libssh2_struct_stat));
      sshc->ssh_channel = libssh2_scp_recv2(sshc->ssh_session,
                                            sshp->path, &sb);
#endif

      if(!sshc->ssh_channel) {
        int ssh_err;
        char *err_msg = NULL;

        if(libssh2_session_last_errno(sshc->ssh_session) ==
           LIBSSH2_ERROR_EAGAIN) {
          rc = LIBSSH2_ERROR_EAGAIN;
          break;
        }


        ssh_err = (int)(libssh2_session_last_error(sshc->ssh_session,
                                                   &err_msg, NULL, 0));
        failf(data, "%s", err_msg);
        state(data, SSH_SCP_CHANNEL_FREE);
        sshc->actualcode = libssh2_session_error_to_CURLE(ssh_err);
        break;
      }

      /* download data */
      bytecount = (curl_off_t)sb.st_size;
      data->req.maxdownload = (curl_off_t)sb.st_size;
      Curl_setup_transfer(data, FIRSTSOCKET, bytecount, FALSE, -1);

      /* not set by Curl_setup_transfer to preserve keepon bits */
      conn->writesockfd = conn->sockfd;

      /* we want to use the _receiving_ function even when the socket turns
         out writableable as the underlying libssh2 recv function will deal
         with both accordingly */
      conn->cselect_bits = CURL_CSELECT_IN;

      if(result) {
        state(data, SSH_SCP_CHANNEL_FREE);
        sshc->actualcode = result;
      }
      else
        state(data, SSH_STOP);
    }
    break;

    case SSH_SCP_DONE:
      if(data->set.upload)
        state(data, SSH_SCP_SEND_EOF);
      else
        state(data, SSH_SCP_CHANNEL_FREE);
      break;

    case SSH_SCP_SEND_EOF:
      if(sshc->ssh_channel) {
        rc = libssh2_channel_send_eof(sshc->ssh_channel);
        if(rc == LIBSSH2_ERROR_EAGAIN) {
          break;
        }
        if(rc) {
          char *err_msg = NULL;
          (void)libssh2_session_last_error(sshc->ssh_session,
                                           &err_msg, NULL, 0);
          infof(data, "Failed to send libssh2 channel EOF: %d %s",
                rc, err_msg);
        }
      }
      state(data, SSH_SCP_WAIT_EOF);
      break;

    case SSH_SCP_WAIT_EOF:
      if(sshc->ssh_channel) {
        rc = libssh2_channel_wait_eof(sshc->ssh_channel);
        if(rc == LIBSSH2_ERROR_EAGAIN) {
          break;
        }
        if(rc) {
          char *err_msg = NULL;
          (void)libssh2_session_last_error(sshc->ssh_session,
                                           &err_msg, NULL, 0);
          infof(data, "Failed to get channel EOF: %d %s", rc, err_msg);
        }
      }
      state(data, SSH_SCP_WAIT_CLOSE);
      break;

    case SSH_SCP_WAIT_CLOSE:
      if(sshc->ssh_channel) {
        rc = libssh2_channel_wait_closed(sshc->ssh_channel);
        if(rc == LIBSSH2_ERROR_EAGAIN) {
          break;
        }
        if(rc) {
          char *err_msg = NULL;
          (void)libssh2_session_last_error(sshc->ssh_session,
                                           &err_msg, NULL, 0);
          infof(data, "Channel failed to close: %d %s", rc, err_msg);
        }
      }
      state(data, SSH_SCP_CHANNEL_FREE);
      break;

    case SSH_SCP_CHANNEL_FREE:
      if(sshc->ssh_channel) {
        rc = libssh2_channel_free(sshc->ssh_channel);
        if(rc == LIBSSH2_ERROR_EAGAIN) {
          break;
        }
        if(rc < 0) {
          char *err_msg = NULL;
          (void)libssh2_session_last_error(sshc->ssh_session,
                                           &err_msg, NULL, 0);
          infof(data, "Failed to free libssh2 scp subsystem: %d %s",
                rc, err_msg);
        }
        sshc->ssh_channel = NULL;
      }
      DEBUGF(infof(data, "SCP DONE phase complete"));
#if 0 /* PREV */
      state(data, SSH_SESSION_DISCONNECT);
#endif
      state(data, SSH_STOP);
      result = sshc->actualcode;
      break;

    case SSH_SESSION_DISCONNECT:
      /* during weird times when we've been prematurely aborted, the channel
         is still alive when we reach this state and we MUST kill the channel
         properly first */
      if(sshc->ssh_channel) {
        rc = libssh2_channel_free(sshc->ssh_channel);
        if(rc == LIBSSH2_ERROR_EAGAIN) {
          break;
        }
        if(rc < 0) {
          char *err_msg = NULL;
          (void)libssh2_session_last_error(sshc->ssh_session,
                                           &err_msg, NULL, 0);
          infof(data, "Failed to free libssh2 scp subsystem: %d %s",
                rc, err_msg);
        }
        sshc->ssh_channel = NULL;
      }

      if(sshc->ssh_session) {
        rc = libssh2_session_disconnect(sshc->ssh_session, "Shutdown");
        if(rc == LIBSSH2_ERROR_EAGAIN) {
          break;
        }
        if(rc < 0) {
          char *err_msg = NULL;
          (void)libssh2_session_last_error(sshc->ssh_session,
                                           &err_msg, NULL, 0);
          infof(data, "Failed to disconnect libssh2 session: %d %s",
                rc, err_msg);
        }
      }

      Curl_safefree(sshc->homedir);
      data->state.most_recent_ftp_entrypath = NULL;

      state(data, SSH_SESSION_FREE);
      break;

    case SSH_SESSION_FREE:
#ifdef HAVE_LIBSSH2_KNOWNHOST_API
      if(sshc->kh) {
        libssh2_knownhost_free(sshc->kh);
        sshc->kh = NULL;
      }
#endif

#ifdef HAVE_LIBSSH2_AGENT_API
      if(sshc->ssh_agent) {
        rc = libssh2_agent_disconnect(sshc->ssh_agent);
        if(rc == LIBSSH2_ERROR_EAGAIN) {
          break;
        }
        if(rc < 0) {
          char *err_msg = NULL;
          (void)libssh2_session_last_error(sshc->ssh_session,
                                           &err_msg, NULL, 0);
          infof(data, "Failed to disconnect from libssh2 agent: %d %s",
                rc, err_msg);
        }
        libssh2_agent_free(sshc->ssh_agent);
        sshc->ssh_agent = NULL;

        /* NB: there is no need to free identities, they are part of internal
           agent stuff */
        sshc->sshagent_identity = NULL;
        sshc->sshagent_prev_identity = NULL;
      }
#endif

      if(sshc->ssh_session) {
        rc = libssh2_session_free(sshc->ssh_session);
        if(rc == LIBSSH2_ERROR_EAGAIN) {
          break;
        }
        if(rc < 0) {
          char *err_msg = NULL;
          (void)libssh2_session_last_error(sshc->ssh_session,
                                           &err_msg, NULL, 0);
          infof(data, "Failed to free libssh2 session: %d %s", rc, err_msg);
        }
        sshc->ssh_session = NULL;
      }

      /* worst-case scenario cleanup */

      DEBUGASSERT(sshc->ssh_session == NULL);
      DEBUGASSERT(sshc->ssh_channel == NULL);
      DEBUGASSERT(sshc->sftp_session == NULL);
      DEBUGASSERT(sshc->sftp_handle == NULL);
#ifdef HAVE_LIBSSH2_KNOWNHOST_API
      DEBUGASSERT(sshc->kh == NULL);
#endif
#ifdef HAVE_LIBSSH2_AGENT_API
      DEBUGASSERT(sshc->ssh_agent == NULL);
#endif

      Curl_safefree(sshc->rsa_pub);
      Curl_safefree(sshc->rsa);

      Curl_safefree(sshc->quote_path1);
      Curl_safefree(sshc->quote_path2);

      Curl_safefree(sshc->homedir);
      Curl_safefree(sshc->readdir_line);

      /* the code we are about to return */
      result = sshc->actualcode;

      memset(sshc, 0, sizeof(struct ssh_conn));

      connclose(conn, "SSH session free");
      sshc->state = SSH_SESSION_FREE; /* current */
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

  if(rc == LIBSSH2_ERROR_EAGAIN) {
    /* we would block, we need to wait for the socket to be ready (in the
       right direction too)! */
    *block = TRUE;
  }

  return result;
}

/* called by the multi interface to figure out what socket(s) to wait for and
   for what actions in the DO_DONE, PERFORM and WAITPERFORM states */
static int ssh_getsock(struct Curl_easy *data,
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

/*
 * When one of the libssh2 functions has returned LIBSSH2_ERROR_EAGAIN this
 * function is used to figure out in what direction and stores this info so
 * that the multi interface can take advantage of it. Make sure to call this
 * function in all cases so that when it _doesn't_ return EAGAIN we can
 * restore the default wait bits.
 */
static void ssh_block2waitfor(struct Curl_easy *data, bool block)
{
  struct connectdata *conn = data->conn;
  struct ssh_conn *sshc = &conn->proto.sshc;
  int dir = 0;
  if(block) {
    dir = libssh2_session_block_directions(sshc->ssh_session);
    if(dir) {
      /* translate the libssh2 define bits into our own bit defines */
      conn->waitfor = ((dir&LIBSSH2_SESSION_BLOCK_INBOUND)?KEEP_RECV:0) |
        ((dir&LIBSSH2_SESSION_BLOCK_OUTBOUND)?KEEP_SEND:0);
    }
  }
  if(!dir)
    /* It didn't block or libssh2 didn't reveal in which direction, put back
       the original set */
    conn->waitfor = sshc->orig_waitfor;
}

/* called repeatedly until done from multi.c */
static CURLcode ssh_multi_statemach(struct Curl_easy *data, bool *done)
{
  struct connectdata *conn = data->conn;
  struct ssh_conn *sshc = &conn->proto.sshc;
  CURLcode result = CURLE_OK;
  bool block; /* we store the status and use that to provide a ssh_getsock()
                 implementation */
  do {
    result = ssh_statemach_act(data, &block);
    *done = (sshc->state == SSH_STOP) ? TRUE : FALSE;
    /* if there's no error, it isn't done and it didn't EWOULDBLOCK, then
       try again */
  } while(!result && !*done && !block);
  ssh_block2waitfor(data, block);

  return result;
}

static CURLcode ssh_block_statemach(struct Curl_easy *data,
                                    struct connectdata *conn,
                                    bool disconnect)
{
  struct ssh_conn *sshc = &conn->proto.sshc;
  CURLcode result = CURLE_OK;
  struct curltime dis = Curl_now();

  while((sshc->state != SSH_STOP) && !result) {
    bool block;
    timediff_t left = 1000;
    struct curltime now = Curl_now();

    result = ssh_statemach_act(data, &block);
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
    else if(Curl_timediff(now, dis) > 1000) {
      /* disconnect timeout */
      failf(data, "Disconnect timed out");
      result = CURLE_OK;
      break;
    }

    if(block) {
      int dir = libssh2_session_block_directions(sshc->ssh_session);
      curl_socket_t sock = conn->sock[FIRSTSOCKET];
      curl_socket_t fd_read = CURL_SOCKET_BAD;
      curl_socket_t fd_write = CURL_SOCKET_BAD;
      if(LIBSSH2_SESSION_BLOCK_INBOUND & dir)
        fd_read = sock;
      if(LIBSSH2_SESSION_BLOCK_OUTBOUND & dir)
        fd_write = sock;
      /* wait for the socket to become ready */
      (void)Curl_socket_check(fd_read, CURL_SOCKET_BAD, fd_write,
                              left>1000?1000:left);
    }
  }

  return result;
}

/*
 * SSH setup and connection
 */
static CURLcode ssh_setup_connection(struct Curl_easy *data,
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

#ifndef CURL_DISABLE_PROXY
static ssize_t ssh_tls_recv(libssh2_socket_t sock, void *buffer,
                            size_t length, int flags, void **abstract)
{
  struct Curl_easy *data = (struct Curl_easy *)*abstract;
  ssize_t nread;
  CURLcode result;
  struct connectdata *conn = data->conn;
  Curl_recv *backup = conn->recv[0];
  struct ssh_conn *ssh = &conn->proto.sshc;
  (void)flags;

  /* swap in the TLS reader function for this call only, and then swap back
     the SSH one again */
  conn->recv[0] = ssh->tls_recv;
  result = Curl_read(data, sock, buffer, length, &nread);
  conn->recv[0] = backup;
  if(result == CURLE_AGAIN)
    return -EAGAIN; /* magic return code for libssh2 */
  else if(result)
    return -1; /* generic error */
  Curl_debug(data, CURLINFO_DATA_IN, (char *)buffer, (size_t)nread);
  return nread;
}

static ssize_t ssh_tls_send(libssh2_socket_t sock, const void *buffer,
                            size_t length, int flags, void **abstract)
{
  struct Curl_easy *data = (struct Curl_easy *)*abstract;
  ssize_t nwrite;
  CURLcode result;
  struct connectdata *conn = data->conn;
  Curl_send *backup = conn->send[0];
  struct ssh_conn *ssh = &conn->proto.sshc;
  (void)flags;

  /* swap in the TLS writer function for this call only, and then swap back
     the SSH one again */
  conn->send[0] = ssh->tls_send;
  result = Curl_write(data, sock, buffer, length, &nwrite);
  conn->send[0] = backup;
  if(result == CURLE_AGAIN)
    return -EAGAIN; /* magic return code for libssh2 */
  else if(result)
    return -1; /* error */
  Curl_debug(data, CURLINFO_DATA_OUT, (char *)buffer, (size_t)nwrite);
  return nwrite;
}
#endif

/*
 * Curl_ssh_connect() gets called from Curl_protocol_connect() to allow us to
 * do protocol-specific actions at connect-time.
 */
static CURLcode ssh_connect(struct Curl_easy *data, bool *done)
{
#ifdef CURL_LIBSSH2_DEBUG
  curl_socket_t sock;
#endif
  struct ssh_conn *sshc;
  CURLcode result;
  struct connectdata *conn = data->conn;

  /* initialize per-handle data if not already */
  if(!data->req.p.ssh) {
    result = ssh_setup_connection(data, conn);
    if(result)
      return result;
  }

  /* We default to persistent connections. We set this already in this connect
     function to make the re-use checks properly be able to check this bit. */
  connkeep(conn, "SSH default");

  sshc = &conn->proto.sshc;

#ifdef CURL_LIBSSH2_DEBUG
  if(conn->user) {
    infof(data, "User: %s", conn->user);
  }
  if(conn->passwd) {
    infof(data, "Password: %s", conn->passwd);
  }
  sock = conn->sock[FIRSTSOCKET];
#endif /* CURL_LIBSSH2_DEBUG */

  sshc->ssh_session = libssh2_session_init_ex(my_libssh2_malloc,
                                              my_libssh2_free,
                                              my_libssh2_realloc, data);
  if(!sshc->ssh_session) {
    failf(data, "Failure initialising ssh session");
    return CURLE_FAILED_INIT;
  }

#ifndef CURL_DISABLE_PROXY
  if(conn->http_proxy.proxytype == CURLPROXY_HTTPS) {
    /*
     * This crazy union dance is here to avoid assigning a void pointer a
     * function pointer as it is invalid C. The problem is of course that
     * libssh2 has such an API...
     */
    union receive {
      void *recvp;
      ssize_t (*recvptr)(libssh2_socket_t, void *, size_t, int, void **);
    };
    union transfer {
      void *sendp;
      ssize_t (*sendptr)(libssh2_socket_t, const void *, size_t, int, void **);
    };
    union receive sshrecv;
    union transfer sshsend;

    sshrecv.recvptr = ssh_tls_recv;
    sshsend.sendptr = ssh_tls_send;

    infof(data, "Uses HTTPS proxy!");
    /*
      Setup libssh2 callbacks to make it read/write TLS from the socket.

      ssize_t
      recvcb(libssh2_socket_t sock, void *buffer, size_t length,
      int flags, void **abstract);

      ssize_t
      sendcb(libssh2_socket_t sock, const void *buffer, size_t length,
      int flags, void **abstract);

    */
    libssh2_session_callback_set(sshc->ssh_session,
                                 LIBSSH2_CALLBACK_RECV, sshrecv.recvp);
    libssh2_session_callback_set(sshc->ssh_session,
                                 LIBSSH2_CALLBACK_SEND, sshsend.sendp);

    /* Store the underlying TLS recv/send function pointers to be used when
       reading from the proxy */
    sshc->tls_recv = conn->recv[FIRSTSOCKET];
    sshc->tls_send = conn->send[FIRSTSOCKET];
  }

#endif /* CURL_DISABLE_PROXY */
  if(conn->handler->protocol & CURLPROTO_SCP) {
    conn->recv[FIRSTSOCKET] = scp_recv;
    conn->send[FIRSTSOCKET] = scp_send;
  }
  else {
    conn->recv[FIRSTSOCKET] = sftp_recv;
    conn->send[FIRSTSOCKET] = sftp_send;
  }

  if(data->set.ssh_compression) {
#if LIBSSH2_VERSION_NUM >= 0x010208
    if(libssh2_session_flag(sshc->ssh_session, LIBSSH2_FLAG_COMPRESS, 1) < 0)
#endif
      infof(data, "Failed to enable compression for ssh session");
  }

#ifdef HAVE_LIBSSH2_KNOWNHOST_API
  if(data->set.str[STRING_SSH_KNOWNHOSTS]) {
    int rc;
    sshc->kh = libssh2_knownhost_init(sshc->ssh_session);
    if(!sshc->kh) {
      libssh2_session_free(sshc->ssh_session);
      sshc->ssh_session = NULL;
      return CURLE_FAILED_INIT;
    }

    /* read all known hosts from there */
    rc = libssh2_knownhost_readfile(sshc->kh,
                                    data->set.str[STRING_SSH_KNOWNHOSTS],
                                    LIBSSH2_KNOWNHOST_FILE_OPENSSH);
    if(rc < 0)
      infof(data, "Failed to read known hosts from %s",
            data->set.str[STRING_SSH_KNOWNHOSTS]);
  }
#endif /* HAVE_LIBSSH2_KNOWNHOST_API */

#ifdef CURL_LIBSSH2_DEBUG
  libssh2_trace(sshc->ssh_session, ~0);
  infof(data, "SSH socket: %d", (int)sock);
#endif /* CURL_LIBSSH2_DEBUG */

  state(data, SSH_INIT);

  result = ssh_multi_statemach(data, done);

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
                     bool *connected,
                     bool *dophase_done)
{
  CURLcode result = CURLE_OK;
  struct connectdata *conn = data->conn;

  DEBUGF(infof(data, "DO phase starts"));

  *dophase_done = FALSE; /* not done yet */

  /* start the first command in the DO phase */
  state(data, SSH_SCP_TRANS_INIT);

  /* run the state-machine */
  result = ssh_multi_statemach(data, dophase_done);

  *connected = conn->bits.tcpconnect[FIRSTSOCKET];

  if(*dophase_done) {
    DEBUGF(infof(data, "DO phase is complete"));
  }

  return result;
}

/* called from multi.c while DOing */
static CURLcode scp_doing(struct Curl_easy *data,
                          bool *dophase_done)
{
  CURLcode result;
  result = ssh_multi_statemach(data, dophase_done);

  if(*dophase_done) {
    DEBUGF(infof(data, "DO phase is complete"));
  }
  return result;
}

/*
 * The DO function is generic for both protocols. There was previously two
 * separate ones but this way means less duplicated code.
 */

static CURLcode ssh_do(struct Curl_easy *data, bool *done)
{
  CURLcode result;
  bool connected = 0;
  struct connectdata *conn = data->conn;
  struct ssh_conn *sshc = &conn->proto.sshc;

  *done = FALSE; /* default to false */

  data->req.size = -1; /* make sure this is unknown at this point */

  sshc->actualcode = CURLE_OK; /* reset error code */
  sshc->secondCreateDirs = 0;   /* reset the create dir attempt state
                                   variable */

  Curl_pgrsSetUploadCounter(data, 0);
  Curl_pgrsSetDownloadCounter(data, 0);
  Curl_pgrsSetUploadSize(data, -1);
  Curl_pgrsSetDownloadSize(data, -1);

  if(conn->handler->protocol & CURLPROTO_SCP)
    result = scp_perform(data, &connected,  done);
  else
    result = sftp_perform(data, &connected,  done);

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
  struct ssh_conn *sshc = &conn->proto.sshc;
  (void) dead_connection;

  if(sshc->ssh_session) {
    /* only if there's a session still around to use! */
    state(data, SSH_SESSION_DISCONNECT);
    result = ssh_block_statemach(data, conn, TRUE);
  }

  return result;
}

/* generic done function for both SCP and SFTP called from their specific
   done functions */
static CURLcode ssh_done(struct Curl_easy *data, CURLcode status)
{
  CURLcode result = CURLE_OK;
  struct SSHPROTO *sshp = data->req.p.ssh;
  struct connectdata *conn = data->conn;

  if(!status)
    /* run the state-machine */
    result = ssh_block_statemach(data, conn, FALSE);
  else
    result = status;

  Curl_safefree(sshp->path);
  Curl_safefree(sshp->readdir_filename);
  Curl_safefree(sshp->readdir_longentry);
  Curl_dyn_free(&sshp->readdir);

  if(Curl_pgrsDone(data))
    return CURLE_ABORTED_BY_CALLBACK;

  data->req.keepon = 0; /* clear all bits */
  return result;
}


static CURLcode scp_done(struct Curl_easy *data, CURLcode status,
                         bool premature)
{
  (void)premature; /* not used */

  if(!status)
    state(data, SSH_SCP_DONE);

  return ssh_done(data, status);

}

static ssize_t scp_send(struct Curl_easy *data, int sockindex,
                        const void *mem, size_t len, CURLcode *err)
{
  ssize_t nwrite;
  struct connectdata *conn = data->conn;
  struct ssh_conn *sshc = &conn->proto.sshc;
  (void)sockindex; /* we only support SCP on the fixed known primary socket */

  /* libssh2_channel_write() returns int! */
  nwrite = (ssize_t) libssh2_channel_write(sshc->ssh_channel, mem, len);

  ssh_block2waitfor(data, (nwrite == LIBSSH2_ERROR_EAGAIN)?TRUE:FALSE);

  if(nwrite == LIBSSH2_ERROR_EAGAIN) {
    *err = CURLE_AGAIN;
    nwrite = 0;
  }
  else if(nwrite < LIBSSH2_ERROR_NONE) {
    *err = libssh2_session_error_to_CURLE((int)nwrite);
    nwrite = -1;
  }

  return nwrite;
}

static ssize_t scp_recv(struct Curl_easy *data, int sockindex,
                        char *mem, size_t len, CURLcode *err)
{
  ssize_t nread;
  struct connectdata *conn = data->conn;
  struct ssh_conn *sshc = &conn->proto.sshc;
  (void)sockindex; /* we only support SCP on the fixed known primary socket */

  /* libssh2_channel_read() returns int */
  nread = (ssize_t) libssh2_channel_read(sshc->ssh_channel, mem, len);

  ssh_block2waitfor(data, (nread == LIBSSH2_ERROR_EAGAIN)?TRUE:FALSE);
  if(nread == LIBSSH2_ERROR_EAGAIN) {
    *err = CURLE_AGAIN;
    nread = -1;
  }

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

  DEBUGF(infof(data, "DO phase starts"));

  *dophase_done = FALSE; /* not done yet */

  /* start the first command in the DO phase */
  state(data, SSH_SFTP_QUOTE_INIT);

  /* run the state-machine */
  result = ssh_multi_statemach(data, dophase_done);

  *connected = data->conn->bits.tcpconnect[FIRSTSOCKET];

  if(*dophase_done) {
    DEBUGF(infof(data, "DO phase is complete"));
  }

  return result;
}

/* called from multi.c while DOing */
static CURLcode sftp_doing(struct Curl_easy *data,
                           bool *dophase_done)
{
  CURLcode result = ssh_multi_statemach(data, dophase_done);

  if(*dophase_done) {
    DEBUGF(infof(data, "DO phase is complete"));
  }
  return result;
}

/* BLOCKING, but the function is using the state machine so the only reason
   this is still blocking is that the multi interface code has no support for
   disconnecting operations that takes a while */
static CURLcode sftp_disconnect(struct Curl_easy *data,
                                struct connectdata *conn, bool dead_connection)
{
  CURLcode result = CURLE_OK;
  struct ssh_conn *sshc = &conn->proto.sshc;
  (void) dead_connection;

  DEBUGF(infof(data, "SSH DISCONNECT starts now"));

  if(sshc->ssh_session) {
    /* only if there's a session still around to use! */
    state(data, SSH_SFTP_SHUTDOWN);
    result = ssh_block_statemach(data, conn, TRUE);
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
  return ssh_done(data, status);
}

/* return number of sent bytes */
static ssize_t sftp_send(struct Curl_easy *data, int sockindex,
                         const void *mem, size_t len, CURLcode *err)
{
  ssize_t nwrite;
  struct connectdata *conn = data->conn;
  struct ssh_conn *sshc = &conn->proto.sshc;
  (void)sockindex;

  nwrite = libssh2_sftp_write(sshc->sftp_handle, mem, len);

  ssh_block2waitfor(data, (nwrite == LIBSSH2_ERROR_EAGAIN)?TRUE:FALSE);

  if(nwrite == LIBSSH2_ERROR_EAGAIN) {
    *err = CURLE_AGAIN;
    nwrite = 0;
  }
  else if(nwrite < LIBSSH2_ERROR_NONE) {
    *err = libssh2_session_error_to_CURLE((int)nwrite);
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
  struct ssh_conn *sshc = &conn->proto.sshc;
  (void)sockindex;

  nread = libssh2_sftp_read(sshc->sftp_handle, mem, len);

  ssh_block2waitfor(data, (nread == LIBSSH2_ERROR_EAGAIN)?TRUE:FALSE);

  if(nread == LIBSSH2_ERROR_EAGAIN) {
    *err = CURLE_AGAIN;
    nread = -1;

  }
  else if(nread < 0) {
    *err = libssh2_session_error_to_CURLE((int)nread);
  }
  return nread;
}

static const char *sftp_libssh2_strerror(unsigned long err)
{
  switch(err) {
    case LIBSSH2_FX_NO_SUCH_FILE:
      return "No such file or directory";

    case LIBSSH2_FX_PERMISSION_DENIED:
      return "Permission denied";

    case LIBSSH2_FX_FAILURE:
      return "Operation failed";

    case LIBSSH2_FX_BAD_MESSAGE:
      return "Bad message from SFTP server";

    case LIBSSH2_FX_NO_CONNECTION:
      return "Not connected to SFTP server";

    case LIBSSH2_FX_CONNECTION_LOST:
      return "Connection to SFTP server lost";

    case LIBSSH2_FX_OP_UNSUPPORTED:
      return "Operation not supported by SFTP server";

    case LIBSSH2_FX_INVALID_HANDLE:
      return "Invalid handle";

    case LIBSSH2_FX_NO_SUCH_PATH:
      return "No such file or directory";

    case LIBSSH2_FX_FILE_ALREADY_EXISTS:
      return "File already exists";

    case LIBSSH2_FX_WRITE_PROTECT:
      return "File is write protected";

    case LIBSSH2_FX_NO_MEDIA:
      return "No media";

    case LIBSSH2_FX_NO_SPACE_ON_FILESYSTEM:
      return "Disk full";

    case LIBSSH2_FX_QUOTA_EXCEEDED:
      return "User quota exceeded";

    case LIBSSH2_FX_UNKNOWN_PRINCIPLE:
      return "Unknown principle";

    case LIBSSH2_FX_LOCK_CONFlICT:
      return "File lock conflict";

    case LIBSSH2_FX_DIR_NOT_EMPTY:
      return "Directory not empty";

    case LIBSSH2_FX_NOT_A_DIRECTORY:
      return "Not a directory";

    case LIBSSH2_FX_INVALID_FILENAME:
      return "Invalid filename";

    case LIBSSH2_FX_LINK_LOOP:
      return "Link points to itself";
  }
  return "Unknown error in libssh2";
}

CURLcode Curl_ssh_init(void)
{
#ifdef HAVE_LIBSSH2_INIT
  if(libssh2_init(0)) {
    DEBUGF(fprintf(stderr, "Error: libssh2_init failed\n"));
    return CURLE_FAILED_INIT;
  }
#endif
  return CURLE_OK;
}

void Curl_ssh_cleanup(void)
{
#ifdef HAVE_LIBSSH2_EXIT
  (void)libssh2_exit();
#endif
}

void Curl_ssh_version(char *buffer, size_t buflen)
{
  (void)msnprintf(buffer, buflen, "libssh2/%s", CURL_LIBSSH2_VERSION);
}

/* The SSH session is associated with the *CONNECTION* but the callback user
 * pointer is an easy handle pointer. This function allows us to reassign the
 * user pointer to the *CURRENT* (new) easy handle.
 */
static void ssh_attach(struct Curl_easy *data, struct connectdata *conn)
{
  DEBUGASSERT(data);
  DEBUGASSERT(conn);
  if(conn->handler->protocol & PROTO_FAMILY_SSH) {
    struct ssh_conn *sshc = &conn->proto.sshc;
    if(sshc->ssh_session) {
      /* only re-attach if the session already exists */
      void **abstract = libssh2_session_abstract(sshc->ssh_session);
      *abstract = data;
    }
  }
}
#endif /* USE_LIBSSH2 */
