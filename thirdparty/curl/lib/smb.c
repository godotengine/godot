/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 2016 - 2021, Daniel Stenberg, <daniel@haxx.se>, et al.
 * Copyright (C) 2014, Bill Nagel <wnagel@tycoint.com>, Exacq Technologies
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

#if !defined(CURL_DISABLE_SMB) && defined(USE_CURL_NTLM_CORE) &&  \
  (SIZEOF_CURL_OFF_T > 4)

#define BUILDING_CURL_SMB_C

#ifdef HAVE_PROCESS_H
#include <process.h>
#ifdef CURL_WINDOWS_APP
#define getpid GetCurrentProcessId
#elif !defined(MSDOS)
#define getpid _getpid
#endif
#endif

#include "smb.h"
#include "urldata.h"
#include "sendf.h"
#include "multiif.h"
#include "connect.h"
#include "progress.h"
#include "transfer.h"
#include "vtls/vtls.h"
#include "curl_ntlm_core.h"
#include "escape.h"
#include "curl_endian.h"

/* The last #include files should be: */
#include "curl_memory.h"
#include "memdebug.h"

/* Local API functions */
static CURLcode smb_setup_connection(struct Curl_easy *data,
                                     struct connectdata *conn);
static CURLcode smb_connect(struct Curl_easy *data, bool *done);
static CURLcode smb_connection_state(struct Curl_easy *data, bool *done);
static CURLcode smb_do(struct Curl_easy *data, bool *done);
static CURLcode smb_request_state(struct Curl_easy *data, bool *done);
static CURLcode smb_done(struct Curl_easy *data, CURLcode status,
                         bool premature);
static CURLcode smb_disconnect(struct Curl_easy *data,
                               struct connectdata *conn, bool dead);
static int smb_getsock(struct Curl_easy *data, struct connectdata *conn,
                       curl_socket_t *socks);
static CURLcode smb_parse_url_path(struct Curl_easy *data,
                                   struct connectdata *conn);

/*
 * SMB handler interface
 */
const struct Curl_handler Curl_handler_smb = {
  "SMB",                                /* scheme */
  smb_setup_connection,                 /* setup_connection */
  smb_do,                               /* do_it */
  smb_done,                             /* done */
  ZERO_NULL,                            /* do_more */
  smb_connect,                          /* connect_it */
  smb_connection_state,                 /* connecting */
  smb_request_state,                    /* doing */
  smb_getsock,                          /* proto_getsock */
  smb_getsock,                          /* doing_getsock */
  ZERO_NULL,                            /* domore_getsock */
  ZERO_NULL,                            /* perform_getsock */
  smb_disconnect,                       /* disconnect */
  ZERO_NULL,                            /* readwrite */
  ZERO_NULL,                            /* connection_check */
  ZERO_NULL,                            /* attach connection */
  PORT_SMB,                             /* defport */
  CURLPROTO_SMB,                        /* protocol */
  CURLPROTO_SMB,                        /* family */
  PROTOPT_NONE                          /* flags */
};

#ifdef USE_SSL
/*
 * SMBS handler interface
 */
const struct Curl_handler Curl_handler_smbs = {
  "SMBS",                               /* scheme */
  smb_setup_connection,                 /* setup_connection */
  smb_do,                               /* do_it */
  smb_done,                             /* done */
  ZERO_NULL,                            /* do_more */
  smb_connect,                          /* connect_it */
  smb_connection_state,                 /* connecting */
  smb_request_state,                    /* doing */
  smb_getsock,                          /* proto_getsock */
  smb_getsock,                          /* doing_getsock */
  ZERO_NULL,                            /* domore_getsock */
  ZERO_NULL,                            /* perform_getsock */
  smb_disconnect,                       /* disconnect */
  ZERO_NULL,                            /* readwrite */
  ZERO_NULL,                            /* connection_check */
  ZERO_NULL,                            /* attach connection */
  PORT_SMBS,                            /* defport */
  CURLPROTO_SMBS,                       /* protocol */
  CURLPROTO_SMB,                        /* family */
  PROTOPT_SSL                           /* flags */
};
#endif

#define MAX_PAYLOAD_SIZE  0x8000
#define MAX_MESSAGE_SIZE  (MAX_PAYLOAD_SIZE + 0x1000)
#define CLIENTNAME        "curl"
#define SERVICENAME       "?????"

/* Append a string to an SMB message */
#define MSGCAT(str)                             \
  do {                                          \
    strcpy(p, (str));                           \
    p += strlen(str);                           \
  } while(0)

/* Append a null-terminated string to an SMB message */
#define MSGCATNULL(str)                         \
  do {                                          \
    strcpy(p, (str));                           \
    p += strlen(str) + 1;                       \
  } while(0)

/* SMB is mostly little endian */
#if (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__) || \
  defined(__OS400__)
static unsigned short smb_swap16(unsigned short x)
{
  return (unsigned short) ((x << 8) | ((x >> 8) & 0xff));
}

static unsigned int smb_swap32(unsigned int x)
{
  return (x << 24) | ((x << 8) & 0xff0000) | ((x >> 8) & 0xff00) |
    ((x >> 24) & 0xff);
}

static curl_off_t smb_swap64(curl_off_t x)
{
  return ((curl_off_t) smb_swap32((unsigned int) x) << 32) |
    smb_swap32((unsigned int) (x >> 32));
}

#else
#  define smb_swap16(x) (x)
#  define smb_swap32(x) (x)
#  define smb_swap64(x) (x)
#endif

/* SMB request state */
enum smb_req_state {
  SMB_REQUESTING,
  SMB_TREE_CONNECT,
  SMB_OPEN,
  SMB_DOWNLOAD,
  SMB_UPLOAD,
  SMB_CLOSE,
  SMB_TREE_DISCONNECT,
  SMB_DONE
};

/* SMB request data */
struct smb_request {
  enum smb_req_state state;
  char *path;
  unsigned short tid; /* Even if we connect to the same tree as another */
  unsigned short fid; /* request, the tid will be different */
  CURLcode result;
};

static void conn_state(struct Curl_easy *data, enum smb_conn_state newstate)
{
  struct smb_conn *smbc = &data->conn->proto.smbc;
#if defined(DEBUGBUILD) && !defined(CURL_DISABLE_VERBOSE_STRINGS)
  /* For debug purposes */
  static const char * const names[] = {
    "SMB_NOT_CONNECTED",
    "SMB_CONNECTING",
    "SMB_NEGOTIATE",
    "SMB_SETUP",
    "SMB_CONNECTED",
    /* LAST */
  };

  if(smbc->state != newstate)
    infof(data, "SMB conn %p state change from %s to %s",
          (void *)smbc, names[smbc->state], names[newstate]);
#endif

  smbc->state = newstate;
}

static void request_state(struct Curl_easy *data,
                          enum smb_req_state newstate)
{
  struct smb_request *req = data->req.p.smb;
#if defined(DEBUGBUILD) && !defined(CURL_DISABLE_VERBOSE_STRINGS)
  /* For debug purposes */
  static const char * const names[] = {
    "SMB_REQUESTING",
    "SMB_TREE_CONNECT",
    "SMB_OPEN",
    "SMB_DOWNLOAD",
    "SMB_UPLOAD",
    "SMB_CLOSE",
    "SMB_TREE_DISCONNECT",
    "SMB_DONE",
    /* LAST */
  };

  if(req->state != newstate)
    infof(data, "SMB request %p state change from %s to %s",
          (void *)req, names[req->state], names[newstate]);
#endif

  req->state = newstate;
}

/* this should setup things in the connection, not in the easy
   handle */
static CURLcode smb_setup_connection(struct Curl_easy *data,
                                     struct connectdata *conn)
{
  struct smb_request *req;

  /* Initialize the request state */
  data->req.p.smb = req = calloc(1, sizeof(struct smb_request));
  if(!req)
    return CURLE_OUT_OF_MEMORY;

  /* Parse the URL path */
  return smb_parse_url_path(data, conn);
}

static CURLcode smb_connect(struct Curl_easy *data, bool *done)
{
  struct connectdata *conn = data->conn;
  struct smb_conn *smbc = &conn->proto.smbc;
  char *slash;

  (void) done;

  /* Check we have a username and password to authenticate with */
  if(!conn->bits.user_passwd)
    return CURLE_LOGIN_DENIED;

  /* Initialize the connection state */
  smbc->state = SMB_CONNECTING;
  smbc->recv_buf = malloc(MAX_MESSAGE_SIZE);
  if(!smbc->recv_buf)
    return CURLE_OUT_OF_MEMORY;

  /* Multiple requests are allowed with this connection */
  connkeep(conn, "SMB default");

  /* Parse the username, domain, and password */
  slash = strchr(conn->user, '/');
  if(!slash)
    slash = strchr(conn->user, '\\');

  if(slash) {
    smbc->user = slash + 1;
    smbc->domain = strdup(conn->user);
    if(!smbc->domain)
      return CURLE_OUT_OF_MEMORY;
    smbc->domain[slash - conn->user] = 0;
  }
  else {
    smbc->user = conn->user;
    smbc->domain = strdup(conn->host.name);
    if(!smbc->domain)
      return CURLE_OUT_OF_MEMORY;
  }

  return CURLE_OK;
}

static CURLcode smb_recv_message(struct Curl_easy *data, void **msg)
{
  struct connectdata *conn = data->conn;
  struct smb_conn *smbc = &conn->proto.smbc;
  char *buf = smbc->recv_buf;
  ssize_t bytes_read;
  size_t nbt_size;
  size_t msg_size;
  size_t len = MAX_MESSAGE_SIZE - smbc->got;
  CURLcode result;

  result = Curl_read(data, FIRSTSOCKET, buf + smbc->got, len, &bytes_read);
  if(result)
    return result;

  if(!bytes_read)
    return CURLE_OK;

  smbc->got += bytes_read;

  /* Check for a 32-bit nbt header */
  if(smbc->got < sizeof(unsigned int))
    return CURLE_OK;

  nbt_size = Curl_read16_be((const unsigned char *)
                            (buf + sizeof(unsigned short))) +
    sizeof(unsigned int);
  if(smbc->got < nbt_size)
    return CURLE_OK;

  msg_size = sizeof(struct smb_header);
  if(nbt_size >= msg_size + 1) {
    /* Add the word count */
    msg_size += 1 + ((unsigned char) buf[msg_size]) * sizeof(unsigned short);
    if(nbt_size >= msg_size + sizeof(unsigned short)) {
      /* Add the byte count */
      msg_size += sizeof(unsigned short) +
        Curl_read16_le((const unsigned char *)&buf[msg_size]);
      if(nbt_size < msg_size)
        return CURLE_READ_ERROR;
    }
  }

  *msg = buf;

  return CURLE_OK;
}

static void smb_pop_message(struct connectdata *conn)
{
  struct smb_conn *smbc = &conn->proto.smbc;

  smbc->got = 0;
}

static void smb_format_message(struct Curl_easy *data, struct smb_header *h,
                               unsigned char cmd, size_t len)
{
  struct connectdata *conn = data->conn;
  struct smb_conn *smbc = &conn->proto.smbc;
  struct smb_request *req = data->req.p.smb;
  unsigned int pid;

  memset(h, 0, sizeof(*h));
  h->nbt_length = htons((unsigned short) (sizeof(*h) - sizeof(unsigned int) +
                                          len));
  memcpy((char *)h->magic, "\xffSMB", 4);
  h->command = cmd;
  h->flags = SMB_FLAGS_CANONICAL_PATHNAMES | SMB_FLAGS_CASELESS_PATHNAMES;
  h->flags2 = smb_swap16(SMB_FLAGS2_IS_LONG_NAME | SMB_FLAGS2_KNOWS_LONG_NAME);
  h->uid = smb_swap16(smbc->uid);
  h->tid = smb_swap16(req->tid);
  pid = getpid();
  h->pid_high = smb_swap16((unsigned short)(pid >> 16));
  h->pid = smb_swap16((unsigned short) pid);
}

static CURLcode smb_send(struct Curl_easy *data, ssize_t len,
                         size_t upload_size)
{
  struct connectdata *conn = data->conn;
  struct smb_conn *smbc = &conn->proto.smbc;
  ssize_t bytes_written;
  CURLcode result;

  result = Curl_write(data, FIRSTSOCKET, data->state.ulbuf,
                      len, &bytes_written);
  if(result)
    return result;

  if(bytes_written != len) {
    smbc->send_size = len;
    smbc->sent = bytes_written;
  }

  smbc->upload_size = upload_size;

  return CURLE_OK;
}

static CURLcode smb_flush(struct Curl_easy *data)
{
  struct connectdata *conn = data->conn;
  struct smb_conn *smbc = &conn->proto.smbc;
  ssize_t bytes_written;
  ssize_t len = smbc->send_size - smbc->sent;
  CURLcode result;

  if(!smbc->send_size)
    return CURLE_OK;

  result = Curl_write(data, FIRSTSOCKET,
                      data->state.ulbuf + smbc->sent,
                      len, &bytes_written);
  if(result)
    return result;

  if(bytes_written != len)
    smbc->sent += bytes_written;
  else
    smbc->send_size = 0;

  return CURLE_OK;
}

static CURLcode smb_send_message(struct Curl_easy *data, unsigned char cmd,
                                 const void *msg, size_t msg_len)
{
  CURLcode result = Curl_get_upload_buffer(data);
  if(result)
    return result;
  smb_format_message(data, (struct smb_header *)data->state.ulbuf,
                     cmd, msg_len);
  memcpy(data->state.ulbuf + sizeof(struct smb_header),
         msg, msg_len);

  return smb_send(data, sizeof(struct smb_header) + msg_len, 0);
}

static CURLcode smb_send_negotiate(struct Curl_easy *data)
{
  const char *msg = "\x00\x0c\x00\x02NT LM 0.12";

  return smb_send_message(data, SMB_COM_NEGOTIATE, msg, 15);
}

static CURLcode smb_send_setup(struct Curl_easy *data)
{
  struct connectdata *conn = data->conn;
  struct smb_conn *smbc = &conn->proto.smbc;
  struct smb_setup msg;
  char *p = msg.bytes;
  unsigned char lm_hash[21];
  unsigned char lm[24];
  unsigned char nt_hash[21];
  unsigned char nt[24];

  size_t byte_count = sizeof(lm) + sizeof(nt);
  byte_count += strlen(smbc->user) + strlen(smbc->domain);
  byte_count += strlen(OS) + strlen(CLIENTNAME) + 4; /* 4 null chars */
  if(byte_count > sizeof(msg.bytes))
    return CURLE_FILESIZE_EXCEEDED;

  Curl_ntlm_core_mk_lm_hash(data, conn->passwd, lm_hash);
  Curl_ntlm_core_lm_resp(lm_hash, smbc->challenge, lm);
#ifdef USE_NTRESPONSES
  Curl_ntlm_core_mk_nt_hash(data, conn->passwd, nt_hash);
  Curl_ntlm_core_lm_resp(nt_hash, smbc->challenge, nt);
#else
  memset(nt, 0, sizeof(nt));
#endif

  memset(&msg, 0, sizeof(msg));
  msg.word_count = SMB_WC_SETUP_ANDX;
  msg.andx.command = SMB_COM_NO_ANDX_COMMAND;
  msg.max_buffer_size = smb_swap16(MAX_MESSAGE_SIZE);
  msg.max_mpx_count = smb_swap16(1);
  msg.vc_number = smb_swap16(1);
  msg.session_key = smb_swap32(smbc->session_key);
  msg.capabilities = smb_swap32(SMB_CAP_LARGE_FILES);
  msg.lengths[0] = smb_swap16(sizeof(lm));
  msg.lengths[1] = smb_swap16(sizeof(nt));
  memcpy(p, lm, sizeof(lm));
  p += sizeof(lm);
  memcpy(p, nt, sizeof(nt));
  p += sizeof(nt);
  MSGCATNULL(smbc->user);
  MSGCATNULL(smbc->domain);
  MSGCATNULL(OS);
  MSGCATNULL(CLIENTNAME);
  byte_count = p - msg.bytes;
  msg.byte_count = smb_swap16((unsigned short)byte_count);

  return smb_send_message(data, SMB_COM_SETUP_ANDX, &msg,
                          sizeof(msg) - sizeof(msg.bytes) + byte_count);
}

static CURLcode smb_send_tree_connect(struct Curl_easy *data)
{
  struct smb_tree_connect msg;
  struct connectdata *conn = data->conn;
  struct smb_conn *smbc = &conn->proto.smbc;
  char *p = msg.bytes;

  size_t byte_count = strlen(conn->host.name) + strlen(smbc->share);
  byte_count += strlen(SERVICENAME) + 5; /* 2 nulls and 3 backslashes */
  if(byte_count > sizeof(msg.bytes))
    return CURLE_FILESIZE_EXCEEDED;

  memset(&msg, 0, sizeof(msg));
  msg.word_count = SMB_WC_TREE_CONNECT_ANDX;
  msg.andx.command = SMB_COM_NO_ANDX_COMMAND;
  msg.pw_len = 0;
  MSGCAT("\\\\");
  MSGCAT(conn->host.name);
  MSGCAT("\\");
  MSGCATNULL(smbc->share);
  MSGCATNULL(SERVICENAME); /* Match any type of service */
  byte_count = p - msg.bytes;
  msg.byte_count = smb_swap16((unsigned short)byte_count);

  return smb_send_message(data, SMB_COM_TREE_CONNECT_ANDX, &msg,
                          sizeof(msg) - sizeof(msg.bytes) + byte_count);
}

static CURLcode smb_send_open(struct Curl_easy *data)
{
  struct smb_request *req = data->req.p.smb;
  struct smb_nt_create msg;
  size_t byte_count;

  if((strlen(req->path) + 1) > sizeof(msg.bytes))
    return CURLE_FILESIZE_EXCEEDED;

  memset(&msg, 0, sizeof(msg));
  msg.word_count = SMB_WC_NT_CREATE_ANDX;
  msg.andx.command = SMB_COM_NO_ANDX_COMMAND;
  byte_count = strlen(req->path);
  msg.name_length = smb_swap16((unsigned short)byte_count);
  msg.share_access = smb_swap32(SMB_FILE_SHARE_ALL);
  if(data->set.upload) {
    msg.access = smb_swap32(SMB_GENERIC_READ | SMB_GENERIC_WRITE);
    msg.create_disposition = smb_swap32(SMB_FILE_OVERWRITE_IF);
  }
  else {
    msg.access = smb_swap32(SMB_GENERIC_READ);
    msg.create_disposition = smb_swap32(SMB_FILE_OPEN);
  }
  msg.byte_count = smb_swap16((unsigned short) ++byte_count);
  strcpy(msg.bytes, req->path);

  return smb_send_message(data, SMB_COM_NT_CREATE_ANDX, &msg,
                          sizeof(msg) - sizeof(msg.bytes) + byte_count);
}

static CURLcode smb_send_close(struct Curl_easy *data)
{
  struct smb_request *req = data->req.p.smb;
  struct smb_close msg;

  memset(&msg, 0, sizeof(msg));
  msg.word_count = SMB_WC_CLOSE;
  msg.fid = smb_swap16(req->fid);

  return smb_send_message(data, SMB_COM_CLOSE, &msg, sizeof(msg));
}

static CURLcode smb_send_tree_disconnect(struct Curl_easy *data)
{
  struct smb_tree_disconnect msg;

  memset(&msg, 0, sizeof(msg));

  return smb_send_message(data, SMB_COM_TREE_DISCONNECT, &msg, sizeof(msg));
}

static CURLcode smb_send_read(struct Curl_easy *data)
{
  struct smb_request *req = data->req.p.smb;
  curl_off_t offset = data->req.offset;
  struct smb_read msg;

  memset(&msg, 0, sizeof(msg));
  msg.word_count = SMB_WC_READ_ANDX;
  msg.andx.command = SMB_COM_NO_ANDX_COMMAND;
  msg.fid = smb_swap16(req->fid);
  msg.offset = smb_swap32((unsigned int) offset);
  msg.offset_high = smb_swap32((unsigned int) (offset >> 32));
  msg.min_bytes = smb_swap16(MAX_PAYLOAD_SIZE);
  msg.max_bytes = smb_swap16(MAX_PAYLOAD_SIZE);

  return smb_send_message(data, SMB_COM_READ_ANDX, &msg, sizeof(msg));
}

static CURLcode smb_send_write(struct Curl_easy *data)
{
  struct smb_write *msg;
  struct smb_request *req = data->req.p.smb;
  curl_off_t offset = data->req.offset;
  curl_off_t upload_size = data->req.size - data->req.bytecount;
  CURLcode result = Curl_get_upload_buffer(data);
  if(result)
    return result;
  msg = (struct smb_write *)data->state.ulbuf;

  if(upload_size >= MAX_PAYLOAD_SIZE - 1) /* There is one byte of padding */
    upload_size = MAX_PAYLOAD_SIZE - 1;

  memset(msg, 0, sizeof(*msg));
  msg->word_count = SMB_WC_WRITE_ANDX;
  msg->andx.command = SMB_COM_NO_ANDX_COMMAND;
  msg->fid = smb_swap16(req->fid);
  msg->offset = smb_swap32((unsigned int) offset);
  msg->offset_high = smb_swap32((unsigned int) (offset >> 32));
  msg->data_length = smb_swap16((unsigned short) upload_size);
  msg->data_offset = smb_swap16(sizeof(*msg) - sizeof(unsigned int));
  msg->byte_count = smb_swap16((unsigned short) (upload_size + 1));

  smb_format_message(data, &msg->h, SMB_COM_WRITE_ANDX,
                     sizeof(*msg) - sizeof(msg->h) + (size_t) upload_size);

  return smb_send(data, sizeof(*msg), (size_t) upload_size);
}

static CURLcode smb_send_and_recv(struct Curl_easy *data, void **msg)
{
  struct connectdata *conn = data->conn;
  struct smb_conn *smbc = &conn->proto.smbc;
  CURLcode result;
  *msg = NULL; /* if it returns early */

  /* Check if there is data in the transfer buffer */
  if(!smbc->send_size && smbc->upload_size) {
    size_t nread = smbc->upload_size > (size_t)data->set.upload_buffer_size ?
      (size_t)data->set.upload_buffer_size : smbc->upload_size;
    data->req.upload_fromhere = data->state.ulbuf;
    result = Curl_fillreadbuffer(data, nread, &nread);
    if(result && result != CURLE_AGAIN)
      return result;
    if(!nread)
      return CURLE_OK;

    smbc->upload_size -= nread;
    smbc->send_size = nread;
    smbc->sent = 0;
  }

  /* Check if there is data to send */
  if(smbc->send_size) {
    result = smb_flush(data);
    if(result)
      return result;
  }

  /* Check if there is still data to be sent */
  if(smbc->send_size || smbc->upload_size)
    return CURLE_AGAIN;

  return smb_recv_message(data, msg);
}

static CURLcode smb_connection_state(struct Curl_easy *data, bool *done)
{
  struct connectdata *conn = data->conn;
  struct smb_conn *smbc = &conn->proto.smbc;
  struct smb_negotiate_response *nrsp;
  struct smb_header *h;
  CURLcode result;
  void *msg = NULL;

  if(smbc->state == SMB_CONNECTING) {
#ifdef USE_SSL
    if((conn->handler->flags & PROTOPT_SSL)) {
      bool ssl_done = FALSE;
      result = Curl_ssl_connect_nonblocking(data, conn, FALSE,
                                            FIRSTSOCKET, &ssl_done);
      if(result && result != CURLE_AGAIN)
        return result;
      if(!ssl_done)
        return CURLE_OK;
    }
#endif

    result = smb_send_negotiate(data);
    if(result) {
      connclose(conn, "SMB: failed to send negotiate message");
      return result;
    }

    conn_state(data, SMB_NEGOTIATE);
  }

  /* Send the previous message and check for a response */
  result = smb_send_and_recv(data, &msg);
  if(result && result != CURLE_AGAIN) {
    connclose(conn, "SMB: failed to communicate");
    return result;
  }

  if(!msg)
    return CURLE_OK;

  h = msg;

  switch(smbc->state) {
  case SMB_NEGOTIATE:
    if((smbc->got < sizeof(*nrsp) + sizeof(smbc->challenge) - 1) ||
       h->status) {
      connclose(conn, "SMB: negotiation failed");
      return CURLE_COULDNT_CONNECT;
    }
    nrsp = msg;
    memcpy(smbc->challenge, nrsp->bytes, sizeof(smbc->challenge));
    smbc->session_key = smb_swap32(nrsp->session_key);
    result = smb_send_setup(data);
    if(result) {
      connclose(conn, "SMB: failed to send setup message");
      return result;
    }
    conn_state(data, SMB_SETUP);
    break;

  case SMB_SETUP:
    if(h->status) {
      connclose(conn, "SMB: authentication failed");
      return CURLE_LOGIN_DENIED;
    }
    smbc->uid = smb_swap16(h->uid);
    conn_state(data, SMB_CONNECTED);
    *done = true;
    break;

  default:
    smb_pop_message(conn);
    return CURLE_OK; /* ignore */
  }

  smb_pop_message(conn);

  return CURLE_OK;
}

/*
 * Convert a timestamp from the Windows world (100 nsec units from 1 Jan 1601)
 * to Posix time. Cap the output to fit within a time_t.
 */
static void get_posix_time(time_t *out, curl_off_t timestamp)
{
  timestamp -= 116444736000000000;
  timestamp /= 10000000;
#if SIZEOF_TIME_T < SIZEOF_CURL_OFF_T
  if(timestamp > TIME_T_MAX)
    *out = TIME_T_MAX;
  else if(timestamp < TIME_T_MIN)
    *out = TIME_T_MIN;
  else
#endif
    *out = (time_t) timestamp;
}

static CURLcode smb_request_state(struct Curl_easy *data, bool *done)
{
  struct connectdata *conn = data->conn;
  struct smb_request *req = data->req.p.smb;
  struct smb_header *h;
  struct smb_conn *smbc = &conn->proto.smbc;
  enum smb_req_state next_state = SMB_DONE;
  unsigned short len;
  unsigned short off;
  CURLcode result;
  void *msg = NULL;
  const struct smb_nt_create_response *smb_m;

  /* Start the request */
  if(req->state == SMB_REQUESTING) {
    result = smb_send_tree_connect(data);
    if(result) {
      connclose(conn, "SMB: failed to send tree connect message");
      return result;
    }

    request_state(data, SMB_TREE_CONNECT);
  }

  /* Send the previous message and check for a response */
  result = smb_send_and_recv(data, &msg);
  if(result && result != CURLE_AGAIN) {
    connclose(conn, "SMB: failed to communicate");
    return result;
  }

  if(!msg)
    return CURLE_OK;

  h = msg;

  switch(req->state) {
  case SMB_TREE_CONNECT:
    if(h->status) {
      req->result = CURLE_REMOTE_FILE_NOT_FOUND;
      if(h->status == smb_swap32(SMB_ERR_NOACCESS))
        req->result = CURLE_REMOTE_ACCESS_DENIED;
      break;
    }
    req->tid = smb_swap16(h->tid);
    next_state = SMB_OPEN;
    break;

  case SMB_OPEN:
    if(h->status || smbc->got < sizeof(struct smb_nt_create_response)) {
      req->result = CURLE_REMOTE_FILE_NOT_FOUND;
      if(h->status == smb_swap32(SMB_ERR_NOACCESS))
        req->result = CURLE_REMOTE_ACCESS_DENIED;
      next_state = SMB_TREE_DISCONNECT;
      break;
    }
    smb_m = (const struct smb_nt_create_response*) msg;
    req->fid = smb_swap16(smb_m->fid);
    data->req.offset = 0;
    if(data->set.upload) {
      data->req.size = data->state.infilesize;
      Curl_pgrsSetUploadSize(data, data->req.size);
      next_state = SMB_UPLOAD;
    }
    else {
      smb_m = (const struct smb_nt_create_response*) msg;
      data->req.size = smb_swap64(smb_m->end_of_file);
      if(data->req.size < 0) {
        req->result = CURLE_WEIRD_SERVER_REPLY;
        next_state = SMB_CLOSE;
      }
      else {
        Curl_pgrsSetDownloadSize(data, data->req.size);
        if(data->set.get_filetime)
          get_posix_time(&data->info.filetime, smb_m->last_change_time);
        next_state = SMB_DOWNLOAD;
      }
    }
    break;

  case SMB_DOWNLOAD:
    if(h->status || smbc->got < sizeof(struct smb_header) + 14) {
      req->result = CURLE_RECV_ERROR;
      next_state = SMB_CLOSE;
      break;
    }
    len = Curl_read16_le(((const unsigned char *) msg) +
                         sizeof(struct smb_header) + 11);
    off = Curl_read16_le(((const unsigned char *) msg) +
                         sizeof(struct smb_header) + 13);
    if(len > 0) {
      if(off + sizeof(unsigned int) + len > smbc->got) {
        failf(data, "Invalid input packet");
        result = CURLE_RECV_ERROR;
      }
      else
        result = Curl_client_write(data, CLIENTWRITE_BODY,
                                   (char *)msg + off + sizeof(unsigned int),
                                   len);
      if(result) {
        req->result = result;
        next_state = SMB_CLOSE;
        break;
      }
    }
    data->req.bytecount += len;
    data->req.offset += len;
    Curl_pgrsSetDownloadCounter(data, data->req.bytecount);
    next_state = (len < MAX_PAYLOAD_SIZE) ? SMB_CLOSE : SMB_DOWNLOAD;
    break;

  case SMB_UPLOAD:
    if(h->status || smbc->got < sizeof(struct smb_header) + 6) {
      req->result = CURLE_UPLOAD_FAILED;
      next_state = SMB_CLOSE;
      break;
    }
    len = Curl_read16_le(((const unsigned char *) msg) +
                         sizeof(struct smb_header) + 5);
    data->req.bytecount += len;
    data->req.offset += len;
    Curl_pgrsSetUploadCounter(data, data->req.bytecount);
    if(data->req.bytecount >= data->req.size)
      next_state = SMB_CLOSE;
    else
      next_state = SMB_UPLOAD;
    break;

  case SMB_CLOSE:
    /* We don't care if the close failed, proceed to tree disconnect anyway */
    next_state = SMB_TREE_DISCONNECT;
    break;

  case SMB_TREE_DISCONNECT:
    next_state = SMB_DONE;
    break;

  default:
    smb_pop_message(conn);
    return CURLE_OK; /* ignore */
  }

  smb_pop_message(conn);

  switch(next_state) {
  case SMB_OPEN:
    result = smb_send_open(data);
    break;

  case SMB_DOWNLOAD:
    result = smb_send_read(data);
    break;

  case SMB_UPLOAD:
    result = smb_send_write(data);
    break;

  case SMB_CLOSE:
    result = smb_send_close(data);
    break;

  case SMB_TREE_DISCONNECT:
    result = smb_send_tree_disconnect(data);
    break;

  case SMB_DONE:
    result = req->result;
    *done = true;
    break;

  default:
    break;
  }

  if(result) {
    connclose(conn, "SMB: failed to send message");
    return result;
  }

  request_state(data, next_state);

  return CURLE_OK;
}

static CURLcode smb_done(struct Curl_easy *data, CURLcode status,
                         bool premature)
{
  (void) premature;
  Curl_safefree(data->req.p.smb);
  return status;
}

static CURLcode smb_disconnect(struct Curl_easy *data,
                               struct connectdata *conn, bool dead)
{
  struct smb_conn *smbc = &conn->proto.smbc;
  (void) dead;
  (void) data;
  Curl_safefree(smbc->share);
  Curl_safefree(smbc->domain);
  Curl_safefree(smbc->recv_buf);
  return CURLE_OK;
}

static int smb_getsock(struct Curl_easy *data,
                       struct connectdata *conn, curl_socket_t *socks)
{
  (void)data;
  socks[0] = conn->sock[FIRSTSOCKET];
  return GETSOCK_READSOCK(0) | GETSOCK_WRITESOCK(0);
}

static CURLcode smb_do(struct Curl_easy *data, bool *done)
{
  struct connectdata *conn = data->conn;
  struct smb_conn *smbc = &conn->proto.smbc;

  *done = FALSE;
  if(smbc->share) {
    return CURLE_OK;
  }
  return CURLE_URL_MALFORMAT;
}

static CURLcode smb_parse_url_path(struct Curl_easy *data,
                                   struct connectdata *conn)
{
  struct smb_request *req = data->req.p.smb;
  struct smb_conn *smbc = &conn->proto.smbc;
  char *path;
  char *slash;

  /* URL decode the path */
  CURLcode result = Curl_urldecode(data, data->state.up.path, 0, &path, NULL,
                                   REJECT_CTRL);
  if(result)
    return result;

  /* Parse the path for the share */
  smbc->share = strdup((*path == '/' || *path == '\\') ? path + 1 : path);
  free(path);
  if(!smbc->share)
    return CURLE_OUT_OF_MEMORY;

  slash = strchr(smbc->share, '/');
  if(!slash)
    slash = strchr(smbc->share, '\\');

  /* The share must be present */
  if(!slash) {
    Curl_safefree(smbc->share);
    return CURLE_URL_MALFORMAT;
  }

  /* Parse the path for the file path converting any forward slashes into
     backslashes */
  *slash++ = 0;
  req->path = slash;

  for(; *slash; slash++) {
    if(*slash == '/')
      *slash = '\\';
  }
  return CURLE_OK;
}

#endif /* CURL_DISABLE_SMB && USE_CURL_NTLM_CORE &&
          SIZEOF_CURL_OFF_T > 4 */
