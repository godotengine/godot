/* GSSAPI/krb5 support for FTP - loosely based on old krb4.c
 *
 * Copyright (c) 1995, 1996, 1997, 1998, 1999 Kungliga Tekniska Högskolan
 * (Royal Institute of Technology, Stockholm, Sweden).
 * Copyright (c) 2004 - 2021 Daniel Stenberg
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the Institute nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE INSTITUTE AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE INSTITUTE OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.  */

#include "curl_setup.h"

#if defined(HAVE_GSSAPI) && !defined(CURL_DISABLE_FTP)

#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif

#include "urldata.h"
#include "curl_base64.h"
#include "ftp.h"
#include "curl_gssapi.h"
#include "sendf.h"
#include "curl_krb5.h"
#include "warnless.h"
#include "non-ascii.h"
#include "strcase.h"
#include "strdup.h"

/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"

static CURLcode ftpsend(struct Curl_easy *data, struct connectdata *conn,
                        const char *cmd)
{
  ssize_t bytes_written;
#define SBUF_SIZE 1024
  char s[SBUF_SIZE];
  size_t write_len;
  char *sptr = s;
  CURLcode result = CURLE_OK;
#ifdef HAVE_GSSAPI
  enum protection_level data_sec = conn->data_prot;
#endif

  if(!cmd)
    return CURLE_BAD_FUNCTION_ARGUMENT;

  write_len = strlen(cmd);
  if(!write_len || write_len > (sizeof(s) -3))
    return CURLE_BAD_FUNCTION_ARGUMENT;

  memcpy(&s, cmd, write_len);
  strcpy(&s[write_len], "\r\n"); /* append a trailing CRLF */
  write_len += 2;
  bytes_written = 0;

  result = Curl_convert_to_network(data, s, write_len);
  /* Curl_convert_to_network calls failf if unsuccessful */
  if(result)
    return result;

  for(;;) {
#ifdef HAVE_GSSAPI
    conn->data_prot = PROT_CMD;
#endif
    result = Curl_write(data, conn->sock[FIRSTSOCKET], sptr, write_len,
                        &bytes_written);
#ifdef HAVE_GSSAPI
    DEBUGASSERT(data_sec > PROT_NONE && data_sec < PROT_LAST);
    conn->data_prot = data_sec;
#endif

    if(result)
      break;

    Curl_debug(data, CURLINFO_HEADER_OUT, sptr, (size_t)bytes_written);

    if(bytes_written != (ssize_t)write_len) {
      write_len -= bytes_written;
      sptr += bytes_written;
    }
    else
      break;
  }

  return result;
}

static int
krb5_init(void *app_data)
{
  gss_ctx_id_t *context = app_data;
  /* Make sure our context is initialized for krb5_end. */
  *context = GSS_C_NO_CONTEXT;
  return 0;
}

static int
krb5_check_prot(void *app_data, int level)
{
  (void)app_data; /* unused */
  if(level == PROT_CONFIDENTIAL)
    return -1;
  return 0;
}

static int
krb5_decode(void *app_data, void *buf, int len,
            int level UNUSED_PARAM,
            struct connectdata *conn UNUSED_PARAM)
{
  gss_ctx_id_t *context = app_data;
  OM_uint32 maj, min;
  gss_buffer_desc enc, dec;

  (void)level;
  (void)conn;

  enc.value = buf;
  enc.length = len;
  maj = gss_unwrap(&min, *context, &enc, &dec, NULL, NULL);
  if(maj != GSS_S_COMPLETE) {
    if(len >= 4)
      strcpy(buf, "599 ");
    return -1;
  }

  memcpy(buf, dec.value, dec.length);
  len = curlx_uztosi(dec.length);
  gss_release_buffer(&min, &dec);

  return len;
}

static int
krb5_encode(void *app_data, const void *from, int length, int level, void **to)
{
  gss_ctx_id_t *context = app_data;
  gss_buffer_desc dec, enc;
  OM_uint32 maj, min;
  int state;
  int len;

  /* NOTE that the cast is safe, neither of the krb5, gnu gss and heimdal
   * libraries modify the input buffer in gss_wrap()
   */
  dec.value = (void *)from;
  dec.length = length;
  maj = gss_wrap(&min, *context,
                 level == PROT_PRIVATE,
                 GSS_C_QOP_DEFAULT,
                 &dec, &state, &enc);

  if(maj != GSS_S_COMPLETE)
    return -1;

  /* malloc a new buffer, in case gss_release_buffer doesn't work as
     expected */
  *to = malloc(enc.length);
  if(!*to)
    return -1;
  memcpy(*to, enc.value, enc.length);
  len = curlx_uztosi(enc.length);
  gss_release_buffer(&min, &enc);
  return len;
}

static int
krb5_auth(void *app_data, struct Curl_easy *data, struct connectdata *conn)
{
  int ret = AUTH_OK;
  char *p;
  const char *host = conn->host.name;
  ssize_t nread;
  curl_socklen_t l = sizeof(conn->local_addr);
  CURLcode result;
  const char *service = data->set.str[STRING_SERVICE_NAME] ?
                        data->set.str[STRING_SERVICE_NAME] :
                        "ftp";
  const char *srv_host = "host";
  gss_buffer_desc input_buffer, output_buffer, _gssresp, *gssresp;
  OM_uint32 maj, min;
  gss_name_t gssname;
  gss_ctx_id_t *context = app_data;
  struct gss_channel_bindings_struct chan;
  size_t base64_sz = 0;
  struct sockaddr_in **remote_addr =
    (struct sockaddr_in **)&conn->ip_addr->ai_addr;
  char *stringp;

  if(getsockname(conn->sock[FIRSTSOCKET],
                 (struct sockaddr *)&conn->local_addr, &l) < 0)
    perror("getsockname()");

  chan.initiator_addrtype = GSS_C_AF_INET;
  chan.initiator_address.length = l - 4;
  chan.initiator_address.value = &conn->local_addr.sin_addr.s_addr;
  chan.acceptor_addrtype = GSS_C_AF_INET;
  chan.acceptor_address.length = l - 4;
  chan.acceptor_address.value = &(*remote_addr)->sin_addr.s_addr;
  chan.application_data.length = 0;
  chan.application_data.value = NULL;

  /* this loop will execute twice (once for service, once for host) */
  for(;;) {
    /* this really shouldn't be repeated here, but can't help it */
    if(service == srv_host) {
      result = ftpsend(data, conn, "AUTH GSSAPI");
      if(result)
        return -2;

      if(Curl_GetFTPResponse(data, &nread, NULL))
        return -1;

      if(data->state.buffer[0] != '3')
        return -1;
    }

    stringp = aprintf("%s@%s", service, host);
    if(!stringp)
      return -2;

    input_buffer.value = stringp;
    input_buffer.length = strlen(stringp);
    maj = gss_import_name(&min, &input_buffer, GSS_C_NT_HOSTBASED_SERVICE,
                          &gssname);
    free(stringp);
    if(maj != GSS_S_COMPLETE) {
      gss_release_name(&min, &gssname);
      if(service == srv_host) {
        failf(data, "Error importing service name %s@%s", service, host);
        return AUTH_ERROR;
      }
      service = srv_host;
      continue;
    }
    /* We pass NULL as |output_name_type| to avoid a leak. */
    gss_display_name(&min, gssname, &output_buffer, NULL);
    infof(data, "Trying against %s", output_buffer.value);
    gssresp = GSS_C_NO_BUFFER;
    *context = GSS_C_NO_CONTEXT;

    do {
      /* Release the buffer at each iteration to avoid leaking: the first time
         we are releasing the memory from gss_display_name. The last item is
         taken care by a final gss_release_buffer. */
      gss_release_buffer(&min, &output_buffer);
      ret = AUTH_OK;
      maj = Curl_gss_init_sec_context(data,
                                      &min,
                                      context,
                                      gssname,
                                      &Curl_krb5_mech_oid,
                                      &chan,
                                      gssresp,
                                      &output_buffer,
                                      TRUE,
                                      NULL);

      if(gssresp) {
        free(_gssresp.value);
        gssresp = NULL;
      }

      if(GSS_ERROR(maj)) {
        infof(data, "Error creating security context");
        ret = AUTH_ERROR;
        break;
      }

      if(output_buffer.length) {
        char *cmd;

        result = Curl_base64_encode(data, (char *)output_buffer.value,
                                    output_buffer.length, &p, &base64_sz);
        if(result) {
          infof(data, "base64-encoding: %s", curl_easy_strerror(result));
          ret = AUTH_ERROR;
          break;
        }

        cmd = aprintf("ADAT %s", p);
        if(cmd)
          result = ftpsend(data, conn, cmd);
        else
          result = CURLE_OUT_OF_MEMORY;

        free(p);
        free(cmd);

        if(result) {
          ret = -2;
          break;
        }

        if(Curl_GetFTPResponse(data, &nread, NULL)) {
          ret = -1;
          break;
        }

        if(data->state.buffer[0] != '2' && data->state.buffer[0] != '3') {
          infof(data, "Server didn't accept auth data");
          ret = AUTH_ERROR;
          break;
        }

        _gssresp.value = NULL; /* make sure it is initialized */
        p = data->state.buffer + 4;
        p = strstr(p, "ADAT=");
        if(p) {
          result = Curl_base64_decode(p + 5,
                                      (unsigned char **)&_gssresp.value,
                                      &_gssresp.length);
          if(result) {
            failf(data, "base64-decoding: %s", curl_easy_strerror(result));
            ret = AUTH_CONTINUE;
            break;
          }
        }

        gssresp = &_gssresp;
      }
    } while(maj == GSS_S_CONTINUE_NEEDED);

    gss_release_name(&min, &gssname);
    gss_release_buffer(&min, &output_buffer);

    if(gssresp)
      free(_gssresp.value);

    if(ret == AUTH_OK || service == srv_host)
      return ret;

    service = srv_host;
  }
  return ret;
}

static void krb5_end(void *app_data)
{
    OM_uint32 min;
    gss_ctx_id_t *context = app_data;
    if(*context != GSS_C_NO_CONTEXT) {
      OM_uint32 maj = gss_delete_sec_context(&min, context, GSS_C_NO_BUFFER);
      (void)maj;
      DEBUGASSERT(maj == GSS_S_COMPLETE);
    }
}

static const struct Curl_sec_client_mech Curl_krb5_client_mech = {
  "GSSAPI",
  sizeof(gss_ctx_id_t),
  krb5_init,
  krb5_auth,
  krb5_end,
  krb5_check_prot,

  krb5_encode,
  krb5_decode
};

static const struct {
  enum protection_level level;
  const char *name;
} level_names[] = {
  { PROT_CLEAR, "clear" },
  { PROT_SAFE, "safe" },
  { PROT_CONFIDENTIAL, "confidential" },
  { PROT_PRIVATE, "private" }
};

static enum protection_level
name_to_level(const char *name)
{
  int i;
  for(i = 0; i < (int)sizeof(level_names)/(int)sizeof(level_names[0]); i++)
    if(curl_strequal(name, level_names[i].name))
      return level_names[i].level;
  return PROT_NONE;
}

/* Convert a protocol |level| to its char representation.
   We take an int to catch programming mistakes. */
static char level_to_char(int level)
{
  switch(level) {
  case PROT_CLEAR:
    return 'C';
  case PROT_SAFE:
    return 'S';
  case PROT_CONFIDENTIAL:
    return 'E';
  case PROT_PRIVATE:
    return 'P';
  case PROT_CMD:
    /* Fall through */
  default:
    /* Those 2 cases should not be reached! */
    break;
  }
  DEBUGASSERT(0);
  /* Default to the most secure alternative. */
  return 'P';
}

/* Send an FTP command defined by |message| and the optional arguments. The
   function returns the ftp_code. If an error occurs, -1 is returned. */
static int ftp_send_command(struct Curl_easy *data, const char *message, ...)
{
  int ftp_code;
  ssize_t nread = 0;
  va_list args;
  char print_buffer[50];

  va_start(args, message);
  mvsnprintf(print_buffer, sizeof(print_buffer), message, args);
  va_end(args);

  if(ftpsend(data, data->conn, print_buffer)) {
    ftp_code = -1;
  }
  else {
    if(Curl_GetFTPResponse(data, &nread, &ftp_code))
      ftp_code = -1;
  }

  (void)nread; /* Unused */
  return ftp_code;
}

/* Read |len| from the socket |fd| and store it in |to|. Return a CURLcode
   saying whether an error occurred or CURLE_OK if |len| was read. */
static CURLcode
socket_read(curl_socket_t fd, void *to, size_t len)
{
  char *to_p = to;
  CURLcode result;
  ssize_t nread = 0;

  while(len > 0) {
    result = Curl_read_plain(fd, to_p, len, &nread);
    if(!result) {
      len -= nread;
      to_p += nread;
    }
    else {
      if(result == CURLE_AGAIN)
        continue;
      return result;
    }
  }
  return CURLE_OK;
}


/* Write |len| bytes from the buffer |to| to the socket |fd|. Return a
   CURLcode saying whether an error occurred or CURLE_OK if |len| was
   written. */
static CURLcode
socket_write(struct Curl_easy *data, curl_socket_t fd, const void *to,
             size_t len)
{
  const char *to_p = to;
  CURLcode result;
  ssize_t written;

  while(len > 0) {
    result = Curl_write_plain(data, fd, to_p, len, &written);
    if(!result) {
      len -= written;
      to_p += written;
    }
    else {
      if(result == CURLE_AGAIN)
        continue;
      return result;
    }
  }
  return CURLE_OK;
}

static CURLcode read_data(struct connectdata *conn,
                          curl_socket_t fd,
                          struct krb5buffer *buf)
{
  int len;
  CURLcode result;

  result = socket_read(fd, &len, sizeof(len));
  if(result)
    return result;

  if(len) {
    /* only realloc if there was a length */
    len = ntohl(len);
    buf->data = Curl_saferealloc(buf->data, len);
  }
  if(!len || !buf->data)
    return CURLE_OUT_OF_MEMORY;

  result = socket_read(fd, buf->data, len);
  if(result)
    return result;
  buf->size = conn->mech->decode(conn->app_data, buf->data, len,
                                 conn->data_prot, conn);
  buf->index = 0;
  return CURLE_OK;
}

static size_t
buffer_read(struct krb5buffer *buf, void *data, size_t len)
{
  if(buf->size - buf->index < len)
    len = buf->size - buf->index;
  memcpy(data, (char *)buf->data + buf->index, len);
  buf->index += len;
  return len;
}

/* Matches Curl_recv signature */
static ssize_t sec_recv(struct Curl_easy *data, int sockindex,
                        char *buffer, size_t len, CURLcode *err)
{
  size_t bytes_read;
  size_t total_read = 0;
  struct connectdata *conn = data->conn;
  curl_socket_t fd = conn->sock[sockindex];

  *err = CURLE_OK;

  /* Handle clear text response. */
  if(conn->sec_complete == 0 || conn->data_prot == PROT_CLEAR)
    return sread(fd, buffer, len);

  if(conn->in_buffer.eof_flag) {
    conn->in_buffer.eof_flag = 0;
    return 0;
  }

  bytes_read = buffer_read(&conn->in_buffer, buffer, len);
  len -= bytes_read;
  total_read += bytes_read;
  buffer += bytes_read;

  while(len > 0) {
    if(read_data(conn, fd, &conn->in_buffer))
      return -1;
    if(conn->in_buffer.size == 0) {
      if(bytes_read > 0)
        conn->in_buffer.eof_flag = 1;
      return bytes_read;
    }
    bytes_read = buffer_read(&conn->in_buffer, buffer, len);
    len -= bytes_read;
    total_read += bytes_read;
    buffer += bytes_read;
  }
  return total_read;
}

/* Send |length| bytes from |from| to the |fd| socket taking care of encoding
   and negotiating with the server. |from| can be NULL. */
static void do_sec_send(struct Curl_easy *data, struct connectdata *conn,
                        curl_socket_t fd, const char *from, int length)
{
  int bytes, htonl_bytes; /* 32-bit integers for htonl */
  char *buffer = NULL;
  char *cmd_buffer;
  size_t cmd_size = 0;
  CURLcode error;
  enum protection_level prot_level = conn->data_prot;
  bool iscmd = (prot_level == PROT_CMD)?TRUE:FALSE;

  DEBUGASSERT(prot_level > PROT_NONE && prot_level < PROT_LAST);

  if(iscmd) {
    if(!strncmp(from, "PASS ", 5) || !strncmp(from, "ACCT ", 5))
      prot_level = PROT_PRIVATE;
    else
      prot_level = conn->command_prot;
  }
  bytes = conn->mech->encode(conn->app_data, from, length, prot_level,
                             (void **)&buffer);
  if(!buffer || bytes <= 0)
    return; /* error */

  if(iscmd) {
    error = Curl_base64_encode(data, buffer, curlx_sitouz(bytes),
                               &cmd_buffer, &cmd_size);
    if(error) {
      free(buffer);
      return; /* error */
    }
    if(cmd_size > 0) {
      static const char *enc = "ENC ";
      static const char *mic = "MIC ";
      if(prot_level == PROT_PRIVATE)
        socket_write(data, fd, enc, 4);
      else
        socket_write(data, fd, mic, 4);

      socket_write(data, fd, cmd_buffer, cmd_size);
      socket_write(data, fd, "\r\n", 2);
      infof(data, "Send: %s%s", prot_level == PROT_PRIVATE?enc:mic,
            cmd_buffer);
      free(cmd_buffer);
    }
  }
  else {
    htonl_bytes = htonl(bytes);
    socket_write(data, fd, &htonl_bytes, sizeof(htonl_bytes));
    socket_write(data, fd, buffer, curlx_sitouz(bytes));
  }
  free(buffer);
}

static ssize_t sec_write(struct Curl_easy *data, struct connectdata *conn,
                         curl_socket_t fd, const char *buffer, size_t length)
{
  ssize_t tx = 0, len = conn->buffer_size;

  if(len <= 0)
    len = length;
  while(length) {
    if(length < (size_t)len)
      len = length;

    do_sec_send(data, conn, fd, buffer, curlx_sztosi(len));
    length -= len;
    buffer += len;
    tx += len;
  }
  return tx;
}

/* Matches Curl_send signature */
static ssize_t sec_send(struct Curl_easy *data, int sockindex,
                        const void *buffer, size_t len, CURLcode *err)
{
  struct connectdata *conn = data->conn;
  curl_socket_t fd = conn->sock[sockindex];
  *err = CURLE_OK;
  return sec_write(data, conn, fd, buffer, len);
}

int Curl_sec_read_msg(struct Curl_easy *data, struct connectdata *conn,
                      char *buffer, enum protection_level level)
{
  /* decoded_len should be size_t or ssize_t but conn->mech->decode returns an
     int */
  int decoded_len;
  char *buf;
  int ret_code = 0;
  size_t decoded_sz = 0;
  CURLcode error;

  (void) data;

  if(!conn->mech)
    /* not initialized, return error */
    return -1;

  DEBUGASSERT(level > PROT_NONE && level < PROT_LAST);

  error = Curl_base64_decode(buffer + 4, (unsigned char **)&buf, &decoded_sz);
  if(error || decoded_sz == 0)
    return -1;

  if(decoded_sz > (size_t)INT_MAX) {
    free(buf);
    return -1;
  }
  decoded_len = curlx_uztosi(decoded_sz);

  decoded_len = conn->mech->decode(conn->app_data, buf, decoded_len,
                                   level, conn);
  if(decoded_len <= 0) {
    free(buf);
    return -1;
  }

  {
    buf[decoded_len] = '\n';
    Curl_debug(data, CURLINFO_HEADER_IN, buf, decoded_len + 1);
  }

  buf[decoded_len] = '\0';
  if(decoded_len <= 3)
    /* suspiciously short */
    return 0;

  if(buf[3] != '-')
    /* safe to ignore return code */
    (void)sscanf(buf, "%d", &ret_code);

  if(buf[decoded_len - 1] == '\n')
    buf[decoded_len - 1] = '\0';
  strcpy(buffer, buf);
  free(buf);
  return ret_code;
}

static int sec_set_protection_level(struct Curl_easy *data)
{
  int code;
  struct connectdata *conn = data->conn;
  enum protection_level level = conn->request_data_prot;

  DEBUGASSERT(level > PROT_NONE && level < PROT_LAST);

  if(!conn->sec_complete) {
    infof(data, "Trying to change the protection level after the"
                " completion of the data exchange.");
    return -1;
  }

  /* Bail out if we try to set up the same level */
  if(conn->data_prot == level)
    return 0;

  if(level) {
    char *pbsz;
    unsigned int buffer_size = 1 << 20; /* 1048576 */

    code = ftp_send_command(data, "PBSZ %u", buffer_size);
    if(code < 0)
      return -1;

    if(code/100 != 2) {
      failf(data, "Failed to set the protection's buffer size.");
      return -1;
    }
    conn->buffer_size = buffer_size;

    pbsz = strstr(data->state.buffer, "PBSZ=");
    if(pbsz) {
      /* ignore return code, use default value if it fails */
      (void)sscanf(pbsz, "PBSZ=%u", &buffer_size);
      if(buffer_size < conn->buffer_size)
        conn->buffer_size = buffer_size;
    }
  }

  /* Now try to negotiate the protection level. */
  code = ftp_send_command(data, "PROT %c", level_to_char(level));

  if(code < 0)
    return -1;

  if(code/100 != 2) {
    failf(data, "Failed to set the protection level.");
    return -1;
  }

  conn->data_prot = level;
  if(level == PROT_PRIVATE)
    conn->command_prot = level;

  return 0;
}

int
Curl_sec_request_prot(struct connectdata *conn, const char *level)
{
  enum protection_level l = name_to_level(level);
  if(l == PROT_NONE)
    return -1;
  DEBUGASSERT(l > PROT_NONE && l < PROT_LAST);
  conn->request_data_prot = l;
  return 0;
}

static CURLcode choose_mech(struct Curl_easy *data, struct connectdata *conn)
{
  int ret;
  void *tmp_allocation;
  const struct Curl_sec_client_mech *mech = &Curl_krb5_client_mech;

  tmp_allocation = realloc(conn->app_data, mech->size);
  if(!tmp_allocation) {
    failf(data, "Failed realloc of size %zu", mech->size);
    mech = NULL;
    return CURLE_OUT_OF_MEMORY;
  }
  conn->app_data = tmp_allocation;

  if(mech->init) {
    ret = mech->init(conn->app_data);
    if(ret) {
      infof(data, "Failed initialization for %s. Skipping it.",
            mech->name);
      return CURLE_FAILED_INIT;
    }
  }

  infof(data, "Trying mechanism %s...", mech->name);
  ret = ftp_send_command(data, "AUTH %s", mech->name);
  if(ret < 0)
    return CURLE_COULDNT_CONNECT;

  if(ret/100 != 3) {
    switch(ret) {
    case 504:
      infof(data, "Mechanism %s is not supported by the server (server "
            "returned ftp code: 504).", mech->name);
      break;
    case 534:
      infof(data, "Mechanism %s was rejected by the server (server returned "
            "ftp code: 534).", mech->name);
      break;
    default:
      if(ret/100 == 5) {
        infof(data, "server does not support the security extensions");
        return CURLE_USE_SSL_FAILED;
      }
      break;
    }
    return CURLE_LOGIN_DENIED;
  }

  /* Authenticate */
  ret = mech->auth(conn->app_data, data, conn);

  if(ret != AUTH_CONTINUE) {
    if(ret != AUTH_OK) {
      /* Mechanism has dumped the error to stderr, don't error here. */
      return CURLE_USE_SSL_FAILED;
    }
    DEBUGASSERT(ret == AUTH_OK);

    conn->mech = mech;
    conn->sec_complete = 1;
    conn->recv[FIRSTSOCKET] = sec_recv;
    conn->send[FIRSTSOCKET] = sec_send;
    conn->recv[SECONDARYSOCKET] = sec_recv;
    conn->send[SECONDARYSOCKET] = sec_send;
    conn->command_prot = PROT_SAFE;
    /* Set the requested protection level */
    /* BLOCKING */
    (void)sec_set_protection_level(data);
  }

  return CURLE_OK;
}

CURLcode
Curl_sec_login(struct Curl_easy *data, struct connectdata *conn)
{
  return choose_mech(data, conn);
}


void
Curl_sec_end(struct connectdata *conn)
{
  if(conn->mech != NULL && conn->mech->end)
    conn->mech->end(conn->app_data);
  free(conn->app_data);
  conn->app_data = NULL;
  if(conn->in_buffer.data) {
    free(conn->in_buffer.data);
    conn->in_buffer.data = NULL;
    conn->in_buffer.size = 0;
    conn->in_buffer.index = 0;
    conn->in_buffer.eof_flag = 0;
  }
  conn->sec_complete = 0;
  conn->data_prot = PROT_CLEAR;
  conn->mech = NULL;
}

#endif /* HAVE_GSSAPI && !CURL_DISABLE_FTP */
