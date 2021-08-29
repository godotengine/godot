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

#ifndef CURL_DISABLE_TFTP

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

#include "urldata.h"
#include <curl/curl.h>
#include "transfer.h"
#include "sendf.h"
#include "tftp.h"
#include "progress.h"
#include "connect.h"
#include "strerror.h"
#include "sockaddr.h" /* required for Curl_sockaddr_storage */
#include "multiif.h"
#include "url.h"
#include "strcase.h"
#include "speedcheck.h"
#include "select.h"
#include "escape.h"

/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"

/* RFC2348 allows the block size to be negotiated */
#define TFTP_BLKSIZE_DEFAULT 512
#define TFTP_BLKSIZE_MIN 8
#define TFTP_BLKSIZE_MAX 65464
#define TFTP_OPTION_BLKSIZE "blksize"

/* from RFC2349: */
#define TFTP_OPTION_TSIZE    "tsize"
#define TFTP_OPTION_INTERVAL "timeout"

typedef enum {
  TFTP_MODE_NETASCII = 0,
  TFTP_MODE_OCTET
} tftp_mode_t;

typedef enum {
  TFTP_STATE_START = 0,
  TFTP_STATE_RX,
  TFTP_STATE_TX,
  TFTP_STATE_FIN
} tftp_state_t;

typedef enum {
  TFTP_EVENT_NONE = -1,
  TFTP_EVENT_INIT = 0,
  TFTP_EVENT_RRQ = 1,
  TFTP_EVENT_WRQ = 2,
  TFTP_EVENT_DATA = 3,
  TFTP_EVENT_ACK = 4,
  TFTP_EVENT_ERROR = 5,
  TFTP_EVENT_OACK = 6,
  TFTP_EVENT_TIMEOUT
} tftp_event_t;

typedef enum {
  TFTP_ERR_UNDEF = 0,
  TFTP_ERR_NOTFOUND,
  TFTP_ERR_PERM,
  TFTP_ERR_DISKFULL,
  TFTP_ERR_ILLEGAL,
  TFTP_ERR_UNKNOWNID,
  TFTP_ERR_EXISTS,
  TFTP_ERR_NOSUCHUSER,  /* This will never be triggered by this code */

  /* The remaining error codes are internal to curl */
  TFTP_ERR_NONE = -100,
  TFTP_ERR_TIMEOUT,
  TFTP_ERR_NORESPONSE
} tftp_error_t;

struct tftp_packet {
  unsigned char *data;
};

struct tftp_state_data {
  tftp_state_t    state;
  tftp_mode_t     mode;
  tftp_error_t    error;
  tftp_event_t    event;
  struct Curl_easy *data;
  curl_socket_t   sockfd;
  int             retries;
  int             retry_time;
  int             retry_max;
  time_t          rx_time;
  struct Curl_sockaddr_storage   local_addr;
  struct Curl_sockaddr_storage   remote_addr;
  curl_socklen_t  remote_addrlen;
  int             rbytes;
  int             sbytes;
  int             blksize;
  int             requested_blksize;
  unsigned short  block;
  struct tftp_packet rpacket;
  struct tftp_packet spacket;
};


/* Forward declarations */
static CURLcode tftp_rx(struct tftp_state_data *state, tftp_event_t event);
static CURLcode tftp_tx(struct tftp_state_data *state, tftp_event_t event);
static CURLcode tftp_connect(struct Curl_easy *data, bool *done);
static CURLcode tftp_disconnect(struct Curl_easy *data,
                                struct connectdata *conn,
                                bool dead_connection);
static CURLcode tftp_do(struct Curl_easy *data, bool *done);
static CURLcode tftp_done(struct Curl_easy *data,
                          CURLcode, bool premature);
static CURLcode tftp_setup_connection(struct Curl_easy *data,
                                      struct connectdata *conn);
static CURLcode tftp_multi_statemach(struct Curl_easy *data, bool *done);
static CURLcode tftp_doing(struct Curl_easy *data, bool *dophase_done);
static int tftp_getsock(struct Curl_easy *data, struct connectdata *conn,
                        curl_socket_t *socks);
static CURLcode tftp_translate_code(tftp_error_t error);


/*
 * TFTP protocol handler.
 */

const struct Curl_handler Curl_handler_tftp = {
  "TFTP",                               /* scheme */
  tftp_setup_connection,                /* setup_connection */
  tftp_do,                              /* do_it */
  tftp_done,                            /* done */
  ZERO_NULL,                            /* do_more */
  tftp_connect,                         /* connect_it */
  tftp_multi_statemach,                 /* connecting */
  tftp_doing,                           /* doing */
  tftp_getsock,                         /* proto_getsock */
  tftp_getsock,                         /* doing_getsock */
  ZERO_NULL,                            /* domore_getsock */
  ZERO_NULL,                            /* perform_getsock */
  tftp_disconnect,                      /* disconnect */
  ZERO_NULL,                            /* readwrite */
  ZERO_NULL,                            /* connection_check */
  ZERO_NULL,                            /* attach connection */
  PORT_TFTP,                            /* defport */
  CURLPROTO_TFTP,                       /* protocol */
  CURLPROTO_TFTP,                       /* family */
  PROTOPT_NOTCPPROXY | PROTOPT_NOURLQUERY /* flags */
};

/**********************************************************
 *
 * tftp_set_timeouts -
 *
 * Set timeouts based on state machine state.
 * Use user provided connect timeouts until DATA or ACK
 * packet is received, then use user-provided transfer timeouts
 *
 *
 **********************************************************/
static CURLcode tftp_set_timeouts(struct tftp_state_data *state)
{
  time_t maxtime, timeout;
  timediff_t timeout_ms;
  bool start = (state->state == TFTP_STATE_START) ? TRUE : FALSE;

  /* Compute drop-dead time */
  timeout_ms = Curl_timeleft(state->data, NULL, start);

  if(timeout_ms < 0) {
    /* time-out, bail out, go home */
    failf(state->data, "Connection time-out");
    return CURLE_OPERATION_TIMEDOUT;
  }

  if(timeout_ms > 0)
    maxtime = (time_t)(timeout_ms + 500) / 1000;
  else
    maxtime = 3600; /* use for calculating block timeouts */

  /* Set per-block timeout to total */
  timeout = maxtime;

  /* Average reposting an ACK after 5 seconds */
  state->retry_max = (int)timeout/5;

  /* But bound the total number */
  if(state->retry_max<3)
    state->retry_max = 3;

  if(state->retry_max>50)
    state->retry_max = 50;

  /* Compute the re-ACK interval to suit the timeout */
  state->retry_time = (int)(timeout/state->retry_max);
  if(state->retry_time<1)
    state->retry_time = 1;

  infof(state->data,
        "set timeouts for state %d; Total % " CURL_FORMAT_CURL_OFF_T
        ", retry %d maxtry %d",
        (int)state->state, timeout_ms, state->retry_time, state->retry_max);

  /* init RX time */
  time(&state->rx_time);

  return CURLE_OK;
}

/**********************************************************
 *
 * tftp_set_send_first
 *
 * Event handler for the START state
 *
 **********************************************************/

static void setpacketevent(struct tftp_packet *packet, unsigned short num)
{
  packet->data[0] = (unsigned char)(num >> 8);
  packet->data[1] = (unsigned char)(num & 0xff);
}


static void setpacketblock(struct tftp_packet *packet, unsigned short num)
{
  packet->data[2] = (unsigned char)(num >> 8);
  packet->data[3] = (unsigned char)(num & 0xff);
}

static unsigned short getrpacketevent(const struct tftp_packet *packet)
{
  return (unsigned short)((packet->data[0] << 8) | packet->data[1]);
}

static unsigned short getrpacketblock(const struct tftp_packet *packet)
{
  return (unsigned short)((packet->data[2] << 8) | packet->data[3]);
}

static size_t tftp_strnlen(const char *string, size_t maxlen)
{
  const char *end = memchr(string, '\0', maxlen);
  return end ? (size_t) (end - string) : maxlen;
}

static const char *tftp_option_get(const char *buf, size_t len,
                                   const char **option, const char **value)
{
  size_t loc;

  loc = tftp_strnlen(buf, len);
  loc++; /* NULL term */

  if(loc >= len)
    return NULL;
  *option = buf;

  loc += tftp_strnlen(buf + loc, len-loc);
  loc++; /* NULL term */

  if(loc > len)
    return NULL;
  *value = &buf[strlen(*option) + 1];

  return &buf[loc];
}

static CURLcode tftp_parse_option_ack(struct tftp_state_data *state,
                                      const char *ptr, int len)
{
  const char *tmp = ptr;
  struct Curl_easy *data = state->data;

  /* if OACK doesn't contain blksize option, the default (512) must be used */
  state->blksize = TFTP_BLKSIZE_DEFAULT;

  while(tmp < ptr + len) {
    const char *option, *value;

    tmp = tftp_option_get(tmp, ptr + len - tmp, &option, &value);
    if(!tmp) {
      failf(data, "Malformed ACK packet, rejecting");
      return CURLE_TFTP_ILLEGAL;
    }

    infof(data, "got option=(%s) value=(%s)", option, value);

    if(checkprefix(option, TFTP_OPTION_BLKSIZE)) {
      long blksize;

      blksize = strtol(value, NULL, 10);

      if(!blksize) {
        failf(data, "invalid blocksize value in OACK packet");
        return CURLE_TFTP_ILLEGAL;
      }
      if(blksize > TFTP_BLKSIZE_MAX) {
        failf(data, "%s (%d)", "blksize is larger than max supported",
              TFTP_BLKSIZE_MAX);
        return CURLE_TFTP_ILLEGAL;
      }
      else if(blksize < TFTP_BLKSIZE_MIN) {
        failf(data, "%s (%d)", "blksize is smaller than min supported",
              TFTP_BLKSIZE_MIN);
        return CURLE_TFTP_ILLEGAL;
      }
      else if(blksize > state->requested_blksize) {
        /* could realloc pkt buffers here, but the spec doesn't call out
         * support for the server requesting a bigger blksize than the client
         * requests */
        failf(data, "%s (%ld)",
              "server requested blksize larger than allocated", blksize);
        return CURLE_TFTP_ILLEGAL;
      }

      state->blksize = (int)blksize;
      infof(data, "%s (%d) %s (%d)", "blksize parsed from OACK",
            state->blksize, "requested", state->requested_blksize);
    }
    else if(checkprefix(option, TFTP_OPTION_TSIZE)) {
      long tsize = 0;

      tsize = strtol(value, NULL, 10);
      infof(data, "%s (%ld)", "tsize parsed from OACK", tsize);

      /* tsize should be ignored on upload: Who cares about the size of the
         remote file? */
      if(!data->set.upload) {
        if(!tsize) {
          failf(data, "invalid tsize -:%s:- value in OACK packet", value);
          return CURLE_TFTP_ILLEGAL;
        }
        Curl_pgrsSetDownloadSize(data, tsize);
      }
    }
  }

  return CURLE_OK;
}

static CURLcode tftp_option_add(struct tftp_state_data *state, size_t *csize,
                                char *buf, const char *option)
{
  if(( strlen(option) + *csize + 1) > (size_t)state->blksize)
    return CURLE_TFTP_ILLEGAL;
  strcpy(buf, option);
  *csize += strlen(option) + 1;
  return CURLE_OK;
}

static CURLcode tftp_connect_for_tx(struct tftp_state_data *state,
                                    tftp_event_t event)
{
  CURLcode result;
#ifndef CURL_DISABLE_VERBOSE_STRINGS
  struct Curl_easy *data = state->data;

  infof(data, "%s", "Connected for transmit");
#endif
  state->state = TFTP_STATE_TX;
  result = tftp_set_timeouts(state);
  if(result)
    return result;
  return tftp_tx(state, event);
}

static CURLcode tftp_connect_for_rx(struct tftp_state_data *state,
                                    tftp_event_t event)
{
  CURLcode result;
#ifndef CURL_DISABLE_VERBOSE_STRINGS
  struct Curl_easy *data = state->data;

  infof(data, "%s", "Connected for receive");
#endif
  state->state = TFTP_STATE_RX;
  result = tftp_set_timeouts(state);
  if(result)
    return result;
  return tftp_rx(state, event);
}

static CURLcode tftp_send_first(struct tftp_state_data *state,
                                tftp_event_t event)
{
  size_t sbytes;
  ssize_t senddata;
  const char *mode = "octet";
  char *filename;
  struct Curl_easy *data = state->data;
  CURLcode result = CURLE_OK;

  /* Set ascii mode if -B flag was used */
  if(data->state.prefer_ascii)
    mode = "netascii";

  switch(event) {

  case TFTP_EVENT_INIT:    /* Send the first packet out */
  case TFTP_EVENT_TIMEOUT: /* Resend the first packet out */
    /* Increment the retry counter, quit if over the limit */
    state->retries++;
    if(state->retries>state->retry_max) {
      state->error = TFTP_ERR_NORESPONSE;
      state->state = TFTP_STATE_FIN;
      return result;
    }

    if(data->set.upload) {
      /* If we are uploading, send an WRQ */
      setpacketevent(&state->spacket, TFTP_EVENT_WRQ);
      state->data->req.upload_fromhere =
        (char *)state->spacket.data + 4;
      if(data->state.infilesize != -1)
        Curl_pgrsSetUploadSize(data, data->state.infilesize);
    }
    else {
      /* If we are downloading, send an RRQ */
      setpacketevent(&state->spacket, TFTP_EVENT_RRQ);
    }
    /* As RFC3617 describes the separator slash is not actually part of the
       file name so we skip the always-present first letter of the path
       string. */
    result = Curl_urldecode(data, &state->data->state.up.path[1], 0,
                            &filename, NULL, REJECT_ZERO);
    if(result)
      return result;

    if(strlen(filename) > (state->blksize - strlen(mode) - 4)) {
      failf(data, "TFTP file name too long");
      free(filename);
      return CURLE_TFTP_ILLEGAL; /* too long file name field */
    }

    msnprintf((char *)state->spacket.data + 2,
              state->blksize,
              "%s%c%s%c", filename, '\0',  mode, '\0');
    sbytes = 4 + strlen(filename) + strlen(mode);

    /* optional addition of TFTP options */
    if(!data->set.tftp_no_options) {
      char buf[64];
      /* add tsize option */
      if(data->set.upload && (data->state.infilesize != -1))
        msnprintf(buf, sizeof(buf), "%" CURL_FORMAT_CURL_OFF_T,
                  data->state.infilesize);
      else
        strcpy(buf, "0"); /* the destination is large enough */

      result = tftp_option_add(state, &sbytes,
                               (char *)state->spacket.data + sbytes,
                               TFTP_OPTION_TSIZE);
      if(result == CURLE_OK)
        result = tftp_option_add(state, &sbytes,
                                 (char *)state->spacket.data + sbytes, buf);

      /* add blksize option */
      msnprintf(buf, sizeof(buf), "%d", state->requested_blksize);
      if(result == CURLE_OK)
        result = tftp_option_add(state, &sbytes,
                                 (char *)state->spacket.data + sbytes,
                                 TFTP_OPTION_BLKSIZE);
      if(result == CURLE_OK)
        result = tftp_option_add(state, &sbytes,
                                 (char *)state->spacket.data + sbytes, buf);

      /* add timeout option */
      msnprintf(buf, sizeof(buf), "%d", state->retry_time);
      if(result == CURLE_OK)
        result = tftp_option_add(state, &sbytes,
                                 (char *)state->spacket.data + sbytes,
                                 TFTP_OPTION_INTERVAL);
      if(result == CURLE_OK)
        result = tftp_option_add(state, &sbytes,
                                 (char *)state->spacket.data + sbytes, buf);

      if(result != CURLE_OK) {
        failf(data, "TFTP buffer too small for options");
        free(filename);
        return CURLE_TFTP_ILLEGAL;
      }
    }

    /* the typecase for the 3rd argument is mostly for systems that do
       not have a size_t argument, like older unixes that want an 'int' */
    senddata = sendto(state->sockfd, (void *)state->spacket.data,
                      (SEND_TYPE_ARG3)sbytes, 0,
                      data->conn->ip_addr->ai_addr,
                      data->conn->ip_addr->ai_addrlen);
    if(senddata != (ssize_t)sbytes) {
      char buffer[STRERROR_LEN];
      failf(data, "%s", Curl_strerror(SOCKERRNO, buffer, sizeof(buffer)));
    }
    free(filename);
    break;

  case TFTP_EVENT_OACK:
    if(data->set.upload) {
      result = tftp_connect_for_tx(state, event);
    }
    else {
      result = tftp_connect_for_rx(state, event);
    }
    break;

  case TFTP_EVENT_ACK: /* Connected for transmit */
    result = tftp_connect_for_tx(state, event);
    break;

  case TFTP_EVENT_DATA: /* Connected for receive */
    result = tftp_connect_for_rx(state, event);
    break;

  case TFTP_EVENT_ERROR:
    state->state = TFTP_STATE_FIN;
    break;

  default:
    failf(state->data, "tftp_send_first: internal error");
    break;
  }

  return result;
}

/* the next blocknum is x + 1 but it needs to wrap at an unsigned 16bit
   boundary */
#define NEXT_BLOCKNUM(x) (((x) + 1)&0xffff)

/**********************************************************
 *
 * tftp_rx
 *
 * Event handler for the RX state
 *
 **********************************************************/
static CURLcode tftp_rx(struct tftp_state_data *state,
                        tftp_event_t event)
{
  ssize_t sbytes;
  int rblock;
  struct Curl_easy *data = state->data;
  char buffer[STRERROR_LEN];

  switch(event) {

  case TFTP_EVENT_DATA:
    /* Is this the block we expect? */
    rblock = getrpacketblock(&state->rpacket);
    if(NEXT_BLOCKNUM(state->block) == rblock) {
      /* This is the expected block.  Reset counters and ACK it. */
      state->retries = 0;
    }
    else if(state->block == rblock) {
      /* This is the last recently received block again. Log it and ACK it
         again. */
      infof(data, "Received last DATA packet block %d again.", rblock);
    }
    else {
      /* totally unexpected, just log it */
      infof(data,
            "Received unexpected DATA packet block %d, expecting block %d",
            rblock, NEXT_BLOCKNUM(state->block));
      break;
    }

    /* ACK this block. */
    state->block = (unsigned short)rblock;
    setpacketevent(&state->spacket, TFTP_EVENT_ACK);
    setpacketblock(&state->spacket, state->block);
    sbytes = sendto(state->sockfd, (void *)state->spacket.data,
                    4, SEND_4TH_ARG,
                    (struct sockaddr *)&state->remote_addr,
                    state->remote_addrlen);
    if(sbytes < 0) {
      failf(data, "%s", Curl_strerror(SOCKERRNO, buffer, sizeof(buffer)));
      return CURLE_SEND_ERROR;
    }

    /* Check if completed (That is, a less than full packet is received) */
    if(state->rbytes < (ssize_t)state->blksize + 4) {
      state->state = TFTP_STATE_FIN;
    }
    else {
      state->state = TFTP_STATE_RX;
    }
    time(&state->rx_time);
    break;

  case TFTP_EVENT_OACK:
    /* ACK option acknowledgement so we can move on to data */
    state->block = 0;
    state->retries = 0;
    setpacketevent(&state->spacket, TFTP_EVENT_ACK);
    setpacketblock(&state->spacket, state->block);
    sbytes = sendto(state->sockfd, (void *)state->spacket.data,
                    4, SEND_4TH_ARG,
                    (struct sockaddr *)&state->remote_addr,
                    state->remote_addrlen);
    if(sbytes < 0) {
      failf(data, "%s", Curl_strerror(SOCKERRNO, buffer, sizeof(buffer)));
      return CURLE_SEND_ERROR;
    }

    /* we're ready to RX data */
    state->state = TFTP_STATE_RX;
    time(&state->rx_time);
    break;

  case TFTP_EVENT_TIMEOUT:
    /* Increment the retry count and fail if over the limit */
    state->retries++;
    infof(data,
          "Timeout waiting for block %d ACK.  Retries = %d",
          NEXT_BLOCKNUM(state->block), state->retries);
    if(state->retries > state->retry_max) {
      state->error = TFTP_ERR_TIMEOUT;
      state->state = TFTP_STATE_FIN;
    }
    else {
      /* Resend the previous ACK */
      sbytes = sendto(state->sockfd, (void *)state->spacket.data,
                      4, SEND_4TH_ARG,
                      (struct sockaddr *)&state->remote_addr,
                      state->remote_addrlen);
      if(sbytes<0) {
        failf(data, "%s", Curl_strerror(SOCKERRNO, buffer, sizeof(buffer)));
        return CURLE_SEND_ERROR;
      }
    }
    break;

  case TFTP_EVENT_ERROR:
    setpacketevent(&state->spacket, TFTP_EVENT_ERROR);
    setpacketblock(&state->spacket, state->block);
    (void)sendto(state->sockfd, (void *)state->spacket.data,
                 4, SEND_4TH_ARG,
                 (struct sockaddr *)&state->remote_addr,
                 state->remote_addrlen);
    /* don't bother with the return code, but if the socket is still up we
     * should be a good TFTP client and let the server know we're done */
    state->state = TFTP_STATE_FIN;
    break;

  default:
    failf(data, "%s", "tftp_rx: internal error");
    return CURLE_TFTP_ILLEGAL; /* not really the perfect return code for
                                  this */
  }
  return CURLE_OK;
}

/**********************************************************
 *
 * tftp_tx
 *
 * Event handler for the TX state
 *
 **********************************************************/
static CURLcode tftp_tx(struct tftp_state_data *state, tftp_event_t event)
{
  struct Curl_easy *data = state->data;
  ssize_t sbytes;
  CURLcode result = CURLE_OK;
  struct SingleRequest *k = &data->req;
  size_t cb; /* Bytes currently read */
  char buffer[STRERROR_LEN];

  switch(event) {

  case TFTP_EVENT_ACK:
  case TFTP_EVENT_OACK:
    if(event == TFTP_EVENT_ACK) {
      /* Ack the packet */
      int rblock = getrpacketblock(&state->rpacket);

      if(rblock != state->block &&
         /* There's a bug in tftpd-hpa that causes it to send us an ack for
          * 65535 when the block number wraps to 0. So when we're expecting
          * 0, also accept 65535. See
          * https://www.syslinux.org/archives/2010-September/015612.html
          * */
         !(state->block == 0 && rblock == 65535)) {
        /* This isn't the expected block.  Log it and up the retry counter */
        infof(data, "Received ACK for block %d, expecting %d",
              rblock, state->block);
        state->retries++;
        /* Bail out if over the maximum */
        if(state->retries>state->retry_max) {
          failf(data, "tftp_tx: giving up waiting for block %d ack",
                state->block);
          result = CURLE_SEND_ERROR;
        }
        else {
          /* Re-send the data packet */
          sbytes = sendto(state->sockfd, (void *)state->spacket.data,
                          4 + state->sbytes, SEND_4TH_ARG,
                          (struct sockaddr *)&state->remote_addr,
                          state->remote_addrlen);
          /* Check all sbytes were sent */
          if(sbytes<0) {
            failf(data, "%s", Curl_strerror(SOCKERRNO,
                                            buffer, sizeof(buffer)));
            result = CURLE_SEND_ERROR;
          }
        }

        return result;
      }
      /* This is the expected packet.  Reset the counters and send the next
         block */
      time(&state->rx_time);
      state->block++;
    }
    else
      state->block = 1; /* first data block is 1 when using OACK */

    state->retries = 0;
    setpacketevent(&state->spacket, TFTP_EVENT_DATA);
    setpacketblock(&state->spacket, state->block);
    if(state->block > 1 && state->sbytes < state->blksize) {
      state->state = TFTP_STATE_FIN;
      return CURLE_OK;
    }

    /* TFTP considers data block size < 512 bytes as an end of session. So
     * in some cases we must wait for additional data to build full (512 bytes)
     * data block.
     * */
    state->sbytes = 0;
    state->data->req.upload_fromhere = (char *)state->spacket.data + 4;
    do {
      result = Curl_fillreadbuffer(data, state->blksize - state->sbytes, &cb);
      if(result)
        return result;
      state->sbytes += (int)cb;
      state->data->req.upload_fromhere += cb;
    } while(state->sbytes < state->blksize && cb);

    sbytes = sendto(state->sockfd, (void *) state->spacket.data,
                    4 + state->sbytes, SEND_4TH_ARG,
                    (struct sockaddr *)&state->remote_addr,
                    state->remote_addrlen);
    /* Check all sbytes were sent */
    if(sbytes<0) {
      failf(data, "%s", Curl_strerror(SOCKERRNO, buffer, sizeof(buffer)));
      return CURLE_SEND_ERROR;
    }
    /* Update the progress meter */
    k->writebytecount += state->sbytes;
    Curl_pgrsSetUploadCounter(data, k->writebytecount);
    break;

  case TFTP_EVENT_TIMEOUT:
    /* Increment the retry counter and log the timeout */
    state->retries++;
    infof(data, "Timeout waiting for block %d ACK. "
          " Retries = %d", NEXT_BLOCKNUM(state->block), state->retries);
    /* Decide if we've had enough */
    if(state->retries > state->retry_max) {
      state->error = TFTP_ERR_TIMEOUT;
      state->state = TFTP_STATE_FIN;
    }
    else {
      /* Re-send the data packet */
      sbytes = sendto(state->sockfd, (void *)state->spacket.data,
                      4 + state->sbytes, SEND_4TH_ARG,
                      (struct sockaddr *)&state->remote_addr,
                      state->remote_addrlen);
      /* Check all sbytes were sent */
      if(sbytes<0) {
        failf(data, "%s", Curl_strerror(SOCKERRNO, buffer, sizeof(buffer)));
        return CURLE_SEND_ERROR;
      }
      /* since this was a re-send, we remain at the still byte position */
      Curl_pgrsSetUploadCounter(data, k->writebytecount);
    }
    break;

  case TFTP_EVENT_ERROR:
    state->state = TFTP_STATE_FIN;
    setpacketevent(&state->spacket, TFTP_EVENT_ERROR);
    setpacketblock(&state->spacket, state->block);
    (void)sendto(state->sockfd, (void *)state->spacket.data, 4, SEND_4TH_ARG,
                 (struct sockaddr *)&state->remote_addr,
                 state->remote_addrlen);
    /* don't bother with the return code, but if the socket is still up we
     * should be a good TFTP client and let the server know we're done */
    state->state = TFTP_STATE_FIN;
    break;

  default:
    failf(data, "tftp_tx: internal error, event: %i", (int)(event));
    break;
  }

  return result;
}

/**********************************************************
 *
 * tftp_translate_code
 *
 * Translate internal error codes to CURL error codes
 *
 **********************************************************/
static CURLcode tftp_translate_code(tftp_error_t error)
{
  CURLcode result = CURLE_OK;

  if(error != TFTP_ERR_NONE) {
    switch(error) {
    case TFTP_ERR_NOTFOUND:
      result = CURLE_TFTP_NOTFOUND;
      break;
    case TFTP_ERR_PERM:
      result = CURLE_TFTP_PERM;
      break;
    case TFTP_ERR_DISKFULL:
      result = CURLE_REMOTE_DISK_FULL;
      break;
    case TFTP_ERR_UNDEF:
    case TFTP_ERR_ILLEGAL:
      result = CURLE_TFTP_ILLEGAL;
      break;
    case TFTP_ERR_UNKNOWNID:
      result = CURLE_TFTP_UNKNOWNID;
      break;
    case TFTP_ERR_EXISTS:
      result = CURLE_REMOTE_FILE_EXISTS;
      break;
    case TFTP_ERR_NOSUCHUSER:
      result = CURLE_TFTP_NOSUCHUSER;
      break;
    case TFTP_ERR_TIMEOUT:
      result = CURLE_OPERATION_TIMEDOUT;
      break;
    case TFTP_ERR_NORESPONSE:
      result = CURLE_COULDNT_CONNECT;
      break;
    default:
      result = CURLE_ABORTED_BY_CALLBACK;
      break;
    }
  }
  else
    result = CURLE_OK;

  return result;
}

/**********************************************************
 *
 * tftp_state_machine
 *
 * The tftp state machine event dispatcher
 *
 **********************************************************/
static CURLcode tftp_state_machine(struct tftp_state_data *state,
                                   tftp_event_t event)
{
  CURLcode result = CURLE_OK;
  struct Curl_easy *data = state->data;

  switch(state->state) {
  case TFTP_STATE_START:
    DEBUGF(infof(data, "TFTP_STATE_START"));
    result = tftp_send_first(state, event);
    break;
  case TFTP_STATE_RX:
    DEBUGF(infof(data, "TFTP_STATE_RX"));
    result = tftp_rx(state, event);
    break;
  case TFTP_STATE_TX:
    DEBUGF(infof(data, "TFTP_STATE_TX"));
    result = tftp_tx(state, event);
    break;
  case TFTP_STATE_FIN:
    infof(data, "%s", "TFTP finished");
    break;
  default:
    DEBUGF(infof(data, "STATE: %d", state->state));
    failf(data, "%s", "Internal state machine error");
    result = CURLE_TFTP_ILLEGAL;
    break;
  }

  return result;
}

/**********************************************************
 *
 * tftp_disconnect
 *
 * The disconnect callback
 *
 **********************************************************/
static CURLcode tftp_disconnect(struct Curl_easy *data,
                                struct connectdata *conn, bool dead_connection)
{
  struct tftp_state_data *state = conn->proto.tftpc;
  (void) data;
  (void) dead_connection;

  /* done, free dynamically allocated pkt buffers */
  if(state) {
    Curl_safefree(state->rpacket.data);
    Curl_safefree(state->spacket.data);
    free(state);
  }

  return CURLE_OK;
}

/**********************************************************
 *
 * tftp_connect
 *
 * The connect callback
 *
 **********************************************************/
static CURLcode tftp_connect(struct Curl_easy *data, bool *done)
{
  struct tftp_state_data *state;
  int blksize;
  int need_blksize;
  struct connectdata *conn = data->conn;

  blksize = TFTP_BLKSIZE_DEFAULT;

  state = conn->proto.tftpc = calloc(1, sizeof(struct tftp_state_data));
  if(!state)
    return CURLE_OUT_OF_MEMORY;

  /* alloc pkt buffers based on specified blksize */
  if(data->set.tftp_blksize) {
    blksize = (int)data->set.tftp_blksize;
    if(blksize > TFTP_BLKSIZE_MAX || blksize < TFTP_BLKSIZE_MIN)
      return CURLE_TFTP_ILLEGAL;
  }

  need_blksize = blksize;
  /* default size is the fallback when no OACK is received */
  if(need_blksize < TFTP_BLKSIZE_DEFAULT)
    need_blksize = TFTP_BLKSIZE_DEFAULT;

  if(!state->rpacket.data) {
    state->rpacket.data = calloc(1, need_blksize + 2 + 2);

    if(!state->rpacket.data)
      return CURLE_OUT_OF_MEMORY;
  }

  if(!state->spacket.data) {
    state->spacket.data = calloc(1, need_blksize + 2 + 2);

    if(!state->spacket.data)
      return CURLE_OUT_OF_MEMORY;
  }

  /* we don't keep TFTP connections up basically because there's none or very
   * little gain for UDP */
  connclose(conn, "TFTP");

  state->data = data;
  state->sockfd = conn->sock[FIRSTSOCKET];
  state->state = TFTP_STATE_START;
  state->error = TFTP_ERR_NONE;
  state->blksize = TFTP_BLKSIZE_DEFAULT; /* Unless updated by OACK response */
  state->requested_blksize = blksize;

  ((struct sockaddr *)&state->local_addr)->sa_family =
    (CURL_SA_FAMILY_T)(conn->ip_addr->ai_family);

  tftp_set_timeouts(state);

  if(!conn->bits.bound) {
    /* If not already bound, bind to any interface, random UDP port. If it is
     * reused or a custom local port was desired, this has already been done!
     *
     * We once used the size of the local_addr struct as the third argument
     * for bind() to better work with IPv6 or whatever size the struct could
     * have, but we learned that at least Tru64, AIX and IRIX *requires* the
     * size of that argument to match the exact size of a 'sockaddr_in' struct
     * when running IPv4-only.
     *
     * Therefore we use the size from the address we connected to, which we
     * assume uses the same IP version and thus hopefully this works for both
     * IPv4 and IPv6...
     */
    int rc = bind(state->sockfd, (struct sockaddr *)&state->local_addr,
                  conn->ip_addr->ai_addrlen);
    if(rc) {
      char buffer[STRERROR_LEN];
      failf(data, "bind() failed; %s",
            Curl_strerror(SOCKERRNO, buffer, sizeof(buffer)));
      return CURLE_COULDNT_CONNECT;
    }
    conn->bits.bound = TRUE;
  }

  Curl_pgrsStartNow(data);

  *done = TRUE;

  return CURLE_OK;
}

/**********************************************************
 *
 * tftp_done
 *
 * The done callback
 *
 **********************************************************/
static CURLcode tftp_done(struct Curl_easy *data, CURLcode status,
                          bool premature)
{
  CURLcode result = CURLE_OK;
  struct connectdata *conn = data->conn;
  struct tftp_state_data *state = conn->proto.tftpc;

  (void)status; /* unused */
  (void)premature; /* not used */

  if(Curl_pgrsDone(data))
    return CURLE_ABORTED_BY_CALLBACK;

  /* If we have encountered an error */
  if(state)
    result = tftp_translate_code(state->error);

  return result;
}

/**********************************************************
 *
 * tftp_getsock
 *
 * The getsock callback
 *
 **********************************************************/
static int tftp_getsock(struct Curl_easy *data,
                        struct connectdata *conn, curl_socket_t *socks)
{
  (void)data;
  socks[0] = conn->sock[FIRSTSOCKET];
  return GETSOCK_READSOCK(0);
}

/**********************************************************
 *
 * tftp_receive_packet
 *
 * Called once select fires and data is ready on the socket
 *
 **********************************************************/
static CURLcode tftp_receive_packet(struct Curl_easy *data)
{
  struct Curl_sockaddr_storage fromaddr;
  curl_socklen_t        fromlen;
  CURLcode              result = CURLE_OK;
  struct connectdata *conn = data->conn;
  struct tftp_state_data *state = conn->proto.tftpc;
  struct SingleRequest  *k = &data->req;

  /* Receive the packet */
  fromlen = sizeof(fromaddr);
  state->rbytes = (int)recvfrom(state->sockfd,
                                (void *)state->rpacket.data,
                                state->blksize + 4,
                                0,
                                (struct sockaddr *)&fromaddr,
                                &fromlen);
  if(state->remote_addrlen == 0) {
    memcpy(&state->remote_addr, &fromaddr, fromlen);
    state->remote_addrlen = fromlen;
  }

  /* Sanity check packet length */
  if(state->rbytes < 4) {
    failf(data, "Received too short packet");
    /* Not a timeout, but how best to handle it? */
    state->event = TFTP_EVENT_TIMEOUT;
  }
  else {
    /* The event is given by the TFTP packet time */
    unsigned short event = getrpacketevent(&state->rpacket);
    state->event = (tftp_event_t)event;

    switch(state->event) {
    case TFTP_EVENT_DATA:
      /* Don't pass to the client empty or retransmitted packets */
      if(state->rbytes > 4 &&
         (NEXT_BLOCKNUM(state->block) == getrpacketblock(&state->rpacket))) {
        result = Curl_client_write(data, CLIENTWRITE_BODY,
                                   (char *)state->rpacket.data + 4,
                                   state->rbytes-4);
        if(result) {
          tftp_state_machine(state, TFTP_EVENT_ERROR);
          return result;
        }
        k->bytecount += state->rbytes-4;
        Curl_pgrsSetDownloadCounter(data, (curl_off_t) k->bytecount);
      }
      break;
    case TFTP_EVENT_ERROR:
    {
      unsigned short error = getrpacketblock(&state->rpacket);
      char *str = (char *)state->rpacket.data + 4;
      size_t strn = state->rbytes - 4;
      state->error = (tftp_error_t)error;
      if(tftp_strnlen(str, strn) < strn)
        infof(data, "TFTP error: %s", str);
      break;
    }
    case TFTP_EVENT_ACK:
      break;
    case TFTP_EVENT_OACK:
      result = tftp_parse_option_ack(state,
                                     (const char *)state->rpacket.data + 2,
                                     state->rbytes-2);
      if(result)
        return result;
      break;
    case TFTP_EVENT_RRQ:
    case TFTP_EVENT_WRQ:
    default:
      failf(data, "%s", "Internal error: Unexpected packet");
      break;
    }

    /* Update the progress meter */
    if(Curl_pgrsUpdate(data)) {
      tftp_state_machine(state, TFTP_EVENT_ERROR);
      return CURLE_ABORTED_BY_CALLBACK;
    }
  }
  return result;
}

/**********************************************************
 *
 * tftp_state_timeout
 *
 * Check if timeouts have been reached
 *
 **********************************************************/
static timediff_t tftp_state_timeout(struct Curl_easy *data,
                                     tftp_event_t *event)
{
  time_t current;
  struct connectdata *conn = data->conn;
  struct tftp_state_data *state = conn->proto.tftpc;
  timediff_t timeout_ms;

  if(event)
    *event = TFTP_EVENT_NONE;

  timeout_ms = Curl_timeleft(state->data, NULL,
                             (state->state == TFTP_STATE_START));
  if(timeout_ms < 0) {
    state->error = TFTP_ERR_TIMEOUT;
    state->state = TFTP_STATE_FIN;
    return 0;
  }
  time(&current);
  if(current > state->rx_time + state->retry_time) {
    if(event)
      *event = TFTP_EVENT_TIMEOUT;
    time(&state->rx_time); /* update even though we received nothing */
  }

  return timeout_ms;
}

/**********************************************************
 *
 * tftp_multi_statemach
 *
 * Handle single RX socket event and return
 *
 **********************************************************/
static CURLcode tftp_multi_statemach(struct Curl_easy *data, bool *done)
{
  tftp_event_t event;
  CURLcode result = CURLE_OK;
  struct connectdata *conn = data->conn;
  struct tftp_state_data *state = conn->proto.tftpc;
  timediff_t timeout_ms = tftp_state_timeout(data, &event);

  *done = FALSE;

  if(timeout_ms < 0) {
    failf(data, "TFTP response timeout");
    return CURLE_OPERATION_TIMEDOUT;
  }
  if(event != TFTP_EVENT_NONE) {
    result = tftp_state_machine(state, event);
    if(result)
      return result;
    *done = (state->state == TFTP_STATE_FIN) ? TRUE : FALSE;
    if(*done)
      /* Tell curl we're done */
      Curl_setup_transfer(data, -1, -1, FALSE, -1);
  }
  else {
    /* no timeouts to handle, check our socket */
    int rc = SOCKET_READABLE(state->sockfd, 0);

    if(rc == -1) {
      /* bail out */
      int error = SOCKERRNO;
      char buffer[STRERROR_LEN];
      failf(data, "%s", Curl_strerror(error, buffer, sizeof(buffer)));
      state->event = TFTP_EVENT_ERROR;
    }
    else if(rc) {
      result = tftp_receive_packet(data);
      if(result)
        return result;
      result = tftp_state_machine(state, state->event);
      if(result)
        return result;
      *done = (state->state == TFTP_STATE_FIN) ? TRUE : FALSE;
      if(*done)
        /* Tell curl we're done */
        Curl_setup_transfer(data, -1, -1, FALSE, -1);
    }
    /* if rc == 0, then select() timed out */
  }

  return result;
}

/**********************************************************
 *
 * tftp_doing
 *
 * Called from multi.c while DOing
 *
 **********************************************************/
static CURLcode tftp_doing(struct Curl_easy *data, bool *dophase_done)
{
  CURLcode result;
  result = tftp_multi_statemach(data, dophase_done);

  if(*dophase_done) {
    DEBUGF(infof(data, "DO phase is complete"));
  }
  else if(!result) {
    /* The multi code doesn't have this logic for the DOING state so we
       provide it for TFTP since it may do the entire transfer in this
       state. */
    if(Curl_pgrsUpdate(data))
      result = CURLE_ABORTED_BY_CALLBACK;
    else
      result = Curl_speedcheck(data, Curl_now());
  }
  return result;
}

/**********************************************************
 *
 * tftp_perform
 *
 * Entry point for transfer from tftp_do, starts state mach
 *
 **********************************************************/
static CURLcode tftp_perform(struct Curl_easy *data, bool *dophase_done)
{
  CURLcode result = CURLE_OK;
  struct connectdata *conn = data->conn;
  struct tftp_state_data *state = conn->proto.tftpc;

  *dophase_done = FALSE;

  result = tftp_state_machine(state, TFTP_EVENT_INIT);

  if((state->state == TFTP_STATE_FIN) || result)
    return result;

  tftp_multi_statemach(data, dophase_done);

  if(*dophase_done)
    DEBUGF(infof(data, "DO phase is complete"));

  return result;
}


/**********************************************************
 *
 * tftp_do
 *
 * The do callback
 *
 * This callback initiates the TFTP transfer
 *
 **********************************************************/

static CURLcode tftp_do(struct Curl_easy *data, bool *done)
{
  struct tftp_state_data *state;
  CURLcode result;
  struct connectdata *conn = data->conn;

  *done = FALSE;

  if(!conn->proto.tftpc) {
    result = tftp_connect(data, done);
    if(result)
      return result;
  }

  state = conn->proto.tftpc;
  if(!state)
    return CURLE_TFTP_ILLEGAL;

  result = tftp_perform(data, done);

  /* If tftp_perform() returned an error, use that for return code. If it
     was OK, see if tftp_translate_code() has an error. */
  if(!result)
    /* If we have encountered an internal tftp error, translate it. */
    result = tftp_translate_code(state->error);

  return result;
}

static CURLcode tftp_setup_connection(struct Curl_easy *data,
                                      struct connectdata *conn)
{
  char *type;

  conn->transport = TRNSPRT_UDP;

  /* TFTP URLs support an extension like ";mode=<typecode>" that
   * we'll try to get now! */
  type = strstr(data->state.up.path, ";mode=");

  if(!type)
    type = strstr(conn->host.rawalloc, ";mode=");

  if(type) {
    char command;
    *type = 0;                   /* it was in the middle of the hostname */
    command = Curl_raw_toupper(type[6]);

    switch(command) {
    case 'A': /* ASCII mode */
    case 'N': /* NETASCII mode */
      data->state.prefer_ascii = TRUE;
      break;

    case 'O': /* octet mode */
    case 'I': /* binary mode */
    default:
      /* switch off ASCII */
      data->state.prefer_ascii = FALSE;
      break;
    }
  }

  return CURLE_OK;
}
#endif
