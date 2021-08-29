/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 2020 - 2021, Daniel Stenberg, <daniel@haxx.se>, et al.
 * Copyright (C) 2019, Björn Stenberg, <bjorn@haxx.se>
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

#ifndef CURL_DISABLE_MQTT

#include "urldata.h"
#include <curl/curl.h>
#include "transfer.h"
#include "sendf.h"
#include "progress.h"
#include "mqtt.h"
#include "select.h"
#include "strdup.h"
#include "url.h"
#include "escape.h"
#include "warnless.h"
#include "curl_printf.h"
#include "curl_memory.h"
#include "multiif.h"
#include "rand.h"

/* The last #include file should be: */
#include "memdebug.h"

#define MQTT_MSG_CONNECT   0x10
#define MQTT_MSG_CONNACK   0x20
#define MQTT_MSG_PUBLISH   0x30
#define MQTT_MSG_SUBSCRIBE 0x82
#define MQTT_MSG_SUBACK    0x90
#define MQTT_MSG_DISCONNECT 0xe0

#define MQTT_CONNACK_LEN 2
#define MQTT_SUBACK_LEN 3
#define MQTT_CLIENTID_LEN 12 /* "curl0123abcd" */

/*
 * Forward declarations.
 */

static CURLcode mqtt_do(struct Curl_easy *data, bool *done);
static CURLcode mqtt_doing(struct Curl_easy *data, bool *done);
static int mqtt_getsock(struct Curl_easy *data, struct connectdata *conn,
                        curl_socket_t *sock);
static CURLcode mqtt_setup_conn(struct Curl_easy *data,
                                struct connectdata *conn);

/*
 * MQTT protocol handler.
 */

const struct Curl_handler Curl_handler_mqtt = {
  "MQTT",                             /* scheme */
  mqtt_setup_conn,                    /* setup_connection */
  mqtt_do,                            /* do_it */
  ZERO_NULL,                          /* done */
  ZERO_NULL,                          /* do_more */
  ZERO_NULL,                          /* connect_it */
  ZERO_NULL,                          /* connecting */
  mqtt_doing,                         /* doing */
  ZERO_NULL,                          /* proto_getsock */
  mqtt_getsock,                       /* doing_getsock */
  ZERO_NULL,                          /* domore_getsock */
  ZERO_NULL,                          /* perform_getsock */
  ZERO_NULL,                          /* disconnect */
  ZERO_NULL,                          /* readwrite */
  ZERO_NULL,                          /* connection_check */
  ZERO_NULL,                          /* attach connection */
  PORT_MQTT,                          /* defport */
  CURLPROTO_MQTT,                     /* protocol */
  CURLPROTO_MQTT,                     /* family */
  PROTOPT_NONE                        /* flags */
};

static CURLcode mqtt_setup_conn(struct Curl_easy *data,
                                struct connectdata *conn)
{
  /* allocate the HTTP-specific struct for the Curl_easy, only to survive
     during this request */
  struct MQTT *mq;
  (void)conn;
  DEBUGASSERT(data->req.p.mqtt == NULL);

  mq = calloc(1, sizeof(struct MQTT));
  if(!mq)
    return CURLE_OUT_OF_MEMORY;
  data->req.p.mqtt = mq;
  return CURLE_OK;
}

static CURLcode mqtt_send(struct Curl_easy *data,
                          char *buf, size_t len)
{
  CURLcode result = CURLE_OK;
  struct connectdata *conn = data->conn;
  curl_socket_t sockfd = conn->sock[FIRSTSOCKET];
  struct MQTT *mq = data->req.p.mqtt;
  ssize_t n;
  result = Curl_write(data, sockfd, buf, len, &n);
  if(!result)
    Curl_debug(data, CURLINFO_HEADER_OUT, buf, (size_t)n);
  if(len != (size_t)n) {
    size_t nsend = len - n;
    char *sendleftovers = Curl_memdup(&buf[n], nsend);
    if(!sendleftovers)
      return CURLE_OUT_OF_MEMORY;
    mq->sendleftovers = sendleftovers;
    mq->nsend = nsend;
  }
  else {
    mq->sendleftovers = NULL;
    mq->nsend = 0;
  }
  return result;
}

/* Generic function called by the multi interface to figure out what socket(s)
   to wait for and for what actions during the DOING and PROTOCONNECT
   states */
static int mqtt_getsock(struct Curl_easy *data,
                        struct connectdata *conn,
                        curl_socket_t *sock)
{
  (void)data;
  sock[0] = conn->sock[FIRSTSOCKET];
  return GETSOCK_READSOCK(FIRSTSOCKET);
}

static int mqtt_encode_len(char *buf, size_t len)
{
  unsigned char encoded;
  int i;

  for(i = 0; (len > 0) && (i<4); i++) {
    encoded = len % 0x80;
    len /= 0x80;
    if(len)
      encoded |= 0x80;
    buf[i] = encoded;
  }

  return i;
}

/* add the passwd to the CONNECT packet */
static int add_passwd(const char *passwd, const size_t plen,
                       char *pkt, const size_t start, int remain_pos)
{
  /* magic number that need to be set properly */
  const size_t conn_flags_pos = remain_pos + 8;
  if(plen > 0xffff)
    return 1;

  /* set password flag */
  pkt[conn_flags_pos] |= 0x40;

  /* length of password provided */
  pkt[start] = (char)((plen >> 8) & 0xFF);
  pkt[start + 1] = (char)(plen & 0xFF);
  memcpy(&pkt[start + 2], passwd, plen);
  return 0;
}

/* add user to the CONN packet */
static int add_user(const char *username, const size_t ulen,
                    unsigned char *pkt, const size_t start, int remain_pos)
{
  /* magic number that need to be set properly */
  const size_t conn_flags_pos = remain_pos + 8;
  if(ulen > 0xffff)
    return 1;

  /* set username flag */
  pkt[conn_flags_pos] |= 0x80;
  /* length of username provided */
  pkt[start] = (unsigned char)((ulen >> 8) & 0xFF);
  pkt[start + 1] = (unsigned char)(ulen & 0xFF);
  memcpy(&pkt[start + 2], username, ulen);
  return 0;
}

/* add client ID to the CONN packet */
static int add_client_id(const char *client_id, const size_t client_id_len,
                         char *pkt, const size_t start)
{
  if(client_id_len != MQTT_CLIENTID_LEN)
    return 1;
  pkt[start] = 0x00;
  pkt[start + 1] = MQTT_CLIENTID_LEN;
  memcpy(&pkt[start + 2], client_id, MQTT_CLIENTID_LEN);
  return 0;
}

/* Set initial values of CONN packet */
static int init_connpack(char *packet, char *remain, int remain_pos)
{
  /* Fixed header starts */
  /* packet type */
  packet[0] = MQTT_MSG_CONNECT;
  /* remaining length field */
  memcpy(&packet[1], remain, remain_pos);
  /* Fixed header ends */

  /* Variable header starts */
  /* protocol length */
  packet[remain_pos + 1] = 0x00;
  packet[remain_pos + 2] = 0x04;
  /* protocol name */
  packet[remain_pos + 3] = 'M';
  packet[remain_pos + 4] = 'Q';
  packet[remain_pos + 5] = 'T';
  packet[remain_pos + 6] = 'T';
  /* protocol level */
  packet[remain_pos + 7] = 0x04;
  /* CONNECT flag: CleanSession */
  packet[remain_pos + 8] = 0x02;
  /* keep-alive 0 = disabled */
  packet[remain_pos + 9] = 0x00;
  packet[remain_pos + 10] = 0x3c;
  /*end of variable header*/
  return remain_pos + 10;
}

static CURLcode mqtt_connect(struct Curl_easy *data)
{
  CURLcode result = CURLE_OK;
  int pos = 0;
  int rc = 0;
  /*remain length*/
  int remain_pos = 0;
  char remain[4] = {0};
  size_t packetlen = 0;
  size_t payloadlen = 0;
  size_t start_user = 0;
  size_t start_pwd = 0;
  char client_id[MQTT_CLIENTID_LEN + 1] = "curl";
  const size_t clen = strlen("curl");
  char *packet = NULL;

  /* extracting username from request */
  const char *username = data->state.aptr.user ?
    data->state.aptr.user : "";
  const size_t ulen = strlen(username);
  /* extracting password from request */
  const char *passwd = data->state.aptr.passwd ?
    data->state.aptr.passwd : "";
  const size_t plen = strlen(passwd);

  payloadlen = ulen + plen + MQTT_CLIENTID_LEN + 2;
  /* The plus 2 are for the MSB and LSB describing the length of the string to
   * be added on the payload. Refer to spec 1.5.2 and 1.5.4 */
  if(ulen)
    payloadlen += 2;
  if(plen)
    payloadlen += 2;

  /* getting how much occupy the remain length */
  remain_pos = mqtt_encode_len(remain, payloadlen + 10);

  /* 10 length of variable header and 1 the first byte of the fixed header */
  packetlen = payloadlen + 10 + remain_pos + 1;

  /* allocating packet */
  if(packetlen > 268435455)
    return CURLE_WEIRD_SERVER_REPLY;
  packet = malloc(packetlen);
  if(!packet)
    return CURLE_OUT_OF_MEMORY;
  memset(packet, 0, packetlen);

  /* set initial values for CONN pack */
  pos = init_connpack(packet, remain, remain_pos);

  result = Curl_rand_hex(data, (unsigned char *)&client_id[clen],
                         MQTT_CLIENTID_LEN - clen + 1);
  /* add client id */
  rc = add_client_id(client_id, strlen(client_id), packet, pos + 1);
  if(rc) {
    failf(data, "Client ID length mismatched: [%lu]", strlen(client_id));
    result = CURLE_WEIRD_SERVER_REPLY;
    goto end;
  }
  infof(data, "Using client id '%s'", client_id);

  /* position where starts the user payload */
  start_user = pos + 3 + MQTT_CLIENTID_LEN;
  /* position where starts the password payload */
  start_pwd = start_user + ulen;
  /* if user name was provided, add it to the packet */
  if(ulen) {
    start_pwd += 2;

    rc = add_user(username, ulen,
                  (unsigned char *)packet, start_user, remain_pos);
    if(rc) {
      failf(data, "Username is too large: [%lu]", ulen);
      result = CURLE_WEIRD_SERVER_REPLY;
      goto end;
    }
  }

  /* if passwd was provided, add it to the packet */
  if(plen) {
    rc = add_passwd(passwd, plen, packet, start_pwd, remain_pos);
    if(rc) {
      failf(data, "Password is too large: [%lu]", plen);
      result = CURLE_WEIRD_SERVER_REPLY;
      goto end;
    }
  }

  if(!result)
    result = mqtt_send(data, packet, packetlen);

end:
  if(packet)
    free(packet);
  Curl_safefree(data->state.aptr.user);
  Curl_safefree(data->state.aptr.passwd);
  return result;
}

static CURLcode mqtt_disconnect(struct Curl_easy *data)
{
  CURLcode result = CURLE_OK;
  result = mqtt_send(data, (char *)"\xe0\x00", 2);
  return result;
}

static CURLcode mqtt_verify_connack(struct Curl_easy *data)
{
  CURLcode result;
  struct connectdata *conn = data->conn;
  curl_socket_t sockfd = conn->sock[FIRSTSOCKET];
  unsigned char readbuf[MQTT_CONNACK_LEN];
  ssize_t nread;

  result = Curl_read(data, sockfd, (char *)readbuf, MQTT_CONNACK_LEN, &nread);
  if(result)
    goto fail;

  Curl_debug(data, CURLINFO_HEADER_IN, (char *)readbuf, (size_t)nread);

  /* fixme */
  if(nread < MQTT_CONNACK_LEN) {
    result = CURLE_WEIRD_SERVER_REPLY;
    goto fail;
  }

  /* verify CONNACK */
  if(readbuf[0] != 0x00 || readbuf[1] != 0x00) {
    failf(data, "Expected %02x%02x but got %02x%02x",
          0x00, 0x00, readbuf[0], readbuf[1]);
    result = CURLE_WEIRD_SERVER_REPLY;
  }

fail:
  return result;
}

static CURLcode mqtt_get_topic(struct Curl_easy *data,
                               char **topic, size_t *topiclen)
{
  char *path = data->state.up.path;
  if(strlen(path) > 1)
    return Curl_urldecode(data, path + 1, 0, topic, topiclen,
                          REJECT_NADA);
  failf(data, "No MQTT topic found. Forgot to URL encode it?");
  return CURLE_URL_MALFORMAT;
}

static CURLcode mqtt_subscribe(struct Curl_easy *data)
{
  CURLcode result = CURLE_OK;
  char *topic = NULL;
  size_t topiclen;
  unsigned char *packet = NULL;
  size_t packetlen;
  char encodedsize[4];
  size_t n;
  struct connectdata *conn = data->conn;

  result = mqtt_get_topic(data, &topic, &topiclen);
  if(result)
    goto fail;

  conn->proto.mqtt.packetid++;

  packetlen = topiclen + 5; /* packetid + topic (has a two byte length field)
                               + 2 bytes topic length + QoS byte */
  n = mqtt_encode_len((char *)encodedsize, packetlen);
  packetlen += n + 1; /* add one for the control packet type byte */

  packet = malloc(packetlen);
  if(!packet) {
    result = CURLE_OUT_OF_MEMORY;
    goto fail;
  }

  packet[0] = MQTT_MSG_SUBSCRIBE;
  memcpy(&packet[1], encodedsize, n);
  packet[1 + n] = (conn->proto.mqtt.packetid >> 8) & 0xff;
  packet[2 + n] = conn->proto.mqtt.packetid & 0xff;
  packet[3 + n] = (topiclen >> 8) & 0xff;
  packet[4 + n ] = topiclen & 0xff;
  memcpy(&packet[5 + n], topic, topiclen);
  packet[5 + n + topiclen] = 0; /* QoS zero */

  result = mqtt_send(data, (char *)packet, packetlen);

fail:
  free(topic);
  free(packet);
  return result;
}

/*
 * Called when the first byte was already read.
 */
static CURLcode mqtt_verify_suback(struct Curl_easy *data)
{
  CURLcode result;
  struct connectdata *conn = data->conn;
  curl_socket_t sockfd = conn->sock[FIRSTSOCKET];
  unsigned char readbuf[MQTT_SUBACK_LEN];
  ssize_t nread;
  struct mqtt_conn *mqtt = &conn->proto.mqtt;

  result = Curl_read(data, sockfd, (char *)readbuf, MQTT_SUBACK_LEN, &nread);
  if(result)
    goto fail;

  Curl_debug(data, CURLINFO_HEADER_IN, (char *)readbuf, (size_t)nread);

  /* fixme */
  if(nread < MQTT_SUBACK_LEN) {
    result = CURLE_WEIRD_SERVER_REPLY;
    goto fail;
  }

  /* verify SUBACK */
  if(readbuf[0] != ((mqtt->packetid >> 8) & 0xff) ||
     readbuf[1] != (mqtt->packetid & 0xff) ||
     readbuf[2] != 0x00)
    result = CURLE_WEIRD_SERVER_REPLY;

fail:
  return result;
}

static CURLcode mqtt_publish(struct Curl_easy *data)
{
  CURLcode result;
  char *payload = data->set.postfields;
  size_t payloadlen;
  char *topic = NULL;
  size_t topiclen;
  unsigned char *pkt = NULL;
  size_t i = 0;
  size_t remaininglength;
  size_t encodelen;
  char encodedbytes[4];
  curl_off_t postfieldsize = data->set.postfieldsize;

  if(!payload)
    return CURLE_BAD_FUNCTION_ARGUMENT;
  if(postfieldsize < 0)
    payloadlen = strlen(payload);
  else
    payloadlen = (size_t)postfieldsize;

  result = mqtt_get_topic(data, &topic, &topiclen);
  if(result)
    goto fail;

  remaininglength = payloadlen + 2 + topiclen;
  encodelen = mqtt_encode_len(encodedbytes, remaininglength);

  /* add the control byte and the encoded remaining length */
  pkt = malloc(remaininglength + 1 + encodelen);
  if(!pkt) {
    result = CURLE_OUT_OF_MEMORY;
    goto fail;
  }

  /* assemble packet */
  pkt[i++] = MQTT_MSG_PUBLISH;
  memcpy(&pkt[i], encodedbytes, encodelen);
  i += encodelen;
  pkt[i++] = (topiclen >> 8) & 0xff;
  pkt[i++] = (topiclen & 0xff);
  memcpy(&pkt[i], topic, topiclen);
  i += topiclen;
  memcpy(&pkt[i], payload, payloadlen);
  i += payloadlen;
  result = mqtt_send(data, (char *)pkt, i);

fail:
  free(pkt);
  free(topic);
  return result;
}

static size_t mqtt_decode_len(unsigned char *buf,
                              size_t buflen, size_t *lenbytes)
{
  size_t len = 0;
  size_t mult = 1;
  size_t i;
  unsigned char encoded = 128;

  for(i = 0; (i < buflen) && (encoded & 128); i++) {
    encoded = buf[i];
    len += (encoded & 127) * mult;
    mult *= 128;
  }

  if(lenbytes)
    *lenbytes = i;

  return len;
}

#ifdef CURLDEBUG
static const char *statenames[]={
  "MQTT_FIRST",
  "MQTT_REMAINING_LENGTH",
  "MQTT_CONNACK",
  "MQTT_SUBACK",
  "MQTT_SUBACK_COMING",
  "MQTT_PUBWAIT",
  "MQTT_PUB_REMAIN",

  "NOT A STATE"
};
#endif

/* The only way to change state */
static void mqstate(struct Curl_easy *data,
                    enum mqttstate state,
                    enum mqttstate nextstate) /* used if state == FIRST */
{
  struct connectdata *conn = data->conn;
  struct mqtt_conn *mqtt = &conn->proto.mqtt;
#ifdef CURLDEBUG
  infof(data, "%s (from %s) (next is %s)",
        statenames[state],
        statenames[mqtt->state],
        (state == MQTT_FIRST)? statenames[nextstate] : "");
#endif
  mqtt->state = state;
  if(state == MQTT_FIRST)
    mqtt->nextstate = nextstate;
}


/* for the publish packet */
#define MQTT_HEADER_LEN 5    /* max 5 bytes */

static CURLcode mqtt_read_publish(struct Curl_easy *data, bool *done)
{
  CURLcode result = CURLE_OK;
  struct connectdata *conn = data->conn;
  curl_socket_t sockfd = conn->sock[FIRSTSOCKET];
  ssize_t nread;
  unsigned char *pkt = (unsigned char *)data->state.buffer;
  size_t remlen;
  struct mqtt_conn *mqtt = &conn->proto.mqtt;
  struct MQTT *mq = data->req.p.mqtt;
  unsigned char packet;

  switch(mqtt->state) {
  MQTT_SUBACK_COMING:
  case MQTT_SUBACK_COMING:
    result = mqtt_verify_suback(data);
    if(result)
      break;

    mqstate(data, MQTT_FIRST, MQTT_PUBWAIT);
    break;

  case MQTT_SUBACK:
  case MQTT_PUBWAIT:
    /* we are expecting PUBLISH or SUBACK */
    packet = mq->firstbyte & 0xf0;
    if(packet == MQTT_MSG_PUBLISH)
      mqstate(data, MQTT_PUB_REMAIN, MQTT_NOSTATE);
    else if(packet == MQTT_MSG_SUBACK) {
      mqstate(data, MQTT_SUBACK_COMING, MQTT_NOSTATE);
      goto MQTT_SUBACK_COMING;
    }
    else if(packet == MQTT_MSG_DISCONNECT) {
      infof(data, "Got DISCONNECT");
      *done = TRUE;
      goto end;
    }
    else {
      result = CURLE_WEIRD_SERVER_REPLY;
      goto end;
    }

    /* -- switched state -- */
    remlen = mq->remaining_length;
    infof(data, "Remaining length: %zd bytes", remlen);
    if(data->set.max_filesize &&
       (curl_off_t)remlen > data->set.max_filesize) {
      failf(data, "Maximum file size exceeded");
      result = CURLE_FILESIZE_EXCEEDED;
      goto end;
    }
    Curl_pgrsSetDownloadSize(data, remlen);
    data->req.bytecount = 0;
    data->req.size = remlen;
    mq->npacket = remlen; /* get this many bytes */
    /* FALLTHROUGH */
  case MQTT_PUB_REMAIN: {
    /* read rest of packet, but no more. Cap to buffer size */
    struct SingleRequest *k = &data->req;
    size_t rest = mq->npacket;
    if(rest > (size_t)data->set.buffer_size)
      rest = (size_t)data->set.buffer_size;
    result = Curl_read(data, sockfd, (char *)pkt, rest, &nread);
    if(result) {
      if(CURLE_AGAIN == result) {
        infof(data, "EEEE AAAAGAIN");
      }
      goto end;
    }
    if(!nread) {
      infof(data, "server disconnected");
      result = CURLE_PARTIAL_FILE;
      goto end;
    }
    Curl_debug(data, CURLINFO_DATA_IN, (char *)pkt, (size_t)nread);

    mq->npacket -= nread;
    k->bytecount += nread;
    Curl_pgrsSetDownloadCounter(data, k->bytecount);

    /* if QoS is set, message contains packet id */

    result = Curl_client_write(data, CLIENTWRITE_BODY, (char *)pkt, nread);
    if(result)
      goto end;

    if(!mq->npacket)
      /* no more PUBLISH payload, back to subscribe wait state */
      mqstate(data, MQTT_FIRST, MQTT_PUBWAIT);
    break;
  }
  default:
    DEBUGASSERT(NULL); /* illegal state */
    result = CURLE_WEIRD_SERVER_REPLY;
    goto end;
  }
  end:
  return result;
}

static CURLcode mqtt_do(struct Curl_easy *data, bool *done)
{
  CURLcode result = CURLE_OK;
  *done = FALSE; /* unconditionally */

  result = mqtt_connect(data);
  if(result) {
    failf(data, "Error %d sending MQTT CONN request", result);
    return result;
  }
  mqstate(data, MQTT_FIRST, MQTT_CONNACK);
  return CURLE_OK;
}

static CURLcode mqtt_doing(struct Curl_easy *data, bool *done)
{
  CURLcode result = CURLE_OK;
  struct connectdata *conn = data->conn;
  struct mqtt_conn *mqtt = &conn->proto.mqtt;
  struct MQTT *mq = data->req.p.mqtt;
  ssize_t nread;
  curl_socket_t sockfd = conn->sock[FIRSTSOCKET];
  unsigned char *pkt = (unsigned char *)data->state.buffer;
  unsigned char byte;

  *done = FALSE;

  if(mq->nsend) {
    /* send the remainder of an outgoing packet */
    char *ptr = mq->sendleftovers;
    result = mqtt_send(data, mq->sendleftovers, mq->nsend);
    free(ptr);
    if(result)
      return result;
  }

  infof(data, "mqtt_doing: state [%d]", (int) mqtt->state);
  switch(mqtt->state) {
  case MQTT_FIRST:
    /* Read the initial byte only */
    result = Curl_read(data, sockfd, (char *)&mq->firstbyte, 1, &nread);
    if(!nread)
      break;
    Curl_debug(data, CURLINFO_HEADER_IN, (char *)&mq->firstbyte, 1);
    /* remember the first byte */
    mq->npacket = 0;
    mqstate(data, MQTT_REMAINING_LENGTH, MQTT_NOSTATE);
    /* FALLTHROUGH */
  case MQTT_REMAINING_LENGTH:
    do {
      result = Curl_read(data, sockfd, (char *)&byte, 1, &nread);
      if(!nread)
        break;
      Curl_debug(data, CURLINFO_HEADER_IN, (char *)&byte, 1);
      pkt[mq->npacket++] = byte;
    } while((byte & 0x80) && (mq->npacket < 4));
    if(nread && (byte & 0x80))
      /* MQTT supports up to 127 * 128^0 + 127 * 128^1 + 127 * 128^2 +
         127 * 128^3 bytes. server tried to send more */
      result = CURLE_WEIRD_SERVER_REPLY;
    if(result)
      break;
    mq->remaining_length = mqtt_decode_len(&pkt[0], mq->npacket, NULL);
    mq->npacket = 0;
    if(mq->remaining_length) {
      mqstate(data, mqtt->nextstate, MQTT_NOSTATE);
      break;
    }
    mqstate(data, MQTT_FIRST, MQTT_FIRST);

    if(mq->firstbyte == MQTT_MSG_DISCONNECT) {
      infof(data, "Got DISCONNECT");
      *done = TRUE;
    }
    break;
  case MQTT_CONNACK:
    result = mqtt_verify_connack(data);
    if(result)
      break;

    if(data->state.httpreq == HTTPREQ_POST) {
      result = mqtt_publish(data);
      if(!result) {
        result = mqtt_disconnect(data);
        *done = TRUE;
      }
      mqtt->nextstate = MQTT_FIRST;
    }
    else {
      result = mqtt_subscribe(data);
      if(!result) {
        mqstate(data, MQTT_FIRST, MQTT_SUBACK);
      }
    }
    break;

  case MQTT_SUBACK:
  case MQTT_PUBWAIT:
  case MQTT_PUB_REMAIN:
    result = mqtt_read_publish(data, done);
    break;

  default:
    failf(data, "State not handled yet");
    *done = TRUE;
    break;
  }

  if(result == CURLE_AGAIN)
    result = CURLE_OK;
  return result;
}

#endif /* CURL_DISABLE_MQTT */
