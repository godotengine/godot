/*
 * Wslay - The WebSocket Library
 *
 * Copyright (c) 2011, 2012 Tatsuhiro Tsujikawa
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#ifndef WSLAY_EVENT_H
#define WSLAY_EVENT_H

#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <wslay/wslay.h>

struct wslay_stack;
struct wslay_queue;

struct wslay_event_byte_chunk {
  uint8_t *data;
  size_t data_length;
};

struct wslay_event_imsg {
  uint8_t fin;
  uint8_t rsv;
  uint8_t opcode;
  uint32_t utf8state;
  struct wslay_queue *chunks;
  size_t msg_length;
};

enum wslay_event_msg_type {
  WSLAY_NON_FRAGMENTED,
  WSLAY_FRAGMENTED
};

struct wslay_event_omsg {
  uint8_t fin;
  uint8_t opcode;
  uint8_t rsv;
  enum wslay_event_msg_type type;

  uint8_t *data;
  size_t data_length;

  union wslay_event_msg_source source;
  wslay_event_fragmented_msg_callback read_callback;
};

struct wslay_event_frame_user_data {
  wslay_event_context_ptr ctx;
  void *user_data;
};

enum wslay_event_close_status {
  WSLAY_CLOSE_RECEIVED = 1 << 0,
  WSLAY_CLOSE_QUEUED = 1 << 1,
  WSLAY_CLOSE_SENT = 1 << 2
};

enum wslay_event_config {
  WSLAY_CONFIG_NO_BUFFERING = 1 << 0
};

struct wslay_event_context {
  /* config status, bitwise OR of enum wslay_event_config values*/
  uint32_t config;
  /* maximum message length that can be received */
  uint64_t max_recv_msg_length;
  /* 1 if initialized for server, otherwise 0 */
  uint8_t server;
  /* bitwise OR of enum wslay_event_close_status values */
  uint8_t close_status;
  /* status code in received close control frame */
  uint16_t status_code_recv;
  /* status code in sent close control frame */
  uint16_t status_code_sent;
  wslay_frame_context_ptr frame_ctx;
  /* 1 if reading is enabled, otherwise 0. Upon receiving close
     control frame this value set to 0. If any errors in read
     operation will also set this value to 0. */
  uint8_t read_enabled;
  /* 1 if writing is enabled, otherwise 0 Upon completing sending
     close control frame, this value set to 0. If any errors in write
     opration will also set this value to 0. */
  uint8_t write_enabled;
  /* imsg buffer to allow interleaved control frame between
     non-control frames. */
  struct wslay_event_imsg imsgs[2];
  /* Pointer to imsgs to indicate current used buffer. */
  struct wslay_event_imsg *imsg;
  /* payload length of frame currently being received. */
  uint64_t ipayloadlen;
  /* next byte offset of payload currently being received. */
  uint64_t ipayloadoff;
  /* error value set by user callback */
  int error;
  /* Pointer to the message currently being sent. NULL if no message
     is currently sent. */
  struct wslay_event_omsg *omsg;
  /* Queue for non-control frames */
  struct wslay_queue/*<wslay_omsg*>*/ *send_queue;
  /* Queue for control frames */
  struct wslay_queue/*<wslay_omsg*>*/ *send_ctrl_queue;
  /* Size of send_queue + size of send_ctrl_queue */
  size_t queued_msg_count;
  /* The sum of message length in send_queue */
  size_t queued_msg_length;
  /* Buffer used for fragmented messages */
  uint8_t obuf[4096];
  uint8_t *obuflimit;
  uint8_t *obufmark;
  /* payload length of frame currently being sent. */
  uint64_t opayloadlen;
  /* next byte offset of payload currently being sent. */
  uint64_t opayloadoff;
  struct wslay_event_callbacks callbacks;
  struct wslay_event_frame_user_data frame_user_data;
  void *user_data;
  uint8_t allowed_rsv_bits;
};

#endif /* WSLAY_EVENT_H */
