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
#include "wslay_event.h"

#include <string.h>
#include <assert.h>
#include <stdio.h>

#include "wslay_queue.h"
#include "wslay_frame.h"
#include "wslay_net.h"
/* Start of utf8 dfa */
/* Copyright (c) 2008-2010 Bjoern Hoehrmann <bjoern@hoehrmann.de>
 * See http://bjoern.hoehrmann.de/utf-8/decoder/dfa/ for details.
 *
 * Copyright (c) 2008-2009 Bjoern Hoehrmann <bjoern@hoehrmann.de>
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#define UTF8_ACCEPT 0
#define UTF8_REJECT 12

static const uint8_t utf8d[] = {
  /*
   * The first part of the table maps bytes to character classes that
   * to reduce the size of the transition table and create bitmasks.
   */
   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
   1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,  9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,
   7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,  7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
   8,8,2,2,2,2,2,2,2,2,2,2,2,2,2,2,  2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
  10,3,3,3,3,3,3,3,3,3,3,3,3,4,3,3, 11,6,6,6,5,8,8,8,8,8,8,8,8,8,8,8,

   /*
    * The second part is a transition table that maps a combination
    * of a state of the automaton and a character class to a state.
    */
   0,12,24,36,60,96,84,12,12,12,48,72, 12,12,12,12,12,12,12,12,12,12,12,12,
  12, 0,12,12,12,12,12, 0,12, 0,12,12, 12,24,12,12,12,12,12,24,12,24,12,12,
  12,12,12,12,12,12,12,24,12,12,12,12, 12,24,12,12,12,12,12,12,12,24,12,12,
  12,12,12,12,12,12,12,36,12,36,12,12, 12,36,12,12,12,12,12,36,12,36,12,12,
  12,36,12,12,12,12,12,12,12,12,12,12,
};

static uint32_t
decode(uint32_t* state, uint32_t* codep, uint32_t byte) {
  uint32_t type = utf8d[byte];

  *codep = (*state != UTF8_ACCEPT) ?
    (byte & 0x3fu) | (*codep << 6) :
    (0xff >> type) & (byte);

  *state = utf8d[256 + *state + type];
  return *state;
}

/* End of utf8 dfa */

static ssize_t wslay_event_frame_recv_callback(uint8_t *buf, size_t len,
                                               int flags, void *user_data)
{
  struct wslay_event_frame_user_data *e =
    (struct wslay_event_frame_user_data*)user_data;
  return e->ctx->callbacks.recv_callback(e->ctx, buf, len, flags, e->user_data);
}

static ssize_t wslay_event_frame_send_callback(const uint8_t *data, size_t len,
                                               int flags, void *user_data)
{
  struct wslay_event_frame_user_data *e =
    (struct wslay_event_frame_user_data*)user_data;
  return e->ctx->callbacks.send_callback(e->ctx, data, len, flags,
                                         e->user_data);
}

static int wslay_event_frame_genmask_callback(uint8_t *buf, size_t len,
                                              void *user_data)
{
  struct wslay_event_frame_user_data *e =
    (struct wslay_event_frame_user_data*)user_data;
  return e->ctx->callbacks.genmask_callback(e->ctx, buf, len, e->user_data);
}

static int wslay_event_byte_chunk_init
(struct wslay_event_byte_chunk **chunk, size_t len)
{
  *chunk = (struct wslay_event_byte_chunk*)malloc
    (sizeof(struct wslay_event_byte_chunk));
  if(*chunk == NULL) {
    return WSLAY_ERR_NOMEM;
  }
  memset(*chunk, 0, sizeof(struct wslay_event_byte_chunk));
  if(len) {
    (*chunk)->data = (uint8_t*)malloc(len);
    if((*chunk)->data == NULL) {
      free(*chunk);
      return WSLAY_ERR_NOMEM;
    }
    (*chunk)->data_length = len;
  }
  return 0;
}

static void wslay_event_byte_chunk_free(struct wslay_event_byte_chunk *c)
{
  if(!c) {
    return;
  }
  free(c->data);
  free(c);
}

static void wslay_event_byte_chunk_copy(struct wslay_event_byte_chunk *c,
                                        size_t off,
                                        const uint8_t *data, size_t data_length)
{
  memcpy(c->data+off, data, data_length);
}

static void wslay_event_imsg_set(struct wslay_event_imsg *m,
                                 uint8_t fin, uint8_t rsv, uint8_t opcode)
{
  m->fin = fin;
  m->rsv = rsv;
  m->opcode = opcode;
  m->msg_length = 0;
}

static void wslay_event_imsg_chunks_free(struct wslay_event_imsg *m)
{
  if(!m->chunks) {
    return;
  }
  while(!wslay_queue_empty(m->chunks)) {
    wslay_event_byte_chunk_free(wslay_queue_top(m->chunks));
    wslay_queue_pop(m->chunks);
  }
}

static void wslay_event_imsg_reset(struct wslay_event_imsg *m)
{
  m->opcode = 0xffu;
  m->utf8state = UTF8_ACCEPT;
  wslay_event_imsg_chunks_free(m);
}

static int wslay_event_imsg_append_chunk(struct wslay_event_imsg *m, size_t len)
{
  if(len == 0) {
    return 0;
  } else {
    int r;
    struct wslay_event_byte_chunk *chunk;
    if((r = wslay_event_byte_chunk_init(&chunk, len)) != 0) {
      return r;
    }
    if((r = wslay_queue_push(m->chunks, chunk)) != 0) {
      return r;
    }
    m->msg_length += len;
    return 0;
  }
}

static int wslay_event_omsg_non_fragmented_init
(struct wslay_event_omsg **m, uint8_t opcode, uint8_t rsv,
 const uint8_t *msg, size_t msg_length)
{
  *m = (struct wslay_event_omsg*)malloc(sizeof(struct wslay_event_omsg));
  if(!*m) {
    return WSLAY_ERR_NOMEM;
  }
  memset(*m, 0, sizeof(struct wslay_event_omsg));
  (*m)->fin = 1;
  (*m)->opcode = opcode;
  (*m)->rsv = rsv;
  (*m)->type = WSLAY_NON_FRAGMENTED;
  if(msg_length) {
    (*m)->data = (uint8_t*)malloc(msg_length);
    if(!(*m)->data) {
      free(*m);
      return WSLAY_ERR_NOMEM;
    }
    memcpy((*m)->data, msg, msg_length);
    (*m)->data_length = msg_length;
  }
  return 0;
}

static int wslay_event_omsg_fragmented_init
(struct wslay_event_omsg **m, uint8_t opcode, uint8_t rsv,
 const union wslay_event_msg_source source,
 wslay_event_fragmented_msg_callback read_callback)
{
  *m = (struct wslay_event_omsg*)malloc(sizeof(struct wslay_event_omsg));
  if(!*m) {
    return WSLAY_ERR_NOMEM;
  }
  memset(*m, 0, sizeof(struct wslay_event_omsg));
  (*m)->opcode = opcode;
  (*m)->rsv = rsv;
  (*m)->type = WSLAY_FRAGMENTED;
  (*m)->source = source;
  (*m)->read_callback = read_callback;
  return 0;
}

static void wslay_event_omsg_free(struct wslay_event_omsg *m)
{
  if(!m) {
    return;
  }
  free(m->data);
  free(m);
}

static uint8_t* wslay_event_flatten_queue(struct wslay_queue *queue, size_t len)
{
  if(len == 0) {
    return NULL;
  } else {
    size_t off = 0;
    uint8_t *buf = (uint8_t*)malloc(len);
    if(!buf) {
      return NULL;
    }
    while(!wslay_queue_empty(queue)) {
      struct wslay_event_byte_chunk *chunk = wslay_queue_top(queue);
      memcpy(buf+off, chunk->data, chunk->data_length);
      off += chunk->data_length;
      wslay_event_byte_chunk_free(chunk);
      wslay_queue_pop(queue);
      assert(off <= len);
    }
    assert(len == off);
    return buf;
  }
}

static int wslay_event_is_msg_queueable(wslay_event_context_ptr ctx)
{
  return ctx->write_enabled && (ctx->close_status & WSLAY_CLOSE_QUEUED) == 0;
}

int wslay_event_queue_close(wslay_event_context_ptr ctx, uint16_t status_code,
                            const uint8_t *reason, size_t reason_length)
{
  if(!wslay_event_is_msg_queueable(ctx)) {
    return WSLAY_ERR_NO_MORE_MSG;
  } else if(reason_length > 123) {
    return WSLAY_ERR_INVALID_ARGUMENT;
  } else {
    uint8_t msg[128];
    size_t msg_length;
    struct wslay_event_msg arg;
    uint16_t ncode;
    int r;
    if(status_code == 0) {
      msg_length = 0;
    } else {
      ncode = htons(status_code);
      memcpy(msg, &ncode, 2);
      if(reason_length) {
        memcpy(msg+2, reason, reason_length);
      }
      msg_length = reason_length+2;
    }
    arg.opcode = WSLAY_CONNECTION_CLOSE;
    arg.msg = msg;
    arg.msg_length = msg_length;
    r = wslay_event_queue_msg(ctx, &arg);
    if(r == 0) {
      ctx->close_status |= WSLAY_CLOSE_QUEUED;
    }
    return r;
  }
}

static int wslay_event_queue_close_wrapper
(wslay_event_context_ptr ctx, uint16_t status_code,
 const uint8_t *reason, size_t reason_length)
{
  int r;
  ctx->read_enabled = 0;
  if((r = wslay_event_queue_close(ctx, status_code, reason, reason_length)) &&
     r != WSLAY_ERR_NO_MORE_MSG) {
    return r;
  }
  return 0;
}

static int wslay_event_verify_rsv_bits(wslay_event_context_ptr ctx, uint8_t rsv)
{
  return ((rsv & ~ctx->allowed_rsv_bits) == 0);
}

int wslay_event_queue_msg(wslay_event_context_ptr ctx,
                          const struct wslay_event_msg *arg)
{
  return wslay_event_queue_msg_ex(ctx, arg, WSLAY_RSV_NONE);
}

int wslay_event_queue_msg_ex(wslay_event_context_ptr ctx,
                              const struct wslay_event_msg *arg, uint8_t rsv)
{
  int r;
  struct wslay_event_omsg *omsg;
  if(!wslay_event_is_msg_queueable(ctx)) {
    return WSLAY_ERR_NO_MORE_MSG;
  }
  /* RSV1 is not allowed for control frames */
  if((wslay_is_ctrl_frame(arg->opcode) &&
      (arg->msg_length > 125 || wslay_get_rsv1(rsv)))
        || !wslay_event_verify_rsv_bits(ctx, rsv)) {
    return WSLAY_ERR_INVALID_ARGUMENT;
  }
  if((r = wslay_event_omsg_non_fragmented_init
      (&omsg, arg->opcode, rsv, arg->msg, arg->msg_length)) != 0) {
    return r;
  }
  if(wslay_is_ctrl_frame(arg->opcode)) {
    if((r = wslay_queue_push(ctx->send_ctrl_queue, omsg)) != 0) {
      return r;
    }
  } else {
    if((r = wslay_queue_push(ctx->send_queue, omsg)) != 0) {
      return r;
    }
  }
  ++ctx->queued_msg_count;
  ctx->queued_msg_length += arg->msg_length;
  return 0;
}

int wslay_event_queue_fragmented_msg
(wslay_event_context_ptr ctx, const struct wslay_event_fragmented_msg *arg)
{
  return wslay_event_queue_fragmented_msg_ex(ctx, arg, WSLAY_RSV_NONE);
}

int wslay_event_queue_fragmented_msg_ex(wslay_event_context_ptr ctx,
    const struct wslay_event_fragmented_msg *arg, uint8_t rsv)
{
  int r;
  struct wslay_event_omsg *omsg;
  if(!wslay_event_is_msg_queueable(ctx)) {
    return WSLAY_ERR_NO_MORE_MSG;
  }
  if(wslay_is_ctrl_frame(arg->opcode) ||
     !wslay_event_verify_rsv_bits(ctx, rsv)) {
    return WSLAY_ERR_INVALID_ARGUMENT;
  }
  if((r = wslay_event_omsg_fragmented_init
      (&omsg, arg->opcode, rsv, arg->source, arg->read_callback)) != 0) {
    return r;
  }
  if((r = wslay_queue_push(ctx->send_queue, omsg)) != 0) {
    return r;
  }
  ++ctx->queued_msg_count;
  return 0;
}

void wslay_event_config_set_callbacks
(wslay_event_context_ptr ctx, const struct wslay_event_callbacks *callbacks)
{
  ctx->callbacks = *callbacks;
}

static int wslay_event_context_init
(wslay_event_context_ptr *ctx,
 const struct wslay_event_callbacks *callbacks,
 void *user_data)
{
  int i, r;
  struct wslay_frame_callbacks frame_callbacks = {
    wslay_event_frame_send_callback,
    wslay_event_frame_recv_callback,
    wslay_event_frame_genmask_callback
  };
  *ctx = (wslay_event_context_ptr)malloc(sizeof(struct wslay_event_context));
  if(!*ctx) {
    return WSLAY_ERR_NOMEM;
  }
  memset(*ctx, 0, sizeof(struct wslay_event_context));
  wslay_event_config_set_callbacks(*ctx, callbacks);
  (*ctx)->user_data = user_data;
  (*ctx)->frame_user_data.ctx = *ctx;
  (*ctx)->frame_user_data.user_data = user_data;
  if((r = wslay_frame_context_init(&(*ctx)->frame_ctx, &frame_callbacks,
                                   &(*ctx)->frame_user_data)) != 0) {
    wslay_event_context_free(*ctx);
    return r;
  }
  (*ctx)->read_enabled = (*ctx)->write_enabled = 1;
  (*ctx)->send_queue = wslay_queue_new();
  if(!(*ctx)->send_queue) {
    wslay_event_context_free(*ctx);
    return WSLAY_ERR_NOMEM;
  }
  (*ctx)->send_ctrl_queue = wslay_queue_new();
  if(!(*ctx)->send_ctrl_queue) {
    wslay_event_context_free(*ctx);
    return WSLAY_ERR_NOMEM;
  }
  (*ctx)->queued_msg_count = 0;
  (*ctx)->queued_msg_length = 0;
  for(i = 0; i < 2; ++i) {
    wslay_event_imsg_reset(&(*ctx)->imsgs[i]);
    (*ctx)->imsgs[i].chunks = wslay_queue_new();
    if(!(*ctx)->imsgs[i].chunks) {
      wslay_event_context_free(*ctx);
      return WSLAY_ERR_NOMEM;
    }
  }
  (*ctx)->imsg = &(*ctx)->imsgs[0];
  (*ctx)->obufmark = (*ctx)->obuflimit = (*ctx)->obuf;
  (*ctx)->status_code_sent = WSLAY_CODE_ABNORMAL_CLOSURE;
  (*ctx)->status_code_recv = WSLAY_CODE_ABNORMAL_CLOSURE;
  (*ctx)->max_recv_msg_length = (1u << 31)-1;
  return 0;
}

int wslay_event_context_server_init
(wslay_event_context_ptr *ctx,
 const struct wslay_event_callbacks *callbacks,
 void *user_data)
{
  int r;
  if((r = wslay_event_context_init(ctx, callbacks, user_data)) != 0) {
    return r;
  }
  (*ctx)->server = 1;
  return 0;
}

int wslay_event_context_client_init
(wslay_event_context_ptr *ctx,
 const struct wslay_event_callbacks *callbacks,
 void *user_data)
{
  int r;
  if((r = wslay_event_context_init(ctx, callbacks, user_data)) != 0) {
    return r;
  }
  (*ctx)->server = 0;
  return 0;
}

void wslay_event_context_free(wslay_event_context_ptr ctx)
{
  int i;
  if(!ctx) {
    return;
  }
  for(i = 0; i < 2; ++i) {
    wslay_event_imsg_chunks_free(&ctx->imsgs[i]);
    wslay_queue_free(ctx->imsgs[i].chunks);
  }
  if(ctx->send_queue) {
    while(!wslay_queue_empty(ctx->send_queue)) {
      wslay_event_omsg_free(wslay_queue_top(ctx->send_queue));
      wslay_queue_pop(ctx->send_queue);
    }
    wslay_queue_free(ctx->send_queue);
  }
  if(ctx->send_ctrl_queue) {
    while(!wslay_queue_empty(ctx->send_ctrl_queue)) {
      wslay_event_omsg_free(wslay_queue_top(ctx->send_ctrl_queue));
      wslay_queue_pop(ctx->send_ctrl_queue);
    }
    wslay_queue_free(ctx->send_ctrl_queue);
  }
  wslay_frame_context_free(ctx->frame_ctx);
  wslay_event_omsg_free(ctx->omsg);
  free(ctx);
}

static void wslay_event_call_on_frame_recv_start_callback
(wslay_event_context_ptr ctx, const struct wslay_frame_iocb *iocb)
{
  if(ctx->callbacks.on_frame_recv_start_callback) {
    struct wslay_event_on_frame_recv_start_arg arg;
    arg.fin = iocb->fin;
    arg.rsv = iocb->rsv;
    arg.opcode = iocb->opcode;
    arg.payload_length = iocb->payload_length;
    ctx->callbacks.on_frame_recv_start_callback(ctx, &arg, ctx->user_data);
  }
}

static void wslay_event_call_on_frame_recv_chunk_callback
(wslay_event_context_ptr ctx, const struct wslay_frame_iocb *iocb)
{
  if(ctx->callbacks.on_frame_recv_chunk_callback) {
    struct wslay_event_on_frame_recv_chunk_arg arg;
    arg.data = iocb->data;
    arg.data_length = iocb->data_length;
    ctx->callbacks.on_frame_recv_chunk_callback(ctx, &arg, ctx->user_data);
  }
}

static void wslay_event_call_on_frame_recv_end_callback
(wslay_event_context_ptr ctx)
{
  if(ctx->callbacks.on_frame_recv_end_callback) {
    ctx->callbacks.on_frame_recv_end_callback(ctx, ctx->user_data);
  }
}

static int wslay_event_is_valid_status_code(uint16_t status_code)
{
  return (1000 <= status_code && status_code <= 1011 &&
          status_code != 1004 && status_code != 1005 && status_code != 1006) ||
    (3000 <= status_code && status_code <= 4999);
}

static int wslay_event_config_get_no_buffering(wslay_event_context_ptr ctx)
{
  return (ctx->config & WSLAY_CONFIG_NO_BUFFERING) > 0;
}

int wslay_event_recv(wslay_event_context_ptr ctx)
{
  struct wslay_frame_iocb iocb;
  ssize_t r;
  while(ctx->read_enabled) {
    memset(&iocb, 0, sizeof(iocb));
    r = wslay_frame_recv(ctx->frame_ctx, &iocb);
    if(r >= 0) {
      int new_frame = 0;
      /* RSV1 is not allowed on control and continuation frames */
      if((!wslay_event_verify_rsv_bits(ctx, iocb.rsv)) ||
          (wslay_get_rsv1(iocb.rsv) && (wslay_is_ctrl_frame(iocb.opcode) ||
             iocb.opcode == WSLAY_CONTINUATION_FRAME)) ||
               (ctx->server && !iocb.mask) || (!ctx->server && iocb.mask)) {
        if((r = wslay_event_queue_close_wrapper
            (ctx, WSLAY_CODE_PROTOCOL_ERROR, NULL, 0)) != 0) {
          return r;
        }
        break;
      }
      if(ctx->imsg->opcode == 0xffu) {
        if(iocb.opcode == WSLAY_TEXT_FRAME ||
           iocb.opcode == WSLAY_BINARY_FRAME ||
           iocb.opcode == WSLAY_CONNECTION_CLOSE ||
           iocb.opcode == WSLAY_PING ||
           iocb.opcode == WSLAY_PONG) {
          wslay_event_imsg_set(ctx->imsg, iocb.fin, iocb.rsv, iocb.opcode);
          new_frame = 1;
        } else {
          if((r = wslay_event_queue_close_wrapper
              (ctx, WSLAY_CODE_PROTOCOL_ERROR, NULL, 0)) != 0) {
            return r;
          }
          break;
        }
      } else if(ctx->ipayloadlen == 0 && ctx->ipayloadoff == 0) {
        if(iocb.opcode == WSLAY_CONTINUATION_FRAME) {
          ctx->imsg->fin = iocb.fin;
        } else if(iocb.opcode == WSLAY_CONNECTION_CLOSE ||
                  iocb.opcode == WSLAY_PING ||
                  iocb.opcode == WSLAY_PONG) {
          ctx->imsg = &ctx->imsgs[1];
          wslay_event_imsg_set(ctx->imsg, iocb.fin, iocb.rsv, iocb.opcode);
        } else {
          if((r = wslay_event_queue_close_wrapper
              (ctx, WSLAY_CODE_PROTOCOL_ERROR, NULL, 0)) != 0) {
            return r;
          }
          break;
        }
        new_frame = 1;
      }
      if(new_frame) {
        if(ctx->imsg->msg_length+iocb.payload_length >
           ctx->max_recv_msg_length) {
          if((r = wslay_event_queue_close_wrapper
              (ctx, WSLAY_CODE_MESSAGE_TOO_BIG, NULL, 0)) != 0) {
            return r;
          }
          break;
        }
        ctx->ipayloadlen = iocb.payload_length;
        wslay_event_call_on_frame_recv_start_callback(ctx, &iocb);
        if(!wslay_event_config_get_no_buffering(ctx) ||
           wslay_is_ctrl_frame(iocb.opcode)) {
          if((r = wslay_event_imsg_append_chunk(ctx->imsg,
                                                iocb.payload_length)) != 0) {
            ctx->read_enabled = 0;
            return r;
          }
        }
      }
      /* If RSV1 bit is set then it is too early for utf-8 validation */
      if((!wslay_get_rsv1(ctx->imsg->rsv) &&
          ctx->imsg->opcode == WSLAY_TEXT_FRAME) ||
            ctx->imsg->opcode == WSLAY_CONNECTION_CLOSE) {
        size_t i;
        if(ctx->imsg->opcode == WSLAY_CONNECTION_CLOSE) {
          i = 2;
        } else {
          i = 0;
        }
        for(; i < iocb.data_length; ++i) {
          uint32_t codep;
          if(decode(&ctx->imsg->utf8state, &codep,
                    iocb.data[i]) == UTF8_REJECT) {
            if((r = wslay_event_queue_close_wrapper
                (ctx, WSLAY_CODE_INVALID_FRAME_PAYLOAD_DATA, NULL, 0)) != 0) {
              return r;
            }
            break;
          }
        }
      }
      if(ctx->imsg->utf8state == UTF8_REJECT) {
        break;
      }
      wslay_event_call_on_frame_recv_chunk_callback(ctx, &iocb);
      if(iocb.data_length > 0) {
        if(!wslay_event_config_get_no_buffering(ctx) ||
           wslay_is_ctrl_frame(iocb.opcode)) {
          struct wslay_event_byte_chunk *chunk;
          chunk = wslay_queue_tail(ctx->imsg->chunks);
          wslay_event_byte_chunk_copy(chunk, ctx->ipayloadoff,
                                      iocb.data, iocb.data_length);
        }
        ctx->ipayloadoff += iocb.data_length;
      }
      if(ctx->ipayloadoff == ctx->ipayloadlen) {
        if(ctx->imsg->fin &&
           (ctx->imsg->opcode == WSLAY_TEXT_FRAME ||
            ctx->imsg->opcode == WSLAY_CONNECTION_CLOSE) &&
           ctx->imsg->utf8state != UTF8_ACCEPT) {
          if((r = wslay_event_queue_close_wrapper
              (ctx, WSLAY_CODE_INVALID_FRAME_PAYLOAD_DATA, NULL, 0)) != 0) {
            return r;
          }
          break;
        }
        wslay_event_call_on_frame_recv_end_callback(ctx);
        if(ctx->imsg->fin) {
          if(ctx->callbacks.on_msg_recv_callback ||
             ctx->imsg->opcode == WSLAY_CONNECTION_CLOSE ||
             ctx->imsg->opcode == WSLAY_PING) {
            struct wslay_event_on_msg_recv_arg arg;
            uint16_t status_code = 0;
            uint8_t *msg = NULL;
            size_t msg_length = 0;
            if(!wslay_event_config_get_no_buffering(ctx) ||
               wslay_is_ctrl_frame(iocb.opcode)) {
              msg = wslay_event_flatten_queue(ctx->imsg->chunks,
                                              ctx->imsg->msg_length);
              if(ctx->imsg->msg_length && !msg) {
                ctx->read_enabled = 0;
                return WSLAY_ERR_NOMEM;
              }
              msg_length = ctx->imsg->msg_length;
            }
            if(ctx->imsg->opcode == WSLAY_CONNECTION_CLOSE) {
              const uint8_t *reason;
              size_t reason_length;
              if(ctx->imsg->msg_length >= 2) {
                memcpy(&status_code, msg, 2);
                status_code = ntohs(status_code);
                if(!wslay_event_is_valid_status_code(status_code)) {
                  free(msg);
                  if((r = wslay_event_queue_close_wrapper
                      (ctx, WSLAY_CODE_PROTOCOL_ERROR, NULL, 0)) != 0) {
                    return r;
                  }
                  break;
                }
                reason = msg+2;
                reason_length = ctx->imsg->msg_length-2;
              } else {
                reason = NULL;
                reason_length = 0;
              }
              ctx->close_status |= WSLAY_CLOSE_RECEIVED;
              ctx->status_code_recv =
                status_code == 0 ? WSLAY_CODE_NO_STATUS_RCVD : status_code;
              if((r = wslay_event_queue_close_wrapper
                  (ctx, status_code, reason, reason_length)) != 0) {
                free(msg);
                return r;
              }
            } else if(ctx->imsg->opcode == WSLAY_PING) {
              struct wslay_event_msg arg;
              arg.opcode = WSLAY_PONG;
              arg.msg = msg;
              arg.msg_length = ctx->imsg->msg_length;
              if((r = wslay_event_queue_msg(ctx, &arg)) &&
                 r != WSLAY_ERR_NO_MORE_MSG) {
                ctx->read_enabled = 0;
                free(msg);
                return r;
              }
            }
            if(ctx->callbacks.on_msg_recv_callback) {
              arg.rsv = ctx->imsg->rsv;
              arg.opcode = ctx->imsg->opcode;
              arg.msg = msg;
              arg.msg_length = msg_length;
              arg.status_code = status_code;
              ctx->error = 0;
              ctx->callbacks.on_msg_recv_callback(ctx, &arg, ctx->user_data);
            }
            free(msg);
          }
          wslay_event_imsg_reset(ctx->imsg);
          if(ctx->imsg == &ctx->imsgs[1]) {
            ctx->imsg = &ctx->imsgs[0];
          }
        }
        ctx->ipayloadlen = ctx->ipayloadoff = 0;
      }
    } else {
      if(r != WSLAY_ERR_WANT_READ ||
         (ctx->error != WSLAY_ERR_WOULDBLOCK && ctx->error != 0)) {
        if((r = wslay_event_queue_close_wrapper(ctx, 0, NULL, 0)) != 0) {
          return r;
        }
        return WSLAY_ERR_CALLBACK_FAILURE;
      }
      break;
    }
  }
  return 0;
}

static void wslay_event_on_non_fragmented_msg_popped
(wslay_event_context_ptr ctx)
{
  ctx->omsg->fin = 1;
  ctx->opayloadlen = ctx->omsg->data_length;
  ctx->opayloadoff = 0;
}

static struct wslay_event_omsg* wslay_event_send_ctrl_queue_pop
(wslay_event_context_ptr ctx)
{
  /*
   * If Close control frame is queued, we don't send any control frame
   * other than Close.
   */
  if(ctx->close_status & WSLAY_CLOSE_QUEUED) {
    while(!wslay_queue_empty(ctx->send_ctrl_queue)) {
      struct wslay_event_omsg *msg = wslay_queue_top(ctx->send_ctrl_queue);
      wslay_queue_pop(ctx->send_ctrl_queue);
      if(msg->opcode == WSLAY_CONNECTION_CLOSE) {
        return msg;
      } else {
        wslay_event_omsg_free(msg);
      }
    }
    return NULL;
  } else {
    struct wslay_event_omsg *msg = wslay_queue_top(ctx->send_ctrl_queue);
    wslay_queue_pop(ctx->send_ctrl_queue);
    return msg;
  }
}

int wslay_event_send(wslay_event_context_ptr ctx)
{
  struct wslay_frame_iocb iocb;
  ssize_t r;
  while(ctx->write_enabled &&
        (!wslay_queue_empty(ctx->send_queue) ||
         !wslay_queue_empty(ctx->send_ctrl_queue) || ctx->omsg)) {
    if(!ctx->omsg) {
      if(wslay_queue_empty(ctx->send_ctrl_queue)) {
        ctx->omsg = wslay_queue_top(ctx->send_queue);
        wslay_queue_pop(ctx->send_queue);
      } else {
        ctx->omsg = wslay_event_send_ctrl_queue_pop(ctx);
        if(ctx->omsg == NULL) {
          break;
        }
      }
      if(ctx->omsg->type == WSLAY_NON_FRAGMENTED) {
        wslay_event_on_non_fragmented_msg_popped(ctx);
      }
    } else if(!wslay_is_ctrl_frame(ctx->omsg->opcode) &&
              ctx->frame_ctx->ostate == PREP_HEADER &&
              !wslay_queue_empty(ctx->send_ctrl_queue)) {
      if((r = wslay_queue_push_front(ctx->send_queue, ctx->omsg)) != 0) {
        ctx->write_enabled = 0;
        return r;
      }
      ctx->omsg = wslay_event_send_ctrl_queue_pop(ctx);
      if(ctx->omsg == NULL) {
        break;
      }
      /* ctrl message has WSLAY_NON_FRAGMENTED */
      wslay_event_on_non_fragmented_msg_popped(ctx);
    }
    if(ctx->omsg->type == WSLAY_NON_FRAGMENTED) {
      memset(&iocb, 0, sizeof(iocb));
      iocb.fin = 1;
      iocb.opcode = ctx->omsg->opcode;
      iocb.rsv = ctx->omsg->rsv;
      iocb.mask = ctx->server^1;
      iocb.data = ctx->omsg->data+ctx->opayloadoff;
      iocb.data_length = ctx->opayloadlen-ctx->opayloadoff;
      iocb.payload_length = ctx->opayloadlen;
      r = wslay_frame_send(ctx->frame_ctx, &iocb);
      if(r >= 0) {
        ctx->opayloadoff += r;
        if(ctx->opayloadoff == ctx->opayloadlen) {
          --ctx->queued_msg_count;
          ctx->queued_msg_length -= ctx->omsg->data_length;
          if(ctx->omsg->opcode == WSLAY_CONNECTION_CLOSE) {
            uint16_t status_code = 0;
            ctx->write_enabled = 0;
            ctx->close_status |= WSLAY_CLOSE_SENT;
            if(ctx->omsg->data_length >= 2) {
              memcpy(&status_code, ctx->omsg->data, 2);
              status_code = ntohs(status_code);
            }
            ctx->status_code_sent =
              status_code == 0 ? WSLAY_CODE_NO_STATUS_RCVD : status_code;
          }
          wslay_event_omsg_free(ctx->omsg);
          ctx->omsg = NULL;
        } else {
          break;
        }
      } else {
        if(r != WSLAY_ERR_WANT_WRITE ||
           (ctx->error != WSLAY_ERR_WOULDBLOCK && ctx->error != 0)) {
          ctx->write_enabled = 0;
          return WSLAY_ERR_CALLBACK_FAILURE;
        }
        break;
      }
    } else {
      if(ctx->omsg->fin == 0 && ctx->obuflimit == ctx->obufmark) {
        int eof = 0;
        r = ctx->omsg->read_callback(ctx, ctx->obuf, sizeof(ctx->obuf),
                                     &ctx->omsg->source,
                                     &eof, ctx->user_data);
        if(r == 0) {
          break;
        } else if(r < 0) {
          ctx->write_enabled = 0;
          return WSLAY_ERR_CALLBACK_FAILURE;
        }
        ctx->obuflimit = ctx->obuf+r;
        if(eof) {
          ctx->omsg->fin = 1;
        }
        ctx->opayloadlen = r;
        ctx->opayloadoff = 0;
      }
      memset(&iocb, 0, sizeof(iocb));
      iocb.fin = ctx->omsg->fin;
      iocb.opcode = ctx->omsg->opcode;
      iocb.rsv = ctx->omsg->rsv;
      iocb.mask = ctx->server ? 0 : 1;
      iocb.data = ctx->obufmark;
      iocb.data_length = ctx->obuflimit-ctx->obufmark;
      iocb.payload_length = ctx->opayloadlen;
      r = wslay_frame_send(ctx->frame_ctx, &iocb);
      if(r >= 0) {
        ctx->obufmark += r;
        if(ctx->obufmark == ctx->obuflimit) {
          ctx->obufmark = ctx->obuflimit = ctx->obuf;
          if(ctx->omsg->fin) {
            --ctx->queued_msg_count;
            wslay_event_omsg_free(ctx->omsg);
            ctx->omsg = NULL;
          } else {
            ctx->omsg->opcode = WSLAY_CONTINUATION_FRAME;
            /* RSV1 is not set on continuation frames */
            ctx->omsg->rsv = ctx->omsg->rsv & ~WSLAY_RSV1_BIT;
          }
        } else {
          break;
        }
      } else {
        if(r != WSLAY_ERR_WANT_WRITE ||
           (ctx->error != WSLAY_ERR_WOULDBLOCK &&
            ctx->error != 0)) {
          ctx->write_enabled = 0;
          return WSLAY_ERR_CALLBACK_FAILURE;
        }
        break;
      }
    }
  }
  return 0;
}

void wslay_event_set_error(wslay_event_context_ptr ctx, int val)
{
  ctx->error = val;
}

int wslay_event_want_read(wslay_event_context_ptr ctx)
{
  return ctx->read_enabled;
}

int wslay_event_want_write(wslay_event_context_ptr ctx)
{
  return ctx->write_enabled &&
    (!wslay_queue_empty(ctx->send_queue) ||
     !wslay_queue_empty(ctx->send_ctrl_queue) || ctx->omsg);
}

void wslay_event_shutdown_read(wslay_event_context_ptr ctx)
{
  ctx->read_enabled = 0;
}

void wslay_event_shutdown_write(wslay_event_context_ptr ctx)
{
  ctx->write_enabled = 0;
}

int wslay_event_get_read_enabled(wslay_event_context_ptr ctx)
{
  return ctx->read_enabled;
}

int wslay_event_get_write_enabled(wslay_event_context_ptr ctx)
{
  return ctx->write_enabled;
}

int wslay_event_get_close_received(wslay_event_context_ptr ctx)
{
  return (ctx->close_status & WSLAY_CLOSE_RECEIVED) > 0;
}

int wslay_event_get_close_sent(wslay_event_context_ptr ctx)
{
  return (ctx->close_status & WSLAY_CLOSE_SENT) > 0;
}

void wslay_event_config_set_allowed_rsv_bits(wslay_event_context_ptr ctx,
                                             uint8_t rsv)
{
  /* We currently only allow WSLAY_RSV1_BIT or WSLAY_RSV_NONE */
  ctx->allowed_rsv_bits = rsv & WSLAY_RSV1_BIT;
}

void wslay_event_config_set_no_buffering(wslay_event_context_ptr ctx, int val)
{
  if(val) {
    ctx->config |= WSLAY_CONFIG_NO_BUFFERING;
  } else {
    ctx->config &= ~WSLAY_CONFIG_NO_BUFFERING;
  }
}

void wslay_event_config_set_max_recv_msg_length(wslay_event_context_ptr ctx,
                                                uint64_t val)
{
  ctx->max_recv_msg_length = val;
}

uint16_t wslay_event_get_status_code_received(wslay_event_context_ptr ctx)
{
  return ctx->status_code_recv;
}

uint16_t wslay_event_get_status_code_sent(wslay_event_context_ptr ctx)
{
  return ctx->status_code_sent;
}

size_t wslay_event_get_queued_msg_count(wslay_event_context_ptr ctx)
{
  return ctx->queued_msg_count;
}

size_t wslay_event_get_queued_msg_length(wslay_event_context_ptr ctx)
{
  return ctx->queued_msg_length;
}
