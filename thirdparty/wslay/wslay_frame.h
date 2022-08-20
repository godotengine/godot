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
#ifndef WSLAY_FRAME_H
#define WSLAY_FRAME_H

#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <wslay/wslay.h>

enum wslay_frame_state {
  PREP_HEADER,
  PREP_HEADER_NOBUF,
  SEND_HEADER,
  SEND_PAYLOAD,
  RECV_HEADER1,
  RECV_PAYLOADLEN,
  RECV_EXT_PAYLOADLEN,
  RECV_MASKKEY,
  RECV_PAYLOAD
};

struct wslay_frame_opcode_memo {
  uint8_t fin;
  uint8_t opcode;
  uint8_t rsv;
};

struct wslay_frame_context {
  uint8_t ibuf[4096];
  uint8_t *ibufmark;
  uint8_t *ibuflimit;
  struct wslay_frame_opcode_memo iom;
  uint64_t ipayloadlen;
  uint64_t ipayloadoff;
  uint8_t imask;
  uint8_t imaskkey[4];
  enum wslay_frame_state istate;
  size_t ireqread;

  uint8_t oheader[14];
  uint8_t *oheadermark;
  uint8_t *oheaderlimit;
  uint64_t opayloadlen;
  uint64_t opayloadoff;
  uint8_t omask;
  uint8_t omaskkey[4];
  enum wslay_frame_state ostate;

  struct wslay_frame_callbacks callbacks;
  void *user_data;
};

#endif /* WSLAY_FRAME_H */
