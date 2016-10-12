/* Copyright (C) 2007 Jean-Marc Valin
      
   File: speex_buffer.h
   This is a very simple ring buffer implementation. It is not thread-safe
   so you need to do your own locking.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:

   1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   3. The name of the author may not be used to endorse or promote products
   derived from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
   IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
   OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
   DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
   STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef SPEEX_BUFFER_H
#define SPEEX_BUFFER_H

#include "speex/speex_types.h"

#ifdef __cplusplus
extern "C" {
#endif

struct SpeexBuffer_;
typedef struct SpeexBuffer_ SpeexBuffer;

SpeexBuffer *speex_buffer_init(int size);

void speex_buffer_destroy(SpeexBuffer *st);

int speex_buffer_write(SpeexBuffer *st, void *data, int len);

int speex_buffer_writezeros(SpeexBuffer *st, int len);

int speex_buffer_read(SpeexBuffer *st, void *data, int len);

int speex_buffer_get_available(SpeexBuffer *st);

int speex_buffer_resize(SpeexBuffer *st, int len);

#ifdef __cplusplus
}
#endif

#endif




