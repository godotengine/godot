/* Copyright (c) 2011 Xiph.Org Foundation
   Written by Jean-Marc Valin */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "opus_multistream.h"
#include "opus.h"
#include "opus_private.h"
#include "stack_alloc.h"
#include <stdarg.h>
#include "float_cast.h"
#include "os_support.h"


int validate_layout(const ChannelLayout *layout)
{
   int i, max_channel;

   max_channel = layout->nb_streams+layout->nb_coupled_streams;
   if (max_channel>255)
      return 0;
   for (i=0;i<layout->nb_channels;i++)
   {
      if (layout->mapping[i] >= max_channel && layout->mapping[i] != 255)
         return 0;
   }
   return 1;
}


int get_left_channel(const ChannelLayout *layout, int stream_id, int prev)
{
   int i;
   i = (prev<0) ? 0 : prev+1;
   for (;i<layout->nb_channels;i++)
   {
      if (layout->mapping[i]==stream_id*2)
         return i;
   }
   return -1;
}

int get_right_channel(const ChannelLayout *layout, int stream_id, int prev)
{
   int i;
   i = (prev<0) ? 0 : prev+1;
   for (;i<layout->nb_channels;i++)
   {
      if (layout->mapping[i]==stream_id*2+1)
         return i;
   }
   return -1;
}

int get_mono_channel(const ChannelLayout *layout, int stream_id, int prev)
{
   int i;
   i = (prev<0) ? 0 : prev+1;
   for (;i<layout->nb_channels;i++)
   {
      if (layout->mapping[i]==stream_id+layout->nb_coupled_streams)
         return i;
   }
   return -1;
}

