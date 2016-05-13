/* Copyright (C) 2002 Jean-Marc Valin 
   File: speex_header.c
   Describes the Speex header

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:
   
   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
   
   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
   
   - Neither the name of the Xiph.org Foundation nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.
   
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/


#include "config.h"


#include "arch.h"
#include <speex/speex_header.h>
#include <speex/speex.h>
#include "os_support.h"

#ifndef NULL
#define NULL 0
#endif

/** Convert little endian */
static SPEEX_INLINE spx_int32_t le_int(spx_int32_t i)
{
#if !defined(__LITTLE_ENDIAN__) && ( defined(WORDS_BIGENDIAN) || defined(__BIG_ENDIAN__) /* || defined(BIG_ENDIAN_ENABLED) */ )
   spx_uint32_t ui, ret;
   ui = i;
   ret =  ui>>24;
   ret |= (ui>>8)&0x0000ff00;
   ret |= (ui<<8)&0x00ff0000;
   ret |= (ui<<24);
   return ret;
#else
   return i;
#endif
}

#define ENDIAN_SWITCH(x) {x=le_int(x);}


/*
typedef struct SpeexHeader {
   char speex_string[8];
   char speex_version[SPEEX_HEADER_VERSION_LENGTH];
   int speex_version_id;
   int header_size;
   int rate;
   int mode;
   int mode_bitstream_version;
   int nb_channels;
   int bitrate;
   int frame_size;
   int vbr;
   int frames_per_packet;
   int extra_headers;
   int reserved1;
   int reserved2;
} SpeexHeader;
*/

EXPORT void speex_init_header(SpeexHeader *header, int rate, int nb_channels, const SpeexMode *m)
{
   int i;
   const char *h="Speex   ";
   /*
   strncpy(header->speex_string, "Speex   ", 8);
   strncpy(header->speex_version, SPEEX_VERSION, SPEEX_HEADER_VERSION_LENGTH-1);
   header->speex_version[SPEEX_HEADER_VERSION_LENGTH-1]=0;
   */
   for (i=0;i<8;i++)
      header->speex_string[i]=h[i];
   for (i=0;i<SPEEX_HEADER_VERSION_LENGTH-1 && SPEEX_VERSION[i];i++)
      header->speex_version[i]=SPEEX_VERSION[i];
   for (;i<SPEEX_HEADER_VERSION_LENGTH;i++)
      header->speex_version[i]=0;
   
   header->speex_version_id = 1;
   header->header_size = sizeof(SpeexHeader);
   
   header->rate = rate;
   header->mode = m->modeID;
   header->mode_bitstream_version = m->bitstream_version;
   if (m->modeID<0)
      speex_warning("This mode is meant to be used alone");
   header->nb_channels = nb_channels;
   header->bitrate = -1;
   speex_mode_query(m, SPEEX_MODE_FRAME_SIZE, &header->frame_size);
   header->vbr = 0;
   
   header->frames_per_packet = 0;
   header->extra_headers = 0;
   header->reserved1 = 0;
   header->reserved2 = 0;
}

EXPORT char *speex_header_to_packet(SpeexHeader *header, int *size)
{
   SpeexHeader *le_header;
   le_header = (SpeexHeader*)speex_alloc(sizeof(SpeexHeader));
   
   SPEEX_COPY(le_header, header, 1);
   
   /*Make sure everything is now little-endian*/
   ENDIAN_SWITCH(le_header->speex_version_id);
   ENDIAN_SWITCH(le_header->header_size);
   ENDIAN_SWITCH(le_header->rate);
   ENDIAN_SWITCH(le_header->mode);
   ENDIAN_SWITCH(le_header->mode_bitstream_version);
   ENDIAN_SWITCH(le_header->nb_channels);
   ENDIAN_SWITCH(le_header->bitrate);
   ENDIAN_SWITCH(le_header->frame_size);
   ENDIAN_SWITCH(le_header->vbr);
   ENDIAN_SWITCH(le_header->frames_per_packet);
   ENDIAN_SWITCH(le_header->extra_headers);

   *size = sizeof(SpeexHeader);
   return (char *)le_header;
}

EXPORT SpeexHeader *speex_packet_to_header(char *packet, int size)
{
   int i;
   SpeexHeader *le_header;
   const char *h = "Speex   ";
   for (i=0;i<8;i++)
      if (packet[i]!=h[i])
      {
         speex_notify("This doesn't look like a Speex file");
         return NULL;
      }
   
   /*FIXME: Do we allow larger headers?*/
   if (size < (int)sizeof(SpeexHeader))
   {
      speex_notify("Speex header too small");
      return NULL;
   }
   
   le_header = (SpeexHeader*)speex_alloc(sizeof(SpeexHeader));
   
   SPEEX_COPY(le_header, (SpeexHeader*)packet, 1);
   
   /*Make sure everything is converted correctly from little-endian*/
   ENDIAN_SWITCH(le_header->speex_version_id);
   ENDIAN_SWITCH(le_header->header_size);
   ENDIAN_SWITCH(le_header->rate);
   ENDIAN_SWITCH(le_header->mode);
   ENDIAN_SWITCH(le_header->mode_bitstream_version);
   ENDIAN_SWITCH(le_header->nb_channels);
   ENDIAN_SWITCH(le_header->bitrate);
   ENDIAN_SWITCH(le_header->frame_size);
   ENDIAN_SWITCH(le_header->vbr);
   ENDIAN_SWITCH(le_header->frames_per_packet);
   ENDIAN_SWITCH(le_header->extra_headers);

   if (le_header->mode >= SPEEX_NB_MODES || le_header->mode < 0)
   {
      speex_notify("Invalid mode specified in Speex header");
      speex_free (le_header);
      return NULL;
   }

   if (le_header->nb_channels>2)
      le_header->nb_channels = 2;
   if (le_header->nb_channels<1)
      le_header->nb_channels = 1;

   return le_header;

}

EXPORT void speex_header_free(void *ptr)
{
   speex_free(ptr);
}
