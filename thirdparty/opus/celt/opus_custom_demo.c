/* Copyright (c) 2007-2008 CSIRO
   Copyright (c) 2007-2009 Xiph.Org Foundation
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

#include "opus_custom.h"
#include "arch.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAX_PACKET 1275

int main(int argc, char *argv[])
{
   int err;
   char *inFile, *outFile;
   FILE *fin, *fout;
   OpusCustomMode *mode=NULL;
   OpusCustomEncoder *enc;
   OpusCustomDecoder *dec;
   int len;
   opus_int32 frame_size, channels, rate;
   int bytes_per_packet;
   unsigned char data[MAX_PACKET];
   int complexity;
#if !(defined (FIXED_POINT) && !defined(CUSTOM_MODES)) && defined(RESYNTH)
   int i;
   double rmsd = 0;
#endif
   int count = 0;
   opus_int32 skip;
   opus_int16 *in, *out;
   if (argc != 9 && argc != 8 && argc != 7)
   {
      fprintf (stderr, "Usage: test_opus_custom <rate> <channels> <frame size> "
               " <bytes per packet> [<complexity> [packet loss rate]] "
               "<input> <output>\n");
      return 1;
   }

   rate = (opus_int32)atol(argv[1]);
   channels = atoi(argv[2]);
   frame_size = atoi(argv[3]);
   mode = opus_custom_mode_create(rate, frame_size, NULL);
   if (mode == NULL)
   {
      fprintf(stderr, "failed to create a mode\n");
      return 1;
   }

   bytes_per_packet = atoi(argv[4]);
   if (bytes_per_packet < 0 || bytes_per_packet > MAX_PACKET)
   {
      fprintf (stderr, "bytes per packet must be between 0 and %d\n",
                        MAX_PACKET);
      return 1;
   }

   inFile = argv[argc-2];
   fin = fopen(inFile, "rb");
   if (!fin)
   {
      fprintf (stderr, "Could not open input file %s\n", argv[argc-2]);
      return 1;
   }
   outFile = argv[argc-1];
   fout = fopen(outFile, "wb+");
   if (!fout)
   {
      fprintf (stderr, "Could not open output file %s\n", argv[argc-1]);
      fclose(fin);
      return 1;
   }

   enc = opus_custom_encoder_create(mode, channels, &err);
   if (err != 0)
   {
      fprintf(stderr, "Failed to create the encoder: %s\n", opus_strerror(err));
      fclose(fin);
      fclose(fout);
      return 1;
   }
   dec = opus_custom_decoder_create(mode, channels, &err);
   if (err != 0)
   {
      fprintf(stderr, "Failed to create the decoder: %s\n", opus_strerror(err));
      fclose(fin);
      fclose(fout);
      return 1;
   }
   opus_custom_decoder_ctl(dec, OPUS_GET_LOOKAHEAD(&skip));

   if (argc>7)
   {
      complexity=atoi(argv[5]);
      opus_custom_encoder_ctl(enc,OPUS_SET_COMPLEXITY(complexity));
   }

   in = (opus_int16*)malloc(frame_size*channels*sizeof(opus_int16));
   out = (opus_int16*)malloc(frame_size*channels*sizeof(opus_int16));

   while (!feof(fin))
   {
      int ret;
      err = fread(in, sizeof(short), frame_size*channels, fin);
      if (feof(fin))
         break;
      len = opus_custom_encode(enc, in, frame_size, data, bytes_per_packet);
      if (len <= 0)
         fprintf (stderr, "opus_custom_encode() failed: %s\n", opus_strerror(len));

      /* This is for simulating bit errors */
#if 0
      int errors = 0;
      int eid = 0;
      /* This simulates random bit error */
      for (i=0;i<len*8;i++)
      {
         if (rand()%atoi(argv[8])==0)
         {
            if (i<64)
            {
               errors++;
               eid = i;
            }
            data[i/8] ^= 1<<(7-(i%8));
         }
      }
      if (errors == 1)
         data[eid/8] ^= 1<<(7-(eid%8));
      else if (errors%2 == 1)
         data[rand()%8] ^= 1<<rand()%8;
#endif

#if 1 /* Set to zero to use the encoder's output instead */
      /* This is to simulate packet loss */
      if (argc==9 && rand()%1000<atoi(argv[argc-3]))
      /*if (errors && (errors%2==0))*/
         ret = opus_custom_decode(dec, NULL, len, out, frame_size);
      else
         ret = opus_custom_decode(dec, data, len, out, frame_size);
      if (ret < 0)
         fprintf(stderr, "opus_custom_decode() failed: %s\n", opus_strerror(ret));
#else
      for (i=0;i<ret*channels;i++)
         out[i] = in[i];
#endif
#if !(defined (FIXED_POINT) && !defined(CUSTOM_MODES)) && defined(RESYNTH)
      for (i=0;i<ret*channels;i++)
      {
         rmsd += (in[i]-out[i])*1.0*(in[i]-out[i]);
         /*out[i] -= in[i];*/
      }
#endif
      count++;
      fwrite(out+skip*channels, sizeof(short), (ret-skip)*channels, fout);
      skip = 0;
   }
   PRINT_MIPS(stderr);

   opus_custom_encoder_destroy(enc);
   opus_custom_decoder_destroy(dec);
   fclose(fin);
   fclose(fout);
   opus_custom_mode_destroy(mode);
   free(in);
   free(out);
#if !(defined (FIXED_POINT) && !defined(CUSTOM_MODES)) && defined(RESYNTH)
   if (rmsd > 0)
   {
      rmsd = sqrt(rmsd/(1.0*frame_size*channels*count));
      fprintf (stderr, "Error: encoder doesn't match decoder\n");
      fprintf (stderr, "RMS mismatch is %f\n", rmsd);
      return 1;
   } else {
      fprintf (stderr, "Encoder matches decoder!!\n");
   }
#endif
   return 0;
}

