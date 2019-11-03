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

#include "opus.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_PACKETOUT 32000

void usage(char *argv0)
{
   fprintf(stderr, "usage: %s [options] input_file output_file\n", argv0);
}

static void int_to_char(opus_uint32 i, unsigned char ch[4])
{
    ch[0] = i>>24;
    ch[1] = (i>>16)&0xFF;
    ch[2] = (i>>8)&0xFF;
    ch[3] = i&0xFF;
}

static opus_uint32 char_to_int(unsigned char ch[4])
{
    return ((opus_uint32)ch[0]<<24) | ((opus_uint32)ch[1]<<16)
         | ((opus_uint32)ch[2]<< 8) |  (opus_uint32)ch[3];
}

int main(int argc, char *argv[])
{
   int i, eof=0;
   FILE *fin, *fout;
   unsigned char packets[48][1500];
   int len[48];
   int rng[48];
   OpusRepacketizer *rp;
   unsigned char output_packet[MAX_PACKETOUT];
   int merge = 1, split=0;

   if (argc < 3)
   {
      usage(argv[0]);
      return EXIT_FAILURE;
   }
   for (i=1;i<argc-2;i++)
   {
      if (strcmp(argv[i], "-merge")==0)
      {
         merge = atoi(argv[i+1]);
         if(merge<1)
         {
            fprintf(stderr, "-merge parameter must be at least 1.\n");
            return EXIT_FAILURE;
         }
         if(merge>48)
         {
            fprintf(stderr, "-merge parameter must be less than 48.\n");
            return EXIT_FAILURE;
         }
         i++;
      } else if (strcmp(argv[i], "-split")==0)
         split = 1;
      else
      {
         fprintf(stderr, "Unknown option: %s\n", argv[i]);
         usage(argv[0]);
         return EXIT_FAILURE;
      }
   }
   fin = fopen(argv[argc-2], "r");
   if(fin==NULL)
   {
     fprintf(stderr, "Error opening input file: %s\n", argv[argc-2]);
     return EXIT_FAILURE;
   }
   fout = fopen(argv[argc-1], "w");
   if(fout==NULL)
   {
     fprintf(stderr, "Error opening output file: %s\n", argv[argc-1]);
     fclose(fin);
     return EXIT_FAILURE;
   }

   rp = opus_repacketizer_create();
   while (!eof)
   {
      int err;
      int nb_packets=merge;
      opus_repacketizer_init(rp);
      for (i=0;i<nb_packets;i++)
      {
         unsigned char ch[4];
         err = fread(ch, 1, 4, fin);
         len[i] = char_to_int(ch);
         /*fprintf(stderr, "in len = %d\n", len[i]);*/
         if (len[i]>1500 || len[i]<0)
         {
             if (feof(fin))
             {
                eof = 1;
             } else {
                fprintf(stderr, "Invalid payload length\n");
                fclose(fin);
                fclose(fout);
                return EXIT_FAILURE;
             }
             break;
         }
         err = fread(ch, 1, 4, fin);
         rng[i] = char_to_int(ch);
         err = fread(packets[i], 1, len[i], fin);
         if (feof(fin))
         {
            eof = 1;
            break;
         }
         err = opus_repacketizer_cat(rp, packets[i], len[i]);
         if (err!=OPUS_OK)
         {
            fprintf(stderr, "opus_repacketizer_cat() failed: %s\n", opus_strerror(err));
            break;
         }
      }
      nb_packets = i;

      if (eof)
         break;

      if (!split)
      {
         err = opus_repacketizer_out(rp, output_packet, MAX_PACKETOUT);
         if (err>0) {
            unsigned char int_field[4];
            int_to_char(err, int_field);
            if(fwrite(int_field, 1, 4, fout)!=4){
               fprintf(stderr, "Error writing.\n");
               return EXIT_FAILURE;
            }
            int_to_char(rng[nb_packets-1], int_field);
            if (fwrite(int_field, 1, 4, fout)!=4) {
               fprintf(stderr, "Error writing.\n");
               return EXIT_FAILURE;
            }
            if (fwrite(output_packet, 1, err, fout)!=(unsigned)err) {
               fprintf(stderr, "Error writing.\n");
               return EXIT_FAILURE;
            }
            /*fprintf(stderr, "out len = %d\n", err);*/
         } else {
            fprintf(stderr, "opus_repacketizer_out() failed: %s\n", opus_strerror(err));
         }
      } else {
         int nb_frames = opus_repacketizer_get_nb_frames(rp);
         for (i=0;i<nb_frames;i++)
         {
            err = opus_repacketizer_out_range(rp, i, i+1, output_packet, MAX_PACKETOUT);
            if (err>0) {
               unsigned char int_field[4];
               int_to_char(err, int_field);
               if (fwrite(int_field, 1, 4, fout)!=4) {
                  fprintf(stderr, "Error writing.\n");
                  return EXIT_FAILURE;
               }
               if (i==nb_frames-1)
                  int_to_char(rng[nb_packets-1], int_field);
               else
                  int_to_char(0, int_field);
               if (fwrite(int_field, 1, 4, fout)!=4) {
                  fprintf(stderr, "Error writing.\n");
                  return EXIT_FAILURE;
               }
               if (fwrite(output_packet, 1, err, fout)!=(unsigned)err) {
                  fprintf(stderr, "Error writing.\n");
                  return EXIT_FAILURE;
               }
               /*fprintf(stderr, "out len = %d\n", err);*/
            } else {
               fprintf(stderr, "opus_repacketizer_out() failed: %s\n", opus_strerror(err));
            }

         }
      }
   }

   fclose(fin);
   fclose(fout);
   return EXIT_SUCCESS;
}
