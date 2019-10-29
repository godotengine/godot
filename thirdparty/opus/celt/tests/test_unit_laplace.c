/* Copyright (c) 2008-2011 Xiph.Org Foundation, Mozilla Corporation
   Written by Jean-Marc Valin and Timothy B. Terriberry */
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

#include <stdio.h>
#include <stdlib.h>
#include "laplace.h"
#define CELT_C
#include "stack_alloc.h"

#include "entenc.c"
#include "entdec.c"
#include "entcode.c"
#include "laplace.c"

#define DATA_SIZE 40000

int ec_laplace_get_start_freq(int decay)
{
   opus_uint32 ft = 32768 - LAPLACE_MINP*(2*LAPLACE_NMIN+1);
   int fs = (ft*(16384-decay))/(16384+decay);
   return fs+LAPLACE_MINP;
}

int main(void)
{
   int i;
   int ret = 0;
   ec_enc enc;
   ec_dec dec;
   unsigned char *ptr;
   int val[10000], decay[10000];
   ALLOC_STACK;
   ptr = (unsigned char *)malloc(DATA_SIZE);
   ec_enc_init(&enc,ptr,DATA_SIZE);

   val[0] = 3; decay[0] = 6000;
   val[1] = 0; decay[1] = 5800;
   val[2] = -1; decay[2] = 5600;
   for (i=3;i<10000;i++)
   {
      val[i] = rand()%15-7;
      decay[i] = rand()%11000+5000;
   }
   for (i=0;i<10000;i++)
      ec_laplace_encode(&enc, &val[i],
            ec_laplace_get_start_freq(decay[i]), decay[i]);

   ec_enc_done(&enc);

   ec_dec_init(&dec,ec_get_buffer(&enc),ec_range_bytes(&enc));

   for (i=0;i<10000;i++)
   {
      int d = ec_laplace_decode(&dec,
            ec_laplace_get_start_freq(decay[i]), decay[i]);
      if (d != val[i])
      {
         fprintf (stderr, "Got %d instead of %d\n", d, val[i]);
         ret = 1;
      }
   }

   free(ptr);
   return ret;
}
