/* Copyright (C) 2002 Jean-Marc Valin 
   File: quant_lsp.c
   LSP vector quantization

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


#include "quant_lsp.h"
#include "os_support.h"
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "arch.h"

#ifdef BFIN_ASM
#include "quant_lsp_bfin.h"
#endif

#ifdef FIXED_POINT

#define LSP_LINEAR(i) (SHL16(i+1,11))
#define LSP_LINEAR_HIGH(i) (ADD16(MULT16_16_16(i,2560),6144))
#define LSP_DIV_256(x) (SHL16((spx_word16_t)x, 5))
#define LSP_DIV_512(x) (SHL16((spx_word16_t)x, 4))
#define LSP_DIV_1024(x) (SHL16((spx_word16_t)x, 3))
#define LSP_PI 25736

#else

#define LSP_LINEAR(i) (.25*(i)+.25)
#define LSP_LINEAR_HIGH(i) (.3125*(i)+.75)
#define LSP_SCALE 256.
#define LSP_DIV_256(x) (0.0039062*(x))
#define LSP_DIV_512(x) (0.0019531*(x))
#define LSP_DIV_1024(x) (0.00097656*(x))
#define LSP_PI M_PI

#endif

static void compute_quant_weights(spx_lsp_t *qlsp, spx_word16_t *quant_weight, int order)
{
   int i;
   spx_word16_t tmp1, tmp2;
   for (i=0;i<order;i++)
   {
      if (i==0)
         tmp1 = qlsp[i];
      else
         tmp1 = qlsp[i]-qlsp[i-1];
      if (i==order-1)
         tmp2 = LSP_PI-qlsp[i];
      else
         tmp2 = qlsp[i+1]-qlsp[i];
      if (tmp2<tmp1)
         tmp1 = tmp2;
#ifdef FIXED_POINT
      quant_weight[i] = DIV32_16(81920,ADD16(300,tmp1));
#else
      quant_weight[i] = 10/(.04+tmp1);
#endif
   }

}

/* Note: x is modified*/
#ifndef OVERRIDE_LSP_QUANT
static int lsp_quant(spx_word16_t *x, const signed char *cdbk, int nbVec, int nbDim)
{
   int i,j;
   spx_word32_t dist;
   spx_word16_t tmp;
   spx_word32_t best_dist=VERY_LARGE32;
   int best_id=0;
   const signed char *ptr=cdbk;
   for (i=0;i<nbVec;i++)
   {
      dist=0;
      for (j=0;j<nbDim;j++)
      {
         tmp=SUB16(x[j],SHL16((spx_word16_t)*ptr++,5));
         dist=MAC16_16(dist,tmp,tmp);
      } 
      if (dist<best_dist)
      {
         best_dist=dist;
         best_id=i;
      }
   }

   for (j=0;j<nbDim;j++)
      x[j] = SUB16(x[j],SHL16((spx_word16_t)cdbk[best_id*nbDim+j],5));
    
   return best_id;
}
#endif

/* Note: x is modified*/
#ifndef OVERRIDE_LSP_WEIGHT_QUANT
static int lsp_weight_quant(spx_word16_t *x, spx_word16_t *weight, const signed char *cdbk, int nbVec, int nbDim)
{
   int i,j;
   spx_word32_t dist;
   spx_word16_t tmp;
   spx_word32_t best_dist=VERY_LARGE32;
   int best_id=0;
   const signed char *ptr=cdbk;
   for (i=0;i<nbVec;i++)
   {
      dist=0;
      for (j=0;j<nbDim;j++)
      {
         tmp=SUB16(x[j],SHL16((spx_word16_t)*ptr++,5));
         dist=MAC16_32_Q15(dist,weight[j],MULT16_16(tmp,tmp));
      }
      if (dist<best_dist)
      {
         best_dist=dist;
         best_id=i;
      }
   }
   
   for (j=0;j<nbDim;j++)
      x[j] = SUB16(x[j],SHL16((spx_word16_t)cdbk[best_id*nbDim+j],5));
   return best_id;
}
#endif

void lsp_quant_nb(spx_lsp_t *lsp, spx_lsp_t *qlsp, int order, SpeexBits *bits)
{
   int i;
   int id;
   spx_word16_t quant_weight[10];
   
   for (i=0;i<order;i++)
      qlsp[i]=lsp[i];

   compute_quant_weights(qlsp, quant_weight, order);

   for (i=0;i<order;i++)
      qlsp[i]=SUB16(qlsp[i],LSP_LINEAR(i));

#ifndef FIXED_POINT
   for (i=0;i<order;i++)
      qlsp[i] = LSP_SCALE*qlsp[i];
#endif
   id = lsp_quant(qlsp, cdbk_nb, NB_CDBK_SIZE, order);
   speex_bits_pack(bits, id, 6);

   for (i=0;i<order;i++)
      qlsp[i]*=2;
 
   id = lsp_weight_quant(qlsp, quant_weight, cdbk_nb_low1, NB_CDBK_SIZE_LOW1, 5);
   speex_bits_pack(bits, id, 6);

   for (i=0;i<5;i++)
      qlsp[i]*=2;

   id = lsp_weight_quant(qlsp, quant_weight, cdbk_nb_low2, NB_CDBK_SIZE_LOW2, 5);
   speex_bits_pack(bits, id, 6);

   id = lsp_weight_quant(qlsp+5, quant_weight+5, cdbk_nb_high1, NB_CDBK_SIZE_HIGH1, 5);
   speex_bits_pack(bits, id, 6);

   for (i=5;i<10;i++)
      qlsp[i]*=2;

   id = lsp_weight_quant(qlsp+5, quant_weight+5, cdbk_nb_high2, NB_CDBK_SIZE_HIGH2, 5);
   speex_bits_pack(bits, id, 6);

#ifdef FIXED_POINT
   for (i=0;i<order;i++)
      qlsp[i]=PSHR16(qlsp[i],2);
#else
   for (i=0;i<order;i++)
      qlsp[i]=qlsp[i] * .00097656;
#endif

   for (i=0;i<order;i++)
      qlsp[i]=lsp[i]-qlsp[i];
}

void lsp_unquant_nb(spx_lsp_t *lsp, int order, SpeexBits *bits)
{
   int i, id;
   for (i=0;i<order;i++)
      lsp[i]=LSP_LINEAR(i);


   id=speex_bits_unpack_unsigned(bits, 6);
   for (i=0;i<10;i++)
      lsp[i] = ADD32(lsp[i], LSP_DIV_256(cdbk_nb[id*10+i]));

   id=speex_bits_unpack_unsigned(bits, 6);
   for (i=0;i<5;i++)
      lsp[i] = ADD16(lsp[i], LSP_DIV_512(cdbk_nb_low1[id*5+i]));

   id=speex_bits_unpack_unsigned(bits, 6);
   for (i=0;i<5;i++)
      lsp[i] = ADD32(lsp[i], LSP_DIV_1024(cdbk_nb_low2[id*5+i]));

   id=speex_bits_unpack_unsigned(bits, 6);
   for (i=0;i<5;i++)
      lsp[i+5] = ADD32(lsp[i+5], LSP_DIV_512(cdbk_nb_high1[id*5+i]));
   
   id=speex_bits_unpack_unsigned(bits, 6);
   for (i=0;i<5;i++)
      lsp[i+5] = ADD32(lsp[i+5], LSP_DIV_1024(cdbk_nb_high2[id*5+i]));
}


void lsp_quant_lbr(spx_lsp_t *lsp, spx_lsp_t *qlsp, int order, SpeexBits *bits)
{
   int i;
   int id;
   spx_word16_t quant_weight[10];

   for (i=0;i<order;i++)
      qlsp[i]=lsp[i];

   compute_quant_weights(qlsp, quant_weight, order);

   for (i=0;i<order;i++)
      qlsp[i]=SUB16(qlsp[i],LSP_LINEAR(i));
#ifndef FIXED_POINT
   for (i=0;i<order;i++)
      qlsp[i]=qlsp[i]*LSP_SCALE;
#endif
   id = lsp_quant(qlsp, cdbk_nb, NB_CDBK_SIZE, order);
   speex_bits_pack(bits, id, 6);
   
   for (i=0;i<order;i++)
      qlsp[i]*=2;
   
   id = lsp_weight_quant(qlsp, quant_weight, cdbk_nb_low1, NB_CDBK_SIZE_LOW1, 5);
   speex_bits_pack(bits, id, 6);

   id = lsp_weight_quant(qlsp+5, quant_weight+5, cdbk_nb_high1, NB_CDBK_SIZE_HIGH1, 5);
   speex_bits_pack(bits, id, 6);

#ifdef FIXED_POINT
   for (i=0;i<order;i++)
      qlsp[i] = PSHR16(qlsp[i],1);
#else
   for (i=0;i<order;i++)
      qlsp[i] = qlsp[i]*0.0019531;
#endif

   for (i=0;i<order;i++)
      qlsp[i]=lsp[i]-qlsp[i];
}

void lsp_unquant_lbr(spx_lsp_t *lsp, int order, SpeexBits *bits)
{
   int i, id;
   for (i=0;i<order;i++)
      lsp[i]=LSP_LINEAR(i);


   id=speex_bits_unpack_unsigned(bits, 6);
   for (i=0;i<10;i++)
      lsp[i] += LSP_DIV_256(cdbk_nb[id*10+i]);

   id=speex_bits_unpack_unsigned(bits, 6);
   for (i=0;i<5;i++)
      lsp[i] += LSP_DIV_512(cdbk_nb_low1[id*5+i]);

   id=speex_bits_unpack_unsigned(bits, 6);
   for (i=0;i<5;i++)
      lsp[i+5] += LSP_DIV_512(cdbk_nb_high1[id*5+i]);
   
}


#ifdef DISABLE_WIDEBAND
void lsp_quant_high(spx_lsp_t *lsp, spx_lsp_t *qlsp, int order, SpeexBits *bits)
{
   speex_fatal("Wideband and Ultra-wideband are disabled");
}
void lsp_unquant_high(spx_lsp_t *lsp, int order, SpeexBits *bits)
{
   speex_fatal("Wideband and Ultra-wideband are disabled");
}
#else
extern const signed char high_lsp_cdbk[];
extern const signed char high_lsp_cdbk2[];


void lsp_quant_high(spx_lsp_t *lsp, spx_lsp_t *qlsp, int order, SpeexBits *bits)
{
   int i;
   int id;
   spx_word16_t quant_weight[10];

   for (i=0;i<order;i++)
      qlsp[i]=lsp[i];

   compute_quant_weights(qlsp, quant_weight, order);

   /*   quant_weight[0] = 10/(qlsp[1]-qlsp[0]);
   quant_weight[order-1] = 10/(qlsp[order-1]-qlsp[order-2]);
   for (i=1;i<order-1;i++)
   {
      tmp1 = 10/(qlsp[i]-qlsp[i-1]);
      tmp2 = 10/(qlsp[i+1]-qlsp[i]);
      quant_weight[i] = tmp1 > tmp2 ? tmp1 : tmp2;
      }*/

   for (i=0;i<order;i++)
      qlsp[i]=SUB16(qlsp[i],LSP_LINEAR_HIGH(i));
#ifndef FIXED_POINT
   for (i=0;i<order;i++)
      qlsp[i] = qlsp[i]*LSP_SCALE;
#endif
   id = lsp_quant(qlsp, high_lsp_cdbk, 64, order);
   speex_bits_pack(bits, id, 6);

   for (i=0;i<order;i++)
      qlsp[i]*=2;

   id = lsp_weight_quant(qlsp, quant_weight, high_lsp_cdbk2, 64, order);
   speex_bits_pack(bits, id, 6);

#ifdef FIXED_POINT
   for (i=0;i<order;i++)
      qlsp[i] = PSHR16(qlsp[i],1);
#else
   for (i=0;i<order;i++)
      qlsp[i] = qlsp[i]*0.0019531;
#endif

   for (i=0;i<order;i++)
      qlsp[i]=lsp[i]-qlsp[i];
}

void lsp_unquant_high(spx_lsp_t *lsp, int order, SpeexBits *bits)
{

   int i, id;
   for (i=0;i<order;i++)
      lsp[i]=LSP_LINEAR_HIGH(i);


   id=speex_bits_unpack_unsigned(bits, 6);
   for (i=0;i<order;i++)
      lsp[i] += LSP_DIV_256(high_lsp_cdbk[id*order+i]);


   id=speex_bits_unpack_unsigned(bits, 6);
   for (i=0;i<order;i++)
      lsp[i] += LSP_DIV_512(high_lsp_cdbk2[id*order+i]);
}

#endif

