/* Copyright (C) 2002-2007 Jean-Marc Valin 
   File: modes.c

   Describes the wideband modes of the codec

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


#include "modes.h"
#include "ltp.h"
#include "quant_lsp.h"
#include "cb_search.h"
#include "sb_celp.h"
#include "nb_celp.h"
#include "vbr.h"
#include "arch.h"
#include <math.h>
#include "os_support.h"


#ifndef NULL
#define NULL 0
#endif

EXPORT const SpeexMode * const speex_mode_list[SPEEX_NB_MODES] = {&speex_nb_mode, &speex_wb_mode, &speex_uwb_mode};

extern const signed char hexc_table[];
extern const signed char hexc_10_32_table[];

#ifndef DISABLE_WIDEBAND

/* Split-VQ innovation for high-band wideband */
static const split_cb_params split_cb_high = {
   8,               /*subvect_size*/
   5,               /*nb_subvect*/
   hexc_table,       /*shape_cb*/
   7,               /*shape_bits*/
   1,
};


/* Split-VQ innovation for high-band wideband */
static const split_cb_params split_cb_high_lbr = {
   10,               /*subvect_size*/
   4,               /*nb_subvect*/
   hexc_10_32_table,       /*shape_cb*/
   5,               /*shape_bits*/
   0,
};

#endif


static const SpeexSubmode wb_submode1 = {
   0,
   0,
   1,
   0,
   /*LSP quantization*/
   lsp_quant_high,
   lsp_unquant_high,
   /*Pitch quantization*/
   NULL,
   NULL,
   NULL,
   /*No innovation quantization*/
   NULL,
   NULL,
   NULL,
   -1,
   36
};


static const SpeexSubmode wb_submode2 = {
   0,
   0,
   1,
   0,
   /*LSP quantization*/
   lsp_quant_high,
   lsp_unquant_high,
   /*Pitch quantization*/
   NULL,
   NULL,
   NULL,
   /*Innovation quantization*/
   split_cb_search_shape_sign,
   split_cb_shape_sign_unquant,
#ifdef DISABLE_WIDEBAND
   NULL,
#else
   &split_cb_high_lbr,
#endif
   -1,
   112
};


static const SpeexSubmode wb_submode3 = {
   0,
   0,
   1,
   0,
   /*LSP quantization*/
   lsp_quant_high,
   lsp_unquant_high,
   /*Pitch quantization*/
   NULL,
   NULL,
   NULL,
   /*Innovation quantization*/
   split_cb_search_shape_sign,
   split_cb_shape_sign_unquant,
#ifdef DISABLE_WIDEBAND
   NULL,
#else
   &split_cb_high,
#endif
   -1,
   192
};

static const SpeexSubmode wb_submode4 = {
   0,
   0,
   1,
   1,
   /*LSP quantization*/
   lsp_quant_high,
   lsp_unquant_high,
   /*Pitch quantization*/
   NULL,
   NULL,
   NULL,
   /*Innovation quantization*/
   split_cb_search_shape_sign,
   split_cb_shape_sign_unquant,
#ifdef DISABLE_WIDEBAND
   NULL,
#else
   &split_cb_high,
#endif
   -1,
   352
};


/* Split-band wideband CELP mode*/
static const SpeexSBMode sb_wb_mode = {
   &speex_nb_mode,
   160,    /*frameSize*/
   40,     /*subframeSize*/
   8,     /*lpcSize*/
#ifdef FIXED_POINT
   29491, 19661, /* gamma1, gamma2 */
#else
   0.9, 0.6, /* gamma1, gamma2 */
#endif
   QCONST16(.0002,15), /*lpc_floor*/
   QCONST16(0.9f,15),
   {NULL, &wb_submode1, &wb_submode2, &wb_submode3, &wb_submode4, NULL, NULL, NULL},
   3,
   {1, 8, 2, 3, 4, 5, 5, 6, 6, 7, 7},
   {1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4},
#ifndef DISABLE_VBR
   vbr_hb_thresh,
#endif
   5
};


EXPORT const SpeexMode speex_wb_mode = {
   &sb_wb_mode,
   wb_mode_query,
   "wideband (sub-band CELP)",
   1,
   4,
   &sb_encoder_init,
   &sb_encoder_destroy,
   &sb_encode,
   &sb_decoder_init,
   &sb_decoder_destroy,
   &sb_decode,
   &sb_encoder_ctl,
   &sb_decoder_ctl,
};



/* "Ultra-wideband" mode stuff */



/* Split-band "ultra-wideband" (32 kbps) CELP mode*/
static const SpeexSBMode sb_uwb_mode = {
   &speex_wb_mode,
   320,    /*frameSize*/
   80,     /*subframeSize*/
   8,     /*lpcSize*/
#ifdef FIXED_POINT
   29491, 19661, /* gamma1, gamma2 */
#else
   0.9, 0.6, /* gamma1, gamma2 */
#endif
   QCONST16(.0002,15), /*lpc_floor*/
   QCONST16(0.7f,15),
   {NULL, &wb_submode1, NULL, NULL, NULL, NULL, NULL, NULL},
   1,
   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
   {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
#ifndef DISABLE_VBR
   vbr_uhb_thresh,
#endif
   2
};

int wb_mode_query(const void *mode, int request, void *ptr)
{
   const SpeexSBMode *m = (const SpeexSBMode*)mode;

   switch (request)
   {
      case SPEEX_MODE_FRAME_SIZE:
         *((int*)ptr)=2*m->frameSize;
         break;
      case SPEEX_SUBMODE_BITS_PER_FRAME:
         if (*((int*)ptr)==0)
            *((int*)ptr) = SB_SUBMODE_BITS+1;
         else if (m->submodes[*((int*)ptr)]==NULL)
            *((int*)ptr) = -1;
         else
            *((int*)ptr) = m->submodes[*((int*)ptr)]->bits_per_frame;
         break;
      default:
         speex_warning_int("Unknown wb_mode_query request: ", request);
         return -1;
   }
   return 0;
}


EXPORT const SpeexMode speex_uwb_mode = {
   &sb_uwb_mode,
   wb_mode_query,
   "ultra-wideband (sub-band CELP)",
   2,
   4,
   &sb_encoder_init,
   &sb_encoder_destroy,
   &sb_encode,
   &sb_decoder_init,
   &sb_decoder_destroy,
   &sb_decode,
   &sb_encoder_ctl,
   &sb_decoder_ctl,
};

/* We have defined speex_lib_get_mode() as a macro in speex.h */
#undef speex_lib_get_mode

EXPORT const SpeexMode * speex_lib_get_mode (int mode)
{
   if (mode < 0 || mode >= SPEEX_NB_MODES) return NULL;

   return speex_mode_list[mode];
}



