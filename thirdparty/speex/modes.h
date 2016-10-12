/* Copyright (C) 2002-2006 Jean-Marc Valin */
/**
   @file modes.h
   @brief Describes the different modes of the codec
*/
/*
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

#ifndef MODES_H
#define MODES_H

#include <speex/speex.h>
#include <speex/speex_bits.h>
#include "arch.h"

#define NB_SUBMODES 16
#define NB_SUBMODE_BITS 4

#define SB_SUBMODES 8
#define SB_SUBMODE_BITS 3

/* Used internally, NOT TO BE USED in applications */
/** Used internally*/
#define SPEEX_GET_PI_GAIN 100
/** Used internally*/
#define SPEEX_GET_EXC     101
/** Used internally*/
#define SPEEX_GET_INNOV   102
/** Used internally*/
#define SPEEX_GET_DTX_STATUS   103
/** Used internally*/
#define SPEEX_SET_INNOVATION_SAVE   104
/** Used internally*/
#define SPEEX_SET_WIDEBAND   105

/** Used internally*/
#define SPEEX_GET_STACK   106


/** Quantizes LSPs */
typedef void (*lsp_quant_func)(spx_lsp_t *, spx_lsp_t *, int, SpeexBits *);

/** Decodes quantized LSPs */
typedef void (*lsp_unquant_func)(spx_lsp_t *, int, SpeexBits *);


/** Long-term predictor quantization */
typedef int (*ltp_quant_func)(spx_word16_t *, spx_word16_t *, spx_coef_t *, spx_coef_t *, 
                              spx_coef_t *, spx_sig_t *, const void *, int, int, spx_word16_t, 
                              int, int, SpeexBits*, char *, spx_word16_t *, spx_word16_t *, int, int, int, spx_word32_t *);

/** Long-term un-quantize */
typedef void (*ltp_unquant_func)(spx_word16_t *, spx_word32_t *, int, int, spx_word16_t, const void *, int, int *,
                                 spx_word16_t *, SpeexBits*, char*, int, int, spx_word16_t, int);


/** Innovation quantization function */
typedef void (*innovation_quant_func)(spx_word16_t *, spx_coef_t *, spx_coef_t *, spx_coef_t *, const void *, int, int, 
                                      spx_sig_t *, spx_word16_t *, SpeexBits *, char *, int, int);

/** Innovation unquantization function */
typedef void (*innovation_unquant_func)(spx_sig_t *, const void *, int, SpeexBits*, char *, spx_int32_t *);

/** Description of a Speex sub-mode (wither narrowband or wideband */
typedef struct SpeexSubmode {
   int     lbr_pitch;          /**< Set to -1 for "normal" modes, otherwise encode pitch using a global pitch and allowing a +- lbr_pitch variation (for low not-rates)*/
   int     forced_pitch_gain;  /**< Use the same (forced) pitch gain for all sub-frames */
   int     have_subframe_gain; /**< Number of bits to use as sub-frame innovation gain */
   int     double_codebook;    /**< Apply innovation quantization twice for higher quality (and higher bit-rate)*/
   /*LSP functions*/
   lsp_quant_func    lsp_quant; /**< LSP quantization function */
   lsp_unquant_func  lsp_unquant; /**< LSP unquantization function */

   /*Long-term predictor functions*/
   ltp_quant_func    ltp_quant; /**< Long-term predictor (pitch) quantizer */
   ltp_unquant_func  ltp_unquant; /**< Long-term predictor (pitch) un-quantizer */
   const void       *ltp_params; /**< Pitch parameters (options) */

   /*Quantization of innovation*/
   innovation_quant_func innovation_quant; /**< Innovation quantization */
   innovation_unquant_func innovation_unquant; /**< Innovation un-quantization */
   const void             *innovation_params; /**< Innovation quantization parameters*/

   spx_word16_t      comb_gain;  /**< Gain of enhancer comb filter */

   int               bits_per_frame; /**< Number of bits per frame after encoding*/
} SpeexSubmode;

/** Struct defining the encoding/decoding mode*/
typedef struct SpeexNBMode {
   int     frameSize;      /**< Size of frames used for encoding */
   int     subframeSize;   /**< Size of sub-frames used for encoding */
   int     lpcSize;        /**< Order of LPC filter */
   int     pitchStart;     /**< Smallest pitch value allowed */
   int     pitchEnd;       /**< Largest pitch value allowed */

   spx_word16_t gamma1;    /**< Perceptual filter parameter #1 */
   spx_word16_t gamma2;    /**< Perceptual filter parameter #2 */
   spx_word16_t   lpc_floor;      /**< Noise floor for LPC analysis */

   const SpeexSubmode *submodes[NB_SUBMODES]; /**< Sub-mode data for the mode */
   int     defaultSubmode; /**< Default sub-mode to use when encoding */
   int     quality_map[11]; /**< Mode corresponding to each quality setting */
} SpeexNBMode;


/** Struct defining the encoding/decoding mode for SB-CELP (wideband) */
typedef struct SpeexSBMode {
   const SpeexMode *nb_mode;    /**< Embedded narrowband mode */
   int     frameSize;     /**< Size of frames used for encoding */
   int     subframeSize;  /**< Size of sub-frames used for encoding */
   int     lpcSize;       /**< Order of LPC filter */
   spx_word16_t gamma1;   /**< Perceptual filter parameter #1 */
   spx_word16_t gamma2;   /**< Perceptual filter parameter #1 */
   spx_word16_t   lpc_floor;     /**< Noise floor for LPC analysis */
   spx_word16_t   folding_gain;

   const SpeexSubmode *submodes[SB_SUBMODES]; /**< Sub-mode data for the mode */
   int     defaultSubmode; /**< Default sub-mode to use when encoding */
   int     low_quality_map[11]; /**< Mode corresponding to each quality setting */
   int     quality_map[11]; /**< Mode corresponding to each quality setting */
#ifndef DISABLE_VBR
   const float (*vbr_thresh)[11];
#endif
   int     nb_modes;
} SpeexSBMode;

int speex_encode_native(void *state, spx_word16_t *in, SpeexBits *bits);
int speex_decode_native(void *state, SpeexBits *bits, spx_word16_t *out);

int nb_mode_query(const void *mode, int request, void *ptr);
int wb_mode_query(const void *mode, int request, void *ptr);

#endif
