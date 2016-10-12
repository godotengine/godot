/* Copyright (C) 2002-2006 Jean-Marc Valin */
/**
    @file nb_celp.h
    @brief Narrowband CELP encoder/decoder
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

#ifndef NB_CELP_H
#define NB_CELP_H

#include "modes.h"
#include <speex/speex_bits.h>
#include <speex/speex_callbacks.h>
#include "vbr.h"
#include "filters.h"

#ifdef VORBIS_PSYCHO
#include "vorbis_psy.h"
#endif

/**Structure representing the full state of the narrowband encoder*/
typedef struct EncState {
   const SpeexMode *mode;        /**< Mode corresponding to the state */
   int    first;                 /**< Is this the first frame? */
   int    frameSize;             /**< Size of frames */
   int    subframeSize;          /**< Size of sub-frames */
   int    nbSubframes;           /**< Number of sub-frames */
   int    windowSize;            /**< Analysis (LPC) window length */
   int    lpcSize;               /**< LPC order */
   int    min_pitch;             /**< Minimum pitch value allowed */
   int    max_pitch;             /**< Maximum pitch value allowed */

   spx_word32_t cumul_gain;      /**< Product of previously used pitch gains (Q10) */
   int    bounded_pitch;         /**< Next frame should not rely on previous frames for pitch */
   int    ol_pitch;              /**< Open-loop pitch */
   int    ol_voiced;             /**< Open-loop voiced/non-voiced decision */
   int   *pitch;

#ifdef VORBIS_PSYCHO
   VorbisPsy *psy;
   float *psy_window;
   float *curve;
   float *old_curve;
#endif

   spx_word16_t  gamma1;         /**< Perceptual filter: A(z/gamma1) */
   spx_word16_t  gamma2;         /**< Perceptual filter: A(z/gamma2) */
   spx_word16_t  lpc_floor;      /**< Noise floor multiplier for A[0] in LPC analysis*/
   char  *stack;                 /**< Pseudo-stack allocation for temporary memory */
   spx_word16_t *winBuf;         /**< Input buffer (original signal) */
   spx_word16_t *excBuf;         /**< Excitation buffer */
   spx_word16_t *exc;            /**< Start of excitation frame */
   spx_word16_t *swBuf;          /**< Weighted signal buffer */
   spx_word16_t *sw;             /**< Start of weighted signal frame */
   const spx_word16_t *window;   /**< Temporary (Hanning) window */
   const spx_word16_t *lagWindow;      /**< Window applied to auto-correlation */
   spx_lsp_t *old_lsp;           /**< LSPs for previous frame */
   spx_lsp_t *old_qlsp;          /**< Quantized LSPs for previous frame */
   spx_mem_t *mem_sp;            /**< Filter memory for signal synthesis */
   spx_mem_t *mem_sw;            /**< Filter memory for perceptually-weighted signal */
   spx_mem_t *mem_sw_whole;      /**< Filter memory for perceptually-weighted signal (whole frame)*/
   spx_mem_t *mem_exc;           /**< Filter memory for excitation (whole frame) */
   spx_mem_t *mem_exc2;          /**< Filter memory for excitation (whole frame) */
   spx_mem_t mem_hp[2];          /**< High-pass filter memory */
   spx_word32_t *pi_gain;        /**< Gain of LPC filter at theta=pi (fe/2) */
   spx_word16_t *innov_rms_save; /**< If non-NULL, innovation RMS is copied here */

#ifndef DISABLE_VBR
   VBRState *vbr;                /**< State of the VBR data */
   float  vbr_quality;           /**< Quality setting for VBR encoding */
   float  relative_quality;      /**< Relative quality that will be needed by VBR */
   spx_int32_t vbr_enabled;      /**< 1 for enabling VBR, 0 otherwise */
   spx_int32_t vbr_max;          /**< Max bit-rate allowed in VBR mode */
   int    vad_enabled;           /**< 1 for enabling VAD, 0 otherwise */
   int    dtx_enabled;           /**< 1 for enabling DTX, 0 otherwise */
   int    dtx_count;             /**< Number of consecutive DTX frames */
   spx_int32_t abr_enabled;      /**< ABR setting (in bps), 0 if off */
   float  abr_drift;
   float  abr_drift2;
   float  abr_count;
#endif /* #ifndef DISABLE_VBR */
   
   int    complexity;            /**< Complexity setting (0-10 from least complex to most complex) */
   spx_int32_t sampling_rate;
   int    plc_tuning;
   int    encode_submode;
   const SpeexSubmode * const *submodes; /**< Sub-mode data */
   int    submodeID;             /**< Activated sub-mode */
   int    submodeSelect;         /**< Mode chosen by the user (may differ from submodeID if VAD is on) */
   int    isWideband;            /**< Is this used as part of the embedded wideband codec */
   int    highpass_enabled;        /**< Is the input filter enabled */
} EncState;

/**Structure representing the full state of the narrowband decoder*/
typedef struct DecState {
   const SpeexMode *mode;       /**< Mode corresponding to the state */
   int    first;                /**< Is this the first frame? */
   int    count_lost;           /**< Was the last frame lost? */
   int    frameSize;            /**< Size of frames */
   int    subframeSize;         /**< Size of sub-frames */
   int    nbSubframes;          /**< Number of sub-frames */
   int    lpcSize;              /**< LPC order */
   int    min_pitch;            /**< Minimum pitch value allowed */
   int    max_pitch;            /**< Maximum pitch value allowed */
   spx_int32_t sampling_rate;

   spx_word16_t  last_ol_gain;  /**< Open-loop gain for previous frame */

   char  *stack;                /**< Pseudo-stack allocation for temporary memory */
   spx_word16_t *excBuf;        /**< Excitation buffer */
   spx_word16_t *exc;           /**< Start of excitation frame */
   spx_lsp_t *old_qlsp;         /**< Quantized LSPs for previous frame */
   spx_coef_t *interp_qlpc;     /**< Interpolated quantized LPCs */
   spx_mem_t *mem_sp;           /**< Filter memory for synthesis signal */
   spx_mem_t mem_hp[2];         /**< High-pass filter memory */
   spx_word32_t *pi_gain;       /**< Gain of LPC filter at theta=pi (fe/2) */
   spx_word16_t *innov_save;    /** If non-NULL, innovation is copied here */
   
   spx_word16_t level;
   spx_word16_t max_level;
   spx_word16_t min_level;
   
   /* This is used in packet loss concealment */
   int    last_pitch;           /**< Pitch of last correctly decoded frame */
   spx_word16_t  last_pitch_gain; /**< Pitch gain of last correctly decoded frame */
   spx_word16_t  pitch_gain_buf[3]; /**< Pitch gain of last decoded frames */
   int    pitch_gain_buf_idx;   /**< Tail of the buffer */
   spx_int32_t seed;            /** Seed used for random number generation */
   
   int    encode_submode;
   const SpeexSubmode * const *submodes; /**< Sub-mode data */
   int    submodeID;            /**< Activated sub-mode */
   int    lpc_enh_enabled;      /**< 1 when LPC enhancer is on, 0 otherwise */
   SpeexCallback speex_callbacks[SPEEX_MAX_CALLBACKS];

   SpeexCallback user_callback;

   /*Vocoder data*/
   spx_word16_t  voc_m1;
   spx_word32_t  voc_m2;
   spx_word16_t  voc_mean;
   int    voc_offset;

   int    dtx_enabled;
   int    isWideband;            /**< Is this used as part of the embedded wideband codec */
   int    highpass_enabled;        /**< Is the input filter enabled */
} DecState;

/** Initializes encoder state*/
void *nb_encoder_init(const SpeexMode *m);

/** De-allocates encoder state resources*/
void nb_encoder_destroy(void *state);

/** Encodes one frame*/
int nb_encode(void *state, void *in, SpeexBits *bits);


/** Initializes decoder state*/
void *nb_decoder_init(const SpeexMode *m);

/** De-allocates decoder state resources*/
void nb_decoder_destroy(void *state);

/** Decodes one frame*/
int nb_decode(void *state, SpeexBits *bits, void *out);

/** ioctl-like function for controlling a narrowband encoder */
int nb_encoder_ctl(void *state, int request, void *ptr);

/** ioctl-like function for controlling a narrowband decoder */
int nb_decoder_ctl(void *state, int request, void *ptr);


#endif
