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
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef ANALYSIS_H
#define ANALYSIS_H

#include "celt.h"
#include "opus_private.h"
#include "mlp.h"

#define NB_FRAMES 8
#define NB_TBANDS 18
#define ANALYSIS_BUF_SIZE 720 /* 30 ms at 24 kHz */

/* At that point we can stop counting frames because it no longer matters. */
#define ANALYSIS_COUNT_MAX 10000

#define DETECT_SIZE 100

/* Uncomment this to print the MLP features on stdout. */
/*#define MLP_TRAINING*/

typedef struct {
   int arch;
   int application;
   opus_int32 Fs;
#define TONALITY_ANALYSIS_RESET_START angle
   float angle[240];
   float d_angle[240];
   float d2_angle[240];
   opus_val32 inmem[ANALYSIS_BUF_SIZE];
   int   mem_fill;                      /* number of usable samples in the buffer */
   float prev_band_tonality[NB_TBANDS];
   float prev_tonality;
   int prev_bandwidth;
   float E[NB_FRAMES][NB_TBANDS];
   float logE[NB_FRAMES][NB_TBANDS];
   float lowE[NB_TBANDS];
   float highE[NB_TBANDS];
   float meanE[NB_TBANDS+1];
   float mem[32];
   float cmean[8];
   float std[9];
   float Etracker;
   float lowECount;
   int E_count;
   int count;
   int analysis_offset;
   int write_pos;
   int read_pos;
   int read_subframe;
   float hp_ener_accum;
   int initialized;
   float rnn_state[MAX_NEURONS];
   opus_val32 downmix_state[3];
   AnalysisInfo info[DETECT_SIZE];
} TonalityAnalysisState;

/** Initialize a TonalityAnalysisState struct.
 *
 * This performs some possibly slow initialization steps which should
 * not be repeated every analysis step. No allocated memory is retained
 * by the state struct, so no cleanup call is required.
 */
void tonality_analysis_init(TonalityAnalysisState *analysis, opus_int32 Fs);

/** Reset a TonalityAnalysisState struct.
 *
 * Call this when there's a discontinuity in the data.
 */
void tonality_analysis_reset(TonalityAnalysisState *analysis);

void tonality_get_info(TonalityAnalysisState *tonal, AnalysisInfo *info_out, int len);

void run_analysis(TonalityAnalysisState *analysis, const CELTMode *celt_mode, const void *analysis_pcm,
                 int analysis_frame_size, int frame_size, int c1, int c2, int C, opus_int32 Fs,
                 int lsb_depth, downmix_func downmix, AnalysisInfo *analysis_info);

#endif
